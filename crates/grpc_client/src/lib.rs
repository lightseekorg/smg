//! gRPC clients for SGLang, vLLM, TensorRT-LLM, and MLX backends
//!
//! This crate provides gRPC client implementations for communicating with
//! SGLang scheduler, vLLM engine, TensorRT-LLM engine, and MLX engine backends.

pub mod common_proto {
    #![allow(clippy::all, clippy::absolute_paths, unused_qualifications)]
    tonic::include_proto!("smg.grpc.common");
}
pub mod abort_on_drop;
pub mod channel;
pub mod mlx_engine;
pub mod sglang_scheduler;
pub mod tokenizer_bundle;
pub mod trtllm_service;
pub mod vllm_engine;

// Re-export clients
use std::sync::Arc;

pub use abort_on_drop::{AbortOnDropClient, AbortOnDropStream};
pub use channel::{connect_channel, normalize_grpc_endpoint};
pub use mlx_engine::{proto as mlx_proto, MlxEngineClient};
pub use sglang_scheduler::{proto as sglang_proto, SglangSchedulerClient};
use tonic::metadata::MetadataMap;
pub use trtllm_service::{proto as trtllm_proto, TrtllmServiceClient};
pub use vllm_engine::{proto as vllm_proto, VllmEngineClient};

/// Shared `get_tokenizer()` implementation for all engine clients.
///
/// Each engine's generated proto client has a `get_tokenizer` RPC method
/// with identical signature (using common proto types). This macro provides
/// the wrapper that calls `collect_bundle_from_rpc` with the standard
/// timeout and chunk extraction.
macro_rules! impl_get_tokenizer {
    () => {
        pub async fn get_tokenizer(
            &self,
        ) -> Result<
            $crate::tokenizer_bundle::StreamBundle,
            Box<dyn std::error::Error + Send + Sync>,
        > {
            use $crate::common_proto::GetTokenizerRequest;
            let request = tonic::Request::new(GetTokenizerRequest {});
            let mut client = self.client.clone();
            $crate::tokenizer_bundle::collect_bundle_from_rpc(
                client.get_tokenizer(request),
                |chunk| (chunk.data, chunk.sha256),
                std::time::Duration::from_secs(120),
            )
            .await
        }
    };
}
pub(crate) use impl_get_tokenizer;

/// Shared `subscribe_kv_events()` implementation for all engine clients.
///
/// Each engine's generated proto client has a `subscribe_kv_events` RPC method
/// with identical signature (using common proto types). This macro provides
/// the wrapper that returns a `tonic::Streaming<KvEventBatch>`.
macro_rules! impl_subscribe_kv_events {
    () => {
        /// Subscribe to KV cache events from the backend.
        /// Returns a long-lived server-streaming response.
        pub async fn subscribe_kv_events(
            &self,
            start_sequence_number: u64,
        ) -> Result<tonic::Streaming<$crate::common_proto::KvEventBatch>, tonic::Status> {
            let request = tonic::Request::new($crate::common_proto::SubscribeKvEventsRequest {
                start_sequence_number,
            });
            let mut client = self.client.clone();
            let response = client.subscribe_kv_events(request).await?;
            Ok(response.into_inner())
        }
    };
}
pub(crate) use impl_subscribe_kv_events;

/// Trait for injecting trace context into gRPC metadata.
///
/// Implement this trait to enable distributed tracing across gRPC calls.
/// The default implementation is a no-op.
pub trait TraceInjector: Send + Sync {
    /// Inject trace context into the given metadata map.
    ///
    /// Returns `Ok(())` on success, or an error if injection fails.
    fn inject(
        &self,
        metadata: &mut MetadataMap,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

/// A no-op trace injector that does nothing.
#[derive(Clone, Default)]
pub struct NoopTraceInjector;

impl TraceInjector for NoopTraceInjector {
    fn inject(
        &self,
        _metadata: &mut MetadataMap,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}

/// Type alias for a boxed trace injector.
pub type BoxedTraceInjector = Arc<dyn TraceInjector>;

/// Generates the boilerplate that every engine client shares: the two
/// `connect` constructors, `with_trace_injector`, and the three "standard"
/// RPCs (`health_check`, `get_model_info`, `get_server_info`) whose
/// request/response types are uniform across the generated proto crates.
///
/// `$proto_client` is the fully-qualified path of the generated tonic
/// client type (which `Self` wraps). `$display_name` is the human-readable
/// name used in the connect log line.
///
/// Each engine's `impl` block invokes this once and then adds engine-
/// specific RPCs (`generate`, `embed`, etc.) below.
macro_rules! impl_engine_client_basics {
    ($proto_client:path, $display_name:literal) => {
        /// Create a new client and connect to the backend.
        pub async fn connect(
            endpoint: &str,
        ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
            Self::connect_with_trace_injector(
                endpoint,
                std::sync::Arc::new($crate::NoopTraceInjector),
            )
            .await
        }

        /// Create a new client with a custom trace injector.
        pub async fn connect_with_trace_injector(
            endpoint: &str,
            trace_injector: $crate::BoxedTraceInjector,
        ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
            tracing::debug!(
                "Connecting to {} gRPC server at {}",
                $display_name,
                endpoint
            );
            let channel = $crate::channel::connect_channel(endpoint).await?;
            let client = <$proto_client>::new(channel);
            Ok(Self {
                client,
                trace_injector,
            })
        }

        /// Set or replace the trace injector.
        #[must_use]
        pub fn with_trace_injector(mut self, trace_injector: $crate::BoxedTraceInjector) -> Self {
            self.trace_injector = trace_injector;
            self
        }

        /// Perform a health check.
        pub async fn health_check(&self) -> Result<proto::HealthCheckResponse, tonic::Status> {
            tracing::debug!("Sending health check request");
            let request = tonic::Request::new(proto::HealthCheckRequest {});
            let mut client = self.client.clone();
            let response = client.health_check(request).await?;
            tracing::debug!("Health check response received");
            Ok(response.into_inner())
        }

        /// Get model information.
        pub async fn get_model_info(&self) -> Result<proto::GetModelInfoResponse, tonic::Status> {
            tracing::debug!("Requesting model info");
            let request = tonic::Request::new(proto::GetModelInfoRequest {});
            let mut client = self.client.clone();
            let response = client.get_model_info(request).await?;
            tracing::debug!("Model info response received");
            Ok(response.into_inner())
        }

        /// Get server information.
        pub async fn get_server_info(&self) -> Result<proto::GetServerInfoResponse, tonic::Status> {
            tracing::debug!("Requesting server info");
            let request = tonic::Request::new(proto::GetServerInfoRequest {});
            let mut client = self.client.clone();
            let response = client.get_server_info(request).await?;
            tracing::debug!("Server info response received");
            Ok(response.into_inner())
        }
    };
}
pub(crate) use impl_engine_client_basics;
