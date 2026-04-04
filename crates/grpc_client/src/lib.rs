//! gRPC clients for SGLang, vLLM, TensorRT-LLM, and MLX backends
//!
//! This crate provides gRPC client implementations for communicating with
//! SGLang scheduler, vLLM engine, TensorRT-LLM engine, and MLX engine backends.

pub mod common_proto {
    #![allow(clippy::all, clippy::absolute_paths, unused_qualifications)]
    tonic::include_proto!("smg.grpc.common");
}
pub mod mlx_engine;
pub mod sglang_scheduler;
pub mod tokenizer_bundle;
pub mod trtllm_service;
pub mod vllm_engine;

// Re-export clients
use std::sync::Arc;

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
