//! gRPC clients for SGLang, vLLM, and TensorRT-LLM backends
//!
//! This crate provides gRPC client implementations for communicating with
//! SGLang scheduler, vLLM engine, and TensorRT-LLM engine backends.

pub mod archive_ops;
pub mod sglang_scheduler;
pub mod stream_bundle;
pub mod trtllm_service;
pub mod vllm_engine;

// Re-export clients
use std::sync::Arc;

pub use sglang_scheduler::{proto as sglang_proto, SglangSchedulerClient};
use tonic::metadata::MetadataMap;
pub use trtllm_service::{proto as trtllm_proto, TrtllmServiceClient};
pub use vllm_engine::{proto as vllm_proto, VllmEngineClient};

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
