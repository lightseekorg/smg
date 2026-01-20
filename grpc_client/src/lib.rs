//! gRPC clients for SGLang and vLLM backends
//!
//! This crate provides gRPC client implementations for communicating with
//! SGLang scheduler and vLLM engine backends.

pub mod sglang_scheduler;
pub mod vllm_engine;

// Re-export clients
use std::sync::Arc;

pub use sglang_scheduler::{proto as sglang_proto, SglangSchedulerClient};
use tonic::metadata::MetadataMap;
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
