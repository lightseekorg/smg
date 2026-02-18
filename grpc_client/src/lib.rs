//! gRPC clients for SGLang, vLLM, and TensorRT-LLM backends
//!
//! This crate provides gRPC client implementations for communicating with
//! SGLang scheduler, vLLM engine, and TensorRT-LLM engine backends.

pub mod sglang_scheduler;
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

/// Generate a `decode_tokenizer_chunk` function for each backend's proto module.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_decode_tokenizer_chunk {
    () => {
        fn decode_tokenizer_chunk(
            chunk: proto::GetTokenizerChunk,
        ) -> Result<TokenizerChunk, String> {
            match chunk.chunk {
                Some(proto::get_tokenizer_chunk::Chunk::Metadata(meta)) => {
                    Ok(TokenizerChunk::Metadata(TokenizerMetadata {
                        model_identifier: meta.model_identifier,
                        fingerprint: meta.fingerprint,
                        files: meta
                            .files
                            .into_iter()
                            .map(|file| TokenizerFileDescriptor {
                                file_name: file.file_name,
                                mime_type: file.mime_type,
                                optional: file.optional,
                            })
                            .collect(),
                        bundle_format: meta.bundle_format,
                    }))
                }
                Some(proto::get_tokenizer_chunk::Chunk::FileChunk(file_chunk)) => {
                    Ok(TokenizerChunk::FileChunk(TokenizerFileChunk {
                        data: file_chunk.data,
                        chunk_index: file_chunk.chunk_index,
                        is_last_chunk: file_chunk.is_last_chunk,
                    }))
                }
                None => Err("Protocol error: empty tokenizer chunk".to_string()),
            }
        }
    };
}

/// Generate a `get_tokenizer` method for each backend client.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_get_tokenizer {
    () => {
        /// Get tokenizer bundle from the backend.
        ///
        /// Streams a tokenizer bundle (metadata + compressed file chunks)
        /// from the backend and validates the streaming output.
        /// The stream is bounded by a 120s timeout to prevent indefinite
        /// blocking on slow or stalled backends.
        pub async fn get_tokenizer(
            &self,
        ) -> Result<TokenizerBundle, Box<dyn std::error::Error + Send + Sync>> {
            tracing::debug!("Requesting tokenizer from backend");
            let request = tonic::Request::new(proto::GetTokenizerRequest {});
            let mut client = self.client.clone();
            let mut stream = client.get_tokenizer(request).await?.into_inner();
            let timeout = std::time::Duration::from_secs(120);
            tokio::time::timeout(
                timeout,
                collect_tokenizer_bundle(&mut stream, decode_tokenizer_chunk),
            )
            .await
            .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
                "get_tokenizer stream timed out after 120s".into()
            })?
            .map_err(|e| e.into())
        }
    };
}
