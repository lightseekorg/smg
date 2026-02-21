//! gRPC router implementations

use std::sync::Arc;

use llm_multimodal::ImageFrame;
use openai_protocol::common::StringOrArray;

pub mod client; // Used by core/
pub(crate) mod common;
pub(crate) mod context;
pub(crate) mod harmony;
pub(crate) mod multimodal;
pub(crate) mod pd_router; // Used by routers/factory
pub(crate) mod pipeline;
pub(crate) mod proto_wrapper;
pub(crate) mod regular;
pub(crate) mod router; // Used by routers/factory
pub mod utils; // Used by routers/http and bindings/golang

// Re-export for convenience
pub use proto_wrapper::{MultimodalData, TensorBytes};

/// Processed chat messages ready for gRPC generation
#[derive(Debug)]
pub struct ProcessedMessages {
    pub text: String,
    /// Raw fetched images (Phase 1). Backend-specific preprocessing
    /// happens in Phase 2 at request building time.
    pub multimodal_images: Option<Vec<Arc<ImageFrame>>>,
    #[allow(dead_code)]
    pub stop_sequences: Option<StringOrArray>,
}
