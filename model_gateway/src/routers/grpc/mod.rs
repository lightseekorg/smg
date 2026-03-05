//! gRPC router implementations

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
    /// Preprocessed multimodal intermediate (deferred assembly).
    /// Populated during preparation when multimodal content is detected.
    /// Assembled into backend-specific `MultimodalData` in request_building.
    pub(crate) multimodal_intermediate: Option<multimodal::MultimodalIntermediate>,
    pub stop_sequences: Option<StringOrArray>,
}
