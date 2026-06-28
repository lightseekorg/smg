//! Multimodal processing integration for gRPC pipeline (chat + messages).
//!
//! Bridges the `llm-multimodal` crate with the gRPC router pipeline, split by
//! processing phase:
//!
//! - [`detect`]: find modalities and extract content parts from chat/messages.
//! - [`config`]: model config-file registry and per-router component bundle.
//! - [`process`]: fetch media → preprocess → expand placeholder tokens →
//!   build the lightweight [`MultimodalIntermediate`].
//! - [`assemble`]: turn the intermediate into backend-specific `MultimodalData`
//!   once the target backend is known (after worker selection).
//! - [`serialize`]: tensor byte/dtype serialization used by assembly.
//! - [`transport`]: SHM-vs-inline transport resolution and `/dev/shm`
//!   namespace verification.

use std::{
    collections::HashMap,
    sync::{Arc, OnceLock},
};

use llm_multimodal::{
    FieldLayout, ImageFrame, Modality, PlaceholderRange, PreprocessedEncoderInputs, VideoClip,
};

mod assemble;
mod config;
mod detect;
mod pixel_cache;
mod process;
mod serialize;
mod transport;

pub(crate) use assemble::{
    assemble_multimodal_data, assemble_multimodal_data_after_encode, assemble_tokenspeed,
    precomputed_encode_routing_hashes,
};
pub(crate) use config::{
    load_preprocessor_config_file, load_video_preprocessor_config, MultimodalComponents,
    MultimodalConfigRegistry, MultimodalModelConfig,
};
pub(crate) use detect::{chat_modalities, has_multimodal_content_messages};
pub(crate) use process::{
    process_multimodal, process_multimodal_messages, resolve_placeholder_token,
};
pub(crate) use transport::init_mm_transport_defaults;

/// Whether verbose multimodal timing logs are enabled via `SMG_LOG_MM_TIMING`.
/// Read from the environment once and cached; the flag is not expected to change
/// at runtime, and this is called on every multimodal request.
fn log_mm_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SMG_LOG_MM_TIMING")
            .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// Output of the multimodal processing pipeline.
pub(crate) struct MultimodalOutput {
    /// Token IDs with placeholder tokens expanded to the correct count per media item.
    pub expanded_token_ids: Vec<u32>,
    /// Lightweight intermediate holding preprocessing results.
    /// Assembled into backend-specific `MultimodalData` in request_building.
    pub intermediate: MultimodalIntermediate,
}

/// Lightweight intermediate from the preparation stage.
///
/// Holds all preprocessing results without serializing tensors to bytes.
/// The assembly stage converts this into a backend-specific `MultimodalData`
/// variant once the target backend is known (after worker selection).
#[derive(Debug)]
pub(crate) enum MultimodalIntermediate {
    Precomputed(PrecomputedMultimodalIntermediate),
}

#[derive(Debug)]
pub(crate) struct PrecomputedMultimodalIntermediate {
    /// Active modality for this preprocessed payload.
    pub modality: Modality,
    /// Preprocessed encoder input and model-specific tensors (not yet serialized).
    pub preprocessed: PreprocessedEncoderInputs,
    /// Raw image frames (bytes + blake3 hashes).
    pub images: Vec<Arc<ImageFrame>>,
    /// Raw video clips (bytes + blake3 hashes + sampled frames).
    pub videos: Vec<Arc<VideoClip>>,
    /// Full structural placeholder ranges (offset, length).
    pub placeholders: Vec<PlaceholderRange>,
    /// Patch-only placeholder offsets for sglang.
    pub patch_offsets: Option<Vec<(u32, u32)>>,
    /// Placeholder token ID from model config for the active modality.
    pub placeholder_token_id: Option<u32>,
    /// Per-tensor field layout classification from the model spec.
    pub field_layouts: HashMap<String, FieldLayout>,
    /// Tensor keys that should remain on CPU (vLLM `keep_on_cpu` hint).
    pub keep_on_cpu_keys: Vec<String>,
}
