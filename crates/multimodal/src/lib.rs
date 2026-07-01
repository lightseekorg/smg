pub mod error;
pub mod hasher;
pub mod hub;
pub mod jpeg_turbo;
pub mod media;
#[cfg(feature = "opencv-video")]
mod opencv_buffer;
pub mod registry;
pub mod runtime;
pub mod tracker;
pub mod types;
pub mod vision;

pub use error::{MediaConnectorError, MultiModalError, MultiModalResult};
pub use media::{
    ImageFetchConfig, MediaConnector, MediaConnectorConfig, MediaSource, VideoFetchConfig,
};
pub use registry::{ModelMetadata, ModelProcessorSpec, ModelRegistry};
pub use runtime::MultimodalRuntime;
pub use tracker::{AsyncMultiModalTracker, TrackerOutput};
pub use types::{
    FieldLayout, ImageDetail, ImageFrame, ImageSize, ImageSource, MediaContentPart, Modality,
    MultiModalData, MultiModalUUIDs, PlaceholderRange, PromptReplacement, RgbFrameRef, TokenId,
    TrackedMedia, VideoClip, VideoSource,
};
// Re-export vision processing components
pub use vision::{
    DeferredNormalizedEncoderInput, EncoderInput, LlavaNextProcessor, LlavaProcessor,
    ModalityInput, ModalityPreProcessor, ModelSpecificValue, OutputPreference, PreProcessorConfig,
    PreprocessRequest, PreprocessedEncoderInputs, TransformError, VideoInput, VisionPreProcessor,
    VisionProcessorRegistry,
};
