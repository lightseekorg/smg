//! Pure Rust vision processing module for multimodal models.
//!
//! This module provides vision preprocessing pipelines that match HuggingFace
//! processor outputs without requiring Python dependencies.
//!
//! # Architecture
//!
//! The vision module is structured as follows:
//!
//! - `transforms`: Core image transformations (resize, normalize, crop, etc.)
//! - `preprocessor_config`: HuggingFace config parsing
//! - `processor`: Trait and output types for processors
//! - `processors`: Model-specific implementations (LLaVA, Qwen-VL, etc.)
//!
//! # Usage
//!
//! ```rust,ignore
//! use smg::multimodal::vision::{
//!     PreProcessorConfig,
//!     processors::LlavaProcessor,
//!     VisionPreProcessor,
//! };
//!
//! // Load config from HuggingFace
//! let config = PreProcessorConfig::from_json(config_json)?;
//!
//! // Create processor and preprocess images
//! let processor = LlavaProcessor::new();
//! let result = processor.preprocess(&images, &config)?;
//! ```

pub mod preprocessor_config;
pub mod processor;
pub mod processors;
pub mod transforms;

// Re-export commonly used types
pub use preprocessor_config::PreProcessorConfig;
pub use processor::{
    ModelSpecificValue, PreprocessedEncoderInputs, VisionPreProcessor, VisionProcessorRegistry,
};
pub use processors::{
    Llama4VisionProcessor, LlavaNextProcessor, LlavaProcessor, Phi3VisionProcessor,
    Phi4VisionProcessor, PixtralProcessor, Qwen2VLProcessor, Qwen3VLProcessor,
};
pub use transforms::TransformError;
