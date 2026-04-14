//! Kimi-K2.5 image processor.
//!
//! Kimi-K2.5 uses MoonViT3d with a NaViT-style architecture very similar
//! to Qwen VL: patch_size=14, merge_kernel_size=(2,2), dynamic resolution.
//! We reuse the QwenVLProcessorBase with Kimi-specific defaults.

use image::DynamicImage;

use super::qwen_vl_base::{QwenVLConfig, QwenVLProcessorBase};
use crate::vision::{
    image_processor::{ImagePreProcessor, PreprocessedImages},
    preprocessor_config::PreProcessorConfig,
    transforms::TransformError,
};

pub struct KimiK25Processor {
    inner: QwenVLProcessorBase,
}

impl Default for KimiK25Processor {
    fn default() -> Self {
        Self::new()
    }
}

impl KimiK25Processor {
    pub fn new() -> Self {
        Self {
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size: 14,
                merge_size: 2,
                min_pixels: 14 * 14 * 4,
                max_pixels: 14 * 14 * 16384,
                temporal_patch_size: 1,
                mean: [0.5, 0.5, 0.5],
                std: [0.5, 0.5, 0.5],
                model_name: "kimi_k25",
            }),
        }
    }
}

impl ImagePreProcessor for KimiK25Processor {
    fn default_mean(&self) -> [f64; 3] {
        [0.5, 0.5, 0.5]
    }

    fn default_std(&self) -> [f64; 3] {
        [0.5, 0.5, 0.5]
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedImages, TransformError> {
        self.inner.preprocess(images, config)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        self.inner.calculate_num_tokens(width, height, config)
    }

    fn model_name(&self) -> &'static str {
        "kimi_k25"
    }
}
