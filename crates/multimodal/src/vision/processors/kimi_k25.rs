//! Kimi-K2.5 image processor.
//!
//! This module provides the Kimi-K2.5 processor which wraps the shared
//! `QwenVLProcessorBase` with Kimi-K2.5 specific default parameters.
//!
//! Kimi-K2.5 uses MoonViT with NaViT-style patchification that is structurally
//! identical to Qwen2-VL's processing pipeline, differing in normalization
//! (simple [0.5, 0.5, 0.5] instead of CLIP mean/std) and default max_pixels
//! (3,211,264 vs 1,003,520).
//!
//! # Kimi-K2.5 Parameters
//!
//! - patch_size: 14
//! - merge_size: 2
//! - factor: 28 (patch_size * merge_size)
//! - normalization: [0.5, 0.5, 0.5] mean/std

use std::ops::Deref;

use image::DynamicImage;

use super::qwen_vl_base::{QwenVLConfig, QwenVLProcessorBase};
use crate::vision::{
    image_processor::{ImagePreProcessor, PreprocessedImages},
    preprocessor_config::PreProcessorConfig,
    transforms::TransformError,
};

/// Kimi-K2.5 normalization mean values.
pub const KIMI_K25_MEAN: [f64; 3] = [0.5, 0.5, 0.5];

/// Kimi-K2.5 normalization std values.
pub const KIMI_K25_STD: [f64; 3] = [0.5, 0.5, 0.5];

/// Default minimum pixels for Kimi-K2.5 (256 * 28 * 28 = 200,704)
pub const DEFAULT_MIN_PIXELS: usize = 256 * 28 * 28;

/// Default maximum pixels for Kimi-K2.5 (16384 * 14 * 14 = 3,211,264)
/// Based on in_patch_limit=16384 from preprocessor config.
pub const DEFAULT_MAX_PIXELS: usize = 16384 * 14 * 14;

/// Default patch size
pub const DEFAULT_PATCH_SIZE: usize = 14;

/// Default merge size for token reduction
pub const DEFAULT_MERGE_SIZE: usize = 2;

/// Default temporal patch size (for video frames)
pub const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 2;

/// Kimi-K2.5 image processor.
///
/// This is a thin wrapper around `QwenVLProcessorBase` with Kimi-K2.5
/// specific default parameters:
/// - patch_size: 14
/// - merge_size: 2
/// - [0.5, 0.5, 0.5] normalization mean/std
#[derive(Debug, Clone)]
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
                patch_size: DEFAULT_PATCH_SIZE,
                merge_size: DEFAULT_MERGE_SIZE,
                min_pixels: DEFAULT_MIN_PIXELS,
                max_pixels: DEFAULT_MAX_PIXELS,
                temporal_patch_size: DEFAULT_TEMPORAL_PATCH_SIZE,
                mean: KIMI_K25_MEAN,
                std: KIMI_K25_STD,
                model_name: "kimi-k2.5",
            }),
        }
    }

    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        Self {
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size: config.get_patch_size(DEFAULT_PATCH_SIZE),
                merge_size: config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE),
                min_pixels: config.min_pixels.unwrap_or(DEFAULT_MIN_PIXELS),
                max_pixels: config.max_pixels.unwrap_or(DEFAULT_MAX_PIXELS),
                temporal_patch_size: config
                    .temporal_patch_size
                    .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE),
                mean: KIMI_K25_MEAN,
                std: KIMI_K25_STD,
                model_name: "kimi-k2.5",
            }),
        }
    }
}

impl Deref for KimiK25Processor {
    type Target = QwenVLProcessorBase;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl ImagePreProcessor for KimiK25Processor {
    fn default_mean(&self) -> [f64; 3] {
        self.inner.default_mean()
    }

    fn default_std(&self) -> [f64; 3] {
        self.inner.default_std()
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedImages, TransformError> {
        let mut result = self.inner.preprocess(images, config)?;

        // Kimi-K2.5 engine expects "grid_thws" instead of Qwen-VL's "image_grid_thw"
        if let Some(val) = result.model_specific.remove("image_grid_thw") {
            result.model_specific.insert("grid_thws".to_string(), val);
        }

        Ok(result)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        self.inner.calculate_num_tokens(width, height, config)
    }

    fn model_name(&self) -> &'static str {
        self.inner.model_name()
    }

    fn get_processed_size(&self, config: &PreProcessorConfig) -> Option<(u32, u32)> {
        self.inner.get_processed_size(config)
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;
    use crate::vision::image_processor::ModelSpecificValue;

    fn create_test_image(width: u32, height: u32, color: Rgb<u8>) -> DynamicImage {
        DynamicImage::from(RgbImage::from_pixel(width, height, color))
    }

    #[test]
    fn test_kimi_k25_processor_default() {
        let processor = KimiK25Processor::new();
        assert_eq!(processor.patch_size(), 14);
        assert_eq!(processor.merge_size(), 2);
        assert_eq!(processor.min_pixels(), DEFAULT_MIN_PIXELS);
        assert_eq!(processor.max_pixels(), DEFAULT_MAX_PIXELS);
        assert_eq!(processor.get_factor(), 28); // 14 * 2
    }

    #[test]
    fn test_default_mean_std() {
        let processor = KimiK25Processor::new();
        assert_eq!(processor.default_mean(), KIMI_K25_MEAN);
        assert_eq!(processor.default_std(), KIMI_K25_STD);
    }

    #[test]
    fn test_model_name() {
        let processor = KimiK25Processor::new();
        assert_eq!(processor.model_name(), "kimi-k2.5");
    }

    #[test]
    fn test_kimi_k25_preprocess() {
        let processor = KimiK25Processor::new();
        let config = PreProcessorConfig {
            do_resize: Some(true),
            do_normalize: Some(true),
            image_mean: Some(KIMI_K25_MEAN.to_vec()),
            image_std: Some(KIMI_K25_STD.to_vec()),
            ..Default::default()
        };

        let image = create_test_image(600, 400, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[image], &config).unwrap();

        // pixel_values is patchified: [total_patches, patch_features]
        assert_eq!(result.pixel_values.ndim(), 2);
        assert!(result.pixel_values.shape()[0] > 0);

        // Check grid_thws and patches_per_image are present
        assert!(result.model_specific.contains_key("grid_thws"));
        assert!(result.model_specific.contains_key("patches_per_image"));

        assert!(result.num_img_tokens[0] > 0);
    }

    #[test]
    fn test_kimi_k25_preprocess_multiple() {
        let processor = KimiK25Processor::new();
        let config = PreProcessorConfig::default();

        let images = vec![
            create_test_image(600, 400, Rgb([100, 100, 100])),
            create_test_image(400, 600, Rgb([150, 150, 150])),
        ];

        let result = processor.preprocess(&images, &config).unwrap();

        assert_eq!(result.image_sizes.len(), 2);
        assert_eq!(result.num_img_tokens.len(), 2);
        assert_eq!(result.pixel_values.ndim(), 2);

        if let Some(ModelSpecificValue::IntTensor { data, shape }) =
            result.model_specific.get("grid_thws")
        {
            assert_eq!(shape, &[2, 3]);
            assert_eq!(data.len(), 6);
        } else {
            panic!("Expected grid_thws to be IntTensor");
        }
    }
}
