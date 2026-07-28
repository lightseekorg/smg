//! Kimi-K2.5 (MoonViT) image processor.
//!
//! Matches the HuggingFace `KimiK25VisionProcessor` preprocessing pipeline,
//! which is implemented in [`super::moonvit`] and shared with Kimi-K3. This
//! module supplies only K2.5's defaults.
//!
//! K2.5's checkpoint declares no `transparent_bg_config`, and its reference
//! processor has no transparency handling at all, so alpha-carrying images are
//! flattened by dropping the alpha channel — matching `image.convert("RGB")`.

use std::ops::Deref;

use image::DynamicImage;

use super::moonvit::{MoonVitConfig, MoonVitProcessorBase};
use crate::vision::{
    preprocessor_config::PreProcessorConfig,
    processor::{PreprocessedEncoderInputs, VisionPreProcessor},
    transforms::TransformError,
};

pub const KIMI_K25_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
pub const KIMI_K25_STD: [f64; 3] = [0.5, 0.5, 0.5];

pub const DEFAULT_PATCH_SIZE: usize = 14;
pub const DEFAULT_MERGE_SIZE: usize = 2;
/// Maximum total patches before merge (from preprocessor_config.json in_patch_limit)
pub const DEFAULT_IN_PATCH_LIMIT: usize = 16384;
/// Maximum patches along one spatial dimension
pub const DEFAULT_PATCH_LIMIT_ON_ONE_SIDE: usize = 512;

#[derive(Debug, Clone)]
pub struct KimiK25Processor {
    inner: MoonVitProcessorBase,
}

impl Default for KimiK25Processor {
    fn default() -> Self {
        Self::new()
    }
}

impl KimiK25Processor {
    /// Create a processor with Kimi-K2.5's published defaults.
    pub fn new() -> Self {
        Self {
            inner: MoonVitProcessorBase::new(MoonVitConfig {
                patch_size: DEFAULT_PATCH_SIZE,
                merge_size: DEFAULT_MERGE_SIZE,
                in_patch_limit: DEFAULT_IN_PATCH_LIMIT,
                patch_limit_on_one_side: DEFAULT_PATCH_LIMIT_ON_ONE_SIDE,
                mean: KIMI_K25_MEAN,
                std: KIMI_K25_STD,
                // The reference K2.5 processor has no transparency handling:
                // alpha is dropped, not composited.
                transparent_bg: None,
                model_name: "kimi-k2.5",
            }),
        }
    }

    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        Self {
            inner: MoonVitProcessorBase::new(MoonVitConfig {
                patch_size: config.get_patch_size(DEFAULT_PATCH_SIZE),
                merge_size: config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE),
                in_patch_limit: config
                    .get_extra::<usize>("in_patch_limit")
                    .unwrap_or(DEFAULT_IN_PATCH_LIMIT),
                patch_limit_on_one_side: config
                    .get_extra::<usize>("patch_limit_on_one_side")
                    .unwrap_or(DEFAULT_PATCH_LIMIT_ON_ONE_SIDE),
                mean: KIMI_K25_MEAN,
                std: KIMI_K25_STD,
                // The reference K2.5 processor has no transparency handling:
                // alpha is dropped, not composited.
                transparent_bg: None,
                model_name: "kimi-k2.5",
            }),
        }
    }

    pub fn patch_size(&self) -> usize {
        self.inner.patch_size()
    }

    pub fn merge_size(&self) -> usize {
        self.inner.merge_size()
    }
}

impl Deref for KimiK25Processor {
    type Target = MoonVitProcessorBase;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl VisionPreProcessor for KimiK25Processor {
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
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.inner.preprocess(images, config)
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
    use crate::vision::{preprocessor_config::PatchSize, processor::ModelSpecificValue};

    fn create_test_image(width: u32, height: u32, color: Rgb<u8>) -> DynamicImage {
        DynamicImage::from(RgbImage::from_pixel(width, height, color))
    }

    #[test]
    fn test_defaults() {
        let p = KimiK25Processor::new();
        assert_eq!(p.patch_size(), 14);
        assert_eq!(p.merge_size(), 2);
        assert_eq!(p.factor(), 28);
    }

    #[test]
    fn test_mean_std() {
        let p = KimiK25Processor::new();
        assert_eq!(p.default_mean(), KIMI_K25_MEAN);
        assert_eq!(p.default_std(), KIMI_K25_STD);
    }

    #[test]
    fn test_model_name() {
        assert_eq!(KimiK25Processor::new().model_name(), "kimi-k2.5");
    }

    #[test]
    fn test_resize_config_no_upscale() {
        let p = KimiK25Processor::new();
        // Small image should NOT be upscaled (scale capped at 1.0)
        let cfg = p.compute_resize_config(100, 100);
        assert!(cfg.new_width <= 100);
        assert!(cfg.new_height <= 100);
        // Padded dimensions must be factor-aligned
        assert_eq!((cfg.new_height + cfg.pad_height) % 28, 0);
        assert_eq!((cfg.new_width + cfg.pad_width) % 28, 0);
    }

    #[test]
    fn test_resize_config_large_image_downscaled() {
        let p = KimiK25Processor::new();
        // Large image should be downscaled
        let cfg = p.compute_resize_config(4000, 3000);
        // Resized dimensions should be smaller than original
        assert!(cfg.new_width < 4000);
        assert!(cfg.new_height < 3000);
        // Per-side patch limit must be respected (HF assertion)
        let padded_h = cfg.new_height + cfg.pad_height;
        let padded_w = cfg.new_width + cfg.pad_width;
        assert!(padded_h / 14 <= DEFAULT_PATCH_LIMIT_ON_ONE_SIDE * 2);
        assert!(padded_w / 14 <= DEFAULT_PATCH_LIMIT_ON_ONE_SIDE * 2);
    }

    #[test]
    fn test_resize_config_matches_hf_reference() {
        let p = KimiK25Processor::new();
        // 600x400 image: scale=1.0 (small enough), resize to 600x400,
        // pad to (600+4=) → let's compute:
        // factor=28, 400 % 28 = 400 - 14*28 = 400-392 = 8, pad_h = 28-8 = 20
        // 600 % 28 = 600 - 21*28 = 600-588 = 12, pad_w = 28-12 = 16
        let cfg = p.compute_resize_config(600, 400);
        assert_eq!(cfg.new_width, 600);
        assert_eq!(cfg.new_height, 400);
        assert_eq!(cfg.pad_height, 20);
        assert_eq!(cfg.pad_width, 16);
        // Padded: 420 x 616, grid: 30 x 44, tokens: (30*44)/(2*2) = 330
        assert_eq!(cfg.num_tokens, 330);
    }

    #[test]
    fn test_preprocess_4d_output() {
        let p = KimiK25Processor::new();
        let config = PreProcessorConfig {
            do_normalize: Some(true),
            image_mean: Some(KIMI_K25_MEAN.to_vec()),
            image_std: Some(KIMI_K25_STD.to_vec()),
            ..Default::default()
        };

        let image = create_test_image(600, 400, Rgb([128, 128, 128]));
        let result = p.preprocess(&[image], &config).unwrap();

        // 4D output: [total_patches, 3, 14, 14]
        assert_eq!(result.encoder_input.ndim(), 4);
        assert_eq!(result.encoder_input.shape()[1], 3);
        assert_eq!(result.encoder_input.shape()[2], 14);
        assert_eq!(result.encoder_input.shape()[3], 14);

        assert!(result.model_specific.contains_key("grid_thws"));
        assert!(result.model_specific.contains_key("patches_per_image"));
        assert!(result.feature_token_counts[0] > 0);
    }

    #[test]
    fn test_preprocess_multiple_images() {
        let p = KimiK25Processor::new();
        let config = PreProcessorConfig::default();
        let images = vec![
            create_test_image(600, 400, Rgb([100, 100, 100])),
            create_test_image(400, 600, Rgb([150, 150, 150])),
        ];

        let result = p.preprocess(&images, &config).unwrap();

        assert_eq!(result.item_sizes.len(), 2);
        assert_eq!(result.feature_token_counts.len(), 2);
        assert_eq!(result.encoder_input.ndim(), 4);
        assert_eq!(result.encoder_input.shape()[1], 3);

        if let Some(ModelSpecificValue::IntTensor { data, shape }) =
            result.model_specific.get("grid_thws")
        {
            assert_eq!(shape, &[2, 3]);
            assert_eq!(data.len(), 6);
        } else {
            panic!("Expected grid_thws to be IntTensor");
        }

        if let Some(ModelSpecificValue::IntTensor { data, .. }) =
            result.model_specific.get("patches_per_image")
        {
            let total: i64 = data.iter().sum();
            assert_eq!(total as usize, result.encoder_input.shape()[0]);
        }
    }

    #[test]
    fn test_calculate_num_tokens() {
        let p = KimiK25Processor::new();
        let config = PreProcessorConfig::default();
        let tokens = p.calculate_num_tokens(600, 400, &config);
        assert_eq!(tokens, 330);
    }

    #[test]
    fn test_from_preprocessor_config() {
        let config = PreProcessorConfig {
            patch_size: Some(PatchSize {
                height: Some(14),
                width: Some(14),
            }),
            merge_size: Some(2),
            ..Default::default()
        };
        let p = KimiK25Processor::from_preprocessor_config(&config);
        assert_eq!(p.patch_size(), 14);
        assert_eq!(p.merge_size(), 2);
    }

    #[test]
    fn test_zero_padding_applied() {
        let p = KimiK25Processor::new();
        let config = PreProcessorConfig {
            image_mean: Some(KIMI_K25_MEAN.to_vec()),
            image_std: Some(KIMI_K25_STD.to_vec()),
            ..Default::default()
        };

        // 100x100 white image — after normalization: (255/255 - 0.5) / 0.5 = 1.0
        // Padded region: (0/255 - 0.5) / 0.5 = -1.0
        let image = create_test_image(100, 100, Rgb([255, 255, 255]));
        let result = p.preprocess(&[image], &config).unwrap();

        let flat = result.encoder_input_flat();
        // Padded region should be normalized black (-1.0)
        let has_neg_ones = flat.iter().any(|&v| (v - (-1.0)).abs() < 1e-6);
        assert!(
            has_neg_ones,
            "Expected normalized-black padding (-1.0) in output"
        );

        // Image region should be normalized white (1.0)
        let has_ones = flat.iter().any(|&v| (v - 1.0).abs() < 1e-6);
        assert!(
            has_ones,
            "Expected normalized-white image values (1.0) in output"
        );
    }

    #[test]
    fn test_preprocess_tiny_image() {
        // 1x1 image should not panic — padded to 28x28
        let p = KimiK25Processor::new();
        let config = PreProcessorConfig {
            image_mean: Some(KIMI_K25_MEAN.to_vec()),
            image_std: Some(KIMI_K25_STD.to_vec()),
            ..Default::default()
        };
        let image = create_test_image(1, 1, Rgb([128, 128, 128]));
        let result = p.preprocess(&[image], &config).unwrap();
        assert_eq!(result.encoder_input.ndim(), 4);
        assert!(result.encoder_input.shape()[0] > 0);
        assert!(result.feature_token_counts[0] > 0);
    }

    #[test]
    fn test_preprocess_empty_batch_returns_error() {
        let p = KimiK25Processor::new();
        let config = PreProcessorConfig::default();
        let result = p.preprocess(&[], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_preprocessor_config_reads_limits() {
        let config = PreProcessorConfig {
            patch_size: Some(PatchSize {
                height: Some(14),
                width: Some(14),
            }),
            merge_size: Some(2),
            extra: [
                ("in_patch_limit".to_string(), serde_json::json!(8192)),
                (
                    "patch_limit_on_one_side".to_string(),
                    serde_json::json!(256),
                ),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        let p = KimiK25Processor::from_preprocessor_config(&config);
        assert_eq!(p.in_patch_limit(), 8192);
        assert_eq!(p.patch_limit_on_one_side(), 256);
    }
}
