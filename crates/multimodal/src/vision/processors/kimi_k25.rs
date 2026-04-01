//! Kimi-K2.5 (MoonViT) image processor.
//!
//! Kimi-K2.5 uses MoonViT which expects pixel_values as
//! `[total_patches, C, patch_size, patch_size]` (4D on the engine side).
//! The model's PatchEmbed3d applies Conv2d on each patch, so patches must NOT
//! be flattened across the temporal dimension in the preprocessor.
//!
//! # Processing Pipeline
//!
//! 1. NaViT-style resize to fit within min/max pixel bounds
//! 2. Align dimensions to (patch_size * merge_size) boundary
//! 3. Convert to tensor and normalize with [0.5, 0.5, 0.5] mean/std
//! 4. Extract patches as [C, patch_size, patch_size] per spatial position
//!
//! # Token Calculation
//!
//! ```text
//! grid_t = 1  (for images)
//! grid_h = resized_height / patch_size
//! grid_w = resized_width / patch_size
//! num_tokens = (grid_t * grid_h * grid_w) / merge_size²
//! ```
//!
//! # Parameters
//!
//! - patch_size: 14
//! - merge_size: 2
//! - factor: 28 (patch_size * merge_size)
//! - normalization: [0.5, 0.5, 0.5] mean/std
//! - max_pixels: 3,211,264 (from in_patch_limit=16384)

use image::{DynamicImage, GenericImageView};
use ndarray::{Array2, Array3};

use crate::vision::{
    image_processor::{ImagePreProcessor, ModelSpecificValue, PreprocessedImages},
    preprocessor_config::PreProcessorConfig,
    transforms::{normalize, pil_to_filter, resize, to_tensor, TransformError},
};

pub const KIMI_K25_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
pub const KIMI_K25_STD: [f64; 3] = [0.5, 0.5, 0.5];

pub const DEFAULT_PATCH_SIZE: usize = 14;
pub const DEFAULT_MERGE_SIZE: usize = 2;
pub const DEFAULT_MIN_PIXELS: usize = 256 * 28 * 28; // 200,704
pub const DEFAULT_MAX_PIXELS: usize = 16384 * 14 * 14; // 3,211,264 (in_patch_limit=16384)

/// Python-compatible rounding (banker's rounding / round half to even).
#[inline]
fn round_half_to_even(x: f64) -> f64 {
    let rounded = x.round();
    if (x - x.floor() - 0.5).abs() < 1e-9 && rounded as i64 % 2 != 0 {
        return rounded - 1.0;
    }
    rounded
}

#[derive(Debug, Clone)]
pub struct KimiK25Processor {
    patch_size: usize,
    merge_size: usize,
    min_pixels: usize,
    max_pixels: usize,
}

impl Default for KimiK25Processor {
    fn default() -> Self {
        Self::new()
    }
}

impl KimiK25Processor {
    pub fn new() -> Self {
        Self {
            patch_size: DEFAULT_PATCH_SIZE,
            merge_size: DEFAULT_MERGE_SIZE,
            min_pixels: DEFAULT_MIN_PIXELS,
            max_pixels: DEFAULT_MAX_PIXELS,
        }
    }

    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        Self {
            patch_size: config.get_patch_size(DEFAULT_PATCH_SIZE),
            merge_size: config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE),
            min_pixels: config.min_pixels.unwrap_or(DEFAULT_MIN_PIXELS),
            max_pixels: config.max_pixels.unwrap_or(DEFAULT_MAX_PIXELS),
        }
    }

    pub fn patch_size(&self) -> usize {
        self.patch_size
    }

    pub fn merge_size(&self) -> usize {
        self.merge_size
    }

    /// factor = patch_size * merge_size — dimensions must be divisible by this.
    #[inline]
    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    /// NaViT-style smart resize: fit within min/max pixel bounds while
    /// preserving aspect ratio and aligning to factor boundary.
    pub fn smart_resize(
        &self,
        height: usize,
        width: usize,
    ) -> Result<(usize, usize), TransformError> {
        let factor = self.factor();

        if height < factor || width < factor {
            return Err(TransformError::InvalidShape {
                expected: format!("dimensions >= {factor} (patch_size * merge_size)"),
                actual: vec![height, width],
            });
        }

        let max_dim = height.max(width) as f64;
        let min_dim = height.min(width) as f64;
        if max_dim / min_dim > 200.0 {
            return Err(TransformError::InvalidShape {
                expected: "aspect ratio < 200:1".to_string(),
                actual: vec![height, width],
            });
        }

        let mut h_bar = round_half_to_even(height as f64 / factor as f64) as usize * factor;
        let mut w_bar = round_half_to_even(width as f64 / factor as f64) as usize * factor;
        h_bar = h_bar.max(factor);
        w_bar = w_bar.max(factor);

        if h_bar * w_bar > self.max_pixels {
            let beta = ((height * width) as f64 / self.max_pixels as f64).sqrt();
            h_bar = ((height as f64 / beta / factor as f64).floor() as usize) * factor;
            w_bar = ((width as f64 / beta / factor as f64).floor() as usize) * factor;
            h_bar = h_bar.max(factor);
            w_bar = w_bar.max(factor);
        } else if h_bar * w_bar < self.min_pixels {
            let beta = (self.min_pixels as f64 / (height * width) as f64).sqrt();
            h_bar = ((height as f64 * beta / factor as f64).ceil() as usize) * factor;
            w_bar = ((width as f64 * beta / factor as f64).ceil() as usize) * factor;
        }

        Ok((h_bar, w_bar))
    }

    /// Extract [C, patch_size, patch_size] patches from a [C, H, W] tensor.
    ///
    /// Returns flattened `Vec<f32>` with layout `[num_patches, C * patch_size * patch_size]`.
    /// The engine reconstructs this as `[num_patches, C, patch_size, patch_size]`.
    fn extract_patches(
        tensor: &Array3<f32>,
        patch_size: usize,
    ) -> Result<Vec<f32>, TransformError> {
        let channels = tensor.shape()[0];
        let height = tensor.shape()[1];
        let width = tensor.shape()[2];

        if !height.is_multiple_of(patch_size) || !width.is_multiple_of(patch_size) {
            return Err(TransformError::ShapeError(format!(
                "Image dimensions [{height}, {width}] not divisible by patch_size {patch_size}"
            )));
        }

        let grid_h = height / patch_size;
        let grid_w = width / patch_size;
        let num_patches = grid_h * grid_w;
        let patch_features = channels * patch_size * patch_size;

        let mut patches = Vec::with_capacity(num_patches * patch_features);

        for gh in 0..grid_h {
            for gw in 0..grid_w {
                let h_start = gh * patch_size;
                let w_start = gw * patch_size;
                for c in 0..channels {
                    for ph in 0..patch_size {
                        for pw in 0..patch_size {
                            patches.push(tensor[[c, h_start + ph, w_start + pw]]);
                        }
                    }
                }
            }
        }

        Ok(patches)
    }
}

impl ImagePreProcessor for KimiK25Processor {
    fn default_mean(&self) -> [f64; 3] {
        KIMI_K25_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        KIMI_K25_STD
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedImages, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        let image_sizes: Vec<(u32, u32)> = images.iter().map(|img| img.dimensions()).collect();
        let mean = config.get_image_mean();
        let std = config.get_image_std();
        let filter = pil_to_filter(config.resampling);

        // Kimi patches are [C, patch_size, patch_size] — no temporal flattening
        let patch_features = 3 * self.patch_size * self.patch_size;

        let mut all_patches: Vec<f32> = Vec::new();
        let mut patches_per_image: Vec<i64> = Vec::with_capacity(images.len());
        let mut grid_thw_data = Vec::with_capacity(images.len() * 3);
        let mut num_img_tokens = Vec::with_capacity(images.len());

        for image in images {
            let (w, h) = image.dimensions();
            let (target_h, target_w) = self.smart_resize(h as usize, w as usize)?;

            let resized = if config.do_resize.unwrap_or(true) {
                resize(image, target_w as u32, target_h as u32, filter)
            } else {
                image.clone()
            };

            let mut tensor = to_tensor(&resized);
            if config.do_normalize.unwrap_or(true) {
                normalize(&mut tensor, &mean, &std);
            }

            let grid_h = target_h / self.patch_size;
            let grid_w = target_w / self.patch_size;
            let grid_t = 1usize; // images always have temporal=1

            grid_thw_data.push(grid_t as i64);
            grid_thw_data.push(grid_h as i64);
            grid_thw_data.push(grid_w as i64);

            let num_patches = grid_h * grid_w;
            // Token count after merge: (grid_t * grid_h * grid_w) / merge_size²
            let tokens = (grid_t * grid_h * grid_w) / (self.merge_size * self.merge_size);
            num_img_tokens.push(tokens);

            let patches = Self::extract_patches(&tensor, self.patch_size)?;
            all_patches.extend(patches);
            patches_per_image.push(num_patches as i64);
        }

        let total_patches: usize = patches_per_image.iter().map(|&n| n as usize).sum();
        let pixel_values = Array2::from_shape_vec((total_patches, patch_features), all_patches)
            .map_err(|e| {
                TransformError::ShapeError(format!(
                    "Failed to create pixel_values [{total_patches}, {patch_features}]: {e}"
                ))
            })?;

        let result =
            PreprocessedImages::new_dynamic(pixel_values.into_dyn(), num_img_tokens, image_sizes)
                .with_extra(
                    "grid_thws",
                    ModelSpecificValue::int_2d(grid_thw_data, images.len(), 3),
                )
                .with_extra(
                    "patches_per_image",
                    ModelSpecificValue::int_1d(patches_per_image),
                );

        Ok(result)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, _config: &PreProcessorConfig) -> usize {
        let (new_height, new_width) = match self.smart_resize(height as usize, width as usize) {
            Ok((h, w)) => (h, w),
            Err(_) => {
                let factor = self.factor();
                (factor, factor)
            }
        };
        let grid_h = new_height / self.patch_size;
        let grid_w = new_width / self.patch_size;
        (grid_h * grid_w) / (self.merge_size * self.merge_size)
    }

    fn model_name(&self) -> &'static str {
        "kimi-k2.5"
    }

    fn get_processed_size(&self, _config: &PreProcessorConfig) -> Option<(u32, u32)> {
        None // dynamic sizing
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;
    use crate::vision::preprocessor_config::PatchSize;

    fn create_test_image(width: u32, height: u32, color: Rgb<u8>) -> DynamicImage {
        DynamicImage::from(RgbImage::from_pixel(width, height, color))
    }

    #[test]
    fn test_kimi_k25_processor_default() {
        let processor = KimiK25Processor::new();
        assert_eq!(processor.patch_size(), 14);
        assert_eq!(processor.merge_size(), 2);
        assert_eq!(processor.factor(), 28);
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
    fn test_smart_resize_within_bounds() {
        let processor = KimiK25Processor::new();
        let (h, w) = processor.smart_resize(500, 500).unwrap();
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
        assert!(h * w >= processor.min_pixels);
        assert!(h * w <= processor.max_pixels);
    }

    #[test]
    fn test_smart_resize_extreme_aspect_ratio() {
        let processor = KimiK25Processor::new();
        assert!(processor.smart_resize(100, 30000).is_err());
    }

    #[test]
    fn test_preprocess_patch_shape() {
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

        // Kimi: [total_patches, 3 * 14 * 14] = [total_patches, 588]
        assert_eq!(result.pixel_values.ndim(), 2);
        assert_eq!(result.pixel_values.shape()[1], 3 * 14 * 14);
        assert!(result.pixel_values.shape()[0] > 0);

        assert!(result.model_specific.contains_key("grid_thws"));
        assert!(result.model_specific.contains_key("patches_per_image"));
        assert!(result.num_img_tokens[0] > 0);
    }

    #[test]
    fn test_preprocess_multiple_images() {
        let processor = KimiK25Processor::new();
        let config = PreProcessorConfig::default();

        let images = vec![
            create_test_image(600, 400, Rgb([100, 100, 100])),
            create_test_image(400, 600, Rgb([150, 150, 150])),
        ];

        let result = processor.preprocess(&images, &config).unwrap();

        assert_eq!(result.image_sizes.len(), 2);
        assert_eq!(result.num_img_tokens.len(), 2);
        assert_eq!(result.pixel_values.shape()[1], 588);

        if let Some(ModelSpecificValue::IntTensor { data, shape }) =
            result.model_specific.get("grid_thws")
        {
            assert_eq!(shape, &[2, 3]);
            assert_eq!(data.len(), 6);
        } else {
            panic!("Expected grid_thws to be IntTensor");
        }

        // patches_per_image sum matches total_patches
        if let Some(ModelSpecificValue::IntTensor { data, .. }) =
            result.model_specific.get("patches_per_image")
        {
            let total: i64 = data.iter().sum();
            assert_eq!(total as usize, result.pixel_values.shape()[0]);
        }
    }

    #[test]
    fn test_calculate_num_tokens() {
        let processor = KimiK25Processor::new();
        let config = PreProcessorConfig::default();
        // 448x448 → grid 32x32 → 1024 patches → 1024/4 = 256 tokens
        let tokens = processor.calculate_num_tokens(448, 448, &config);
        assert_eq!(tokens, 256);
    }

    #[test]
    fn test_from_preprocessor_config() {
        let config = PreProcessorConfig {
            patch_size: Some(PatchSize {
                height: Some(14),
                width: Some(14),
            }),
            merge_size: Some(2),
            min_pixels: Some(100000),
            max_pixels: Some(500000),
            ..Default::default()
        };
        let processor = KimiK25Processor::from_preprocessor_config(&config);
        assert_eq!(processor.patch_size(), 14);
        assert_eq!(processor.merge_size(), 2);
        assert_eq!(processor.min_pixels, 100000);
        assert_eq!(processor.max_pixels, 500000);
    }
}
