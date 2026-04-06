//! Kimi-K2.5 (MoonViT) image processor.
//!
//! Matches the HuggingFace `KimiK25VisionProcessor` preprocessing pipeline:
//!
//! 1. Compute scale to fit within patch limits (never upscale)
//! 2. Resize with BICUBIC interpolation
//! 3. Zero-pad to make dimensions divisible by factor (patch_size * merge_size)
//! 4. Normalize with [0.5, 0.5, 0.5] mean/std
//! 5. Extract patches as [N, C, patch_size, patch_size]
//!
//! Kimi resizes then zero-pads to make dimensions divisible by the alignment
//! factor (patch_size * merge_size). The model was trained with zero-padded
//! images, so using direct resize-to-aligned would degrade image quality.

use image::{DynamicImage, GenericImageView};
use ndarray::Array3;

use crate::vision::{
    image_processor::{ImagePreProcessor, ModelSpecificValue, PreprocessedImages},
    preprocessor_config::PreProcessorConfig,
    transforms::{self, TransformError},
};

pub const KIMI_K25_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
pub const KIMI_K25_STD: [f64; 3] = [0.5, 0.5, 0.5];

pub const DEFAULT_PATCH_SIZE: usize = 14;
pub const DEFAULT_MERGE_SIZE: usize = 2;
/// Maximum total patches before merge (from preprocessor_config.json in_patch_limit)
pub const DEFAULT_IN_PATCH_LIMIT: usize = 16384;
/// Maximum patches along one spatial dimension
pub const DEFAULT_PATCH_LIMIT_ON_ONE_SIDE: usize = 512;

/// Kimi-K2.5 resize configuration for a single image.
struct ResizeConfig {
    new_width: usize,
    new_height: usize,
    pad_width: usize,
    pad_height: usize,
    num_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct KimiK25Processor {
    patch_size: usize,
    merge_size: usize,
    in_patch_limit: usize,
    patch_limit_on_one_side: usize,
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
            in_patch_limit: DEFAULT_IN_PATCH_LIMIT,
            patch_limit_on_one_side: DEFAULT_PATCH_LIMIT_ON_ONE_SIDE,
        }
    }

    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        Self {
            patch_size: config.get_patch_size(DEFAULT_PATCH_SIZE),
            merge_size: config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE),
            in_patch_limit: config
                .get_extra::<usize>("in_patch_limit")
                .unwrap_or(DEFAULT_IN_PATCH_LIMIT),
            patch_limit_on_one_side: config
                .get_extra::<usize>("patch_limit_on_one_side")
                .unwrap_or(DEFAULT_PATCH_LIMIT_ON_ONE_SIDE),
        }
    }

    pub fn patch_size(&self) -> usize {
        self.patch_size
    }

    pub fn merge_size(&self) -> usize {
        self.merge_size
    }

    #[inline]
    fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    /// Compute resize dimensions and padding, matching HF `navit_resize_image`.
    ///
    /// Never upscales (scale capped at 1.0). Pads with zeros to align to factor.
    fn compute_resize_config(&self, width: usize, height: usize) -> ResizeConfig {
        let ps = self.patch_size;
        let patches_w = (width / ps).max(1) as f64;
        let patches_h = (height / ps).max(1) as f64;

        let s1 = (self.in_patch_limit as f64 / (patches_w * patches_h)).sqrt();
        let s2 = (self.patch_limit_on_one_side * ps) as f64 / width as f64;
        let s3 = (self.patch_limit_on_one_side * ps) as f64 / height as f64;
        let scale = f64::min(1.0, f64::min(s1, f64::min(s2, s3)));

        let new_w = ((width as f64 * scale) as usize).max(1);
        let new_h = ((height as f64 * scale) as usize).max(1);
        let new_w = new_w.min(self.patch_limit_on_one_side * ps);
        let new_h = new_h.min(self.patch_limit_on_one_side * ps);

        let factor = self.factor();
        let pad_width = (factor - new_w % factor) % factor;
        let pad_height = (factor - new_h % factor) % factor;

        let token_height = (new_h + pad_height) / factor;
        let token_width = (new_w + pad_width) / factor;
        let num_tokens = token_height * token_width;

        ResizeConfig {
            new_width: new_w,
            new_height: new_h,
            pad_width,
            pad_height,
            num_tokens,
        }
    }

    /// Fused resize + zero-pad + normalize into a single [C, H_padded, W_padded] tensor.
    ///
    /// Avoids intermediate allocations by:
    /// 1. Allocating the final padded canvas directly
    /// 2. Pre-filling with normalized black (bias value)
    /// 3. Deinterleaving + normalizing the image region in one pass
    fn resize_pad_and_normalize(
        image: &DynamicImage,
        cfg: &ResizeConfig,
        mean: &[f64; 3],
        std: &[f64; 3],
    ) -> Array3<f32> {
        let canvas_h = cfg.new_height + cfg.pad_height;
        let canvas_w = cfg.new_width + cfg.pad_width;

        // Resize using SIMD-accelerated BICUBIC (fast_image_resize)
        let resized = transforms::resize(
            image,
            cfg.new_width as u32,
            cfg.new_height as u32,
            image::imageops::FilterType::CatmullRom,
        );

        let (img_w, img_h, raw) = transforms::rgb_bytes(&resized);
        let canvas_pixels = canvas_h * canvas_w;

        // Precompute fused scale/bias: pixel/255 → normalized
        // output[c][i] = raw[i*3+c] / 255.0 * (1/std[c]) + (-mean[c]/std[c])
        let scale: [f32; 3] = std::array::from_fn(|c| 1.0 / (255.0 * std[c] as f32));
        let bias: [f32; 3] = std::array::from_fn(|c| -(mean[c] as f32) / (std[c] as f32));

        let mut data = vec![0.0f32; 3 * canvas_pixels];
        let (r_plane, rest) = data.split_at_mut(canvas_pixels);
        let (g_plane, b_plane) = rest.split_at_mut(canvas_pixels);

        // Pre-fill with normalized black: (0/255 - mean) / std = bias
        r_plane.fill(bias[0]);
        g_plane.fill(bias[1]);
        b_plane.fill(bias[2]);

        // Overwrite image region row-by-row using vectorized deinterleave
        let rw = img_w.min(canvas_w);
        let rh = img_h.min(canvas_h);
        for y in 0..rh {
            let src_row = &raw[y * img_w * 3..y * img_w * 3 + rw * 3];
            let dst_offset = y * canvas_w;
            transforms::deinterleave_rgb_to_planes(
                src_row,
                &mut r_plane[dst_offset..dst_offset + rw],
                &mut g_plane[dst_offset..dst_offset + rw],
                &mut b_plane[dst_offset..dst_offset + rw],
                scale,
                bias,
            );
        }

        #[expect(
            clippy::expect_used,
            reason = "data has exactly 3*canvas_h*canvas_w elements by construction"
        )]
        Array3::from_shape_vec((3, canvas_h, canvas_w), data)
            .expect("shape matches pre-allocated buffer")
    }

    /// Extract [C, patch_size, patch_size] patches from a contiguous [C, H, W] tensor.
    ///
    /// Uses row-based `copy_from_slice` instead of per-element indexing so the
    /// compiler can auto-vectorize the inner copy.
    fn extract_patches(tensor: &Array3<f32>, patch_size: usize) -> Vec<f32> {
        let channels = tensor.shape()[0];
        let height = tensor.shape()[1];
        let width = tensor.shape()[2];

        let grid_h = height / patch_size;
        let grid_w = width / patch_size;
        let num_patches = grid_h * grid_w;
        let patch_features = channels * patch_size * patch_size;

        let mut patches = Vec::with_capacity(num_patches * patch_features);

        // Get contiguous slice for direct row addressing
        let flat = tensor.as_standard_layout();
        #[expect(
            clippy::expect_used,
            reason = "as_standard_layout guarantees contiguous C-order memory"
        )]
        let data = flat
            .as_slice()
            .expect("as_standard_layout guarantees contiguous memory");

        for gh in 0..grid_h {
            for gw in 0..grid_w {
                let h_start = gh * patch_size;
                let w_start = gw * patch_size;
                for c in 0..channels {
                    let plane_offset = c * height * width;
                    for ph in 0..patch_size {
                        let row_start = plane_offset + (h_start + ph) * width + w_start;
                        patches.extend_from_slice(&data[row_start..row_start + patch_size]);
                    }
                }
            }
        }

        patches
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

        let mut all_patches: Vec<f32> = Vec::new();
        let mut patches_per_image: Vec<i64> = Vec::with_capacity(images.len());
        let mut grid_thw_data = Vec::with_capacity(images.len() * 3);
        let mut num_img_tokens = Vec::with_capacity(images.len());

        for image in images {
            let (w, h) = image.dimensions();
            let cfg = self.compute_resize_config(w as usize, h as usize);

            // Fused resize + pad + normalize in one pass (avoids 2 extra allocations)
            let tensor = Self::resize_pad_and_normalize(image, &cfg, &mean, &std);

            let padded_h = cfg.new_height + cfg.pad_height;
            let padded_w = cfg.new_width + cfg.pad_width;
            let grid_h = padded_h / self.patch_size;
            let grid_w = padded_w / self.patch_size;
            let grid_t = 1usize;

            grid_thw_data.push(grid_t as i64);
            grid_thw_data.push(grid_h as i64);
            grid_thw_data.push(grid_w as i64);

            let num_patches = grid_h * grid_w;
            num_img_tokens.push(cfg.num_tokens);

            let patches = Self::extract_patches(&tensor, self.patch_size);
            all_patches.extend(patches);
            patches_per_image.push(num_patches as i64);
        }

        let total_patches: usize = patches_per_image.iter().map(|&n| n as usize).sum();
        let pixel_values = ndarray::Array4::from_shape_vec(
            (total_patches, 3, self.patch_size, self.patch_size),
            all_patches,
        )
        .map_err(|e| {
            TransformError::ShapeError(format!(
                "Failed to create pixel_values [{total_patches}, 3, {}, {}]: {e}",
                self.patch_size, self.patch_size
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
        self.compute_resize_config(width as usize, height as usize)
            .num_tokens
    }

    fn model_name(&self) -> &'static str {
        "kimi-k2.5"
    }

    fn get_processed_size(&self, _config: &PreProcessorConfig) -> Option<(u32, u32)> {
        None
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
        assert_eq!(result.pixel_values.ndim(), 4);
        assert_eq!(result.pixel_values.shape()[1], 3);
        assert_eq!(result.pixel_values.shape()[2], 14);
        assert_eq!(result.pixel_values.shape()[3], 14);

        assert!(result.model_specific.contains_key("grid_thws"));
        assert!(result.model_specific.contains_key("patches_per_image"));
        assert!(result.num_img_tokens[0] > 0);
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

        assert_eq!(result.image_sizes.len(), 2);
        assert_eq!(result.num_img_tokens.len(), 2);
        assert_eq!(result.pixel_values.ndim(), 4);
        assert_eq!(result.pixel_values.shape()[1], 3);

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
            assert_eq!(total as usize, result.pixel_values.shape()[0]);
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

        let flat = result.pixel_values_flat();
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
        assert_eq!(result.pixel_values.ndim(), 4);
        assert!(result.pixel_values.shape()[0] > 0);
        assert!(result.num_img_tokens[0] > 0);
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
        assert_eq!(p.in_patch_limit, 8192);
        assert_eq!(p.patch_limit_on_one_side, 256);
    }
}
