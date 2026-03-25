//! Shared base implementation for Qwen VL family image processors.
//!
//! This module provides a generic processor that handles the common logic
//! for Qwen2-VL, Qwen2.5-VL, and Qwen3-VL models. The specific variants
//! differ only in their default parameters (patch_size, normalization values).
//!
//! # Processing Pipeline
//!
//! 1. Validate aspect ratio (must be < 200:1)
//! 2. Smart resize to fit within min/max pixel bounds
//! 3. Align dimensions to (patch_size * merge_size) boundary
//! 4. Convert to tensor and normalize
//! 5. Reshape into patches for the vision encoder
//!
//! # Token Calculation
//!
//! ```text
//! grid_t = 1  (for images, temporal dimension is 1)
//! grid_h = resized_height / patch_size
//! grid_w = resized_width / patch_size
//! num_tokens = (grid_t * grid_h * grid_w) / merge_size²
//! ```

use image::{DynamicImage, GenericImageView};
use ndarray::{Array2, Array3};

use crate::vision::{
    image_processor::{ImagePreProcessor, ModelSpecificValue, PreprocessedImages},
    preprocessor_config::PreProcessorConfig,
    transforms::{pil_to_filter, resize, to_tensor, to_tensor_and_normalize, TransformError},
};

/// Python-compatible rounding (banker's rounding / round half to even).
///
/// This matches Python's `round()` behavior where 0.5 is rounded to the nearest
/// even number, unlike Rust's `f64::round()` which rounds half away from zero.
///
/// Examples:
/// - round_half_to_even(12.5) = 12 (not 13)
/// - round_half_to_even(13.5) = 14 (not 14)
/// - round_half_to_even(12.4) = 12
/// - round_half_to_even(12.6) = 13
#[inline]
fn round_half_to_even(x: f64) -> f64 {
    let rounded = x.round();
    // Check if we're exactly at a .5 case
    if (x - x.floor() - 0.5).abs() < 1e-9 {
        // Round to nearest even
        if rounded as i64 % 2 != 0 {
            return rounded - 1.0;
        }
    }
    rounded
}

/// Configuration for a Qwen VL processor variant.
#[derive(Debug, Clone)]
pub struct QwenVLConfig {
    /// Vision encoder patch size
    pub patch_size: usize,
    /// Merge size for token reduction
    pub merge_size: usize,
    /// Minimum total pixels allowed
    pub min_pixels: usize,
    /// Maximum total pixels allowed
    pub max_pixels: usize,
    /// Temporal patch size for video
    pub temporal_patch_size: usize,
    /// Normalization mean values
    pub mean: [f64; 3],
    /// Normalization std values
    pub std: [f64; 3],
    /// Model name for identification
    pub model_name: &'static str,
}

/// Generic Qwen VL image processor.
///
/// This struct implements the shared preprocessing logic for all Qwen VL
/// model variants. Each variant (Qwen2-VL, Qwen3-VL, etc.) uses this with
/// different configuration values.
#[derive(Debug, Clone)]
pub struct QwenVLProcessorBase {
    config: QwenVLConfig,
}

impl QwenVLProcessorBase {
    /// Create a new processor with the given configuration.
    pub fn new(config: QwenVLConfig) -> Self {
        Self { config }
    }

    /// Get the patch size.
    pub fn patch_size(&self) -> usize {
        self.config.patch_size
    }

    /// Get the merge size.
    pub fn merge_size(&self) -> usize {
        self.config.merge_size
    }

    /// Get the minimum pixels.
    pub fn min_pixels(&self) -> usize {
        self.config.min_pixels
    }

    /// Get the maximum pixels.
    pub fn max_pixels(&self) -> usize {
        self.config.max_pixels
    }

    /// Get the temporal patch size.
    pub fn temporal_patch_size(&self) -> usize {
        self.config.temporal_patch_size
    }

    /// Get the factor for dimension alignment.
    ///
    /// Dimensions must be divisible by (patch_size * merge_size).
    #[inline]
    pub fn get_factor(&self) -> usize {
        self.config.patch_size * self.config.merge_size
    }

    /// Smart resize algorithm for Qwen VL models.
    ///
    /// Resizes image dimensions to fit within min/max pixel bounds while:
    /// - Preserving aspect ratio
    /// - Aligning to (patch_size * merge_size) boundaries
    ///
    /// # Arguments
    /// * `height` - Original image height
    /// * `width` - Original image width
    ///
    /// # Returns
    /// (new_height, new_width) or error if aspect ratio is too extreme
    ///
    /// # Errors
    /// - If height or width is zero
    /// - If aspect ratio exceeds 200:1
    pub fn smart_resize(
        &self,
        height: usize,
        width: usize,
    ) -> Result<(usize, usize), TransformError> {
        let factor = self.get_factor();

        // Validate non-zero dimensions
        if height == 0 || width == 0 {
            return Err(TransformError::InvalidShape {
                expected: "non-zero dimensions".to_string(),
                actual: vec![height, width],
            });
        }

        // Validate aspect ratio
        let max_dim = height.max(width) as f64;
        let min_dim = height.min(width) as f64;
        let aspect_ratio = max_dim / min_dim;
        if aspect_ratio > 200.0 {
            return Err(TransformError::InvalidShape {
                expected: "aspect ratio < 200:1".to_string(),
                actual: vec![height, width],
            });
        }

        // Round to nearest factor multiple using Python-compatible rounding
        // Python uses banker's rounding (round half to even), which affects
        // edge cases like 400/32 = 12.5 -> 12 (not 13)
        let mut h_bar = round_half_to_even(height as f64 / factor as f64) as usize * factor;
        let mut w_bar = round_half_to_even(width as f64 / factor as f64) as usize * factor;

        // Ensure minimum size
        h_bar = h_bar.max(factor);
        w_bar = w_bar.max(factor);

        // Scale down if exceeding max_pixels
        if h_bar * w_bar > self.config.max_pixels {
            let beta = ((height * width) as f64 / self.config.max_pixels as f64).sqrt();
            h_bar = ((height as f64 / beta / factor as f64).floor() as usize) * factor;
            w_bar = ((width as f64 / beta / factor as f64).floor() as usize) * factor;
            // Ensure minimum size after scaling down
            h_bar = h_bar.max(factor);
            w_bar = w_bar.max(factor);
        }
        // Scale up if below min_pixels
        else if h_bar * w_bar < self.config.min_pixels {
            let beta = (self.config.min_pixels as f64 / (height * width) as f64).sqrt();
            h_bar = ((height as f64 * beta / factor as f64).ceil() as usize) * factor;
            w_bar = ((width as f64 * beta / factor as f64).ceil() as usize) * factor;
        }

        Ok((h_bar, w_bar))
    }

    /// Calculate the grid dimensions (T, H, W) for an image.
    ///
    /// For single images, T=1. For video, T = num_frames / temporal_patch_size.
    ///
    /// # Arguments
    /// * `height` - Resized image height
    /// * `width` - Resized image width
    /// * `num_frames` - Number of frames (1 for images)
    ///
    /// # Returns
    /// (grid_t, grid_h, grid_w)
    pub fn calculate_grid_thw(
        &self,
        height: usize,
        width: usize,
        num_frames: usize,
    ) -> (usize, usize, usize) {
        let grid_t =
            num_frames.max(self.config.temporal_patch_size) / self.config.temporal_patch_size;
        let grid_h = height / self.config.patch_size;
        let grid_w = width / self.config.patch_size;
        (grid_t, grid_h, grid_w)
    }

    /// Calculate the number of image tokens after merge.
    ///
    /// tokens = (grid_t * grid_h * grid_w) / merge_size²
    pub fn calculate_tokens_from_grid(&self, grid_t: usize, grid_h: usize, grid_w: usize) -> usize {
        (grid_t * grid_h * grid_w) / (self.config.merge_size * self.config.merge_size)
    }

    /// Patchify tensor directly into an output buffer (avoids intermediate Vec allocation).
    #[expect(
        clippy::expect_used,
        reason = "as_standard_layout guarantees contiguous memory"
    )]
    pub fn patchify_into(
        &self,
        tensor: &Array3<f32>,
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
        output: &mut Vec<f32>,
    ) {
        let channel = tensor.shape()[0];
        let height = tensor.shape()[1];
        let width = tensor.shape()[2];
        let patch_size = self.config.patch_size;
        let merge_size = self.config.merge_size;
        let temporal_patch_size = self.config.temporal_patch_size;

        debug_assert_eq!(height, grid_h * patch_size);
        debug_assert_eq!(width, grid_w * patch_size);

        let grid_h_m = grid_h / merge_size;
        let grid_w_m = grid_w / merge_size;
        let num_patches = grid_t * grid_h * grid_w;
        let patch_features = channel * temporal_patch_size * patch_size * patch_size;

        output.reserve(num_patches * patch_features);
        let base_idx = output.len();
        output.resize(base_idx + num_patches * patch_features, 0.0);

        let data = tensor.as_standard_layout();
        let flat = data
            .as_slice()
            .expect("standard layout tensor must be contiguous");
        let planes: Vec<&[f32]> = (0..channel)
            .map(|c| &flat[c * height * width..(c + 1) * height * width])
            .collect();

        let mut out_idx = base_idx;
        for _gt in 0..grid_t {
            for gh_m in 0..grid_h_m {
                for gw_m in 0..grid_w_m {
                    for mh in 0..merge_size {
                        for mw in 0..merge_size {
                            let y_base = (gh_m * merge_size + mh) * patch_size;
                            let x_base = (gw_m * merge_size + mw) * patch_size;
                            for plane in &planes {
                                for _tp in 0..temporal_patch_size {
                                    for ph in 0..patch_size {
                                        let row_start = (y_base + ph) * width + x_base;
                                        output[out_idx..out_idx + patch_size].copy_from_slice(
                                            &plane[row_start..row_start + patch_size],
                                        );
                                        out_idx += patch_size;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[expect(
        clippy::expect_used,
        reason = "as_standard_layout guarantees contiguous memory"
    )]
    pub fn reshape_to_patches(
        &self,
        tensor: &Array3<f32>,
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
    ) -> Vec<f32> {
        let channel = tensor.shape()[0];
        let height = tensor.shape()[1];
        let width = tensor.shape()[2];

        let patch_size = self.config.patch_size;
        let merge_size = self.config.merge_size;
        let temporal_patch_size = self.config.temporal_patch_size;

        debug_assert_eq!(height, grid_h * patch_size);
        debug_assert_eq!(width, grid_w * patch_size);

        // Direct index computation: write output in the permuted order without
        // materializing intermediate arrays. This replaces the 9D reshape +
        // permute + as_standard_layout() chain which caused two full copies.
        //
        // Output layout (flattened):
        //   [grid_t, grid_h/merge, grid_w/merge, merge, merge, C, temporal, patch_h, patch_w]
        // For images: grid_t=1, temporal=1, so the outer two loops are trivial.

        let grid_h_m = grid_h / merge_size;
        let grid_w_m = grid_w / merge_size;
        let num_patches = grid_t * grid_h * grid_w;
        let patch_features = channel * temporal_patch_size * patch_size * patch_size;

        let mut output = vec![0.0f32; num_patches * patch_features];
        // tensor is [C, H, W] in row-major order.
        let data = tensor.as_standard_layout();
        let flat = data
            .as_slice()
            .expect("standard layout tensor must be contiguous");

        // Pre-compute channel plane pointers for fast access
        let planes: Vec<&[f32]> = (0..channel)
            .map(|c| &flat[c * height * width..(c + 1) * height * width])
            .collect();

        let mut out_idx = 0;
        for _gt in 0..grid_t {
            for gh_m in 0..grid_h_m {
                for gw_m in 0..grid_w_m {
                    for mh in 0..merge_size {
                        for mw in 0..merge_size {
                            let y_base = (gh_m * merge_size + mh) * patch_size;
                            let x_base = (gw_m * merge_size + mw) * patch_size;
                            for plane in &planes {
                                for _tp in 0..temporal_patch_size {
                                    for ph in 0..patch_size {
                                        let row_start = (y_base + ph) * width + x_base;
                                        output[out_idx..out_idx + patch_size].copy_from_slice(
                                            &plane[row_start..row_start + patch_size],
                                        );
                                        out_idx += patch_size;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        output
    }
}

impl ImagePreProcessor for QwenVLProcessorBase {
    fn default_mean(&self) -> [f64; 3] {
        self.config.mean
    }

    fn default_std(&self) -> [f64; 3] {
        self.config.std
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedImages, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        // Store original sizes
        let image_sizes: Vec<(u32, u32)> = images.iter().map(|img| img.dimensions()).collect();

        let mean = config.get_image_mean();
        let std = config.get_image_std();
        let filter = pil_to_filter(config.resampling);

        let patch_size = self.config.patch_size;
        let temporal_patch_size = self.config.temporal_patch_size;
        let patch_features = 3 * temporal_patch_size * patch_size * patch_size;

        // Pre-allocate based on total pixel count to avoid repeated Vec growth
        let estimated_total: usize = images
            .iter()
            .map(|img| {
                let (w, h) = img.dimensions();
                (w as usize * h as usize) / (self.config.merge_size * self.config.merge_size)
                    * patch_features
                    / (patch_size * patch_size)
            })
            .sum();
        let mut all_patches: Vec<f32> = Vec::with_capacity(estimated_total);
        let mut patches_per_image: Vec<i64> = Vec::with_capacity(images.len());
        let mut grid_thw_data = Vec::with_capacity(images.len() * 3);
        let mut num_img_tokens = Vec::with_capacity(images.len());

        for image in images {
            let (w, h) = image.dimensions();
            let (target_h, target_w) = self.smart_resize(h as usize, w as usize)?;

            // Resize to the image's own target size (skip if dimensions match)
            let (tw32, th32) = (target_w as u32, target_h as u32);
            let needs_resize = config.do_resize.unwrap_or(true) && (w != tw32 || h != th32);
            let resized;
            let img_ref = if needs_resize {
                resized = resize(image, tw32, th32, filter);
                &resized
            } else {
                image
            };

            // Grid dimensions based on the target size
            let (grid_t, grid_h, grid_w) = self.calculate_grid_thw(target_h, target_w, 1);
            grid_thw_data.push(grid_t as i64);
            grid_thw_data.push(grid_h as i64);
            grid_thw_data.push(grid_w as i64);

            let num_patches = grid_t * grid_h * grid_w;
            let tokens = self.calculate_tokens_from_grid(grid_t, grid_h, grid_w);
            num_img_tokens.push(tokens);

            // Convert to tensor [C, H, W] and normalize in one fused pass
            let tensor = if config.do_normalize.unwrap_or(true) {
                to_tensor_and_normalize(img_ref, &mean, &std)
            } else {
                to_tensor(img_ref)
            };

            // Patchify directly into all_patches to avoid intermediate Vec + copy
            self.patchify_into(&tensor, grid_t, grid_h, grid_w, &mut all_patches);
            patches_per_image.push(num_patches as i64);
        }

        let total_patches: usize = patches_per_image.iter().map(|&n| n as usize).sum();
        let pixel_values =
            Array2::from_shape_vec((total_patches, patch_features), all_patches).map_err(|e| {
                TransformError::ShapeError(format!(
                    "Failed to create patchified pixel_values [{total_patches}, {patch_features}]: {e}"
                ))
            })?;

        let result =
            PreprocessedImages::new_dynamic(pixel_values.into_dyn(), num_img_tokens, image_sizes)
                .with_extra(
                    "image_grid_thw",
                    ModelSpecificValue::int_2d(grid_thw_data, images.len(), 3),
                )
                .with_extra(
                    "patches_per_image",
                    ModelSpecificValue::int_1d(patches_per_image),
                );

        Ok(result)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, _config: &PreProcessorConfig) -> usize {
        // Calculate resized dimensions
        let (new_height, new_width) = match self.smart_resize(height as usize, width as usize) {
            Ok((h, w)) => (h, w),
            Err(_) => {
                // Fallback: use minimum size
                let factor = self.get_factor();
                (factor, factor)
            }
        };

        // Calculate grid and tokens
        let (grid_t, grid_h, grid_w) = self.calculate_grid_thw(new_height, new_width, 1);
        self.calculate_tokens_from_grid(grid_t, grid_h, grid_w)
    }

    fn model_name(&self) -> &'static str {
        self.config.model_name
    }

    fn get_processed_size(&self, _config: &PreProcessorConfig) -> Option<(u32, u32)> {
        // Qwen VL models have dynamic sizing, no fixed output size
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> QwenVLConfig {
        QwenVLConfig {
            patch_size: 14,
            merge_size: 2,
            min_pixels: 256 * 28 * 28,
            max_pixels: 1280 * 28 * 28,
            temporal_patch_size: 2,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            model_name: "test-qwen-vl",
        }
    }

    #[test]
    fn test_qwen_vl_base_factor() {
        let processor = QwenVLProcessorBase::new(create_test_config());
        assert_eq!(processor.get_factor(), 28); // 14 * 2
    }

    #[test]
    fn test_smart_resize_within_bounds() {
        let processor = QwenVLProcessorBase::new(create_test_config());
        let (h, w) = processor.smart_resize(500, 500).unwrap();

        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
        assert!(h * w >= processor.min_pixels());
        assert!(h * w <= processor.max_pixels());
    }

    #[test]
    fn test_smart_resize_extreme_aspect_ratio_error() {
        let processor = QwenVLProcessorBase::new(create_test_config());
        let result = processor.smart_resize(100, 30000);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_grid_thw() {
        let processor = QwenVLProcessorBase::new(create_test_config());
        let (t, h, w) = processor.calculate_grid_thw(448, 448, 1);

        assert_eq!(t, 1);
        assert_eq!(h, 448 / 14);
        assert_eq!(w, 448 / 14);
    }

    #[test]
    fn test_calculate_tokens() {
        let processor = QwenVLProcessorBase::new(create_test_config());
        let tokens = processor.calculate_tokens_from_grid(1, 32, 32);
        assert_eq!(tokens, (32 * 32) / 4);
    }
}
