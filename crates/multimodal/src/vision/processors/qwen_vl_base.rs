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

use std::borrow::Cow;

use image::{DynamicImage, GenericImageView};
use ndarray::{Array2, Array3};

use crate::{
    types::RgbFrameRef,
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{ModelSpecificValue, PreprocessedEncoderInputs, VisionPreProcessor},
        transforms::{
            pil_to_filter, resize, resize_rgb_bytes, rgb_bytes, to_tensor, to_tensor_and_normalize,
            TransformError,
        },
    },
};

/// Python-compatible rounding (banker's rounding / round half to even).
///
/// This matches Python's `round()` behavior where 0.5 is rounded to the nearest
/// even number (e.g. `12.5 -> 12`, `13.5 -> 14`), unlike Rust's `f64::round()`
/// which rounds half away from zero.
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

struct VideoFrameRgb<'a> {
    width: usize,
    height: usize,
    data: Cow<'a, [u8]>,
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

    /// Smart resize for Qwen3-style video processors.
    ///
    /// Unlike image resize, the pixel budget is applied to the full sampled
    /// video volume (`T * H * W`), matching HuggingFace's Qwen3 video
    /// processor.
    pub fn smart_resize_video(
        &self,
        num_frames: usize,
        height: usize,
        width: usize,
    ) -> Result<(usize, usize), TransformError> {
        let factor = self.get_factor();

        if num_frames == 0 {
            return Err(TransformError::InvalidShape {
                expected: "num_frames > 0".to_string(),
                actual: vec![num_frames],
            });
        }

        if height < factor || width < factor {
            return Err(TransformError::InvalidShape {
                expected: format!("height and width >= factor ({factor})"),
                actual: vec![height, width],
            });
        }

        let max_dim = height.max(width) as f64;
        let min_dim = height.min(width) as f64;
        let aspect_ratio = max_dim / min_dim;
        if aspect_ratio > 200.0 {
            return Err(TransformError::InvalidShape {
                expected: "aspect ratio < 200:1".to_string(),
                actual: vec![height, width],
            });
        }

        let mut h_bar = round_half_to_even(height as f64 / factor as f64) as usize * factor;
        let mut w_bar = round_half_to_even(width as f64 / factor as f64) as usize * factor;
        h_bar = h_bar.max(factor);
        w_bar = w_bar.max(factor);

        let t_bar =
            num_frames.div_ceil(self.config.temporal_patch_size) * self.config.temporal_patch_size;
        let resized_pixels = (t_bar * h_bar * w_bar) as f64;
        if resized_pixels > self.config.max_pixels as f64 {
            let beta = ((t_bar * height * width) as f64 / self.config.max_pixels as f64).sqrt();
            h_bar = ((height as f64 / beta / factor as f64).floor() as usize) * factor;
            w_bar = ((width as f64 / beta / factor as f64).floor() as usize) * factor;
            h_bar = h_bar.max(factor);
            w_bar = w_bar.max(factor);
        } else if resized_pixels < self.config.min_pixels as f64 {
            let beta = (self.config.min_pixels as f64 / (t_bar * height * width) as f64).sqrt();
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
    /// Patchify a [C, H, W] tensor and append the patches to `output`.
    ///
    /// Output layout per image:
    ///   `[grid_t, patch_rows, patch_cols, merge_h, merge_w, C, temporal, patch_h, patch_w]`
    ///
    /// Each "merged patch" covers a `(merge_size * patch_size)²` spatial region.
    /// Within it, `merge_size²` sub-patches are emitted, each containing all channels.
    pub fn patchify_into(
        &self,
        tensor: &Array3<f32>,
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
        output: &mut Vec<f32>,
    ) -> Result<(), TransformError> {
        let channel = tensor.shape()[0];
        let height = tensor.shape()[1];
        let width = tensor.shape()[2];
        let patch_size = self.config.patch_size;
        let merge_size = self.config.merge_size;
        let temporal_patch_size = self.config.temporal_patch_size;

        debug_assert_eq!(
            height,
            grid_h * patch_size,
            "Height must match grid_h * patch_size"
        );
        debug_assert_eq!(
            width,
            grid_w * patch_size,
            "Width must match grid_w * patch_size"
        );

        let num_patches = grid_t * grid_h * grid_w;
        let patch_features = channel * temporal_patch_size * patch_size * patch_size;
        let base_idx = output.len();
        output.resize(base_idx + num_patches * patch_features, 0.0);

        let data = tensor.as_standard_layout();
        let flat = data.as_slice().ok_or_else(|| {
            TransformError::ShapeError("tensor not contiguous after as_standard_layout".to_string())
        })?;
        let planes: Vec<&[f32]> = (0..channel)
            .map(|c| &flat[c * height * width..(c + 1) * height * width])
            .collect();

        let merged_patch = merge_size * patch_size;
        let mut out_idx = base_idx;

        for _gt in 0..grid_t {
            for pr in 0..grid_h / merge_size {
                for pc in 0..grid_w / merge_size {
                    let y0 = pr * merged_patch;
                    let x0 = pc * merged_patch;

                    for mh in 0..merge_size {
                        for mw in 0..merge_size {
                            for plane in &planes {
                                for _tp in 0..temporal_patch_size {
                                    for py in 0..patch_size {
                                        let row = (y0 + mh * patch_size + py) * width
                                            + x0
                                            + mw * patch_size;
                                        output[out_idx..out_idx + patch_size]
                                            .copy_from_slice(&plane[row..row + patch_size]);
                                        out_idx += patch_size;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Patchify a sequence of frame tensors into Qwen's video patch layout.
    ///
    /// `tensors` must already be resized and normalized to the same spatial
    /// shape. If the frame count is not divisible by `temporal_patch_size`,
    /// the caller should pad by repeating the final frame before calling this.
    pub fn patchify_video_into(
        &self,
        tensors: &[Array3<f32>],
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
        output: &mut Vec<f32>,
    ) -> Result<(), TransformError> {
        if tensors.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        let channel = tensors[0].shape()[0];
        let height = tensors[0].shape()[1];
        let width = tensors[0].shape()[2];
        let patch_size = self.config.patch_size;
        let merge_size = self.config.merge_size;
        let temporal_patch_size = self.config.temporal_patch_size;

        debug_assert_eq!(height, grid_h * patch_size);
        debug_assert_eq!(width, grid_w * patch_size);
        debug_assert_eq!(tensors.len(), grid_t * temporal_patch_size);

        let num_patches = grid_t * grid_h * grid_w;
        let patch_features = channel * temporal_patch_size * patch_size * patch_size;
        let base_idx = output.len();
        output.resize(base_idx + num_patches * patch_features, 0.0);

        let frame_planes: Vec<Vec<&[f32]>> = tensors
            .iter()
            .map(|tensor| {
                let flat = tensor.as_slice().ok_or_else(|| {
                    TransformError::ShapeError("video frame tensor is not contiguous".to_string())
                })?;
                Ok((0..channel)
                    .map(|c| &flat[c * height * width..(c + 1) * height * width])
                    .collect::<Vec<_>>())
            })
            .collect::<Result<_, TransformError>>()?;

        let merged_patch = merge_size * patch_size;
        let mut out_idx = base_idx;

        for gt in 0..grid_t {
            let frame_start = gt * temporal_patch_size;
            for pr in 0..grid_h / merge_size {
                for pc in 0..grid_w / merge_size {
                    let y0 = pr * merged_patch;
                    let x0 = pc * merged_patch;

                    for mh in 0..merge_size {
                        for mw in 0..merge_size {
                            let frame_window =
                                &frame_planes[frame_start..frame_start + temporal_patch_size];
                            for channel_frames in (0..channel).map(|channel_idx| {
                                frame_window.iter().map(move |planes| planes[channel_idx])
                            }) {
                                for plane in channel_frames {
                                    for py in 0..patch_size {
                                        let row = (y0 + mh * patch_size + py) * width
                                            + x0
                                            + mw * patch_size;
                                        output[out_idx..out_idx + patch_size]
                                            .copy_from_slice(&plane[row..row + patch_size]);
                                        out_idx += patch_size;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn patchify_video_rgb_chunk_into(
        &self,
        frames: &[VideoFrameRgb<'_>],
        grid_h: usize,
        grid_w: usize,
        output: &mut [f32],
        out_idx: &mut usize,
        lut: &[[f32; 256]; 3],
    ) -> Result<(), TransformError> {
        let patch_size = self.config.patch_size;
        let merge_size = self.config.merge_size;
        let temporal_patch_size = self.config.temporal_patch_size;
        if frames.len() != temporal_patch_size {
            return Err(TransformError::InvalidShape {
                expected: format!("{temporal_patch_size} video frames in temporal patch"),
                actual: vec![frames.len()],
            });
        }

        let height = grid_h * patch_size;
        let width = grid_w * patch_size;
        for frame in frames {
            if frame.height != height || frame.width != width {
                return Err(TransformError::InvalidShape {
                    expected: format!("video frame size {width}x{height}"),
                    actual: vec![frame.width, frame.height],
                });
            }
        }

        let merged_patch = merge_size * patch_size;
        for pr in 0..grid_h / merge_size {
            for pc in 0..grid_w / merge_size {
                let y0 = pr * merged_patch;
                let x0 = pc * merged_patch;

                for mh in 0..merge_size {
                    for mw in 0..merge_size {
                        for (c, lut_c) in lut.iter().enumerate().take(3) {
                            for frame in frames {
                                let raw = frame.data.as_ref();
                                for py in 0..patch_size {
                                    let row =
                                        (y0 + mh * patch_size + py) * width + x0 + mw * patch_size;
                                    let mut src_idx = row * 3 + c;
                                    let dst_end = *out_idx + patch_size;
                                    for dst in &mut output[*out_idx..dst_end] {
                                        *dst = lut_c[raw[src_idx] as usize];
                                        src_idx += 3;
                                    }
                                    *out_idx = dst_end;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl VisionPreProcessor for QwenVLProcessorBase {
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
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        // Store original sizes
        let item_sizes: Vec<(u32, u32)> = images.iter().map(|img| img.dimensions()).collect();

        let mean = config.get_image_mean();
        let std = config.get_image_std();
        // Qwen2VL/Qwen3VL image processors default to BICUBIC (PIL resample=3)
        // when the preprocessor config omits `resample`. The global pil_to_filter
        // fallback is bilinear, which yields smoother features and measurably
        // degrades VLM accuracy, so pin the HF-correct default here.
        let filter = pil_to_filter(config.resampling.or(Some(3)));

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
        let mut feature_token_counts = Vec::with_capacity(images.len());

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
            feature_token_counts.push(tokens);

            // Convert to tensor [C, H, W] and normalize in one fused pass
            let tensor = if config.do_normalize.unwrap_or(true) {
                to_tensor_and_normalize(img_ref, &mean, &std)
            } else {
                to_tensor(img_ref)
            };

            // Patchify directly into all_patches to avoid intermediate Vec + copy
            self.patchify_into(&tensor, grid_t, grid_h, grid_w, &mut all_patches)?;
            patches_per_image.push(num_patches as i64);
        }

        let total_patches: usize = patches_per_image.iter().map(|&n| n as usize).sum();
        let encoder_input =
            Array2::from_shape_vec((total_patches, patch_features), all_patches).map_err(|e| {
                TransformError::ShapeError(format!(
                    "Failed to create patchified encoder_input [{total_patches}, {patch_features}]: {e}"
                ))
            })?;

        let result = PreprocessedEncoderInputs::new_dynamic(
            encoder_input.into_dyn(),
            feature_token_counts,
            item_sizes,
        )
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

    fn preprocess_video(
        &self,
        frames: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if frames.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        let (w, h) = frames[0].dimensions();
        let item_sizes = vec![(w, h)];
        let mean = config.get_image_mean();
        let std = config.get_image_std();
        // Qwen2VL/Qwen3VL image processors default to BICUBIC (PIL resample=3)
        // when the preprocessor config omits `resample`. The global pil_to_filter
        // fallback is bilinear, which yields smoother features and measurably
        // degrades VLM accuracy, so pin the HF-correct default here.
        let filter = pil_to_filter(config.resampling.or(Some(3)));

        let temporal_patch_size = self.config.temporal_patch_size;
        let padded_frames = frames.len().div_ceil(temporal_patch_size) * temporal_patch_size;
        let (target_h, target_w) = self.smart_resize_video(frames.len(), h as usize, w as usize)?;
        let (tw32, th32) = (target_w as u32, target_h as u32);
        let (grid_t, grid_h, grid_w) = self.calculate_grid_thw(target_h, target_w, padded_frames);

        let patch_size = self.config.patch_size;
        let patch_features = 3 * temporal_patch_size * patch_size * patch_size;
        let num_patches = grid_t * grid_h * grid_w;
        let tokens = self.calculate_tokens_from_grid(grid_t, grid_h, grid_w);
        let mut all_patches = vec![0.0; num_patches * patch_features];

        let do_normalize = config.do_normalize.unwrap_or(true);
        let scale: [f32; 3] = if do_normalize {
            std::array::from_fn(|c| 1.0 / (255.0 * std[c] as f32))
        } else {
            [1.0 / 255.0; 3]
        };
        let bias: [f32; 3] = if do_normalize {
            std::array::from_fn(|c| -(mean[c] as f32) / (std[c] as f32))
        } else {
            [0.0; 3]
        };
        let lut: [[f32; 256]; 3] =
            std::array::from_fn(|c| std::array::from_fn(|v| v as f32 * scale[c] + bias[c]));

        let mut out_idx = 0;
        let mut frame_rgbs = Vec::with_capacity(temporal_patch_size);
        for gt in 0..grid_t {
            frame_rgbs.clear();
            for tp in 0..temporal_patch_size {
                let idx = (gt * temporal_patch_size + tp).min(frames.len() - 1);
                let frame = &frames[idx];
                let needs_resize = config.do_resize.unwrap_or(true)
                    && (frame.width() != tw32 || frame.height() != th32);
                if needs_resize {
                    let resized = resize(frame, tw32, th32, filter);
                    let (width, height, data) = rgb_bytes(&resized);
                    frame_rgbs.push(VideoFrameRgb {
                        width,
                        height,
                        data: Cow::Owned(data.into_owned()),
                    });
                } else {
                    let (width, height, data) = rgb_bytes(frame);
                    frame_rgbs.push(VideoFrameRgb {
                        width,
                        height,
                        data,
                    });
                }
            }

            self.patchify_video_rgb_chunk_into(
                &frame_rgbs,
                grid_h,
                grid_w,
                &mut all_patches,
                &mut out_idx,
                &lut,
            )?;
        }
        debug_assert_eq!(out_idx, all_patches.len());

        let encoder_input = Array2::from_shape_vec((num_patches, patch_features), all_patches)
            .map_err(|e| {
                TransformError::ShapeError(format!(
                    "Failed to create video encoder_input [{num_patches}, {patch_features}]: {e}"
                ))
            })?;

        let result = PreprocessedEncoderInputs::new_dynamic(
            encoder_input.into_dyn(),
            vec![tokens],
            item_sizes,
        )
        .with_extra(
            "video_grid_thw",
            ModelSpecificValue::int_2d(vec![grid_t as i64, grid_h as i64, grid_w as i64], 1, 3),
        )
        .with_extra(
            "patches_per_video",
            ModelSpecificValue::int_1d(vec![num_patches as i64]),
        )
        .with_extra(
            "patches_per_image",
            ModelSpecificValue::int_1d(vec![num_patches as i64]),
        );

        Ok(result)
    }

    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if frames.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        let w = frames[0].width;
        let h = frames[0].height;
        let item_sizes = vec![(w, h)];
        let mean = config.get_image_mean();
        let std = config.get_image_std();
        // Qwen2VL/Qwen3VL image processors default to BICUBIC (PIL resample=3)
        // when the preprocessor config omits `resample`. The global pil_to_filter
        // fallback is bilinear, which yields smoother features and measurably
        // degrades VLM accuracy, so pin the HF-correct default here.
        let filter = pil_to_filter(config.resampling.or(Some(3)));

        let temporal_patch_size = self.config.temporal_patch_size;
        let padded_frames = frames.len().div_ceil(temporal_patch_size) * temporal_patch_size;
        let (target_h, target_w) = self.smart_resize_video(frames.len(), h as usize, w as usize)?;
        let (tw32, th32) = (target_w as u32, target_h as u32);
        let (grid_t, grid_h, grid_w) = self.calculate_grid_thw(target_h, target_w, padded_frames);

        let patch_size = self.config.patch_size;
        let patch_features = 3 * temporal_patch_size * patch_size * patch_size;
        let num_patches = grid_t * grid_h * grid_w;
        let tokens = self.calculate_tokens_from_grid(grid_t, grid_h, grid_w);
        let mut all_patches = vec![0.0; num_patches * patch_features];

        let do_normalize = config.do_normalize.unwrap_or(true);
        let scale: [f32; 3] = if do_normalize {
            std::array::from_fn(|c| 1.0 / (255.0 * std[c] as f32))
        } else {
            [1.0 / 255.0; 3]
        };
        let bias: [f32; 3] = if do_normalize {
            std::array::from_fn(|c| -(mean[c] as f32) / (std[c] as f32))
        } else {
            [0.0; 3]
        };
        let lut: [[f32; 256]; 3] =
            std::array::from_fn(|c| std::array::from_fn(|v| v as f32 * scale[c] + bias[c]));

        let mut needs_resize_any = false;
        let do_resize = config.do_resize.unwrap_or(true);
        for frame in frames {
            let expected_len = (frame.width as usize)
                .checked_mul(frame.height as usize)
                .and_then(|pixels| pixels.checked_mul(3))
                .ok_or_else(|| {
                    TransformError::ShapeError(format!(
                        "video frame dimensions are too large: {}x{}",
                        frame.width, frame.height
                    ))
                })?;
            if frame.data.len() != expected_len {
                return Err(TransformError::InvalidShape {
                    expected: format!(
                        "RGB frame byte length {expected_len} for {}x{}",
                        frame.width, frame.height
                    ),
                    actual: vec![frame.data.len()],
                });
            }
            needs_resize_any |= do_resize && (frame.width != tw32 || frame.height != th32);
        }

        let mut out_idx = 0;
        if needs_resize_any {
            let mut frame_rgbs = Vec::with_capacity(temporal_patch_size);
            for gt in 0..grid_t {
                frame_rgbs.clear();
                for tp in 0..temporal_patch_size {
                    let idx = (gt * temporal_patch_size + tp).min(frames.len() - 1);
                    let frame = frames[idx];
                    let needs_resize = do_resize && (frame.width != tw32 || frame.height != th32);
                    if needs_resize {
                        let resized = resize_rgb_bytes(
                            frame.data,
                            frame.width,
                            frame.height,
                            tw32,
                            th32,
                            filter,
                        )?;
                        frame_rgbs.push(VideoFrameRgb {
                            width: tw32 as usize,
                            height: th32 as usize,
                            data: Cow::Owned(resized.into_raw()),
                        });
                    } else {
                        frame_rgbs.push(VideoFrameRgb {
                            width: frame.width as usize,
                            height: frame.height as usize,
                            data: Cow::Borrowed(frame.data),
                        });
                    }
                }

                self.patchify_video_rgb_chunk_into(
                    &frame_rgbs,
                    grid_h,
                    grid_w,
                    &mut all_patches,
                    &mut out_idx,
                    &lut,
                )?;
            }
        } else {
            let mut frame_rgbs = Vec::with_capacity(temporal_patch_size);
            for gt in 0..grid_t {
                frame_rgbs.clear();
                for tp in 0..temporal_patch_size {
                    let idx = (gt * temporal_patch_size + tp).min(frames.len() - 1);
                    let frame = frames[idx];
                    frame_rgbs.push(VideoFrameRgb {
                        width: frame.width as usize,
                        height: frame.height as usize,
                        data: Cow::Borrowed(frame.data),
                    });
                }

                self.patchify_video_rgb_chunk_into(
                    &frame_rgbs,
                    grid_h,
                    grid_w,
                    &mut all_patches,
                    &mut out_idx,
                    &lut,
                )?;
            }
        }
        debug_assert_eq!(out_idx, all_patches.len());

        let encoder_input = Array2::from_shape_vec((num_patches, patch_features), all_patches)
            .map_err(|e| {
                TransformError::ShapeError(format!(
                    "Failed to create video encoder_input [{num_patches}, {patch_features}]: {e}"
                ))
            })?;

        let result = PreprocessedEncoderInputs::new_dynamic(
            encoder_input.into_dyn(),
            vec![tokens],
            item_sizes,
        )
        .with_extra(
            "video_grid_thw",
            ModelSpecificValue::int_2d(vec![grid_t as i64, grid_h as i64, grid_w as i64], 1, 3),
        )
        .with_extra(
            "patches_per_video",
            ModelSpecificValue::int_1d(vec![num_patches as i64]),
        )
        .with_extra(
            "patches_per_image",
            ModelSpecificValue::int_1d(vec![num_patches as i64]),
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
    use image::RgbImage;

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

    fn create_video_test_config() -> QwenVLConfig {
        QwenVLConfig {
            patch_size: 2,
            merge_size: 1,
            min_pixels: 1,
            max_pixels: 1024 * 1024,
            temporal_patch_size: 2,
            mean: [0.5, 0.25, 0.75],
            std: [0.5, 0.25, 0.5],
            model_name: "test-qwen-vl-video",
        }
    }

    fn create_pattern_frame(seed: u8) -> DynamicImage {
        let mut image = RgbImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                image.put_pixel(
                    x,
                    y,
                    image::Rgb([
                        seed.wrapping_add((x * 3 + y) as u8),
                        seed.wrapping_add((x + y * 5) as u8),
                        seed.wrapping_add((x * 7 + y * 11) as u8),
                    ]),
                );
            }
        }
        DynamicImage::ImageRgb8(image)
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

    #[test]
    fn test_preprocess_video_matches_tensor_patchify() {
        let processor = QwenVLProcessorBase::new(create_video_test_config());
        let config = PreProcessorConfig {
            image_mean: Some(processor.default_mean().to_vec()),
            image_std: Some(processor.default_std().to_vec()),
            ..Default::default()
        };
        let frames = vec![create_pattern_frame(3), create_pattern_frame(101)];

        let result = processor.preprocess_video(&frames, &config).unwrap();
        let actual = result.encoder_input.as_slice_memory_order().unwrap();

        let tensors = frames
            .iter()
            .map(|frame| {
                to_tensor_and_normalize(frame, &processor.default_mean(), &processor.default_std())
            })
            .collect::<Vec<_>>();
        let (grid_t, grid_h, grid_w) = processor.calculate_grid_thw(4, 4, frames.len());
        let mut expected = Vec::new();
        processor
            .patchify_video_into(&tensors, grid_t, grid_h, grid_w, &mut expected)
            .unwrap();

        assert_eq!(actual.len(), expected.len());
        for (idx, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "video patch value differs at index {idx}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn test_preprocess_video_rgb_matches_dynamic_video() {
        let processor = QwenVLProcessorBase::new(create_video_test_config());
        let config = PreProcessorConfig {
            image_mean: Some(processor.default_mean().to_vec()),
            image_std: Some(processor.default_std().to_vec()),
            ..Default::default()
        };
        let frames = vec![
            create_pattern_frame(3),
            create_pattern_frame(101),
            create_pattern_frame(177),
        ];

        let rgb_frames = frames
            .iter()
            .map(|frame| {
                let DynamicImage::ImageRgb8(rgb) = frame else {
                    panic!("test frame is not RGB8");
                };
                RgbFrameRef {
                    width: rgb.width(),
                    height: rgb.height(),
                    data: rgb.as_raw(),
                }
            })
            .collect::<Vec<_>>();

        let dynamic_result = processor.preprocess_video(&frames, &config).unwrap();
        let rgb_result = processor
            .preprocess_video_rgb(&rgb_frames, &config)
            .unwrap();

        assert_eq!(
            dynamic_result.encoder_input.shape(),
            rgb_result.encoder_input.shape()
        );
        let mut dynamic_keys = dynamic_result.model_specific.keys().collect::<Vec<_>>();
        let mut rgb_keys = rgb_result.model_specific.keys().collect::<Vec<_>>();
        dynamic_keys.sort();
        rgb_keys.sort();
        assert_eq!(dynamic_keys, rgb_keys);

        let dynamic_values = dynamic_result
            .encoder_input
            .as_slice_memory_order()
            .unwrap();
        let rgb_values = rgb_result.encoder_input.as_slice_memory_order().unwrap();
        for (idx, (&got, &want)) in rgb_values.iter().zip(dynamic_values.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "RGB video patch value differs at index {idx}: got {got}, want {want}"
            );
        }
    }
}
