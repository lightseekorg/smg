//! Shared base implementation for MoonViT-based image processors.
//!
//! Kimi-K2.5 and Kimi-K3 both encode images with MoonViT and differ only in
//! their preprocessing parameters. This mirrors how the reference
//! implementation factors the NaViT resize/patchify helpers into a shared
//! `media_utils` module and keeps a thin per-model processor class on top, and
//! how [`super::qwen_vl_base`] serves the Qwen VL family here.
//!
//! # Processing Pipeline
//!
//! 1. Compute scale to fit within patch limits (never upscale)
//! 2. Resize with BICUBIC interpolation
//! 3. Zero-pad to make dimensions divisible by factor (patch_size * merge_size)
//! 4. Normalize with the checkpoint's mean/std
//! 5. Extract patches as [N, C, patch_size, patch_size]
//!
//! MoonViT resizes then zero-pads to make dimensions divisible by the alignment
//! factor (patch_size * merge_size). The models were trained with zero-padded
//! images, so using direct resize-to-aligned would degrade image quality.

use image::{DynamicImage, GenericImageView};
use ndarray::Array3;

use crate::vision::{
    preprocessor_config::PreProcessorConfig,
    processor::{ModelSpecificValue, PreprocessedEncoderInputs, VisionPreProcessor},
    scratch,
    transforms::{self, TransformError},
};

/// Parameters distinguishing one MoonViT processor variant from another.
#[derive(Debug, Clone)]
pub struct MoonVitConfig {
    /// Vision encoder patch size.
    pub patch_size: usize,
    /// Merge size for token reduction.
    pub merge_size: usize,
    /// Maximum total patches before merge (`in_patch_limit`).
    pub in_patch_limit: usize,
    /// Maximum patches along one spatial dimension.
    pub patch_limit_on_one_side: usize,
    /// Default normalization mean, used when the checkpoint declares none.
    pub mean: [f64; 3],
    /// Default normalization std, used when the checkpoint declares none.
    pub std: [f64; 3],
    /// Model name for identification.
    pub model_name: &'static str,
}

/// MoonViT resize configuration for a single image.
pub(super) struct ResizeConfig {
    pub(super) new_width: usize,
    pub(super) new_height: usize,
    pub(super) pad_width: usize,
    pub(super) pad_height: usize,
    pub(super) num_tokens: usize,
}

/// Shared MoonViT preprocessing pipeline, parameterized by [`MoonVitConfig`].
#[derive(Debug, Clone)]
pub struct MoonVitProcessorBase {
    config: MoonVitConfig,
}

impl MoonVitProcessorBase {
    pub fn new(config: MoonVitConfig) -> Self {
        Self { config }
    }

    pub fn patch_size(&self) -> usize {
        self.config.patch_size
    }

    pub fn merge_size(&self) -> usize {
        self.config.merge_size
    }

    pub fn in_patch_limit(&self) -> usize {
        self.config.in_patch_limit
    }

    pub fn patch_limit_on_one_side(&self) -> usize {
        self.config.patch_limit_on_one_side
    }

    /// Dimension alignment factor: `patch_size * merge_size`.
    #[inline]
    pub fn factor(&self) -> usize {
        self.config.patch_size * self.config.merge_size
    }

    /// Compute resize dimensions and padding, matching HF `navit_resize_image`.
    ///
    /// Never upscales (scale capped at 1.0). Pads with zeros to align to factor.
    pub(super) fn compute_resize_config(&self, width: usize, height: usize) -> ResizeConfig {
        let ps = self.config.patch_size;
        let patches_w = (width / ps).max(1) as f64;
        let patches_h = (height / ps).max(1) as f64;

        let s1 = (self.config.in_patch_limit as f64 / (patches_w * patches_h)).sqrt();
        let s2 = (self.config.patch_limit_on_one_side * ps) as f64 / width as f64;
        let s3 = (self.config.patch_limit_on_one_side * ps) as f64 / height as f64;
        let scale = f64::min(1.0, f64::min(s1, f64::min(s2, s3)));

        let new_w = ((width as f64 * scale) as usize).max(1);
        let new_h = ((height as f64 * scale) as usize).max(1);
        let new_w = new_w.min(self.config.patch_limit_on_one_side * ps);
        let new_h = new_h.min(self.config.patch_limit_on_one_side * ps);

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

        // Pooled: this per-image CHW buffer (tens of MB) is recycled by the
        // caller after patch extraction, keeping its pages mapped and hot.
        let mut data = scratch::take_f32(3 * canvas_pixels);
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
    /// Append this image's patches directly into `out` (no per-image intermediate
    /// Vec): `out` is the pooled batch buffer pre-sized for the whole request.
    fn extract_patches_into(tensor: &Array3<f32>, patch_size: usize, out: &mut Vec<f32>) {
        let channels = tensor.shape()[0];
        let height = tensor.shape()[1];
        let width = tensor.shape()[2];

        let grid_h = height / patch_size;
        let grid_w = width / patch_size;

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
                        out.extend_from_slice(&data[row_start..row_start + patch_size]);
                    }
                }
            }
        }
    }
}

impl VisionPreProcessor for MoonVitProcessorBase {
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

        let patch_size = self.config.patch_size;
        let item_sizes: Vec<(u32, u32)> = images.iter().map(|img| img.dimensions()).collect();
        let mean = config.get_image_mean();
        let std = config.get_image_std();

        // Pre-size the pooled batch buffer exactly (patch_features per patch =
        // 3 * patch_size^2; this is the data plane's hottest allocation).
        let patch_features = 3 * patch_size * patch_size;
        let mut estimated_total = 0usize;
        for image in images {
            let (w, h) = image.dimensions();
            let cfg = self.compute_resize_config(w as usize, h as usize);
            let grid_h = (cfg.new_height + cfg.pad_height) / patch_size;
            let grid_w = (cfg.new_width + cfg.pad_width) / patch_size;
            estimated_total += grid_h * grid_w * patch_features;
        }
        let mut all_patches: Vec<f32> = scratch::take_f32_cap(estimated_total);
        let mut patches_per_image: Vec<i64> = Vec::with_capacity(images.len());
        let mut grid_thw_data = Vec::with_capacity(images.len() * 3);
        let mut feature_token_counts = Vec::with_capacity(images.len());

        for image in images {
            let (w, h) = image.dimensions();
            let cfg = self.compute_resize_config(w as usize, h as usize);

            // Fused resize + pad + normalize in one pass (avoids 2 extra allocations)
            let tensor = Self::resize_pad_and_normalize(image, &cfg, &mean, &std);

            let padded_h = cfg.new_height + cfg.pad_height;
            let padded_w = cfg.new_width + cfg.pad_width;
            let grid_h = padded_h / patch_size;
            let grid_w = padded_w / patch_size;
            let grid_t = 1usize;

            grid_thw_data.push(grid_t as i64);
            grid_thw_data.push(grid_h as i64);
            grid_thw_data.push(grid_w as i64);

            let num_patches = grid_h * grid_w;
            feature_token_counts.push(cfg.num_tokens);

            // Patchify directly into the pooled batch buffer, then recycle the
            // CHW tensor's storage (standard layout, offset 0) for the next image.
            Self::extract_patches_into(&tensor, patch_size, &mut all_patches);
            let (storage, _offset) = tensor.into_raw_vec_and_offset();
            scratch::give_f32(storage);
            patches_per_image.push(num_patches as i64);
        }

        let total_patches: usize = patches_per_image.iter().map(|&n| n as usize).sum();
        let encoder_input = ndarray::Array4::from_shape_vec(
            (total_patches, 3, patch_size, patch_size),
            all_patches,
        )
        .map_err(|e| {
            TransformError::ShapeError(format!(
                "Failed to create encoder_input [{total_patches}, 3, {patch_size}, \
                         {patch_size}]: {e}"
            ))
        })?;

        let result =
            PreprocessedEncoderInputs::new(encoder_input, feature_token_counts, item_sizes)
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
        self.config.model_name
    }

    fn get_processed_size(&self, _config: &PreProcessorConfig) -> Option<(u32, u32)> {
        None
    }
}
