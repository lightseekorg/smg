//! Kimi-K3 (MoonViT) image processor.
//!
//! K3 encodes images with the same MoonViT stack as K2.5 — implemented in
//! [`super::moonvit`] — but its reference processor is a separate class with its
//! own parameters, so it gets its own thin wrapper here rather than sharing
//! K2.5's. This mirrors how [`super::qwen2_vl`] and [`super::qwen3_vl`] sit on
//! [`super::qwen_vl_base`].
//!
//! The difference that matters for output: K3 composites alpha-carrying images
//! over a background the checkpoint describes, where K2.5 just drops the alpha
//! channel. Dropping it is not equivalent — a fully transparent pixel usually
//! stores RGB `(0,0,0)`, so the model sees solid black where it was trained to
//! see a chessboard. The settings are read from the request's
//! `PreProcessorConfig` rather than hardcoded, matching the reference, which
//! falls back to dropping alpha when `transparent_bg_config` is absent.
//!
//! Known remaining difference, deliberately left for a follow-up change: K3's
//! checkpoint raises `in_patch_limit` to 65536, so large images should keep
//! roughly 4x the visual tokens they get from K2.5's budget. Reading the
//! budget from the checkpoint touches token accounting on both models and is
//! tracked separately from this module's transparency handling.

use std::{borrow::Cow, ops::Deref};

use image::DynamicImage;

use super::{
    kimi_k25::{
        DEFAULT_IN_PATCH_LIMIT, DEFAULT_MERGE_SIZE, DEFAULT_PATCH_LIMIT_ON_ONE_SIDE,
        DEFAULT_PATCH_SIZE, KIMI_K25_MEAN, KIMI_K25_STD,
    },
    moonvit::{MoonVitConfig, MoonVitProcessorBase},
};
use crate::vision::{
    preprocessor_config::PreProcessorConfig,
    processor::{PreprocessedEncoderInputs, VisionPreProcessor},
    transforms::TransformError,
};

/// K3 ships the same MoonViT normalization constants as K2.5.
pub const KIMI_K3_MEAN: [f64; 3] = KIMI_K25_MEAN;
pub const KIMI_K3_STD: [f64; 3] = KIMI_K25_STD;

#[derive(Debug, Clone)]
pub struct KimiK3Processor {
    inner: MoonVitProcessorBase,
}

impl Default for KimiK3Processor {
    fn default() -> Self {
        Self::new()
    }
}

impl KimiK3Processor {
    /// Create a processor with Kimi-K3's defaults.
    pub fn new() -> Self {
        Self {
            inner: MoonVitProcessorBase::new(Self::base_config(
                DEFAULT_PATCH_SIZE,
                DEFAULT_MERGE_SIZE,
                DEFAULT_IN_PATCH_LIMIT,
                DEFAULT_PATCH_LIMIT_ON_ONE_SIDE,
            )),
        }
    }

    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        Self {
            inner: MoonVitProcessorBase::new(Self::base_config(
                config.get_patch_size(DEFAULT_PATCH_SIZE),
                config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE),
                config
                    .get_extra::<usize>("in_patch_limit")
                    .unwrap_or(DEFAULT_IN_PATCH_LIMIT),
                config
                    .get_extra::<usize>("patch_limit_on_one_side")
                    .unwrap_or(DEFAULT_PATCH_LIMIT_ON_ONE_SIDE),
            )),
        }
    }

    fn base_config(
        patch_size: usize,
        merge_size: usize,
        in_patch_limit: usize,
        patch_limit_on_one_side: usize,
    ) -> MoonVitConfig {
        MoonVitConfig {
            patch_size,
            merge_size,
            in_patch_limit,
            patch_limit_on_one_side,
            mean: KIMI_K3_MEAN,
            std: KIMI_K3_STD,
            // Resolved per request from the checkpoint, never hardcoded here:
            // the reference reads `transparent_bg_config` out of the model's
            // config and falls back to dropping alpha when the key is absent.
            transparent_bg: None,
            model_name: "kimi-k3",
        }
    }

    /// Apply the checkpoint's transparency settings, if it declares any.
    ///
    /// The registry hands out a default-constructed processor, so a request's
    /// `PreProcessorConfig` is the first point at which these values are
    /// known. Mirrors [`super::qwen2_vl`]'s call-time re-resolution, narrowed
    /// to transparency: nothing else here depends on the checkpoint.
    fn resolved(&self, config: &PreProcessorConfig) -> Cow<'_, MoonVitProcessorBase> {
        match config.transparent_bg() {
            Some(bg) => Cow::Owned(self.inner.with_transparent_bg(Some(bg))),
            None => Cow::Borrowed(&self.inner),
        }
    }

    pub fn patch_size(&self) -> usize {
        self.inner.patch_size()
    }

    pub fn merge_size(&self) -> usize {
        self.inner.merge_size()
    }
}

impl Deref for KimiK3Processor {
    type Target = MoonVitProcessorBase;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl VisionPreProcessor for KimiK3Processor {
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
        self.resolved(config).preprocess(images, config)
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
    use image::{Rgb, RgbImage, Rgba, RgbaImage};

    use super::*;
    use crate::vision::processors::KimiK25Processor;

    fn kimi_config() -> PreProcessorConfig {
        PreProcessorConfig {
            image_mean: Some(KIMI_K3_MEAN.to_vec()),
            image_std: Some(KIMI_K3_STD.to_vec()),
            ..Default::default()
        }
    }

    /// A config carrying the transparency keys as a checkpoint would ship them.
    fn kimi_config_with_bg(bg: serde_json::Value, stage: &str) -> PreProcessorConfig {
        PreProcessorConfig {
            extra: [
                ("transparent_bg_config".to_string(), bg),
                (
                    "transparent_bg_fill_stage".to_string(),
                    serde_json::json!(stage),
                ),
            ]
            .into_iter()
            .collect(),
            ..kimi_config()
        }
    }

    /// With mean = std = 0.5, a channel byte `v` normalizes to `v/127.5 - 1`:
    /// black is -1.0 and white is +1.0.
    fn normalized(byte: u8) -> f32 {
        f32::from(byte) / 127.5 - 1.0
    }

    #[test]
    fn defaults_match_the_moonvit_stack() {
        let p = KimiK3Processor::new();
        assert_eq!(p.patch_size(), 14);
        assert_eq!(p.merge_size(), 2);
        assert_eq!(p.factor(), 28);
        assert_eq!(p.default_mean(), KIMI_K3_MEAN);
        assert_eq!(p.default_std(), KIMI_K3_STD);
    }

    #[test]
    fn model_name_is_distinct_from_k25() {
        assert_eq!(KimiK3Processor::new().model_name(), "kimi-k3");
        assert_ne!(
            KimiK3Processor::new().model_name(),
            KimiK25Processor::new().model_name()
        );
    }

    #[test]
    fn from_preprocessor_config_reads_limits() {
        let config = PreProcessorConfig {
            extra: [
                ("in_patch_limit".to_string(), serde_json::json!(65536)),
                (
                    "patch_limit_on_one_side".to_string(),
                    serde_json::json!(512),
                ),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        let p = KimiK3Processor::from_preprocessor_config(&config);
        assert_eq!(p.in_patch_limit(), 65536);
        assert_eq!(p.patch_limit_on_one_side(), 512);
    }

    /// Opaque images must go through K3 exactly as they go through K2.5: the
    /// two processors share one pipeline, and only alpha-carrying inputs are
    /// allowed to diverge.
    #[test]
    fn opaque_images_match_k25_byte_for_byte() {
        let image = DynamicImage::from(RgbImage::from_pixel(600, 400, Rgb([37, 128, 220])));
        let config = kimi_config();

        let k3 = KimiK3Processor::new()
            .preprocess(std::slice::from_ref(&image), &config)
            .unwrap();
        let k25 = KimiK25Processor::new()
            .preprocess(std::slice::from_ref(&image), &config)
            .unwrap();

        assert_eq!(k3.encoder_input.shape(), k25.encoder_input.shape());
        assert_eq!(k3.feature_token_counts, k25.feature_token_counts);
        assert_eq!(k3.encoder_input_flat(), k25.encoder_input_flat());
    }

    #[test]
    fn transparent_pixels_composite_over_the_configured_background() {
        // A fully transparent PNG stores RGB (0,0,0) under the alpha channel,
        // so dropping alpha normalizes it to -1.0 — solid black. Compositing
        // over the checkpoint's white background must instead give +1.0.
        let image = DynamicImage::from(RgbaImage::from_pixel(56, 56, Rgba([0, 0, 0, 0])));
        let config = kimi_config_with_bg(serde_json::json!({"pattern": "white"}), "after_resize");

        let out = KimiK3Processor::new()
            .preprocess(std::slice::from_ref(&image), &config)
            .unwrap();

        assert!(
            out.encoder_input_flat()
                .iter()
                .all(|v| (v - normalized(255)).abs() < 1e-6),
            "transparent pixels must read as the background, not as black"
        );
    }

    #[test]
    fn transparency_config_does_not_reach_k25() {
        // K2.5's reference processor has no transparency handling at all, so
        // the same config must leave it dropping alpha.
        let image = DynamicImage::from(RgbaImage::from_pixel(56, 56, Rgba([0, 0, 0, 0])));
        let config = kimi_config_with_bg(serde_json::json!({"pattern": "white"}), "after_resize");

        let out = KimiK25Processor::new()
            .preprocess(std::slice::from_ref(&image), &config)
            .unwrap();

        assert!(
            out.encoder_input_flat()
                .iter()
                .all(|v| (v - normalized(0)).abs() < 1e-6),
            "K2.5 must ignore a background it does not declare"
        );
    }

    #[test]
    fn opaque_images_ignore_the_transparency_config() {
        // Declaring a background must not perturb images without alpha, which
        // is the overwhelming majority of traffic.
        let image = DynamicImage::from(RgbImage::from_pixel(600, 400, Rgb([37, 128, 220])));
        let bg = serde_json::json!({"pattern": "chessboard"});

        let plain = KimiK3Processor::new()
            .preprocess(std::slice::from_ref(&image), &kimi_config())
            .unwrap();
        let with_bg = KimiK3Processor::new()
            .preprocess(
                std::slice::from_ref(&image),
                &kimi_config_with_bg(bg, "after_resize"),
            )
            .unwrap();

        assert_eq!(plain.encoder_input_flat(), with_bg.encoder_input_flat());
    }

    #[test]
    fn fill_stage_paints_the_board_at_its_own_resolution() {
        // The chessboard is generated at the resolution of the image it lands
        // on, so flattening before vs. after the resize changes the square
        // size relative to the content. Force a 4x downscale and check the two
        // stages disagree — the key is not cosmetic.
        let image = DynamicImage::from(RgbaImage::from_pixel(112, 112, Rgba([0, 0, 0, 0])));
        let bg = serde_json::json!({
            "pattern": "chessboard",
            "chessboard_square_size": 8,
        });

        let outputs = ["before_resize", "after_resize"].map(|stage| {
            let mut config = kimi_config_with_bg(bg.clone(), stage);
            config
                .extra
                .insert("patch_limit_on_one_side".to_string(), serde_json::json!(2));
            KimiK3Processor::from_preprocessor_config(&config)
                .preprocess(std::slice::from_ref(&image), &config)
                .unwrap()
        });

        assert_eq!(
            outputs[0].encoder_input.shape(),
            outputs[1].encoder_input.shape(),
            "only the pixel values should differ"
        );
        assert_ne!(
            outputs[0].encoder_input_flat(),
            outputs[1].encoder_input_flat()
        );
    }
}
