use std::collections::HashMap;

use llm_tokenizer::TokenizerTrait;
use once_cell::sync::Lazy;
use serde_json::{json, Value};
use thiserror::Error;

use super::types::{FieldLayout, ImageSize, Modality, PromptReplacement, TokenId};
use crate::vision::image_processor::{ModelSpecificValue, PreprocessedImages};

#[derive(Debug, Error)]
pub enum ModelRegistryError {
    #[error("unsupported model: {0}")]
    UnsupportedModel(String),
    #[error("token '{token}' not found in tokenizer vocabulary")]
    TokenNotFound { token: String },
    #[error("missing config field '{field}'")]
    MissingConfigField { field: String },
}

pub type RegistryResult<T> = Result<T, ModelRegistryError>;

/// Metadata about the current model used to derive tokenizer/config dependent fields.
pub struct ModelMetadata<'a> {
    pub model_id: &'a str,
    pub tokenizer: &'a dyn TokenizerTrait,
    pub config: &'a Value,
}

impl<'a> ModelMetadata<'a> {
    pub fn token_id(&self, token: &str) -> RegistryResult<TokenId> {
        self.tokenizer
            .token_to_id(token)
            .map(|id| id as TokenId)
            .ok_or_else(|| ModelRegistryError::TokenNotFound {
                token: token.to_string(),
            })
    }

    pub fn config_u32(&self, path: &[&str]) -> Option<u32> {
        Self::find_value(self.config, path).and_then(|value| value.as_u64().map(|v| v as u32))
    }

    fn find_value<'v>(value: &'v Value, path: &[&str]) -> Option<&'v Value> {
        let mut current = value;
        for key in path {
            current = current.get(*key)?;
        }
        Some(current)
    }
}

pub trait ModelProcessorSpec: Send + Sync {
    fn name(&self) -> &'static str;
    fn matches(&self, metadata: &ModelMetadata) -> bool;
    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String>;
    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId>;
    fn modality_limits(&self, metadata: &ModelMetadata)
        -> RegistryResult<HashMap<Modality, usize>>;
    fn processor_kwargs(&self, metadata: &ModelMetadata) -> RegistryResult<Value>;
    /// Compute per-image prompt replacement token sequences.
    ///
    /// Receives the full preprocessed output so each model can extract whatever
    /// metadata it needs (e.g. aspect_ratios for tile-based models).  This
    /// mirrors vLLM's `_get_prompt_updates(out_mm_kwargs)` pattern.
    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedImages,
    ) -> RegistryResult<Vec<PromptReplacement>>;

    /// Declare how each tensor's first dimension maps to images.
    ///
    /// Keys not listed are treated as shared (replicated across all images).
    /// The `"pixel_values"` key should be included when it differs from batched.
    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // Default: pixel_values is batched (most models).
        HashMap::from([("pixel_values".to_string(), FieldLayout::Batched)])
    }
}

pub struct ModelRegistry {
    specs: Vec<LazySpec>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            specs: vec![
                LazySpec::new("llama4", || Box::new(Llama4Spec)),
                LazySpec::new("llava", || Box::new(LlavaSpec)),
                // Qwen3-VL must be registered before QwenVL so "qwen3" matches first.
                LazySpec::new("qwen3_vl", || Box::new(Qwen3VLVisionSpec)),
                LazySpec::new("qwen_vl", || Box::new(QwenVLVisionSpec)),
                LazySpec::new("phi3_v", || Box::new(Phi3VisionSpec)),
            ],
        }
    }

    pub fn lookup<'a>(&'a self, metadata: &ModelMetadata) -> Option<&'a dyn ModelProcessorSpec> {
        for spec in &self.specs {
            let spec_ref = spec.get();
            if spec_ref.matches(metadata) {
                return Some(spec_ref);
            }
        }
        None
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

struct LazySpec {
    inner: Lazy<Box<dyn ModelProcessorSpec>>,
}

impl LazySpec {
    fn new(_id: &'static str, factory: fn() -> Box<dyn ModelProcessorSpec>) -> Self {
        Self {
            inner: Lazy::new(factory),
        }
    }

    fn get(&self) -> &dyn ModelProcessorSpec {
        self.inner.as_ref()
    }
}

/// Convert preprocessor `(height, width)` tuples to `ImageSize` values.
fn image_sizes_hw(preprocessed: &PreprocessedImages) -> Vec<ImageSize> {
    preprocessed
        .image_sizes
        .iter()
        .map(|&(h, w)| ImageSize {
            width: w,
            height: h,
        })
        .collect()
}

struct LlavaSpec;

impl LlavaSpec {
    fn patch_size(metadata: &ModelMetadata) -> u32 {
        metadata
            .config_u32(&["vision_config", "patch_size"])
            .unwrap_or(14)
    }

    fn tokens_per_image(metadata: &ModelMetadata, size: ImageSize) -> usize {
        let patch = Self::patch_size(metadata);
        let cols = size.width.div_ceil(patch) as usize;
        let rows = size.height.div_ceil(patch) as usize;
        cols * rows
    }
}

impl ModelProcessorSpec for LlavaSpec {
    fn name(&self) -> &'static str {
        "llava"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata.model_id.to_ascii_lowercase().contains("llava")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<image>".to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        if let Some(value) = metadata.config_u32(&["image_token_index"]) {
            return Ok(value as TokenId);
        }
        metadata.token_id("<image>")
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 4)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedImages,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let token_id = self.placeholder_token_id(metadata)?;
        let token = self.placeholder_token(metadata)?;
        let image_sizes = image_sizes_hw(preprocessed);
        Ok(image_sizes
            .iter()
            .map(|size| {
                let count = Self::tokens_per_image(metadata, *size);
                PromptReplacement::repeated(Modality::Image, &token, token_id, count)
            })
            .collect())
    }
}

struct Qwen3VLVisionSpec;

impl Qwen3VLVisionSpec {
    fn pad_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["image_token_id"])
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "image_token_id".to_string(),
            })
    }

    fn start_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["vision_start_token_id"])
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "vision_start_token_id".to_string(),
            })
    }

    fn end_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["vision_end_token_id"])
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "vision_end_token_id".to_string(),
            })
    }
}

impl ModelProcessorSpec for Qwen3VLVisionSpec {
    fn name(&self) -> &'static str {
        "qwen3_vl"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        id.contains("qwen3") && id.contains("vl")
    }

    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String> {
        let token_id = Self::pad_token_id(metadata)? as u32;
        metadata
            .tokenizer
            .id_to_token(token_id)
            .ok_or_else(|| ModelRegistryError::TokenNotFound {
                token: format!("image_token_id:{token_id}"),
            })
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::pad_token_id(metadata)
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 10)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedImages,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let start_token_id = Self::start_token_id(metadata)?;
        let pad_token_id = Self::pad_token_id(metadata)?;
        let end_token_id = Self::end_token_id(metadata)?;
        let placeholder_token = self.placeholder_token(metadata)?;
        Ok(preprocessed
            .num_img_tokens
            .iter()
            .map(|&num_tokens| {
                let mut tokens = Vec::with_capacity(num_tokens + 2);
                tokens.push(start_token_id);
                tokens.extend(std::iter::repeat_n(pad_token_id, num_tokens));
                tokens.push(end_token_id);
                PromptReplacement::sequence(Modality::Image, &placeholder_token, tokens)
            })
            .collect())
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // pixel_values is patchified: [total_patches, patch_features].
        // patches_per_image tells how many patches belong to each image.
        // image_grid_thw is [num_images, 3].
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("image_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_image".to_string(), FieldLayout::Batched),
        ])
    }
}

struct QwenVLVisionSpec;

impl QwenVLVisionSpec {
    fn pad_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["vision_token_id"])
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "vision_token_id".to_string(),
            })
    }

    fn start_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["vision_start_token_id"])
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "vision_start_token_id".to_string(),
            })
    }
}

impl ModelProcessorSpec for QwenVLVisionSpec {
    fn name(&self) -> &'static str {
        "qwen_vl"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata.model_id.to_ascii_lowercase().contains("qwen")
            && metadata.model_id.to_ascii_lowercase().contains("vl")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<image>".to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        // Must match pad_token_id (vision_token_id) — this is the repeated token
        // in the expanded sequence. image_token_id is a distinct token in Qwen2-VL.
        Self::pad_token_id(metadata)
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 10)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedImages,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let start_token_id = Self::start_token_id(metadata)?;
        let pad_token_id = Self::pad_token_id(metadata)?;
        let placeholder_token = self.placeholder_token(metadata)?;
        Ok(preprocessed
            .num_img_tokens
            .iter()
            .map(|&num_tokens| {
                let mut tokens = Vec::with_capacity(num_tokens + 1);
                tokens.push(start_token_id);
                tokens.extend(std::iter::repeat_n(pad_token_id, num_tokens));
                PromptReplacement::sequence(Modality::Image, &placeholder_token, tokens)
            })
            .collect())
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // pixel_values is patchified: [total_patches, patch_features].
        // patches_per_image tells how many patches belong to each image.
        // image_grid_thw is [num_images, 3].
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("image_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_image".to_string(), FieldLayout::Batched),
        ])
    }
}

struct Phi3VisionSpec;

impl Phi3VisionSpec {
    fn tokens_per_image(metadata: &ModelMetadata) -> usize {
        metadata
            .config_u32(&["img_processor", "num_img_tokens"])
            .unwrap_or(256) as usize
    }
}

impl ModelProcessorSpec for Phi3VisionSpec {
    fn name(&self) -> &'static str {
        "phi3_v"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        id.contains("phi") && id.contains("vision")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<image>".to_owned())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.token_id("<image>")
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 4)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedImages,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let token_id = self.placeholder_token_id(metadata)?;
        let token = self.placeholder_token(metadata)?;
        let count = Self::tokens_per_image(metadata);
        Ok(preprocessed
            .image_sizes
            .iter()
            .map(|_| PromptReplacement::repeated(Modality::Image, &token, token_id, count))
            .collect())
    }
}

struct Llama4Spec;

impl Llama4Spec {
    fn patch_size(metadata: &ModelMetadata) -> u32 {
        metadata
            .config_u32(&["vision_config", "patch_size"])
            .unwrap_or(14)
    }

    fn tile_size(metadata: &ModelMetadata) -> u32 {
        metadata
            .config_u32(&["vision_config", "image_size"])
            .filter(|v| *v > 0)
            .unwrap_or(336)
    }

    fn pixel_shuffle_ratio(metadata: &ModelMetadata) -> f64 {
        metadata
            .config
            .get("vision_config")
            .and_then(|v| v.get("pixel_shuffle_ratio"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5)
    }

    fn tokens_per_tile(metadata: &ModelMetadata) -> usize {
        let tile = Self::tile_size(metadata) as usize;
        let patch = Self::patch_size(metadata) as usize;
        if patch == 0 {
            return 0;
        }
        let patches = (tile / patch).pow(2);
        // Pixel shuffle reduces spatial dims by ratio, so token count by ratio^2
        let ratio = Self::pixel_shuffle_ratio(metadata);
        let downsample = (1.0 / (ratio * ratio)).round().max(1.0) as usize;
        patches / downsample
    }

    /// Extract per-image `(h_tiles, w_tiles)` from the preprocessor's
    /// `aspect_ratios` tensor.  Falls back to deriving tile counts from
    /// the original image sizes when aspect_ratios are unavailable.
    fn extract_aspect_ratios(
        preprocessed: &PreprocessedImages,
        tile_size: usize,
    ) -> Vec<(usize, usize)> {
        if let Some(ModelSpecificValue::IntTensor { data, shape }) =
            preprocessed.model_specific.get("aspect_ratios")
        {
            if shape.len() == 2 && shape[1] == 2 && data.len() == shape[0] * 2 {
                return data
                    .chunks_exact(2)
                    .map(|chunk| (chunk[0] as usize, chunk[1] as usize))
                    .collect();
            }
        }
        // Fallback: derive from original image sizes (height, width).
        preprocessed
            .image_sizes
            .iter()
            .map(|&(h, w)| {
                let h_tiles = (h as usize).div_ceil(tile_size);
                let w_tiles = (w as usize).div_ceil(tile_size);
                (h_tiles, w_tiles)
            })
            .collect()
    }
}

impl ModelProcessorSpec for Llama4Spec {
    fn name(&self) -> &'static str {
        "llama4"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        // Match "llama-4", "llama4", "Llama-4-Maverick", "Llama-4-Scout", etc.
        id.contains("llama-4") || id.contains("llama4")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<|image|>".to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        if let Some(value) = metadata.config_u32(&["image_token_index"]) {
            return Ok(value as TokenId);
        }
        metadata.token_id("<|image|>")
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 8)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedImages,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let patch_token_id = self.placeholder_token_id(metadata)?;
        let placeholder = self.placeholder_token(metadata)?;
        let tokens_per_tile = Self::tokens_per_tile(metadata);
        let tile_size = Self::tile_size(metadata) as usize;

        // Structural token IDs matching HF _prompt_split_image format.
        let image_start_id = metadata.token_id("<|image_start|>")?;
        let image_end_id = metadata.token_id("<|image_end|>")?;
        let image_id = metadata.token_id("<|image|>")?;
        let tile_x_sep_id = metadata.token_id("<|tile_x_separator|>")?;
        let tile_y_sep_id = metadata.token_id("<|tile_y_separator|>")?;

        // Extract aspect_ratios from preprocessor output (computed by
        // get_best_fit, respecting max_patches cap).  This is the Llama 4
        // analog of vLLM's out_mm_kwargs["image"][i]["aspect_ratios"].data.
        let aspect_ratios = Self::extract_aspect_ratios(preprocessed, tile_size);

        Ok(aspect_ratios
            .iter()
            .map(|&(h_tiles, w_tiles)| {
                let num_tiles = h_tiles * w_tiles;

                let mut tokens = Vec::new();

                // <|image_start|>
                tokens.push(image_start_id);

                // Grid tiles with separators (only for multi-tile images)
                if num_tiles > 1 {
                    for _row in 0..h_tiles {
                        for col in 0..w_tiles {
                            tokens.extend(std::iter::repeat_n(patch_token_id, tokens_per_tile));
                            if col < w_tiles - 1 {
                                tokens.push(tile_x_sep_id);
                            }
                        }
                        tokens.push(tile_y_sep_id);
                    }
                }

                // Global/cover tile: <|image|> + <|patch|> * tokens_per_tile
                tokens.push(image_id);
                tokens.extend(std::iter::repeat_n(patch_token_id, tokens_per_tile));

                // <|image_end|>
                tokens.push(image_end_id);

                PromptReplacement::sequence(Modality::Image, &placeholder, tokens)
            })
            .collect())
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // pixel_values is [total_tiles, C, H, W] — variable tiles per image.
        // aspect_ratios and patches_per_image are [num_images, ...].
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("aspect_ratios".to_string(), FieldLayout::Batched),
            ("patches_per_image".to_string(), FieldLayout::Batched),
        ])
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use llm_tokenizer::{Decoder, Encoder, Encoding, SpecialTokens, TokenizerTrait};
    use serde_json::json;

    use super::*;

    struct TestTokenizer {
        vocab: HashMap<String, u32>,
    }

    impl TestTokenizer {
        fn new(pairs: &[(&str, u32)]) -> Self {
            let vocab = pairs
                .iter()
                .map(|(token, id)| ((*token).to_string(), *id))
                .collect();
            Self { vocab }
        }
    }

    impl Encoder for TestTokenizer {
        fn encode(&self, _input: &str, _add_special_tokens: bool) -> anyhow::Result<Encoding> {
            Ok(Encoding::Plain(Vec::new()))
        }

        fn encode_batch(
            &self,
            inputs: &[&str],
            add_special_tokens: bool,
        ) -> anyhow::Result<Vec<Encoding>> {
            inputs
                .iter()
                .map(|_| self.encode("", add_special_tokens))
                .collect()
        }
    }

    impl Decoder for TestTokenizer {
        fn decode(&self, _token_ids: &[u32], _skip_special_tokens: bool) -> anyhow::Result<String> {
            Ok(String::new())
        }
    }

    impl TokenizerTrait for TestTokenizer {
        fn vocab_size(&self) -> usize {
            self.vocab.len()
        }

        fn get_special_tokens(&self) -> &SpecialTokens {
            static TOKENS: Lazy<SpecialTokens> = Lazy::new(|| SpecialTokens {
                bos_token: None,
                eos_token: None,
                unk_token: None,
                sep_token: None,
                pad_token: None,
                cls_token: None,
                mask_token: None,
                additional_special_tokens: vec![],
            });
            &TOKENS
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.vocab.get(token).copied()
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            self.vocab
                .iter()
                .find(|(_, &v)| v == id)
                .map(|(k, _)| k.clone())
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// Build a minimal `PreprocessedImages` for testing prompt_replacements.
    fn test_preprocessed(image_sizes: &[ImageSize]) -> PreprocessedImages {
        test_preprocessed_with_tokens(image_sizes, &vec![0; image_sizes.len()])
    }

    fn test_preprocessed_with_tokens(
        image_sizes: &[ImageSize],
        num_img_tokens: &[usize],
    ) -> PreprocessedImages {
        let sizes: Vec<(u32, u32)> = image_sizes.iter().map(|s| (s.height, s.width)).collect();
        PreprocessedImages {
            pixel_values: ndarray::ArrayD::zeros(vec![1, 3, 336, 336]),
            num_img_tokens: num_img_tokens.to_vec(),
            image_sizes: sizes,
            model_specific: std::collections::HashMap::new(),
        }
    }

    /// Build `PreprocessedImages` with explicit aspect_ratios (for Llama4 tests).
    fn test_preprocessed_with_aspects(
        image_sizes: &[ImageSize],
        aspect_ratios: &[(i64, i64)],
    ) -> PreprocessedImages {
        use crate::vision::image_processor::ModelSpecificValue;
        let sizes: Vec<(u32, u32)> = image_sizes.iter().map(|s| (s.height, s.width)).collect();
        let flat: Vec<i64> = aspect_ratios
            .iter()
            .flat_map(|&(h, w)| vec![h, w])
            .collect();
        let batch = aspect_ratios.len();
        let mut model_specific = std::collections::HashMap::new();
        model_specific.insert(
            "aspect_ratios".to_string(),
            ModelSpecificValue::IntTensor {
                data: flat,
                shape: vec![batch, 2],
            },
        );
        PreprocessedImages {
            pixel_values: ndarray::ArrayD::zeros(vec![1, 3, 336, 336]),
            num_img_tokens: vec![0; sizes.len()],
            image_sizes: sizes,
            model_specific,
        }
    }

    #[test]
    fn llava_prompt_replacement_uses_config_ids() {
        let tokenizer = TestTokenizer::new(&[("<image>", 32000)]);
        let config = json!({
            "model_type": "llava",
            "image_token_index": 32000,
            "vision_config": {"patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "llava-v1.5",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("llava spec");
        let replacements = spec
            .prompt_replacements(&metadata, &test_preprocessed(&[ImageSize::new(336, 336)]))
            .unwrap();
        assert_eq!(replacements[0].tokens.len(), 576);
    }

    #[test]
    fn qwen_vision_uses_config_token_ids() {
        let tokenizer = TestTokenizer::new(&[("<image>", 999)]);
        let config = json!({
            "model_type": "qwen2_vl",
            "vision_start_token_id": 151652,
            "vision_token_id": 151654,
            "image_token_id": 151655,
            "vision_config": {"patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "Qwen2-VL-7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen spec");
        // 448/14 = 32 grid, merge_size=2 => (32*32)/4 = 256 tokens
        let replacements = spec
            .prompt_replacements(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(448, 448)], &[256]),
            )
            .unwrap();
        // 256 pad tokens + 1 start token = 257
        assert_eq!(replacements[0].tokens.len(), 257);
        assert_eq!(replacements[0].tokens[0], 151652);
        assert_eq!(replacements[0].tokens[1], 151654);
    }

    #[test]
    fn qwen3_vl_includes_end_token() {
        let tokenizer = TestTokenizer::new(&[("<image>", 999), ("<|image_pad|>", 151655)]);
        let config = json!({
            "model_type": "qwen3_vl",
            "vision_start_token_id": 151652,
            "image_token_id": 151655,
            "vision_end_token_id": 151653,
            "vision_config": {"patch_size": 16}
        });
        let metadata = ModelMetadata {
            model_id: "Qwen3-VL-7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen3 spec");
        assert_eq!(spec.name(), "qwen3_vl");
        // 448/16 = 28 grid, merge_size=2 => (28*28)/4 = 196 tokens
        let replacements = spec
            .prompt_replacements(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(448, 448)], &[196]),
            )
            .unwrap();
        // 196 pad tokens + 1 start + 1 end = 198
        assert_eq!(replacements[0].tokens.len(), 198);
        assert_eq!(replacements[0].tokens[0], 151652); // start
        assert_eq!(replacements[0].tokens[1], 151655); // pad (image_token_id)
        assert_eq!(*replacements[0].tokens.last().unwrap(), 151653); // end
    }

    #[test]
    fn qwen2_vl_does_not_match_qwen3() {
        let tokenizer = TestTokenizer::new(&[("<image>", 999)]);
        let config = json!({
            "model_type": "qwen3_vl",
            "vision_start_token_id": 151652,
            "image_token_id": 151655,
            "vision_end_token_id": 151653,
            "vision_config": {"patch_size": 16}
        });
        let metadata = ModelMetadata {
            model_id: "Qwen3-VL-7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("should match qwen3");
        // Must match qwen3_vl spec, not qwen_vl
        assert_eq!(spec.name(), "qwen3_vl");
    }

    #[test]
    fn phi3_uses_num_img_tokens() {
        let tokenizer = TestTokenizer::new(&[("<image>", 555)]);
        let config = json!({
            "model_type": "phi3_v",
            "img_processor": {"num_img_tokens": 144}
        });
        let metadata = ModelMetadata {
            model_id: "Phi-3-vision",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("phi3 spec");
        let replacements = spec
            .prompt_replacements(&metadata, &test_preprocessed(&[ImageSize::new(336, 336)]))
            .unwrap();
        assert_eq!(replacements[0].tokens.len(), 144);
        assert_eq!(replacements[0].tokens[0], 555);
    }

    #[test]
    fn llama4_single_tile_token_count() {
        let tokenizer = TestTokenizer::new(&[
            ("<|image|>", 200090),
            ("<|image_start|>", 200088),
            ("<|image_end|>", 200089),
            ("<|patch|>", 200092),
            ("<|tile_x_separator|>", 200093),
            ("<|tile_y_separator|>", 200094),
        ]);
        let config = json!({
            "model_type": "llama4",
            "image_token_index": 200092,
            "vision_config": {"image_size": 336, "patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "/models/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("llama4 spec");
        assert_eq!(spec.name(), "llama4");

        // Single tile (336x336): <|image_start|> <|image|> <|patch|>*144 <|image_end|>
        // = 1 + 1 + 144 + 1 = 147 tokens
        let pp = test_preprocessed_with_aspects(&[ImageSize::new(336, 336)], &[(1, 1)]);
        let replacements = spec.prompt_replacements(&metadata, &pp).unwrap();
        assert_eq!(replacements[0].tokens.len(), 147);
        assert_eq!(replacements[0].tokens[0], 200088); // <|image_start|>
        assert_eq!(replacements[0].tokens[1], 200090); // <|image|>
        assert_eq!(replacements[0].tokens[2], 200092); // <|patch|> (first)
        assert_eq!(replacements[0].tokens[145], 200092); // <|patch|> (last)
        assert_eq!(replacements[0].tokens[146], 200089); // <|image_end|>
    }

    #[test]
    fn llama4_multi_tile_adds_global() {
        let tokenizer = TestTokenizer::new(&[
            ("<|image|>", 200090),
            ("<|image_start|>", 200088),
            ("<|image_end|>", 200089),
            ("<|patch|>", 200092),
            ("<|tile_x_separator|>", 200093),
            ("<|tile_y_separator|>", 200094),
        ]);
        let config = json!({
            "model_type": "llama4",
            "image_token_index": 200092,
            "vision_config": {"image_size": 336, "patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "Llama-4-Scout-Vision",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("llama4 spec");

        // 672x672 = 2x2 tiles + 1 global:
        //   <|image_start|>                                  = 1
        //   row0: <|patch|>*144 <|tile_x_sep|> <|patch|>*144 <|tile_y_sep|>  = 290
        //   row1: <|patch|>*144 <|tile_x_sep|> <|patch|>*144 <|tile_y_sep|>  = 290
        //   <|image|> <|patch|>*144                          = 145
        //   <|image_end|>                                    = 1
        //   Total = 1 + 290 + 290 + 145 + 1 = 727
        let pp = test_preprocessed_with_aspects(&[ImageSize::new(672, 672)], &[(2, 2)]);
        let replacements = spec.prompt_replacements(&metadata, &pp).unwrap();
        assert_eq!(replacements[0].tokens.len(), 727);
        // Verify structure: starts with image_start, ends with image_end
        assert_eq!(replacements[0].tokens[0], 200088); // <|image_start|>
        assert_eq!(*replacements[0].tokens.last().unwrap(), 200089); // <|image_end|>
                                                                     // The token before the last patch block is <|image|> (global tile marker)
                                                                     // Position: 1 + 290 + 290 = 581
        assert_eq!(replacements[0].tokens[581], 200090); // <|image|>
    }
}
