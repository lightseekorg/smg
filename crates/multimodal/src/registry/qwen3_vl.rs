use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
    vision::image_processor::PreprocessedImages,
};

pub(super) struct Qwen3VLVisionSpec;

impl Qwen3VLVisionSpec {
    fn pad_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["image_token_id"])
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "image_token_id".to_string(),
            })
    }
}

impl ModelProcessorSpec for Qwen3VLVisionSpec {
    fn name(&self) -> &'static str {
        "qwen3_vl"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        // Qwen3.5 shares Qwen3-VL's image pipeline in sglang.
        let id = metadata.model_id.to_ascii_lowercase();
        let id_match = id.contains("qwen3")
            && (id.contains("vl") || id.contains("3.5") || id.contains("3-5"));
        let type_match = metadata.config_model_type().is_some_and(|mt| {
            matches!(mt, "qwen3_vl" | "qwen3_vl_moe" | "qwen3_5" | "qwen3_5_moe")
        });
        id_match || type_match
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
        let pad_token_id = Self::pad_token_id(metadata)?;
        let placeholder_token = self.placeholder_token(metadata)?;
        // The chat template already wraps each image with <|vision_start|> ... <|vision_end|>,
        // so we only expand the single <|image_pad|> placeholder to N pad tokens.
        Ok(preprocessed
            .num_img_tokens
            .iter()
            .map(|&num_tokens| {
                let tokens = vec![pad_token_id; num_tokens];
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

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec!["image_grid_thw".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{
        registry::{test_helpers::*, ModelMetadata, ModelRegistry},
        types::ImageSize,
    };

    #[test]
    fn qwen3_vl_pad_only_replacement() {
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
        // Only pad tokens — vision_start/vision_end are already in the chat template
        assert_eq!(replacements[0].tokens.len(), 196);
        assert_eq!(replacements[0].tokens[0], 151655); // pad (image_token_id)
        assert_eq!(*replacements[0].tokens.last().unwrap(), 151655); // pad
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
    fn qwen3_vl_matches_alias_via_model_type() {
        let tokenizer = TestTokenizer::new(&[("<|image_pad|>", 151655)]);
        let config = json!({
            "model_type": "qwen3_vl",
            "vision_start_token_id": 151652,
            "image_token_id": 151655,
            "vision_end_token_id": 151653
        });
        let metadata = ModelMetadata {
            model_id: "custom-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry
            .lookup(&metadata)
            .expect("should match qwen3 alias");
        assert_eq!(spec.name(), "qwen3_vl");
    }

    #[test]
    fn qwen3_5_matches_via_model_type() {
        let tokenizer = TestTokenizer::new(&[("<|image_pad|>", 248056)]);
        let config = json!({
            "model_type": "qwen3_5",
            "vision_start_token_id": 248053,
            "image_token_id": 248056,
            "vision_end_token_id": 248054,
            "vision_config": {"patch_size": 16, "spatial_merge_size": 2}
        });
        let metadata = ModelMetadata {
            model_id: "shadow/qwen3-5-9b-smg-grpc",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen3_5 spec");
        assert_eq!(spec.name(), "qwen3_vl");
    }

    #[test]
    fn qwen3_5_moe_matches_via_model_type() {
        let tokenizer = TestTokenizer::new(&[("<|image_pad|>", 248056)]);
        let config = json!({
            "model_type": "qwen3_5_moe",
            "vision_start_token_id": 248053,
            "image_token_id": 248056,
            "vision_end_token_id": 248054,
            "vision_config": {"patch_size": 16, "spatial_merge_size": 2}
        });
        let metadata = ModelMetadata {
            model_id: "some-qwen-moe-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry
            .lookup(&metadata)
            .expect("should match qwen3_5_moe");
        assert_eq!(spec.name(), "qwen3_vl");
    }

    #[test]
    fn qwen3_5_matches_by_model_id_dotted() {
        let tokenizer = TestTokenizer::new(&[("<|image_pad|>", 248056)]);
        let config = json!({
            "model_type": "qwen3_5",
            "image_token_id": 248056,
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054
        });
        let metadata = ModelMetadata {
            model_id: "togethercomputer/Qwen3.5-9B-FP8",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry
            .lookup(&metadata)
            .expect("should match Qwen3.5 model id");
        assert_eq!(spec.name(), "qwen3_vl");
    }
}
