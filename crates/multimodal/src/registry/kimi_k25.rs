use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
    vision::image_processor::PreprocessedImages,
};

pub(super) struct KimiK25Spec;

impl KimiK25Spec {
    fn media_placeholder_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["media_placeholder_token_id"])
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "media_placeholder_token_id".to_string(),
            })
    }
}

impl ModelProcessorSpec for KimiK25Spec {
    fn name(&self) -> &'static str {
        "kimi_k25"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata
            .config_model_type()
            .is_some_and(|mt| mt == "kimi_k25")
            || {
                let id = metadata.model_id.to_ascii_lowercase();
                id.contains("kimi") && (id.contains("k2.5") || id.contains("k25"))
            }
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<|media_pad|>".to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::media_placeholder_token_id(metadata)
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
        let pad_token_id = Self::media_placeholder_token_id(metadata)?;
        let placeholder_token = self.placeholder_token(metadata)?;
        // Keep 1 placeholder per image — TRT-LLM's KimiK25InputProcessor
        // handles expansion to N vision tokens server-side based on grid_thws.
        // SMG must NOT pre-expand or TRT-LLM will see N placeholders and
        // attempt to expand each one again.
        Ok(preprocessed
            .num_img_tokens
            .iter()
            .map(|_| {
                let tokens = vec![pad_token_id; 1];
                PromptReplacement::sequence(Modality::Image, &placeholder_token, tokens)
            })
            .collect())
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // pixel_values is patchified: [total_patches, C, patch_h, patch_w].
        // grid_thws is [num_images, 3] with (t, h, w) per image.
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("grid_thws".to_string(), FieldLayout::Batched),
            ("patches_per_image".to_string(), FieldLayout::Batched),
        ])
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec!["grid_thws".to_string()]
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
    fn kimi_k25_matches_by_model_type() {
        let tokenizer = TestTokenizer::new(&[("<|media_pad|>", 163605)]);
        let config = json!({
            "model_type": "kimi_k25",
            "media_placeholder_token_id": 163605,
        });
        let metadata = ModelMetadata {
            model_id: "some-custom-name",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("should match kimi_k25");
        assert_eq!(spec.name(), "kimi_k25");
    }

    #[test]
    fn kimi_k25_matches_by_model_id() {
        let tokenizer = TestTokenizer::new(&[("<|media_pad|>", 163605)]);
        let config = json!({
            "model_type": "unknown",
            "media_placeholder_token_id": 163605,
        });
        let metadata = ModelMetadata {
            model_id: "nvidia/Kimi-K2.5-NVFP4",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry
            .lookup(&metadata)
            .expect("should match kimi by name");
        assert_eq!(spec.name(), "kimi_k25");
    }

    #[test]
    fn kimi_k25_prompt_replacements() {
        let tokenizer = TestTokenizer::new(&[("<|media_pad|>", 163605)]);
        let config = json!({
            "model_type": "kimi_k25",
            "media_placeholder_token_id": 163605,
        });
        let metadata = ModelMetadata {
            model_id: "nvidia/Kimi-K2.5-NVFP4",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("kimi spec");
        // 1 placeholder per image (TRT-LLM expands server-side)
        let replacements = spec
            .prompt_replacements(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(448, 448)], &[256]),
            )
            .unwrap();
        assert_eq!(replacements.len(), 1);
        assert_eq!(replacements[0].tokens.len(), 1);
        assert_eq!(replacements[0].tokens[0], 163605);
    }
}
