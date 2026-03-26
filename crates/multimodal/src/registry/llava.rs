use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    registry::{image_sizes_hw, ModelMetadata, ModelProcessorSpec, RegistryResult},
    types::{ImageSize, Modality, PromptReplacement, TokenId},
    vision::image_processor::PreprocessedImages,
};

pub(super) struct LlavaSpec;

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
            || metadata
                .config_model_type()
                .is_some_and(|mt| mt == "llava" || mt == "llava_next")
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{
        registry::{test_helpers::*, ModelMetadata, ModelRegistry},
        types::ImageSize,
    };

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
    fn llava_matches_alias_via_model_type() {
        let tokenizer = TestTokenizer::new(&[("<image>", 32000)]);
        let config = json!({
            "model_type": "llava",
            "image_token_index": 32000,
            "vision_config": {"patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "custom-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("llava alias");
        assert_eq!(spec.name(), "llava");
    }
}
