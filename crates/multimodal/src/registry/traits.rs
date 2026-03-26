use std::collections::HashMap;

use llm_tokenizer::TokenizerTrait;
use serde_json::Value;
use thiserror::Error;

use crate::{
    types::{FieldLayout, ImageSize, Modality, PromptReplacement, TokenId},
    vision::image_processor::PreprocessedImages,
};

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

    pub fn config_model_type(&self) -> Option<&str> {
        Self::find_value(self.config, &["model_type"]).and_then(Value::as_str)
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

    /// Tensor keys that should remain on CPU (not transferred to GPU).
    ///
    /// In vLLM, certain model-specific tensors are marked `keep_on_cpu=True`
    /// in their `MultiModalFieldConfig`.  This method mirrors that per-model
    /// knowledge so the router can send the hint via gRPC, avoiding the need
    /// for the backend to instantiate a Python processor just to query it.
    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec![]
    }
}

/// Convert preprocessor `(height, width)` tuples to `ImageSize` values.
pub fn image_sizes_hw(preprocessed: &PreprocessedImages) -> Vec<ImageSize> {
    preprocessed
        .image_sizes
        .iter()
        .map(|&(h, w)| ImageSize {
            width: w,
            height: h,
        })
        .collect()
}
