use std::collections::HashMap;

use llm_tokenizer::TokenizerTrait;
use serde_json::Value;
use thiserror::Error;

use crate::{
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
    vision::processor::PreprocessedEncoderInputs,
};

#[derive(Debug, Error)]
pub enum ModelRegistryError {
    #[error("unsupported model: {0}")]
    UnsupportedModel(String),
    #[error("token '{token}' not found in tokenizer vocabulary")]
    TokenNotFound { token: String },
    #[error("missing config field '{field}'")]
    MissingConfigField { field: String },
    #[error("modality {modality} is not supported by model spec {spec}")]
    UnsupportedModality {
        spec: &'static str,
        modality: Modality,
    },
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
    fn placeholder_token_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => self.placeholder_token(metadata),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }
    fn placeholder_token_id_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<TokenId> {
        match modality {
            Modality::Image => self.placeholder_token_id(metadata),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }
    fn modality_limits(&self, metadata: &ModelMetadata)
        -> RegistryResult<HashMap<Modality, usize>>;
    fn processor_kwargs(&self, metadata: &ModelMetadata) -> RegistryResult<Value>;
    /// Compute per-media prompt replacement token sequences.
    ///
    /// Receives the full preprocessed output so each model can extract whatever
    /// metadata it needs (e.g. aspect_ratios for tile-based models).  This
    /// mirrors vLLM's `_get_prompt_updates(out_mm_kwargs)` pattern.
    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>>;
    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        match modality {
            Modality::Image => self.prompt_replacements(metadata, preprocessed),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    /// Declare how each tensor's first dimension maps to media items.
    ///
    /// Keys not listed are treated as shared (replicated across all media items).
    /// The `"pixel_values"` key mirrors HF/vLLM vision kwargs and should be
    /// included when the primary encoder input differs from batched layout.
    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // Default: encoder_input is batched (most models).
        HashMap::from([("pixel_values".to_string(), FieldLayout::Batched)])
    }

    /// Model-specific tensor keys that should remain CPU-resident.
    ///
    /// Backend adapters translate this model-level placement requirement to
    /// their native wire or runtime representation.
    fn cpu_resident_tensor_keys(&self) -> Vec<String> {
        vec![]
    }
}
