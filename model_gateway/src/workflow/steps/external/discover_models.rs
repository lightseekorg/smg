//! Model discovery step for external API endpoints.

use std::time::Duration;

use async_trait::async_trait;
use once_cell::sync::Lazy;
use openai_protocol::{model_card::ModelCard, model_type::ModelType, worker::ProviderType};
use reqwest::Client;
use serde::{Deserialize, Deserializer};
use tracing::{debug, info};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::workflow::data::{WorkerKind, WorkerWorkflowData};

// HTTP client for API calls
#[expect(
    clippy::expect_used,
    reason = "Lazy static initialization — reqwest::Client::build() only fails on TLS backend misconfiguration which is unrecoverable"
)]
static HTTP_CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to create HTTP client")
});

/// OpenAI /v1/models response format.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelsResponse {
    #[serde(default, deserialize_with = "deserialize_model_rows")]
    pub data: Vec<ModelInfo>,
    #[serde(default)]
    pub object: String,
}

/// Individual model information from /v1/models.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(default, deserialize_with = "deserialize_aliases")]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: Option<u64>,
    #[serde(default)]
    pub owned_by: Option<String>,
}

fn deserialize_aliases<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum AliasesField {
        Values(Vec<String>),
        Null(Option<()>),
        Other(serde::de::IgnoredAny),
    }

    Ok(match AliasesField::deserialize(deserializer)? {
        AliasesField::Values(values) => values,
        AliasesField::Null(_) | AliasesField::Other(_) => Vec::new(),
    })
}

fn deserialize_model_rows<'de, D>(deserializer: D) -> Result<Vec<ModelInfo>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum DataField {
        Rows(Vec<serde_json::Value>),
        Other(serde::de::IgnoredAny),
    }

    Ok(match DataField::deserialize(deserializer)? {
        DataField::Rows(rows) => rows
            .into_iter()
            .filter_map(|row| serde_json::from_value::<ModelInfo>(row).ok())
            .collect(),
        DataField::Other(_) => Vec::new(),
    })
}

/// Convert a flat upstream `/v1/models` list into `ModelCard`s.
///
/// Upstream IDs are preserved as-is. Virtual short aliases like `grok-4` are
/// resolved later by `WorkerModels::find` via prefix fallback.
pub fn build_model_cards(models: Vec<ModelInfo>) -> Vec<ModelCard> {
    models
        .into_iter()
        .map(|model| {
            let mut card = ModelCard::new(&model.id)
                .with_model_type(infer_model_type_from_id(&model.id))
                .with_created_at(model.created.unwrap_or(0));

            if !model.aliases.is_empty() {
                card = card.with_aliases(model.aliases);
            }

            card.provider = infer_provider_from_id(&card.id);
            card
        })
        .collect()
}

pub(crate) fn apply_provider_hint(model_cards: &mut [ModelCard], provider: Option<&ProviderType>) {
    if let Some(provider) = provider {
        for card in model_cards {
            card.provider = Some(provider.clone());
        }
    }
}

/// Kept for backwards compatibility with existing call sites.
#[inline]
pub fn group_models_into_cards(models: Vec<ModelInfo>) -> Vec<ModelCard> {
    build_model_cards(models)
}

/// Infer ModelType from model ID string.
pub fn infer_model_type_from_id(id: &str) -> ModelType {
    let id_lower = id.to_lowercase();

    // Embedding models
    if id_lower.contains("embed") || id_lower.contains("ada-002") {
        return ModelType::EMBED_MODEL;
    }

    // Rerank models
    if id_lower.contains("rerank") {
        return ModelType::RERANK_MODEL;
    }

    // Image generation models
    if id_lower.starts_with("dall-e")
        || id_lower.starts_with("sora")
        || (id_lower.contains("image") && !id_lower.contains("vision"))
    {
        return ModelType::IMAGE_MODEL;
    }

    // Audio models
    if id_lower.starts_with("tts")
        || id_lower.starts_with("whisper")
        || id_lower.contains("audio")
        || id_lower.contains("realtime")
        || id_lower.contains("transcribe")
    {
        return ModelType::AUDIO_MODEL;
    }

    // Moderation models
    if id_lower.contains("moderation") {
        return ModelType::MODERATION_MODEL;
    }

    // Reasoning models
    let is_reasoning = id_lower.starts_with("o1")
        || id_lower.starts_with("o3")
        || (id_lower.contains("reasoning") && !id_lower.contains("non-reasoning"));

    // Vision LLM
    let is_vision = id_lower.contains("vision") || id_lower.contains("4o");

    if is_reasoning && is_vision {
        return ModelType::FULL_LLM;
    }

    if is_vision {
        return ModelType::VISION_LLM;
    }

    if is_reasoning {
        return ModelType::REASONING_LLM;
    }

    // Default to standard LLM
    ModelType::LLM
}

/// Infer provider type from model ID string.
fn infer_provider_from_id(id: &str) -> Option<ProviderType> {
    let id_lower = id.to_lowercase();

    // OpenAI models
    if id_lower.starts_with("gpt")
        || id_lower.starts_with("o1")
        || id_lower.starts_with("o3")
        || id_lower.starts_with("dall-e")
        || id_lower.starts_with("whisper")
        || id_lower.starts_with("tts")
        || id_lower.starts_with("text-embedding")
        || id_lower.starts_with("babbage")
        || id_lower.starts_with("davinci")
        || id_lower.contains("omni")
    {
        return Some(ProviderType::OpenAI);
    }

    // xAI/Grok models
    if id_lower.starts_with("grok") {
        return Some(ProviderType::XAI);
    }

    // Anthropic Claude models
    if id_lower.starts_with("claude") {
        return Some(ProviderType::Anthropic);
    }

    // Google Gemini models
    if id_lower.starts_with("gemini") {
        return Some(ProviderType::Gemini);
    }

    None
}

/// Resolve the API key to use for model discovery.
///
/// Priority: per-provider env var > config.api_key > None (wildcard).
fn resolve_discovery_api_key(
    provider: Option<&ProviderType>,
    url: &str,
    config_api_key: Option<&str>,
) -> Option<String> {
    // 1. Try per-provider admin key from env var
    if let Some(env_name) = provider.and_then(|p| p.admin_key_env_var()) {
        if let Some(key) = std::env::var(env_name).ok().filter(|k| !k.is_empty()) {
            debug!("Using {} for model discovery on {}", env_name, url);
            return Some(key);
        }
    }

    // 2. Fall back to config api_key (from --api-key)
    if let Some(key) = config_api_key {
        debug!("Using --api-key for model discovery on {}", url);
        return Some(key.to_string());
    }

    None
}

/// Fetch models from /v1/models endpoint.
async fn fetch_models(
    url: &str,
    api_key: Option<&str>,
    provider: Option<&ProviderType>,
) -> Result<Vec<ModelCard>, String> {
    let base_url = url.trim_end_matches('/');
    let models_url = format!("{base_url}/v1/models");

    let mut req = HTTP_CLIENT.get(&models_url);
    if let Some(key) = api_key {
        if provider.is_some_and(|p| p.uses_x_api_key()) {
            req = req.header("x-api-key", key);
        } else {
            req = req.bearer_auth(key);
        }
    }

    let response = req
        .send()
        .await
        .map_err(|e| format!("Failed to connect to {models_url}: {e}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "Server returned status {} from {}",
            response.status(),
            models_url
        ));
    }

    let models_response: ModelsResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse models response: {e}"))?;

    debug!(
        "Fetched {} raw models from {}",
        models_response.data.len(),
        url
    );

    let mut model_cards = group_models_into_cards(models_response.data);
    apply_provider_hint(&mut model_cards, provider);

    debug!(
        "Grouped into {} model cards with aliases",
        model_cards.len()
    );

    Ok(model_cards)
}

/// Step 1: Discover models from external /v1/models endpoint.
pub struct DiscoverModelsStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for DiscoverModelsStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        if context.data.worker_kind != Some(WorkerKind::External) {
            return Ok(StepResult::Skip);
        }

        let config = &context.data.config;
        let provider = ProviderType::from_url(&config.url);

        // Resolve discovery API key: env var admin key > config.api_key > None (wildcard)
        let discovery_key =
            resolve_discovery_api_key(provider.as_ref(), &config.url, config.api_key.as_deref());

        if discovery_key.is_none() {
            info!(
                "No API key provided for {} - using wildcard mode (accepts any model). \
                 User's Authorization header will be forwarded to backend.",
                config.url
            );
            // Leave model_cards empty for wildcard mode
            return Ok(StepResult::Success);
        }

        debug!("Discovering models from external endpoint {}", config.url);

        let model_cards = fetch_models(&config.url, discovery_key.as_deref(), provider.as_ref())
            .await
            .map_err(|e| WorkflowError::StepFailed {
                step_id: StepId::new("discover_models"),
                message: format!("Failed to discover models from {}: {}", config.url, e),
            })?;

        if model_cards.is_empty() {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("discover_models"),
                message: format!("No models discovered from {}", config.url),
            });
        }

        info!(
            "Discovered {} models from {}: {:?}",
            model_cards.len(),
            config.url,
            model_cards.iter().map(|c| &c.id).collect::<Vec<_>>()
        );

        context.data.model_cards = model_cards;
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::ProviderType;
    use serde_json::json;

    use super::{apply_provider_hint, build_model_cards, ModelInfo, ModelsResponse};

    #[test]
    fn apply_provider_hint_uses_worker_provider_for_prefixed_ids() {
        let mut cards = build_model_cards(vec![ModelInfo {
            id: "models/gemini-2.5-pro".to_string(),
            aliases: Vec::new(),
            object: "model".to_string(),
            created: Some(1_752_019_200),
            owned_by: None,
        }]);

        assert_eq!(cards[0].provider, None);

        apply_provider_hint(&mut cards, Some(&ProviderType::Gemini));

        assert_eq!(cards[0].provider, Some(ProviderType::Gemini));
    }

    #[test]
    fn models_response_tolerates_malformed_aliases_and_skips_bad_rows() {
        let response: ModelsResponse = serde_json::from_value(json!({
            "data": [
                {
                    "id": "grok-4-0709",
                    "aliases": null,
                    "object": "model",
                    "created": 1_752_019_200
                },
                {
                    "id": "grok-4-fast-reasoning",
                    "aliases": "grok-4",
                    "object": "model",
                    "created": 1_752_019_200
                },
                {
                    "id": 42,
                    "object": "model",
                    "created": 1_752_019_199
                }
            ],
            "object": "list"
        }))
        .expect("bad rows should be skipped and bad aliases tolerated");

        assert_eq!(response.data.len(), 2);
        assert_eq!(response.data[0].id, "grok-4-0709");
        assert_eq!(response.data[0].aliases, Vec::<String>::new());
        assert_eq!(response.data[1].id, "grok-4-fast-reasoning");
        assert_eq!(response.data[1].aliases, Vec::<String>::new());
    }
}
