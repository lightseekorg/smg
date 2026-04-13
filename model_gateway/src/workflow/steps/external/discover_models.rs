//! Model discovery step for external API endpoints.

use std::{collections::HashMap, time::Duration};

use async_trait::async_trait;
use once_cell::sync::Lazy;
use openai_protocol::{model_card::ModelCard, model_type::ModelType, worker::ProviderType};
use regex::Regex;
use reqwest::Client;
use serde::Deserialize;
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

// Regex to strip date suffix: -YYYY-MM-DD or -YYYY-MM
#[expect(
    clippy::expect_used,
    reason = "Lazy static initialization — compile-time constant regex pattern cannot fail"
)]
static DATE_SUFFIX_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"-\d{4}-\d{2}(-\d{2})?$").expect("Invalid date regex"));

// xAI Grok revision suffixes commonly look like -MMDD (for example grok-4-0709).
#[expect(
    clippy::expect_used,
    reason = "Lazy static initialization — compile-time constant regex pattern cannot fail"
)]
static XAI_REVISION_SUFFIX_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(grok-\d+)-(\d{4})$").expect("Invalid xAI revision regex"));

/// OpenAI /v1/models response format.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelsResponse {
    pub data: Vec<ModelInfo>,
    #[serde(default)]
    pub object: String,
}

/// Individual model information from /v1/models.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: Option<u64>,
    #[serde(default)]
    pub owned_by: Option<String>,
}

fn alias_group_key(id: &str) -> String {
    if let Some(captures) = XAI_REVISION_SUFFIX_PATTERN.captures(id) {
        if let Some(base) = captures.get(1) {
            return base.as_str().to_string();
        }
    }

    DATE_SUFFIX_PATTERN.replace(id, "").to_string()
}

fn xai_revision_rank(id: &str) -> Option<u16> {
    XAI_REVISION_SUFFIX_PATTERN
        .captures(id)
        .and_then(|captures| captures.get(2))
        .and_then(|suffix| suffix.as_str().parse::<u16>().ok())
}

fn select_primary_and_aliases(group_key: &str, variants: Vec<ModelInfo>) -> (String, Vec<String>) {
    let primary_id = if variants.iter().any(|variant| variant.id == group_key) {
        group_key.to_string()
    } else if variants
        .iter()
        .all(|variant| xai_revision_rank(&variant.id).is_some())
    {
        variants
            .iter()
            .max_by_key(|variant| {
                (
                    variant.created.unwrap_or(0),
                    xai_revision_rank(&variant.id).unwrap_or(0),
                    variant.id.as_str(),
                )
            })
            .map(|variant| variant.id.clone())
            .unwrap_or_else(|| group_key.to_string())
    } else {
        variants
            .iter()
            .map(|variant| variant.id.as_str())
            .min_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)))
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| group_key.to_string())
    };

    let mut aliases: Vec<String> = variants
        .iter()
        .filter_map(|variant| (variant.id != primary_id).then(|| variant.id.clone()))
        .collect();

    if group_key != primary_id && !aliases.iter().any(|alias| alias == group_key) {
        aliases.push(group_key.to_string());
    }

    aliases.sort();
    aliases.dedup();

    (primary_id, aliases)
}

/// Group models by base name and create ModelCards with aliases.
///
/// # Example
/// Input:  `["gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20"]`
/// Output: `ModelCard { id: "gpt-4o", aliases: ["gpt-4o-2024-05-13", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20"] }`
pub fn group_models_into_cards(models: Vec<ModelInfo>) -> Vec<ModelCard> {
    // Group model IDs by alias family key.
    let mut groups: HashMap<String, Vec<ModelInfo>> = HashMap::new();
    for model in models {
        let base = alias_group_key(&model.id);
        groups.entry(base).or_default().push(model);
    }

    // Create ModelCard for each group
    groups
        .into_iter()
        .map(|(group_key, variants)| {
            let (primary_id, aliases) = select_primary_and_aliases(&group_key, variants);
            let model_type = infer_model_type_from_id(&primary_id);
            let provider = infer_provider_from_id(&primary_id);

            let mut card = ModelCard::new(&primary_id)
                .with_aliases(aliases)
                .with_model_type(model_type);

            if let Some(p) = provider {
                card = card.with_provider(p);
            }

            card
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{group_models_into_cards, ModelInfo};

    fn model(id: &str) -> ModelInfo {
        ModelInfo {
            id: id.to_string(),
            object: "model".to_string(),
            created: None,
            owned_by: None,
        }
    }

    #[test]
    fn groups_openai_date_variants_under_stable_name() {
        let cards = group_models_into_cards(vec![
            model("gpt-4o"),
            model("gpt-4o-2024-08-06"),
            model("gpt-4o-2024-11-20"),
        ]);

        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].id, "gpt-4o");
        assert!(cards[0]
            .aliases
            .iter()
            .any(|alias| alias == "gpt-4o-2024-08-06"));
        assert!(cards[0]
            .aliases
            .iter()
            .any(|alias| alias == "gpt-4o-2024-11-20"));
    }

    #[test]
    fn groups_xai_revisioned_model_under_family_alias() {
        let cards = group_models_into_cards(vec![model("grok-4-0709")]);

        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].id, "grok-4-0709");
        assert!(cards[0].aliases.iter().any(|alias| alias == "grok-4"));
    }

    #[test]
    fn picks_latest_xai_revision_as_primary_model() {
        let mut older = model("grok-4-0709");
        older.created = Some(100);
        let mut newer = model("grok-4-0812");
        newer.created = Some(200);

        let cards = group_models_into_cards(vec![older, newer]);

        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].id, "grok-4-0812");
        assert!(cards[0].aliases.iter().any(|alias| alias == "grok-4"));
        assert!(cards[0].aliases.iter().any(|alias| alias == "grok-4-0709"));
    }
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

    // Vision LLM
    if id_lower.contains("vision") || id_lower.contains("4o") {
        return ModelType::VISION_LLM;
    }

    // Reasoning models
    if id_lower.starts_with("o1") || id_lower.starts_with("o3") {
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

    let model_cards = group_models_into_cards(models_response.data);

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
