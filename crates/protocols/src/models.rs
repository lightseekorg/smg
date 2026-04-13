//! OpenAI-compatible `/v1/models` response types.
//!
//! Provides [`ListModelsResponse`] built from in-memory [`ModelCard`] data,
//! ensuring every router returns the same format.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{model_card::ModelCard, worker::ProviderType};

#[derive(Debug, Clone)]
struct UpstreamModelInfo {
    id: String,
    created: Option<u64>,
}

/// A single model entry in the `/v1/models` response.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ModelObject {
    /// Model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct").
    pub id: String,
    /// Always `"model"`.
    pub object: String,
    /// Unix timestamp of model creation (0 = unknown).
    pub created: i64,
    /// Who owns/hosts the model.
    pub owned_by: String,
}

/// Response body for `GET /v1/models`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ListModelsResponse {
    /// Always `"list"`.
    pub object: String,
    /// Available models.
    pub data: Vec<ModelObject>,
}

impl ListModelsResponse {
    /// Build a response from an iterator of [`ModelCard`]s.
    ///
    /// Deduplicates by model ID (first occurrence wins).
    pub fn from_model_cards(cards: impl IntoIterator<Item = ModelCard>) -> Self {
        let mut seen = HashSet::new();
        let data = cards
            .into_iter()
            .filter(|card| seen.insert(card.id.clone()))
            .map(ModelCard::into_model_object)
            .collect();
        Self {
            object: "list".to_owned(),
            data,
        }
    }

    /// Parse an upstream `/v1/models` JSON response into [`ModelCard`]s.
    ///
    /// Handles both OpenAI and Anthropic response schemas — both use
    /// `data[].id` for the model identifier. Returns an empty vec if
    /// the JSON does not contain a valid `data` array.
    pub fn parse_upstream(json: &Value, provider: Option<ProviderType>) -> Vec<ModelCard> {
        let Some(data) = json.get("data").and_then(|d| d.as_array()) else {
            return Vec::new();
        };
        let models: Vec<_> = data
            .iter()
            .filter_map(|model| {
                let id = model.get("id").and_then(|id| id.as_str())?;
                let created = model.get("created").and_then(Value::as_u64);
                Some(UpstreamModelInfo {
                    id: id.to_string(),
                    created,
                })
            })
            .collect();

        group_upstream_models_into_cards(models, provider)
    }
}

fn is_ascii_digits(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|b| b.is_ascii_digit())
}

fn strip_date_suffix(id: &str) -> Option<String> {
    let parts: Vec<_> = id.split('-').collect();

    if parts.len() >= 4
        && is_ascii_digits(parts[parts.len() - 3], 4)
        && is_ascii_digits(parts[parts.len() - 2], 2)
        && is_ascii_digits(parts[parts.len() - 1], 2)
    {
        return Some(parts[..parts.len() - 3].join("-"));
    }

    if parts.len() >= 3
        && is_ascii_digits(parts[parts.len() - 2], 4)
        && is_ascii_digits(parts[parts.len() - 1], 2)
    {
        return Some(parts[..parts.len() - 2].join("-"));
    }

    None
}

fn xai_group_key_and_rank(id: &str) -> Option<(String, u16)> {
    let parts: Vec<_> = id.split('-').collect();
    if parts.len() == 3
        && parts[0] == "grok"
        && !parts[1].is_empty()
        && parts[1].bytes().all(|b| b.is_ascii_digit())
        && is_ascii_digits(parts[2], 4)
    {
        return parts[2]
            .parse::<u16>()
            .ok()
            .map(|rank| (format!("{}-{}", parts[0], parts[1]), rank));
    }

    None
}

fn alias_group_key(id: &str) -> String {
    if let Some((group_key, _)) = xai_group_key_and_rank(id) {
        return group_key;
    }

    strip_date_suffix(id).unwrap_or_else(|| id.to_string())
}

fn xai_revision_rank(id: &str) -> Option<u16> {
    xai_group_key_and_rank(id).map(|(_, rank)| rank)
}

fn select_primary_and_aliases(
    group_key: &str,
    variants: &[UpstreamModelInfo],
) -> (String, Vec<String>) {
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

fn group_upstream_models_into_cards(
    models: Vec<UpstreamModelInfo>,
    provider: Option<ProviderType>,
) -> Vec<ModelCard> {
    let mut groups = std::collections::BTreeMap::<String, Vec<UpstreamModelInfo>>::new();
    for model in models {
        groups
            .entry(alias_group_key(&model.id))
            .or_default()
            .push(model);
    }

    groups
        .into_iter()
        .map(|(group_key, variants)| {
            let (primary_id, aliases) = select_primary_and_aliases(&group_key, &variants);
            let mut card = ModelCard::new(primary_id).with_aliases(aliases);
            card.provider.clone_from(&provider);
            card
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::ListModelsResponse;
    use crate::worker::ProviderType;

    #[test]
    fn parse_upstream_groups_openai_date_variants_under_stable_name() {
        let json = json!({
            "object": "list",
            "data": [
                {"id": "gpt-4o", "object": "model"},
                {"id": "gpt-4o-2024-08-06", "object": "model"},
                {"id": "gpt-4o-2024-11-20", "object": "model"}
            ]
        });

        let cards = ListModelsResponse::parse_upstream(&json, Some(ProviderType::OpenAI));

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
        assert_eq!(cards[0].provider, Some(ProviderType::OpenAI));
    }

    #[test]
    fn parse_upstream_adds_xai_family_alias_for_revisioned_model() {
        let json = json!({
            "object": "list",
            "data": [
                {"id": "grok-4-0709", "object": "model", "created": 100}
            ]
        });

        let cards = ListModelsResponse::parse_upstream(&json, Some(ProviderType::XAI));

        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].id, "grok-4-0709");
        assert!(cards[0].aliases.iter().any(|alias| alias == "grok-4"));
        assert_eq!(cards[0].provider, Some(ProviderType::XAI));
    }

    #[test]
    fn parse_upstream_picks_latest_xai_revision_as_primary_model() {
        let json = json!({
            "object": "list",
            "data": [
                {"id": "grok-4-0709", "object": "model", "created": 100},
                {"id": "grok-4-0812", "object": "model", "created": 200}
            ]
        });

        let cards = ListModelsResponse::parse_upstream(&json, Some(ProviderType::XAI));

        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].id, "grok-4-0812");
        assert!(cards[0].aliases.iter().any(|alias| alias == "grok-4"));
        assert!(cards[0].aliases.iter().any(|alias| alias == "grok-4-0709"));
    }
}
