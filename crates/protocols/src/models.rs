//! OpenAI-compatible `/v1/models` response types.
//!
//! Provides [`ListModelsResponse`] built from in-memory [`ModelCard`] data,
//! ensuring every router returns the same format.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::model_card::ModelCard;

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
}
