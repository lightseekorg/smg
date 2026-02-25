use serde::{Deserialize, Serialize};

use crate::{transport::Transport, SmgError};

/// A model object returned by `/v1/models`.
///
/// Accepts both OpenAI format (`object`, `created`, `owned_by`) and
/// Anthropic format (`display_name`, `created_at`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelObject {
    pub id: String,
    #[serde(default = "default_model_object")]
    pub object: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owned_by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root: Option<String>,
    /// Display name (Anthropic format).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// ISO 8601 creation timestamp (Anthropic format).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
}

fn default_model_object() -> String {
    "model".to_string()
}

/// Response from `GET /v1/models`.
///
/// Accepts both OpenAI format (`object: "list"`) and Anthropic format
/// (`has_more`, `first_id`, `last_id` pagination fields).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelList {
    #[serde(default = "default_list_object")]
    pub object: String,
    pub data: Vec<ModelObject>,
    /// Whether more results are available (Anthropic pagination).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub has_more: Option<bool>,
    /// ID of the first model in the list (Anthropic pagination).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_id: Option<String>,
    /// ID of the last model in the list (Anthropic pagination).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_id: Option<String>,
}

fn default_list_object() -> String {
    "list".to_string()
}

/// Models API (`/v1/models`).
pub struct Models {
    transport: Transport,
}

impl Models {
    pub(crate) fn new(transport: Transport) -> Self {
        Self { transport }
    }

    /// List all available models.
    pub async fn list(&self) -> Result<ModelList, SmgError> {
        let resp = self.transport.get("/v1/models").await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }
}
