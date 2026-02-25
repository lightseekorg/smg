use serde::{Deserialize, Serialize};

use crate::{transport::Transport, SmgError};

/// A model object returned by `/v1/models`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owned_by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root: Option<String>,
}

/// Response from `GET /v1/models`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelObject>,
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
