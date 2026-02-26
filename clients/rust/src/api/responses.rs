use openai_protocol::responses::{ResponsesRequest, ResponsesResponse};

use crate::{transport::Transport, SmgError};

/// Responses API (`/v1/responses`).
pub struct Responses {
    transport: Transport,
}

impl Responses {
    pub(crate) fn new(transport: Transport) -> Self {
        Self { transport }
    }

    /// Create a new response.
    pub async fn create(&self, request: &ResponsesRequest) -> Result<ResponsesResponse, SmgError> {
        let resp = self.transport.post("/v1/responses", request).await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }

    /// Retrieve an existing response by ID.
    pub async fn get(&self, response_id: &str) -> Result<ResponsesResponse, SmgError> {
        let resp = self
            .transport
            .get(&format!("/v1/responses/{response_id}"))
            .await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }

    /// Delete a response by ID.
    ///
    /// Note: this endpoint is not yet implemented on the server (returns 501).
    pub async fn delete(&self, response_id: &str) -> Result<(), SmgError> {
        self.transport
            .delete(&format!("/v1/responses/{response_id}"))
            .await?;
        Ok(())
    }

    /// Cancel an in-progress response.
    pub async fn cancel(&self, response_id: &str) -> Result<ResponsesResponse, SmgError> {
        let resp = self
            .transport
            .post(
                &format!("/v1/responses/{response_id}/cancel"),
                &serde_json::json!({}),
            )
            .await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }

    /// List input items for a response.
    pub async fn list_input_items(&self, response_id: &str) -> Result<serde_json::Value, SmgError> {
        let resp = self
            .transport
            .get(&format!("/v1/responses/{response_id}/input_items"))
            .await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }
}
