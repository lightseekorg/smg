use openai_protocol::embedding::{EmbeddingRequest, EmbeddingResponse};

use crate::{transport::Transport, SmgError};

/// Embeddings API (`/v1/embeddings`).
pub struct Embeddings {
    transport: Transport,
}

impl Embeddings {
    pub(crate) fn new(transport: Transport) -> Self {
        Self { transport }
    }

    /// Create embeddings for the given input.
    pub async fn create(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse, SmgError> {
        let resp = self.transport.post("/v1/embeddings", request).await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }
}
