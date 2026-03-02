use openai_protocol::rerank::{RerankRequest, RerankResponse};

use crate::{transport::Transport, SmgError};

/// Rerank API (`/v1/rerank`).
pub struct Rerank {
    transport: Transport,
}

impl Rerank {
    pub(crate) fn new(transport: Transport) -> Self {
        Self { transport }
    }

    /// Rerank documents by relevance to a query.
    pub async fn create(&self, request: &RerankRequest) -> Result<RerankResponse, SmgError> {
        let resp = self.transport.post("/v1/rerank", request).await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }
}
