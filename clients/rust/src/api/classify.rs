use openai_protocol::classify::{ClassifyRequest, ClassifyResponse};

use crate::{transport::Transport, SmgError};

/// Classify API (`/v1/classify`).
pub struct Classify {
    transport: Transport,
}

impl Classify {
    pub(crate) fn new(transport: Transport) -> Self {
        Self { transport }
    }

    /// Classify input text into predefined categories.
    pub async fn create(&self, request: &ClassifyRequest) -> Result<ClassifyResponse, SmgError> {
        let resp = self.transport.post("/v1/classify", request).await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }
}
