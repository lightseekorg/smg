use openai_protocol::completion::{
    CompletionRequest, CompletionResponse, CompletionStreamResponse,
};

use crate::{
    streaming::{sse_stream, TypedStream},
    transport::Transport,
    SmgError,
};

/// Legacy completions API (`/v1/completions`).
pub struct Completions {
    transport: Transport,
}

impl Completions {
    pub(crate) fn new(transport: Transport) -> Self {
        Self { transport }
    }

    /// Create a non-streaming completion.
    pub async fn create(
        &self,
        request: &CompletionRequest,
    ) -> Result<CompletionResponse, SmgError> {
        let resp = self.transport.post("/v1/completions", request).await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }

    /// Create a streaming completion.
    pub async fn create_stream(
        &self,
        request: &CompletionRequest,
    ) -> Result<TypedStream<CompletionStreamResponse>, SmgError> {
        let resp = self
            .transport
            .post_stream("/v1/completions", request)
            .await?;
        let byte_stream = resp.bytes_stream();
        let events = sse_stream(byte_stream);
        Ok(TypedStream::new(events))
    }
}
