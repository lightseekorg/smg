use openai_protocol::chat::{
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamResponse,
};

use crate::{
    streaming::{sse_stream, TypedStream},
    transport::Transport,
    SmgError,
};

/// Chat completions API.
pub struct Chat {
    transport: Transport,
}

impl Chat {
    pub(crate) fn new(transport: Transport) -> Self {
        Self { transport }
    }

    /// Create a non-streaming chat completion.
    pub async fn create(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, SmgError> {
        let resp = self.transport.post("/v1/chat/completions", request).await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }

    /// Create a streaming chat completion.
    ///
    /// Returns a `TypedStream` that yields `ChatCompletionStreamResponse` chunks.
    ///
    /// ```no_run
    /// # use futures::StreamExt;
    /// # async fn example(client: smg_client::SmgClient) -> Result<(), smg_client::SmgError> {
    /// let req = openai_protocol::chat::ChatCompletionRequest {
    ///     stream: true,
    ///     ..Default::default()
    /// };
    /// let mut stream = client.chat().create_stream(&req).await?;
    /// while let Some(chunk) = stream.next().await {
    ///     let chunk = chunk?;
    ///     // Process chunk
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn create_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<TypedStream<ChatCompletionStreamResponse>, SmgError> {
        let resp = self
            .transport
            .post_stream("/v1/chat/completions", request)
            .await?;
        let byte_stream = resp.bytes_stream();
        let events = sse_stream(byte_stream);
        Ok(TypedStream::new(events))
    }
}
