use openai_protocol::messages::{CreateMessageRequest, Message, MessageStreamEvent};

use crate::{
    streaming::{sse_stream, SseEvent, TypedStream},
    transport::Transport,
    SmgError,
};

/// Anthropic Messages API (`/v1/messages`).
pub struct Messages {
    transport: Transport,
}

impl Messages {
    pub(crate) fn new(transport: Transport) -> Self {
        Self { transport }
    }

    /// Create a non-streaming message.
    pub async fn create(&self, request: &CreateMessageRequest) -> Result<Message, SmgError> {
        let resp = self.transport.post("/v1/messages", request).await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }

    /// Create a streaming message.
    ///
    /// Returns a `TypedStream` that yields `MessageStreamEvent` variants.
    /// Anthropic SSE uses `event: type\ndata: {...}` format — the event type
    /// is embedded in each `MessageStreamEvent` variant via serde tag.
    pub async fn create_stream(
        &self,
        request: &CreateMessageRequest,
    ) -> Result<TypedStream<MessageStreamEvent>, SmgError> {
        let resp = self.transport.post_stream("/v1/messages", request).await?;
        let byte_stream = resp.bytes_stream();
        let raw_events = sse_stream(byte_stream);

        // Anthropic events need the SSE `event:` field injected into JSON
        // as `"type"` for serde deserialization of the tagged enum.
        let typed_events = futures::stream::StreamExt::map(raw_events, inject_event_type);
        Ok(TypedStream::new(typed_events))
    }
}

/// For Anthropic SSE, the event type is in the `event:` field, not in the JSON body.
/// Inject it as `"type"` into the JSON data so `MessageStreamEvent` can deserialize.
fn inject_event_type(result: Result<SseEvent, SmgError>) -> Result<SseEvent, SmgError> {
    let mut event = result?;
    if let Some(ref event_type) = event.event {
        // Try to parse and inject; if it fails, just pass through.
        if let Ok(mut value) = serde_json::from_str::<serde_json::Value>(&event.data) {
            if let Some(obj) = value.as_object_mut() {
                if !obj.contains_key("type") {
                    obj.insert("type".to_string(), serde_json::json!(event_type));
                    event.data = value.to_string();
                }
            }
        }
    }
    Ok(event)
}
