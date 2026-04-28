//! Builder for ResponsesResponse
//!
//! Provides an ergonomic fluent API for constructing ResponsesResponse instances.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    common::{ConversationRef, PromptCacheRetention},
    responses::*,
};

/// Builder for ResponsesResponse
///
/// Provides a fluent interface for constructing responses with sensible defaults.
#[must_use = "Builder does nothing until .build() is called"]
#[derive(Clone, Debug)]
pub struct ResponsesResponseBuilder {
    id: String,
    object: String,
    created_at: i64,
    completed_at: Option<i64>,
    background: Option<bool>,
    conversation: Option<ConversationRef>,
    status: ResponseStatus,
    error: Option<Value>,
    incomplete_details: Option<Value>,
    billing: Option<Value>,
    instructions: Option<String>,
    max_output_tokens: Option<u32>,
    max_tool_calls: Option<u32>,
    frequency_penalty: Option<f32>,
    model: String,
    output: Vec<ResponseOutputItem>,
    parallel_tool_calls: bool,
    previous_response_id: Option<String>,
    moderation: Option<Value>,
    presence_penalty: Option<f32>,
    prompt_cache_key: Option<String>,
    prompt_cache_retention: Option<PromptCacheRetention>,
    reasoning: Option<Value>,
    store: bool,
    service_tier: Option<ServiceTier>,
    temperature: Option<f32>,
    text: Option<TextConfig>,
    tool_choice: Value,
    tools: Vec<ResponseTool>,
    top_logprobs: Option<u32>,
    top_p: Option<f32>,
    truncation: Option<Truncation>,
    usage: Option<ResponsesUsage>,
    user: Option<String>,
    safety_identifier: Option<String>,
    metadata: HashMap<String, Value>,
}

impl ResponsesResponseBuilder {
    /// Create a new builder with required fields
    ///
    /// # Arguments
    /// - `id`: Response ID (e.g., "resp_abc123")
    /// - `model`: Model name used for generation
    pub fn new(id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "response".to_string(),
            created_at: chrono::Utc::now().timestamp(),
            completed_at: None,
            background: None,
            conversation: None,
            status: ResponseStatus::InProgress,
            error: None,
            incomplete_details: None,
            billing: None,
            instructions: None,
            max_output_tokens: None,
            max_tool_calls: None,
            frequency_penalty: None,
            model: model.into(),
            output: Vec::new(),
            parallel_tool_calls: true,
            previous_response_id: None,
            moderation: None,
            presence_penalty: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            reasoning: None,
            store: true,
            service_tier: None,
            temperature: None,
            text: None,
            tool_choice: json!("auto"),
            tools: Vec::new(),
            top_logprobs: None,
            top_p: None,
            truncation: None,
            usage: None,
            user: None,
            safety_identifier: None,
            metadata: HashMap::new(),
        }
    }

    /// Copy common fields from a ResponsesRequest
    ///
    /// This populates fields like instructions, max_output_tokens, temperature, etc.
    /// from the original request, making it easy to construct a response that mirrors
    /// the request parameters.
    ///
    pub fn copy_from_request(mut self, request: &ResponsesRequest) -> Self {
        self.instructions.clone_from(&request.instructions);
        self.max_output_tokens = request.max_output_tokens;
        self.max_tool_calls = request.max_tool_calls;
        self.frequency_penalty = request.frequency_penalty;
        self.parallel_tool_calls = request.parallel_tool_calls.unwrap_or(true);
        self.previous_response_id
            .clone_from(&request.previous_response_id);
        self.presence_penalty = request.presence_penalty;
        self.prompt_cache_key.clone_from(&request.prompt_cache_key);
        self.prompt_cache_retention = request.prompt_cache_retention;
        self.store = request.store.unwrap_or(true);
        self.service_tier.clone_from(&request.service_tier);
        self.background = request.background;
        // OpenAI's response shape echoes `conversation` as
        // `{ "id": "..." }`; render the canonical Object form
        // regardless of whether the request used the `Id(...)` or
        // `Object { id }` wire shape.
        self.conversation = request
            .conversation
            .as_ref()
            .map(|c| ConversationRef::Object {
                id: c.as_id().to_string(),
            });
        self.temperature = request.temperature;
        self.tool_choice = ResponsesToolChoice::serialize_to_value(request.tool_choice.as_ref());
        self.tools = request.tools.clone().unwrap_or_default();
        self.top_logprobs = request.top_logprobs;
        self.top_p = request.top_p;
        self.truncation.clone_from(&request.truncation);
        self.text.clone_from(&request.text);
        self.user.clone_from(&request.user);
        self.safety_identifier
            .clone_from(&request.safety_identifier);
        self.reasoning = request
            .reasoning
            .as_ref()
            .and_then(|reasoning| serde_json::to_value(reasoning).ok());
        self.metadata = request.metadata.clone().unwrap_or_default();
        self
    }

    /// Set the object type (default: "response")
    pub fn object(mut self, object: impl Into<String>) -> Self {
        self.object = object.into();
        self
    }

    /// Set the creation timestamp (default: current time)
    pub fn created_at(mut self, timestamp: i64) -> Self {
        self.created_at = timestamp;
        self
    }

    /// Set the completion timestamp. Populate when the response reaches a
    /// terminal status (`completed`, `incomplete`, `failed`, `cancelled`).
    pub fn completed_at(mut self, timestamp: i64) -> Self {
        self.completed_at = Some(timestamp);
        self
    }

    /// Mark the response as created in background mode.
    pub fn background(mut self, background: bool) -> Self {
        self.background = Some(background);
        self
    }

    /// Set the linked conversation ID. Stored in `Object` form so the
    /// echoed `conversation` field serializes as `{ "id": "..." }`,
    /// matching OpenAI's response shape.
    pub fn conversation(mut self, conversation: impl Into<String>) -> Self {
        self.conversation = Some(ConversationRef::Object {
            id: conversation.into(),
        });
        self
    }

    /// Set the conversation from a [`ConversationRef`], normalised to the
    /// canonical `{ "id": "..." }` `Object` form. Mirrors the
    /// `copy_from_request` echo path so every response builder route emits
    /// the same wire shape regardless of which input variant the caller
    /// originally received.
    pub fn conversation_ref(mut self, conversation: ConversationRef) -> Self {
        self.conversation = Some(ConversationRef::Object {
            id: conversation.as_id().to_string(),
        });
        self
    }

    /// Set the response status
    pub fn status(mut self, status: ResponseStatus) -> Self {
        self.status = status;
        self
    }

    /// Set error information (if status is failed)
    pub fn error(mut self, error: Value) -> Self {
        self.error = Some(error);
        self
    }

    /// Set incomplete details (if response was truncated)
    pub fn incomplete_details(mut self, details: Value) -> Self {
        self.incomplete_details = Some(details);
        self
    }

    /// Set system instructions
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Set max output tokens
    pub fn max_output_tokens(mut self, tokens: u32) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    /// Set output items
    pub fn output(mut self, output: Vec<ResponseOutputItem>) -> Self {
        self.output = output;
        self
    }

    /// Add a single output item
    pub fn add_output(mut self, item: ResponseOutputItem) -> Self {
        self.output.push(item);
        self
    }

    /// Set whether parallel tool calls are enabled
    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = enabled;
        self
    }

    /// Set previous response ID (if continuation)
    pub fn previous_response_id(mut self, id: impl Into<String>) -> Self {
        self.previous_response_id = Some(id.into());
        self
    }

    /// Set reasoning information
    pub fn reasoning(mut self, reasoning: ReasoningInfo) -> Self {
        self.reasoning = Some(json!(reasoning));
        self
    }

    /// Set whether the response is stored
    pub fn store(mut self, store: bool) -> Self {
        self.store = store;
        self
    }

    /// Set temperature setting
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set text format settings if provided (handles Option)
    pub fn maybe_text(mut self, text: Option<TextConfig>) -> Self {
        if let Some(t) = text {
            self.text = Some(t);
        }
        self
    }

    /// Set tool choice setting
    pub fn tool_choice(mut self, tool_choice: impl Into<String>) -> Self {
        self.tool_choice = Value::String(tool_choice.into());
        self
    }

    /// Set available tools
    pub fn tools(mut self, tools: Vec<ResponseTool>) -> Self {
        self.tools = tools;
        self
    }

    /// Set top-p setting
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set truncation strategy
    pub fn truncation(mut self, truncation: Truncation) -> Self {
        self.truncation = Some(truncation);
        self
    }

    /// Set usage information
    pub fn usage(mut self, usage: ResponsesUsage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Set usage if provided (handles Option)
    pub fn maybe_usage(mut self, usage: Option<ResponsesUsage>) -> Self {
        if let Some(u) = usage {
            self.usage = Some(u);
        }
        self
    }

    /// Copy from request if provided (handles Option)
    pub fn maybe_copy_from_request(mut self, request: Option<&ResponsesRequest>) -> Self {
        if let Some(req) = request {
            self = self.copy_from_request(req);
        }
        self
    }

    /// Set user identifier
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Set safety identifier
    pub fn safety_identifier(mut self, identifier: impl Into<String>) -> Self {
        self.safety_identifier = Some(identifier.into());
        self
    }

    /// Set metadata
    pub fn metadata(mut self, metadata: HashMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Add a single metadata entry
    pub fn add_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Build the ResponsesResponse
    pub fn build(self) -> ResponsesResponse {
        ResponsesResponse {
            id: self.id,
            object: self.object,
            created_at: self.created_at,
            completed_at: self.completed_at,
            background: self.background,
            conversation: self.conversation,
            status: self.status,
            error: self.error,
            incomplete_details: self.incomplete_details,
            billing: self.billing,
            instructions: self.instructions,
            max_output_tokens: self.max_output_tokens,
            max_tool_calls: self.max_tool_calls,
            frequency_penalty: self.frequency_penalty,
            model: self.model,
            output: self.output,
            parallel_tool_calls: self.parallel_tool_calls,
            previous_response_id: self.previous_response_id,
            moderation: self.moderation,
            presence_penalty: self.presence_penalty,
            prompt_cache_key: self.prompt_cache_key,
            prompt_cache_retention: self.prompt_cache_retention,
            reasoning: self.reasoning,
            store: self.store,
            service_tier: self.service_tier,
            temperature: self.temperature,
            text: self.text,
            tool_choice: self.tool_choice,
            tools: self.tools,
            top_logprobs: self.top_logprobs,
            top_p: self.top_p,
            truncation: self.truncation,
            usage: self.usage,
            user: self.user,
            safety_identifier: self.safety_identifier,
            metadata: self.metadata,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_minimal() {
        let response = ResponsesResponse::builder("resp_123", "gpt-4").build();

        assert_eq!(response.id, "resp_123");
        assert_eq!(response.model, "gpt-4");
        assert_eq!(response.object, "response");
        assert_eq!(response.status, ResponseStatus::InProgress);
        assert!(response.output.is_empty());
        assert!(response.parallel_tool_calls);
        assert!(response.store);
    }

    #[test]
    fn test_build_complete() {
        let response = ResponsesResponse::builder("resp_123", "gpt-4")
            .status(ResponseStatus::Completed)
            .instructions("You are a helpful assistant")
            .max_output_tokens(1000)
            .temperature(0.7)
            .top_p(0.9)
            .parallel_tool_calls(false)
            .store(false)
            .build();

        assert_eq!(response.status, ResponseStatus::Completed);
        assert_eq!(
            response.instructions.as_ref().unwrap(),
            "You are a helpful assistant"
        );
        assert_eq!(response.max_output_tokens, Some(1000));
        assert_eq!(response.temperature, Some(0.7));
        assert_eq!(response.top_p, Some(0.9));
        assert!(!response.parallel_tool_calls);
        assert!(!response.store);
    }

    #[test]
    fn test_copy_from_request() {
        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("test".to_string()),
            instructions: Some("Be helpful".to_string()),
            max_output_tokens: Some(500),
            temperature: Some(0.8),
            top_p: Some(0.95),
            parallel_tool_calls: Some(false),
            store: Some(false),
            user: Some("user_123".to_string()),
            metadata: Some(HashMap::from([(
                "key".to_string(),
                serde_json::json!("value"),
            )])),
            ..Default::default()
        };

        let response = ResponsesResponse::builder("resp_456", "gpt-4")
            .copy_from_request(&request)
            .status(ResponseStatus::Completed)
            .build();

        assert_eq!(response.instructions.as_ref().unwrap(), "Be helpful");
        assert_eq!(response.max_output_tokens, Some(500));
        assert_eq!(response.temperature, Some(0.8));
        assert_eq!(response.top_p, Some(0.95));
        assert!(!response.parallel_tool_calls);
        assert!(!response.store);
        assert_eq!(response.user.as_ref().unwrap(), "user_123");
        assert_eq!(
            response.metadata.get("key").unwrap(),
            &serde_json::json!("value")
        );
    }

    #[test]
    fn test_add_output_items() {
        let response = ResponsesResponse::builder("resp_789", "gpt-4")
            .add_output(ResponseOutputItem::Message {
                id: "msg_1".to_string(),
                role: "assistant".to_string(),
                content: vec![],
                status: "completed".to_string(),
                phase: None,
            })
            .add_output(ResponseOutputItem::Message {
                id: "msg_2".to_string(),
                role: "assistant".to_string(),
                content: vec![],
                status: "completed".to_string(),
                phase: None,
            })
            .build();

        assert_eq!(response.output.len(), 2);
    }

    #[test]
    fn test_add_metadata() {
        let response = ResponsesResponse::builder("resp_101", "gpt-4")
            .add_metadata("key1", serde_json::json!("value1"))
            .add_metadata("key2", serde_json::json!(42))
            .build();

        assert_eq!(response.metadata.len(), 2);
        assert_eq!(response.metadata.get("key1").unwrap(), "value1");
        assert_eq!(response.metadata.get("key2").unwrap(), 42);
    }
}
