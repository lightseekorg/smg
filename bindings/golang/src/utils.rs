//! Utility functions for FFI

use llm_tokenizer::{
    chat_template::{ThinkingKeyName, ThinkingToggle},
    traits::Tokenizer,
};
use openai_protocol::chat::ChatCompletionRequest;
use uuid::Uuid;

/// Helper function to generate tool call ID (matches router implementation)
pub fn generate_tool_call_id(
    model: &str,
    function_name: &str,
    index: usize,
    history_tool_calls_count: usize,
) -> String {
    if model.to_lowercase().contains("kimi") {
        // KimiK2 format: functions.{name}:{global_index}
        format!(
            "functions.{}:{}",
            function_name,
            history_tool_calls_count + index
        )
    } else {
        // Standard OpenAI format: call_{24-char-uuid}
        format!("call_{}", &Uuid::now_v7().simple().to_string()[..24])
    }
}

/// Determine whether the SGLang gRPC request should ask the backend to count
/// reasoning tokens for this chat request.
pub(crate) fn chat_requires_reasoning(
    request: &ChatCompletionRequest,
    tokenizer: &dyn Tokenizer,
) -> bool {
    let user_thinking = request
        .chat_template_kwargs
        .as_ref()
        .and_then(|kwargs| match tokenizer.thinking_key_name() {
            Some(ThinkingKeyName::EnableThinking) => kwargs.get("enable_thinking"),
            Some(ThinkingKeyName::Thinking) => kwargs.get("thinking"),
            None => None,
        })
        .and_then(|value| value.as_bool());

    match tokenizer.thinking_toggle() {
        ThinkingToggle::None => false,
        ThinkingToggle::DefaultOn => user_thinking != Some(false),
        ThinkingToggle::DefaultOff => user_thinking == Some(true),
    }
}
