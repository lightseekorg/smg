//! Conversion utilities for translating between /v1/responses and /v1/chat/completions formats
//!
//! This module implements the conversion approach where:
//! 1. ResponsesRequest → ChatCompletionRequest (for backend processing)
//! 2. ChatCompletionResponse → ResponsesResponse (for client response)
//!
//! This allows the gRPC router to reuse the existing chat pipeline infrastructure
//! without requiring Python backend changes.

use openai_protocol::{
    chat::{ChatCompletionRequest, ChatMessage, MessageContent},
    common::{FunctionCallResponse, JsonSchemaFormat, ResponseFormat, ToolCall},
    responses::{
        ResponseContentPart, ResponseInput, ResponseInputOutputItem,
        ResponseReasoningContent::ReasoningText, ResponsesRequest, StringOrContentParts,
        TextConfig, TextFormat,
    },
    UNKNOWN_MODEL_ID,
};
use tracing::warn;

use crate::routers::grpc::common::responses::utils::extract_tools_from_response_tools;

/// Convert a ResponsesRequest to ChatCompletionRequest for processing through the chat pipeline
///
/// # Conversion Logic
/// - `input` (text/items) → `messages` (chat messages)
/// - `instructions` → system message (prepended)
/// - `max_output_tokens` → `max_completion_tokens`
/// - `tools` → function tools extracted from ResponseTools
/// - `tool_choice` → passed through from request
/// - Response-specific fields (previous_response_id, conversation) are handled by router
pub(crate) fn responses_to_chat(req: &ResponsesRequest) -> Result<ChatCompletionRequest, String> {
    let mut messages = Vec::new();

    // 1. Add system message if instructions provided
    if let Some(instructions) = &req.instructions {
        messages.push(ChatMessage::System {
            content: MessageContent::Text(instructions.clone()),
            name: None,
        });
    }

    // 2. Convert input to chat messages
    match &req.input {
        ResponseInput::Text(text) => {
            // Simple text input → user message
            messages.push(ChatMessage::User {
                content: MessageContent::Text(text.clone()),
                name: None,
            });
        }
        ResponseInput::Items(items) => {
            // Structured items → convert each to appropriate chat message
            for item in items {
                match item {
                    ResponseInputOutputItem::SimpleInputMessage { content, role, .. } => {
                        // Convert SimpleInputMessage to chat message
                        let text = match content {
                            StringOrContentParts::String(s) => s.clone(),
                            StringOrContentParts::Array(parts) => {
                                // Extract text from content parts (only InputText supported)
                                parts
                                    .iter()
                                    .filter_map(|part| match part {
                                        ResponseContentPart::InputText { text } => {
                                            Some(text.as_str())
                                        }
                                        _ => None,
                                    })
                                    .collect::<Vec<_>>()
                                    .join(" ")
                            }
                        };

                        messages.push(role_to_chat_message(role.as_str(), text));
                    }
                    ResponseInputOutputItem::Message { role, content, .. } => {
                        // Extract text from content parts
                        let text = extract_text_from_content(content);

                        messages.push(role_to_chat_message(role.as_str(), text));
                    }
                    ResponseInputOutputItem::FunctionToolCall {
                        id,
                        name,
                        arguments,
                        output,
                        ..
                    } => {
                        // Tool call from history - add as assistant message with tool call
                        // followed by tool response if output exists

                        // Add assistant message with tool_calls (the LLM's decision)
                        messages.push(ChatMessage::Assistant {
                            content: None,
                            name: None,
                            tool_calls: Some(vec![ToolCall {
                                id: id.clone(),
                                tool_type: "function".to_string(),
                                function: FunctionCallResponse {
                                    name: name.clone(),
                                    arguments: Some(arguments.clone()),
                                },
                            }]),
                            reasoning_content: None,
                        });

                        // Add tool result message if output exists
                        if let Some(output_text) = output {
                            messages.push(ChatMessage::Tool {
                                content: MessageContent::Text(output_text.clone()),
                                tool_call_id: id.clone(),
                            });
                        }
                    }
                    ResponseInputOutputItem::Reasoning { content, .. } => {
                        // Reasoning content - add as assistant message with reasoning_content
                        let reasoning_text = content
                            .iter()
                            .map(|c| match c {
                                ReasoningText { text } => text.as_str(),
                            })
                            .collect::<Vec<_>>()
                            .join("\n");

                        messages.push(ChatMessage::Assistant {
                            content: None,
                            name: None,
                            tool_calls: None,
                            reasoning_content: Some(reasoning_text),
                        });
                    }
                    ResponseInputOutputItem::FunctionCallOutput {
                        call_id, output, ..
                    } => {
                        // Function call output - add as tool message
                        // Note: The function name is looked up from prev_outputs in Harmony path
                        // For Chat path, we just use the call_id
                        messages.push(ChatMessage::Tool {
                            content: MessageContent::Text(output.clone()),
                            tool_call_id: call_id.clone(),
                        });
                    }
                    ResponseInputOutputItem::McpCall { .. }
                    | ResponseInputOutputItem::McpListTools { .. }
                    | ResponseInputOutputItem::McpApprovalRequest { .. }
                    | ResponseInputOutputItem::McpApprovalResponse { .. }
                    | ResponseInputOutputItem::WebSearchCall { .. }
                    | ResponseInputOutputItem::CodeInterpreterCall { .. }
                    | ResponseInputOutputItem::FileSearchCall { .. }
                    | ResponseInputOutputItem::ImageGenerationCall { .. } => {
                        // Hosted-tool item types are projected by the
                        // shared `transcript_lower` pass before this
                        // function runs (see
                        // `routers/common/transcript_lower.rs`):
                        // hosted calls → `function_call (+ output)` pair,
                        // hosted-tool metadata (`mcp_list_tools`,
                        // approvals) is dropped. Reaching this arm means
                        // the lower pass was bypassed; treat it as a
                        // programming error rather than papering over it
                        // again.
                        warn!(
                            function = "responses_to_chat",
                            "Hosted-tool item reached chat conversion despite transcript_lower"
                        );
                        return Err("Unsupported input item type".to_string());
                    }
                    ResponseInputOutputItem::ComputerCall { .. }
                    | ResponseInputOutputItem::ComputerCallOutput { .. } => {
                        warn!(
                            function = "responses_to_chat",
                            "computer_call item reached chat conversion"
                        );
                        return Err("Unsupported input item type".to_string());
                    }
                    ResponseInputOutputItem::Compaction { .. }
                    | ResponseInputOutputItem::ItemReference { .. } => {
                        return Err("Unsupported input item type".to_string());
                    }
                    ResponseInputOutputItem::CustomToolCall { .. }
                    | ResponseInputOutputItem::CustomToolCallOutput { .. } => {
                        warn!(
                            function = "responses_to_chat",
                            "Custom tool item reached chat conversion"
                        );
                        return Err("Unsupported input item type".to_string());
                    }
                    ResponseInputOutputItem::ShellCall { .. }
                    | ResponseInputOutputItem::ShellCallOutput { .. } => {
                        warn!(
                            function = "responses_to_chat",
                            "Shell tool item reached chat conversion"
                        );
                        return Err("Unsupported input item type".to_string());
                    }
                    ResponseInputOutputItem::ApplyPatchCall { .. }
                    | ResponseInputOutputItem::ApplyPatchCallOutput { .. } => {
                        warn!(
                            function = "responses_to_chat",
                            "apply_patch item reached chat conversion"
                        );
                        return Err("Unsupported input item type".to_string());
                    }
                    // T5 schema-only: forced-cascade arm, no behavior.
                    ResponseInputOutputItem::LocalShellCall { .. }
                    | ResponseInputOutputItem::LocalShellCallOutput { .. } => {
                        return Err("Unsupported input item type".to_string());
                    }
                }
            }
        }
    }

    // Ensure we have at least one message
    if messages.is_empty() {
        return Err("Request must contain at least one message".to_string());
    }

    // 3. Extract function tools from ResponseTools.
    // MCP tools are merged by the regular agent-loop adapters after conversion.
    let function_tools = extract_tools_from_response_tools(req.tools.as_deref());
    let tools = if function_tools.is_empty() {
        None
    } else {
        Some(function_tools)
    };

    // 4. Build ChatCompletionRequest
    let is_streaming = req.stream.unwrap_or(false);

    Ok(ChatCompletionRequest {
        messages,
        model: if req.model.is_empty() {
            UNKNOWN_MODEL_ID.to_string()
        } else {
            req.model.clone()
        },
        temperature: req.temperature,
        max_completion_tokens: req.max_output_tokens,
        stream: is_streaming,
        // Preserve caller-provided stream_options (e.g. `include_obfuscation: false`
        // on the Responses API) and only default `include_usage` when the caller
        // did not set it. Non-streaming requests intentionally drop stream_options.
        stream_options: if is_streaming {
            let mut opts = req.stream_options.clone().unwrap_or_default();
            if opts.include_usage.is_none() {
                opts.include_usage = Some(true);
            }
            Some(opts)
        } else {
            None
        },
        parallel_tool_calls: req.parallel_tool_calls,
        top_logprobs: req.top_logprobs,
        top_p: req.top_p,
        skip_special_tokens: true,
        tools,
        tool_choice: req.tool_choice.as_ref().map(|tc| tc.to_chat_tool_choice()),
        response_format: map_text_to_response_format(req.text.as_ref()),
        ..Default::default()
    })
}

/// Extract text content from ResponseContentPart array. `Refusal` is
/// losslessly representable as text and is preserved verbatim. Image / file
/// parts are currently dropped; the gRPC regular path is text-only and
/// relies on the multimodal pipeline for media handling (R1/R2/R3 will
/// implement full media handling).
fn extract_text_from_content(content: &[ResponseContentPart]) -> String {
    content
        .iter()
        .filter_map(|part| match part {
            ResponseContentPart::InputText { text } => Some(text.as_str()),
            ResponseContentPart::OutputText { text, .. } => Some(text.as_str()),
            ResponseContentPart::Refusal { refusal } => Some(refusal.as_str()),
            // R1/R2/R3 will implement full media handling
            ResponseContentPart::InputImage { .. } | ResponseContentPart::InputFile { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Convert role and text to ChatMessage
fn role_to_chat_message(role: &str, text: String) -> ChatMessage {
    match role {
        "user" => ChatMessage::User {
            content: MessageContent::Text(text),
            name: None,
        },
        "assistant" => ChatMessage::Assistant {
            content: Some(MessageContent::Text(text)),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        },
        "system" => ChatMessage::System {
            content: MessageContent::Text(text),
            name: None,
        },
        _ => {
            // Unknown role, treat as user message
            ChatMessage::User {
                content: MessageContent::Text(text),
                name: None,
            }
        }
    }
}

/// Map TextConfig from Responses API to ResponseFormat for Chat API
///
/// Converts the structured output configuration from the Responses API format
/// to the Chat API format for non-Harmony models.
fn map_text_to_response_format(text: Option<&TextConfig>) -> Option<ResponseFormat> {
    let text_config = text?;
    let format = text_config.format.as_ref()?;

    match format {
        TextFormat::Text => Some(ResponseFormat::Text),
        TextFormat::JsonObject => Some(ResponseFormat::JsonObject),
        TextFormat::JsonSchema {
            name,
            schema,
            description: _,
            strict,
        } => Some(ResponseFormat::JsonSchema {
            json_schema: JsonSchemaFormat {
                name: name.clone(),
                schema: schema.clone(),
                strict: *strict,
            },
        }),
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::StreamOptions;

    use super::*;

    #[test]
    fn test_text_input_conversion() {
        let req = ResponsesRequest {
            input: ResponseInput::Text("Hello, world!".to_string()),
            instructions: Some("You are a helpful assistant.".to_string()),
            model: "gpt-4".to_string(),
            temperature: Some(0.7),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.messages.len(), 2); // system + user
        assert_eq!(chat_req.model, "gpt-4");
        assert_eq!(chat_req.temperature, Some(0.7));
    }

    #[test]
    fn test_items_input_conversion() {
        let req = ResponsesRequest {
            input: ResponseInput::Items(vec![
                ResponseInputOutputItem::Message {
                    id: "msg_1".to_string(),
                    role: "user".to_string(),
                    content: vec![ResponseContentPart::InputText {
                        text: "Hello!".to_string(),
                    }],
                    status: None,
                    phase: None,
                },
                ResponseInputOutputItem::Message {
                    id: "msg_2".to_string(),
                    role: "assistant".to_string(),
                    content: vec![ResponseContentPart::OutputText {
                        text: "Hi there!".to_string(),
                        annotations: vec![],
                        logprobs: None,
                    }],
                    status: None,
                    phase: None,
                },
            ]),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.messages.len(), 2); // user + assistant
    }

    #[test]
    fn test_empty_input_error() {
        let req = ResponsesRequest {
            input: ResponseInput::Text(String::new()),
            ..Default::default()
        };

        // Empty text should still create a user message, so this should succeed
        let result = responses_to_chat(&req);
        assert!(result.is_ok());
    }

    #[test]
    fn test_stream_options_include_obfuscation_roundtrip() {
        // Regression: ensure caller-provided stream_options (e.g. `include_obfuscation`)
        // are preserved through the Responses → Chat conversion when streaming.
        let req = ResponsesRequest {
            input: ResponseInput::Text("hi".to_string()),
            stream: Some(true),
            stream_options: Some(StreamOptions {
                include_usage: None,
                include_obfuscation: Some(false),
            }),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert!(chat_req.stream);
        let opts = chat_req
            .stream_options
            .expect("stream_options populated when streaming");
        // Caller-provided value is preserved verbatim.
        assert_eq!(opts.include_obfuscation, Some(false));
        // include_usage defaults to true when absent so downstream consumers
        // still emit the usage block at end-of-stream.
        assert_eq!(opts.include_usage, Some(true));
    }

    #[test]
    fn test_stream_options_caller_include_usage_preserved() {
        // Caller-set `include_usage` must not be clobbered by the conversion layer.
        let req = ResponsesRequest {
            input: ResponseInput::Text("hi".to_string()),
            stream: Some(true),
            stream_options: Some(StreamOptions {
                include_usage: Some(false),
                include_obfuscation: Some(true),
            }),
            ..Default::default()
        };

        let opts = responses_to_chat(&req).unwrap().stream_options.unwrap();
        assert_eq!(opts.include_usage, Some(false));
        assert_eq!(opts.include_obfuscation, Some(true));
    }

    #[test]
    fn test_stream_options_non_streaming_dropped() {
        // stream=false must produce None stream_options even if caller set it.
        let req = ResponsesRequest {
            input: ResponseInput::Text("hi".to_string()),
            stream: Some(false),
            stream_options: Some(StreamOptions {
                include_usage: Some(true),
                include_obfuscation: Some(false),
            }),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert!(!chat_req.stream);
        assert!(chat_req.stream_options.is_none());
    }

    #[test]
    fn test_mcp_call_history_errors_when_lower_pass_skipped() {
        // The shared `transcript_lower` pass is supposed to project
        // hosted-tool items into core `function_call` (+ optional
        // `function_call_output`) pairs before this conversion runs.
        // Reaching `responses_to_chat` with a raw `McpCall` therefore
        // means the lower pass was bypassed — we surface that as a
        // hard error rather than silently re-projecting again here.
        let req = ResponsesRequest {
            input: ResponseInput::Items(vec![ResponseInputOutputItem::McpCall {
                id: "mcp_abc".to_string(),
                arguments: "{}".to_string(),
                name: "search".to_string(),
                server_label: "brave".to_string(),
                approval_request_id: None,
                error: None,
                output: None,
                status: None,
            }]),
            ..Default::default()
        };
        let err = responses_to_chat(&req).unwrap_err();
        assert!(err.contains("Unsupported"));
    }
}
