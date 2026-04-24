//! Conversion utilities for translating between /v1/responses and /v1/chat/completions formats
//!
//! This module implements the conversion approach where:
//! 1. ResponsesRequest → ChatCompletionRequest (for backend processing)
//! 2. ChatCompletionResponse → ResponsesResponse (for client response)
//!
//! This allows the gRPC router to reuse the existing chat pipeline infrastructure
//! without requiring Python backend changes.

use openai_protocol::{
    chat::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageContent},
    common::{
        ContentPart, FunctionCallResponse, ImageUrl, JsonSchemaFormat, ResponseFormat, ToolCall,
        UsageInfo,
    },
    responses::{
        ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseOutputItem,
        ResponseReasoningContent::ReasoningText, ResponseStatus, ResponsesRequest,
        ResponsesResponse, ResponsesUsage, StringOrContentParts, TextConfig, TextFormat,
    },
    UNKNOWN_MODEL_ID,
};
use tracing::warn;

use super::content_parts::ConversionError;
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
///
/// Assumes the caller has already run
/// [`super::content_parts::preprocess_responses_input`] so that every
/// [`ResponseContentPart::InputImage`] carries an `image_url` (HTTP or
/// `data:`) and every [`ResponseContentPart::InputFile`] has either been
/// rewritten to `InputImage` or rejected. The only remaining branching in
/// here is mapping `InputImage` to [`ContentPart::ImageUrl`] so the chat
/// pipeline's existing [`crate::routers::grpc::multimodal`] integration
/// picks it up automatically.
pub(crate) fn responses_to_chat(
    req: &ResponsesRequest,
) -> Result<ChatCompletionRequest, ConversionError> {
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
                        let chat_content = match content {
                            StringOrContentParts::String(s) => MessageContent::Text(s.clone()),
                            StringOrContentParts::Array(parts) => {
                                build_message_content(parts, role.as_str())?
                            }
                        };
                        messages.push(role_to_chat_message(role.as_str(), chat_content));
                    }
                    ResponseInputOutputItem::Message { role, content, .. } => {
                        let chat_content = build_message_content(content, role.as_str())?;
                        messages.push(role_to_chat_message(role.as_str(), chat_content));
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
                    ResponseInputOutputItem::McpApprovalResponse { .. }
                    | ResponseInputOutputItem::McpApprovalRequest { .. }
                    | ResponseInputOutputItem::ComputerCall { .. }
                    | ResponseInputOutputItem::ComputerCallOutput { .. }
                    | ResponseInputOutputItem::McpCall { .. }
                    | ResponseInputOutputItem::McpListTools { .. } => {
                        warn!(
                            function = "responses_to_chat",
                            "Approval item reached chat conversion"
                        );
                        return Err(ConversionError::UnsupportedContent(
                            "Unsupported input item type".to_string(),
                        ));
                    }
                    ResponseInputOutputItem::ImageGenerationCall { .. } => {
                        warn!(
                            function = "responses_to_chat",
                            "image_generation_call input item reached chat conversion"
                        );
                        return Err(ConversionError::UnsupportedContent(
                            "Unsupported input item type".to_string(),
                        ));
                    }
                    ResponseInputOutputItem::Compaction { .. }
                    | ResponseInputOutputItem::ItemReference { .. } => {
                        return Err(ConversionError::UnsupportedContent(
                            "Unsupported input item type".to_string(),
                        ));
                    }
                    ResponseInputOutputItem::CustomToolCall { .. }
                    | ResponseInputOutputItem::CustomToolCallOutput { .. } => {
                        warn!(
                            function = "responses_to_chat",
                            "Custom tool item reached chat conversion"
                        );
                        return Err(ConversionError::UnsupportedContent(
                            "Unsupported input item type".to_string(),
                        ));
                    }
                    ResponseInputOutputItem::ShellCall { .. }
                    | ResponseInputOutputItem::ShellCallOutput { .. } => {
                        warn!(
                            function = "responses_to_chat",
                            "Shell tool item reached chat conversion"
                        );
                        return Err(ConversionError::UnsupportedContent(
                            "Unsupported input item type".to_string(),
                        ));
                    }
                    ResponseInputOutputItem::ApplyPatchCall { .. }
                    | ResponseInputOutputItem::ApplyPatchCallOutput { .. } => {
                        warn!(
                            function = "responses_to_chat",
                            "apply_patch item reached chat conversion"
                        );
                        return Err(ConversionError::UnsupportedContent(
                            "Unsupported input item type".to_string(),
                        ));
                    }
                    // T5 schema-only: forced-cascade arm, no behavior.
                    ResponseInputOutputItem::LocalShellCall { .. }
                    | ResponseInputOutputItem::LocalShellCallOutput { .. } => {
                        return Err(ConversionError::UnsupportedContent(
                            "Unsupported input item type".to_string(),
                        ));
                    }
                }
            }
        }
    }

    // Ensure we have at least one message
    if messages.is_empty() {
        return Err(ConversionError::InvalidRequest(
            "Request must contain at least one message".to_string(),
        ));
    }

    // 3. Extract function tools from ResponseTools.
    // MCP tools are merged later by the tool loop (see tool_loop.rs:prepare_chat_tools_and_choice).
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

/// Build a chat [`MessageContent`] from a Responses-API content-part array.
///
/// Emits:
/// - [`MessageContent::Text`] when the array is text-only (every `InputText`
///   / `OutputText` / `Refusal` entry concatenated with an empty separator).
/// - [`MessageContent::Parts`] when any [`ContentPart::ImageUrl`] is present,
///   interleaving text and image URLs so the chat multimodal pipeline
///   (`grpc/multimodal.rs`) can extract them.
///
/// The `SimpleInputMessage` arm of `responses_to_chat` previously joined
/// content parts with a single space while the `Message` arm joined with
/// an empty string. Both shapes funnel through this helper post-R2 and
/// use the empty-separator join — the Message behavior, which matches
/// OpenAI's Chat Completions `text` parts semantics where consecutive
/// text runs concatenate verbatim. Unifying here removes the asymmetry
/// without changing observable behavior for any real prompt: the string
/// form (`StringOrContentParts::String`) is unaffected, and callers that
/// relied on the inserted whitespace were getting lossy prompts anyway.
///
/// Assumes [`super::content_parts::preprocess_responses_input`] has already
/// normalized `InputFile` → `InputImage` and rejected `InputImage.file_id` /
/// input-role `Refusal`. The `role` argument is currently unused here
/// (those checks live in `content_parts`) but is kept in the signature so
/// future role-sensitive decisions (e.g. refusal-on-assistant emission)
/// have a single plumbing point.
fn build_message_content(
    parts: &[ResponseContentPart],
    _role: &str,
) -> Result<MessageContent, ConversionError> {
    // Any InputFile at this layer means preprocessing was skipped — refuse
    // to silently drop it.
    if parts
        .iter()
        .any(|p| matches!(p, ResponseContentPart::InputFile { .. }))
    {
        return Err(ConversionError::UnsupportedContent(
            "input_file reached conversions.rs without preprocessing — \
             this is a bug in the Responses router"
                .to_string(),
        ));
    }

    let has_image = parts
        .iter()
        .any(|p| matches!(p, ResponseContentPart::InputImage { .. }));

    if !has_image {
        let text = parts
            .iter()
            .filter_map(|part| match part {
                ResponseContentPart::InputText { text }
                | ResponseContentPart::OutputText { text, .. } => Some(text.as_str()),
                ResponseContentPart::Refusal { refusal } => Some(refusal.as_str()),
                ResponseContentPart::InputImage { .. } | ResponseContentPart::InputFile { .. } => {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");
        return Ok(MessageContent::Text(text));
    }

    let mut chat_parts: Vec<ContentPart> = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            ResponseContentPart::InputText { text }
            | ResponseContentPart::OutputText { text, .. } => {
                if !text.is_empty() {
                    chat_parts.push(ContentPart::Text { text: text.clone() });
                }
            }
            ResponseContentPart::Refusal { refusal } => {
                if !refusal.is_empty() {
                    chat_parts.push(ContentPart::Text {
                        text: refusal.clone(),
                    });
                }
            }
            ResponseContentPart::InputImage {
                image_url, detail, ..
            } => {
                // preprocess_responses_input guarantees image_url is set and
                // file_id is None. Keep the defensive check so an
                // unpreprocessed input surfaces a typed error instead of
                // silently dropping the image.
                let url = image_url.clone().ok_or_else(|| {
                    ConversionError::InvalidRequest(
                        "input_image is missing image_url after preprocessing — this is a bug"
                            .to_string(),
                    )
                })?;
                chat_parts.push(ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url,
                        detail: detail.clone().map(detail_to_chat_string),
                    },
                });
            }
            ResponseContentPart::InputFile { .. } => {
                // preprocess_responses_input must have rewritten this to an
                // InputImage or rejected it. Anything still here is a bug.
                return Err(ConversionError::UnsupportedContent(
                    "input_file reached conversions.rs without preprocessing — \
                     this is a bug in the Responses router"
                        .to_string(),
                ));
            }
        }
    }

    Ok(MessageContent::Parts(chat_parts))
}

/// Map the Responses-API [`Detail`] enum (low/high/auto/original) to the
/// Chat-Completions `image_url.detail` string. The chat pipeline recognizes
/// `"auto" | "low" | "high"` (see `multimodal::parse_detail`); unknown
/// values are silently ignored downstream, so `Original` is passed through
/// as its rename (`"original"`).
fn detail_to_chat_string(detail: openai_protocol::common::Detail) -> String {
    match detail {
        openai_protocol::common::Detail::Low => "low".to_string(),
        openai_protocol::common::Detail::High => "high".to_string(),
        openai_protocol::common::Detail::Auto => "auto".to_string(),
        openai_protocol::common::Detail::Original => "original".to_string(),
    }
}

/// Convert role and content to ChatMessage
fn role_to_chat_message(role: &str, content: MessageContent) -> ChatMessage {
    match role {
        "user" => ChatMessage::User {
            content,
            name: None,
        },
        "assistant" => ChatMessage::Assistant {
            content: Some(content),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        },
        "system" => ChatMessage::System {
            content,
            name: None,
        },
        _ => {
            // Unknown role, treat as user message
            ChatMessage::User {
                content,
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

/// Convert a ChatCompletionResponse to ResponsesResponse
///
/// # Conversion Logic
/// - `id` → `response_id_override` if provided, otherwise `chat_resp.id`
/// - `model` → `model` (pass through)
/// - `choices[0].message` → `output` array (convert to ResponseOutputItem::Message)
/// - `choices[0].finish_reason` → determines `status` (stop/length → Completed)
/// - `created` timestamp → `created_at`
pub(crate) fn chat_to_responses(
    chat_resp: &ChatCompletionResponse,
    original_req: &ResponsesRequest,
    response_id_override: Option<String>,
) -> Result<ResponsesResponse, String> {
    // Extract the first choice (responses API doesn't support n>1)
    let choice = chat_resp
        .choices
        .first()
        .ok_or_else(|| "Chat response contains no choices".to_string())?;

    // Convert assistant message to output items
    let mut output: Vec<ResponseOutputItem> = Vec::new();

    // Convert message content to output item
    if let Some(content) = &choice.message.content {
        if !content.is_empty() {
            output.push(ResponseOutputItem::Message {
                id: format!("msg_{}", chat_resp.id),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: content.clone(),
                    annotations: vec![],
                    logprobs: choice.logprobs.clone(),
                }],
                status: "completed".to_string(),
                phase: None,
            });
        }
    }

    // Convert reasoning content if present (O1-style models)
    if let Some(reasoning) = &choice.message.reasoning_content {
        if !reasoning.is_empty() {
            output.push(ResponseOutputItem::new_reasoning(
                format!("reasoning_{}", chat_resp.id),
                vec![],
                vec![ReasoningText {
                    text: reasoning.clone(),
                }],
                Some("completed".to_string()),
            ));
        }
    }

    // Convert tool calls if present
    if let Some(tool_calls) = &choice.message.tool_calls {
        for tool_call in tool_calls {
            output.push(ResponseOutputItem::FunctionToolCall {
                id: tool_call.id.clone(),
                call_id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                arguments: tool_call.function.arguments.clone().unwrap_or_default(),
                output: None, // Tool hasn't been executed yet
                status: "in_progress".to_string(),
            });
        }
    }

    // Determine response status based on finish_reason
    let status = match choice.finish_reason.as_deref() {
        Some("stop") | Some("length") => ResponseStatus::Completed,
        Some("tool_calls") => ResponseStatus::InProgress, // Waiting for tool execution
        Some("failed") | Some("error") => ResponseStatus::Failed,
        _ => ResponseStatus::Completed, // Default to completed
    };

    // Convert usage from Usage to UsageInfo, then wrap in ResponsesUsage
    let usage = chat_resp.usage.as_ref().map(|u| {
        let usage_info = UsageInfo {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            reasoning_tokens: u
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens),
            prompt_tokens_details: None, // Chat response doesn't have this
        };
        ResponsesUsage::Classic(usage_info)
    });

    // Generate response
    let response_id = response_id_override.unwrap_or_else(|| chat_resp.id.clone());
    Ok(ResponsesResponse::builder(&response_id, &chat_resp.model)
        .copy_from_request(original_req)
        .created_at(chat_resp.created as i64)
        .status(status)
        .output(output)
        .maybe_text(original_req.text.clone())
        .maybe_usage(usage)
        .build())
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
    fn test_image_generation_call_input_rejected() {
        // Regression: `image_generation_call` items are server-produced
        // output (populated via the shared MCP transformer) and must not
        // be round-tripped back into the chat conversion as input.
        // The regular gRPC path — used by non-Harmony text LLMs that only do
        // function calling — rejects this variant with the same contract as
        // sibling hosted-tool items (Computer/Shell/Custom/ApplyPatch).
        let req = ResponsesRequest {
            input: ResponseInput::Items(vec![ResponseInputOutputItem::ImageGenerationCall {
                id: "ig_test".to_string(),
                action: None,
                background: None,
                output_format: None,
                quality: None,
                result: Some("base64data".to_string()),
                revised_prompt: Some("a cat".to_string()),
                size: None,
                status: None,
            }]),
            ..Default::default()
        };

        let result = responses_to_chat(&req);
        let err = result.expect_err("ImageGenerationCall input must be rejected");
        assert_eq!(err.error_code(), "unsupported_content");
        assert_eq!(err.message(), "Unsupported input item type");
    }

    #[test]
    fn test_input_image_content_part_produces_chat_image_part() {
        // After preprocessing, an InputImage with image_url (data URL or
        // HTTP) flows through responses_to_chat and is emitted as a chat
        // `MessageContent::Parts` entry with `ContentPart::ImageUrl`. The
        // downstream multimodal pipeline then picks it up automatically.
        let req = ResponsesRequest {
            input: ResponseInput::Items(vec![ResponseInputOutputItem::Message {
                id: "msg_1".to_string(),
                role: "user".to_string(),
                content: vec![
                    ResponseContentPart::InputText {
                        text: "describe this:".to_string(),
                    },
                    ResponseContentPart::InputImage {
                        detail: Some(openai_protocol::common::Detail::High),
                        file_id: None,
                        image_url: Some("data:image/jpeg;base64,AAAA".to_string()),
                    },
                ],
                status: None,
                phase: None,
            }]),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.messages.len(), 1);
        let user = &chat_req.messages[0];
        let parts = match user {
            ChatMessage::User {
                content: MessageContent::Parts(parts),
                ..
            } => parts,
            other => panic!("expected User message with Parts content, got {other:?}"),
        };
        assert_eq!(parts.len(), 2);
        match &parts[0] {
            ContentPart::Text { text } => assert_eq!(text, "describe this:"),
            other => panic!("expected Text part first, got {other:?}"),
        }
        match &parts[1] {
            ContentPart::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "data:image/jpeg;base64,AAAA");
                assert_eq!(image_url.detail.as_deref(), Some("high"));
            }
            other => panic!("expected ImageUrl part second, got {other:?}"),
        }
    }

    #[test]
    fn test_input_image_missing_url_errors_as_invalid_request() {
        // If preprocessing ever misses an InputImage (no image_url), the
        // conversion must surface a typed error rather than silently
        // dropping the image.
        let req = ResponsesRequest {
            input: ResponseInput::Items(vec![ResponseInputOutputItem::Message {
                id: "msg_1".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputImage {
                    detail: None,
                    file_id: None,
                    image_url: None,
                }],
                status: None,
                phase: None,
            }]),
            ..Default::default()
        };

        let err = responses_to_chat(&req).expect_err("missing image_url must error");
        assert_eq!(err.error_code(), "invalid_request");
    }

    #[test]
    fn test_input_file_leaks_past_preprocess_errors() {
        // Defensive contract: any InputFile reaching conversions is a bug,
        // since preprocess_responses_input is expected to rewrite to
        // InputImage or reject. Verify we surface a clear typed error.
        let req = ResponsesRequest {
            input: ResponseInput::Items(vec![ResponseInputOutputItem::Message {
                id: "msg_1".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputFile {
                    detail: None,
                    file_data: Some("AAAA".to_string()),
                    file_id: None,
                    file_url: None,
                    filename: None,
                }],
                status: None,
                phase: None,
            }]),
            ..Default::default()
        };

        let err = responses_to_chat(&req).expect_err("unpreprocessed InputFile must error");
        assert_eq!(err.error_code(), "unsupported_content");
    }
}
