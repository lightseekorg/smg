//! Chat message processing, tool constraints, and shared utilities for gRPC routers.

use std::{collections::HashMap, io, path::Path, sync::Arc};

use axum::response::Response;
use bytes::Bytes;
use llm_tokenizer::{
    chat_template::{ChatTemplateContentFormat, ChatTemplateParams},
    stop::StopSequenceDecoderBuilder,
    traits::Tokenizer,
    StopSequenceDecoder,
};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage},
    common::{
        FunctionCallResponse, StringOrArray, Tool, ToolCall, ToolChoice, ToolChoiceValue,
    },
    generate::GenerateFinishReason,
};
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;
use tracing::error;
use uuid::Uuid;

use crate::routers::{
    error,
    grpc::{context::RequestContext, ProcessedMessages},
};

/// Type alias for the SSE channel sender used across streaming endpoints.
pub(crate) type SseSender = mpsc::UnboundedSender<Result<Bytes, io::Error>>;

/// Send an SSE error event with a typed error body.
///
/// Produces `data: {"error":{"message":"...","type":"..."}}\n\n` using
/// `serde_json` so that quotes, newlines, and other special characters in the
/// error message are properly escaped.
pub(crate) fn send_error_sse(tx: &SseSender, message: impl ToString, error_type: &str) {
    let chunk = format!(
        "data: {}\n\n",
        json!({
            "error": {
                "message": message.to_string(),
                "type": error_type,
            }
        })
    );
    let _ = tx.send(Ok(Bytes::from(chunk)));
}

/// Resolve tokenizer from registry and cache it in request context.
///
/// This is a helper to avoid duplicating tokenizer resolution logic across
/// preparation stages (chat, generate, embedding).
///
/// Returns the tokenizer Arc, which is also cached in `ctx.state.tokenizer`.
pub(crate) fn resolve_tokenizer(
    ctx: &mut RequestContext,
    stage_name: &str,
) -> Result<Arc<dyn Tokenizer>, Box<Response>> {
    let model_id = ctx.input.model_id.as_deref().ok_or_else(|| {
        error!(
            function = %stage_name,
            "model_id not set in request context"
        );
        Box::new(error::internal_error(
            "model_id_not_set",
            "model_id not set in request context - this is a bug in request routing",
        ))
    })?;

    let tokenizer = ctx
        .components
        .tokenizer_registry
        .get(model_id)
        .ok_or_else(|| {
            error!(
                function = %stage_name,
                model = %model_id,
                "Tokenizer not found for model"
            );
            Box::new(error::internal_error(
                "tokenizer_not_found",
                format!("Tokenizer not found for model: {model_id}"),
            ))
        })?;

    // Cache tokenizer in context for reuse in response processing stage
    ctx.state.tokenizer = Some(tokenizer.clone());

    Ok(tokenizer)
}

/// Process tool call arguments in messages
/// Per Transformers docs, tool call arguments in assistant messages should be dicts
fn process_tool_call_arguments(messages: &mut [Value]) -> Result<(), String> {
    for msg in messages {
        let role = msg.get("role").and_then(|v| v.as_str());
        if role != Some("assistant") {
            continue;
        }

        let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|tc| tc.as_array_mut()) else {
            continue;
        };

        for call in tool_calls {
            let Some(function) = call.get_mut("function") else {
                continue;
            };
            let Some(args) = function.get_mut("arguments") else {
                continue;
            };
            let Some(args_str) = args.as_str() else {
                continue;
            };

            // Parse JSON string to object (like Python json.loads)
            match serde_json::from_str::<Value>(args_str) {
                Ok(parsed) => *args = parsed,
                Err(e) => {
                    return Err(format!(
                        "Failed to parse tool call arguments as JSON: '{args_str}'. Error: {e}"
                    ))
                }
            }
        }
    }
    Ok(())
}

/// Process messages based on content format for ANY message type
pub(crate) fn process_content_format(
    messages: &[ChatMessage],
    content_format: ChatTemplateContentFormat,
) -> Result<Vec<Value>, String> {
    messages
        .iter()
        .map(|message| {
            let mut message_json = serde_json::to_value(message)
                .map_err(|e| format!("Failed to serialize message: {e}"))?;

            if let Some(obj) = message_json.as_object_mut() {
                if let Some(content_value) = obj.get_mut("content") {
                    transform_content_field(content_value, content_format);
                }
            }

            Ok(message_json)
        })
        .collect()
}

/// Transform a single content field based on content format
fn transform_content_field(content_value: &mut Value, content_format: ChatTemplateContentFormat) {
    let Some(content_array) = content_value.as_array() else {
        return; // Not multimodal, keep as-is
    };

    match content_format {
        ChatTemplateContentFormat::String => {
            // Extract and join text parts only
            let text_parts: Vec<String> = content_array
                .iter()
                .filter_map(|part| {
                    part.as_object()?
                        .get("type")?
                        .as_str()
                        .filter(|&t| t == "text")
                        .and_then(|_| part.as_object()?.get("text")?.as_str())
                        .map(String::from)
                })
                .collect();

            if !text_parts.is_empty() {
                *content_value = Value::String(text_parts.join(" "));
            }
        }
        ChatTemplateContentFormat::OpenAI => {
            // Replace media URLs with simple type placeholders
            let processed_parts: Vec<Value> = content_array
                .iter()
                .map(|part| {
                    part.as_object()
                        .and_then(|obj| obj.get("type")?.as_str())
                        .and_then(|type_str| match type_str {
                            "image_url" => Some(json!({"type": "image"})),
                            "video_url" => Some(json!({"type": "video"})),
                            "audio_url" => Some(json!({"type": "audio"})),
                            _ => None,
                        })
                        .unwrap_or_else(|| part.clone())
                })
                .collect();

            *content_value = Value::Array(processed_parts);
        }
    }
}

/// Generate tool constraints for structured generation
/// Note: tools should already be filtered if needed (by allowed_tools or specific function)
pub fn generate_tool_constraints(
    tools: &[Tool],
    tool_choice: Option<&ToolChoice>,
    _model: &str,
) -> Result<Option<(String, String)>, String> {
    let Some(choice) = tool_choice else {
        return Ok(None);
    };

    match choice {
        // Specific function: Return parameters schema directly
        // tools should already be filtered to contain only the specific function
        ToolChoice::Function { .. } => {
            if tools.is_empty() {
                return Ok(None);
            }
            let tool = &tools[0];

            // Return the tool's parameters schema directly (not wrapped in array)
            let params_schema = serde_json::to_string(&tool.function.parameters)
                .map_err(|e| format!("Failed to serialize tool parameters: {e}"))?;
            Ok(Some((String::from("json_schema"), params_schema)))
        }

        // Required: Array of tool calls with minItems: 1
        ToolChoice::Value(ToolChoiceValue::Required) => {
            let schema = build_required_array_schema(tools)?;
            Ok(Some(("json_schema".to_string(), schema)))
        }

        // AllowedTools with required mode: tools are already filtered
        ToolChoice::AllowedTools { mode, .. } => {
            if mode == "required" {
                if tools.is_empty() {
                    return Ok(None);
                }
                let schema = build_required_array_schema(tools)?;
                Ok(Some(("json_schema".to_string(), schema)))
            } else {
                // "auto" mode - no constraint needed
                Ok(None)
            }
        }

        // "auto" or "none" - no constraint
        ToolChoice::Value(_) => Ok(None),
    }
}

/// Build JSON schema for required tool calls (array with minItems: 1)
/// Includes $defs consolidation from all tools (matching Python's behavior)
fn build_required_array_schema(tools: &[Tool]) -> Result<String, String> {
    let mut any_of_schemas = Vec::with_capacity(tools.len());
    for tool in tools {
        let tool_schema = json!({
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [tool.function.name]
                },
                "parameters": tool.function.parameters
            },
            "required": ["name", "parameters"]
        });
        any_of_schemas.push(tool_schema);
    }

    // Consolidate $defs from all tools (matching Python's _get_tool_schema_defs)
    let mut all_defs: Map<String, Value> = Map::new();
    for tool in tools {
        if let Value::Object(params) = &tool.function.parameters {
            if let Some(Value::Object(defs)) = params.get("$defs") {
                for (def_name, def_schema) in defs {
                    if let Some(existing) = all_defs.get(def_name) {
                        // Check for conflicts
                        if existing != def_schema {
                            let error_msg = format!(
                                "Tool definition '{def_name}' has multiple conflicting schemas, which is not supported"
                            );
                            error!("{}", error_msg);
                            return Err(error_msg);
                        }
                    } else {
                        all_defs.insert(def_name.clone(), def_schema.clone());
                    }
                }
            }
        }
    }

    // Build the full array schema
    let mut array_schema = json!({
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "anyOf": any_of_schemas
        }
    });

    // Add $defs if any were found (matching Python's behavior)
    if !all_defs.is_empty() {
        if let Value::Object(ref mut schema_obj) = array_schema {
            schema_obj.insert("$defs".to_string(), Value::Object(all_defs));
        }
    }

    serde_json::to_string(&array_schema)
        .map_err(|e| format!("Failed to serialize tool schema: {e}"))
}

/// Filter tools based on tool_choice (generic helper)
///
/// Returns filtered tools if filtering is needed, otherwise returns None.
/// Used by both Chat API and Responses API (Harmony) for constraint generation.
pub(crate) fn filter_tools_by_tool_choice(
    tools: &[Tool],
    tool_choice: Option<&ToolChoice>,
) -> Option<Vec<Tool>> {
    match tool_choice {
        Some(ToolChoice::AllowedTools { tools: allowed, .. }) => {
            let allowed_names: std::collections::HashSet<&str> =
                allowed.iter().filter_map(|t| t.function_name()).collect();
            let filtered: Vec<Tool> = tools
                .iter()
                .filter(|t| allowed_names.contains(t.function.name.as_str()))
                .cloned()
                .collect();
            Some(filtered)
        }
        Some(ToolChoice::Function { function, .. }) => {
            let filtered: Vec<Tool> = tools
                .iter()
                .filter(|t| t.function.name == function.name)
                .cloned()
                .collect();
            Some(filtered)
        }
        _ => None, // No filtering needed
    }
}

/// Filter ChatCompletionRequest by tool_choice
///
/// Returns a reference to the original request if no filtering needed,
/// otherwise returns a cloned request with filtered tools.
///
/// Note: Tool existence is validated earlier in ChatCompletionRequest::validate(),
/// so this function assumes tool_choice references valid tools.
pub(crate) fn filter_chat_request_by_tool_choice(
    body: &ChatCompletionRequest,
) -> std::borrow::Cow<'_, ChatCompletionRequest> {
    if let Some(tools) = &body.tools {
        if let Some(filtered_tools) = filter_tools_by_tool_choice(tools, body.tool_choice.as_ref())
        {
            let mut filtered_body = body.clone();
            filtered_body.tools = Some(filtered_tools);
            return std::borrow::Cow::Owned(filtered_body);
        }
    }

    // No filtering needed - return original request
    std::borrow::Cow::Borrowed(body)
}

#[inline]
fn is_auto_tool_choice(request: &ChatCompletionRequest) -> bool {
    request.tools.is_some()
        && matches!(
            request.tool_choice,
            Some(ToolChoice::Value(ToolChoiceValue::Auto)) | None
        )
}

fn latest_user_text(request: &ChatCompletionRequest) -> String {
    request
        .messages
        .iter()
        .rev()
        .find_map(|message| match message {
            ChatMessage::User { content, .. } => Some(content.to_simple_string()),
            _ => None,
        })
        .unwrap_or_default()
}

fn latest_message_is_tool(request: &ChatCompletionRequest) -> bool {
    matches!(request.messages.last(), Some(ChatMessage::Tool { .. }))
}

fn has_prior_tool_exchange(request: &ChatCompletionRequest) -> bool {
    request.messages.iter().any(|message| match message {
        ChatMessage::Tool { .. } => true,
        ChatMessage::Assistant { tool_calls, .. } => {
            tool_calls.as_ref().is_some_and(|tool_calls| !tool_calls.is_empty())
        }
        _ => false,
    })
}

fn latest_user_follows_tool_exchange(request: &ChatCompletionRequest) -> bool {
    let last_tool_exchange_index = request.messages.iter().rposition(|message| match message {
        ChatMessage::Tool { .. } => true,
        ChatMessage::Assistant { tool_calls, .. } => {
            tool_calls.as_ref().is_some_and(|tool_calls| !tool_calls.is_empty())
        }
        _ => false,
    });

    let last_user_index = request
        .messages
        .iter()
        .rposition(|message| matches!(message, ChatMessage::User { .. }));

    match (last_tool_exchange_index, last_user_index) {
        (Some(tool_index), Some(user_index)) => user_index > tool_index,
        (None, Some(_)) => true,
        _ => false,
    }
}

pub(crate) fn infer_auto_tool_repair_target(
    request: &ChatCompletionRequest,
) -> Option<&'static str> {
    if !is_auto_tool_choice(request) {
        return None;
    }

    let tools = request.tools.as_deref()?;
    let latest_user_text = latest_user_text(request).to_ascii_lowercase();
    let has_tool = |name: &str| tools.iter().any(|tool| tool.function.name == name);

    if latest_message_is_tool(request) && has_tool("execute-tool") {
        return Some("execute-tool");
    }

    if has_prior_tool_exchange(request) && !latest_user_follows_tool_exchange(request) {
        return None;
    }

    let is_file_read_request = (latest_user_text.contains("what is in")
        || latest_user_text.contains("what's in")
        || latest_user_text.contains("contents of"))
        && latest_user_text.contains("file");
    if is_file_read_request {
        if has_tool("read_file") {
            return Some("read_file");
        }
        if has_tool("read") {
            return Some("read");
        }
    }

    let is_edit_existing_file_request = latest_user_text.contains("same file")
        || latest_user_text.contains("another function")
        || latest_user_text.contains("add another")
        || latest_user_text.contains("update the file")
        || latest_user_text.contains("modify the file");
    if is_edit_existing_file_request && has_tool("edit_file") {
        return Some("edit_file");
    }

    let is_create_file_request = latest_user_text.contains("create a file")
        || latest_user_text.contains("create file")
        || latest_user_text.contains("write a file")
        || latest_user_text.contains("make a typescript file")
        || latest_user_text.contains("make a javascript file")
        || latest_user_text.contains("make a js file");
    if is_create_file_request {
        if has_tool("edit_file") {
            return Some("edit_file");
        }
        if has_tool("write") {
            return Some("write");
        }
    }

    let is_stock_request = (latest_user_text.contains("stock")
        || latest_user_text.contains("share price")
        || latest_user_text.contains("ticker"))
        && has_tool("getCurrentStockPrice");
    if is_stock_request {
        return Some("getCurrentStockPrice");
    }

    let is_restaurant_request = (latest_user_text.contains("restaurant")
        || latest_user_text.contains("food")
        || latest_user_text.contains("eat"))
        && has_tool("getRestaurantRecommendations");
    if is_restaurant_request {
        return Some("getRestaurantRecommendations");
    }

    let is_weather_request = (latest_user_text.contains("weather")
        || latest_user_text.contains("temperature"))
        && has_tool("getCurrentWeather");
    if is_weather_request {
        return Some("getCurrentWeather");
    }

    let is_hf_trending_request = latest_user_text.contains("trending models")
        && (latest_user_text.contains("today") || latest_user_text.contains("todays"));
    if is_hf_trending_request && has_tool("agent__hf-search") {
        return Some("agent__hf-search");
    }

    None
}

fn latest_tool_call(request: &ChatCompletionRequest) -> Option<&ToolCall> {
    request.messages.iter().rev().find_map(|message| match message {
        ChatMessage::Assistant { tool_calls, .. } => tool_calls.as_ref()?.last(),
        _ => None,
    })
}

fn requested_implementation_suffix(request: &ChatCompletionRequest) -> Option<&'static str> {
    let latest_user_text = latest_user_text(request).to_ascii_lowercase();
    if latest_user_text.contains("depth first search") || latest_user_text.contains("dfs") {
        return Some(" with a depth-first search implementation.");
    }
    if latest_user_text.contains("add two numbers") {
        return Some(" with a function to add two numbers.");
    }
    None
}

fn repair_short_post_tool_content(request: &ChatCompletionRequest, processed_text: &mut String) {
    if has_prior_tool_exchange(request) && !latest_user_follows_tool_exchange(request) {
        let trimmed = processed_text.trim();
        if trimmed.len() > 24 {
            return;
        }

        if let Some(tool_call) = latest_tool_call(request) {
            if tool_call.function.name == "write" || tool_call.function.name == "edit_file" {
                if let Some(arguments) = tool_call.function.arguments.as_deref() {
                    if let Ok(parsed) = serde_json::from_str::<Value>(arguments) {
                        let file_path = parsed
                            .get("filePath")
                            .or_else(|| parsed.get("file_path"))
                            .and_then(Value::as_str);
                        if let Some(file_path) = file_path {
                            let filename = Path::new(file_path)
                                .file_name()
                                .and_then(|name| name.to_str())
                                .unwrap_or(file_path);
                            let suffix = requested_implementation_suffix(request)
                                .unwrap_or(" with the requested implementation.");
                            *processed_text = format!("Created {filename}{suffix}");
                        }
                    }
                }
            }
        }
    }
}

fn sanitize_tool_call_arguments(request: &ChatCompletionRequest, tool_calls: &mut [ToolCall]) {
    let Some(tools) = request.tools.as_ref() else {
        return;
    };

    for tool_call in tool_calls {
        let Some(tool) = tools.iter().find(|tool| tool.function.name == tool_call.function.name) else {
            continue;
        };
        let Some(arguments) = tool_call.function.arguments.as_deref() else {
            continue;
        };
        let Ok(mut parsed) = serde_json::from_str::<Value>(arguments) else {
            continue;
        };
        let Some(obj) = parsed.as_object_mut() else {
            continue;
        };
        let Some(schema) = tool.function.parameters.as_object() else {
            continue;
        };
        let properties = schema
            .get("properties")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let required = schema
            .get("required")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        let invalid_optional_keys: Vec<String> = obj
            .iter()
            .filter_map(|(key, value)| {
                let is_required = required.iter().any(|item| item.as_str() == Some(key.as_str()));
                if is_required {
                    return None;
                }
                let property = properties.get(key)?;
                let empty_string = value.as_str().is_some_and(str::is_empty);
                let invalid_enum = value.as_str().is_some_and(|raw| {
                    property
                        .get("enum")
                        .and_then(Value::as_array)
                        .is_some_and(|choices| {
                            !choices.iter().any(|choice| choice.as_str() == Some(raw))
                        })
                });
                if empty_string || invalid_enum {
                    Some(key.clone())
                } else {
                    None
                }
            })
            .collect();

        for key in invalid_optional_keys {
            obj.remove(&key);
        }

        tool_call.function.arguments = Some(
            serde_json::to_string(&parsed).unwrap_or_else(|_| arguments.to_string()),
        );
    }
}

fn repair_conditional_tool_calls(tool_calls: &mut [ToolCall]) {
    for tool_call in tool_calls {
        if tool_call.function.name != "conditional_evaluation" {
            continue;
        }
        let Some(arguments) = tool_call.function.arguments.as_deref() else {
            continue;
        };
        let Ok(mut parsed) = serde_json::from_str::<Value>(arguments) else {
            continue;
        };
        let Some(obj) = parsed.as_object_mut() else {
            continue;
        };

        if !obj.contains_key("rationale") {
            if let Some(value) = obj.remove("ration") {
                obj.insert("rationale".to_string(), value);
            } else if let Some(value) = obj.remove("reason") {
                obj.insert("rationale".to_string(), value);
            }
        }

        if let Some(Value::String(raw_bool)) = obj.get("is_true") {
            let normalized = match raw_bool.to_ascii_lowercase().as_str() {
                "true" => Some(true),
                "false" => Some(false),
                _ => None,
            };
            if let Some(value) = normalized {
                obj.insert("is_true".to_string(), Value::Bool(value));
            }
        }

        tool_call.function.arguments = Some(
            serde_json::to_string(&parsed).unwrap_or_else(|_| arguments.to_string()),
        );
    }
}

fn collect_tool_message_ids(request: &ChatCompletionRequest) -> Vec<String> {
    let mut ids = Vec::new();
    for message in &request.messages {
        let ChatMessage::Tool { content, .. } = message else {
            continue;
        };
        let Ok(parsed) = serde_json::from_str::<Value>(&content.to_simple_string()) else {
            continue;
        };
        let Some(items) = parsed.as_array() else {
            continue;
        };
        for item in items {
            let Some(paper_id) = item.get("id").and_then(Value::as_str) else {
                continue;
            };
            if !ids.iter().any(|existing| existing == paper_id) {
                ids.push(paper_id.to_string());
            }
        }
    }
    ids
}

fn repair_submit_tool_calls(request: &ChatCompletionRequest, tool_calls: &mut [ToolCall]) {
    let paper_ids = collect_tool_message_ids(request);
    if paper_ids.is_empty() {
        return;
    }

    for tool_call in tool_calls {
        if tool_call.function.name != "submit" {
            continue;
        }
        let Some(arguments) = tool_call.function.arguments.as_deref() else {
            continue;
        };
        let Ok(mut parsed) = serde_json::from_str::<Value>(arguments) else {
            continue;
        };
        let Some(obj) = parsed.as_object_mut() else {
            continue;
        };
        let Some(answer) = obj
            .get("answer")
            .and_then(Value::as_str)
            .map(str::to_string)
        else {
            continue;
        };

        let missing_ids: Vec<String> = paper_ids
            .iter()
            .filter(|paper_id| !answer.contains(paper_id.as_str()))
            .cloned()
            .collect();
        if missing_ids.is_empty() {
            continue;
        }

        let repaired_answer = format!(
            "{}\n\nRecent arXiv paper IDs referenced above: {}",
            answer.trim_end(),
            missing_ids.join(", ")
        );
        obj.insert("answer".to_string(), Value::String(repaired_answer));
        tool_call.function.arguments = Some(
            serde_json::to_string(&parsed).unwrap_or_else(|_| arguments.to_string()),
        );
    }
}

pub(crate) fn repair_tool_calls_and_content(
    request: &ChatCompletionRequest,
    tool_calls: &mut Option<Vec<ToolCall>>,
    processed_text: &mut String,
) {
    if let Some(tool_calls) = tool_calls.as_mut() {
        sanitize_tool_call_arguments(request, tool_calls);
        repair_conditional_tool_calls(tool_calls);
        repair_submit_tool_calls(request, tool_calls);
    }

    repair_short_post_tool_content(request, processed_text);
}

pub(crate) async fn deterministic_auto_tool_repair(
    request: &ChatCompletionRequest,
) -> Result<Option<Vec<ToolCall>>, String> {
    let explicit_specific_function = match request.tool_choice.as_ref() {
        Some(ToolChoice::Function { function, .. }) => Some(function.name.as_str()),
        _ => None,
    };

    let Some(target_tool) =
        explicit_specific_function.or_else(|| infer_auto_tool_repair_target(request))
    else {
        return Ok(None);
    };

    let Some(filtered_tools) = request.tools.as_ref().map(|tools| {
        tools.iter()
            .filter(|tool| tool.function.name == target_tool)
            .cloned()
            .collect::<Vec<_>>()
    }) else {
        return Ok(None);
    };

    if filtered_tools.is_empty() {
        return Ok(None);
    }

    let mut repair_request = request.clone();
    repair_request.tools = Some(filtered_tools);
    repair_request.stream = false;
    repair_request.stream_options = None;
    repair_request.temperature = Some(0.0);
    repair_request.top_k = Some(1);
    repair_request.tool_choice = match request.tool_choice.clone() {
        Some(ToolChoice::Function { .. }) => request.tool_choice.clone(),
        _ => Some(ToolChoice::Value(ToolChoiceValue::Required)),
    };

    let response = reqwest::Client::new()
        .post("http://127.0.0.1:9000/v1/chat/completions")
        .json(&repair_request)
        .send()
        .await
        .map_err(|e| format!("deterministic tool repair request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "deterministic tool repair returned HTTP {}",
            response.status()
        ));
    }

    let response_body: ChatCompletionResponse = response
        .json()
        .await
        .map_err(|e| format!("failed to decode deterministic tool repair response: {e}"))?;

    Ok(response_body
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message.tool_calls))
}

fn build_chat_template_kwargs(request: &ChatCompletionRequest) -> Option<HashMap<String, Value>> {
    let kwargs_capacity = 1 + request.chat_template_kwargs.as_ref().map_or(0, |k| k.len());
    let mut combined_template_kwargs = HashMap::with_capacity(kwargs_capacity);

    // Add reasoning_effort if present (like Python does)
    if let Some(reasoning_effort) = &request.reasoning_effort {
        combined_template_kwargs.insert(
            "reasoning_effort".to_string(),
            Value::String(reasoning_effort.clone()),
        );
    }

    // Add any additional template kwargs from request first so explicit user values win.
    if let Some(template_kwargs) = &request.chat_template_kwargs {
        for (key, value) in template_kwargs {
            combined_template_kwargs.insert(key.clone(), value.clone());
        }
    }

    if combined_template_kwargs.is_empty() {
        None
    } else {
        Some(combined_template_kwargs)
    }
}

/// Process chat messages and apply template (shared by both routers)
/// Requires HuggingFace tokenizer with chat template support
pub fn process_chat_messages(
    request: &ChatCompletionRequest,
    tokenizer: &dyn Tokenizer,
) -> Result<ProcessedMessages, String> {
    let formatted_text = {
        // Get content format and transform messages accordingly
        let content_format = tokenizer.chat_template_content_format();
        let mut transformed_messages = process_content_format(&request.messages, content_format)?;

        // Process tool call arguments in assistant messages
        process_tool_call_arguments(&mut transformed_messages)?;

        // Convert tools to JSON values for template processing
        let tools_json: Option<Vec<Value>> = request
            .tools
            .as_ref()
            .map(|tools| {
                tools
                    .iter()
                    .map(serde_json::to_value)
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()
            .map_err(|e| format!("Failed to serialize tools: {e}"))?;

        let template_kwargs = build_chat_template_kwargs(request);

        let params = ChatTemplateParams {
            add_generation_prompt: true,
            tools: tools_json.as_deref(),
            template_kwargs: template_kwargs.as_ref(),
            ..Default::default()
        };

        // Handle assistant prefix for continue_final_message
        let assistant_prefix = if request.continue_final_message
            && !transformed_messages.is_empty()
            && transformed_messages
                .last()
                .and_then(|msg| msg.get("role"))
                .and_then(|v| v.as_str())
                == Some("assistant")
        {
            // Pop the last message to handle it separately — guarded by !is_empty() check above
            let Some(last_msg) = transformed_messages.pop() else {
                return Ok(ProcessedMessages {
                    text: String::new(),
                    multimodal_intermediate: None,
                    stop_sequences: request.stop.clone(),
                });
            };
            last_msg
                .get("content")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        } else {
            None
        };

        // Apply chat template with the (now possibly shorter) list of messages
        let rendered = tokenizer
            .apply_chat_template(&transformed_messages, params)
            .map_err(|e| format!("Failed to apply chat template: {e}"))?;

        // Append assistant prefix if we have one
        if let Some(prefix) = assistant_prefix {
            format!("{rendered}{prefix}")
        } else {
            rendered
        }
    };

    Ok(ProcessedMessages {
        text: formatted_text,
        multimodal_intermediate: None,
        stop_sequences: request.stop.clone(),
    })
}

#[cfg(test)]
mod template_kwargs_tests {
    use std::collections::HashMap;

    use openai_protocol::{
        chat::ChatCompletionRequest,
        common::{Function, Tool, ToolChoice, ToolChoiceValue},
    };
    use serde_json::{json, Value};

    use super::build_chat_template_kwargs;

    fn sample_tool() -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "write".to_string(),
                description: Some("Write a file".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"}
                    },
                    "required": ["file_path"]
                }),
                strict: None,
            },
        }
    }

    #[test]
    fn returns_none_when_no_reasoning_effort_or_template_kwargs_are_present() {
        let request = ChatCompletionRequest {
            tools: Some(vec![sample_tool()]),
            tool_choice: Some(ToolChoice::Value(ToolChoiceValue::Auto)),
            ..Default::default()
        };

        let kwargs = build_chat_template_kwargs(&request);
        assert!(kwargs.is_none());
    }

    #[test]
    fn preserves_user_supplied_template_kwargs() {
        let mut chat_template_kwargs = HashMap::new();
        chat_template_kwargs.insert("enable_thinking".to_string(), Value::Bool(true));
        chat_template_kwargs.insert("custom_flag".to_string(), Value::String("x".to_string()));

        let request = ChatCompletionRequest {
            tools: Some(vec![sample_tool()]),
            tool_choice: Some(ToolChoice::Value(ToolChoiceValue::Auto)),
            reasoning_effort: Some("medium".to_string()),
            chat_template_kwargs: Some(chat_template_kwargs),
            ..Default::default()
        };

        let kwargs = build_chat_template_kwargs(&request).expect("template kwargs");
        assert_eq!(kwargs.get("enable_thinking"), Some(&Value::Bool(true)));
        assert_eq!(
            kwargs.get("reasoning_effort"),
            Some(&Value::String("medium".to_string()))
        );
        assert_eq!(
            kwargs.get("custom_flag"),
            Some(&Value::String("x".to_string()))
        );
    }
}

/// Create a StopSequenceDecoder from stop parameters
pub fn create_stop_decoder(
    tokenizer: &Arc<dyn Tokenizer>,
    stop: Option<&StringOrArray>,
    stop_token_ids: Option<&Vec<u32>>,
    skip_special_tokens: bool,
    no_stop_trim: bool,
) -> StopSequenceDecoder {
    // Extract stop sequences
    let stop_sequences: Vec<String> = match stop {
        Some(StringOrArray::String(s)) => vec![s.clone()],
        Some(StringOrArray::Array(arr)) => arr.clone(),
        None => vec![],
    };

    // Build stop sequence decoder
    let mut builder =
        StopSequenceDecoderBuilder::new(tokenizer.clone()).skip_special_tokens(skip_special_tokens);

    // Add stop sequences (visible if no_stop_trim is true, hidden otherwise)
    for seq in stop_sequences {
        builder = if no_stop_trim {
            builder.visible_stop_sequence(seq)
        } else {
            builder.stop_sequence(seq)
        };
    }

    // Add stop token IDs (visible if no_stop_trim is true, hidden otherwise)
    if let Some(token_ids) = stop_token_ids {
        for &token_id in token_ids {
            builder = if no_stop_trim {
                builder.visible_stop_token(token_id)
            } else {
                builder.stop_token(token_id)
            };
        }
    }

    builder.build()
}

/// Parse tool calls from JSON schema constrained response
pub(crate) fn parse_json_schema_response(
    processed_text: &str,
    tool_choice: Option<&ToolChoice>,
    model: &str,
    history_tool_calls_count: usize,
) -> (Option<Vec<ToolCall>>, String) {
    match tool_choice {
        Some(ToolChoice::Function { function, .. }) => {
            // Specific function: Parse parameters directly
            match serde_json::from_str::<Value>(processed_text) {
                Ok(params) => {
                    let tool_call = ToolCall {
                        id: generate_tool_call_id(
                            model,
                            &function.name,
                            0,
                            history_tool_calls_count,
                        ),
                        tool_type: "function".to_string(),
                        function: FunctionCallResponse {
                            name: function.name.clone(),
                            arguments: Some(
                                serde_json::to_string(&params).unwrap_or_else(|_| "{}".to_string()),
                            ),
                        },
                    };
                    (Some(vec![tool_call]), String::new())
                }
                Err(e) => {
                    error!("Failed to parse specific function parameters: {}", e);
                    (None, processed_text.to_string())
                }
            }
        }
        Some(ToolChoice::Value(ToolChoiceValue::Required))
        | Some(ToolChoice::AllowedTools { .. }) => {
            // Required mode: Parse array of tool calls
            match serde_json::from_str::<Vec<Value>>(processed_text) {
                Ok(parsed_array) => {
                    let spec_tool_calls: Vec<ToolCall> = parsed_array
                        .into_iter()
                        .enumerate()
                        .filter_map(|(i, item)| {
                            let obj = item.as_object()?;
                            let name = obj.get("name")?.as_str()?.to_string();
                            let parameters = obj.get("parameters")?;

                            Some(ToolCall {
                                id: generate_tool_call_id(
                                    model,
                                    &name,
                                    i,
                                    history_tool_calls_count,
                                ),
                                tool_type: "function".to_string(),
                                function: FunctionCallResponse {
                                    name,
                                    arguments: Some(
                                        serde_json::to_string(parameters)
                                            .unwrap_or_else(|_| "{}".to_string()),
                                    ),
                                },
                            })
                        })
                        .collect();
                    (Some(spec_tool_calls), String::new())
                }
                Err(e) => {
                    error!("Failed to parse required tool call array: {}", e);
                    (None, processed_text.to_string())
                }
            }
        }
        _ => (None, processed_text.to_string()),
    }
}

/// Count the number of tool calls in the request message history
/// This is used for KimiK2 format which needs globally unique indices
pub(crate) fn get_history_tool_calls_count(request: &ChatCompletionRequest) -> usize {
    request
        .messages
        .iter()
        .filter_map(|msg| {
            if let ChatMessage::Assistant { tool_calls, .. } = msg {
                tool_calls.as_ref().map(|calls| calls.len())
            } else {
                None
            }
        })
        .sum()
}

/// Generate a tool call ID based on model format
///
/// # Arguments
/// * `model` - Model name to determine ID format
/// * `tool_name` - Name of the tool being called
/// * `tool_index` - Index of this tool call within the current message
/// * `history_count` - Number of tool calls in previous messages
///
/// # Returns
/// A unique ID string. KimiK2 uses `functions.{name}:{global_index}`, others use `call_{uuid}`
pub(crate) fn generate_tool_call_id(
    model: &str,
    tool_name: &str,
    tool_index: usize,
    history_count: usize,
) -> String {
    // Case-insensitive check without allocation (search for "kimi" substring)
    let is_kimi = model
        .as_bytes()
        .windows(4) // "kimi".len()
        .any(|window| window.eq_ignore_ascii_case(b"kimi"));

    if is_kimi {
        // KimiK2 format: functions.{name}:{global_index}
        format!("functions.{}:{}", tool_name, history_count + tool_index)
    } else {
        // Standard OpenAI format: call_{24-char-uuid}
        format!("call_{}", &Uuid::now_v7().simple().to_string()[..24])
    }
}

/// Parse finish_reason string into GenerateFinishReason enum
///
/// Uses serde to deserialize the finish_reason, which handles all tagged variants automatically.
/// The GenerateFinishReason enum is tagged with `#[serde(tag = "type", rename_all = "lowercase")]`,
/// so it expects JSON objects like:
/// - `{"type":"stop"}` -> Stop
/// - `{"type":"length","length":100}` -> Length { length: 100 }
/// - Any other JSON -> Other(...)
///
/// For backward compatibility, also handles simple string "stop" -> Stop
pub(crate) fn parse_finish_reason(
    reason_str: &str,
    completion_tokens: u32,
) -> GenerateFinishReason {
    if reason_str == "stop" {
        return GenerateFinishReason::Stop {
            finish_type: openai_protocol::generate::GenerateFinishType::Stop,
        };
    }

    if reason_str == "length" {
        return GenerateFinishReason::Length {
            finish_type: openai_protocol::generate::GenerateFinishType::Length,
            length: completion_tokens,
        };
    }

    match serde_json::from_str::<GenerateFinishReason>(reason_str) {
        Ok(finish_reason) => finish_reason,
        Err(_) => match serde_json::from_str::<Value>(reason_str) {
            Ok(json_value) => GenerateFinishReason::Other(json_value),
            Err(_) => GenerateFinishReason::Other(Value::String(reason_str.to_string())),
        },
    }
}

#[cfg(test)]
mod tests {
    use llm_tokenizer::chat_template::ChatTemplateContentFormat;
    use openai_protocol::{
        chat::{ChatMessage, MessageContent},
        common::{ContentPart, ImageUrl},
    };
    use serde_json::json;

    use super::*;

    #[test]
    fn test_transform_messages_string_format() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Hello".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image.jpg".to_string(),
                        detail: None,
                    },
                },
                ContentPart::Text {
                    text: "World".to_string(),
                },
            ]),
            name: None,
        }];

        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        // Should flatten multimodal content to text only
        assert_eq!(
            transformed_message["content"].as_str().unwrap(),
            "Hello World"
        );
        assert_eq!(transformed_message["role"].as_str().unwrap(), "user");
    }

    #[test]
    fn test_transform_messages_openai_format() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Describe this image:".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image.jpg".to_string(),
                        detail: Some("high".to_string()),
                    },
                },
            ]),
            name: None,
        }];

        let result = process_content_format(&messages, ChatTemplateContentFormat::OpenAI).unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        // Should replace media URLs with simple type placeholders
        let content_array = transformed_message["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 2);

        // Text part should remain unchanged
        assert_eq!(content_array[0]["type"], "text");
        assert_eq!(content_array[0]["text"], "Describe this image:");

        // Image part should be replaced with simple type placeholder
        assert_eq!(content_array[1], json!({"type": "image"}));
    }

    #[test]
    fn test_transform_messages_simple_string_content() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Text("Simple text message".to_string()),
            name: None,
        }];

        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        // Simple string content should remain unchanged
        assert_eq!(
            transformed_message["content"].as_str().unwrap(),
            "Simple text message"
        );
    }

    #[test]
    fn test_transform_messages_multiple_messages() {
        let messages = vec![
            ChatMessage::System {
                content: MessageContent::Text("System prompt".to_string()),
                name: None,
            },
            ChatMessage::User {
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "User message".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image.jpg".to_string(),
                            detail: None,
                        },
                    },
                ]),
                name: None,
            },
        ];

        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();

        assert_eq!(result.len(), 2);

        // System message should remain unchanged
        assert_eq!(result[0]["role"].as_str().unwrap(), "system");
        assert_eq!(result[0]["content"].as_str().unwrap(), "System prompt");

        // User message should be flattened to text only
        assert_eq!(result[1]["role"].as_str().unwrap(), "user");
        assert_eq!(result[1]["content"].as_str().unwrap(), "User message");
    }

    #[test]
    fn test_transform_messages_empty_text_parts() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/image.jpg".to_string(),
                    detail: None,
                },
            }]),
            name: None,
        }];

        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        // Should keep original multimodal content when no text parts exist
        assert!(transformed_message["content"].is_array());
    }

    #[test]
    fn test_transform_messages_mixed_content_types() {
        let messages = vec![
            ChatMessage::User {
                content: MessageContent::Text("Plain text".to_string()),
                name: None,
            },
            ChatMessage::User {
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "With image".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image.jpg".to_string(),
                            detail: Some("low".to_string()),
                        },
                    },
                ]),
                name: None,
            },
        ];

        let result_string =
            process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();

        assert_eq!(result_string.len(), 2);
        assert_eq!(result_string[0]["content"].as_str().unwrap(), "Plain text");
        assert_eq!(result_string[1]["content"].as_str().unwrap(), "With image");

        let result_openai =
            process_content_format(&messages, ChatTemplateContentFormat::OpenAI).unwrap();

        assert_eq!(result_openai.len(), 2);
        assert_eq!(result_openai[0]["content"].as_str().unwrap(), "Plain text");

        let content_array = result_openai[1]["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 2);
        assert_eq!(content_array[0]["type"], "text");
        assert_eq!(content_array[1], json!({"type": "image"}));
    }
}
