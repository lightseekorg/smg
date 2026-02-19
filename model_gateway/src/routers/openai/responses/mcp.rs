//! MCP (Model Context Protocol) Integration Module
//!
//! This module contains all MCP-related functionality for the OpenAI router:
//! - Tool loop state management for multi-turn tool calling
//! - MCP tool execution and result handling
//! - Output item builders for MCP-specific response formats
//! - SSE event generation for streaming MCP operations
//! - Payload transformation for MCP tool interception
//! - Metadata injection for MCP operations

use std::io;

use axum::http::HeaderMap;
use bytes::Bytes;
use openai_protocol::{
    event_types::{
        is_function_call_type, CodeInterpreterCallEvent, FileSearchCallEvent, ItemType, McpEvent,
        OutputItemEvent, WebSearchCallEvent,
    },
    responses::{generate_id, ResponseInput, ResponsesRequest},
};
use serde_json::{json, to_value, Value};
use smg_mcp::{McpToolSession, ResponseFormat, ResponseTransformer, ToolExecutionInput};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use super::tool_handler::FunctionCallInProgress;
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::{error, header_utils::apply_request_headers, mcp_utils::DEFAULT_MAX_ITERATIONS},
};

/// State for tracking multi-turn tool calling loop
pub(super) struct ToolLoopState {
    /// Current iteration number (starts at 0, increments with each tool call)
    pub iteration: usize,
    /// Total number of tool calls executed
    pub total_calls: usize,
    /// Conversation history (function_call and function_call_output items)
    pub conversation_history: Vec<Value>,
    /// Original user input (preserved for building resume payloads)
    pub original_input: ResponseInput,
    /// Transformed output items (mcp_call, web_search_call, etc.) - stored to avoid reconstruction
    pub mcp_call_items: Vec<Value>,
}

impl ToolLoopState {
    pub fn new(original_input: ResponseInput) -> Self {
        Self {
            iteration: 0,
            total_calls: 0,
            conversation_history: Vec::new(),
            original_input,
            mcp_call_items: Vec::new(),
        }
    }

    /// Record a tool call in the loop state
    ///
    /// Stores both the conversation history (for resume payloads) and the
    /// transformed output item (to avoid re-transformation later).
    pub fn record_call(
        &mut self,
        call_id: String,
        tool_name: String,
        args_json_str: String,
        output_str: String,
        transformed_item: Value,
    ) {
        let func_item = json!({
            "type": ItemType::FUNCTION_CALL,
            "call_id": call_id,
            "name": tool_name,
            "arguments": args_json_str
        });
        self.conversation_history.push(func_item);

        let output_item = json!({
            "type": "function_call_output",
            "call_id": call_id,
            "output": output_str
        });
        self.conversation_history.push(output_item);

        self.mcp_call_items.push(transformed_item);
    }
}

/// Execute detected tool calls and send completion events to client
/// Returns false if client disconnected during execution
pub(super) async fn execute_streaming_tool_calls(
    pending_calls: Vec<FunctionCallInProgress>,
    session: &McpToolSession<'_>,
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    state: &mut ToolLoopState,
    sequence_number: &mut u64,
    model_id: &str,
) -> bool {
    for call in pending_calls {
        if call.name.is_empty() {
            warn!(
                "Skipping incomplete tool call: name is empty, args_len={}",
                call.arguments_buffer.len()
            );
            continue;
        }

        info!(
            "Executing tool call during streaming: {} ({})",
            call.name, call.call_id
        );

        let args_str = if call.arguments_buffer.is_empty() {
            "{}"
        } else {
            &call.arguments_buffer
        };

        let response_format = session.tool_response_format(&call.name);
        let server_label = session.resolve_tool_server_label(&call.name);

        let arguments: Value = match serde_json::from_str(args_str) {
            Ok(v) => v,
            Err(e) => {
                let err_str = format!("Failed to parse tool arguments: {}", e);
                warn!("{}", err_str);
                let error_output = json!({ "error": &err_str });
                let mcp_call_item = build_transformed_mcp_call_item(
                    &error_output,
                    &response_format,
                    &call.call_id,
                    &server_label,
                    &call.name,
                    &call.arguments_buffer,
                );
                if !send_tool_call_completion_events(
                    tx,
                    &call,
                    mcp_call_item.clone(),
                    sequence_number,
                ) {
                    return false;
                }
                state.record_call(
                    call.call_id,
                    call.name,
                    call.arguments_buffer,
                    error_output.to_string(),
                    mcp_call_item,
                );
                continue;
            }
        };

        if !send_tool_call_intermediate_event(tx, &call, &response_format, sequence_number) {
            return false;
        }

        debug!("Calling MCP tool '{}' with args: {}", call.name, args_str);
        let tool_output = session
            .execute_tool(ToolExecutionInput {
                call_id: call.call_id.clone(),
                tool_name: call.name.clone(),
                arguments,
            })
            .await;

        Metrics::record_mcp_tool_duration(model_id, &tool_output.tool_name, tool_output.duration);
        Metrics::record_mcp_tool_call(
            model_id,
            &tool_output.tool_name,
            if tool_output.is_error {
                metrics_labels::RESULT_ERROR
            } else {
                metrics_labels::RESULT_SUCCESS
            },
        );

        let output_str = tool_output.output.to_string();
        let mcp_call_item = to_value(tool_output.to_response_item()).unwrap_or_else(|e| {
            warn!(tool = %call.name, error = %e, "Failed to convert item to Value");
            json!({})
        });

        if !send_tool_call_completion_events(tx, &call, mcp_call_item.clone(), sequence_number) {
            return false;
        }

        state.record_call(
            call.call_id,
            call.name,
            call.arguments_buffer,
            output_str,
            mcp_call_item,
        );
    }
    true
}

/// Transform payload to replace MCP/builtin tools with function tools.
///
/// Retains existing function tools from the request, removes non-function tools
/// (MCP, builtin), and appends function tools for discovered MCP server tools.
pub(super) fn prepare_mcp_tools_as_functions(payload: &mut Value, session: &McpToolSession<'_>) {
    let Some(obj) = payload.as_object_mut() else {
        return;
    };

    let mut retained_tools: Vec<Value> = Vec::new();
    if let Some(v) = obj.get_mut("tools") {
        if let Some(arr) = v.as_array_mut() {
            retained_tools = arr
                .drain(..)
                .filter(|item| {
                    item.get("type")
                        .and_then(|v| v.as_str())
                        .map(|s| s == ItemType::FUNCTION)
                        .unwrap_or(false)
                })
                .collect();
        }
    }

    let session_tools = session.build_function_tools_json();
    let mut tools_json = Vec::with_capacity(retained_tools.len() + session_tools.len());
    tools_json.append(&mut retained_tools);
    tools_json.extend(session_tools);

    if !tools_json.is_empty() {
        obj.insert("tools".to_string(), Value::Array(tools_json));
        obj.insert("tool_choice".to_string(), Value::String("auto".to_string()));
    }
}

/// Build a resume payload with conversation history
pub(super) fn build_resume_payload(
    base_payload: &Value,
    conversation_history: &[Value],
    original_input: &ResponseInput,
    tools_json: &Value,
    is_streaming: bool,
) -> Result<Value, String> {
    let mut payload = base_payload.clone();

    let obj = payload
        .as_object_mut()
        .ok_or_else(|| "payload not an object".to_string())?;

    let mut input_array = Vec::with_capacity(1 + conversation_history.len());

    match original_input {
        ResponseInput::Text(text) => {
            let user_item = json!({
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": text }]
            });
            input_array.push(user_item);
        }
        ResponseInput::Items(items) => {
            let items_value =
                to_value(items).map_err(|e| format!("Failed to serialize input items: {}", e))?;
            if let Some(items_arr) = items_value.as_array() {
                input_array.extend_from_slice(items_arr);
            }
        }
    }

    input_array.extend_from_slice(conversation_history);
    obj.insert("input".to_string(), Value::Array(input_array));

    if let Some(tools_arr) = tools_json.as_array() {
        if !tools_arr.is_empty() {
            obj.insert("tools".to_string(), tools_json.clone());
        }
    }

    obj.insert("stream".to_string(), Value::Bool(is_streaming));
    obj.insert("store".to_string(), Value::Bool(false));

    Ok(payload)
}

/// Send mcp_list_tools events to client at the start of streaming
/// Returns false if client disconnected
pub(super) fn send_mcp_list_tools_events(
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    session: &McpToolSession<'_>,
    server_label: &str,
    output_index: usize,
    sequence_number: &mut u64,
    server_key: &str,
) -> bool {
    let tools_item_full = session.build_mcp_list_tools_json(server_label, server_key);
    let item_id = tools_item_full
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // Create empty tools version for the initial added event
    let mut tools_item_empty = tools_item_full.clone();
    if let Some(obj) = tools_item_empty.as_object_mut() {
        obj.insert("tools".to_string(), json!([]));
    }

    // Event 1: response.output_item.added with empty tools
    let event1_payload = json!({
        "type": OutputItemEvent::ADDED,
        "sequence_number": *sequence_number,
        "output_index": output_index,
        "item": tools_item_empty
    });
    *sequence_number += 1;
    let event1 = format!(
        "event: {}\ndata: {}\n\n",
        OutputItemEvent::ADDED,
        event1_payload
    );
    if tx.send(Ok(Bytes::from(event1))).is_err() {
        return false; // Client disconnected
    }

    // Event 2: response.mcp_list_tools.in_progress
    let event2_payload = json!({
        "type": McpEvent::LIST_TOOLS_IN_PROGRESS,
        "sequence_number": *sequence_number,
        "output_index": output_index,
        "item_id": item_id
    });
    *sequence_number += 1;
    let event2 = format!(
        "event: {}\ndata: {}\n\n",
        McpEvent::LIST_TOOLS_IN_PROGRESS,
        event2_payload
    );
    if tx.send(Ok(Bytes::from(event2))).is_err() {
        return false;
    }

    // Event 3: response.mcp_list_tools.completed
    let event3_payload = json!({
        "type": McpEvent::LIST_TOOLS_COMPLETED,
        "sequence_number": *sequence_number,
        "output_index": output_index,
        "item_id": item_id
    });
    *sequence_number += 1;
    let event3 = format!(
        "event: {}\ndata: {}\n\n",
        McpEvent::LIST_TOOLS_COMPLETED,
        event3_payload
    );
    if tx.send(Ok(Bytes::from(event3))).is_err() {
        return false;
    }

    // Event 4: response.output_item.done with full tools list
    let event4_payload = json!({
        "type": OutputItemEvent::DONE,
        "sequence_number": *sequence_number,
        "output_index": output_index,
        "item": tools_item_full
    });
    *sequence_number += 1;
    let event4 = format!(
        "event: {}\ndata: {}\n\n",
        OutputItemEvent::DONE,
        event4_payload
    );
    tx.send(Ok(Bytes::from(event4))).is_ok()
}

/// Send intermediate event during tool execution (searching/interpreting).
/// Returns false if client disconnected.
fn send_tool_call_intermediate_event(
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    call: &FunctionCallInProgress,
    response_format: &ResponseFormat,
    sequence_number: &mut u64,
) -> bool {
    // Determine event type and ID prefix based on response format
    let (event_type, id_prefix) = match response_format {
        ResponseFormat::WebSearchCall => (WebSearchCallEvent::SEARCHING, "ws_"),
        ResponseFormat::CodeInterpreterCall => (CodeInterpreterCallEvent::INTERPRETING, "ci_"),
        ResponseFormat::FileSearchCall => (FileSearchCallEvent::SEARCHING, "fs_"),
        ResponseFormat::Passthrough => return true, // mcp_call has no intermediate event
    };

    let effective_output_index = call.effective_output_index();

    // Transform call_id from fc_* to appropriate prefix
    let item_id = call
        .call_id
        .strip_prefix("fc_")
        .map(|stripped| format!("{}{}", id_prefix, stripped))
        .unwrap_or_else(|| call.call_id.clone());

    let event_payload = json!({
        "type": event_type,
        "sequence_number": *sequence_number,
        "output_index": effective_output_index,
        "item_id": item_id
    });
    *sequence_number += 1;

    let event = format!("event: {}\ndata: {}\n\n", event_type, event_payload);
    tx.send(Ok(Bytes::from(event))).is_ok()
}

/// Send tool call completion events after tool execution.
/// Handles mcp_call, web_search_call, code_interpreter_call, and file_search_call items.
/// Returns false if client disconnected.
pub(super) fn send_tool_call_completion_events(
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    call: &FunctionCallInProgress,
    tool_call_item: Value,
    sequence_number: &mut u64,
) -> bool {
    let effective_output_index = call.effective_output_index();

    let item_id = tool_call_item
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // Determine the completion event type based on item type
    let item_type = tool_call_item
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let completed_event_type: &str = match item_type {
        ItemType::WEB_SEARCH_CALL => WebSearchCallEvent::COMPLETED,
        ItemType::CODE_INTERPRETER_CALL => CodeInterpreterCallEvent::COMPLETED,
        ItemType::FILE_SEARCH_CALL => FileSearchCallEvent::COMPLETED,
        _ => McpEvent::CALL_COMPLETED, // Default to mcp_call for mcp_call and unknown types
    };

    // Event 1: response.<type>.completed
    let completed_payload = json!({
        "type": completed_event_type,
        "sequence_number": *sequence_number,
        "output_index": effective_output_index,
        "item_id": item_id
    });
    *sequence_number += 1;

    let completed_event = format!(
        "event: {}\ndata: {}\n\n",
        completed_event_type, completed_payload
    );
    if tx.send(Ok(Bytes::from(completed_event))).is_err() {
        return false;
    }

    // Event 2: response.output_item.done (with completed tool call)
    let done_payload = json!({
        "type": OutputItemEvent::DONE,
        "sequence_number": *sequence_number,
        "output_index": effective_output_index,
        "item": tool_call_item
    });
    *sequence_number += 1;

    let done_event = format!(
        "event: {}\ndata: {}\n\n",
        OutputItemEvent::DONE,
        done_payload
    );
    tx.send(Ok(Bytes::from(done_event))).is_ok()
}

/// Inject MCP metadata into a streaming response
pub(super) fn inject_mcp_metadata_streaming(
    response: &mut Value,
    state: &ToolLoopState,
    session: &McpToolSession<'_>,
) {
    let mcp_servers = session.mcp_servers();

    if let Some(output_array) = response.get_mut("output").and_then(|v| v.as_array_mut()) {
        output_array.retain(|item| {
            item.get("type").and_then(|t| t.as_str()) != Some(ItemType::MCP_LIST_TOOLS)
        });

        for (label, key) in mcp_servers.iter().rev() {
            let list_tools_item = session.build_mcp_list_tools_json(label, key);
            output_array.insert(0, list_tools_item);
        }

        // Use stored transformed items (no reconstruction needed)
        let mut insert_pos = mcp_servers.len();
        for item in &state.mcp_call_items {
            output_array.insert(insert_pos, item.clone());
            insert_pos += 1;
        }
    } else if let Some(obj) = response.as_object_mut() {
        let mut output_items = Vec::new();
        for (label, key) in mcp_servers.iter() {
            output_items.push(session.build_mcp_list_tools_json(label, key));
        }
        // Use stored transformed items (no reconstruction needed)
        output_items.extend(state.mcp_call_items.iter().cloned());
        obj.insert("output".to_string(), Value::Array(output_items));
    }
}

/// Execute the tool calling loop
pub(super) async fn execute_tool_loop(
    client: &reqwest::Client,
    url: &str,
    headers: Option<&HeaderMap>,
    initial_payload: Value,
    original_body: &ResponsesRequest,
    session: &McpToolSession<'_>,
) -> Result<Value, String> {
    let mut state = ToolLoopState::new(original_body.input.clone());
    let max_tool_calls = original_body.max_tool_calls.map(|n| n as usize);
    let base_payload = initial_payload.clone();
    let tools_json = base_payload.get("tools").cloned().unwrap_or(json!([]));
    let mut current_payload = initial_payload;

    info!(
        "Starting tool loop: max_tool_calls={:?}, max_iterations={}",
        max_tool_calls, DEFAULT_MAX_ITERATIONS
    );

    loop {
        let request_builder = client.post(url).json(&current_payload);
        let request_builder = if let Some(headers) = headers {
            apply_request_headers(headers, request_builder, true)
        } else {
            request_builder
        };

        let response = request_builder
            .send()
            .await
            .map_err(|e| format!("upstream request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let body = error::sanitize_error_body(&body);
            return Err(format!("upstream error {}: {}", status, body));
        }

        let mut response_json = response
            .json::<Value>()
            .await
            .map_err(|e| format!("parse response: {}", e))?;

        let function_calls = extract_function_calls(&response_json);
        if function_calls.is_empty() {
            info!(
                "Tool loop completed: {} iterations, {} total calls",
                state.iteration, state.total_calls
            );
            if state.total_calls > 0 {
                inject_mcp_metadata_streaming(&mut response_json, &state, session);
            }
            return Ok(response_json);
        }

        state.iteration += 1;
        Metrics::record_mcp_tool_iteration(&original_body.model);

        info!(
            "Tool loop iteration {}: {} function call(s) detected",
            state.iteration,
            function_calls.len()
        );

        let effective_limit = match max_tool_calls {
            Some(user_max) => user_max.min(DEFAULT_MAX_ITERATIONS),
            None => DEFAULT_MAX_ITERATIONS,
        };

        for call in function_calls {
            state.total_calls += 1;

            if state.total_calls > effective_limit {
                warn!(
                    "Reached tool call limit ({}) after {} calls",
                    effective_limit, state.total_calls
                );
                return build_incomplete_response(
                    response_json,
                    state,
                    "max_tool_calls",
                    session,
                    original_body,
                );
            }
            let arguments: Value = match serde_json::from_str(&call.arguments) {
                Ok(v) => v,
                Err(e) => {
                    warn!(tool = %call.name, error = %e, "Failed to parse tool arguments as JSON");
                    let error_output = format!("Invalid tool arguments: {}", e);
                    let response_format = session.tool_response_format(&call.name);
                    let server_label = session.resolve_tool_server_label(&call.name);
                    let error_json = json!({ "error": &error_output });
                    let transformed_item = build_transformed_mcp_call_item(
                        &error_json,
                        &response_format,
                        &call.call_id,
                        &server_label,
                        &call.name,
                        &call.arguments,
                    );

                    Metrics::record_mcp_tool_call(
                        &original_body.model,
                        &call.name,
                        metrics_labels::RESULT_ERROR,
                    );

                    state.record_call(
                        call.call_id,
                        call.name,
                        call.arguments,
                        error_output,
                        transformed_item,
                    );
                    continue;
                }
            };

            debug!(
                "Calling MCP tool '{}' with args: {}",
                call.name, call.arguments
            );
            let tool_output = session
                .execute_tool(ToolExecutionInput {
                    call_id: call.call_id.clone(),
                    tool_name: call.name.clone(),
                    arguments,
                })
                .await;

            Metrics::record_mcp_tool_duration(
                &original_body.model,
                &tool_output.tool_name,
                tool_output.duration,
            );
            Metrics::record_mcp_tool_call(
                &original_body.model,
                &tool_output.tool_name,
                if tool_output.is_error {
                    metrics_labels::RESULT_ERROR
                } else {
                    metrics_labels::RESULT_SUCCESS
                },
            );

            let output_str = tool_output.output.to_string();
            let transformed_item = to_value(tool_output.to_response_item()).unwrap_or_else(|e| {
                warn!(tool = %call.name, error = %e, "Failed to convert item to Value");
                json!({})
            });

            state.record_call(
                call.call_id,
                call.name,
                call.arguments,
                output_str,
                transformed_item,
            );
        }

        current_payload = build_resume_payload(
            &base_payload,
            &state.conversation_history,
            &state.original_input,
            &tools_json,
            false,
        )?;
    }
}

/// Build an incomplete response when limits are exceeded
pub(super) fn build_incomplete_response(
    mut response: Value,
    state: ToolLoopState,
    reason: &str,
    session: &McpToolSession<'_>,
    _original_body: &ResponsesRequest,
) -> Result<Value, String> {
    let obj = response
        .as_object_mut()
        .ok_or_else(|| "response not an object".to_string())?;

    // Set status to completed (not failed - partial success)
    obj.insert("status".to_string(), Value::String("completed".to_string()));

    obj.insert(
        "incomplete_details".to_string(),
        json!({ "reason": reason }),
    );

    let mcp_servers = session.mcp_servers();

    // Convert any function_call in output to mcp_call format
    if let Some(output_array) = obj.get_mut("output").and_then(|v| v.as_array_mut()) {
        // Find any function_call items and convert them to mcp_call (incomplete)
        let mut incomplete_items = Vec::new();
        for item in output_array.iter() {
            let item_type = item.get("type").and_then(|t| t.as_str());
            if item_type.is_some_and(is_function_call_type) {
                let tool_name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let args = item
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}");

                // Mark as incomplete - not executed
                let resolved_label = session.resolve_tool_server_label(tool_name);
                let mcp_call_item = build_mcp_call_item(
                    tool_name,
                    args,
                    "", // No output - wasn't executed
                    &resolved_label,
                    false, // Not successful
                    Some("Not executed - response stopped due to limit"),
                );
                incomplete_items.push(mcp_call_item);
            }
        }

        // Add mcp_list_tools and executed mcp_call items at the beginning
        if state.total_calls > 0 || !incomplete_items.is_empty() {
            for (label, key) in mcp_servers.iter().rev() {
                let list_tools_item = session.build_mcp_list_tools_json(label, key);
                output_array.insert(0, list_tools_item);
            }

            // Insert stored transformed items for executed calls (no reconstruction needed)
            let mut insert_pos = mcp_servers.len();
            for item in &state.mcp_call_items {
                output_array.insert(insert_pos, item.clone());
                insert_pos += 1;
            }

            // Add incomplete mcp_call items (never executed, so no stored item)
            for item in incomplete_items {
                output_array.insert(insert_pos, item);
                insert_pos += 1;
            }
        }
    }

    if let Some(metadata_val) = obj.get_mut("metadata") {
        if let Some(metadata_obj) = metadata_val.as_object_mut() {
            if let Some(mcp_val) = metadata_obj.get_mut("mcp") {
                if let Some(mcp_obj) = mcp_val.as_object_mut() {
                    mcp_obj.insert(
                        "truncation_warning".to_string(),
                        Value::String(format!(
                            "Loop terminated at {} iterations, {} total calls (reason: {})",
                            state.iteration, state.total_calls, reason
                        )),
                    );
                }
            }
        }
    }

    Ok(response)
}

/// Build a mcp_call output item
pub(super) fn build_mcp_call_item(
    tool_name: &str,
    arguments: &str,
    output: &str,
    server_label: &str,
    success: bool,
    error: Option<&str>,
) -> Value {
    json!({
        "id": generate_id("mcp"),
        "type": ItemType::MCP_CALL,
        "status": if success { "completed" } else { "failed" },
        "approval_request_id": Value::Null,
        "arguments": arguments,
        "error": error,
        "name": tool_name,
        "output": output,
        "server_label": server_label
    })
}

/// Build a transformed output item using ResponseTransformer
///
/// Converts the output using the tool's response_format to the correctly-typed
/// output item (mcp_call, web_search_call, code_interpreter_call, file_search_call).
/// Returns the result as a JSON Value for SSE event streaming.
pub(super) fn build_transformed_mcp_call_item(
    output: &Value,
    response_format: &ResponseFormat,
    call_id: &str,
    server_label: &str,
    tool_name: &str,
    arguments: &str,
) -> Value {
    let output_item = ResponseTransformer::transform(
        output,
        response_format,
        call_id,
        server_label,
        tool_name,
        arguments,
    );
    to_value(&output_item).unwrap_or_else(|e| {
        warn!(tool = %tool_name, error = %e, "Failed to serialize transformed output item");
        json!({})
    })
}

/// A function call extracted from a non-streaming response
pub(super) struct ExtractedFunctionCall {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

/// Extract all function calls from a response
pub(super) fn extract_function_calls(resp: &Value) -> Vec<ExtractedFunctionCall> {
    let Some(output) = resp.get("output").and_then(|v| v.as_array()) else {
        return Vec::new();
    };

    let mut calls = Vec::new();
    for item in output {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let Some(t) = obj.get("type").and_then(|v| v.as_str()) else {
            continue;
        };
        if !is_function_call_type(t) {
            continue;
        }

        let call_id = obj
            .get("call_id")
            .and_then(|v| v.as_str())
            .or_else(|| obj.get("id").and_then(|v| v.as_str()));
        let name = obj.get("name").and_then(|v| v.as_str());
        let arguments = obj.get("arguments").and_then(|v| v.as_str());

        if let (Some(call_id), Some(name), Some(arguments)) = (call_id, name, arguments) {
            calls.push(ExtractedFunctionCall {
                call_id: call_id.to_string(),
                name: name.to_string(),
                arguments: arguments.to_string(),
            });
        }
    }

    calls
}
