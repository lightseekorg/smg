//! Streaming response handling for OpenAI-compatible responses
//!
//! This module handles all streaming-related functionality including:
//! - SSE (Server-Sent Events) parsing and forwarding
//! - Streaming response accumulation for persistence
//! - Tool call detection and interception during streaming
//! - MCP tool execution loops within streaming responses
//! - Event transformation and output index remapping

use std::io;

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, HeaderValue, StatusCode},
    response::Response,
};
use bytes::Bytes;
use openai_protocol::{
    event_types::{
        is_function_call_type, is_response_event, CodeInterpreterCallEvent, FileSearchCallEvent,
        FunctionCallEvent, ImageGenerationCallEvent, ItemType, McpEvent, OutputItemEvent,
        ResponseEvent, WebSearchCallEvent,
    },
    responses::{ResponseTool, ResponsesRequest},
};
use serde_json::{json, Value};
use smg_mcp::McpToolSession;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::warn;

use super::{
    common::{extract_output_index, get_event_type},
    utils::response_tool_to_value,
};
const SSE_DONE: &str = "data: [DONE]\n\n";

use crate::routers::{
    common::agent_loop::{normalize_output_item_id, OutputFamily},
    error,
    openai::{
        context::{RequestContext, StreamingEventContext},
        mcp::StreamingToolHandler,
    },
};

fn output_item_type_for_family(family: OutputFamily) -> &'static str {
    match family {
        OutputFamily::McpCall => ItemType::MCP_CALL,
        OutputFamily::WebSearchCall => ItemType::WEB_SEARCH_CALL,
        OutputFamily::CodeInterpreterCall => ItemType::CODE_INTERPRETER_CALL,
        OutputFamily::FileSearchCall => ItemType::FILE_SEARCH_CALL,
        OutputFamily::ImageGenerationCall => ItemType::IMAGE_GENERATION_CALL,
        OutputFamily::Function => ItemType::FUNCTION_CALL,
    }
}

/// Apply all transformations to event data in-place (rewrite + transform)
/// Optimized to parse JSON only once instead of multiple times
/// Returns true if any changes were made
pub(super) fn apply_event_transformations_inplace(
    parsed_data: &mut Value,
    ctx: &StreamingEventContext<'_>,
) -> bool {
    let mut changed = false;

    // 1. Apply rewrite_streaming_block logic (store, previous_response_id, tools masking)
    let event_type = parsed_data
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let should_patch = is_response_event(event_type);
    // Owned copy needed for the match below since we mutate parsed_data
    let event_type = event_type.to_string();

    if should_patch {
        if let Some(response_obj) = parsed_data
            .get_mut("response")
            .and_then(|v| v.as_object_mut())
        {
            let desired_store = Value::Bool(ctx.original_request.store.unwrap_or(true));
            if response_obj.get("store") != Some(&desired_store) {
                response_obj.insert("store".to_string(), desired_store);
                changed = true;
            }

            if let Some(prev_id) = ctx.previous_response_id {
                let needs_previous = response_obj
                    .get("previous_response_id")
                    .map(|v| v.is_null() || v.as_str().map(|s| s.is_empty()).unwrap_or(false))
                    .unwrap_or(true);

                if needs_previous {
                    response_obj.insert(
                        "previous_response_id".to_string(),
                        Value::String(prev_id.to_string()),
                    );
                    changed = true;
                }
            }

            // Mask tools from function to MCP format (optimized without cloning)
            if response_obj.get("tools").is_some() {
                let requested_mcp = ctx
                    .original_request
                    .tools
                    .as_ref()
                    .map(|tools| tools.iter().any(|t| matches!(t, ResponseTool::Mcp(_))))
                    .unwrap_or(false);

                if requested_mcp {
                    if let Some(mcp_tools) = build_mcp_tools_value(ctx.original_request) {
                        response_obj.insert("tools".to_string(), mcp_tools);
                        response_obj
                            .entry("tool_choice".to_string())
                            .or_insert(Value::String("auto".to_string()));
                        changed = true;
                    }
                }
            }
        }
    }

    // 2. Apply transform_streaming_event logic (function_call → mcp_call/web_search_call)
    match event_type.as_str() {
        OutputItemEvent::ADDED | OutputItemEvent::DONE => {
            if let Some(item) = parsed_data.get_mut("item") {
                if let Some(item_type) = item.get("type").and_then(|v| v.as_str()) {
                    if is_function_call_type(item_type) {
                        // Look up response_format for the tool
                        let tool_name = item
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();

                        // Only transform if this is an MCP tool; keep function_call unchanged
                        if let Some(session) =
                            ctx.session.filter(|s| s.has_exposed_tool(&tool_name))
                        {
                            let response_format = session.tool_response_format(&tool_name);
                            let family = OutputFamily::from_mcp_format(&response_format);
                            let new_type = output_item_type_for_family(family);

                            item["type"] = json!(new_type);
                            if new_type == ItemType::MCP_CALL {
                                let label = session.resolve_tool_server_label(&tool_name);
                                item["server_label"] = json!(label);
                            }

                            // Normalize to the shared family-specific external id.
                            if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                                item["id"] = json!(normalize_output_item_id(family, id));
                            }

                            changed = true;
                        }
                    }
                }
            }
        }
        _ => {}
    }

    changed
}

/// Helper to build MCP tools value
fn build_mcp_tools_value(original_body: &ResponsesRequest) -> Option<Value> {
    let tools = original_body.tools.as_ref()?;
    let mcp_tools: Vec<Value> = tools
        .iter()
        .filter(|t| matches!(t, ResponseTool::Mcp(_)))
        .filter_map(response_tool_to_value)
        .collect();

    if mcp_tools.is_empty() {
        None
    } else {
        Some(Value::Array(mcp_tools))
    }
}

/// Send an SSE event to the client channel
/// Returns false if client disconnected
#[inline]
fn send_sse_event(
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    event_name: &str,
    data: &Value,
) -> bool {
    let block = format!("event: {event_name}\ndata: {data}\n\n");
    tx.send(Ok(Bytes::from(block))).is_ok()
}

/// Map function_call event names to mcp_call event names
#[inline]
fn map_event_name(event_name: &str) -> &str {
    match event_name {
        FunctionCallEvent::ARGUMENTS_DELTA => McpEvent::CALL_ARGUMENTS_DELTA,
        FunctionCallEvent::ARGUMENTS_DONE => McpEvent::CALL_ARGUMENTS_DONE,
        other => other,
    }
}

/// Send buffered function call arguments as a synthetic delta event.
/// Returns false if client disconnected.
fn send_buffered_arguments(
    parsed_data: &mut Value,
    handler: &StreamingToolHandler,
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ctx: &StreamingEventContext<'_>,
    sequence_number: &mut u64,
    mapped_output_index: &mut Option<usize>,
) -> bool {
    let Some(output_index) = extract_output_index(parsed_data) else {
        return true;
    };

    let assigned_index = handler
        .mapped_output_index(output_index)
        .unwrap_or(output_index);
    *mapped_output_index = Some(assigned_index);

    let Some(call) = handler
        .pending_calls
        .iter()
        .find(|c| c.output_index == output_index)
    else {
        return true;
    };

    let is_gateway_call = ctx
        .session
        .is_some_and(|session| session.has_exposed_tool(&call.name));

    let arguments_value = if call.arguments_buffer.is_empty() {
        "{}".to_string()
    } else {
        call.arguments_buffer.clone()
    };

    // Update the done event with full arguments
    parsed_data["arguments"] = Value::String(arguments_value.clone());

    let item_id = parsed_data
        .get("item_id")
        .and_then(|v| v.as_str())
        .or(call.item_id.as_deref())
        .unwrap_or(call.call_id.as_str());
    let transformed_item_id = if is_gateway_call {
        normalize_output_item_id(OutputFamily::McpCall, item_id)
    } else {
        item_id.to_string()
    };
    if is_gateway_call {
        parsed_data["type"] = json!(McpEvent::CALL_ARGUMENTS_DONE);
        parsed_data["item_id"] = Value::String(transformed_item_id.clone());
    }

    // Build synthetic delta event
    let mut delta_event = json!({
        "type": if is_gateway_call {
            McpEvent::CALL_ARGUMENTS_DELTA
        } else {
            FunctionCallEvent::ARGUMENTS_DELTA
        },
        "sequence_number": *sequence_number,
        "output_index": assigned_index,
        "item_id": transformed_item_id,
        "delta": arguments_value,
    });

    // Add obfuscation if present
    let obfuscation = call
        .last_obfuscation
        .as_ref()
        .map(|s| Value::String(s.clone()))
        .or_else(|| parsed_data.get("obfuscation").cloned());

    if let Some(obf) = obfuscation {
        if let Some(obj) = delta_event.as_object_mut() {
            obj.insert("obfuscation".to_string(), obf);
        }
    }

    let delta_event_name = if is_gateway_call {
        McpEvent::CALL_ARGUMENTS_DELTA
    } else {
        FunctionCallEvent::ARGUMENTS_DELTA
    };
    if !send_sse_event(tx, delta_event_name, &delta_event) {
        return false;
    }

    *sequence_number += 1;
    true
}

/// An SSE event to be forwarded to the client, with optional pre-parsed JSON.
pub(super) struct SseEventData<'a> {
    pub raw_block: &'a str,
    pub event_name: Option<&'a str>,
    pub data: &'a str,
    /// Pre-parsed JSON value. When `Some`, avoids re-parsing `data`.
    pub pre_parsed: Option<Value>,
}

/// Forward and transform a streaming event to the client.
/// Returns false if client disconnected.
pub(super) fn forward_streaming_event(
    event: SseEventData<'_>,
    handler: &mut StreamingToolHandler,
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ctx: &StreamingEventContext<'_>,
    sequence_number: &mut u64,
) -> bool {
    let SseEventData {
        raw_block,
        event_name,
        data,
        pre_parsed,
    } = event;

    if event_name == Some(FunctionCallEvent::ARGUMENTS_DELTA) {
        return true;
    }

    // Use pre-parsed value or parse JSON data
    let mut parsed_data: Value = match pre_parsed {
        Some(v) => v,
        None => match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => {
                let chunk = format!("{raw_block}\n\n");
                return tx.send(Ok(Bytes::from(chunk))).is_ok();
            }
        },
    };

    let event_type = get_event_type(event_name, &parsed_data);
    if event_type == ResponseEvent::COMPLETED {
        return true;
    }

    // Handle function_call_arguments.done - send buffered args first
    let mut mapped_output_index: Option<usize> = None;
    if event_name == Some(FunctionCallEvent::ARGUMENTS_DONE)
        && !send_buffered_arguments(
            &mut parsed_data,
            handler,
            tx,
            ctx,
            sequence_number,
            &mut mapped_output_index,
        )
    {
        return false;
    }

    if mapped_output_index.is_none() {
        if let Some(idx) = extract_output_index(&parsed_data) {
            mapped_output_index = handler.mapped_output_index(idx);
        }
    }
    if let Some(mapped) = mapped_output_index {
        parsed_data["output_index"] = json!(mapped);
    }

    apply_event_transformations_inplace(&mut parsed_data, ctx);

    if let Some(response_obj) = parsed_data
        .get_mut("response")
        .and_then(|v| v.as_object_mut())
    {
        if let Some(original_id) = handler.original_response_id() {
            response_obj.insert("id".to_string(), Value::String(original_id.to_string()));
        }
    }

    if parsed_data.get("sequence_number").is_some() {
        parsed_data["sequence_number"] = json!(*sequence_number);
        *sequence_number += 1;
    }

    let final_data = match serde_json::to_string(&parsed_data) {
        Ok(s) => s,
        Err(_) => {
            let chunk = format!("{raw_block}\n\n");
            return tx.send(Ok(Bytes::from(chunk))).is_ok();
        }
    };

    let final_block = match event_name {
        Some(FunctionCallEvent::ARGUMENTS_DONE) => format!(
            "event: {}\ndata: {}\n\n",
            parsed_data
                .get("type")
                .and_then(|value| value.as_str())
                .unwrap_or(FunctionCallEvent::ARGUMENTS_DONE),
            final_data
        ),
        Some(evt) => format!("event: {}\ndata: {}\n\n", map_event_name(evt), final_data),
        None => format!("data: {final_data}\n\n"),
    };

    if tx.send(Ok(Bytes::from(final_block))).is_err() {
        return false;
    }

    if event_name == Some(OutputItemEvent::ADDED)
        && !maybe_inject_tool_in_progress(&parsed_data, tx, sequence_number)
    {
        return false;
    }

    true
}

/// Inject in_progress event after a tool call item is added.
/// Handles mcp_call, web_search_call, code_interpreter_call, file_search_call,
/// and image_generation_call items.
/// Returns false if client disconnected.
fn maybe_inject_tool_in_progress(
    parsed_data: &Value,
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    sequence_number: &mut u64,
) -> bool {
    let Some(item) = parsed_data.get("item") else {
        return true;
    };

    let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");

    // Determine the in_progress event type based on item type
    let event_type = match item_type {
        ItemType::MCP_CALL => McpEvent::CALL_IN_PROGRESS,
        ItemType::WEB_SEARCH_CALL => WebSearchCallEvent::IN_PROGRESS,
        ItemType::CODE_INTERPRETER_CALL => CodeInterpreterCallEvent::IN_PROGRESS,
        ItemType::FILE_SEARCH_CALL => FileSearchCallEvent::IN_PROGRESS,
        ItemType::IMAGE_GENERATION_CALL => ImageGenerationCallEvent::IN_PROGRESS,
        _ => return true, // Not a tool call item, nothing to inject
    };

    let Some(item_id) = item.get("id").and_then(|v| v.as_str()) else {
        return true;
    };
    let Some(output_index) = parsed_data.get("output_index").and_then(|v| v.as_u64()) else {
        return true;
    };

    let event = json!({
        "type": event_type,
        "sequence_number": *sequence_number,
        "output_index": output_index,
        "item_id": item_id
    });
    *sequence_number += 1;

    send_sse_event(tx, event_type, &event)
}

/// Main entry point for streaming responses.
pub async fn handle_streaming_response(mut ctx: RequestContext) -> Response {
    use serde_json::to_value;
    use uuid::Uuid;

    use super::agent_loop_adapter::{
        OpenAiResponseStreamSink, OpenAiStreamingAdapter, OpenAiUpstreamHandle,
    };
    use crate::routers::{
        common::{
            agent_loop::{run_agent_loop, AgentLoopContext, AgentLoopState, PreparedLoopInput},
            header_utils::extract_forwardable_request_headers,
            mcp_utils::ensure_mcp_connection,
            persistence_utils::persist_conversation_items,
        },
        openai::context::{ResponsesPayloadState, StorageHandles},
    };

    let payload_state = match ctx.take_payload() {
        Some(state) => state,
        None => return error::internal_error("internal_error", "Payload not prepared"),
    };
    let ResponsesPayloadState {
        previous_response_id: _,
        existing_mcp_list_tools_labels,
        prepared_input,
        control_items,
    } = ctx.take_responses_payload().unwrap_or_default();
    let original_request = match ctx.responses_request() {
        Some(request) => request.clone(),
        None => return error::internal_error("internal_error", "Expected responses request"),
    };
    let prepared_input = prepared_input.unwrap_or_else(|| original_request.input.clone());
    let worker = match ctx.worker() {
        Some(worker) => worker.clone(),
        None => return error::internal_error("internal_error", "Worker not selected"),
    };
    let mcp_orchestrator = match ctx.components.mcp_orchestrator() {
        Some(orchestrator) => orchestrator.clone(),
        None => return error::internal_error("internal_error", "MCP orchestrator required"),
    };

    let (_has_gateway_tools, mcp_servers) =
        match ensure_mcp_connection(&mcp_orchestrator, original_request.tools.as_deref()).await {
            Ok(result) => result,
            Err(response) => return response,
        };

    let prepared = PreparedLoopInput::new(prepared_input, control_items);
    let state = AgentLoopState::new(
        prepared.upstream_input.clone(),
        existing_mcp_list_tools_labels.into_iter().collect(),
    );
    let loop_ctx_max_tool_calls = original_request.max_tool_calls.map(|value| value as usize);
    let loop_ctx_request = original_request.clone();
    let session_request_id = original_request
        .request_id
        .clone()
        .unwrap_or_else(|| format!("resp_{}", Uuid::now_v7()));
    let forwarded_headers = extract_forwardable_request_headers(ctx.headers());
    let client = ctx.components.client().clone();
    let request_headers = ctx.headers().cloned();

    let storage = match (
        ctx.components.response_storage(),
        ctx.components.conversation_storage(),
        ctx.components.conversation_item_storage(),
        ctx.components.conversation_memory_writer(),
    ) {
        (Some(response), Some(conversation), Some(conversation_item), Some(_memory_writer)) => {
            StorageHandles {
                response: response.clone(),
                conversation: conversation.clone(),
                conversation_item: conversation_item.clone(),
                request_context: ctx.storage_request_context.clone(),
            }
        }
        _ => {
            return error::internal_error(
                "internal_error",
                "Streaming storage handles are not configured",
            );
        }
    };
    let upstream = OpenAiUpstreamHandle {
        client,
        url: payload_state.url,
        headers: request_headers,
        api_key: worker.api_key().cloned(),
        base_payload: payload_state.json,
    };

    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

    #[expect(
        clippy::disallowed_methods,
        reason = "fire-and-forget streaming loop; gateway shutdown need not wait for individual response streams"
    )]
    tokio::spawn(async move {
        let mut session = McpToolSession::new_with_headers(
            &mcp_orchestrator,
            mcp_servers,
            &session_request_id,
            forwarded_headers,
        );
        if let Some(tools) = loop_ctx_request.tools.as_deref() {
            session.configure_approval_policy(tools);
        }
        let loop_ctx = AgentLoopContext {
            prepared: &prepared,
            session: Some(&session),
            original_request: &loop_ctx_request,
            max_tool_calls: loop_ctx_max_tool_calls,
        };
        let adapter = OpenAiStreamingAdapter::new(&loop_ctx_request, upstream);
        let sink = OpenAiResponseStreamSink::new(tx.clone());
        match run_agent_loop(adapter, loop_ctx, state, sink).await {
            Ok(response) => match to_value(&response) {
                Ok(response_json) => {
                    if let Err(err) = persist_conversation_items(
                        storage.conversation.clone(),
                        storage.conversation_item.clone(),
                        storage.response.clone(),
                        &response_json,
                        &loop_ctx_request,
                        storage.request_context.clone(),
                    )
                    .await
                    {
                        warn!("Failed to persist streaming response items: {}", err);
                    }
                }
                Err(err) => {
                    // Don't silently drop persistence on a serialize
                    // failure; otherwise a follow-up turn that sets
                    // `previous_response_id` to this id would observe
                    // a missing chain link with no diagnostic trail.
                    warn!(
                        "Failed to serialize streaming response for persistence: {}",
                        err
                    );
                }
            },
            Err(err) => {
                // `into_response().status().to_string()` would discard
                // the actual error message and only forward the HTTP
                // status string. Pull the human-readable message off
                // the `AgentLoopError` first so SSE clients see the
                // diagnostic detail that drove the failure.
                let message = err.message();
                let _ = send_sse_event(&tx, "error", &json!({"error": {"message": message}}));
            }
        }

        let _ = tx.send(Ok(Bytes::from_static(SSE_DONE.as_bytes())));
    });

    let body_stream = UnboundedReceiverStream::new(rx);
    let mut response = Response::new(Body::from_stream(body_stream));
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response
}
