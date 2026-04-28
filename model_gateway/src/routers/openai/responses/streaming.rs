//! Streaming response handling for OpenAI-compatible responses
//!
//! This module handles all streaming-related functionality including:
//! - SSE (Server-Sent Events) parsing and forwarding
//! - Streaming response accumulation for persistence
//! - Tool call detection and interception during streaming
//! - MCP tool execution loops within streaming responses
//! - Event transformation and output index remapping

use std::{borrow::Cow, io, sync::Arc};

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::StreamExt;
use openai_protocol::{
    event_types::{
        is_function_call_type, is_response_event, CodeInterpreterCallEvent, FileSearchCallEvent,
        FunctionCallEvent, ImageGenerationCallEvent, ItemType, McpEvent, OutputItemEvent,
        ResponseEvent, WebSearchCallEvent,
    },
    responses::{ResponseTool, ResponsesRequest},
};
use serde_json::{json, Value};
use smg_mcp::{
    mcp_response_item_id, McpOrchestrator, McpServerBinding, McpToolSession, ResponseFormat,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::warn;

use super::{
    accumulator::StreamingResponseAccumulator,
    common::{extract_output_index, parse_sse_block, ChunkProcessor},
    utils::{
        patch_response_with_request_metadata, response_tool_to_value, restore_original_tools,
        rewrite_streaming_block,
    },
};
const SSE_DONE: &str = "data: [DONE]\n\n";

use super::super::provider::Provider;
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::{
        common::{
            header_utils::{
                extract_forwardable_request_headers, preserve_response_headers, ApiProvider,
            },
            mcp_utils::DEFAULT_MAX_ITERATIONS,
            persistence_utils::persist_conversation_items,
        },
        error,
        openai::{
            context::{RequestContext, StreamingEventContext, StreamingRequest},
            mcp::{
                build_resume_payload, execute_streaming_tool_calls, inject_mcp_metadata_streaming,
                mcp_list_tools_bindings_to_emit, prepare_mcp_tools_as_functions,
                send_mcp_list_tools_events, StreamAction, StreamingToolHandler, ToolLoopState,
            },
        },
    },
    worker::Endpoint,
};

fn maybe_transform_final_response_for_persistence(
    response: &mut Value,
    provider: Option<&dyn Provider>,
    model_id: &str,
) {
    if let Some(provider) =
        provider.filter(|p| !p.is_openai_response_shape(response, Endpoint::Responses))
    {
        if let Err(err) = provider.transform_response(response, Endpoint::Responses) {
            warn!(
                model = %model_id,
                provider = ?provider.provider_type(),
                error = %err,
                "Failed to transform final response for persistence"
            );
            Metrics::record_router_error(
                metrics_labels::ROUTER_OPENAI,
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::CONNECTION_HTTP,
                model_id,
                metrics_labels::ENDPOINT_RESPONSES,
                metrics_labels::ERROR_INTERNAL,
            );
        }
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

                            // Determine item type and ID prefix based on response_format
                            let (new_type, id_prefix) = match response_format {
                                ResponseFormat::WebSearchCall => (ItemType::WEB_SEARCH_CALL, "ws_"),
                                ResponseFormat::ImageGenerationCall => {
                                    (ItemType::IMAGE_GENERATION_CALL, "ig_")
                                }
                                _ => (ItemType::MCP_CALL, "mcp_"),
                            };

                            item["type"] = json!(new_type);
                            if new_type == ItemType::MCP_CALL {
                                let label = session.resolve_tool_server_label(&tool_name);
                                item["server_label"] = json!(label);
                            }

                            // Transform ID from fc_* to appropriate prefix
                            if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                                if new_type == ItemType::MCP_CALL {
                                    item["id"] = json!(mcp_response_item_id(id));
                                } else if let Some(stripped) = id.strip_prefix("fc_") {
                                    let new_id = format!("{id_prefix}{stripped}");
                                    item["id"] = json!(new_id);
                                }
                            }

                            changed = true;
                        }
                    }
                }
            }
        }
        FunctionCallEvent::ARGUMENTS_DONE => {
            parsed_data["type"] = json!(McpEvent::CALL_ARGUMENTS_DONE);

            // Transform item_id from fc_* to mcp_*
            if let Some(item_id) = parsed_data.get("item_id").and_then(|v| v.as_str()) {
                parsed_data["item_id"] = json!(mcp_response_item_id(item_id));
            }

            changed = true;
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

    let arguments_value = if call.arguments_buffer.is_empty() {
        "{}".to_string()
    } else {
        call.arguments_buffer.clone()
    };

    // Update the done event with full arguments
    parsed_data["arguments"] = Value::String(arguments_value.clone());

    // Transform item_id
    let item_id = parsed_data
        .get("item_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let mcp_item_id = mcp_response_item_id(item_id);

    // Build synthetic delta event
    let mut delta_event = json!({
        "type": McpEvent::CALL_ARGUMENTS_DELTA,
        "sequence_number": *sequence_number,
        "output_index": assigned_index,
        "item_id": mcp_item_id,
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

    if !send_sse_event(tx, McpEvent::CALL_ARGUMENTS_DELTA, &delta_event) {
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
    let parsed_data: Value = match pre_parsed {
        Some(v) => v,
        None => match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => {
                let chunk = format!("{raw_block}\n\n");
                return tx.send(Ok(Bytes::from(chunk))).is_ok();
            }
        },
    };

    // Provider normalization comes first so downstream gateway rewrites
    // operate on OpenAI-shaped events regardless of upstream provider format.
    let mut transformed_events = vec![parsed_data];
    if let Some(provider) = &ctx.provider {
        handler.ensure_provider_stream_state(provider.as_ref());
        let state = handler.provider_stream_state_mut();
        match provider.transform_stream_event(&transformed_events[0], state, Endpoint::Responses) {
            Ok(events) => {
                if events.is_empty() {
                    warn!(
                        provider = ?provider.provider_type(),
                        "Provider returned empty transformed stream event list; forwarding original event"
                    );
                } else {
                    transformed_events = events;
                }
            }
            Err(err) => {
                warn!(
                    provider = ?provider.provider_type(),
                    error = %err,
                    "Provider stream event transformation failed; forwarding original event"
                );
            }
        }
    }

    for mut parsed_data in transformed_events {
        let transformed_event_type = parsed_data
            .get("type")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        if transformed_event_type.as_deref() == Some(ResponseEvent::COMPLETED)
            || event_name == Some(ResponseEvent::COMPLETED)
        {
            // Preserve transformed completion payload for synthetic-final emission
            // and persistence paths, then suppress inline forwarding to keep one
            // canonical completion event at stream end.
            if let Ok(serialized) = serde_json::to_string(&parsed_data) {
                let completed_block =
                    format!("event: {}\ndata: {serialized}", ResponseEvent::COMPLETED);
                handler.accumulator.ingest_block(&completed_block);
            }
            continue;
        }

        // Handle function_call_arguments.done - send buffered args first
        let mut mapped_output_index: Option<usize> = None;
        if (event_name == Some(FunctionCallEvent::ARGUMENTS_DONE)
            || parsed_data
                .get("type")
                .and_then(|v| v.as_str())
                .is_some_and(|t| t == FunctionCallEvent::ARGUMENTS_DONE))
            && !send_buffered_arguments(
                &mut parsed_data,
                handler,
                tx,
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

        let outgoing_event_name = parsed_data
            .get("type")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .or_else(|| event_name.map(str::to_string));
        let final_block = match outgoing_event_name.as_deref() {
            Some(evt) => format!("event: {}\ndata: {}\n\n", map_event_name(evt), final_data),
            None => format!("data: {final_data}\n\n"),
        };

        if tx.send(Ok(Bytes::from(final_block))).is_err() {
            return false;
        }

        let is_output_item_added = event_name == Some(OutputItemEvent::ADDED)
            || parsed_data
                .get("type")
                .and_then(|v| v.as_str())
                .is_some_and(|t| t == OutputItemEvent::ADDED);
        if is_output_item_added && !maybe_inject_tool_in_progress(&parsed_data, tx, sequence_number)
        {
            return false;
        }
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

/// Send final response.completed event to client
/// Returns false if client disconnected
pub(super) fn send_final_response_event(
    handler: &StreamingToolHandler,
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    sequence_number: &mut u64,
    state: &ToolLoopState,
    ctx: &StreamingEventContext<'_>,
) -> bool {
    let mut final_response = handler.snapshot_final_response().unwrap_or_else(|| {
        warn!("Final response snapshot unavailable; synthesizing minimal completion payload");
        json!({
            "id": handler
                .original_response_id()
                .map(ToString::to_string)
                .unwrap_or_else(|| format!("resp_{}", uuid::Uuid::now_v7())),
            "object": "response",
            "status": "completed",
            "output": []
        })
    });

    if let Some(original_id) = handler.original_response_id() {
        if let Some(obj) = final_response.as_object_mut() {
            obj.insert("id".to_string(), Value::String(original_id.to_string()));
        }
    }

    if let Some(session) = ctx.session {
        inject_mcp_metadata_streaming(&mut final_response, state, session);
    }

    restore_original_tools(&mut final_response, ctx.original_request, ctx.session);
    patch_response_with_request_metadata(
        &mut final_response,
        ctx.original_request,
        ctx.previous_response_id,
    );

    if let Some(obj) = final_response.as_object_mut() {
        obj.insert("status".to_string(), Value::String("completed".to_string()));
    }

    let completed_payload = json!({
        "type": ResponseEvent::COMPLETED,
        "sequence_number": *sequence_number,
        "response": final_response
    });
    *sequence_number += 1;

    let completed_event = format!(
        "event: {}\ndata: {}\n\n",
        ResponseEvent::COMPLETED,
        completed_payload
    );
    tx.send(Ok(Bytes::from(completed_event))).is_ok()
}

/// Simple pass-through streaming without MCP interception
pub(super) async fn handle_simple_streaming_passthrough(
    client: &reqwest::Client,
    worker: &Arc<dyn crate::worker::Worker>,
    headers: Option<&HeaderMap>,
    req: StreamingRequest,
) -> Response {
    let provider_hook = req.provider.clone();
    let mut request_builder = client.post(&req.url).json(&req.payload);
    let api_provider = ApiProvider::from_url(&req.url);
    let auth_header = api_provider.extract_auth_header(headers, worker.api_key());
    request_builder = api_provider.apply_headers(request_builder, auth_header.as_ref());
    if let Some(provider) = &provider_hook {
        request_builder = provider.apply_headers(request_builder, auth_header.as_ref());
    }

    request_builder = request_builder.header("Accept", "text/event-stream");

    let response = match request_builder.send().await {
        Ok(resp) => resp,
        Err(err) => {
            worker.record_outcome(502);
            return (
                StatusCode::BAD_GATEWAY,
                format!("Failed to forward request to OpenAI: {err}"),
            )
                .into_response();
        }
    };

    let status = response.status();
    let status_code =
        StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    worker.record_outcome(status.as_u16());

    if !status.is_success() {
        let error_body = response
            .text()
            .await
            .unwrap_or_else(|err| format!("Failed to read upstream error body: {err}"));
        let error_body = error::sanitize_error_body(&error_body);
        return (status_code, error_body).into_response();
    }

    let preserved_headers = preserve_response_headers(response.headers());
    let mut upstream_stream = response.bytes_stream();

    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

    let should_store = req.original_body.store.unwrap_or(true);
    let original_request = req.original_body;
    let previous_response_id = req.previous_response_id;
    let provider = req.provider;
    let storage = req.storage;

    #[expect(
        clippy::disallowed_methods,
        reason = "fire-and-forget stream processing; gateway shutdown need not wait for individual response streams"
    )]
    tokio::spawn(async move {
        let mut accumulator = StreamingResponseAccumulator::new();
        let mut upstream_failed = false;
        let mut receiver_connected = true;
        let mut chunk_processor = ChunkProcessor::new();
        let mut provider_stream_state = provider.as_ref().and_then(|p| p.new_stream_state());

        while let Some(chunk_result) = upstream_stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    chunk_processor.push_chunk(&chunk);

                    while let Some(raw_block) = chunk_processor.next_block() {
                        if let Some(provider_ref) = provider.as_deref() {
                            let (event_name, data) = parse_sse_block(&raw_block);
                            if !data.is_empty() {
                                if let Ok(parsed_data) =
                                    serde_json::from_str::<Value>(data.as_ref())
                                {
                                    let state = provider_stream_state
                                        .as_deref_mut()
                                        .map(|s| s as &mut (dyn std::any::Any + Send));
                                    match provider_ref.transform_stream_event(
                                        &parsed_data,
                                        state,
                                        Endpoint::Responses,
                                    ) {
                                        Ok(transformed_events) => {
                                            if transformed_events.is_empty() {
                                                warn!(
                                                    provider = ?provider_ref.provider_type(),
                                                    "Provider returned empty transformed stream event list in passthrough mode; forwarding original event"
                                                );
                                                // Fallback to original raw block flow below.
                                            } else {
                                                for transformed in transformed_events {
                                                    match serde_json::to_string(&transformed) {
                                                        Ok(serialized) => {
                                                            let raw_transformed_block =
                                                                match event_name {
                                                                    Some(evt) => {
                                                                        format!("event: {evt}\ndata: {serialized}")
                                                                    }
                                                                    None => format!("data: {serialized}"),
                                                                };

                                                            let chunk_to_send = match rewrite_streaming_block(
                                                                &raw_transformed_block,
                                                                &original_request,
                                                                previous_response_id.as_deref(),
                                                            ) {
                                                                Some(modified) => format!("{modified}\n\n"),
                                                                None => format!("{raw_transformed_block}\n\n"),
                                                            };

                                                            if should_store {
                                                                accumulator.ingest_block(&chunk_to_send);
                                                            }

                                                            if receiver_connected
                                                                && tx
                                                                    .send(Ok(Bytes::from(chunk_to_send)))
                                                                    .is_err()
                                                            {
                                                                receiver_connected = false;
                                                            }
                                                        }
                                                        Err(err) => {
                                                            warn!(
                                                                "Failed to serialize transformed streaming event in passthrough mode: {}",
                                                                err
                                                            );
                                                        }
                                                    }

                                                    if !receiver_connected && !should_store {
                                                        break;
                                                    }
                                                }

                                                if !receiver_connected && !should_store {
                                                    break;
                                                }
                                                continue;
                                            }
                                        }
                                        Err(err) => warn!(
                                            "Failed to transform provider streaming event in passthrough mode: {}",
                                            err
                                        ),
                                    }
                                }
                            }
                        }

                        let block_cow = match rewrite_streaming_block(
                            &raw_block,
                            &original_request,
                            previous_response_id.as_deref(),
                        ) {
                            Some(modified) => Cow::Owned(modified),
                            None => Cow::Borrowed(raw_block.as_str()),
                        };

                        if should_store {
                            accumulator.ingest_block(&block_cow);
                        }

                        if receiver_connected {
                            let chunk_to_send = format!("{block_cow}\n\n");
                            if tx.send(Ok(Bytes::from(chunk_to_send))).is_err() {
                                receiver_connected = false;
                            }
                        }

                        if !receiver_connected && !should_store {
                            break;
                        }
                    }

                    if !receiver_connected && !should_store {
                        break;
                    }
                }
                Err(err) => {
                    upstream_failed = true;
                    let io_err = io::Error::other(err);
                    let _ = tx.send(Err(io_err));
                    break;
                }
            }
        }

        if should_store && !upstream_failed {
            if chunk_processor.has_remaining() {
                accumulator.ingest_block(&chunk_processor.take_remaining());
            }
            let encountered_error = accumulator.encountered_error().cloned();
            if let Some(mut response_json) = accumulator.into_final_response() {
                maybe_transform_final_response_for_persistence(
                    &mut response_json,
                    provider.as_deref(),
                    &original_request.model,
                );
                patch_response_with_request_metadata(
                    &mut response_json,
                    &original_request,
                    previous_response_id.as_deref(),
                );

                // Always persist conversation items and response (even without conversation)
                if let Err(err) = persist_conversation_items(
                    storage.conversation.clone(),
                    storage.conversation_item.clone(),
                    storage.response.clone(),
                    &response_json,
                    &original_request,
                    storage.request_context.clone(),
                )
                .await
                {
                    warn!("Failed to persist conversation items (stream): {}", err);
                }
            } else if let Some(error_payload) = encountered_error {
                warn!("Upstream streaming error payload: {}", error_payload);
            } else {
                warn!("Streaming completed without a final response payload");
            }
        }
    });

    let body_stream = UnboundedReceiverStream::new(rx);
    let mut response = Response::new(Body::from_stream(body_stream));
    *response.status_mut() = status_code;

    let headers_mut = response.headers_mut();
    for (name, value) in &preserved_headers {
        headers_mut.insert(name, value.clone());
    }

    if !headers_mut.contains_key(CONTENT_TYPE) {
        headers_mut.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    }

    response
}

/// Handle streaming WITH MCP tool call interception and execution
pub(super) fn handle_streaming_with_tool_interception(
    client: &reqwest::Client,
    worker_api_key: Option<String>,
    headers: Option<&HeaderMap>,
    req: StreamingRequest,
    orchestrator: &Arc<McpOrchestrator>,
    mcp_servers: Vec<McpServerBinding>,
) -> Response {
    let payload = req.payload;

    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();
    let should_store = req.original_body.store.unwrap_or(true);
    let original_request = req.original_body;
    let previous_response_id = req.previous_response_id;
    let existing_mcp_list_tools_labels = req.existing_mcp_list_tools_labels;
    let provider = req.provider.clone();
    let url = req.url;
    let storage = req.storage;

    let client_clone = client.clone();
    let url_clone = url.clone();
    let headers_opt = headers.cloned();
    let payload_clone = payload.clone();
    let orchestrator_clone = Arc::clone(orchestrator);
    let forwarded_headers = extract_forwardable_request_headers(headers);

    #[expect(
        clippy::disallowed_methods,
        reason = "fire-and-forget MCP tool loop; gateway shutdown need not wait for individual tool loops"
    )]
    tokio::spawn(async move {
        macro_rules! send_done_once {
            () => {{
                let _ = tx.send(Ok(Bytes::from_static(SSE_DONE.as_bytes())));
            }};
        }
        macro_rules! send_error_and_done {
            ($message:expr) => {{
                let _ = send_sse_event(&tx, "error", &json!({"error": {"message": $message}}));
                send_done_once!();
            }};
        }

        let mut state = ToolLoopState::new(
            original_request.input.clone(),
            existing_mcp_list_tools_labels,
        );
        let max_tool_calls = original_request.max_tool_calls.map(|n| n as usize);

        // Create session inside spawned task (borrows from orchestrator_clone which lives in closure)
        let session_request_id = format!("resp_{}", uuid::Uuid::now_v7());
        let session = McpToolSession::new_with_headers(
            &orchestrator_clone,
            mcp_servers.clone(),
            &session_request_id,
            forwarded_headers.clone(),
        );
        let mut current_payload = payload_clone;
        prepare_mcp_tools_as_functions(&mut current_payload, &session);
        let tools_json = current_payload.get("tools").cloned().unwrap_or(json!([]));
        let base_payload = current_payload.clone();
        let mut mcp_list_tools_sent = false;
        let mut is_first_iteration = true;
        let mut sequence_number: u64 = 0;
        let mut next_output_index: usize = 0;
        let mut preserved_response_id: Option<String> = None;
        let list_tools_bindings = mcp_list_tools_bindings_to_emit(
            &state.existing_mcp_list_tools_labels,
            session.mcp_servers(),
        );

        let streaming_ctx = StreamingEventContext {
            original_request: &original_request,
            previous_response_id: previous_response_id.as_deref(),
            session: Some(&session),
            provider: provider.clone(),
        };
        let api_provider = ApiProvider::from_url(&url_clone);
        let auth_header =
            api_provider.extract_auth_header(headers_opt.as_ref(), worker_api_key.as_ref());

        loop {
            // Make streaming request
            let mut request_builder = client_clone.post(&url_clone).json(&current_payload);
            request_builder = api_provider.apply_headers(request_builder, auth_header.as_ref());
            if let Some(provider) = &streaming_ctx.provider {
                request_builder = provider.apply_headers(request_builder, auth_header.as_ref());
            }
            request_builder = request_builder.header("Accept", "text/event-stream");

            let response = match request_builder.send().await {
                Ok(r) => r,
                Err(e) => {
                    send_error_and_done!(e.to_string());
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                let body = error::sanitize_error_body(&body);
                send_error_and_done!(format!("Upstream error {}: {}", status, body));
                return;
            }

            let mut upstream_stream = response.bytes_stream();
            let mut handler = StreamingToolHandler::with_starting_index(next_output_index);
            if let Some(ref id) = preserved_response_id {
                handler.original_response_id = Some(id.clone());
            }
            let mut chunk_processor = ChunkProcessor::new();
            let mut tool_calls_detected = false;
            let mut seen_in_progress = false;

            while let Some(chunk_result) = upstream_stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        chunk_processor.push_chunk(&chunk);

                        while let Some(raw_block) = chunk_processor.next_block() {
                            // Parse event
                            let (event_name, data) = parse_sse_block(&raw_block);

                            if data.is_empty() {
                                continue;
                            }
                            // In interception mode we synthesize the terminal sentinel after
                            // the final response payload; ignore upstream [DONE] to avoid
                            // duplicate/misordered completion framing.
                            if data.trim() == "[DONE]" {
                                continue;
                            }

                            // Process through handler
                            let action = handler.process_event(event_name, data.as_ref());

                            match action {
                                StreamAction::Forward => {
                                    // Parse data once and reuse for skip check, forwarding, and in_progress check
                                    let parsed = serde_json::from_str::<Value>(data.as_ref()).ok();

                                    // Skip response.created and response.in_progress on subsequent iterations
                                    let should_skip = if is_first_iteration {
                                        false
                                    } else {
                                        parsed.as_ref().is_some_and(|v| {
                                            matches!(
                                                v.get("type").and_then(|t| t.as_str()),
                                                Some(ResponseEvent::CREATED)
                                                    | Some(ResponseEvent::IN_PROGRESS)
                                            )
                                        })
                                    };

                                    // Check in_progress before moving parsed into SseEventData
                                    let is_in_progress = !seen_in_progress
                                        && parsed.as_ref().is_some_and(|v| {
                                            v.get("type").and_then(|t| t.as_str())
                                                == Some(ResponseEvent::IN_PROGRESS)
                                        });

                                    if !should_skip {
                                        // Forward the event with pre-parsed value (moved, not cloned)
                                        if !forward_streaming_event(
                                            SseEventData {
                                                raw_block: &raw_block,
                                                event_name,
                                                data: data.as_ref(),
                                                pre_parsed: parsed,
                                            },
                                            &mut handler,
                                            &tx,
                                            &streaming_ctx,
                                            &mut sequence_number,
                                        ) {
                                            return;
                                        }
                                    }

                                    if is_in_progress {
                                        seen_in_progress = true;
                                        if !mcp_list_tools_sent {
                                            for (server_label, server_key) in &list_tools_bindings {
                                                let list_tools_index =
                                                    handler.allocate_synthetic_output_index();
                                                if !send_mcp_list_tools_events(
                                                    &tx,
                                                    &session,
                                                    server_label,
                                                    list_tools_index,
                                                    &mut sequence_number,
                                                    server_key,
                                                ) {
                                                    // Client disconnected
                                                    return;
                                                }
                                            }
                                            mcp_list_tools_sent = true;
                                        }
                                    }
                                }
                                StreamAction::Buffer => {
                                    // Don't forward, just buffer
                                }
                                StreamAction::Drop => {
                                    // R6.7c native passthrough: upstream
                                    // emitted an `output_item.done` for a
                                    // hosted tool-call item that we did
                                    // NOT wrap as a function_call (and so
                                    // did not track in `pending_calls`).
                                    // The upstream envelope arrives
                                    // mis-ordered BEFORE
                                    // `response.<type>.completed`, so we
                                    // drop it here to preserve the wire
                                    // invariant that `output_item.done`
                                    // is the LAST event for a given item.
                                    // Unlike `ExecuteTools`, this does NOT
                                    // kick the tool loop — there is no
                                    // dispatched call to finish.
                                }
                                StreamAction::ExecuteTools {
                                    forward_triggering_event,
                                } => {
                                    // When the upstream signals tool completion via
                                    // `output_item.done` (instead of a preceding
                                    // `function_call_arguments.done`), forwarding the
                                    // event here would emit an umbrella
                                    // `response.output_item.done` BEFORE
                                    // `response.<tool>.completed`, violating spec
                                    // sub-event ordering. The tool loop emits its own
                                    // `output_item.done` at the correct position after
                                    // the `.completed` sub-event; suppress here.
                                    if forward_triggering_event
                                        && !forward_streaming_event(
                                            SseEventData {
                                                raw_block: &raw_block,
                                                event_name,
                                                data: data.as_ref(),
                                                pre_parsed: None,
                                            },
                                            &mut handler,
                                            &tx,
                                            &streaming_ctx,
                                            &mut sequence_number,
                                        )
                                    {
                                        return;
                                    }
                                    tool_calls_detected = true;
                                    break; // Exit stream processing to execute tools
                                }
                            }
                        }

                        if tool_calls_detected {
                            break;
                        }
                    }
                    Err(e) => {
                        send_error_and_done!(format!("Stream error: {}", e));
                        return;
                    }
                }
            }

            next_output_index = handler.next_output_index();
            if let Some(id) = handler.original_response_id().map(|s| s.to_string()) {
                preserved_response_id = Some(id);
            }

            // If no tool calls, we're done - stream is complete
            if !tool_calls_detected {
                if !send_final_response_event(
                    &handler,
                    &tx,
                    &mut sequence_number,
                    &state,
                    &streaming_ctx,
                ) {
                    return;
                }

                let final_response_json = if should_store {
                    handler.accumulator.into_final_response()
                } else {
                    None
                };

                if let Some(mut response_json) = final_response_json {
                    maybe_transform_final_response_for_persistence(
                        &mut response_json,
                        streaming_ctx.provider.as_deref(),
                        &original_request.model,
                    );
                    if let Some(ref id) = preserved_response_id {
                        if let Some(obj) = response_json.as_object_mut() {
                            obj.insert("id".to_string(), Value::String(id.clone()));
                        }
                    }
                    inject_mcp_metadata_streaming(&mut response_json, &state, &session);

                    restore_original_tools(&mut response_json, &original_request, Some(&session));
                    patch_response_with_request_metadata(
                        &mut response_json,
                        &original_request,
                        previous_response_id.as_deref(),
                    );

                    // Always persist conversation items and response (even without conversation)
                    if let Err(err) = persist_conversation_items(
                        storage.conversation.clone(),
                        storage.conversation_item.clone(),
                        storage.response.clone(),
                        &response_json,
                        &original_request,
                        storage.request_context.clone(),
                    )
                    .await
                    {
                        warn!(
                            "Failed to persist conversation items (stream + MCP): {}",
                            err
                        );
                    }
                }

                let _ = tx.send(Ok(Bytes::from_static(SSE_DONE.as_bytes())));
                return;
            }

            // Execute tools
            let pending_calls = handler.take_pending_calls();

            // Check iteration limit
            state.iteration += 1;
            state.total_calls += pending_calls.len();

            // Record tool loop iteration metric
            Metrics::record_mcp_tool_iteration(&original_request.model);

            let effective_limit = match max_tool_calls {
                Some(user_max) => user_max.min(DEFAULT_MAX_ITERATIONS),
                None => DEFAULT_MAX_ITERATIONS,
            };

            if state.total_calls > effective_limit {
                warn!(
                    "Reached tool call limit during streaming: {}",
                    effective_limit
                );
                send_sse_event(
                    &tx,
                    "error",
                    &json!({"error": {"message": "Exceeded max_tool_calls limit"}}),
                );
                send_done_once!();
                return;
            }

            // Execute all pending tool calls. Pass the caller-declared tools so
            // hosted-tool overrides (e.g. image_generation size/quality) are
            // merged into dispatch args before MCP execution. The request-level
            // `user` is also forwarded into hosted-tool dispatch args.
            if !execute_streaming_tool_calls(
                pending_calls,
                &session,
                &tx,
                &mut state,
                &mut sequence_number,
                &original_request.model,
                original_request.tools.as_deref().unwrap_or(&[]),
                original_request.user.as_deref(),
            )
            .await
            {
                return;
            }

            // Build resume payload
            match build_resume_payload(
                &base_payload,
                &state.conversation_history,
                &state.original_input,
                &tools_json,
                true, // is_streaming = true
            ) {
                Ok(resume_payload) => {
                    current_payload = resume_payload;
                    is_first_iteration = false;
                }
                Err(e) => {
                    send_error_and_done!(format!("Failed to build resume payload: {}", e));
                    return;
                }
            }
        }
    });

    let body_stream = UnboundedReceiverStream::new(rx);
    let mut response = Response::new(Body::from_stream(body_stream));
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response
}

/// Main entry point for streaming responses
pub async fn handle_streaming_response(ctx: RequestContext) -> Response {
    use crate::routers::common::mcp_utils::ensure_request_mcp_client;

    let worker = match ctx.worker() {
        Some(w) => w.clone(),
        None => {
            return error::internal_error("internal_error", "Worker not selected");
        }
    };
    let headers = ctx.headers().cloned();
    let original_body = match ctx.responses_request() {
        Some(r) => r,
        None => {
            return error::internal_error("internal_error", "Expected responses request");
        }
    };
    let mcp_orchestrator = match ctx.components.mcp_orchestrator() {
        Some(m) => m.clone(),
        None => {
            return error::internal_error("internal_error", "MCP orchestrator required");
        }
    };

    // Check for MCP tools and create request context if needed
    let mcp_servers = if let Some(tools) = original_body.tools.as_deref() {
        ensure_request_mcp_client(&mcp_orchestrator, tools).await
    } else {
        None
    };

    let client = ctx.components.client().clone();
    let req = match ctx.into_streaming_context() {
        Ok(r) => r,
        Err(msg) => {
            return error::internal_error("internal_error", msg);
        }
    };

    let Some(mcp_servers) = mcp_servers else {
        return handle_simple_streaming_passthrough(&client, &worker, headers.as_ref(), req).await;
    };

    handle_streaming_with_tool_interception(
        &client,
        worker.api_key().cloned(),
        headers.as_ref(),
        req,
        &mcp_orchestrator,
        mcp_servers,
    )
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use openai_protocol::responses::{ResponseInput, ResponsesRequest};
    use serde_json::json;
    use tokio::sync::{mpsc, mpsc::error::TryRecvError};

    use super::*;
    use crate::{routers::openai::provider::ProviderError, worker::ProviderType};

    struct CountingProvider {
        transform_calls: Arc<AtomicUsize>,
    }

    impl Provider for CountingProvider {
        fn provider_type(&self) -> ProviderType {
            ProviderType::Gemini
        }

        fn transform_response(
            &self,
            _response: &mut Value,
            _endpoint: Endpoint,
        ) -> Result<(), ProviderError> {
            self.transform_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn is_openai_response_shape(&self, response: &Value, _endpoint: Endpoint) -> bool {
            !(response.get("candidates").is_some()
                || response.get("modelVersion").is_some()
                || response.get("responseId").is_some()
                || response.get("usageMetadata").is_some())
        }
    }

    #[test]
    fn persistence_transform_runs_for_provider_native_final_response() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = CountingProvider {
            transform_calls: Arc::clone(&calls),
        };

        let mut native = json!({
            "responseId": "resp_upstream",
            "modelVersion": "gemini-2.5-flash",
            "candidates": [{"finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 1}
        });

        maybe_transform_final_response_for_persistence(&mut native, Some(&provider), "test-model");

        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn persistence_transform_skips_openai_shaped_final_response() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = CountingProvider {
            transform_calls: Arc::clone(&calls),
        };

        let mut openai_shaped = json!({
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]
        });

        maybe_transform_final_response_for_persistence(
            &mut openai_shaped,
            Some(&provider),
            "test-model",
        );

        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    struct CreatedEventProvider;

    impl Provider for CreatedEventProvider {
        fn provider_type(&self) -> ProviderType {
            ProviderType::Gemini
        }

        fn transform_stream_event(
            &self,
            _event: &Value,
            _state: Option<&mut (dyn std::any::Any + Send)>,
            _endpoint: Endpoint,
        ) -> Result<Vec<Value>, ProviderError> {
            Ok(vec![json!({
                "type": ResponseEvent::CREATED,
                "response": {
                    "id": "resp_provider"
                }
            })])
        }
    }

    #[test]
    fn forward_streaming_event_applies_rewrites_after_provider_normalization() {
        let provider: Arc<dyn Provider> = Arc::new(CreatedEventProvider);
        let request = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hi".to_string()),
            store: Some(false),
            ..Default::default()
        };
        let ctx = StreamingEventContext {
            original_request: &request,
            previous_response_id: Some("resp_prev"),
            session: None,
            provider: Some(provider),
        };
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut handler = StreamingToolHandler::with_starting_index(0);
        let mut sequence_number = 0;

        let forwarded = forward_streaming_event(
            SseEventData {
                raw_block: "data: {\"native\": true}",
                event_name: None,
                data: "{\"native\": true}",
                pre_parsed: None,
            },
            &mut handler,
            &tx,
            &ctx,
            &mut sequence_number,
        );

        assert!(forwarded);
        let chunk = rx
            .try_recv()
            .expect("expected forwarded event")
            .expect("event should be Ok");
        let payload = String::from_utf8(chunk.to_vec()).expect("valid UTF-8");
        assert!(
            payload.contains("\"store\":false"),
            "provider-created response.created should be patched with request store flag: {payload}"
        );
        assert!(
            payload.contains("\"previous_response_id\":\"resp_prev\""),
            "provider-created response.created should be patched with previous_response_id: {payload}"
        );
    }

    struct CompletedEventProvider;

    impl Provider for CompletedEventProvider {
        fn provider_type(&self) -> ProviderType {
            ProviderType::Gemini
        }

        fn transform_stream_event(
            &self,
            _event: &Value,
            _state: Option<&mut (dyn std::any::Any + Send)>,
            _endpoint: Endpoint,
        ) -> Result<Vec<Value>, ProviderError> {
            Ok(vec![json!({
                "type": ResponseEvent::COMPLETED,
                "response": {"id": "resp_provider_done"}
            })])
        }
    }

    #[test]
    fn forward_streaming_event_skips_provider_generated_response_completed() {
        let provider: Arc<dyn Provider> = Arc::new(CompletedEventProvider);
        let request = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hi".to_string()),
            ..Default::default()
        };
        let ctx = StreamingEventContext {
            original_request: &request,
            previous_response_id: None,
            session: None,
            provider: Some(provider),
        };
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut handler = StreamingToolHandler::with_starting_index(0);
        let mut sequence_number = 0;

        let forwarded = forward_streaming_event(
            SseEventData {
                raw_block: "data: {\"native\": true}",
                event_name: None,
                data: "{\"native\": true}",
                pre_parsed: None,
            },
            &mut handler,
            &tx,
            &ctx,
            &mut sequence_number,
        );

        assert!(forwarded);
        assert!(
            matches!(rx.try_recv(), Err(TryRecvError::Empty)),
            "response.completed events should be suppressed and synthesized once at stream end"
        );
    }

    #[test]
    fn forward_streaming_event_skips_provider_completed_even_with_mismatched_event_name() {
        let provider: Arc<dyn Provider> = Arc::new(CompletedEventProvider);
        let request = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hi".to_string()),
            ..Default::default()
        };
        let ctx = StreamingEventContext {
            original_request: &request,
            previous_response_id: None,
            session: None,
            provider: Some(provider),
        };
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut handler = StreamingToolHandler::with_starting_index(0);
        let mut sequence_number = 0;

        let forwarded = forward_streaming_event(
            SseEventData {
                raw_block: "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\"}",
                event_name: Some("response.output_item.done"),
                data: "{\"type\":\"response.output_item.done\"}",
                pre_parsed: None,
            },
            &mut handler,
            &tx,
            &ctx,
            &mut sequence_number,
        );

        assert!(forwarded);
        assert!(
            matches!(rx.try_recv(), Err(TryRecvError::Empty)),
            "provider-transformed response.completed must be suppressed even if incoming event name differs"
        );
    }

    struct EmptyEventsProvider;

    impl Provider for EmptyEventsProvider {
        fn provider_type(&self) -> ProviderType {
            ProviderType::Gemini
        }

        fn transform_stream_event(
            &self,
            _event: &Value,
            _state: Option<&mut (dyn std::any::Any + Send)>,
            _endpoint: Endpoint,
        ) -> Result<Vec<Value>, ProviderError> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn forward_streaming_event_falls_back_when_provider_returns_empty_events() {
        let provider: Arc<dyn Provider> = Arc::new(EmptyEventsProvider);
        let request = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hi".to_string()),
            store: Some(false),
            ..Default::default()
        };
        let ctx = StreamingEventContext {
            original_request: &request,
            previous_response_id: Some("resp_prev"),
            session: None,
            provider: Some(provider),
        };
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut handler = StreamingToolHandler::with_starting_index(0);
        let mut sequence_number = 0;

        let forwarded = forward_streaming_event(
            SseEventData {
                raw_block: "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_upstream\"}}",
                event_name: Some("response.created"),
                data: "{\"type\":\"response.created\",\"response\":{\"id\":\"resp_upstream\"}}",
                pre_parsed: None,
            },
            &mut handler,
            &tx,
            &ctx,
            &mut sequence_number,
        );

        assert!(forwarded);
        let chunk = rx
            .try_recv()
            .expect("expected forwarded fallback event")
            .expect("event should be Ok");
        let payload = String::from_utf8(chunk.to_vec()).expect("valid UTF-8");
        assert!(
            payload.contains("\"type\":\"response.created\""),
            "fallback should forward original event type: {payload}"
        );
        assert!(
            payload.contains("\"store\":false"),
            "fallback path should still patch response metadata: {payload}"
        );
    }
}
