//! Streaming execution for Regular Responses API
//!
//! This module handles streaming request execution:
//! - `execute_tool_loop_streaming` - MCP tool loop with streaming
//! - `convert_chat_stream_to_responses_stream` - Non-MCP streaming conversion
//! - Streaming accumulators for response building

use std::{
    collections::HashMap,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Body,
    http::{header, StatusCode},
    response::Response,
};
use bytes::Bytes;
use futures_util::StreamExt;
use openai_protocol::{
    chat::{
        ChatChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse,
        ChatCompletionStreamResponse,
    },
    common::{FunctionCallResponse, ToolCall, Usage, UsageInfo},
    responses::{
        ResponseContentPart, ResponseOutputItem, ResponseReasoningContent, ResponseStatus,
        ResponsesRequest, ResponsesResponse, ResponsesUsage,
    },
};
use serde_json::{json, Value};
use smg_data_connector::{
    ConversationItemStorage, ConversationStorage, RequestContext as StorageRequestContext,
    ResponseStorage,
};
use smg_mcp::{McpServerBinding, McpToolSession, ToolExecutionInput};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, trace, warn};
use uuid::Uuid;

use super::{
    common::{
        build_next_request, convert_mcp_tools_to_chat_tools, extract_all_tool_calls_from_chat,
        prepare_chat_tools_and_choice, ExtractedToolCall, ResponsesCallContext, ToolLoopState,
    },
    conversions,
};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::{
        common::{
            mcp_utils::{prepare_hosted_dispatch_args, DEFAULT_MAX_ITERATIONS},
            openai_bridge::{self, ResponseFormat},
        },
        grpc::{
            common::responses::{
                build_sse_response, persist_response_if_needed,
                streaming::{
                    attach_mcp_server_label, OutputItemKind, ResponseEventSink,
                    ResponseStreamEventEmitter, WsResponseEventSink,
                },
                ResponsesContext,
            },
            utils,
        },
    },
};

// ============================================================================
// Non-MCP Streaming Path
// ============================================================================

/// Convert chat streaming response to responses streaming format
///
/// This function:
/// 1. Gets chat SSE stream from pipeline
/// 2. Intercepts and parses each SSE event
/// 3. Converts ChatCompletionStreamResponse → ResponsesResponse delta
/// 4. Accumulates response state for final persistence
/// 5. Emits transformed SSE events in responses format
pub(super) async fn convert_chat_stream_to_responses_stream(
    ctx: &ResponsesContext,
    chat_request: Arc<ChatCompletionRequest>,
    params: ResponsesCallContext,
    original_request: &ResponsesRequest,
) -> Response {
    debug!("Converting chat SSE stream to responses SSE format");

    // Get chat streaming response
    let chat_response = ctx
        .pipeline
        .execute_chat(
            chat_request,
            params.headers,
            params.model_id,
            ctx.components.clone(),
            Some(params.tenant_request_meta),
        )
        .await;

    // Extract body from chat response
    let (_parts, body) = chat_response.into_parts();

    // Create channel for transformed SSE events
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();

    // Spawn background task to transform stream
    let original_request_clone = original_request.clone();
    let response_storage = ctx.response_storage.clone();
    let conversation_storage = ctx.conversation_storage.clone();
    let conversation_item_storage = ctx.conversation_item_storage.clone();
    let request_context = ctx.request_context.clone();

    #[expect(
        clippy::disallowed_methods,
        reason = "streaming task is fire-and-forget; client disconnect terminates it"
    )]
    tokio::spawn(async move {
        if let Err(e) = process_and_transform_sse_stream(
            body,
            original_request_clone,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            request_context,
            tx.clone(),
        )
        .await
        {
            warn!("Error transforming SSE stream: {}", e);
            utils::send_error_sse(&tx, &e, "stream_error");
        }

        // Send final [DONE] event
        let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
    });

    // Build SSE response with transformed stream
    build_sse_response(rx)
}

/// Process chat SSE stream and transform to responses format
async fn process_and_transform_sse_stream(
    body: Body,
    original_request: ResponsesRequest,
    response_storage: Arc<dyn ResponseStorage>,
    conversation_storage: Arc<dyn ConversationStorage>,
    conversation_item_storage: Arc<dyn ConversationItemStorage>,
    request_context: Option<StorageRequestContext>,
    tx: mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) -> Result<(), String> {
    // Drive the transport-neutral core over the SSE sender. `process_chunk`
    // emits `completed` (draining state) on this path for byte-identical SSE
    // output; the finalized response is then persisted.
    let final_response = drive_non_mcp_stream(
        body,
        &original_request,
        &tx,
        /* drain_completed */ true,
    )
    .await?;

    // Finalize and persist accumulated response
    persist_response_if_needed(
        conversation_storage,
        conversation_item_storage,
        response_storage,
        &final_response,
        &original_request,
        request_context,
    )
    .await;

    Ok(())
}

/// Sink-generic core for the non-MCP streaming path.
///
/// Emits `response.created` / `response.in_progress`, then transforms each chat
/// SSE chunk into Responses events on `sink`, accumulates the response, and
/// (when `drain_completed`) emits the terminal `response.completed`. Returns the
/// finalized [`ResponsesResponse`] so the WS path can populate its connection
/// cache (it sets `drain_completed = false` and emits `completed`/`failed`
/// itself with an explicit status).
async fn drive_non_mcp_stream(
    body: Body,
    original_request: &ResponsesRequest,
    sink: &impl ResponseEventSink,
    drain_completed: bool,
) -> Result<ResponsesResponse, String> {
    // Create accumulator for final response
    let mut accumulator = StreamingResponseAccumulator::new(original_request);

    // Create event emitter for OpenAI-compatible streaming
    let response_id = format!("resp_{}", Uuid::now_v7());
    let model = original_request.model.clone();
    let created_at = chrono::Utc::now().timestamp() as u64;
    let mut event_emitter = ResponseStreamEventEmitter::new(response_id, model, created_at);
    event_emitter.set_original_request(original_request.clone());

    // Emit initial response.created and response.in_progress events
    let event = event_emitter.emit_created();
    event_emitter
        .send_event(&event, sink)
        .map_err(|_| "Failed to send response.created event".to_string())?;

    let event = event_emitter.emit_in_progress();
    event_emitter
        .send_event(&event, sink)
        .map_err(|_| "Failed to send response.in_progress event".to_string())?;

    // Convert body to data stream
    let mut stream = body.into_data_stream();

    // Process stream chunks (each chunk is a complete SSE event)
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Stream read error: {e}"))?;

        // Convert chunk to string
        let event_str = String::from_utf8_lossy(&chunk);
        let event = event_str.trim();

        // Check for end of stream
        if event == "data: [DONE]" {
            break;
        }

        // Parse SSE event (format: "data: {...}\n\n" or "data: {...}")
        if let Some(json_str) = event.strip_prefix("data: ") {
            let json_str = json_str.trim();

            // Try to parse as ChatCompletionStreamResponse
            match serde_json::from_str::<ChatCompletionStreamResponse>(json_str) {
                Ok(chat_chunk) => {
                    // Update accumulator
                    accumulator.process_chunk(&chat_chunk);

                    // Process chunk through event emitter (emits proper OpenAI events)
                    event_emitter.process_chunk(&chat_chunk, sink)?;
                }
                Err(_) => {
                    // Not a valid chat chunk - might be error event, pass through.
                    // Forward the data:-stripped payload (`json_str`), NOT `event`:
                    // `send_raw_json` re-frames as `data: {payload}` for SSE and
                    // sends bare JSON text on WS, so passing the prefixed line would
                    // double-frame to `data: data: ...`.
                    debug!("Non-chunk SSE event, passing through: {}", json_str);
                    sink.send_raw_json(json_str)
                        .map_err(|_| "Client disconnected".to_string())?;
                }
            }
        }
    }

    // Emit final response.completed event with accumulated usage
    let usage_json = accumulator.usage.as_ref().map(|u| {
        let mut usage_obj = json!({
            "input_tokens": u.prompt_tokens,
            "output_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens
        });

        // Include reasoning_tokens if present
        if let Some(details) = &u.completion_tokens_details {
            if let Some(reasoning_tokens) = details.reasoning_tokens {
                usage_obj["output_tokens_details"] =
                    json!({ "reasoning_tokens": reasoning_tokens });
            }
        }

        usage_obj
    });

    if drain_completed {
        let completed_event = event_emitter.emit_completed(usage_json.as_ref());
        event_emitter.send_event(&completed_event, sink)?;
    } else {
        // WS path: emit the terminal event with an explicit status (CLONE — does
        // not drain — so the finalized cache response below sees the items).
        let status = accumulator.status();
        let completed_event = event_emitter.emit_completed_with_status(status, usage_json.as_ref());
        event_emitter.send_event(&completed_event, sink)?;
    }

    // Finalize accumulated response (used for persistence and/or WS cache).
    //
    // The accumulator adopts the worker's chat-completion id (`chatcmpl-…`) from
    // the first chunk, but the client received the emitter's `resp_…` id in the
    // streamed events. Stamp the finalized response with the emitter id so the
    // persisted/cached response is keyed by the id the client holds — required
    // for `previous_response_id` continuation (durable + WS connection cache).
    let mut final_response = accumulator.finalize();
    final_response.id.clone_from(&event_emitter.response_id);
    Ok(final_response)
}

/// Execute the non-MCP streaming pipeline over an arbitrary sink and return the
/// finalized response. Used by the WebSocket Responses executor; the chat
/// pipeline is driven inline (no fire-and-forget spawn / SSE body) so the caller
/// can await the materialized [`ResponsesResponse`] for its connection cache.
pub(crate) async fn execute_non_mcp_stream_with_sink(
    ctx: &ResponsesContext,
    chat_request: Arc<ChatCompletionRequest>,
    original_request: ResponsesRequest,
    headers: Option<header::HeaderMap>,
    model_id: Option<String>,
    sink: &WsResponseEventSink,
) -> Result<ResponsesResponse, String> {
    let chat_response = ctx
        .pipeline
        .execute_chat(
            chat_request,
            headers,
            model_id.unwrap_or_else(|| original_request.model.clone()),
            ctx.components.clone(),
            None,
        )
        .await;

    let (_parts, body) = chat_response.into_parts();

    let final_response = drive_non_mcp_stream(
        body,
        &original_request,
        sink,
        /* drain_completed */ false,
    )
    .await?;

    // Persistence is handled once by the caller (`execute_response_create`)
    // after the MCP/non-MCP branch, so both WS paths persist identically.
    Ok(final_response)
}

/// Response accumulator for streaming responses (non-MCP path)
struct StreamingResponseAccumulator {
    // Response metadata
    response_id: String,
    model: String,
    created_at: i64,

    // Accumulated content
    content_buffer: String,
    reasoning_buffer: String,
    tool_calls: Vec<ResponseOutputItem>,

    // Completion state
    finish_reason: Option<String>,
    usage: Option<Usage>,

    // Original request for final response construction
    original_request: ResponsesRequest,
}

impl StreamingResponseAccumulator {
    fn new(original_request: &ResponsesRequest) -> Self {
        Self {
            response_id: String::new(),
            model: String::new(),
            created_at: 0,
            content_buffer: String::new(),
            reasoning_buffer: String::new(),
            tool_calls: Vec::new(),
            finish_reason: None,
            usage: None,
            original_request: original_request.clone(),
        }
    }

    fn process_chunk(&mut self, chunk: &ChatCompletionStreamResponse) {
        // Initialize metadata on first chunk
        if self.response_id.is_empty() {
            self.response_id.clone_from(&chunk.id);
            self.model.clone_from(&chunk.model);
            self.created_at = chunk.created as i64;
        }

        // Process first choice (responses API doesn't support n>1)
        if let Some(choice) = chunk.choices.first() {
            // Accumulate content
            if let Some(content) = &choice.delta.content {
                self.content_buffer.push_str(content);
            }

            // Accumulate reasoning
            if let Some(reasoning) = &choice.delta.reasoning_content {
                self.reasoning_buffer.push_str(reasoning);
            }

            // Process tool call deltas
            if let Some(tool_call_deltas) = &choice.delta.tool_calls {
                for delta in tool_call_deltas {
                    // Use index directly (it's a u32, not Option<u32>)
                    let index = delta.index as usize;

                    // Ensure we have enough tool calls
                    while self.tool_calls.len() <= index {
                        self.tool_calls.push(ResponseOutputItem::FunctionToolCall {
                            id: None,
                            call_id: String::new(),
                            name: String::new(),
                            arguments: String::new(),
                            output: None,
                            status: "in_progress".to_string(),
                        });
                    }

                    // Update the tool call at this index
                    if let ResponseOutputItem::FunctionToolCall {
                        id,
                        call_id,
                        name,
                        arguments,
                        ..
                    } = &mut self.tool_calls[index]
                    {
                        if let Some(delta_id) = &delta.id {
                            id.get_or_insert_with(String::new).push_str(delta_id);
                            call_id.push_str(delta_id);
                        }
                        if let Some(function) = &delta.function {
                            if let Some(delta_name) = &function.name {
                                name.push_str(delta_name);
                            }
                            if let Some(delta_args) = &function.arguments {
                                arguments.push_str(delta_args);
                            }
                        }
                    }
                }
            }

            // Update finish reason
            if let Some(reason) = &choice.finish_reason {
                self.finish_reason = Some(reason.clone());
            }
        }

        // Update usage
        if let Some(usage) = &chunk.usage {
            self.usage = Some(usage.clone());
        }
    }

    /// Terminal status derived from the accumulated `finish_reason`.
    ///
    /// Borrows `&self` so the WS path can read the status before `finalize`
    /// consumes the accumulator. Mirrors the mapping in `finalize`.
    fn status(&self) -> ResponseStatus {
        match self.finish_reason.as_deref() {
            Some("stop") | Some("length") => ResponseStatus::Completed,
            Some("tool_calls") => ResponseStatus::InProgress,
            Some("failed") | Some("error") => ResponseStatus::Failed,
            _ => ResponseStatus::Completed,
        }
    }

    fn finalize(self) -> ResponsesResponse {
        // Determine final status before consuming fields out of `self`.
        let status = self.status();

        let mut output: Vec<ResponseOutputItem> = Vec::new();

        // Add message content if present
        if !self.content_buffer.is_empty() {
            output.push(ResponseOutputItem::Message {
                id: format!("msg_{}", self.response_id),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: self.content_buffer,
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
                phase: None,
            });
        }

        // Add reasoning if present
        if !self.reasoning_buffer.is_empty() {
            output.push(ResponseOutputItem::new_reasoning(
                format!("reasoning_{}", self.response_id),
                vec![],
                vec![ResponseReasoningContent::ReasoningText {
                    text: self.reasoning_buffer,
                }],
                Some("completed".to_string()),
            ));
        }

        // Add tool calls
        output.extend(self.tool_calls);

        // Convert usage
        let usage = self.usage.as_ref().map(|u| {
            let usage_info = UsageInfo {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
                reasoning_tokens: u
                    .completion_tokens_details
                    .as_ref()
                    .and_then(|d| d.reasoning_tokens),
                prompt_tokens_details: None,
            };
            ResponsesUsage::Classic(usage_info)
        });

        ResponsesResponse::builder(&self.response_id, &self.model)
            .copy_from_request(&self.original_request)
            .created_at(self.created_at)
            .status(status)
            .output(output)
            .maybe_usage(usage)
            .build()
    }
}

// ============================================================================
// MCP Streaming Path
// ============================================================================

/// Execute MCP tool loop with streaming support
///
/// This streams each iteration's response to the client while accumulating
/// to check for tool calls. If tool calls are found, executes them and
/// continues with the next streaming iteration.
pub(super) fn execute_tool_loop_streaming(
    ctx: &ResponsesContext,
    current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    params: ResponsesCallContext,
    mcp_servers: Vec<McpServerBinding>,
) -> Response {
    // Create SSE channel for client
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();

    // Clone data for background task
    let ctx_clone = ctx.clone();
    let original_request_clone = original_request.clone();

    // Spawn background task for tool loop
    #[expect(
        clippy::disallowed_methods,
        reason = "streaming task is fire-and-forget; client disconnect terminates it"
    )]
    tokio::spawn(async move {
        let result = execute_tool_loop_streaming_internal(
            &ctx_clone,
            current_request,
            &original_request_clone,
            params,
            mcp_servers,
            &tx,
            /* drain_completed */ true,
        )
        .await;

        match result {
            // Persist (when store=true) the finalized tool-using response so
            // GET /v1/responses/{id} retrieval and durable previous_response_id
            // resolution work on the SSE MCP path too — mirroring the non-MCP
            // SSE path (`process_and_transform_sse_stream`) and the WS executor
            // (`websocket.rs`). The sink-generic refactor previously dropped this,
            // silently breaking store=true durability for streaming tool-call
            // responses (default store=true → GET 404 + broken chaining).
            Ok(final_response) => {
                persist_response_if_needed(
                    ctx_clone.conversation_storage.clone(),
                    ctx_clone.conversation_item_storage.clone(),
                    ctx_clone.response_storage.clone(),
                    &final_response,
                    &original_request_clone,
                    ctx_clone.request_context.clone(),
                )
                .await;
            }
            Err(e) => {
                warn!("Streaming tool loop error: {}", e);
                utils::send_error_sse(&tx, &e, "tool_loop_error");
            }
        }

        // Send [DONE]
        let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
    });

    // Build SSE response
    let stream = UnboundedReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    #[expect(
        clippy::expect_used,
        reason = "Response::builder with valid status and no invalid headers is infallible"
    )]
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .body(body)
        .expect("infallible: valid status code, no invalid headers");

    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("text/event-stream"),
    );
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-cache"),
    );
    response.headers_mut().insert(
        header::CONNECTION,
        header::HeaderValue::from_static("keep-alive"),
    );

    response
}

/// Execute the MCP streaming tool loop over a WebSocket sink and return the
/// finalized response for the connection cache.
///
/// Mirrors [`execute_tool_loop_streaming`] but drives the loop inline (no
/// fire-and-forget spawn / SSE body) so the caller can await the materialized
/// [`ResponsesResponse`]. The WebSocket transport is inherently event-streamed,
/// so the terminal `response.completed`/`failed` is emitted with an explicit
/// status (no draining) and the response is materialized non-destructively.
pub(crate) async fn execute_tool_loop_streaming_with_sink(
    ctx: &ResponsesContext,
    current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    headers: Option<header::HeaderMap>,
    model_id: Option<String>,
    mcp_servers: Vec<McpServerBinding>,
    sink: &WsResponseEventSink,
) -> Result<ResponsesResponse, String> {
    // WS sessions resolve tenant via the route_request_meta middleware on the
    // realtime-style route group, but the meta is not threaded down to the
    // executor; use an anonymous charge identity for the inline pipeline call.
    let tenant_request_meta = crate::middleware::TenantRequestMeta::new(
        crate::tenant::TenantIdentity::Anonymous.into_key(),
    );
    let params = ResponsesCallContext {
        headers,
        model_id: model_id.unwrap_or_else(|| current_request.model.clone()),
        response_id: None,
        tenant_request_meta,
    };

    execute_tool_loop_streaming_internal(
        ctx,
        current_request,
        original_request,
        params,
        mcp_servers,
        sink,
        /* drain_completed */ false,
    )
    .await
}

/// Internal streaming tool loop implementation
///
/// Sink-generic so both the SSE and WebSocket Responses paths reuse the same
/// MCP tool-loop, event-emission and accumulation logic. When `drain_completed`
/// is true (SSE) the terminal `response.completed` event is emitted by draining
/// emitter state; when false (WS) it is emitted with an explicit status via a
/// clone so the returned [`ResponsesResponse`] still carries the full output.
async fn execute_tool_loop_streaming_internal(
    ctx: &ResponsesContext,
    mut current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    params: ResponsesCallContext,
    mcp_servers: Vec<McpServerBinding>,
    sink: &impl ResponseEventSink,
    drain_completed: bool,
) -> Result<ResponsesResponse, String> {
    let mut state = ToolLoopState::new(original_request.input.clone());
    let max_tool_calls = original_request.max_tool_calls.map(|n| n as usize);

    // Generate response ID first so we can use it for both emitter and session
    let response_id = format!("resp_{}", Uuid::now_v7());

    // Create session once — bundles orchestrator, request_ctx, server_keys, mcp_tools
    let session = McpToolSession::new(&ctx.mcp_orchestrator, mcp_servers, &response_id);

    // Create response event emitter
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut emitter =
        ResponseStreamEventEmitter::new(response_id, current_request.model.clone(), created_at);
    emitter.set_original_request(original_request.clone());

    // Emit initial response.created and response.in_progress events
    let event = emitter.emit_created();
    emitter.send_event(&event, sink)?;
    let event = emitter.emit_in_progress();
    emitter.send_event(&event, sink)?;

    // Get MCP tools and convert to chat format (do this once before loop)
    let mcp_chat_tools = convert_mcp_tools_to_chat_tools(&session);
    trace!(
        "Streaming: Converted {} MCP tools to chat format",
        mcp_chat_tools.len()
    );

    // Flag to track if mcp_list_tools has been emitted
    let mut mcp_list_tools_emitted = false;

    // Terminal (usage, status) captured at whichever `break` the loop exits
    // through, used to materialize the finalized response for the WS cache /
    // persistence. The function-tool-call and tool-limit exits leave the turn
    // pending (`InProgress`); only natural completion reports `Completed`.
    let (terminal_usage, terminal_status) = loop {
        state.iteration += 1;

        // Record tool loop iteration metric
        Metrics::record_mcp_tool_iteration(&current_request.model);

        if state.iteration > DEFAULT_MAX_ITERATIONS {
            return Err(format!(
                "Tool loop exceeded maximum iterations ({DEFAULT_MAX_ITERATIONS})"
            ));
        }

        trace!("Streaming MCP tool loop iteration {}", state.iteration);

        // Emit mcp_list_tools as first output item (only once, on first iteration)
        if !mcp_list_tools_emitted {
            for binding in session.mcp_servers() {
                let tools_for_server = session.list_tools_for_server(&binding.server_key);

                emitter.emit_mcp_list_tools_sequence(&binding.label, &tools_for_server, sink)?;
            }
            mcp_list_tools_emitted = true;
        }

        // Convert to chat request
        let mut chat_request = conversions::responses_to_chat(&current_request)
            .map_err(|e| format!("Failed to convert request: {e}"))?;

        // Prepare tools and tool_choice for this iteration (same logic as non-streaming)
        prepare_chat_tools_and_choice(&mut chat_request, &mcp_chat_tools, state.iteration);

        // Execute chat streaming
        let response = ctx
            .pipeline
            .execute_chat(
                Arc::new(chat_request),
                params.headers.clone(),
                params.model_id.clone(),
                ctx.components.clone(),
                Some(params.tenant_request_meta.clone()),
            )
            .await;

        // Convert chat stream to Responses API events while accumulating for tool call detection
        // Stream text naturally - it only appears on final iteration (tool iterations have empty content)
        let accumulated_response =
            convert_and_accumulate_stream(response.into_body(), &mut emitter, sink).await?;

        // Check for tool calls (extract all of them for parallel execution)
        let tool_calls = extract_all_tool_calls_from_chat(&accumulated_response);

        if !tool_calls.is_empty() {
            trace!(
                "Tool loop iteration {}: found {} tool call(s)",
                state.iteration,
                tool_calls.len()
            );

            // Separate MCP and function tool calls using session-exposed names.
            let (mcp_tool_calls, function_tool_calls): (Vec<ExtractedToolCall>, Vec<_>) =
                tool_calls
                    .into_iter()
                    .partition(|tc| session.has_exposed_tool(tc.name.as_str()));

            trace!(
                "Separated tool calls: {} MCP, {} function",
                mcp_tool_calls.len(),
                function_tool_calls.len()
            );

            // Check combined limit (only count MCP tools since function tools will be returned)
            let effective_limit = match max_tool_calls {
                Some(user_max) => user_max.min(DEFAULT_MAX_ITERATIONS),
                None => DEFAULT_MAX_ITERATIONS,
            };

            if state.total_calls + mcp_tool_calls.len() > effective_limit {
                warn!(
                    "Reached tool call limit: {} + {} > {} (max_tool_calls={:?}, safety_limit={})",
                    state.total_calls,
                    mcp_tool_calls.len(),
                    effective_limit,
                    max_tool_calls,
                    DEFAULT_MAX_ITERATIONS
                );
                // TODO(phase2): max_tool_calls should surface status=failed + an
                // error payload per main's contract (NOT incomplete_details).
                // For now keep main's current behavior: stop the loop and return
                // the accumulated turn unchanged (pending → InProgress).
                break (accumulated_response.usage, ResponseStatus::InProgress);
            }

            // Process each MCP tool call
            for tool_call in mcp_tool_calls {
                state.total_calls += 1;

                trace!(
                    "Executing tool call {}/{}: {} (call_id: {})",
                    state.total_calls,
                    state.total_calls,
                    tool_call.name,
                    tool_call.call_id
                );

                let response_format = openai_bridge::lookup_tool_format(
                    &session,
                    &ctx.mcp_format_registry,
                    &tool_call.name,
                );

                // Use emitter helpers to determine correct type and allocate index
                let item_type =
                    ResponseStreamEventEmitter::type_str_for_format(Some(&response_format));
                let resolved_label = session.resolve_tool_server_label(&tool_call.name);

                // Allocate output_index with the format's id-prefix discriminator
                // (e.g. `ws_…` for web_search_call); see FormatDescriptor.
                let (output_index, item_id) =
                    emitter.allocate_output_index_for_format(Some(response_format));

                // Build initial tool call item
                let mut item = json!({
                    "id": item_id,
                    "type": item_type,
                    "name": tool_call.name,
                    "status": "in_progress",
                    "arguments": ""
                });
                attach_mcp_server_label(
                    &mut item,
                    Some(resolved_label.as_str()),
                    Some(&response_format),
                );

                // Emit output_item.added
                let event = emitter.emit_output_item_added(output_index, &item);
                emitter.send_event(&event, sink)?;

                // Emit tool_call.in_progress
                let event =
                    emitter.emit_tool_call_in_progress(output_index, &item_id, response_format);
                emitter.send_event(&event, sink)?;

                // Emit arguments events for mcp_call only (skip for builtin tools)
                if matches!(response_format, ResponseFormat::Passthrough) {
                    // Emit mcp_call_arguments.delta (simulate streaming by sending full arguments)
                    let event = emitter.emit_mcp_call_arguments_delta(
                        output_index,
                        &item_id,
                        &tool_call.arguments,
                    );
                    emitter.send_event(&event, sink)?;

                    // Emit mcp_call_arguments.done
                    let event = emitter.emit_mcp_call_arguments_done(
                        output_index,
                        &item_id,
                        &tool_call.arguments,
                    );
                    emitter.send_event(&event, sink)?;
                }

                // Emit searching/interpreting event for builtin tools
                if let Some(event) =
                    emitter.emit_tool_call_searching(output_index, &item_id, response_format)
                {
                    emitter.send_event(&event, sink)?;
                }

                // Execute the MCP tool
                trace!(
                    "Calling MCP tool '{}' with args: {}",
                    tool_call.name,
                    tool_call.arguments
                );
                // Parse arguments to Value, coercing scalar/array/null payloads
                // to an empty object so hosted-tool override merge can actually
                // apply. `apply_hosted_tool_overrides` is a no-op on non-objects;
                // silently dropping caller-declared config would be surprising.
                let mut arguments = match serde_json::from_str::<Value>(&tool_call.arguments) {
                    Ok(Value::Object(map)) => Value::Object(map),
                    _ => json!({}),
                };
                prepare_hosted_dispatch_args(
                    &mut arguments,
                    response_format,
                    original_request.tools.as_deref().unwrap_or(&[]),
                    original_request.user.as_deref(),
                );

                // Execute the single tool via the normalized MCP execution API.
                // This avoids custom serialization and manual re-transformation in streaming paths.
                let tool_output = session
                    .execute_tool(ToolExecutionInput {
                        call_id: tool_call.call_id.clone(),
                        tool_name: tool_call.name.clone(),
                        arguments,
                    })
                    .await;

                let success = !tool_output.is_error;
                let output_str = tool_output.output.to_string();

                let output_item =
                    openai_bridge::transform_tool_output(&tool_output, response_format);
                let mut item_done = serde_json::to_value(&output_item).unwrap_or_else(|e| {
                    warn!(
                        tool = %tool_output.tool_name,
                        error = %e,
                        "Failed to serialize transformed output item; falling back to a minimal stub",
                    );
                    json!({
                        "id": item_id,
                        "type": item_type,
                        "status": if success { "completed" } else { "failed" },
                    })
                });
                // Override the typed item's id so output_item.done matches the
                // streaming-allocated id used by the earlier output_item.added.
                if let Some(obj) = item_done.as_object_mut() {
                    obj.insert("id".to_string(), json!(&item_id));
                }
                attach_mcp_server_label(
                    &mut item_done,
                    Some(tool_output.server_label.as_str()),
                    Some(&response_format),
                );

                if success {
                    let event =
                        emitter.emit_tool_call_completed(output_index, &item_id, response_format);
                    emitter.send_event(&event, sink)?;
                } else {
                    let err_text = tool_output
                        .error_message
                        .clone()
                        .unwrap_or_else(|| output_str.clone());
                    warn!("Tool execution returned error: {}", err_text);

                    // `response.mcp_call.failed` is the only `*.failed` event
                    // in the Responses API. Hosted-builtin families close via
                    // `*.completed` to mirror OpenAI cloud's wire shape;
                    // the failure context (when present) lives in the item
                    // content.
                    if matches!(response_format, ResponseFormat::Passthrough) {
                        let event = emitter.emit_mcp_call_failed(output_index, &item_id, &err_text);
                        emitter.send_event(&event, sink)?;
                    } else {
                        let event = emitter.emit_tool_call_completed(
                            output_index,
                            &item_id,
                            response_format,
                        );
                        emitter.send_event(&event, sink)?;
                    }
                }

                let event = emitter.emit_output_item_done(output_index, &item_done);
                emitter.send_event(&event, sink)?;
                emitter.complete_output_item(output_index);

                Metrics::record_mcp_tool_duration(
                    &current_request.model,
                    &tool_output.tool_name,
                    tool_output.duration,
                );
                Metrics::record_mcp_tool_call(
                    &current_request.model,
                    &tool_output.tool_name,
                    if success {
                        metrics_labels::RESULT_SUCCESS
                    } else {
                        metrics_labels::RESULT_ERROR
                    },
                );

                state.record_call(
                    tool_output.call_id,
                    tool_output.tool_name,
                    tool_output.arguments_str,
                    output_str,
                    output_item,
                    success,
                );
            }

            // If there are function tool calls, emit events and exit MCP loop
            if !function_tool_calls.is_empty() {
                trace!(
                    "Found {} function tool call(s) - emitting events and exiting MCP loop",
                    function_tool_calls.len()
                );

                // Emit function_tool_call events for each function tool
                for tool_call in function_tool_calls {
                    // Allocate output_index for this function_tool_call item
                    let (output_index, item_id) =
                        emitter.allocate_output_index(OutputItemKind::FunctionCall);

                    // Build initial function_call item
                    let item = json!({
                        "id": item_id,
                        "type": "function_call",
                        "call_id": tool_call.call_id,
                        "name": tool_call.name,
                        "status": "in_progress",
                        "arguments": ""
                    });

                    // Emit output_item.added
                    let event = emitter.emit_output_item_added(output_index, &item);
                    emitter.send_event(&event, sink)?;

                    // Emit function_call_arguments.delta
                    let event = emitter.emit_function_call_arguments_delta(
                        output_index,
                        &item_id,
                        &tool_call.arguments,
                    );
                    emitter.send_event(&event, sink)?;

                    // Emit function_call_arguments.done
                    let event = emitter.emit_function_call_arguments_done(
                        output_index,
                        &item_id,
                        &tool_call.arguments,
                    );
                    emitter.send_event(&event, sink)?;

                    // Build complete item
                    let item_complete = json!({
                        "id": item_id,
                        "type": "function_call",
                        "call_id": tool_call.call_id,
                        "name": tool_call.name,
                        "status": "completed",
                        "arguments": tool_call.arguments
                    });

                    // Emit output_item.done
                    let event = emitter.emit_output_item_done(output_index, &item_complete);
                    emitter.send_event(&event, sink)?;

                    emitter.complete_output_item(output_index);
                }

                // Break loop to return response to caller. Function tool calls
                // leave the turn pending on the client; status stays InProgress.
                break (accumulated_response.usage, ResponseStatus::InProgress);
            }

            // Build next request with conversation history
            current_request = build_next_request(&state, current_request);

            continue;
        }

        // No tool calls, this is the final response
        trace!("No tool calls found, ending streaming MCP loop");

        // Check for reasoning content
        let reasoning_content = accumulated_response
            .choices
            .first()
            .and_then(|c| c.message.reasoning_content.clone());

        // Emit reasoning item if present
        if let Some(reasoning) = reasoning_content {
            if !reasoning.is_empty() {
                emitter.emit_reasoning_item(sink, Some(reasoning))?;
            }
        }

        // Text message events already emitted naturally by process_chunk during stream processing
        // (OpenAI router approach - text only appears on final iteration when no tool calls)

        // No tool calls → final turn. The terminal `response.completed` event is
        // emitted once after the loop so every exit path emits it (see below).
        break (accumulated_response.usage, ResponseStatus::Completed);
    };

    // Emit the terminal response event for EVERY loop exit. Previously only the
    // normal-completion path emitted it inline; the max-tool-calls and
    // function-tool-call early exits broke out WITHOUT emitting, hanging
    // streaming clients that block on `response.completed`/`error` until timeout.
    // Emit once here for all exits.
    let terminal_usage_json = terminal_usage.as_ref().map(|u| {
        json!({
            "input_tokens": u.prompt_tokens,
            "output_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens
        })
    });
    // SSE drains emitter state (terminal, byte-identical to prior output) for a
    // normal `Completed` turn, but on the early exits (`InProgress`:
    // function-tool-call / max-tool-calls) it must preserve the explicit status so
    // clients see the turn is still pending rather than a hardcoded `completed`.
    // WS always emits with the explicit status (clone) so `finalize_with_status`
    // below still materializes the full output.
    let event = if drain_completed && matches!(terminal_status, ResponseStatus::Completed) {
        emitter.emit_completed(terminal_usage_json.as_ref())
    } else {
        emitter.emit_completed_with_status(terminal_status.clone(), terminal_usage_json.as_ref())
    };
    emitter.send_event(&event, sink)?;

    // Materialize the finalized response from the (non-drained) emitter state so
    // the WS connection cache and persistence see the full output. Both
    // transports consume this value: the WS caller caches and persists it, and
    // the SSE caller persists it (store=true) in the spawned tool-loop task.
    Ok(emitter.finalize_with_status(terminal_usage, terminal_status))
}

/// Convert chat stream to Responses API events while accumulating for tool call detection
async fn convert_and_accumulate_stream(
    body: Body,
    emitter: &mut ResponseStreamEventEmitter,
    sink: &impl ResponseEventSink,
) -> Result<ChatCompletionResponse, String> {
    let mut accumulator = ChatResponseAccumulator::new();
    let mut stream = body.into_data_stream();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Stream read error: {e}"))?;

        // Parse chunk
        let event_str = String::from_utf8_lossy(&chunk);
        let event = event_str.trim();

        if event == "data: [DONE]" {
            break;
        }

        if let Some(json_str) = event.strip_prefix("data: ") {
            let json_str = json_str.trim();
            match serde_json::from_str::<ChatCompletionStreamResponse>(json_str) {
                Ok(chat_chunk) => {
                    // Convert chat chunk to Responses API events and emit
                    emitter.process_chunk(&chat_chunk, sink)?;

                    // Accumulate for tool call detection
                    accumulator.process_chunk(&chat_chunk);
                }
                Err(_) => {
                    // A frame that isn't a chat-completion delta but carries an
                    // `error` object is a mid-stream worker error. The MCP tool
                    // loop must surface it rather than silently drop the frame and
                    // finalize a bogus `response.completed` (the non-MCP path
                    // forwards such frames to the client; here we fail the turn so
                    // the caller emits `response.failed` / an error event). Other
                    // unrecognized informational frames stay lenient (ignored).
                    if let Some(message) = serde_json::from_str::<Value>(json_str)
                        .ok()
                        .as_ref()
                        .and_then(|value| value.get("error"))
                        .map(|error| {
                            error
                                .get("message")
                                .and_then(Value::as_str)
                                .unwrap_or("upstream worker stream error")
                                .to_string()
                        })
                    {
                        return Err(message);
                    }
                }
            }
        }
    }

    Ok(accumulator.finalize())
}

/// Accumulates chat streaming chunks into complete ChatCompletionResponse
struct ChatResponseAccumulator {
    id: String,
    model: String,
    content: String,
    reasoning_content: Option<String>,
    tool_calls: HashMap<usize, ToolCall>,
    finish_reason: Option<String>,
    usage: Option<Usage>,
}

impl ChatResponseAccumulator {
    fn new() -> Self {
        Self {
            id: String::new(),
            model: String::new(),
            content: String::new(),
            reasoning_content: None,
            tool_calls: HashMap::new(),
            finish_reason: None,
            usage: None,
        }
    }

    fn process_chunk(&mut self, chunk: &ChatCompletionStreamResponse) {
        if !chunk.id.is_empty() {
            self.id.clone_from(&chunk.id);
        }
        if !chunk.model.is_empty() {
            self.model.clone_from(&chunk.model);
        }

        if let Some(choice) = chunk.choices.first() {
            // Accumulate content
            if let Some(content) = &choice.delta.content {
                self.content.push_str(content);
            }

            // Accumulate reasoning content
            if let Some(reasoning) = &choice.delta.reasoning_content {
                self.reasoning_content
                    .get_or_insert_with(String::new)
                    .push_str(reasoning);
            }

            // Accumulate tool calls
            if let Some(tool_call_deltas) = &choice.delta.tool_calls {
                for delta in tool_call_deltas {
                    let index = delta.index as usize;
                    let entry = self.tool_calls.entry(index).or_insert_with(|| ToolCall {
                        id: String::new(),
                        tool_type: "function".to_string(),
                        function: FunctionCallResponse {
                            name: String::new(),
                            arguments: Some(String::new()),
                        },
                    });

                    if let Some(id) = &delta.id {
                        entry.id.clone_from(id);
                    }
                    if let Some(function) = &delta.function {
                        if let Some(name) = &function.name {
                            entry.function.name.clone_from(name);
                        }
                        if let Some(args) = &function.arguments {
                            if let Some(ref mut existing_args) = entry.function.arguments {
                                existing_args.push_str(args);
                            }
                        }
                    }
                }
            }

            // Capture finish reason
            if let Some(reason) = &choice.finish_reason {
                self.finish_reason = Some(reason.clone());
            }
        }

        // Update usage
        if let Some(usage) = &chunk.usage {
            self.usage = Some(usage.clone());
        }
    }

    fn finalize(self) -> ChatCompletionResponse {
        let mut tool_calls_vec: Vec<_> = self.tool_calls.into_iter().collect();
        tool_calls_vec.sort_by_key(|(index, _)| *index);
        let tool_calls: Vec<_> = tool_calls_vec.into_iter().map(|(_, call)| call).collect();

        ChatCompletionResponse::builder(&self.id, &self.model)
            .choices(vec![ChatChoice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: if self.content.is_empty() {
                        None
                    } else {
                        Some(self.content)
                    },
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    reasoning_content: self.reasoning_content,
                },
                finish_reason: self.finish_reason,
                logprobs: None,
                matched_stop: None,
                hidden_states: None,
            }])
            .maybe_usage(self.usage)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_accumulator_populates_call_id_from_tool_delta_id() {
        let request = ResponsesRequest::default();
        let mut accumulator = StreamingResponseAccumulator::new(&request);
        let chunk = ChatCompletionStreamResponse::builder("chatcmpl_test", "test-model")
            .add_choice_tool_name(0, "call_streamed", "lookup")
            .build();

        accumulator.process_chunk(&chunk);

        assert_eq!(accumulator.tool_calls.len(), 1);
        match &accumulator.tool_calls[0] {
            ResponseOutputItem::FunctionToolCall {
                id, call_id, name, ..
            } => {
                assert_eq!(id.as_deref(), Some("call_streamed"));
                assert_eq!(call_id, "call_streamed");
                assert_eq!(name, "lookup");
            }
            other => panic!("expected function tool call, got {other:?}"),
        }
    }

    /// Collecting [`ResponseEventSink`] that records every emitted event so a
    /// test can inspect the wire frames the client would have received.
    #[derive(Default)]
    struct CollectingSink {
        events: std::sync::Mutex<Vec<Value>>,
        raw_jsons: std::sync::Mutex<Vec<String>>,
    }

    impl CollectingSink {
        /// The `id` carried by the most recent `response.completed` frame — the
        /// id the client actually observed on the wire.
        fn streamed_completed_id(&self) -> Option<String> {
            self.events
                .lock()
                .expect("sink mutex poisoned")
                .iter()
                .rev()
                .find(|event| event["type"] == "response.completed")
                .and_then(|event| event["response"]["id"].as_str().map(str::to_owned))
        }

        /// Raw pass-through payloads handed to `send_raw_json` (the exact bytes a
        /// client would receive after the sink's own framing).
        fn raw_jsons(&self) -> Vec<String> {
            self.raw_jsons.lock().expect("sink mutex poisoned").clone()
        }
    }

    impl ResponseEventSink for CollectingSink {
        fn send_event(&self, event: &Value) -> Result<(), String> {
            self.events
                .lock()
                .expect("sink mutex poisoned")
                .push(event.clone());
            Ok(())
        }

        fn send_raw_json(&self, payload: &str) -> Result<(), String> {
            self.raw_jsons
                .lock()
                .expect("sink mutex poisoned")
                .push(payload.to_string());
            Ok(())
        }
    }

    /// Frame an SSE body out of chat chunks whose ids begin with `chatcmpl-`,
    /// the same shape the worker emits.
    fn sse_body_from_chunks(chunks: &[ChatCompletionStreamResponse]) -> Body {
        let mut payload = String::new();
        for chunk in chunks {
            let json = serde_json::to_string(chunk).expect("chunk serializes");
            payload.push_str("data: ");
            payload.push_str(&json);
            payload.push_str("\n\n");
        }
        payload.push_str("data: [DONE]\n\n");
        Body::from(payload)
    }

    /// Regression: the worker stamps each chat chunk with its own
    /// `chatcmpl-…` id, but the client receives (and continuation relies on)
    /// the emitter's `resp_…` id. `drive_non_mcp_stream` must return a response
    /// keyed by the emitter id — never the chunk id — and that returned id must
    /// match the id streamed in `response.completed`. Caught only by live
    /// inference before this test existed (resp_/chatcmpl- mismatch broke
    /// `previous_response_id` continuation).
    #[tokio::test]
    async fn drive_non_mcp_stream_stamps_emitter_resp_id_not_chunk_id() {
        let chunk = ChatCompletionStreamResponse::builder("chatcmpl-fake-worker-123", "test-model")
            .add_choice_content(0, "assistant", "hello world")
            .build();
        // A second chunk carrying the terminal finish_reason and usage.
        let mut finish_chunk =
            ChatCompletionStreamResponse::builder("chatcmpl-fake-worker-123", "test-model").build();
        finish_chunk
            .choices
            .push(openai_protocol::chat::ChatStreamChoice {
                index: 0,
                delta: openai_protocol::chat::ChatMessageDelta {
                    role: None,
                    content: None,
                    tool_calls: None,
                    reasoning_content: None,
                },
                logprobs: None,
                finish_reason: Some("stop".to_string()),
                matched_stop: None,
            });

        let body = sse_body_from_chunks(&[chunk, finish_chunk]);
        let request = ResponsesRequest {
            model: "test-model".to_string(),
            ..Default::default()
        };
        let sink = CollectingSink::default();

        // `drain_completed = false` exercises the WS path, which is the one that
        // populates the connection cache from the returned response.
        let final_response = drive_non_mcp_stream(body, &request, &sink, false)
            .await
            .expect("stream drives to completion");

        assert!(
            final_response.id.starts_with("resp_"),
            "finalized id must be the emitter resp_ id, got {:?}",
            final_response.id
        );
        assert!(
            !final_response.id.contains("chatcmpl-"),
            "finalized id must not adopt the worker chunk id, got {:?}",
            final_response.id
        );

        // The id the client received on the wire must equal the id we cache /
        // persist, so a follow-up `previous_response_id` resolves.
        let streamed_id = sink
            .streamed_completed_id()
            .expect("a response.completed frame was streamed");
        assert_eq!(
            streamed_id, final_response.id,
            "streamed completed id must match the finalized/cached id"
        );
    }

    /// Regression for the double-`data:` framing bug: a non-chat SSE event (e.g.
    /// the worker's mid-stream `data: {"error":…}`) fails
    /// `ChatCompletionStreamResponse` parsing and is forwarded via
    /// `send_raw_json`. It must be forwarded as the data:-STRIPPED JSON, NOT the
    /// prefixed line — otherwise `send_raw_json` re-frames it to `data: data: …`
    /// on SSE (and ships non-JSON text to WS clients). Caught only on the wire
    /// before this test existed.
    #[tokio::test]
    async fn drive_non_mcp_stream_forwards_non_chat_events_single_framed() {
        let chunk = ChatCompletionStreamResponse::builder("chatcmpl-x", "test-model")
            .add_choice_content(0, "assistant", "hi")
            .build();
        let error_json =
            r#"{"error":{"message":"boom","type":"server_error","code":"internal_error"}}"#;
        // Each HTTP body frame is treated as one complete SSE event, so emit the
        // valid chunk, the non-chat error event, and [DONE] as SEPARATE frames.
        let frames: Vec<Result<bytes::Bytes, std::io::Error>> = vec![
            Ok(bytes::Bytes::from(format!(
                "data: {}\n\n",
                serde_json::to_string(&chunk).expect("chunk serializes")
            ))),
            Ok(bytes::Bytes::from(format!("data: {error_json}\n\n"))),
            Ok(bytes::Bytes::from("data: [DONE]\n\n")),
        ];
        let body = Body::from_stream(futures_util::stream::iter(frames));

        let request = ResponsesRequest {
            model: "test-model".to_string(),
            ..Default::default()
        };
        let sink = CollectingSink::default();
        drive_non_mcp_stream(body, &request, &sink, false)
            .await
            .expect("stream drives to completion");

        let raw = sink.raw_jsons();
        assert_eq!(
            raw.len(),
            1,
            "exactly one non-chat event should be forwarded, got {raw:?}"
        );
        assert_eq!(
            raw[0], error_json,
            "non-chat event must be forwarded as stripped JSON (no `data: ` prefix)"
        );
        assert!(
            !raw[0].starts_with("data: "),
            "forwarded payload must not retain the SSE `data: ` prefix (would double-frame), got {:?}",
            raw[0]
        );
    }

    /// Frame chat chunks + extra raw `data:` lines into a multi-frame SSE body
    /// (each `Bytes` is one SSE event), so a non-chat frame can be interleaved.
    fn sse_body_from_lines(lines: &[String]) -> Body {
        let frames: Vec<Result<Bytes, std::io::Error>> = lines
            .iter()
            .map(|line| Ok(Bytes::from(format!("data: {line}\n\n"))))
            .chain(std::iter::once(Ok(Bytes::from("data: [DONE]\n\n"))))
            .collect();
        Body::from_stream(futures_util::stream::iter(frames))
    }

    /// Regression: on the MCP tool-loop path a mid-stream worker error frame
    /// (`data: {"error":…}`) must FAIL the turn rather than be silently dropped
    /// and finalized as a bogus successful response. The non-MCP path forwards
    /// such frames to the client; the MCP accumulator (which feeds tool-call
    /// detection) had no `else` branch, so the error chunk failed
    /// `ChatCompletionStreamResponse` parsing and was ignored, and the loop
    /// returned `Ok(empty completed)` — the caller then emitted a spurious
    /// `response.completed`. The error must now surface as `Err` so the caller
    /// emits `response.failed` / an error event.
    #[tokio::test]
    async fn convert_and_accumulate_stream_surfaces_worker_error_frame() {
        let valid = ChatCompletionStreamResponse::builder("chatcmpl-x", "test-model")
            .add_choice_content(0, "assistant", "partial")
            .build();
        let error_json = r#"{"error":{"message":"worker exploded","type":"server_error","code":"internal_error"}}"#;
        let body = sse_body_from_lines(&[
            serde_json::to_string(&valid).expect("chunk serializes"),
            error_json.to_string(),
        ]);

        let mut emitter =
            ResponseStreamEventEmitter::new("resp_test".to_string(), "test-model".to_string(), 0);
        let sink = CollectingSink::default();

        let err = convert_and_accumulate_stream(body, &mut emitter, &sink)
            .await
            .expect_err("a mid-stream worker error frame must fail the turn");
        assert!(
            err.contains("worker exploded"),
            "surfaced error must carry the upstream worker message, got {err:?}"
        );
        assert!(
            sink.streamed_completed_id().is_none(),
            "no terminal response.completed may be emitted for a failed turn"
        );
    }

    /// Companion to the worker-error test: a non-chat frame WITHOUT an `error`
    /// object (e.g. an unrecognized informational/keepalive event) must stay
    /// lenient — ignored, not turned into a turn failure — so the fix does not
    /// regress benign streams.
    #[tokio::test]
    async fn convert_and_accumulate_stream_ignores_benign_unknown_frame() {
        let valid = ChatCompletionStreamResponse::builder("chatcmpl-x", "test-model")
            .add_choice_content(0, "assistant", "hello")
            .build();
        let body = sse_body_from_lines(&[
            serde_json::to_string(&valid).expect("chunk serializes"),
            r#"{"note":"keepalive","seq":7}"#.to_string(),
        ]);

        let mut emitter =
            ResponseStreamEventEmitter::new("resp_test".to_string(), "test-model".to_string(), 0);
        let sink = CollectingSink::default();

        convert_and_accumulate_stream(body, &mut emitter, &sink)
            .await
            .expect("a benign non-error frame must not fail the turn");
    }
}
