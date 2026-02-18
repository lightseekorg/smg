//! Streaming processor for Anthropic Messages API
//!
//! Handles both passthrough (no MCP) and MCP tool loop streaming paths,
//! composing worker, sse, and mcp primitives.

use std::{io, sync::Arc, time::Instant};

use axum::{
    body::Body,
    http::{header, StatusCode},
    response::Response,
};
use bytes::Bytes;
use openai_protocol::messages::{InputContent, InputMessage, Role};
use smg_mcp::McpToolSession;
use tokio::sync::mpsc;
use tracing::{debug, error, warn};

use super::{
    context::{RequestContext, RouterContext},
    mcp, sse, worker,
};
use crate::{
    core::Worker,
    observability::metrics::{metrics_labels, Metrics},
    routers::{
        error::{self as router_error, extract_error_code_from_response},
        mcp_utils::DEFAULT_MAX_ITERATIONS,
    },
};

/// Channel buffer size for SSE events sent to the client.
const SSE_CHANNEL_SIZE: usize = 128;

/// Execute a streaming Messages API request, handling both
/// passthrough (no MCP) and MCP tool loop paths.
pub(crate) async fn execute(router: &RouterContext, req_ctx: RequestContext) -> Response {
    if req_ctx.mcp_servers.is_some() {
        return execute_mcp_streaming(router, req_ctx).await;
    }
    execute_passthrough(router, &req_ctx).await
}

// ============================================================================
// Passthrough streaming (no MCP)
// ============================================================================

/// Direct streaming: select worker, send, wrap in load-tracking SSE response.
async fn execute_passthrough(router: &RouterContext, req_ctx: &RequestContext) -> Response {
    let model_id = &req_ctx.model_id;
    let start_time = Instant::now();

    let worker_arc = match worker::select_worker(&router.worker_registry, model_id) {
        Ok(w) => w,
        Err(resp) => return resp,
    };
    worker::record_router_request(model_id, true);
    let (url, req_headers) = worker::build_request(&*worker_arc, req_ctx.headers.as_ref());
    let response = match worker::send_request(
        &router.http_client,
        &url,
        &req_headers,
        &req_ctx.request,
        router.request_timeout,
        &*worker_arc,
    )
    .await
    {
        Ok(r) => r,
        Err(resp) => return resp,
    };

    build_streaming_response(response, model_id, start_time, worker_arc).await
}

/// Build a streaming SSE response with load tracking.
async fn build_streaming_response(
    response: reqwest::Response,
    model_id: &str,
    start_time: Instant,
    worker: Arc<dyn Worker>,
) -> Response {
    let status = response.status();

    if !status.is_success() {
        return build_streaming_error_response(response, model_id, start_time, worker).await;
    }

    debug!(model = %model_id, status = %status, "Starting streaming response");

    Metrics::record_router_duration(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model_id,
        "messages",
        start_time.elapsed(),
    );

    let headers = response.headers().clone();
    let stream = response.bytes_stream();
    let load_stream = sse::LoadTrackingStream::new(stream, worker);
    let body = Body::from_stream(load_stream);

    sse::build_sse_response(status, headers, body)
}

/// Handle a non-success streaming response.
async fn build_streaming_error_response(
    response: reqwest::Response,
    model_id: &str,
    start_time: Instant,
    worker: Arc<dyn Worker>,
) -> Response {
    let status = response.status();
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    // If it's an SSE error stream, pass it through (with worker load tracking)
    if content_type
        .to_ascii_lowercase()
        .contains("text/event-stream")
    {
        warn!(model = %model_id, status = %status, "Streaming error response (SSE)");

        Metrics::record_router_duration(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model_id,
            "messages",
            start_time.elapsed(),
        );
        Metrics::record_router_error(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model_id,
            "messages",
            "streaming_error",
        );

        let headers = response.headers().clone();
        let stream = response.bytes_stream();
        let load_stream = sse::LoadTrackingStream::new_force_failure(stream, worker);
        let body = Body::from_stream(load_stream);

        return sse::build_sse_response(status, headers, body);
    }

    // Non-SSE error: read the body and return a proper error response
    worker::handle_error_response(response, model_id, start_time, &*worker).await
}

// ============================================================================
// MCP streaming (tool loop)
// ============================================================================

/// Spawn the MCP tool loop in a background task and return an SSE response.
async fn execute_mcp_streaming(router: &RouterContext, req_ctx: RequestContext) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(SSE_CHANNEL_SIZE);

    let router = router.clone();

    tokio::spawn(async move {
        if let Err(e) = run_tool_loop(tx.clone(), router, req_ctx).await {
            warn!(error = %e, "Streaming tool loop failed");
            let _ = sse::send_error(&tx, &e).await;
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(body)
        .unwrap_or_else(|e| {
            error!("Failed to build streaming response: {}", e);
            router_error::internal_error("response_build_failed", "Failed to build response")
        })
}

/// Run the MCP tool loop, sending SSE events to the client via `tx`.
async fn run_tool_loop(
    tx: mpsc::Sender<Result<Bytes, io::Error>>,
    router: RouterContext,
    mut req_ctx: RequestContext,
) -> Result<(), String> {
    let session_id = format!("msg_{}", uuid::Uuid::new_v4());
    let mcp_servers = req_ctx.mcp_servers.take().unwrap_or_default();
    let session = McpToolSession::new(&router.mcp_orchestrator, mcp_servers, &session_id);

    let mut global_index: u32 = 0;
    let mut total_input_tokens: u32 = 0;
    let mut total_output_tokens: u32 = 0;
    let mut is_first_iteration = true;

    for _iteration in 0..DEFAULT_MAX_ITERATIONS {
        Metrics::record_mcp_tool_iteration(&req_ctx.model_id);

        let (response, selected_worker) = send_streaming_request(&router, &req_ctx).await?;

        // Consume the upstream SSE stream
        let result = sse::consume_and_forward(
            &tx,
            response,
            &mut global_index,
            is_first_iteration,
            |name| session.resolve_tool_server_label(name),
        )
        .await;

        // Worker load management
        selected_worker.decrement_load();
        match &result {
            Ok(_) => selected_worker.record_outcome(true),
            Err(_) => selected_worker.record_outcome(false),
        }

        let consumed = result?;
        is_first_iteration = false;

        // Accumulate usage
        if let Some(ref usage) = consumed.usage {
            total_output_tokens += usage.output_tokens;
            if let Some(input) = usage.input_tokens {
                total_input_tokens += input;
            }
        }

        // Check if we should continue the tool loop
        match mcp::process_iteration(&consumed.iteration, &session, &req_ctx.model_id).await {
            mcp::ToolLoopAction::Done => {
                sse::emit_final(
                    &tx,
                    consumed.iteration.stop_reason.as_ref(),
                    total_input_tokens,
                    total_output_tokens,
                )
                .await;
                return Ok(());
            }
            mcp::ToolLoopAction::Error(msg) => {
                return Err(msg);
            }
            mcp::ToolLoopAction::Continue(cont) => {
                // Emit mcp_tool_result events for each completed tool call
                for call in &cont.mcp_calls {
                    if !sse::emit_mcp_tool_result(&tx, call, &mut global_index).await {
                        return Ok(());
                    }
                }

                // Append assistant + tool_result messages for next iteration
                req_ctx.request.messages.push(InputMessage {
                    role: Role::Assistant,
                    content: InputContent::Blocks(cont.assistant_blocks),
                });
                req_ctx.request.messages.push(InputMessage {
                    role: Role::User,
                    content: InputContent::Blocks(cont.tool_result_blocks),
                });
            }
        }
    }

    // Max iterations exceeded
    warn!(
        "Streaming MCP tool loop exceeded max iterations ({})",
        DEFAULT_MAX_ITERATIONS
    );
    let error_msg = format!(
        "MCP tool loop exceeded maximum iterations ({})",
        DEFAULT_MAX_ITERATIONS
    );
    let _ = sse::send_error(&tx, &error_msg).await;
    Ok(())
}

/// Select a worker, send a streaming request, and return the response.
///
/// Returns the worker alongside the response for load tracking after
/// the stream is consumed.
async fn send_streaming_request(
    router: &RouterContext,
    req_ctx: &RequestContext,
) -> Result<(reqwest::Response, Arc<dyn Worker>), String> {
    let model_id = &req_ctx.model_id;

    let selected_worker =
        worker::select_worker(&router.worker_registry, model_id).map_err(|e| {
            format!(
                "No healthy workers available for model '{}' ({})",
                model_id,
                extract_error_code_from_response(&e)
            )
        })?;

    worker::record_router_request(model_id, true);

    let (url, req_headers) = worker::build_request(&*selected_worker, req_ctx.headers.as_ref());
    let response = worker::send_request(
        &router.http_client,
        &url,
        &req_headers,
        &req_ctx.request,
        router.request_timeout,
        &*selected_worker,
    )
    .await
    .map_err(|e| {
        format!(
            "Failed to send request to worker ({})",
            extract_error_code_from_response(&e)
        )
    })?;

    if !response.status().is_success() {
        let status = response.status();
        selected_worker.decrement_load();
        selected_worker.record_outcome(false);
        return Err(format!("Worker returned error status: {}", status));
    }

    Ok((response, selected_worker))
}
