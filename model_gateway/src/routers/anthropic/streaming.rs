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
use futures_util::Stream;
use openai_protocol::messages::{InputContent, InputMessage, Role};
use smg_mcp::McpToolSession;
use tokio::sync::mpsc;
use tracing::{debug, error, warn};

use super::{
    context::{RequestContext, RouterContext},
    mcp, sse, worker,
};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::{
        common::{
            mcp_utils::DEFAULT_MAX_ITERATIONS,
            sse::{observe_event_type, update_event_boundary, SseDecoder, SseEncoder},
            stream_timeout::{StreamDeadline, StreamTimeoutKind},
        },
        error::{self as router_error, extract_error_code_from_response},
    },
    worker::Worker as WorkerTrait,
};

/// Channel buffer size for SSE events sent to the client.
const SSE_CHANNEL_SIZE: usize = 128;

/// Execute a streaming Messages API request, handling both
/// passthrough (no MCP) and MCP tool loop paths.
pub(crate) async fn execute(router: &RouterContext, req_ctx: RequestContext) -> Response {
    if req_ctx.mcp_servers.is_some() {
        return execute_mcp_streaming(router, req_ctx);
    }
    execute_passthrough(router, &req_ctx).await
}

// ============================================================================
// Passthrough streaming (no MCP)
// ============================================================================

/// Direct streaming: send request and wrap in SSE response.
async fn execute_passthrough(router: &RouterContext, req_ctx: &RequestContext) -> Response {
    let model_id = &req_ctx.model_id;
    let start_time = Instant::now();

    worker::record_router_request(model_id, true);
    let (url, req_headers) = worker::build_request(&*req_ctx.worker, req_ctx.headers.as_ref());
    let stream_deadline = StreamDeadline::new(router.request_timeout, router.stream_idle_timeout);
    let response = match stream_deadline
        .until_total(worker::send_request(
            &router.streaming_http_client,
            &url,
            &req_headers,
            &req_ctx.request,
            None,
        ))
        .await
    {
        Ok(result) => match result {
            Ok(r) => r,
            Err(resp) => return resp,
        },
        Err(_) => {
            record_streaming_timeout_metrics(model_id, start_time);
            record_streaming_worker_outcome(
                Some(req_ctx.worker.as_ref()),
                StatusCode::GATEWAY_TIMEOUT,
            );
            return router_error::gateway_timeout(
                "streaming_timeout",
                stream_deadline.message(StreamTimeoutKind::Total),
            );
        }
    };

    build_streaming_response(
        response,
        req_ctx.worker.clone(),
        model_id,
        start_time,
        stream_deadline,
    )
    .await
}

/// Build a streaming SSE response.
async fn build_streaming_response(
    response: reqwest::Response,
    worker: Arc<dyn WorkerTrait>,
    model_id: &str,
    start_time: Instant,
    stream_deadline: StreamDeadline,
) -> Response {
    let status = response.status();

    if !status.is_success() {
        return build_streaming_error_response(response, model_id, start_time, stream_deadline)
            .await;
    }

    debug!(model = %model_id, status = %status, "Starting streaming response");

    let headers = response.headers().clone();
    let upstream_stream = response.bytes_stream();
    let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(SSE_CHANNEL_SIZE);

    #[expect(
        clippy::disallowed_methods,
        reason = "fire-and-forget stream relay; gateway shutdown need not wait for individual stream forwarding"
    )]
    tokio::spawn(relay_stream_with_deadline(
        tx,
        upstream_stream,
        stream_deadline,
        model_id.to_string(),
        start_time,
        Some(worker),
        true,
    ));

    let body = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));

    sse::build_sse_response(status, headers, body)
}

fn record_streaming_error_metric(model_id: &str, error_type: &'static str) {
    Metrics::record_router_error(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model_id,
        "messages",
        error_type,
    );
}

fn record_streaming_timeout_metrics(model_id: &str, _start_time: Instant) {
    record_streaming_error_metric(model_id, "streaming_timeout");
}

fn record_streaming_router_outcome(model_id: &str, start_time: Instant, status: StatusCode) {
    if status.is_success() {
        Metrics::record_router_duration(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model_id,
            "messages",
            start_time.elapsed(),
        );
    } else {
        let error_type = if status == StatusCode::GATEWAY_TIMEOUT {
            "streaming_timeout"
        } else {
            metrics_labels::ERROR_BACKEND
        };
        record_streaming_error_metric(model_id, error_type);
    }
}

fn record_streaming_worker_outcome(worker: Option<&dyn WorkerTrait>, status: StatusCode) {
    let Some(worker) = worker else {
        return;
    };
    worker.record_outcome(status.as_u16());
    if status.is_server_error() {
        Metrics::record_worker_error(
            metrics_labels::WORKER_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            if status == StatusCode::GATEWAY_TIMEOUT {
                metrics_labels::ERROR_TIMEOUT
            } else {
                metrics_labels::ERROR_BACKEND
            },
        );
    }
}

fn is_streaming_timeout_message(message: &str) -> bool {
    message.starts_with("Streaming request exceeded configured ")
}

async fn relay_stream_with_deadline<S>(
    tx: mpsc::Sender<Result<Bytes, io::Error>>,
    mut upstream_stream: S,
    stream_deadline: StreamDeadline,
    model_id: String,
    start_time: Instant,
    worker: Option<Arc<dyn WorkerTrait>>,
    record_router_outcome: bool,
) where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin + Send + 'static,
{
    let mut encoder = SseEncoder::new();
    let mut boundary_tail = Vec::new();
    let mut at_event_boundary = true;
    let mut terminal_decoder = SseDecoder::new();
    let mut stream_failure_status = None;
    loop {
        let chunk = match stream_deadline.next(&mut upstream_stream).await {
            Ok(Some(chunk)) => chunk,
            Ok(None) => break,
            Err(timeout) => {
                let message = stream_deadline.message(timeout);
                stream_failure_status = Some(StatusCode::GATEWAY_TIMEOUT);
                if at_event_boundary {
                    let _ = sse::send_error(&tx, &mut encoder, &message).await;
                } else {
                    let _ = tx.send(Err(io::Error::other(message))).await;
                }
                break;
            }
        };
        match chunk {
            Ok(bytes) => {
                let stream_done =
                    observe_event_type(&mut terminal_decoder, bytes.as_ref(), "message_stop");
                at_event_boundary = update_event_boundary(&mut boundary_tail, bytes.as_ref());
                if tx.send(Ok(bytes)).await.is_err() {
                    break;
                }
                if stream_done {
                    break;
                }
            }
            Err(e) => {
                stream_failure_status = Some(StatusCode::BAD_GATEWAY);
                let _ = tx.send(Err(io::Error::other(e))).await;
                break;
            }
        }
    }
    let effective_status = stream_failure_status.unwrap_or(StatusCode::OK);
    record_streaming_worker_outcome(worker.as_deref(), effective_status);
    if record_router_outcome {
        record_streaming_router_outcome(&model_id, start_time, effective_status);
    }
}

/// Handle a non-success streaming response.
async fn build_streaming_error_response(
    response: reqwest::Response,
    model_id: &str,
    start_time: Instant,
    stream_deadline: StreamDeadline,
) -> Response {
    let status = response.status();
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    // If it's an SSE error stream, pass it through
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
        let upstream_stream = response.bytes_stream();
        let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(SSE_CHANNEL_SIZE);

        #[expect(
            clippy::disallowed_methods,
            reason = "fire-and-forget error stream relay; gateway shutdown need not wait for individual stream forwarding"
        )]
        tokio::spawn(relay_stream_with_deadline(
            tx,
            upstream_stream,
            stream_deadline,
            model_id.to_string(),
            start_time,
            None,
            false,
        ));

        let body = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));

        return sse::build_sse_response(status, headers, body);
    }

    // Non-SSE error: read the body and return a proper error response
    worker::handle_streaming_error_response(response, model_id, start_time, stream_deadline).await
}

// ============================================================================
// MCP streaming (tool loop)
// ============================================================================

/// Spawn the MCP tool loop in a background task and return an SSE response.
fn execute_mcp_streaming(router: &RouterContext, req_ctx: RequestContext) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(SSE_CHANNEL_SIZE);

    let router = router.clone();

    #[expect(
        clippy::disallowed_methods,
        reason = "fire-and-forget streaming task; gateway shutdown need not wait for individual MCP tool loops"
    )]
    tokio::spawn(async move {
        if let Err(e) = Box::pin(run_tool_loop(tx.clone(), router, req_ctx)).await {
            if e == sse::CLIENT_DISCONNECTED_ERROR {
                debug!(error = %e, "Streaming tool loop ended: client disconnected");
                return;
            }
            warn!(error = %e, "Streaming tool loop failed");
            let mut enc = SseEncoder::new();
            let _ = sse::send_error(&tx, &mut enc, &e).await;
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
    let session_id = format!("msg_{}", uuid::Uuid::now_v7());
    let mcp_servers = req_ctx.mcp_servers.take().unwrap_or_default();
    let session = McpToolSession::new(&router.mcp_orchestrator, mcp_servers, &session_id);

    // Inject MCP tools into the request as regular tools
    mcp::inject_mcp_tools_into_request(&mut req_ctx.request, &session);

    let mut global_index: u32 = 0;
    let mut total_input_tokens: u32 = 0;
    let mut total_output_tokens: u32 = 0;
    let mut is_first_iteration = true;
    // Reusable SSE encoder shared across every event emitted for this stream.
    let mut encoder = SseEncoder::new();
    let stream_deadline = StreamDeadline::new(router.request_timeout, router.stream_idle_timeout);

    for _iteration in 0..DEFAULT_MAX_ITERATIONS {
        Metrics::record_mcp_tool_iteration(&req_ctx.model_id);

        let iteration_start = Instant::now();
        let response = send_streaming_request(
            &router,
            &req_ctx,
            stream_deadline,
            iteration_start,
            is_first_iteration,
        )
        .await?;

        if !response.status().is_success() {
            return Err(format!(
                "Worker returned error status: {}",
                response.status()
            ));
        }

        // Consume the upstream SSE stream
        let result = sse::consume_and_forward(
            &tx,
            &mut encoder,
            response,
            &mut global_index,
            is_first_iteration,
            stream_deadline,
            |name| session.resolve_tool_server_label(name),
        )
        .await;

        let consumed = match result {
            Ok(consumed) => consumed,
            Err(err) => {
                if is_streaming_timeout_message(&err) {
                    record_streaming_timeout_metrics(&req_ctx.model_id, iteration_start);
                    record_streaming_worker_outcome(
                        Some(req_ctx.worker.as_ref()),
                        StatusCode::GATEWAY_TIMEOUT,
                    );
                }
                return Err(err);
            }
        };
        is_first_iteration = false;

        // Accumulate usage
        if let Some(ref usage) = consumed.usage {
            total_output_tokens += usage.output_tokens;
            if let Some(input) = usage.input_tokens {
                total_input_tokens += input;
            }
        }

        // Check if we should continue the tool loop
        let action = match Box::pin(stream_deadline.until_activity(mcp::process_iteration(
            &consumed.iteration,
            &session,
            &req_ctx.model_id,
        )))
        .await
        {
            Ok(action) => action,
            Err(timeout) => {
                record_streaming_timeout_metrics(&req_ctx.model_id, iteration_start);
                let _ = sse::send_error(&tx, &mut encoder, &stream_deadline.message(timeout)).await;
                return Ok(());
            }
        };
        match action {
            mcp::ToolLoopAction::Done => {
                sse::emit_final(
                    &tx,
                    &mut encoder,
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
                    if !sse::emit_mcp_tool_result(&tx, &mut encoder, call, &mut global_index).await
                    {
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
    let error_msg = format!("MCP tool loop exceeded maximum iterations ({DEFAULT_MAX_ITERATIONS})");
    let _ = sse::send_error(&tx, &mut encoder, &error_msg).await;
    Ok(())
}

/// Send a streaming request to the pre-selected worker.
async fn send_streaming_request(
    router: &RouterContext,
    req_ctx: &RequestContext,
    stream_deadline: StreamDeadline,
    start_time: Instant,
    is_first_iteration: bool,
) -> Result<reqwest::Response, String> {
    worker::record_router_request(&req_ctx.model_id, true);

    let (url, req_headers) = worker::build_request(&*req_ctx.worker, req_ctx.headers.as_ref());
    let send_future = worker::send_request(
        &router.streaming_http_client,
        &url,
        &req_headers,
        &req_ctx.request,
        None,
    );
    let response = if is_first_iteration {
        stream_deadline.until_total(send_future).await
    } else {
        stream_deadline.until_activity(send_future).await
    }
    .map_err(|timeout| {
        record_streaming_timeout_metrics(&req_ctx.model_id, start_time);
        record_streaming_worker_outcome(Some(req_ctx.worker.as_ref()), StatusCode::GATEWAY_TIMEOUT);
        stream_deadline.message(timeout)
    })?
    .map_err(|e| {
        format!(
            "Failed to send request to worker ({})",
            extract_error_code_from_response(&e)
        )
    })?;

    Ok(response)
}
