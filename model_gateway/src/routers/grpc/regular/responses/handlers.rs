//! Handler functions for /v1/responses endpoints.
//!
//! Both streaming and non-streaming requests now flow through the same
//! shared agent-loop driver:
//!
//! - sync → `non_streaming::route_responses_internal` runs the loop
//!   with `RegularAdapter` + `NoopSink`
//! - stream → `route_responses_streaming` spawns a task running the
//!   loop with `RegularStreamingAdapter` + `GrpcResponseStreamSink`
//!
//! Background mode is rejected at this surface boundary.

use std::sync::Arc;

use axum::{
    body::Body,
    http::{self, header, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use openai_protocol::responses::ResponsesRequest;
use smg_mcp::McpToolSession;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use uuid::Uuid;

use super::{
    agent_streaming_adapter::{RegularStreamingAdapter, RegularStreamingUpstreamHandle},
    common::ResponsesCallContext,
    non_streaming,
};
use crate::routers::{
    common::{
        agent_loop::{run_agent_loop, AgentLoopContext, AgentLoopState, ToolTransferDescriptor},
        responses_streaming::ResponseStreamEventEmitter,
    },
    error,
    grpc::common::responses::{
        ensure_mcp_connection, prepare_request_history, GrpcResponseStreamSink, ResponsesContext,
    },
};

pub(crate) async fn route_responses(
    ctx: &ResponsesContext,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    tenant_request_meta: crate::middleware::TenantRequestMeta,
    model_id: String,
) -> Response {
    let is_background = request.background.unwrap_or(false);
    if is_background {
        return error::bad_request(
            "unsupported_parameter",
            "Background mode is not supported. Please set 'background' to false or omit it.",
        );
    }

    let is_streaming = request.stream.unwrap_or(false);
    if is_streaming {
        let params = ResponsesCallContext {
            headers,
            model_id,
            response_id: None,
            tenant_request_meta,
        };
        route_responses_streaming(ctx, request, params).await
    } else {
        let params = ResponsesCallContext {
            headers,
            model_id,
            response_id: Some(format!("resp_{}", Uuid::now_v7())),
            tenant_request_meta,
        };
        match non_streaming::route_responses_internal(ctx, request, params).await {
            Ok(responses_response) => axum::Json(responses_response).into_response(),
            Err(response) => response,
        }
    }
}

#[expect(
    clippy::disallowed_methods,
    reason = "streaming task is fire-and-forget; client disconnect terminates it"
)]
async fn route_responses_streaming(
    ctx: &ResponsesContext,
    request: Arc<ResponsesRequest>,
    params: ResponsesCallContext,
) -> Response {
    let loaded = match prepare_request_history(ctx, &request).await {
        Ok(history) => history,
        Err(response) => return response,
    };
    let modified_request = loaded.request;
    let emitted_mcp_server_labels = loaded.existing_mcp_list_tools_labels;
    let prepared = loaded.prepared;

    let (_, mcp_servers) =
        match ensure_mcp_connection(&ctx.mcp_orchestrator, request.tools.as_deref()).await {
            Ok(result) => result,
            Err(response) => return response,
        };

    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();
    let response_id = format!("resp_{}", Uuid::now_v7());
    let model = modified_request.model.clone();
    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut emitter = ResponseStreamEventEmitter::new(response_id.clone(), model, created_at);

    let ctx_clone = ctx.clone();
    let original_request = (*request).clone();
    // Feed the emitter the user-provided request shape (with
    // `previous_response_id` / `conversation` still set) so the
    // `response.completed` payload echoes them.
    emitter.set_original_request(original_request.clone());

    tokio::spawn(async move {
        let ctx = &ctx_clone;
        let session_request_id = format!("resp_{}", Uuid::now_v7());
        let mut session =
            McpToolSession::new(&ctx.mcp_orchestrator, mcp_servers, &session_request_id);
        if let Some(tools) = original_request.tools.as_deref() {
            session.configure_approval_policy(tools);
        }
        let max_tool_calls = modified_request.max_tool_calls.map(|n| n as usize);

        let upstream_handle = RegularStreamingUpstreamHandle {
            headers: params.headers.clone(),
            model_id: params.model_id.clone(),
            tenant_request_meta: params.tenant_request_meta.clone(),
        };
        let adapter = RegularStreamingAdapter::new(ctx, upstream_handle, &session);

        let state = AgentLoopState::new(prepared.upstream_input.clone(), emitted_mcp_server_labels);
        // `loop_ctx.original_request` carries the user-provided shape
        // (with `previous_response_id` / `conversation` set) so the
        // adapter's `render_final` can echo those fields onto the
        // response. Iteration-request rebuilding pulls input items
        // from `state.upstream_input` (already merged with history).
        let loop_ctx = AgentLoopContext {
            prepared: &prepared,
            session: Some(&session),
            original_request: &original_request,
            max_tool_calls,
        };

        let sink = GrpcResponseStreamSink::new(
            emitter,
            tx.clone(),
            ToolTransferDescriptor::map_from_mcp_snapshots(
                session.presentation_snapshot(),
                session.approval_snapshot(),
            ),
            session.server_label_snapshot(),
        );
        match run_agent_loop(adapter, loop_ctx, state, sink).await {
            Ok(()) => {
                // Adapter's `render_final` already persisted the
                // accumulated response via `persist_response_if_needed`
                // (see `agent_streaming_adapter.rs::render_final`).
                // Nothing left to do here — the SSE channel keeps
                // streaming until the spawned task drops its sender
                // and the receiver below sees end-of-stream.
            }
            Err(e) => {
                // Uniform `/v1/responses` error payload across surfaces.
                let _ = tx.send(Ok(Bytes::from(format!(
                    "event: error\ndata: {{\"error\":\"agent_loop_failed\",\"status\":{}}}\n\n",
                    e.into_response().status().as_u16()
                ))));
            }
        }

        let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
    });

    // Return SSE response with the receiver stream.
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
