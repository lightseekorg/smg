//! Streaming Harmony Responses API entry point.
//!
//! Mirrors the non-streaming surface: hydrate history, connect MCP,
//! build a `HarmonyStreamingAdapter`, and run the shared agent-loop
//! driver. The driver pumps `LoopEvent`s into the gRPC sink which
//! translates them to Responses-API SSE; chunk-level events
//! (text/reasoning deltas, function-call argument deltas) are emitted
//! by the adapter directly through the sink's underlying emitter.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::response::Response;
use openai_protocol::responses::ResponsesRequest;
use tokio::sync::mpsc;
use uuid::Uuid;

use super::agent_streaming_adapter::HarmonyStreamingAdapter;
use crate::{
    middleware::TenantRequestMeta,
    routers::{
        common::{
            agent_loop::{run_agent_loop, AgentLoopContext, ToolTransferDescriptor},
            responses_loop_setup::ResponsesLoopSetup,
            responses_streaming::ResponseStreamEventEmitter,
        },
        grpc::common::responses::{
            build_sse_response, ensure_mcp_connection, prepare_request_history,
            GrpcResponseStreamSink, ResponsesContext,
        },
    },
};

#[expect(
    clippy::disallowed_methods,
    reason = "streaming task is fire-and-forget; client disconnect terminates it"
)]
pub(crate) async fn serve_harmony_responses_stream(
    ctx: &ResponsesContext,
    request: ResponsesRequest,
    tenant_request_meta: TenantRequestMeta,
) -> Response {
    let original_request = request.clone();

    let loaded = match prepare_request_history(ctx, &request).await {
        Ok(history) => history,
        Err(err_response) => return err_response,
    };

    let (_, mcp_servers) =
        match ensure_mcp_connection(&ctx.mcp_orchestrator, original_request.tools.as_deref()).await
        {
            Ok(result) => result,
            Err(response) => return response,
        };
    let setup = ResponsesLoopSetup::from_history(loaded, mcp_servers);

    let (tx, rx) = mpsc::unbounded_channel();
    let response_id = format!("resp_{}", Uuid::now_v7());
    let model = setup.current_request.model.clone();
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut emitter = ResponseStreamEventEmitter::new(response_id.clone(), model, created_at);
    // Feed the emitter the user-provided request shape (with
    // `previous_response_id` / `conversation` still set) so the
    // `response.completed` payload echoes them. The post-load
    // `current_request` strips those fields for in-loop sub-calls.
    emitter.set_original_request(original_request.clone());

    let ctx_clone = ctx.clone();
    let original_for_persist = original_request.clone();

    tokio::spawn(async move {
        let ctx = &ctx_clone;
        let session_request_id = format!("resp_{}", Uuid::now_v7());
        let session = setup.session(
            &ctx.mcp_orchestrator,
            &session_request_id,
            &original_for_persist,
        );
        let ResponsesLoopSetup {
            current_request,
            prepared,
            state,
            max_tool_calls,
            ..
        } = setup;
        let adapter =
            HarmonyStreamingAdapter::new(ctx, tenant_request_meta, &current_request, &session);
        // `loop_ctx.original_request` carries the user-provided shape
        // (with `previous_response_id` / `conversation` set) so the
        // streaming adapter's `render_final` can echo those fields.
        // Iteration request rebuilding pulls input items from
        // `state.upstream_input` (already merged with history).
        let loop_ctx = AgentLoopContext {
            prepared: &prepared,
            session: Some(&session),
            original_request: &original_for_persist,
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
        // The driver returns once Finish/ApprovalInterrupt/Incomplete is
        // reached; the sink itself has already emitted the terminal SSE
        // event by then.
        let result = run_agent_loop(adapter, loop_ctx, state, sink).await;

        match result {
            Ok(()) => {
                // Adapter's `render_final` already persisted the
                // accumulated response via `persist_response_if_needed`.
                // Nothing left to do here.
            }
            Err(e) => {
                let body = e.into_response();
                let _ = tx.send(Ok(bytes::Bytes::from(format!(
                    "event: error\ndata: {{\"error\":\"agent_loop_failed\",\"status\":{}}}\n\n",
                    body.status().as_u16()
                ))));
            }
        }

        let _ = tx.send(Ok(bytes::Bytes::from("data: [DONE]\n\n")));
    });

    build_sse_response(rx)
}
