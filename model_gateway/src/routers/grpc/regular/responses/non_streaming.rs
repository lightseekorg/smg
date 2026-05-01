//! Non-streaming Regular Responses entry point.
//!
//! Drives the shared `run_agent_loop` state machine via
//! `RegularAdapter`. Surface-specific concerns (chat-completion
//! conversion, MCP merge into chat tools) live inside the adapter;
//! this module handles only hydrate / connect / loop / persist.

use std::sync::Arc;

use axum::response::Response;
use openai_protocol::responses::{ResponsesRequest, ResponsesResponse};
use uuid::Uuid;

use super::{
    agent_loop_adapter::{RegularAdapter, RegularUpstreamHandle},
    common::ResponsesCallContext,
};
use crate::routers::{
    common::{
        agent_loop::{run_agent_loop, AgentLoopContext, NoopSink},
        responses_loop_setup::ResponsesLoopSetup,
    },
    grpc::common::responses::{
        ensure_mcp_connection, persist_response_if_needed, prepare_request_history,
        ResponsesContext,
    },
};

pub(super) async fn route_responses_internal(
    ctx: &ResponsesContext,
    request: Arc<ResponsesRequest>,
    params: ResponsesCallContext,
) -> Result<ResponsesResponse, Response> {
    let original_request = (*request).clone();

    let loaded = prepare_request_history(ctx, &request).await?;

    let (_, mcp_servers) =
        ensure_mcp_connection(&ctx.mcp_orchestrator, original_request.tools.as_deref()).await?;
    let setup = ResponsesLoopSetup::from_history(loaded, mcp_servers);
    let session_request_id = params
        .response_id
        .clone()
        .unwrap_or_else(|| format!("resp_{}", Uuid::now_v7()));
    let session = setup.session(
        &ctx.mcp_orchestrator,
        &session_request_id,
        &original_request,
    );

    let ResponsesLoopSetup {
        current_request: modified_request,
        prepared,
        state,
        max_tool_calls,
        ..
    } = setup;

    let upstream_handle = RegularUpstreamHandle {
        headers: params.headers.clone(),
        model_id: params.model_id.clone(),
        response_id: params.response_id.clone(),
        tenant_request_meta: params.tenant_request_meta.clone(),
    };
    let adapter = RegularAdapter::new(ctx, upstream_handle, &modified_request, &session);

    // `loop_ctx.original_request` keeps the user-provided request
    // shape (with `previous_response_id` / `conversation` still set)
    // so `render_final` can echo those fields. `build_iteration_request`
    // rebuilds per-iteration requests from `state.upstream_input`
    // (already merged with history) and clears those resolution
    // sources, so passing the unmodified original here is safe.
    let loop_ctx = AgentLoopContext {
        prepared: &prepared,
        session: Some(&session),
        original_request: &original_request,
        max_tool_calls,
    };

    let responses_response = run_agent_loop(adapter, loop_ctx, state, NoopSink)
        .await
        .map_err(|e| e.into_response())?;

    persist_response_if_needed(
        ctx.conversation_storage.clone(),
        ctx.conversation_item_storage.clone(),
        ctx.response_storage.clone(),
        &responses_response,
        &original_request,
        ctx.request_context.clone(),
    )
    .await;

    Ok(responses_response)
}
