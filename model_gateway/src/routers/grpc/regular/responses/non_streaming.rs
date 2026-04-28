//! Non-streaming Regular Responses entry point.
//!
//! Drives the shared `run_agent_loop` state machine via
//! `RegularAdapter`. Surface-specific concerns (chat-completion
//! conversion, MCP merge into chat tools) live inside the adapter;
//! this module handles only hydrate / connect / loop / persist.

use std::sync::Arc;

use axum::response::Response;
use openai_protocol::responses::{ResponsesRequest, ResponsesResponse};
use smg_mcp::McpToolSession;
use uuid::Uuid;

use super::{
    agent_loop_adapter::{RegularAdapter, RegularUpstreamHandle},
    common::{load_conversation_history, ResponsesCallContext},
};
use crate::routers::{
    common::agent_loop::{
        run_agent_loop, AgentLoopContext, AgentLoopState, NoopSink, PreparedLoopInput,
    },
    grpc::common::responses::{
        ensure_mcp_connection, persist_response_if_needed, ResponsesContext,
    },
};

pub(super) async fn route_responses_internal(
    ctx: &ResponsesContext,
    request: Arc<ResponsesRequest>,
    params: ResponsesCallContext,
) -> Result<ResponsesResponse, Response> {
    let original_request = (*request).clone();

    let loaded = load_conversation_history(ctx, &request).await?;
    let modified_request = loaded.request;
    let emitted_mcp_server_labels = loaded.existing_mcp_list_tools_labels;
    let control_items = loaded.control_items;

    let (_has_mcp_tools, mcp_servers) =
        ensure_mcp_connection(&ctx.mcp_orchestrator, request.tools.as_deref()).await?;

    // Always create the session — empty `mcp_servers` yields a session
    // with no exposed tools, which the loop driver tolerates (every
    // model turn returns zero `pending_gateway_tool_calls`).
    let session_request_id = params
        .response_id
        .clone()
        .unwrap_or_else(|| format!("resp_{}", Uuid::now_v7()));
    let mut session = McpToolSession::new(&ctx.mcp_orchestrator, mcp_servers, &session_request_id);
    if let Some(tools) = modified_request.tools.as_deref() {
        session.configure_approval_policy(tools);
    }

    let upstream_handle = RegularUpstreamHandle {
        headers: params.headers.clone(),
        model_id: params.model_id.clone(),
        response_id: params.response_id.clone(),
        tenant_request_meta: params.tenant_request_meta.clone(),
    };
    let max_tool_calls = modified_request.max_tool_calls.map(|n| n as usize);
    let adapter = RegularAdapter::new(ctx, upstream_handle, &modified_request, &session);

    let prepared = PreparedLoopInput::new(modified_request.input.clone(), control_items);
    let state = AgentLoopState::new(prepared.upstream_input.clone(), emitted_mcp_server_labels);
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
