//! Non-streaming Harmony Responses API entry point.
//!
//! All control flow lives in `routers/common/agent_loop`; this module
//! is the thin surface adapter wiring: hydrate history, connect MCP,
//! build a `HarmonyAdapter`, run the shared agent loop, and persist.

use axum::response::Response;
use openai_protocol::responses::{ResponsesRequest, ResponsesResponse};
use smg_mcp::McpToolSession;
use uuid::Uuid;

use super::agent_loop_adapter::HarmonyAdapter;
use crate::{
    middleware::TenantRequestMeta,
    routers::{
        common::{
            agent_loop::{run_agent_loop, AgentLoopContext, NoopSink},
            responses_loop_setup::ResponsesLoopSetup,
        },
        grpc::common::responses::{
            ensure_mcp_connection, persist_response_if_needed, prepare_request_history,
            ResponsesContext,
        },
    },
};

pub(crate) async fn serve_harmony_responses(
    ctx: &ResponsesContext,
    request: ResponsesRequest,
    tenant_request_meta: TenantRequestMeta,
) -> Result<ResponsesResponse, Response> {
    let original_request = request.clone();

    let loaded = prepare_request_history(ctx, &request).await?;

    let (_, mcp_servers) =
        ensure_mcp_connection(&ctx.mcp_orchestrator, original_request.tools.as_deref()).await?;
    let ResponsesLoopSetup {
        current_request,
        prepared,
        state,
        max_tool_calls,
        mcp_servers,
    } = ResponsesLoopSetup::from_history(loaded, mcp_servers);

    // Always create a session so the loop driver can speak to MCP
    // unconditionally; an empty `mcp_servers` list yields a session
    // with no exposed tools, which the loop tolerates (every model
    // turn produces zero `pending_gateway_tool_calls`).
    let session_request_id = format!("resp_{}", Uuid::now_v7());
    let mut session = McpToolSession::new(&ctx.mcp_orchestrator, mcp_servers, &session_request_id);
    if let Some(tools) = original_request.tools.as_deref() {
        session.configure_approval_policy(tools);
    }

    let adapter = HarmonyAdapter::new(ctx, tenant_request_meta, &current_request, &session);

    // `loop_ctx.original_request` carries the *user-provided* request
    // shape (still holding `previous_response_id` / `conversation`)
    // so the adapter's `render_final` can echo those fields onto the
    // response. `build_iteration_request` rebuilds each iteration's
    // request from `state.upstream_input` (already merged with
    // history) and explicitly clears those resolution-source fields,
    // so passing the unmodified original here is safe.
    let loop_ctx = AgentLoopContext {
        prepared: &prepared,
        session: Some(&session),
        original_request: &original_request,
        max_tool_calls,
    };

    let response = run_agent_loop(adapter, loop_ctx, state, NoopSink)
        .await
        .map_err(|e| e.into_response())?;

    persist_response_if_needed(
        ctx.conversation_storage.clone(),
        ctx.conversation_item_storage.clone(),
        ctx.response_storage.clone(),
        &response,
        &original_request,
        ctx.request_context.clone(),
    )
    .await;

    Ok(response)
}
