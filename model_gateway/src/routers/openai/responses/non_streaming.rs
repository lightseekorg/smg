//! Non-streaming response handling for OpenAI-compatible responses.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::to_value;
use tracing::warn;
use uuid::Uuid;

use super::agent_loop_adapter::{OpenAiNonStreamingAdapter, OpenAiUpstreamHandle};
use crate::routers::{
    common::{
        agent_loop::{run_agent_loop, AgentLoopContext, NoopSink},
        header_utils::extract_forwardable_request_headers,
        mcp_utils::ensure_mcp_connection,
        persistence_utils::persist_conversation_items,
        responses_loop_setup::ResponsesLoopSetup,
    },
    error,
    openai::context::{RequestContext, ResponsesPayloadState},
};

/// Handle a non-streaming responses request.
pub async fn handle_non_streaming_response(mut ctx: RequestContext) -> Response {
    let payload_state = match ctx.state.payload.take() {
        Some(ps) => ps,
        None => {
            return error::internal_error("internal_error", "Payload not prepared");
        }
    };
    let ResponsesPayloadState {
        existing_mcp_list_tools_labels,
        current_request,
        prepared,
    } = match ctx.take_responses_payload() {
        Some(state) => state,
        None => {
            return error::internal_error("internal_error", "Responses payload not prepared");
        }
    };

    let original_request = match ctx.responses_request() {
        Some(r) => r.clone(),
        None => {
            return error::internal_error("internal_error", "Expected responses request");
        }
    };
    let (worker, provider) = match ctx.state.worker.as_ref() {
        Some(selection) => (selection.worker.clone(), selection.provider.clone()),
        None => {
            return error::internal_error("internal_error", "Worker not selected");
        }
    };
    let mcp_orchestrator = match ctx.components.mcp_orchestrator() {
        Some(m) => m.clone(),
        None => {
            return error::internal_error("internal_error", "MCP orchestrator required");
        }
    };

    let (_, mcp_servers) =
        match ensure_mcp_connection(&mcp_orchestrator, original_request.tools.as_deref()).await {
            Ok(result) => result,
            Err(response) => return response,
        };

    let setup = ResponsesLoopSetup::new(
        current_request,
        prepared,
        existing_mcp_list_tools_labels,
        mcp_servers,
    );

    let session_request_id = original_request
        .request_id
        .clone()
        .unwrap_or_else(|| format!("resp_{}", Uuid::now_v7()));
    let forwarded_headers = extract_forwardable_request_headers(ctx.headers());
    let session = setup.session_with_headers(
        &mcp_orchestrator,
        &session_request_id,
        forwarded_headers,
        &original_request,
    );

    let ResponsesLoopSetup {
        prepared,
        state,
        max_tool_calls,
        ..
    } = setup;

    let loop_ctx = AgentLoopContext {
        prepared: &prepared,
        session: Some(&session),
        original_request: &original_request,
        max_tool_calls,
    };
    let adapter = OpenAiNonStreamingAdapter::new(
        &original_request,
        OpenAiUpstreamHandle {
            client: ctx.components.client().clone(),
            url: payload_state.url,
            headers: ctx.headers().cloned(),
            api_key: worker.api_key().cloned(),
            provider,
            base_payload: payload_state.json,
        },
    );

    let response = match run_agent_loop(adapter, loop_ctx, state, NoopSink).await {
        Ok(response) => response,
        Err(err) => {
            let response = err.into_response();
            worker.record_outcome(response.status().as_u16());
            return response;
        }
    };

    if let (Some(conv_storage), Some(item_storage), Some(resp_storage)) = (
        ctx.components.conversation_storage(),
        ctx.components.conversation_item_storage(),
        ctx.components.response_storage(),
    ) {
        let response_json = match to_value(&response) {
            Ok(value) => value,
            Err(err) => {
                return error::internal_error(
                    "serialize_error",
                    format!("Failed to serialize responses payload for persistence: {err}"),
                );
            }
        };
        if let Err(err) = persist_conversation_items(
            conv_storage.clone(),
            item_storage.clone(),
            resp_storage.clone(),
            &response_json,
            &original_request,
            ctx.storage_request_context.clone(),
        )
        .await
        {
            warn!("Failed to persist conversation items: {}", err);
        }
    } else {
        warn!("Storage not configured, skipping conversation persistence");
    }

    worker.record_outcome(StatusCode::OK.as_u16());
    (StatusCode::OK, Json(response)).into_response()
}
