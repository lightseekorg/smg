//! Non-streaming response handling for OpenAI-compatible responses
//!
//! This module handles non-streaming Responses API requests with MCP tool support.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::Value;
use smg_mcp::McpToolSession;
use tracing::warn;

use super::{
    mcp::{execute_tool_loop, prepare_mcp_tools_as_functions},
    utils::{patch_response_with_request_metadata, restore_original_tools},
};
use crate::routers::{
    error,
    header_utils::{apply_provider_headers, extract_auth_header},
    mcp_utils::ensure_request_mcp_client,
    openai::context::{PayloadState, RequestContext},
    persistence_utils::persist_conversation_items,
};

/// Handle a non-streaming responses request
pub async fn handle_non_streaming_response(mut ctx: RequestContext) -> Response {
    let payload_state = match ctx.state.payload.take() {
        Some(ps) => ps,
        None => {
            return error::internal_error("internal_error", "Payload not prepared");
        }
    };

    let PayloadState {
        json: mut payload,
        url,
        previous_response_id,
    } = payload_state;

    let original_body = match ctx.responses_request() {
        Some(r) => r,
        None => {
            return error::internal_error("internal_error", "Expected responses request");
        }
    };
    let worker = match ctx.worker() {
        Some(w) => w.clone(),
        None => {
            return error::internal_error("internal_error", "Worker not selected");
        }
    };
    let mcp_orchestrator = match ctx.components.mcp_orchestrator() {
        Some(m) => m,
        None => {
            return error::internal_error("internal_error", "MCP orchestrator required");
        }
    };

    // Check for MCP tools and create session if needed
    let mcp_servers = if let Some(tools) = original_body.tools.as_deref() {
        ensure_request_mcp_client(mcp_orchestrator, tools).await
    } else {
        None
    };

    let mut response_json: Value;

    if let Some(mcp_servers) = mcp_servers {
        let session_request_id = original_body
            .request_id
            .clone()
            .unwrap_or_else(|| format!("req_{}", uuid::Uuid::new_v4()));
        let session = McpToolSession::new(mcp_orchestrator, mcp_servers, &session_request_id);
        prepare_mcp_tools_as_functions(&mut payload, &session);

        match execute_tool_loop(
            ctx.components.client(),
            &url,
            ctx.headers(),
            payload,
            original_body,
            &session,
        )
        .await
        {
            Ok(resp) => {
                worker.circuit_breaker().record_success();
                response_json = resp;
            }
            Err(err) => {
                worker.circuit_breaker().record_failure();
                return error::internal_error("upstream_error", err);
            }
        }
    } else {
        let mut request_builder = ctx.components.client().post(&url).json(&payload);
        let auth_header = extract_auth_header(ctx.headers(), worker.api_key());
        request_builder = apply_provider_headers(request_builder, &url, auth_header.as_ref());

        let response = match request_builder.send().await {
            Ok(r) => r,
            Err(e) => {
                worker.circuit_breaker().record_failure();
                tracing::error!(
                    url = %url,
                    error = %e,
                    "Failed to forward request to OpenAI"
                );
                return error::bad_gateway(
                    "upstream_error",
                    format!("Failed to forward request to OpenAI: {}", e),
                );
            }
        };

        if !response.status().is_success() {
            worker.circuit_breaker().record_failure();
            let status = StatusCode::from_u16(response.status().as_u16())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let body = response.text().await.unwrap_or_default();
            return (status, body).into_response();
        }

        response_json = match response.json::<Value>().await {
            Ok(r) => r,
            Err(e) => {
                worker.circuit_breaker().record_failure();
                return error::internal_error(
                    "parse_error",
                    format!("Failed to parse upstream response: {}", e),
                );
            }
        };

        worker.circuit_breaker().record_success();
    }

    restore_original_tools(&mut response_json, original_body);
    patch_response_with_request_metadata(
        &mut response_json,
        original_body,
        previous_response_id.as_deref(),
    );

    if let (Some(conv_storage), Some(item_storage), Some(resp_storage)) = (
        ctx.components.conversation_storage(),
        ctx.components.conversation_item_storage(),
        ctx.components.response_storage(),
    ) {
        if let Err(err) = persist_conversation_items(
            conv_storage.clone(),
            item_storage.clone(),
            resp_storage.clone(),
            &response_json,
            original_body,
        )
        .await
        {
            warn!("Failed to persist conversation items: {}", err);
        }
    } else {
        warn!("Storage not configured, skipping conversation persistence");
    }

    (StatusCode::OK, Json(response_json)).into_response()
}
