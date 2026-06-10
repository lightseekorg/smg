//! Handler functions for /v1/responses endpoints
//!
//! # Public API
//!
//! - `route_responses()` - POST /v1/responses (main entry point)
//!
//! # Architecture
//!
//! This module provides the entry point for the /v1/responses endpoint.
//! It supports two execution modes:
//!
//! 1. **Synchronous** - Returns complete response immediately (non_streaming.rs)
//! 2. **Streaming** - Returns SSE stream with real-time events (streaming.rs)
//!
//! # Request Flow
//!
//! ```text
//! route_responses()
//!   ├─► route_responses_sync()  → non_streaming::route_responses_internal()
//!   └─► route_responses_streaming()
//!       ├─► streaming::execute_tool_loop_streaming() (MCP tools)
//!       └─► streaming::convert_chat_stream_to_responses_stream() (no MCP)
//! ```

use std::sync::Arc;

use axum::{
    http,
    response::{IntoResponse, Response},
};
use openai_protocol::responses::{ResponsesRequest, ResponsesResponse};
use tokio_util::sync::CancellationToken;
use tracing::debug;
use uuid::Uuid;

use super::{
    common::{load_conversation_history, ResponsesCallContext},
    conversions, non_streaming, streaming,
};
use crate::{
    middleware::TenantRequestMeta,
    routers::{
        error,
        grpc::common::responses::{
            ensure_mcp_connection, persist_response_if_needed, ResponsesContext,
        },
    },
};

/// Main handler for POST /v1/responses
///
/// Validates request, determines execution mode (sync/streaming), and delegates
pub(crate) async fn route_responses(
    ctx: &ResponsesContext,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    tenant_request_meta: TenantRequestMeta,
    model_id: String,
) -> Response {
    let is_background = request.background.unwrap_or(false);
    if is_background {
        return error::bad_request(
            "unsupported_parameter",
            "Background mode is not supported on the gRPC router; use the HTTP API.",
        );
    }

    // 2. Route based on execution mode
    let is_streaming = request.stream.unwrap_or(false);
    if is_streaming {
        let params = ResponsesCallContext {
            headers,
            model_id,
            response_id: None,
            tenant_request_meta,
            // Live path: a never-cancelled token (no background cooperative cancel).
            cancel: CancellationToken::new(),
        };
        route_responses_streaming(ctx, request, params).await
    } else {
        let params = ResponsesCallContext {
            headers,
            model_id,
            response_id: Some(format!("resp_{}", Uuid::now_v7())),
            tenant_request_meta,
            // Live path: a never-cancelled token (no background cooperative cancel).
            cancel: CancellationToken::new(),
        };
        route_responses_sync(ctx, request, params).await
    }
}

// ============================================================================
// Synchronous Entry Point
// ============================================================================

/// Execute synchronous responses request
async fn route_responses_sync(
    ctx: &ResponsesContext,
    request: Arc<ResponsesRequest>,
    params: ResponsesCallContext,
) -> Response {
    // Keep a handle to the original (pre-execute) request for the persist call;
    // the execute-core consumes the `Arc` it receives.
    let original_request = Arc::clone(&request);
    match non_streaming::route_responses_internal(ctx, request, params).await {
        Ok(responses_response) => {
            // Live path persists exactly as before — now from the caller, using
            // the original request (matching the prior in-function behavior).
            persist_response_if_needed(
                ctx.conversation_storage.clone(),
                ctx.conversation_item_storage.clone(),
                ctx.response_storage.clone(),
                &responses_response,
                &original_request,
                ctx.request_context.clone(),
            )
            .await;
            axum::Json(responses_response).into_response()
        }
        Err(response) => response, // Already a Response with proper status code
    }
}

/// Headless (background) non-streaming execution for the regular pipeline.
///
/// Runs the same core path as [`route_responses_sync`] but:
/// - returns a typed `Result<ResponsesResponse, Response>` (the background
///   worker maps it to a `finalize`, rather than serializing an HTTP response),
/// - does NOT persist the response row — the worker's `finalize` is the
///   authoritative terminal write under the durable id (execute-core never
///   persists),
/// - uses the caller-supplied `response_id` (the durable background id) instead
///   of minting a fresh one,
/// - threads the cooperative-cancel token into the MCP tool loop.
pub(crate) async fn execute_responses_headless(
    ctx: &ResponsesContext,
    request: Arc<ResponsesRequest>,
    tenant_request_meta: TenantRequestMeta,
    response_id: String,
    cancel: CancellationToken,
) -> Result<ResponsesResponse, Response> {
    let params = ResponsesCallContext {
        headers: None,
        model_id: request.model.clone(),
        response_id: Some(response_id),
        tenant_request_meta,
        cancel,
    };
    non_streaming::route_responses_internal(ctx, request, params).await
}

// ============================================================================
// Streaming Entry Point
// ============================================================================

/// Execute streaming responses request
async fn route_responses_streaming(
    ctx: &ResponsesContext,
    request: Arc<ResponsesRequest>,
    params: ResponsesCallContext,
) -> Response {
    // 1. Load conversation history
    let modified_request = match load_conversation_history(ctx, &request).await {
        Ok(req) => req,
        Err(response) => return response, // Already a Response with proper status code
    };

    // 2. Check MCP connection and get whether MCP tools are present
    let (has_mcp_tools, mcp_servers) = match ensure_mcp_connection(
        &ctx.mcp_orchestrator,
        &ctx.mcp_format_registry,
        request.tools.as_deref(),
    )
    .await
    {
        Ok(result) => result,
        Err(response) => return response,
    };

    if has_mcp_tools {
        debug!("MCP tools detected in streaming mode, using streaming tool loop");

        return streaming::execute_tool_loop_streaming(
            ctx,
            modified_request,
            &request,
            params,
            mcp_servers,
        );
    }

    // 3. Convert ResponsesRequest → ChatCompletionRequest
    let chat_request = match conversions::responses_to_chat(&modified_request) {
        Ok(req) => Arc::new(req),
        Err(e) => {
            return error::bad_request(
                "convert_request_failed",
                format!("Failed to convert request: {e}"),
            );
        }
    };

    // 4. Execute chat pipeline and convert streaming format (no MCP tools)
    streaming::convert_chat_stream_to_responses_stream(ctx, chat_request, params, &request).await
}
