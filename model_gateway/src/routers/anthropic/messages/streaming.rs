//! Messages API streaming support (SSE)
//!
//! This module will handle Server-Sent Events (SSE) streaming for the Messages API.
//! Implementation coming in PR #3.

use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};

use crate::{protocols::messages::CreateMessageRequest, routers::anthropic::AnthropicRouter};

/// Handle streaming Messages API request (stub for PR #3)
pub async fn handle_streaming(
    _router: &AnthropicRouter,
    _headers: Option<&HeaderMap>,
    _request: &CreateMessageRequest,
    _model_id: Option<&str>,
) -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        "Messages API streaming handler coming in PR #3",
    )
        .into_response()
}
