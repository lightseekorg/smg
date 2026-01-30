//! Anthropic Messages API implementation
//!
//! This module implements the Anthropic Messages API (`/v1/messages`), providing
//! support for streaming, tool use, and advanced features like extended thinking.
//!
//! ## Architecture
//!
//! - `non_streaming.rs`: Non-streaming request handling with validation, worker selection, and forwarding
//! - `streaming.rs`: SSE streaming support (PR #3)
//! - `tools.rs`: Tool use and MCP integration (PR #4)

pub mod non_streaming;
pub mod streaming;
pub mod tools;

use axum::http::HeaderMap;

use crate::{protocols::messages::CreateMessageRequest, routers::anthropic::AnthropicRouter};

/// Handle non-streaming Messages API request
pub async fn handle_non_streaming(
    router: &AnthropicRouter,
    headers: Option<&HeaderMap>,
    request: &CreateMessageRequest,
    model_id: Option<&str>,
) -> axum::response::Response {
    non_streaming::handle_non_streaming(router, headers, request, model_id).await
}

/// Handle streaming Messages API request
pub async fn handle_streaming(
    router: &AnthropicRouter,
    headers: Option<&HeaderMap>,
    request: &CreateMessageRequest,
    model_id: Option<&str>,
) -> axum::response::Response {
    streaming::handle_streaming(router, headers, request, model_id).await
}
