//! Anthropic Messages API implementation
//!
//! This module implements the Anthropic Messages API (`/v1/messages`), providing
//! support for streaming, tool use, and advanced features like extended thinking.
//!
//! ## Architecture
//!
//! - `handler.rs`: Core request handling (non-streaming and streaming)
//! - `streaming.rs`: SSE streaming support
//! - `tools.rs`: Tool use and MCP integration

pub mod handler;
pub mod streaming;
pub mod tools;

pub use handler::MessagesHandler;

/// Handle non-streaming Messages API request
pub async fn handle_non_streaming(
    headers: Option<&axum::http::HeaderMap>,
    request: &crate::protocols::messages::CreateMessageRequest,
) -> axum::response::Response {
    MessagesHandler::handle_non_streaming(headers, request).await
}

/// Handle streaming Messages API request
pub async fn handle_streaming(
    headers: Option<&axum::http::HeaderMap>,
    request: &crate::protocols::messages::CreateMessageRequest,
) -> axum::response::Response {
    MessagesHandler::handle_streaming(headers, request).await
}
