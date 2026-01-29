//! Messages API streaming support (SSE)
//!
//! This module handles Server-Sent Events (SSE) streaming for the Messages API.
//! Implementation coming in PR #3.

use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};

use crate::protocols::messages::CreateMessageRequest;

/// Create SSE stream for Messages API (placeholder)
///
/// # Arguments
///
/// * `headers` - Request headers
/// * `request` - Messages API request
///
/// # Returns
///
/// SSE stream (currently not implemented)
pub async fn create_sse_stream(
    _headers: Option<&HeaderMap>,
    _request: &CreateMessageRequest,
) -> Response {
    (StatusCode::NOT_IMPLEMENTED, "SSE streaming coming in PR #3").into_response()
}
