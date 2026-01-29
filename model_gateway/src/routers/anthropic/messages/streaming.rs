//! Messages API streaming support (SSE)

use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};

use crate::protocols::messages::CreateMessageRequest;

pub async fn create_sse_stream(
    _headers: Option<&HeaderMap>,
    _request: &CreateMessageRequest,
) -> Response {
    (StatusCode::NOT_IMPLEMENTED, "SSE streaming coming in PR #3").into_response()
}
