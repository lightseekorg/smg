//! Shared helpers for Realtime API REST proxy responses.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use tracing::error;

/// Convert an upstream reqwest Response into an axum Response,
/// preserving status code and body.
pub(crate) async fn proxy_response(resp: reqwest::Response) -> Response {
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();

    match resp.bytes().await {
        Ok(body) => (status, [(http::header::CONTENT_TYPE, content_type)], body).into_response(),
        Err(e) => {
            error!(error = %e, "Failed to read upstream response body");
            StatusCode::BAD_GATEWAY.into_response()
        }
    }
}
