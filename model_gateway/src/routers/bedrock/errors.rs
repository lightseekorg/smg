use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, StatusCode},
    response::Response,
};
use serde::Deserialize;

use crate::routers::error;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockErrorPayload {
    #[serde(default)]
    message: Option<String>,
}

pub(crate) fn map_upstream_error(status: StatusCode, body: &[u8]) -> Response {
    let parsed = serde_json::from_slice::<BedrockErrorPayload>(body).ok();
    let message = parsed
        .and_then(|p| p.message)
        .unwrap_or_else(|| String::from_utf8_lossy(body).to_string());
    let code = status
        .canonical_reason()
        .unwrap_or("bedrock_error")
        .to_lowercase()
        .replace(' ', "_");

    error::create_error(status, code, message)
}

pub(crate) fn map_send_error(err: impl std::fmt::Display) -> Response {
    error::service_unavailable("bedrock_upstream_unreachable", format!("{err}"))
}

pub(crate) fn map_bad_mapping_error(err: impl std::fmt::Display) -> Response {
    error::bad_gateway("bedrock_response_mapping_failed", format!("{err}"))
}

pub(crate) fn map_signing_error(err: impl std::fmt::Display) -> Response {
    error::internal_error("bedrock_signing_error", format!("{err}"))
}

pub(crate) fn unsupported_endpoint() -> Response {
    Response::builder()
        .status(StatusCode::NOT_IMPLEMENTED)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            "{\"error\":{\"code\":\"not_supported\",\"message\":\"Endpoint not yet supported for bedrock router\"}}",
        ))
        .unwrap_or_else(|_| error::not_implemented("not_supported", "Unsupported endpoint"))
}
