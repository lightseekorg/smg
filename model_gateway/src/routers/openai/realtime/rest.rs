//! REST handlers for Realtime API token generation endpoints.
//!
//! These endpoints generate ephemeral tokens for browser-safe authentication.
//! They do NOT create sessions — sessions are created implicitly when the
//! client connects (WebSocket or WebRTC).

use std::sync::Arc;

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde_json::Value;
use tracing::{debug, error};

use crate::{
    core::worker::{RuntimeType, Worker, WorkerLoadGuard},
    routers::{
        error,
        header_utils::{extract_auth_header, should_forward_request_header},
    },
    server::AppState,
};

/// `POST /v1/realtime/client_secrets` — GA ephemeral token generation.
///
/// Generates a short-lived token for browser-safe auth. The session config
/// (model, voice, tools) is included in the request body, pre-configuring
/// the session. MCP tool definitions are injected before forwarding.
pub async fn create_client_secret(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let model = match body.pointer("/session/model").and_then(|v| v.as_str()) {
        Some(m) => m.to_string(),
        None => return error::bad_request("missing_model", "session.model is required"),
    };

    // TODO(Phase 3): Inject MCP tool definitions into body.session.tools

    proxy_realtime_rest(
        &state,
        &headers,
        &body,
        &model,
        "/v1/realtime/client_secrets",
    )
    .await
}

/// `POST /v1/realtime/sessions` — Legacy ephemeral token generation.
///
/// Kept for backwards compatibility; prefer `create_client_secret` for new integrations.
pub async fn create_session(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let model = match body.get("model").and_then(|v| v.as_str()) {
        Some(m) => m.to_string(),
        None => return error::bad_request("missing_model", "model is required"),
    };

    // TODO(Phase 3): Inject MCP tool definitions into body.tools

    proxy_realtime_rest(&state, &headers, &body, &model, "/v1/realtime/sessions").await
}

/// `POST /v1/realtime/transcription_sessions` — Legacy ephemeral token generation for transcription.
///
/// Kept for backwards compatibility; prefer `create_client_secret` for new integrations.
pub async fn create_transcription_session(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let model = match body.get("model").and_then(|v| v.as_str()) {
        Some(m) => m.to_string(),
        None => return error::bad_request("missing_model", "model is required"),
    };

    // TODO(Phase 3): Inject MCP tool definitions into config

    proxy_realtime_rest(
        &state,
        &headers,
        &body,
        &model,
        "/v1/realtime/transcription_sessions",
    )
    .await
}

/// Shared proxy logic for all realtime REST endpoints.
///
/// Handles worker selection, load tracking, auth, header forwarding,
/// upstream request, circuit breaker outcome recording, and response proxying.
async fn proxy_realtime_rest(
    state: &AppState,
    headers: &HeaderMap,
    body: &Value,
    model: &str,
    path: &str,
) -> Response {
    let worker = match select_worker(state, model) {
        Some(w) => w,
        None => {
            error!(model, path, "No available worker for realtime model");
            return StatusCode::SERVICE_UNAVAILABLE.into_response();
        }
    };

    let auth = match extract_auth_header(Some(headers), worker.api_key()) {
        Some(v) => v,
        None => return StatusCode::UNAUTHORIZED.into_response(),
    };

    // Track load for the duration of the upstream request
    let _guard = WorkerLoadGuard::new(worker.clone(), Some(headers));

    let upstream_url = format!("{}{path}", worker.url().trim_end_matches('/'));
    debug!(model, upstream_url, "Forwarding realtime REST request");

    let result = forward_post(&state.context.client, &upstream_url, &auth, headers, body)
        .send()
        .await;

    match result {
        Ok(resp) => {
            let success = resp.status().is_success();
            let response = proxy_response(resp).await;
            worker.record_outcome(success);
            response
        }
        Err(e) => {
            error!(error = %e, path, "Failed to forward realtime REST request");
            worker.record_outcome(false);
            StatusCode::BAD_GATEWAY.into_response()
        }
    }
}

/// Build a POST request to upstream, forwarding whitelisted headers
/// (x-request-id, traceparent, etc.) for trace correlation.
fn forward_post(
    client: &reqwest::Client,
    upstream_url: &str,
    auth: &http::HeaderValue,
    headers: &HeaderMap,
    body: &Value,
) -> reqwest::RequestBuilder {
    let mut req = client
        .post(upstream_url)
        .header("Authorization", auth)
        .json(body);

    for (name, value) in headers.iter() {
        if should_forward_request_header(name.as_str()) {
            req = req.header(name.clone(), value.clone());
        }
    }

    req
}

/// Select the best available worker for a realtime model.
///
/// Uses `supports_model()` which checks `models_override` (populated by lazy
/// discovery), unlike `get_by_model()` which relies on `model_index` (not
/// populated for lazily-discovered workers).
fn select_worker(state: &AppState, model: &str) -> Option<Arc<dyn Worker>> {
    state
        .context
        .worker_registry
        .get_workers_filtered(None, None, None, Some(RuntimeType::External), true)
        .into_iter()
        .filter(|w| w.supports_model(model) && w.circuit_breaker().can_execute())
        .min_by_key(|w| w.load())
}

/// Convert an upstream reqwest Response into an axum Response,
/// preserving status code and body.
pub(super) async fn proxy_response(resp: reqwest::Response) -> Response {
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
