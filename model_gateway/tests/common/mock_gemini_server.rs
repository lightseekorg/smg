//! Mock Gemini API server for testing the Interactions router.

#![allow(dead_code, clippy::allow_attributes)]

use std::{net::SocketAddr, sync::Arc};

use axum::{
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use tokio::net::TcpListener;

/// Mock Gemini API server for testing
pub struct MockGeminiServer {
    addr: SocketAddr,
    handle: tokio::task::JoinHandle<()>,
}

impl Drop for MockGeminiServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

#[derive(Clone)]
struct MockServerState {
    require_auth: bool,
    expected_api_key: Option<String>,
}

impl MockGeminiServer {
    /// Create and start a new mock Gemini server
    pub async fn new() -> Self {
        Self::new_with_auth(None).await
    }

    /// Create and start a new mock Gemini server with optional auth requirement
    #[expect(
        clippy::unwrap_used,
        clippy::disallowed_methods,
        reason = "test helper - panicking on failure is intentional"
    )]
    pub async fn new_with_auth(expected_api_key: Option<String>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let state = Arc::new(MockServerState {
            require_auth: expected_api_key.is_some(),
            expected_api_key,
        });

        let app = Router::new()
            .route("/v1beta/interactions", post(mock_interactions))
            .route("/v1beta/models", get(mock_models))
            .with_state(state);

        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        Self { addr, handle }
    }

    /// Get the base URL for this mock server
    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }
}

/// Validate auth header if auth is required.
/// Accepts both `x-goog-api-key` (production) and `Authorization: Bearer` (test
/// environment where localhost URLs resolve to the Generic provider).
fn check_auth(state: &MockServerState, req: &Request<Body>) -> Option<Response> {
    if !state.require_auth {
        return None;
    }

    // Try x-goog-api-key first, then fall back to Authorization: Bearer
    let api_key = req
        .headers()
        .get("x-goog-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .or_else(|| {
            req.headers()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .map(|s| {
                    s.split_once(' ')
                        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
                        .map_or_else(|| s.to_string(), |(_, token)| token.to_string())
                })
        });

    let auth_ok = match (&state.expected_api_key, api_key) {
        (Some(expected), Some(got)) => &got == expected,
        (None, Some(_)) => true,
        _ => false,
    };

    if !auth_ok {
        let body = json!({
            "error": {
                "code": 401,
                "message": "API key not valid. Please pass a valid API key.",
                "status": "UNAUTHENTICATED"
            }
        });
        let mut response = Response::new(Body::from(body.to_string()));
        *response.status_mut() = StatusCode::UNAUTHORIZED;
        return Some(response);
    }
    None
}

/// Mock interactions endpoint
async fn mock_interactions(
    State(state): State<Arc<MockServerState>>,
    req: Request<Body>,
) -> Response {
    if let Some(err_resp) = check_auth(&state, &req) {
        return err_resp;
    }

    let (_, body) = req.into_parts();
    let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };

    let request: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(req) => req,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };

    let model = request
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let agent = request
        .get("agent")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let store = request
        .get("store")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    // Match real Gemini API behavior: only include `id` when store=true.
    // When store=false, the Gemini API omits the id field.
    let mut response = json!({
        "object": "interaction",
        "status": "completed",
        "created": "2025-01-01T00:00:00Z",
        "role": "model",
        "outputs": [{
            "type": "text",
            "text": "Hello! I'm a mock Gemini response."
        }],
        "usage": {
            "total_input_tokens": 10,
            "total_output_tokens": 15,
            "total_tokens": 25
        },
        "store": store
    });

    if store {
        response["id"] = json!("interaction_mock_123");
    }

    if !model.is_empty() {
        response["model"] = json!(model);
    }
    if !agent.is_empty() {
        response["agent"] = json!(agent);
    }

    Json(response).into_response()
}

/// Mock models endpoint (Gemini `/v1beta/models` format)
async fn mock_models(State(state): State<Arc<MockServerState>>, req: Request<Body>) -> Response {
    if let Some(err_resp) = check_auth(&state, &req) {
        return err_resp;
    }

    let response = json!({
        "models": [
            {
                "name": "models/gemini-2.5-flash",
                "displayName": "Gemini 2.5 Flash",
                "supportedGenerationMethods": ["generateContent"]
            },
            {
                "name": "models/gemini-2.5-pro",
                "displayName": "Gemini 2.5 Pro",
                "supportedGenerationMethods": ["generateContent"]
            },
            {
                "name": "models/deep-research-pro-preview-12-2025",
                "displayName": "Deep Research Pro",
                "supportedGenerationMethods": ["generateContent"]
            }
        ]
    });

    Json(response).into_response()
}
