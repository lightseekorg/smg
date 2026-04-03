//! Score pipeline stage for vLLM `/v1/score` endpoint
//!
//! vLLM exposes `/v1/score` as an HTTP REST endpoint — there is no gRPC proto
//! for it even when the worker is operating in gRPC mode.  This stage handles
//! score requests by:
//!
//! 1. Extracting the selected worker URL from `ctx.state.workers`.
//! 2. Converting the gRPC URL to an HTTP URL using the existing `http_base_url`
//!    utility (same host:port, scheme swap only).
//! 3. Forwarding the original JSON body to `POST /v1/score` via a shared
//!    `reqwest::Client`.
//! 4. Returning `Ok(Some(response))` to short-circuit the pipeline — the raw
//!    backend response is returned verbatim to the caller, preserving status,
//!    headers, and body.
//!
//! This pattern mirrors how streaming responses short-circuit the pipeline in
//! `ChatGenerateResponseProcessingStage`.

use std::sync::Arc;

use axum::{
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::Value;
use tracing::{debug, error};

use crate::{
    core::steps::worker::util::http_base_url,
    routers::{
        error,
        grpc::{
            common::stages::PipelineStage,
            context::{RequestContext, WorkerSelection},
        },
    },
};

/// Pipeline stage that forwards `/v1/score` requests to the vLLM worker via
/// HTTP regardless of whether the gateway is running in gRPC connection mode.
///
/// This stage short-circuits the pipeline by returning `Ok(Some(response))`
/// after forwarding, so no downstream stages are executed.
pub(crate) struct ScoreHttpForwardStage {
    /// Shared HTTP client (cheap to clone — Arc-backed internally)
    client: reqwest::Client,
}

impl ScoreHttpForwardStage {
    pub fn new(client: reqwest::Client) -> Self {
        Self { client }
    }
}

#[async_trait::async_trait]
impl PipelineStage for ScoreHttpForwardStage {
    fn name(&self) -> &'static str {
        "ScoreHttpForwardStage"
    }

    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Get selected worker
        let worker = match &ctx.state.workers {
            Some(WorkerSelection::Single { worker }) => Arc::clone(worker),
            Some(WorkerSelection::Dual { decode, .. }) => {
                // Score is single-worker only; for PD setups use decode worker
                Arc::clone(decode)
            }
            None => {
                error!(
                    stage = self.name(),
                    "No worker selected before ScoreHttpForwardStage"
                );
                return Err(error::internal_error(
                    "score_no_worker",
                    "No worker available for score request",
                ));
            }
        };

        // Get score request body
        let score_req = ctx.score_request_arc();

        // Derive HTTP URL from worker URL
        // http_base_url converts grpc://host:port → http://host:port
        // and is a no-op for http:// URLs.
        let base_url = http_base_url(worker.url());
        let target_url = format!("{base_url}/v1/score");

        debug!(
            stage = self.name(),
            worker_url = worker.url(),
            target_url = %target_url,
            model = %score_req.model,
            "Forwarding /v1/score request via HTTP"
        );

        // Serialize original request body
        let body_json = match serde_json::to_vec(score_req.as_ref()) {
            Ok(b) => b,
            Err(e) => {
                error!(stage = self.name(), error = %e, "Failed to serialize ScoreRequest");
                return Err(error::internal_error(
                    "score_serialize_error",
                    "Failed to serialize score request",
                ));
            }
        };

        // Forward request headers (auth passthrough)
        let mut req_builder = self
            .client
            .post(&target_url)
            .header("Content-Type", "application/json");

        // Forward authorization header if present
        if let Some(headers) = &ctx.input.headers {
            for key in &[
                "authorization",
                "x-api-key",
                "x-request-id",
                "x-correlation-id",
            ] {
                if let Some(val) = headers.get(*key) {
                    req_builder = req_builder.header(*key, val);
                }
            }
        }

        req_builder = req_builder.body(body_json);

        // Execute HTTP request
        let upstream_response = match req_builder.send().await {
            Ok(r) => r,
            Err(e) => {
                error!(
                    stage = self.name(),
                    target_url = %target_url,
                    error = %e,
                    "HTTP forward to /v1/score failed"
                );
                let status = if e.is_timeout() {
                    StatusCode::GATEWAY_TIMEOUT
                } else {
                    StatusCode::BAD_GATEWAY
                };
                return Err(error::create_error(
                    status,
                    "score_upstream_error",
                    format!("Failed to reach upstream /v1/score: {e}"),
                ));
            }
        };

        // Convert upstream response to axum Response
        let status = StatusCode::from_u16(upstream_response.status().as_u16())
            .unwrap_or(StatusCode::BAD_GATEWAY);

        // Collect response headers we want to forward
        let mut response_headers = HeaderMap::new();
        for (name, value) in upstream_response.headers() {
            // Forward content-type so callers get correct JSON MIME
            let name_str = name.as_str();
            if name_str == "content-type" || name_str == "x-request-id" {
                if let (Ok(n), Ok(v)) = (
                    HeaderName::from_bytes(name.as_str().as_bytes()),
                    HeaderValue::from_bytes(value.as_bytes()),
                ) {
                    response_headers.insert(n, v);
                }
            }
        }

        // Read body — for score responses this is always a small JSON object
        let response_bytes = match upstream_response.bytes().await {
            Ok(b) => b,
            Err(e) => {
                error!(
                    stage = self.name(),
                    error = %e,
                    "Failed to read /v1/score upstream response body"
                );
                return Err(error::internal_error(
                    "score_body_read_error",
                    "Failed to read upstream score response",
                ));
            }
        };

        // If upstream returned an error, pass it through with original status
        if !status.is_success() {
            // Try to parse as JSON for a clean error, otherwise raw text
            let error_body: Value = serde_json::from_slice(&response_bytes).unwrap_or_else(|_| {
                Value::String(String::from_utf8_lossy(&response_bytes).into_owned())
            });
            let axum_response = (status, axum::Json(error_body)).into_response();
            // Return as Err to trigger error metrics in the pipeline runner
            return Err(axum_response);
        }

        // Return proxied success response (short-circuits pipeline)
        let mut axum_response = (status, axum::body::Body::from(response_bytes)).into_response();
        // Merge forwarded headers
        axum_response.headers_mut().extend(response_headers);

        Ok(Some(axum_response))
    }
}
