//! Request handling functions for Anthropic router
//!
//! Plain async functions that replace the pipeline/stages architecture.
//! Each function performs one step of request processing:
//! 1. Worker selection
//! 2. Request building (URL + header propagation)
//! 3. Request execution (HTTP POST with load tracking)
//! 4. Response processing (parse JSON, stream SSE, handle errors)

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures::Stream;
use tracing::{debug, error, info, warn};

use super::{
    context::{RequestContext, RouterContext},
    utils::{find_best_worker_for_model, read_response_body_limited, should_propagate_header},
};
use openai_protocol::messages::{CreateMessageRequest, Message};

use crate::{
    core::{Worker, WorkerRegistry},
    observability::metrics::{bool_to_static_str, metrics_labels, Metrics},
    routers::error,
};

/// Maximum error response body size to prevent DoS (1 MB)
const MAX_ERROR_RESPONSE_SIZE: usize = 1024 * 1024;

// ============================================================================
// Composed Entry Points
// ============================================================================

/// Execute a non-streaming Messages API request end-to-end.
pub(crate) async fn execute_for_messages(
    router: &RouterContext,
    req_ctx: &RequestContext,
) -> Result<Message, Response> {
    let model_id = &req_ctx.model_id;
    let start_time = Instant::now();
    record_router_request(model_id, false);

    let worker = select_worker(&router.worker_registry, model_id)?;
    let (url, req_headers) = build_worker_request(&*worker, req_ctx.headers.as_ref());
    let response = send_worker_request(
        &router.http_client,
        &url,
        &req_headers,
        &req_ctx.request,
        router.request_timeout,
        &*worker,
    )
    .await?;

    if !response.status().is_success() {
        return Err(handle_error_response(response, model_id, start_time, &*worker).await);
    }

    parse_message_response(response, model_id, start_time, &*worker).await
}

/// Execute a streaming Messages API request, returning an SSE `Response`.
pub(crate) async fn execute_streaming(
    router: &RouterContext,
    req_ctx: &RequestContext,
) -> Response {
    let model_id = &req_ctx.model_id;
    let start_time = Instant::now();
    record_router_request(model_id, true);

    let worker = match select_worker(&router.worker_registry, model_id) {
        Ok(w) => w,
        Err(resp) => return resp,
    };
    let (url, req_headers) = build_worker_request(&*worker, req_ctx.headers.as_ref());
    let response = match send_worker_request(
        &router.http_client,
        &url,
        &req_headers,
        &req_ctx.request,
        router.request_timeout,
        &*worker,
    )
    .await
    {
        Ok(r) => r,
        Err(resp) => return resp,
    };

    build_streaming_response(response, model_id, start_time, worker).await
}

// ============================================================================
// Individual Steps
// ============================================================================

/// Select the best worker for the given model.
#[allow(clippy::result_large_err)]
pub(crate) fn select_worker(
    worker_registry: &WorkerRegistry,
    model_id: &str,
) -> Result<Arc<dyn Worker>, Response> {
    debug!(model = %model_id, "Selecting worker for request");

    match find_best_worker_for_model(worker_registry, model_id) {
        Some(w) => {
            debug!(
                model = %model_id,
                worker_url = %w.url(),
                worker_load = %w.load(),
                "Selected worker for request"
            );
            Ok(w)
        }
        None => {
            warn!(model = %model_id, "No healthy workers available for model");
            Err(error::service_unavailable(
                "no_workers",
                format!("No healthy workers available for model '{}'", model_id),
            ))
        }
    }
}

/// Build the target URL and propagated headers for a worker request.
fn build_worker_request(worker: &dyn Worker, headers: Option<&HeaderMap>) -> (String, HeaderMap) {
    let url = format!("{}/v1/messages", worker.url());
    let mut propagated = HeaderMap::new();
    if let Some(input_headers) = headers {
        for (key, value) in input_headers {
            if should_propagate_header(key.as_str()) {
                propagated.insert(key.clone(), value.clone());
            }
        }
    }
    debug!(url = %url, header_count = %propagated.len(), "Request built");
    (url, propagated)
}

/// Send the HTTP request to the worker. Increments load on send; decrements + records
/// circuit breaker failure on connection/timeout errors.
async fn send_worker_request(
    http_client: &reqwest::Client,
    url: &str,
    headers: &HeaderMap,
    request: &CreateMessageRequest,
    timeout: Duration,
    worker: &dyn Worker,
) -> Result<reqwest::Response, Response> {
    debug!(url = %url, "Sending request to worker");
    worker.increment_load();

    let mut builder = http_client.post(url).json(request).timeout(timeout);
    for (key, value) in headers {
        builder = builder.header(key, value);
    }

    match builder.send().await {
        Ok(response) => {
            debug!(url = %url, status = %response.status(), "Received response from worker");
            Ok(response)
        }
        Err(e) => {
            warn!(url = %url, error = %e, "Request to worker failed");
            worker.decrement_load();
            worker.record_outcome(false);

            if e.is_timeout() {
                Err(error::gateway_timeout(
                    "timeout",
                    format!("Request timeout: {}", e),
                ))
            } else if e.is_connect() {
                Err(error::bad_gateway(
                    "connection_failed",
                    format!("Connection failed: {}", e),
                ))
            } else {
                Err(error::bad_gateway(
                    "request_failed",
                    format!("Request failed: {}", e),
                ))
            }
        }
    }
}

/// Parse a successful non-streaming JSON response. Decrements load and records circuit breaker.
async fn parse_message_response(
    response: reqwest::Response,
    model_id: &str,
    start_time: Instant,
    worker: &dyn Worker,
) -> Result<Message, Response> {
    let result = match response.json::<Message>().await {
        Ok(message) => {
            debug!(
                model = %model_id,
                message_id = %message.id,
                "Successfully parsed response"
            );
            worker.record_outcome(true);
            record_success_metrics(model_id, &message, start_time);
            info!(
                model = %model_id,
                message_id = %message.id,
                input_tokens = %message.usage.input_tokens,
                output_tokens = %message.usage.output_tokens,
                "Completed non-streaming request"
            );
            Ok(message)
        }
        Err(e) => {
            error!(model = %model_id, error = %e, "Failed to parse response");
            worker.record_outcome(false);
            Metrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::CONNECTION_HTTP,
                model_id,
                "messages",
                start_time.elapsed(),
            );
            Metrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::CONNECTION_HTTP,
                model_id,
                "messages",
                "parse_error",
            );
            Err(error::bad_gateway(
                "parse_error",
                "Invalid response from backend",
            ))
        }
    };

    worker.decrement_load();
    result
}

/// Handle a non-success response: read body (limited), record metrics, decrement load.
async fn handle_error_response(
    response: reqwest::Response,
    model_id: &str,
    start_time: Instant,
    worker: &dyn Worker,
) -> Response {
    use super::utils::ReadBodyResult;

    let status = response.status();

    let body = match read_response_body_limited(response, MAX_ERROR_RESPONSE_SIZE).await {
        ReadBodyResult::Ok(b) if b.is_empty() => format!("Backend returned error: {}", status),
        ReadBodyResult::Ok(b) => b,
        ReadBodyResult::TooLarge => {
            warn!(
                model = %model_id,
                max_size = %MAX_ERROR_RESPONSE_SIZE,
                "Error response body too large"
            );
            format!("Backend returned error: {} (response too large)", status)
        }
        ReadBodyResult::Error(e) => {
            warn!(model = %model_id, error = %e, "Failed to read error response body");
            format!("Backend returned error: {}", status)
        }
    };

    warn!(
        model = %model_id,
        status = %status,
        body_preview = %body.chars().take(200).collect::<String>(),
        "Backend error"
    );

    worker.record_outcome(false);

    Metrics::record_router_duration(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model_id,
        "messages",
        start_time.elapsed(),
    );
    Metrics::record_router_error(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model_id,
        "messages",
        metrics_labels::ERROR_BACKEND,
    );

    worker.decrement_load();

    (status, body).into_response()
}

/// Build a streaming SSE response with load tracking.
async fn build_streaming_response(
    response: reqwest::Response,
    model_id: &str,
    start_time: Instant,
    worker: Arc<dyn Worker>,
) -> Response {
    let status = response.status();

    if !status.is_success() {
        return build_streaming_error_response(response, model_id, start_time, worker).await;
    }

    debug!(model = %model_id, status = %status, "Starting streaming response");

    Metrics::record_router_duration(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model_id,
        "messages",
        start_time.elapsed(),
    );

    let headers = response.headers().clone();
    let stream = response.bytes_stream();
    let load_stream = LoadTrackingStream::new(stream, worker);
    let body = Body::from_stream(load_stream);

    build_sse_response(status, headers, body)
}

/// Handle a non-success streaming response.
async fn build_streaming_error_response(
    response: reqwest::Response,
    model_id: &str,
    start_time: Instant,
    worker: Arc<dyn Worker>,
) -> Response {
    let status = response.status();
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    // If it's an SSE error stream, pass it through (with worker load tracking)
    if content_type
        .to_ascii_lowercase()
        .contains("text/event-stream")
    {
        warn!(model = %model_id, status = %status, "Streaming error response (SSE)");

        Metrics::record_router_duration(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model_id,
            "messages",
            start_time.elapsed(),
        );
        Metrics::record_router_error(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model_id,
            "messages",
            "streaming_error",
        );

        let headers = response.headers().clone();
        let stream = response.bytes_stream();
        let load_stream = LoadTrackingStream::new_force_failure(stream, worker);
        let body = Body::from_stream(load_stream);

        return build_sse_response(status, headers, body);
    }

    // Non-SSE error: read the body and return a proper error response
    handle_error_response(response, model_id, start_time, &*worker).await
}

// ============================================================================
// Metrics Helpers
// ============================================================================

fn record_router_request(model_id: &str, streaming: bool) {
    Metrics::record_router_request(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model_id,
        "messages",
        bool_to_static_str(streaming),
    );
}

fn record_success_metrics(model_id: &str, message: &Message, start_time: Instant) {
    Metrics::record_router_duration(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model_id,
        "messages",
        start_time.elapsed(),
    );
    Metrics::record_router_tokens(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        model_id,
        "messages",
        metrics_labels::TOKEN_INPUT,
        message.usage.input_tokens as u64,
    );
    Metrics::record_router_tokens(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        model_id,
        "messages",
        metrics_labels::TOKEN_OUTPUT,
        message.usage.output_tokens as u64,
    );
}

// ============================================================================
// SSE Response Builder
// ============================================================================

fn build_sse_response(status: StatusCode, upstream_headers: HeaderMap, body: Body) -> Response {
    let mut builder = Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive");

    for (key, value) in upstream_headers.iter() {
        let key_str = key.as_str();
        if !matches!(
            key_str,
            "content-type"
                | "cache-control"
                | "connection"
                | "transfer-encoding"
                | "content-length"
        ) {
            builder = builder.header(key, value);
        }
    }

    builder.body(body).unwrap_or_else(|e| {
        error!("Failed to build streaming response: {}", e);
        error::internal_error("response_build_failed", "Failed to build response")
    })
}

// ============================================================================
// Load Tracking Stream Wrapper
// ============================================================================

/// Stream wrapper that tracks worker load and circuit breaker outcome.
///
/// Decrements worker load when the stream completes or is dropped, and records
/// circuit breaker outcome based on whether the stream completed successfully.
struct LoadTrackingStream<S> {
    inner: Pin<Box<S>>,
    /// Worker is wrapped in `Option` so `Drop` can `.take()` it exactly once.
    /// It is always `Some` during the stream's lifetime.
    worker: Option<Arc<dyn Worker>>,
    completed_successfully: bool,
    encountered_error: bool,
    /// If true, always record failure regardless of stream completion
    force_failure: bool,
}

impl<S> LoadTrackingStream<S> {
    fn new(inner: S, worker: Arc<dyn Worker>) -> Self {
        Self {
            inner: Box::pin(inner),
            worker: Some(worker),
            completed_successfully: false,
            encountered_error: false,
            force_failure: false,
        }
    }

    fn new_force_failure(inner: S, worker: Arc<dyn Worker>) -> Self {
        Self {
            inner: Box::pin(inner),
            worker: Some(worker),
            completed_successfully: false,
            encountered_error: false,
            force_failure: true,
        }
    }
}

impl<S> Stream for LoadTrackingStream<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>>,
{
    type Item = Result<Bytes, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => Poll::Ready(Some(Ok(bytes))),
            Poll::Ready(Some(Err(e))) => {
                self.encountered_error = true;
                Poll::Ready(Some(Err(std::io::Error::other(e.to_string()))))
            }
            Poll::Ready(None) => {
                self.completed_successfully = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<S> Drop for LoadTrackingStream<S> {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.decrement_load();

            if self.force_failure {
                worker.record_outcome(false);
                debug!(
                    completed = %self.completed_successfully,
                    "LoadTrackingStream (force_failure) completed, recorded failure"
                );
            } else if self.completed_successfully && !self.encountered_error {
                worker.record_outcome(true);
                debug!("LoadTrackingStream completed successfully, recorded success");
            } else {
                worker.record_outcome(false);
                debug!(
                    completed = %self.completed_successfully,
                    error = %self.encountered_error,
                    "LoadTrackingStream interrupted or errored, recorded failure"
                );
            }
        }
    }
}
