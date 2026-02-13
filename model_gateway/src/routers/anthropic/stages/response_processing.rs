//! Response processing stage for Anthropic router pipeline
//!
//! This stage processes the worker response:
//! - For non-streaming: Parse JSON response, extract usage, return
//! - For streaming: Forward SSE events to client
//! - Records metrics (tokens, duration, errors)
//! - Handles error responses with size limits

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use async_trait::async_trait;
use axum::{
    body::Body,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use futures::Stream;
use openai_protocol::messages::Message;
use tracing::{debug, error, info, warn};

use super::{PipelineStage, StageResult};
use crate::{
    observability::metrics::{bool_to_static_str, metrics_labels, Metrics},
    routers::{
        anthropic::{
            context::RequestContext,
            utils::{read_response_body_limited, ReadBodyResult},
        },
        error,
    },
};

/// Maximum error response body size to prevent DoS (1 MB)
const MAX_ERROR_RESPONSE_SIZE: usize = 1024 * 1024;

pub(crate) struct ResponseProcessingStage;

impl ResponseProcessingStage {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ResponseProcessingStage {
    fn default() -> Self {
        Self::new()
    }
}

use crate::core::Worker;

#[async_trait]
impl PipelineStage for ResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> StageResult {
        // Take the response from context
        let response = ctx.state.response.worker_response.take().ok_or_else(|| {
            error!("Response processing stage called without worker response");
            error::internal_error("no_response", "Internal error: no worker response")
        })?;

        // Get worker for load tracking (cloned Arc, not taken)
        let worker = ctx.state.worker.clone();

        let status = response.status();
        let is_streaming = ctx.is_streaming();
        let model_id = &ctx.input.model_id;

        // Record request metric
        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model_id,
            "messages",
            bool_to_static_str(is_streaming),
        );

        // Get start time for duration calculation
        let start_time = ctx.start_time;

        if is_streaming {
            // For streaming, pass worker to stream so load is decremented when done
            return self
                .handle_streaming_response(response, model_id, start_time, worker)
                .await;
        }

        // Non-streaming response handling
        if !status.is_success() {
            return self
                .handle_error_response(response, model_id, start_time, worker)
                .await;
        }

        // Parse successful JSON response
        self.handle_success_response(response, model_id, start_time, worker)
            .await
    }

    fn name(&self) -> &'static str {
        "response_processing"
    }
}

impl ResponseProcessingStage {
    /// Handle streaming response - pass through SSE events to client
    async fn handle_streaming_response(
        &self,
        response: reqwest::Response,
        model_id: &str,
        start_time: std::time::Instant,
        worker: Option<Arc<dyn Worker>>,
    ) -> StageResult {
        let status = response.status();

        // Handle non-success streaming responses
        if !status.is_success() {
            return self
                .handle_streaming_error(response, model_id, start_time, worker)
                .await;
        }

        debug!(
            model = %model_id,
            status = %status,
            "Starting streaming response"
        );

        // Record duration metric when streaming starts
        // Note: This records time-to-first-byte, not total stream duration
        Metrics::record_router_duration(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model_id,
            "messages",
            start_time.elapsed(),
        );

        let headers = response.headers().clone();

        // Wrap the stream with worker load tracking
        // Worker load will be decremented when the stream completes (via Drop)
        let inner_stream = response.bytes_stream();
        let load_stream = LoadTrackingStream::new(inner_stream, worker);

        // Create body from the load-tracking stream
        let body = Body::from_stream(load_stream);

        // Build response with proper SSE headers
        self.build_sse_response(status, headers, body)
    }

    /// Handle streaming error response
    async fn handle_streaming_error(
        &self,
        response: reqwest::Response,
        model_id: &str,
        start_time: std::time::Instant,
        worker: Option<Arc<dyn Worker>>,
    ) -> StageResult {
        let status = response.status();
        let content_type = response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        // If it's an SSE error stream, pass it through (with worker load tracking)
        // Case-insensitive check: backends may return mixed-case Content-Type values
        if content_type
            .to_ascii_lowercase()
            .contains("text/event-stream")
        {
            warn!(
                model = %model_id,
                status = %status,
                "Streaming error response (SSE)"
            );

            // Record duration metric for SSE error stream
            Metrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::CONNECTION_HTTP,
                model_id,
                "messages",
                start_time.elapsed(),
            );

            // Record error metric for SSE error stream
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

            // Use LoadTrackingStream::new_force_failure to ensure circuit breaker records failure
            // even if the SSE error stream completes normally
            let load_stream = LoadTrackingStream::new_force_failure(stream, worker);
            let body = Body::from_stream(load_stream);

            return self.build_sse_response(status, headers, body);
        }

        // Regular error response - handle normally
        self.handle_error_response(response, model_id, start_time, worker)
            .await
    }

    #[allow(clippy::result_large_err)]
    fn build_sse_response(
        &self,
        status: StatusCode,
        upstream_headers: HeaderMap,
        body: Body,
    ) -> StageResult {
        let mut builder = Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header(header::CONNECTION, "keep-alive");

        // Copy relevant headers from upstream, excluding ones we set ourselves
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

        let response = builder.body(body).map_err(|e| {
            error!("Failed to build streaming response: {}", e);
            error::internal_error("response_build_failed", "Failed to build response")
        })?;

        Ok(Some(response))
    }

    /// Handle successful non-streaming response
    async fn handle_success_response(
        &self,
        response: reqwest::Response,
        model_id: &str,
        start_time: std::time::Instant,
        worker: Option<Arc<dyn Worker>>,
    ) -> StageResult {
        // Parse JSON response
        let result = match response.json::<Message>().await {
            Ok(message) => {
                debug!(
                    model = %model_id,
                    message_id = %message.id,
                    "Successfully parsed response"
                );

                // Record circuit breaker success - parsing succeeded
                if let Some(ref w) = worker {
                    w.record_outcome(true);
                }

                // Record metrics
                self.record_success_metrics(model_id, &message, start_time);

                info!(
                    model = %model_id,
                    message_id = %message.id,
                    input_tokens = %message.usage.input_tokens,
                    output_tokens = %message.usage.output_tokens,
                    "Completed non-streaming request"
                );

                Ok(Some((StatusCode::OK, Json(message)).into_response()))
            }
            Err(e) => {
                error!(model = %model_id, error = %e, "Failed to parse response");

                // Record circuit breaker failure - parsing failed
                if let Some(ref w) = worker {
                    w.record_outcome(false);
                }

                // Record duration metric for failed parse
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_HTTP,
                    metrics_labels::BACKEND_EXTERNAL,
                    metrics_labels::CONNECTION_HTTP,
                    model_id,
                    "messages",
                    start_time.elapsed(),
                );

                // Record error metric
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

        // Decrement worker load after response is fully processed
        if let Some(w) = worker {
            w.decrement_load();
        }

        result
    }

    async fn handle_error_response(
        &self,
        response: reqwest::Response,
        model_id: &str,
        start_time: std::time::Instant,
        worker: Option<Arc<dyn Worker>>,
    ) -> StageResult {
        let status = response.status();

        // SECURITY: Read error response incrementally with size limit to prevent DoS
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

        // Record circuit breaker failure - backend returned error
        if let Some(ref w) = worker {
            w.record_outcome(false);
        }

        // Record duration metric
        Metrics::record_router_duration(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model_id,
            "messages",
            start_time.elapsed(),
        );

        // Record error metric
        Metrics::record_router_error(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model_id,
            "messages",
            metrics_labels::ERROR_BACKEND,
        );

        // Decrement worker load after response is fully processed
        if let Some(w) = worker {
            w.decrement_load();
        }

        // Pass through error to client with original status code
        Err((status, body).into_response())
    }

    /// Record success metrics
    fn record_success_metrics(
        &self,
        model_id: &str,
        message: &Message,
        start_time: std::time::Instant,
    ) {
        // Record duration
        Metrics::record_router_duration(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model_id,
            "messages",
            start_time.elapsed(),
        );

        // Record token usage
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
}

// ============================================================================
// Load Tracking Stream Wrapper
// ============================================================================

/// Stream wrapper that tracks worker load and circuit breaker outcome
///
/// Used for pass-through streams that need to:
/// - Decrement worker load when stream completes or is dropped
/// - Record circuit breaker outcome based on whether stream completed successfully
///
/// Note: Uses `Pin<Box<S>>` to support `!Unpin` streams like `reqwest::Response::bytes_stream()`.
struct LoadTrackingStream<S> {
    inner: Pin<Box<S>>,
    worker: Option<Arc<dyn Worker>>,
    /// Tracks whether the stream completed successfully (received None from inner stream)
    /// vs being interrupted (dropped before completion)
    completed_successfully: bool,
    /// Tracks whether we encountered an error during streaming
    encountered_error: bool,
    /// If true, always record failure regardless of stream completion
    force_failure: bool,
}

impl<S> LoadTrackingStream<S> {
    fn new(inner: S, worker: Option<Arc<dyn Worker>>) -> Self {
        Self::with_force_failure(inner, worker, false)
    }

    /// Create a stream that always records circuit breaker failure on drop
    fn new_force_failure(inner: S, worker: Option<Arc<dyn Worker>>) -> Self {
        Self::with_force_failure(inner, worker, true)
    }

    fn with_force_failure(inner: S, worker: Option<Arc<dyn Worker>>, force_failure: bool) -> Self {
        Self {
            inner: Box::pin(inner),
            worker,
            completed_successfully: false,
            encountered_error: false,
            force_failure,
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
                // Stream completed normally
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
            // Decrement worker load
            worker.decrement_load();

            // Record circuit breaker outcome based on completion state
            // force_failure streams always record failure, even if stream completed normally
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
                // Stream was interrupted (client disconnect, timeout, etc.) or had an error
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
