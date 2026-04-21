//! Per-request concurrency limiting via a token bucket, with optional
//! queuing for backpressure.
//!
//! `ConcurrencyLimiter` wires a bounded `mpsc` channel that
//! `concurrency_limit_middleware` uses to enqueue requests when the
//! bucket is empty; `QueueProcessor` drains that channel and hands tokens
//! back to waiters. `TokenGuardBody` wraps the response body so the token
//! is only released after the entire stream has been delivered.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use http_body::Frame;
use serde_json::json;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, warn};

use super::token_bucket::TokenBucket;
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    server::AppState,
};

/// A body wrapper that holds a token and returns it when the body is fully consumed or dropped.
/// This ensures that for streaming responses, the token is only returned after the entire
/// stream has been sent to the client.
pub struct TokenGuardBody {
    inner: Body,
    /// The token bucket to return tokens to. Uses Option so we can take() on drop.
    token_bucket: Option<Arc<TokenBucket>>,
    /// Number of tokens to return.
    tokens: f64,
}

impl TokenGuardBody {
    /// Create a new TokenGuardBody that will return tokens when dropped.
    pub fn new(inner: Body, token_bucket: Arc<TokenBucket>, tokens: f64) -> Self {
        Self {
            inner,
            token_bucket: Some(token_bucket),
            tokens,
        }
    }
}

impl Drop for TokenGuardBody {
    fn drop(&mut self) {
        if let Some(bucket) = self.token_bucket.take() {
            debug!(
                "TokenGuardBody: stream ended, returning {} tokens to bucket",
                self.tokens
            );
            // Use lock-free sync return - no runtime needed, guaranteed token return
            bucket.return_tokens_sync(self.tokens);
        }
    }
}

impl http_body::Body for TokenGuardBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        // SAFETY: We never move the inner body, and Body is Unpin
        // (it's a type alias for UnsyncBoxBody which is Unpin)
        let this = self.get_mut();
        Pin::new(&mut this.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

/// Request queue entry
pub struct QueuedRequest {
    /// Time when the request was queued
    queued_at: Instant,
    /// Channel to send the permit back when acquired
    permit_tx: oneshot::Sender<Result<(), StatusCode>>,
}

/// Queue processor that handles queued requests
pub struct QueueProcessor {
    token_bucket: Arc<TokenBucket>,
    queue_rx: mpsc::Receiver<QueuedRequest>,
    queue_timeout: Duration,
}

impl QueueProcessor {
    pub fn new(
        token_bucket: Arc<TokenBucket>,
        queue_rx: mpsc::Receiver<QueuedRequest>,
        queue_timeout: Duration,
    ) -> Self {
        Self {
            token_bucket,
            queue_rx,
            queue_timeout,
        }
    }

    pub async fn run(mut self) {
        debug!("Starting concurrency queue processor");

        // Process requests in a single task to reduce overhead
        while let Some(queued) = self.queue_rx.recv().await {
            // Check timeout immediately
            let elapsed = queued.queued_at.elapsed();
            if elapsed >= self.queue_timeout {
                warn!("Request already timed out in queue");
                let _ = queued.permit_tx.send(Err(StatusCode::REQUEST_TIMEOUT));
                continue;
            }

            let remaining_timeout = self.queue_timeout - elapsed;

            // Try to acquire token for this request
            if self.token_bucket.try_acquire(1.0).is_ok() {
                // Got token immediately
                debug!("Queue: acquired token immediately for queued request");
                let _ = queued.permit_tx.send(Ok(()));
            } else {
                // Need to wait for token
                let token_bucket = self.token_bucket.clone();

                // Spawn task only when we actually need to wait
                #[expect(
                    clippy::disallowed_methods,
                    reason = "fire-and-forget permit acquisition: task is bounded by remaining_timeout and communicates via oneshot; dropping the JoinHandle detaches the task but it self-terminates"
                )]
                tokio::spawn(async move {
                    if token_bucket
                        .acquire_timeout(1.0, remaining_timeout)
                        .await
                        .is_ok()
                    {
                        debug!("Queue: acquired token after waiting");
                        let _ = queued.permit_tx.send(Ok(()));
                    } else {
                        warn!("Queue: request timed out waiting for token");
                        let _ = queued.permit_tx.send(Err(StatusCode::REQUEST_TIMEOUT));
                    }
                });
            }
        }

        warn!("Concurrency queue processor shutting down");
    }
}

/// State for the concurrency limiter
pub struct ConcurrencyLimiter {
    pub queue_tx: Option<mpsc::Sender<QueuedRequest>>,
}

impl ConcurrencyLimiter {
    /// Create new concurrency limiter with optional queue
    pub fn new(
        token_bucket: Option<Arc<TokenBucket>>,
        queue_size: usize,
        queue_timeout: Duration,
    ) -> (Self, Option<QueueProcessor>) {
        match (token_bucket, queue_size) {
            (None, _) => (Self { queue_tx: None }, None),
            (Some(bucket), size) if size > 0 => {
                let (queue_tx, queue_rx) = mpsc::channel(size);
                let processor = QueueProcessor::new(bucket, queue_rx, queue_timeout);
                (
                    Self {
                        queue_tx: Some(queue_tx),
                    },
                    Some(processor),
                )
            }
            (Some(_), _) => (Self { queue_tx: None }, None),
        }
    }
}

/// Middleware function for concurrency limiting with optional queuing
pub async fn concurrency_limit_middleware(
    State(app_state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    // Check mesh global rate limit first if mesh is enabled
    // If mesh is not enabled, this check is skipped and local rate limiting is used
    if let Some(mesh_handler) = &app_state.mesh_handler {
        let (is_exceeded, current_count, limit) =
            mesh_handler.sync_manager.check_global_rate_limit();
        if is_exceeded {
            debug!(
                "Global rate limit exceeded: {}/{} req/s",
                current_count, limit
            );
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(json!({
                    "error": "Rate limit exceeded",
                    "current_count": current_count,
                    "limit": limit
                })),
            )
                .into_response();
        }
    }

    let token_bucket = match &app_state.context.rate_limiter {
        Some(bucket) => bucket.clone(),
        None => {
            // Rate limiting disabled, pass through immediately
            return next.run(request).await;
        }
    };

    // Try to acquire token immediately
    if token_bucket.try_acquire(1.0).is_ok() {
        debug!("Acquired token immediately");
        Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_ALLOWED);
        let response = next.run(request).await;

        // Wrap the response body with TokenGuardBody to return token when stream ends
        // This ensures that for streaming responses, the token is only returned
        // after the entire stream has been sent to the client.
        let (parts, body) = response.into_parts();
        let guarded_body = TokenGuardBody::new(body, token_bucket, 1.0);
        Response::from_parts(parts, Body::new(guarded_body))
    } else {
        // No tokens available, try to queue if enabled
        if let Some(queue_tx) = &app_state.concurrency_queue_tx {
            debug!("No tokens available, attempting to queue request");

            // Create a channel for the token response
            let (permit_tx, permit_rx) = oneshot::channel();

            let queued = QueuedRequest {
                queued_at: Instant::now(),
                permit_tx,
            };

            // Try to send to queue
            match queue_tx.try_send(queued) {
                Ok(()) => {
                    // Wait for token from queue processor
                    match permit_rx.await {
                        Ok(Ok(())) => {
                            debug!("Acquired token from queue");
                            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_ALLOWED);
                            let response = next.run(request).await;

                            // Wrap the response body with TokenGuardBody to return token when stream ends
                            let (parts, body) = response.into_parts();
                            let guarded_body = TokenGuardBody::new(body, token_bucket, 1.0);
                            Response::from_parts(parts, Body::new(guarded_body))
                        }
                        Ok(Err(status)) => {
                            warn!("Queue returned error status: {}", status);
                            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
                            status.into_response()
                        }
                        Err(_) => {
                            error!("Queue response channel closed");
                            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
                            StatusCode::INTERNAL_SERVER_ERROR.into_response()
                        }
                    }
                }
                Err(_) => {
                    warn!("Request queue is full, returning 429");
                    Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
                    StatusCode::TOO_MANY_REQUESTS.into_response()
                }
            }
        } else {
            warn!("No tokens available and queuing is disabled, returning 429");
            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
            StatusCode::TOO_MANY_REQUESTS.into_response()
        }
    }
}
