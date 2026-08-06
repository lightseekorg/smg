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
};
use bytes::Bytes;
use http_body::Frame;
use tokio::{
    sync::{mpsc, oneshot},
    time::error::Elapsed,
};
use tracing::{debug, error, warn};

use super::token_bucket::TokenBucket;
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    server::AppState,
};

/// Returns an acquired token when the request is cancelled or the response body is dropped.
struct TokenPermit {
    token_bucket: Arc<TokenBucket>,
    /// Number of tokens to return.
    tokens: f64,
}

impl TokenPermit {
    fn try_acquire(token_bucket: Arc<TokenBucket>, tokens: f64) -> Result<Self, ()> {
        token_bucket.try_acquire(tokens)?;
        Ok(Self {
            token_bucket,
            tokens,
        })
    }

    async fn acquire_timeout(
        token_bucket: Arc<TokenBucket>,
        tokens: f64,
        timeout: Duration,
    ) -> Result<Self, Elapsed> {
        token_bucket.acquire_timeout(tokens, timeout).await?;
        Ok(Self {
            token_bucket,
            tokens,
        })
    }
}

impl Drop for TokenPermit {
    fn drop(&mut self) {
        debug!(
            "TokenPermit: request ended, returning {} tokens to bucket",
            self.tokens
        );
        // Use lock-free sync return - no runtime needed, guaranteed token return
        self.token_bucket.return_tokens_sync(self.tokens);
    }
}

/// A body wrapper that holds a token until the body is fully consumed or dropped.
pub struct TokenGuardBody {
    inner: Body,
    _permit: TokenPermit,
}

impl TokenGuardBody {
    fn with_permit(inner: Body, permit: TokenPermit) -> Self {
        Self {
            inner,
            _permit: permit,
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

async fn run_with_permit(next: Next, request: Request<Body>, permit: TokenPermit) -> Response {
    let (parts, body) = next.run(request).await.into_parts();
    let body = TokenGuardBody::with_permit(body, permit);
    Response::from_parts(parts, Body::new(body))
}

/// Request queue entry
pub struct QueuedRequest {
    /// Time when the request was queued
    queued_at: Instant,
    /// Channel to send the permit back when acquired
    permit_tx: oneshot::Sender<Result<TokenPermit, StatusCode>>,
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
            if let Ok(permit) = TokenPermit::try_acquire(self.token_bucket.clone(), 1.0) {
                // Got token immediately
                debug!("Queue: acquired token immediately for queued request");
                let _ = queued.permit_tx.send(Ok(permit));
            } else {
                // Need to wait for token
                let token_bucket = self.token_bucket.clone();

                // Spawn task only when we actually need to wait
                #[expect(
                    clippy::disallowed_methods,
                    reason = "fire-and-forget permit acquisition: task is bounded by remaining_timeout and communicates via oneshot; dropping the JoinHandle detaches the task but it self-terminates"
                )]
                tokio::spawn(async move {
                    if let Ok(permit) =
                        TokenPermit::acquire_timeout(token_bucket, 1.0, remaining_timeout).await
                    {
                        debug!("Queue: acquired token after waiting");
                        let _ = queued.permit_tx.send(Ok(permit));
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
    // Cluster-wide rate limiting was previously enforced via the
    // v1 `MeshSyncManager::check_global_rate_limit` path. That hook
    // is removed in this PR. Local per-node token-bucket rate
    // limiting below still applies; cluster aggregation will return
    // through the v2 `RateLimitSyncAdapter` in a follow-up PR.

    let token_bucket = match &app_state.context.rate_limiter {
        Some(bucket) => bucket.clone(),
        None => {
            // Rate limiting disabled, pass through immediately
            return next.run(request).await;
        }
    };

    // Try to acquire token immediately
    if let Ok(permit) = TokenPermit::try_acquire(token_bucket.clone(), 1.0) {
        debug!("Acquired token immediately");
        Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_ALLOWED);
        run_with_permit(next, request, permit).await
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
                        Ok(Ok(permit)) => {
                            debug!("Acquired token from queue");
                            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_ALLOWED);
                            run_with_permit(next, request, permit).await
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_body_holds_token_until_dropped() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let permit = TokenPermit::try_acquire(bucket.clone(), 1.0).unwrap();
        let body = TokenGuardBody::with_permit(Body::empty(), permit);

        assert_eq!(bucket.available_tokens(), 0.0);
        drop(body);
        assert_eq!(bucket.available_tokens(), 1.0);
    }

    #[tokio::test]
    async fn cancellation_returns_acquired_token() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let task_bucket = bucket.clone();
        let (acquired_tx, acquired_rx) = oneshot::channel();
        #[expect(
            clippy::disallowed_methods,
            reason = "Test helper: the spawned task is explicitly aborted and awaited before the test ends"
        )]
        let task = tokio::spawn(async move {
            let _permit = TokenPermit::try_acquire(task_bucket, 1.0).unwrap();
            let _ = acquired_tx.send(());
            std::future::pending::<()>().await;
        });

        acquired_rx.await.unwrap();
        assert_eq!(bucket.available_tokens(), 0.0);
        task.abort();
        let _ = task.await;
        assert_eq!(bucket.available_tokens(), 1.0);
    }

    #[tokio::test]
    async fn cancelled_queued_request_returns_acquired_token() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let (queue_tx, queue_rx) = mpsc::channel(1);
        let (permit_tx, permit_rx) = oneshot::channel();
        drop(permit_rx);

        assert!(queue_tx
            .send(QueuedRequest {
                queued_at: Instant::now(),
                permit_tx,
            })
            .await
            .is_ok());
        drop(queue_tx);

        QueueProcessor::new(bucket.clone(), queue_rx, Duration::from_secs(1))
            .run()
            .await;
        assert_eq!(bucket.available_tokens(), 1.0);
    }
}
