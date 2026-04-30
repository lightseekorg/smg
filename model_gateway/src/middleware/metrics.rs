//! HTTP metrics collection (SMG Layer 1 metrics).
//!
//! `HttpMetricsLayer` wraps the inner service to record per-request
//! duration plus the in-flight connection count via
//! `InFlightRequestTracker`. The path is normalized to bound metric
//! cardinality.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Instant,
};

use axum::{extract::Request, response::Response};
use tower::{Layer, Service};

use crate::{
    observability::{
        inflight_tracker::InFlightRequestTracker,
        metrics::{method_to_static_str, Metrics},
    },
    routers::error::extract_error_code_from_response,
};

/// Tower Layer for HTTP metrics collection (SMG Layer 1 metrics)
#[derive(Clone)]
pub struct HttpMetricsLayer {
    tracker: Arc<InFlightRequestTracker>,
}

impl HttpMetricsLayer {
    pub fn new(tracker: Arc<InFlightRequestTracker>) -> Self {
        Self { tracker }
    }
}

impl<S> Layer<S> for HttpMetricsLayer {
    type Service = HttpMetricsMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        HttpMetricsMiddleware {
            inner,
            in_flight_request_tracker: self.tracker.clone(),
        }
    }
}

/// Tower Service for HTTP metrics collection
#[derive(Clone)]
pub struct HttpMetricsMiddleware<S> {
    inner: S,
    in_flight_request_tracker: Arc<InFlightRequestTracker>,
}

impl<S> Service<Request> for HttpMetricsMiddleware<S>
where
    S: Service<Request, Response = Response> + Send + Clone + 'static,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future =
        Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request) -> Self::Future {
        // Convert method to static string to avoid allocation
        let method = method_to_static_str(req.method().as_str());
        let path = normalize_path_for_metrics(req.uri().path());
        let start = Instant::now();

        let mut inner = self.inner.clone();
        let in_flight_request_tracker = self.in_flight_request_tracker.clone();

        Box::pin(async move {
            let guard = in_flight_request_tracker.track();
            Metrics::set_http_connections_active(in_flight_request_tracker.len());

            // Capture result before dropping guard to ensure decrement happens on error too
            let result = inner.call(req).await;

            drop(guard);
            Metrics::set_http_connections_active(in_flight_request_tracker.len());

            let response = result?;

            let duration = start.elapsed();
            Metrics::record_http_response(
                &path,
                response.status().as_u16(),
                extract_error_code_from_response(&response),
            );
            Metrics::record_http_duration(method, &path, duration);

            Ok(response)
        })
    }
}

/// Normalize path for metrics to avoid high cardinality.
/// Replaces dynamic segments (IDs, UUIDs) with `{id}` placeholder.
/// Only allocates when normalization is needed; uses single-pass with byte offsets.
pub(super) fn normalize_path_for_metrics(path: &str) -> String {
    let bytes = path.as_bytes();
    let mut segment_start = 0;
    let mut segment_idx = 0;
    let mut result: Option<String> = None;

    for (pos, &b) in bytes.iter().enumerate() {
        if b == b'/' || pos == bytes.len() - 1 {
            // Determine segment end (include last char if not a slash)
            let segment_end = if b == b'/' { pos } else { pos + 1 };
            let segment = &path[segment_start..segment_end];

            // Check segments after index 2 for dynamic IDs
            if segment_idx > 2 && !segment.is_empty() && is_dynamic_id(segment) {
                // Initialize result with everything before this segment
                let result = result.get_or_insert_with(|| {
                    let mut s = String::with_capacity(path.len());
                    s.push_str(&path[..segment_start]);
                    s
                });
                result.push_str("{id}");
            } else if let Some(ref mut r) = result {
                // Already normalizing, append this segment as-is
                r.push_str(segment);
            }

            // Add slash after segment (except at end)
            if b == b'/' {
                if let Some(ref mut r) = result {
                    r.push('/');
                }
                segment_start = pos + 1;
                segment_idx += 1;
            }
        }
    }

    result.unwrap_or_else(|| path.to_owned())
}

/// Check if segment looks like a dynamic ID (prefixed ID, UUID, or numeric).
#[inline]
fn is_dynamic_id(s: &str) -> bool {
    // Prefixed IDs: resp_xxx, chatcmpl_xxx (len > 10 with underscore)
    if s.len() > 10 && s.contains('_') {
        return true;
    }
    // UUIDs: 32+ hex chars with dashes
    if s.len() >= 32 && s.bytes().all(|b| b.is_ascii_hexdigit() || b == b'-') {
        return true;
    }
    // Numeric IDs
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_path_no_ids() {
        // Common API paths should pass through unchanged
        assert_eq!(
            normalize_path_for_metrics("/v1/chat/completions"),
            "/v1/chat/completions"
        );
        assert_eq!(
            normalize_path_for_metrics("/v1/completions"),
            "/v1/completions"
        );
        assert_eq!(normalize_path_for_metrics("/v1/models"), "/v1/models");
        assert_eq!(normalize_path_for_metrics("/health"), "/health");
    }

    #[test]
    fn test_normalize_path_with_prefixed_id() {
        // Prefixed IDs (resp_xxx, chatcmpl_xxx) should be normalized
        assert_eq!(
            normalize_path_for_metrics("/v1/responses/resp_abc123def456"),
            "/v1/responses/{id}"
        );
        assert_eq!(
            normalize_path_for_metrics("/v1/chat/completions/chatcmpl_abc123xyz"),
            "/v1/chat/completions/{id}"
        );
    }

    #[test]
    fn test_normalize_path_with_uuid() {
        assert_eq!(
            normalize_path_for_metrics("/v1/responses/550e8400-e29b-41d4-a716-446655440000"),
            "/v1/responses/{id}"
        );
    }

    #[test]
    fn test_normalize_path_with_numeric_id() {
        assert_eq!(
            normalize_path_for_metrics("/v1/workers/12345"),
            "/v1/workers/{id}"
        );
    }

    #[test]
    fn test_is_dynamic_id() {
        // Prefixed IDs
        assert!(is_dynamic_id("resp_abc123def"));
        assert!(is_dynamic_id("chatcmpl_xyz789"));
        assert!(!is_dynamic_id("short_id")); // Too short

        // UUIDs
        assert!(is_dynamic_id("550e8400-e29b-41d4-a716-446655440000"));
        assert!(is_dynamic_id("550e8400e29b41d4a716446655440000")); // No dashes

        // Numeric
        assert!(is_dynamic_id("12345"));
        assert!(!is_dynamic_id("")); // Empty

        // Regular words
        assert!(!is_dynamic_id("completions"));
        assert!(!is_dynamic_id("chat"));
    }
}
