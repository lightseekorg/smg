//! HTTP metrics collection (SMG Layer 1 metrics).
//!
//! `HttpMetricsLayer` wraps the inner service to record per-request
//! duration plus the in-flight connection count via
//! `InFlightRequestTracker`. The path label is the matched axum route
//! template (or `"other"` when unmatched) to bound metric cardinality.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Instant,
};

use axum::{
    extract::{MatchedPath, Request},
    response::Response,
};
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
        let method = method_to_static_str(req.method().as_str());
        let path = matched_path_label(req.extensions()).to_owned();
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

/// Bounded path label for HTTP metrics: the matched axum route template, or
/// `"other"` when no route matched. Labeling by raw request path would let
/// attacker-controlled URIs create unbounded distinct labels.
pub(super) fn matched_path_label(extensions: &http::Extensions) -> &str {
    extensions
        .get::<MatchedPath>()
        .map_or("other", MatchedPath::as_str)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use axum::{body::Body, http::Request, routing::get, Router};
    use tower::{ServiceBuilder, ServiceExt};

    use super::*;
    use crate::observability::metrics::interner_size;

    #[test]
    fn matched_path_label_defaults_to_other_when_absent() {
        // No routing has run, so there is no MatchedPath extension.
        let extensions = http::Extensions::new();
        assert_eq!(matched_path_label(&extensions), "other");
    }

    /// Drive `request_uri` through a router that has one dynamic route and a
    /// fallback, returning the label `matched_path_label` observes at a
    /// `Router::layer`-applied middleware. The layer is the outermost wrap (it
    /// also covers the fallback) so both the matched and unmatched branches are
    /// exercised at the same layer the production metrics middleware uses.
    async fn label_at_layer(request_uri: &str) -> String {
        let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let sink = captured.clone();

        let app = Router::new()
            .route("/v1/responses/{response_id}", get(|| async { "ok" }))
            .fallback(|| async { "fallback" })
            .layer(
                ServiceBuilder::new().map_request(move |req: Request<Body>| {
                    *sink.lock().unwrap() = Some(matched_path_label(req.extensions()).to_owned());
                    req
                }),
            );

        let response = app
            .oneshot(
                Request::builder()
                    .uri(request_uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(response.status().is_success());

        let label = captured.lock().unwrap().clone();
        label.expect("label-capturing layer ran")
    }

    #[tokio::test]
    async fn matched_route_uses_template_label() {
        // A matched dynamic route is labeled by its template, never the raw id.
        assert_eq!(
            label_at_layer("/v1/responses/resp_abc123").await,
            "/v1/responses/{response_id}"
        );
    }

    #[tokio::test]
    async fn unmatched_path_collapses_to_other() {
        // An unmatched path must collapse to "other", not echo the raw URI.
        assert_eq!(label_at_layer("/totally/unregistered/aaaa").await, "other");
    }

    #[tokio::test]
    async fn distinct_ids_on_matched_route_do_not_grow_interner() {
        use crate::observability::inflight_tracker::InFlightRequestTracker;

        // Drive the real `HttpMetricsLayer`. Every request matches the dynamic
        // route `/v1/responses/{response_id}`, so each distinct id must record
        // the bounded template label and leave the never-evicted interner flat.
        let app = Router::new()
            .route("/v1/responses/{response_id}", get(|| async { "ok" }))
            .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()));

        let send = |uri: String| {
            let app = app.clone();
            async move {
                let response = app
                    .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
                    .await
                    .unwrap();
                assert!(response.status().is_success());
            }
        };

        // Warm up so the template label and the empty error_code are interned.
        send("/v1/responses/resp_warmup".to_owned()).await;
        let size_before = interner_size();

        const ITERS: usize = 1000;
        for i in 0..ITERS {
            send(format!("/v1/responses/resp_{i}")).await;
        }

        // Slack tolerates strings unrelated parallel tests may intern; an
        // unbounded label would instead grow the interner by ~ITERS.
        let growth = interner_size().saturating_sub(size_before);
        assert!(
            growth < 100,
            "interner grew by {growth} for {ITERS} distinct request ids"
        );
    }
}
