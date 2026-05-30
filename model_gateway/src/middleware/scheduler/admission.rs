//! `priority_admission_middleware`: the axum layer that runs the priority
//! scheduler on protected routes when it is enabled.
//!
//! Pipeline position (route_layer order): runs after tenant resolution
//! (so `RouteRequestMeta.tenant_key` is in extensions for the clamp) and
//! before the handler. On admission it inserts the permit's cancel token
//! into request extensions (so long-running handlers can `select!` against
//! it for preemption) and wraps the response body in `SchedulerGuardBody`
//! (TTFT marking + slot release).

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use axum::{
    body::Body,
    extract::State,
    http::Request,
    middleware::Next,
    response::{IntoResponse, Response},
};
use smg_auth::RequestId;
use tokio_util::sync::CancellationToken;

use super::{
    state::SchedulerState, AdmitOutcome, Class, SchedulerError, SchedulerGuardBody, PRIORITY_HEADER,
};
use crate::{
    middleware::RouteRequestMeta,
    observability::metrics::{metrics_labels, Metrics},
    tenant::TenantKey,
};

/// Monotonic source of registry keys for admitted requests. Each admission
/// gets a unique key, so the inflight registry can never be clobbered by a
/// duplicate client-supplied `x-request-id` (the collision hazard flagged
/// in earlier review). The client request id is still used for logging.
static NEXT_ADMISSION_ID: AtomicU64 = AtomicU64::new(0);

fn next_registry_id() -> RequestId {
    let n = NEXT_ADMISSION_ID.fetch_add(1, Ordering::Relaxed);
    RequestId(format!("sched-{n}"))
}

/// Resolve the effective class: parse the priority header, then clamp it
/// down to the tenant's configured `max_class` (a low-tier tenant cannot
/// self-promote by setting the header). `min` is the clamp because of the
/// `Ord` derive on `Class`.
fn effective_class(req: &Request<Body>, state: &SchedulerState) -> Class {
    let header_class = req
        .headers()
        .get(PRIORITY_HEADER)
        .and_then(|h| h.to_str().ok())
        .map(Class::parse_header)
        .unwrap_or(Class::Default);
    let tenant = req
        .extensions()
        .get::<RouteRequestMeta>()
        .map(|m| m.tenant_key().clone())
        .unwrap_or_else(|| TenantKey::new("anonymous"));
    header_class.min(state.resolver.policy(&tenant).max_class)
}

pub async fn priority_admission_middleware(
    State(state): State<Arc<SchedulerState>>,
    mut req: Request<Body>,
    next: Next,
) -> Response {
    // RPS sibling check (only set when an explicit per-second limit is
    // configured). Checked before admission so a rejected request never
    // consumes a slot. Tokens are not returned — refill is time-based.
    if let Some(bucket) = &state.rate_limiter {
        if bucket.try_acquire(1.0).is_err() {
            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
            return SchedulerError::QueueFull.into_response();
        }
    }

    let class = effective_class(&req, &state);
    let request_id = next_registry_id();

    // NOTE: client-disconnect detection during the queue wait is not yet
    // wired (axum does not surface it to a middleware pre-`next.run`), so we
    // pass a fresh token; queued waits are bounded by `queue_timeout`. A
    // real disconnect drops the response future (and the SchedulerGuardBody)
    // once admitted, releasing the slot.
    let cancel = CancellationToken::new();

    match state.scheduler.admit(class, request_id, cancel).await {
        AdmitOutcome::Admitted(permit) => {
            // Hand the handler the cancel token (for preemption select!).
            req.extensions_mut().insert(permit.cancel_token());
            let response = next.run(req).await;
            let (parts, body) = response.into_parts();
            Response::from_parts(parts, Body::new(SchedulerGuardBody::new(body, permit)))
        }
        AdmitOutcome::Rejected(reason) => SchedulerError::from(reason).into_response(),
    }
}
