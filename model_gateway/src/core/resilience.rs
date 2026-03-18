//! Resolved per-worker resilience configuration.
//!
//! Merges router-level defaults with per-worker overrides from `ResilienceUpdate`.
//! Created at worker construction time — immutable after that.

use std::{collections::HashSet, time::Duration};

use axum::{http::StatusCode, response::IntoResponse};
use openai_protocol::worker::ResilienceUpdate;
use tracing::debug;

use crate::{
    config::types::RetryConfig,
    core::{circuit_breaker::CircuitBreakerConfig, retry::RetryExecutor, worker::Worker},
    observability::metrics::Metrics,
};

/// Default retryable HTTP status codes.
pub const DEFAULT_RETRYABLE_STATUS_CODES: &[u16] = &[408, 429, 500, 502, 503, 504];

/// Fully resolved resilience configuration for a worker.
///
/// Created by merging `RouterConfig` defaults with `WorkerSpec.resilience` overrides.
/// Immutable after construction.
#[derive(Debug, Clone)]
pub struct ResolvedResilience {
    /// Effective retry config.
    pub retry: RetryConfig,
    /// Whether retries are enabled.
    pub retry_enabled: bool,
    /// Whether circuit breaker is enabled.
    pub circuit_breaker_enabled: bool,
    /// Set of HTTP status codes considered retryable.
    pub retryable_status_codes: HashSet<u16>,
}

impl Default for ResolvedResilience {
    fn default() -> Self {
        Self {
            retry: RetryConfig::default(),
            retry_enabled: true,
            circuit_breaker_enabled: true,
            retryable_status_codes: DEFAULT_RETRYABLE_STATUS_CODES.iter().copied().collect(),
        }
    }
}

/// Resolve per-worker resilience config by merging router defaults with worker overrides.
pub fn resolve_resilience(
    base_retry: &RetryConfig,
    base_cb: &CircuitBreakerConfig,
    base_retry_enabled: bool,
    base_cb_enabled: bool,
    overrides: &ResilienceUpdate,
) -> (ResolvedResilience, CircuitBreakerConfig) {
    // Resolve retry config
    let retry = RetryConfig {
        max_retries: overrides.max_retries.unwrap_or(base_retry.max_retries),
        initial_backoff_ms: overrides
            .initial_backoff_ms
            .unwrap_or(base_retry.initial_backoff_ms),
        max_backoff_ms: overrides
            .max_backoff_ms
            .unwrap_or(base_retry.max_backoff_ms),
        backoff_multiplier: overrides
            .backoff_multiplier
            .unwrap_or(base_retry.backoff_multiplier),
        jitter_factor: overrides.jitter_factor.unwrap_or(base_retry.jitter_factor),
    };

    // Resolve circuit breaker config
    let cb_config = CircuitBreakerConfig {
        failure_threshold: overrides
            .cb_failure_threshold
            .unwrap_or(base_cb.failure_threshold),
        success_threshold: overrides
            .cb_success_threshold
            .unwrap_or(base_cb.success_threshold),
        timeout_duration: overrides
            .cb_timeout_secs
            .map(Duration::from_secs)
            .unwrap_or(base_cb.timeout_duration),
        window_duration: overrides
            .cb_window_secs
            .map(Duration::from_secs)
            .unwrap_or(base_cb.window_duration),
    };

    // Resolve enabled flags (per-worker overrides router-level)
    let retry_enabled = overrides
        .disable_retry
        .map(|d| !d)
        .unwrap_or(base_retry_enabled);

    let cb_enabled = overrides
        .disable_circuit_breaker
        .map(|d| !d)
        .unwrap_or(base_cb_enabled);

    // Resolve retryable status codes
    let retryable_status_codes = overrides
        .retryable_status_codes
        .as_ref()
        .map(|codes| codes.iter().copied().collect())
        .unwrap_or_else(|| DEFAULT_RETRYABLE_STATUS_CODES.iter().copied().collect());

    let resolved = ResolvedResilience {
        retry,
        retry_enabled,
        circuit_breaker_enabled: cb_enabled,
        retryable_status_codes,
    };

    (resolved, cb_config)
}

/// Execute a request with the worker's retry and circuit breaker config.
///
/// This is the primary resilience API. Routers call this instead of
/// using `RetryExecutor` directly.
///
/// Circuit breaker outcomes are recorded automatically.
/// Retry decisions use the worker's `is_retryable()` predicate.
pub async fn execute_with_resilience<F, Fut>(
    worker: &(dyn Worker + '_),
    operation: F,
) -> axum::response::Response
where
    F: FnMut(u32) -> Fut + Send,
    Fut: std::future::Future<Output = axum::response::Response> + Send,
{
    let resilience = worker.resilience();
    let worker_url = worker.url();

    // Check circuit breaker before first attempt
    if resilience.circuit_breaker_enabled && !worker.circuit_breaker().can_execute() {
        debug!(
            worker_url = worker_url,
            "Circuit breaker open, rejecting request"
        );
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            format!("Circuit breaker open for worker {worker_url}"),
        )
            .into_response();
    }

    if !resilience.retry_enabled {
        // Single attempt, no retries
        let response = execute_single(worker, operation).await;
        if resilience.circuit_breaker_enabled {
            let success = !worker.is_retryable(&response);
            worker.circuit_breaker().record_outcome(success);
        }
        return response;
    }

    // Retry loop — delegate to RetryExecutor with worker-aware hooks
    RetryExecutor::execute_response_with_retry(
        &resilience.retry,
        operation,
        |res, _attempt| worker.is_retryable(res),
        |delay, attempt| {
            Metrics::record_worker_retry_backoff(attempt, delay);
            debug!(
                worker_url = worker_url,
                attempt = attempt,
                delay_ms = delay.as_millis() as u64,
                "Worker retry backoff"
            );
        },
        || {
            debug!(worker_url = worker_url, "Worker retries exhausted");
        },
    )
    .await
}

/// Execute a single attempt of an operation (no retry).
async fn execute_single<F, Fut>(
    _worker: &(dyn Worker + '_),
    mut operation: F,
) -> axum::response::Response
where
    F: FnMut(u32) -> Fut + Send,
    Fut: std::future::Future<Output = axum::response::Response> + Send,
{
    operation(0).await
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    };

    use axum::{http::StatusCode, response::IntoResponse};

    use super::*;
    use crate::core::worker_builder::BasicWorkerBuilder;

    #[tokio::test]
    async fn test_execute_with_resilience_success() {
        let worker = BasicWorkerBuilder::new("http://test:8080").build();
        let response = execute_with_resilience(&worker, |_attempt| async {
            (StatusCode::OK, "ok").into_response()
        })
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_execute_with_resilience_circuit_open() {
        let worker = BasicWorkerBuilder::new("http://test:8080").build();
        worker.circuit_breaker().force_open();
        let response = execute_with_resilience(&worker, |_attempt| async {
            (StatusCode::OK, "ok").into_response()
        })
        .await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn test_execute_with_resilience_retries_disabled() {
        let resolved = ResolvedResilience {
            retry_enabled: false,
            ..Default::default()
        };
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .resilience(resolved)
            .build();

        let call_count = Arc::new(AtomicU32::new(0));
        let cc = call_count.clone();
        let response = execute_with_resilience(&worker, move |_attempt| {
            cc.fetch_add(1, Ordering::Relaxed);
            async { (StatusCode::SERVICE_UNAVAILABLE, "fail").into_response() }
        })
        .await;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(call_count.load(Ordering::Relaxed), 1); // No retries
    }

    #[tokio::test]
    async fn test_execute_with_resilience_retries_on_retryable_status() {
        let resolved = ResolvedResilience {
            retry: RetryConfig {
                max_retries: 3,
                initial_backoff_ms: 1,
                max_backoff_ms: 2,
                backoff_multiplier: 1.0,
                jitter_factor: 0.0,
            },
            ..Default::default()
        };
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .resilience(resolved)
            .build();

        let call_count = Arc::new(AtomicU32::new(0));
        let cc = call_count.clone();
        let response = execute_with_resilience(&worker, move |_attempt| {
            let count = cc.fetch_add(1, Ordering::Relaxed);
            async move {
                if count < 2 {
                    (StatusCode::SERVICE_UNAVAILABLE, "fail").into_response()
                } else {
                    (StatusCode::OK, "ok").into_response()
                }
            }
        })
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(call_count.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn test_execute_with_resilience_cb_disabled_ignores_open() {
        let resolved = ResolvedResilience {
            circuit_breaker_enabled: false,
            ..Default::default()
        };
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .resilience(resolved)
            .build();
        worker.circuit_breaker().force_open();

        // CB is disabled — request should go through even though CB is open
        let response = execute_with_resilience(&worker, |_attempt| async {
            (StatusCode::OK, "ok").into_response()
        })
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn test_default_resilience() {
        let resolved = ResolvedResilience::default();
        assert!(resolved.retry_enabled);
        assert!(resolved.circuit_breaker_enabled);
        assert_eq!(resolved.retryable_status_codes.len(), 6);
        assert!(resolved.retryable_status_codes.contains(&429));
        assert!(resolved.retryable_status_codes.contains(&503));
    }

    #[test]
    fn test_resolve_with_no_overrides() {
        let base_retry = RetryConfig::default();
        let base_cb = CircuitBreakerConfig::default();
        let overrides = ResilienceUpdate::default();

        let (resolved, cb_config) =
            resolve_resilience(&base_retry, &base_cb, true, true, &overrides);

        assert_eq!(resolved.retry.max_retries, base_retry.max_retries);
        assert_eq!(cb_config.failure_threshold, base_cb.failure_threshold);
        assert!(resolved.retry_enabled);
        assert!(resolved.circuit_breaker_enabled);
    }

    #[test]
    fn test_resolve_with_retry_overrides() {
        let base_retry = RetryConfig::default();
        let base_cb = CircuitBreakerConfig::default();
        let overrides = ResilienceUpdate {
            max_retries: Some(10),
            initial_backoff_ms: Some(200),
            ..Default::default()
        };

        let (resolved, _) = resolve_resilience(&base_retry, &base_cb, true, true, &overrides);

        assert_eq!(resolved.retry.max_retries, 10);
        assert_eq!(resolved.retry.initial_backoff_ms, 200);
        // Non-overridden fields keep base values
        assert_eq!(resolved.retry.max_backoff_ms, base_retry.max_backoff_ms);
    }

    #[test]
    fn test_resolve_disable_retry() {
        let base_retry = RetryConfig::default();
        let base_cb = CircuitBreakerConfig::default();
        let overrides = ResilienceUpdate {
            disable_retry: Some(true),
            ..Default::default()
        };

        let (resolved, _) = resolve_resilience(&base_retry, &base_cb, true, true, &overrides);

        assert!(!resolved.retry_enabled);
    }

    #[test]
    fn test_resolve_worker_enables_when_router_disables() {
        let base_retry = RetryConfig::default();
        let base_cb = CircuitBreakerConfig::default();
        let overrides = ResilienceUpdate {
            disable_retry: Some(false),
            ..Default::default()
        };

        // Router has retries disabled, but worker explicitly enables
        let (resolved, _) = resolve_resilience(&base_retry, &base_cb, false, true, &overrides);

        assert!(resolved.retry_enabled);
    }

    #[test]
    fn test_resolve_custom_retryable_codes() {
        let base_retry = RetryConfig::default();
        let base_cb = CircuitBreakerConfig::default();
        let overrides = ResilienceUpdate {
            retryable_status_codes: Some(vec![502, 503]),
            ..Default::default()
        };

        let (resolved, _) = resolve_resilience(&base_retry, &base_cb, true, true, &overrides);

        assert_eq!(resolved.retryable_status_codes.len(), 2);
        assert!(resolved.retryable_status_codes.contains(&502));
        assert!(resolved.retryable_status_codes.contains(&503));
        assert!(!resolved.retryable_status_codes.contains(&429));
    }

    #[test]
    fn test_resolve_cb_overrides() {
        let base_retry = RetryConfig::default();
        let base_cb = CircuitBreakerConfig::default();
        let overrides = ResilienceUpdate {
            cb_failure_threshold: Some(10),
            cb_timeout_secs: Some(60),
            ..Default::default()
        };

        let (_, cb_config) = resolve_resilience(&base_retry, &base_cb, true, true, &overrides);

        assert_eq!(cb_config.failure_threshold, 10);
        assert_eq!(cb_config.timeout_duration, Duration::from_secs(60));
        // Non-overridden
        assert_eq!(cb_config.success_threshold, base_cb.success_threshold);
    }
}
