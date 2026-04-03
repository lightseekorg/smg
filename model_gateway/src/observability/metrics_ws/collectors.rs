//! Background collector tasks that publish state to [`WatchRegistry`].
//!
//! - **Event-driven**: workers and models — listen on `WorkerRegistry` broadcast
//! - **Polled**: loads, metrics, rate_limits — read from `AppContext` on intervals

use std::{sync::Arc, time::Duration};

use metrics_exporter_prometheus::PrometheusHandle;
use serde_json::{json, Value};
use tokio::{sync::broadcast::error::RecvError, task::JoinHandle};
use tracing::{debug, warn};

use super::{registry::WatchRegistry, types::Topic};
use crate::{app_context::AppContext, core::worker_event::WorkerEvent};

/// Configuration for collector intervals.
pub struct CollectorConfig {
    /// Interval for the loads collector.
    pub loads_interval: Duration,
    /// Interval for the rate-limits collector.
    pub rate_limits_interval: Duration,
    /// Interval for the metrics (Prometheus) collector.
    pub metrics_interval: Duration,
    /// Checkpoint interval for the worker collector.
    /// Catches health changes that bypass the broadcast (e.g., `set_healthy()`
    /// called directly by the health checker or circuit breaker).
    pub worker_checkpoint_interval: Duration,
}

impl Default for CollectorConfig {
    fn default() -> Self {
        Self {
            loads_interval: Duration::from_secs(3),
            rate_limits_interval: Duration::from_secs(5),
            metrics_interval: Duration::from_secs(3),
            worker_checkpoint_interval: Duration::from_secs(3),
        }
    }
}

/// Start all collector tasks. Returns join handles (caller keeps them alive).
pub fn start_collectors(
    context: Arc<AppContext>,
    registry: Arc<WatchRegistry>,
    config: CollectorConfig,
    prometheus_handle: PrometheusHandle,
) -> Vec<JoinHandle<()>> {
    vec![
        // Event-driven: workers + models (single task, shared broadcast)
        // Also polls on checkpoint interval to catch health changes that bypass broadcast.
        spawn_worker_collector(
            context.clone(),
            registry.clone(),
            config.worker_checkpoint_interval,
        ),
        // Polled
        spawn_interval_collector(
            "loads",
            context.clone(),
            registry.clone(),
            Topic::Loads,
            config.loads_interval,
            collect_loads,
        ),
        spawn_interval_collector(
            "rate_limits",
            context.clone(),
            registry.clone(),
            Topic::RateLimits,
            config.rate_limits_interval,
            collect_rate_limits,
        ),
        spawn_metrics_collector(registry.clone(), config.metrics_interval, prometheus_handle),
    ]
}

// ── Event-driven collector ──────────────────────────────────────────────

/// Listens on WorkerRegistry broadcast for instant push on register/remove.
/// Also polls on a checkpoint interval to catch health changes that bypass
/// the broadcast (e.g., `set_healthy()` called directly by health checker
/// or circuit breaker).
fn spawn_worker_collector(
    context: Arc<AppContext>,
    registry: Arc<WatchRegistry>,
    checkpoint_interval: Duration,
) -> JoinHandle<()> {
    let mut rx = context.worker_registry.subscribe_events();

    #[expect(
        clippy::disallowed_methods,
        reason = "collector runs for the lifetime of the server"
    )]
    tokio::spawn(async move {
        // Publish initial snapshot
        publish_workers(&context, &registry);
        publish_models(&context, &registry);

        let mut checkpoint = tokio::time::interval(checkpoint_interval);
        checkpoint.tick().await; // skip first immediate tick

        loop {
            tokio::select! {
                event = rx.recv() => {
                    match event {
                        Ok(event) => {
                            debug!("worker event: {event:?}");
                            publish_workers(&context, &registry);
                            // Only publish models on membership changes, not health
                            if matches!(event, WorkerEvent::Registered { .. } | WorkerEvent::Removed { .. }) {
                                publish_models(&context, &registry);
                            }
                        }
                        Err(RecvError::Lagged(n)) => {
                            warn!("worker collector lagged by {n} events, publishing full snapshot");
                            publish_workers(&context, &registry);
                            publish_models(&context, &registry);
                        }
                        Err(RecvError::Closed) => {
                            debug!("worker broadcast closed, collector stopping");
                            break;
                        }
                    }
                }
                _ = checkpoint.tick() => {
                    // Catch changes that bypass the broadcast channel
                    publish_workers(&context, &registry);
                    publish_models(&context, &registry);
                }
            }
        }
    })
}

fn publish_workers(context: &AppContext, registry: &WatchRegistry) {
    registry.publish(Topic::Workers, collect_workers(context));
}

fn publish_models(context: &AppContext, registry: &WatchRegistry) {
    registry.publish(Topic::Models, collect_models(context));
}

// ── Polled collectors ───────────────────────────────────────────────────

fn spawn_interval_collector(
    name: &'static str,
    context: Arc<AppContext>,
    registry: Arc<WatchRegistry>,
    topic: Topic,
    interval: Duration,
    collect_fn: fn(&AppContext) -> Value,
) -> JoinHandle<()> {
    #[expect(
        clippy::disallowed_methods,
        reason = "collector runs for the lifetime of the server"
    )]
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        loop {
            ticker.tick().await;
            registry.publish(topic, collect_fn(&context));
            debug!("{name} collector: published");
        }
    })
}

fn spawn_metrics_collector(
    registry: Arc<WatchRegistry>,
    interval: Duration,
    prometheus_handle: PrometheusHandle,
) -> JoinHandle<()> {
    #[expect(
        clippy::disallowed_methods,
        reason = "collector runs for the lifetime of the server"
    )]
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        loop {
            ticker.tick().await;
            let raw = prometheus_handle.render();
            registry.publish(Topic::Metrics, json!({ "raw": raw }));
            debug!("metrics collector: published");
        }
    })
}

// ── Collection functions ────────────────────────────────────────────────

fn collect_workers(context: &AppContext) -> Value {
    let workers = context.worker_registry.get_all();
    let mut healthy = 0usize;
    let worker_data: Vec<Value> = workers
        .iter()
        .map(|w| {
            if w.is_healthy() {
                healthy += 1;
            }
            json!({
                "url": w.url(),
                "model_id": w.model_id(),
                "worker_type": w.worker_type().to_string(),
                "connection_mode": w.connection_mode().to_string(),
                "is_healthy": w.is_healthy(),
                "load": w.load(),
                "processed_requests": w.processed_requests(),
                "circuit_breaker": w.circuit_breaker().state().to_string(),
            })
        })
        .collect();
    let total = worker_data.len();
    json!({
        "workers": worker_data,
        "total": total,
        "healthy": healthy,
        "unhealthy": total - healthy,
    })
}

fn collect_loads(context: &AppContext) -> Value {
    let workers = context.worker_registry.get_all();
    let mut total_load = 0usize;
    let worker_data: Vec<Value> = workers
        .iter()
        .map(|w| {
            let load = w.load();
            total_load += load;
            json!({
                "url": w.url(),
                "load": load,
                "is_healthy": w.is_healthy(),
            })
        })
        .collect();
    json!({
        "workers": worker_data,
        "total_load": total_load,
    })
}

fn collect_rate_limits(context: &AppContext) -> Value {
    match &context.rate_limiter {
        Some(limiter) => json!({
            "enabled": true,
            "available_tokens": limiter.available_tokens(),
        }),
        None => json!({ "enabled": false }),
    }
}

fn collect_models(context: &AppContext) -> Value {
    let models = context.worker_registry.get_models();
    let total = models.len();
    json!({
        "models": models,
        "total": total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_sensible_intervals() {
        let config = CollectorConfig::default();
        assert_eq!(config.loads_interval, Duration::from_secs(3));
        assert_eq!(config.rate_limits_interval, Duration::from_secs(5));
        assert_eq!(config.metrics_interval, Duration::from_secs(3));
        assert_eq!(config.worker_checkpoint_interval, Duration::from_secs(3));
    }
}
