//! Background collector tasks that publish state to [`WatchRegistry`].
//!
//! - **Event-driven**: workers and models — listen on `WorkerRegistry` broadcast
//! - **Polled**: loads, metrics, rate_limits — read from `AppContext` on intervals

use std::{sync::Arc, time::Duration};

use metrics_exporter_prometheus::PrometheusHandle;
use serde_json::{json, Value};
use tokio::{
    sync::broadcast::{
        error::{RecvError, TryRecvError},
        Receiver,
    },
    task::JoinHandle,
};
use tracing::{debug, warn};

use super::{registry::WatchRegistry, types::Topic};
use crate::{app_context::AppContext, worker::event::WorkerEvent};

/// Configuration for collector intervals.
pub struct CollectorConfig {
    /// Interval for the loads collector.
    pub loads_interval: Duration,
    /// Interval for the rate-limits collector.
    pub rate_limits_interval: Duration,
    /// Interval for the metrics (Prometheus) collector.
    pub metrics_interval: Duration,
    /// Checkpoint interval for the worker collector.
    /// Catches health changes that bypass the broadcast (e.g.,
    /// `set_status()` called directly by FFI bindings, the registry
    /// teardown path, or the mesh subscriber).
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
///
/// Covers 5 of 7 topics. Cluster and mesh topics are deferred — they require
/// `MeshServerHandler` access (cross-crate) and change infrequently.
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
/// Also polls on a checkpoint interval to catch health changes that
/// bypass the broadcast (e.g., `set_status()` called directly by FFI
/// bindings, the registry teardown path, or the mesh subscriber).
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
                            // Coalesce a burst of events into a single rebuild, and
                            // publish models only if any of them changed membership.
                            let models_changed = drain_pending(&event, &mut rx);
                            publish_workers(&context, &registry);
                            if models_changed {
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

/// Membership changes (add/remove/replace) affect the model list; status
/// changes do not.
fn is_membership_change(event: &WorkerEvent) -> bool {
    matches!(
        event,
        WorkerEvent::Registered { .. } | WorkerEvent::Removed { .. } | WorkerEvent::Replaced { .. }
    )
}

/// Drain every event already queued behind `first` so a burst collapses into a
/// single rebuild, returning whether any of them (including `first`) changed
/// membership. A drain-time lag conservatively forces a model rebuild.
fn drain_pending(first: &WorkerEvent, rx: &mut Receiver<WorkerEvent>) -> bool {
    let mut models_changed = is_membership_change(first);
    loop {
        match rx.try_recv() {
            Ok(event) => models_changed |= is_membership_change(&event),
            Err(TryRecvError::Empty | TryRecvError::Closed) => break,
            Err(TryRecvError::Lagged(n)) => {
                warn!("worker collector lagged by {n} events while draining");
                models_changed = true;
            }
        }
    }
    models_changed
}

/// Collect and publish `topic` only when a WS client is subscribed, so the
/// collector never builds a snapshot nobody will read.
fn publish_if_subscribed(registry: &WatchRegistry, topic: Topic, collect: impl FnOnce() -> Value) {
    if registry.has_receivers(topic) {
        registry.publish(topic, collect());
    }
}

fn publish_workers(context: &AppContext, registry: &WatchRegistry) {
    publish_if_subscribed(registry, Topic::Workers, || collect_workers(context));
}

fn publish_models(context: &AppContext, registry: &WatchRegistry) {
    publish_if_subscribed(registry, Topic::Models, || collect_models(context));
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
            publish_if_subscribed(&registry, topic, || collect_fn(&context));
            debug!("{name} collector: tick");
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
            publish_if_subscribed(
                &registry,
                Topic::Metrics,
                || json!({ "raw": prometheus_handle.render() }),
            );
            debug!("metrics collector: tick");
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
                "circuit_breaker": w.circuit_breaker_state().to_string(),
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
    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};
    use tokio::sync::broadcast;

    use super::*;
    use crate::worker::{registry::WorkerId, BasicWorkerBuilder, Worker, WorkerType};

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    fn dummy_worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .build(),
        )
    }

    fn status_event(url: &str) -> WorkerEvent {
        WorkerEvent::StatusChanged {
            worker_id: WorkerId::from_string(url.to_string()),
            worker: dummy_worker(url),
            old_status: WorkerStatus::Ready,
            new_status: WorkerStatus::NotReady,
        }
    }

    fn registered_event(url: &str) -> WorkerEvent {
        WorkerEvent::Registered {
            worker_id: WorkerId::from_string(url.to_string()),
            worker: dummy_worker(url),
        }
    }

    #[test]
    fn default_config_has_sensible_intervals() {
        let config = CollectorConfig::default();
        assert_eq!(config.loads_interval, Duration::from_secs(3));
        assert_eq!(config.rate_limits_interval, Duration::from_secs(5));
        assert_eq!(config.metrics_interval, Duration::from_secs(3));
        assert_eq!(config.worker_checkpoint_interval, Duration::from_secs(3));
    }

    #[test]
    fn publish_skipped_without_receivers() {
        let registry = WatchRegistry::new();
        let mut collected = 0;
        publish_if_subscribed(&registry, Topic::Workers, || {
            collected += 1;
            json!({ "workers": [] })
        });
        assert_eq!(
            collected, 0,
            "must not collect when no client is subscribed"
        );
        // A fresh subscriber sees the initial None: nothing was published.
        assert!(registry.subscribe(Topic::Workers).borrow().is_none());
    }

    #[test]
    fn publish_runs_with_receiver() {
        let registry = WatchRegistry::new();
        let rx = registry.subscribe(Topic::Workers);
        let mut collected = 0;
        publish_if_subscribed(&registry, Topic::Workers, || {
            collected += 1;
            json!({ "workers": [] })
        });
        assert_eq!(collected, 1, "must collect when a client is subscribed");
        assert!(rx.borrow().is_some());
    }

    #[test]
    fn drain_pending_coalesces_burst_and_flags_membership() {
        let (tx, mut rx) = broadcast::channel(16);
        let first = status_event("http://w1");
        tx.send(status_event("http://w1")).unwrap();
        tx.send(registered_event("http://w2")).unwrap();
        tx.send(status_event("http://w1")).unwrap();

        let models_changed = drain_pending(&first, &mut rx);

        assert!(
            models_changed,
            "a Registered event in the burst must flag a models rebuild"
        );
        assert!(
            matches!(rx.try_recv(), Err(TryRecvError::Empty)),
            "the whole burst must be drained into a single rebuild"
        );
    }

    #[test]
    fn drain_pending_status_only_keeps_models_untouched() {
        let (tx, mut rx) = broadcast::channel(16);
        let first = status_event("http://w1");
        tx.send(status_event("http://w1")).unwrap();
        tx.send(status_event("http://w1")).unwrap();

        let models_changed = drain_pending(&first, &mut rx);

        assert!(
            !models_changed,
            "a status-only burst must not rebuild models"
        );
        assert!(matches!(rx.try_recv(), Err(TryRecvError::Empty)));
    }
}
