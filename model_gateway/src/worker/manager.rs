//! Worker Management Module
//!
//! Provides worker lifecycle operations and fan-out request utilities.

use std::{
    collections::{HashMap, HashSet},
    future::Future,
    pin::Pin,
    sync::Arc,
    time::Duration,
};

use axum::response::{IntoResponse, Response};
use futures::{
    future,
    stream::{self, FuturesUnordered, StreamExt},
};
use http::StatusCode;
use openai_protocol::worker::{
    FlushCacheResult, HealthCheckConfig, WorkerLoadInfo, WorkerLoadsResult, WorkerStatus,
};
use tokio::{
    sync::{broadcast, Notify},
    task::JoinHandle,
};
use tracing::{debug, error, info, warn};

use crate::{
    observability::metrics::{metrics_labels, Metrics},
    worker::{
        event::WorkerEvent,
        metrics_aggregator::{self, MetricPack},
        monitor::WorkerMonitor,
        registry::{WorkerDescriptor, WorkerId},
        worker::WorkerTypeExt,
        ConnectionMode, Worker, WorkerRegistry, WorkerType,
    },
    workflow::{Job, JobQueue},
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CONCURRENT: usize = 32;
const MAX_CONCURRENT_HEALTH_PROBES: usize = 128;

/// Result of a fan-out request to a single worker
struct WorkerResponse {
    url: String,
    result: Result<reqwest::Response, reqwest::Error>,
}

/// Fan out requests to workers in parallel
async fn fan_out(
    workers: &[Arc<dyn Worker>],
    client: &reqwest::Client,
    endpoint: &str,
    method: reqwest::Method,
) -> Vec<WorkerResponse> {
    let futures: Vec<_> = workers
        .iter()
        .map(|worker| {
            let client = client.clone();
            let url = worker.url().to_string();
            let full_url = format!("{url}/{endpoint}");
            let api_key = worker.api_key().cloned();
            let method = method.clone();

            async move {
                let mut req = client.request(method, &full_url).timeout(REQUEST_TIMEOUT);
                if let Some(key) = api_key {
                    req = req.bearer_auth(key);
                }
                WorkerResponse {
                    url,
                    result: req.send().await,
                }
            }
        })
        .collect();

    stream::iter(futures)
        .buffer_unordered(MAX_CONCURRENT)
        .collect()
        .await
}

pub enum EngineMetricsResult {
    Ok(String),
    Err(String),
}

impl IntoResponse for EngineMetricsResult {
    fn into_response(self) -> Response {
        match self {
            Self::Ok(text) => (StatusCode::OK, text).into_response(),
            Self::Err(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response(),
        }
    }
}

/// Lifecycle coordinator for the worker fleet.
///
/// Owns the background health check loop, applies the state machine to
/// probe outcomes, and triggers removal of `Failed` workers when
/// `--remove-unhealthy-workers` is set. Subscribes to `WorkerRegistry`
/// events to keep its internal schedule in sync with registrations,
/// removals, and replacements.
///
/// The static fan-out helpers (`get_worker_urls`, `flush_cache_all`,
/// `get_all_worker_loads`, `get_engine_metrics`) are operational commands
/// that don't depend on lifecycle state and remain associated functions.
pub struct WorkerManager {
    handle: Option<JoinHandle<()>>,
    shutdown_notify: Arc<Notify>,
}

impl std::fmt::Debug for WorkerManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerManager").finish()
    }
}

/// Configuration for the WorkerManager health check loop.
#[derive(Debug, Clone)]
pub struct WorkerManagerConfig {
    /// Default check interval used when a worker has no override.
    pub default_check_interval_secs: u64,
    /// If true, submit `Job::RemoveWorker` for workers that reach Failed.
    pub remove_unhealthy: bool,
}

impl WorkerManager {
    /// Create and start the WorkerManager background loop.
    ///
    /// Spawns a single task that:
    ///   - Subscribes to `WorkerRegistry` events to maintain a per-worker
    ///     deadline schedule.
    ///   - Probes due workers via `Worker::check_health_async()`.
    ///   - Applies the state machine to probe outcomes and calls
    ///     `WorkerRegistry::transition_status()` to publish StatusChanged.
    ///   - Submits `Job::RemoveWorker` for Failed workers when removal is
    ///     enabled.
    pub fn start(
        registry: Arc<WorkerRegistry>,
        config: WorkerManagerConfig,
        job_queue: Option<Arc<JobQueue>>,
    ) -> Self {
        let shutdown_notify = Arc::new(Notify::new());
        let shutdown_clone = shutdown_notify.clone();

        // Subscribe BEFORE snapshotting the registry. Any registration that
        // lands after this line either (a) is already in the snapshot below
        // because it happened synchronously on this thread, or (b) arrives
        // as a Registered event in the broadcast buffer and is idempotently
        // applied by the event loop. The "a or b" dichotomy is what makes
        // startup deterministic regardless of task scheduling.
        let events_rx = registry.subscribe_events();

        // Run the bootstrap reconcile synchronously on the caller's thread
        // so the initial schedule is captured deterministically — not
        // whenever the spawned task happens to be scheduled. A worker
        // registered between WorkerManager::start() returning and the task
        // running (e.g. the mesh replay loop in server.rs, which runs
        // synchronously right after start()) would otherwise race the
        // task's own reconcile call.
        let mut next_check: HashMap<WorkerId, tokio::time::Instant> = HashMap::new();
        reconcile_from_registry(&registry, &mut next_check, &config);

        let job_queue = if config.remove_unhealthy {
            job_queue
        } else {
            None
        };

        #[expect(
            clippy::disallowed_methods,
            reason = "WorkerManager loop runs for the lifetime of the registry; handle is stored and abort() runs on drop"
        )]
        let handle = tokio::spawn(async move {
            run_health_loop(
                registry,
                events_rx,
                next_check,
                config,
                job_queue,
                shutdown_clone,
            )
            .await;
        });

        Self {
            handle: Some(handle),
            shutdown_notify,
        }
    }

    /// Gracefully shut down the WorkerManager loop, awaiting the task.
    /// Prefer this over dropping when an async context is available — it
    /// lets the in-flight probe iteration finish instead of aborting.
    pub async fn shutdown(&mut self) {
        self.shutdown_notify.notify_one();
        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
    }
}

impl Drop for WorkerManager {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

struct ProbeCompletion {
    worker_id: WorkerId,
    worker: Arc<dyn Worker>,
    expected_revision: u64,
    launched_status: WorkerStatus,
    health_config: HealthCheckConfig,
    probe_result: crate::worker::WorkerResult<()>,
}

struct RemovalCandidate {
    worker_id: WorkerId,
    url: String,
    expected_revision: u64,
}

enum ProbeApplyResult {
    Applied(Option<(WorkerStatus, WorkerStatus)>),
    Stale,
}

type ProbeFutures = FuturesUnordered<Pin<Box<dyn Future<Output = ProbeCompletion> + Send>>>;

/// Background task body: deadline-driven probe loop + event subscription.
///
/// The loop keeps deadline scheduling, in-flight probe tracking, and event
/// handling in one place. Probes run concurrently via `FuturesUnordered` so a
/// slow worker does not block unrelated health checks or registry events.
///
/// `next_check` is seeded by `WorkerManager::start()` via a synchronous
/// `reconcile_from_registry` call, so this function never runs a bootstrap
/// reconcile of its own — by the time the task is scheduled, the caller's
/// thread has already captured a consistent registry snapshot. The only
/// in-loop reconcile is the lag-recovery rebuild triggered by
/// `RecvError::Lagged`.
async fn run_health_loop(
    registry: Arc<WorkerRegistry>,
    mut events_rx: broadcast::Receiver<WorkerEvent>,
    mut next_check: HashMap<WorkerId, tokio::time::Instant>,
    config: WorkerManagerConfig,
    job_queue: Option<Arc<JobQueue>>,
    shutdown: Arc<Notify>,
) {
    let mut probes: ProbeFutures = FuturesUnordered::new();
    let mut in_flight: HashSet<WorkerId> = HashSet::new();

    loop {
        let now = tokio::time::Instant::now();
        let removals = queue_due_probes(
            &registry,
            &config,
            &mut next_check,
            &mut in_flight,
            &mut probes,
            now,
        );
        for removal in removals {
            if let Some(jq) = job_queue.as_ref() {
                submit_removal_job(
                    &removal.worker_id,
                    &removal.url,
                    removal.expected_revision,
                    jq,
                )
                .await;
            }
        }

        let sleep_until = next_check
            .values()
            .min()
            .copied()
            .unwrap_or_else(|| now + Duration::from_secs(config.default_check_interval_secs));

        tokio::select! {
            Some(completion) = probes.next(), if !probes.is_empty() => {
                let worker_id = completion.worker_id.clone();
                in_flight.remove(&worker_id);
                if matches!(
                    apply_probe_completion(&registry, completion, job_queue.as_ref()).await,
                    ProbeApplyResult::Applied(Some((_, WorkerStatus::Failed)))
                ) {
                    next_check.remove(&worker_id);
                }
            }
            () = tokio::time::sleep_until(sleep_until) => {}
            event = events_rx.recv() => {
                match event {
                    Ok(WorkerEvent::Registered { worker_id, worker }) => {
                        schedule_worker_at(
                            &mut next_check,
                            worker_id,
                            worker.status(),
                            &worker.metadata().health_config,
                            &config,
                            tokio::time::Instant::now(),
                            true,
                        );
                    }
                    Ok(WorkerEvent::Removed { worker_id, .. }) => {
                        next_check.remove(&worker_id);
                    }
                    Ok(WorkerEvent::Replaced { worker_id, new, .. }) => {
                        schedule_worker_at(
                            &mut next_check,
                            worker_id,
                            new.status(),
                            &new.metadata().health_config,
                            &config,
                            tokio::time::Instant::now(),
                            true,
                        );
                    }
                    Ok(WorkerEvent::StatusChanged { .. }) => {
                        // Self-published; nothing to do.
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!(
                            "WorkerManager lagged {n} events; rebuilding schedule from registry"
                        );
                        next_check.clear();
                        reconcile_from_registry(&registry, &mut next_check, &config);
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        debug!("WorkerEvent channel closed; WorkerManager exiting");
                        return;
                    }
                }
            }
            () = shutdown.notified() => {
                debug!("WorkerManager received shutdown signal");
                return;
            }
        }
    }
}

/// Side-effect-free startup reconcile: rebuild the schedule from the
/// current registry snapshot without publishing new events or removals.
fn reconcile_from_registry(
    registry: &Arc<WorkerRegistry>,
    next_check: &mut HashMap<WorkerId, tokio::time::Instant>,
    config: &WorkerManagerConfig,
) {
    let now = tokio::time::Instant::now();
    for descriptor in registry.reconcile_snapshot() {
        schedule_descriptor_at(next_check, descriptor, config, now, false);
    }
}

fn queue_due_probes(
    registry: &Arc<WorkerRegistry>,
    config: &WorkerManagerConfig,
    next_check: &mut HashMap<WorkerId, tokio::time::Instant>,
    in_flight: &mut HashSet<WorkerId>,
    probes: &mut ProbeFutures,
    now: tokio::time::Instant,
) -> Vec<RemovalCandidate> {
    let capacity = MAX_CONCURRENT_HEALTH_PROBES.saturating_sub(in_flight.len());
    if capacity == 0 {
        return Vec::new();
    }

    let due_ids: Vec<WorkerId> = next_check
        .iter()
        .filter(|(worker_id, deadline)| now >= **deadline && !in_flight.contains(*worker_id))
        .map(|(worker_id, _)| worker_id.clone())
        .take(capacity)
        .collect();

    let mut removals = Vec::new();
    for worker_id in due_ids {
        let Some(worker) = registry.get(&worker_id) else {
            next_check.remove(&worker_id);
            continue;
        };

        let health_config = worker.metadata().health_config.clone();
        if health_config.disable_health_check {
            next_check.remove(&worker_id);
            continue;
        }

        let launched_status = worker.status();
        let expected_revision = worker.revision();
        if launched_status == WorkerStatus::Failed {
            next_check.remove(&worker_id);
            if config.remove_unhealthy {
                removals.push(RemovalCandidate {
                    worker_id: worker_id.clone(),
                    url: worker.url().to_string(),
                    expected_revision,
                });
            }
            continue;
        }

        let next_deadline = now
            + Duration::from_secs(resolved_interval_secs(
                &health_config,
                config.default_check_interval_secs,
            ));
        next_check.insert(worker_id.clone(), next_deadline);
        in_flight.insert(worker_id.clone());
        probes.push(Box::pin(async move {
            let probe_result = worker.check_health_async().await;
            ProbeCompletion {
                worker_id,
                worker,
                expected_revision,
                launched_status,
                health_config,
                probe_result,
            }
        }));
    }

    removals
}

async fn apply_probe_completion(
    registry: &Arc<WorkerRegistry>,
    completion: ProbeCompletion,
    job_queue: Option<&Arc<JobQueue>>,
) -> ProbeApplyResult {
    let ProbeCompletion {
        worker_id,
        worker,
        expected_revision,
        launched_status,
        health_config,
        probe_result,
    } = completion;

    let probe_ok = match probe_result {
        Ok(()) => true,
        Err(err) => {
            warn!(
                worker_url = %worker.url(),
                error = %err,
                "Health probe failed"
            );
            false
        }
    };
    Metrics::record_worker_health_check(
        worker.worker_type().as_metric_label(),
        if probe_ok {
            metrics_labels::CB_SUCCESS
        } else {
            metrics_labels::CB_FAILURE
        },
    );

    let Some(((), transition)) =
        registry.apply_if_revision(&worker_id, expected_revision, |current_worker| {
            if launched_status == WorkerStatus::Pending {
                current_worker.total_pending_probes_increment();
            }
            (
                (),
                compute_next_status(current_worker, probe_ok, &health_config),
            )
        })
    else {
        debug!(
            worker_url = %worker.url(),
            expected_revision,
            "Discarding stale probe outcome after worker replacement"
        );
        return ProbeApplyResult::Stale;
    };

    if let Some((old, new)) = transition {
        debug!(
            worker_url = %worker.url(),
            ?old,
            ?new,
            "Worker status transition"
        );
        if new == WorkerStatus::Failed {
            if let Some(jq) = job_queue {
                submit_removal_job(&worker_id, worker.url(), expected_revision, jq).await;
            }
        }
    }

    ProbeApplyResult::Applied(transition)
}

fn resolved_interval_secs(health_config: &HealthCheckConfig, default_interval_secs: u64) -> u64 {
    if health_config.check_interval_secs > 0 {
        health_config.check_interval_secs
    } else {
        default_interval_secs
    }
}

fn schedule_descriptor_at(
    next_check: &mut HashMap<WorkerId, tokio::time::Instant>,
    descriptor: WorkerDescriptor,
    config: &WorkerManagerConfig,
    now: tokio::time::Instant,
    immediate: bool,
) {
    if descriptor.disable_health_check {
        next_check.remove(&descriptor.worker_id);
        return;
    }
    if descriptor.status == WorkerStatus::Failed {
        // Startup reconcile and lagged rebuild must be side-effect-free:
        // do not reschedule already-failed workers for probing or removal.
        next_check.remove(&descriptor.worker_id);
        return;
    }

    let delay = if immediate {
        Duration::ZERO
    } else {
        Duration::from_secs(if descriptor.check_interval_secs > 0 {
            descriptor.check_interval_secs
        } else {
            config.default_check_interval_secs
        })
    };
    next_check.insert(descriptor.worker_id, now + delay);
}

fn schedule_worker_at(
    next_check: &mut HashMap<WorkerId, tokio::time::Instant>,
    worker_id: WorkerId,
    status: WorkerStatus,
    health_config: &HealthCheckConfig,
    config: &WorkerManagerConfig,
    now: tokio::time::Instant,
    immediate: bool,
) {
    schedule_descriptor_at(
        next_check,
        WorkerDescriptor {
            worker_id,
            status,
            disable_health_check: health_config.disable_health_check,
            check_interval_secs: health_config.check_interval_secs,
        },
        config,
        now,
        immediate,
    );
}

/// Apply the state machine to a probe outcome. Returns the next status if
/// a transition is needed, `None` if the worker stays in its current state.
///
/// State machine rules:
///   - Pending → Ready on `success_threshold` consecutive successes
///   - Pending → Failed on `max_pending_probes` (10 × failure_threshold) total
///   - NotReady → Ready on `success_threshold` consecutive successes
///   - NotReady → Failed on `liveness_failure_threshold` (3 × failure_threshold)
///   - Ready → NotReady on `failure_threshold` consecutive failures
///   - Failed: terminal (handled outside this function — no transitions)
fn compute_next_status(
    worker: &Arc<dyn Worker>,
    probe_ok: bool,
    health_config: &HealthCheckConfig,
) -> Option<WorkerStatus> {
    let current_status = worker.status();
    let success_threshold = health_config.success_threshold as usize;
    let failure_threshold = health_config.failure_threshold as usize;
    // Liveness threshold: tolerate longer outages before declaring Failed,
    // analogous to K8s having separate readiness and liveness probes.
    let liveness_threshold = failure_threshold * 3;
    // Pending cap: prevent misconfigured URLs from sitting in Pending forever.
    let max_pending_probes = failure_threshold * 10;

    if probe_ok {
        worker.consecutive_failures_reset();
        let successes = worker.consecutive_successes_increment();

        if matches!(
            current_status,
            WorkerStatus::Pending | WorkerStatus::NotReady
        ) && successes >= success_threshold
        {
            worker.consecutive_successes_reset();
            worker.total_pending_probes_reset();
            return Some(WorkerStatus::Ready);
        }

        // Even on a successful probe, enforce the Pending cap. A worker
        // that flaps F,S,F,S,... never reaches success_threshold and would
        // otherwise grow `total_pending_probes` without bound.
        if current_status == WorkerStatus::Pending
            && worker.total_pending_probes() >= max_pending_probes
        {
            worker.consecutive_successes_reset();
            worker.consecutive_failures_reset();
            return Some(WorkerStatus::Failed);
        }

        None
    } else {
        worker.consecutive_successes_reset();
        let failures = worker.consecutive_failures_increment();

        match current_status {
            WorkerStatus::Ready => {
                if failures >= failure_threshold {
                    worker.consecutive_failures_reset();
                    return Some(WorkerStatus::NotReady);
                }
            }
            WorkerStatus::NotReady => {
                if failures >= liveness_threshold {
                    worker.consecutive_failures_reset();
                    return Some(WorkerStatus::Failed);
                }
            }
            WorkerStatus::Pending => {
                if worker.total_pending_probes() >= max_pending_probes {
                    worker.consecutive_failures_reset();
                    return Some(WorkerStatus::Failed);
                }
            }
            WorkerStatus::Failed => {
                // Terminal — handled outside.
            }
        }

        None
    }
}

async fn submit_removal_job(
    worker_id: &WorkerId,
    worker_url: &str,
    expected_revision: u64,
    job_queue: &Arc<JobQueue>,
) {
    let url = worker_url.to_string();
    warn!(
        worker_id = %worker_id.as_str(),
        worker_url = %url,
        expected_revision,
        "Removing failed worker from registry"
    );
    if let Err(e) = job_queue
        .submit(Job::RemoveWorker {
            url: url.clone(),
            expected_revision: Some(expected_revision),
        })
        .await
    {
        error!(
            worker_url = %url,
            error = %e,
            "Failed to submit worker removal job"
        );
    }
}

impl WorkerManager {
    pub fn get_worker_urls(registry: &Arc<WorkerRegistry>) -> Vec<String> {
        registry
            .get_all()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    pub async fn flush_cache_all(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> FlushCacheResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        let http_workers: Vec<_> = workers
            .into_iter()
            .filter(|w| matches!(w.connection_mode(), ConnectionMode::Http))
            .collect();

        if http_workers.is_empty() {
            return FlushCacheResult {
                successful: vec![],
                failed: vec![],
                total_workers,
                http_workers: 0,
                message: "No HTTP workers available for cache flush".to_string(),
            };
        }

        info!(
            "Flushing cache on {} HTTP workers (out of {} total)",
            http_workers.len(),
            total_workers
        );

        let responses = fan_out(&http_workers, client, "flush_cache", reqwest::Method::POST).await;

        let mut successful = Vec::new();
        let mut failed = Vec::new();

        for resp in responses {
            match resp.result {
                Ok(r) if r.status().is_success() => successful.push(resp.url),
                Ok(r) => failed.push((resp.url, format!("HTTP {}", r.status()))),
                Err(e) => failed.push((resp.url, e.to_string())),
            }
        }

        let message = if failed.is_empty() {
            format!(
                "Successfully flushed cache on all {} HTTP workers",
                successful.len()
            )
        } else {
            format!(
                "Cache flush: {} succeeded, {} failed",
                successful.len(),
                failed.len()
            )
        };

        info!("{}", message);

        FlushCacheResult {
            successful,
            failed,
            total_workers,
            http_workers: http_workers.len(),
            message,
        }
    }

    pub async fn get_all_worker_loads(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> WorkerLoadsResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        let futures: Vec<_> = workers
            .iter()
            .map(|worker| {
                let worker_type = match worker.worker_type() {
                    WorkerType::Regular => None,
                    WorkerType::Prefill => Some("prefill".to_string()),
                    WorkerType::Decode => Some("decode".to_string()),
                };
                let connection_mode = worker.connection_mode();
                let client = client.clone();
                let worker = Arc::clone(worker);

                async move {
                    let details = match connection_mode {
                        ConnectionMode::Http => {
                            WorkerMonitor::fetch_http_load(&client, &worker).await
                        }
                        ConnectionMode::Grpc => WorkerMonitor::fetch_grpc_load(&worker).await,
                    };
                    let load = details
                        .as_ref()
                        .map(|d| d.total_used_tokens() as isize)
                        .unwrap_or(-1);
                    WorkerLoadInfo {
                        worker: worker.url().to_string(),
                        worker_type,
                        load,
                        details,
                    }
                }
            })
            .collect();

        let loads = future::join_all(futures).await;
        let successful = loads.iter().filter(|l| l.load >= 0).count();
        let failed = loads.iter().filter(|l| l.load < 0).count();

        WorkerLoadsResult {
            loads,
            total_workers,
            successful,
            failed,
        }
    }

    pub async fn get_engine_metrics(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> EngineMetricsResult {
        let workers = worker_registry.get_all();

        if workers.is_empty() {
            return EngineMetricsResult::Err("No available workers".to_string());
        }

        let responses = fan_out(&workers, client, "metrics", reqwest::Method::GET).await;

        let mut metric_packs = Vec::new();
        for resp in responses {
            if let Ok(r) = resp.result {
                if r.status().is_success() {
                    if let Ok(text) = r.text().await {
                        metric_packs.push(MetricPack {
                            labels: vec![("worker_addr".into(), resp.url)],
                            metrics_text: text,
                        });
                    }
                }
            }
        }

        if metric_packs.is_empty() {
            return EngineMetricsResult::Err("All backend requests failed".to_string());
        }

        match metrics_aggregator::aggregate_metrics(metric_packs) {
            Ok(text) => EngineMetricsResult::Ok(text),
            Err(e) => EngineMetricsResult::Err(format!("Failed to aggregate metrics: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};

    use super::*;
    use crate::worker::{BasicWorkerBuilder, Worker, WorkerError, WorkerRegistry, WorkerType};

    fn make_worker(url: &str, success_threshold: u32, failure_threshold: u32) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .health_config(HealthCheckConfig {
                    success_threshold,
                    failure_threshold,
                    timeout_secs: 1,
                    check_interval_secs: 1,
                    disable_health_check: false,
                })
                .build(),
        )
    }

    fn cfg(success_threshold: u32, failure_threshold: u32) -> HealthCheckConfig {
        HealthCheckConfig {
            success_threshold,
            failure_threshold,
            timeout_secs: 1,
            check_interval_secs: 1,
            disable_health_check: false,
        }
    }

    #[test]
    fn test_state_machine_pending_to_ready_after_success_threshold() {
        let worker = make_worker("http://w:1", 2, 3);
        assert_eq!(worker.status(), WorkerStatus::Pending);

        // First success: not yet promoted (1 < 2)
        assert_eq!(compute_next_status(&worker, true, &cfg(2, 3)), None);
        assert_eq!(worker.status(), WorkerStatus::Pending);

        // Second success: promoted Pending → Ready
        let next = compute_next_status(&worker, true, &cfg(2, 3));
        assert_eq!(next, Some(WorkerStatus::Ready));
    }

    #[test]
    fn test_state_machine_ready_to_notready_after_failure_threshold() {
        let worker = make_worker("http://w:1", 2, 3);
        worker.set_status(WorkerStatus::Ready);

        // 1 fail, 2 fail: still Ready
        assert_eq!(compute_next_status(&worker, false, &cfg(2, 3)), None);
        assert_eq!(compute_next_status(&worker, false, &cfg(2, 3)), None);
        assert_eq!(worker.status(), WorkerStatus::Ready);

        // 3rd fail: Ready → NotReady
        assert_eq!(
            compute_next_status(&worker, false, &cfg(2, 3)),
            Some(WorkerStatus::NotReady)
        );
    }

    #[test]
    fn test_state_machine_notready_to_failed_after_liveness_threshold() {
        let worker = make_worker("http://w:1", 2, 3);
        worker.set_status(WorkerStatus::NotReady);

        // liveness_threshold = 3 × failure_threshold = 9
        for i in 1..9 {
            assert_eq!(
                compute_next_status(&worker, false, &cfg(2, 3)),
                None,
                "iteration {i}"
            );
        }

        // 9th consecutive failure → Failed
        assert_eq!(
            compute_next_status(&worker, false, &cfg(2, 3)),
            Some(WorkerStatus::Failed)
        );
    }

    #[test]
    fn test_state_machine_pending_to_failed_after_max_pending_probes() {
        let worker = make_worker("http://w:1", 2, 3);
        // max_pending_probes = 10 × failure_threshold = 30

        // Simulate 30 failed probes — increment counter manually since the
        // loop usually does this before calling compute_next_status.
        for _ in 0..29 {
            worker.total_pending_probes_increment();
            assert_eq!(compute_next_status(&worker, false, &cfg(2, 3)), None);
        }
        worker.total_pending_probes_increment();
        // 30th: Pending → Failed
        assert_eq!(
            compute_next_status(&worker, false, &cfg(2, 3)),
            Some(WorkerStatus::Failed)
        );
    }

    #[test]
    fn test_state_machine_pending_to_failed_on_success_when_cap_exceeded() {
        // Flapping pattern: a Pending worker that flaps F,S,F,S,... and
        // never reaches success_threshold should still hit max_pending_probes.
        let worker = make_worker("http://w:1", 2, 3);

        // Simulate 30 attempts with the counter, then call compute on success.
        for _ in 0..30 {
            worker.total_pending_probes_increment();
        }
        // Even on success, the cap fires.
        assert_eq!(
            compute_next_status(&worker, true, &cfg(2, 3)),
            Some(WorkerStatus::Failed)
        );
    }

    #[test]
    fn test_state_machine_failed_is_terminal() {
        let worker = make_worker("http://w:1", 2, 3);
        worker.set_status(WorkerStatus::Failed);

        // Successful probes don't recover Failed.
        assert_eq!(compute_next_status(&worker, true, &cfg(2, 3)), None);
        assert_eq!(compute_next_status(&worker, true, &cfg(2, 3)), None);
        assert_eq!(worker.status(), WorkerStatus::Failed);

        // Failed probes don't transition Failed anywhere either.
        assert_eq!(compute_next_status(&worker, false, &cfg(2, 3)), None);
    }

    #[test]
    fn test_state_machine_notready_to_ready_on_success_threshold() {
        let worker = make_worker("http://w:1", 2, 3);
        worker.set_status(WorkerStatus::NotReady);

        assert_eq!(compute_next_status(&worker, true, &cfg(2, 3)), None);
        assert_eq!(
            compute_next_status(&worker, true, &cfg(2, 3)),
            Some(WorkerStatus::Ready)
        );
    }

    #[test]
    fn test_state_machine_success_resets_failure_counter() {
        let worker = make_worker("http://w:1", 2, 3);
        worker.set_status(WorkerStatus::Ready);

        // 2 failures (not yet at threshold)
        assert_eq!(compute_next_status(&worker, false, &cfg(2, 3)), None);
        assert_eq!(compute_next_status(&worker, false, &cfg(2, 3)), None);

        // Single success resets the counter
        assert_eq!(compute_next_status(&worker, true, &cfg(2, 3)), None);

        // Now 2 failures again — still no transition because counter was reset
        assert_eq!(compute_next_status(&worker, false, &cfg(2, 3)), None);
        assert_eq!(compute_next_status(&worker, false, &cfg(2, 3)), None);
        assert_eq!(worker.status(), WorkerStatus::Ready);

        // 3rd failure now triggers transition
        assert_eq!(
            compute_next_status(&worker, false, &cfg(2, 3)),
            Some(WorkerStatus::NotReady)
        );
    }

    #[test]
    fn test_reconcile_from_registry_skips_failed_workers_on_bootstrap() {
        let registry = Arc::new(WorkerRegistry::new());
        let failed_worker = make_worker("http://failed:1", 2, 3);
        failed_worker.set_status(WorkerStatus::Failed);
        let failed_id = registry.register(failed_worker).unwrap();

        let mut next_check = HashMap::new();
        reconcile_from_registry(
            &registry,
            &mut next_check,
            &WorkerManagerConfig {
                default_check_interval_secs: 5,
                remove_unhealthy: true,
            },
        );

        assert!(
            !next_check.contains_key(&failed_id),
            "bootstrap reconcile must not reschedule failed workers"
        );
    }

    #[test]
    fn test_reconcile_from_registry_captures_pending_and_ready_workers() {
        // Positive complement to the "skips failed" test: the startup
        // reconcile must pick up Pending (not-yet-probed) and Ready
        // workers that existed at registry snapshot time. This is what
        // makes WorkerManager::start() deterministic — the schedule is
        // captured on the caller's thread, not whenever the spawned task
        // happens to run.
        let registry = Arc::new(WorkerRegistry::new());

        let pending_worker = make_worker("http://pending:1", 2, 3);
        assert_eq!(pending_worker.status(), WorkerStatus::Pending);
        let pending_id = registry.register(pending_worker).unwrap();

        let ready_worker = make_worker("http://ready:1", 2, 3);
        ready_worker.set_status(WorkerStatus::Ready);
        let ready_id = registry.register(ready_worker).unwrap();

        let mut next_check = HashMap::new();
        reconcile_from_registry(
            &registry,
            &mut next_check,
            &WorkerManagerConfig {
                default_check_interval_secs: 5,
                remove_unhealthy: true,
            },
        );

        assert!(
            next_check.contains_key(&pending_id),
            "pending worker must be in the bootstrap schedule"
        );
        assert!(
            next_check.contains_key(&ready_id),
            "ready worker must be in the bootstrap schedule"
        );
    }

    #[tokio::test]
    async fn test_worker_manager_start_is_deterministic_with_preexisting_workers() {
        // End-to-end contract for fix #2: a worker that exists in the
        // registry before WorkerManager::start() returns must be on the
        // schedule the moment the spawned task begins running — no race
        // with task scheduling. We can't observe `next_check` directly,
        // so we run the full start/shutdown lifecycle with a very long
        // probe interval (so no probe actually fires) and verify the
        // happy path doesn't panic. Together with the reconcile unit
        // test above, this covers both the "reconcile captures workers"
        // and "start() calls reconcile synchronously" invariants.
        let registry = Arc::new(WorkerRegistry::new());
        let worker = make_worker("http://pre-existing:1", 2, 3);
        worker.set_status(WorkerStatus::Ready);
        registry.register(worker).unwrap();

        let mut manager = WorkerManager::start(
            registry,
            WorkerManagerConfig {
                default_check_interval_secs: 3600,
                remove_unhealthy: false,
            },
            None,
        );
        manager.shutdown().await;
    }

    #[tokio::test]
    async fn test_apply_probe_completion_discards_stale_probe_after_replace() {
        let registry = Arc::new(WorkerRegistry::new());
        let worker = make_worker("http://w:1", 1, 1);
        worker.set_status(WorkerStatus::Ready);
        let worker_id = registry.register(worker.clone()).unwrap();
        let expected_revision = worker.revision();

        let completion = ProbeCompletion {
            worker_id: worker_id.clone(),
            worker: worker.clone(),
            expected_revision,
            launched_status: WorkerStatus::Ready,
            health_config: cfg(1, 1),
            probe_result: Err(WorkerError::HealthCheckFailed {
                url: worker.url().to_string(),
                reason: "stale probe".to_string(),
            }),
        };

        let replacement = make_worker("http://w:1", 1, 1);
        assert!(registry.replace(&worker_id, replacement));

        let current = registry.get(&worker_id).unwrap();
        assert_eq!(current.status(), WorkerStatus::Ready);
        assert_eq!(current.revision(), expected_revision + 1);

        let result = apply_probe_completion(&registry, completion, None).await;
        assert!(matches!(result, ProbeApplyResult::Stale));
        assert_eq!(
            registry.get(&worker_id).unwrap().status(),
            WorkerStatus::Ready
        );
    }
}
