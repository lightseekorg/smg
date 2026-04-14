//! Worker load monitoring service.
//!
//! `WorkerMonitor` consolidates the previous `LoadMonitor` (per-group
//! polling loops) and `WorkerLoadManager` (DP rank load cache) into a
//! single coordinator that subscribes to `WorkerRegistry` events.
//!
//! ## Lifecycle
//!
//! - `new()` creates the cache and watch channel without spawning any
//!   tasks. The factory layer reads `worker_load_manager` immediately so
//!   policies that need the DP cache can be wired before workers exist.
//! - `start_event_loop()` subscribes to the registry, runs a synchronous
//!   bootstrap reconcile, and spawns the background event-handling task.
//!   It must be called after the initial worker population (mesh replay,
//!   K8s discovery, etc.) has finished — same ordering rule as
//!   `WorkerManager::start`.
//! - `Drop` aborts the event task and every per-group polling loop.
//!
//! ## Group lifecycle (event-driven)
//!
//! Polling is keyed by `WorkerGroupKey = (model_id, worker_type,
//! connection_mode)`. The event loop reacts to:
//!
//! - `Registered` / `Replaced`: reconcile every group the worker
//!   participates in. New groups start a polling loop; existing groups
//!   restart with a new interval if the per-worker override changed.
//! - `Removed`: reconcile every group the removed worker participated
//!   in. Empty groups stop their loops and evict cached state.
//! - `StatusChanged`: workers leaving `Ready` are evicted from the
//!   watch-channel snapshot and the DP cache (the group loop is left
//!   alone and will skip non-Ready workers on its next tick).
//! - `RecvError::Lagged`: stop every loop, clear shared state, and
//!   rebuild from the current registry snapshot. Monitoring state is
//!   derived data; full rebuild is the recovery mechanism.
//!
//! ## Polling
//!
//! Each group runs a single `tokio::time::interval` loop. Every tick:
//!
//! 1. Skip if no load-aware policy is currently active for this group
//!    (matches the original `LoadMonitor` policy gate).
//! 2. Fetch loads concurrently from every `Ready` worker in the group.
//! 3. Update PowerOfTwo policies and the DP cache.
//! 4. Atomically clear stale entries for the group from the watch
//!    channel and merge in the fresh loads.

use std::{
    collections::HashMap,
    fmt::Debug,
    sync::{Arc, Weak},
    time::Duration,
};

use futures::future;
use openai_protocol::worker::{WorkerGroupKey, WorkerLoadResponse, WorkerStatus};
use parking_lot::{Mutex, RwLock};
use tokio::{
    sync::{broadcast, watch},
    task::JoinHandle,
};
use tracing::{debug, info, warn};

use crate::{
    policies::PolicyRegistry,
    worker::{event::WorkerEvent, ConnectionMode, Worker, WorkerRegistry},
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// DP rank load cache used by load-aware routing policies.
///
/// Pure in-memory data structure with no I/O. `WorkerMonitor` owns the
/// shared `Arc<WorkerLoadManager>` and updates it on every successful
/// poll; routing policies read from it via
/// `select_and_increment_lowest_dp_load`.
#[derive(Debug, Default)]
pub struct WorkerLoadManager {
    /// `<worker_url, <dp_rank, load>>`
    dp_cached_loads: RwLock<HashMap<String, HashMap<isize, isize>>>,
}

impl WorkerLoadManager {
    pub fn new() -> Self {
        Self {
            dp_cached_loads: RwLock::new(HashMap::new()),
        }
    }

    pub fn update_dp_loads(&self, loads: &HashMap<String, HashMap<isize, isize>>) {
        debug!("WorkerLoadManager update_dp_loads map:{:?}", loads);
        let mut cached = self.dp_cached_loads.write();
        cached.extend(loads.iter().map(|(k, v)| (k.clone(), v.clone())));
    }

    pub fn select_and_increment_lowest_dp_load(
        &self,
        worker: &dyn Worker,
        increment: isize,
    ) -> Option<isize> {
        let mut cached = self.dp_cached_loads.write();
        let loads = cached.get_mut(worker.url())?;
        let (&dp_rank, _) = loads.iter().min_by_key(|&(rank, load)| (*load, *rank))?;
        if let Some(v) = loads.get_mut(&dp_rank) {
            *v += increment;
        }
        Some(dp_rank)
    }

    pub fn remove_workers(&self, urls: &[String]) {
        let mut cached = self.dp_cached_loads.write();
        for url in urls {
            cached.remove(url);
        }
    }

    /// Drop a single worker's cached DP loads. Avoids the
    /// `&[String]` allocation that `remove_workers` requires when the
    /// caller only has a single `&str`.
    pub fn remove_worker(&self, url: &str) {
        self.dp_cached_loads.write().remove(url);
    }

    /// Drop every cached DP load entry. Used by `WorkerMonitor` during
    /// `stop_all_groups` and lag-recovery rebuilds so the cache cannot
    /// hand out stale per-rank loads after a full reconcile.
    pub fn clear(&self) {
        self.dp_cached_loads.write().clear();
    }
}

/// Per-group polling loop state.
struct GroupState {
    handle: JoinHandle<()>,
    interval: Duration,
}

/// Load monitoring service that subscribes to `WorkerRegistry` events.
pub struct WorkerMonitor {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    pub worker_load_manager: Arc<WorkerLoadManager>,
    client: reqwest::Client,
    default_interval: Duration,
    load_tx: watch::Sender<HashMap<String, WorkerLoadResponse>>,
    load_rx: watch::Receiver<HashMap<String, WorkerLoadResponse>>,
    group_handles: Mutex<HashMap<WorkerGroupKey, GroupState>>,
    event_task: Mutex<Option<JoinHandle<()>>>,
}

impl Debug for WorkerMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerMonitor")
            .field("default_interval", &self.default_interval)
            .finish_non_exhaustive()
    }
}

impl WorkerMonitor {
    /// Construct a `WorkerMonitor` without spawning any background work.
    ///
    /// The caller must invoke [`Self::start_event_loop`] once initial
    /// workers have been registered. Until then, `worker_load_manager`
    /// is still safe to read; it just stays empty.
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        default_interval_secs: u64,
    ) -> Self {
        let (load_tx, load_rx) = watch::channel(HashMap::new());
        Self {
            worker_registry,
            policy_registry,
            worker_load_manager: Arc::new(WorkerLoadManager::new()),
            client,
            default_interval: Duration::from_secs(default_interval_secs.max(1)),
            load_tx,
            load_rx,
            group_handles: Mutex::new(HashMap::new()),
            event_task: Mutex::new(None),
        }
    }

    /// Subscribe to the snapshot of per-worker loads.
    ///
    /// The watch receiver returns the most recent fully merged map;
    /// stale entries are pruned on each tick of the relevant group.
    pub fn subscribe(&self) -> watch::Receiver<HashMap<String, WorkerLoadResponse>> {
        self.load_rx.clone()
    }

    /// Subscribe to registry events, run a synchronous bootstrap
    /// reconcile, and spawn the background event task.
    ///
    /// Subscribing **before** the bootstrap reconcile guarantees the
    /// "registered between subscribe and reconcile" race resolves
    /// idempotently: any registration that lands after this call either
    /// is already in the snapshot (because it ran on this thread) or
    /// arrives as a `Registered` event that the loop applies on top of
    /// the snapshot.
    pub fn start_event_loop(self: &Arc<Self>) {
        // Capture the receiver before reconciling so events that fire
        // during reconcile are buffered, not lost.
        let events_rx = self.worker_registry.subscribe_events();

        // Synchronous bootstrap: build initial group set from the
        // registry snapshot and start polling loops.
        self.reconcile_from_registry();

        // Hand the spawned task a `Weak<Self>` so it does not pin the
        // monitor in memory across drops. If we passed an `Arc<Self>`
        // the spawned task would form a cycle: the monitor stores the
        // task's `JoinHandle`, and the task closure holds an `Arc`
        // back to the monitor. The strong count would never reach
        // zero, `Drop` would never run, and the abort calls would be
        // unreachable outside of full runtime shutdown.
        let monitor = Arc::downgrade(self);
        #[expect(
            clippy::disallowed_methods,
            reason = "WorkerMonitor event loop runs for the lifetime of the registry; the JoinHandle is stored on the monitor and aborted in Drop"
        )]
        let handle = tokio::spawn(async move {
            run_event_loop(monitor, events_rx).await;
        });

        *self.event_task.lock() = Some(handle);
    }

    /// Stop every per-group polling loop and clear the shared load
    /// snapshot + DP-rank cache.
    ///
    /// Called from `Drop`, the bootstrap reconcile path, and the
    /// `RecvError::Lagged` rebuild. Always clears both caches even
    /// when no groups were running, so a lag-recovery rebuild cannot
    /// inherit stale per-rank loads from a previous incarnation.
    /// Idempotent.
    pub fn stop_all_groups(&self) {
        let drained: Vec<(WorkerGroupKey, GroupState)> = {
            let mut handles = self.group_handles.lock();
            handles.drain().collect()
        };

        if !drained.is_empty() {
            info!("Stopping all {} load monitor groups", drained.len());
            for (key, state) in drained {
                debug!("Stopping load monitor group: {key}");
                state.handle.abort();
            }
        }

        // Always clear both caches. Skipping when `drained.is_empty()`
        // would leave stale per-rank loads behind for any caller that
        // seeded the cache without going through a group loop, and
        // makes the function harder to reason about as a "reset".
        self.load_tx.send_modify(|map| map.clear());
        self.worker_load_manager.clear();
    }

    /// Recompute the polling state for every currently-known group.
    ///
    /// Used as the synchronous bootstrap path and as the lag-recovery
    /// rebuild after `RecvError::Lagged`. Stops every existing loop,
    /// clears the cached snapshot, then walks the registry to start
    /// fresh loops for each non-empty group.
    fn reconcile_from_registry(self: &Arc<Self>) {
        // Stop everything first so the rebuild starts from a clean slate.
        self.stop_all_groups();

        // Walk every worker once and bucket them by group.
        let workers = self.worker_registry.get_all();
        let mut group_keys: HashMap<WorkerGroupKey, ()> = HashMap::new();
        for worker in workers {
            for key in group_keys_for_worker(&worker) {
                group_keys.insert(key, ());
            }
        }

        for key in group_keys.into_keys() {
            self.reconcile_group(&key);
        }
    }

    /// Bring a single group's polling state into sync with the registry.
    ///
    /// - Empty group: stop the loop (if any), evict the group's cached
    ///   loads from the watch channel and DP cache.
    /// - Non-empty group, no current loop: spawn one with the group's
    ///   desired interval.
    /// - Non-empty group, loop exists with a stale interval: stop and
    ///   restart so the new interval takes effect.
    /// - Non-empty group, loop exists with the correct interval: no-op.
    fn reconcile_group(self: &Arc<Self>, key: &WorkerGroupKey) {
        let workers = self.worker_registry.get_workers_filtered(
            Some(&key.model_id),
            Some(key.worker_type),
            Some(key.connection_mode),
            None,
            false,
        );

        if workers.is_empty() {
            self.stop_group(key);
            return;
        }

        let desired_interval = group_interval(&workers, self.default_interval);

        let needs_start = {
            let mut handles = self.group_handles.lock();
            match handles.get(key) {
                Some(state) if state.interval == desired_interval => false,
                Some(_) => {
                    // Interval changed — stop the old loop and fall
                    // through to start a fresh one below.
                    if let Some(old) = handles.remove(key) {
                        debug!("Restarting load monitor group {key} with new interval {desired_interval:?}");
                        old.handle.abort();
                    }
                    true
                }
                None => true,
            }
        };

        if needs_start {
            self.spawn_group_loop(key.clone(), desired_interval);
        }
    }

    /// Stop a single group's polling loop.
    ///
    /// Does **not** evict per-worker cached loads from the watch
    /// channel or DP cache. Callers that know the affected URLs (e.g.
    /// the event loop on `Removed` / `Replaced` / `StatusChanged`)
    /// must invoke [`Self::evict_worker_loads`] separately. The
    /// per-group polling loop itself prunes URLs on the next tick when
    /// the worker is gone, but a stopped loop never gets a next tick.
    fn stop_group(&self, key: &WorkerGroupKey) {
        let removed = {
            let mut handles = self.group_handles.lock();
            handles.remove(key)
        };

        if let Some(state) = removed {
            info!("Stopping load monitor for empty group {key}");
            state.handle.abort();
        }
    }

    /// Evict a single worker's cached loads from both the watch
    /// channel snapshot and the DP cache. Used by the event loop on
    /// `Removed`, `Replaced`, and `StatusChanged` away from `Ready`.
    fn evict_worker_loads(&self, url: &str) {
        self.load_tx.send_modify(|map| {
            map.remove(url);
        });
        self.worker_load_manager.remove_worker(url);
    }

    /// Spawn the polling loop for a single group.
    fn spawn_group_loop(self: &Arc<Self>, key: WorkerGroupKey, interval: Duration) {
        info!("Starting load monitor for group {key} with interval {interval:?}");

        // Same `Weak<Self>` rationale as `start_event_loop`: the
        // group loop is owned by `group_handles` on the monitor, so
        // it must not own a strong reference back to the monitor.
        let monitor = Arc::downgrade(self);
        let group_key = key.clone();

        #[expect(
            clippy::disallowed_methods,
            reason = "Group polling loop runs for the lifetime of the group; the JoinHandle is stored in group_handles and aborted on group removal or monitor drop"
        )]
        let handle = tokio::spawn(async move {
            group_monitor_loop(monitor, group_key, interval).await;
        });

        let mut handles = self.group_handles.lock();
        handles.insert(key, GroupState { handle, interval });
    }

    /// Fetch load via HTTP `GET /v1/loads?include=core`.
    ///
    /// Returns `None` on transport failure, non-success status, JSON
    /// parse failure, or an empty `loads` array.
    pub(crate) async fn fetch_http_load(
        client: &reqwest::Client,
        worker: &Arc<dyn Worker>,
    ) -> Option<WorkerLoadResponse> {
        let url = worker.url();
        let load_url = format!("{url}/v1/loads?include=core");
        let mut req = client.get(&load_url).timeout(REQUEST_TIMEOUT);
        if let Some(key) = worker.api_key() {
            req = req.bearer_auth(key);
        }

        let resp = match req.send().await {
            Ok(r) if r.status().is_success() => r,
            _ => return None,
        };

        let response: WorkerLoadResponse = resp.json().await.ok()?;

        if response.loads.is_empty() {
            return None;
        }

        Some(response)
    }

    /// Fetch load via the gRPC `GetLoads` RPC. Only supported for SGLang
    /// backends. Returns `None` on missing client, RPC error, or empty
    /// `loads` array.
    pub(crate) async fn fetch_grpc_load(worker: &Arc<dyn Worker>) -> Option<WorkerLoadResponse> {
        let grpc_client = match worker.get_grpc_client().await {
            Ok(Some(client)) => client,
            Ok(None) => {
                debug!("No gRPC client for worker {}", worker.url());
                return None;
            }
            Err(e) => {
                debug!("Failed to get gRPC client for {}: {e}", worker.url());
                return None;
            }
        };

        match grpc_client.get_loads().await {
            Ok(load) if !load.loads.is_empty() => Some(load),
            Ok(_) => None,
            Err(e) => {
                debug!("gRPC GetLoads failed for {}: {e}", worker.url());
                None
            }
        }
    }
}

impl Drop for WorkerMonitor {
    fn drop(&mut self) {
        if let Some(handle) = self.event_task.get_mut().take() {
            handle.abort();
        }
        for (_, state) in self.group_handles.get_mut().drain() {
            state.handle.abort();
        }
    }
}

/// Compute the set of `WorkerGroupKey`s a worker participates in.
///
/// A worker can serve multiple models (multimodel deployments), so it
/// can belong to multiple groups simultaneously. Each `(model, type,
/// connection)` triple is one group.
fn group_keys_for_worker(worker: &Arc<dyn Worker>) -> Vec<WorkerGroupKey> {
    WorkerRegistry::worker_model_ids(worker)
        .into_iter()
        .map(|model_id| WorkerGroupKey {
            model_id,
            worker_type: *worker.worker_type(),
            connection_mode: *worker.connection_mode(),
        })
        .collect()
}

/// Compute the polling interval for a group from per-worker overrides.
///
/// Uses the smallest `load_monitor_interval_secs` across the group so
/// the fastest worker's polling cadence wins. Falls back to
/// `default_interval` when no worker sets an override. Always floored
/// to one second to prevent tight-loop DoS.
fn group_interval(workers: &[Arc<dyn Worker>], default_interval: Duration) -> Duration {
    let override_secs = workers
        .iter()
        .filter_map(|w| w.metadata().spec.load_monitor_interval_secs)
        .min();

    let interval = override_secs
        .map(|s| Duration::from_secs(s.max(1)))
        .unwrap_or(default_interval);
    interval.max(Duration::from_secs(1))
}

/// Background event handler. Lives as long as the `WorkerMonitor`.
///
/// Holds a `Weak<WorkerMonitor>` rather than an `Arc<WorkerMonitor>`
/// so the spawned task does not pin the monitor in memory after every
/// strong reference has been dropped. Each iteration upgrades the
/// `Weak` only for the duration of the work and drops the temporary
/// `Arc` before the next `recv().await`, so the monitor's `Drop` can
/// fire as soon as the owning `AppContext` goes away.
async fn run_event_loop(
    monitor: Weak<WorkerMonitor>,
    mut events_rx: broadcast::Receiver<WorkerEvent>,
) {
    loop {
        let event = events_rx.recv().await;
        let Some(monitor) = monitor.upgrade() else {
            debug!("WorkerMonitor was dropped; exiting event loop");
            return;
        };
        match event {
            Ok(WorkerEvent::Registered { worker, .. }) => {
                for key in group_keys_for_worker(&worker) {
                    monitor.reconcile_group(&key);
                }
            }
            Ok(WorkerEvent::Removed { worker, .. }) => {
                // Evict the worker from both caches BEFORE reconciling
                // the group. If this was the last worker in its group,
                // `reconcile_group` will stop the polling loop, and a
                // stopped loop cannot prune the stale entry on a later
                // tick — it would persist forever otherwise.
                monitor.evict_worker_loads(worker.url());
                for key in group_keys_for_worker(&worker) {
                    monitor.reconcile_group(&key);
                }
            }
            Ok(WorkerEvent::Replaced { old, new, .. }) => {
                // Registry guarantees `old.url() == new.url()` (replace
                // rejects URL changes), but the model list may have
                // shrunk so some old groups disappear. Evict the URL
                // from caches before reconciling so the disappearing
                // groups do not leak the entry; the surviving / new
                // groups will repopulate it on their next poll tick.
                monitor.evict_worker_loads(old.url());
                let mut keys = group_keys_for_worker(&old);
                keys.extend(group_keys_for_worker(&new));
                keys.sort_by(|a, b| {
                    a.model_id
                        .cmp(&b.model_id)
                        .then_with(|| (a.worker_type as u8).cmp(&(b.worker_type as u8)))
                        .then_with(|| (a.connection_mode as u8).cmp(&(b.connection_mode as u8)))
                });
                keys.dedup();
                for key in keys {
                    monitor.reconcile_group(&key);
                }
            }
            Ok(WorkerEvent::StatusChanged {
                worker,
                new_status,
                old_status: _,
                ..
            }) => {
                if new_status != WorkerStatus::Ready {
                    monitor.evict_worker_loads(worker.url());
                }
                // No action needed when transitioning *into* Ready: the
                // group's polling loop reads from the registry on every
                // tick and will pick the worker up automatically.
            }
            Err(broadcast::error::RecvError::Lagged(n)) => {
                warn!(
                    skipped = n,
                    "WorkerMonitor lagged behind registry events; rebuilding from snapshot"
                );
                monitor.reconcile_from_registry();
            }
            Err(broadcast::error::RecvError::Closed) => {
                debug!("WorkerMonitor event channel closed; exiting event loop");
                return;
            }
        }
        // Drop the temporary strong reference so we do not keep the
        // monitor alive while parked on the next `recv().await`.
        drop(monitor);
    }
}

/// Polling loop body for a single worker group.
///
/// Holds a `Weak<WorkerMonitor>` for the same cycle-breaking reason
/// as `run_event_loop`. The temporary `Arc` is upgraded after the
/// timer tick and dropped before the next tick so the monitor's
/// `Drop` is reachable.
async fn group_monitor_loop(
    monitor: Weak<WorkerMonitor>,
    group_key: WorkerGroupKey,
    interval: Duration,
) {
    let mut interval_timer = tokio::time::interval(interval);

    loop {
        interval_timer.tick().await;

        let Some(monitor) = monitor.upgrade() else {
            debug!("WorkerMonitor was dropped; exiting group loop for {group_key}");
            return;
        };

        let power_of_two_policies = monitor.policy_registry.get_all_power_of_two_policies();
        if power_of_two_policies.is_empty()
            && monitor.policy_registry.get_dp_rank_policy().is_none()
        {
            debug!("No load-aware policies, skipping load fetch for group {group_key}");
            drop(monitor);
            continue;
        }

        // Only poll Ready workers — Pending/NotReady/Failed do not
        // serve traffic and should not contribute load samples.
        let workers: Vec<Arc<dyn Worker>> = monitor
            .worker_registry
            .get_workers_filtered(
                Some(&group_key.model_id),
                Some(group_key.worker_type),
                Some(group_key.connection_mode),
                None,
                false,
            )
            .into_iter()
            .filter(|w| w.status() == WorkerStatus::Ready)
            .collect();

        if workers.is_empty() {
            debug!("No Ready workers in group {group_key}, skipping");
            drop(monitor);
            continue;
        }

        let futures: Vec<_> = workers
            .iter()
            .map(|worker| {
                let client = monitor.client.clone();
                let worker = Arc::clone(worker);
                let connection_mode = group_key.connection_mode;
                async move {
                    let response = match connection_mode {
                        ConnectionMode::Http => {
                            WorkerMonitor::fetch_http_load(&client, &worker).await
                        }
                        ConnectionMode::Grpc => WorkerMonitor::fetch_grpc_load(&worker).await,
                    };
                    (worker.url().to_string(), response)
                }
            })
            .collect();

        let results = future::join_all(futures).await;

        let mut group_loads: HashMap<String, WorkerLoadResponse> = HashMap::new();
        let mut group_dp_loads: HashMap<String, HashMap<isize, isize>> = HashMap::new();
        for (url, response) in results {
            if let Some(load) = response {
                group_loads.insert(url.clone(), load.clone());
                let dp_rank_loads = load.dp_rank_loads();
                group_dp_loads.insert(url, dp_rank_loads);
            }
        }

        // Compute the URL set up front so both the success and
        // empty-fetch branches can prune stale entries from the watch
        // snapshot. Without the empty-fetch prune, a group that
        // starts timing out keeps publishing its previous tick's
        // loads forever — subscribers see a stale snapshot indefinitely.
        let all_group_urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();

        if group_loads.is_empty() {
            debug!("No loads fetched for group {group_key}, pruning stale entries");
            monitor.load_tx.send_modify(|map| {
                for url in &all_group_urls {
                    map.remove(url);
                }
            });
            // The DP cache deliberately keeps last-known-good entries
            // so routing decisions still have a hint to fall back to
            // when the upstream is briefly unreachable.
            drop(monitor);
            continue;
        }

        debug!(
            "Fetched loads from {}/{} workers in group {group_key}",
            group_loads.len(),
            workers.len()
        );

        for policy in &power_of_two_policies {
            policy.update_loads(&group_loads);
        }
        monitor.worker_load_manager.update_dp_loads(&group_dp_loads);

        // Atomically merge into the shared watch channel: clear stale
        // entries for *this group's* URLs first, then insert the fresh
        // loads. Workers that failed this tick get their stale entries
        // pruned along with the rest.
        monitor.load_tx.send_modify(|map| {
            for url in &all_group_urls {
                map.remove(url);
            }
            map.extend(group_loads);
        });

        // Drop the temporary strong reference so we do not keep the
        // monitor alive across the next `interval_timer.tick().await`.
        drop(monitor);
    }
}

#[cfg(test)]
mod worker_load_manager_tests {
    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    #[test]
    fn test_new_dp_load_manager_instance() {
        let dp_load_manager = WorkerLoadManager::new();
        let cached = dp_load_manager.dp_cached_loads.read();
        assert!(cached.is_empty());
    }

    #[test]
    fn test_update_dp_load() {
        let manager = WorkerLoadManager::new();
        let mut loads = HashMap::new();

        let mut worker1_load = HashMap::new();
        worker1_load.insert(0, 2);
        worker1_load.insert(1, 1);
        loads.insert("http://worker1:8080".to_string(), worker1_load);

        let mut worker2_load = HashMap::new();
        worker2_load.insert(0, 3);
        loads.insert("http://worker2:8080".to_string(), worker2_load);

        manager.update_dp_loads(&loads);

        let cached = manager.dp_cached_loads.read();
        assert_eq!(cached.len(), 2);

        let worker2_cache = cached.get("http://worker2:8080").unwrap();
        assert_eq!(worker2_cache.get(&0), Some(&3));
    }

    #[test]
    fn test_select_and_increment_lowest_dp_load_multiple() {
        let worker = BasicWorkerBuilder::new("http://worker:8080")
            .worker_type(WorkerType::Regular)
            .api_key("test_key")
            .build();

        let manager = WorkerLoadManager::new();
        let mut loads = HashMap::new();
        let mut worker_load = HashMap::new();
        worker_load.insert(0, 10);
        worker_load.insert(1, 3);
        worker_load.insert(2, 7);
        loads.insert(worker.url().to_string(), worker_load);
        manager.update_dp_loads(&loads);

        let selected = manager.select_and_increment_lowest_dp_load(&worker, 4);

        assert_eq!(selected, Some(1));
        let cached = manager.dp_cached_loads.read();
        assert_eq!(*cached.get(worker.url()).unwrap().get(&1).unwrap(), 3 + 4);
    }

    #[test]
    fn test_select_and_increment_lowest_dp_load_none_worker() {
        let worker = BasicWorkerBuilder::new("http://nonexist:8080")
            .worker_type(WorkerType::Regular)
            .api_key("test")
            .build();

        let manager = WorkerLoadManager::new();
        let result = manager.select_and_increment_lowest_dp_load(&worker, 1);
        assert_eq!(result, None);
    }
}

#[cfg(test)]
mod worker_monitor_tests {
    use std::collections::HashMap;

    use openai_protocol::{
        model_card::ModelCard,
        worker::{HealthCheckConfig, WorkerStatus},
    };

    use super::*;
    use crate::{
        config::types::PolicyConfig,
        policies::PolicyRegistry,
        worker::{BasicWorkerBuilder, ConnectionMode, WorkerType},
    };

    fn ready_worker(url: &str, model: &str) -> Arc<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Http)
            .model(ModelCard::new(model))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        worker.set_status(WorkerStatus::Ready);
        Arc::new(worker)
    }

    fn build_monitor() -> (Arc<WorkerRegistry>, Arc<WorkerMonitor>) {
        let registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        let monitor = Arc::new(WorkerMonitor::new(
            registry.clone(),
            policy_registry,
            reqwest::Client::new(),
            5,
        ));
        (registry, monitor)
    }

    #[tokio::test]
    async fn bootstrap_reconcile_starts_loops_for_existing_workers() {
        let (registry, monitor) = build_monitor();
        registry
            .register(ready_worker("http://w1:8080", "llama-3"))
            .unwrap();
        registry
            .register(ready_worker("http://w2:8080", "llama-3"))
            .unwrap();
        registry
            .register(ready_worker("http://w3:8080", "gpt-4"))
            .unwrap();

        monitor.start_event_loop();

        // Two model groups should now have polling loops.
        let handles = monitor.group_handles.lock();
        assert_eq!(handles.len(), 2);
        let keys: Vec<&WorkerGroupKey> = handles.keys().collect();
        assert!(keys.iter().any(|k| k.model_id == "llama-3"));
        assert!(keys.iter().any(|k| k.model_id == "gpt-4"));
    }

    #[tokio::test]
    async fn registered_event_starts_a_new_group() {
        let (registry, monitor) = build_monitor();
        monitor.start_event_loop();

        // Registry was empty at bootstrap, so no groups yet.
        assert!(monitor.group_handles.lock().is_empty());

        registry
            .register(ready_worker("http://w:8080", "llama-3"))
            .unwrap();

        // Give the event loop a moment to process the broadcast.
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(50)).await;

        let handles = monitor.group_handles.lock();
        assert_eq!(handles.len(), 1);
        assert!(handles
            .keys()
            .any(|k| k.model_id == "llama-3" && k.worker_type == WorkerType::Regular));
    }

    #[tokio::test]
    async fn removed_event_stops_empty_group() {
        let (registry, monitor) = build_monitor();
        let id = registry
            .register(ready_worker("http://w:8080", "llama-3"))
            .unwrap();
        monitor.start_event_loop();
        assert_eq!(monitor.group_handles.lock().len(), 1);

        registry.remove(&id);
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert!(monitor.group_handles.lock().is_empty());
    }

    #[tokio::test]
    async fn removed_event_evicts_cached_loads() {
        // Regression test for the bug where a removed worker's last
        // known load would persist forever in `load_tx` and the DP
        // cache because the polling loop that pruned it had been
        // stopped by the same removal.
        let (registry, monitor) = build_monitor();
        let worker = ready_worker("http://w:8080", "llama-3");
        let url = worker.url().to_string();
        let id = registry.register(worker).unwrap();
        monitor.start_event_loop();

        monitor.load_tx.send_modify(|map| {
            map.insert(url.clone(), WorkerLoadResponse::default());
        });
        let mut dp_loads: HashMap<String, HashMap<isize, isize>> = HashMap::new();
        let mut inner = HashMap::new();
        inner.insert(0, 5);
        dp_loads.insert(url.clone(), inner);
        monitor.worker_load_manager.update_dp_loads(&dp_loads);

        registry.remove(&id);
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(50)).await;

        let snapshot = monitor.load_rx.borrow().clone();
        assert!(
            !snapshot.contains_key(&url),
            "load_tx must not retain entries for removed workers"
        );
        let cached = monitor.worker_load_manager.dp_cached_loads.read();
        assert!(
            !cached.contains_key(&url),
            "DP cache must not retain entries for removed workers"
        );
    }

    #[tokio::test]
    async fn stop_all_groups_clears_dp_cache() {
        // Regression test for the bug where a `RecvError::Lagged`
        // rebuild would leave stale DP cache entries because
        // stop_all_groups only cleared the watch snapshot.
        let (registry, monitor) = build_monitor();
        registry
            .register(ready_worker("http://w1:8080", "llama-3"))
            .unwrap();

        let mut dp_loads: HashMap<String, HashMap<isize, isize>> = HashMap::new();
        let mut inner = HashMap::new();
        inner.insert(0, 7);
        dp_loads.insert("http://w1:8080".to_string(), inner);
        monitor.worker_load_manager.update_dp_loads(&dp_loads);

        monitor.stop_all_groups();

        assert!(monitor.load_rx.borrow().is_empty());
        assert!(monitor
            .worker_load_manager
            .dp_cached_loads
            .read()
            .is_empty());
    }

    #[tokio::test]
    async fn status_changed_away_from_ready_evicts_worker() {
        let (registry, monitor) = build_monitor();
        let worker = ready_worker("http://w:8080", "llama-3");
        let url = worker.url().to_string();
        let id = registry.register(worker).unwrap();
        monitor.start_event_loop();

        // Seed the watch channel + DP cache as if a poll had succeeded.
        monitor.load_tx.send_modify(|map| {
            map.insert(url.clone(), WorkerLoadResponse::default());
        });
        let mut dp_loads: HashMap<String, HashMap<isize, isize>> = HashMap::new();
        let mut inner = HashMap::new();
        inner.insert(0, 5);
        dp_loads.insert(url.clone(), inner);
        monitor.worker_load_manager.update_dp_loads(&dp_loads);

        registry.transition_status(&id, WorkerStatus::NotReady);
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Watch channel entry pruned.
        let snapshot = monitor.load_rx.borrow().clone();
        assert!(!snapshot.contains_key(&url));

        // DP cache entry pruned.
        let cached = monitor.worker_load_manager.dp_cached_loads.read();
        assert!(!cached.contains_key(&url));
    }

    #[tokio::test]
    async fn spawned_tasks_use_weak_references() {
        // Regression test for the Arc cycle bug: the spawned event
        // loop and the per-group polling loops must hold
        // `Weak<WorkerMonitor>` rather than `Arc<WorkerMonitor>`. If
        // they held strong references, the cycle through
        // `event_task` / `group_handles` would keep the strong count
        // pinned and `Drop` would never fire outside of full runtime
        // shutdown.
        let (registry, monitor) = build_monitor();
        registry
            .register(ready_worker("http://w:8080", "llama-3"))
            .unwrap();
        monitor.start_event_loop();

        // Bootstrap should have started a polling loop for the
        // worker's group, on top of the event task itself.
        assert_eq!(monitor.group_handles.lock().len(), 1);

        // After spawning, only the test's `Arc` should be strong:
        // both the event task and the group loop downgraded to
        // `Weak<Self>`.
        assert_eq!(
            Arc::strong_count(&monitor),
            1,
            "spawned tasks must not hold strong references to the monitor"
        );
        assert!(
            Arc::weak_count(&monitor) >= 2,
            "expected at least one Weak from the event task and one from the group loop"
        );
    }
}
