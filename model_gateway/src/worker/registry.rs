//! Worker Registry for multi-router support
//!
//! Provides centralized registry for workers with model-based indexing.
//!
//! # Performance Optimizations
//! The model index uses immutable Arc snapshots instead of RwLock for lock-free reads.
//! This is critical for high-concurrency scenarios where many requests query the same model.
//!
//! # Consistent Hash Ring
//! The registry maintains a pre-computed [`HashRing`] per model for O(log n)
//! consistent hashing. The ring is rebuilt only when workers are added or
//! removed, not per request. See [`crate::worker::hash_ring`] for the ring
//! itself — this file only wires registry events to ring rebuilds.

use std::{collections::HashSet, sync::Arc};

use dashmap::{mapref::entry::Entry, DashMap};
use openai_protocol::worker::WorkerStatus;
use parking_lot::RwLock;
use smg_mesh::OptionalMeshSyncManager;
use tokio::sync::broadcast;
use uuid::Uuid;

use crate::{
    config::types::RetryConfig,
    observability::metrics::Metrics,
    worker::{
        circuit_breaker::CircuitState,
        event::WorkerEvent,
        hash_ring::HashRing,
        worker::{RuntimeType, WorkerType},
        ConnectionMode, Worker,
    },
};

/// Unique identifier for a worker
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct WorkerId(String);

impl WorkerId {
    /// Create a new worker ID
    pub fn new() -> Self {
        Self(Uuid::now_v7().to_string())
    }

    /// Create a worker ID from a string
    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    /// Get the ID as a string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for WorkerId {
    fn default() -> Self {
        Self::new()
    }
}

/// Side-effect-free worker snapshot for subscriber bootstrap or lag recovery.
#[derive(Debug, Clone)]
pub struct WorkerDescriptor {
    pub worker_id: WorkerId,
    pub status: WorkerStatus,
    pub disable_health_check: bool,
    pub check_interval_secs: u64,
}

/// Model index using immutable snapshots for lock-free reads.
/// Each model maps to an Arc'd slice of workers that can be read without locking.
/// Updates create new snapshots (copy-on-write semantics).
type ModelIndex = Arc<DashMap<String, Arc<[Arc<dyn Worker>]>>>;

/// Worker registry with model-based indexing
#[derive(Debug)]
pub struct WorkerRegistry {
    /// All workers indexed by ID
    workers: Arc<DashMap<WorkerId, Arc<dyn Worker>>>,

    /// Model index for O(1) lookups using immutable snapshots.
    /// Uses Arc<[T]> instead of Arc<RwLock<Vec<T>>> for lock-free reads.
    model_index: ModelIndex,

    /// Consistent hash rings per model for O(log n) routing.
    /// Rebuilt on worker add/remove (copy-on-write).
    hash_rings: Arc<DashMap<String, Arc<HashRing>>>,

    /// Workers indexed by worker type
    type_workers: Arc<DashMap<WorkerType, Vec<WorkerId>>>,

    /// Workers indexed by connection mode
    connection_workers: Arc<DashMap<ConnectionMode, Vec<WorkerId>>>,

    /// URL to worker ID mapping
    url_to_id: Arc<DashMap<String, WorkerId>>,

    /// Per-worker-ID locks for serializing replace() operations.
    /// Only held during the in-memory model index diff (no I/O, microseconds).
    worker_mutation_locks: Arc<DashMap<WorkerId, Arc<parking_lot::Mutex<()>>>>,

    /// Optional mesh sync manager for state synchronization
    /// When None, the registry works independently without mesh synchronization
    /// Uses RwLock for thread-safe access when setting mesh_sync after initialization
    mesh_sync: Arc<RwLock<OptionalMeshSyncManager>>,

    /// Per-model retry config (last write wins).
    /// Updated when a worker with non-empty retry overrides registers.
    /// Cleaned up when the last worker for a model is removed.
    /// When retries are disabled, max_retries is set to 1.
    model_retry_configs: Arc<DashMap<String, RetryConfig>>,

    /// Broadcast channel for worker state change events.
    event_tx: broadcast::Sender<WorkerEvent>,
}

impl WorkerRegistry {
    // ───────────────────────────────────────────────────────────────────
    // 1. Construction & subscription
    // ───────────────────────────────────────────────────────────────────

    /// Create an empty worker registry.
    ///
    /// Initialises all indexes and a broadcast channel with capacity 64
    /// for `WorkerEvent` delivery. Holds no locks. Emits no events.
    pub fn new() -> Self {
        Self {
            workers: Arc::new(DashMap::new()),
            model_index: Arc::new(DashMap::new()),
            hash_rings: Arc::new(DashMap::new()),
            type_workers: Arc::new(DashMap::new()),
            connection_workers: Arc::new(DashMap::new()),
            url_to_id: Arc::new(DashMap::new()),
            worker_mutation_locks: Arc::new(DashMap::new()),
            mesh_sync: Arc::new(RwLock::new(None)),
            model_retry_configs: Arc::new(DashMap::new()),
            event_tx: broadcast::Sender::new(64),
        }
    }

    /// Subscribe to the `WorkerEvent` broadcast stream.
    ///
    /// Returns a `broadcast::Receiver` that observes every future mutation
    /// event emitted by `register` / `replace` / `remove` / `transition_status`.
    /// Late subscribers miss past events — callers that need historical
    /// state should combine this with [`Self::reconcile_snapshot`] on startup
    /// and on `RecvError::Lagged`. Holds no locks. Emits no events.
    pub fn subscribe_events(&self) -> broadcast::Receiver<WorkerEvent> {
        self.event_tx.subscribe()
    }

    // ───────────────────────────────────────────────────────────────────
    // 2. Read — single worker
    // ───────────────────────────────────────────────────────────────────

    /// Look up a worker by its ID.
    ///
    /// Returns `Some(worker)` when the ID exists, `None` otherwise.
    /// Read-only, lock-free. Emits no events.
    pub fn get(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        self.workers.get(worker_id).map(|entry| entry.clone())
    }

    /// Look up a worker by its URL.
    ///
    /// Returns `Some(worker)` when a worker with this URL is registered,
    /// `None` otherwise. Read-only, lock-free. Emits no events.
    pub fn get_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        self.url_to_id.get(url).and_then(|id| self.get(&id))
    }

    /// Reverse-lookup the URL for a given worker ID.
    ///
    /// Prefers the URL stored on the live worker object; falls back to
    /// scanning `url_to_id` so pre-reserved IDs (from
    /// [`Self::reserve_id_for_url`]) can still be resolved before a worker
    /// is installed. Read-only, lock-free. Emits no events.
    pub fn get_url_by_id(&self, worker_id: &WorkerId) -> Option<String> {
        if let Some(worker) = self.get(worker_id) {
            return Some(worker.url().to_string());
        }
        self.url_to_id
            .iter()
            .find_map(|entry| (entry.value() == worker_id).then(|| entry.key().clone()))
    }

    /// Get the consistent hash ring for a model (O(1) lookup).
    ///
    /// Returns `Some(ring)` if any workers are registered for this model,
    /// `None` otherwise. The ring is pre-built and updated on worker add
    /// or remove, so reads are allocation-free apart from the Arc clone.
    /// Read-only, lock-free. Emits no events.
    pub fn get_hash_ring(&self, model_id: &str) -> Option<Arc<HashRing>> {
        self.hash_rings.get(model_id).map(|r| Arc::clone(&r))
    }

    // ───────────────────────────────────────────────────────────────────
    // 3. Read — collections
    // ───────────────────────────────────────────────────────────────────

    /// Empty worker slice constant returned when a lookup has no matches.
    const EMPTY_WORKERS: &'static [Arc<dyn Worker>] = &[];

    /// Return all workers serving a model as an immutable shared slice.
    ///
    /// This is the fastest possible read path: the model index already
    /// stores the slice as an `Arc<[_]>`, so the return value is just an
    /// atomic refcount bump with zero contention. Returns an empty shared
    /// slice when the model is unknown. Read-only, lock-free. Emits no
    /// events.
    pub fn get_by_model(&self, model_id: &str) -> Arc<[Arc<dyn Worker>]> {
        self.model_index
            .get(model_id)
            .map(|workers| Arc::clone(&workers))
            .unwrap_or_else(|| Arc::from(Self::EMPTY_WORKERS))
    }

    /// Return all workers of a given type as an immutable shared slice.
    ///
    /// Unified with [`Self::get_by_model`] on `Arc<[_]>` so callers can
    /// treat all worker collections uniformly. Builds a fresh slice per
    /// call (one boxed-slice allocation). Read-only, lock-free. Emits no
    /// events.
    pub fn get_by_type(&self, worker_type: WorkerType) -> Arc<[Arc<dyn Worker>]> {
        let workers: Vec<Arc<dyn Worker>> = self
            .type_workers
            .get(&worker_type)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default();
        Arc::from(workers.into_boxed_slice())
    }

    /// Return all workers using a given connection mode (HTTP or gRPC).
    ///
    /// Returned as an immutable shared slice for uniformity with the other
    /// collection getters. Builds a fresh slice per call. Read-only,
    /// lock-free. Emits no events.
    pub fn get_by_connection(&self, connection_mode: ConnectionMode) -> Arc<[Arc<dyn Worker>]> {
        let workers: Vec<Arc<dyn Worker>> = self
            .connection_workers
            .get(&connection_mode)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default();
        Arc::from(workers.into_boxed_slice())
    }

    /// Return every prefill worker, regardless of which model they serve.
    ///
    /// Thin wrapper over [`Self::get_by_type`] with `WorkerType::Prefill`.
    /// Read-only, lock-free. Emits no events.
    pub fn get_prefill_workers(&self) -> Arc<[Arc<dyn Worker>]> {
        self.get_by_type(WorkerType::Prefill)
    }

    /// Return every decode worker, regardless of which model they serve.
    ///
    /// Thin wrapper over [`Self::get_by_type`] with `WorkerType::Decode`.
    /// Read-only, lock-free. Emits no events.
    pub fn get_decode_workers(&self) -> Arc<[Arc<dyn Worker>]> {
        self.get_by_type(WorkerType::Decode)
    }

    /// Return workers matching every supplied filter.
    ///
    /// Filters:
    /// - `model_id`: scope to a single model (uses the O(1) model index)
    /// - `worker_type`: `Regular` / `Prefill` / `Decode`
    /// - `connection_mode`: `Http` / `Grpc`
    /// - `runtime_type`: `Sglang` / `Vllm` / `External` / …
    /// - `healthy_only`: skip workers whose `is_healthy()` is false
    ///
    /// **Cost note on `runtime_type`:** the registry keeps no runtime-type
    /// index. This filter is applied in-memory after fetching by model or
    /// iterating all workers, so the whole candidate set is cloned before
    /// filtering. Callers on hot paths should prefer pre-filtering by
    /// model or type when possible.
    ///
    /// Always returns an owned `Vec` because each call applies a unique
    /// filter combination. Read-only, lock-free. Emits no events.
    pub fn get_workers_filtered(
        &self,
        model_id: Option<&str>,
        worker_type: Option<WorkerType>,
        connection_mode: Option<ConnectionMode>,
        runtime_type: Option<RuntimeType>,
        healthy_only: bool,
    ) -> Vec<Arc<dyn Worker>> {
        // Start with the most efficient collection based on filters
        // Use model index when possible as it's O(1) lookup
        let workers: Vec<Arc<dyn Worker>> = if let Some(model) = model_id {
            self.get_by_model(model).to_vec()
        } else {
            self.get_all()
        };

        workers
            .into_iter()
            .filter(|w| {
                if let Some(ref wtype) = worker_type {
                    if *w.worker_type() != *wtype {
                        return false;
                    }
                }
                if let Some(ref conn) = connection_mode {
                    if w.connection_mode() != conn {
                        return false;
                    }
                }
                if let Some(ref rt) = runtime_type {
                    if w.metadata().spec.runtime_type != *rt {
                        return false;
                    }
                }
                if healthy_only && !w.is_healthy() {
                    return false;
                }
                true
            })
            .collect()
    }

    /// Return an owned snapshot of every registered worker.
    ///
    /// Allocates a fresh `Vec` by cloning each Arc. Intended for cold
    /// paths (admin endpoints, diagnostics). Hot routing paths should
    /// prefer [`Self::get_by_model`]. Read-only, lock-free. Emits no
    /// events.
    pub fn get_all(&self) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Return every worker paired with its ID.
    ///
    /// Used by bootstrap/reconcile paths that need to correlate workers
    /// with their IDs. Read-only, lock-free. Emits no events.
    pub fn get_all_with_ids(&self) -> Vec<(WorkerId, Arc<dyn Worker>)> {
        self.workers
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Return every worker's URL as a freshly allocated `Vec`.
    ///
    /// Used by admin endpoints and tests. Read-only, lock-free. Emits no
    /// events.
    pub fn get_all_urls(&self) -> Vec<String> {
        self.workers
            .iter()
            .map(|entry| entry.value().url().to_string())
            .collect()
    }

    /// Return every worker's URL paired with its optional API key.
    ///
    /// Used by the gateway when proxying to upstream workers that require
    /// per-worker credentials. Read-only, lock-free. Emits no events.
    pub fn get_all_urls_with_api_key(&self) -> Vec<(String, Option<String>)> {
        self.workers
            .iter()
            .map(|entry| {
                (
                    entry.value().url().to_string(),
                    entry.value().api_key().cloned(),
                )
            })
            .collect()
    }

    /// Return a side-effect-free descriptor snapshot for reconcile paths.
    ///
    /// Each `WorkerDescriptor` captures the fields a subscriber needs to
    /// rebuild its in-memory state from scratch (e.g. health scheduling
    /// after `RecvError::Lagged`) without re-reading the worker objects.
    /// Read-only, lock-free. Emits no events.
    pub fn reconcile_snapshot(&self) -> Vec<WorkerDescriptor> {
        self.workers
            .iter()
            .map(|entry| {
                let worker = entry.value();
                WorkerDescriptor {
                    worker_id: entry.key().clone(),
                    status: worker.status(),
                    disable_health_check: worker.metadata().health_config.disable_health_check,
                    check_interval_secs: worker.metadata().health_config.check_interval_secs,
                }
            })
            .collect()
    }

    /// Return the set of model IDs that currently have at least one
    /// worker serving them.
    ///
    /// Skips model entries whose worker slice has become empty (those are
    /// eventually evicted by the removal path). Read-only, lock-free.
    /// Emits no events.
    pub fn get_models(&self) -> Vec<String> {
        self.model_index
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Return the number of registered workers.
    ///
    /// Read-only, lock-free. Emits no events.
    pub fn len(&self) -> usize {
        self.workers.len()
    }

    /// Return `true` when no workers are registered.
    ///
    /// Read-only, lock-free. Emits no events.
    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    /// Return a consolidated snapshot of registry statistics.
    ///
    /// Iterates the `workers` map once, counting totals per worker type,
    /// connection mode, circuit-breaker state, and health status. Used by
    /// `/v1/stats` and monitoring dashboards. Read-only, lock-free.
    /// Emits no events.
    pub fn stats(&self) -> WorkerRegistryStats {
        let total_workers = self.workers.len();
        // Count models directly instead of allocating Vec via get_models() (lock-free)
        let total_models = self
            .model_index
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .count();

        let mut healthy_count = 0;
        let mut total_load = 0;
        let mut regular_count = 0;
        let mut prefill_count = 0;
        let mut decode_count = 0;
        let mut http_count = 0;
        let mut grpc_count = 0;
        let mut cb_open_count = 0;
        let mut cb_half_open_count = 0;

        // Iterate DashMap directly to avoid cloning all workers via get_all()
        for entry in self.workers.iter() {
            let worker = entry.value();
            if worker.is_healthy() {
                healthy_count += 1;
            }
            total_load += worker.load();

            match worker.worker_type() {
                WorkerType::Regular => regular_count += 1,
                WorkerType::Prefill => prefill_count += 1,
                WorkerType::Decode => decode_count += 1,
            }

            match worker.connection_mode() {
                ConnectionMode::Http => http_count += 1,
                ConnectionMode::Grpc => grpc_count += 1,
            }

            match worker.circuit_breaker_state() {
                CircuitState::Open => cb_open_count += 1,
                CircuitState::HalfOpen => cb_half_open_count += 1,
                CircuitState::Closed => {}
            }
        }

        WorkerRegistryStats {
            total_workers,
            total_models,
            healthy_workers: healthy_count,
            unhealthy_workers: total_workers.saturating_sub(healthy_count),
            total_load,
            regular_workers: regular_count,
            prefill_workers: prefill_count,
            decode_workers: decode_count,
            http_workers: http_count,
            grpc_workers: grpc_count,
            circuit_breaker_open: cb_open_count,
            circuit_breaker_half_open: cb_half_open_count,
        }
    }

    /// Return `(regular_count, pd_count)` using the type index directly.
    ///
    /// Avoids allocating the full worker list the way [`Self::stats`] does.
    /// `pd_count` is any worker that is not `Regular`. Read-only, lock-free.
    /// Emits no events.
    pub fn get_worker_distribution(&self) -> (usize, usize) {
        // Use the existing type_workers index for O(1) lookup
        let regular_count = self
            .type_workers
            .get(&WorkerType::Regular)
            .map(|v| v.len())
            .unwrap_or(0);

        let total_workers = self.workers.len();
        let pd_count = total_workers.saturating_sub(regular_count);

        (regular_count, pd_count)
    }

    // ───────────────────────────────────────────────────────────────────
    // 4. Read — config
    // ───────────────────────────────────────────────────────────────────

    /// Get the per-model retry config override, if any.
    ///
    /// Returns `None` when no worker in this model group set a retry
    /// override. When retries are disabled for the group, the stored
    /// `max_retries` is always 1. Read-only, lock-free. Emits no events.
    pub fn get_retry_config(&self, model_id: &str) -> Option<RetryConfig> {
        self.model_retry_configs
            .get(model_id)
            .map(|entry| entry.value().clone())
    }

    // ───────────────────────────────────────────────────────────────────
    // 5. Write — mutation primitives
    //
    // Every method in this section holds the per-worker mutation lock
    // (`worker_mutation_locks`) and emits exactly one `WorkerEvent` before
    // releasing the lock. New mutation methods MUST follow this pattern.
    // Manual publish at each call site is intentional — there are only a
    // handful of mutation methods, and the simplicity beats a generic
    // helper layer.
    // ───────────────────────────────────────────────────────────────────

    /// Register a new worker (create-only).
    ///
    /// Returns the new `WorkerId` on success, or `None` if a worker with
    /// the same URL is already registered and active. A URL that was
    /// pre-reserved via [`Self::reserve_id_for_url`] but has no worker yet
    /// is treated as a new registration (reuses the reserved ID).
    ///
    /// Emits [`WorkerEvent::Registered`] on success. Holds the per-worker
    /// mutation lock for the entire `register_inner` call — the index
    /// updates, mesh sync, and event broadcast all run under the same
    /// lock so subscribers cannot observe `Removed` / `Replaced` /
    /// `StatusChanged` events before the `Registered` event for a
    /// concurrent same-ID operation.
    pub fn register(&self, worker: Arc<dyn Worker>) -> Option<WorkerId> {
        self.register_inner(worker, true)
    }

    /// Register or replace a worker (upsert).
    ///
    /// Returns the resulting `WorkerId`. Used by internal callers (K8s
    /// discovery, startup) that need idempotent registration. If the URL
    /// is new (or pre-reserved), behaves like [`Self::register`] and emits
    /// [`WorkerEvent::Registered`]. If the URL already has an active
    /// worker, delegates to [`Self::replace`] and emits
    /// [`WorkerEvent::Replaced`].
    ///
    /// Holds the per-worker mutation lock for the duration of the
    /// underlying `register` or `replace` call.
    pub fn register_or_replace(&self, worker: Arc<dyn Worker>) -> WorkerId {
        // Try to create first — succeeds for fresh URLs and pre-reserved IDs
        // (where url_to_id has an entry but workers does not).
        if let Some(id) = self.register(worker.clone()) {
            return id;
        }

        // URL exists with an active worker — replace it
        if let Some(existing_id) = self.url_to_id.get(worker.url()).map(|e| e.clone()) {
            if !self.replace(&existing_id, worker) {
                // replace() returned false — worker was removed concurrently.
                // The mutation lock prevents stale indexes, so this is safe to ignore.
                tracing::warn!(
                    "register_or_replace: worker {} was removed during replace",
                    existing_id.as_str()
                );
            }
            return existing_id;
        }

        // Should not reach here: register() returned None means URL is in url_to_id.
        // Recover by clearing the stale entry and retrying full registration.
        tracing::error!(
            "register_or_replace: inconsistent state for URL {}, clearing stale entry",
            worker.url()
        );
        self.url_to_id.remove(worker.url());
        // register() will now succeed since we cleared the entry.
        // If it still fails, something is deeply wrong — return a default ID.
        self.register(worker).unwrap_or_default()
    }

    /// Replace an existing worker with a new one (overwrite-then-diff).
    ///
    /// Used by `PUT /workers/{id}` and K8s discovery when a worker with
    /// the same URL already exists. Updates the worker object in-place and
    /// diffs the model index to avoid a transient gap where the worker is
    /// missing from indexes.
    ///
    /// Returns `true` if the worker was replaced, `false` if the ID was
    /// not found or the URL would change (URL changes require
    /// remove + register instead).
    ///
    /// Emits [`WorkerEvent::Replaced`] on success. Holds the per-worker
    /// mutation lock for the entire diff + broadcast sequence.
    pub fn replace(&self, worker_id: &WorkerId, new_worker: Arc<dyn Worker>) -> bool {
        // Serialize concurrent replacements for the same worker ID.
        // Lock is held only during the in-memory diff (no I/O, microseconds).
        let lock = self
            .worker_mutation_locks
            .entry(worker_id.clone())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone();
        let _guard = lock.lock();

        let old_worker = match self.workers.get(worker_id) {
            Some(entry) => entry.clone(),
            None => return false,
        };

        let old_models: HashSet<String> = Self::worker_model_ids(&old_worker).into_iter().collect();
        let new_models: HashSet<String> = Self::worker_model_ids(&new_worker).into_iter().collect();

        // URL changes are not supported via replace — use remove + register instead
        if old_worker.url() != new_worker.url() {
            tracing::error!(
                old_url = old_worker.url(),
                new_url = new_worker.url(),
                "replace() does not support URL changes"
            );
            return false;
        }

        if !new_worker.inherit_shared_state_from(&*old_worker) {
            tracing::warn!(
                worker_id = %worker_id.as_str(),
                worker_url = old_worker.url(),
                "replace() did not preserve shared mutable worker state"
            );
        }

        // Overwrite worker object atomically
        self.workers.insert(worker_id.clone(), new_worker.clone());

        // Diff model indexes: remove stale, add new
        for removed_model in old_models.difference(&new_models) {
            self.remove_worker_from_model_index(removed_model, old_worker.url());
            // Mirror `remove()`: drop any per-model retry override when
            // the replacement leaves the model with no workers. Without
            // this, `get_retry_config()` would keep returning a stale
            // override for a model that is no longer served.
            let model_empty = self
                .model_index
                .get(removed_model)
                .is_none_or(|workers| workers.is_empty());
            if model_empty {
                self.model_retry_configs.remove(removed_model);
            }
        }
        for added_model in new_models.difference(&old_models) {
            self.add_worker_to_model_index(added_model, new_worker.clone());
            self.rebuild_hash_ring(added_model);
        }
        // For models that stayed the same, update the worker reference in the index
        for kept_model in old_models.intersection(&new_models) {
            self.add_worker_to_model_index(kept_model, new_worker.clone());
            self.rebuild_hash_ring(kept_model);
        }

        if old_worker.worker_type() != new_worker.worker_type() {
            if let Some(mut type_workers) = self.type_workers.get_mut(old_worker.worker_type()) {
                type_workers.retain(|id| id != worker_id);
            }
            self.type_workers
                .entry(*new_worker.worker_type())
                .or_default()
                .push(worker_id.clone());
        }

        if old_worker.connection_mode() != new_worker.connection_mode() {
            if let Some(mut conn_workers) = self
                .connection_workers
                .get_mut(old_worker.connection_mode())
            {
                conn_workers.retain(|id| id != worker_id);
            }
            self.connection_workers
                .entry(*new_worker.connection_mode())
                .or_default()
                .push(worker_id.clone());
        }

        // Sync to mesh if enabled (no-op if mesh is not enabled)
        {
            let guard = self.mesh_sync.read();
            if let Some(ref mesh_sync) = *guard {
                mesh_sync.sync_worker_state(
                    worker_id.as_str().to_string(),
                    new_worker.model_id().to_string(),
                    new_worker.url().to_string(),
                    new_worker.is_healthy(),
                    0.0,
                    bincode::serialize(&new_worker.metadata().spec).unwrap_or_default(),
                );
            }
        }

        let _ = self.event_tx.send(WorkerEvent::Replaced {
            worker_id: worker_id.clone(),
            old: old_worker,
            new: new_worker,
        });

        true
    }

    /// Atomically transition a worker's lifecycle status and emit a
    /// `StatusChanged` event if it actually changed.
    ///
    /// This is a pure mutation primitive — the registry has no opinion on
    /// when a worker should transition. The caller (typically
    /// `WorkerManager`) owns the state machine logic.
    ///
    /// The per-worker mutation lock guarantees:
    ///   1. The status read, write, and event emission are atomic per
    ///      worker.
    ///   2. Two concurrent calls cannot interleave to publish events out
    ///      of order for the same worker.
    ///
    /// Returns `Some((old, new))` if the status changed, `None` if the
    /// worker is gone or the status was already `new_status`.
    ///
    /// Emits [`WorkerEvent::StatusChanged`] on transition.
    pub fn transition_status(
        &self,
        worker_id: &WorkerId,
        new_status: WorkerStatus,
    ) -> Option<(WorkerStatus, WorkerStatus)> {
        self.transition_status_inner(worker_id, None, new_status)
    }

    /// Same as [`Self::transition_status`], but becomes a no-op if the
    /// currently installed worker revision no longer matches
    /// `expected_revision`.
    ///
    /// Used by health probes that must discard stale probe outcomes
    /// after a same-URL `replace()`.
    ///
    /// Emits [`WorkerEvent::StatusChanged`] on transition. Holds the
    /// per-worker mutation lock.
    pub fn transition_status_if_revision(
        &self,
        worker_id: &WorkerId,
        expected_revision: u64,
        new_status: WorkerStatus,
    ) -> Option<(WorkerStatus, WorkerStatus)> {
        self.transition_status_inner(worker_id, Some(expected_revision), new_status)
    }

    /// Apply a worker-local mutation while holding the per-worker lock
    /// and optionally emit a `StatusChanged` event under the same lock.
    ///
    /// Used by `WorkerManager` so counter mutation and revision-checked
    /// status transitions cannot race a same-URL `replace()`. The closure
    /// returns `(result, Option<new_status>)`; a transition is emitted
    /// only when the candidate status differs from the current one.
    ///
    /// Returns `None` when the worker is gone or the revision no longer
    /// matches. Otherwise returns `Some((result, transition))` where
    /// `transition` is `Some((old, new))` if a `StatusChanged` event was
    /// emitted.
    ///
    /// Emits [`WorkerEvent::StatusChanged`] only when the candidate
    /// status differs. Holds the per-worker mutation lock for the whole
    /// closure.
    pub fn apply_if_revision<T, F>(
        &self,
        worker_id: &WorkerId,
        expected_revision: u64,
        f: F,
    ) -> Option<(T, Option<(WorkerStatus, WorkerStatus)>)>
    where
        F: FnOnce(&Arc<dyn Worker>) -> (T, Option<WorkerStatus>),
    {
        let lock = self
            .worker_mutation_locks
            .entry(worker_id.clone())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone();
        let _guard = lock.lock();

        let worker = self.workers.get(worker_id)?.clone();
        if worker.revision() != expected_revision {
            return None;
        }

        let old_status = worker.status();
        let (result, candidate_status) = f(&worker);
        let transition = match candidate_status {
            Some(new_status) if new_status != old_status => {
                worker.set_status(new_status);
                let _ = self.event_tx.send(WorkerEvent::StatusChanged {
                    worker_id: worker_id.clone(),
                    worker: worker.clone(),
                    old_status,
                    new_status,
                });
                Some((old_status, new_status))
            }
            _ => None,
        };

        Some((result, transition))
    }

    // ───────────────────────────────────────────────────────────────────
    // 6. Update — config (no event)
    // ───────────────────────────────────────────────────────────────────

    /// Update the retry config for a model group (last write wins).
    ///
    /// Called during worker registration when the worker carries non-empty
    /// retry overrides. If `enabled` is false, `max_retries` is normalised
    /// to 1 before storage. Holds no registry locks. Emits no events.
    pub fn set_model_retry_config(&self, model_id: &str, mut config: RetryConfig, enabled: bool) {
        if !enabled {
            config.max_retries = 1;
        }
        self.model_retry_configs
            .insert(model_id.to_string(), config);
    }

    /// Reserve (or retrieve) a stable UUID for a worker URL.
    ///
    /// Used by `WorkerService::create_worker()` to return a worker ID in
    /// the 202 response before the async workflow runs. The workflow's
    /// `register_or_replace()` call will find the pre-reserved entry and
    /// create the worker under this ID. Idempotent — repeated calls for
    /// the same URL return the same ID. Emits no events.
    pub fn reserve_id_for_url(&self, url: &str) -> WorkerId {
        self.url_to_id.entry(url.to_string()).or_default().clone()
    }

    /// Set (or clear) the mesh sync manager after initialisation.
    ///
    /// Thread-safe via an internal `RwLock`. The registry forwards worker
    /// add/replace/remove events to the manager when one is installed.
    /// Scheduled for removal when `WorkerSyncAdapter` (mesh v2) replaces
    /// this hook; the registry will then have zero mesh awareness. Emits
    /// no events.
    pub fn set_mesh_sync(&self, mesh_sync: OptionalMeshSyncManager) {
        let mut guard = self.mesh_sync.write();
        *guard = mesh_sync;
    }

    // ───────────────────────────────────────────────────────────────────
    // 7. Remove
    // ───────────────────────────────────────────────────────────────────

    /// Remove a worker by ID and clean up every index entry.
    ///
    /// Returns `Some(worker)` if the ID existed, `None` otherwise. Tears
    /// down the URL mapping, per-worker mutation lock, model/type/
    /// connection indexes, and per-model retry config when the last
    /// worker for a model is removed. Also forwards the removal to mesh
    /// sync and clears per-worker Prometheus metrics.
    ///
    /// Emits [`WorkerEvent::Removed`] on success. Holds the per-worker
    /// mutation lock for the whole teardown so it cannot race a
    /// concurrent `replace()`.
    pub fn remove(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        // Acquire the same per-worker lock used by replace() to prevent
        // remove racing with a concurrent replace that has already snapshot
        // the old worker and is about to re-insert.
        let lock = self
            .worker_mutation_locks
            .entry(worker_id.clone())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone();
        let _guard = lock.lock();

        if let Some((_, worker)) = self.workers.remove(worker_id) {
            self.url_to_id.remove(worker.url());
            // We hold _guard; drop the DashMap entry but the Mutex stays alive via Arc.
            self.worker_mutation_locks.remove(worker_id);

            for model_id in Self::worker_model_ids(&worker) {
                self.remove_worker_from_model_index(&model_id, worker.url());
                // Drop the per-model retry config when the last worker leaves.
                let model_empty = self.model_index.get(&model_id).is_none_or(|w| w.is_empty());
                if model_empty {
                    self.model_retry_configs.remove(&model_id);
                }
            }

            if let Some(mut type_workers) = self.type_workers.get_mut(worker.worker_type()) {
                type_workers.retain(|id| id != worker_id);
            }

            if let Some(mut conn_workers) =
                self.connection_workers.get_mut(worker.connection_mode())
            {
                conn_workers.retain(|id| id != worker_id);
            }

            // Mark the worker as not-ready before tearing down its
            // metrics so any in-flight `is_healthy()` callers that
            // still hold an `Arc` see the correct state. Skip the
            // transition for `Pending` (hasn't proven itself) and
            // `Failed` (already terminal); only Ready warrants the
            // explicit demotion. Mirrors the legacy `set_healthy(false)`
            // semantics without going through the deprecated shim.
            if worker.status() == WorkerStatus::Ready {
                worker.set_status(WorkerStatus::NotReady);
            }
            Metrics::remove_worker_metrics(worker.url());

            // Sync removal to mesh if enabled (no-op if mesh is not enabled)
            {
                let guard = self.mesh_sync.read();
                if let Some(ref mesh_sync) = *guard {
                    mesh_sync.remove_worker_state(worker_id.as_str());
                }
            }

            let _ = self.event_tx.send(WorkerEvent::Removed {
                worker_id: worker_id.clone(),
                worker: worker.clone(),
            });

            Some(worker)
        } else {
            None
        }
    }

    /// Remove a worker by URL.
    ///
    /// Thin wrapper over [`Self::remove`] that first resolves the URL to
    /// a `WorkerId`. Returns `None` if no worker is registered at this
    /// URL. Emits [`WorkerEvent::Removed`] on success via the underlying
    /// `remove()` call.
    ///
    /// Only *reads* the `url_to_id` mapping here — the actual removal is
    /// performed inside `remove()` while the per-worker mutation lock is
    /// held. Pre-removing the mapping would open a race where a
    /// concurrent `register()` could reclaim the URL under a new
    /// `WorkerId` before `remove()` takes the lock, and the subsequent
    /// teardown would then delete the new mapping.
    pub fn remove_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        let worker_id = self.url_to_id.get(url).map(|entry| entry.clone())?;
        self.remove(&worker_id)
    }

    // ───────────────────────────────────────────────────────────────────
    // 8. Internal helpers
    // ───────────────────────────────────────────────────────────────────

    /// Collect the unique model IDs advertised by a worker.
    ///
    /// Public so workflow steps can share the same de-duplication rule
    /// the registry uses internally when building the model index. Falls
    /// back to the worker's primary `model_id()` if the richer
    /// `models()` list is empty. Does not touch the registry; emits no
    /// events.
    pub fn worker_model_ids(worker: &Arc<dyn Worker>) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut model_ids: Vec<String> = worker
            .models()
            .into_iter()
            .map(|model| model.id)
            .filter(|model_id| seen.insert(model_id.clone()))
            .collect();

        if model_ids.is_empty() {
            model_ids.push(worker.model_id().to_string());
        }

        model_ids
    }

    /// Core registration logic shared by local and mesh paths.
    ///
    /// Acquires the per-worker mutation lock before making the worker
    /// visible in any index, and holds it for the full sequence — insert,
    /// index updates, optional outgoing mesh sync, and the `Registered`
    /// event broadcast. Releasing the lock only after the event is sent
    /// guarantees subscribers cannot observe a mutation event for this
    /// `WorkerId` before the `Registered` event that created it.
    ///
    /// `sync_mesh` is `true` for local workflow registrations and
    /// `false` for mesh-imported workers (the mesh subscriber must not
    /// re-broadcast incoming state to avoid a CRDT version-bump loop).
    fn register_inner(&self, worker: Arc<dyn Worker>, sync_mesh: bool) -> Option<WorkerId> {
        // Resolve (or reserve) the worker_id from url_to_id. The entry
        // API is atomic per bucket, so concurrent callers either reuse
        // the same existing_id or serialize on vacant insertion.
        let worker_id = match self.url_to_id.entry(worker.url().to_string()) {
            Entry::Occupied(entry) => entry.get().clone(),
            Entry::Vacant(entry) => {
                let new_id = WorkerId::new();
                entry.insert(new_id.clone());
                new_id
            }
        };

        // Acquire the per-worker mutation lock BEFORE making the worker
        // visible in `workers`. The lock is keyed on `worker_id`, so
        // concurrent registrations for the same URL serialize here.
        let lock = self
            .worker_mutation_locks
            .entry(worker_id.clone())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone();
        let _guard = lock.lock();

        // Under the lock, reject if the URL already has an active
        // worker. A pre-reserved ID (from `reserve_id_for_url`) or a
        // same-ID re-entry from a racing caller both hit this check.
        if self.workers.contains_key(&worker_id) {
            return None;
        }

        self.workers.insert(worker_id.clone(), worker.clone());

        // Update model index for O(1) lookups using copy-on-write.
        for model_id in Self::worker_model_ids(&worker) {
            self.add_worker_to_model_index(&model_id, worker.clone());
            self.rebuild_hash_ring(&model_id);
        }

        // Update type index (clone needed for DashMap key ownership)
        self.type_workers
            .entry(*worker.worker_type())
            .or_default()
            .push(worker_id.clone());

        // Update connection mode index (clone needed for DashMap key ownership)
        self.connection_workers
            .entry(*worker.connection_mode())
            .or_default()
            .push(worker_id.clone());

        // Outgoing mesh sync happens under the lock so mesh observers
        // cannot see a later mutation (Replaced/Removed/StatusChanged)
        // for this worker_id before the initial state is published.
        if sync_mesh {
            let guard = self.mesh_sync.read();
            if let Some(ref mesh_sync) = *guard {
                mesh_sync.sync_worker_state(
                    worker_id.as_str().to_string(),
                    worker.model_id().to_string(),
                    worker.url().to_string(),
                    worker.is_healthy(),
                    0.0,
                    bincode::serialize(&worker.metadata().spec).unwrap_or_default(),
                );
            }
        }

        // Broadcast under the lock so event order per worker_id is
        // strictly: Registered → (Replaced | StatusChanged | Removed).
        let _ = self.event_tx.send(WorkerEvent::Registered {
            worker_id: worker_id.clone(),
            worker: worker.clone(),
        });

        Some(worker_id)
    }

    /// Rebuild the hash ring for a model based on current workers in the model index.
    fn rebuild_hash_ring(&self, model_id: &str) {
        if let Some(workers) = self.model_index.get(model_id) {
            let ring = HashRing::new(workers.value().iter().map(|w| w.url()));
            self.hash_rings.insert(model_id.to_string(), Arc::new(ring));
        } else {
            // No workers for this model, remove the ring
            self.hash_rings.remove(model_id);
        }
    }

    /// Append `worker` to the copy-on-write model index slice for `model_id`.
    /// Replaces any existing entry with the same URL so updates via replace()
    /// do not leave duplicate rows.
    fn add_worker_to_model_index(&self, model_id: &str, worker: Arc<dyn Worker>) {
        self.model_index
            .entry(model_id.to_string())
            .and_modify(|existing| {
                let mut new_workers: Vec<Arc<dyn Worker>> = existing
                    .iter()
                    .filter(|w| w.url() != worker.url())
                    .cloned()
                    .collect();
                new_workers.push(worker.clone());
                *existing = Arc::from(new_workers.into_boxed_slice());
            })
            .or_insert_with(|| Arc::from(vec![worker].into_boxed_slice()));
    }

    /// Drop `worker_url` from the copy-on-write model index slice for `model_id`
    /// and rebuild the hash ring. Evicts the whole model entry when empty.
    fn remove_worker_from_model_index(&self, model_id: &str, worker_url: &str) {
        let mut should_remove_entry = false;

        if let Some(mut entry) = self.model_index.get_mut(model_id) {
            let new_workers: Vec<Arc<dyn Worker>> = entry
                .iter()
                .filter(|w| w.url() != worker_url)
                .cloned()
                .collect();

            if new_workers.is_empty() {
                *entry = Arc::from(Vec::<Arc<dyn Worker>>::new().into_boxed_slice());
                should_remove_entry = true;
            } else {
                *entry = Arc::from(new_workers.into_boxed_slice());
            }
        }

        if should_remove_entry {
            self.model_index
                .remove_if(model_id, |_, workers| workers.is_empty());
        }

        self.rebuild_hash_ring(model_id);
    }

    /// Shared backend for [`Self::transition_status`] and
    /// [`Self::transition_status_if_revision`]. Holds the per-worker
    /// mutation lock for the full read-modify-emit sequence.
    fn transition_status_inner(
        &self,
        worker_id: &WorkerId,
        expected_revision: Option<u64>,
        new_status: WorkerStatus,
    ) -> Option<(WorkerStatus, WorkerStatus)> {
        let lock = self
            .worker_mutation_locks
            .entry(worker_id.clone())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone();
        let _guard = lock.lock();

        let worker = self.workers.get(worker_id)?.clone();
        if expected_revision.is_some_and(|revision| worker.revision() != revision) {
            return None;
        }

        let old_status = worker.status();
        if old_status == new_status {
            return None;
        }

        worker.set_status(new_status);

        let _ = self.event_tx.send(WorkerEvent::StatusChanged {
            worker_id: worker_id.clone(),
            worker: worker.clone(),
            old_status,
            new_status,
        });

        Some((old_status, new_status))
    }
}

// `Default` delegates to `new()` so there is a single source of truth.
// We cannot `#[derive(Default)]` on `WorkerRegistry` because
// `broadcast::Sender` has no `Default` impl — it needs an explicit
// capacity.
impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl smg_mesh::WorkerStateSubscriber for WorkerRegistry {
    fn on_remote_worker_state(&self, state: &smg_mesh::WorkerState) {
        use openai_protocol::model_card::ModelCard;

        // If worker already exists at this URL, update its health
        // status from the mesh state. Don't re-register — the existing
        // worker has full config from its creation workflow. The
        // transition rules mirror the legacy `set_healthy(state.health)`
        // semantics intentionally: `true` always promotes to `Ready`,
        // `false` only demotes from `Ready` to `NotReady` and leaves
        // `Pending` / `Failed` alone (those are owned by the local
        // state machine, not by mesh hints).
        if let Some(existing) = self.get_by_url(&state.url) {
            if state.health {
                existing.set_status(WorkerStatus::Ready);
            } else if existing.status() == WorkerStatus::Ready {
                existing.set_status(WorkerStatus::NotReady);
            }
            tracing::debug!(
                url = %state.url,
                healthy = state.health,
                "Updated health for existing mesh-synced worker"
            );
            return;
        }

        // New worker — build from the full WorkerSpec if available,
        // otherwise fall back to minimal builder (old nodes / rolling upgrade).
        let worker = match bincode::deserialize::<openai_protocol::worker::WorkerSpec>(&state.spec)
        {
            Ok(spec) if !state.spec.is_empty() => {
                super::builder::BasicWorkerBuilder::from_spec(spec).build()
            }
            _ => super::builder::BasicWorkerBuilder::new(&state.url)
                .model(ModelCard::new(&state.model_id))
                .build(),
        };

        // Same legacy `set_healthy(state.health)` semantics for the
        // pre-registration newly built worker: `true` → `Ready`,
        // `false` → leave the builder default (typically `Pending`).
        // The state machine takes over after registration.
        if state.health {
            worker.set_status(WorkerStatus::Ready);
        }

        // `register_inner(worker, false)` skips OUTGOING mesh sync to
        // avoid a version-bump loop on the CRDT, but still publishes
        // the local `Registered` event under the per-worker mutation
        // lock. In-process subscribers (WorkerManager's health
        // scheduler, etc.) pick up mesh-imported workers via the same
        // event path as any other registration. The event is a local
        // broadcast only; it does not re-enter the mesh.
        let worker: Arc<dyn Worker> = Arc::new(worker);
        if let Some(id) = self.register_inner(worker, false) {
            tracing::info!(
                worker_id = %id.as_str(),
                url = %state.url,
                model = %state.model_id,
                healthy = state.health,
                has_spec = !state.spec.is_empty(),
                "Registered mesh-synced worker into local registry"
            );
        }
    }
}

/// Statistics for the worker registry
#[derive(Debug, Clone)]
pub struct WorkerRegistryStats {
    /// Total number of registered workers
    pub total_workers: usize,
    /// Number of unique models served
    pub total_models: usize,
    /// Number of workers passing health checks
    pub healthy_workers: usize,
    /// Number of workers failing health checks
    pub unhealthy_workers: usize,
    /// Sum of current load across all workers
    pub total_load: usize,
    /// Number of regular (non-PD) workers
    pub regular_workers: usize,
    /// Number of prefill workers (PD mode)
    pub prefill_workers: usize,
    /// Number of decode workers (PD mode)
    pub decode_workers: usize,
    /// Number of HTTP-connected workers
    pub http_workers: usize,
    /// Number of gRPC-connected workers
    pub grpc_workers: usize,
    /// Number of workers with circuit breaker in Open state (not accepting requests)
    pub circuit_breaker_open: usize,
    /// Number of workers with circuit breaker in HalfOpen state (testing recovery)
    pub circuit_breaker_half_open: usize,
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use openai_protocol::model_card::ModelCard;

    use super::*;
    use crate::worker::{
        circuit_breaker::{CircuitBreakerConfig, CircuitState},
        BasicWorkerBuilder, WorkerLoadGuard,
    };

    fn no_health_check() -> openai_protocol::worker::HealthCheckConfig {
        openai_protocol::worker::HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_worker_registry() {
        let registry = WorkerRegistry::new();

        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "llama-3-8b".to_string());
        labels.insert("priority".to_string(), "50".to_string());
        labels.insert("cost".to_string(), "0.8".to_string());

        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        // Register worker
        let worker_id = registry.register(Arc::from(worker)).unwrap();

        assert!(registry.get(&worker_id).is_some());
        assert!(registry.get_by_url("http://worker1:8080").is_some());
        assert_eq!(registry.get_by_model("llama-3-8b").len(), 1);
        assert_eq!(registry.get_by_type(WorkerType::Regular).len(), 1);
        assert_eq!(registry.get_by_connection(ConnectionMode::Http).len(), 1);

        let stats = registry.stats();
        assert_eq!(stats.total_workers, 1);
        assert_eq!(stats.total_models, 1);

        registry.remove(&worker_id);
        assert!(registry.get(&worker_id).is_none());
    }

    #[test]
    fn test_model_index_fast_lookup() {
        let registry = WorkerRegistry::new();

        let mut labels1 = HashMap::new();
        labels1.insert("model_id".to_string(), "llama-3".to_string());
        let worker1: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels1)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        let mut labels2 = HashMap::new();
        labels2.insert("model_id".to_string(), "llama-3".to_string());
        let worker2: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker2:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels2)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        let mut labels3 = HashMap::new();
        labels3.insert("model_id".to_string(), "gpt-4".to_string());
        let worker3: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker3:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels3)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        // Register workers
        registry.register(Arc::from(worker1)).unwrap();
        registry.register(Arc::from(worker2)).unwrap();
        registry.register(Arc::from(worker3)).unwrap();

        let llama_workers = registry.get_by_model("llama-3");
        assert_eq!(llama_workers.len(), 2);
        let urls: Vec<String> = llama_workers.iter().map(|w| w.url().to_string()).collect();
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));

        let gpt_workers = registry.get_by_model("gpt-4");
        assert_eq!(gpt_workers.len(), 1);
        assert_eq!(gpt_workers[0].url(), "http://worker3:8080");

        let unknown_workers = registry.get_by_model("unknown-model");
        assert_eq!(unknown_workers.len(), 0);

        registry.remove_by_url("http://worker1:8080");
        let llama_workers_after = registry.get_by_model("llama-3");
        assert_eq!(llama_workers_after.len(), 1);
        assert_eq!(llama_workers_after[0].url(), "http://worker2:8080");
    }

    // Health-checker integration tests moved to worker/manager.rs along with
    // the loop itself. The registry is now a pure collection — see
    // `worker::manager::WorkerManager` tests.

    #[test]
    fn test_transition_status_emits_event_and_changes_status() {
        let registry = WorkerRegistry::new();
        let mut rx = registry.subscribe_events();

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w1:8080")
                .worker_type(WorkerType::Regular)
                .health_config(openai_protocol::worker::HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        let worker_id = registry.register(worker.clone()).unwrap();
        // Drain Registered event
        let _ = rx.try_recv().unwrap();

        // Initial status is Ready (disable_health_check). Transition to NotReady.
        let result = registry.transition_status(&worker_id, WorkerStatus::NotReady);
        assert_eq!(result, Some((WorkerStatus::Ready, WorkerStatus::NotReady)));
        assert_eq!(worker.status(), WorkerStatus::NotReady);

        let event = rx.try_recv().unwrap();
        match event {
            WorkerEvent::StatusChanged {
                old_status,
                new_status,
                ..
            } => {
                assert_eq!(old_status, WorkerStatus::Ready);
                assert_eq!(new_status, WorkerStatus::NotReady);
            }
            other => panic!("Expected StatusChanged, got {other:?}"),
        }
    }

    #[test]
    fn test_transition_status_no_op_when_status_unchanged() {
        let registry = WorkerRegistry::new();
        let mut rx = registry.subscribe_events();

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w1:8080")
                .worker_type(WorkerType::Regular)
                .health_config(openai_protocol::worker::HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        let worker_id = registry.register(worker).unwrap();
        let _ = rx.try_recv().unwrap();

        // Already Ready — transition to Ready is a no-op
        assert_eq!(
            registry.transition_status(&worker_id, WorkerStatus::Ready),
            None
        );
        assert!(rx.try_recv().is_err(), "no event should be emitted");
    }

    #[test]
    fn test_transition_status_returns_none_for_missing_worker() {
        let registry = WorkerRegistry::new();
        let missing = WorkerId::from_string("nonexistent".to_string());
        assert_eq!(
            registry.transition_status(&missing, WorkerStatus::Ready),
            None
        );
    }

    #[test]
    fn test_transition_status_if_revision_rejects_stale_transition() {
        let registry = WorkerRegistry::new();

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w1:8080")
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .build(),
        );
        let worker_id = registry.register(worker.clone()).unwrap();
        let stale_revision = worker.revision();

        let replacement: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w1:8080")
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .priority(99)
                .build(),
        );
        assert!(registry.replace(&worker_id, replacement));

        assert_eq!(
            registry.transition_status_if_revision(
                &worker_id,
                stale_revision,
                WorkerStatus::NotReady
            ),
            None
        );

        let current = registry.get(&worker_id).unwrap();
        assert_eq!(current.status(), WorkerStatus::Ready);
        assert_eq!(current.priority(), 99);
        assert_eq!(current.revision(), stale_revision + 1);
    }

    #[test]
    fn test_multi_model_worker_is_indexed_for_each_model() {
        let registry = WorkerRegistry::new();

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("https://api.openai.com")
                .worker_type(WorkerType::Regular)
                .models(vec![
                    ModelCard::new("gpt-4o"),
                    ModelCard::new("text-embedding-3-large"),
                ])
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        );

        let worker_id = registry.register(worker).unwrap();

        assert!(registry.get(&worker_id).is_some());
        assert_eq!(registry.get_by_model("gpt-4o").len(), 1);
        assert_eq!(registry.get_by_model("text-embedding-3-large").len(), 1);
        assert_eq!(
            registry.get_by_model("gpt-4o")[0].url(),
            "https://api.openai.com"
        );
        assert_eq!(
            registry.get_by_model("text-embedding-3-large")[0].url(),
            "https://api.openai.com"
        );

        let mut models = registry.get_models();
        models.sort();
        assert_eq!(
            models,
            vec!["gpt-4o".to_string(), "text-embedding-3-large".to_string()]
        );

        let stats = registry.stats();
        assert_eq!(stats.total_workers, 1);
        assert_eq!(stats.total_models, 2);
    }

    #[test]
    fn test_replace_same_url_refreshes_all_model_indexes() {
        let registry = WorkerRegistry::new();

        let first: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("https://api.openai.com")
                .worker_type(WorkerType::Regular)
                .models(vec![ModelCard::new("gpt-4o")])
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        );
        let second: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("https://api.openai.com")
                .worker_type(WorkerType::Regular)
                .models(vec![ModelCard::new("o3"), ModelCard::new("o4-mini")])
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        );

        // First registration creates the worker
        let first_id = registry.register(first).unwrap();

        // Second registration with same URL should be rejected
        assert!(registry.register(second.clone()).is_none());

        // Use replace() to update the worker
        assert!(registry.replace(&first_id, second));

        assert_eq!(registry.len(), 1);
        assert!(registry.get_by_model("gpt-4o").is_empty());
        assert_eq!(registry.get_by_model("o3").len(), 1);
        assert_eq!(registry.get_by_model("o4-mini").len(), 1);
        assert_eq!(registry.get_by_type(WorkerType::Regular).len(), 1);
        assert_eq!(registry.get_by_connection(ConnectionMode::Http).len(), 1);

        let mut models = registry.get_models();
        models.sort();
        assert_eq!(models, vec!["o3".to_string(), "o4-mini".to_string()]);
    }

    #[test]
    fn test_register_or_replace_upsert() {
        let registry = WorkerRegistry::new();

        let first: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("https://api.openai.com")
                .worker_type(WorkerType::Regular)
                .models(vec![ModelCard::new("gpt-4o")])
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        );
        let second: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("https://api.openai.com")
                .worker_type(WorkerType::Regular)
                .models(vec![ModelCard::new("o3"), ModelCard::new("o4-mini")])
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        );

        let first_id = registry.register_or_replace(first);
        let second_id = registry.register_or_replace(second);

        // Same URL → same ID (upsert)
        assert_eq!(first_id, second_id);
        assert_eq!(registry.len(), 1);
        // Old model gone, new models present
        assert!(registry.get_by_model("gpt-4o").is_empty());
        assert_eq!(registry.get_by_model("o3").len(), 1);
        assert_eq!(registry.get_by_model("o4-mini").len(), 1);
    }

    #[test]
    fn test_builder_status_override_on_replace() {
        // Regression test: metadata-only updates must not kick a healthy
        // worker back to Pending. The builder exposes a `.status()` setter
        // so callers (e.g. UpdateWorkerPropertiesStep) can pass the old
        // worker's status when constructing the replacement.
        let registry = WorkerRegistry::new();

        // First worker starts Pending (health checks enabled by default),
        // then gets promoted to Ready (simulating what the health checker does).
        let first: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker:8080")
                .worker_type(WorkerType::Regular)
                .model(ModelCard::new("llama-3"))
                .build(),
        );
        assert_eq!(first.status(), WorkerStatus::Pending);
        first.set_status(WorkerStatus::Ready);

        let first_id = registry.register(first.clone()).unwrap();
        assert_eq!(
            registry.get(&first_id).unwrap().status(),
            WorkerStatus::Ready
        );

        // Caller (e.g. UpdateWorkerPropertiesStep) reads the old status and
        // passes it to the builder. The builder honors the override instead
        // of defaulting to Pending.
        let preserved_status = first.status();
        let second: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker:8080")
                .worker_type(WorkerType::Regular)
                .model(ModelCard::new("llama-3"))
                .priority(99)
                .status(preserved_status)
                .build(),
        );
        assert_eq!(
            second.status(),
            WorkerStatus::Ready,
            "builder must honor explicit status override"
        );

        assert!(registry.replace(&first_id, second));

        let after = registry.get(&first_id).unwrap();
        assert_eq!(after.status(), WorkerStatus::Ready);
        assert_eq!(after.priority(), 99, "new priority should be applied");
    }

    #[test]
    fn test_replace_preserves_runtime_state_and_circuit_breaker() {
        let registry = WorkerRegistry::new();
        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-routing-key", "sticky-key".parse().unwrap());

        let first: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker:8080")
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .build(),
        );
        let first_id = registry.register(first.clone()).unwrap();
        let initial_revision = first.revision();

        first.set_status(WorkerStatus::NotReady);
        first.increment_processed();
        let load_guard = WorkerLoadGuard::new(first.clone(), Some(&headers));

        for _ in 0..5 {
            first.record_outcome(503);
        }
        assert_eq!(first.circuit_breaker_state(), CircuitState::Open);

        let replacement: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker:8080")
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .priority(99)
                .build(),
        );
        assert!(registry.replace(&first_id, replacement));

        let current = registry.get(&first_id).unwrap();
        assert_eq!(current.priority(), 99);
        assert_eq!(current.status(), WorkerStatus::NotReady);
        assert_eq!(current.load(), 1);
        assert_eq!(current.routing_key_load(), 1);
        assert_eq!(current.processed_requests(), 1);
        assert_eq!(current.circuit_breaker_state(), CircuitState::Open);
        assert_eq!(current.revision(), initial_revision + 1);

        first.increment_processed();
        assert_eq!(current.processed_requests(), 2);

        drop(load_guard);
        assert_eq!(current.load(), 0);
        assert_eq!(current.routing_key_load(), 0);
    }

    #[test]
    fn test_builder_default_status_is_pending() {
        // Without an explicit override, health-checked workers start Pending.
        let worker = BasicWorkerBuilder::new("http://worker:8080")
            .worker_type(WorkerType::Regular)
            .build();
        assert_eq!(worker.status(), WorkerStatus::Pending);
    }

    #[test]
    fn test_register_rejects_duplicate_url() {
        let registry = WorkerRegistry::new();

        let first: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        let second: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        assert!(registry.register(first).is_some());
        assert!(registry.register(second).is_none());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_model_retry_config_last_write_wins() {
        let registry = WorkerRegistry::new();

        assert!(registry.get_retry_config("llama-3").is_none());

        let config1 = RetryConfig {
            max_retries: 3,
            ..RetryConfig::default()
        };
        registry.set_model_retry_config("llama-3", config1, true);

        let stored = registry.get_retry_config("llama-3").unwrap();
        assert_eq!(stored.max_retries, 3);

        // Last write wins — overwrite with different config
        let config2 = RetryConfig {
            max_retries: 10,
            initial_backoff_ms: 200,
            ..RetryConfig::default()
        };
        registry.set_model_retry_config("llama-3", config2, true);

        let stored = registry.get_retry_config("llama-3").unwrap();
        assert_eq!(stored.max_retries, 10);
        assert_eq!(stored.initial_backoff_ms, 200);

        // Disable retries — max_retries should be set to 1
        let config3 = RetryConfig {
            max_retries: 10,
            ..RetryConfig::default()
        };
        registry.set_model_retry_config("llama-3", config3, false);

        let stored = registry.get_retry_config("llama-3").unwrap();
        assert_eq!(stored.max_retries, 1); // disabled → max_retries = 1
    }

    #[test]
    fn test_model_retry_config_cleanup_on_last_worker_removal() {
        let registry = WorkerRegistry::new();

        let worker1 = Arc::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .model(ModelCard::new("llama-3"))
                .build(),
        ) as Arc<dyn Worker>;

        let worker2 = Arc::new(
            BasicWorkerBuilder::new("http://worker2:8080")
                .model(ModelCard::new("llama-3"))
                .build(),
        ) as Arc<dyn Worker>;

        let id1 = registry.register(worker1).unwrap();
        let id2 = registry.register(worker2).unwrap();

        registry.set_model_retry_config(
            "llama-3",
            RetryConfig {
                max_retries: 5,
                ..RetryConfig::default()
            },
            true,
        );
        assert!(registry.get_retry_config("llama-3").is_some());

        // Remove first worker — config should still exist
        registry.remove(&id1);
        assert!(registry.get_retry_config("llama-3").is_some());

        // Remove last worker — config should be cleaned up
        registry.remove(&id2);
        assert!(registry.get_retry_config("llama-3").is_none());
    }

    #[test]
    fn test_mesh_worker_state_subscriber() {
        use smg_mesh::{WorkerState, WorkerStateSubscriber};

        let registry = WorkerRegistry::new();

        // Simulate a remote mesh worker state arriving
        let state = WorkerState {
            worker_id: "mesh-w1".into(),
            model_id: "llama-3".into(),
            url: "http://mesh-worker1:8080".into(),
            health: true,
            load: 0.5,
            version: 1,
            spec: vec![],
        };

        // Should register the worker locally
        registry.on_remote_worker_state(&state);
        assert_eq!(registry.get_by_model("llama-3").len(), 1);
        assert!(registry.get_by_url("http://mesh-worker1:8080").is_some());

        // Duplicate URL: should be a no-op (create-only semantics)
        let state_v2 = WorkerState {
            version: 2,
            ..state.clone()
        };
        registry.on_remote_worker_state(&state_v2);
        assert_eq!(registry.get_by_model("llama-3").len(), 1);

        // Pre-existing k8s worker at the same URL: mesh should not overwrite
        let registry2 = WorkerRegistry::new();
        let k8s_worker = Arc::new(
            BasicWorkerBuilder::new("http://mesh-worker1:8080")
                .model(ModelCard::new("llama-3"))
                .label("source", "k8s")
                .build(),
        );
        registry2.register(k8s_worker);
        registry2.on_remote_worker_state(&state);
        // Still only one worker, and it's the original k8s one with labels
        assert_eq!(registry2.get_by_model("llama-3").len(), 1);
        let w = registry2.get_by_url("http://mesh-worker1:8080").unwrap();
        assert_eq!(
            w.metadata().spec.labels.get("source"),
            Some(&"k8s".to_string())
        );
    }

    #[test]
    fn test_mesh_imported_worker_emits_registered_event() {
        use smg_mesh::{WorkerState, WorkerStateSubscriber};

        // Mesh-imported workers must emit `WorkerEvent::Registered` so
        // event-driven subscribers (WorkerManager's health scheduler)
        // pick them up via the same path as any other registration.
        // Without this event, a mesh worker would be routable but never
        // probed locally, and `--remove-unhealthy-workers` could not
        // reach it.
        let registry = WorkerRegistry::new();
        let mut rx = registry.subscribe_events();

        let state = WorkerState {
            worker_id: "mesh-w1".into(),
            model_id: "llama-3".into(),
            url: "http://mesh-worker-event:8080".into(),
            health: true,
            load: 0.5,
            version: 1,
            spec: vec![],
        };

        registry.on_remote_worker_state(&state);

        let event = rx
            .try_recv()
            .expect("mesh import must broadcast a Registered event");
        match event {
            WorkerEvent::Registered { worker, .. } => {
                assert_eq!(worker.url(), "http://mesh-worker-event:8080");
                assert_eq!(worker.model_id(), "llama-3");
            }
            other => panic!("Expected Registered event from mesh import, got: {other:?}"),
        }
    }

    #[test]
    fn test_mesh_worker_state_update_is_silent_on_existing_worker() {
        use smg_mesh::{WorkerState, WorkerStateSubscriber};

        // Health updates for an already-registered worker mutate the
        // local status field directly (via `set_status`) without
        // emitting an event — the existing worker is already on
        // WorkerManager's schedule, and local probes will reconcile
        // the state on the next tick. We verify the no-event behavior
        // so a future regression (e.g. adding a `StatusChanged` emit
        // here) is caught.
        let registry = WorkerRegistry::new();

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://mesh-existing:8080")
                .model(ModelCard::new("llama-3"))
                .health_config(openai_protocol::worker::HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        registry.register(worker).unwrap();

        let mut rx = registry.subscribe_events();

        let state = WorkerState {
            worker_id: "mesh-w1".into(),
            model_id: "llama-3".into(),
            url: "http://mesh-existing:8080".into(),
            health: false,
            load: 0.0,
            version: 2,
            spec: vec![],
        };
        registry.on_remote_worker_state(&state);

        // No event is expected on the update path.
        assert!(
            rx.try_recv().is_err(),
            "existing-worker update path should not broadcast an event"
        );
    }

    #[test]
    fn test_worker_event_broadcast() {
        let registry = WorkerRegistry::new();
        let mut rx = registry.subscribe_events();

        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "test-model".to_string());

        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://event-worker:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        let worker_id = registry.register(Arc::from(worker)).unwrap();

        // Should receive Registered event
        let event = rx.try_recv().unwrap();
        match event {
            WorkerEvent::Registered { worker, .. } => {
                assert_eq!(worker.url(), "http://event-worker:8080");
            }
            other => panic!("Expected Registered event, got: {other:?}"),
        }

        registry.remove(&worker_id);

        // Should receive Removed event
        let event = rx.try_recv().unwrap();
        match event {
            WorkerEvent::Removed { worker, .. } => {
                assert_eq!(worker.url(), "http://event-worker:8080");
            }
            other => panic!("Expected Removed event, got: {other:?}"),
        }
    }
}
