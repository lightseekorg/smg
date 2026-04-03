//! Worker Registry for multi-router support
//!
//! Provides centralized registry for workers with model-based indexing
//!
//! # Performance Optimizations
//! The model index uses immutable Arc snapshots instead of RwLock for lock-free reads.
//! This is critical for high-concurrency scenarios where many requests query the same model.
//!
//! # Consistent Hash Ring
//! The registry maintains a pre-computed hash ring per model for O(log n) consistent hashing.
//! The ring is rebuilt only when workers are added/removed, not per-request.
//! Uses virtual nodes (150 per worker) for even distribution and blake3 for stable hashing.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use dashmap::{mapref::entry::Entry, DashMap};
use parking_lot::RwLock;
use smg_mesh::OptionalMeshSyncManager;
use tokio::sync::broadcast;
use uuid::Uuid;

use crate::{
    config::types::RetryConfig,
    core::{
        circuit_breaker::CircuitState,
        worker::{HealthChecker, RuntimeType, WorkerType},
        worker_event::WorkerEvent,
        ConnectionMode, Job, JobQueue, Worker,
    },
    observability::metrics::Metrics,
};

/// Number of virtual nodes per physical worker for even distribution.
/// 150 is a common choice that provides good balance between memory and distribution.
const VIRTUAL_NODES_PER_WORKER: usize = 150;

/// Consistent hash ring for O(log n) worker selection.
///
/// Each worker is placed at multiple positions (virtual nodes) on the ring
/// based on hash(worker_url + vnode_index). This provides:
/// - Even key distribution across workers
/// - Minimal key redistribution when workers are added/removed (~1/N keys move)
/// - O(log n) lookup via binary search
///
/// Uses blake3 for stable, fast hashing that's consistent across Rust versions.
#[derive(Debug, Clone)]
pub struct HashRing {
    /// Sorted list of (ring_position, worker_url)
    /// Multiple entries per worker (virtual nodes) for even distribution.
    /// Uses Arc<str> to share URL across all virtual nodes (150 refs vs 150 copies).
    entries: Arc<[(u64, Arc<str>)]>,
}

impl HashRing {
    /// Build a hash ring from a list of workers.
    /// Creates VIRTUAL_NODES_PER_WORKER entries per worker for even distribution.
    pub fn new(workers: &[Arc<dyn Worker>]) -> Self {
        let mut entries: Vec<(u64, Arc<str>)> =
            Vec::with_capacity(workers.len() * VIRTUAL_NODES_PER_WORKER);

        for worker in workers {
            // Create Arc<str> once per worker, share across all virtual nodes
            let url: Arc<str> = Arc::from(worker.url());

            // Create multiple virtual nodes per worker
            for vnode in 0..VIRTUAL_NODES_PER_WORKER {
                let vnode_key = format!("{url}#{vnode}");
                let pos = Self::hash_position(&vnode_key);
                entries.push((pos, Arc::clone(&url)));
            }
        }

        // Sort by ring position for binary search
        entries.sort_unstable_by_key(|(pos, _)| *pos);

        Self {
            entries: Arc::from(entries.into_boxed_slice()),
        }
    }

    /// Hash a string to a ring position using blake3 (stable across versions).
    #[inline]
    #[expect(
        clippy::expect_used,
        reason = "blake3 always produces 32 bytes — converting a fixed 8-byte slice to [u8; 8] is infallible"
    )]
    fn hash_position(s: &str) -> u64 {
        let hash = blake3::hash(s.as_bytes());
        // Take first 8 bytes as u64
        u64::from_le_bytes(
            hash.as_bytes()[..8]
                .try_into()
                .expect("blake3 hash is always 32 bytes, slicing first 8 is infallible"),
        )
    }

    /// Find worker URL for a key using consistent hashing.
    /// Returns the first healthy worker URL at or after the key's position (clockwise).
    ///
    /// - `key`: The routing key to hash
    /// - `is_healthy`: Function to check if a worker URL is healthy
    pub fn find_healthy_url<F>(&self, key: &str, is_healthy: F) -> Option<&str>
    where
        F: Fn(&str) -> bool,
    {
        if self.entries.is_empty() {
            return None;
        }

        let key_pos = Self::hash_position(key);

        // Binary search to find first entry at or after key_pos
        let start = self.entries.partition_point(|(pos, _)| *pos < key_pos);

        // Walk clockwise from start, wrapping around
        // Track visited URLs to avoid checking same worker multiple times (virtual nodes)
        let mut checked_urls = HashSet::with_capacity(self.worker_count().min(16));

        for i in 0..self.entries.len() {
            let (_, url) = &self.entries[(start + i) % self.entries.len()];
            let url_str: &str = url;

            // Skip if we already checked this worker (from another virtual node)
            if !checked_urls.insert(url_str) {
                continue;
            }

            if is_healthy(url_str) {
                return Some(url_str);
            }
        }

        None
    }

    /// Check if the ring is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the number of entries in the ring (including virtual nodes)
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Get the number of unique workers in the ring
    pub fn worker_count(&self) -> usize {
        self.entries.len() / VIRTUAL_NODES_PER_WORKER.max(1)
    }
}

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
    /// Create a new worker registry
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

    /// Subscribe to worker state change events.
    pub fn subscribe_events(&self) -> broadcast::Receiver<WorkerEvent> {
        self.event_tx.subscribe()
    }

    /// Rebuild the hash ring for a model based on current workers in the model index
    fn rebuild_hash_ring(&self, model_id: &str) {
        if let Some(workers) = self.model_index.get(model_id) {
            let ring = HashRing::new(&workers);
            self.hash_rings.insert(model_id.to_string(), Arc::new(ring));
        } else {
            // No workers for this model, remove the ring
            self.hash_rings.remove(model_id);
        }
    }

    /// Get the hash ring for a model (O(1) lookup)
    pub fn get_hash_ring(&self, model_id: &str) -> Option<Arc<HashRing>> {
        self.hash_rings.get(model_id).map(|r| Arc::clone(&r))
    }

    /// Set mesh sync manager (thread-safe, can be called after initialization)
    pub fn set_mesh_sync(&self, mesh_sync: OptionalMeshSyncManager) {
        let mut guard = self.mesh_sync.write();
        *guard = mesh_sync;
    }

    /// Get the retry config for a model, if a worker group override exists.
    /// When retries are disabled for the group, max_retries will be 1.
    pub fn get_retry_config(&self, model_id: &str) -> Option<RetryConfig> {
        self.model_retry_configs
            .get(model_id)
            .map(|entry| entry.value().clone())
    }

    /// Update the retry config for a model group (last write wins).
    /// Called during worker registration when the worker has non-empty retry overrides.
    /// If retries are disabled, max_retries is set to 1 before storing.
    pub fn set_model_retry_config(&self, model_id: &str, mut config: RetryConfig, enabled: bool) {
        if !enabled {
            config.max_retries = 1;
        }
        self.model_retry_configs
            .insert(model_id.to_string(), config);
    }

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

    /// Register a new worker (create-only).
    ///
    /// Returns the new `WorkerId` on success, or `None` if a worker with
    /// the same URL is already registered and active. A URL that was
    /// pre-reserved via `reserve_id_for_url()` but has no worker yet is
    /// treated as a new registration (reuses the reserved ID).
    pub fn register(&self, worker: Arc<dyn Worker>) -> Option<WorkerId> {
        let worker_id = self.register_inner(worker.clone())?;

        // Sync to mesh if enabled (no-op if mesh is not enabled)
        {
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

        let _ = self.event_tx.send(WorkerEvent::Registered {
            url: worker.url().to_string(),
        });

        Some(worker_id)
    }

    /// Core registration logic shared by local and mesh paths.
    /// Does NOT sync to mesh — callers that need mesh sync do it themselves.
    fn register_inner(&self, worker: Arc<dyn Worker>) -> Option<WorkerId> {
        // Atomic check-and-insert via entry API to avoid TOCTOU races.
        // If URL already has an ID AND a worker object, it's a duplicate.
        // If URL has a reserved ID but no worker, it's a pre-reserved slot.
        // Atomic check-and-insert: reject if URL already has an active worker.
        // A pre-reserved ID (from reserve_id_for_url) with no worker is allowed.
        let worker_id = match self.url_to_id.entry(worker.url().to_string()) {
            Entry::Occupied(entry) => {
                let existing_id = entry.get().clone();
                if self.workers.contains_key(&existing_id) {
                    // URL has an active worker — reject
                    return None;
                }
                // Pre-reserved ID with no worker yet — use it
                existing_id
            }
            Entry::Vacant(entry) => {
                let new_id = WorkerId::new();
                entry.insert(new_id.clone());
                new_id
            }
        };

        // Store worker
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

        Some(worker_id)
    }

    /// Replace an existing worker with a new one (overwrite-then-diff).
    ///
    /// Used by `PUT /workers/{id}` and K8s discovery when a worker with
    /// the same URL already exists. Updates the worker object in-place and
    /// diffs the model index to avoid a transient gap where the worker is
    /// missing from indexes.
    ///
    /// Returns `true` if the worker was replaced, `false` if the ID was not found.
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

        // Overwrite worker object atomically
        self.workers.insert(worker_id.clone(), new_worker.clone());

        // Diff model indexes: remove stale, add new
        for removed_model in old_models.difference(&new_models) {
            self.remove_worker_from_model_index(removed_model, old_worker.url());
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

        // Update type index if changed
        if old_worker.worker_type() != new_worker.worker_type() {
            if let Some(mut type_workers) = self.type_workers.get_mut(old_worker.worker_type()) {
                type_workers.retain(|id| id != worker_id);
            }
            self.type_workers
                .entry(*new_worker.worker_type())
                .or_default()
                .push(worker_id.clone());
        }

        // Update connection mode index if changed
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

        let _ = self.event_tx.send(WorkerEvent::Registered {
            url: new_worker.url().to_string(),
        });

        true
    }

    /// Register or replace a worker (upsert).
    ///
    /// Used by internal callers (K8s discovery, startup) that need idempotent
    /// registration. If the URL already exists, replaces the worker via
    /// overwrite-then-diff. Otherwise, creates a new worker.
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

    /// Reserve (or retrieve) a stable UUID for a worker URL.
    ///
    /// Used by `WorkerService::create_worker()` to return a worker ID in
    /// the 202 response before the async workflow runs. The workflow's
    /// `register_or_replace()` call will find the pre-reserved entry and
    /// create the worker under this ID.
    pub fn reserve_id_for_url(&self, url: &str) -> WorkerId {
        self.url_to_id.entry(url.to_string()).or_default().clone()
    }

    /// Best-effort lookup of the URL for a given worker ID.
    pub fn get_url_by_id(&self, worker_id: &WorkerId) -> Option<String> {
        if let Some(worker) = self.get(worker_id) {
            return Some(worker.url().to_string());
        }
        self.url_to_id
            .iter()
            .find_map(|entry| (entry.value() == worker_id).then(|| entry.key().clone()))
    }

    /// Remove a worker by ID
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
            // Remove from URL mapping
            self.url_to_id.remove(worker.url());
            // Clean up replace lock (after we release it)
            // Note: we hold _guard, so drop the DashMap entry but the Mutex stays alive via Arc
            self.worker_mutation_locks.remove(worker_id);

            for model_id in Self::worker_model_ids(&worker) {
                self.remove_worker_from_model_index(&model_id, worker.url());
                // Clean up per-model retry config when no workers remain for this model
                let model_empty = self.model_index.get(&model_id).is_none_or(|w| w.is_empty());
                if model_empty {
                    self.model_retry_configs.remove(&model_id);
                }
            }

            // Remove from type index
            if let Some(mut type_workers) = self.type_workers.get_mut(worker.worker_type()) {
                type_workers.retain(|id| id != worker_id);
            }

            // Remove from connection mode index
            if let Some(mut conn_workers) =
                self.connection_workers.get_mut(worker.connection_mode())
            {
                conn_workers.retain(|id| id != worker_id);
            }

            worker.set_healthy(false);
            Metrics::remove_worker_metrics(worker.url());

            // Sync removal to mesh if enabled (no-op if mesh is not enabled)
            {
                let guard = self.mesh_sync.read();
                if let Some(ref mesh_sync) = *guard {
                    mesh_sync.remove_worker_state(worker_id.as_str());
                }
            }

            let _ = self.event_tx.send(WorkerEvent::Removed {
                url: worker.url().to_string(),
            });

            Some(worker)
        } else {
            None
        }
    }

    /// Remove a worker by URL
    pub fn remove_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        if let Some((_, worker_id)) = self.url_to_id.remove(url) {
            self.remove(&worker_id)
        } else {
            None
        }
    }

    /// Get a worker by ID
    pub fn get(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        self.workers.get(worker_id).map(|entry| entry.clone())
    }

    /// Get a worker by URL
    pub fn get_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        self.url_to_id.get(url).and_then(|id| self.get(&id))
    }

    /// Empty worker slice constant for returning when no workers found
    const EMPTY_WORKERS: &'static [Arc<dyn Worker>] = &[];

    /// Get all workers for a model (O(1) optimized, lock-free)
    /// Returns an Arc to the immutable worker slice - just an atomic refcount bump.
    /// This is the fastest possible read path with zero contention.
    pub fn get_by_model(&self, model_id: &str) -> Arc<[Arc<dyn Worker>]> {
        self.model_index
            .get(model_id)
            .map(|workers| Arc::clone(&workers))
            .unwrap_or_else(|| Arc::from(Self::EMPTY_WORKERS))
    }

    /// Get all workers by worker type
    pub fn get_by_type(&self, worker_type: WorkerType) -> Vec<Arc<dyn Worker>> {
        self.type_workers
            .get(&worker_type)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Update worker health status and sync to mesh
    pub fn update_worker_health(&self, worker_id: &WorkerId, is_healthy: bool) {
        if let Some(worker) = self.workers.get(worker_id) {
            // Update worker health (if Worker trait has a method for this)
            // For now, we'll just sync to mesh

            // Sync to mesh if enabled (no-op if mesh is not enabled)
            {
                let guard = self.mesh_sync.read();
                if let Some(ref mesh_sync) = *guard {
                    mesh_sync.sync_worker_state(
                        worker_id.as_str().to_string(),
                        worker.model_id().to_string(),
                        worker.url().to_string(),
                        is_healthy,
                        0.0, // TODO: Get actual load
                        bincode::serialize(&worker.metadata().spec).unwrap_or_default(),
                    );
                }
            }

            let _ = self.event_tx.send(WorkerEvent::HealthChanged {
                url: worker.url().to_string(),
                healthy: is_healthy,
            });
        }
    }

    /// Get all prefill workers (regardless of bootstrap_port)
    pub fn get_prefill_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .filter_map(|entry| {
                let worker = entry.value();
                match worker.worker_type() {
                    WorkerType::Prefill => Some(worker.clone()),
                    _ => None,
                }
            })
            .collect()
    }

    /// Get all decode workers
    pub fn get_decode_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.get_by_type(WorkerType::Decode)
    }

    /// Get all workers by connection mode
    pub fn get_by_connection(&self, connection_mode: ConnectionMode) -> Vec<Arc<dyn Worker>> {
        self.connection_workers
            .get(&connection_mode)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get the number of workers in the registry
    pub fn len(&self) -> usize {
        self.workers.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    /// Get all workers
    pub fn get_all(&self) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get all workers with their IDs
    pub fn get_all_with_ids(&self) -> Vec<(WorkerId, Arc<dyn Worker>)> {
        self.workers
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Get all worker URLs
    pub fn get_all_urls(&self) -> Vec<String> {
        self.workers
            .iter()
            .map(|entry| entry.value().url().to_string())
            .collect()
    }

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

    /// Get all model IDs with workers (lock-free)
    pub fn get_models(&self) -> Vec<String> {
        self.model_index
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get workers filtered by multiple criteria
    ///
    /// This method allows flexible filtering of workers based on:
    /// - model_id: Filter by specific model
    /// - worker_type: Filter by worker type (Regular, Prefill, Decode)
    /// - connection_mode: Filter by connection mode (Http, Grpc)
    /// - runtime_type: Filter by runtime type (Sglang, Vllm, External)
    /// - healthy_only: Only return healthy workers
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

        // Apply remaining filters
        workers
            .into_iter()
            .filter(|w| {
                // Check worker_type if specified
                if let Some(ref wtype) = worker_type {
                    if *w.worker_type() != *wtype {
                        return false;
                    }
                }

                // Check connection_mode if specified
                if let Some(ref conn) = connection_mode {
                    if w.connection_mode() != conn {
                        return false;
                    }
                }

                // Check runtime_type if specified
                if let Some(ref rt) = runtime_type {
                    if w.metadata().spec.runtime_type != *rt {
                        return false;
                    }
                }

                // Check health if required
                if healthy_only && !w.is_healthy() {
                    return false;
                }

                true
            })
            .collect()
    }

    /// Get worker statistics (lock-free)
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

            match worker.circuit_breaker().state() {
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

    /// Get counts of regular and PD workers efficiently (O(1))
    /// This avoids the overhead of get_all() which allocates memory and iterates all workers
    pub fn get_worker_distribution(&self) -> (usize, usize) {
        // Use the existing type_workers index for O(1) lookup
        let regular_count = self
            .type_workers
            .get(&WorkerType::Regular)
            .map(|v| v.len())
            .unwrap_or(0);

        // Get total workers count efficiently from DashMap
        let total_workers = self.workers.len();

        // PD workers are any workers that are not Regular
        let pd_count = total_workers.saturating_sub(regular_count);

        (regular_count, pd_count)
    }

    /// Start a deadline-driven health checker for all workers in the registry.
    ///
    /// Each worker is checked according to its own `health_config.check_interval_secs`.
    /// The task sleeps until the next worker is due, so it only wakes when there is
    /// actual work to do — zero CPU when idle, no polling.
    pub(crate) fn start_health_checker(
        &self,
        default_interval_secs: u64,
        remove_unhealthy: bool,
        job_queue: Option<Arc<JobQueue>>,
    ) -> HealthChecker {
        let shutdown_notify = Arc::new(tokio::sync::Notify::new());
        let shutdown_clone = shutdown_notify.clone();
        let workers_ref = self.workers.clone();
        let job_queue = if remove_unhealthy { job_queue } else { None };

        #[expect(
            clippy::disallowed_methods,
            reason = "Health checker loop: runs for the lifetime of the registry, handle is stored in HealthChecker and abort() is called on drop"
        )]
        let handle = tokio::spawn(async move {
            // next_check[url] = Instant when the worker is next due for a health check.
            let mut next_check: HashMap<String, tokio::time::Instant> = HashMap::new();

            loop {
                let now = tokio::time::Instant::now();

                // Snapshot current workers from the registry
                let workers: Vec<Arc<dyn Worker>> = workers_ref
                    .iter()
                    .map(|entry| entry.value().clone())
                    .collect();

                // Sync schedule with registry: add new workers, prune removed
                // and disabled ones so stale deadlines don't cause wakeups.
                let checkable_urls: HashSet<String> = workers
                    .iter()
                    .filter(|w| !w.metadata().health_config.disable_health_check)
                    .map(|w| w.url().to_string())
                    .collect();
                next_check.retain(|url, _| checkable_urls.contains(url));
                for url in &checkable_urls {
                    next_check.entry(url.clone()).or_insert(now);
                }

                // Collect workers whose deadline has passed
                let due_workers: Vec<_> = workers
                    .iter()
                    .filter(|w| !w.metadata().health_config.disable_health_check)
                    .filter(|w| {
                        next_check
                            .get(w.url())
                            .is_some_and(|deadline| now >= *deadline)
                    })
                    .cloned()
                    .collect();

                // Run due health checks in parallel and schedule the next deadline
                if !due_workers.is_empty() {
                    for worker in &due_workers {
                        let secs = worker.metadata().health_config.check_interval_secs;
                        let secs = if secs > 0 {
                            secs
                        } else {
                            default_interval_secs
                        };
                        next_check.insert(
                            worker.url().to_string(),
                            now + tokio::time::Duration::from_secs(secs),
                        );
                    }
                    let futs: Vec<_> = due_workers
                        .into_iter()
                        .map(|w| async move {
                            let _ = w.check_health_async().await;
                            w
                        })
                        .collect();
                    let checked_workers = futures::future::join_all(futs).await;

                    // Submit removal jobs for workers that transitioned to unhealthy.
                    // Goes through the full removal workflow (policy registry,
                    // worker registry, metrics, load monitor).
                    if let Some(ref job_queue) = job_queue {
                        for worker in &checked_workers {
                            if !worker.is_healthy() {
                                let url = worker.url().to_string();
                                tracing::warn!(
                                    worker_url = %url,
                                    "Removing unhealthy worker from registry"
                                );
                                next_check.remove(&url);
                                if let Err(e) = job_queue
                                    .submit(Job::RemoveWorker { url: url.clone() })
                                    .await
                                {
                                    tracing::error!(
                                        worker_url = %url,
                                        error = %e,
                                        "Failed to submit worker removal job"
                                    );
                                }
                            }
                        }
                    }
                }

                // Sleep until the earliest deadline or until shutdown is signalled.
                // If the registry is empty, sleep for the default interval then re-scan
                // (new workers may have been added).
                let sleep_until = next_check.values().min().copied().unwrap_or_else(|| {
                    now + tokio::time::Duration::from_secs(default_interval_secs)
                });

                tokio::select! {
                    () = tokio::time::sleep_until(sleep_until) => {}
                    () = shutdown_clone.notified() => {
                        tracing::debug!("Registry health checker shutting down");
                        break;
                    }
                }
            }
        });

        HealthChecker::new(handle, shutdown_notify)
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl smg_mesh::WorkerStateSubscriber for WorkerRegistry {
    fn on_remote_worker_state(&self, state: &smg_mesh::WorkerState) {
        use openai_protocol::model_card::ModelCard;

        // If worker already exists at this URL, update its health status
        // from the mesh state. Don't re-register — the existing worker has
        // full config from its creation workflow.
        if let Some(existing) = self.get_by_url(&state.url) {
            existing.set_healthy(state.health);
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
                super::worker_builder::BasicWorkerBuilder::from_spec(spec).build()
            }
            _ => super::worker_builder::BasicWorkerBuilder::new(&state.url)
                .model(ModelCard::new(&state.model_id))
                .build(),
        };

        worker.set_healthy(state.health);

        // register_inner skips mesh sync to avoid version-bump loop.
        if let Some(id) = self.register_inner(Arc::new(worker)) {
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
    use crate::core::{circuit_breaker::CircuitBreakerConfig, BasicWorkerBuilder};

    #[test]
    fn test_worker_registry() {
        let registry = WorkerRegistry::new();

        // Create a worker with labels
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

        // Remove worker
        registry.remove(&worker_id);
        assert!(registry.get(&worker_id).is_none());
    }

    #[test]
    fn test_model_index_fast_lookup() {
        let registry = WorkerRegistry::new();

        // Create workers for different models
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

    #[tokio::test]
    async fn test_health_checker_removes_unhealthy_workers() {
        use openai_protocol::worker::HealthCheckConfig;

        let registry = WorkerRegistry::new();

        // Worker pointing at a non-existent URL so health checks fail immediately.
        // failure_threshold=1 so it becomes unhealthy after the first failed check.
        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "test-model".to_string());

        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://127.0.0.1:1")
                .worker_type(WorkerType::Regular)
                .labels(labels)
                .health_config(HealthCheckConfig {
                    failure_threshold: 1,
                    success_threshold: 1,
                    timeout_secs: 1,
                    check_interval_secs: 1,
                    disable_health_check: false,
                })
                .build(),
        );

        registry.register(Arc::from(worker)).unwrap();
        assert_eq!(registry.stats().total_workers, 1);

        // Start health checker with remove_unhealthy=true but no job queue.
        // Without a job queue the removal job can't be submitted, but the worker
        // should still be marked unhealthy by the health check.
        let _hc = registry.start_health_checker(1, true, None);

        // Wait for the health check to run and mark the worker unhealthy
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

        assert_eq!(
            registry.stats().healthy_workers,
            0,
            "Worker should be marked unhealthy after failed health checks"
        );
    }

    #[tokio::test]
    async fn test_health_checker_keeps_unhealthy_workers_when_disabled() {
        use openai_protocol::worker::HealthCheckConfig;

        let registry = WorkerRegistry::new();

        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "test-model".to_string());

        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://127.0.0.1:1")
                .worker_type(WorkerType::Regular)
                .labels(labels)
                .health_config(HealthCheckConfig {
                    failure_threshold: 1,
                    success_threshold: 1,
                    timeout_secs: 1,
                    check_interval_secs: 1,
                    disable_health_check: false,
                })
                .build(),
        );

        registry.register(Arc::from(worker)).unwrap();
        assert_eq!(registry.stats().total_workers, 1);

        // Start health checker with remove_unhealthy=false (default behavior)
        let _hc = registry.start_health_checker(1, false, None);

        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

        assert_eq!(
            registry.stats().total_workers,
            1,
            "Worker should remain in the registry when remove_unhealthy is false"
        );
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

        // No config initially
        assert!(registry.get_retry_config("llama-3").is_none());

        // Set config for a model (retries enabled)
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

        // Set retry config for the model
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
    fn test_worker_event_broadcast() {
        let registry = WorkerRegistry::new();
        let mut rx = registry.subscribe_events();

        // Create and register a worker
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
            WorkerEvent::Registered { url } => {
                assert_eq!(url, "http://event-worker:8080");
            }
            other => panic!("Expected Registered event, got: {other:?}"),
        }

        // Remove the worker
        registry.remove(&worker_id);

        // Should receive Removed event
        let event = rx.try_recv().unwrap();
        match event {
            WorkerEvent::Removed { url } => {
                assert_eq!(url, "http://event-worker:8080");
            }
            other => panic!("Expected Removed event, got: {other:?}"),
        }
    }
}
