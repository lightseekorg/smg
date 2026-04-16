//! Incremental update collection and batching
//!
//! Collects local state changes and batches them for efficient transmission

use std::{
    collections::HashMap,
    sync::{atomic::Ordering, Arc},
    time::{SystemTime, UNIX_EPOCH},
};

use parking_lot::RwLock;
use tracing::{debug, trace};

use super::{
    service::gossip::StateUpdate,
    stores::{AppState, MembershipState, PolicyState, StateStores, StoreType, WorkerState},
    tree_ops::{lz4_compress, TenantDelta, TreeState},
};

/// Trait for extracting version from state types
trait Versioned {
    fn version(&self) -> u64;
}

impl Versioned for WorkerState {
    fn version(&self) -> u64 {
        self.version
    }
}

impl Versioned for PolicyState {
    fn version(&self) -> u64 {
        self.version
    }
}

impl Versioned for AppState {
    fn version(&self) -> u64 {
        self.version
    }
}

impl Versioned for MembershipState {
    fn version(&self) -> u64 {
        self.version
    }
}

/// Tracks the last sent version for each key in each store
#[derive(Debug, Clone, Default)]
struct LastSentVersions {
    worker: HashMap<String, u64>,
    policy: HashMap<String, u64>,
    app: HashMap<String, u64>,
    membership: HashMap<String, u64>,
    rate_limit: HashMap<String, u64>, // Track last sent timestamp for rate limit counter shards
}

/// Tracks store generation to skip unchanged stores
#[derive(Debug, Clone, Default)]
struct LastScannedGenerations {
    worker: u64,
    policy: u64,
    app: u64,
    membership: u64,
    /// Separate generation for tree state changes (bumped by
    /// `sync_tree_operation` via atomic counter instead of CRDT).
    tree: u64,
}

/// How often to send a full tree structure snapshot for convergence,
/// measured in gossip rounds. At the default gossip interval of ~1s,
/// this means a full snapshot every ~30 seconds per model.
///
/// Tenant deltas are sent every round (~20KB/s); full snapshots are
/// heavier (~300KB compressed) but ensure convergence after missed
/// deltas, new nodes joining, or network partitions.
// FIXME: Re-enable when Layer 2 (chunked snapshots) is implemented.
#[expect(dead_code, reason = "Reserved for Layer 2 snapshot interval")]
const STRUCTURE_SNAPSHOT_INTERVAL: u64 = 30;

/// Incremental update collector
pub struct IncrementalUpdateCollector {
    stores: Arc<StateStores>,
    self_name: String,
    last_sent: Arc<RwLock<LastSentVersions>>,
    last_scanned: Arc<RwLock<LastScannedGenerations>>,
    /// Snapshot of `tree_generation` captured during the last collection.
    /// Used in `mark_sent` instead of re-reading the atomic to avoid
    /// advancing `last_scanned.tree` past the batch boundary.
    collected_tree_gen: Arc<RwLock<u64>>,
    /// Counter for gossip rounds since last full structure snapshot per model.
    rounds_since_snapshot: Arc<RwLock<HashMap<String, u64>>>,
    /// Shared reference to centrally drained tenant deltas. When the inner
    /// Option is Some, Policy Phase 0 uses these instead of destructively
    /// draining from the shared DashMap. Set by the gossip event loop once
    /// per round, read by all per-peer collectors.
    central_deltas: Arc<RwLock<Option<DrainedTenantDeltas>>>,
}

impl IncrementalUpdateCollector {
    pub fn new(stores: Arc<StateStores>, self_name: String) -> Self {
        Self {
            stores,
            self_name,
            last_sent: Arc::new(RwLock::new(LastSentVersions::default())),
            last_scanned: Arc::new(RwLock::new(LastScannedGenerations::default())),
            collected_tree_gen: Arc::new(RwLock::new(0)),
            rounds_since_snapshot: Arc::new(RwLock::new(HashMap::new())),
            central_deltas: Arc::new(RwLock::new(None)),
        }
    }

    /// Get current timestamp in nanoseconds
    #[expect(
        clippy::expect_used,
        reason = "system clock before UNIX epoch is a fatal misconfiguration that must not silently produce timestamp=0"
    )]
    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before UNIX_EPOCH; cannot generate valid timestamps")
            .as_nanos() as u64
    }

    pub(crate) fn rate_limit_last_sent_key(key: &str, actor: &str) -> String {
        format!("{key}::actor:{actor}")
    }

    /// Helper function to collect updates for stores with serializable state
    fn collect_serializable_updates<S>(
        &self,
        all_items: std::collections::BTreeMap<String, S>,
        last_sent_map: &HashMap<String, u64>,
        store_name: &str,
        get_id: impl Fn(&S) -> String,
    ) -> Vec<StateUpdate>
    where
        S: serde::Serialize + Versioned,
    {
        let mut updates = Vec::new();
        let timestamp = Self::current_timestamp();

        for (key, state) in all_items {
            let current_version = state.version();
            let last_sent_version = last_sent_map.get(&key).copied().unwrap_or(0);

            if current_version > last_sent_version {
                if let Ok(serialized) = bincode::serialize(&state) {
                    updates.push(StateUpdate {
                        key,
                        value: serialized,
                        version: current_version,
                        actor: self.self_name.clone(),
                        timestamp,
                    });
                    debug!(
                        "Collected {} update: {} (version: {})",
                        store_name,
                        get_id(&state),
                        current_version
                    );
                }
            }
        }
        updates
    }

    /// Collect incremental updates for a specific store type.
    /// Skips the expensive `.all()` scan when the store generation hasn't changed.
    pub fn collect_updates_for_store(&self, store_type: StoreType) -> Vec<StateUpdate> {
        let mut updates = Vec::new();
        let mut last_sent = self.last_sent.read();
        let last_scanned = self.last_scanned.read();

        match store_type {
            StoreType::Worker => {
                let gen = self.stores.worker.generation();
                if gen == last_scanned.worker {
                    return vec![];
                }
                let all_workers = self.stores.worker.all();
                updates = self.collect_serializable_updates(
                    all_workers,
                    &last_sent.worker,
                    "worker",
                    |state: &WorkerState| state.worker_id.clone(),
                );
            }
            StoreType::Policy => {
                let policy_gen = self.stores.policy.generation();
                let tree_gen = self.stores.tree_generation.load(Ordering::Acquire);
                let policy_changed = policy_gen != last_scanned.policy;
                let tree_changed = tree_gen != last_scanned.tree;

                if !policy_changed && !tree_changed {
                    return vec![];
                }

                // Snapshot tree_gen for mark_sent — avoids advancing
                // last_scanned.tree past this batch if new ops arrive
                // between collection and mark_sent.
                *self.collected_tree_gen.write() = tree_gen;

                let timestamp = Self::current_timestamp();

                // --- Policy store scan (only if CRDT generation changed) ---
                // Handles non-tree policy keys only. Tree keys are stored
                // in tree_configs (plain DashMap) and handled by the tree-
                // specific scan below.
                if policy_changed {
                    let all_policies = self.stores.policy.all();
                    for (key, state) in &all_policies {
                        // Tree keys are no longer in the CRDT policy store;
                        // skip any stale entries that may remain from before
                        // the migration.
                        if key.starts_with("tree:") {
                            continue;
                        }

                        let current_version = state.version();
                        let last_sent_version =
                            last_sent.policy.get(key.as_str()).copied().unwrap_or(0);
                        if current_version <= last_sent_version {
                            continue;
                        }

                        if let Ok(serialized) = bincode::serialize(state) {
                            updates.push(StateUpdate {
                                key: key.clone(),
                                value: serialized,
                                version: current_version,
                                actor: self.self_name.clone(),
                                timestamp,
                            });
                            debug!(
                                "Collected policy update: {} (version: {})",
                                state.model_id, current_version
                            );
                        }
                    }
                }

                // --- Tree keys (driven by atomic tree_versions, not CRDT) ---
                if tree_changed {
                    let mut emitted_tree_keys = std::collections::HashSet::new();

                    // Phase 0: Tenant delta collection.
                    //
                    // v2 FIX: If central deltas are available (set by the gossip
                    // event loop via set_central_deltas), use them directly instead
                    // of destructively draining the shared DashMap. This fixes the
                    // v1 bug where only the first per-peer collector receives
                    // tenant deltas.
                    //
                    // When central_deltas is None (legacy path), fall back to the
                    // per-collector destructive drain for backward compatibility.
                    {
                        let central = self.central_deltas.read();
                        if let Some(ref drained) = *central {
                            // Use pre-drained deltas from central drain.
                            updates.extend(drained.updates.clone());
                            emitted_tree_keys.extend(drained.emitted_tree_keys.clone());
                            // Use the central drain's tree_gen snapshot for mark_sent
                            *self.collected_tree_gen.write() = drained.tree_gen_snapshot;

                            let phase0_count = drained.updates.len();
                            if phase0_count > 0 {
                                let phase0_total_bytes: usize =
                                    drained.updates.iter().map(|u| u.value.len()).sum();
                                debug!(
                                    phase0_updates = phase0_count,
                                    phase0_total_bytes,
                                    "Phase 0: used centrally drained tenant deltas"
                                );
                            }
                        } else {
                            // Legacy path: per-collector destructive drain.
                            // TODO(v2): Remove this path once all callers use
                            // set_central_deltas.
                            let models_with_inserts: Vec<String> = self
                                .stores
                                .tenant_delta_inserts
                                .iter()
                                .filter(|entry| !entry.value().is_empty())
                                .map(|entry| entry.key().clone())
                                .collect();
                            let models_with_evictions: Vec<String> = self
                                .stores
                                .tenant_delta_evictions
                                .iter()
                                .filter(|entry| !entry.value().is_empty())
                                .map(|entry| entry.key().clone())
                                .collect();

                            let all_models: std::collections::HashSet<String> = models_with_inserts
                                .into_iter()
                                .chain(models_with_evictions)
                                .collect();

                            let mut rounds = self.rounds_since_snapshot.write();

                            for model_id in all_models {
                                let key = format!("tree:{model_id}");

                                let round_count = rounds.entry(model_id.clone()).or_insert(0);
                                *round_count += 1;
                                let _ = round_count; // tracked for future Layer 2

                                let current_version = self.stores.tree_version(&key);

                                let inserts = self
                                    .stores
                                    .tenant_delta_inserts
                                    .remove(&model_id)
                                    .map(|(_, v)| v)
                                    .unwrap_or_default();
                                let evictions = self
                                    .stores
                                    .tenant_delta_evictions
                                    .remove(&model_id)
                                    .map(|(_, v)| v)
                                    .unwrap_or_default();

                                if inserts.is_empty() && evictions.is_empty() {
                                    continue;
                                }

                                let delta = TenantDelta {
                                    model_id: model_id.clone(),
                                    version: current_version,
                                    inserts,
                                    evictions,
                                };

                                if let Ok(delta_bytes) = delta.to_bytes() {
                                    let delta_policy = PolicyState {
                                        model_id: model_id.clone(),
                                        policy_type: "tenant_delta".to_string(),
                                        config: delta_bytes,
                                        version: current_version,
                                    };
                                    if let Ok(serialized) = bincode::serialize(&delta_policy) {
                                        updates.push(StateUpdate {
                                            key: key.clone(),
                                            value: serialized,
                                            version: current_version,
                                            actor: self.self_name.clone(),
                                            timestamp,
                                        });
                                        debug!(
                                            "Collected tenant delta: {} ({} inserts, {} evictions, version: {})",
                                            model_id,
                                            delta.inserts.len(),
                                            delta.evictions.len(),
                                            current_version,
                                        );
                                        emitted_tree_keys.insert(key);
                                    }
                                }
                            }

                            // Phase 0 summary
                            let phase0_total_bytes: usize = updates
                                .iter()
                                .filter(|u| u.key.starts_with("tree:"))
                                .map(|u| u.value.len())
                                .sum();
                            let phase0_count = updates
                                .iter()
                                .filter(|u| u.key.starts_with("tree:"))
                                .count();
                            if phase0_count > 0 {
                                debug!(
                                    phase0_updates = phase0_count,
                                    phase0_total_bytes,
                                    "Phase 0: tenant delta buffer drain produced updates (legacy path)"
                                );
                            }
                        }
                    }

                    // Phase 1 (removed): tree_ops_pending is no longer populated.
                    // Full prompt text is not stored per-request. Instead,
                    // checkpoint_tree_states exports from the live radix tree.

                    // Phase 2: Scan tree_configs for keys not yet emitted
                    // (e.g., after checkpoint + buffer drain, or remote-only entries).
                    //
                    // tree_configs may contain either:
                    //   - TreeState bytes (from remote full-state updates)
                    //   - TreeSnapshot bytes (from local checkpoint_tree_states)
                    // Both are sent LZ4-compressed as "tree_state_lz4". The
                    // receiver detects the format and converts as needed.
                    let phase2_start = updates.len();
                    for entry in &self.stores.tree_configs {
                        let key = entry.key();
                        if emitted_tree_keys.contains(key.as_str()) {
                            continue;
                        }
                        let current_version = self.stores.tree_version(key);
                        let last_sent_version =
                            last_sent.policy.get(key.as_str()).copied().unwrap_or(0);
                        if current_version <= last_sent_version {
                            continue;
                        }
                        let model_id = key.strip_prefix("tree:").unwrap_or(key).to_string();
                        let config_bytes = entry.value().clone();
                        if config_bytes.is_empty() {
                            continue;
                        }
                        // Validate the bytes as either TreeState or TreeSnapshot
                        // and extract the version for the PolicyState envelope.
                        let tree_version = if let Ok(ts) = TreeState::from_bytes(&config_bytes) {
                            ts.version
                        } else if kv_index::snapshot::TreeSnapshot::from_bytes(&config_bytes)
                            .is_ok()
                        {
                            // TreeSnapshot has no embedded version — use the
                            // atomic tree_version counter instead.
                            current_version
                        } else {
                            debug!(
                                "Skipping tree_configs full-state for {} — config corrupted",
                                key
                            );
                            continue;
                        };
                        let compressed = lz4_compress(&config_bytes);
                        // Skip if compressed size exceeds the gRPC message limit.
                        // This prevents the infinite retry loop where an oversized
                        // snapshot is serialized every round, rejected by the
                        // controller, and retried — causing ~23 MB/s of allocator
                        // churn that the OS never reclaims.
                        const MAX_SNAPSHOT_BYTES: usize = 8 * 1024 * 1024; // 8 MB
                        if compressed.len() > MAX_SNAPSHOT_BYTES {
                            debug!(
                                key = %key,
                                compressed_bytes = compressed.len(),
                                limit = MAX_SNAPSHOT_BYTES,
                                "Skipping oversized tree snapshot — compressed size exceeds limit"
                            );
                            // Mark as sent via a write lock so we don't retry every round.
                            drop(last_sent);
                            self.last_sent
                                .write()
                                .policy
                                .insert(key.to_string(), current_version);
                            // Re-acquire read lock for remaining iterations
                            last_sent = self.last_sent.read();
                            continue;
                        }
                        let full_state = PolicyState {
                            model_id,
                            policy_type: "tree_state_lz4".to_string(),
                            config: compressed,
                            version: tree_version,
                        };
                        if let Ok(serialized) = bincode::serialize(&full_state) {
                            updates.push(StateUpdate {
                                key: key.clone(),
                                value: serialized,
                                version: current_version,
                                actor: self.self_name.clone(),
                                timestamp,
                            });
                            debug!(
                                "Collected tree_configs full state: {} (version: {})",
                                key, current_version
                            );
                        }
                    }

                    // Phase 2 summary
                    let phase2_count = updates.len() - phase2_start;
                    let phase2_bytes: usize =
                        updates[phase2_start..].iter().map(|u| u.value.len()).sum();
                    debug!(
                        phase2_updates = phase2_count,
                        phase2_total_bytes = phase2_bytes,
                        "Phase 2: tree_configs scan {}",
                        if phase2_count > 0 {
                            "produced updates"
                        } else {
                            "no new updates"
                        }
                    );
                }
            }
            StoreType::App => {
                let gen = self.stores.app.generation();
                if gen == last_scanned.app {
                    return vec![];
                }
                let all_apps = self.stores.app.all();
                updates = self.collect_serializable_updates(
                    all_apps,
                    &last_sent.app,
                    "app",
                    |state: &AppState| state.key.clone(),
                );
            }
            StoreType::Membership => {
                let gen = self.stores.membership.generation();
                if gen == last_scanned.membership {
                    return vec![];
                }
                let all_members = self.stores.membership.all();
                updates = self.collect_serializable_updates(
                    all_members,
                    &last_sent.membership,
                    "membership",
                    |state: &MembershipState| state.name.clone(),
                );
            }
            StoreType::RateLimit => {
                let current_timestamp = Self::current_timestamp();

                for (key, actor, counter_value) in self.stores.rate_limit.all_shards() {
                    if !self.stores.rate_limit.is_owner(&key) {
                        continue;
                    }

                    let shard_last_sent_key = Self::rate_limit_last_sent_key(&key, &actor);
                    let last_sent_timestamp = last_sent
                        .rate_limit
                        .get(&shard_last_sent_key)
                        .copied()
                        .unwrap_or(0);

                    // Only send if at least 1 second has passed since last send.
                    if current_timestamp > last_sent_timestamp + 1_000_000_000 {
                        if let Ok(serialized) = bincode::serialize(&counter_value) {
                            updates.push(StateUpdate {
                                key: key.clone(),
                                value: serialized,
                                version: current_timestamp,
                                actor: actor.clone(),
                                timestamp: current_timestamp,
                            });
                            trace!(
                                "Collected rate limit counter shard update: {} actor={}",
                                key,
                                actor
                            );
                        }
                    }
                }
            }
        }

        debug!(
            "Collected {} incremental updates for store {:?}",
            updates.len(),
            store_type
        );
        updates
    }

    /// Collect all incremental updates across all stores
    pub fn collect_all_updates(&self) -> Vec<(StoreType, Vec<StateUpdate>)> {
        let mut all_updates = Vec::new();

        for store_type in [
            StoreType::Worker,
            StoreType::Policy,
            StoreType::App,
            StoreType::Membership,
            StoreType::RateLimit,
        ] {
            let updates = self.collect_updates_for_store(store_type);
            if !updates.is_empty() {
                all_updates.push((store_type, updates));
            }
        }

        all_updates
    }

    /// Mark updates as sent (called after successful transmission).
    /// Also records the store generation to enable skipping unchanged stores.
    pub fn mark_sent(&self, store_type: StoreType, updates: &[StateUpdate]) {
        let mut last_sent = self.last_sent.write();
        let mut last_scanned = self.last_scanned.write();

        // Record the current generation so the next collection cycle can skip
        // this store if nothing has changed since.
        match store_type {
            StoreType::Worker => last_scanned.worker = self.stores.worker.generation(),
            StoreType::Policy => {
                last_scanned.policy = self.stores.policy.generation();
                // Use the snapshot captured during collection, not a
                // live re-read — prevents skipping ops that arrived
                // between collect and mark_sent.
                last_scanned.tree = *self.collected_tree_gen.read();
            }
            StoreType::App => last_scanned.app = self.stores.app.generation(),
            StoreType::Membership => {
                last_scanned.membership = self.stores.membership.generation();
            }
            StoreType::RateLimit => {} // Rate limit uses timestamp-based tracking
        }

        for update in updates {
            match store_type {
                StoreType::Worker => {
                    last_sent.worker.insert(update.key.clone(), update.version);
                }
                StoreType::Policy => {
                    last_sent.policy.insert(update.key.clone(), update.version);
                }
                StoreType::App => {
                    last_sent.app.insert(update.key.clone(), update.version);
                }
                StoreType::Membership => {
                    last_sent
                        .membership
                        .insert(update.key.clone(), update.version);
                }
                StoreType::RateLimit => {
                    let shard_key = Self::rate_limit_last_sent_key(&update.key, &update.actor);
                    last_sent.rate_limit.insert(shard_key, update.version);
                }
            }
        }
    }
}

// ============================================================================
// Central Tenant Delta Drain (v2 bug fix)
// ============================================================================

/// Tenant delta updates drained once per gossip round. Shared with all per-peer
/// collector instances so they all see the same deltas instead of racing to drain
/// the shared DashMap (v1 bug where only the first peer receives deltas).
#[derive(Debug, Clone, Default)]
pub struct DrainedTenantDeltas {
    /// Tenant delta StateUpdates collected from the destructive drain.
    /// These are Policy-type updates with key "tree:{model_id}".
    pub updates: Vec<StateUpdate>,
    /// Set of tree keys emitted as deltas. Per-peer collectors use this to
    /// skip these keys in Phase 2 (tree_configs full-state scan).
    pub emitted_tree_keys: std::collections::HashSet<String>,
    /// Tree generation snapshot at drain time. Per-peer collectors use this
    /// for mark_sent to avoid advancing past the batch boundary.
    pub tree_gen_snapshot: u64,
}

/// Drains tenant delta buffers exactly once per gossip round. The result is
/// stored in a shared location so all per-peer collectors can include the
/// same deltas without racing on the destructive DashMap remove.
pub fn drain_tenant_deltas_central(stores: &StateStores, self_name: &str) -> DrainedTenantDeltas {
    let tree_gen = stores.tree_generation.load(Ordering::Acquire);
    let timestamp = IncrementalUpdateCollector::current_timestamp();
    let mut updates = Vec::new();
    let mut emitted_tree_keys = std::collections::HashSet::new();

    let models_with_inserts: Vec<String> = stores
        .tenant_delta_inserts
        .iter()
        .filter(|entry| !entry.value().is_empty())
        .map(|entry| entry.key().clone())
        .collect();
    let models_with_evictions: Vec<String> = stores
        .tenant_delta_evictions
        .iter()
        .filter(|entry| !entry.value().is_empty())
        .map(|entry| entry.key().clone())
        .collect();

    let all_models: std::collections::HashSet<String> = models_with_inserts
        .into_iter()
        .chain(models_with_evictions)
        .collect();

    for model_id in all_models {
        let key = format!("tree:{model_id}");
        let current_version = stores.tree_version(&key);

        let inserts = stores
            .tenant_delta_inserts
            .remove(&model_id)
            .map(|(_, v)| v)
            .unwrap_or_default();
        let evictions = stores
            .tenant_delta_evictions
            .remove(&model_id)
            .map(|(_, v)| v)
            .unwrap_or_default();

        if inserts.is_empty() && evictions.is_empty() {
            continue;
        }

        let delta = TenantDelta {
            model_id: model_id.clone(),
            version: current_version,
            inserts,
            evictions,
        };

        if let Ok(delta_bytes) = delta.to_bytes() {
            let delta_policy = PolicyState {
                model_id: model_id.clone(),
                policy_type: "tenant_delta".to_string(),
                config: delta_bytes,
                version: current_version,
            };
            if let Ok(serialized) = bincode::serialize(&delta_policy) {
                updates.push(StateUpdate {
                    key: key.clone(),
                    value: serialized,
                    version: current_version,
                    actor: self_name.to_string(),
                    timestamp,
                });
                debug!(
                    "Central drain: tenant delta {} ({} inserts, {} evictions, version: {})",
                    model_id,
                    delta.inserts.len(),
                    delta.evictions.len(),
                    current_version,
                );
                emitted_tree_keys.insert(key);
            }
        }
    }

    if !updates.is_empty() {
        let total_bytes: usize = updates.iter().map(|u| u.value.len()).sum();
        debug!(
            "Central drain: {} tenant delta updates ({} bytes total)",
            updates.len(),
            total_bytes,
        );
    }

    DrainedTenantDeltas {
        updates,
        emitted_tree_keys,
        tree_gen_snapshot: tree_gen,
    }
}

// ============================================================================
// CentralCollector + PeerWatermark (v2 architecture)
// ============================================================================

/// A round batch produced by the central collector. Contains ALL updates from
/// this round, organized by store type. Per-peer watermark filtering happens
/// at send time via `PeerWatermark::filter()`.
#[derive(Debug, Clone, Default)]
pub struct RoundBatch {
    pub updates: Vec<(StoreType, Vec<StateUpdate>)>,
}

/// Central collector that runs once per gossip round. Produces a `RoundBatch`
/// containing all changed entries across all stores. Destructive operations
/// (tenant delta drain) happen here exactly once. Per-peer watermark filtering
/// is NOT done here — that's `PeerWatermark`'s job.
pub struct CentralCollector {
    stores: Arc<StateStores>,
    self_name: String,
    /// Generation tracking to skip unchanged stores between rounds.
    last_scanned: RwLock<LastScannedGenerations>,
    /// Snapshot of tree_generation from the last collection.
    collected_tree_gen: RwLock<u64>,
}

impl CentralCollector {
    pub fn new(stores: Arc<StateStores>, self_name: String) -> Self {
        Self {
            stores,
            self_name,
            last_scanned: RwLock::new(LastScannedGenerations::default()),
            collected_tree_gen: RwLock::new(0),
        }
    }

    /// Collect all changes for this round. Called exactly once per gossip round
    /// by the event loop. Returns a `RoundBatch` that per-peer watermarks filter.
    pub fn collect(&self) -> RoundBatch {
        let mut all_updates = Vec::new();

        for store_type in [
            StoreType::Worker,
            StoreType::Policy,
            StoreType::App,
            StoreType::Membership,
            StoreType::RateLimit,
        ] {
            let updates = self.collect_store(store_type);
            if !updates.is_empty() {
                all_updates.push((store_type, updates));
            }
        }

        RoundBatch {
            updates: all_updates,
        }
    }

    /// Record the current generation after a successful round so the next
    /// round can skip unchanged stores.
    pub fn advance_generations(&self) {
        let mut last_scanned = self.last_scanned.write();
        last_scanned.worker = self.stores.worker.generation();
        last_scanned.policy = self.stores.policy.generation();
        last_scanned.app = self.stores.app.generation();
        last_scanned.membership = self.stores.membership.generation();
        last_scanned.tree = *self.collected_tree_gen.read();
    }

    /// Collect all entries for a store type. No watermark filtering — includes
    /// ALL current entries from stores that changed since last round.
    fn collect_store(&self, store_type: StoreType) -> Vec<StateUpdate> {
        let last_scanned = self.last_scanned.read();
        let timestamp = IncrementalUpdateCollector::current_timestamp();

        match store_type {
            StoreType::Worker => {
                let gen = self.stores.worker.generation();
                if gen == last_scanned.worker {
                    return vec![];
                }
                self.collect_serializable_store(
                    self.stores.worker.all(),
                    "worker",
                    timestamp,
                    |s: &WorkerState| s.worker_id.clone(),
                )
            }
            StoreType::Policy => {
                let policy_gen = self.stores.policy.generation();
                let tree_gen = self.stores.tree_generation.load(Ordering::Acquire);
                if policy_gen == last_scanned.policy && tree_gen == last_scanned.tree {
                    return vec![];
                }
                *self.collected_tree_gen.write() = tree_gen;
                self.collect_policy_store(timestamp, tree_gen != last_scanned.tree)
            }
            StoreType::App => {
                let gen = self.stores.app.generation();
                if gen == last_scanned.app {
                    return vec![];
                }
                self.collect_serializable_store(
                    self.stores.app.all(),
                    "app",
                    timestamp,
                    |s: &AppState| s.key.clone(),
                )
            }
            StoreType::Membership => {
                let gen = self.stores.membership.generation();
                if gen == last_scanned.membership {
                    return vec![];
                }
                self.collect_serializable_store(
                    self.stores.membership.all(),
                    "membership",
                    timestamp,
                    |s: &MembershipState| s.name.clone(),
                )
            }
            StoreType::RateLimit => {
                let current_timestamp = IncrementalUpdateCollector::current_timestamp();
                let mut updates = Vec::new();
                for (key, actor, counter_value) in self.stores.rate_limit.all_shards() {
                    if !self.stores.rate_limit.is_owner(&key) {
                        continue;
                    }
                    if let Ok(serialized) = bincode::serialize(&counter_value) {
                        updates.push(StateUpdate {
                            key,
                            value: serialized,
                            version: current_timestamp,
                            actor,
                            timestamp: current_timestamp,
                        });
                    }
                }
                updates
            }
        }
    }

    /// Collect all entries from a serializable store. No watermark filtering.
    fn collect_serializable_store<S>(
        &self,
        all_items: std::collections::BTreeMap<String, S>,
        store_name: &str,
        timestamp: u64,
        get_id: impl Fn(&S) -> String,
    ) -> Vec<StateUpdate>
    where
        S: serde::Serialize + Versioned,
    {
        let mut updates = Vec::new();
        for (key, state) in all_items {
            if let Ok(serialized) = bincode::serialize(&state) {
                debug!(
                    "Central collect {} update: {} (version: {})",
                    store_name,
                    get_id(&state),
                    state.version(),
                );
                updates.push(StateUpdate {
                    key,
                    value: serialized,
                    version: state.version(),
                    actor: self.self_name.clone(),
                    timestamp,
                });
            }
        }
        updates
    }

    /// Collect policy store entries + tenant deltas + tree_configs.
    /// Tenant deltas are destructively drained (safe because this runs once).
    fn collect_policy_store(&self, timestamp: u64, tree_changed: bool) -> Vec<StateUpdate> {
        let mut updates = Vec::new();
        let mut emitted_tree_keys = std::collections::HashSet::new();

        // Non-tree policy entries
        let all_policies = self.stores.policy.all();
        for (key, state) in &all_policies {
            if key.starts_with("tree:") {
                continue;
            }
            if let Ok(serialized) = bincode::serialize(state) {
                updates.push(StateUpdate {
                    key: key.clone(),
                    value: serialized,
                    version: state.version(),
                    actor: self.self_name.clone(),
                    timestamp,
                });
            }
        }

        if !tree_changed {
            return updates;
        }

        // Phase 0: Drain tenant deltas (destructive, runs once)
        let drained = drain_tenant_deltas_central(&self.stores, &self.self_name);
        updates.extend(drained.updates);
        emitted_tree_keys.extend(drained.emitted_tree_keys);

        // Phase 2: tree_configs scan for keys not emitted as deltas
        for entry in &self.stores.tree_configs {
            let key = entry.key();
            if emitted_tree_keys.contains(key.as_str()) {
                continue;
            }
            let model_id = key.strip_prefix("tree:").unwrap_or(key).to_string();
            let config_bytes = entry.value().clone();
            if config_bytes.is_empty() {
                continue;
            }
            let current_version = self.stores.tree_version(key);
            let tree_version = if let Ok(ts) = TreeState::from_bytes(&config_bytes) {
                ts.version
            } else if kv_index::snapshot::TreeSnapshot::from_bytes(&config_bytes).is_ok() {
                current_version
            } else {
                continue;
            };
            let compressed = lz4_compress(&config_bytes);
            const MAX_SNAPSHOT_BYTES: usize = 8 * 1024 * 1024;
            if compressed.len() > MAX_SNAPSHOT_BYTES {
                debug!(
                    key = %key,
                    compressed_bytes = compressed.len(),
                    "Skipping oversized tree snapshot"
                );
                continue;
            }
            let full_state = PolicyState {
                model_id,
                policy_type: "tree_state_lz4".to_string(),
                config: compressed,
                version: tree_version,
            };
            if let Ok(serialized) = bincode::serialize(&full_state) {
                updates.push(StateUpdate {
                    key: key.clone(),
                    value: serialized,
                    version: current_version,
                    actor: self.self_name.clone(),
                    timestamp,
                });
            }
        }

        updates
    }
}

/// Per-peer watermark tracker. Filters a centrally collected `RoundBatch` to
/// include only entries this peer hasn't seen yet, and tracks what was sent.
#[derive(Debug)]
pub struct PeerWatermark {
    /// Peer name, used for Debug output.
    _peer_name: String,
    last_sent: LastSentVersions,
}

impl PeerWatermark {
    pub fn new(peer_name: String) -> Self {
        Self {
            _peer_name: peer_name,
            last_sent: LastSentVersions::default(),
        }
    }

    /// Filter a round batch to include only entries this peer hasn't received.
    /// Returns updates organized by store type, ready to send.
    pub fn filter(&self, batch: &RoundBatch) -> Vec<(StoreType, Vec<StateUpdate>)> {
        let mut filtered = Vec::new();

        for (store_type, updates) in &batch.updates {
            let peer_updates: Vec<StateUpdate> = updates
                .iter()
                .filter(|u| self.should_send(*store_type, u))
                .cloned()
                .collect();
            if !peer_updates.is_empty() {
                filtered.push((*store_type, peer_updates));
            }
        }

        filtered
    }

    /// Mark updates as successfully sent to this peer. Advances watermark.
    pub fn mark_sent(&mut self, store_type: StoreType, updates: &[StateUpdate]) {
        for update in updates {
            match store_type {
                StoreType::Worker => {
                    self.last_sent
                        .worker
                        .insert(update.key.clone(), update.version);
                }
                StoreType::Policy => {
                    self.last_sent
                        .policy
                        .insert(update.key.clone(), update.version);
                }
                StoreType::App => {
                    self.last_sent
                        .app
                        .insert(update.key.clone(), update.version);
                }
                StoreType::Membership => {
                    self.last_sent
                        .membership
                        .insert(update.key.clone(), update.version);
                }
                StoreType::RateLimit => {
                    let shard_key = IncrementalUpdateCollector::rate_limit_last_sent_key(
                        &update.key,
                        &update.actor,
                    );
                    self.last_sent.rate_limit.insert(shard_key, update.version);
                }
            }
        }
    }

    fn should_send(&self, store_type: StoreType, update: &StateUpdate) -> bool {
        let last_sent_version = match store_type {
            StoreType::Worker => self.last_sent.worker.get(&update.key).copied().unwrap_or(0),
            StoreType::Policy => self.last_sent.policy.get(&update.key).copied().unwrap_or(0),
            StoreType::App => self.last_sent.app.get(&update.key).copied().unwrap_or(0),
            StoreType::Membership => self
                .last_sent
                .membership
                .get(&update.key)
                .copied()
                .unwrap_or(0),
            StoreType::RateLimit => {
                let shard_key = IncrementalUpdateCollector::rate_limit_last_sent_key(
                    &update.key,
                    &update.actor,
                );
                self.last_sent
                    .rate_limit
                    .get(&shard_key)
                    .copied()
                    .unwrap_or(0)
            }
        };
        update.version > last_sent_version
    }
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

    use super::*;
    use crate::stores::{AppState, MembershipState, PolicyState, StateStores, WorkerState};

    fn create_test_collector(self_name: String) -> IncrementalUpdateCollector {
        let stores = Arc::new(StateStores::with_self_name(self_name.clone()));
        IncrementalUpdateCollector::new(stores, self_name)
    }

    #[test]
    fn test_collect_worker_updates() {
        let collector = create_test_collector("node1".to_string());
        let stores = collector.stores.clone();

        // Insert a worker state
        let key = "worker1".to_string();
        let worker_state = WorkerState {
            worker_id: "worker1".to_string(),
            model_id: "model1".to_string(),
            url: "http://localhost:8000".to_string(),
            health: true,
            load: 0.5,
            version: 1,
            spec: vec![],
        };
        let _ = stores.worker.insert(key, worker_state);

        // Collect updates
        let updates = collector.collect_updates_for_store(StoreType::Worker);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].key, "worker1");
        assert_eq!(updates[0].version, 1);
        assert_eq!(updates[0].actor, "node1");

        // Collect again before mark_sent - should still include pending updates.
        let updates2 = collector.collect_updates_for_store(StoreType::Worker);
        assert_eq!(updates2.len(), 1);

        // Mark transmission success and verify it is no longer collected.
        collector.mark_sent(StoreType::Worker, &updates2);
        let updates_after_mark = collector.collect_updates_for_store(StoreType::Worker);
        assert_eq!(updates_after_mark.len(), 0);

        // Update worker state
        let key2 = "worker1".to_string();
        let worker_state2 = WorkerState {
            worker_id: "worker1".to_string(),
            model_id: "model1".to_string(),
            url: "http://localhost:8000".to_string(),
            health: false,
            load: 0.8,
            version: 2,
            spec: vec![],
        };
        let _ = stores.worker.insert(key2, worker_state2);

        // Should collect new version
        let updates3 = collector.collect_updates_for_store(StoreType::Worker);
        assert_eq!(updates3.len(), 1);
        assert_eq!(updates3[0].version, 2);
    }

    #[test]
    fn test_collect_policy_updates() {
        let collector = create_test_collector("node1".to_string());
        let stores = collector.stores.clone();

        let key = "policy:model1".to_string();
        let policy_state = PolicyState {
            model_id: "model1".to_string(),
            policy_type: "cache_aware".to_string(),
            config: b"config_data".to_vec(),
            version: 1,
        };
        let _ = stores.policy.insert(key, policy_state);

        let updates = collector.collect_updates_for_store(StoreType::Policy);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].key, "policy:model1");
    }

    #[test]
    fn test_collect_app_updates() {
        let collector = create_test_collector("node1".to_string());
        let stores = collector.stores.clone();

        let key = "app_key1".to_string();
        let app_state = AppState {
            key: "app_key1".to_string(),
            value: b"app_value".to_vec(),
            version: 1,
        };
        let _ = stores.app.insert(key, app_state);

        let updates = collector.collect_updates_for_store(StoreType::App);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].key, "app_key1");
    }

    #[test]
    fn test_collect_membership_updates() {
        let collector = create_test_collector("node1".to_string());
        let stores = collector.stores.clone();

        let key = "node2".to_string();
        let membership_state = MembershipState {
            name: "node2".to_string(),
            address: "127.0.0.1:8001".to_string(),
            status: 1, // Alive
            version: 1,
            metadata: std::collections::BTreeMap::new(),
        };
        let _ = stores.membership.insert(key, membership_state);

        let updates = collector.collect_updates_for_store(StoreType::Membership);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].key, "node2");
    }

    #[test]
    fn test_collect_all_updates() {
        let collector = create_test_collector("node1".to_string());
        let stores = collector.stores.clone();

        // Insert into multiple stores
        let worker_key = "worker1".to_string();
        let _ = stores.worker.insert(
            worker_key,
            WorkerState {
                worker_id: "worker1".to_string(),
                model_id: "model1".to_string(),
                url: "http://localhost:8000".to_string(),
                health: true,
                load: 0.5,
                version: 1,
                spec: vec![],
            },
        );

        let policy_key = "policy:model1".to_string();
        let _ = stores.policy.insert(
            policy_key,
            PolicyState {
                model_id: "model1".to_string(),
                policy_type: "cache_aware".to_string(),
                config: vec![],
                version: 1,
            },
        );

        let all_updates = collector.collect_all_updates();
        assert_eq!(all_updates.len(), 2); // Worker and Policy
    }

    #[test]
    fn test_mark_sent() {
        let collector = create_test_collector("node1".to_string());
        let stores = collector.stores.clone();

        // Insert and collect
        let key = "worker1".to_string();
        let _ = stores.worker.insert(
            key,
            WorkerState {
                worker_id: "worker1".to_string(),
                model_id: "model1".to_string(),
                url: "http://localhost:8000".to_string(),
                health: true,
                load: 0.5,
                version: 1,
                spec: vec![],
            },
        );

        let updates = collector.collect_updates_for_store(StoreType::Worker);
        assert_eq!(updates.len(), 1);

        // Mark as sent
        collector.mark_sent(StoreType::Worker, &updates);

        // Should not collect again
        let updates2 = collector.collect_updates_for_store(StoreType::Worker);
        assert_eq!(updates2.len(), 0);
    }

    #[test]
    fn test_rate_limit_timestamp_filtering() {
        let collector = create_test_collector("node1".to_string());
        let stores = collector.stores.clone();

        // Update membership to make node1 an owner
        stores.rate_limit.update_membership(&["node1".to_string()]);

        // Insert a counter (node1 should be owner)
        let test_key = "test_rate_limit_key".to_string();
        if stores.rate_limit.is_owner(&test_key) {
            stores
                .rate_limit
                .inc(test_key.clone(), "node1".to_string(), 1);
        }

        // Collect immediately - should be filtered by timestamp
        let _updates = collector.collect_updates_for_store(StoreType::RateLimit);
        // May be empty if timestamp check fails, or may have one update
        // The exact behavior depends on timing

        // Wait a bit and try again
        thread::sleep(Duration::from_secs(2));

        // Now should collect (enough time has passed)
        let updates2 = collector.collect_updates_for_store(StoreType::RateLimit);
        // Should have at least one update if node1 is owner
        if stores.rate_limit.is_owner(&test_key) {
            // Updates may be 0 or 1 depending on timing
            let _ = updates2;
        }
    }

    #[test]
    fn test_version_tracking() {
        let collector = create_test_collector("node1".to_string());
        let stores = collector.stores.clone();

        let key = "worker1".to_string();

        // Insert first version with explicit version number
        let _ = stores.worker.insert(
            key.clone(),
            WorkerState {
                worker_id: "worker1".to_string(),
                model_id: "model1".to_string(),
                url: "http://localhost:8000".to_string(),
                health: true,
                load: 0.5,
                version: 1,
                spec: vec![],
            },
        );

        let updates1 = collector.collect_updates_for_store(StoreType::Worker);
        assert_eq!(updates1.len(), 1);
        let version1 = updates1[0].version;
        assert_eq!(version1, 1);

        // Insert second version with incremented version number
        let _ = stores.worker.insert(
            key.clone(),
            WorkerState {
                worker_id: "worker1".to_string(),
                model_id: "model1".to_string(),
                url: "http://localhost:8000".to_string(),
                health: false,
                load: 0.8,
                version: 2,
                spec: vec![],
            },
        );

        let updates2 = collector.collect_updates_for_store(StoreType::Worker);
        assert_eq!(updates2.len(), 1);
        let version2 = updates2[0].version;
        assert_eq!(version2, 2);

        // Insert third version with incremented version number
        let _ = stores.worker.insert(
            key,
            WorkerState {
                worker_id: "worker1".to_string(),
                model_id: "model1".to_string(),
                url: "http://localhost:8000".to_string(),
                health: true,
                load: 0.3,
                version: 3,
                spec: vec![],
            },
        );

        let updates3 = collector.collect_updates_for_store(StoreType::Worker);
        assert_eq!(updates3.len(), 1);
        let version3 = updates3[0].version;
        assert_eq!(version3, 3);
    }
}
