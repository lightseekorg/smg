//! Mesh state synchronization module
//!
//! Handles synchronization of worker and policy states across mesh cluster nodes

use std::{
    fmt::Debug,
    sync::{atomic::Ordering, Arc},
};

use parking_lot::RwLock;
use tracing::{debug, warn};

use super::{
    service::gossip::NodeStatus,
    stores::{
        policy_key, tree_state_key, PolicyState, RateLimitConfig, StateStores, WorkerState,
        GLOBAL_RATE_LIMIT_COUNTER_KEY, GLOBAL_RATE_LIMIT_KEY,
    },
    tree_ops::{
        hash_node_path, hash_token_path, TenantDelta, TenantEvict, TenantInsert, TreeKey,
        TreeOperation, TreeState, TreeStateDelta,
    },
};

pub trait TreeStateSubscriber: Send + Sync + Debug {
    fn apply_remote_tree_state(&self, model_id: &str, tree_state: &TreeState);

    /// Apply lightweight tenant delta — inserts and evictions by hash.
    /// Default: process global evictions only (where `node_path_hash == GLOBAL_EVICTION_HASH`).
    /// Inserts require the actual tree to resolve hashes to nodes,
    /// so they are dropped here; implementations that maintain a
    /// hash→node index (e.g. `CacheAwarePolicy`) should override.
    fn apply_tenant_delta(
        &self,
        model_id: &str,
        _inserts: &[TenantInsert],
        evictions: &[TenantEvict],
    ) {
        // Default: only convert global evictions (hash=GLOBAL_EVICTION_HASH)
        // into Remove ops. Targeted evictions (non-zero hash) are skipped
        // because we can't resolve the hash without a path index.
        let global_evictions: Vec<&TenantEvict> = evictions
            .iter()
            .filter(|e| e.node_path_hash == crate::tree_ops::GLOBAL_EVICTION_HASH)
            .collect();

        if !global_evictions.is_empty() {
            let mut tree_state = TreeState::new(model_id.to_string());
            for evict in global_evictions {
                tree_state.add_operation(TreeOperation::Remove(crate::tree_ops::TreeRemoveOp {
                    tenant: evict.worker_url.clone(),
                }));
            }
            self.apply_remote_tree_state(model_id, &tree_state);
        }
    }

    /// Export the current tree state for a model from the live radix tree.
    /// Used by `checkpoint_tree_states` to build periodic structure snapshots
    /// WITHOUT accumulating full prompt text in memory on every request.
    /// Returns None if the subscriber doesn't have a tree for this model.
    fn export_tree_state(&self, _model_id: &str) -> Option<TreeState> {
        None
    }

    /// Export a compact tree snapshot for a model from the live radix tree.
    /// Returns a [`kv_index::snapshot::TreeSnapshot`] that encodes the tree
    /// structure with shared prefixes — much smaller than the flat
    /// `TreeState` returned by [`export_tree_state`].
    ///
    /// Used by `checkpoint_tree_states` to populate `tree_configs` for
    /// Layer 2 periodic snapshots.
    fn export_tree_snapshot(&self, _model_id: &str) -> Option<kv_index::snapshot::TreeSnapshot> {
        None
    }
}

pub trait WorkerStateSubscriber: Send + Sync + Debug {
    fn on_remote_worker_state(&self, state: &WorkerState);
}

/// Mesh sync manager for coordinating state synchronization
#[derive(Clone, Debug)]
pub struct MeshSyncManager {
    pub(crate) stores: Arc<StateStores>,
    self_name: String,
    tree_state_subscribers: Arc<RwLock<Vec<Arc<dyn TreeStateSubscriber>>>>,
    worker_state_subscribers: Arc<RwLock<Vec<Arc<dyn WorkerStateSubscriber>>>>,
}

impl MeshSyncManager {
    pub fn new(stores: Arc<StateStores>, self_name: String) -> Self {
        Self {
            stores,
            self_name,
            tree_state_subscribers: Arc::new(RwLock::new(Vec::new())),
            worker_state_subscribers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn register_tree_state_subscriber(&self, subscriber: Arc<dyn TreeStateSubscriber>) {
        self.tree_state_subscribers.write().push(subscriber);
    }

    fn notify_tree_state_subscribers(&self, model_id: &str, tree_state: &TreeState) {
        let subscribers = self.tree_state_subscribers.read().clone();
        for subscriber in subscribers {
            subscriber.apply_remote_tree_state(model_id, tree_state);
        }
    }

    pub fn register_worker_state_subscriber(&self, subscriber: Arc<dyn WorkerStateSubscriber>) {
        self.worker_state_subscribers.write().push(subscriber);
    }

    fn notify_worker_state_subscribers(&self, state: &WorkerState) {
        let subscribers = self.worker_state_subscribers.read().clone();
        for subscriber in subscribers {
            subscriber.on_remote_worker_state(state);
        }
    }

    /// Get the node name (actor) for this sync manager
    pub fn self_name(&self) -> &str {
        &self.self_name
    }

    /// Sync worker state to mesh stores
    pub fn sync_worker_state(
        &self,
        worker_id: String,
        model_id: String,
        url: String,
        health: bool,
        load: f64,
        spec: Vec<u8>,
    ) {
        let key = worker_id.clone();

        let updated_state = self.stores.worker.update(key, |current| {
            let new_version = current
                .map(|state| state.version)
                .unwrap_or(0)
                .saturating_add(1);

            WorkerState {
                worker_id: worker_id.clone(),
                model_id,
                url,
                health,
                load,
                version: new_version,
                spec,
            }
        });

        match updated_state {
            Ok(Some(state)) => {
                debug!(
                    "Synced worker state to mesh {} (version: {})",
                    state.worker_id, state.version
                );
            }
            Ok(None) => {}
            Err(err) => {
                debug!(error = %err, worker_id = %worker_id, "Failed to sync worker state");
            }
        }
    }

    /// Remove worker state from mesh stores
    pub fn remove_worker_state(&self, worker_id: &str) {
        self.stores.worker.remove(worker_id);
        debug!("Removed worker state from mesh {}", worker_id);
    }

    /// Sync policy state to mesh stores
    pub fn sync_policy_state(&self, model_id: String, policy_type: String, config: Vec<u8>) {
        let key = policy_key(&model_id);
        let model_id_for_update = model_id.clone();

        let updated_state = self.stores.policy.update(key, move |current| {
            let new_version = current
                .map(|state| state.version)
                .unwrap_or(0)
                .saturating_add(1);

            PolicyState {
                model_id: model_id_for_update,
                policy_type,
                config,
                version: new_version,
            }
        });

        match updated_state {
            Ok(Some(state)) => {
                debug!(
                    "Synced policy state to mesh model={} (version: {})",
                    state.model_id, state.version
                );
            }
            Ok(None) => {}
            Err(err) => {
                debug!(error = %err, model_id = %model_id, "Failed to sync policy state");
            }
        }
    }

    /// Remove policy state from mesh stores
    pub fn remove_policy_state(&self, model_id: &str) {
        let key = policy_key(model_id);
        self.stores.policy.remove(&key);
        debug!("Removed policy state from mesh model={}", model_id);
    }

    /// Get worker state from mesh stores
    pub fn get_worker_state(&self, worker_id: &str) -> Option<WorkerState> {
        self.stores.worker.get(worker_id)
    }

    /// Get all worker states from mesh stores
    pub fn get_all_worker_states(&self) -> Vec<WorkerState> {
        self.stores.worker.all().into_values().collect()
    }

    /// Get policy state from mesh stores
    pub fn get_policy_state(&self, model_id: &str) -> Option<PolicyState> {
        let key = policy_key(model_id);
        self.stores.policy.get(&key)
    }

    /// Get all policy states from mesh stores
    pub fn get_all_policy_states(&self) -> Vec<PolicyState> {
        self.stores.policy.all().into_values().collect()
    }

    /// Apply worker state update from remote node
    /// The actor should be extracted from the state update context (e.g., from StateUpdate message)
    pub fn apply_remote_worker_state(&self, state: WorkerState, actor: Option<String>) {
        let key = state.worker_id.clone();
        let actor = actor.unwrap_or_else(|| "remote".to_string());
        let mut current_version = 0;

        let update_result = self.stores.worker.update_if(key, |current| {
            current_version = current
                .as_ref()
                .map(|existing| existing.version)
                .unwrap_or(0);
            if state.version > current_version {
                Some(state.clone())
            } else {
                None
            }
        });

        match update_result {
            Ok((_, true)) => {
                debug!(
                    "Applied remote worker state update: {} (version: {} -> {})",
                    state.worker_id, current_version, state.version
                );
                self.notify_worker_state_subscribers(&state);
            }
            Ok((_, false)) => {
                debug!(
                    "Skipped remote worker state update: {} (version {} <= current {})",
                    state.worker_id, state.version, current_version
                );
            }
            Err(err) => {
                debug!(error = %err, worker_id = %state.worker_id, actor = %actor, "Failed to apply remote worker state update");
            }
        }
    }

    /// Apply policy state update from remote node
    /// The actor should be extracted from the state update context (e.g., from StateUpdate message)
    pub fn apply_remote_policy_state(&self, state: PolicyState, actor: Option<String>) {
        let key = policy_key(&state.model_id);
        let actor = actor.unwrap_or_else(|| "remote".to_string());
        let mut current_version = 0;

        let update_result = self.stores.policy.update_if(key, |current| {
            current_version = current
                .as_ref()
                .map(|existing| existing.version)
                .unwrap_or(0);
            if state.version > current_version {
                Some(state.clone())
            } else {
                None
            }
        });

        match update_result {
            Ok((_, true)) => {
                debug!(
                    "Applied remote policy state update: {} (version: {} -> {})",
                    state.model_id, current_version, state.version
                );
            }
            Ok((_, false)) => {
                debug!(
                    "Skipped remote policy state update: {} (version {} <= current {})",
                    state.model_id, state.version, current_version
                );
            }
            Err(err) => {
                debug!(error = %err, model_id = %state.model_id, actor = %actor, "Failed to apply remote policy state update");
            }
        }
    }

    /// Update rate-limit hash ring with current membership
    pub fn update_rate_limit_membership(&self) {
        // Get all alive nodes from membership store
        let all_members = self.stores.membership.all();
        let alive_nodes: Vec<String> = all_members
            .values()
            .filter(|m| m.status == NodeStatus::Alive as i32)
            .map(|m| m.name.clone())
            .collect();

        self.stores.rate_limit.update_membership(&alive_nodes);
        debug!(
            "Updated rate-limit hash ring with {} alive nodes",
            alive_nodes.len()
        );
    }

    /// Handle node failure and transfer rate-limit ownership
    pub fn handle_node_failure(&self, failed_nodes: &[String]) {
        if failed_nodes.is_empty() {
            return;
        }

        debug!("Handling node failure for rate-limit: {:?}", failed_nodes);

        // Check which keys need ownership transfer
        let affected_keys = self
            .stores
            .rate_limit
            .check_ownership_transfer(failed_nodes);

        if !affected_keys.is_empty() {
            debug!(
                "Ownership transfer needed for {} rate-limit keys",
                affected_keys.len()
            );

            // Update membership to reflect node failures
            self.update_rate_limit_membership();

            // For each affected key, we may need to initialize counters if we're now an owner
            for key in &affected_keys {
                if self.stores.rate_limit.is_owner(key) {
                    debug!("This node is now owner of rate-limit key: {}", key);
                    // Counter will be created on first inc() call
                }
            }
        }
    }

    /// Sync rate-limit counter increment (only if this node is an owner)
    pub fn sync_rate_limit_inc(&self, key: String, delta: i64) {
        if !self.stores.rate_limit.is_owner(&key) {
            // Not an owner, skip
            return;
        }

        self.stores
            .rate_limit
            .inc(key.clone(), self.self_name.clone(), delta);
        debug!("Synced rate-limit increment: key={}, delta={}", key, delta);
    }

    /// Apply remote rate-limit counter update (merge CRDT)
    pub fn apply_remote_rate_limit_counter(&self, log: &super::crdt_kv::OperationLog) {
        // Merge operation log regardless of ownership (for CRDT consistency)
        self.stores.rate_limit.merge(log);
        debug!("Applied remote rate-limit counter update");
    }

    /// Apply remote rate-limit counter snapshot encoded as raw i64.
    pub fn apply_remote_rate_limit_counter_value(&self, key: String, counter_value: i64) {
        self.apply_remote_rate_limit_counter_value_with_actor_and_timestamp(
            key,
            "remote".to_string(),
            counter_value,
            0,
        );
    }

    pub fn apply_remote_rate_limit_counter_value_with_actor(
        &self,
        key: String,
        actor: String,
        counter_value: i64,
    ) {
        self.apply_remote_rate_limit_counter_value_with_actor_and_timestamp(
            key,
            actor,
            counter_value,
            0,
        );
    }

    pub fn apply_remote_rate_limit_counter_value_with_actor_and_timestamp(
        &self,
        key: String,
        actor: String,
        counter_value: i64,
        timestamp: u64,
    ) {
        if let Some((shard_key, payload)) =
            super::stores::RateLimitStore::snapshot_payload_for_counter_value(
                key,
                actor.clone(),
                counter_value,
            )
        {
            self.stores
                .rate_limit
                .apply_counter_snapshot_payload(shard_key, &actor, timestamp, &payload);
            debug!("Applied remote rate-limit counter snapshot payload");
        }
    }

    /// Get rate-limit value (aggregate from all owners)
    pub fn get_rate_limit_value(&self, key: &str) -> Option<i64> {
        self.stores.rate_limit.value(key)
    }

    /// Get global rate limit configuration from AppStore
    pub fn get_global_rate_limit_config(&self) -> Option<RateLimitConfig> {
        self.stores
            .app
            .get(GLOBAL_RATE_LIMIT_KEY)
            .and_then(|app_state| bincode::deserialize::<RateLimitConfig>(&app_state.value).ok())
    }

    /// Check if global rate limit is exceeded
    /// Returns (is_exceeded, current_count, limit)
    pub fn check_global_rate_limit(&self) -> (bool, i64, u64) {
        let config = self.get_global_rate_limit_config().unwrap_or_default();

        if config.limit_per_second == 0 {
            // Rate limit disabled
            return (false, 0, 0);
        }

        // Increment counter if this node is an owner
        self.sync_rate_limit_inc(GLOBAL_RATE_LIMIT_COUNTER_KEY.to_string(), 1);

        // Get aggregated counter value from all owners
        let current_count = self
            .get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY)
            .unwrap_or(0);

        let is_exceeded = current_count > config.limit_per_second as i64;
        (is_exceeded, current_count, config.limit_per_second)
    }

    /// Reset global rate limit counter (called periodically for time window reset)
    pub fn reset_global_rate_limit_counter(&self) {
        // Reset by decrementing the current value
        // Since we use PNCounter, we can't directly reset, but we can track the window
        // For simplicity, we'll use a time-based approach where counters are reset periodically
        // The actual reset logic will be handled by the window manager
        let current_count = self
            .get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY)
            .unwrap_or(0);

        if current_count > 0 {
            // Decrement by current count to effectively reset
            // Note: This is a workaround since PNCounter doesn't support direct reset
            // In production, you might want to use a different approach like timestamped counters
            self.sync_rate_limit_inc(GLOBAL_RATE_LIMIT_COUNTER_KEY.to_string(), -current_count);
        }
    }

    /// Sync tree operation to mesh stores.
    ///
    /// This is called on every request (hot path). The operation is appended to
    /// the pending buffer for delta sync — the collector serializes and sends it
    /// to peers. We do NOT read/deserialize/serialize the full TreeState here,
    /// because that is O(tree_size) per request and caused multi-GB memory usage
    /// at 200+ rps.
    ///
    /// The policy store version is bumped so the generation-based collector
    /// detects the change, but the `config` blob is NOT updated on every call.
    /// It is rebuilt lazily by the collector when a full-state fallback is needed.
    /// Lightweight sync: accepts a pre-computed hash + tenant, avoiding
    /// the 80k+ String allocation from TreeKey::Text on every request.
    pub fn sync_tree_insert_hash(&self, model_id: &str, path_hash: u64, tenant: &str) {
        let key = tree_state_key(model_id);

        self.stores
            .tenant_delta_inserts
            .entry(model_id.to_string())
            .or_default()
            .push(TenantInsert {
                node_path_hash: path_hash,
                worker_url: tenant.to_string(),
                epoch: self.stores.tree_version(&key),
            });

        self.stores.bump_tree_version(&key);
    }

    #[expect(
        clippy::unnecessary_wraps,
        reason = "Public API — callers handle Result; changing return type is a cross-crate break"
    )]
    pub fn sync_tree_operation(
        &self,
        model_id: String,
        operation: TreeOperation,
    ) -> Result<(), String> {
        let key = tree_state_key(&model_id);

        // Buffer a lightweight tenant delta — 24 bytes per insert (hash + epoch)
        // instead of 80k+ bytes (full prompt text).
        match &operation {
            TreeOperation::Insert(insert) => {
                let path_hash = match &insert.key {
                    TreeKey::Text(text) => hash_node_path(text),
                    TreeKey::Tokens(tokens) => hash_token_path(tokens),
                };
                self.stores
                    .tenant_delta_inserts
                    .entry(model_id.clone())
                    .or_default()
                    .push(TenantInsert {
                        node_path_hash: path_hash,
                        worker_url: insert.tenant.clone(),
                        epoch: self.stores.tree_version(&key),
                    });
            }
            TreeOperation::Remove(remove) => {
                // TODO: capture the specific prefix hash being evicted.
                // For now, 0 means "evict from all nodes" (global eviction).
                // This is overly aggressive but correct — the next structure
                // snapshot will restore any wrongly evicted entries.
                self.stores
                    .tenant_delta_evictions
                    .entry(model_id.clone())
                    .or_default()
                    .push(TenantEvict {
                        node_path_hash: crate::tree_ops::GLOBAL_EVICTION_HASH,
                        worker_url: remove.tenant.clone(),
                    });
            }
        }

        // NOTE: We intentionally do NOT push to tree_ops_pending here.
        // That would store the full TreeOperation (including 20KB+ prompt text)
        // on every request — 40MB between checkpoints at 200 rps.
        // Instead, checkpoint_tree_states exports the live tree via subscribers.

        // Bump the lightweight atomic version counter (O(1), no serialization).
        self.stores.bump_tree_version(&key);

        Ok(())
    }

    /// Load the materialized TreeState from `tree_configs`.
    /// Returns None if no checkpoint exists for this key.
    ///
    /// Handles two storage formats:
    /// - `TreeState` bytes (from remote full-state updates)
    /// - `TreeSnapshot` bytes (from local `checkpoint_tree_states`)
    fn materialize_tree_state(&self, key: &str, model_id: &str) -> Option<TreeState> {
        let config_bytes = self.stores.tree_configs.get(key)?;
        let bytes = config_bytes.value();
        if bytes.is_empty() {
            return Some(TreeState::new(model_id.to_string()));
        }
        // Try TreeState first (remote full-state updates store this format).
        if let Ok(ts) = TreeState::from_bytes(bytes) {
            return Some(ts);
        }
        // Fall back to TreeSnapshot (local checkpoint format).
        if let Ok(snap) = kv_index::snapshot::TreeSnapshot::from_bytes(bytes) {
            let version = self.stores.tree_version(key);
            return Some(TreeState::from_snapshot(
                model_id.to_string(),
                &snap,
                version,
            ));
        }
        None
    }

    /// Get tree state for a model from mesh stores.
    /// Reads from `tree_configs` (populated by periodic checkpoint from live tree).
    pub fn get_tree_state(&self, model_id: &str) -> Option<TreeState> {
        let key = tree_state_key(model_id);
        self.materialize_tree_state(&key, model_id)
    }

    pub fn get_all_tree_states(&self) -> Vec<TreeState> {
        let mut results = Vec::new();

        for entry in &self.stores.tree_configs {
            let key = entry.key().clone();
            let model_id = key.strip_prefix("tree:").unwrap_or(&key).to_string();
            if let Some(ts) = self.materialize_tree_state(&key, &model_id) {
                results.push(ts);
            }
        }

        results
    }

    /// Apply remote tree operation to local stores.
    /// This is called when receiving full tree state updates from other nodes.
    ///
    /// Writes to `tree_configs` (plain DashMap) instead of the CRDT policy
    /// store to avoid operation log memory accumulation.
    ///
    /// Uses `DashMap::entry()` for atomic read-modify-write on `tree_configs`
    /// to avoid the TOCTOU gap between `get()` and `insert()`.
    pub fn apply_remote_tree_operation(
        &self,
        model_id: String,
        tree_state: TreeState,
        actor: Option<String>,
    ) {
        use dashmap::mapref::entry::Entry;

        let key = tree_state_key(&model_id);
        let _actor = actor.unwrap_or_else(|| "remote".to_string());

        let serialized = match tree_state.to_bytes() {
            Ok(bytes) => bytes,
            Err(err) => {
                debug!(error = %err, model_id = %model_id, "Failed to serialize remote tree state");
                return;
            }
        };

        // Atomic read-modify-write via entry() — version check and insert
        // happen under the same shard lock, closing the TOCTOU gap.
        let applied = match self.stores.tree_configs.entry(key.clone()) {
            Entry::Occupied(mut entry) => {
                // tree_configs may hold TreeState bytes (from remote) or
                // TreeSnapshot bytes (from local checkpoint). Fall back to
                // the authoritative atomic version counter if deserialization fails.
                let current_version = TreeState::from_bytes(entry.get())
                    .ok()
                    .map(|ts| ts.version)
                    .unwrap_or_else(|| self.stores.tree_version(&key));
                if tree_state.version > current_version {
                    entry.insert(serialized);
                    debug!(
                        "Applied remote tree state update: model={} (version: {} -> {})",
                        model_id, current_version, tree_state.version
                    );
                    true
                } else {
                    debug!(
                        "Skipped remote tree state update: model={} (version {} <= current {})",
                        model_id, tree_state.version, current_version
                    );
                    false
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(serialized);
                debug!(
                    "Applied remote tree state update (new): model={} (version: {})",
                    model_id, tree_state.version
                );
                true
            }
        };

        // Subscriber notification and version advancement happen after
        // dropping the entry (shard lock released).
        if applied {
            self.stores.advance_tree_version(&key, tree_state.version);
            self.stores.tree_generation.fetch_add(1, Ordering::Release);
            self.notify_tree_state_subscribers(&model_id, &tree_state);
        }
    }

    /// Apply a delta (incremental operations) from a remote node.
    /// Merges the delta operations into the existing local tree state,
    /// avoiding the cost of replacing the entire tree state on every sync.
    ///
    /// Uses `DashMap::entry()` for atomic read-modify-write on `tree_configs`
    /// to avoid the TOCTOU gap between `get()` and `insert()`.
    pub fn apply_remote_tree_delta(&self, delta: TreeStateDelta, actor: Option<String>) {
        use dashmap::mapref::entry::Entry;

        let key = tree_state_key(&delta.model_id);
        let _actor = actor.unwrap_or_else(|| "remote".to_string());
        let model_id = delta.model_id.clone();
        let ops_count = delta.operations.len();

        // Perform the atomic read-modify-write inside the entry block.
        // Tree construction and serialization happen while holding the
        // shard write lock; subscriber notification happens after.
        let result: Option<(TreeState, u64)> = match self.stores.tree_configs.entry(key.clone()) {
            Entry::Occupied(mut entry) => {
                let bytes = entry.get();
                let current_version = if bytes.is_empty() {
                    0
                } else {
                    match TreeState::from_bytes(bytes) {
                        Ok(ts) => ts.version,
                        Err(_) => 0,
                    }
                };

                // Version checks
                if delta.base_version > current_version || current_version >= delta.new_version {
                    debug!(
                        "Skipped remote tree delta: model={} (base_version={}, new_version={}, current={})",
                        model_id, delta.base_version, delta.new_version, current_version
                    );
                    return;
                }

                // Build base tree from config only.
                let mut tree_state = if bytes.is_empty() {
                    if current_version > 0 {
                        debug!(
                            "Skipped remote tree delta: model={} (base_version={}, new_version={}, current={})",
                            model_id, delta.base_version, delta.new_version, current_version
                        );
                        return;
                    }
                    TreeState::new(delta.model_id.clone())
                } else {
                    match TreeState::from_bytes(bytes) {
                        Ok(state) => state,
                        Err(err) => {
                            warn!(
                                model_id = %delta.model_id,
                                error = %err,
                                "Corrupted tree state — rejecting delta to avoid data loss"
                            );
                            return;
                        }
                    }
                };

                let old_version = current_version;
                for op in &delta.operations {
                    tree_state.add_operation(op.clone());
                }
                let new_version = tree_state.version;

                match tree_state.to_bytes() {
                    Ok(serialized) => {
                        entry.insert(serialized);
                        debug!(
                            "Applied remote tree delta: model={} (version: {} -> +{} ops)",
                            model_id, old_version, ops_count
                        );
                        Some((tree_state, new_version))
                    }
                    Err(err) => {
                        debug!(error = %err, model_id = %model_id, "Failed to serialize tree state after delta apply");
                        None
                    }
                }
            }
            Entry::Vacant(entry) => {
                // No existing config — new tree from delta.
                if delta.base_version > 0 {
                    debug!(
                        "Skipped remote tree delta: model={} (base_version={}, new_version={}, no local state)",
                        model_id, delta.base_version, delta.new_version
                    );
                    return;
                }
                let mut tree_state = TreeState::new(delta.model_id.clone());
                for op in &delta.operations {
                    tree_state.add_operation(op.clone());
                }
                let new_version = tree_state.version;

                match tree_state.to_bytes() {
                    Ok(serialized) => {
                        entry.insert(serialized);
                        debug!(
                            "Applied remote tree delta (new tree): model={} (+{} ops)",
                            model_id, ops_count
                        );
                        Some((tree_state, new_version))
                    }
                    Err(err) => {
                        debug!(error = %err, model_id = %model_id, "Failed to serialize new tree state from delta");
                        None
                    }
                }
            }
        };

        // Notification happens outside the entry block (shard lock released).
        if let Some((tree_state, new_version)) = result {
            self.stores.advance_tree_version(&key, new_version);
            self.stores.tree_generation.fetch_add(1, Ordering::Release);
            self.notify_tree_state_subscribers(&model_id, &tree_state);
        }
    }

    /// Apply a lightweight tenant delta from a remote node.
    /// Updates the local radix tree directly via subscribers without
    /// going through the CRDT or the full TreeState machinery.
    pub fn apply_remote_tenant_delta(&self, delta: TenantDelta, _actor: Option<String>) {
        let key = tree_state_key(&delta.model_id);

        if delta.inserts.is_empty() && delta.evictions.is_empty() {
            return;
        }

        // No version check — both routers independently bump tree_version
        // on local inserts, so the remote delta's version can be lower than
        // the local version even though it contains novel inserts. Tenant
        // inserts are idempotent (insert_text is a no-op if the tenant
        // already exists at the node), so applying "stale" deltas is safe.

        debug!(
            model_id = %delta.model_id,
            inserts = delta.inserts.len(),
            evictions = delta.evictions.len(),
            version = delta.version,
            "Applying remote tenant delta"
        );

        // Clone subscriber list before calling back — same pattern as
        // notify_tree_state_subscribers — so we don't hold the read guard
        // during potentially expensive subscriber callbacks.
        let subscribers = self.tree_state_subscribers.read().clone();
        for subscriber in &subscribers {
            subscriber.apply_tenant_delta(&delta.model_id, &delta.inserts, &delta.evictions);
        }

        // Advance version and bump generation so collector re-scans
        self.stores.advance_tree_version(&key, delta.version);
        self.stores.tree_generation.fetch_add(1, Ordering::Release);
    }

    /// Checkpoint tree state by exporting compact snapshots from the live
    /// radix tree via subscribers.
    ///
    /// Called periodically (~every 10s) to keep `tree_configs` fresh for
    /// the periodic structure snapshot (every 30 gossip rounds).
    ///
    /// Uses [`TreeStateSubscriber::export_tree_snapshot`] to obtain a
    /// compact [`kv_index::snapshot::TreeSnapshot`] that preserves shared
    /// prefixes.  This is much smaller than the flat `TreeState` produced
    /// by `export_tree_state` (~2-4 MB vs ~40 MB for 2048 entries sharing
    /// 80% prefixes) and avoids accumulating full prompt text in memory.
    #[expect(
        clippy::unused_self,
        reason = "Public API called by controller — removing &self is a breaking change"
    )]
    pub fn checkpoint_tree_states(&self) {
        // FIXME: Layer 2 (full tree snapshots) is disabled because the
        // snapshot can be 170+ MB for large trees with long prompts, and
        // allocating it every 10s causes allocator fragmentation. Tree data
        // currently syncs via Layer 1 only (tenant deltas, ~50 bytes each).
        // TODO: implement chunked snapshots or incremental tree diffs so
        // Layer 2 works for large trees without excessive memory allocation.
    }
}

/// Optional mesh sync manager (can be None if mesh is not enabled)
pub type OptionalMeshSyncManager = Option<Arc<MeshSyncManager>>;

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
    };

    use super::*;
    use crate::{
        collector::CentralCollector,
        service::gossip::StateUpdate,
        stores::{
            AppState, MembershipState, RateLimitConfig, StateStores, StoreType,
            GLOBAL_RATE_LIMIT_COUNTER_KEY, GLOBAL_RATE_LIMIT_KEY,
        },
    };

    /// Test-only helper: collect Policy updates via CentralCollector.
    /// Skips PeerWatermark since these tests don't exercise watermark filtering.
    fn collect_policy_updates(stores: Arc<StateStores>, self_name: &str) -> Vec<StateUpdate> {
        let central = CentralCollector::new(stores, self_name.to_string());
        let batch = central.collect();
        batch
            .updates
            .into_iter()
            .find(|(t, _)| *t == StoreType::Policy)
            .map(|(_, v)| v)
            .unwrap_or_default()
    }

    fn create_test_sync_manager() -> MeshSyncManager {
        let stores = Arc::new(StateStores::new());
        MeshSyncManager::new(stores, "test_node".to_string())
    }

    fn create_test_manager(self_name: String) -> MeshSyncManager {
        let stores = Arc::new(StateStores::with_self_name(self_name.clone()));
        MeshSyncManager::new(stores, self_name)
    }

    #[derive(Debug)]
    struct LockCheckingSubscriber {
        manager: Arc<MeshSyncManager>,
        can_acquire_write_lock: Arc<AtomicBool>,
    }

    impl TreeStateSubscriber for LockCheckingSubscriber {
        fn apply_remote_tree_state(&self, _model_id: &str, _tree_state: &TreeState) {
            let can_acquire_write_lock = self.manager.tree_state_subscribers.try_write().is_some();
            self.can_acquire_write_lock
                .store(can_acquire_write_lock, Ordering::SeqCst);
        }
    }

    #[test]
    fn test_sync_manager_new() {
        let manager = create_test_sync_manager();
        // Should create without panicking
        assert_eq!(manager.get_all_worker_states().len(), 0);
        assert_eq!(manager.get_all_policy_states().len(), 0);
    }

    #[test]
    fn test_sync_worker_state() {
        let manager = create_test_manager("node1".to_string());

        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://localhost:8000".to_string(),
            true,
            0.5,
            vec![],
        );

        let state = manager.get_worker_state("worker1").unwrap();
        assert_eq!(state.worker_id, "worker1");
        assert_eq!(state.model_id, "model1");
        assert_eq!(state.url, "http://localhost:8000");
        assert!(state.health);
        assert_eq!(state.load, 0.5);
        assert_eq!(state.version, 1);
    }

    #[test]
    fn test_sync_multiple_worker_states() {
        let manager = create_test_sync_manager();

        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://localhost:8000".to_string(),
            true,
            0.5,
            vec![],
        );

        manager.sync_worker_state(
            "worker2".to_string(),
            "model1".to_string(),
            "http://localhost:8001".to_string(),
            false,
            0.8,
            vec![],
        );

        manager.sync_worker_state(
            "worker3".to_string(),
            "model2".to_string(),
            "http://localhost:8002".to_string(),
            true,
            0.3,
            vec![],
        );

        let all_states = manager.get_all_worker_states();
        assert_eq!(all_states.len(), 3);

        let worker1 = manager.get_worker_state("worker1").unwrap();
        assert_eq!(worker1.worker_id, "worker1");
        assert!(worker1.health);

        let worker2 = manager.get_worker_state("worker2").unwrap();
        assert_eq!(worker2.worker_id, "worker2");
        assert!(!worker2.health);

        let worker3 = manager.get_worker_state("worker3").unwrap();
        assert_eq!(worker3.worker_id, "worker3");
        assert_eq!(worker3.model_id, "model2");
    }

    #[test]
    fn test_sync_worker_state_version_increment() {
        let manager = create_test_manager("node1".to_string());

        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://localhost:8000".to_string(),
            true,
            0.5,
            vec![],
        );

        let state1 = manager.get_worker_state("worker1").unwrap();
        assert_eq!(state1.version, 1);

        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://localhost:8000".to_string(),
            false,
            0.8,
            vec![],
        );

        let state2 = manager.get_worker_state("worker1").unwrap();
        assert_eq!(state2.version, 2);
        assert!(!state2.health);
        assert_eq!(state2.load, 0.8);
    }

    #[test]
    fn test_remove_worker_state() {
        let manager = create_test_manager("node1".to_string());

        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://localhost:8000".to_string(),
            true,
            0.5,
            vec![],
        );

        assert!(manager.get_worker_state("worker1").is_some());

        manager.remove_worker_state("worker1");

        assert!(manager.get_worker_state("worker1").is_none());
        assert_eq!(manager.get_all_worker_states().len(), 0);
    }

    #[test]
    fn test_remove_nonexistent_worker_state() {
        let manager = create_test_sync_manager();

        // Should not panic
        manager.remove_worker_state("nonexistent");
        assert!(manager.get_worker_state("nonexistent").is_none());
    }

    #[test]
    fn test_sync_policy_state() {
        let manager = create_test_manager("node1".to_string());

        manager.sync_policy_state(
            "model1".to_string(),
            "cache_aware".to_string(),
            b"config_data".to_vec(),
        );

        let state = manager.get_policy_state("model1").unwrap();
        assert_eq!(state.model_id, "model1");
        assert_eq!(state.policy_type, "cache_aware");
        assert_eq!(state.config, b"config_data");
        assert_eq!(state.version, 1);
    }

    #[test]
    fn test_sync_multiple_policy_states() {
        let manager = create_test_sync_manager();

        manager.sync_policy_state(
            "model1".to_string(),
            "round_robin".to_string(),
            b"config1".to_vec(),
        );

        manager.sync_policy_state(
            "model2".to_string(),
            "random".to_string(),
            b"config2".to_vec(),
        );

        manager.sync_policy_state(
            "model3".to_string(),
            "consistent_hash".to_string(),
            b"config3".to_vec(),
        );

        let all_states = manager.get_all_policy_states();
        assert_eq!(all_states.len(), 3);

        let policy1 = manager.get_policy_state("model1").unwrap();
        assert_eq!(policy1.model_id, "model1");
        assert_eq!(policy1.policy_type, "round_robin");

        let policy2 = manager.get_policy_state("model2").unwrap();
        assert_eq!(policy2.model_id, "model2");
        assert_eq!(policy2.policy_type, "random");
    }

    #[test]
    fn test_remove_policy_state() {
        let manager = create_test_sync_manager();

        manager.sync_policy_state(
            "model1".to_string(),
            "round_robin".to_string(),
            b"config".to_vec(),
        );

        assert!(manager.get_policy_state("model1").is_some());

        manager.remove_policy_state("model1");

        assert!(manager.get_policy_state("model1").is_none());
        assert_eq!(manager.get_all_policy_states().len(), 0);
    }

    #[test]
    fn test_remove_nonexistent_policy_state() {
        let manager = create_test_sync_manager();

        // Should not panic
        manager.remove_policy_state("nonexistent");
        assert!(manager.get_policy_state("nonexistent").is_none());
    }

    #[test]
    fn test_apply_remote_worker_state() {
        let manager = create_test_manager("node1".to_string());

        // Apply remote state with higher version
        let remote_state = WorkerState {
            worker_id: "worker1".to_string(),
            model_id: "model1".to_string(),
            url: "http://localhost:8000".to_string(),
            health: true,
            load: 0.5,
            version: 5,
            spec: vec![],
        };

        manager.apply_remote_worker_state(remote_state.clone(), Some("node2".to_string()));

        let state = manager.get_worker_state("worker1").unwrap();
        assert_eq!(state.version, 5);
    }

    #[test]
    fn test_apply_remote_worker_state_basic() {
        let manager = create_test_sync_manager();

        let remote_state = WorkerState {
            worker_id: "remote_worker1".to_string(),
            model_id: "model1".to_string(),
            url: "http://localhost:8000".to_string(),
            health: true,
            load: 0.6,
            version: 1,
            spec: vec![],
        };

        manager.apply_remote_worker_state(remote_state.clone(), None);

        let state = manager.get_worker_state("remote_worker1");
        assert!(state.is_some());
        let state = state.unwrap();
        assert_eq!(state.worker_id, "remote_worker1");
        assert_eq!(state.model_id, "model1");
        assert_eq!(state.url, "http://localhost:8000");
        assert!(state.health);
        assert_eq!(state.load, 0.6);
    }

    #[test]
    fn test_apply_remote_worker_state_version_check() {
        let manager = create_test_manager("node1".to_string());

        // First insert local state
        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://localhost:8000".to_string(),
            true,
            0.5,
            vec![],
        );

        // Try to apply older version - should be skipped
        let old_state = WorkerState {
            worker_id: "worker1".to_string(),
            model_id: "model1".to_string(),
            url: "http://localhost:8000".to_string(),
            health: false,
            load: 0.8,
            version: 0, // Older version
            spec: vec![],
        };

        manager.apply_remote_worker_state(old_state, Some("node2".to_string()));

        // Should still have version 1
        let state = manager.get_worker_state("worker1").unwrap();
        assert_eq!(state.version, 1);
        assert!(state.health); // Not updated
    }

    #[test]
    fn test_apply_remote_policy_state() {
        let manager = create_test_sync_manager();

        let remote_state = PolicyState {
            model_id: "model1".to_string(),
            policy_type: "remote_policy".to_string(),
            config: b"remote_config".to_vec(),
            version: 1,
        };

        manager.apply_remote_policy_state(remote_state.clone(), None);

        let state = manager.get_policy_state("model1");
        assert!(state.is_some());
        let state = state.unwrap();
        assert_eq!(state.model_id, "model1");
        assert_eq!(state.policy_type, "remote_policy");
        assert_eq!(state.config, b"remote_config");
    }

    #[test]
    fn test_mixed_local_and_remote_states() {
        let manager = create_test_sync_manager();

        // Add local worker
        manager.sync_worker_state(
            "local_worker".to_string(),
            "model1".to_string(),
            "http://localhost:8000".to_string(),
            true,
            0.5,
            vec![],
        );

        // Add remote worker
        let remote_state = WorkerState {
            worker_id: "remote_worker".to_string(),
            model_id: "model1".to_string(),
            url: "http://localhost:8001".to_string(),
            health: true,
            load: 0.7,
            version: 1,
            spec: vec![],
        };
        manager.apply_remote_worker_state(remote_state, None);

        let all_states = manager.get_all_worker_states();
        assert_eq!(all_states.len(), 2);

        assert!(manager.get_worker_state("local_worker").is_some());
        assert!(manager.get_worker_state("remote_worker").is_some());
    }

    #[test]
    fn test_update_worker_state() {
        let manager = create_test_sync_manager();

        // Initial state
        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://localhost:8000".to_string(),
            true,
            0.5,
            vec![],
        );

        // Update state
        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://localhost:8000".to_string(),
            false,
            0.9,
            vec![],
        );

        let state = manager.get_worker_state("worker1").unwrap();
        assert!(!state.health);
        assert_eq!(state.load, 0.9);
        assert_eq!(manager.get_all_worker_states().len(), 1);
    }

    #[test]
    fn test_update_policy_state() {
        let manager = create_test_sync_manager();

        // Initial state
        manager.sync_policy_state(
            "model1".to_string(),
            "round_robin".to_string(),
            b"config1".to_vec(),
        );

        // Update state
        manager.sync_policy_state(
            "model1".to_string(),
            "random".to_string(),
            b"config2".to_vec(),
        );

        let state = manager.get_policy_state("model1").unwrap();
        assert_eq!(state.policy_type, "random");
        assert_eq!(state.config, b"config2");
        assert_eq!(manager.get_all_policy_states().len(), 1);
    }

    #[test]
    fn test_get_all_worker_states_empty() {
        let manager = create_test_sync_manager();
        let states = manager.get_all_worker_states();
        assert!(states.is_empty());
    }

    #[test]
    fn test_get_all_policy_states_empty() {
        let manager = create_test_sync_manager();
        let states = manager.get_all_policy_states();
        assert!(states.is_empty());
    }

    #[test]
    fn test_update_rate_limit_membership() {
        let manager = create_test_manager("node1".to_string());

        // Add membership nodes
        let _ = manager.stores.membership.insert(
            "node1".to_string(),
            MembershipState {
                name: "node1".to_string(),
                address: "127.0.0.1:8000".to_string(),
                status: NodeStatus::Alive as i32,
                version: 1,
                metadata: BTreeMap::new(),
            },
        );

        let _ = manager.stores.membership.insert(
            "node2".to_string(),
            MembershipState {
                name: "node2".to_string(),
                address: "127.0.0.1:8001".to_string(),
                status: NodeStatus::Alive as i32,
                version: 1,
                metadata: BTreeMap::new(),
            },
        );

        manager.update_rate_limit_membership();

        // Check that hash ring was updated
        let owners = manager.stores.rate_limit.get_owners("test_key");
        assert!(!owners.is_empty());
    }

    #[test]
    fn test_handle_node_failure() {
        let manager = create_test_manager("node1".to_string());

        // Setup membership
        let _ = manager.stores.membership.insert(
            "node1".to_string(),
            MembershipState {
                name: "node1".to_string(),
                address: "127.0.0.1:8000".to_string(),
                status: NodeStatus::Alive as i32,
                version: 1,
                metadata: BTreeMap::new(),
            },
        );

        let _ = manager.stores.membership.insert(
            "node2".to_string(),
            MembershipState {
                name: "node2".to_string(),
                address: "127.0.0.1:8001".to_string(),
                status: NodeStatus::Alive as i32,
                version: 1,
                metadata: BTreeMap::new(),
            },
        );

        manager.update_rate_limit_membership();

        // Handle node failure
        manager.handle_node_failure(&["node2".to_string()]);

        // Membership should be updated
        manager.update_rate_limit_membership();
    }

    #[test]
    fn test_sync_rate_limit_inc() {
        let manager = create_test_manager("node1".to_string());

        // Setup membership to make node1 an owner
        manager
            .stores
            .rate_limit
            .update_membership(&["node1".to_string()]);

        let test_key = "test_key".to_string();
        if manager.stores.rate_limit.is_owner(&test_key) {
            manager.sync_rate_limit_inc(test_key.clone(), 5);

            let value = manager.get_rate_limit_value(&test_key);
            assert_eq!(value, Some(5));
        }
    }

    #[test]
    fn test_sync_rate_limit_inc_non_owner() {
        let manager = create_test_manager("node1".to_string());

        // Setup membership without node1
        manager
            .stores
            .rate_limit
            .update_membership(&["node2".to_string(), "node3".to_string()]);

        let test_key = "test_key".to_string();
        if !manager.stores.rate_limit.is_owner(&test_key) {
            manager.sync_rate_limit_inc(test_key.clone(), 5);

            // Should not increment if not owner
            let value = manager.get_rate_limit_value(&test_key);
            assert_eq!(value, None);
        }
    }

    #[test]
    fn test_get_global_rate_limit_config() {
        let manager = create_test_manager("node1".to_string());

        // Initially should be None
        assert!(manager.get_global_rate_limit_config().is_none());

        // Set config
        let config = RateLimitConfig {
            limit_per_second: 100,
        };
        let serialized = bincode::serialize(&config).unwrap();
        let _ = manager.stores.app.insert(
            GLOBAL_RATE_LIMIT_KEY.to_string(),
            AppState {
                key: GLOBAL_RATE_LIMIT_KEY.to_string(),
                value: serialized,
                version: 1,
            },
        );

        let retrieved = manager.get_global_rate_limit_config().unwrap();
        assert_eq!(retrieved.limit_per_second, 100);
    }

    #[test]
    fn test_check_global_rate_limit() {
        let manager = create_test_manager("node1".to_string());

        // Setup config
        let config = RateLimitConfig {
            limit_per_second: 10,
        };
        let serialized = bincode::serialize(&config).unwrap();
        let _ = manager.stores.app.insert(
            GLOBAL_RATE_LIMIT_KEY.to_string(),
            AppState {
                key: GLOBAL_RATE_LIMIT_KEY.to_string(),
                value: serialized,
                version: 1,
            },
        );

        // Setup membership
        manager
            .stores
            .rate_limit
            .update_membership(&["node1".to_string()]);

        // Check rate limit
        let (is_exceeded, _current_count, limit) = manager.check_global_rate_limit();
        assert!(!is_exceeded); // First check should not exceed
        assert_eq!(limit, 10);

        // Increment multiple times
        for _ in 0..15 {
            manager.check_global_rate_limit();
        }

        let (is_exceeded2, current_count2, _) = manager.check_global_rate_limit();
        // Should exceed after many increments
        assert!(is_exceeded2 || current_count2 > 10);
    }

    #[test]
    fn test_reset_global_rate_limit_counter() {
        let manager = create_test_manager("node1".to_string());

        // Setup membership
        manager
            .stores
            .rate_limit
            .update_membership(&["node1".to_string()]);

        // Increment counter
        if manager
            .stores
            .rate_limit
            .is_owner(GLOBAL_RATE_LIMIT_COUNTER_KEY)
        {
            manager.sync_rate_limit_inc(GLOBAL_RATE_LIMIT_COUNTER_KEY.to_string(), 10);
            let value = manager.get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY);
            assert!(value.is_some() && value.unwrap() > 0);

            // Reset
            manager.reset_global_rate_limit_counter();
            let value_after = manager.get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY);
            // Should be reset (0 or negative)
            assert!(value_after.is_none() || value_after.unwrap() <= 0);
        }
    }

    #[test]
    fn test_sync_tree_operation() {
        let manager = create_test_manager("node1".to_string());

        use crate::tree_ops::{TreeInsertOp, TreeKey, TreeOperation};

        let op = TreeOperation::Insert(TreeInsertOp {
            key: TreeKey::Text("test_text".to_string()),
            tenant: "http://localhost:8000".to_string(),
        });

        let result = manager.sync_tree_operation("model1".to_string(), op);
        assert!(result.is_ok());

        // sync_tree_operation no longer populates tree_configs (no subscribers
        // in unit tests), so get_tree_state returns None.  Instead, verify
        // that the tenant delta was buffered.
        let inserts = manager.stores.tenant_delta_inserts.get("model1").unwrap();
        assert_eq!(inserts.len(), 1);
        assert_eq!(inserts[0].worker_url, "http://localhost:8000");
        assert_eq!(inserts[0].node_path_hash, hash_node_path("test_text"));
    }

    #[test]
    fn test_get_tree_state() {
        let manager = create_test_manager("node1".to_string());

        // Initially should be None
        assert!(manager.get_tree_state("model1").is_none());

        // sync_tree_operation only buffers tenant deltas and bumps the version
        // counter — it does NOT populate tree_configs (that requires a
        // subscriber-backed checkpoint).  Verify get_tree_state returns None
        // after sync, but the tenant delta was buffered.
        use crate::tree_ops::{TreeInsertOp, TreeKey, TreeOperation};
        let op = TreeOperation::Insert(TreeInsertOp {
            key: TreeKey::Text("test_text".to_string()),
            tenant: "http://localhost:8000".to_string(),
        });
        manager
            .sync_tree_operation("model1".to_string(), op)
            .unwrap();

        // get_tree_state reads from tree_configs which is empty (no subscriber)
        assert!(manager.get_tree_state("model1").is_none());
        // But the tenant delta insert was buffered
        assert!(manager.stores.tenant_delta_inserts.get("model1").is_some());
    }

    #[test]
    fn test_apply_remote_tree_operation() {
        let manager = create_test_manager("node1".to_string());

        use crate::tree_ops::{TreeInsertOp, TreeKey, TreeOperation, TreeState};

        let mut tree_state = TreeState::new("model1".to_string());
        tree_state.version = 5;
        tree_state.add_operation(TreeOperation::Insert(TreeInsertOp {
            key: TreeKey::Text("remote_text".to_string()),
            tenant: "http://localhost:8001".to_string(),
        }));
        // add_operation increments version, so version is now 6

        manager.apply_remote_tree_operation(
            "model1".to_string(),
            tree_state,
            Some("node2".to_string()),
        );

        let retrieved = manager.get_tree_state("model1").unwrap();
        assert_eq!(retrieved.version, 6); // add_operation increments version from 5 to 6
        assert_eq!(retrieved.operations.len(), 1);
    }

    #[test]
    fn test_notify_tree_state_subscribers_drops_lock_before_callback() {
        let manager = Arc::new(create_test_manager("node1".to_string()));
        let can_acquire_write_lock = Arc::new(AtomicBool::new(false));
        let subscriber = Arc::new(LockCheckingSubscriber {
            manager: manager.clone(),
            can_acquire_write_lock: can_acquire_write_lock.clone(),
        });

        manager.register_tree_state_subscriber(subscriber);
        manager.notify_tree_state_subscribers("model1", &TreeState::new("model1".to_string()));

        assert!(can_acquire_write_lock.load(Ordering::SeqCst));
    }

    #[test]
    fn test_get_all_tree_states() {
        let manager = create_test_manager("node1".to_string());

        // get_all_tree_states reads from tree_configs. In unit tests there are
        // no subscribers, so sync_tree_operation won't populate tree_configs.
        // Instead, insert TreeStates directly into tree_configs.
        let mut ts1 = TreeState::new("model1".to_string());
        ts1.add_operation(make_insert_op("alpha", "http://localhost:8000"));
        let mut ts2 = TreeState::new("model2".to_string());
        ts2.add_operation(make_insert_op("beta", "http://localhost:8001"));

        manager
            .stores
            .tree_configs
            .insert("tree:model1".to_string(), ts1.to_bytes().unwrap());
        manager
            .stores
            .tree_configs
            .insert("tree:model2".to_string(), ts2.to_bytes().unwrap());

        let mut states = manager.get_all_tree_states();
        states.sort_by(|left, right| left.model_id.cmp(&right.model_id));

        assert_eq!(states.len(), 2);
        assert_eq!(states[0].model_id, "model1");
        assert_eq!(states[1].model_id, "model2");
    }

    #[test]
    fn test_get_all_worker_states() {
        let manager = create_test_manager("node1".to_string());

        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://localhost:8000".to_string(),
            true,
            0.5,
            vec![],
        );

        manager.sync_worker_state(
            "worker2".to_string(),
            "model2".to_string(),
            "http://localhost:8001".to_string(),
            false,
            0.8,
            vec![],
        );

        let all_states = manager.get_all_worker_states();
        assert_eq!(all_states.len(), 2);
    }

    #[test]
    fn test_get_all_policy_states() {
        let manager = create_test_manager("node1".to_string());

        manager.sync_policy_state("model1".to_string(), "cache_aware".to_string(), vec![]);

        manager.sync_policy_state("model2".to_string(), "round_robin".to_string(), vec![]);

        let all_states = manager.get_all_policy_states();
        assert_eq!(all_states.len(), 2);
    }

    // ── Delta encoding tests ────────────────────────────────────────────

    use crate::tree_ops::{TreeInsertOp, TreeKey, TreeOperation, TreeRemoveOp, TreeStateDelta};

    fn make_insert_op(text: &str, tenant: &str) -> TreeOperation {
        TreeOperation::Insert(TreeInsertOp {
            key: TreeKey::Text(text.to_string()),
            tenant: tenant.to_string(),
        })
    }

    fn make_delta(model_id: &str, ops: Vec<TreeOperation>, base: u64, new: u64) -> TreeStateDelta {
        TreeStateDelta {
            model_id: model_id.to_string(),
            operations: ops,
            base_version: base,
            new_version: new,
        }
    }

    #[test]
    fn test_delta_basic_apply() {
        let manager = create_test_manager("node1".to_string());

        let ops = vec![
            make_insert_op("a", "http://w1:8000"),
            make_insert_op("b", "http://w2:8000"),
            make_insert_op("c", "http://w3:8000"),
        ];

        let delta = make_delta("model1", ops, 0, 3);
        manager.apply_remote_tree_delta(delta, Some("node2".to_string()));

        let tree = manager.get_tree_state("model1").unwrap();
        assert_eq!(tree.version, 3);
        assert_eq!(tree.operations.len(), 3);
    }

    #[test]
    fn test_delta_version_check_rejects_gap() {
        let manager = create_test_manager("node1".to_string());

        // Seed tree at version 10
        let mut seed = TreeState::new("model1".to_string());
        for i in 0..10 {
            seed.add_operation(make_insert_op(&format!("seed_{i}"), "http://w:8000"));
        }
        assert_eq!(seed.version, 10);
        manager.apply_remote_tree_operation("model1".to_string(), seed, Some("seed".to_string()));

        // Delta with base_version=5 should be accepted (base <= current)
        let delta_ok = make_delta("model1", vec![make_insert_op("ok", "http://w:8000")], 5, 11);
        manager.apply_remote_tree_delta(delta_ok, None);
        let tree = manager.get_tree_state("model1").unwrap();
        assert_eq!(tree.version, 11);

        // Delta with base_version=20 should be rejected (gap: base > current)
        let delta_gap = make_delta(
            "model1",
            vec![make_insert_op("gap", "http://w:8000")],
            20,
            21,
        );
        manager.apply_remote_tree_delta(delta_gap, None);
        let tree = manager.get_tree_state("model1").unwrap();
        // Version should still be 11 — the gap delta was rejected
        assert_eq!(tree.version, 11);
    }

    #[test]
    fn test_delta_concurrent_apply() {
        let manager = Arc::new(create_test_manager("node1".to_string()));

        // Both deltas target the same empty tree.  At least one must succeed,
        // and the resulting version must reflect the applied operations.
        let m1 = manager.clone();
        let m2 = manager.clone();

        let t1 = std::thread::spawn(move || {
            let delta = make_delta("model1", vec![make_insert_op("t1", "http://w1:8000")], 0, 1);
            m1.apply_remote_tree_delta(delta, Some("thread1".to_string()));
        });

        let t2 = std::thread::spawn(move || {
            let delta = make_delta("model1", vec![make_insert_op("t2", "http://w2:8000")], 0, 1);
            m2.apply_remote_tree_delta(delta, Some("thread2".to_string()));
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // At least one delta should have been applied
        let tree = manager.get_tree_state("model1").unwrap();
        assert!(tree.version >= 1);
        assert!(!tree.operations.is_empty());
    }

    #[test]
    fn test_delta_empty_tree() {
        let manager = create_test_manager("node1".to_string());

        // No pre-existing tree for "new_model"
        assert!(manager.get_tree_state("new_model").is_none());

        let delta = make_delta(
            "new_model",
            vec![make_insert_op("first", "http://w1:8000")],
            0,
            1,
        );
        manager.apply_remote_tree_delta(delta, None);

        let tree = manager.get_tree_state("new_model").unwrap();
        assert_eq!(tree.model_id, "new_model");
        assert_eq!(tree.version, 1);
        assert_eq!(tree.operations.len(), 1);
    }

    #[test]
    fn test_delta_notifies_subscribers() {
        let manager = Arc::new(create_test_manager("node1".to_string()));
        let notified = Arc::new(AtomicBool::new(false));

        #[derive(Debug)]
        struct FlagSubscriber(Arc<AtomicBool>);
        impl TreeStateSubscriber for FlagSubscriber {
            fn apply_remote_tree_state(&self, _model_id: &str, _tree_state: &TreeState) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        manager.register_tree_state_subscriber(Arc::new(FlagSubscriber(notified.clone())));

        let delta = make_delta("model1", vec![make_insert_op("x", "http://w:8000")], 0, 1);
        manager.apply_remote_tree_delta(delta, None);

        assert!(
            notified.load(Ordering::SeqCst),
            "subscriber was not notified after delta apply"
        );
    }

    #[test]
    fn test_collector_sends_tenant_delta() {
        use crate::tree_ops::TenantDelta;

        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let manager = MeshSyncManager::new(stores.clone(), "node1".to_string());

        // Sync a tree operation — buffers a tenant insert
        manager
            .sync_tree_operation(
                "model1".to_string(),
                make_insert_op("hello world", "http://w:8000"),
            )
            .unwrap();

        let updates = collect_policy_updates(stores.clone(), "node1");

        assert!(!updates.is_empty(), "expected at least one policy update");

        // The update should be a tenant delta (not full tree state)
        let tree_update = updates
            .iter()
            .find(|u| u.key.starts_with("tree:"))
            .expect("expected a tree key update");

        let policy_state: PolicyState =
            bincode::deserialize(&tree_update.value).expect("deserialize PolicyState");
        assert_eq!(
            policy_state.policy_type, "tenant_delta",
            "expected tenant_delta, got {}",
            policy_state.policy_type
        );

        // Verify the tenant delta deserializes and contains the insert
        let delta = TenantDelta::from_bytes(&policy_state.config).expect("deserialize TenantDelta");
        assert_eq!(delta.model_id, "model1");
        assert_eq!(delta.inserts.len(), 1);
        assert_eq!(delta.inserts[0].worker_url, "http://w:8000");
        assert_eq!(
            delta.inserts[0].node_path_hash,
            hash_node_path("hello world")
        );
        assert!(delta.evictions.is_empty());
    }

    #[test]
    fn test_collector_falls_back_to_full_state() {
        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));

        // Directly insert a tree state into tree_configs WITHOUT going through
        // sync_tree_operation (so tree_ops_pending is empty).  This simulates
        // a remote tree state received via apply_remote_tree_operation.
        let mut tree = TreeState::new("model1".to_string());
        tree.add_operation(make_insert_op("direct", "http://w:8000"));
        let serialized = tree.to_bytes().unwrap();
        stores
            .tree_configs
            .insert("tree:model1".to_string(), serialized);
        // Advance tree version so the collector sees it as changed.
        stores.advance_tree_version("tree:model1", tree.version);
        // Bump tree_generation so the collector's tree_changed check fires.
        stores.bump_tree_version("tree:model1");

        let updates = collect_policy_updates(stores.clone(), "node1");

        assert!(!updates.is_empty(), "expected at least one policy update");

        let tree_update = updates
            .iter()
            .find(|u| u.key.starts_with("tree:"))
            .expect("expected a tree key update");

        // Since there are no pending ops, it should fall back to full PolicyState
        let policy_state: PolicyState =
            bincode::deserialize(&tree_update.value).expect("deserialize PolicyState");
        assert_eq!(
            policy_state.policy_type, "tree_state_lz4",
            "expected full state fallback, got delta"
        );
    }

    // test_collector_buffer_survives_mark_sent removed: tested tree_ops_pending
    // buffer survival across mark_sent calls, which is a dead code path now
    // that sync_tree_operation no longer pushes to tree_ops_pending.

    #[test]
    fn test_receiver_dispatches_delta_vs_full() {
        let manager = create_test_manager("node1".to_string());

        // 1. Apply via delta path
        let delta = make_delta(
            "model_d",
            vec![make_insert_op("delta_op", "http://w:8000")],
            0,
            1,
        );
        manager.apply_remote_tree_delta(delta, Some("remote".to_string()));

        let tree_d = manager.get_tree_state("model_d").unwrap();
        assert_eq!(tree_d.version, 1);
        assert_eq!(tree_d.operations.len(), 1);

        // 2. Apply via full state path
        let mut full_tree = TreeState::new("model_f".to_string());
        full_tree.add_operation(make_insert_op("full_op1", "http://w1:8000"));
        full_tree.add_operation(make_insert_op("full_op2", "http://w2:8000"));

        manager.apply_remote_tree_operation(
            "model_f".to_string(),
            full_tree,
            Some("remote".to_string()),
        );

        let tree_f = manager.get_tree_state("model_f").unwrap();
        assert_eq!(tree_f.version, 2);
        assert_eq!(tree_f.operations.len(), 2);
    }

    #[test]
    fn test_delta_backward_compatible_full_state() {
        let manager = create_test_manager("node1".to_string());

        // Simulate receiving a full TreeState (the old, pre-delta format)
        let mut old_format_tree = TreeState::new("legacy_model".to_string());
        old_format_tree.add_operation(make_insert_op("old1", "http://w:8000"));
        old_format_tree.add_operation(make_insert_op("old2", "http://w:8000"));

        // The full-state path (apply_remote_tree_operation) should handle it
        manager.apply_remote_tree_operation(
            "legacy_model".to_string(),
            old_format_tree.clone(),
            Some("old_node".to_string()),
        );

        let tree = manager.get_tree_state("legacy_model").unwrap();
        assert_eq!(tree.version, old_format_tree.version);
        assert_eq!(tree.operations.len(), 2);
        assert_eq!(tree.model_id, "legacy_model");
    }

    // ── Edge-case delta encoding tests ─────────────────────────────────

    #[test]
    fn test_delta_reconnect_falls_back_to_full_state() {
        // Simulate a reconnected peer scenario: tree_configs has a materialized
        // tree state but tenant delta buffers are empty.  The collector should
        // produce a full PolicyState (lz4-compressed), not a delta.
        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));

        // Directly insert a tree state into tree_configs (simulating a
        // checkpoint that ran with real subscribers in production).
        let mut tree = TreeState::new("model1".to_string());
        for i in 0..10 {
            tree.add_operation(make_insert_op(&format!("op_{i}"), "http://w:8000"));
        }
        let serialized = tree.to_bytes().unwrap();
        stores
            .tree_configs
            .insert("tree:model1".to_string(), serialized);
        stores.advance_tree_version("tree:model1", tree.version);
        stores.bump_tree_version("tree:model1");

        // Ensure tenant delta buffers are empty (simulating buffer drain)
        stores.tenant_delta_inserts.remove("model1");
        stores.tenant_delta_evictions.remove("model1");

        // Collect via v2 central collector (simulating reconnected peer)
        let updates = collect_policy_updates(stores.clone(), "node1");

        assert!(!updates.is_empty(), "expected at least one update");

        let tree_update = updates
            .iter()
            .find(|u| u.key.starts_with("tree:"))
            .expect("expected a tree key update");

        let policy_state: PolicyState =
            bincode::deserialize(&tree_update.value).expect("deserialize PolicyState");
        assert_eq!(
            policy_state.policy_type, "tree_state_lz4",
            "expected full state fallback when tenant delta buffers are empty, got: {}",
            policy_state.policy_type
        );
    }

    // test_delta_compaction_divergence removed: tested TreeState compaction
    // via sync_tree_operation + get_tree_state, which relied on tree_ops_pending
    // replay. sync_tree_operation no longer pushes to tree_ops_pending, and
    // get_tree_state reads only from tree_configs (populated by subscribers).

    #[test]
    fn test_delta_out_of_order_delivery() {
        // Create tree at version 0.  Apply delta [0→5], then apply stale
        // delta [0→3].  The second delta should be rejected because the
        // tree is already at version 5.
        let manager = create_test_manager("node1".to_string());

        let ops_1_to_5: Vec<_> = (1..=5)
            .map(|i| make_insert_op(&format!("op_{i}"), "http://w:8000"))
            .collect();
        let delta1 = make_delta("model1", ops_1_to_5, 0, 5);
        manager.apply_remote_tree_delta(delta1, Some("peer_a".to_string()));

        let tree = manager.get_tree_state("model1").unwrap();
        assert_eq!(tree.version, 5);
        assert_eq!(tree.operations.len(), 5);

        // Late-arriving delta with lower new_version
        let ops_1_to_3: Vec<_> = (1..=3)
            .map(|i| make_insert_op(&format!("late_op_{i}"), "http://w:8000"))
            .collect();
        let delta2 = make_delta("model1", ops_1_to_3, 0, 3);
        manager.apply_remote_tree_delta(delta2, Some("peer_b".to_string()));

        // Tree should be unchanged — stale delta rejected
        let tree_after = manager.get_tree_state("model1").unwrap();
        assert_eq!(tree_after.version, 5);
        assert_eq!(tree_after.operations.len(), 5);
    }

    #[test]
    fn test_delta_duplicate_delivery() {
        // Apply the same delta twice.  The second application must be a
        // no-op because current version >= delta.new_version.
        let manager = create_test_manager("node1".to_string());

        let ops = vec![
            make_insert_op("dup1", "http://w:8000"),
            make_insert_op("dup2", "http://w:8000"),
        ];
        let delta = make_delta("model1", ops.clone(), 0, 2);

        manager.apply_remote_tree_delta(delta.clone(), Some("peer".to_string()));
        let tree1 = manager.get_tree_state("model1").unwrap();
        assert_eq!(tree1.version, 2);
        assert_eq!(tree1.operations.len(), 2);

        // Second apply — duplicate
        manager.apply_remote_tree_delta(delta, Some("peer".to_string()));
        let tree2 = manager.get_tree_state("model1").unwrap();
        assert_eq!(
            tree2.version, 2,
            "duplicate delta should not change version"
        );
        assert_eq!(
            tree2.operations.len(),
            2,
            "duplicate delta should not add extra ops"
        );
    }

    #[test]
    fn test_delta_split_brain_recovery() {
        // Node A and Node B both start at version 5.
        // A processes 3 ops (version 8).  B has the seed at version 5
        // in tree_configs (local ops via sync_tree_operation only bump
        // the atomic counter, not tree_configs).
        // A sends delta(base=5, new=8) to B.
        // B's tree_configs version is 5.
        //   base(5) <= current(5) ✓
        //   current(5) < new(8) ✓
        // So B accepts and applies the 3 ops.
        let manager = create_test_manager("nodeB".to_string());

        // Seed the tree at version 5 (common ancestor) — writes to tree_configs
        let mut seed = TreeState::new("model1".to_string());
        for i in 0..5 {
            seed.add_operation(make_insert_op(&format!("seed_{i}"), "http://w:8000"));
        }
        assert_eq!(seed.version, 5);
        manager.apply_remote_tree_operation("model1".to_string(), seed, Some("origin".to_string()));

        // Verify tree_configs has version 5
        let tree_b = manager.get_tree_state("model1").unwrap();
        assert_eq!(tree_b.version, 5);

        // A's delta: base=5, new=8, 3 ops
        let a_ops: Vec<_> = (0..3)
            .map(|i| make_insert_op(&format!("A_op_{i}"), "http://wA:8000"))
            .collect();
        let delta_a = make_delta("model1", a_ops, 5, 8);
        manager.apply_remote_tree_delta(delta_a, Some("nodeA".to_string()));

        // After apply, tree should have seed ops + A's ops.
        let tree_merged = manager.get_tree_state("model1").unwrap();
        assert_eq!(
            tree_merged.version, 8,
            "tree_configs version should be 8 (seed 5 + 3 delta ops), got {}",
            tree_merged.version
        );
        assert_eq!(tree_merged.operations.len(), 8);
    }

    // test_delta_buffer_trim_multi_peer removed: tested tree_ops_pending trim
    // behavior across multiple peer collectors. sync_tree_operation no longer
    // pushes to tree_ops_pending, making this a dead code path.

    // test_delta_empty_pending_vec removed: tested empty tree_ops_pending
    // fallback to full state. sync_tree_operation no longer pushes to
    // tree_ops_pending, making this a dead code path.

    #[test]
    fn test_delta_concurrent_write_and_collect() {
        // Spawn a thread that adds 100 ops via sync_tree_operation.
        // Simultaneously run the collector.  The collector should get a
        // consistent snapshot — either some ops or all ops, but never
        // corrupted data.
        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let manager = Arc::new(MeshSyncManager::new(stores.clone(), "node1".to_string()));

        let manager_clone = manager.clone();
        let writer = std::thread::spawn(move || {
            for i in 0..100 {
                manager_clone
                    .sync_tree_operation(
                        "model1".to_string(),
                        make_insert_op(&format!("concurrent_op_{i}"), "http://w:8000"),
                    )
                    .unwrap();
            }
        });

        // Collect multiple times while writer is active
        let mut collected_any = false;
        for _ in 0..10 {
            let updates = collect_policy_updates(stores.clone(), "node1");
            for update in &updates {
                if update.key.starts_with("tree:") {
                    // Verify the data deserializes without corruption
                    let policy_state: PolicyState =
                        bincode::deserialize(&update.value).expect("data should not be corrupted");
                    match policy_state.policy_type.as_str() {
                        "tenant_delta" => {
                            TenantDelta::from_bytes(&policy_state.config)
                                .expect("tenant delta should deserialize cleanly");
                        }
                        "tree_state_delta" => {
                            let delta = TreeStateDelta::from_bytes(&policy_state.config)
                                .expect("delta should deserialize cleanly");
                            assert!(!delta.operations.is_empty());
                        }
                        "tree_state_lz4" => {
                            let decompressed =
                                crate::tree_ops::lz4_decompress(&policy_state.config)
                                    .expect("lz4 should decompress cleanly");
                            let tree = TreeState::from_bytes(&decompressed)
                                .expect("tree state should deserialize cleanly");
                            assert!(!tree.operations.is_empty());
                        }
                        "tree_state" => {
                            let tree = TreeState::from_bytes(&policy_state.config)
                                .expect("tree state should deserialize cleanly");
                            assert!(!tree.operations.is_empty());
                        }
                        other => panic!("unexpected policy_type: {other}"),
                    }
                    collected_any = true;
                }
            }
        }

        writer.join().unwrap();

        // After writer finishes, one final collect should succeed
        let final_updates = collect_policy_updates(stores.clone(), "node1");
        if !collected_any {
            // Writer may have been too fast; at least final collection must succeed
            assert!(
                !final_updates.is_empty(),
                "final collection after writer finished should have updates"
            );
        }
    }

    // test_delta_oversized_mark_sent_trims_buffer removed: tested
    // tree_ops_pending trim threshold during mark_sent. sync_tree_operation
    // no longer pushes to tree_ops_pending, making this a dead code path.

    // test_delta_version_monotonic_after_compaction removed: tested version
    // monotonicity across compaction by calling sync_tree_operation 3000 times
    // and reading back via get_tree_state. Both paths relied on tree_ops_pending
    // replay, which is a dead code path now.

    #[test]
    fn test_delta_with_remove_operations() {
        // Verify that deltas containing Remove operations work correctly
        let manager = create_test_manager("node1".to_string());

        let ops = vec![
            make_insert_op("text1", "http://w1:8000"),
            TreeOperation::Remove(TreeRemoveOp {
                tenant: "http://w1:8000".to_string(),
            }),
            make_insert_op("text2", "http://w2:8000"),
        ];

        let delta = make_delta("model1", ops, 0, 3);
        manager.apply_remote_tree_delta(delta, Some("peer".to_string()));

        let tree = manager.get_tree_state("model1").unwrap();
        assert_eq!(tree.version, 3);
        assert_eq!(tree.operations.len(), 3);
        // Verify the remove op is present
        assert!(matches!(
            tree.operations[1],
            TreeOperation::Remove(TreeRemoveOp { .. })
        ));
    }

    #[test]
    fn test_delta_multiple_models_independent() {
        // Verify that deltas for different models don't interfere with
        // each other
        let manager = create_test_manager("node1".to_string());

        let delta_a = make_delta(
            "model_a",
            vec![make_insert_op("a_op", "http://w:8000")],
            0,
            1,
        );
        let delta_b = make_delta(
            "model_b",
            vec![
                make_insert_op("b_op1", "http://w:8000"),
                make_insert_op("b_op2", "http://w:8000"),
            ],
            0,
            2,
        );

        manager.apply_remote_tree_delta(delta_a, None);
        manager.apply_remote_tree_delta(delta_b, None);

        let tree_a = manager.get_tree_state("model_a").unwrap();
        let tree_b = manager.get_tree_state("model_b").unwrap();

        assert_eq!(tree_a.version, 1);
        assert_eq!(tree_a.operations.len(), 1);
        assert_eq!(tree_b.version, 2);
        assert_eq!(tree_b.operations.len(), 2);
    }

    #[test]
    fn test_delta_incremental_chain() {
        // Apply a chain of sequential deltas: 0→3, 3→5, 5→8
        // Each should be accepted and the tree should accumulate all ops.
        let manager = create_test_manager("node1".to_string());

        let delta1 = make_delta(
            "model1",
            (0..3)
                .map(|i| make_insert_op(&format!("batch1_op_{i}"), "http://w:8000"))
                .collect(),
            0,
            3,
        );
        manager.apply_remote_tree_delta(delta1, None);
        let tree = manager.get_tree_state("model1").unwrap();
        assert_eq!(tree.version, 3);

        let delta2 = make_delta(
            "model1",
            (0..2)
                .map(|i| make_insert_op(&format!("batch2_op_{i}"), "http://w:8000"))
                .collect(),
            3,
            5,
        );
        manager.apply_remote_tree_delta(delta2, None);
        let tree = manager.get_tree_state("model1").unwrap();
        assert_eq!(tree.version, 5);

        let delta3 = make_delta(
            "model1",
            (0..3)
                .map(|i| make_insert_op(&format!("batch3_op_{i}"), "http://w:8000"))
                .collect(),
            5,
            8,
        );
        manager.apply_remote_tree_delta(delta3, None);
        let tree = manager.get_tree_state("model1").unwrap();
        assert_eq!(tree.version, 8);
        assert_eq!(tree.operations.len(), 8);
    }

    #[test]
    fn test_delta_token_key_serialization_round_trip() {
        // Verify that deltas with TreeKey::Tokens survive serialization
        // through the full delta encode/decode path.
        use crate::tree_ops::TreeInsertOp;

        let tokens = vec![42u32, 100, 200, 999, u32::MAX];
        let ops = vec![TreeOperation::Insert(TreeInsertOp {
            key: TreeKey::Tokens(tokens.clone()),
            tenant: "http://w:8000".to_string(),
        })];

        let delta = TreeStateDelta {
            model_id: "token_model".to_string(),
            operations: ops,
            base_version: 0,
            new_version: 1,
        };

        // Serialize and deserialize
        let bytes = delta.to_bytes().unwrap();
        let restored = TreeStateDelta::from_bytes(&bytes).unwrap();
        assert_eq!(restored.operations.len(), 1);

        match &restored.operations[0] {
            TreeOperation::Insert(op) => {
                assert_eq!(op.key, TreeKey::Tokens(tokens));
            }
            TreeOperation::Remove(_) => panic!("expected Insert operation"),
        }

        // Apply the delta to a manager and verify the tree
        let manager = create_test_manager("node1".to_string());
        manager.apply_remote_tree_delta(restored, None);

        let tree = manager.get_tree_state("token_model").unwrap();
        assert_eq!(tree.version, 1);
        assert_eq!(tree.operations.len(), 1);
    }
}
