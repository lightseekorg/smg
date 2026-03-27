//! Mesh state synchronization module
//!
//! Handles synchronization of worker and policy states across mesh cluster nodes

use std::{fmt::Debug, sync::Arc};

use parking_lot::RwLock;
use tracing::{debug, warn};

use super::{
    service::gossip::NodeStatus,
    stores::{
        policy_key, tree_state_key, PolicyState, RateLimitConfig, StateStores, WorkerState,
        GLOBAL_RATE_LIMIT_COUNTER_KEY, GLOBAL_RATE_LIMIT_KEY,
    },
    tree_ops::{TreeOperation, TreeState, TreeStateDelta},
};

pub trait TreeStateSubscriber: Send + Sync + Debug {
    fn apply_remote_tree_state(&self, model_id: &str, tree_state: &TreeState);
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

        // Append to pending buffer for delta sync (O(1)).
        self.stores
            .tree_ops_pending
            .entry(key.clone())
            .or_default()
            .push(operation);

        // Bump the lightweight atomic version counter (O(1), no serialization).
        //
        // Previously this called `stores.policy.update()` which goes through
        // the full CRDT machinery: deserialize PolicyState (including the
        // potentially large config blob), re-serialize, clone for operation
        // log — costing ~5× the config size per request.  After checkpoint
        // materializes the TreeState (~1 MB), this was ~1 GB/s of allocation
        // churn at 200 rps.
        //
        // The collector now detects tree changes via `tree_generation` and
        // reads version from `tree_versions` instead of PolicyState.version.
        self.stores.bump_tree_version(&key);

        Ok(())
    }

    /// Materialize a TreeState for the given tree key by combining the
    /// policy store's config blob (may be stale) with any pending operations.
    fn materialize_tree_state(&self, key: &str, model_id: &str) -> Option<TreeState> {
        let policy_state = self.stores.policy.get(key);
        let mut tree_state = match policy_state {
            Some(ps) if !ps.config.is_empty() => TreeState::from_bytes(&ps.config).ok()?,
            Some(_) => TreeState::new(model_id.to_string()),
            None => {
                // No policy store entry yet — tree may only exist in the
                // pending buffer (sync_tree_operation no longer creates
                // policy entries on the hot path).
                if self.stores.tree_ops_pending.contains_key(key) {
                    TreeState::new(model_id.to_string())
                } else {
                    return None;
                }
            }
        };
        if let Some(pending) = self.stores.tree_ops_pending.get(key) {
            for op in pending.iter() {
                tree_state.add_operation(op.clone());
            }
        }
        Some(tree_state)
    }

    /// Get tree state for a model from mesh stores
    pub fn get_tree_state(&self, model_id: &str) -> Option<TreeState> {
        let key = tree_state_key(model_id);
        self.materialize_tree_state(&key, model_id)
    }

    pub fn get_all_tree_states(&self) -> Vec<TreeState> {
        let mut seen = std::collections::HashSet::new();
        let mut results = Vec::new();

        // Collect from policy store (checkpointed trees)
        for (key, state) in self.stores.policy.all() {
            if state.policy_type != "tree_state" {
                continue;
            }
            seen.insert(key.clone());
            if let Some(ts) = self.materialize_tree_state(&key, &state.model_id) {
                results.push(ts);
            }
        }

        // Also collect trees that only exist in the pending buffer
        // (not yet checkpointed into policy store)
        for entry in &self.stores.tree_ops_pending {
            let key = entry.key();
            if seen.contains(key.as_str()) {
                continue;
            }
            let model_id = key.strip_prefix("tree:").unwrap_or(key);
            if let Some(ts) = self.materialize_tree_state(key, model_id) {
                results.push(ts);
            }
        }

        results
    }

    /// Apply remote tree operation to local policy
    /// This is called when receiving tree state updates from other nodes
    pub fn apply_remote_tree_operation(
        &self,
        model_id: String,
        tree_state: TreeState,
        actor: Option<String>,
    ) {
        let key = tree_state_key(&model_id);
        let actor = actor.unwrap_or_else(|| "remote".to_string());

        let serialized = match tree_state.to_bytes() {
            Ok(bytes) => bytes,
            Err(err) => {
                debug!(error = %err, model_id = %model_id, "Failed to serialize remote tree state");
                return;
            }
        };

        let mut current_version = 0;
        let update_result = self.stores.policy.update_if(key.clone(), |current| {
            current_version = current
                .as_ref()
                .map(|existing| existing.version)
                .unwrap_or(0);
            if tree_state.version > current_version {
                Some(PolicyState {
                    model_id: model_id.clone(),
                    policy_type: "tree_state".to_string(),
                    config: serialized.clone(),
                    version: tree_state.version,
                })
            } else {
                None
            }
        });

        match update_result {
            Ok((_, true)) => {
                // Note: we do NOT clear tree_ops_pending here — concurrent
                // local ops may have been appended since update_if ran.
                // The periodic checkpoint (every ~60s) handles cleanup.
                debug!(
                    "Applied remote tree state update: model={} (version: {} -> {})",
                    model_id, current_version, tree_state.version
                );
                // Advance atomic tree version so subsequent local ops
                // start above the remote baseline.
                self.stores.advance_tree_version(&key, tree_state.version);
                self.notify_tree_state_subscribers(&model_id, &tree_state);
            }
            Ok((_, false)) => {
                debug!(
                    "Skipped remote tree state update: model={} (version {} <= current {})",
                    model_id, tree_state.version, current_version
                );
            }
            Err(err) => {
                debug!(error = %err, model_id = %model_id, actor = %actor, "Failed to apply remote tree state update");
            }
        }
    }

    /// Apply a delta (incremental operations) from a remote node.
    /// Merges the delta operations into the existing local tree state,
    /// avoiding the cost of replacing the entire tree state on every sync.
    /// Uses atomic compare-and-swap via `update_if` to prevent concurrent
    /// read/modify/write races.
    pub fn apply_remote_tree_delta(&self, delta: TreeStateDelta, actor: Option<String>) {
        let key = tree_state_key(&delta.model_id);
        let actor = actor.unwrap_or_else(|| "remote".to_string());
        let model_id = delta.model_id.clone();
        let ops_count = delta.operations.len();

        let mut old_version = 0u64;

        let update_result = self.stores.policy.update_if(key.clone(), |current| {
            let mut tree_state = if let Some(existing) = current {
                old_version = existing.version;

                // If the delta's base_version is ahead of our current version,
                // there is a gap — we are missing operations.  Skip the delta;
                // the next full-state sync will catch us up.
                if delta.base_version > existing.version {
                    return None;
                }

                // Skip if we already have a version >= the delta's new_version
                if existing.version >= delta.new_version {
                    return None;
                }

                // Build base tree from config only — do NOT replay
                // tree_ops_pending here. Pending ops are local and will be
                // included by materialize_tree_state() when the tree is
                // read for routing or sync.
                if existing.config.is_empty() {
                    if existing.version > 0 {
                        // Config not materialized but we have a version —
                        // local ops exist only in pending buffer. Reject the
                        // delta; the periodic checkpoint will materialize the
                        // config, then the next delta will apply cleanly.
                        return None;
                    }
                    TreeState::new(delta.model_id.clone())
                } else {
                    match TreeState::from_bytes(&existing.config) {
                        Ok(state) => state,
                        Err(err) => {
                            warn!(
                                model_id = %delta.model_id,
                                error = %err,
                                "Corrupted tree state — rejecting delta to avoid data loss"
                            );
                            return None;
                        }
                    }
                }
            } else {
                TreeState::new(delta.model_id.clone())
            };

            for op in &delta.operations {
                tree_state.add_operation(op.clone());
            }

            let serialized = tree_state.to_bytes().ok()?;
            Some(PolicyState {
                model_id: delta.model_id.clone(),
                policy_type: "tree_state".to_string(),
                config: serialized,
                version: tree_state.version,
            })
        });

        match update_result {
            Ok((_, true)) => {
                // Note: we do NOT clear tree_ops_pending here — concurrent
                // local ops may have been appended since update_if ran.
                // The periodic checkpoint (every ~60s) handles cleanup.
                debug!(
                    "Applied remote tree delta: model={} (version: {} -> +{} ops)",
                    model_id, old_version, ops_count
                );

                // Advance atomic tree version so subsequent local ops
                // start above the remote baseline.
                if let Some(ps) = self.stores.policy.get(&key) {
                    self.stores.advance_tree_version(&key, ps.version);
                }

                // Re-read the committed state for subscriber notification
                if let Some(policy_state) = self.stores.policy.get(&key) {
                    if let Ok(tree_state) = TreeState::from_bytes(&policy_state.config) {
                        self.notify_tree_state_subscribers(&model_id, &tree_state);
                    }
                }
            }
            Ok((_, false)) => {
                debug!(
                    "Skipped remote tree delta: model={} (base_version={}, new_version={}, current={})",
                    model_id, delta.base_version, delta.new_version, old_version
                );
            }
            Err(err) => {
                debug!(error = %err, model_id = %model_id, actor = %actor, "Failed to apply remote tree delta");
            }
        }
    }

    /// Checkpoint pending tree operations into the policy store config blob.
    /// Called periodically (e.g., from the GC cycle) to keep the config blob
    /// reasonably fresh and limit how much the pending buffer can drift.
    /// Materialize pending tree ops into policy store config blobs.
    /// Called periodically (every ~60 gossip rounds) to keep the config
    /// blob fresh and bound pending buffer growth.
    pub fn checkpoint_tree_states(&self) {
        let keys: Vec<String> = self
            .stores
            .tree_ops_pending
            .iter()
            .map(|e| e.key().clone())
            .collect();

        for key in keys {
            let model_id = key.strip_prefix("tree:").unwrap_or(&key);
            if let Some(tree_state) = self.materialize_tree_state(&key, model_id) {
                if let Ok(serialized) = tree_state.to_bytes() {
                    // Preserve version monotonicity: use max of current policy
                    // version and tree version to avoid regressing.
                    let _ = self.stores.policy.update(key.clone(), |current| {
                        let current_version = current.map(|c| c.version).unwrap_or(0);
                        PolicyState {
                            model_id: tree_state.model_id.clone(),
                            policy_type: "tree_state".to_string(),
                            config: serialized,
                            version: current_version.max(tree_state.version),
                        }
                    });
                    // Drain materialized ops from the pending buffer.
                    // Use `alter` to atomically keep only ops appended AFTER
                    // materialize_tree_state() read the buffer. The number of
                    // ops materialized is `tree_state.operations.len()`.
                    let materialized_count = tree_state.operations.len();
                    self.stores.tree_ops_pending.alter(&key, |_, mut ops| {
                        if ops.len() <= materialized_count {
                            ops.clear();
                        } else {
                            ops.drain(..materialized_count);
                        }
                        ops
                    });
                }
            }
        }
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
    use crate::stores::{
        AppState, MembershipState, RateLimitConfig, StateStores, GLOBAL_RATE_LIMIT_COUNTER_KEY,
        GLOBAL_RATE_LIMIT_KEY,
    };

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

        // Verify tree state was stored
        let tree_state = manager.get_tree_state("model1");
        assert!(tree_state.is_some());
        let tree = tree_state.unwrap();
        assert_eq!(tree.model_id, "model1");
        assert_eq!(tree.operations.len(), 1);
    }

    #[test]
    fn test_get_tree_state() {
        let manager = create_test_manager("node1".to_string());

        // Initially should be None
        assert!(manager.get_tree_state("model1").is_none());

        // Sync an operation
        use crate::tree_ops::{TreeInsertOp, TreeKey, TreeOperation};
        let op = TreeOperation::Insert(TreeInsertOp {
            key: TreeKey::Text("test_text".to_string()),
            tenant: "http://localhost:8000".to_string(),
        });
        manager
            .sync_tree_operation("model1".to_string(), op)
            .unwrap();

        let tree_state = manager.get_tree_state("model1");
        assert!(tree_state.is_some());
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

        use crate::tree_ops::{TreeInsertOp, TreeKey, TreeOperation};

        manager
            .sync_tree_operation(
                "model1".to_string(),
                TreeOperation::Insert(TreeInsertOp {
                    key: TreeKey::Text("alpha".to_string()),
                    tenant: "http://localhost:8000".to_string(),
                }),
            )
            .unwrap();
        manager
            .sync_tree_operation(
                "model2".to_string(),
                TreeOperation::Insert(TreeInsertOp {
                    key: TreeKey::Tokens(vec![1, 2, 3, 4]),
                    tenant: "http://localhost:8001".to_string(),
                }),
            )
            .unwrap();

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
    fn test_collector_sends_delta_not_full_state() {
        use crate::{incremental::IncrementalUpdateCollector, stores::StoreType};

        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let manager = MeshSyncManager::new(stores.clone(), "node1".to_string());

        // Sync a tree operation (this populates both the policy store and tree_ops_pending)
        manager
            .sync_tree_operation("model1".to_string(), make_insert_op("a", "http://w:8000"))
            .unwrap();

        let collector = IncrementalUpdateCollector::new(stores.clone(), "node1".to_string());
        let updates = collector.collect_updates_for_store(StoreType::Policy);

        assert!(!updates.is_empty(), "expected at least one policy update");

        // The update should be a delta (policy_type = "tree_state_delta")
        let tree_update = updates
            .iter()
            .find(|u| u.key.starts_with("tree:"))
            .expect("expected a tree key update");

        let policy_state: PolicyState =
            bincode::deserialize(&tree_update.value).expect("deserialize PolicyState");
        assert_eq!(
            policy_state.policy_type, "tree_state_delta",
            "expected delta, got full state"
        );

        // Verify the delta deserializes correctly
        let delta =
            TreeStateDelta::from_bytes(&policy_state.config).expect("deserialize TreeStateDelta");
        assert_eq!(delta.model_id, "model1");
        assert_eq!(delta.operations.len(), 1);
    }

    #[test]
    fn test_collector_falls_back_to_full_state() {
        use crate::{incremental::IncrementalUpdateCollector, stores::StoreType};

        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));

        // Directly insert a tree state into the policy store WITHOUT going through
        // sync_tree_operation (so tree_ops_pending is empty).
        let mut tree = TreeState::new("model1".to_string());
        tree.add_operation(make_insert_op("direct", "http://w:8000"));
        let serialized = tree.to_bytes().unwrap();
        let _ = stores.policy.insert(
            "tree:model1".to_string(),
            PolicyState {
                model_id: "model1".to_string(),
                policy_type: "tree_state".to_string(),
                config: serialized,
                version: tree.version,
            },
        );

        let collector = IncrementalUpdateCollector::new(stores.clone(), "node1".to_string());
        let updates = collector.collect_updates_for_store(StoreType::Policy);

        assert!(!updates.is_empty(), "expected at least one policy update");

        let tree_update = updates
            .iter()
            .find(|u| u.key.starts_with("tree:"))
            .expect("expected a tree key update");

        // Since there are no pending ops, it should fall back to full PolicyState
        let policy_state: PolicyState =
            bincode::deserialize(&tree_update.value).expect("deserialize PolicyState");
        assert_eq!(
            policy_state.policy_type, "tree_state",
            "expected full state fallback, got delta"
        );
    }

    #[test]
    fn test_collector_buffer_survives_mark_sent() {
        use crate::{incremental::IncrementalUpdateCollector, stores::StoreType};

        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let manager = MeshSyncManager::new(stores.clone(), "node1".to_string());

        // Sync an operation (populates tree_ops_pending)
        manager
            .sync_tree_operation("model1".to_string(), make_insert_op("a", "http://w:8000"))
            .unwrap();

        // Collector A collects and marks sent
        let collector_a = IncrementalUpdateCollector::new(stores.clone(), "node1".to_string());
        let updates_a = collector_a.collect_updates_for_store(StoreType::Policy);
        assert!(!updates_a.is_empty());
        collector_a.mark_sent(StoreType::Policy, &updates_a);

        // Collector B (simulating a second peer's collector) should still be
        // able to collect, because the buffer was NOT cleared by A's mark_sent.
        let collector_b = IncrementalUpdateCollector::new(stores.clone(), "node1".to_string());
        let updates_b = collector_b.collect_updates_for_store(StoreType::Policy);
        assert!(
            !updates_b.is_empty(),
            "collector B lost updates after collector A marked sent — buffer was prematurely cleared"
        );

        // Verify the buffer still has the pending ops
        let tree_update_b = updates_b
            .iter()
            .find(|u| u.key.starts_with("tree:"))
            .expect("expected tree update for collector B");

        let policy_b: PolicyState = bincode::deserialize(&tree_update_b.value).unwrap();
        // It may be a delta (if pending buffer survived) or full state (fallback).
        // Either way, it must be present.
        assert!(
            policy_b.policy_type == "tree_state_delta" || policy_b.policy_type == "tree_state",
            "unexpected policy_type: {}",
            policy_b.policy_type
        );
    }

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
        // Simulate: add ops via sync_tree_operation, checkpoint (materializes
        // config in policy store), then clear tree_ops_pending (simulating
        // buffer trim), then create a new collector (simulating reconnected
        // peer).  The collector should produce a full PolicyState, not a delta.
        use crate::{incremental::IncrementalUpdateCollector, stores::StoreType};

        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let manager = MeshSyncManager::new(stores.clone(), "node1".to_string());

        for i in 0..10 {
            manager
                .sync_tree_operation(
                    "model1".to_string(),
                    make_insert_op(&format!("op_{i}"), "http://w:8000"),
                )
                .unwrap();
        }

        // Checkpoint to materialize into policy store (like production)
        manager.checkpoint_tree_states();

        // Clear tree_ops_pending to simulate buffer drain
        stores.tree_ops_pending.remove("tree:model1");

        // New collector (simulating reconnected peer)
        let collector = IncrementalUpdateCollector::new(stores.clone(), "node1".to_string());
        let updates = collector.collect_updates_for_store(StoreType::Policy);

        assert!(!updates.is_empty(), "expected at least one update");

        let tree_update = updates
            .iter()
            .find(|u| u.key.starts_with("tree:"))
            .expect("expected a tree key update");

        let policy_state: PolicyState =
            bincode::deserialize(&tree_update.value).expect("deserialize PolicyState");
        assert_eq!(
            policy_state.policy_type, "tree_state",
            "expected full state fallback when pending buffer is empty, got: {}",
            policy_state.policy_type
        );
    }

    #[test]
    fn test_delta_compaction_divergence() {
        // Add 2048 + 100 operations to trigger compaction.  Version must
        // remain monotonically increasing regardless of compaction.
        let manager = create_test_manager("node1".to_string());
        let total_ops = 2048 + 100;

        for i in 0..total_ops {
            manager
                .sync_tree_operation(
                    "model1".to_string(),
                    make_insert_op(&format!("op_{i}"), "http://w:8000"),
                )
                .unwrap();
        }

        let tree = manager.get_tree_state("model1").unwrap();
        // Version must equal total operations added (monotonic, not reset)
        assert_eq!(tree.version, total_ops as u64);
        // Operations should have been compacted — fewer than total_ops
        assert!(
            tree.operations.len() < total_ops,
            "expected compaction to reduce ops count, got {}",
            tree.operations.len()
        );

        // Checkpoint to materialize config before applying delta
        // (delta is rejected when config is empty but version > 0).
        manager.checkpoint_tree_states();

        // Apply a delta referencing a post-compaction version.
        // Even though many old operations were compacted away, the version
        // is still valid because it monotonically increases.
        let delta = make_delta(
            "model1",
            vec![make_insert_op("post_compaction", "http://w2:8000")],
            total_ops as u64 - 1,
            total_ops as u64 + 1,
        );

        // The local version is total_ops (the policy store version), and the
        // delta's new_version must exceed it for acceptance.
        // sync_tree_operation bumps the PolicyState.version each call, so
        // the policy version equals total_ops.
        let pre_apply_version = {
            let key = tree_state_key("model1");
            manager.stores.policy.get(&key).unwrap().version
        };
        assert_eq!(pre_apply_version, total_ops as u64);

        manager.apply_remote_tree_delta(delta, Some("remote".to_string()));

        let tree_after = manager.get_tree_state("model1").unwrap();
        // The delta should have been applied — version increased
        assert!(
            tree_after.version > total_ops as u64,
            "delta after compaction should have been accepted, version={}",
            tree_after.version
        );
    }

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
        // A processes 3 ops (version 8). B processes 2 ops (version 7).
        // A sends delta(base=5, new=8) to B.
        // B's current version is 7.
        //   base(5) <= current(7) ✓
        //   current(7) < new(8) ✓
        // So B accepts and applies the 3 ops.  The resulting tree has
        // version = 7 + 3 = 10 (B's state plus A's ops applied on top).
        let manager = create_test_manager("nodeB".to_string());

        // Seed the tree at version 5 (common ancestor)
        let mut seed = TreeState::new("model1".to_string());
        for i in 0..5 {
            seed.add_operation(make_insert_op(&format!("seed_{i}"), "http://w:8000"));
        }
        assert_eq!(seed.version, 5);
        manager.apply_remote_tree_operation("model1".to_string(), seed, Some("origin".to_string()));

        // B locally processes 2 more ops (version 5→7)
        for i in 0..2 {
            manager
                .sync_tree_operation(
                    "model1".to_string(),
                    make_insert_op(&format!("B_op_{i}"), "http://wB:8000"),
                )
                .unwrap();
        }
        let tree_b = manager.get_tree_state("model1").unwrap();
        // sync_tree_operation bumps the atomic tree_version counter (not
        // PolicyState.version). The seed was at version 5 (from remote),
        // then 2 local ops bumped tree_version to 2.  The materialized
        // tree version is 7 (seed 5 + 2 pending ops).
        assert_eq!(tree_b.version, 7);

        // A's delta: base=5, new=8, 3 ops
        let a_ops: Vec<_> = (0..3)
            .map(|i| make_insert_op(&format!("A_op_{i}"), "http://wA:8000"))
            .collect();
        let delta_a = make_delta("model1", a_ops, 5, 8);
        manager.apply_remote_tree_delta(delta_a, Some("nodeA".to_string()));

        // After apply, tree should have B's ops + A's ops (both sets).
        // Note: pending buffer may include ops already in the config,
        // so version and op count may be higher than the minimal merge.
        // The tree content is correct (duplicate inserts are idempotent).
        let tree_merged = manager.get_tree_state("model1").unwrap();
        assert!(
            tree_merged.version >= tree_b.version + 3,
            "version should be at least {} (B's version + A's 3 ops), got {}",
            tree_b.version + 3,
            tree_merged.version
        );
    }

    #[test]
    fn test_delta_buffer_trim_multi_peer() {
        // Create a pending buffer with >4096 ops.  Collector A sends and
        // marks_sent (which trims at the PENDING_TRIM_THRESHOLD of 4096).
        // Collector B should still get data — either delta from remaining
        // buffer or full state fallback.
        use crate::{incremental::IncrementalUpdateCollector, stores::StoreType};

        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let manager = MeshSyncManager::new(stores.clone(), "node1".to_string());

        let op_count = 5000;
        for i in 0..op_count {
            manager
                .sync_tree_operation(
                    "model1".to_string(),
                    make_insert_op(&format!("op_{i}"), "http://w:8000"),
                )
                .unwrap();
        }

        // Verify pending buffer has all ops
        let pending_before = stores
            .tree_ops_pending
            .get("tree:model1")
            .map(|v| v.len())
            .unwrap_or(0);
        assert_eq!(pending_before, op_count);

        // Collector A collects and marks sent (triggers trim)
        let collector_a = IncrementalUpdateCollector::new(stores.clone(), "node1".to_string());
        let updates_a = collector_a.collect_updates_for_store(StoreType::Policy);
        assert!(!updates_a.is_empty());
        collector_a.mark_sent(StoreType::Policy, &updates_a);

        // Pending buffer should have been trimmed (oldest half removed)
        let pending_after = stores
            .tree_ops_pending
            .get("tree:model1")
            .map(|v| v.len())
            .unwrap_or(0);
        assert!(
            pending_after < pending_before,
            "mark_sent should have trimmed the buffer: before={pending_before}, after={pending_after}",
        );

        // Collector B (new peer) should still get data
        let collector_b = IncrementalUpdateCollector::new(stores.clone(), "node1".to_string());
        let updates_b = collector_b.collect_updates_for_store(StoreType::Policy);
        assert!(
            !updates_b.is_empty(),
            "collector B should still get updates after A's trim"
        );

        let tree_update_b = updates_b
            .iter()
            .find(|u| u.key.starts_with("tree:"))
            .expect("expected tree update for collector B");
        let policy_b: PolicyState = bincode::deserialize(&tree_update_b.value).unwrap();
        // Should get a delta (from remaining buffer) or full state fallback
        assert!(
            policy_b.policy_type == "tree_state_delta" || policy_b.policy_type == "tree_state",
            "unexpected policy_type: {}",
            policy_b.policy_type
        );
    }

    #[test]
    fn test_delta_empty_pending_vec() {
        // Add an op, checkpoint (materializes config in policy store), then
        // replace pending with empty vec.  The collector should NOT produce
        // a delta — it should fall back to full state from the policy store.
        use crate::{incremental::IncrementalUpdateCollector, stores::StoreType};

        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let manager = MeshSyncManager::new(stores.clone(), "node1".to_string());

        // Add one op via normal path
        manager
            .sync_tree_operation(
                "model1".to_string(),
                make_insert_op("real_op", "http://w:8000"),
            )
            .unwrap();

        // Checkpoint to materialize into policy store
        manager.checkpoint_tree_states();

        // Overwrite pending buffer with empty vec
        stores
            .tree_ops_pending
            .insert("tree:model1".to_string(), Vec::new());

        let collector = IncrementalUpdateCollector::new(stores.clone(), "node1".to_string());
        let updates = collector.collect_updates_for_store(StoreType::Policy);

        assert!(!updates.is_empty(), "expected at least one update");

        let tree_update = updates
            .iter()
            .find(|u| u.key.starts_with("tree:"))
            .expect("expected a tree key update");

        let policy_state: PolicyState =
            bincode::deserialize(&tree_update.value).expect("deserialize PolicyState");
        // Empty pending vec should cause fallback to full state
        assert_eq!(
            policy_state.policy_type, "tree_state",
            "empty pending vec should fall back to full state, got: {}",
            policy_state.policy_type
        );
    }

    #[test]
    fn test_delta_concurrent_write_and_collect() {
        // Spawn a thread that adds 100 ops via sync_tree_operation.
        // Simultaneously run the collector.  The collector should get a
        // consistent snapshot — either some ops or all ops, but never
        // corrupted data.
        use crate::{incremental::IncrementalUpdateCollector, stores::StoreType};

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
            let collector = IncrementalUpdateCollector::new(stores.clone(), "node1".to_string());
            let updates = collector.collect_updates_for_store(StoreType::Policy);
            for update in &updates {
                if update.key.starts_with("tree:") {
                    // Verify the data deserializes without corruption
                    let policy_state: PolicyState =
                        bincode::deserialize(&update.value).expect("data should not be corrupted");
                    assert!(
                        policy_state.policy_type == "tree_state_delta"
                            || policy_state.policy_type == "tree_state"
                    );

                    if policy_state.policy_type == "tree_state_delta" {
                        let delta = TreeStateDelta::from_bytes(&policy_state.config)
                            .expect("delta should deserialize cleanly");
                        assert!(!delta.operations.is_empty());
                    } else {
                        let tree = TreeState::from_bytes(&policy_state.config)
                            .expect("tree state should deserialize cleanly");
                        assert!(!tree.operations.is_empty());
                    }
                    collected_any = true;
                }
            }
        }

        writer.join().unwrap();

        // After writer finishes, one final collect should succeed
        let collector = IncrementalUpdateCollector::new(stores.clone(), "node1".to_string());
        let final_updates = collector.collect_updates_for_store(StoreType::Policy);
        if !collected_any {
            // Writer may have been too fast; at least final collection must succeed
            assert!(
                !final_updates.is_empty(),
                "final collection after writer finished should have updates"
            );
        }
    }

    #[test]
    fn test_delta_oversized_mark_sent_trims_buffer() {
        // Verify that when the pending buffer exceeds PENDING_TRIM_THRESHOLD
        // (4096) and mark_sent is called, the buffer gets trimmed so that
        // subsequent collections can produce smaller batches.
        use crate::{
            incremental::IncrementalUpdateCollector, service::gossip::StateUpdate,
            stores::StoreType,
        };

        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let manager = MeshSyncManager::new(stores.clone(), "node1".to_string());

        // Add more ops than PENDING_TRIM_THRESHOLD (4096)
        let op_count = 4200;
        for i in 0..op_count {
            manager
                .sync_tree_operation(
                    "model1".to_string(),
                    make_insert_op(&format!("op_{i}"), "http://w:8000"),
                )
                .unwrap();
        }

        let pending_before = stores
            .tree_ops_pending
            .get("tree:model1")
            .map(|v| v.len())
            .unwrap_or(0);
        assert_eq!(pending_before, op_count);

        // Simulate mark_sent with a fake update for the tree key
        let collector = IncrementalUpdateCollector::new(stores.clone(), "node1".to_string());
        let fake_update = StateUpdate {
            key: "tree:model1".to_string(),
            value: vec![],
            version: op_count as u64,
            actor: "node1".to_string(),
            timestamp: 0,
        };
        collector.mark_sent(StoreType::Policy, &[fake_update]);

        let pending_after = stores
            .tree_ops_pending
            .get("tree:model1")
            .map(|v| v.len())
            .unwrap_or(0);

        // Buffer should have been trimmed: oldest half removed
        assert!(
            pending_after < pending_before,
            "buffer should be trimmed after mark_sent: before={pending_before}, after={pending_after}",
        );
        assert_eq!(
            pending_after,
            pending_before - pending_before / 2,
            "expected oldest half to be drained"
        );

        // Subsequent collection should succeed with the reduced buffer
        let collector2 = IncrementalUpdateCollector::new(stores.clone(), "node1".to_string());
        let updates = collector2.collect_updates_for_store(StoreType::Policy);
        assert!(
            !updates.is_empty(),
            "collection after trim should produce updates"
        );
    }

    #[test]
    fn test_delta_version_monotonic_after_compaction() {
        // Add 3000 operations (triggers compaction at MAX_TREE_OPERATIONS=2048).
        // Verify version is 3000 (not reset by compaction).  Then apply a
        // delta with base_version=2999 — should succeed.
        let manager = create_test_manager("node1".to_string());

        for i in 0..3000 {
            manager
                .sync_tree_operation(
                    "model1".to_string(),
                    make_insert_op(&format!("op_{i}"), "http://w:8000"),
                )
                .unwrap();
        }

        let tree = manager.get_tree_state("model1").unwrap();
        // TreeState.version should be 3000 — monotonic despite compaction
        assert_eq!(tree.version, 3000);
        // Operations compacted: fewer than 3000 stored
        assert!(tree.operations.len() < 3000);

        // Retrieve and re-store to verify persistence round-trip
        let bytes = tree.to_bytes().unwrap();
        let restored = TreeState::from_bytes(&bytes).unwrap();
        assert_eq!(restored.version, 3000);

        // Checkpoint to materialize config before applying delta
        manager.checkpoint_tree_states();

        // Apply delta with base_version=2999 (one less than current policy version)
        let key = tree_state_key("model1");
        let policy_version = manager.stores.policy.get(&key).unwrap().version;

        let delta = make_delta(
            "model1",
            vec![make_insert_op("post_compact_op", "http://w2:8000")],
            policy_version - 1,
            policy_version + 1,
        );
        manager.apply_remote_tree_delta(delta, Some("remote".to_string()));

        let tree_after = manager.get_tree_state("model1").unwrap();
        // Version may be higher than 3001 due to pending buffer replay
        assert!(
            tree_after.version >= 3001,
            "version should be at least 3001, got {}",
            tree_after.version
        );
    }

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
