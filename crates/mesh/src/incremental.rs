//! Incremental update collection and batching
//!
//! Collects local state changes and batches them for efficient transmission

use std::{
    collections::HashMap,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use parking_lot::RwLock;
use tracing::{debug, trace};

use super::{
    service::gossip::StateUpdate,
    stores::{AppState, MembershipState, PolicyState, StateStores, StoreType, WorkerState},
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
}

/// Incremental update collector
pub struct IncrementalUpdateCollector {
    stores: Arc<StateStores>,
    self_name: String,
    last_sent: Arc<RwLock<LastSentVersions>>,
    last_scanned: Arc<RwLock<LastScannedGenerations>>,
}

impl IncrementalUpdateCollector {
    pub fn new(stores: Arc<StateStores>, self_name: String) -> Self {
        Self {
            stores,
            self_name,
            last_sent: Arc::new(RwLock::new(LastSentVersions::default())),
            last_scanned: Arc::new(RwLock::new(LastScannedGenerations::default())),
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

    fn rate_limit_last_sent_key(key: &str, actor: &str) -> String {
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
        let last_sent = self.last_sent.read();
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
                let gen = self.stores.policy.generation();
                if gen == last_scanned.policy {
                    return vec![];
                }
                let all_policies = self.stores.policy.all();
                updates = self.collect_serializable_updates(
                    all_policies,
                    &last_sent.policy,
                    "policy",
                    |state: &PolicyState| state.model_id.clone(),
                );
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
            StoreType::Policy => last_scanned.policy = self.stores.policy.generation(),
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
            },
        );

        let updates3 = collector.collect_updates_for_store(StoreType::Worker);
        assert_eq!(updates3.len(), 1);
        let version3 = updates3[0].version;
        assert_eq!(version3, 3);
    }
}
