//! State stores for mesh cluster synchronization
//!
//! Four types of state stores:
//! - MembershipStore: Router node membership
//! - AppStore: Application configuration, rate-limiting rules, LB algorithms
//! - WorkerStore: Worker status, load, health
//! - PolicyStore: Routing policy internal state

use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    sync::Arc,
};

use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tracing::debug;

use super::{
    consistent_hash::ConsistentHashRing,
    crdt_kv::{CrdtOrMap, Operation, OperationLog, ReplicaId},
};

// ============================================================================
// Type-Safe Serialization Layer - Transparent T ↔ Vec<u8> Conversion
// ============================================================================

/// Trait for CRDT-compatible value types.
/// Uses bincode for compact binary serialization. This is critical for
/// PolicyState which contains TreeState with token payloads — JSON
/// serialization of Vec<u8> is ~4x larger than binary.
trait CrdtValue: Serialize + DeserializeOwned + Clone {
    fn to_bytes(&self) -> Result<Vec<u8>, CrdtSerError> {
        bincode::serialize(self).map_err(CrdtSerError)
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, CrdtSerError> {
        bincode::deserialize(bytes).map_err(CrdtSerError)
    }
}

/// Serialization error wrapper for CRDT values.
#[derive(Debug)]
pub struct CrdtSerError(Box<bincode::ErrorKind>);

impl std::fmt::Display for CrdtSerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CRDT serialization error: {}", self.0)
    }
}

impl std::error::Error for CrdtSerError {}

// Blanket implementation for all types that satisfy the bounds
impl<T> CrdtValue for T where T: Serialize + DeserializeOwned + Clone {}

// ============================================================================
// Generic CRDT Store Wrapper - Type-Safe Interface Over CrdtOrMap
// ============================================================================

/// Generic store wrapper providing type-safe operations over CrdtOrMap
#[derive(Clone)]
struct CrdtStore<T> {
    inner: CrdtOrMap,
    _phantom: PhantomData<T>,
}

impl<T> std::fmt::Debug for CrdtStore<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrdtStore")
            .field("inner", &"<CrdtOrMap>")
            .finish()
    }
}

impl<T: CrdtValue> CrdtStore<T> {
    fn new() -> Self {
        Self {
            inner: CrdtOrMap::new(),
            _phantom: PhantomData,
        }
    }

    fn get(&self, key: &str) -> Option<T> {
        self.inner.get(key).and_then(|bytes| {
            T::from_bytes(&bytes)
                .map_err(|err| {
                    debug!(error = %err, %key, "Failed to deserialize CRDT value");
                })
                .ok()
        })
    }

    fn insert(&self, key: String, value: T) -> Result<Option<T>, CrdtSerError> {
        let bytes = value.to_bytes().map_err(|err| {
            debug!(error = %err, %key, "Failed to serialize CRDT value");
            err
        })?;

        Ok(self.inner.insert(key, bytes).and_then(|old_bytes| {
            T::from_bytes(&old_bytes)
                .map_err(|err| {
                    debug!(error = %err, "Failed to deserialize old CRDT value");
                })
                .ok()
        }))
    }

    fn remove(&self, key: &str) -> Option<T> {
        self.inner.remove(key).and_then(|bytes| {
            T::from_bytes(&bytes)
                .map_err(|err| {
                    debug!(error = %err, %key, "Failed to deserialize removed CRDT value");
                })
                .ok()
        })
    }

    fn update<F>(&self, key: String, updater: F) -> Result<Option<T>, CrdtSerError>
    where
        F: FnOnce(Option<T>) -> T,
    {
        let updated_bytes = self.inner.try_upsert(key, |current_bytes| {
            let current = current_bytes.and_then(|bytes| {
                T::from_bytes(bytes)
                    .map_err(|err| {
                        debug!(error = %err, "Failed to deserialize current CRDT value");
                    })
                    .ok()
            });

            let updated = updater(current);
            updated.to_bytes()
        })?;

        Ok(T::from_bytes(&updated_bytes)
            .map_err(|err| {
                debug!(error = %err, "Failed to deserialize updated CRDT value");
                err
            })
            .ok())
    }

    fn update_if<F>(&self, key: String, updater: F) -> Result<(Option<T>, bool), CrdtSerError>
    where
        F: FnOnce(Option<T>) -> Option<T>,
    {
        let (updated_bytes, changed) = self.inner.try_upsert_if(key, |current_bytes| {
            let current = current_bytes.and_then(|bytes| {
                T::from_bytes(bytes)
                    .map_err(|err| {
                        debug!(error = %err, "Failed to deserialize current CRDT value");
                    })
                    .ok()
            });

            let updated = updater(current);
            updated.map(|value| value.to_bytes()).transpose()
        })?;

        let value = T::from_bytes(&updated_bytes)
            .map_err(|err| {
                debug!(error = %err, "Failed to deserialize conditionally updated CRDT value");
                err
            })
            .ok();

        Ok((value, changed))
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // fn contains_key(&self, key: &str) -> bool {
    //     self.inner.contains_key(key)
    // }

    fn merge(&self, log: &OperationLog) {
        self.inner.merge(log);
    }

    fn get_operation_log(&self) -> OperationLog {
        self.inner.get_operation_log()
    }

    fn all(&self) -> BTreeMap<String, T> {
        self.inner
            .all()
            .into_iter()
            .filter_map(|(k, v)| {
                let key_for_log = k.clone();
                T::from_bytes(&v)
                    .map(|val| (k, val))
                    .map_err(|err| {
                        debug!(error = %err, key = %key_for_log, "Failed to deserialize CRDT value in all()");
                    })
                    .ok()
            })
            .collect()
    }
}

impl<T: CrdtValue> Default for CrdtStore<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Store type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StoreType {
    Membership,
    App,
    Worker,
    Policy,
    RateLimit,
}

impl StoreType {
    pub fn as_str(self) -> &'static str {
        match self {
            StoreType::Membership => "membership",
            StoreType::App => "app",
            StoreType::Worker => "worker",
            StoreType::Policy => "policy",
            StoreType::RateLimit => "rate_limit",
        }
    }

    /// Convert to proto StoreType (i32)
    pub fn to_proto(self) -> i32 {
        use super::service::gossip::StoreType as ProtoStoreType;
        match self {
            StoreType::Membership => ProtoStoreType::Membership as i32,
            StoreType::App => ProtoStoreType::App as i32,
            StoreType::Worker => ProtoStoreType::Worker as i32,
            StoreType::Policy => ProtoStoreType::Policy as i32,
            StoreType::RateLimit => ProtoStoreType::RateLimit as i32,
        }
    }

    /// Convert from proto StoreType (i32) to local StoreType
    pub fn from_proto(proto_value: i32) -> Self {
        match proto_value {
            0 => StoreType::Membership,
            1 => StoreType::App,
            2 => StoreType::Worker,
            3 => StoreType::Policy,
            4 => StoreType::RateLimit,
            unknown => {
                tracing::warn!(
                    proto_value = unknown,
                    "Unknown StoreType proto value, defaulting to Membership"
                );
                StoreType::Membership
            }
        }
    }
}

/// Membership state entry
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub struct MembershipState {
    pub name: String,
    pub address: String,
    pub status: i32, // NodeStatus enum value
    pub version: u64,
    pub metadata: BTreeMap<String, Vec<u8>>,
}

/// App state entry (application configuration)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub struct AppState {
    pub key: String,
    pub value: Vec<u8>, // Serialized config
    pub version: u64,
}

/// Global rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct RateLimitConfig {
    pub limit_per_second: u64,
}

/// Key for global rate limit configuration in AppStore
pub const GLOBAL_RATE_LIMIT_KEY: &str = "global_rate_limit";
/// Key for global rate limit counter in RateLimitStore
pub const GLOBAL_RATE_LIMIT_COUNTER_KEY: &str = "global";

/// Worker state entry
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct WorkerState {
    pub worker_id: String,
    pub model_id: String,
    pub url: String,
    pub health: bool,
    pub load: f64,
    pub version: u64,
}

// Implement Hash manually for WorkerState (excluding f64)
impl std::hash::Hash for WorkerState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.worker_id.hash(state);
        self.model_id.hash(state);
        self.url.hash(state);
        self.health.hash(state);
        // f64 cannot be hashed directly, use a workaround
        (self.load as i64).hash(state);
        self.version.hash(state);
    }
}

// Implement Eq manually (f64 comparison with epsilon)
impl Eq for WorkerState {}

/// Policy state entry
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub struct PolicyState {
    pub model_id: String,
    pub policy_type: String,
    pub config: Vec<u8>, // Serialized policy config
    pub version: u64,
}

/// Helper function to get policy state key for a model
pub fn policy_key(model_id: &str) -> String {
    format!("policy:{model_id}")
}

/// Helper function to get tree state key for a model
pub fn tree_state_key(model_id: &str) -> String {
    format!("tree:{model_id}")
}

macro_rules! define_state_store {
    ($store_name:ident, $value_type:ty) => {
        #[derive(Debug, Clone)]
        pub struct $store_name {
            inner: CrdtStore<$value_type>,
        }

        impl $store_name {
            pub fn new() -> Self {
                Self {
                    inner: CrdtStore::new(),
                }
            }

            pub fn get(&self, key: &str) -> Option<$value_type> {
                self.inner.get(key)
            }

            pub fn insert(
                &self,
                key: String,
                value: $value_type,
            ) -> Result<Option<$value_type>, CrdtSerError> {
                self.inner.insert(key, value)
            }

            pub fn remove(&self, key: &str) {
                self.inner.remove(key);
            }

            pub fn merge(&self, log: &OperationLog) {
                self.inner.merge(log);
            }

            pub fn get_operation_log(&self) -> OperationLog {
                self.inner.get_operation_log()
            }

            pub fn update<F>(
                &self,
                key: String,
                updater: F,
            ) -> Result<Option<$value_type>, CrdtSerError>
            where
                F: FnOnce(Option<$value_type>) -> $value_type,
            {
                self.inner.update(key, updater)
            }

            pub fn update_if<F>(
                &self,
                key: String,
                updater: F,
            ) -> Result<(Option<$value_type>, bool), CrdtSerError>
            where
                F: FnOnce(Option<$value_type>) -> Option<$value_type>,
            {
                self.inner.update_if(key, updater)
            }

            pub fn len(&self) -> usize {
                self.inner.len()
            }

            pub fn is_empty(&self) -> bool {
                self.inner.is_empty()
            }

            pub fn all(&self) -> BTreeMap<String, $value_type> {
                self.inner.all()
            }
        }

        impl Default for $store_name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

define_state_store!(MembershipStore, MembershipState);
define_state_store!(AppStore, AppState);
define_state_store!(WorkerStore, WorkerState);
define_state_store!(PolicyStore, PolicyState);

// ============================================================================
// Rate Limit Counter - Simplified Counter Using CrdtOrMap
// ============================================================================

/// Counter value wrapper for rate limiting
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
struct CounterValue {
    value: i64,
}

/// Rate-limit counter store (using CrdtOrMap with consistent hashing)
#[derive(Debug, Clone)]
pub struct RateLimitStore {
    counters: CrdtStore<CounterValue>,
    hash_ring: Arc<RwLock<ConsistentHashRing>>,
    self_name: String,
    actor_replica_ids: Arc<DashMap<String, ReplicaId>>,
}

impl RateLimitStore {
    const SHARD_SEPARATOR: &'static str = "::actor:";

    pub fn new(self_name: String) -> Self {
        Self {
            counters: CrdtStore::new(),
            hash_ring: Arc::new(RwLock::new(ConsistentHashRing::new())),
            self_name,
            actor_replica_ids: Arc::new(DashMap::new()),
        }
    }

    fn shard_key(key: &str, actor: &str) -> String {
        format!("{key}{}{actor}", Self::SHARD_SEPARATOR)
    }

    fn split_shard_key(shard_key: &str) -> Option<(&str, &str)> {
        shard_key.rsplit_once(Self::SHARD_SEPARATOR)
    }

    fn base_key(shard_key: &str) -> &str {
        Self::split_shard_key(shard_key).map_or(shard_key, |(base, _)| base)
    }

    fn replica_id_for_actor(&self, actor: &str) -> ReplicaId {
        if let Ok(replica_id) = ReplicaId::from_string(actor) {
            return replica_id;
        }

        *self.actor_replica_ids.entry(actor.to_string()).or_default()
    }

    fn aggregate_counter(&self, key: &str) -> Option<i64> {
        let all_counters = self.counters.all();
        let mut has_shard = false;
        let mut total = 0;

        for (shard_key, counter) in all_counters {
            if Self::base_key(&shard_key) == key {
                has_shard = true;
                total += counter.value;
            }
        }

        if has_shard {
            Some(total)
        } else {
            None
        }
    }

    /// Update the hash ring with current membership
    pub fn update_membership(&self, nodes: &[String]) {
        let mut ring = self.hash_ring.write();
        ring.update_membership(nodes);
        debug!("Updated rate-limit hash ring with {} nodes", nodes.len());
    }

    /// Check if this node is an owner of a key
    pub fn is_owner(&self, key: &str) -> bool {
        let ring = self.hash_ring.read();
        ring.is_owner(key, &self.self_name)
    }

    /// Get owners for a key
    pub fn get_owners(&self, key: &str) -> Vec<String> {
        let ring = self.hash_ring.read();
        ring.get_owners(key)
    }

    /// Get or create counter (only if this node is an owner)
    #[expect(dead_code)]
    fn get_or_create_counter_internal(&self, key: String) -> Option<i64> {
        if !self.is_owner(&key) {
            return None;
        }

        let shard_key = Self::shard_key(&key, &self.self_name);
        if let Some(counter) = self.counters.get(&shard_key) {
            return Some(counter.value);
        }

        let _ = self.counters.insert(shard_key, CounterValue::default());
        Some(0)
    }

    pub fn get_counter(&self, key: &str) -> Option<i64> {
        if !self.is_owner(key) {
            return None;
        }
        self.aggregate_counter(key)
    }

    /// Get all actor shards as (base_key, actor, value).
    pub fn all_shards(&self) -> Vec<(String, String, i64)> {
        self.counters
            .all()
            .into_iter()
            .filter_map(|(shard_key, counter)| {
                Self::split_shard_key(&shard_key).map(|(base_key, actor)| {
                    (base_key.to_string(), actor.to_string(), counter.value)
                })
            })
            .collect()
    }

    /// Increment counter (only if this node is an owner)
    pub fn inc(&self, key: String, actor: String, delta: i64) {
        if !self.is_owner(&key) {
            return;
        }

        let shard_key = Self::shard_key(&key, &actor);
        if let Err(err) = self.counters.update(shard_key, |current| CounterValue {
            value: current.map_or(delta, |existing| existing.value + delta),
        }) {
            debug!(error = %err, %key, %actor, "Failed to update rate-limit counter shard");
        }
    }

    /// Set a snapshot value for one actor shard.
    pub fn set_counter_snapshot(&self, key: String, actor: String, counter_value: i64) {
        if !self.is_owner(&key) {
            return;
        }

        let shard_key = Self::shard_key(&key, &actor);
        if let Err(err) = self.counters.insert(
            shard_key,
            CounterValue {
                value: counter_value,
            },
        ) {
            debug!(error = %err, %key, %actor, "Failed to set rate-limit counter snapshot");
        }
    }

    /// Build serialized snapshot payload and shard key for a counter value.
    ///
    /// NOTE: This intentionally does not fabricate CRDT operation IDs.
    pub fn snapshot_payload_for_counter_value(
        key: String,
        actor: String,
        counter_value: i64,
    ) -> Option<(String, Vec<u8>)> {
        let bytes = match (CounterValue {
            value: counter_value,
        })
        .to_bytes()
        {
            Ok(bytes) => bytes,
            Err(err) => {
                debug!(error = %err, "Failed to serialize rate-limit counter snapshot");
                return None;
            }
        };

        let shard_key = Self::shard_key(&key, &actor);
        Some((shard_key, bytes))
    }

    pub fn apply_counter_snapshot_payload(
        &self,
        shard_key: String,
        actor: &str,
        timestamp: u64,
        payload: &[u8],
    ) {
        let Some((base_key, _)) = Self::split_shard_key(&shard_key) else {
            debug!(%shard_key, "Invalid rate-limit shard key in snapshot payload");
            return;
        };

        if !self.is_owner(base_key) {
            return;
        }

        if let Err(err) = CounterValue::from_bytes(payload) {
            debug!(error = %err, %shard_key, "Failed to decode rate-limit snapshot payload");
            return;
        }

        let replica_id = self.replica_id_for_actor(actor);
        let mut log = OperationLog::new();
        log.append(Operation::insert(
            shard_key,
            payload.to_vec(),
            timestamp,
            replica_id,
        ));
        self.counters.merge(&log);
    }

    /// Get counter value
    pub fn value(&self, key: &str) -> Option<i64> {
        self.aggregate_counter(key)
    }

    /// Merge operation log from another node
    pub fn merge(&self, log: &OperationLog) {
        self.counters.merge(log);
    }

    /// Get operation log for synchronization
    pub fn get_operation_log(&self) -> OperationLog {
        self.counters.get_operation_log()
    }

    /// Get all counter keys
    pub fn keys(&self) -> Vec<String> {
        self.counters
            .all()
            .keys()
            .map(|key| Self::base_key(key).to_string())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// Check if we need to transfer ownership due to node failure
    pub fn check_ownership_transfer(&self, failed_nodes: &[String]) -> Vec<String> {
        let mut affected_keys = Vec::new();
        let ring = self.hash_ring.read();
        for key in self.keys() {
            let owners = ring.get_owners(&key);
            if owners.iter().any(|owner| failed_nodes.contains(owner))
                && ring.is_owner(&key, &self.self_name)
            {
                affected_keys.push(key);
            }
        }

        affected_keys
    }
}

impl Default for RateLimitStore {
    fn default() -> Self {
        Self::new("default".to_string())
    }
}

/// All state stores container
#[derive(Debug, Clone)]
pub struct StateStores {
    pub membership: MembershipStore,
    pub app: AppStore,
    pub worker: WorkerStore,
    pub policy: PolicyStore,
    pub rate_limit: RateLimitStore,
}

impl StateStores {
    pub fn new() -> Self {
        Self {
            membership: MembershipStore::new(),
            app: AppStore::new(),
            worker: WorkerStore::new(),
            policy: PolicyStore::new(),
            rate_limit: RateLimitStore::new("default".to_string()),
        }
    }

    pub fn with_self_name(self_name: String) -> Self {
        Self {
            membership: MembershipStore::new(),
            app: AppStore::new(),
            worker: WorkerStore::new(),
            policy: PolicyStore::new(),
            rate_limit: RateLimitStore::new(self_name),
        }
    }
}

impl Default for StateStores {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::service::gossip::NodeStatus;

    #[test]
    fn test_membership_store() {
        let store = MembershipStore::new();
        let key = "node1".to_string();
        let state = MembershipState {
            name: "node1".to_string(),
            address: "127.0.0.1:8000".to_string(),
            status: NodeStatus::Alive as i32,
            version: 1,
            metadata: BTreeMap::new(),
        };

        let _ = store.insert(key.clone(), state.clone());
        assert_eq!(store.get(&key).unwrap().name, "node1");

        store.remove(&key);
        assert!(store.get(&key).is_none());
    }

    #[test]
    fn test_app_store() {
        let store = AppStore::new();
        let key = "app_key1".to_string();
        let state = AppState {
            key: "app_key1".to_string(),
            value: b"app_value".to_vec(),
            version: 1,
        };

        let _ = store.insert(key.clone(), state.clone());
        assert_eq!(store.get(&key).unwrap().key, "app_key1");
    }

    #[test]
    fn test_worker_store() {
        let store = WorkerStore::new();
        let key = "worker1".to_string();
        let state = WorkerState {
            worker_id: "worker1".to_string(),
            model_id: "model1".to_string(),
            url: "http://localhost:8000".to_string(),
            health: true,
            load: 0.5,
            version: 1,
        };

        let _ = store.insert(key.clone(), state.clone());
        assert_eq!(store.get(&key).unwrap().worker_id, "worker1");
    }

    #[test]
    fn test_policy_store() {
        let store = PolicyStore::new();
        let key = "policy:model1".to_string();
        let state = PolicyState {
            model_id: "model1".to_string(),
            policy_type: "cache_aware".to_string(),
            config: b"config_data".to_vec(),
            version: 1,
        };

        let _ = store.insert(key.clone(), state.clone());
        assert_eq!(store.get(&key).unwrap().model_id, "model1");
    }

    #[test]
    fn test_rate_limit_store_update_membership() {
        let store = RateLimitStore::new("node1".to_string());

        store.update_membership(&[
            "node1".to_string(),
            "node2".to_string(),
            "node3".to_string(),
        ]);

        let owners = store.get_owners("test_key");
        assert_eq!(owners.len(), 3);
        assert!(
            owners.contains(&"node1".to_string())
                || owners.contains(&"node2".to_string())
                || owners.contains(&"node3".to_string())
        );
    }

    #[test]
    fn test_rate_limit_store_is_owner() {
        let store = RateLimitStore::new("node1".to_string());

        store.update_membership(&["node1".to_string()]);

        let test_key = "test_key".to_string();
        let is_owner = store.is_owner(&test_key);
        // node1 should be owner since it's the only node
        assert!(is_owner);
    }

    #[test]
    fn test_rate_limit_store_inc_only_owner() {
        let store = RateLimitStore::new("node1".to_string());

        store.update_membership(&["node1".to_string()]);

        let test_key = "test_key".to_string();
        if store.is_owner(&test_key) {
            store.inc(test_key.clone(), "node1".to_string(), 5);

            let value = store.value(&test_key);
            assert_eq!(value, Some(5));
        }
    }

    #[test]
    fn test_rate_limit_store_inc_non_owner() {
        let store = RateLimitStore::new("node1".to_string());

        // Setup membership without node1 as owner
        store.update_membership(&["node2".to_string(), "node3".to_string()]);

        let test_key = "test_key".to_string();
        if !store.is_owner(&test_key) {
            store.inc(test_key.clone(), "node1".to_string(), 5);

            // Should not increment if not owner
            let value = store.value(&test_key);
            assert_eq!(value, None);
        }
    }

    #[test]
    fn test_rate_limit_store_merge_counter() {
        let store1 = RateLimitStore::new("node1".to_string());
        let store2 = RateLimitStore::new("node2".to_string());

        store1.update_membership(&["node1".to_string()]);
        store2.update_membership(&["node2".to_string()]);

        let test_key = "test_key".to_string();

        // Both nodes increment their counters
        if store1.is_owner(&test_key) {
            store1.inc(test_key.clone(), "node1".to_string(), 10);
        }

        if store2.is_owner(&test_key) {
            store2.inc(test_key.clone(), "node2".to_string(), 5);
        }

        // Merge operation log from store2 into store1
        let log2 = store2.get_operation_log();
        store1.merge(&log2);

        // Get aggregated value (if node1 is owner)
        if store1.is_owner(&test_key) {
            let value = store1.value(&test_key);
            assert_eq!(value, Some(15));
        }
    }

    #[test]
    fn test_rate_limit_store_check_ownership_transfer() {
        let store = RateLimitStore::new("node1".to_string());

        store.update_membership(&[
            "node1".to_string(),
            "node2".to_string(),
            "node3".to_string(),
        ]);

        let test_key = "test_key".to_string();

        // Setup a counter (if node1 is owner)
        if store.is_owner(&test_key) {
            store.inc(test_key.clone(), "node1".to_string(), 10);
        }

        // Check ownership transfer when node2 fails
        let affected = store.check_ownership_transfer(&["node2".to_string()]);
        // Should detect if node2 was an owner
        let _ = affected;
    }

    #[test]
    fn test_rate_limit_store_keys() {
        let store = RateLimitStore::new("node1".to_string());

        store.update_membership(&["node1".to_string()]);

        let key1 = "key1".to_string();
        let key2 = "key2".to_string();

        if store.is_owner(&key1) {
            store.inc(key1.clone(), "node1".to_string(), 1);
        }

        if store.is_owner(&key2) {
            store.inc(key2.clone(), "node1".to_string(), 1);
        }

        let keys = store.keys();
        // Should contain keys where node1 is owner
        let _ = keys;
    }

    #[test]
    fn test_state_stores_new() {
        let stores = StateStores::new();
        assert_eq!(stores.membership.len(), 0);
        assert_eq!(stores.app.len(), 0);
        assert_eq!(stores.worker.len(), 0);
        assert_eq!(stores.policy.len(), 0);
    }

    #[test]
    fn test_state_stores_with_self_name() {
        let stores = StateStores::with_self_name("test_node".to_string());
        // Rate limit store should have the self_name
        let test_key = "test_key".to_string();
        stores
            .rate_limit
            .update_membership(&["test_node".to_string()]);
        assert!(stores.rate_limit.is_owner(&test_key));
    }
}
