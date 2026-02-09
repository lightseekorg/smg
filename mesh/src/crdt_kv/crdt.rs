use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

use super::{
    kv_store::KvStore,
    operation::{Operation, OperationLog},
    replica::{LamportClock, ReplicaId},
};

// ============================================================================
// CRDT OR-Map - Observed-Remove Map Implementation
// ============================================================================

/// Value metadata for CRDT OR-Map
#[derive(Debug, Clone, PartialEq, Eq)]
struct ValueMetadata {
    value: Vec<u8>,
    timestamp: u64,
    replica_id: ReplicaId,
    is_tombstone: bool, // Marks if this version is a tombstone (deletion)
}

impl ValueMetadata {
    fn new(value: Vec<u8>, timestamp: u64, replica_id: ReplicaId) -> Self {
        Self {
            value,
            timestamp,
            replica_id,
            is_tombstone: false,
        }
    }

    fn tombstone(timestamp: u64, replica_id: ReplicaId) -> Self {
        Self {
            value: Vec::new(),
            timestamp,
            replica_id,
            is_tombstone: true,
        }
    }
}

/// CRDT OR-Map
#[derive(Clone)]
pub struct CrdtOrMap {
    store: KvStore,
    metadata: Arc<DashMap<String, Vec<ValueMetadata>>>, // Key to list of versions
    replica_id: ReplicaId,
    clock: LamportClock,
    operation_log: Arc<RwLock<OperationLog>>,
}

impl CrdtOrMap {
    /// Create new CRDT OR-Map
    pub fn new() -> Self {
        Self::with_replica_id(ReplicaId::new())
    }

    /// Create new CRDT OR-Map with specified replica ID
    pub fn with_replica_id(replica_id: ReplicaId) -> Self {
        info!("Creating CRDT OR-Map, Replica ID: {}", replica_id);
        Self {
            store: KvStore::new(),
            metadata: Arc::new(DashMap::new()),
            replica_id,
            clock: LamportClock::new(),
            operation_log: Arc::new(RwLock::new(OperationLog::new())),
        }
    }

    /// Insert key-value pair (transparent operation)
    pub fn insert(&self, key: String, value: Vec<u8>) -> Option<Vec<u8>> {
        let prev = self.store.get(&key);
        let timestamp = self.clock.tick();
        let operation = Operation::insert(key.clone(), value.clone(), timestamp, self.replica_id);

        self.operation_log.write().append(operation.clone());

        debug!(
            "Insert: key={}, timestamp={}, replica={}",
            key, timestamp, self.replica_id
        );

        self.apply_insert(&key, value.clone(), timestamp, self.replica_id);

        prev
    }

    /// Atomically update a key under the same DashMap entry lock.
    pub fn upsert<F>(&self, key: String, updater: F) -> Vec<u8>
    where
        F: FnOnce(Option<&[u8]>) -> Vec<u8>,
    {
        let updated_value = self.store.upsert(key.clone(), updater);
        let timestamp = self.clock.tick();
        let operation = Operation::insert(
            key.clone(),
            updated_value.clone(),
            timestamp,
            self.replica_id,
        );

        self.operation_log.write().append(operation);
        let _ =
            self.record_insert_metadata(&key, updated_value.clone(), timestamp, self.replica_id);

        updated_value
    }

    /// Remove key (transparent operation)
    pub fn remove(&self, key: &str) -> Option<Vec<u8>> {
        let timestamp = self.clock.tick();
        let operation = Operation::remove(key.to_string(), timestamp, self.replica_id);

        self.operation_log.write().append(operation.clone());

        debug!(
            "Remove: key={}, timestamp={}, replica={}",
            key, timestamp, self.replica_id
        );

        self.apply_remove(key, timestamp, self.replica_id)
    }

    /// Get value by key
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.store.get(key)
    }

    /// Check if key exists
    pub fn contains_key(&self, key: &str) -> bool {
        self.store.contains_key(key)
    }

    /// Get all key-value pairs
    pub fn all(&self) -> std::collections::BTreeMap<String, Vec<u8>> {
        self.store.all()
    }

    /// Get the replica ID
    pub fn replica_id(&self) -> ReplicaId {
        self.replica_id
    }

    /// Get the operation log
    pub fn get_operation_log(&self) -> OperationLog {
        self.operation_log.read().clone()
    }

    /// Apply a single operation
    fn apply_operation(&self, operation: &Operation) {
        match operation {
            Operation::Insert {
                key,
                value,
                timestamp,
                replica_id,
            } => {
                self.clock.update(*timestamp);
                self.apply_insert(key, value.clone(), *timestamp, *replica_id);
            }
            Operation::Remove {
                key,
                timestamp,
                replica_id,
            } => {
                self.clock.update(*timestamp);
                let _ = self.apply_remove(key, *timestamp, *replica_id);
            }
        }
    }

    /// Merge operation log from another replica
    /// This is the core CRDT merge operation - state is derived from log
    pub fn merge(&self, log: &OperationLog) {
        info!(
            "Merging {} operations into replica {}",
            log.len(),
            self.replica_id
        );

        // Merge into local operation log
        self.operation_log.write().merge(log);

        // Apply operations to update state
        for operation in log.operations() {
            self.apply_operation(operation);
        }
    }

    /// Convenience method: merge from another replica instance
    /// In distributed systems, prefer using merge(&log) with serialized logs
    pub fn merge_replica(&self, other: &CrdtOrMap) {
        let other_log = other.get_operation_log();
        self.merge(&other_log);
    }

    // ========================================================================
    // Internal methods for applying operations
    // ========================================================================

    /// Apply insert (Add-wins semantic)
    fn apply_insert(&self, key: &str, value: Vec<u8>, timestamp: u64, replica_id: ReplicaId) {
        if self.record_insert_metadata(key, value.clone(), timestamp, replica_id) {
            self.store.insert(key.to_string(), value);
        }
    }

    fn record_insert_metadata(
        &self,
        key: &str,
        value: Vec<u8>,
        timestamp: u64,
        replica_id: ReplicaId,
    ) -> bool {
        let new_metadata = ValueMetadata::new(value, timestamp, replica_id);
        let mut should_store = false;

        self.metadata
            .entry(key.to_string())
            .and_modify(|versions| {
                let has_newer_remove = versions.iter().any(|v| {
                    v.is_tombstone
                        && (v.timestamp > timestamp
                            || (v.timestamp == timestamp && v.replica_id > replica_id))
                });

                if !has_newer_remove {
                    // Add-wins: only add if there's no newer remove
                    versions.push(new_metadata.clone());
                    should_store = true;
                } else {
                    warn!("Insert was overwritten by Remove: key={}", key);
                }
            })
            .or_insert_with(|| {
                should_store = true;
                vec![new_metadata]
            });

        should_store
    }

    /// Apply remove
    fn apply_remove(&self, key: &str, timestamp: u64, replica_id: ReplicaId) -> Option<Vec<u8>> {
        let tombstone = ValueMetadata::tombstone(timestamp, replica_id);
        let mut removed_value = None;

        self.metadata
            .entry(key.to_string())
            .and_modify(|versions| {
                versions.push(tombstone.clone());
                removed_value = self.store.remove(key);
            })
            .or_insert_with(|| vec![tombstone]);

        removed_value
    }
}

impl Default for CrdtOrMap {
    fn default() -> Self {
        Self::new()
    }
}
