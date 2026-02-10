use std::sync::Arc;

use dashmap::{mapref::entry::Entry as MapEntry, DashMap};
use parking_lot::{Mutex, RwLock};
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

    fn version_key(&self) -> (u64, ReplicaId) {
        (self.timestamp, self.replica_id)
    }

    fn matches_version(&self, timestamp: u64, replica_id: ReplicaId) -> bool {
        self.timestamp == timestamp && self.replica_id == replica_id
    }

    fn is_newer_than(&self, timestamp: u64, replica_id: ReplicaId) -> bool {
        self.version_key() > (timestamp, replica_id)
    }
}

/// CRDT OR-Map
#[derive(Clone)]
pub struct CrdtOrMap {
    store: KvStore,
    metadata: Arc<DashMap<String, Vec<ValueMetadata>>>, // Key to list of versions
    key_locks: Arc<DashMap<String, Arc<Mutex<()>>>>,    // Per-key critical section lock
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
            key_locks: Arc::new(DashMap::new()),
            replica_id,
            clock: LamportClock::new(),
            operation_log: Arc::new(RwLock::new(OperationLog::new())),
        }
    }

    fn key_lock_for(&self, key: &str) -> Arc<Mutex<()>> {
        self.key_locks
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    /// Insert key-value pair (transparent operation)
    pub fn insert(&self, key: String, value: Vec<u8>) -> Option<Vec<u8>> {
        let key_lock = self.key_lock_for(&key);
        let _key_guard = key_lock.lock();

        let timestamp = self.clock.tick();
        if !self.record_insert_metadata(&key, value.clone(), timestamp, self.replica_id) {
            return self.store.get(&key).map(|bytes| bytes.to_vec());
        }

        let mut prev = None;
        let value_for_store = value.clone();
        let _ = self.store.upsert(key.clone(), |current| {
            prev = current.map(|bytes| bytes.to_vec());
            value_for_store
        });

        let operation = Operation::insert(key.clone(), value, timestamp, self.replica_id);
        self.operation_log.write().append(operation);

        debug!(
            "Insert: key={}, timestamp={}, replica={}",
            key, timestamp, self.replica_id
        );

        prev
    }

    /// Update a key using the current store value and CRDT insert semantics.
    pub fn upsert<F>(&self, key: String, updater: F) -> Vec<u8>
    where
        F: FnOnce(Option<&[u8]>) -> Vec<u8>,
    {
        let key_lock = self.key_lock_for(&key);
        let _key_guard = key_lock.lock();

        let current_value = self.store.get(&key);
        let updated_value = updater(current_value.as_deref());
        let timestamp = self.clock.tick();

        if !self.record_insert_metadata(&key, updated_value.clone(), timestamp, self.replica_id) {
            return self.store.get(&key).unwrap_or_default();
        }

        let operation = Operation::insert(
            key.clone(),
            updated_value.clone(),
            timestamp,
            self.replica_id,
        );

        self.store.insert(key, updated_value.clone());
        self.operation_log.write().append(operation);

        updated_value
    }

    /// Remove key (transparent operation)
    pub fn remove(&self, key: &str) -> Option<Vec<u8>> {
        let key_lock = self.key_lock_for(key);
        let _key_guard = key_lock.lock();

        let timestamp = self.clock.tick();
        let operation = Operation::remove(key.to_string(), timestamp, self.replica_id);

        debug!(
            "Remove: key={}, timestamp={}, replica={}",
            key, timestamp, self.replica_id
        );

        let removed = if self.record_remove_metadata(key, timestamp, self.replica_id) {
            self.store.remove(key)
        } else {
            None
        };

        self.operation_log.write().append(operation);

        removed
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
        let key_lock = self.key_lock_for(key);
        let _key_guard = key_lock.lock();

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

        match self.metadata.entry(key.to_string()) {
            MapEntry::Occupied(mut entry) => {
                let versions = entry.get_mut();

                let has_existing_entry = versions
                    .iter()
                    .any(|v| !v.is_tombstone && v.matches_version(timestamp, replica_id));
                if has_existing_entry {
                    return false;
                }

                let current_winner = versions
                    .iter()
                    .filter(|v| !v.is_tombstone)
                    .max_by_key(|v| v.version_key());

                if current_winner.is_some_and(|winner| winner.is_newer_than(timestamp, replica_id))
                {
                    return false;
                }

                let has_newer_remove = versions
                    .iter()
                    .any(|v| v.is_tombstone && v.is_newer_than(timestamp, replica_id));
                if has_newer_remove {
                    warn!("Insert was overwritten by Remove: key={}", key);
                    return false;
                }

                // Add-wins: only add if there is no newer remove.
                versions.push(new_metadata);
                true
            }
            MapEntry::Vacant(entry) => {
                entry.insert(vec![new_metadata]);
                true
            }
        }
    }

    /// Apply remove
    fn apply_remove(&self, key: &str, timestamp: u64, replica_id: ReplicaId) -> Option<Vec<u8>> {
        let key_lock = self.key_lock_for(key);
        let _key_guard = key_lock.lock();

        if self.record_remove_metadata(key, timestamp, replica_id) {
            return self.store.remove(key);
        }

        None
    }

    fn record_remove_metadata(&self, key: &str, timestamp: u64, replica_id: ReplicaId) -> bool {
        let tombstone = ValueMetadata::tombstone(timestamp, replica_id);

        match self.metadata.entry(key.to_string()) {
            MapEntry::Occupied(mut entry) => {
                let versions = entry.get_mut();
                let has_existing_entry = versions
                    .iter()
                    .any(|v| v.is_tombstone && v.matches_version(timestamp, replica_id));
                if has_existing_entry {
                    return false;
                }

                let has_newer_insert = versions
                    .iter()
                    .any(|v| !v.is_tombstone && v.is_newer_than(timestamp, replica_id));
                if has_newer_insert {
                    return false;
                }

                versions.push(tombstone);
                true
            }
            MapEntry::Vacant(entry) => {
                entry.insert(vec![tombstone]);
                true
            }
        }
    }
}

impl Default for CrdtOrMap {
    fn default() -> Self {
        Self::new()
    }
}
