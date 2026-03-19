use std::{collections::HashSet, sync::Arc};

use dashmap::{mapref::entry::Entry as MapEntry, DashMap};
use parking_lot::{Mutex, RwLock};
use tracing::{debug, info};

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
    timestamp: u64,
    replica_id: ReplicaId,
    is_tombstone: bool, // Marks if this version is a tombstone (deletion)
}

impl ValueMetadata {
    fn new(timestamp: u64, replica_id: ReplicaId) -> Self {
        Self {
            timestamp,
            replica_id,
            is_tombstone: false,
        }
    }

    fn tombstone(timestamp: u64, replica_id: ReplicaId) -> Self {
        Self {
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

    fn key_is_tombstoned_or_unknown(&self, key: &str) -> bool {
        self.metadata.get(key).is_none_or(|versions| {
            versions
                .iter()
                .max_by_key(|version| version.version_key())
                .is_none_or(|winner| winner.is_tombstone)
        })
    }

    fn try_cleanup_key_lock(&self, key: &str, key_lock: &Arc<Mutex<()>>) {
        if self.store.contains_key(key) || !self.key_is_tombstoned_or_unknown(key) {
            return;
        }

        let _ = self.key_locks.remove_if(key, |_, stored_lock| {
            Arc::ptr_eq(stored_lock, key_lock)
                && Arc::strong_count(stored_lock) <= 2
                && stored_lock.try_lock().is_some()
        });
    }

    /// Insert key-value pair (transparent operation)
    pub fn insert(&self, key: String, value: Vec<u8>) -> Option<Vec<u8>> {
        let key_lock = self.key_lock_for(&key);
        let key_guard = key_lock.lock();

        let timestamp = self.clock.tick();
        let result = if self.record_insert_metadata(&key, timestamp, self.replica_id) {
            let mut prev = None;
            let value_for_operation = value.clone();
            let _ = self.store.upsert(key.clone(), |current| {
                prev = current.map(|bytes| bytes.to_vec());
                value
            });

            let operation =
                Operation::insert(key.clone(), value_for_operation, timestamp, self.replica_id);
            self.operation_log.write().append(operation);

            debug!(
                "Insert: key={}, timestamp={}, replica={}",
                key, timestamp, self.replica_id
            );

            prev
        } else {
            self.store.get(&key).map(|bytes| bytes.to_vec())
        };

        drop(key_guard);
        self.try_cleanup_key_lock(&key, &key_lock);
        result
    }

    /// Update a key using the current store value and CRDT insert semantics.
    pub fn upsert<F>(&self, key: String, updater: F) -> Vec<u8>
    where
        F: FnOnce(Option<&[u8]>) -> Vec<u8>,
    {
        let key_lock = self.key_lock_for(&key);
        let key_guard = key_lock.lock();

        let current_value = self.store.get(&key);
        let updated_value = updater(current_value.as_deref());
        let timestamp = self.clock.tick();

        let result = if self.record_insert_metadata(&key, timestamp, self.replica_id) {
            let operation = Operation::insert(
                key.clone(),
                updated_value.clone(),
                timestamp,
                self.replica_id,
            );

            self.store.insert(key.clone(), updated_value.clone());
            self.operation_log.write().append(operation);

            updated_value
        } else {
            self.store.get(&key).unwrap_or_default()
        };

        drop(key_guard);
        self.try_cleanup_key_lock(&key, &key_lock);
        result
    }

    /// Fallible variant of upsert that returns serializer/updater errors.
    pub fn try_upsert<F, E>(&self, key: String, updater: F) -> Result<Vec<u8>, E>
    where
        F: FnOnce(Option<&[u8]>) -> Result<Vec<u8>, E>,
    {
        let key_lock = self.key_lock_for(&key);
        let key_guard = key_lock.lock();

        let current_value = self.store.get(&key);
        let updated_value = match updater(current_value.as_deref()) {
            Ok(value) => value,
            Err(err) => {
                drop(key_guard);
                self.try_cleanup_key_lock(&key, &key_lock);
                return Err(err);
            }
        };
        let timestamp = self.clock.tick();

        let result = if self.record_insert_metadata(&key, timestamp, self.replica_id) {
            let operation = Operation::insert(
                key.clone(),
                updated_value.clone(),
                timestamp,
                self.replica_id,
            );

            self.store.insert(key.clone(), updated_value.clone());
            self.operation_log.write().append(operation);

            updated_value
        } else {
            self.store.get(&key).unwrap_or_default()
        };

        drop(key_guard);
        self.try_cleanup_key_lock(&key, &key_lock);
        Ok(result)
    }

    /// Fallible atomic upsert with conditional write. Returning `Ok(None)` skips CRDT write.
    pub fn try_upsert_if<F, E>(&self, key: String, updater: F) -> Result<(Vec<u8>, bool), E>
    where
        F: FnOnce(Option<&[u8]>) -> Result<Option<Vec<u8>>, E>,
    {
        let key_lock = self.key_lock_for(&key);
        let key_guard = key_lock.lock();

        let current_value = self.store.get(&key);
        let maybe_updated_value = match updater(current_value.as_deref()) {
            Ok(value) => value,
            Err(err) => {
                drop(key_guard);
                self.try_cleanup_key_lock(&key, &key_lock);
                return Err(err);
            }
        };

        let (result, changed) = if let Some(updated_value) = maybe_updated_value {
            let timestamp = self.clock.tick();
            if self.record_insert_metadata(&key, timestamp, self.replica_id) {
                let operation = Operation::insert(
                    key.clone(),
                    updated_value.clone(),
                    timestamp,
                    self.replica_id,
                );
                self.store.insert(key.clone(), updated_value.clone());
                self.operation_log.write().append(operation);
                (updated_value, true)
            } else {
                (self.store.get(&key).unwrap_or_default(), false)
            }
        } else {
            (self.store.get(&key).unwrap_or_default(), false)
        };

        drop(key_guard);
        self.try_cleanup_key_lock(&key, &key_lock);
        Ok((result, changed))
    }

    /// Remove key (transparent operation)
    pub fn remove(&self, key: &str) -> Option<Vec<u8>> {
        let key_lock = self.key_lock_for(key);
        let key_guard = key_lock.lock();

        let timestamp = self.clock.tick();

        debug!(
            "Remove: key={}, timestamp={}, replica={}",
            key, timestamp, self.replica_id
        );

        let removed = if self.record_remove_metadata(key, timestamp, self.replica_id) {
            let operation = Operation::remove(key.to_string(), timestamp, self.replica_id);
            self.operation_log.write().append(operation);
            self.store.remove(key)
        } else {
            None
        };

        drop(key_guard);
        self.try_cleanup_key_lock(key, &key_lock);
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

    /// Mutation generation counter. Increments on every insert/remove/upsert.
    pub fn generation(&self) -> u64 {
        self.store.generation()
    }

    /// Get all key-value pairs
    pub fn all(&self) -> std::collections::BTreeMap<String, Vec<u8>> {
        self.store.all()
    }

    /// Get number of live keys in the local store.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

        let seen_operations: HashSet<(ReplicaId, u64)> = {
            let local_log = self.operation_log.read();
            local_log
                .operations()
                .iter()
                .map(|operation| (operation.replica_id(), operation.timestamp()))
                .collect()
        };

        let unseen_operations: Vec<Operation> = {
            let mut local_log = self.operation_log.write();
            local_log.merge(log);
            local_log.compact();

            let mut unseen: Vec<Operation> = local_log
                .operations()
                .iter()
                .filter(|operation| {
                    !seen_operations.contains(&(operation.replica_id(), operation.timestamp()))
                })
                .cloned()
                .collect();
            unseen.sort_by_key(|operation| (operation.timestamp(), operation.replica_id()));
            unseen
        };

        // Apply only new operations in deterministic order.
        for operation in &unseen_operations {
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

    /// Apply insert (LWW semantic; newer tombstones can suppress older inserts).
    fn apply_insert(&self, key: &str, value: Vec<u8>, timestamp: u64, replica_id: ReplicaId) {
        let key_lock = self.key_lock_for(key);
        let key_guard = key_lock.lock();

        if self.record_insert_metadata(key, timestamp, replica_id) {
            self.store.insert(key.to_string(), value);
        }

        drop(key_guard);
        self.try_cleanup_key_lock(key, &key_lock);
    }

    fn compact_key_metadata(versions: &mut Vec<ValueMetadata>) {
        if versions.len() <= 1 {
            return;
        }

        if let Some(winner) = versions.iter().max_by_key(|v| v.version_key()).cloned() {
            versions.clear();
            versions.push(winner);
        }
    }

    fn record_insert_metadata(&self, key: &str, timestamp: u64, replica_id: ReplicaId) -> bool {
        let new_metadata = ValueMetadata::new(timestamp, replica_id);

        match self.metadata.entry(key.to_string()) {
            MapEntry::Occupied(mut entry) => {
                let versions = entry.get_mut();

                let has_existing_entry = versions
                    .iter()
                    .any(|v| v.matches_version(timestamp, replica_id));
                if has_existing_entry {
                    Self::compact_key_metadata(versions);
                    return false;
                }

                let current_winner = versions.iter().max_by_key(|v| v.version_key());

                if current_winner.is_some_and(|winner| winner.is_newer_than(timestamp, replica_id))
                {
                    Self::compact_key_metadata(versions);
                    return false;
                }

                versions.push(new_metadata);
                Self::compact_key_metadata(versions);
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
        let key_guard = key_lock.lock();

        let removed = if self.record_remove_metadata(key, timestamp, replica_id) {
            self.store.remove(key)
        } else {
            None
        };

        drop(key_guard);
        self.try_cleanup_key_lock(key, &key_lock);
        removed
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
                    Self::compact_key_metadata(versions);
                    return false;
                }

                let has_newer_version = versions
                    .iter()
                    .any(|v| v.is_newer_than(timestamp, replica_id));
                if has_newer_version {
                    Self::compact_key_metadata(versions);
                    return false;
                }

                versions.push(tombstone);
                Self::compact_key_metadata(versions);
                true
            }
            MapEntry::Vacant(entry) => {
                if self.store.contains_key(key) {
                    entry.insert(vec![tombstone]);
                    true
                } else {
                    false
                }
            }
        }
    }
}

impl Default for CrdtOrMap {
    fn default() -> Self {
        Self::new()
    }
}
