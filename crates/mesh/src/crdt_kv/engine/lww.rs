//! Last-writer-wins engine.
//!
//! Conflicts are resolved by `(timestamp, replica_id)` strictly. Tombstones
//! and live writes follow the same ordering; the newer wins.
//!
//! State owned by this engine:
//! - [`KvStore`] for live bytes
//! - per-key metadata vec ([`ValueMetadata`]) carrying timestamp / replica /
//!   tombstone flag / GC clock
//! - per-key locks (so same-key writes serialise with metadata updates)
//! - a [`LamportClock`] for stamping local writes
//! - an [`OperationLog`] for replication

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use dashmap::{mapref::entry::Entry as MapEntry, DashMap};
use parking_lot::{Mutex, RwLock};
use tracing::debug;

use super::NamespaceCrdtEngine;
use crate::crdt_kv::{
    kv_store::KvStore,
    operation::{Operation, OperationLog},
    replica::{LamportClock, ReplicaId},
};

// Shared per-node Lamport clock. Op-id `(replica_id, timestamp)` must be unique
// across every operation this node emits, regardless of which engine handled
// the write — otherwise a peer that routes both keys into one engine (e.g. has
// not yet registered the second prefix) deduplicates two unrelated ops by op-id
// and silently drops one. See PR #1539 codex P1.

#[derive(Debug, Clone)]
struct ValueMetadata {
    timestamp: u64,
    replica_id: ReplicaId,
    is_tombstone: bool,
    created_at: Instant,
}

impl PartialEq for ValueMetadata {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp == other.timestamp
            && self.replica_id == other.replica_id
            && self.is_tombstone == other.is_tombstone
    }
}

impl Eq for ValueMetadata {}

impl ValueMetadata {
    fn new(timestamp: u64, replica_id: ReplicaId) -> Self {
        Self {
            timestamp,
            replica_id,
            is_tombstone: false,
            created_at: Instant::now(),
        }
    }

    fn tombstone(timestamp: u64, replica_id: ReplicaId) -> Self {
        Self {
            timestamp,
            replica_id,
            is_tombstone: true,
            created_at: Instant::now(),
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

pub(crate) struct LwwEngine {
    store: KvStore,
    metadata: Arc<DashMap<String, Vec<ValueMetadata>>>,
    key_locks: Arc<DashMap<String, Arc<Mutex<()>>>>,
    log: Arc<RwLock<OperationLog>>,
    clock: Arc<LamportClock>,
    replica_id: ReplicaId,
}

impl LwwEngine {
    pub(crate) fn new(replica_id: ReplicaId, clock: Arc<LamportClock>) -> Self {
        Self {
            store: KvStore::new(),
            metadata: Arc::new(DashMap::new()),
            key_locks: Arc::new(DashMap::new()),
            log: Arc::new(RwLock::new(OperationLog::new())),
            clock,
            replica_id,
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
                // Tombstone for a never-seen key still records ordering so a
                // delayed older insert is suppressed (PR #1469).
                entry.insert(vec![tombstone]);
                true
            }
        }
    }

    fn apply_insert(&self, key: &str, value: Vec<u8>, timestamp: u64, replica_id: ReplicaId) {
        let key_lock = self.key_lock_for(key);
        let key_guard = key_lock.lock();
        if self.record_insert_metadata(key, timestamp, replica_id) {
            self.store.insert(key.to_string(), value);
        }
        drop(key_guard);
        self.try_cleanup_key_lock(key, &key_lock);
    }

    fn apply_remove_inner(
        &self,
        key: &str,
        timestamp: u64,
        replica_id: ReplicaId,
    ) -> Option<Vec<u8>> {
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

    fn append_op(&self, op: Operation) {
        let mut log = self.log.write();
        log.append(op);
        if log.len() > OperationLog::AUTO_COMPACT_THRESHOLD {
            Self::compact_log(&mut log);
            // Safety valve for the many-thousand-unique-key pathology: if
            // compaction couldn't bring the log below the threshold, drop
            // the oldest entries. Only safe on this local-write path -
            // truncating in `apply_remote_ops` would silently drop
            // remotely-learned keys from the log even though they are live
            // in state, breaking downstream sync from this node.
            if log.len() > OperationLog::AUTO_COMPACT_THRESHOLD {
                let keep = OperationLog::AUTO_COMPACT_THRESHOLD * 3 / 4;
                let drain_count = log.len() - keep;
                tracing::warn!(
                    total = log.len(),
                    draining = drain_count,
                    keeping = keep,
                    "LwwEngine log still over threshold after compaction, truncating oldest entries"
                );
                log.operations_mut().drain(..drain_count);
            }
        }
    }

    /// LWW per-key fold: the winning op is the one with the maximum
    /// `(timestamp, replica_id)` tuple. Tombstones and inserts share the
    /// ordering - the newer wins regardless of kind.
    fn lww_fold(ops: &[Operation]) -> Option<Operation> {
        ops.iter()
            .max_by_key(|op| (op.timestamp(), op.replica_id()))
            .cloned()
    }

    /// Compact the log per LWW rules: collapse all ops for a key down to the
    /// one with maximum `(timestamp, replica_id)`. Never drops keys outright -
    /// the truncate-oldest safety valve lives only on the `append_op`
    /// local-write path.
    fn compact_log(log: &mut OperationLog) {
        log.compact_by_key(Self::lww_fold);
    }
}

impl NamespaceCrdtEngine for LwwEngine {
    fn put_local(&self, key: &str, value: Vec<u8>) -> Option<Vec<u8>> {
        let key_lock = self.key_lock_for(key);
        let key_guard = key_lock.lock();

        let previous = self.store.get(key);
        let timestamp = self.clock.tick();
        let accepted = self.record_insert_metadata(key, timestamp, self.replica_id);
        let result = if accepted {
            let op = Operation::insert(key.to_string(), value.clone(), timestamp, self.replica_id);
            self.store.insert(key.to_string(), value);
            self.append_op(op);
            debug!(
                "LwwEngine insert: key={}, timestamp={}, replica={}",
                key, timestamp, self.replica_id
            );
            previous
        } else {
            self.store.get(key).map(|bytes| bytes.to_vec())
        };

        drop(key_guard);
        self.try_cleanup_key_lock(key, &key_lock);
        result
    }

    fn delete_local(&self, key: &str) -> Option<Vec<u8>> {
        let key_lock = self.key_lock_for(key);
        let key_guard = key_lock.lock();

        let timestamp = self.clock.tick();
        debug!(
            "LwwEngine remove: key={}, timestamp={}, replica={}",
            key, timestamp, self.replica_id
        );
        let removed = if self.record_remove_metadata(key, timestamp, self.replica_id) {
            let op = Operation::remove(key.to_string(), timestamp, self.replica_id);
            self.append_op(op);
            self.store.remove(key)
        } else {
            None
        };

        drop(key_guard);
        self.try_cleanup_key_lock(key, &key_lock);
        removed
    }

    fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.store.get(key)
    }

    fn contains_key(&self, key: &str) -> bool {
        self.store.contains_key(key)
    }

    fn keys(&self) -> Vec<String> {
        self.store.keys()
    }

    fn len(&self) -> usize {
        self.store.len()
    }

    fn generation(&self) -> u64 {
        self.store.generation()
    }

    fn export_ops(&self) -> Vec<Operation> {
        self.log.read().operations().to_vec()
    }

    fn apply_remote_ops(&self, ops: Vec<Operation>) {
        if ops.is_empty() {
            return;
        }

        // Determine which incoming ops the local log has not yet seen. LWW
        // dedups by op-id; an op already in the log is a no-op.
        let seen: std::collections::HashSet<(ReplicaId, u64)> = self
            .log
            .read()
            .operations()
            .iter()
            .map(|op| (op.replica_id(), op.timestamp()))
            .collect();

        let mut unseen: Vec<Operation> = ops
            .iter()
            .filter(|op| !seen.contains(&(op.replica_id(), op.timestamp())))
            .cloned()
            .collect();
        unseen.sort_by_key(|op| (op.timestamp(), op.replica_id()));

        // LWW op-id collision policy is dedup: an op already in the log by
        // `(replica_id, timestamp)` is a no-op. `unseen` was already filtered
        // against the local log above; append it directly and compact (no
        // truncate - this path must not drop remotely-learned keys).
        {
            let mut log = self.log.write();
            for op in &unseen {
                log.append(op.clone());
            }
            Self::compact_log(&mut log);
        }

        // Apply unseen ops to live state. Lamport clock observes each remote
        // timestamp so subsequent local ticks beat it.
        for op in unseen {
            self.clock.update(op.timestamp());
            match op {
                Operation::Insert {
                    key,
                    value,
                    timestamp,
                    replica_id,
                } => {
                    self.apply_insert(&key, value, timestamp, replica_id);
                }
                Operation::Remove {
                    key,
                    timestamp,
                    replica_id,
                } => {
                    let _ = self.apply_remove_inner(&key, timestamp, replica_id);
                }
            }
        }
    }

    fn gc_tombstones(&self, grace: Duration) -> usize {
        let now = Instant::now();
        let mut removed = 0;
        let keys_to_check: Vec<String> = self
            .metadata
            .iter()
            .filter(|entry| !self.store.contains_key(entry.key()))
            .map(|entry| entry.key().clone())
            .collect();

        for key in keys_to_check {
            if !self.key_is_tombstoned_or_unknown(&key) {
                continue;
            }
            self.key_locks.remove_if(&key, |_, lock| {
                Arc::strong_count(lock) <= 2 && lock.try_lock().is_some()
            });
            let was_removed = self.metadata.remove_if(&key, |_, versions| {
                !self.store.contains_key(&key)
                    && versions
                        .iter()
                        .max_by_key(|v| v.version_key())
                        .is_none_or(|winner| {
                            winner.is_tombstone
                                && now.saturating_duration_since(winner.created_at) >= grace
                        })
            });
            if was_removed.is_some() {
                removed += 1;
            }
        }
        removed
    }
}
