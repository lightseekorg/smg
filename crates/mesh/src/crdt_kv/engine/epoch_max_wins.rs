//! Transitional EpochMaxWins engine.
//!
//! This is a behavior-preserving wrapper around the legacy EpochMaxWins logic
//! that lived inline in `CrdtOrMap`. It owns its own state (so the engine
//! trait surface is exercised end-to-end), but its internal shape still
//! mirrors the LWW layout (one shared `ValueMetadata` vec per key, etc.).
//! PR #3 will replace this with a real `RateLimitEngine` that holds a typed
//! `RateLimitShard` per key without the LWW-shaped metadata layer.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use dashmap::{mapref::entry::Entry as MapEntry, DashMap};
use parking_lot::{Mutex, RwLock};
use tracing::debug;

use super::NamespaceCrdtEngine;
use crate::crdt_kv::{
    epoch_max_wins as ratelimit,
    kv_store::KvStore,
    operation::{Operation, OperationLog},
    replica::{LamportClock, ReplicaId},
};

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
    fn from_rate_limit_live_version(version: ratelimit::RateLimitVersion) -> Self {
        Self {
            timestamp: version.timestamp,
            replica_id: version.replica_id,
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

    fn as_rate_limit_version(&self) -> ratelimit::RateLimitVersion {
        ratelimit::RateLimitVersion::new(self.timestamp, self.replica_id)
    }

    fn matches_version(&self, timestamp: u64, replica_id: ReplicaId) -> bool {
        self.timestamp == timestamp && self.replica_id == replica_id
    }
}

pub(crate) struct EpochMaxWinsLegacyEngine {
    store: KvStore,
    metadata: Arc<DashMap<String, Vec<ValueMetadata>>>,
    key_locks: Arc<DashMap<String, Arc<Mutex<()>>>>,
    log: Arc<RwLock<OperationLog>>,
    // Shared per-node Lamport clock — see the same note in `engine::lww`.
    clock: Arc<LamportClock>,
    replica_id: ReplicaId,
}

impl EpochMaxWinsLegacyEngine {
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

    fn newest_tombstone_version(versions: &[ValueMetadata]) -> Option<ratelimit::RateLimitVersion> {
        versions
            .iter()
            .filter(|version| version.is_tombstone)
            .max_by_key(|version| version.version_key())
            .map(ValueMetadata::as_rate_limit_version)
    }

    fn record_epoch_insert_metadata(
        &self,
        key: &str,
        value: &[u8],
        timestamp: u64,
        replica_id: ReplicaId,
    ) -> Option<Vec<u8>> {
        let incoming_version = ratelimit::RateLimitVersion::new(timestamp, replica_id);
        let current = self.store.get(key);

        match self.metadata.entry(key.to_string()) {
            MapEntry::Occupied(mut entry) => {
                let versions = entry.get_mut();
                let current_tombstone = Self::newest_tombstone_version(versions);
                let Some(merged) = ratelimit::merge_live_value(
                    current.as_deref(),
                    current_tombstone,
                    value,
                    incoming_version,
                ) else {
                    Self::compact_key_metadata(versions);
                    return None;
                };
                if !merged.changed {
                    Self::compact_key_metadata(versions);
                    return None;
                }
                versions.clear();
                versions.push(ValueMetadata::from_rate_limit_live_version(
                    merged.live_version,
                ));
                Some(merged.value)
            }
            MapEntry::Vacant(entry) => {
                let merged = ratelimit::merge_live_value(None, None, value, incoming_version)?;
                entry.insert(vec![ValueMetadata::from_rate_limit_live_version(
                    merged.live_version,
                )]);
                Some(merged.value)
            }
        }
    }

    fn apply_epoch_remove_locked(&self, key: &str, timestamp: u64, replica_id: ReplicaId) -> bool {
        let incoming_tombstone = ratelimit::RateLimitVersion::new(timestamp, replica_id);
        let current = self.store.get(key);

        match self.metadata.entry(key.to_string()) {
            MapEntry::Occupied(mut entry) => {
                let versions = entry.get_mut();
                let already_recorded = versions
                    .iter()
                    .any(|v| v.is_tombstone && v.matches_version(timestamp, replica_id));
                if already_recorded {
                    Self::compact_key_metadata(versions);
                    return false;
                }
                let current_tombstone = Self::newest_tombstone_version(versions);
                let result = ratelimit::apply_tombstone(
                    current.as_deref(),
                    current_tombstone,
                    incoming_tombstone,
                );
                match result {
                    ratelimit::TombstoneApply::Surviving {
                        value,
                        live_version,
                    } => {
                        versions.clear();
                        versions.push(ValueMetadata::from_rate_limit_live_version(live_version));
                        self.store.insert(key.to_string(), value);
                    }
                    ratelimit::TombstoneApply::Empty { tombstone_version } => {
                        // Preserve `created_at` on a dominated tombstone (PR #1469
                        // codex P2: older delayed Removes must not refresh GC).
                        let already_matches = versions.iter().any(|v| {
                            v.is_tombstone
                                && v.matches_version(
                                    tombstone_version.timestamp,
                                    tombstone_version.replica_id,
                                )
                        });
                        if !already_matches {
                            versions.clear();
                            versions.push(ValueMetadata::tombstone(
                                tombstone_version.timestamp,
                                tombstone_version.replica_id,
                            ));
                        }
                        self.store.remove(key);
                    }
                }
                true
            }
            MapEntry::Vacant(entry) => {
                // Tombstone for a never-seen key still records ordering so a
                // delayed pre-tombstone insert is suppressed (PR #1469).
                let result =
                    ratelimit::apply_tombstone(current.as_deref(), None, incoming_tombstone);
                let mut versions = Vec::new();
                match result {
                    ratelimit::TombstoneApply::Surviving {
                        value,
                        live_version,
                    } => {
                        versions.push(ValueMetadata::from_rate_limit_live_version(live_version));
                        self.store.insert(key.to_string(), value);
                    }
                    ratelimit::TombstoneApply::Empty { tombstone_version } => {
                        versions.push(ValueMetadata::tombstone(
                            tombstone_version.timestamp,
                            tombstone_version.replica_id,
                        ));
                        self.store.remove(key);
                    }
                }
                entry.insert(versions);
                true
            }
        }
    }

    fn apply_remote_insert(
        &self,
        key: &str,
        value: Vec<u8>,
        timestamp: u64,
        replica_id: ReplicaId,
    ) {
        let key_lock = self.key_lock_for(key);
        let key_guard = key_lock.lock();
        if let Some(stored) = self.record_epoch_insert_metadata(key, &value, timestamp, replica_id)
        {
            self.store.insert(key.to_string(), stored);
        }
        drop(key_guard);
        self.try_cleanup_key_lock(key, &key_lock);
    }

    fn apply_remote_remove(&self, key: &str, timestamp: u64, replica_id: ReplicaId) {
        let key_lock = self.key_lock_for(key);
        let key_guard = key_lock.lock();
        self.apply_epoch_remove_locked(key, timestamp, replica_id);
        drop(key_guard);
        self.try_cleanup_key_lock(key, &key_lock);
    }

    fn append_op(&self, op: Operation) {
        self.log
            .write()
            .append_with_strategy(op, |_| crate::crdt_kv::MergeStrategy::EpochMaxWins);
    }
}

impl NamespaceCrdtEngine for EpochMaxWinsLegacyEngine {
    fn put_local(&self, key: &str, value: Vec<u8>) -> Option<Vec<u8>> {
        let key_lock = self.key_lock_for(key);
        let key_guard = key_lock.lock();

        let previous = self.store.get(key);
        let timestamp = self.clock.tick();
        let result = if let Some(stored) =
            self.record_epoch_insert_metadata(key, &value, timestamp, self.replica_id)
        {
            let op = Operation::insert(key.to_string(), value, timestamp, self.replica_id);
            self.store.insert(key.to_string(), stored);
            self.append_op(op);
            debug!(
                "EpochMaxWinsLegacyEngine insert: key={}, timestamp={}, replica={}",
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
            "EpochMaxWinsLegacyEngine remove: key={}, timestamp={}, replica={}",
            key, timestamp, self.replica_id
        );
        if self.apply_epoch_remove_locked(key, timestamp, self.replica_id) {
            let op = Operation::remove(key.to_string(), timestamp, self.replica_id);
            self.append_op(op);
        }

        drop(key_guard);
        self.try_cleanup_key_lock(key, &key_lock);
        // EpochMaxWins removes are per-point; no clean "what was removed" answer.
        None
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

        // EpochMaxWins always replays incoming ops because a compacted snapshot
        // can carry an embedded tombstone_version at the same op-id as a
        // previously-seen raw payload. `merge_live_value.changed` gates the
        // store update so identical bytes are still a no-op (PR #1469).
        let mut to_apply = ops.clone();
        to_apply.sort_by_key(|op| (op.timestamp(), op.replica_id()));

        {
            let mut log = self.log.write();
            let incoming = OperationLog::from_operations(ops);
            log.merge_with_strategy(&incoming, |_| crate::crdt_kv::MergeStrategy::EpochMaxWins);
            log.compact_with_strategy(|_| crate::crdt_kv::MergeStrategy::EpochMaxWins);
        }

        for op in to_apply {
            self.clock.update(op.timestamp());
            match op {
                Operation::Insert {
                    key,
                    value,
                    timestamp,
                    replica_id,
                } => {
                    self.apply_remote_insert(&key, value, timestamp, replica_id);
                }
                Operation::Remove {
                    key,
                    timestamp,
                    replica_id,
                } => {
                    self.apply_remote_remove(&key, timestamp, replica_id);
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
