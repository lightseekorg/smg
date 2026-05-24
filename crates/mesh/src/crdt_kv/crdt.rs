use std::{
    cmp::Reverse,
    collections::HashSet,
    sync::Arc,
    time::{Duration, Instant},
};

use dashmap::{mapref::entry::Entry as MapEntry, DashMap};
use parking_lot::{Mutex, RwLock};
use tracing::{debug, info};

use super::{
    epoch_max_wins,
    kv_store::KvStore,
    merge_strategy::MergeStrategy,
    operation::{Operation, OperationLog},
    replica::{LamportClock, ReplicaId},
};

// ============================================================================
// CRDT OR-Map - Observed-Remove Map Implementation
// ============================================================================

/// Default tombstone grace period. Tombstones younger than this are not
/// garbage collected, preventing data resurrection from stale peers.
/// Gossip converges in seconds for small clusters, so 5 minutes is very
/// conservative.
pub const DEFAULT_TOMBSTONE_GRACE: Duration = Duration::from_secs(300);

/// Value metadata for CRDT OR-Map
#[derive(Debug, Clone)]
struct ValueMetadata {
    timestamp: u64,
    replica_id: ReplicaId,
    is_tombstone: bool, // Marks if this version is a tombstone (deletion)
    /// Monotonic timestamp for tombstone GC. Tombstones younger than
    /// `tombstone_grace` are not garbage collected to prevent data resurrection.
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

    fn from_rate_limit_live_version(version: epoch_max_wins::RateLimitVersion) -> Self {
        Self::new(version.timestamp, version.replica_id)
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

    fn as_rate_limit_version(&self) -> epoch_max_wins::RateLimitVersion {
        epoch_max_wins::RateLimitVersion::new(self.timestamp, self.replica_id)
    }

    fn matches_version(&self, timestamp: u64, replica_id: ReplicaId) -> bool {
        self.timestamp == timestamp && self.replica_id == replica_id
    }

    fn is_newer_than(&self, timestamp: u64, replica_id: ReplicaId) -> bool {
        self.version_key() > (timestamp, replica_id)
    }
}

/// Immutable snapshot of registered prefix→strategy mappings.
/// `register_merge_strategy` builds a new snapshot copy-on-write; readers
/// take a cheap `Arc::clone` and traverse it without holding any lock.
type StrategyTable = Arc<[(String, MergeStrategy)]>;

/// CRDT OR-Map
#[derive(Clone)]
pub struct CrdtOrMap {
    store: KvStore,
    metadata: Arc<DashMap<String, Vec<ValueMetadata>>>, // Key to list of versions
    key_locks: Arc<DashMap<String, Arc<Mutex<()>>>>,    // Per-key critical section lock
    merge_strategies: Arc<RwLock<StrategyTable>>,
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
            merge_strategies: Arc::new(RwLock::new(Arc::from(Vec::new()))),
            replica_id,
            clock: LamportClock::new(),
            operation_log: Arc::new(RwLock::new(OperationLog::new())),
        }
    }

    /// Register the merge strategy for a key prefix. Copy-on-write: builds
    /// a new immutable snapshot so readers (`compact`, `append`, `merge`)
    /// can grab a cheap `Arc::clone` of the current snapshot instead of
    /// cloning the underlying Vec on every gossip round.
    pub(crate) fn register_merge_strategy(&self, prefix: String, strategy: MergeStrategy) {
        let mut guard = self.merge_strategies.write();
        let mut next: Vec<(String, MergeStrategy)> = guard.iter().cloned().collect();
        if let Some((_, existing)) = next
            .iter_mut()
            .find(|(registered_prefix, _)| registered_prefix == &prefix)
        {
            *existing = strategy;
        } else {
            next.push((prefix, strategy));
        }
        next.sort_by_key(|(prefix, _)| Reverse(prefix.len()));
        *guard = Arc::from(next);
    }

    fn merge_strategies_snapshot(&self) -> StrategyTable {
        Arc::clone(&self.merge_strategies.read())
    }

    fn merge_strategy_for_key(&self, key: &str) -> MergeStrategy {
        let strategies = self.merge_strategies_snapshot();
        Self::merge_strategy_for_key_from(&strategies, key)
    }

    fn merge_strategy_for_key_from(
        strategies: &[(String, MergeStrategy)],
        key: &str,
    ) -> MergeStrategy {
        strategies
            .iter()
            .find_map(|(prefix, strategy)| key.starts_with(prefix).then_some(*strategy))
            .unwrap_or(MergeStrategy::LastWriterWins)
    }

    fn compact_operation_log(&self, operation_log: &mut OperationLog) {
        let strategies = self.merge_strategies_snapshot();
        operation_log
            .compact_with_strategy(|key| Self::merge_strategy_for_key_from(&strategies, key));
    }

    fn append_operation(&self, operation: Operation) {
        let mut operation_log = self.operation_log.write();
        let strategies = self.merge_strategies_snapshot();
        operation_log.append_with_strategy(operation, |key| {
            Self::merge_strategy_for_key_from(&strategies, key)
        });
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

        let previous = self.store.get(&key);
        let timestamp = self.clock.tick();
        let operation = Operation::insert(key.clone(), value, timestamp, self.replica_id);
        let result = if self.apply_insert_locked(&key, operation.clone()) {
            self.append_operation(operation);

            debug!(
                "Insert: key={}, timestamp={}, replica={}",
                key, timestamp, self.replica_id
            );

            previous
        } else {
            self.store.get(&key).map(|bytes| bytes.to_vec())
        };

        drop(key_guard);
        self.try_cleanup_key_lock(&key, &key_lock);
        result
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

        let removed = match self.merge_strategy_for_key(key) {
            MergeStrategy::EpochMaxWins => {
                if self.apply_epoch_remove_locked(key, timestamp, self.replica_id) {
                    let operation = Operation::remove(key.to_string(), timestamp, self.replica_id);
                    self.append_operation(operation);
                }
                None
            }
            MergeStrategy::LastWriterWins => {
                if self.record_remove_metadata(key, timestamp, self.replica_id) {
                    let operation = Operation::remove(key.to_string(), timestamp, self.replica_id);
                    self.append_operation(operation);
                    self.store.remove(key)
                } else {
                    None
                }
            }
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

    /// Get all keys without cloning values.
    pub fn keys(&self) -> Vec<String> {
        self.store.keys()
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

    /// Remove tombstoned keys from the metadata and key_locks maps.
    /// Keys that are not in the live store and whose latest metadata entry
    /// is a tombstone are cleaned up to prevent unbounded memory growth.
    ///
    /// Tombstones younger than `grace` are NOT removed, preventing data
    /// resurrection from stale peers that haven't received the tombstone yet
    /// Default grace period: 5 minutes.
    ///
    /// Returns the number of entries removed.
    pub fn gc_tombstones(&self) -> usize {
        self.gc_tombstones_with_grace(DEFAULT_TOMBSTONE_GRACE)
    }

    /// Like `gc_tombstones` but with a custom grace period.
    /// Useful for testing with shorter durations.
    pub fn gc_tombstones_with_grace(&self, grace: Duration) -> usize {
        let now = Instant::now();
        let mut removed = 0;
        // Collect-then-remove: collect keys to check first (read-only iteration),
        // then remove individually. This avoids locking all DashMap shards
        // simultaneously, which would stall concurrent writers.
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
            // Only remove key_locks if no other task holds the lock.
            // Uses the same safety pattern as try_cleanup_key_lock:
            // check strong_count and try_lock before removing.
            self.key_locks.remove_if(&key, |_, lock| {
                Arc::strong_count(lock) <= 2 && lock.try_lock().is_some()
            });
            // Atomically remove metadata only if the key is still not in the
            // live store AND still tombstoned AND the tombstone is older than
            // the grace period. The remove_if closure runs under the DashMap
            // shard lock, preventing a concurrent insert from racing between
            // check and remove.
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

        let strategies = self.merge_strategies_snapshot();
        let seen_operations: HashSet<(ReplicaId, u64)> = {
            let local_log = self.operation_log.read();
            local_log
                .operations()
                .iter()
                .map(|operation| (operation.replica_id(), operation.timestamp()))
                .collect()
        };

        // EpochMaxWins re-applies same-op-id operations because a compacted
        // payload may carry an embedded tombstone_version that the receiver's
        // raw-payload version is missing (`merge_live_value.changed` gates the
        // store update so identical bytes are still a no-op).
        let mut unseen_operations: Vec<Operation> =
            log.operations()
                .iter()
                .filter(|operation| {
                    match Self::merge_strategy_for_key_from(&strategies, operation.key()) {
                        MergeStrategy::LastWriterWins => !seen_operations
                            .contains(&(operation.replica_id(), operation.timestamp())),
                        MergeStrategy::EpochMaxWins => true,
                    }
                })
                .cloned()
                .collect();
        unseen_operations.sort_by_key(|operation| (operation.timestamp(), operation.replica_id()));

        {
            let mut local_log = self.operation_log.write();
            local_log.merge_with_strategy(log, |key| {
                Self::merge_strategy_for_key_from(&strategies, key)
            });
            self.compact_operation_log(&mut local_log);
        }

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
        let operation = Operation::insert(key.to_string(), value, timestamp, replica_id);

        self.apply_insert_locked(key, operation);

        drop(key_guard);
        self.try_cleanup_key_lock(key, &key_lock);
    }

    fn apply_insert_locked(&self, key: &str, operation: Operation) -> bool {
        let Operation::Insert {
            value,
            timestamp,
            replica_id,
            ..
        } = operation
        else {
            return false;
        };

        match self.merge_strategy_for_key(key) {
            MergeStrategy::EpochMaxWins => {
                if let Some(merged) =
                    self.record_epoch_insert_metadata(key, &value, timestamp, replica_id)
                {
                    self.store.insert(key.to_string(), merged);
                    true
                } else {
                    false
                }
            }
            MergeStrategy::LastWriterWins => {
                if self.record_insert_metadata(key, timestamp, replica_id) {
                    self.store.insert(key.to_string(), value);
                    true
                } else {
                    false
                }
            }
        }
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

    fn newest_rate_limit_tombstone_version(
        versions: &[ValueMetadata],
    ) -> Option<epoch_max_wins::RateLimitVersion> {
        versions
            .iter()
            .filter(|version| version.is_tombstone)
            .max_by_key(|version| version.version_key())
            .map(ValueMetadata::as_rate_limit_version)
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

    // Tombstones for EpochMaxWins keys are tracked in two places:
    //   (a) `ValueMetadata { is_tombstone: true, .. }` in `metadata`
    //       — used locally for LWW ordering + tombstone GC.
    //   (b) `tombstone_version` embedded in the stored shard payload
    //       — propagates across replicas via snapshot/compaction so
    //       a peer that receives only the post-tombstone Insert
    //       (the Remove op gone after compaction) still filters
    //       pre-tombstone inserts. See
    //       `test_epoch_max_wins_snapshot_only_propagation_preserves_tombstone_boundary`.
    fn record_epoch_insert_metadata(
        &self,
        key: &str,
        value: &[u8],
        timestamp: u64,
        replica_id: ReplicaId,
    ) -> Option<Vec<u8>> {
        let incoming_version = epoch_max_wins::RateLimitVersion::new(timestamp, replica_id);
        let current = self.store.get(key);

        match self.metadata.entry(key.to_string()) {
            MapEntry::Occupied(mut entry) => {
                let versions = entry.get_mut();
                // No op-id short-circuit: a same-(timestamp, replica_id) op
                // may carry a richer payload (e.g. a compacted shard with an
                // embedded tombstone_version). `merge_live_value.changed`
                // gates the store update so identical bytes are still a no-op.
                let current_tombstone = Self::newest_rate_limit_tombstone_version(versions);
                let Some(merged) = epoch_max_wins::merge_live_value(
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
                let merged = epoch_max_wins::merge_live_value(None, None, value, incoming_version)?;
                entry.insert(vec![ValueMetadata::from_rate_limit_live_version(
                    merged.live_version,
                )]);
                Some(merged.value)
            }
        }
    }

    /// Apply remove
    fn apply_remove(&self, key: &str, timestamp: u64, replica_id: ReplicaId) -> Option<Vec<u8>> {
        let key_lock = self.key_lock_for(key);
        let key_guard = key_lock.lock();

        let removed = match self.merge_strategy_for_key(key) {
            MergeStrategy::EpochMaxWins => {
                self.apply_epoch_remove_locked(key, timestamp, replica_id);
                None
            }
            MergeStrategy::LastWriterWins => {
                if self.record_remove_metadata(key, timestamp, replica_id) {
                    self.store.remove(key)
                } else {
                    None
                }
            }
        };

        drop(key_guard);
        self.try_cleanup_key_lock(key, &key_lock);
        removed
    }

    // Per-point tombstone application for EpochMaxWins keys. The stored shard
    // is filtered against the merged (existing ∪ incoming) tombstone so live
    // path and `compact_operations` agree (spec §2.5). Returns whether the
    // tombstone was newly accepted (used by `remove` to decide whether to
    // append an operation).
    fn apply_epoch_remove_locked(&self, key: &str, timestamp: u64, replica_id: ReplicaId) -> bool {
        let incoming_tombstone = epoch_max_wins::RateLimitVersion::new(timestamp, replica_id);
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
                let current_tombstone = Self::newest_rate_limit_tombstone_version(versions);
                let result = epoch_max_wins::apply_tombstone(
                    current.as_deref(),
                    current_tombstone,
                    incoming_tombstone,
                );
                match result {
                    epoch_max_wins::TombstoneApply::Surviving {
                        value,
                        live_version,
                    } => {
                        versions.clear();
                        versions.push(ValueMetadata::from_rate_limit_live_version(live_version));
                        self.store.insert(key.to_string(), value);
                    }
                    epoch_max_wins::TombstoneApply::Empty { tombstone_version } => {
                        // Preserve `created_at` on an existing tombstone whose
                        // (timestamp, replica_id) already matches the merged
                        // result - older delayed Removes from a lagging peer
                        // must not refresh the GC clock.
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
                // A tombstone for a never-seen key still records ordering
                // information so a delayed pre-tombstone insert is suppressed.
                let result =
                    epoch_max_wins::apply_tombstone(current.as_deref(), None, incoming_tombstone);
                let mut versions = Vec::new();
                match result {
                    epoch_max_wins::TombstoneApply::Surviving {
                        value,
                        live_version,
                    } => {
                        versions.push(ValueMetadata::from_rate_limit_live_version(live_version));
                        self.store.insert(key.to_string(), value);
                    }
                    epoch_max_wins::TombstoneApply::Empty { tombstone_version } => {
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
                // Record the tombstone even for a never-seen key so a delayed
                // older insert cannot resurrect it (CRDT OR-map semantics).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_equal_value_insert_does_not_rewind_metadata() {
        let replica = CrdtOrMap::new();
        replica.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);

        let key = "rl:global:node-a";
        let newer_insert_replica = ReplicaId::new();
        let older_insert_replica = ReplicaId::new();
        let tombstone_replica = ReplicaId::new();

        assert!(replica.apply_insert_locked(
            key,
            Operation::insert(
                key.to_string(),
                epoch_max_wins::encode(6, 0).to_vec(),
                100,
                newer_insert_replica
            ),
        ));
        assert!(!replica.apply_insert_locked(
            key,
            Operation::insert(
                key.to_string(),
                epoch_max_wins::encode(6, 0).to_vec(),
                10,
                older_insert_replica
            ),
        ));

        assert_eq!(replica.apply_remove(key, 50, tombstone_replica), None);
        assert_eq!(
            replica.get(key).and_then(|value| epoch_max_wins::decode(&value)),
            Some(epoch_max_wins::EpochCount { epoch: 6, count: 0 }),
            "older equal-value insert must not let an intermediate tombstone delete the newer live value",
        );
    }
}
