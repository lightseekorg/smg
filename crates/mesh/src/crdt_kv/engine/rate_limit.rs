//! Rate-limit engine.
//!
//! Holds typed `RateLimitState` per key. Replaces the transitional
//! `EpochMaxWinsLegacyEngine` (LWW-shaped `Vec<ValueMetadata>` + separate
//! bytes store + per-key `Mutex` table) with state that matches the
//! EpochMaxWins CRDT directly: each key is either `Live(shard)` carrying a
//! live-points frontier plus an optional tombstone boundary, or
//! `Tombstone(version)` past which dominated inserts are suppressed.
//!
//! State owned by this engine:
//! - `entries: DashMap<String, ShardEntry>` — typed per-key state, atomic via
//!   DashMap's per-shard locks (entry API). No separate `KvStore`, no LWW
//!   metadata vec, no per-key mutex table.
//! - `log: OperationLog` — gossip-visible operation log
//! - shared `LamportClock` (per node, cloned from `CrdtOrMap`)
//! - `generation: AtomicU64` — mutation counter for change-detection callers

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use dashmap::{mapref::entry::Entry as MapEntry, DashMap};
use parking_lot::RwLock;
use tracing::debug;

use super::NamespaceCrdtEngine;
use crate::crdt_kv::{
    epoch_max_wins::{self as ratelimit, RateLimitState, RateLimitVersion},
    operation::{Operation, OperationLog},
    replica::{LamportClock, ReplicaId},
};

struct ShardEntry {
    state: RateLimitState,
    /// Local-clock moment the entry first transitioned to `Tombstone`. `None`
    /// for live entries. Used by `gc_tombstones`; never refreshed on a
    /// subsequent tombstone-to-tombstone merge so a late delayed Remove cannot
    /// restart the grace clock (PR #1469 codex P2).
    tombstoned_at: Option<Instant>,
}

pub(crate) struct RateLimitEngine {
    entries: Arc<DashMap<String, ShardEntry>>,
    log: Arc<RwLock<OperationLog>>,
    // Shared per-node Lamport clock — same Arc held by the router and every
    // other engine. See the equivalent note in `engine::lww`.
    clock: Arc<LamportClock>,
    replica_id: ReplicaId,
    generation: AtomicU64,
}

impl RateLimitEngine {
    pub(crate) fn new(replica_id: ReplicaId, clock: Arc<LamportClock>) -> Self {
        Self {
            entries: Arc::new(DashMap::new()),
            log: Arc::new(RwLock::new(OperationLog::new())),
            clock,
            replica_id,
            generation: AtomicU64::new(0),
        }
    }

    fn append_op(&self, op: Operation) {
        self.log
            .write()
            .append_with_strategy(op, |_| crate::crdt_kv::MergeStrategy::EpochMaxWins);
    }

    fn current_encoded(&self, key: &str) -> Option<Vec<u8>> {
        self.entries
            .get(key)
            .and_then(|entry| entry.state.encode_live())
    }

    /// Merge an insert (value + version) into the entry for `key`.
    /// Returns `None` if the payload is malformed (decoder rejected it).
    /// Otherwise returns `Some(changed)` — `true` iff the merged state differs
    /// from the prior state.
    fn merge_insert(&self, key: &str, value: &[u8], version: RateLimitVersion) -> Option<bool> {
        let incoming = ratelimit::state_from_insert_value(value, version)?;
        let changed = match self.entries.entry(key.to_string()) {
            MapEntry::Occupied(mut occupied) => {
                let entry = occupied.get_mut();
                // `RateLimitState::merge` returns `None` only when both operands
                // carry no live points and no tombstone. Both `entry.state` and
                // `incoming` always carry content, so this can only happen on a
                // contract violation - treat as no-op rather than panicking.
                let Some(merged) = entry.state.clone().merge(incoming) else {
                    return Some(false);
                };
                if merged == entry.state {
                    false
                } else {
                    update_entry(entry, merged);
                    true
                }
            }
            MapEntry::Vacant(vacant) => {
                let tombstoned_at =
                    matches!(&incoming, RateLimitState::Tombstone(_)).then(Instant::now);
                vacant.insert(ShardEntry {
                    state: incoming,
                    tombstoned_at,
                });
                true
            }
        };
        Some(changed)
    }

    /// Merge a remove (tombstone version) into the entry for `key`. Returns
    /// `true` iff the merged state differs from the prior state.
    fn merge_remove(&self, key: &str, version: RateLimitVersion) -> bool {
        let incoming = RateLimitState::Tombstone(version);
        match self.entries.entry(key.to_string()) {
            MapEntry::Occupied(mut occupied) => {
                let entry = occupied.get_mut();
                // See `merge_insert`: `None` requires both operands to be empty,
                // which is impossible here.
                let Some(merged) = entry.state.clone().merge(incoming) else {
                    return false;
                };
                if merged == entry.state {
                    false
                } else {
                    update_entry(entry, merged);
                    true
                }
            }
            MapEntry::Vacant(vacant) => {
                vacant.insert(ShardEntry {
                    state: incoming,
                    tombstoned_at: Some(Instant::now()),
                });
                true
            }
        }
    }
}

/// Apply a merged state to `entry`, updating `tombstoned_at` per the
/// transition. Preserves the existing `tombstoned_at` when the entry stays in
/// `Tombstone` (PR #1469 codex P2: never refresh GC on a same-key delayed
/// Remove).
fn update_entry(entry: &mut ShardEntry, merged: RateLimitState) {
    let was_tombstone = matches!(&entry.state, RateLimitState::Tombstone(_));
    let now_tombstone = matches!(&merged, RateLimitState::Tombstone(_));
    entry.state = merged;
    match (was_tombstone, now_tombstone) {
        (false, true) => entry.tombstoned_at = Some(Instant::now()),
        (true, false) => entry.tombstoned_at = None,
        // (false, false) or (true, true): keep tombstoned_at as-is.
        _ => {}
    }
}

impl NamespaceCrdtEngine for RateLimitEngine {
    fn put_local(&self, key: &str, value: Vec<u8>) -> Option<Vec<u8>> {
        let previous = self.current_encoded(key);
        let timestamp = self.clock.tick();
        let version = RateLimitVersion::new(timestamp, self.replica_id);

        match self.merge_insert(key, &value, version) {
            Some(changed) => {
                if changed {
                    let op = Operation::insert(key.to_string(), value, timestamp, self.replica_id);
                    self.append_op(op);
                    self.generation.fetch_add(1, Ordering::Release);
                    debug!(
                        "RateLimitEngine insert: key={}, timestamp={}, replica={}",
                        key, timestamp, self.replica_id
                    );
                }
                previous
            }
            None => self.current_encoded(key),
        }
    }

    fn delete_local(&self, key: &str) -> Option<Vec<u8>> {
        let timestamp = self.clock.tick();
        let version = RateLimitVersion::new(timestamp, self.replica_id);
        debug!(
            "RateLimitEngine remove: key={}, timestamp={}, replica={}",
            key, timestamp, self.replica_id
        );
        if self.merge_remove(key, version) {
            let op = Operation::remove(key.to_string(), timestamp, self.replica_id);
            self.append_op(op);
            self.generation.fetch_add(1, Ordering::Release);
        }
        None
    }

    fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.current_encoded(key)
    }

    fn contains_key(&self, key: &str) -> bool {
        self.entries
            .get(key)
            .is_some_and(|entry| matches!(&entry.state, RateLimitState::Live(_)))
    }

    fn keys(&self) -> Vec<String> {
        self.entries
            .iter()
            .filter(|entry| matches!(&entry.state, RateLimitState::Live(_)))
            .map(|entry| entry.key().clone())
            .collect()
    }

    fn len(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| matches!(&entry.state, RateLimitState::Live(_)))
            .count()
    }

    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn export_ops(&self) -> Vec<Operation> {
        self.log.read().operations().to_vec()
    }

    fn apply_remote_ops(&self, mut ops: Vec<Operation>) {
        if ops.is_empty() {
            return;
        }

        // EpochMaxWins always replays incoming ops because a compacted snapshot
        // can carry an embedded tombstone_version at the same op-id as a
        // previously-seen raw payload. `merge_insert` / `merge_remove` return
        // `changed=false` for byte-identical re-applies so generation only
        // bumps when state truly changes (PR #1469).
        ops.sort_by_key(|op| (op.timestamp(), op.replica_id()));

        {
            let mut log = self.log.write();
            let incoming = OperationLog::from_operations(ops.clone());
            log.merge_with_strategy(&incoming, |_| crate::crdt_kv::MergeStrategy::EpochMaxWins);
            log.compact_with_strategy(|_| crate::crdt_kv::MergeStrategy::EpochMaxWins);
        }

        for op in ops {
            self.clock.update(op.timestamp());
            let changed = match op {
                Operation::Insert {
                    key,
                    value,
                    timestamp,
                    replica_id,
                } => {
                    let version = RateLimitVersion::new(timestamp, replica_id);
                    self.merge_insert(&key, &value, version).unwrap_or(false)
                }
                Operation::Remove {
                    key,
                    timestamp,
                    replica_id,
                } => {
                    let version = RateLimitVersion::new(timestamp, replica_id);
                    self.merge_remove(&key, version)
                }
            };
            if changed {
                self.generation.fetch_add(1, Ordering::Release);
            }
        }
    }

    fn gc_tombstones(&self, grace: Duration) -> usize {
        let now = Instant::now();
        let candidates: Vec<String> = self
            .entries
            .iter()
            .filter(|entry| {
                matches!(&entry.state, RateLimitState::Tombstone(_))
                    && entry
                        .tombstoned_at
                        .is_some_and(|at| now.saturating_duration_since(at) >= grace)
            })
            .map(|entry| entry.key().clone())
            .collect();

        let mut removed = 0;
        for key in candidates {
            let was_removed = self.entries.remove_if(&key, |_, entry| {
                matches!(&entry.state, RateLimitState::Tombstone(_))
                    && entry
                        .tombstoned_at
                        .is_some_and(|at| now.saturating_duration_since(at) >= grace)
            });
            if was_removed.is_some() {
                removed += 1;
            }
        }
        removed
    }
}
