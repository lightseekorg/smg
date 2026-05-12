//! Rate-limit shard merge for epoch-aware counters.
//!
//! Gateway code writes the simple application payload `(epoch, count)` as
//! 16 bytes: `u64` big-endian epoch followed by `i64` big-endian count.
//! Inside the CRDT, `rl:` values are normalized into a rate-limit shard
//! state that also carries a normalized frontier of live points plus the
//! newest tombstone boundary. That extra metadata is what lets operation-log
//! compaction keep deletes meaningful: a delayed insert from before a
//! tombstone cannot be resurrected just because the log compacted to one live
//! value.
//!
//! Raw epoch/count payloads are accepted only at the insert boundary. Stored
//! and gossiped `rl:` values are always serialized [`RateLimitShard`] states.
//! Malformed stored input: if one side decodes, it wins. If both fail, keep
//! `local` per the `MergeStrategy::EpochMaxWins` contract in `kv.rs` - a
//! no-op on the store.

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use super::{operation::Operation, replica::ReplicaId};

/// Fixed application payload size: 8-byte epoch + 8-byte count.
pub const EPOCH_MAX_WINS_ENCODED_LEN: usize = 16;

/// Parsed value returned owned so callers don't need to keep the source slice
/// alive across the merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpochCount {
    pub epoch: u64,
    pub count: i64,
}

/// Lamport version for a rate-limit shard state component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(super) struct RateLimitVersion {
    pub timestamp: u64,
    pub replica_id: ReplicaId,
}

impl RateLimitVersion {
    pub(super) fn new(timestamp: u64, replica_id: ReplicaId) -> Self {
        Self {
            timestamp,
            replica_id,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct LivePoint {
    value: EpochCount,
    version: RateLimitVersion,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RateLimitShard {
    live_points: Vec<LivePoint>,
    tombstone_version: Option<RateLimitVersion>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RateLimitState {
    Live(RateLimitShard),
    Tombstone(RateLimitVersion),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ValueWinner {
    Local,
    Remote,
    Equal,
}

pub(super) struct LiveMerge {
    pub value: Vec<u8>,
    pub live_version: RateLimitVersion,
    pub changed: bool,
}

/// Encode `(epoch, count)` to the 16-byte application payload. `rl:` CRDT
/// inserts normalize this payload into a [`RateLimitShard`] state before
/// storing it.
#[must_use]
pub fn encode(epoch: u64, count: i64) -> [u8; EPOCH_MAX_WINS_ENCODED_LEN] {
    let mut buf = [0u8; EPOCH_MAX_WINS_ENCODED_LEN];
    buf[0..8].copy_from_slice(&epoch.to_be_bytes());
    buf[8..16].copy_from_slice(&count.to_be_bytes());
    buf
}

/// Decode a normalized CRDT shard state. `None` means malformed.
#[must_use]
pub fn decode(bytes: &[u8]) -> Option<EpochCount> {
    decode_shard(bytes).and_then(|shard| shard.current_value())
}

fn decode_raw_epoch_count(bytes: &[u8]) -> Option<EpochCount> {
    if bytes.len() != EPOCH_MAX_WINS_ENCODED_LEN {
        return None;
    }
    let epoch = u64::from_be_bytes(bytes[0..8].try_into().ok()?);
    let count = i64::from_be_bytes(bytes[8..16].try_into().ok()?);
    Some(EpochCount { epoch, count })
}

fn encode_shard(shard: &RateLimitShard) -> Option<Vec<u8>> {
    bincode::serialize(shard).ok()
}

fn decode_shard(bytes: &[u8]) -> Option<RateLimitShard> {
    let shard: RateLimitShard = bincode::deserialize(bytes).ok()?;
    (!shard.live_points.is_empty()).then_some(shard)
}

fn decode_stored_live(bytes: &[u8]) -> Option<RateLimitShard> {
    decode_shard(bytes)
}

fn compare_epoch_count(local: EpochCount, remote: EpochCount) -> ValueWinner {
    match local.epoch.cmp(&remote.epoch) {
        Ordering::Greater => ValueWinner::Local,
        Ordering::Less => ValueWinner::Remote,
        Ordering::Equal => match local.count.cmp(&remote.count) {
            Ordering::Greater => ValueWinner::Local,
            Ordering::Less => ValueWinner::Remote,
            Ordering::Equal => ValueWinner::Equal,
        },
    }
}

impl RateLimitShard {
    fn from_live_point(point: LivePoint) -> Self {
        Self {
            live_points: vec![point],
            tombstone_version: None,
        }
    }

    fn current_value(&self) -> Option<EpochCount> {
        self.live_points
            .iter()
            .map(|point| point.value)
            .reduce(
                |current, candidate| match compare_epoch_count(current, candidate) {
                    ValueWinner::Remote => candidate,
                    ValueWinner::Local | ValueWinner::Equal => current,
                },
            )
    }

    fn newest_live_version(&self) -> Option<RateLimitVersion> {
        self.live_points.iter().map(|point| point.version).max()
    }

    fn merged(
        mut points: Vec<LivePoint>,
        tombstone_version: Option<RateLimitVersion>,
    ) -> Option<Self> {
        points.retain(|point| tombstone_version.is_none_or(|tombstone| point.version > tombstone));
        if points.is_empty() {
            return None;
        }

        points.sort_by_key(|point| std::cmp::Reverse(point.version));
        let mut suffix_best: Option<EpochCount> = None;
        let mut frontier = Vec::new();
        for point in points {
            let keep = suffix_best.is_none_or(|best| {
                matches!(compare_epoch_count(point.value, best), ValueWinner::Local)
            });
            if keep {
                suffix_best = Some(match suffix_best {
                    Some(best) => match compare_epoch_count(best, point.value) {
                        ValueWinner::Remote => point.value,
                        ValueWinner::Local | ValueWinner::Equal => best,
                    },
                    None => point.value,
                });
                frontier.push(point);
            }
        }
        frontier.sort_by_key(|point| point.version);

        Some(Self {
            live_points: frontier,
            tombstone_version,
        })
    }

    fn live_points_after_tombstone(
        &self,
        tombstone_version: Option<RateLimitVersion>,
    ) -> Vec<LivePoint> {
        self.live_points
            .iter()
            .filter(|point| tombstone_version.is_none_or(|tombstone| point.version > tombstone))
            .cloned()
            .collect()
    }
}

impl RateLimitState {
    fn tombstone_version(&self) -> Option<RateLimitVersion> {
        match self {
            Self::Live(shard) => shard.tombstone_version,
            Self::Tombstone(version) => Some(*version),
        }
    }

    fn live_points_after_tombstone(
        &self,
        tombstone_version: Option<RateLimitVersion>,
    ) -> Vec<LivePoint> {
        match self {
            Self::Live(shard) => shard.live_points_after_tombstone(tombstone_version),
            Self::Tombstone(_) => Vec::new(),
        }
    }

    fn merge(self, other: Self) -> Option<Self> {
        let tombstone_version = self.tombstone_version().max(other.tombstone_version());
        let mut live_points = self.live_points_after_tombstone(tombstone_version);
        live_points.extend(other.live_points_after_tombstone(tombstone_version));

        match RateLimitShard::merged(live_points, tombstone_version) {
            Some(shard) => Some(Self::Live(shard)),
            None => tombstone_version.map(Self::Tombstone),
        }
    }

    fn into_operation(self, key: String) -> Option<Operation> {
        match self {
            Self::Live(shard) => {
                let live_version = shard.newest_live_version()?;
                Some(Operation::insert(
                    key,
                    encode_shard(&shard)?,
                    live_version.timestamp,
                    live_version.replica_id,
                ))
            }
            Self::Tombstone(version) => Some(Operation::remove(
                key,
                version.timestamp,
                version.replica_id,
            )),
        }
    }
}

fn state_from_insert_value(value: &[u8], version: RateLimitVersion) -> Option<RateLimitState> {
    if let Some(shard) = decode_shard(value) {
        return Some(RateLimitState::Live(shard));
    }
    decode_raw_epoch_count(value).map(|value| {
        RateLimitState::Live(RateLimitShard::from_live_point(LivePoint {
            value,
            version,
        }))
    })
}

fn state_from_stored_value(value: &[u8]) -> Option<RateLimitState> {
    decode_stored_live(value).map(RateLimitState::Live)
}

pub(super) fn merge_live_value(
    current_value: Option<&[u8]>,
    current_tombstone_version: Option<RateLimitVersion>,
    incoming_value: &[u8],
    incoming_version: RateLimitVersion,
) -> Option<LiveMerge> {
    let incoming = state_from_insert_value(incoming_value, incoming_version)?;
    let current = current_value.and_then(state_from_stored_value);
    let current = match (current, current_tombstone_version) {
        (Some(current), Some(tombstone_version)) => {
            current.merge(RateLimitState::Tombstone(tombstone_version))
        }
        (Some(current), None) => Some(current),
        (None, Some(tombstone_version)) => Some(RateLimitState::Tombstone(tombstone_version)),
        (None, None) => None,
    };

    let merged = match current {
        Some(current) => current.merge(incoming)?,
        None => incoming,
    };
    let RateLimitState::Live(shard) = merged else {
        return None;
    };
    let value = encode_shard(&shard)?;
    let changed = current_value != Some(value.as_slice());
    let live_version = shard.newest_live_version()?;
    Some(LiveMerge {
        value,
        live_version,
        changed,
    })
}

pub(super) fn compact_operations<'a>(
    operations: impl IntoIterator<Item = &'a Operation>,
) -> Option<Operation> {
    let mut key = None;
    let mut state: Option<RateLimitState> = None;

    for operation in operations {
        key.get_or_insert_with(|| operation.key().to_string());
        let operation_state = match operation {
            Operation::Insert {
                value,
                timestamp,
                replica_id,
                ..
            } => state_from_insert_value(value, RateLimitVersion::new(*timestamp, *replica_id))?,
            Operation::Remove {
                timestamp,
                replica_id,
                ..
            } => RateLimitState::Tombstone(RateLimitVersion::new(*timestamp, *replica_id)),
        };
        state = Some(match state {
            Some(current) => current.merge(operation_state)?,
            None => operation_state,
        });
    }

    state.and_then(|state| state.into_operation(key?))
}

/// Merge two stored rate-limit shard values.
///
/// Both decode: merge their shard frontiers. One decodes: the well-formed
/// side wins. Neither decodes: keep `local` (no-op, per the `EpochMaxWins`
/// contract in `kv.rs`).
#[must_use]
pub fn merge(local: &[u8], remote: &[u8]) -> Vec<u8> {
    match (decode_stored_live(local), decode_stored_live(remote)) {
        (Some(local_state), Some(remote_state)) => {
            let Some(RateLimitState::Live(shard)) =
                RateLimitState::Live(local_state).merge(RateLimitState::Live(remote_state))
            else {
                return local.to_vec();
            };
            encode_shard(&shard).unwrap_or_else(|| local.to_vec())
        }
        (Some(_), None) | (None, None) => local.to_vec(),
        (None, Some(_)) => remote.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rate_limit_version(timestamp: u64) -> RateLimitVersion {
        RateLimitVersion::new(timestamp, ReplicaId::new())
    }

    fn stored(epoch: u64, count: i64, timestamp: u64) -> Vec<u8> {
        merge_live_value(
            None,
            None,
            &encode(epoch, count),
            rate_limit_version(timestamp),
        )
        .expect("raw epoch/count insert normalizes")
        .value
    }

    #[test]
    fn raw_epoch_count_payload_round_trip() {
        for (epoch, count) in [
            (0_u64, 0_i64),
            (1, 1),
            (5, 30),
            (u64::MAX, i64::MAX),
            (u64::MAX, i64::MIN),
            (42, -1),
        ] {
            let buf = encode(epoch, count);
            assert_eq!(buf.len(), EPOCH_MAX_WINS_ENCODED_LEN);
            let decoded = decode_raw_epoch_count(&buf).expect("encoded buffer is 16 bytes");
            assert_eq!(decoded, EpochCount { epoch, count });
        }
    }

    #[test]
    fn public_decode_rejects_raw_epoch_count_payload() {
        assert_eq!(decode(&encode(1, 2)), None);
    }

    #[test]
    fn raw_epoch_count_decode_rejects_wrong_lengths() {
        assert_eq!(decode_raw_epoch_count(&[]), None);
        assert_eq!(decode_raw_epoch_count(&[0u8; 15]), None);
        assert_eq!(decode_raw_epoch_count(&[0u8; 17]), None);
        assert!(decode_raw_epoch_count(&[0u8; 16]).is_some());
    }

    #[test]
    fn normalized_shard_decodes_to_epoch_count() {
        let merged = merge_live_value(None, None, &encode(7, 42), rate_limit_version(10))
            .expect("raw epoch/count insert normalizes to shard state");
        assert_ne!(merged.value.len(), EPOCH_MAX_WINS_ENCODED_LEN);
        assert_eq!(
            decode(&merged.value),
            Some(EpochCount {
                epoch: 7,
                count: 42
            })
        );
    }

    #[test]
    fn same_epoch_max_count_wins() {
        let local = stored(5, 30, 1);
        let remote = stored(5, 42, 2);
        let merged = merge(&local, &remote);
        assert_eq!(
            decode(&merged).unwrap(),
            EpochCount {
                epoch: 5,
                count: 42
            }
        );
        assert_eq!(merge(&remote, &local), merged);
    }

    #[test]
    fn higher_epoch_wins_even_with_lower_count() {
        let merged = merge(&stored(5, 30, 1), &stored(6, 0, 2));
        assert_eq!(decode(&merged).unwrap(), EpochCount { epoch: 6, count: 0 });
    }

    #[test]
    fn lower_epoch_loses_to_local_newer_window() {
        let merged = merge(&stored(6, 10, 1), &stored(5, 100, 2));
        assert_eq!(
            decode(&merged).unwrap(),
            EpochCount {
                epoch: 6,
                count: 10
            }
        );
    }

    #[test]
    fn near_simultaneous_reset_both_at_zero() {
        let merged = merge(&stored(5, 0, 1), &stored(5, 0, 2));
        assert_eq!(decode(&merged).unwrap(), EpochCount { epoch: 5, count: 0 });
    }

    #[test]
    fn malformed_remote_keeps_local() {
        let local = stored(5, 30, 1);
        let merged = merge(&local, &[0xFFu8; 15]);
        assert_eq!(merged, local);
    }

    #[test]
    fn malformed_local_is_replaced_by_remote() {
        let remote = stored(5, 30, 1);
        let merged = merge(&[], &remote);
        assert_eq!(merged, remote);
    }

    #[test]
    fn both_malformed_keeps_local_no_panic() {
        let corrupt_local = vec![1u8, 2, 3];
        let merged = merge(&corrupt_local, &[0xFFu8; 17]);
        assert_eq!(merged, corrupt_local);
    }

    #[test]
    fn signed_count_preserves_sign() {
        let merged = merge(&stored(5, -10, 1), &stored(5, -5, 2));
        assert_eq!(
            decode(&merged).unwrap(),
            EpochCount {
                epoch: 5,
                count: -5
            }
        );
    }

    #[test]
    fn merge_is_idempotent() {
        let value = stored(42, 7, 1);
        assert_eq!(merge(&value, &value), value);
    }

    #[test]
    fn merge_is_associative_on_three_values() {
        let a = stored(5, 10, 1);
        let b = stored(6, 3, 2);
        let c = stored(6, 9, 3);
        let ab_then_c = merge(&merge(&a, &b), &c);
        let a_then_bc = merge(&a, &merge(&b, &c));
        assert_eq!(ab_then_c, a_then_bc);
        assert_eq!(
            decode(&ab_then_c).unwrap(),
            EpochCount { epoch: 6, count: 9 }
        );
    }

    #[test]
    fn compacted_live_state_remembers_tombstone_boundary() {
        let key = "rl:global:node-a".to_string();
        let ops = [
            Operation::insert(key.clone(), encode(9, 99).to_vec(), 10, ReplicaId::new()),
            Operation::remove(key.clone(), 20, ReplicaId::new()),
            Operation::insert(key.clone(), encode(1, 1).to_vec(), 30, ReplicaId::new()),
        ];

        let compacted =
            compact_operations(ops.iter()).expect("post-tombstone live insert remains live");
        assert!(matches!(compacted, Operation::Insert { .. }));

        let delayed = Operation::insert(key.clone(), encode(9, 99).to_vec(), 10, ReplicaId::new());
        let compacted_again = compact_operations([compacted, delayed].iter())
            .expect("compacted live shard remains live");
        let Operation::Insert { value, .. } = compacted_again else {
            panic!("expected live compacted shard");
        };
        assert_eq!(
            decode(&value),
            Some(EpochCount { epoch: 1, count: 1 }),
            "pre-tombstone high-epoch insert must stay suppressed after compaction",
        );
    }

    #[test]
    fn compacted_live_state_uses_newest_live_version() {
        let key = "rl:global:node-a".to_string();
        let ops = [
            Operation::remove(key.clone(), 50, ReplicaId::new()),
            Operation::insert(key.clone(), encode(7, 100).to_vec(), 60, ReplicaId::new()),
            Operation::insert(key.clone(), encode(6, 1).to_vec(), 70, ReplicaId::new()),
        ];

        let compacted = compact_operations(ops.iter()).expect("live state wins");
        let Operation::Insert {
            value, timestamp, ..
        } = compacted
        else {
            panic!("expected live compacted shard");
        };
        assert_eq!(timestamp, 70);
        assert_eq!(
            decode(&value),
            Some(EpochCount {
                epoch: 7,
                count: 100
            })
        );
    }
}
