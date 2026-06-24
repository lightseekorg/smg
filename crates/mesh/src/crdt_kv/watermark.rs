use std::collections::HashMap;

use super::{operation::Operation, replica::ReplicaId};

/// A CRDT write's identity and order: `(Lamport timestamp, replica_id)`. This
/// is exactly the key LWW uses to pick a winner (`max_by_key((timestamp,
/// replica_id))`), so a watermark built from it advances in lockstep with the
/// merge — including the replica tie-break when two writes share a timestamp.
pub type CrdtVersion = (u64, ReplicaId);

/// Per-peer CRDT send watermark, keyed by KEY (not by author replica). Tracks
/// the op-id `(timestamp, replica_id)` a peer has acknowledged for each key, so
/// the sender can skip re-broadcasting keys the peer already has.
///
/// Keying by key — not by replica — keeps this correct on our compacted,
/// multi-author op-log: each key is tracked independently, so a dropped or late
/// op only delays that one key (resent next round) and can never be hidden
/// behind a higher version on a different key.
///
/// The op-id, not just the timestamp, is the watermark value: equal timestamps
/// from different replicas are ordered by `replica_id` (the LWW tie-break), so
/// a same-timestamp winner change must still be sent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CrdtWatermark {
    versions: HashMap<String, CrdtVersion>,
}

impl CrdtWatermark {
    /// Empty watermark — every op is allowed (a fresh peer gets the full set).
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from a set of ops, keeping the max op-id per key. Used by the
    /// receiver to ack the versions it just merged. Clones a key only when
    /// first seen, so allocation is O(distinct keys), not O(ops).
    pub fn from_ops(ops: &[Operation]) -> Self {
        let mut versions: HashMap<String, CrdtVersion> = HashMap::new();
        for op in ops {
            let id = (op.timestamp(), op.replica_id());
            match versions.get_mut(op.key()) {
                Some(existing) => *existing = (*existing).max(id),
                None => {
                    versions.insert(op.key().to_owned(), id);
                }
            }
        }
        Self { versions }
    }

    /// True if `op` is a newer write than what the peer has acked for its key,
    /// by `(timestamp, replica_id)` order. An unacked key always sends.
    pub fn allows(&self, op: &Operation) -> bool {
        match self.get(op.key()) {
            Some(acked) => (op.timestamp(), op.replica_id()) > acked,
            None => true,
        }
    }

    /// The acked op-id for `key`, if the peer has acked it.
    pub fn get(&self, key: &str) -> Option<CrdtVersion> {
        self.versions.get(key).copied()
    }

    /// Advance toward `other`, taking the per-key maximum op-id. Monotone,
    /// idempotent, and commutative, so out-of-order or duplicate acks are
    /// self-correcting.
    pub fn merge_max(&mut self, other: &CrdtWatermark) {
        for (key, &id) in &other.versions {
            match self.versions.get_mut(key) {
                Some(existing) => *existing = (*existing).max(id),
                None => {
                    self.versions.insert(key.clone(), id);
                }
            }
        }
    }

    /// Advance from owned `(key, op-id)` pairs in place, taking the key only
    /// when inserting a new entry. Avoids cloning keys / a temp map when
    /// merging a received ack on the hot path.
    pub fn merge_max_owned<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (String, CrdtVersion)>,
    {
        for (key, id) in iter {
            match self.versions.get_mut(&key) {
                Some(existing) => *existing = (*existing).max(id),
                None => {
                    self.versions.insert(key, id);
                }
            }
        }
    }

    /// Iterate `(key, op-id)` pairs — used by the wire codec.
    pub fn iter(&self) -> impl Iterator<Item = (&str, CrdtVersion)> {
        self.versions.iter().map(|(k, &id)| (k.as_str(), id))
    }

    /// True if no key has been acked yet.
    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }
}

impl FromIterator<(String, CrdtVersion)> for CrdtWatermark {
    /// Build from `(key, op-id)` pairs, keeping the max op-id per key. Used by
    /// the wire codec to rebuild a watermark from a received ack.
    fn from_iter<I: IntoIterator<Item = (String, CrdtVersion)>>(iter: I) -> Self {
        let mut watermark = Self::new();
        watermark.merge_max_owned(iter);
        watermark
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic replica id `00…00NN` so tie-break tests are reproducible
    /// (higher `n` => higher `ReplicaId`).
    fn replica(n: u8) -> ReplicaId {
        ReplicaId::from_string(&format!("00000000-0000-0000-0000-0000000000{n:02x}")).unwrap()
    }

    fn op(key: &str, ts: u64) -> Operation {
        op_r(key, ts, 1)
    }

    fn op_r(key: &str, ts: u64, n: u8) -> Operation {
        Operation::insert(key.to_owned(), vec![1], ts, replica(n))
    }

    #[test]
    fn unknown_key_is_unset_and_allowed() {
        let wm = CrdtWatermark::new();
        assert_eq!(wm.get("worker:a"), None);
        assert!(wm.allows(&op("worker:a", 1)));
    }

    #[test]
    fn allows_only_strictly_newer_versions() {
        let wm = CrdtWatermark::from_ops(&[op("k", 5)]);
        assert!(!wm.allows(&op("k", 4)), "older version already acked");
        assert!(!wm.allows(&op("k", 5)), "equal op-id already acked");
        assert!(wm.allows(&op("k", 6)), "newer version must be sent");
        assert!(
            wm.allows(&op("other", 1)),
            "other key tracked independently"
        );
    }

    #[test]
    fn same_timestamp_higher_replica_is_a_newer_winner() {
        // The replica tie-break: at equal timestamps LWW orders by replica_id,
        // so a same-timestamp, higher-replica write is a newer winner and must
        // be sent — a timestamp-only watermark would wrongly drop it.
        let wm = CrdtWatermark::from_ops(&[op_r("k", 5, 1)]);
        assert!(!wm.allows(&op_r("k", 5, 1)), "same op-id already acked");
        assert!(
            wm.allows(&op_r("k", 5, 2)),
            "same timestamp, higher replica is a newer winner"
        );
        assert!(
            !wm.allows(&op_r("k", 5, 0)),
            "same timestamp, lower replica is older"
        );
    }

    #[test]
    fn from_ops_keeps_max_op_id_per_key() {
        let wm = CrdtWatermark::from_ops(&[op("k", 3), op("k", 9), op("k", 7), op("j", 2)]);
        assert_eq!(wm.get("k"), Some((9, replica(1))));
        assert_eq!(wm.get("j"), Some((2, replica(1))));
    }

    #[test]
    fn merge_max_takes_per_key_maximum() {
        let mut a = CrdtWatermark::from_ops(&[op("k", 5), op("j", 8)]);
        let b = CrdtWatermark::from_ops(&[op("k", 3), op("j", 12), op("new", 1)]);
        a.merge_max(&b);
        assert_eq!(
            a.get("k"),
            Some((5, replica(1))),
            "higher op-id not lowered"
        );
        assert_eq!(a.get("j"), Some((12, replica(1))), "op-id raised");
        assert_eq!(a.get("new"), Some((1, replica(1))), "new key added");
    }

    #[test]
    fn merge_max_is_idempotent() {
        let mut a = CrdtWatermark::from_ops(&[op("k", 5)]);
        let b = a.clone();
        a.merge_max(&b);
        a.merge_max(&b);
        assert_eq!(a.get("k"), Some((5, replica(1))));
    }

    #[test]
    fn merge_max_owned_takes_per_key_maximum() {
        let mut a = CrdtWatermark::from_ops(&[op("k", 5)]);
        a.merge_max_owned([
            ("k".to_owned(), (3, replica(1))),
            ("j".to_owned(), (8, replica(2))),
        ]);
        assert_eq!(a.get("k"), Some((5, replica(1))), "lower op-id ignored");
        assert_eq!(a.get("j"), Some((8, replica(2))), "new key added");
    }

    #[test]
    fn iter_yields_all_key_versions() {
        let wm = CrdtWatermark::from_ops(&[op("k", 5), op("j", 8)]);
        let mut pairs: Vec<(String, CrdtVersion)> =
            wm.iter().map(|(k, id)| (k.to_owned(), id)).collect();
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                ("j".to_owned(), (8, replica(1))),
                ("k".to_owned(), (5, replica(1))),
            ]
        );
    }

    #[test]
    fn is_empty_reflects_contents() {
        assert!(CrdtWatermark::new().is_empty());
        assert!(!CrdtWatermark::from_ops(&[op("k", 1)]).is_empty());
    }

    #[test]
    fn from_iter_takes_max_per_key() {
        let wm: CrdtWatermark = [
            ("k".to_owned(), (3, replica(1))),
            ("k".to_owned(), (9, replica(1))),
            ("j".to_owned(), (2, replica(1))),
        ]
        .into_iter()
        .collect();
        assert_eq!(wm.get("k"), Some((9, replica(1))));
        assert_eq!(wm.get("j"), Some((2, replica(1))));
    }
}
