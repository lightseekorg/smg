use std::collections::HashMap;

use super::operation::Operation;

/// Per-peer CRDT send watermark, keyed by KEY (not by author replica). Tracks
/// the highest version (Lamport timestamp) a peer has acknowledged for each
/// key, so the sender can skip re-broadcasting keys the peer already has.
///
/// Keying by key — not by replica — is what keeps this correct on our
/// compacted, multi-author op-log: each key is tracked independently, so a
/// dropped or late op only delays that one key (resent next round) and can
/// never be hidden behind a higher version on a different key.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CrdtWatermark {
    versions: HashMap<String, u64>,
}

impl CrdtWatermark {
    /// Empty watermark — every op is allowed (a fresh peer gets the full set).
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from a set of ops, keeping the max version per key. Used by the
    /// receiver to ack the versions it just merged.
    pub fn from_ops(ops: &[Operation]) -> Self {
        let mut versions = HashMap::new();
        for op in ops {
            let entry = versions.entry(op.key().to_owned()).or_insert(0);
            *entry = (*entry).max(op.timestamp());
        }
        Self { versions }
    }

    /// True if `op` is newer than what the peer has acked for its key. An
    /// unacked key counts as version 0, so it is always allowed.
    pub fn allows(&self, op: &Operation) -> bool {
        op.timestamp() > self.get(op.key())
    }

    /// The acked version for `key` (0 if never acked).
    pub fn get(&self, key: &str) -> u64 {
        self.versions.get(key).copied().unwrap_or(0)
    }

    /// Advance toward `other`, taking the per-key maximum. Monotone,
    /// idempotent, and commutative, so out-of-order or duplicate acks are
    /// self-correcting.
    pub fn merge_max(&mut self, other: &CrdtWatermark) {
        for (key, &version) in &other.versions {
            let entry = self.versions.entry(key.clone()).or_insert(0);
            *entry = (*entry).max(version);
        }
    }

    /// Iterate `(key, version)` pairs — used by the wire codec.
    pub fn iter(&self) -> impl Iterator<Item = (&str, u64)> {
        self.versions.iter().map(|(k, &v)| (k.as_str(), v))
    }

    /// True if no key has been acked yet.
    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }
}

impl FromIterator<(String, u64)> for CrdtWatermark {
    /// Build from `(key, version)` pairs, keeping the max version per key. Used
    /// by the wire codec to rebuild a watermark from a received ack.
    fn from_iter<I: IntoIterator<Item = (String, u64)>>(iter: I) -> Self {
        let mut versions = HashMap::new();
        for (key, version) in iter {
            let entry = versions.entry(key).or_insert(0);
            *entry = (*entry).max(version);
        }
        Self { versions }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crdt_kv::ReplicaId;

    fn op(key: &str, ts: u64) -> Operation {
        Operation::insert(key.to_owned(), vec![1], ts, ReplicaId::new())
    }

    #[test]
    fn unknown_key_is_version_zero_and_allowed() {
        let wm = CrdtWatermark::new();
        assert_eq!(wm.get("worker:a"), 0);
        assert!(wm.allows(&op("worker:a", 1)));
    }

    #[test]
    fn allows_only_strictly_newer_versions() {
        let wm = CrdtWatermark::from_ops(&[op("k", 5)]);
        assert!(!wm.allows(&op("k", 4)), "older version already acked");
        assert!(!wm.allows(&op("k", 5)), "equal version already acked");
        assert!(wm.allows(&op("k", 6)), "newer version must be sent");
        assert!(
            wm.allows(&op("other", 1)),
            "other key tracked independently"
        );
    }

    #[test]
    fn from_ops_keeps_max_version_per_key() {
        let wm = CrdtWatermark::from_ops(&[op("k", 3), op("k", 9), op("k", 7), op("j", 2)]);
        assert_eq!(wm.get("k"), 9);
        assert_eq!(wm.get("j"), 2);
    }

    #[test]
    fn merge_max_takes_per_key_maximum() {
        let mut a = CrdtWatermark::from_ops(&[op("k", 5), op("j", 8)]);
        let b = CrdtWatermark::from_ops(&[op("k", 3), op("j", 12), op("new", 1)]);
        a.merge_max(&b);
        assert_eq!(a.get("k"), 5, "existing higher value is not lowered");
        assert_eq!(a.get("j"), 12, "existing value is raised");
        assert_eq!(a.get("new"), 1, "new key is added");
    }

    #[test]
    fn merge_max_is_idempotent() {
        let mut a = CrdtWatermark::from_ops(&[op("k", 5)]);
        let b = a.clone();
        a.merge_max(&b);
        a.merge_max(&b);
        assert_eq!(a.get("k"), 5);
    }

    #[test]
    fn iter_yields_all_key_versions() {
        let wm = CrdtWatermark::from_ops(&[op("k", 5), op("j", 8)]);
        let mut pairs: Vec<(String, u64)> = wm.iter().map(|(k, v)| (k.to_owned(), v)).collect();
        pairs.sort();
        assert_eq!(pairs, vec![("j".to_owned(), 8), ("k".to_owned(), 5)]);
    }

    #[test]
    fn is_empty_reflects_contents() {
        assert!(CrdtWatermark::new().is_empty());
        assert!(!CrdtWatermark::from_ops(&[op("k", 1)]).is_empty());
    }

    #[test]
    fn from_iter_takes_max_per_key() {
        let wm: CrdtWatermark = [
            ("k".to_owned(), 3),
            ("k".to_owned(), 9),
            ("j".to_owned(), 2),
        ]
        .into_iter()
        .collect();
        assert_eq!(wm.get("k"), 9);
        assert_eq!(wm.get("j"), 2);
    }
}
