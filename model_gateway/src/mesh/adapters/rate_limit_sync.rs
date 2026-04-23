//! `rl:` CRDT adapter: gateway ↔ mesh bridge for rate-limit counters.
//!
//! Per-actor sharding is preserved from v1. Each node writes its own
//! count under `rl:{counter}:{node_name}`; cluster-wide totals come
//! from summing shards with [`get_aggregate`](RateLimitSyncAdapter::get_aggregate).
//! The merge strategy for the `rl:` namespace is `EpochMaxWins`
//! (higher epoch wins; max count inside the same epoch), which lets
//! window resets propagate without undoing the reset via a naive
//! `max(old, new)`.
//!
//! Wire format is the shared 16-byte layout from the mesh crate:
//! `u64` big-endian epoch in bytes 0..8, `i64` big-endian count in
//! bytes 8..16. The helpers [`encode_epoch_count`] /
//! [`decode_epoch_count`] match what the mesh merge itself reads, so
//! this adapter never drifts from the merge format.
//!
//! The caller owns the epoch clock — typically
//! `now.as_secs() / window.as_secs()`. `sync_counter` writes the
//! current (epoch, count) for the local shard; a new epoch with
//! count = 0 is how a node signals "window reset for this shard".
//! No window-reset timer lives in the adapter; that scheduling is a
//! concern for whoever drives rate-limit increments.
//!
//! Aggregate reads are epoch-aware: only shards at the highest
//! observed epoch contribute. Stale shards from dead nodes or slow
//! resets are ignored, matching the v1 behaviour after window
//! advance.

use std::sync::Arc;

use bytes::Bytes;
use smg_mesh::{
    decode_epoch_count, encode_epoch_count, CrdtNamespace, EpochCount, EPOCH_MAX_WINS_ENCODED_LEN,
};
use tracing::{debug, warn};

const PREFIX: &str = "rl:";

/// Bridge between the `rl:` CRDT namespace and the gateway's
/// rate-limit enforcement path. Writes the local shard on each
/// increment and reads cluster-wide aggregates on demand.
pub struct RateLimitSyncAdapter {
    rate_limits: Arc<CrdtNamespace>,
    node_name: String,
}

impl std::fmt::Debug for RateLimitSyncAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RateLimitSyncAdapter")
            .field("prefix", &self.rate_limits.prefix())
            .field("node_name", &self.node_name)
            .finish()
    }
}

impl RateLimitSyncAdapter {
    /// Build an adapter wrapping an `rl:`-scoped namespace and the
    /// local node name. Panics if the namespace prefix is wrong
    /// (fails fast at startup instead of routing writes to the wrong
    /// CRDT), or if `node_name` contains `:` — the separator is how
    /// shards are keyed, so a colon in the node name would make
    /// `get_aggregate` filter results ambiguous.
    pub fn new(rate_limits: Arc<CrdtNamespace>, node_name: String) -> Arc<Self> {
        assert_eq!(
            rate_limits.prefix(),
            PREFIX,
            "RateLimitSyncAdapter requires a namespace scoped to `{PREFIX}`",
        );
        assert!(
            !node_name.is_empty(),
            "RateLimitSyncAdapter node_name must not be empty",
        );
        assert!(
            !node_name.contains(':'),
            "RateLimitSyncAdapter node_name must not contain ':' (got {node_name:?})",
        );
        Arc::new(Self {
            rate_limits,
            node_name,
        })
    }

    /// Start the inbound path. Subscribes first so no live event is
    /// lost, spawns the recv loop to flag malformed wire values, and
    /// backfills existing shards on the caller's thread so the recv
    /// loop can drain concurrently on a multi-threaded runtime.
    /// Aggregate reads always hit the CRDT store directly, so
    /// subscription is observability-only today: it logs remote
    /// shards at debug and warns when a value doesn't match the
    /// 16-byte wire format.
    pub fn start(self: &Arc<Self>) {
        let mut sub = self.rate_limits.subscribe("");
        #[expect(
            clippy::disallowed_methods,
            reason = "subscription task ends automatically when the mesh KV drops and closes the channel; no handle needed"
        )]
        tokio::spawn(async move {
            while let Some((key, value)) = sub.receiver.recv().await {
                let Some(shard) = key.strip_prefix(PREFIX).filter(|s| !s.is_empty()) else {
                    warn!(key, "rl: subscription yielded unexpected key shape");
                    continue;
                };
                match value {
                    Some(fragments) => {
                        let total = fragments.iter().map(Bytes::len).sum();
                        let mut bytes = Vec::with_capacity(total);
                        for frag in fragments {
                            bytes.extend_from_slice(&frag);
                        }
                        Self::observe_shard(shard, &bytes);
                    }
                    None => debug!(shard, "remote rate-limit tombstone"),
                }
            }
            debug!("RateLimitSyncAdapter subscription closed");
        });
        self.backfill_existing();
    }

    /// Replay current shards for observability. Same reasoning as
    /// `WorkerSyncAdapter::backfill_existing`: catches data-shape
    /// issues at startup rather than waiting for the next live
    /// write.
    fn backfill_existing(&self) {
        for key in self.rate_limits.keys("") {
            let Some(shard) = key.strip_prefix(PREFIX).filter(|s| !s.is_empty()) else {
                warn!(key, "rl: backfill yielded unexpected key shape");
                continue;
            };
            if let Some(bytes) = self.rate_limits.get(&key) {
                Self::observe_shard(shard, &bytes);
            }
        }
    }

    fn observe_shard(shard: &str, bytes: &[u8]) {
        match decode_epoch_count(bytes) {
            Some(EpochCount { epoch, count }) => {
                debug!(shard, epoch, count, "remote rate-limit shard");
            }
            None => warn!(
                shard,
                len = bytes.len(),
                "rate-limit value must be exactly {EPOCH_MAX_WINS_ENCODED_LEN} bytes",
            ),
        }
    }

    /// Publish this node's shard for a counter at the given epoch
    /// and count. The caller advances the epoch on window rollover
    /// (a new epoch with count = 0 signals reset); inside a window
    /// the count only grows.
    pub fn sync_counter(&self, counter_name: &str, epoch: u64, count: i64) {
        debug_assert!(
            !counter_name.contains(':'),
            "counter_name must not contain ':' (got {counter_name:?})",
        );
        let shard_key = format!("{PREFIX}{counter_name}:{}", self.node_name);
        let bytes = encode_epoch_count(epoch, count).to_vec();
        self.rate_limits.put(&shard_key, bytes);
    }

    /// Cluster-wide aggregate for a counter, summing shards only at
    /// the highest observed epoch. Shards from older epochs (dead
    /// nodes, slow resets) are ignored so the aggregate doesn't mix
    /// windows — a brief under-count while a slow node catches up is
    /// safer than an over-count that triggers rate limiting
    /// prematurely.
    pub fn get_aggregate(&self, counter_name: &str) -> i64 {
        debug_assert!(
            !counter_name.contains(':'),
            "counter_name must not contain ':' (got {counter_name:?})",
        );
        let sub_prefix = format!("{counter_name}:");
        let mut shards = Vec::new();
        for key in self.rate_limits.keys(&sub_prefix) {
            if let Some(bytes) = self.rate_limits.get(&key) {
                if let Some(value) = decode_epoch_count(&bytes) {
                    shards.push(value);
                }
            }
        }
        let Some(max_epoch) = shards.iter().map(|s| s.epoch).max() else {
            return 0;
        };
        // Saturate at i64::MAX instead of wrapping: a signed wrap
        // would flip the aggregate negative and effectively disable
        // rate limiting. Realistic cluster totals are nowhere near
        // overflow, but under-limiting is the failure we must never
        // accept — capping at i64::MAX keeps the gate closed.
        shards
            .iter()
            .filter(|s| s.epoch == max_epoch)
            .try_fold(0i64, |acc, s| acc.checked_add(s.count))
            .unwrap_or(i64::MAX)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use smg_mesh::{MergeStrategy, MeshKV};
    use tokio::time::sleep;

    use super::*;

    fn rl_namespace(mesh: &MeshKV) -> Arc<CrdtNamespace> {
        mesh.configure_crdt_prefix(PREFIX, MergeStrategy::EpochMaxWins)
    }

    #[tokio::test]
    async fn sync_counter_writes_sixteen_byte_big_endian_value() {
        let mesh = MeshKV::new("node-a".into());
        let ns = rl_namespace(&mesh);
        let adapter = RateLimitSyncAdapter::new(ns.clone(), "node-a".into());

        adapter.sync_counter("global", 7, 42);

        let raw = ns
            .get("rl:global:node-a")
            .expect("shard is written under rl:{counter}:{node}");
        assert_eq!(
            raw.len(),
            EPOCH_MAX_WINS_ENCODED_LEN,
            "wire format is fixed-size"
        );
        let decoded = decode_epoch_count(&raw).unwrap();
        assert_eq!(
            decoded,
            EpochCount {
                epoch: 7,
                count: 42
            }
        );
    }

    #[tokio::test]
    async fn get_aggregate_sums_only_max_epoch_shards() {
        // Cluster-wide aggregate must follow the EpochMaxWins
        // discipline: a dead node with a stale epoch cannot inflate
        // the current window.
        let mesh = MeshKV::new("node-a".into());
        let ns = rl_namespace(&mesh);
        let adapter = RateLimitSyncAdapter::new(ns.clone(), "node-a".into());

        // Simulate shards from three nodes: two at the current epoch,
        // one lagging at the previous epoch.
        ns.put("rl:global:node-a", encode_epoch_count(6, 10).to_vec());
        ns.put("rl:global:node-b", encode_epoch_count(6, 8).to_vec());
        ns.put("rl:global:node-c", encode_epoch_count(5, 25).to_vec());

        // Naive sum would be 10 + 8 + 25 = 43. Correct epoch-aware
        // sum is 10 + 8 = 18.
        assert_eq!(adapter.get_aggregate("global"), 18);
    }

    #[tokio::test]
    async fn get_aggregate_ignores_other_counters() {
        let mesh = MeshKV::new("node-a".into());
        let ns = rl_namespace(&mesh);
        let adapter = RateLimitSyncAdapter::new(ns.clone(), "node-a".into());

        ns.put("rl:global:node-a", encode_epoch_count(1, 5).to_vec());
        ns.put("rl:users:node-a", encode_epoch_count(1, 99).to_vec());

        assert_eq!(adapter.get_aggregate("global"), 5);
        assert_eq!(adapter.get_aggregate("users"), 99);
    }

    #[tokio::test]
    async fn get_aggregate_ignores_prefix_overlap() {
        // `global` must not match `globalish`. The trailing `:` in
        // the sub_prefix is what keeps them disjoint.
        let mesh = MeshKV::new("node-a".into());
        let ns = rl_namespace(&mesh);
        let adapter = RateLimitSyncAdapter::new(ns.clone(), "node-a".into());

        ns.put("rl:global:node-a", encode_epoch_count(1, 3).to_vec());
        ns.put("rl:globalish:node-a", encode_epoch_count(1, 100).to_vec());

        assert_eq!(adapter.get_aggregate("global"), 3);
    }

    #[tokio::test]
    async fn get_aggregate_empty_counter_returns_zero() {
        let mesh = MeshKV::new("node-a".into());
        let ns = rl_namespace(&mesh);
        let adapter = RateLimitSyncAdapter::new(ns, "node-a".into());
        assert_eq!(adapter.get_aggregate("never-written"), 0);
    }

    #[tokio::test]
    async fn get_aggregate_skips_malformed_shards() {
        // A corrupt entry must not poison the aggregate — other
        // well-formed shards still contribute.
        let mesh = MeshKV::new("node-a".into());
        let ns = rl_namespace(&mesh);
        let adapter = RateLimitSyncAdapter::new(ns.clone(), "node-a".into());

        ns.put("rl:global:node-a", encode_epoch_count(5, 7).to_vec());
        ns.put("rl:global:node-bad", b"not-16-bytes".to_vec());

        assert_eq!(adapter.get_aggregate("global"), 7);
    }

    #[tokio::test]
    async fn sync_counter_reset_via_new_epoch() {
        // Caller-driven reset: bumping the epoch with count = 0
        // supersedes any older-epoch shard.
        let mesh = MeshKV::new("node-a".into());
        let ns = rl_namespace(&mesh);
        let adapter = RateLimitSyncAdapter::new(ns.clone(), "node-a".into());

        adapter.sync_counter("global", 5, 100);
        assert_eq!(adapter.get_aggregate("global"), 100);

        adapter.sync_counter("global", 6, 0);
        assert_eq!(
            adapter.get_aggregate("global"),
            0,
            "new epoch dominates stale count"
        );
    }

    #[tokio::test]
    async fn start_backfills_existing_shards() {
        // Pre-seed the namespace before starting, exercise the
        // backfill path (observability only — aggregate still works
        // regardless of whether start has run).
        let mesh = MeshKV::new("node-a".into());
        let ns = rl_namespace(&mesh);
        ns.put("rl:global:node-a", encode_epoch_count(1, 5).to_vec());

        let adapter = RateLimitSyncAdapter::new(ns, "node-a".into());
        adapter.start();

        // Aggregation is independent of subscription wiring — it
        // reads straight from the store.
        assert_eq!(adapter.get_aggregate("global"), 5);
        // Give the spawned task a tick to drain any queued events so
        // logs surface in the test output; no functional assertion
        // beyond "doesn't panic or hang".
        sleep(Duration::from_millis(10)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "RateLimitSyncAdapter requires a namespace scoped to `rl:`")]
    async fn new_rejects_wrong_prefix() {
        let mesh = MeshKV::new("node-a".into());
        let ns = mesh.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
        let _ = RateLimitSyncAdapter::new(ns, "node-a".into());
    }

    #[tokio::test]
    #[should_panic(expected = "node_name must not contain ':'")]
    async fn new_rejects_colon_in_node_name() {
        let mesh = MeshKV::new("node-a".into());
        let ns = rl_namespace(&mesh);
        let _ = RateLimitSyncAdapter::new(ns, "bad:node".into());
    }

    #[tokio::test]
    #[should_panic(expected = "node_name must not be empty")]
    async fn new_rejects_empty_node_name() {
        let mesh = MeshKV::new("node-a".into());
        let ns = rl_namespace(&mesh);
        let _ = RateLimitSyncAdapter::new(ns, String::new());
    }

    #[tokio::test]
    async fn get_aggregate_saturates_on_overflow() {
        // Overflow is unreachable with realistic counts, but wrapping
        // a signed sum would flip to negative and disable rate
        // limiting. The aggregator must cap at i64::MAX instead.
        let mesh = MeshKV::new("node-a".into());
        let ns = rl_namespace(&mesh);
        let adapter = RateLimitSyncAdapter::new(ns.clone(), "node-a".into());

        ns.put("rl:global:node-a", encode_epoch_count(1, i64::MAX).to_vec());
        ns.put("rl:global:node-b", encode_epoch_count(1, 1).to_vec());

        assert_eq!(adapter.get_aggregate("global"), i64::MAX);
    }
}
