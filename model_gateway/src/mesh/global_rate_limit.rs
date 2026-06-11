//! Cluster-wide rate limiting (d-3b).
//!
//! Each node counts the requests it admits into a wall-clock window and
//! publishes the monotone cumulative count as its `rl:` shard; enforcement
//! compares the epoch-aligned cluster aggregate (plus this node's unflushed
//! delta) against the limit stored under `config:rate_limit`.
//!
//! The EpochMaxWins merge pins the per-shard maximum within an epoch, so a
//! published count must never regress inside a window — the counter only
//! ever increments, and a window roll advances the epoch so the fresh zero
//! replaces (rather than fights) the old peak. Windows are wall-clock
//! seconds so they align across nodes without coordination; clock skew at a
//! boundary briefly under-counts, which the aggregate's max-epoch rule
//! already tolerates by design.

use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use smg_mesh::CrdtNamespace;

use super::adapters::RateLimitSyncAdapter;

/// Cluster-wide rate-limit configuration, stored under
/// [`RATE_LIMIT_CONFIG_KEY`] (LWW) and read per request — a DashMap lookup,
/// not a network call.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum admitted requests per second across the whole cluster.
    pub limit_per_second: u64,
}

/// Config key carrying a bincode-encoded [`RateLimitConfig`]. Absent or
/// undecodable config disables cluster-wide limiting.
pub const RATE_LIMIT_CONFIG_KEY: &str = "config:rate_limit";

/// The cluster-wide counter name every node shards into.
const GLOBAL_COUNTER: &str = "global";

/// One node's view of the current window.
struct Window {
    /// Wall-clock second this window counts (the EpochMaxWins epoch).
    epoch: u64,
    /// Requests admitted this window on this node. Monotone within the
    /// window (the d-3b constraint).
    count: i64,
    /// The count last published via `sync_counter`; the delta above it is
    /// what the cluster aggregate does not see yet.
    flushed: i64,
}

impl Window {
    /// Roll forward when the wall clock entered a new epoch.
    fn rotate(&mut self, now_secs: u64) {
        if now_secs > self.epoch {
            self.epoch = now_secs;
            self.count = 0;
            self.flushed = 0;
        }
    }
}

/// Cluster-wide request limiter. One per gateway, owned by `MeshAdapters`;
/// the concurrency middleware consults it before the local token bucket.
pub struct GlobalRateLimiter {
    rate_limit: Arc<RateLimitSyncAdapter>,
    configs: Arc<CrdtNamespace>,
    window: Mutex<Window>,
}

impl std::fmt::Debug for GlobalRateLimiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlobalRateLimiter").finish_non_exhaustive()
    }
}

impl GlobalRateLimiter {
    pub(crate) fn new(
        rate_limit: Arc<RateLimitSyncAdapter>,
        configs: Arc<CrdtNamespace>,
    ) -> Arc<Self> {
        Arc::new(Self {
            rate_limit,
            configs,
            window: Mutex::new(Window {
                epoch: 0,
                count: 0,
                flushed: 0,
            }),
        })
    }

    /// The configured cluster-wide limit, if any.
    fn limit(&self) -> Option<u64> {
        self.configs
            .get(RATE_LIMIT_CONFIG_KEY)
            .and_then(|bytes| bincode::deserialize::<RateLimitConfig>(&bytes).ok())
            .map(|config| config.limit_per_second)
    }

    /// Admit or reject a request arriving at `now_secs`, counting it only
    /// when admitted (rejected requests do not consume budget). The check
    /// is the epoch-aligned cluster aggregate — which already includes this
    /// node's last-flushed shard — plus the local delta not yet flushed.
    pub fn try_admit(&self, now_secs: u64) -> bool {
        let Some(limit) = self.limit() else {
            return true;
        };
        let mut window = self.window.lock();
        window.rotate(now_secs);
        // Pinned to this window's epoch: after a roll, the stale window's
        // shards must not gate admission (nothing would ever publish the
        // new epoch and the cluster would reject forever).
        let aggregate = self
            .rate_limit
            .get_aggregate_at(GLOBAL_COUNTER, window.epoch);
        let unflushed = (window.count - window.flushed).max(0);
        if aggregate.saturating_add(unflushed) >= limit as i64 {
            return false;
        }
        window.count += 1;
        true
    }

    /// Publish this node's shard when it grew. Driven once per second by
    /// the flush task `MeshAdapters::start` spawns.
    pub(crate) fn flush(&self, now_secs: u64) {
        let mut window = self.window.lock();
        window.rotate(now_secs);
        if window.count > window.flushed {
            self.rate_limit
                .sync_counter(GLOBAL_COUNTER, window.epoch, window.count);
            window.flushed = window.count;
        }
    }
}

/// Wall-clock seconds since the unix epoch; the cross-node window identity.
pub(crate) fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use smg_mesh::{MergeStrategy, MeshKV};

    use super::*;

    fn limiter_with(limit: Option<u64>) -> Arc<GlobalRateLimiter> {
        let mesh = MeshKV::new("node-a".into());
        let rl_ns = mesh.configure_crdt_prefix("rl:", MergeStrategy::EpochMaxWins);
        let rate_limit = RateLimitSyncAdapter::new(rl_ns, "node-a".into());
        let configs = mesh.configs();
        if let Some(limit_per_second) = limit {
            let config = RateLimitConfig { limit_per_second };
            configs.put(
                RATE_LIMIT_CONFIG_KEY,
                bincode::serialize(&config).expect("config encodes"),
            );
        }
        GlobalRateLimiter::new(rate_limit, configs)
    }

    #[test]
    fn no_config_means_unlimited() {
        let limiter = limiter_with(None);
        for _ in 0..1000 {
            assert!(limiter.try_admit(100));
        }
    }

    #[test]
    fn admits_up_to_the_limit_then_rejects() {
        let limiter = limiter_with(Some(5));
        for _ in 0..5 {
            assert!(limiter.try_admit(100), "under the limit admits");
        }
        assert!(!limiter.try_admit(100), "at the limit rejects");
        assert!(!limiter.try_admit(100), "rejections consume no budget");
    }

    #[test]
    fn window_roll_resets_the_budget() {
        let limiter = limiter_with(Some(2));
        assert!(limiter.try_admit(100));
        assert!(limiter.try_admit(100));
        assert!(!limiter.try_admit(100));

        assert!(limiter.try_admit(101), "a new window admits again");
    }

    #[test]
    fn flush_publishes_monotone_cumulative_counts() {
        let limiter = limiter_with(Some(100));
        assert!(limiter.try_admit(100));
        assert!(limiter.try_admit(100));
        limiter.flush(100);
        assert_eq!(
            limiter.rate_limit.get_aggregate("global"),
            2,
            "the flushed shard carries the cumulative window count"
        );

        assert!(limiter.try_admit(100));
        limiter.flush(100);
        assert_eq!(
            limiter.rate_limit.get_aggregate("global"),
            3,
            "the count grows monotonically within the window"
        );
    }

    #[test]
    fn enforcement_combines_aggregate_and_unflushed_delta() {
        let limiter = limiter_with(Some(10));
        for _ in 0..4 {
            assert!(limiter.try_admit(100));
        }
        limiter.flush(100);

        // The aggregate carries the flushed 4; six unflushed admits exhaust
        // the cluster budget of 10 without double-counting the shard.
        for _ in 0..6 {
            assert!(limiter.try_admit(100));
        }
        assert!(!limiter.try_admit(100), "cluster budget exhausted");
    }

    #[test]
    fn unflushed_delta_counts_against_the_limit() {
        let limiter = limiter_with(Some(3));
        assert!(limiter.try_admit(100));
        assert!(limiter.try_admit(100));
        assert!(limiter.try_admit(100));
        // Nothing flushed yet: the local delta alone must enforce.
        assert!(!limiter.try_admit(100));
    }

    #[test]
    fn stale_epoch_shards_do_not_inflate_the_budget_check() {
        let limiter = limiter_with(Some(2));
        assert!(limiter.try_admit(100));
        assert!(limiter.try_admit(100));
        limiter.flush(100);
        assert!(!limiter.try_admit(100), "window 100 exhausted");

        // The flushed shard (epoch 100, count 2) still sits in the store,
        // but the budget check is pinned to the rolled window's epoch — a
        // stale window must never wedge the cluster into rejecting forever.
        assert!(limiter.try_admit(101));
        limiter.flush(101);
        assert_eq!(limiter.rate_limit.get_aggregate("global"), 1);
    }
}
