//! Cluster-wide rate limiting.
//!
//! Each node counts the requests it admits into a wall-clock window and
//! publishes the monotone cumulative count as its `rl:` shard; enforcement
//! compares the epoch-pinned cluster aggregate (plus this node's unflushed
//! delta) against the limit stored under `config:rate_limit`.
//!
//! The EpochMaxWins merge pins the per-shard maximum within an epoch, so a
//! published count must never regress inside a window — the counter only
//! ever increments, and a window roll advances the epoch so the fresh zero
//! replaces (rather than fights) the old peak.
//!
//! Coordination model, stated honestly: windows are wall-clock seconds, so
//! alignment assumes NTP-disciplined clocks. Sub-second boundary jitter
//! costs a brief under-count; a node skewed by a full second or more
//! enforces its own window island (worst case: islands × limit admitted
//! cluster-wide). A peer's spend reaches this node only after that peer's
//! publish (≤ flush interval) AND the next gossip round that ships its
//! op-log (≤ gossip period, ~1s) — the gossip term dominates, and with a
//! 1s window equals it. So the effective cluster guarantee is between
//! `limit` and roughly `limit + nodes × (flush-interval + gossip-period)
//! admits` per window — coordination-free by design, not an exact global
//! counter, and under sustained load it trends toward the per-node-island
//! worst case above. A wall-clock step is absorbed by the rotation's
//! monotone grace: it costs at most ~one grace period, not the step's
//! magnitude.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use smg_mesh::CrdtNamespace;
use tracing::warn;

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
/// undecodable config disables cluster-wide limiting (fail-open; decode
/// failures warn once per failure streak).
pub const RATE_LIMIT_CONFIG_KEY: &str = "config:rate_limit";

/// The cluster-wide counter name every node shards into.
const GLOBAL_COUNTER: &str = "global";

/// Real time a window must survive a wall clock running behind it before
/// the epoch is re-anchored to the wall. Absorbs NTP steps and spurious
/// readings: a backward step or a forward glitch wedges admission for at
/// most this long instead of the step's magnitude.
const REGRESSION_GRACE: Duration = Duration::from_secs(2);

/// Pace of inline shard publishing from the admission path. Faster than
/// the 1s window so peers see this node's spend mid-window (tightening the
/// cross-node overshoot), slow enough that the per-publish CRDT put and
/// op-log append stay negligible. Publishing is traffic-driven, so no
/// background task is needed: an admit advances the count and publishes;
/// once admits stop, the last published value is frozen — it lags the
/// window's true final count by at most one flush interval of admits, and
/// a window roll discards that unpublished tail. Harmless for enforcement
/// (this node's own check adds the unflushed delta; peers under-count by a
/// bounded amount in the over-admit direction), but a future consumer of
/// the `rl:` shard history (observability, billing) would under-read every
/// window's tail — wire a flush-on-roll then.
const FLUSH_INTERVAL: Duration = Duration::from_millis(250);

/// One node's view of the current window.
struct Window {
    /// Wall-clock second this window counts (the EpochMaxWins epoch).
    epoch: u64,
    /// Requests admitted this window on this node. Monotone within the
    /// window — EpochMaxWins pins the per-shard maximum, so a published
    /// count must never regress inside an epoch.
    count: i64,
    /// The count last published via `sync_counter`; the delta above it is
    /// what the cluster aggregate does not see yet.
    flushed: i64,
    /// Monotone moment of the last rotation, the regression detector.
    rotated_at: Instant,
    /// Monotone moment of the last publish, pacing the inline flush.
    flushed_at: Instant,
}

impl Window {
    /// Roll forward when the wall clock entered a new epoch — or re-anchor
    /// to the wall when it has run *behind* the epoch for a full grace of
    /// real time (a backward NTP step, or a prior spurious forward
    /// reading). Re-anchoring may republish a lower count at an epoch this
    /// node already used; EpochMaxWins masks it by keeping the old peak,
    /// which over-counts this node for that one window — the conservative
    /// direction.
    fn rotate(&mut self, now_secs: u64, grace: Duration) {
        let regressed = now_secs < self.epoch && self.rotated_at.elapsed() >= grace;
        if now_secs > self.epoch || regressed {
            self.epoch = now_secs;
            self.count = 0;
            self.flushed = 0;
            self.rotated_at = Instant::now();
        }
    }
}

/// Cluster-wide request limiter. One per gateway, owned by `MeshAdapters`;
/// the concurrency middleware checks it before the local token bucket and
/// records only requests the node actually admits.
pub struct GlobalRateLimiter {
    rate_limit: Arc<RateLimitSyncAdapter>,
    configs: Arc<CrdtNamespace>,
    window: Mutex<Window>,
    regression_grace: Duration,
    /// Latches decode failures so the warn fires once per failure streak.
    config_decode_failed: AtomicBool,
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
        Self::with_regression_grace(rate_limit, configs, REGRESSION_GRACE)
    }

    fn with_regression_grace(
        rate_limit: Arc<RateLimitSyncAdapter>,
        configs: Arc<CrdtNamespace>,
        regression_grace: Duration,
    ) -> Arc<Self> {
        Arc::new(Self {
            rate_limit,
            configs,
            window: Mutex::new(Window {
                epoch: 0,
                count: 0,
                flushed: 0,
                rotated_at: Instant::now(),
                flushed_at: Instant::now(),
            }),
            regression_grace,
            config_decode_failed: AtomicBool::new(false),
        })
    }

    /// The configured cluster-wide limit, if any. Fail-open on a missing
    /// key; a present-but-undecodable value also fails open but warns once
    /// per failure streak — silently disabled limiting must not pass for
    /// unconfigured.
    fn limit(&self) -> Option<u64> {
        let bytes = self.configs.get(RATE_LIMIT_CONFIG_KEY)?;
        match bincode::deserialize::<RateLimitConfig>(&bytes) {
            Ok(config) => {
                self.config_decode_failed.store(false, Ordering::Release);
                Some(config.limit_per_second)
            }
            Err(err) => {
                if !self.config_decode_failed.swap(true, Ordering::AcqRel) {
                    warn!(
                        %err,
                        "config:rate_limit present but undecodable; cluster-wide \
                         rate limiting is DISABLED until it decodes"
                    );
                }
                None
            }
        }
    }

    /// Whether a request arriving at `now_secs` fits the cluster budget.
    /// Does NOT consume budget — the caller counts via
    /// [`Self::record_admit`] only after the request clears local admission
    /// too, so requests the local limiter then rejects never burn the
    /// cluster's budget (a saturated node must not starve its peers).
    ///
    /// `check` and `record_admit` are separate lock acquisitions, so up to
    /// the node's in-flight concurrency can pass `check` before any records;
    /// per-node overshoot is bounded by that concurrency. Deliberate — the
    /// split is what keeps locally-rejected requests from burning budget —
    /// and dwarfed by the gossip-cadence slack documented on the module.
    pub fn check(&self, now_secs: u64) -> bool {
        let Some(limit) = self.limit() else {
            return true;
        };
        let mut window = self.window.lock();
        window.rotate(now_secs, self.regression_grace);
        // Pinned to this window's epoch: after a roll, the stale window's
        // shards must not gate admission (nothing would ever publish the
        // new epoch and the cluster would reject forever).
        let aggregate = self
            .rate_limit
            .get_aggregate_at(GLOBAL_COUNTER, window.epoch);
        let unflushed = (window.count - window.flushed).max(0);
        aggregate.saturating_add(unflushed) < limit as i64
    }

    /// Count one locally-admitted request into the current window,
    /// publishing the grown shard at most once per [`FLUSH_INTERVAL`].
    pub fn record_admit(&self, now_secs: u64) {
        if self.limit().is_none() {
            return;
        }
        let mut window = self.window.lock();
        window.rotate(now_secs, self.regression_grace);
        window.count += 1;
        if window.flushed_at.elapsed() >= FLUSH_INTERVAL {
            self.publish(&mut window);
        }
    }

    /// Force publication regardless of pacing. Production publishes inline
    /// from `record_admit`; only tests need to flush on demand.
    #[cfg(test)]
    pub(crate) fn flush(&self, now_secs: u64) {
        let mut window = self.window.lock();
        window.rotate(now_secs, self.regression_grace);
        self.publish(&mut window);
    }

    // Runs under the window Mutex and calls into the CRDT store
    // (window-lock -> store-lock order). Safe because the store's subscriber
    // notify is `try_send` with no synchronous callback back into the
    // limiter; if a future store path called back in, that order would need
    // revisiting. EpochMaxWins tolerates same-epoch out-of-order publishes,
    // so the put could move outside the lock should this ever contend.
    fn publish(&self, window: &mut Window) {
        if window.count > window.flushed {
            self.rate_limit
                .sync_counter(GLOBAL_COUNTER, window.epoch, window.count);
            window.flushed = window.count;
            window.flushed_at = Instant::now();
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
        limiter_with_grace(limit, REGRESSION_GRACE)
    }

    fn limiter_with_grace(limit: Option<u64>, grace: Duration) -> Arc<GlobalRateLimiter> {
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
        GlobalRateLimiter::with_regression_grace(rate_limit, configs, grace)
    }

    /// The middleware's two-step admission as one call.
    fn admit(limiter: &GlobalRateLimiter, now_secs: u64) -> bool {
        if limiter.check(now_secs) {
            limiter.record_admit(now_secs);
            true
        } else {
            false
        }
    }

    #[test]
    fn no_config_means_unlimited() {
        let limiter = limiter_with(None);
        for _ in 0..1000 {
            assert!(admit(&limiter, 100));
        }
    }

    #[test]
    fn undecodable_config_fails_open() {
        let limiter = limiter_with(None);
        limiter
            .configs
            .put(RATE_LIMIT_CONFIG_KEY, b"not-bincode!".to_vec());
        assert!(admit(&limiter, 100), "corrupt config fails open");
    }

    #[test]
    fn admits_up_to_the_limit_then_rejects() {
        let limiter = limiter_with(Some(5));
        for _ in 0..5 {
            assert!(admit(&limiter, 100), "under the limit admits");
        }
        assert!(!admit(&limiter, 100), "at the limit rejects");
        assert!(!admit(&limiter, 100), "rejections consume no budget");
    }

    #[test]
    fn check_alone_consumes_no_budget() {
        // The middleware checks before local admission and records only
        // after it: requests the local limiter rejects must not burn the
        // cluster budget.
        let limiter = limiter_with(Some(2));
        for _ in 0..100 {
            assert!(limiter.check(100));
        }
        assert!(admit(&limiter, 100));
        assert!(admit(&limiter, 100));
        assert!(!admit(&limiter, 100));
    }

    #[test]
    fn window_roll_resets_the_budget() {
        let limiter = limiter_with(Some(2));
        assert!(admit(&limiter, 100));
        assert!(admit(&limiter, 100));
        assert!(!admit(&limiter, 100));

        assert!(admit(&limiter, 101), "a new window admits again");
    }

    #[test]
    fn flush_publishes_monotone_cumulative_counts() {
        let limiter = limiter_with(Some(100));
        assert!(admit(&limiter, 100));
        assert!(admit(&limiter, 100));
        limiter.flush(100);
        assert_eq!(
            limiter.rate_limit.get_aggregate("global"),
            2,
            "the flushed shard carries the cumulative window count"
        );

        assert!(admit(&limiter, 100));
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
            assert!(admit(&limiter, 100));
        }
        limiter.flush(100);

        // The aggregate carries the flushed 4; six unflushed admits exhaust
        // the cluster budget of 10 without double-counting the shard.
        for _ in 0..6 {
            assert!(admit(&limiter, 100));
        }
        assert!(!admit(&limiter, 100), "cluster budget exhausted");
    }

    #[test]
    fn stale_epoch_shards_do_not_inflate_the_budget_check() {
        let limiter = limiter_with(Some(2));
        assert!(admit(&limiter, 100));
        assert!(admit(&limiter, 100));
        limiter.flush(100);
        assert!(!admit(&limiter, 100), "window 100 exhausted");

        // The flushed shard (epoch 100, count 2) still sits in the store,
        // but the budget check is pinned to the rolled window's epoch — a
        // stale window must never wedge the cluster into rejecting forever.
        assert!(admit(&limiter, 101));
        limiter.flush(101);
        assert_eq!(limiter.rate_limit.get_aggregate("global"), 1);
    }

    #[test]
    fn backward_clock_step_recovers_after_the_grace() {
        // A backward NTP step must wedge admission for at most the
        // regression grace, not the step's magnitude.
        let limiter = limiter_with_grace(Some(2), Duration::from_millis(50));
        assert!(admit(&limiter, 100));
        assert!(admit(&limiter, 100));
        assert!(!admit(&limiter, 100), "window 100 exhausted");

        // The wall clock steps back 50 seconds; within the grace the window
        // stays pinned (and exhausted).
        assert!(!admit(&limiter, 50), "regression inside the grace holds");

        std::thread::sleep(Duration::from_millis(60));
        assert!(
            admit(&limiter, 50),
            "past the grace the window re-anchors to the wall clock"
        );
    }

    #[test]
    fn forward_clock_glitch_recovers_after_the_grace() {
        // One spurious far-future reading must not wedge the node until
        // real time catches up.
        let limiter = limiter_with_grace(Some(2), Duration::from_millis(50));
        assert!(admit(&limiter, 100));
        assert!(admit(&limiter, 4000), "glitch rolls the window forward");

        std::thread::sleep(Duration::from_millis(60));
        assert!(
            admit(&limiter, 101),
            "after the grace the window re-anchors to real wall time"
        );
    }
}
