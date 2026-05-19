use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use dashmap::DashMap;
use parking_lot::Mutex;

use super::types::{MultiTenantRateLimitConfig, TenantTokenPolicy};

pub const TERMINAL_REJECTION_RETRY_AFTER_SECS: u64 = u64::MAX;
const STALE_BUCKET_TTL_MULTIPLIER: u32 = 10;
const CLEANUP_INTERVAL: Duration = Duration::from_secs(60);

#[derive(Debug)]
struct BucketState {
    available_tokens: f64,
    available_requests: f64,
    last_refill: Instant,
}

#[derive(Debug)]
struct Bucket {
    policy: TenantTokenPolicy,
    state: Mutex<BucketState>,
}

/// In-memory tenant token bucket rate limiter.
///
/// Bucket cardinality grows with the number of distinct `tenant_key` values observed.
/// To avoid unbounded growth from untrusted tenant identifiers, pair this limiter with
/// upstream tenant validation or keep `RouterConfigBuilder::trust_tenant_header(false)`.
///
/// Each bucket stores a cloned `TenantTokenPolicy` at allocation time. Runtime policy
/// updates do not affect already-allocated buckets until they are evicted by cleanup or
/// otherwise replaced, so policy-change flows may need explicit bucket invalidation.
#[derive(Debug)]
pub struct LocalTokenRateLimiter {
    config: MultiTenantRateLimitConfig,
    buckets: DashMap<String, Arc<Bucket>>,
    started_at: Instant,
    last_cleanup_ms: AtomicU64,
}

impl LocalTokenRateLimiter {
    #[must_use]
    pub fn new(config: MultiTenantRateLimitConfig) -> Self {
        Self {
            config,
            buckets: DashMap::new(),
            started_at: Instant::now(),
            last_cleanup_ms: AtomicU64::new(0),
        }
    }

    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    pub fn check_and_consume(&self, tenant_key: &str, estimated_tokens: u32) -> Result<(), u64> {
        self.maybe_cleanup_stale_buckets();
        let Some(policy) = self.config.policy_for(tenant_key) else {
            return Ok(());
        };

        if policy.tokens_per_minute == 0 && policy.requests_per_minute == 0 {
            return Ok(());
        }

        let bucket = if let Some(bucket) = self.buckets.get(tenant_key) {
            bucket.value().clone()
        } else {
            self.buckets
                .entry(tenant_key.to_string())
                .or_insert_with(|| {
                    Arc::new(Bucket {
                        policy: policy.clone(),
                        state: Mutex::new(BucketState {
                            available_tokens: policy.tokens_per_minute as f64,
                            available_requests: policy.requests_per_minute as f64,
                            last_refill: Instant::now(),
                        }),
                    })
                })
                .value()
                .clone()
        };

        let mut state = bucket.state.lock();
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_refill).as_secs_f64();
        state.last_refill = now;

        if bucket.policy.tokens_per_minute > 0 {
            let refill = elapsed * (bucket.policy.tokens_per_minute as f64 / 60.0);
            state.available_tokens =
                (state.available_tokens + refill).min(bucket.policy.tokens_per_minute as f64);
        }
        if bucket.policy.requests_per_minute > 0 {
            let refill = elapsed * (bucket.policy.requests_per_minute as f64 / 60.0);
            state.available_requests =
                (state.available_requests + refill).min(bucket.policy.requests_per_minute as f64);
        }

        let need_tokens = estimated_tokens as f64;
        let exceeds_token_capacity = bucket.policy.tokens_per_minute > 0
            && need_tokens > bucket.policy.tokens_per_minute as f64;

        if exceeds_token_capacity {
            return Err(TERMINAL_REJECTION_RETRY_AFTER_SECS);
        }

        let token_denied =
            bucket.policy.tokens_per_minute > 0 && state.available_tokens < need_tokens;
        let request_denied =
            bucket.policy.requests_per_minute > 0 && state.available_requests < 1.0;

        if token_denied || request_denied {
            let token_retry = if token_denied && bucket.policy.tokens_per_minute > 0 {
                let debt = (need_tokens - state.available_tokens).max(0.0);
                (debt / (bucket.policy.tokens_per_minute as f64 / 60.0)).ceil() as u64
            } else {
                0
            };
            let request_retry = if request_denied && bucket.policy.requests_per_minute > 0 {
                let debt = (1.0 - state.available_requests).max(0.0);
                (debt / (bucket.policy.requests_per_minute as f64 / 60.0)).ceil() as u64
            } else {
                0
            };
            return Err(token_retry.max(request_retry).max(1));
        }

        if bucket.policy.tokens_per_minute > 0 {
            state.available_tokens -= need_tokens;
        }
        if bucket.policy.requests_per_minute > 0 {
            state.available_requests -= 1.0;
        }

        Ok(())
    }

    fn maybe_cleanup_stale_buckets(&self) {
        let now = Instant::now();
        let now_ms = duration_millis(now.duration_since(self.started_at));
        let cleanup_interval_ms = duration_millis(CLEANUP_INTERVAL);
        let last_cleanup_ms = self.last_cleanup_ms.load(Ordering::Relaxed);

        if now_ms.saturating_sub(last_cleanup_ms) < cleanup_interval_ms {
            return;
        }

        if self
            .last_cleanup_ms
            .compare_exchange(last_cleanup_ms, now_ms, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return;
        }

        self.cleanup_stale_buckets_at(now);
    }

    fn cleanup_stale_buckets_at(&self, now: Instant) {
        let stale_keys: Vec<String> = self
            .buckets
            .iter()
            .filter_map(|entry| {
                let state = entry.value().state.lock();
                let ttl = stale_bucket_ttl(&entry.value().policy);
                (now.duration_since(state.last_refill) > ttl).then(|| entry.key().clone())
            })
            .collect();

        for key in stale_keys {
            let _ = self.buckets.remove(&key);
        }
    }

    #[cfg(test)]
    fn force_cleanup_stale_buckets(&self) {
        self.cleanup_stale_buckets_at(Instant::now());
    }
}

fn duration_millis(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

fn stale_bucket_ttl(policy: &TenantTokenPolicy) -> Duration {
    let token_window_secs = (policy.tokens_per_minute > 0).then_some(60);
    let request_window_secs = (policy.requests_per_minute > 0).then_some(60);
    let refill_window_secs = token_window_secs
        .into_iter()
        .chain(request_window_secs)
        .max()
        .unwrap_or(60);
    Duration::from_secs(u64::from(refill_window_secs * STALE_BUCKET_TTL_MULTIPLIER))
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        thread,
        time::{Duration, Instant},
    };

    use super::{
        LocalTokenRateLimiter, STALE_BUCKET_TTL_MULTIPLIER, TERMINAL_REJECTION_RETRY_AFTER_SECS,
    };
    use crate::rate_limit::types::{MultiTenantRateLimitConfig, TenantTokenPolicy};

    fn limiter(policy: TenantTokenPolicy) -> LocalTokenRateLimiter {
        LocalTokenRateLimiter::new(MultiTenantRateLimitConfig {
            enabled: true,
            default_tokens_per_minute: 0,
            default_requests_per_minute: 0,
            tenants: HashMap::from([("tenant-a".to_string(), policy)]),
        })
    }

    #[test]
    fn allows_unknown_tenant_without_policy() {
        let limiter = LocalTokenRateLimiter::new(MultiTenantRateLimitConfig {
            enabled: true,
            default_tokens_per_minute: 0,
            default_requests_per_minute: 0,
            tenants: HashMap::new(),
        });

        assert!(limiter.check_and_consume("tenant-a", 10).is_ok());
    }

    #[test]
    fn enforces_request_limit_per_tenant() {
        let limiter = limiter(TenantTokenPolicy {
            tokens_per_minute: 0,
            requests_per_minute: 1,
        });

        assert!(limiter.check_and_consume("tenant-a", 0).is_ok());
        let retry_after = limiter
            .check_and_consume("tenant-a", 0)
            .expect_err("second request should be rejected");

        assert_eq!(retry_after, 60);
    }

    #[test]
    fn enforces_token_limit_per_tenant() {
        let limiter = limiter(TenantTokenPolicy {
            tokens_per_minute: 5,
            requests_per_minute: 0,
        });

        assert!(limiter.check_and_consume("tenant-a", 3).is_ok());
        let retry_after = limiter
            .check_and_consume("tenant-a", 3)
            .expect_err("second token-heavy request should be rejected");

        assert_eq!(retry_after, 12);
    }

    #[test]
    fn rejects_requests_larger_than_token_bucket_capacity_as_terminal() {
        let limiter = limiter(TenantTokenPolicy {
            tokens_per_minute: 5,
            requests_per_minute: 0,
        });

        let retry_after = limiter
            .check_and_consume("tenant-a", 6)
            .expect_err("request larger than full bucket capacity should be terminal");

        assert_eq!(retry_after, TERMINAL_REJECTION_RETRY_AFTER_SECS);
    }

    #[test]
    fn tracks_tenants_independently() {
        let limiter = LocalTokenRateLimiter::new(MultiTenantRateLimitConfig {
            enabled: true,
            default_tokens_per_minute: 0,
            default_requests_per_minute: 0,
            tenants: HashMap::from([
                (
                    "tenant-a".to_string(),
                    TenantTokenPolicy {
                        tokens_per_minute: 0,
                        requests_per_minute: 1,
                    },
                ),
                (
                    "tenant-b".to_string(),
                    TenantTokenPolicy {
                        tokens_per_minute: 0,
                        requests_per_minute: 1,
                    },
                ),
            ]),
        });

        assert!(limiter.check_and_consume("tenant-a", 0).is_ok());
        assert!(limiter.check_and_consume("tenant-b", 0).is_ok());
        assert!(limiter.check_and_consume("tenant-a", 0).is_err());
        assert!(limiter.check_and_consume("tenant-b", 0).is_err());
    }

    #[test]
    fn evicts_stale_buckets_during_periodic_cleanup() {
        let limiter = LocalTokenRateLimiter::new(MultiTenantRateLimitConfig {
            enabled: true,
            default_tokens_per_minute: 0,
            default_requests_per_minute: 0,
            tenants: HashMap::from([
                (
                    "tenant-a".to_string(),
                    TenantTokenPolicy {
                        tokens_per_minute: 60,
                        requests_per_minute: 0,
                    },
                ),
                (
                    "tenant-b".to_string(),
                    TenantTokenPolicy {
                        tokens_per_minute: 60,
                        requests_per_minute: 0,
                    },
                ),
            ]),
        });

        assert!(limiter.check_and_consume("tenant-a", 1).is_ok());
        {
            let bucket = limiter
                .buckets
                .get("tenant-a")
                .expect("bucket should exist");
            let mut state = bucket.state.lock();
            state.last_refill = Instant::now()
                - Duration::from_secs(u64::from(60 * STALE_BUCKET_TTL_MULTIPLIER + 1));
        }
        limiter.force_cleanup_stale_buckets();

        assert!(limiter.buckets.get("tenant-a").is_none());
    }

    #[test]
    fn refills_capacity_over_time() {
        let limiter = limiter(TenantTokenPolicy {
            tokens_per_minute: 60,
            requests_per_minute: 60,
        });

        assert!(limiter.check_and_consume("tenant-a", 60).is_ok());
        let retry_after = limiter
            .check_and_consume("tenant-a", 1)
            .expect_err("bucket should be empty immediately after consuming full capacity");
        assert_eq!(retry_after, 1);

        thread::sleep(Duration::from_millis(1100));

        assert!(limiter.check_and_consume("tenant-a", 1).is_ok());
    }
}
