use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

/// Default forwarding failures before a remote region breaker opens.
const DEFAULT_FAILURE_THRESHOLD: u32 = 1;
/// Default successful half-open probes before a remote region breaker closes.
const DEFAULT_SUCCESS_THRESHOLD: u32 = 1;
/// Default time an open remote-region breaker suppresses attempts.
const DEFAULT_OPEN_TIMEOUT_MS: u64 = 30_000;

/// Per-remote-region request-forward circuit breaker state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BreakerState {
    #[default]
    Closed,
    HalfOpen,
    Open,
}

/// Runtime configuration for the per-remote-region request-forward breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrossRegionBreakerConfig {
    failure_threshold: u32,
    success_threshold: u32,
    open_timeout_ms: u64,
}

impl Default for CrossRegionBreakerConfig {
    /// Create the default request-forward breaker configuration.
    fn default() -> Self {
        Self {
            failure_threshold: DEFAULT_FAILURE_THRESHOLD,
            success_threshold: DEFAULT_SUCCESS_THRESHOLD,
            open_timeout_ms: DEFAULT_OPEN_TIMEOUT_MS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RegionBreakerState {
    state: BreakerState,
    consecutive_failures: u32,
    consecutive_successes: u32,
    opened_at_ms: Option<u64>,
}

impl RegionBreakerState {
    /// Create per-region breaker counters from an initial state.
    fn new(state: BreakerState) -> Self {
        Self {
            state,
            consecutive_failures: 0,
            consecutive_successes: 0,
            opened_at_ms: None,
        }
    }
}

/// Per-target-region request-forward breaker.
#[derive(Debug, Clone)]
pub struct CrossRegionBreaker {
    default_state: BreakerState,
    config: CrossRegionBreakerConfig,
    regions: Arc<RwLock<HashMap<String, RegionBreakerState>>>,
}

impl Default for CrossRegionBreaker {
    /// Create a breaker whose default remote-region state is closed.
    fn default() -> Self {
        Self {
            default_state: BreakerState::Closed,
            config: CrossRegionBreakerConfig::default(),
            regions: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl CrossRegionBreaker {
    /// Create a no-op breaker that allows all attempts.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a breaker facade with a fixed default state for every remote region.
    pub fn with_default_state(default_state: BreakerState) -> Self {
        Self {
            default_state,
            ..Self::default()
        }
    }

    /// Create a breaker with a custom failure threshold.
    pub fn with_failure_threshold(failure_threshold: u32) -> Self {
        let mut breaker = Self::new();
        breaker.config.failure_threshold = failure_threshold.max(1);
        breaker
    }

    /// Override the open-state timeout in milliseconds.
    pub fn with_open_timeout_ms(mut self, open_timeout_ms: u64) -> Self {
        self.config.open_timeout_ms = open_timeout_ms;
        self
    }

    /// Return the current state for a region; the skeleton defaults to closed.
    pub fn state_for(&self, region_id: &str) -> BreakerState {
        let mut regions = self
            .regions
            .write()
            .expect("cross-region breaker lock should not be poisoned");
        let Some(region) = regions.get_mut(region_id) else {
            return self.default_state;
        };
        self.refresh_open_region(region);
        region.state
    }

    /// Return true when candidate calculation may attempt the remote region.
    pub fn can_attempt(&self, region_id: &str) -> bool {
        self.state_for(region_id) != BreakerState::Open
    }

    /// Record a successful request-forward attempt for a target region.
    pub fn record_success(&self, region_id: &str) {
        let mut regions = self
            .regions
            .write()
            .expect("cross-region breaker lock should not be poisoned");
        let region = regions
            .entry(region_id.to_string())
            .or_insert_with(|| RegionBreakerState::new(self.default_state));
        self.refresh_open_region(region);
        match region.state {
            BreakerState::Closed => {
                region.consecutive_failures = 0;
            }
            BreakerState::HalfOpen => {
                region.consecutive_successes = region.consecutive_successes.saturating_add(1);
                region.consecutive_failures = 0;
                if region.consecutive_successes >= self.config.success_threshold.max(1) {
                    close_region(region);
                }
            }
            BreakerState::Open => {}
        }
    }

    /// Record a failed request-forward attempt for a target region.
    pub fn record_failure(&self, region_id: &str) {
        let mut regions = self
            .regions
            .write()
            .expect("cross-region breaker lock should not be poisoned");
        let region = regions
            .entry(region_id.to_string())
            .or_insert_with(|| RegionBreakerState::new(self.default_state));
        self.refresh_open_region(region);
        match region.state {
            BreakerState::Closed | BreakerState::HalfOpen => {
                region.consecutive_failures = region.consecutive_failures.saturating_add(1);
                region.consecutive_successes = 0;
                if region.state == BreakerState::HalfOpen
                    || region.consecutive_failures >= self.config.failure_threshold.max(1)
                {
                    open_region(region);
                }
            }
            BreakerState::Open => {}
        }
    }

    /// Move an expired open breaker into half-open probe state.
    fn refresh_open_region(&self, region: &mut RegionBreakerState) {
        if region.state != BreakerState::Open {
            return;
        }
        let Some(opened_at_ms) = region.opened_at_ms else {
            return;
        };
        if now_ms().saturating_sub(opened_at_ms) >= self.config.open_timeout_ms {
            region.state = BreakerState::HalfOpen;
            region.consecutive_failures = 0;
            region.consecutive_successes = 0;
            region.opened_at_ms = None;
        }
    }
}

/// Move a per-region breaker into open state after forwarding failure.
fn open_region(region: &mut RegionBreakerState) {
    region.state = BreakerState::Open;
    region.consecutive_successes = 0;
    region.opened_at_ms = Some(now_ms());
}

/// Move a per-region breaker back to closed state after a successful probe.
fn close_region(region: &mut RegionBreakerState) {
    region.state = BreakerState::Closed;
    region.consecutive_failures = 0;
    region.consecutive_successes = 0;
    region.opened_at_ms = None;
}

/// Return wall-clock milliseconds for breaker cooldown checks.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_op_breaker_allows_attempts() {
        let breaker = CrossRegionBreaker::new();

        assert_eq!(breaker.state_for("us-chicago-1"), BreakerState::Closed);
        assert!(breaker.can_attempt("us-chicago-1"));
    }

    #[test]
    fn breaker_opens_after_configured_forwarding_failures() {
        let breaker = CrossRegionBreaker::with_failure_threshold(2);

        breaker.record_failure("us-chicago-1");
        assert_eq!(breaker.state_for("us-chicago-1"), BreakerState::Closed);
        assert!(breaker.can_attempt("us-chicago-1"));

        breaker.record_failure("us-chicago-1");
        assert_eq!(breaker.state_for("us-chicago-1"), BreakerState::Open);
        assert!(!breaker.can_attempt("us-chicago-1"));
    }

    #[test]
    fn breaker_moves_from_open_to_half_open_then_closed_after_success() {
        let breaker = CrossRegionBreaker::with_failure_threshold(1).with_open_timeout_ms(0);

        breaker.record_failure("us-chicago-1");
        assert_eq!(breaker.state_for("us-chicago-1"), BreakerState::HalfOpen);
        assert!(breaker.can_attempt("us-chicago-1"));

        breaker.record_success("us-chicago-1");
        assert_eq!(breaker.state_for("us-chicago-1"), BreakerState::Closed);
        assert!(breaker.can_attempt("us-chicago-1"));
    }
}
