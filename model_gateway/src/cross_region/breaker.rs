use serde::{Deserialize, Serialize};

/// Per-remote-region request-forward circuit breaker state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BreakerState {
    #[default]
    Closed,
    HalfOpen,
    Open,
}

/// No-op breaker boundary for candidate gating.
#[derive(Debug, Clone)]
pub struct CrossRegionBreaker {
    default_state: BreakerState,
}

impl Default for CrossRegionBreaker {
    /// Create a breaker whose default remote-region state is closed.
    fn default() -> Self {
        Self {
            default_state: BreakerState::Closed,
        }
    }
}

impl CrossRegionBreaker {
    /// Create a no-op breaker that allows all attempts.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the current state for a region; the skeleton defaults to closed.
    pub fn state_for(&self, _region_id: &str) -> BreakerState {
        self.default_state
    }

    /// Return true when candidate calculation may attempt the remote region.
    pub fn can_attempt(&self, region_id: &str) -> bool {
        self.state_for(region_id) != BreakerState::Open
    }
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
}
