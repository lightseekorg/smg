use serde::{Deserialize, Serialize};

/// Outcome labels for cross-region route decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteDecisionOutcome {
    Local,
    Remote,
    NoCandidate,
    Rejected,
}

/// Stable reason labels for candidate gating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateGatedReason {
    PolicyMismatch,
    StaleSignal,
    BreakerOpen,
    PeerUnavailable,
    NoCapacity,
}

/// Common cross-region metric labels.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CrossRegionMetricLabels {
    pub source_region: String,
    pub target_region: String,
    pub model_id: String,
    pub outcome: RouteDecisionOutcome,
}

impl CrossRegionMetricLabels {
    /// Build stable route-decision labels.
    pub fn new(
        source_region: impl Into<String>,
        target_region: impl Into<String>,
        model_id: impl Into<String>,
        outcome: RouteDecisionOutcome,
    ) -> Self {
        Self {
            source_region: source_region.into(),
            target_region: target_region.into(),
            model_id: model_id.into(),
            outcome,
        }
    }
}

/// No-op metrics facade for future route and sync instrumentation.
#[derive(Debug, Clone, Default)]
pub struct CrossRegionMetrics {
    enabled: bool,
}

impl CrossRegionMetrics {
    /// Create a no-op cross-region metrics facade.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a route decision; the skeleton intentionally emits no metrics.
    pub fn record_route_decision(&self, labels: &CrossRegionMetricLabels) {
        if self.enabled {
            let _ = labels;
        }
    }

    /// Record a gated candidate; the skeleton intentionally emits no metrics.
    pub fn record_candidate_gated(&self, reason: CandidateGatedReason) {
        if self.enabled {
            let _ = reason;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_labels_preserve_route_dimensions() {
        let labels = CrossRegionMetricLabels::new(
            "us-ashburn-1",
            "us-chicago-1",
            "cohere.command-r-plus",
            RouteDecisionOutcome::Remote,
        );

        assert_eq!(labels.target_region, "us-chicago-1");
        assert_eq!(labels.outcome, RouteDecisionOutcome::Remote);
    }
}
