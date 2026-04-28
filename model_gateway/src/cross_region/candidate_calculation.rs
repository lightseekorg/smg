use serde::{Deserialize, Serialize};

use super::{CrossRegionError, CrossRegionResult, RequestMode, RoutingProfileContext};
use crate::config::CrossRegionFailoverMode;

/// Region-level execution target. Remote variants carry a region, never a worker URL.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExecutionTarget {
    LocalRegion,
    RemoteRegion { region_id: String },
}

/// Region-level route decision emitted by cross-region candidate calculation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionRouteDecision {
    pub route_id: String,
    pub target_region: String,
    pub model_id: String,
    pub execution_target: ExecutionTarget,
}

/// Committed route metadata attached to settled remote requests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteCommit {
    pub route_id: String,
    pub entry_region: String,
    pub target_region: String,
    pub model_id: String,
    pub request_mode: RequestMode,
    pub attempt: u32,
    pub failover_mode: CrossRegionFailoverMode,
}

/// Candidate region scoring input produced before a route decision is committed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionCandidate {
    pub region_id: String,
    pub model_id: String,
    pub healthy: bool,
    pub has_capacity: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_latency_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_load: Option<isize>,
}

/// Input bundle for the future candidate calculation implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateCalculationInput {
    pub profile: RoutingProfileContext,
    pub local_region: String,
}

/// No-op candidate calculator placeholder for later request-plane implementation.
#[derive(Debug, Clone)]
pub struct CandidateCalculator {
    enabled: bool,
}

impl Default for CandidateCalculator {
    /// Create the default no-op calculator boundary.
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl CandidateCalculator {
    /// Create a no-op candidate calculator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return no route decision until SMG-07 wires real candidate logic.
    pub fn calculate(
        &self,
        input: CandidateCalculationInput,
    ) -> CrossRegionResult<Option<RegionRouteDecision>> {
        if input.local_region.trim().is_empty() {
            return Err(CrossRegionError::InvalidConfig {
                reason: "local_region must not be empty".to_string(),
            });
        }
        if !self.enabled {
            return Ok(None);
        }
        Ok(None)
    }
}

impl RouteCommit {
    /// Build committed route metadata from a route decision and request metadata.
    pub fn from_decision(
        decision: RegionRouteDecision,
        entry_region: impl Into<String>,
        request_mode: RequestMode,
        attempt: u32,
        failover_mode: CrossRegionFailoverMode,
    ) -> Self {
        Self {
            route_id: decision.route_id,
            entry_region: entry_region.into(),
            target_region: decision.target_region,
            model_id: decision.model_id,
            request_mode,
            attempt,
            failover_mode,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cross_region::{FailoverPolicy, ModalityPolicy};

    #[test]
    fn route_decision_serializes_without_worker_url() {
        let decision = RegionRouteDecision {
            route_id: "route-1".to_string(),
            target_region: "us-chicago-1".to_string(),
            model_id: "cohere.command-r-plus".to_string(),
            execution_target: ExecutionTarget::RemoteRegion {
                region_id: "us-chicago-1".to_string(),
            },
        };

        let json = serde_json::to_string(&decision).expect("serialize route decision");

        assert!(json.contains("us-chicago-1"));
        assert!(!json.contains("worker_url"));
    }

    #[test]
    fn no_op_calculator_returns_no_decision() {
        let profile = RoutingProfileContext::new(
            vec!["us-ashburn-1".to_string()],
            vec!["cohere.command-r-plus".to_string()],
            FailoverPolicy::new(CrossRegionFailoverMode::Manual, 1),
            ModalityPolicy::default(),
        )
        .expect("profile should be valid");
        let calculator = CandidateCalculator::new();

        assert!(calculator
            .calculate(CandidateCalculationInput {
                profile,
                local_region: "us-ashburn-1".to_string(),
            })
            .expect("no-op calculation should succeed")
            .is_none());
    }
}
