//! Cross-region smart-router boundaries.
//!
//! This module is intentionally inert until the later request-plane and
//! sync-plane tasks wire it into serving paths.

pub mod breaker;
pub mod candidate_calculation;
pub mod config;
pub mod forwarding;
pub mod headers;
pub mod metrics;
pub mod peers;
pub mod profile;
pub mod signals;
pub mod state;
pub mod sync;

pub use breaker::{BreakerState, CrossRegionBreaker};
pub use candidate_calculation::{
    CandidateCalculationInput, CandidateCalculator, ExecutionTarget, RegionCandidate,
    RegionRouteDecision, RouteCommit,
};
pub use config::{
    CrossRegionContext, CrossRegionMtlsRuntimeConfig, CrossRegionRuntimeConfig,
    RequestPlaneRuntimeConfig, SyncPlaneRuntimeConfig,
};
pub use forwarding::{CrossRegionForwarder, ForwardingRequest};
pub use headers::{CrossRegionHeaders, RequestMode};
pub use metrics::{
    CandidateGatedReason, CrossRegionMetricLabels, CrossRegionMetrics, RouteDecisionOutcome,
};
pub use peers::{RegionPeer, RegionPeerRegistry};
pub use profile::{FailoverPolicy, ModalityPolicy, RoutingProfileContext};
pub use signals::{
    ClientLatencySignal, SignalEnvelope, SignalKey, SmgReadinessSignal, WorkerHealthSignal,
    WorkerLoadSignal, SIGNAL_CONTRACT_VERSION,
};
pub use state::{CrossRegionState, SignalVersion};
pub use sync::{CrossRegionSyncService, SyncEvent};

/// Cross-region module result type.
pub type CrossRegionResult<T> = Result<T, CrossRegionError>;

/// Error variants used by cross-region boundary code before HTTP mapping exists.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CrossRegionError {
    #[error("cross-region config is invalid: {reason}")]
    InvalidConfig { reason: String },

    #[error("cross-region header is invalid: {reason}")]
    InvalidHeader { reason: String },

    #[error("routing profile is invalid: {reason}")]
    InvalidProfile { reason: String },

    #[error("region peer '{region_id}' is not configured")]
    PeerNotFound { region_id: String },

    #[error("region peer '{region_id}' is invalid: {reason}")]
    InvalidPeer { region_id: String, reason: String },

    #[error("forwarding target is invalid: {reason}")]
    InvalidForwardingTarget { reason: String },

    #[error("no eligible cross-region candidate: {reason}")]
    NoCandidate { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_reexports_are_usable_from_module_root() {
        let peer = RegionPeer::new(
            "us-chicago-1",
            "https://smg-region-agent.us-chicago-1.internal:8443",
            "https://smg-region-agent.us-chicago-1.internal:9443",
            "oc1",
            "prod",
            None,
        )
        .expect("peer should parse");
        let registry = RegionPeerRegistry::new(vec![peer]).expect("registry should build");

        assert_eq!(RequestMode::Unresolved.to_string(), "UNRESOLVED");
        assert!(registry.contains_region("us-chicago-1"));
    }

    #[test]
    fn cross_region_errors_have_stable_messages() {
        let error = CrossRegionError::PeerNotFound {
            region_id: "us-phoenix-1".to_string(),
        };

        assert_eq!(
            error.to_string(),
            "region peer 'us-phoenix-1' is not configured"
        );
    }
}
