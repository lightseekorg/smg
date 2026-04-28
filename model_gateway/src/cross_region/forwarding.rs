use super::{
    CrossRegionError, CrossRegionResult, ExecutionTarget, RegionPeer, RegionPeerRegistry,
    RegionRouteDecision,
};

/// Minimal request envelope reserved for later remote SMG forwarding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardingRequest {
    pub existing_smg_path: String,
    pub body: Vec<u8>,
}

/// No-op remote forwarder boundary that resolves only by region id.
#[derive(Debug, Clone)]
pub struct CrossRegionForwarder {
    peer_registry: RegionPeerRegistry,
}

impl CrossRegionForwarder {
    /// Create a forwarder boundary from a peer registry.
    pub fn new(peer_registry: RegionPeerRegistry) -> Self {
        Self { peer_registry }
    }

    /// Resolve the peer for a remote decision without performing network I/O.
    pub fn peer_for_decision(
        &self,
        decision: &RegionRouteDecision,
    ) -> CrossRegionResult<&RegionPeer> {
        match &decision.execution_target {
            ExecutionTarget::RemoteRegion { region_id } => self.peer_registry.get(region_id),
            ExecutionTarget::LocalRegion => Err(CrossRegionError::InvalidForwardingTarget {
                reason: "local execution must use the existing local router path".to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a remote route decision fixture.
    fn remote_decision(region_id: &str) -> RegionRouteDecision {
        RegionRouteDecision {
            route_id: "route-1".to_string(),
            target_region: region_id.to_string(),
            model_id: "cohere.command-r-plus".to_string(),
            execution_target: ExecutionTarget::RemoteRegion {
                region_id: region_id.to_string(),
            },
        }
    }

    #[test]
    fn forwarder_resolves_remote_peer_by_region() {
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
        let forwarder = CrossRegionForwarder::new(registry);

        assert_eq!(
            forwarder
                .peer_for_decision(&remote_decision("us-chicago-1"))
                .expect("peer should resolve")
                .region_id,
            "us-chicago-1"
        );
    }

    #[test]
    fn forwarder_rejects_local_execution_target() {
        let forwarder = CrossRegionForwarder::new(RegionPeerRegistry::empty());
        let decision = RegionRouteDecision {
            route_id: "route-1".to_string(),
            target_region: "us-ashburn-1".to_string(),
            model_id: "cohere.command-r-plus".to_string(),
            execution_target: ExecutionTarget::LocalRegion,
        };

        let error = forwarder
            .peer_for_decision(&decision)
            .expect_err("local target should be rejected");

        assert!(error.to_string().contains("local router path"));
    }
}
