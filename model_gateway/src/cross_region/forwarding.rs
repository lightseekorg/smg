use url::Url;

use super::{
    peers::RegionPeerRequestTarget, CrossRegionError, CrossRegionResult, ExecutionTarget,
    RegionPeerRegistry, RegionRouteDecision,
};

/// Existing SMG endpoint path that cannot be an absolute worker URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExistingSmgPath(String);

impl ExistingSmgPath {
    /// Build a forwarding path for an existing SMG endpoint.
    pub fn new(path: impl Into<String>) -> CrossRegionResult<Self> {
        let path = path.into();
        if path.trim().is_empty() {
            return Err(CrossRegionError::InvalidForwardingTarget {
                reason: "existing SMG path must not be empty".to_string(),
            });
        }
        if path.trim() != path {
            return Err(CrossRegionError::InvalidForwardingTarget {
                reason: "existing SMG path must not contain surrounding whitespace".to_string(),
            });
        }
        if Url::parse(&path).is_ok() || path.starts_with("//") {
            return Err(CrossRegionError::InvalidForwardingTarget {
                reason: "forwarding path must be an existing SMG endpoint path, not a URL"
                    .to_string(),
            });
        }
        if !path.starts_with('/') {
            return Err(CrossRegionError::InvalidForwardingTarget {
                reason: "existing SMG path must start with '/'".to_string(),
            });
        }

        Ok(Self(path))
    }

    /// Return the validated path string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Minimal request envelope reserved for later remote SMG forwarding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardingRequest {
    existing_smg_path: ExistingSmgPath,
    body: Vec<u8>,
}

impl ForwardingRequest {
    /// Build a forwarding request for an existing SMG endpoint path.
    pub fn new(existing_smg_path: impl Into<String>, body: Vec<u8>) -> CrossRegionResult<Self> {
        Ok(Self {
            existing_smg_path: ExistingSmgPath::new(existing_smg_path)?,
            body,
        })
    }

    /// Return the existing SMG endpoint path for the remote request.
    pub fn existing_smg_path(&self) -> &ExistingSmgPath {
        &self.existing_smg_path
    }

    /// Return the forwarding request body bytes.
    pub fn body(&self) -> &[u8] {
        &self.body
    }
}

/// Resolved forwarding target that pairs the internal URL with peer identity.
#[derive(Clone, PartialEq, Eq)]
pub struct ForwardingTarget {
    url: Url,
    expected_mtls_identity: String,
}

impl ForwardingTarget {
    /// Build a forwarding target after request path and peer resolution validation.
    fn new(url: Url, expected_mtls_identity: impl Into<String>) -> Self {
        Self {
            url,
            expected_mtls_identity: expected_mtls_identity.into(),
        }
    }

    /// Return the peer identity that outbound mTLS must enforce.
    pub fn expected_mtls_identity(&self) -> &str {
        &self.expected_mtls_identity
    }

    /// Return the endpoint origin for internal diagnostics without exposing a forwarding URL.
    pub(crate) fn endpoint_origin(&self) -> String {
        self.url.origin().ascii_serialization()
    }
}

impl std::fmt::Debug for ForwardingTarget {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ForwardingTarget")
            .field("endpoint_origin", &self.endpoint_origin())
            .field("expected_mtls_identity", &self.expected_mtls_identity)
            .finish()
    }
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

    /// Resolve the request-plane target for a remote decision.
    pub(crate) fn request_target_for_decision(
        &self,
        decision: &RegionRouteDecision,
    ) -> CrossRegionResult<RegionPeerRequestTarget> {
        match &decision.execution_target {
            ExecutionTarget::RemoteRegion { region_id } => {
                self.peer_registry.request_target(region_id)
            }
            ExecutionTarget::LocalRegion => Err(CrossRegionError::InvalidForwardingTarget {
                reason: "local execution must use the existing local router path".to_string(),
            }),
        }
    }

    /// Resolve the remote Region Agent target for a validated forwarding request.
    pub fn forwarding_target_for_decision(
        &self,
        decision: &RegionRouteDecision,
        request: &ForwardingRequest,
    ) -> CrossRegionResult<ForwardingTarget> {
        let target = self.request_target_for_decision(decision)?;
        let url = target
            .request_url()
            .join(request.existing_smg_path().as_str())
            .map_err(|e| CrossRegionError::InvalidForwardingTarget {
                reason: format!("existing SMG path could not be joined to peer request_url: {e}"),
            })?;

        Ok(ForwardingTarget::new(url, target.expected_mtls_identity()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cross_region::RegionPeer;

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
    fn forwarder_resolves_request_target_without_exposing_peer() {
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

        let _target = forwarder
            .request_target_for_decision(&remote_decision("us-chicago-1"))
            .expect("target should resolve");
    }

    #[test]
    fn forwarder_resolves_request_target_by_region_only() {
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

        let target = forwarder
            .request_target_for_decision(&remote_decision("us-chicago-1"))
            .expect("target should resolve");

        assert_eq!(
            target.request_url().as_str(),
            "https://smg-region-agent.us-chicago-1.internal:8443/"
        );
    }

    #[test]
    fn forwarding_target_uses_peer_identity_and_existing_smg_path() {
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
        let request = ForwardingRequest::new("/v1/chat/completions", b"{}".to_vec())
            .expect("path should parse");

        let target = forwarder
            .forwarding_target_for_decision(&remote_decision("us-chicago-1"), &request)
            .expect("target should resolve");

        assert_eq!(
            target.endpoint_origin(),
            "https://smg-region-agent.us-chicago-1.internal:8443"
        );
        assert_eq!(
            target.expected_mtls_identity(),
            "spiffe://oraclecorp.com/oci/oc1/prod/region/us-chicago-1/service/smg-region-agent"
        );
    }

    #[test]
    fn forwarding_target_debug_hides_joined_path() {
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
        let request = ForwardingRequest::new("/v1/chat/completions", b"{}".to_vec())
            .expect("path should parse");

        let debug = format!(
            "{:?}",
            forwarder
                .forwarding_target_for_decision(&remote_decision("us-chicago-1"), &request)
                .expect("target should resolve")
        );

        assert!(debug.contains("endpoint_origin"));
        assert!(debug.contains("expected_mtls_identity"));
        assert!(!debug.contains("/v1/chat/completions"));
        assert!(!debug.contains("url"));
    }

    #[test]
    fn forwarding_request_rejects_raw_worker_url_as_path() {
        let error = ForwardingRequest::new(
            "https://remote-worker.us-chicago-1.internal:8000/v1/chat/completions",
            b"{}".to_vec(),
        )
        .expect_err("raw worker URL should be rejected");

        assert!(error.to_string().contains("not a URL"));
    }

    #[test]
    fn forwarding_request_rejects_scheme_relative_worker_url_as_path() {
        let error = ForwardingRequest::new(
            "//remote-worker.us-chicago-1.internal:8000/v1/chat/completions",
            b"{}".to_vec(),
        )
        .expect_err("scheme-relative worker URL should be rejected");

        assert!(error.to_string().contains("not a URL"));
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
            .request_target_for_decision(&decision)
            .expect_err("local target should be rejected");

        assert!(error.to_string().contains("local router path"));
    }
}
