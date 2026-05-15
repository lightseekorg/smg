use std::collections::{hash_map::Entry, HashMap};

use url::Url;

use super::{CrossRegionError, CrossRegionResult};
use crate::config::CrossRegionPeerConfig;

/// Remote Region Agent endpoint metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionPeer {
    region_id: String,
    request_url: String,
    sync_url: String,
    realm: String,
    environment: String,
    expected_mtls_identity: String,
    enabled: bool,
}

impl RegionPeer {
    /// Build a peer after validating that only SMG Region Agent endpoints are modeled.
    pub fn new(
        region_id: impl Into<String>,
        request_url: impl Into<String>,
        sync_url: impl Into<String>,
        realm: impl Into<String>,
        environment: impl Into<String>,
        expected_mtls_identity: Option<String>,
    ) -> CrossRegionResult<Self> {
        let region_id = region_id.into();
        let realm = realm.into();
        let environment = environment.into();
        let expected_mtls_identity = expected_mtls_identity
            .unwrap_or_else(|| derive_expected_mtls_identity(&region_id, &realm, &environment));
        let peer = Self {
            region_id,
            request_url: request_url.into(),
            sync_url: sync_url.into(),
            realm,
            environment,
            expected_mtls_identity,
            enabled: true,
        };
        peer.validate()?;
        Ok(peer)
    }

    /// Return a copy of this peer with explicit routing eligibility.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Return the peer region id.
    pub fn region_id(&self) -> &str {
        &self.region_id
    }

    /// Return true when this peer can be used for request forwarding.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Return the mTLS URI SAN expected for this peer.
    pub fn expected_mtls_identity(&self) -> &str {
        &self.expected_mtls_identity
    }

    /// Convert SMG config peer data into the runtime peer registry shape.
    pub fn from_config(config: &CrossRegionPeerConfig) -> CrossRegionResult<Self> {
        Ok(Self::new(
            required("region", config.region_id.as_deref())?,
            required("request_url", config.request_url.as_deref())?,
            required("sync_url", config.sync_url.as_deref())?,
            required("realm", config.realm.as_deref())?,
            required("environment", config.environment.as_deref())?,
            config.expected_mtls_identity.clone(),
        )?
        .with_enabled(config.enabled))
    }

    /// Validate the peer identity and the request/sync endpoint URL forms.
    pub fn validate(&self) -> CrossRegionResult<()> {
        require_non_empty_peer_field(&self.region_id, "region")?;
        require_non_empty_peer_field(&self.realm, "realm")?;
        require_non_empty_peer_field(&self.environment, "environment")?;
        require_non_empty_peer_field(&self.expected_mtls_identity, "expected_mtls_identity")?;
        validate_expected_mtls_identity(
            &self.region_id,
            &self.realm,
            &self.environment,
            &self.expected_mtls_identity,
        )?;
        validate_peer_endpoint(&self.region_id, "request_url", &self.request_url)?;
        validate_peer_endpoint(&self.region_id, "sync_url", &self.sync_url)?;
        Ok(())
    }
}

/// Resolved request-plane target for a remote SMG Region Agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RegionPeerRequestTarget {
    request_url: Url,
    expected_mtls_identity: String,
}

impl RegionPeerRequestTarget {
    /// Build a request target from a validated, enabled region peer.
    fn from_peer(peer: &RegionPeer) -> CrossRegionResult<Self> {
        let request_url =
            Url::parse(&peer.request_url).map_err(|e| CrossRegionError::InvalidPeer {
                region_id: peer.region_id.clone(),
                reason: format!("request_url has invalid URL format: {e}"),
            })?;

        Ok(Self {
            request_url,
            expected_mtls_identity: peer.expected_mtls_identity.clone(),
        })
    }

    /// Return the internally configured request-plane Region Agent URL.
    pub(crate) fn request_url(&self) -> &Url {
        &self.request_url
    }

    /// Return the mTLS URI SAN expected for the target peer.
    pub(crate) fn expected_mtls_identity(&self) -> &str {
        &self.expected_mtls_identity
    }
}

/// Registry that resolves a target region into a remote SMG Region Agent peer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RegionPeerRegistry {
    peers_by_region: HashMap<String, RegionPeer>,
}

impl RegionPeerRegistry {
    /// Build a registry from unique peer region entries.
    pub fn new(peers: Vec<RegionPeer>) -> CrossRegionResult<Self> {
        let mut peers_by_region = HashMap::new();
        for peer in peers {
            peer.validate()?;
            match peers_by_region.entry(peer.region_id.clone()) {
                Entry::Vacant(entry) => {
                    entry.insert(peer);
                }
                Entry::Occupied(entry) => {
                    return Err(CrossRegionError::InvalidPeer {
                        region_id: entry.key().clone(),
                        reason: "duplicate peer region".to_string(),
                    });
                }
            }
        }

        Ok(Self { peers_by_region })
    }

    /// Return an empty registry for disabled/no-op cross-region contexts.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Resolve a remote region to its configured peer endpoint metadata.
    fn get(&self, region_id: &str) -> CrossRegionResult<&RegionPeer> {
        let peer =
            self.peers_by_region
                .get(region_id)
                .ok_or_else(|| CrossRegionError::PeerNotFound {
                    region_id: region_id.to_string(),
                })?;

        if !peer.is_enabled() {
            return Err(CrossRegionError::PeerDisabled {
                region_id: region_id.to_string(),
            });
        }

        Ok(peer)
    }

    /// Resolve a remote region to the request-plane Region Agent target.
    pub(crate) fn request_target(
        &self,
        region_id: &str,
    ) -> CrossRegionResult<RegionPeerRequestTarget> {
        RegionPeerRequestTarget::from_peer(self.get(region_id)?)
    }

    /// Return true when a region is present in the peer registry.
    pub fn contains_region(&self, region_id: &str) -> bool {
        self.peers_by_region.contains_key(region_id)
    }

    /// Return true when a configured peer exists and is eligible for routing.
    pub fn is_enabled(&self, region_id: &str) -> bool {
        self.peers_by_region
            .get(region_id)
            .is_some_and(RegionPeer::is_enabled)
    }

    /// Return the number of configured remote region peers.
    pub fn len(&self) -> usize {
        self.peers_by_region.len()
    }

    /// Return true when no remote region peers are configured.
    pub fn is_empty(&self) -> bool {
        self.peers_by_region.is_empty()
    }

    /// Return configured regions in stable order for deterministic diagnostics.
    pub fn regions(&self) -> Vec<&str> {
        let mut regions = self
            .peers_by_region
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        regions.sort_unstable();
        regions
    }
}

/// Derive the SPIFFE-style URI SAN expected for a Region Agent peer.
fn derive_expected_mtls_identity(region_id: &str, realm: &str, environment: &str) -> String {
    format!(
        "spiffe://oraclecorp.com/oci/{realm}/{environment}/region/{region_id}/service/smg-region-agent"
    )
}

/// Validate that a configured peer identity matches the expected Region Agent identity.
fn validate_expected_mtls_identity(
    region_id: &str,
    realm: &str,
    environment: &str,
    value: &str,
) -> CrossRegionResult<()> {
    let expected = derive_expected_mtls_identity(region_id, realm, environment);
    if value != expected {
        return Err(CrossRegionError::InvalidPeer {
            region_id: region_id.to_string(),
            reason: format!("expected_mtls_identity must be {expected}"),
        });
    }

    Ok(())
}

/// Return a required peer config field or an invalid peer error.
fn required<'a>(field: &str, value: Option<&'a str>) -> CrossRegionResult<&'a str> {
    let value = value.ok_or_else(|| CrossRegionError::InvalidPeer {
        region_id: "<unknown>".to_string(),
        reason: format!("{field} is required"),
    })?;
    if value.trim().is_empty() {
        return Err(CrossRegionError::InvalidPeer {
            region_id: "<unknown>".to_string(),
            reason: format!("{field} must not be empty"),
        });
    }
    Ok(value)
}

/// Reject blank peer fields before URL-specific validation runs.
fn require_non_empty_peer_field(value: &str, field: &str) -> CrossRegionResult<()> {
    if value.trim().is_empty() {
        return Err(CrossRegionError::InvalidPeer {
            region_id: "<unknown>".to_string(),
            reason: format!("{field} must not be empty"),
        });
    }
    Ok(())
}

/// Validate that a peer endpoint is HTTPS and includes an explicit port.
fn validate_peer_endpoint(region_id: &str, field: &str, value: &str) -> CrossRegionResult<()> {
    let parsed = Url::parse(value).map_err(|e| CrossRegionError::InvalidPeer {
        region_id: region_id.to_string(),
        reason: format!("{field} has invalid URL format: {e}"),
    })?;

    if parsed.scheme() != "https" {
        return Err(CrossRegionError::InvalidPeer {
            region_id: region_id.to_string(),
            reason: format!("{field} must use https"),
        });
    }
    let Some(host) = parsed.host_str() else {
        return Err(CrossRegionError::InvalidPeer {
            region_id: region_id.to_string(),
            reason: format!("{field} must include a host"),
        });
    };
    if is_worker_like_host(host) {
        return Err(CrossRegionError::InvalidPeer {
            region_id: region_id.to_string(),
            reason: format!("{field} must identify a Region Agent endpoint, not a worker endpoint"),
        });
    }
    if parsed.port().is_none() {
        return Err(CrossRegionError::InvalidPeer {
            region_id: region_id.to_string(),
            reason: format!("{field} must include an explicit port"),
        });
    }
    if parsed.path() != "/" || parsed.query().is_some() || parsed.fragment().is_some() {
        return Err(CrossRegionError::InvalidPeer {
            region_id: region_id.to_string(),
            reason: format!("{field} must be a Region Agent endpoint base URL"),
        });
    }

    Ok(())
}

/// Return true when a peer URL hostname appears to identify a model worker.
fn is_worker_like_host(host: &str) -> bool {
    let service_label = host.split('.').next().unwrap_or(host);
    service_label
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .any(|label| {
            let label = label.to_ascii_lowercase();
            label.starts_with("worker") || label.ends_with("worker")
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a valid peer fixture for registry tests.
    fn valid_peer(region_id: &str) -> RegionPeer {
        RegionPeer::new(
            region_id,
            format!("https://smg-region-agent.{region_id}.internal:8443"),
            format!("https://smg-region-agent.{region_id}.internal:9443"),
            "oc1",
            "prod",
            None,
        )
        .expect("peer should parse")
    }

    #[test]
    fn registry_rejects_missing_peer_lookup() {
        let registry = RegionPeerRegistry::new(vec![valid_peer("us-chicago-1")])
            .expect("registry should build");

        let error = registry
            .request_target("us-phoenix-1")
            .expect_err("missing peer should fail");

        assert!(error.to_string().contains("not configured"));
    }

    #[test]
    fn registry_resolves_request_target_by_region() {
        let registry = RegionPeerRegistry::new(vec![valid_peer("us-chicago-1")])
            .expect("registry should build");

        let target = registry
            .request_target("us-chicago-1")
            .expect("target should resolve");

        assert_eq!(
            target.request_url().as_str(),
            "https://smg-region-agent.us-chicago-1.internal:8443/"
        );
    }

    #[test]
    fn registry_rejects_duplicate_regions() {
        let error =
            RegionPeerRegistry::new(vec![valid_peer("us-chicago-1"), valid_peer("us-chicago-1")])
                .expect_err("duplicate peer should fail");

        assert!(error.to_string().contains("duplicate peer region"));
    }

    #[test]
    fn registry_rejects_disabled_peer_lookup() {
        let registry =
            RegionPeerRegistry::new(vec![valid_peer("us-chicago-1").with_enabled(false)])
                .expect("registry should build");

        let error = registry
            .request_target("us-chicago-1")
            .expect_err("disabled peer should not resolve");

        assert!(error.to_string().contains("disabled"));
        assert!(registry.contains_region("us-chicago-1"));
        assert!(!registry.is_enabled("us-chicago-1"));
    }

    #[test]
    fn registry_revalidates_peer_identity_before_insert() {
        let mut peer = valid_peer("us-chicago-1");
        peer.expected_mtls_identity =
            "spiffe://oraclecorp.com/oci/oc1/prod/region/us-phoenix-1/service/smg-region-agent"
                .to_string();

        let error = RegionPeerRegistry::new(vec![peer]).expect_err("mutated peer should fail");

        assert!(error.to_string().contains("expected_mtls_identity"));
    }

    #[test]
    fn registry_rejects_mutated_worker_api_request_url() {
        let mut peer = valid_peer("us-chicago-1");
        peer.request_url =
            "https://remote-worker.us-chicago-1.internal:8000/v1/chat/completions".to_string();

        let error = RegionPeerRegistry::new(vec![peer]).expect_err("worker URL should fail");

        assert!(error.to_string().contains("not a worker endpoint"));
    }

    #[test]
    fn peer_rejects_non_https_endpoint() {
        let error = RegionPeer::new(
            "us-chicago-1",
            "http://smg-region-agent.us-chicago-1.internal:8443",
            "https://smg-region-agent.us-chicago-1.internal:9443",
            "oc1",
            "prod",
            None,
        )
        .expect_err("http endpoint should fail");

        assert!(error.to_string().contains("https"));
    }

    #[test]
    fn peer_rejects_request_url_with_path() {
        let error = RegionPeer::new(
            "us-chicago-1",
            "https://smg-region-agent.us-chicago-1.internal:8443/v1/chat/completions",
            "https://smg-region-agent.us-chicago-1.internal:9443",
            "oc1",
            "prod",
            None,
        )
        .expect_err("request endpoint with path should fail");

        assert!(error.to_string().contains("endpoint base URL"));
    }

    #[test]
    fn peer_rejects_worker_like_endpoint_host() {
        let error = RegionPeer::new(
            "us-chicago-1",
            "https://remote-worker.us-chicago-1.internal:18443",
            "https://smg-region-agent.us-chicago-1.internal:19443",
            "oc1",
            "prod",
            None,
        )
        .expect_err("worker-like endpoint host should fail");

        assert!(error.to_string().contains("not a worker endpoint"));
    }

    #[test]
    fn peer_accepts_region_agent_host_in_worker_named_namespace() {
        let peer = RegionPeer::new(
            "us-chicago-1",
            "https://smg-region-agent.smg-worker.svc.cluster.local:18443",
            "https://smg-region-agent.smg-worker.svc.cluster.local:19443",
            "oc1",
            "prod",
            None,
        )
        .expect("worker-named namespace should not make Region Agent host fail");

        assert_eq!(peer.region_id(), "us-chicago-1");
    }

    #[test]
    fn peer_accepts_custom_listener_ports() {
        let peer = RegionPeer::new(
            "us-chicago-1",
            "https://smg-region-agent.us-chicago-1.internal:18443",
            "https://smg-region-agent.us-chicago-1.internal:19443",
            "oc1",
            "prod",
            None,
        )
        .expect("custom listener ports should parse");

        assert_eq!(peer.region_id(), "us-chicago-1");
    }

    #[test]
    fn peer_derives_expected_mtls_identity() {
        let peer = valid_peer("us-chicago-1");

        assert_eq!(
            peer.expected_mtls_identity(),
            "spiffe://oraclecorp.com/oci/oc1/prod/region/us-chicago-1/service/smg-region-agent"
        );
    }

    #[test]
    fn peer_rejects_mismatched_expected_mtls_identity() {
        let error = RegionPeer::new(
            "us-chicago-1",
            "https://smg-region-agent.us-chicago-1.internal:8443",
            "https://smg-region-agent.us-chicago-1.internal:9443",
            "oc1",
            "prod",
            Some(
                "spiffe://oraclecorp.com/oci/oc1/prod/region/us-ashburn-1/service/smg-region-agent"
                    .to_string(),
            ),
        )
        .expect_err("mismatched identity should fail");

        assert!(error.to_string().contains("expected_mtls_identity"));
    }
}
