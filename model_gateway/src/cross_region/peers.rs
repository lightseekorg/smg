use std::collections::{hash_map::Entry, HashMap};

use serde::{Deserialize, Serialize};

use super::{CrossRegionError, CrossRegionResult};
use crate::config::CrossRegionPeerConfig;

/// Remote Region Agent endpoint metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionPeer {
    pub region_id: String,
    pub request_url: String,
    pub sync_url: String,
    pub realm: String,
    pub environment: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_mtls_identity: Option<String>,
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
        let peer = Self {
            region_id: region_id.into(),
            request_url: request_url.into(),
            sync_url: sync_url.into(),
            realm: realm.into(),
            environment: environment.into(),
            expected_mtls_identity,
        };
        peer.validate()?;
        Ok(peer)
    }

    /// Convert SMG config peer data into the runtime peer registry shape.
    pub fn from_config(config: &CrossRegionPeerConfig) -> CrossRegionResult<Self> {
        Self::new(
            required("region", config.region_id.as_deref())?,
            required("request_url", config.request_url.as_deref())?,
            required("sync_url", config.sync_url.as_deref())?,
            required("realm", config.realm.as_deref())?,
            required("environment", config.environment.as_deref())?,
            None,
        )
    }

    /// Validate the peer identity and the request/sync endpoint URL forms.
    pub fn validate(&self) -> CrossRegionResult<()> {
        require_non_empty_peer_field(&self.region_id, "region")?;
        require_non_empty_peer_field(&self.realm, "realm")?;
        require_non_empty_peer_field(&self.environment, "environment")?;
        validate_peer_endpoint(&self.region_id, "request_url", &self.request_url)?;
        validate_peer_endpoint(&self.region_id, "sync_url", &self.sync_url)?;
        Ok(())
    }
}

/// Registry that resolves a target region into a remote SMG Region Agent peer.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionPeerRegistry {
    peers_by_region: HashMap<String, RegionPeer>,
}

impl RegionPeerRegistry {
    /// Build a registry from unique peer region entries.
    pub fn new(peers: Vec<RegionPeer>) -> CrossRegionResult<Self> {
        let mut peers_by_region = HashMap::new();
        for peer in peers {
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
    pub fn get(&self, region_id: &str) -> CrossRegionResult<&RegionPeer> {
        self.peers_by_region
            .get(region_id)
            .ok_or_else(|| CrossRegionError::PeerNotFound {
                region_id: region_id.to_string(),
            })
    }

    /// Return true when a region is present in the peer registry.
    pub fn contains_region(&self, region_id: &str) -> bool {
        self.peers_by_region.contains_key(region_id)
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
    let parsed = url::Url::parse(value).map_err(|e| CrossRegionError::InvalidPeer {
        region_id: region_id.to_string(),
        reason: format!("{field} has invalid URL format: {e}"),
    })?;

    if parsed.scheme() != "https" {
        return Err(CrossRegionError::InvalidPeer {
            region_id: region_id.to_string(),
            reason: format!("{field} must use https"),
        });
    }
    if parsed.host_str().is_none() {
        return Err(CrossRegionError::InvalidPeer {
            region_id: region_id.to_string(),
            reason: format!("{field} must include a host"),
        });
    }
    if parsed.port().is_none() {
        return Err(CrossRegionError::InvalidPeer {
            region_id: region_id.to_string(),
            reason: format!("{field} must include an explicit port"),
        });
    }

    Ok(())
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
    fn registry_resolves_peer_by_region() {
        let registry = RegionPeerRegistry::new(vec![valid_peer("us-chicago-1")])
            .expect("registry should build");

        assert_eq!(
            registry
                .get("us-chicago-1")
                .expect("peer should exist")
                .request_url,
            "https://smg-region-agent.us-chicago-1.internal:8443"
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
}
