use super::{CrossRegionError, CrossRegionResult, RegionPeer, RegionPeerRegistry};
use crate::config::{CrossRegionConfig, CrossRegionFailoverMode};

/// Runtime-friendly request-plane settings derived from RouterConfig.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestPlaneRuntimeConfig {
    pub enabled: bool,
    pub listen_port: u16,
    pub max_platform_retries: u32,
    pub default_failover_mode: CrossRegionFailoverMode,
    pub local_first_tie_break: bool,
}

/// Runtime-friendly sync-plane settings derived from RouterConfig.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncPlaneRuntimeConfig {
    pub enabled: bool,
    pub listen_port: u16,
    pub full_resync_interval_seconds: u64,
    pub signal_stale_after_seconds: u64,
}

/// Runtime mTLS file paths for cross-region request and sync planes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossRegionMtlsRuntimeConfig {
    pub ca_cert_path: String,
    pub server_cert_path: String,
    pub server_key_path: String,
    pub client_cert_path: String,
    pub client_key_path: String,
}

/// Runtime cross-region identity and tuning config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossRegionRuntimeConfig {
    pub region_id: String,
    pub realm: String,
    pub environment: String,
    pub local_only_on_degraded_sync: bool,
    pub request_plane: RequestPlaneRuntimeConfig,
    pub sync_plane: SyncPlaneRuntimeConfig,
    pub mtls: CrossRegionMtlsRuntimeConfig,
}

impl CrossRegionRuntimeConfig {
    /// Convert enabled RouterConfig cross-region settings into runtime settings.
    pub fn from_router_config(config: &CrossRegionConfig) -> CrossRegionResult<Option<Self>> {
        if !config.enabled {
            return Ok(None);
        }

        Ok(Some(Self {
            region_id: required("region", config.region_id.as_deref())?.to_string(),
            realm: required("realm", config.realm.as_deref())?.to_string(),
            environment: required("environment", config.environment.as_deref())?.to_string(),
            local_only_on_degraded_sync: config.local_only_on_degraded_sync,
            request_plane: RequestPlaneRuntimeConfig {
                enabled: config.request_plane.enabled,
                listen_port: config.request_plane.listen_port,
                max_platform_retries: config.request_plane.max_platform_retries,
                default_failover_mode: config.request_plane.default_failover_mode,
                local_first_tie_break: config.request_plane.local_first_tie_break,
            },
            sync_plane: SyncPlaneRuntimeConfig {
                enabled: config.sync_plane.enabled,
                listen_port: config.sync_plane.listen_port,
                full_resync_interval_seconds: config.sync_plane.full_resync_interval_seconds,
                signal_stale_after_seconds: config.sync_plane.signal_stale_after_seconds,
            },
            mtls: CrossRegionMtlsRuntimeConfig {
                ca_cert_path: required("mtls.ca_cert_path", config.mtls.ca_cert_path.as_deref())?
                    .to_string(),
                server_cert_path: required(
                    "mtls.server_cert_path",
                    config.mtls.server_cert_path.as_deref(),
                )?
                .to_string(),
                server_key_path: required(
                    "mtls.server_key_path",
                    config.mtls.server_key_path.as_deref(),
                )?
                .to_string(),
                client_cert_path: required(
                    "mtls.client_cert_path",
                    config.mtls.client_cert_path.as_deref(),
                )?
                .to_string(),
                client_key_path: required(
                    "mtls.client_key_path",
                    config.mtls.client_key_path.as_deref(),
                )?
                .to_string(),
            },
        }))
    }
}

/// Top-level cross-region runtime context consumed by later service wiring tasks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossRegionContext {
    pub config: CrossRegionRuntimeConfig,
    pub peers: RegionPeerRegistry,
}

impl CrossRegionContext {
    /// Build an optional runtime context from RouterConfig cross-region settings.
    pub fn from_router_config(config: &CrossRegionConfig) -> CrossRegionResult<Option<Self>> {
        let Some(runtime_config) = CrossRegionRuntimeConfig::from_router_config(config)? else {
            return Ok(None);
        };

        let peers = config
            .peers
            .iter()
            .map(RegionPeer::from_config)
            .collect::<CrossRegionResult<Vec<_>>>()?;

        Ok(Some(Self {
            config: runtime_config,
            peers: RegionPeerRegistry::new(peers)?,
        }))
    }
}

/// Return a required config value or a field-specific config error.
fn required<'a>(field: &str, value: Option<&'a str>) -> CrossRegionResult<&'a str> {
    let value = value.ok_or_else(|| CrossRegionError::InvalidConfig {
        reason: format!("cross_region.{field} is required"),
    })?;
    if value.trim().is_empty() {
        return Err(CrossRegionError::InvalidConfig {
            reason: format!("cross_region.{field} must not be empty"),
        });
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        CrossRegionMtlsConfig, CrossRegionPeerConfig, CrossRegionRequestPlaneConfig,
        CrossRegionSyncPlaneConfig,
    };

    /// Build a valid cross-region config fixture for runtime conversion tests.
    fn valid_config() -> CrossRegionConfig {
        CrossRegionConfig {
            enabled: true,
            region_id: Some("us-ashburn-1".to_string()),
            realm: Some("oc1".to_string()),
            environment: Some("prod".to_string()),
            local_only_on_degraded_sync: true,
            request_plane: CrossRegionRequestPlaneConfig::default(),
            sync_plane: CrossRegionSyncPlaneConfig::default(),
            peers: vec![CrossRegionPeerConfig {
                region_id: Some("us-chicago-1".to_string()),
                request_url: Some(
                    "https://smg-region-agent.us-chicago-1.internal:8443".to_string(),
                ),
                sync_url: Some("https://smg-region-agent.us-chicago-1.internal:9443".to_string()),
                realm: Some("oc1".to_string()),
                environment: Some("prod".to_string()),
            }],
            mtls: CrossRegionMtlsConfig {
                ca_cert_path: Some("/etc/smg/certs/ca.crt".to_string()),
                server_cert_path: Some("/etc/smg/certs/tls.crt".to_string()),
                server_key_path: Some("/etc/smg/certs/tls.key".to_string()),
                client_cert_path: Some("/etc/smg/certs/client.crt".to_string()),
                client_key_path: Some("/etc/smg/certs/client.key".to_string()),
            },
        }
    }

    #[test]
    fn disabled_config_returns_no_context() {
        let context = CrossRegionContext::from_router_config(&CrossRegionConfig::default())
            .expect("disabled config should be accepted");

        assert!(context.is_none());
    }

    #[test]
    fn enabled_config_builds_runtime_context() {
        let context = CrossRegionContext::from_router_config(&valid_config())
            .expect("enabled config should convert")
            .expect("context should be present");

        assert_eq!(context.config.region_id, "us-ashburn-1");
        assert!(context.peers.contains_region("us-chicago-1"));
    }

    #[test]
    fn enabled_config_requires_region_identity() {
        let mut config = valid_config();
        config.region_id = None;

        let error = CrossRegionContext::from_router_config(&config)
            .expect_err("missing region should fail");

        assert!(error.to_string().contains("cross_region.region"));
    }
}
