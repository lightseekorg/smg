//! Per-worker HTTP client construction.
//!
//! Builds a `reqwest::Client` for each worker by merging per-worker
//! `HttpPoolConfig` overrides with router-level TLS and timeout defaults.

use std::time::Duration;

use openai_protocol::worker::HttpPoolConfig;
use tracing::debug;

use crate::config::RouterConfig;

/// Default pool settings for per-worker HTTP clients.
const DEFAULT_POOL_MAX_IDLE_PER_HOST: usize = 8;
const DEFAULT_POOL_IDLE_TIMEOUT_SECS: u64 = 50;
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;

/// Build a per-worker `reqwest::Client`.
///
/// Merges `HttpPoolConfig` overrides with router-level TLS settings.
/// Each worker gets its own connection pool, isolated from other workers.
pub fn build_worker_http_client(
    pool_config: &HttpPoolConfig,
    router_config: &RouterConfig,
) -> Result<reqwest::Client, String> {
    let timeout_secs = pool_config
        .timeout_secs
        .unwrap_or(router_config.request_timeout_secs);
    let connect_timeout_secs = pool_config
        .connect_timeout_secs
        .unwrap_or(DEFAULT_CONNECT_TIMEOUT_SECS);
    let pool_max_idle = pool_config
        .pool_max_idle_per_host
        .unwrap_or(DEFAULT_POOL_MAX_IDLE_PER_HOST);
    let pool_idle_timeout = pool_config
        .pool_idle_timeout_secs
        .unwrap_or(DEFAULT_POOL_IDLE_TIMEOUT_SECS);

    let has_tls =
        router_config.client_identity.is_some() || !router_config.ca_certificates.is_empty();

    let mut builder = reqwest::Client::builder()
        .pool_max_idle_per_host(pool_max_idle)
        .pool_idle_timeout(Some(Duration::from_secs(pool_idle_timeout)))
        .timeout(Duration::from_secs(timeout_secs))
        .connect_timeout(Duration::from_secs(connect_timeout_secs))
        .tcp_nodelay(true)
        .tcp_keepalive(Some(Duration::from_secs(30)));

    if has_tls {
        builder = builder.use_rustls_tls();
    }

    if let Some(identity_pem) = &router_config.client_identity {
        let identity = reqwest::Identity::from_pem(identity_pem)
            .map_err(|e| format!("Failed to create client identity: {e}"))?;
        builder = builder.identity(identity);
    }

    for ca_cert in &router_config.ca_certificates {
        let cert = reqwest::Certificate::from_pem(ca_cert)
            .map_err(|e| format!("Failed to add CA certificate: {e}"))?;
        builder = builder.add_root_certificate(cert);
    }

    debug!(
        pool_max_idle = pool_max_idle,
        timeout_secs = timeout_secs,
        "Building per-worker HTTP client"
    );

    builder
        .build()
        .map_err(|e| format!("Failed to create per-worker HTTP client: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::{PolicyConfig, RoutingMode};

    fn test_router_config() -> RouterConfig {
        RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec![],
            },
            PolicyConfig::Random,
        )
    }

    #[test]
    fn test_build_client_with_defaults() {
        let pool = HttpPoolConfig::default();
        let config = test_router_config();
        let client = build_worker_http_client(&pool, &config);
        assert!(client.is_ok());
    }

    #[test]
    fn test_build_client_with_overrides() {
        let pool = HttpPoolConfig {
            pool_max_idle_per_host: Some(16),
            timeout_secs: Some(60),
            ..Default::default()
        };
        let config = test_router_config();
        let client = build_worker_http_client(&pool, &config);
        assert!(client.is_ok());
    }
}
