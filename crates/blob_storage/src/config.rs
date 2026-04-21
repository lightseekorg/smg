use serde::{Deserialize, Serialize};

/// Supported blob-store backend families.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum BlobStoreBackend {
    #[default]
    Filesystem,
    S3,
    Gcs,
    Azure,
    Oci,
}

/// Backend-neutral blob-store configuration shared by the skills subsystem.
///
/// The fields stay intentionally small in this first PR and cover the existing
/// skills config surface. Provider-specific auth and richer backend configs
/// arrive with the concrete backend implementations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct BlobStoreConfig {
    pub backend: BlobStoreBackend,
    pub path: String,
    pub bucket: Option<String>,
    pub prefix: Option<String>,
    pub region: Option<String>,
    pub endpoint: Option<String>,
    pub read_retry_window_ms: u64,
    pub read_retry_max_attempts: u32,
}

impl Default for BlobStoreConfig {
    fn default() -> Self {
        Self {
            backend: BlobStoreBackend::Filesystem,
            path: "/var/smg/skills".to_string(),
            bucket: None,
            prefix: None,
            region: None,
            endpoint: None,
            read_retry_window_ms: 2000,
            read_retry_max_attempts: 3,
        }
    }
}

/// Local read-cache configuration layered in front of blob reads.
///
/// The cache stays disabled by default until the operator opts in with a
/// positive `max_size_mb`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct BlobCacheConfig {
    pub path: String,
    pub max_size_mb: usize,
}

impl BlobCacheConfig {
    #[must_use]
    pub fn max_size_bytes(&self) -> u64 {
        (self.max_size_mb as u64).saturating_mul(1024 * 1024)
    }

    #[must_use]
    pub fn enabled(&self) -> bool {
        self.max_size_mb > 0
    }
}

impl Default for BlobCacheConfig {
    fn default() -> Self {
        Self {
            path: "/var/smg/cache/skills".to_string(),
            max_size_mb: 0,
        }
    }
}
