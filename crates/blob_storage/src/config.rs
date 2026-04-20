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
