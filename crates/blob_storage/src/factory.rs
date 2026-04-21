use std::sync::Arc;

use thiserror::Error;

use crate::{
    cache::CachedBlobStore,
    config::{BlobCacheConfig, BlobStoreBackend, BlobStoreConfig},
    filesystem::FilesystemBlobStore,
    store::BlobStore,
};

/// Initialization-time failures for blob-store construction.
#[derive(Debug, Error)]
pub enum BlobStoreInitError {
    #[error("blob store backend `{backend:?}` is not implemented yet")]
    UnsupportedBackend { backend: BlobStoreBackend },

    #[error("invalid blob store configuration: {message}")]
    InvalidConfig { message: String },

    #[error("blob store initialization failed at `{path}`: {message}")]
    Io { path: String, message: String },
}

/// Build the configured blob store, optionally layering in the local read
/// cache for successful blob reads.
pub fn create_blob_store(
    store_config: &BlobStoreConfig,
    cache_config: Option<&BlobCacheConfig>,
) -> Result<Arc<dyn BlobStore>, BlobStoreInitError> {
    let backend: Arc<dyn BlobStore> = match store_config.backend {
        BlobStoreBackend::Filesystem => Arc::new(FilesystemBlobStore::new(&store_config.path)?),
        backend => return Err(BlobStoreInitError::UnsupportedBackend { backend }),
    };

    match cache_config.filter(|config| config.enabled()) {
        Some(config) => Ok(Arc::new(CachedBlobStore::new(backend, config)?)),
        None => Ok(backend),
    }
}
