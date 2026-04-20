//! Generic blob/object storage contracts for the skills subsystem.
//!
//! This crate defines the backend-neutral contract plus the first concrete
//! single-process implementation path: filesystem-backed blobs with a local
//! read cache.

mod cache;
mod config;
mod factory;
mod filesystem;
mod store;
mod types;

pub use cache::CachedBlobStore;
pub use config::{BlobCacheConfig, BlobStoreBackend, BlobStoreConfig};
pub use factory::{create_blob_store, BlobStoreInitError};
pub use filesystem::FilesystemBlobStore;
pub use store::{BlobStore, BlobStoreError};
pub use types::{
    BlobKey, BlobMetadata, BlobPrefix, GetBlobResponse, ListBlobsPage, PutBlobRequest,
};
