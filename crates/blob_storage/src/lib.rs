//! Generic blob/object storage contracts for the skills subsystem.
//!
//! This crate intentionally defines only backend-neutral configuration and
//! trait types. Concrete filesystem and cloud-provider implementations land in
//! later PRs.

mod config;
mod store;
mod types;

pub use config::{BlobStoreBackend, BlobStoreConfig};
pub use store::{BlobStore, BlobStoreError};
pub use types::{
    BlobKey, BlobMetadata, BlobPrefix, GetBlobResponse, ListBlobsPage, PutBlobRequest,
};
