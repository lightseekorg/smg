use async_trait::async_trait;

use crate::types::{
    BlobKey, BlobMetadata, BlobPrefix, GetBlobResponse, ListBlobsPage, PutBlobRequest,
};

/// Blob/object-store error surface used by the skills subsystem.
#[derive(Debug, thiserror::Error)]
pub enum BlobStoreError {
    #[error("blob not found: {key}")]
    NotFound { key: String },

    #[error("invalid blob key `{key}`: {message}")]
    InvalidKey { key: String, message: String },

    #[error("blob store operation `{operation}` failed: {message}")]
    Operation {
        operation: &'static str,
        message: String,
    },
}

/// Backend-neutral blob/object storage contract.
#[async_trait]
pub trait BlobStore: Send + Sync + 'static {
    async fn put_stream(
        &self,
        key: &BlobKey,
        request: PutBlobRequest,
    ) -> Result<BlobMetadata, BlobStoreError>;

    async fn get(&self, key: &BlobKey) -> Result<GetBlobResponse, BlobStoreError>;

    async fn head(&self, key: &BlobKey) -> Result<Option<BlobMetadata>, BlobStoreError>;

    async fn delete(&self, key: &BlobKey) -> Result<(), BlobStoreError>;

    async fn list_prefix(
        &self,
        prefix: &BlobPrefix,
        cursor: Option<String>,
        limit: usize,
    ) -> Result<ListBlobsPage, BlobStoreError>;
}
