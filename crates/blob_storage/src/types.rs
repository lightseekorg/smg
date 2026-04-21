use std::pin::Pin;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncRead;

/// Canonical blob key used across blob-store backends.
///
/// This type is an opaque label, not a validated filesystem path. Concrete
/// backends are responsible for validating and rejecting dangerous keys such as
/// traversal attempts or absolute paths before using them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlobKey(pub String);

impl From<String> for BlobKey {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for BlobKey {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

/// Prefix used for paginated blob listings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlobPrefix(pub String);

impl From<String> for BlobPrefix {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for BlobPrefix {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

/// Backend-neutral blob metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlobMetadata {
    pub key: BlobKey,
    pub size_bytes: u64,
    pub etag: Option<String>,
    pub last_modified: Option<DateTime<Utc>>,
}

/// Streaming upload request for a single blob.
pub struct PutBlobRequest {
    pub reader: Pin<Box<dyn AsyncRead + Send>>,
    pub content_length: u64,
    pub content_type: Option<String>,
}

/// Streaming download response for a single blob.
pub struct GetBlobResponse {
    pub reader: Pin<Box<dyn AsyncRead + Send>>,
    pub metadata: BlobMetadata,
}

/// One page of blob-listing results.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ListBlobsPage {
    pub blobs: Vec<BlobMetadata>,
    pub next_cursor: Option<String>,
}
