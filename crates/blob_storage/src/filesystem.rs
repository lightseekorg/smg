use std::{
    path::{Path, PathBuf},
    time::SystemTime,
};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use tokio::fs::{self, File};
use uuid::Uuid;

use crate::{
    factory::BlobStoreInitError,
    store::{BlobStore, BlobStoreError},
    types::{BlobKey, BlobMetadata, BlobPrefix, GetBlobResponse, ListBlobsPage, PutBlobRequest},
};

/// Local filesystem-backed blob store used for development and early testing.
#[derive(Debug, Clone)]
pub struct FilesystemBlobStore {
    root_dir: PathBuf,
}

impl FilesystemBlobStore {
    pub fn new(root_dir: impl AsRef<Path>) -> Result<Self, BlobStoreInitError> {
        let root_dir = root_dir.as_ref();
        if root_dir.as_os_str().is_empty() {
            return Err(BlobStoreInitError::InvalidConfig {
                message: "filesystem blob-store path must not be empty".to_string(),
            });
        }

        std::fs::create_dir_all(root_dir).map_err(|error| BlobStoreInitError::Io {
            path: root_dir.display().to_string(),
            message: error.to_string(),
        })?;

        Ok(Self {
            root_dir: root_dir.to_path_buf(),
        })
    }

    fn resolve_blob_path(&self, key: &BlobKey) -> Result<PathBuf, BlobStoreError> {
        validate_blob_key(&key.0)?;
        let mut path = self.root_dir.clone();
        for segment in key.0.split('/') {
            path.push(segment);
        }
        Ok(path)
    }

    fn metadata_for_path(key: &BlobKey, metadata: std::fs::Metadata) -> BlobMetadata {
        BlobMetadata {
            key: key.clone(),
            size_bytes: metadata.len(),
            etag: None,
            last_modified: metadata.modified().ok().map(to_utc),
        }
    }
}

#[async_trait]
impl BlobStore for FilesystemBlobStore {
    async fn put_stream(
        &self,
        key: &BlobKey,
        mut request: PutBlobRequest,
    ) -> Result<BlobMetadata, BlobStoreError> {
        let path = self.resolve_blob_path(key)?;
        let parent = path.parent().ok_or_else(|| BlobStoreError::InvalidKey {
            key: key.0.clone(),
            message: "blob key must contain at least one path segment".to_string(),
        })?;

        fs::create_dir_all(parent)
            .await
            .map_err(|error| BlobStoreError::Operation {
                operation: "create_dir_all",
                message: error.to_string(),
            })?;

        let temp_path = path.with_extension(format!("{}.tmp", Uuid::now_v7()));
        let copied_result = async {
            let mut file =
                File::create(&temp_path)
                    .await
                    .map_err(|error| BlobStoreError::Operation {
                        operation: "create",
                        message: error.to_string(),
                    })?;

            let copied = tokio::io::copy(&mut request.reader, &mut file)
                .await
                .map_err(|error| BlobStoreError::Operation {
                    operation: "write",
                    message: error.to_string(),
                })?;
            file.sync_all()
                .await
                .map_err(|error| BlobStoreError::Operation {
                    operation: "sync_all",
                    message: error.to_string(),
                })?;
            drop(file);

            fs::rename(&temp_path, &path)
                .await
                .map_err(|error| BlobStoreError::Operation {
                    operation: "rename",
                    message: error.to_string(),
                })?;

            Ok::<u64, BlobStoreError>(copied)
        }
        .await;
        let copied = match copied_result {
            Ok(copied) => copied,
            Err(error) => {
                let _ = fs::remove_file(&temp_path).await;
                return Err(error);
            }
        };

        let metadata = fs::metadata(&path)
            .await
            .map_err(|error| BlobStoreError::Operation {
                operation: "metadata",
                message: error.to_string(),
            })?;
        let mut blob_metadata = Self::metadata_for_path(key, metadata);
        blob_metadata.size_bytes = copied;
        Ok(blob_metadata)
    }

    async fn get(&self, key: &BlobKey) -> Result<GetBlobResponse, BlobStoreError> {
        let path = self.resolve_blob_path(key)?;
        let file = File::open(&path).await.map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                BlobStoreError::NotFound { key: key.0.clone() }
            } else {
                BlobStoreError::Operation {
                    operation: "open",
                    message: error.to_string(),
                }
            }
        })?;
        let metadata = fs::metadata(&path)
            .await
            .map_err(|error| BlobStoreError::Operation {
                operation: "metadata",
                message: error.to_string(),
            })?;

        Ok(GetBlobResponse {
            reader: Box::pin(file),
            metadata: Self::metadata_for_path(key, metadata),
        })
    }

    async fn head(&self, key: &BlobKey) -> Result<Option<BlobMetadata>, BlobStoreError> {
        let path = self.resolve_blob_path(key)?;
        match fs::metadata(&path).await {
            Ok(metadata) => Ok(Some(Self::metadata_for_path(key, metadata))),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(BlobStoreError::Operation {
                operation: "metadata",
                message: error.to_string(),
            }),
        }
    }

    async fn delete(&self, key: &BlobKey) -> Result<(), BlobStoreError> {
        let path = self.resolve_blob_path(key)?;
        fs::remove_file(&path).await.map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                BlobStoreError::NotFound { key: key.0.clone() }
            } else {
                BlobStoreError::Operation {
                    operation: "remove_file",
                    message: error.to_string(),
                }
            }
        })?;
        Ok(())
    }

    async fn list_prefix(
        &self,
        prefix: &BlobPrefix,
        cursor: Option<String>,
        limit: usize,
    ) -> Result<ListBlobsPage, BlobStoreError> {
        let normalized_prefix = normalize_prefix(&prefix.0)?;
        let mut blob_entries = Vec::new();
        let search_root = search_root_for_prefix(&self.root_dir, &normalized_prefix);
        match fs::metadata(&search_root).await {
            Ok(metadata) if metadata.is_dir() => {
                collect_blob_entries(&self.root_dir, &search_root, &mut blob_entries).await?;
            }
            Ok(_) => {
                return Ok(ListBlobsPage {
                    blobs: Vec::new(),
                    next_cursor: None,
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(ListBlobsPage {
                    blobs: Vec::new(),
                    next_cursor: None,
                });
            }
            Err(error) => {
                return Err(BlobStoreError::Operation {
                    operation: "metadata",
                    message: error.to_string(),
                });
            }
        }
        blob_entries.retain(|entry| entry.key.starts_with(&normalized_prefix));
        blob_entries.sort_by(|left, right| left.key.cmp(&right.key));

        let start_index = match cursor.as_ref() {
            Some(cursor_key) => match blob_entries
                .iter()
                .position(|entry| entry.key > *cursor_key)
            {
                Some(index) => index,
                None => {
                    return Ok(ListBlobsPage {
                        blobs: Vec::new(),
                        next_cursor: None,
                    });
                }
            },
            None => 0,
        };

        let end_index =
            start_index.saturating_add(limit.min(blob_entries.len().saturating_sub(start_index)));
        let mut blobs = Vec::with_capacity(end_index.saturating_sub(start_index));
        for entry in &blob_entries[start_index..end_index] {
            match fs::metadata(&entry.path).await {
                Ok(metadata) => {
                    blobs.push(Self::metadata_for_path(
                        &BlobKey(entry.key.clone()),
                        metadata,
                    ));
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => {
                    return Err(BlobStoreError::Operation {
                        operation: "metadata",
                        message: error.to_string(),
                    });
                }
            }
        }

        let next_cursor = if end_index < blob_entries.len() && end_index > start_index {
            blob_entries
                .get(end_index - 1)
                .map(|entry| entry.key.clone())
        } else {
            None
        };

        Ok(ListBlobsPage { blobs, next_cursor })
    }
}

fn search_root_for_prefix(root_dir: &Path, normalized_prefix: &str) -> PathBuf {
    let Some((parent_prefix, _)) = normalized_prefix.rsplit_once('/') else {
        return root_dir.to_path_buf();
    };
    let mut path = root_dir.to_path_buf();
    for segment in parent_prefix.split('/') {
        path.push(segment);
    }
    path
}

#[derive(Debug, Clone)]
struct BlobListEntry {
    key: String,
    path: PathBuf,
}

async fn collect_blob_entries(
    root_dir: &Path,
    current_dir: &Path,
    blob_entries: &mut Vec<BlobListEntry>,
) -> Result<(), BlobStoreError> {
    let mut entries =
        fs::read_dir(current_dir)
            .await
            .map_err(|error| BlobStoreError::Operation {
                operation: "read_dir",
                message: error.to_string(),
            })?;

    while let Some(entry) =
        entries
            .next_entry()
            .await
            .map_err(|error| BlobStoreError::Operation {
                operation: "read_dir",
                message: error.to_string(),
            })?
    {
        let path = entry.path();
        let file_type = match entry.file_type().await {
            Ok(file_type) => file_type,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(BlobStoreError::Operation {
                    operation: "file_type",
                    message: error.to_string(),
                });
            }
        };
        if file_type.is_dir() {
            Box::pin(collect_blob_entries(root_dir, &path, blob_entries)).await?;
            continue;
        }
        if !file_type.is_file() {
            continue;
        }

        let relative_path =
            path.strip_prefix(root_dir)
                .map_err(|error| BlobStoreError::Operation {
                    operation: "strip_prefix",
                    message: error.to_string(),
                })?;
        let key = relative_path
            .iter()
            .map(|segment| segment.to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        blob_entries.push(BlobListEntry { key, path });
    }

    Ok(())
}

fn validate_blob_key(raw_key: &str) -> Result<(), BlobStoreError> {
    if raw_key.is_empty() {
        return Err(BlobStoreError::InvalidKey {
            key: raw_key.to_string(),
            message: "blob key must not be empty".to_string(),
        });
    }
    if raw_key.starts_with('/') {
        return Err(BlobStoreError::InvalidKey {
            key: raw_key.to_string(),
            message: "blob key must be relative".to_string(),
        });
    }

    for segment in raw_key.split('/') {
        if segment.is_empty() {
            return Err(BlobStoreError::InvalidKey {
                key: raw_key.to_string(),
                message: "blob key must not contain empty path segments".to_string(),
            });
        }
        if matches!(segment, "." | "..") {
            return Err(BlobStoreError::InvalidKey {
                key: raw_key.to_string(),
                message: "blob key must not contain traversal segments".to_string(),
            });
        }
        if segment.contains('\\') || segment.contains('\0') {
            return Err(BlobStoreError::InvalidKey {
                key: raw_key.to_string(),
                message: "blob key contains unsupported path characters".to_string(),
            });
        }
    }

    Ok(())
}

fn normalize_prefix(raw_prefix: &str) -> Result<String, BlobStoreError> {
    let trimmed = raw_prefix.trim_start_matches('/');
    if trimmed.is_empty() {
        return Ok(String::new());
    }

    let has_trailing_slash = trimmed.ends_with('/');
    let core = trimmed.trim_end_matches('/');
    if core.is_empty() {
        return Ok(String::new());
    }

    validate_blob_key(core)?;
    if has_trailing_slash {
        Ok(format!("{core}/"))
    } else {
        Ok(core.to_string())
    }
}

fn to_utc(system_time: SystemTime) -> DateTime<Utc> {
    DateTime::<Utc>::from(system_time)
}
