use std::{
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
};

use async_trait::async_trait;
use lru::LruCache;
use parking_lot::Mutex;
use tokio::{fs, fs::File};
use uuid::Uuid;

use crate::{
    config::BlobCacheConfig,
    factory::BlobStoreInitError,
    store::{BlobStore, BlobStoreError},
    types::{BlobKey, BlobMetadata, BlobPrefix, GetBlobResponse, ListBlobsPage, PutBlobRequest},
};

#[derive(Debug)]
struct CacheEntry {
    path: PathBuf,
    size_bytes: u64,
    metadata: BlobMetadata,
}

#[derive(Debug)]
struct CacheState {
    current_size_bytes: u64,
    entries: LruCache<String, CacheEntry>,
}

impl Default for CacheState {
    fn default() -> Self {
        Self {
            current_size_bytes: 0,
            entries: LruCache::unbounded(),
        }
    }
}

#[derive(Debug)]
struct LocalBlobCache {
    cache_dir: PathBuf,
    max_size_bytes: u64,
    state: Mutex<CacheState>,
}

impl LocalBlobCache {
    fn new(config: &BlobCacheConfig) -> Result<Self, BlobStoreInitError> {
        let cache_root = PathBuf::from(&config.path);
        if cache_root.as_os_str().is_empty() {
            return Err(BlobStoreInitError::InvalidConfig {
                message: "blob cache path must not be empty".to_string(),
            });
        }
        if config.max_size_bytes() == 0 {
            return Err(BlobStoreInitError::InvalidConfig {
                message: "blob cache max_size_mb must be greater than zero".to_string(),
            });
        }
        std::fs::create_dir_all(&cache_root).map_err(|error| BlobStoreInitError::Io {
            path: cache_root.display().to_string(),
            message: error.to_string(),
        })?;
        // Keep cache state process-local without deleting the configured root.
        // Each instance gets its own empty subdirectory under that root.
        let cache_dir = cache_root.join(format!("instance-{}", Uuid::now_v7()));
        std::fs::create_dir_all(&cache_dir).map_err(|error| BlobStoreInitError::Io {
            path: cache_dir.display().to_string(),
            message: error.to_string(),
        })?;

        Ok(Self {
            cache_dir,
            max_size_bytes: config.max_size_bytes(),
            state: Mutex::new(CacheState {
                current_size_bytes: 0,
                entries: LruCache::unbounded(),
            }),
        })
    }

    async fn get(&self, key: &BlobKey) -> Result<Option<GetBlobResponse>, BlobStoreError> {
        let cache_key = cache_key_for(key);
        let cached = {
            let mut state = self.state.lock();
            state.entries.get(&cache_key).map(|entry| CachedLookup {
                path: entry.path.clone(),
                metadata: entry.metadata.clone(),
            })
        };

        let Some(cached) = cached else {
            return Ok(None);
        };

        match File::open(&cached.path).await {
            Ok(file) => Ok(Some(GetBlobResponse {
                reader: Box::pin(file),
                metadata: cached.metadata,
            })),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                self.remove_cache_key(&cache_key).await?;
                Ok(None)
            }
            Err(error) => Err(BlobStoreError::Operation {
                operation: "cache_open",
                message: error.to_string(),
            }),
        }
    }

    async fn insert(
        &self,
        key: &BlobKey,
        response: GetBlobResponse,
    ) -> Result<GetBlobResponse, BlobStoreError> {
        if response.metadata.size_bytes > self.max_size_bytes {
            return Ok(response);
        }

        let cache_key = cache_key_for(key);
        let cache_path = cache_path_for(&self.cache_dir, &cache_key);
        let temp_path = cache_path.with_extension(format!("{}.tmp", Uuid::now_v7()));

        let mut reader = response.reader;
        let copied_result = async {
            let mut file =
                File::create(&temp_path)
                    .await
                    .map_err(|error| BlobStoreError::Operation {
                        operation: "cache_create",
                        message: error.to_string(),
                    })?;
            let copied = tokio::io::copy(&mut reader, &mut file)
                .await
                .map_err(|error| BlobStoreError::Operation {
                    operation: "cache_write",
                    message: error.to_string(),
                })?;
            file.sync_all()
                .await
                .map_err(|error| BlobStoreError::Operation {
                    operation: "cache_sync_all",
                    message: error.to_string(),
                })?;
            drop(file);

            fs::rename(&temp_path, &cache_path).await.map_err(|error| {
                BlobStoreError::Operation {
                    operation: "cache_rename",
                    message: error.to_string(),
                }
            })?;

            Ok::<u64, BlobStoreError>(copied)
        }
        .await;
        let copied = match copied_result {
            Ok(copied) => copied,
            Err(error) => {
                let _ = remove_file_if_present(&temp_path).await;
                return Err(error);
            }
        };

        let mut metadata = response.metadata;
        metadata.size_bytes = copied;

        {
            let mut state = self.state.lock();
            if let Some(previous) = state.entries.put(
                cache_key.clone(),
                CacheEntry {
                    path: cache_path.clone(),
                    size_bytes: copied,
                    metadata: metadata.clone(),
                },
            ) {
                state.current_size_bytes =
                    state.current_size_bytes.saturating_sub(previous.size_bytes);
            }
            state.current_size_bytes = state.current_size_bytes.saturating_add(copied);
        }

        self.evict_if_needed(Some(&cache_key)).await?;

        let file = File::open(&cache_path)
            .await
            .map_err(|error| BlobStoreError::Operation {
                operation: "cache_open",
                message: error.to_string(),
            })?;

        Ok(GetBlobResponse {
            reader: Box::pin(file),
            metadata,
        })
    }

    async fn invalidate(&self, key: &BlobKey) -> Result<(), BlobStoreError> {
        self.remove_cache_key(&cache_key_for(key)).await
    }

    async fn evict_if_needed(
        &self,
        protected_cache_key: Option<&str>,
    ) -> Result<(), BlobStoreError> {
        loop {
            let evicted = {
                let mut state = self.state.lock();
                if state.current_size_bytes <= self.max_size_bytes {
                    None
                } else {
                    match state.entries.pop_lru() {
                        Some((cache_key, entry))
                            if protected_cache_key
                                .is_some_and(|protected| protected == cache_key) =>
                        {
                            state.entries.put(cache_key, entry);
                            None
                        }
                        Some((_, entry)) => {
                            state.current_size_bytes =
                                state.current_size_bytes.saturating_sub(entry.size_bytes);
                            Some(entry)
                        }
                        None => None,
                    }
                }
            };

            let Some(entry) = evicted else {
                return Ok(());
            };

            remove_file_if_present(&entry.path).await?;
        }
    }

    async fn remove_cache_key(&self, cache_key: &str) -> Result<(), BlobStoreError> {
        let removed = {
            let mut state = self.state.lock();
            state.entries.pop(cache_key).inspect(|entry| {
                state.current_size_bytes =
                    state.current_size_bytes.saturating_sub(entry.size_bytes);
            })
        };

        if let Some(entry) = removed {
            remove_file_if_present(&entry.path).await?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
struct CachedLookup {
    path: PathBuf,
    metadata: BlobMetadata,
}

/// Generic local read-cache wrapper layered in front of any blob store.
pub struct CachedBlobStore {
    inner: Arc<dyn BlobStore>,
    cache: LocalBlobCache,
}

impl fmt::Debug for CachedBlobStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CachedBlobStore").finish_non_exhaustive()
    }
}

impl CachedBlobStore {
    pub fn new(
        inner: Arc<dyn BlobStore>,
        config: &BlobCacheConfig,
    ) -> Result<Self, BlobStoreInitError> {
        Ok(Self {
            inner,
            cache: LocalBlobCache::new(config)?,
        })
    }
}

#[async_trait]
impl BlobStore for CachedBlobStore {
    async fn put_stream(
        &self,
        key: &BlobKey,
        request: PutBlobRequest,
    ) -> Result<BlobMetadata, BlobStoreError> {
        let _ = self.cache.invalidate(key).await;
        self.inner.put_stream(key, request).await
    }

    async fn get(&self, key: &BlobKey) -> Result<GetBlobResponse, BlobStoreError> {
        if let Ok(Some(response)) = self.cache.get(key).await {
            return Ok(response);
        }

        let response = self.inner.get(key).await?;
        match self.cache.insert(key, response).await {
            Ok(cached) => Ok(cached),
            Err(_) => self.inner.get(key).await,
        }
    }

    async fn head(&self, key: &BlobKey) -> Result<Option<BlobMetadata>, BlobStoreError> {
        self.inner.head(key).await
    }

    async fn delete(&self, key: &BlobKey) -> Result<(), BlobStoreError> {
        let _ = self.cache.invalidate(key).await;
        self.inner.delete(key).await
    }

    async fn list_prefix(
        &self,
        prefix: &BlobPrefix,
        cursor: Option<String>,
        limit: usize,
    ) -> Result<ListBlobsPage, BlobStoreError> {
        self.inner.list_prefix(prefix, cursor, limit).await
    }
}

async fn remove_file_if_present(path: &Path) -> Result<(), BlobStoreError> {
    match fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(BlobStoreError::Operation {
            operation: "cache_remove_file",
            message: error.to_string(),
        }),
    }
}

fn cache_key_for(key: &BlobKey) -> String {
    // This local cache is invalidation-driven within the current process, so it
    // keys by the logical blob key rather than embedding the backend etag.
    // Doing otherwise makes cached etag-bearing reads unreachable because the
    // lookup path does not know the current etag ahead of time.
    key.0.clone()
}

fn cache_path_for(cache_dir: &Path, cache_key: &str) -> PathBuf {
    let digest = blake3::hash(cache_key.as_bytes());
    cache_dir.join(digest.to_hex().as_str())
}
