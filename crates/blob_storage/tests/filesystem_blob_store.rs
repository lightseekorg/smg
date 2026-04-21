use std::{
    io::Cursor,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use anyhow::Result;
use async_trait::async_trait;
use smg_blob_storage::{
    BlobCacheConfig, BlobKey, BlobMetadata, BlobPrefix, BlobStore, FilesystemBlobStore,
    GetBlobResponse, ListBlobsPage, PutBlobRequest,
};
use tempfile::TempDir;
use tokio::io::AsyncReadExt;

fn put_request(bytes: &[u8]) -> PutBlobRequest {
    PutBlobRequest {
        reader: Box::pin(Cursor::new(bytes.to_vec())),
        content_length: bytes.len() as u64,
        content_type: None,
    }
}

async fn read_all(response: GetBlobResponse) -> Result<Vec<u8>> {
    let mut reader = response.reader;
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer).await?;
    Ok(buffer)
}

#[tokio::test]
async fn filesystem_store_round_trips_and_lists_prefixes() -> Result<()> {
    let root = TempDir::new()?;
    let store = FilesystemBlobStore::new(root.path())?;
    let key = BlobKey::from("skills/t1/s1/v1/SKILL.md");
    let aux_key = BlobKey::from("skills/t1/s1/v1/scripts/run.py");

    let metadata = store.put_stream(&key, put_request(b"hello skill")).await?;
    assert_eq!(metadata.key, key);
    assert_eq!(metadata.size_bytes, 11);

    store
        .put_stream(&aux_key, put_request(b"print('hi')"))
        .await?;

    let head = store
        .head(&key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("missing head"))?;
    assert_eq!(head.size_bytes, 11);

    let body = read_all(store.get(&key).await?).await?;
    assert_eq!(body, b"hello skill");

    let page = store
        .list_prefix(&BlobPrefix::from("skills/t1/s1"), None, 10)
        .await?;
    assert_eq!(page.blobs.len(), 2);

    store.delete(&key).await?;
    assert!(store.head(&key).await?.is_none());
    Ok(())
}

#[tokio::test]
async fn filesystem_store_rejects_invalid_blob_keys() -> Result<()> {
    let root = TempDir::new()?;
    let store = FilesystemBlobStore::new(root.path())?;

    let error = store
        .put_stream(&BlobKey::from("../escape"), put_request(b"bad"))
        .await;
    let Err(error) = error else {
        return Err(anyhow::anyhow!("invalid blob key should fail"));
    };
    assert!(matches!(
        error,
        smg_blob_storage::BlobStoreError::InvalidKey { .. }
    ));
    Ok(())
}

struct CountingBlobStore {
    inner: FilesystemBlobStore,
    get_calls: AtomicUsize,
    etag: Option<&'static str>,
}

#[async_trait]
impl BlobStore for CountingBlobStore {
    async fn put_stream(
        &self,
        key: &BlobKey,
        request: PutBlobRequest,
    ) -> Result<BlobMetadata, smg_blob_storage::BlobStoreError> {
        self.inner.put_stream(key, request).await
    }

    async fn get(
        &self,
        key: &BlobKey,
    ) -> Result<GetBlobResponse, smg_blob_storage::BlobStoreError> {
        self.get_calls.fetch_add(1, Ordering::SeqCst);
        let mut response = self.inner.get(key).await?;
        response.metadata.etag = self.etag.map(str::to_string);
        Ok(response)
    }

    async fn head(
        &self,
        key: &BlobKey,
    ) -> Result<Option<BlobMetadata>, smg_blob_storage::BlobStoreError> {
        self.inner.head(key).await
    }

    async fn delete(&self, key: &BlobKey) -> Result<(), smg_blob_storage::BlobStoreError> {
        self.inner.delete(key).await
    }

    async fn list_prefix(
        &self,
        prefix: &BlobPrefix,
        cursor: Option<String>,
        limit: usize,
    ) -> Result<ListBlobsPage, smg_blob_storage::BlobStoreError> {
        self.inner.list_prefix(prefix, cursor, limit).await
    }
}

#[tokio::test]
async fn cached_store_avoids_repeated_backend_reads() -> Result<()> {
    let blob_root = TempDir::new()?;
    let cache_root = TempDir::new()?;
    let inner = Arc::new(CountingBlobStore {
        inner: FilesystemBlobStore::new(blob_root.path())?,
        get_calls: AtomicUsize::new(0),
        etag: None,
    });
    let key = BlobKey::from("skills/t1/s1/v1/scripts/run.py");

    inner
        .put_stream(&key, put_request(b"print('cached')"))
        .await?;

    // Replace the factory-created backend with the counting backend so the test
    // can prove the cache short-circuits the second read.
    let store = Arc::new(smg_blob_storage::CachedBlobStore::new(
        inner.clone() as Arc<dyn BlobStore>,
        &BlobCacheConfig {
            path: cache_root.path().display().to_string(),
            max_size_mb: 8,
        },
    )?) as Arc<dyn BlobStore>;

    let first = read_all(store.get(&key).await?).await?;
    let second = read_all(store.get(&key).await?).await?;

    assert_eq!(first, b"print('cached')");
    assert_eq!(second, b"print('cached')");
    assert_eq!(inner.get_calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn cached_store_rejects_zero_sized_cache_config() -> Result<()> {
    let blob_root = TempDir::new()?;
    let inner = Arc::new(FilesystemBlobStore::new(blob_root.path())?) as Arc<dyn BlobStore>;
    let cache_root = TempDir::new()?;

    let error = smg_blob_storage::CachedBlobStore::new(
        inner,
        &BlobCacheConfig {
            path: cache_root.path().display().to_string(),
            max_size_mb: 0,
        },
    )
    .expect_err("zero-sized cache config should be rejected");

    assert!(matches!(
        error,
        smg_blob_storage::BlobStoreInitError::InvalidConfig { .. }
    ));
    Ok(())
}

#[tokio::test]
async fn cached_store_keeps_oversized_latest_blob_readable() -> Result<()> {
    let blob_root = TempDir::new()?;
    let cache_root = TempDir::new()?;
    let inner = Arc::new(CountingBlobStore {
        inner: FilesystemBlobStore::new(blob_root.path())?,
        get_calls: AtomicUsize::new(0),
        etag: None,
    });
    let key = BlobKey::from("skills/t1/s1/v1/large.bin");
    let payload = vec![b'x'; 2 * 1024 * 1024];

    inner.put_stream(&key, put_request(&payload)).await?;

    let store = Arc::new(smg_blob_storage::CachedBlobStore::new(
        inner.clone() as Arc<dyn BlobStore>,
        &BlobCacheConfig {
            path: cache_root.path().display().to_string(),
            max_size_mb: 1,
        },
    )?) as Arc<dyn BlobStore>;

    let first = read_all(store.get(&key).await?).await?;
    let second = read_all(store.get(&key).await?).await?;

    assert_eq!(first.len(), payload.len());
    assert_eq!(second.len(), payload.len());
    assert_eq!(inner.get_calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test]
async fn cached_store_hits_when_backend_returns_etag_metadata() -> Result<()> {
    let blob_root = TempDir::new()?;
    let cache_root = TempDir::new()?;
    let inner = Arc::new(CountingBlobStore {
        inner: FilesystemBlobStore::new(blob_root.path())?,
        get_calls: AtomicUsize::new(0),
        etag: Some("etag-v1"),
    });
    let key = BlobKey::from("skills/t1/s1/v1/SKILL.md");

    inner.put_stream(&key, put_request(b"etagged")).await?;

    let store = Arc::new(smg_blob_storage::CachedBlobStore::new(
        inner.clone() as Arc<dyn BlobStore>,
        &BlobCacheConfig {
            path: cache_root.path().display().to_string(),
            max_size_mb: 8,
        },
    )?) as Arc<dyn BlobStore>;

    let first = read_all(store.get(&key).await?).await?;
    let second = read_all(store.get(&key).await?).await?;

    assert_eq!(first, b"etagged");
    assert_eq!(second, b"etagged");
    assert_eq!(inner.get_calls.load(Ordering::SeqCst), 1);
    Ok(())
}
