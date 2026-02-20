use std::{future::Future, time::Duration};

use futures::{Stream, StreamExt};
use tracing::debug;

pub const MAX_ZIP_ENTRIES: usize = 50;
pub const MAX_UNCOMPRESSED_SIZE: u64 = 500 * 1024 * 1024;
pub const MAX_STREAM_BUNDLE_SIZE: usize = 200 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct StreamBundle {
    pub sha256: String,
    pub compressed_data: Vec<u8>,
}

pub async fn collect_stream_bundle<S, C, E, F>(
    stream: &mut S,
    extract: F,
) -> Result<StreamBundle, String>
where
    S: Stream<Item = Result<C, E>> + Unpin,
    E: std::fmt::Display,
    F: Fn(C) -> (Vec<u8>, String),
{
    let mut sha256 = String::new();
    let mut data = Vec::new();
    let mut saw_chunk = false;
    let mut last_chunk_had_sha = false;

    while let Some(result) = stream.next().await {
        let chunk = result.map_err(|e| format!("Stream error: {}", e))?;
        let (chunk_data, chunk_sha) = extract(chunk);
        saw_chunk = true;

        last_chunk_had_sha = !chunk_sha.is_empty();
        if last_chunk_had_sha {
            sha256 = chunk_sha;
        }

        let new_total = data
            .len()
            .checked_add(chunk_data.len())
            .ok_or_else(|| "Stream bundle size overflow".to_string())?;
        if new_total > MAX_STREAM_BUNDLE_SIZE {
            return Err(format!(
                "Stream bundle exceeds maximum size limit ({} bytes > {} bytes)",
                new_total, MAX_STREAM_BUNDLE_SIZE
            ));
        }

        data.extend_from_slice(&chunk_data);
    }

    if !saw_chunk {
        return Err("Empty stream: no chunks received".to_string());
    }

    if !last_chunk_had_sha {
        return Err("Stream ended without terminal sha256 fingerprint".to_string());
    }

    if data.is_empty() {
        return Err("Received empty stream bundle".to_string());
    }

    debug!(
        "Stream bundle received: {} bytes, sha256={}",
        data.len(),
        sha256
    );

    Ok(StreamBundle {
        sha256,
        compressed_data: data,
    })
}

/// Wraps both the RPC handshake and stream collection inside a single timeout.
///
/// This ensures the entire operation (connection + streaming) is bounded, eliminating
/// the gap where the RPC handshake could hang without any timeout.
pub async fn collect_bundle_from_rpc<S, C, E, F>(
    rpc_future: impl Future<Output = Result<tonic::Response<S>, tonic::Status>>,
    extract: F,
    timeout_duration: Duration,
) -> Result<StreamBundle, Box<dyn std::error::Error + Send + Sync>>
where
    S: Stream<Item = Result<C, E>> + Unpin,
    E: std::fmt::Display,
    F: Fn(C) -> (Vec<u8>, String),
{
    tokio::time::timeout(timeout_duration, async {
        let mut stream = rpc_future.await?.into_inner();
        collect_stream_bundle(&mut stream, extract)
            .await
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })
    })
    .await
    .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
        format!(
            "get_tokenizer timed out after {}s",
            timeout_duration.as_secs()
        )
        .into()
    })?
}

#[cfg(test)]
mod tests {
    use futures::{executor::block_on, stream};

    use super::*;

    fn identity(chunk: (Vec<u8>, String)) -> (Vec<u8>, String) {
        chunk
    }

    type ChunkResult = Result<(Vec<u8>, String), &'static str>;

    #[test]
    fn test_collect_single_chunk() {
        let mut s = stream::iter(vec![
            Ok((b"zipdata".to_vec(), "abc123".to_string())) as ChunkResult
        ]);

        let bundle = block_on(collect_stream_bundle(&mut s, identity)).unwrap();
        assert_eq!(bundle.compressed_data, b"zipdata");
        assert_eq!(bundle.sha256, "abc123");
    }

    #[test]
    fn test_collect_multiple_chunks() {
        let mut s = stream::iter(vec![
            Ok((b"abc".to_vec(), String::new())) as ChunkResult,
            Ok((b"def".to_vec(), String::new())),
            Ok((b"ghi".to_vec(), "sha".to_string())),
        ]);

        let bundle = block_on(collect_stream_bundle(&mut s, identity)).unwrap();
        assert_eq!(bundle.compressed_data, b"abcdefghi");
        assert_eq!(bundle.sha256, "sha");
    }

    #[test]
    fn test_collect_uses_last_non_empty_sha256() {
        let mut s = stream::iter(vec![
            Ok((b"abc".to_vec(), "sha-old".to_string())) as ChunkResult,
            Ok((b"def".to_vec(), String::new())),
            Ok((b"ghi".to_vec(), "sha-new".to_string())),
        ]);

        let bundle = block_on(collect_stream_bundle(&mut s, identity)).unwrap();
        assert_eq!(bundle.compressed_data, b"abcdefghi");
        assert_eq!(bundle.sha256, "sha-new");
    }

    #[test]
    fn test_collect_rejects_missing_terminal_sha256() {
        let mut s = stream::iter(vec![
            Ok((b"abc".to_vec(), "sha-old".to_string())) as ChunkResult,
            Ok((b"def".to_vec(), String::new())),
        ]);

        let err = block_on(collect_stream_bundle(&mut s, identity)).unwrap_err();
        assert!(err.contains("without terminal sha256"));
    }

    #[test]
    fn test_collect_empty_stream() {
        let mut s = stream::iter(Vec::<ChunkResult>::new());

        let err = block_on(collect_stream_bundle(&mut s, identity)).unwrap_err();
        assert!(err.contains("no chunks received"));
    }

    #[test]
    fn test_collect_stream_error() {
        let mut s = stream::iter(vec![
            Ok((b"abc".to_vec(), String::new())),
            Err("socket closed"),
        ]);

        let err = block_on(collect_stream_bundle(&mut s, identity)).unwrap_err();
        assert!(err.contains("Stream error: socket closed"));
    }

    #[test]
    fn test_collect_exceeds_max_size() {
        let oversized = vec![0u8; MAX_STREAM_BUNDLE_SIZE + 1];
        let mut s = stream::iter(vec![Ok((oversized, String::new())) as ChunkResult]);

        let err = block_on(collect_stream_bundle(&mut s, identity)).unwrap_err();
        assert!(err.contains("exceeds maximum size limit"));
    }

    #[test]
    fn test_collect_rejects_no_sha256() {
        let mut s = stream::iter(vec![Ok((b"data".to_vec(), String::new())) as ChunkResult]);

        let err = block_on(collect_stream_bundle(&mut s, identity)).unwrap_err();
        assert!(err.contains("without terminal sha256"));
    }
}
