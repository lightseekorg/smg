use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tracing::debug;

/// Maximum number of entries allowed in a tokenizer zip archive.
pub const MAX_ZIP_ENTRIES: usize = 50;

/// Maximum total uncompressed size for all extracted files (500 MB).
pub const MAX_UNCOMPRESSED_SIZE: u64 = 500 * 1024 * 1024;

/// Maximum total size for streamed tokenizer bundle data (200 MB).
pub const MAX_TOKENIZER_BUNDLE_SIZE: usize = 200 * 1024 * 1024;

/// Descriptor for a file in the tokenizer bundle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerFileDescriptor {
    pub file_name: String,
    pub mime_type: String,
    pub optional: bool,
}

/// Metadata for a tokenizer bundle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerMetadata {
    pub model_identifier: String,
    pub fingerprint: String,
    pub files: Vec<TokenizerFileDescriptor>,
    pub bundle_format: String,
}

/// Tokenizer bundle containing metadata and compressed data
#[derive(Debug, Clone)]
pub struct TokenizerBundle {
    pub metadata: TokenizerMetadata,
    pub compressed_data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct TokenizerFileChunk {
    pub data: Vec<u8>,
    pub chunk_index: u32,
    pub is_last_chunk: bool,
}

#[derive(Debug, Clone)]
pub enum TokenizerChunk {
    Metadata(TokenizerMetadata),
    FileChunk(TokenizerFileChunk),
}

/// Stateful collector for tokenizer chunks.
#[derive(Debug, Default)]
pub struct TokenizerBundleCollector {
    metadata: Option<TokenizerMetadata>,
    compressed_data: Vec<u8>,
    expected_chunk_index: u32,
    last_chunk_received: bool,
}

impl TokenizerBundleCollector {
    pub fn push_chunk(&mut self, chunk: TokenizerChunk) -> Result<(), String> {
        if self.last_chunk_received {
            return Err("Protocol error: received chunk after final chunk".to_string());
        }

        match chunk {
            TokenizerChunk::Metadata(meta) => {
                if self.metadata.is_some() {
                    return Err(format!(
                        "Protocol error: unexpected metadata chunk at position {}",
                        self.expected_chunk_index
                    ));
                }

                debug!(
                    "Received tokenizer metadata: model={}, fingerprint={}, files={}, format={}",
                    meta.model_identifier,
                    meta.fingerprint,
                    meta.files.len(),
                    meta.bundle_format
                );

                if meta.bundle_format != "zip" {
                    return Err(format!(
                        "Unsupported tokenizer bundle format '{}', expected 'zip'",
                        meta.bundle_format
                    ));
                }

                self.metadata = Some(meta);
                Ok(())
            }
            TokenizerChunk::FileChunk(file_chunk) => {
                if self.metadata.is_none() {
                    return Err(
                        "Protocol error: first chunk must be metadata, got file chunk".to_string(),
                    );
                }

                if file_chunk.chunk_index != self.expected_chunk_index {
                    return Err(format!(
                        "Protocol error: expected chunk index {}, got {}",
                        self.expected_chunk_index, file_chunk.chunk_index
                    ));
                }

                let new_total = self.compressed_data.len() + file_chunk.data.len();
                if new_total > MAX_TOKENIZER_BUNDLE_SIZE {
                    return Err(format!(
                        "Tokenizer bundle exceeds maximum size limit ({} bytes > {} bytes)",
                        new_total, MAX_TOKENIZER_BUNDLE_SIZE
                    ));
                }

                debug!(
                    "Received file chunk {}: {} bytes, is_last={}",
                    file_chunk.chunk_index,
                    file_chunk.data.len(),
                    file_chunk.is_last_chunk
                );

                self.compressed_data.extend_from_slice(&file_chunk.data);
                self.last_chunk_received = file_chunk.is_last_chunk;
                self.expected_chunk_index += 1;

                Ok(())
            }
        }
    }

    pub fn finish(self) -> Result<TokenizerBundle, String> {
        let metadata = self
            .metadata
            .ok_or_else(|| "Empty stream: expected metadata chunk".to_string())?;

        if !self.last_chunk_received {
            return Err("Protocol error: stream ended without receiving final chunk".to_string());
        }

        if self.compressed_data.is_empty() {
            return Err("Protocol error: received empty tokenizer bundle".to_string());
        }

        debug!(
            "Tokenizer bundle received successfully: {} bytes compressed",
            self.compressed_data.len()
        );

        Ok(TokenizerBundle {
            metadata,
            compressed_data: self.compressed_data,
        })
    }
}

/// Parse tokenizer chunks from an async stream into a validated `TokenizerBundle`.
pub async fn collect_tokenizer_bundle<S, C, E, DecodeFn>(
    stream: &mut S,
    decode_chunk: DecodeFn,
) -> Result<TokenizerBundle, String>
where
    S: Stream<Item = Result<C, E>> + Unpin,
    E: std::fmt::Display,
    DecodeFn: Fn(C) -> Result<TokenizerChunk, String>,
{
    let mut collector = TokenizerBundleCollector::default();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Tokenizer stream error: {}", e))?;
        let decoded = decode_chunk(chunk)?;
        collector.push_chunk(decoded)?;
    }

    collector.finish()
}

#[cfg(test)]
mod tests {
    use futures::{executor::block_on, stream};

    use super::*;

    fn metadata() -> TokenizerMetadata {
        TokenizerMetadata {
            model_identifier: "test-model".to_string(),
            fingerprint: "test-fingerprint".to_string(),
            files: vec![TokenizerFileDescriptor {
                file_name: "tokenizer.json".to_string(),
                mime_type: "application/json".to_string(),
                optional: false,
            }],
            bundle_format: "zip".to_string(),
        }
    }

    #[test]
    fn test_collector_accepts_valid_metadata_and_chunks() {
        let mut collector = TokenizerBundleCollector::default();

        collector
            .push_chunk(TokenizerChunk::Metadata(metadata()))
            .unwrap();
        collector
            .push_chunk(TokenizerChunk::FileChunk(TokenizerFileChunk {
                data: b"abc".to_vec(),
                chunk_index: 0,
                is_last_chunk: false,
            }))
            .unwrap();
        collector
            .push_chunk(TokenizerChunk::FileChunk(TokenizerFileChunk {
                data: b"def".to_vec(),
                chunk_index: 1,
                is_last_chunk: true,
            }))
            .unwrap();

        let bundle = collector.finish().unwrap();
        assert_eq!(bundle.metadata.model_identifier, "test-model");
        assert_eq!(bundle.compressed_data, b"abcdef");
    }

    #[test]
    fn test_collector_rejects_file_chunk_before_metadata() {
        let mut collector = TokenizerBundleCollector::default();

        let err = collector
            .push_chunk(TokenizerChunk::FileChunk(TokenizerFileChunk {
                data: vec![1, 2, 3],
                chunk_index: 0,
                is_last_chunk: true,
            }))
            .unwrap_err();

        assert!(err.contains("first chunk must be metadata"));
    }

    #[test]
    fn test_collector_rejects_out_of_order_chunk_index() {
        let mut collector = TokenizerBundleCollector::default();
        collector
            .push_chunk(TokenizerChunk::Metadata(metadata()))
            .unwrap();

        let err = collector
            .push_chunk(TokenizerChunk::FileChunk(TokenizerFileChunk {
                data: vec![1, 2, 3],
                chunk_index: 1,
                is_last_chunk: true,
            }))
            .unwrap_err();

        assert!(err.contains("expected chunk index 0, got 1"));
    }

    #[test]
    fn test_collector_rejects_chunk_after_last_chunk() {
        let mut collector = TokenizerBundleCollector::default();
        collector
            .push_chunk(TokenizerChunk::Metadata(metadata()))
            .unwrap();
        collector
            .push_chunk(TokenizerChunk::FileChunk(TokenizerFileChunk {
                data: vec![1, 2, 3],
                chunk_index: 0,
                is_last_chunk: true,
            }))
            .unwrap();

        let err = collector
            .push_chunk(TokenizerChunk::FileChunk(TokenizerFileChunk {
                data: vec![4, 5, 6],
                chunk_index: 1,
                is_last_chunk: true,
            }))
            .unwrap_err();

        assert!(err.contains("received chunk after final chunk"));
    }

    #[test]
    fn test_collector_finish_requires_final_chunk() {
        let mut collector = TokenizerBundleCollector::default();
        collector
            .push_chunk(TokenizerChunk::Metadata(metadata()))
            .unwrap();
        collector
            .push_chunk(TokenizerChunk::FileChunk(TokenizerFileChunk {
                data: vec![1, 2, 3],
                chunk_index: 0,
                is_last_chunk: false,
            }))
            .unwrap();

        let err = collector.finish().unwrap_err();
        assert!(err.contains("stream ended without receiving final chunk"));
    }

    #[test]
    fn test_collector_rejects_unsupported_bundle_format() {
        let mut collector = TokenizerBundleCollector::default();

        let mut meta = metadata();
        meta.bundle_format = "tar.gz".to_string();

        let err = collector
            .push_chunk(TokenizerChunk::Metadata(meta))
            .unwrap_err();

        assert!(err.contains("Unsupported tokenizer bundle format 'tar.gz'"));
    }

    #[test]
    fn test_collector_rejects_bundle_exceeding_max_size() {
        let mut collector = TokenizerBundleCollector::default();
        collector
            .push_chunk(TokenizerChunk::Metadata(metadata()))
            .unwrap();

        // Send a single chunk that exceeds the limit
        let oversized = vec![0u8; MAX_TOKENIZER_BUNDLE_SIZE + 1];
        let err = collector
            .push_chunk(TokenizerChunk::FileChunk(TokenizerFileChunk {
                data: oversized,
                chunk_index: 0,
                is_last_chunk: true,
            }))
            .unwrap_err();

        assert!(err.contains("exceeds maximum size limit"));
    }

    #[test]
    fn test_collect_tokenizer_bundle_propagates_stream_errors() {
        let mut s = stream::iter(vec![
            Ok(TokenizerChunk::Metadata(metadata())),
            Err("socket closed"),
        ]);

        let err = block_on(collect_tokenizer_bundle(&mut s, Ok)).unwrap_err();
        assert!(err.contains("Tokenizer stream error: socket closed"));
    }
}
