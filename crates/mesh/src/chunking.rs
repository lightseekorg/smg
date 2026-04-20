//! Sender-side stream chunking helpers and receiver-side dispatch.
//!
//! Sender: [`chunk_value`] splits an oversized value into a sequence
//! of `StreamEntry`s; [`build_stream_batches`] packs them into
//! `StreamBatch` messages respecting a per-batch cap.
//!
//! Receiver: [`dispatch_stream_batch`] routes inbound entries —
//! single-chunk entries fire subscribers directly, multi-chunk entries
//! go through the [`ChunkAssembler`](crate::chunk_assembler::ChunkAssembler)
//! and fire subscribers only on full reassembly.

use bytes::Bytes;

use crate::{
    flow_control::MAX_MESSAGE_SIZE,
    kv::MeshKV,
    service::gossip::{StreamBatch, StreamEntry},
};

/// Per-round cap on chunks emitted across all stream entries. Prevents
/// a burst of large values from monopolising the gossip channel.
pub const DEFAULT_MAX_CHUNKS_PER_ROUND: usize = 5;

/// Headroom reserved below `MAX_MESSAGE_SIZE` for protobuf envelope
/// overhead (StreamMessage wrapper, StreamEntry metadata, per-field
/// tags and length prefixes). Without this margin a chunk sized
/// exactly at `MAX_MESSAGE_SIZE` pushes the serialised message past
/// the receiver's `max_decoding_message_size` and tonic rejects it.
pub const STREAM_CHUNK_OVERHEAD_MARGIN: usize = 64 * 1024;

/// Maximum payload bytes per `StreamEntry`. Callers pass this to
/// [`chunk_value`] so every emitted entry leaves room for the
/// protobuf envelope within the gRPC message budget.
pub const MAX_STREAM_CHUNK_BYTES: usize = MAX_MESSAGE_SIZE - STREAM_CHUNK_OVERHEAD_MARGIN;

/// Split a stream value into one or more `StreamEntry`s.
///
/// Values with `value.len() <= max_chunk_bytes` produce a single entry
/// with `total_chunks = 1`. Larger values split into
/// `ceil(len / max_chunk_bytes)` chunks of at most `max_chunk_bytes`
/// bytes each. Chunks are emitted in index order 0..N. `generation` is
/// a per-publish monotonic tag chosen by the caller.
pub fn chunk_value(
    key: String,
    generation: u64,
    value: Bytes,
    max_chunk_bytes: usize,
) -> Vec<StreamEntry> {
    assert!(max_chunk_bytes > 0, "max_chunk_bytes must be non-zero");

    if value.len() <= max_chunk_bytes {
        return vec![StreamEntry {
            key,
            generation,
            chunk_index: 0,
            total_chunks: 1,
            data: value.to_vec(),
        }];
    }

    debug_assert!(
        !key.starts_with("tree:page:"),
        "tree:page:* values must be bounded by max_tree_repair_page_bytes \
         and never enter the multi-chunk split path"
    );

    let total_chunks = value.len().div_ceil(max_chunk_bytes);
    let mut entries = Vec::with_capacity(total_chunks);
    for index in 0..total_chunks {
        let start = index * max_chunk_bytes;
        let end = (start + max_chunk_bytes).min(value.len());
        entries.push(StreamEntry {
            key: key.clone(),
            generation,
            chunk_index: index as u32,
            total_chunks: total_chunks as u32,
            data: value.slice(start..end).to_vec(),
        });
    }
    entries
}

/// Receiver-side dispatch for `StreamBatch` entries. Single-chunk
/// entries (`total_chunks == 1`) fire subscribers directly — no state
/// in the chunk assembler. Multi-chunk entries route through the
/// assembler and fire subscribers only on full reassembly.
///
/// Takes ownership of the entries so chunk payloads move into `Bytes`
/// without cloning. The chunk assembler scopes in-flight state by
/// `peer_id` so concurrent chunked values from different senders under
/// the same key don't collide.
pub fn dispatch_stream_batch(
    mesh_kv: &MeshKV,
    peer_id: &str,
    entries: impl IntoIterator<Item = StreamEntry>,
) {
    for entry in entries {
        if entry.total_chunks == 1 {
            mesh_kv.notify_subscribers(&entry.key, Some(vec![Bytes::from(entry.data)]));
        } else {
            let key = entry.key.clone();
            if let Some(fragments) = mesh_kv.chunk_assembler().receive_chunk(
                peer_id,
                &key,
                entry.generation,
                entry.chunk_index,
                entry.total_chunks,
                entry.data,
            ) {
                mesh_kv.notify_subscribers(&key, Some(fragments));
            }
        }
    }
}

/// Pack `StreamEntry`s into one or more `StreamBatch`es of at most
/// `max_chunks_per_batch` entries each. Every entry is emitted —
/// truncating mid-sequence would strand trailing chunks of a value
/// undeliverable (receiver never completes reassembly; buffer was
/// already drained so the sender has nothing to retry with).
pub fn build_stream_batches(
    entries: Vec<StreamEntry>,
    max_chunks_per_batch: usize,
) -> Vec<StreamBatch> {
    debug_assert!(
        max_chunks_per_batch > 0,
        "max_chunks_per_batch must be non-zero; callers should pass \
         DEFAULT_MAX_CHUNKS_PER_ROUND or another positive cap"
    );
    if max_chunks_per_batch == 0 || entries.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut iter = entries.into_iter().peekable();
    while iter.peek().is_some() {
        let batch: Vec<StreamEntry> = iter.by_ref().take(max_chunks_per_batch).collect();
        if !batch.is_empty() {
            out.push(StreamBatch { entries: batch });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_chunk_fast_path() {
        let value = Bytes::from(vec![0u8; 100]);
        let entries = chunk_value("key".into(), 42, value.clone(), 1024);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].key, "key");
        assert_eq!(entries[0].generation, 42);
        assert_eq!(entries[0].chunk_index, 0);
        assert_eq!(entries[0].total_chunks, 1);
        assert_eq!(entries[0].data, value.to_vec());
    }

    #[test]
    fn test_empty_value_is_single_chunk() {
        let entries = chunk_value("k".into(), 1, Bytes::new(), 1024);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].total_chunks, 1);
        assert!(entries[0].data.is_empty());
    }

    #[test]
    fn test_multi_chunk_split_even_boundary() {
        // 3 × 10 bytes = 30 bytes total, max_chunk_bytes = 10 → 3 chunks
        let value = Bytes::from((0u8..30).collect::<Vec<_>>());
        let entries = chunk_value("k".into(), 7, value.clone(), 10);
        assert_eq!(entries.len(), 3);
        for (i, e) in entries.iter().enumerate() {
            assert_eq!(e.chunk_index, i as u32);
            assert_eq!(e.total_chunks, 3);
            assert_eq!(e.generation, 7);
        }
        let reassembled: Vec<u8> = entries.iter().flat_map(|e| e.data.clone()).collect();
        assert_eq!(reassembled, value.to_vec());
    }

    #[test]
    fn test_multi_chunk_split_uneven_boundary() {
        // 25 bytes, max_chunk_bytes = 10 → 3 chunks (10+10+5)
        let value = Bytes::from((0u8..25).collect::<Vec<_>>());
        let entries = chunk_value("k".into(), 1, value.clone(), 10);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].data.len(), 10);
        assert_eq!(entries[1].data.len(), 10);
        assert_eq!(entries[2].data.len(), 5);
        let reassembled: Vec<u8> = entries.iter().flat_map(|e| e.data.clone()).collect();
        assert_eq!(reassembled, value.to_vec());
    }

    #[test]
    fn test_chunk_at_exact_boundary() {
        // Value exactly equals max_chunk_bytes → single chunk (fast path).
        let value = Bytes::from(vec![0u8; 10]);
        let entries = chunk_value("k".into(), 1, value, 10);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].total_chunks, 1);
    }

    #[test]
    fn test_one_byte_over_boundary_splits() {
        let value = Bytes::from(vec![0u8; 11]);
        let entries = chunk_value("k".into(), 1, value, 10);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].data.len(), 10);
        assert_eq!(entries[1].data.len(), 1);
    }

    #[test]
    #[should_panic(expected = "max_chunk_bytes must be non-zero")]
    fn test_zero_max_chunk_bytes_panics() {
        let _ = chunk_value("k".into(), 1, Bytes::from_static(b"x"), 0);
    }

    #[test]
    fn test_tree_page_fast_path() {
        // tree:page:* stays under max_chunk_bytes by contract (2 MB vs 10 MB),
        // so it takes the single-chunk path and the debug_assert does not fire.
        let value = Bytes::from(vec![0u8; 1_000_000]);
        let entries = chunk_value("tree:page:model1:0".into(), 1, value, 10 * 1024 * 1024);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].total_chunks, 1);
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "tree:page:*"))]
    fn test_tree_page_split_panics_in_debug() {
        // If a tree page ever grows past max_chunk_bytes, debug build
        // catches it. Release build would just emit the split (valid
        // semantically, but violates the tree-sync contract).
        let value = Bytes::from(vec![0u8; 20]);
        let _ = chunk_value("tree:page:model1:0".into(), 1, value, 10);
    }

    #[test]
    fn test_build_stream_batches_empty() {
        assert!(build_stream_batches(vec![], 5).is_empty());
    }

    #[test]
    fn test_build_stream_batches_under_cap() {
        let entries = vec![
            StreamEntry {
                key: "a".into(),
                generation: 1,
                chunk_index: 0,
                total_chunks: 1,
                data: vec![1],
            },
            StreamEntry {
                key: "b".into(),
                generation: 1,
                chunk_index: 0,
                total_chunks: 1,
                data: vec![2],
            },
        ];
        let batches = build_stream_batches(entries, 5);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].entries.len(), 2);
    }

    #[test]
    fn test_build_stream_batches_packs_over_cap() {
        let entries: Vec<StreamEntry> = (0..10)
            .map(|i| StreamEntry {
                key: format!("k{i}"),
                generation: 1,
                chunk_index: 0,
                total_chunks: 1,
                data: vec![i as u8],
            })
            .collect();
        // 10 entries, cap=3 → 3 + 3 + 3 + 1 = 4 batches. No entry dropped.
        let batches = build_stream_batches(entries, 3);
        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0].entries.len(), 3);
        assert_eq!(batches[1].entries.len(), 3);
        assert_eq!(batches[2].entries.len(), 3);
        assert_eq!(batches[3].entries.len(), 1);
        let total: usize = batches.iter().map(|b| b.entries.len()).sum();
        assert_eq!(total, 10, "every entry must be emitted (no truncation)");
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "max_chunks_per_batch"))]
    fn test_build_stream_batches_zero_cap() {
        // Debug build: assert fires (contract violation).
        // Release build: runtime zero-check still returns empty as a safety net.
        let entries = vec![StreamEntry {
            key: "k".into(),
            generation: 1,
            chunk_index: 0,
            total_chunks: 1,
            data: vec![1],
        }];
        let out = build_stream_batches(entries, 0);
        assert!(out.is_empty());
    }
}
