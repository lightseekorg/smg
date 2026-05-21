//! Shared SyncStream payload construction helpers.
//!
//! Both the outbound [`gossip_controller`](crate::gossip_controller)
//! and the inbound [`gossip_service`](crate::gossip_service) need
//! to build per-peer stream batches from a drained
//! [`RoundBatch`](crate::kv::RoundBatch) and wrap them in
//! `StreamMessage` envelopes. Putting that logic in one place keeps
//! the chunking + batching + envelope shape consistent across both
//! directions of the stream.

use crate::{
    kv::RoundBatch,
    service::gossip::{
        stream_message::Payload as StreamPayload, StreamBatch, StreamMessage, StreamMessageType,
    },
    transport::{
        chunking::{build_stream_batches, chunk_value, next_generation},
        limits::{DEFAULT_MAX_CHUNKS_PER_BATCH, MAX_STREAM_CHUNK_BYTES},
    },
};

/// Build the `StreamBatch`es that should be sent to `peer_id` for
/// the current round. Returns an empty `Vec` when neither the
/// broadcast drain nor any targeted entry is addressed to this peer.
///
/// `drain_entries` are broadcast: every peer's emitter includes
/// them. `targeted_entries` are only included when their target
/// matches `peer_id` — pass `""` to skip all targeted entries
/// (e.g. when the inbound peer identity is not yet learned).
/// Oversized values are split via [`chunk_value`]; the returned
/// batches respect the `DEFAULT_MAX_CHUNKS_PER_BATCH` /
/// `MAX_STREAM_CHUNK_BYTES` caps.
pub fn build_peer_stream_batches(round_batch: &RoundBatch, peer_id: &str) -> Vec<StreamBatch> {
    let mut entries = Vec::new();
    for (key, value) in &round_batch.drain_entries {
        entries.extend(chunk_value(
            key.clone(),
            next_generation(),
            value.clone(),
            MAX_STREAM_CHUNK_BYTES,
        ));
    }
    for (target, key, value) in &round_batch.targeted_entries {
        if target == peer_id {
            entries.extend(chunk_value(
                key.clone(),
                next_generation(),
                value.clone(),
                MAX_STREAM_CHUNK_BYTES,
            ));
        }
    }
    if entries.is_empty() {
        return Vec::new();
    }
    build_stream_batches(
        entries,
        DEFAULT_MAX_CHUNKS_PER_BATCH,
        MAX_STREAM_CHUNK_BYTES,
    )
}

/// Wrap a `StreamBatch` in a `StreamMessage` envelope.
pub fn wrap_stream_batch(batch: StreamBatch, sequence: u64, self_name: String) -> StreamMessage {
    StreamMessage {
        message_type: StreamMessageType::StreamBatch as i32,
        payload: Some(StreamPayload::StreamBatch(batch)),
        sequence,
        peer_id: self_name,
    }
}

/// Build a heartbeat `StreamMessage` (no payload, message_type = Heartbeat).
pub fn build_heartbeat(sequence: u64, self_name: String) -> StreamMessage {
    StreamMessage {
        message_type: StreamMessageType::Heartbeat as i32,
        payload: None,
        sequence,
        peer_id: self_name,
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;

    fn round_batch_with(
        drain: Vec<(&str, &[u8])>,
        targeted: Vec<(&str, &str, &[u8])>,
    ) -> RoundBatch {
        RoundBatch {
            drain_entries: drain
                .into_iter()
                .map(|(k, v)| (k.to_string(), Bytes::copy_from_slice(v)))
                .collect(),
            targeted_entries: targeted
                .into_iter()
                .map(|(t, k, v)| (t.to_string(), k.to_string(), Bytes::copy_from_slice(v)))
                .collect(),
        }
    }

    #[test]
    fn empty_round_batch_emits_nothing() {
        let rb = round_batch_with(vec![], vec![]);
        assert!(build_peer_stream_batches(&rb, "peer1").is_empty());
    }

    #[test]
    fn drain_only_is_emitted_to_every_peer() {
        let rb = round_batch_with(vec![("td:abc", b"hello")], vec![]);
        let a = build_peer_stream_batches(&rb, "peer1");
        let b = build_peer_stream_batches(&rb, "peer2");
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        assert_eq!(a[0].entries.len(), 1);
        assert_eq!(a[0].entries[0].key, "td:abc");
    }

    #[test]
    fn targeted_entries_filter_by_peer() {
        let rb = round_batch_with(
            vec![],
            vec![
                ("peer1", "tree:req:a", b"req-a".as_slice()),
                ("peer2", "tree:req:b", b"req-b".as_slice()),
            ],
        );
        let a = build_peer_stream_batches(&rb, "peer1");
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].entries.len(), 1);
        assert_eq!(a[0].entries[0].key, "tree:req:a");

        let none = build_peer_stream_batches(&rb, "");
        assert!(none.is_empty());
    }

    #[test]
    fn drain_emitted_even_when_peer_unknown() {
        let rb = round_batch_with(
            vec![("td:abc", b"hello".as_slice())],
            vec![("peer1", "tree:req:a", b"req-a".as_slice())],
        );
        // Empty peer_id -> targeted entries excluded, drain still emitted.
        let batches = build_peer_stream_batches(&rb, "");
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].entries.len(), 1);
        assert_eq!(batches[0].entries[0].key, "td:abc");
    }

    #[test]
    fn wrap_stream_batch_envelope_shape() {
        let batch = StreamBatch::default();
        let msg = wrap_stream_batch(batch, 42, "node-1".to_string());
        assert_eq!(msg.message_type, StreamMessageType::StreamBatch as i32);
        assert_eq!(msg.sequence, 42);
        assert_eq!(msg.peer_id, "node-1");
        assert!(matches!(msg.payload, Some(StreamPayload::StreamBatch(_))));
    }

    #[test]
    fn build_heartbeat_envelope_shape() {
        let msg = build_heartbeat(7, "node-2".to_string());
        assert_eq!(msg.message_type, StreamMessageType::Heartbeat as i32);
        assert_eq!(msg.sequence, 7);
        assert_eq!(msg.peer_id, "node-2");
        assert!(msg.payload.is_none());
    }
}
