use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use tracing as log;

use crate::{
    chunking::{
        build_stream_batches, chunk_value, next_generation, DEFAULT_MAX_CHUNKS_PER_BATCH,
        MAX_STREAM_CHUNK_BYTES,
    },
    crdt_kv::OperationLog,
    kv::{MeshKV, RoundBatch},
    service::gossip::{
        stream_message::Payload as StreamPayload, CrdtBatch, StreamAck, StreamBatch, StreamMessage,
        StreamMessageType,
    },
};

pub(crate) fn next_sequence(sequence: &AtomicU64) -> u64 {
    sequence.fetch_add(1, Ordering::Relaxed)
}

pub(crate) fn heartbeat(self_name: &str, sequence: &AtomicU64) -> StreamMessage {
    StreamMessage {
        message_type: StreamMessageType::Heartbeat as i32,
        payload: None,
        sequence: next_sequence(sequence),
        peer_id: self_name.to_string(),
    }
}

pub(crate) fn ack(
    self_name: &str,
    sequence: &AtomicU64,
    acked_sequence: u64,
    success: bool,
    error_message: String,
) -> StreamMessage {
    StreamMessage {
        message_type: StreamMessageType::Ack as i32,
        payload: Some(StreamPayload::Ack(StreamAck {
            sequence: acked_sequence,
            success,
            error_message,
        })),
        sequence: next_sequence(sequence),
        peer_id: self_name.to_string(),
    }
}

pub(crate) fn crdt_batch_message(
    mesh_kv: &MeshKV,
    self_name: &str,
    sequence: &AtomicU64,
) -> Option<StreamMessage> {
    let operation_log = mesh_kv.crdt_operation_log();
    if operation_log.is_empty() {
        return None;
    }

    let operation_log = match operation_log.to_bytes() {
        Ok(bytes) => bytes,
        Err(err) => {
            log::warn!(%err, "failed to serialize CRDT operation log");
            return None;
        }
    };

    Some(StreamMessage {
        message_type: StreamMessageType::CrdtBatch as i32,
        payload: Some(StreamPayload::CrdtBatch(CrdtBatch { operation_log })),
        sequence: next_sequence(sequence),
        peer_id: self_name.to_string(),
    })
}

pub(crate) fn apply_crdt_batch(mesh_kv: &MeshKV, batch: &CrdtBatch) {
    match OperationLog::from_bytes(&batch.operation_log) {
        Ok(log) => mesh_kv.apply_crdt_operation_log(&log),
        Err(err) => log::warn!(%err, "failed to decode CRDT operation log"),
    }
}

pub(crate) fn build_stream_messages(
    batch: &RoundBatch,
    target_peer: Option<&str>,
    self_name: &str,
    sequence: &AtomicU64,
) -> Vec<StreamMessage> {
    let mut entries = Vec::new();

    for (key, value) in &batch.drain_entries {
        entries.extend(chunk_value(
            key.clone(),
            next_generation(),
            value.clone(),
            MAX_STREAM_CHUNK_BYTES,
        ));
    }

    if let Some(peer) = target_peer {
        for (target, key, value) in &batch.targeted_entries {
            if target == peer {
                entries.extend(chunk_value(
                    key.clone(),
                    next_generation(),
                    value.clone(),
                    MAX_STREAM_CHUNK_BYTES,
                ));
            }
        }
    }

    build_stream_batches(
        entries,
        DEFAULT_MAX_CHUNKS_PER_BATCH,
        MAX_STREAM_CHUNK_BYTES,
    )
    .into_iter()
    .map(|batch| StreamMessage {
        message_type: StreamMessageType::StreamBatch as i32,
        payload: Some(StreamPayload::StreamBatch(batch)),
        sequence: next_sequence(sequence),
        peer_id: self_name.to_string(),
    })
    .collect()
}

pub(crate) fn dispatch_stream_payload(mesh_kv: &Arc<MeshKV>, peer_id: &str, batch: StreamBatch) {
    crate::chunking::dispatch_stream_batch(mesh_kv, peer_id, batch.entries);
}
