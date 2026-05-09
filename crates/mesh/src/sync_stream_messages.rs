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
    crdt_kv::{Operation, OperationLog, ReplicaId},
    kv::{MeshKV, RoundBatch},
    service::gossip::{
        stream_message::Payload as StreamPayload, CrdtBatch, CrdtBatchEntry, StreamAck,
        StreamBatch, StreamMessage, StreamMessageType,
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

    let entries = operation_log_to_entries(&operation_log);

    Some(StreamMessage {
        message_type: StreamMessageType::CrdtBatch as i32,
        payload: Some(StreamPayload::CrdtBatch(CrdtBatch { entries })),
        sequence: next_sequence(sequence),
        peer_id: self_name.to_string(),
    })
}

pub(crate) fn apply_crdt_batch(mesh_kv: &MeshKV, batch: &CrdtBatch) {
    if batch.entries.is_empty() {
        return;
    }

    let operations = batch
        .entries
        .iter()
        .filter_map(crdt_entry_to_operation)
        .collect::<Vec<_>>();
    mesh_kv.apply_crdt_operation_log(&OperationLog::from_operations(operations));
}

pub(crate) fn crdt_batch_encoded_len(batch: &CrdtBatch) -> usize {
    prost::Message::encoded_len(batch)
}

fn operation_log_to_entries(log: &OperationLog) -> Vec<CrdtBatchEntry> {
    log.operations()
        .iter()
        .map(|operation| match operation {
            Operation::Insert {
                key,
                value,
                timestamp,
                replica_id,
            } => CrdtBatchEntry {
                key: key.clone(),
                value: value.clone(),
                tombstone: false,
                timestamp: *timestamp,
                replica_id: replica_id.to_string(),
            },
            Operation::Remove {
                key,
                timestamp,
                replica_id,
            } => CrdtBatchEntry {
                key: key.clone(),
                value: Vec::new(),
                tombstone: true,
                timestamp: *timestamp,
                replica_id: replica_id.to_string(),
            },
        })
        .collect()
}

fn crdt_entry_to_operation(entry: &CrdtBatchEntry) -> Option<Operation> {
    let replica_id = match ReplicaId::from_string(&entry.replica_id) {
        Ok(replica_id) => replica_id,
        Err(err) => {
            log::warn!(
                %err,
                replica_id = %entry.replica_id,
                key = %entry.key,
                "dropping CRDT batch entry with invalid replica_id"
            );
            return None;
        }
    };

    Some(if entry.tombstone {
        Operation::remove(entry.key.clone(), entry.timestamp, replica_id)
    } else {
        Operation::insert(
            entry.key.clone(),
            entry.value.clone(),
            entry.timestamp,
            replica_id,
        )
    })
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU64;

    use super::*;
    use crate::kv::MeshKV;

    #[test]
    fn crdt_batch_uses_explicit_entries_for_put_and_delete() {
        let source = MeshKV::new("source".to_string());
        let target = MeshKV::new("target".to_string());
        let source_config = source.configs();
        let target_config = target.configs();

        source_config.put("config:feature", b"enabled".to_vec());
        let sequence = AtomicU64::new(0);
        let msg = crdt_batch_message(&source, "source", &sequence).expect("batch");
        let Some(StreamPayload::CrdtBatch(batch)) = msg.payload else {
            panic!("expected CRDT batch");
        };

        assert_eq!(batch.entries.len(), 1);
        assert_eq!(batch.entries[0].key, "config:feature");
        assert_eq!(batch.entries[0].value, b"enabled");
        assert!(!batch.entries[0].tombstone);
        assert!(!batch.entries[0].replica_id.is_empty());

        apply_crdt_batch(&target, &batch);
        assert_eq!(
            target_config.get("config:feature"),
            Some(b"enabled".to_vec())
        );

        source_config.delete("config:feature");
        let delete_msg = crdt_batch_message(&source, "source", &sequence).expect("delete batch");
        let Some(StreamPayload::CrdtBatch(delete_batch)) = delete_msg.payload else {
            panic!("expected CRDT batch");
        };
        assert!(delete_batch
            .entries
            .iter()
            .any(|entry| entry.key == "config:feature" && entry.tombstone));

        apply_crdt_batch(&target, &delete_batch);
        assert_eq!(target_config.get("config:feature"), None);
    }

    #[test]
    fn apply_crdt_batch_skips_entries_with_invalid_replica_ids() {
        let target = MeshKV::new("target".to_string());
        let batch = CrdtBatch {
            entries: vec![CrdtBatchEntry {
                key: "config:feature".to_string(),
                value: b"enabled".to_vec(),
                tombstone: false,
                timestamp: 1,
                replica_id: "not-a-uuid".to_string(),
            }],
        };

        apply_crdt_batch(&target, &batch);
        assert_eq!(target.configs().get("config:feature"), None);
    }
}
