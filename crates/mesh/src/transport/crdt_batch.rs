//! CRDT batch wire-message helpers for the `Gossip::sync_stream` RPC.
//!
//! Mirrors [`sync_stream`](crate::transport::sync_stream) for the CRDT data
//! path:
//! - Outbound: [`build_crdt_batches`] converts an [`Operation`] slice (the
//!   local op-log snapshot carried in [`RoundBatch`](crate::kv::RoundBatch))
//!   into one or more wire [`CrdtBatch`]es, each bounded below the gRPC
//!   message cap; [`wrap_crdt_batch`] builds the `StreamMessage` envelope.
//! - Inbound: [`dispatch_crdt_batch`] decodes a received `CrdtBatch` back into
//!   `Operation`s and merges them into the local CRDT store via `MeshKV`.
//!
//! The op-log is broadcast in full each round; merge is idempotent by op-id,
//! so re-sending already-seen ops is a no-op. Per-peer watermark filtering (to
//! send only ops the peer has not acked) is a follow-up.

use crate::{
    crdt_kv::{Operation, ReplicaId},
    kv::MeshKV,
    service::gossip::{
        stream_message::Payload as StreamPayload, CrdtBatch, CrdtOp, StreamMessage,
        StreamMessageType,
    },
};

/// Convert an in-crate [`Operation`] into its wire form. A `Remove` carries an
/// empty `value` and `tombstone = true`; an `Insert` carries the value bytes
/// and `tombstone = false`. The `replica_id` UUID is rendered to text.
fn op_to_proto(op: &Operation) -> CrdtOp {
    match op {
        Operation::Insert {
            key,
            value,
            timestamp,
            replica_id,
        } => CrdtOp {
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
        } => CrdtOp {
            key: key.clone(),
            value: Vec::new(),
            tombstone: true,
            timestamp: *timestamp,
            replica_id: replica_id.to_string(),
        },
    }
}

/// Convert a wire [`CrdtOp`] back into an [`Operation`]. Returns `None` if the
/// `replica_id` field is not a valid UUID — a malformed/hostile peer's op is
/// dropped rather than poisoning the merge.
fn proto_to_op(op: CrdtOp) -> Option<Operation> {
    let replica_id = ReplicaId::from_string(&op.replica_id).ok()?;
    Some(if op.tombstone {
        Operation::remove(op.key, op.timestamp, replica_id)
    } else {
        Operation::insert(op.key, op.value, op.timestamp, replica_id)
    })
}

/// Conservative estimate of a `CrdtOp`'s encoded size: the variable-length
/// fields plus a constant covering proto field tags, length varints, the
/// timestamp/tombstone fields, and the repeated-field wrapper in `CrdtBatch`.
fn estimated_op_size(op: &CrdtOp) -> usize {
    op.key.len() + op.value.len() + op.replica_id.len() + 24
}

/// Split an op-log snapshot into one or more [`CrdtBatch`]es, each estimated to
/// stay under `max_bytes` so the wrapped `StreamMessage` does not exceed the
/// gRPC message cap. Returns an empty `Vec` for an empty snapshot. Without this
/// bound, a large op-log (broadcast in full each round until per-peer watermark
/// filtering lands) could produce a single frame above `MAX_MESSAGE_SIZE`,
/// which tonic rejects on encode/decode and which would tear down the
/// sync_stream. A single op larger than `max_bytes` is emitted alone (best
/// effort); `worker:`/`rl:`/`config:` values are far below the cap, so in
/// practice this only bounds the op count per frame.
pub fn build_crdt_batches(ops: &[Operation], max_bytes: usize) -> Vec<CrdtBatch> {
    let mut out = Vec::new();
    let mut current: Vec<CrdtOp> = Vec::new();
    let mut current_bytes = 0usize;
    for op in ops {
        let proto = op_to_proto(op);
        let size = estimated_op_size(&proto);
        if !current.is_empty() && current_bytes + size > max_bytes {
            out.push(CrdtBatch {
                ops: std::mem::take(&mut current),
            });
            current_bytes = 0;
        }
        current_bytes += size;
        current.push(proto);
    }
    if !current.is_empty() {
        out.push(CrdtBatch { ops: current });
    }
    out
}

/// Wrap a [`CrdtBatch`] in a `StreamMessage` envelope.
pub fn wrap_crdt_batch(batch: CrdtBatch, sequence: u64, self_name: &str) -> StreamMessage {
    StreamMessage {
        message_type: StreamMessageType::CrdtBatch as i32,
        payload: Some(StreamPayload::CrdtBatch(batch)),
        sequence,
        peer_id: self_name.to_owned(),
    }
}

/// Receiver-side dispatch for a `CrdtBatch`: decode each op and merge the batch
/// into the local CRDT store, firing subscribers for keys whose value changed
/// (via `MeshKV::merge_crdt_ops`). Ops with an unparsable `replica_id` are
/// skipped. Merge is idempotent by op-id, so a batch the node has already
/// absorbed is a no-op and fires no subscriber event.
pub fn dispatch_crdt_batch(mesh_kv: &MeshKV, batch: CrdtBatch) {
    let ops: Vec<Operation> = batch.ops.into_iter().filter_map(proto_to_op).collect();
    if ops.is_empty() {
        return;
    }
    mesh_kv.merge_crdt_ops(ops);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_round_trips_through_proto() {
        let replica = ReplicaId::new();
        let insert = Operation::insert("worker:a".to_string(), b"v".to_vec(), 5, replica);
        let remove = Operation::remove("worker:b".to_string(), 7, replica);

        let back_insert = proto_to_op(op_to_proto(&insert)).expect("valid insert round-trips");
        let back_remove = proto_to_op(op_to_proto(&remove)).expect("valid remove round-trips");
        assert_eq!(back_insert, insert);
        assert_eq!(back_remove, remove);
    }

    #[test]
    fn build_crdt_batches_empty_and_single() {
        assert!(build_crdt_batches(&[], 1024).is_empty());
        let replica = ReplicaId::new();
        let ops = vec![Operation::insert(
            "rl:c".to_string(),
            b"x".to_vec(),
            1,
            replica,
        )];
        let batches = build_crdt_batches(&ops, 1024);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].ops.len(), 1);
        assert_eq!(batches[0].ops[0].key, "rl:c");
        assert!(!batches[0].ops[0].tombstone);
    }

    #[test]
    fn build_crdt_batches_splits_over_budget() {
        let replica = ReplicaId::new();
        let ops: Vec<Operation> = (0..50)
            .map(|i| Operation::insert(format!("worker:{i}"), vec![0u8; 100], i, replica))
            .collect();
        // Each op is ~100 bytes of value + overhead; a 300-byte budget forces
        // several batches.
        let batches = build_crdt_batches(&ops, 300);
        assert!(
            batches.len() > 1,
            "large op-log must split into many batches"
        );
        // Every op is preserved exactly once across all batches.
        let total: usize = batches.iter().map(|b| b.ops.len()).sum();
        assert_eq!(total, ops.len());
        // No batch exceeds the budget (except a lone oversized op, which none
        // of these are).
        for batch in &batches {
            let bytes: usize = batch.ops.iter().map(estimated_op_size).sum();
            assert!(bytes <= 300 || batch.ops.len() == 1);
        }
    }

    #[test]
    fn proto_to_op_rejects_bad_replica_id() {
        let bad = CrdtOp {
            key: "worker:a".to_string(),
            value: Vec::new(),
            tombstone: false,
            timestamp: 1,
            replica_id: "not-a-uuid".to_string(),
        };
        assert!(proto_to_op(bad).is_none());
    }

    #[test]
    fn wrap_crdt_batch_envelope_shape() {
        let msg = wrap_crdt_batch(CrdtBatch::default(), 11, "node-1");
        assert_eq!(msg.message_type, StreamMessageType::CrdtBatch as i32);
        assert_eq!(msg.sequence, 11);
        assert_eq!(msg.peer_id, "node-1");
        assert!(matches!(msg.payload, Some(StreamPayload::CrdtBatch(_))));
    }
}
