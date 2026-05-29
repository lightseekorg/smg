//! CRDT batch wire-message helpers for the `Gossip::sync_stream` RPC.
//!
//! Mirrors [`sync_stream`](crate::transport::sync_stream) for the CRDT data
//! path:
//! - Outbound: [`build_crdt_batch`] converts an [`Operation`] slice (the local
//!   op-log snapshot carried in [`RoundBatch`](crate::kv::RoundBatch)) into a
//!   wire [`CrdtBatch`]; [`wrap_crdt_batch`] builds the `StreamMessage`
//!   envelope.
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

/// Build a [`CrdtBatch`] from an op-log snapshot. Returns `None` for an empty
/// snapshot so the caller can skip sending an empty message.
pub fn build_crdt_batch(ops: &[Operation]) -> Option<CrdtBatch> {
    if ops.is_empty() {
        return None;
    }
    Some(CrdtBatch {
        ops: ops.iter().map(op_to_proto).collect(),
    })
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
/// into the local CRDT store. Ops with an unparsable `replica_id` are skipped.
/// Merge is idempotent by op-id, so a batch the node has already absorbed is a
/// no-op. (Subscriber notification on remote merge is a follow-up — d-3a-2.)
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
    fn build_crdt_batch_skips_empty() {
        assert!(build_crdt_batch(&[]).is_none());
        let replica = ReplicaId::new();
        let ops = vec![Operation::insert(
            "rl:c".to_string(),
            b"x".to_vec(),
            1,
            replica,
        )];
        let batch = build_crdt_batch(&ops).expect("non-empty op-log yields a batch");
        assert_eq!(batch.ops.len(), 1);
        assert_eq!(batch.ops[0].key, "rl:c");
        assert!(!batch.ops[0].tombstone);
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
