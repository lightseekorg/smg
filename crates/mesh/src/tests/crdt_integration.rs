//! End-to-end tests for the CRDT-over-gossip path (d-3a).
//!
//! These exercise the full producer→consumer round trip without gRPC: the
//! sender's op-log snapshot is encoded with
//! [`build_crdt_batch`](crate::transport::crdt_batch::build_crdt_batch) and fed
//! into the receiver's
//! [`dispatch_crdt_batch`](crate::transport::crdt_batch::dispatch_crdt_batch),
//! which decodes, merges into the receiver's store, and fires subscribers. In
//! production the batch serialises through the `Gossip::sync_stream` RPC; here
//! we bypass prost and route the ops directly, matching the chunking
//! integration tests.

use bytes::Bytes;
use tokio::sync::mpsc::error::TryRecvError;

use crate::{
    crdt_kv::{decode as decode_epoch_count, encode as encode_epoch_count, EpochCount},
    kv::MeshKV,
    transport::crdt_batch::{build_crdt_batch, dispatch_crdt_batch},
    MergeStrategy,
};

/// Simulate one gossip round of CRDT delivery: snapshot the sender's op-log,
/// encode it, and dispatch it into the receiver.
fn deliver_crdt(sender: &MeshKV, receiver: &MeshKV) {
    let ops = sender.collect_round_batch().crdt_ops;
    if let Some(batch) = build_crdt_batch(&ops) {
        dispatch_crdt_batch(receiver, batch);
    }
}

fn flatten(fragments: &[Bytes]) -> Vec<u8> {
    fragments.iter().flat_map(|b| b.iter().copied()).collect()
}

#[test]
fn remote_crdt_batch_converges_store() {
    let sender = MeshKV::new("sender".to_string());
    let s_ns = sender.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    s_ns.put("worker:a", b"v1".to_vec());

    let receiver = MeshKV::new("receiver".to_string());
    let r_ns = receiver.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    assert_eq!(r_ns.get("worker:a"), None);

    deliver_crdt(&sender, &receiver);
    assert_eq!(
        r_ns.get("worker:a"),
        Some(b"v1".to_vec()),
        "CRDT op-log delivered over the wire converges the receiver's store"
    );
}

#[tokio::test]
async fn remote_merge_fires_subscriber_with_value() {
    let sender = MeshKV::new("sender".to_string());
    let s_ns = sender.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    s_ns.put("worker:a", b"v1".to_vec());

    let receiver = MeshKV::new("receiver".to_string());
    let r_ns = receiver.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    let mut sub = r_ns.subscribe("");

    deliver_crdt(&sender, &receiver);

    let (key, payload) = sub
        .receiver
        .recv()
        .await
        .expect("remote merge fires a subscriber event");
    assert_eq!(key, "worker:a");
    assert_eq!(flatten(&payload.expect("insert delivers a value")), b"v1");
}

#[tokio::test]
async fn redelivering_same_batch_fires_no_new_event() {
    let sender = MeshKV::new("sender".to_string());
    let s_ns = sender.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    s_ns.put("worker:a", b"v1".to_vec());

    let receiver = MeshKV::new("receiver".to_string());
    let r_ns = receiver.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    let mut sub = r_ns.subscribe("");

    deliver_crdt(&sender, &receiver);
    let _ = sub.receiver.recv().await.expect("first delivery fires");

    // Merge is idempotent by op-id: re-delivering the same batch changes no
    // live value, so no subscriber event fires.
    deliver_crdt(&sender, &receiver);
    assert!(
        matches!(sub.receiver.try_recv(), Err(TryRecvError::Empty)),
        "idempotent re-delivery must not fire a subscriber event"
    );
}

#[tokio::test]
async fn rl_remote_merge_subscriber_sees_canonical_shard() {
    // The sender writes the raw 16-byte (epoch, count) payload; the engine
    // normalises it into a shard. The remote-merge subscriber must see the
    // canonical shard shape (matching `get`), not the raw input — migration
    // step 7's value-shape alignment.
    let sender = MeshKV::new("sender".to_string());
    let s_ns = sender.configure_crdt_prefix("rl:", MergeStrategy::EpochMaxWins);
    s_ns.put("rl:global:node-a", encode_epoch_count(7, 42).to_vec());

    let receiver = MeshKV::new("receiver".to_string());
    let r_ns = receiver.configure_crdt_prefix("rl:", MergeStrategy::EpochMaxWins);
    let mut sub = r_ns.subscribe("");

    deliver_crdt(&sender, &receiver);

    let (key, payload) = sub
        .receiver
        .recv()
        .await
        .expect("rl remote merge fires a subscriber event");
    assert_eq!(key, "rl:global:node-a");
    let bytes = flatten(&payload.expect("insert delivers a value"));
    assert_ne!(
        bytes.len(),
        encode_epoch_count(7, 42).len(),
        "subscriber sees the encoded shard, not the raw 16-byte payload"
    );
    assert_eq!(
        decode_epoch_count(&bytes),
        Some(EpochCount {
            epoch: 7,
            count: 42
        })
    );
    assert_eq!(
        r_ns.get("rl:global:node-a"),
        Some(bytes),
        "remote-merge notification shape matches get()"
    );
}

#[tokio::test]
async fn remote_tombstone_after_insert_notifies_none() {
    let sender = MeshKV::new("sender".to_string());
    let s_ns = sender.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    s_ns.put("worker:a", b"v1".to_vec());

    let receiver = MeshKV::new("receiver".to_string());
    let r_ns = receiver.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    let mut sub = r_ns.subscribe("");

    deliver_crdt(&sender, &receiver);
    let (_, payload) = sub.receiver.recv().await.expect("insert fires");
    assert!(payload.is_some());

    // Delete on the sender; the tombstone propagates and the receiver fires a
    // `None` event as worker:a transitions live -> tombstoned.
    s_ns.delete("worker:a");
    deliver_crdt(&sender, &receiver);

    let (key, payload) = sub.receiver.recv().await.expect("tombstone fires");
    assert_eq!(key, "worker:a");
    assert!(payload.is_none(), "tombstone notifies None");
    assert_eq!(r_ns.get("worker:a"), None);
}
