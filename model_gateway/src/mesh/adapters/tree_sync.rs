//! `td:` / `tree:*` stream adapter: gateway ↔ mesh bridge for the
//! distributed prefix tree.
//!
//! First slice (this PR): tenant-delta fast path only.
//! - Outbound: `on_local_insert` buffers per-model `TreeDelta`
//!   entries. The mesh drain callback, called once per gossip round,
//!   batches each model's buffer into a single `td:{model_id}`
//!   stream entry (bincode-serialised `Vec<TreeDelta>`).
//! - Inbound: a spawned task subscribes to `td:` and decodes
//!   incoming batches. Apply-to-local-tree is deferred to the next
//!   slice (hash index + resolution); for now, remote deltas are
//!   logged for observability.
//!
//! Repair sessions (unknown-hash recovery, cold-start export/apply)
//! are tracked by later PRs in the outer sequence. The adapter is
//! intentionally scoped to the drain-and-subscribe shape here so the
//! wire format and the outbound path can soak without pulling in the
//! full session state machine.

use std::sync::{Arc, OnceLock};

use bytes::Bytes;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use smg_mesh::{DrainHandle, StreamNamespace};
use tracing::{debug, trace, warn};

const PREFIX: &str = "td:";

/// Which local tree a delta originated from. String and token trees
/// have disjoint hash spaces, so the adapter keeps them separate
/// across every data path (hash index, repair sessions, apply logic).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TreeKind {
    String,
    Token,
}

/// One prefix-tree change observed on a producing node.
///
/// `node_hash` is the blake3-derived 8-byte identifier for the tree
/// node scoped by `(model_id, tree_kind, path)`. The scope is carried
/// implicitly: `model_id` is encoded in the stream key (`td:{model_id}`)
/// and `tree_kind` sits inside this struct. The receiver resolves the
/// hash through its own `(model_id, tree_kind)` index; unknown hashes
/// trigger a repair request in a later slice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeDelta {
    pub tree_kind: TreeKind,
    pub node_hash: u64,
    /// Worker URL that cached the prefix.
    pub worker_url: String,
    /// Cache-event epoch. The merge rule is max-epoch-wins on the
    /// same node; this field is irrelevant to the stream transport
    /// but kept on the payload so the receiver can order within a
    /// batch.
    pub epoch: u64,
}

/// Bridge between the `td:` broadcast stream namespace and the
/// gateway's per-model tenant buffers.
pub struct TreeSyncAdapter {
    tenant_deltas: Arc<StreamNamespace>,
    pending_deltas: DashMap<String, Vec<TreeDelta>>,
    node_name: String,
    /// Keeps the drain registration alive for the adapter's lifetime.
    /// Dropping the handle unregisters from the mesh drain registry;
    /// `OnceLock` guards against a second `start` call.
    drain_handle: OnceLock<DrainHandle>,
}

impl std::fmt::Debug for TreeSyncAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeSyncAdapter")
            .field("prefix", &self.tenant_deltas.prefix())
            .field("node_name", &self.node_name)
            .field("pending_models", &self.pending_deltas.len())
            .finish()
    }
}

impl TreeSyncAdapter {
    /// Build an adapter wrapping a `td:`-scoped broadcast namespace
    /// and the local node name. Panics if the namespace prefix is
    /// wrong so a mis-wired caller fails loudly at startup instead
    /// of fanning deltas into the wrong stream.
    pub fn new(tenant_deltas: Arc<StreamNamespace>, node_name: String) -> Arc<Self> {
        assert_eq!(
            tenant_deltas.prefix(),
            PREFIX,
            "TreeSyncAdapter requires a tenant-delta namespace scoped to `{PREFIX}`",
        );
        assert!(
            !node_name.is_empty(),
            "TreeSyncAdapter node_name must not be empty",
        );
        Arc::new(Self {
            tenant_deltas,
            pending_deltas: DashMap::new(),
            node_name,
            drain_handle: OnceLock::new(),
        })
    }

    /// Register the drain callback and start the inbound task. Must
    /// be called exactly once per adapter — a second call panics via
    /// the mesh's one-drain-per-prefix invariant. The spawned recv
    /// loop runs until the namespace's subscriber channel closes
    /// (only on `MeshKV` drop).
    pub fn start(self: &Arc<Self>) {
        // Capture a `Weak` ref so the drain closure doesn't keep the
        // adapter alive. A strong `Arc` here would cycle:
        // `TreeSyncAdapter → DrainHandle → DrainRegistry → drain
        // closure → TreeSyncAdapter`, and `DrainHandle::drop` would
        // never fire, leaking the drain registration. The upgrade
        // check returns an empty batch if the adapter has already
        // been dropped — the drain is a no-op until the mesh tears
        // the `DrainHandle` down on its own `Drop`.
        let drain_owner = Arc::downgrade(self);
        let handle = self.tenant_deltas.register_drain(Box::new(move || {
            drain_owner
                .upgrade()
                .map(|this| this.drain_pending_deltas())
                .unwrap_or_default()
        }));
        assert!(
            self.drain_handle.set(handle).is_ok(),
            "TreeSyncAdapter::start called more than once",
        );

        let mut sub = self.tenant_deltas.subscribe("");
        #[expect(
            clippy::disallowed_methods,
            reason = "subscription task ends automatically when the mesh KV drops and closes the channel; no handle needed"
        )]
        tokio::spawn(async move {
            while let Some((key, value)) = sub.receiver.recv().await {
                let Some(model_id) = key.strip_prefix(PREFIX).filter(|s| !s.is_empty()) else {
                    warn!(key, "td: subscription yielded unexpected key shape");
                    continue;
                };
                match value {
                    Some(fragments) => Self::handle_incoming_batch(model_id, &fragments),
                    None => {
                        // Stream namespaces don't emit tombstones in
                        // the current mesh, but the subscription API
                        // type is shared with CRDT. Log once and
                        // keep going — nothing to apply.
                        debug!(model_id, "unexpected td: tombstone event");
                    }
                }
            }
            debug!("TreeSyncAdapter tenant-delta subscription closed");
        });
    }

    /// Buffer a local tree insert for the next gossip round. Called
    /// by `CacheAwarePolicy` on every local cache event; the batch
    /// is flushed by the drain callback and must not do anything
    /// heavy here.
    ///
    /// `delta.node_hash` must not be zero: `GLOBAL_EVICTION_HASH` in
    /// `smg_mesh::tree_ops` reserves 0 as the "evict everywhere"
    /// sentinel, and the `hash_node_path` / `hash_token_path`
    /// producers both remap 0→1 to keep the space disjoint. A zero
    /// hash here would collide with the sentinel in the apply path
    /// landing next slice.
    pub fn on_local_insert(&self, model_id: &str, delta: TreeDelta) {
        debug_assert!(
            !model_id.is_empty(),
            "TreeSyncAdapter::on_local_insert requires non-empty model_id",
        );
        debug_assert_ne!(
            delta.node_hash, 0,
            "TreeDelta.node_hash must be non-zero (0 is reserved for GLOBAL_EVICTION_HASH)",
        );
        self.pending_deltas
            .entry(model_id.to_string())
            .or_default()
            .push(delta);
    }

    /// Collect each model's buffer into a single `td:{model_id}`
    /// stream entry. Called exactly once per gossip round by the
    /// mesh. The iterate→remove pattern avoids iterating while
    /// mutating, which would deadlock the `DashMap` shards.
    fn drain_pending_deltas(&self) -> Vec<(String, Bytes)> {
        let model_ids: Vec<String> = self
            .pending_deltas
            .iter()
            .filter(|e| !e.value().is_empty())
            .map(|e| e.key().clone())
            .collect();

        let mut entries = Vec::with_capacity(model_ids.len());
        for model_id in model_ids {
            let Some((_, deltas)) = self.pending_deltas.remove(&model_id) else {
                continue;
            };
            if deltas.is_empty() {
                continue;
            }
            match bincode::serialize(&deltas) {
                Ok(bytes) => {
                    entries.push((format!("{PREFIX}{model_id}"), Bytes::from(bytes)));
                }
                Err(err) => {
                    // Serialisation should never fail for this
                    // schema; if it does, drop the batch so we don't
                    // re-enter and log on every round.
                    warn!(model_id, %err, "failed to serialize tenant deltas");
                }
            }
        }
        entries
    }

    fn handle_incoming_batch(model_id: &str, fragments: &[Bytes]) {
        let total = fragments.iter().map(Bytes::len).sum();
        let mut bytes = Vec::with_capacity(total);
        for frag in fragments {
            bytes.extend_from_slice(frag);
        }
        match bincode::deserialize::<Vec<TreeDelta>>(&bytes) {
            Ok(batch) => {
                debug!(
                    model_id,
                    count = batch.len(),
                    "remote tenant-delta batch received"
                );
                // Apply + unknown-hash repair are in the next slice.
                // For now, trace-log each delta so observability is
                // available during manual soak.
                for delta in &batch {
                    trace!(
                        model_id,
                        kind = ?delta.tree_kind,
                        hash = delta.node_hash,
                        worker_url = %delta.worker_url,
                        epoch = delta.epoch,
                        "remote tenant delta",
                    );
                }
            }
            Err(err) => warn!(model_id, %err, "failed to decode tenant-delta batch"),
        }
    }
}

#[cfg(test)]
mod tests {
    use smg_mesh::{MeshKV, StreamConfig, StreamRouting};

    use super::*;

    fn td_namespace(mesh: &MeshKV) -> Arc<StreamNamespace> {
        mesh.configure_stream_prefix(
            PREFIX,
            StreamConfig {
                max_buffer_bytes: 64 * 1024,
                routing: StreamRouting::Broadcast,
            },
        )
    }

    fn delta(hash: u64, worker: &str) -> TreeDelta {
        TreeDelta {
            tree_kind: TreeKind::String,
            node_hash: hash,
            worker_url: worker.into(),
            epoch: 1,
        }
    }

    #[tokio::test]
    async fn tree_delta_bincode_round_trip() {
        let batch = vec![
            TreeDelta {
                tree_kind: TreeKind::String,
                node_hash: 7,
                worker_url: "http://w1".into(),
                epoch: 42,
            },
            TreeDelta {
                tree_kind: TreeKind::Token,
                node_hash: u64::MAX,
                worker_url: "http://w2".into(),
                epoch: 0,
            },
        ];
        let bytes = bincode::serialize(&batch).unwrap();
        let decoded: Vec<TreeDelta> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded, batch);
    }

    #[tokio::test]
    async fn on_local_insert_buffers_per_model() {
        let mesh = MeshKV::new("node-a".into());
        let ns = td_namespace(&mesh);
        let adapter = TreeSyncAdapter::new(ns, "node-a".into());

        adapter.on_local_insert("model-1", delta(1, "http://w1"));
        adapter.on_local_insert("model-1", delta(2, "http://w1"));
        adapter.on_local_insert("model-2", delta(3, "http://w2"));

        assert_eq!(adapter.pending_deltas.get("model-1").unwrap().len(), 2);
        assert_eq!(adapter.pending_deltas.get("model-2").unwrap().len(), 1);
    }

    #[tokio::test]
    async fn drain_batches_per_model_and_clears_buffer() {
        let mesh = MeshKV::new("node-a".into());
        let ns = td_namespace(&mesh);
        let adapter = TreeSyncAdapter::new(ns, "node-a".into());

        adapter.on_local_insert("model-1", delta(1, "http://w1"));
        adapter.on_local_insert("model-1", delta(2, "http://w1"));
        adapter.on_local_insert("model-2", delta(3, "http://w2"));

        let entries = adapter.drain_pending_deltas();
        assert_eq!(entries.len(), 2, "one batch per model");

        // Each batch round-trips into the original per-model deltas.
        let mut by_key: std::collections::HashMap<String, Vec<TreeDelta>> =
            std::collections::HashMap::new();
        for (key, bytes) in entries {
            let batch: Vec<TreeDelta> = bincode::deserialize(&bytes).unwrap();
            by_key.insert(key, batch);
        }
        assert_eq!(by_key.get("td:model-1").unwrap().len(), 2);
        assert_eq!(by_key.get("td:model-2").unwrap().len(), 1);

        // Buffer is emptied on drain so the next round starts fresh.
        assert!(adapter.pending_deltas.is_empty());
    }

    #[tokio::test]
    async fn drain_skips_empty_model_buffers() {
        // `on_local_insert` creates an entry; if a test ever fills
        // and then clears a model, the next drain must not emit an
        // empty batch (empty batches would burn gossip bandwidth).
        let mesh = MeshKV::new("node-a".into());
        let ns = td_namespace(&mesh);
        let adapter = TreeSyncAdapter::new(ns, "node-a".into());

        adapter
            .pending_deltas
            .insert("model-empty".into(), Vec::new());

        let entries = adapter.drain_pending_deltas();
        assert!(entries.is_empty());
    }

    #[tokio::test]
    async fn start_registers_drain_with_mesh_round_collector() {
        // Exercise the end-to-end outbound path: start() → drain
        // registration → mesh.collect_round_batch() pulls our entries.
        let mesh = MeshKV::new("node-a".into());
        let ns = td_namespace(&mesh);
        let adapter = TreeSyncAdapter::new(ns, "node-a".into());
        adapter.start();

        adapter.on_local_insert("model-1", delta(10, "http://w1"));
        adapter.on_local_insert("model-2", delta(20, "http://w2"));

        let round = mesh.collect_round_batch();
        let keys: std::collections::HashSet<String> =
            round.drain_entries.iter().map(|(k, _)| k.clone()).collect();
        assert!(keys.contains("td:model-1"));
        assert!(keys.contains("td:model-2"));
    }

    #[tokio::test]
    async fn drain_closure_uses_weak_reference() {
        // Dropping the only strong `Arc` to the adapter must actually
        // drop it. If the drain closure held `Arc<Self>`, the cycle
        // through the DrainRegistry would keep the adapter alive
        // until MeshKV drop. Verify via a `Weak::upgrade` check.
        let mesh = MeshKV::new("node-a".into());
        let ns = td_namespace(&mesh);
        let adapter = TreeSyncAdapter::new(ns, "node-a".into());
        adapter.start();

        let weak = Arc::downgrade(&adapter);
        drop(adapter);
        assert!(
            weak.upgrade().is_none(),
            "drain closure must not strongly hold the adapter",
        );

        // The drain is now a no-op. Collecting a round should not
        // panic and should not yield any entries from this prefix.
        let round = mesh.collect_round_batch();
        let td_entries: Vec<_> = round
            .drain_entries
            .iter()
            .filter(|(k, _)| k.starts_with("td:"))
            .collect();
        assert!(td_entries.is_empty());
    }

    #[tokio::test]
    #[should_panic(expected = "TreeSyncAdapter requires a tenant-delta namespace scoped to `td:`")]
    async fn new_rejects_wrong_prefix() {
        let mesh = MeshKV::new("node-a".into());
        let ns = mesh.configure_stream_prefix(
            "tree:req:",
            StreamConfig {
                max_buffer_bytes: 64 * 1024,
                routing: StreamRouting::Targeted,
            },
        );
        let _ = TreeSyncAdapter::new(ns, "node-a".into());
    }

    #[tokio::test]
    #[should_panic(expected = "node_name must not be empty")]
    async fn new_rejects_empty_node_name() {
        let mesh = MeshKV::new("node-a".into());
        let ns = td_namespace(&mesh);
        let _ = TreeSyncAdapter::new(ns, String::new());
    }

    #[tokio::test]
    #[should_panic(expected = "drain already registered for prefix 'td:'")]
    async fn start_is_fused() {
        // The mesh drain registry allows only one callback per
        // prefix, so a second start() on the same adapter must fail
        // loudly rather than register a phantom drain that the
        // OnceLock silently drops.
        let mesh = MeshKV::new("node-a".into());
        let ns = td_namespace(&mesh);
        let adapter = TreeSyncAdapter::new(ns, "node-a".into());
        adapter.start();
        adapter.start();
    }
}
