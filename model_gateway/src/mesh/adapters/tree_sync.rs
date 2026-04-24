//! `td:` / `tree:*` stream adapter: gateway ↔ mesh bridge for the
//! distributed prefix tree.
//!
//! Current scope: tenant-delta fast path + inbound hash resolution
//! via [`crate::policies::cache_aware::TreeHandle`]. The adapter
//! holds no tree-membership state of its own — the tree-owning
//! component (`CacheAwarePolicy` in production) exposes the handle
//! and the adapter queries through it. This keeps the dependency
//! direction adapter → policy; the policy never imports from this
//! module.
//!
//! - Outbound: `on_local_insert` buffers per-model `TreeDelta`
//!   entries. The mesh drain callback, called once per gossip round,
//!   batches each model's buffer into a single `td:{model_id}`
//!   stream entry (bincode-serialised `Vec<TreeDelta>`).
//! - Inbound: a spawned task subscribes to `td:` and decodes
//!   incoming batches. For each delta, the adapter asks the tree
//!   handle "do you have this node locally?" — known hashes log
//!   at trace (the apply sink lands in the next slice), unknown
//!   hashes log at debug (the repair request lands in the slice
//!   after).
//!
//! Repair sessions and the apply sink are deliberately not in this
//! file yet so the handle contract can soak on its own.

use std::sync::{Arc, OnceLock};

use bytes::Bytes;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use smg_mesh::{DrainHandle, StreamNamespace};
use tracing::{debug, trace, warn};

use crate::policies::{TreeHandle, TreeKind};

const PREFIX: &str = "td:";

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
///
/// The adapter holds no tree-membership state of its own. The
/// tree-owning component (`CacheAwarePolicy` in production) exposes
/// a [`TreeHandle`] and the adapter queries through it. Layering
/// flows: `adapter → policies::cache_aware::TreeHandle ←
/// CacheAwarePolicy`, so the policy never imports from this module.
pub struct TreeSyncAdapter {
    tenant_deltas: Arc<StreamNamespace>,
    pending_deltas: DashMap<String, Vec<TreeDelta>>,
    /// Hash-membership oracle provided by the tree owner.
    tree: Arc<dyn TreeHandle>,
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
    pub fn new(
        tenant_deltas: Arc<StreamNamespace>,
        tree: Arc<dyn TreeHandle>,
        node_name: String,
    ) -> Arc<Self> {
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
            tree,
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

        // Same `Weak` pattern for the subscription task: a strong
        // `Arc<Self>` captured in the spawned future would keep the
        // adapter alive until MeshKV drops the subscription channel,
        // even if the caller has explicitly dropped its last
        // reference. Upgrade per iteration and exit the loop on
        // first-None so the task releases cleanly.
        let sub_owner = Arc::downgrade(self);
        let mut sub = self.tenant_deltas.subscribe("");
        #[expect(
            clippy::disallowed_methods,
            reason = "subscription task ends automatically when the mesh KV drops and closes the channel; no handle needed"
        )]
        tokio::spawn(async move {
            while let Some((key, value)) = sub.receiver.recv().await {
                let Some(this) = sub_owner.upgrade() else {
                    debug!("TreeSyncAdapter dropped, exiting tenant-delta subscription");
                    break;
                };
                let Some(model_id) = key.strip_prefix(PREFIX).filter(|s| !s.is_empty()) else {
                    warn!(key, "td: subscription yielded unexpected key shape");
                    continue;
                };
                match value {
                    Some(fragments) => this.handle_incoming_batch(model_id, &fragments),
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

    fn handle_incoming_batch(&self, model_id: &str, fragments: &[Bytes]) {
        let total = fragments.iter().map(Bytes::len).sum();
        let mut bytes = Vec::with_capacity(total);
        for frag in fragments {
            bytes.extend_from_slice(frag);
        }
        let batch: Vec<TreeDelta> = match bincode::deserialize(&bytes) {
            Ok(batch) => batch,
            Err(err) => {
                warn!(model_id, %err, "failed to decode tenant-delta batch");
                return;
            }
        };
        debug!(
            model_id,
            count = batch.len(),
            "remote tenant-delta batch received"
        );
        for delta in &batch {
            if self
                .tree
                .contains_hash(model_id, delta.tree_kind, delta.node_hash)
            {
                // Hash is locally known. Applying the tenant to the
                // matched tree node lands in the next slice via an
                // apply sink; for now, just log.
                trace!(
                    model_id,
                    kind = ?delta.tree_kind,
                    hash = delta.node_hash,
                    worker_url = %delta.worker_url,
                    epoch = delta.epoch,
                    "resolved remote tenant delta against local tree handle",
                );
            } else {
                // Unknown hash — the repair-request slice will turn
                // this into a `tree:req:` message. For now, debug-log
                // so soak runs can see unresolved deltas without
                // inflating the trace stream.
                debug!(
                    model_id,
                    kind = ?delta.tree_kind,
                    hash = delta.node_hash,
                    worker_url = %delta.worker_url,
                    epoch = delta.epoch,
                    "unknown remote tenant delta hash (repair deferred to next slice)",
                );
            }
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

    /// Test-only implementation of [`TreeHandle`]. Keyed by
    /// `(model_id, TreeKind)` with a set of known hashes. The real
    /// handle lives on `CacheAwarePolicy` and is backed by the
    /// policy's `path_hash_index` / `token_path_hash_index` — this
    /// mock only answers the contract, not the full policy.
    #[derive(Debug, Default)]
    struct MockTreeHandle {
        known: DashMap<(String, TreeKind), DashMap<u64, ()>>,
    }

    impl MockTreeHandle {
        fn insert(&self, model_id: &str, tree_kind: TreeKind, node_hash: u64) {
            self.known
                .entry((model_id.to_string(), tree_kind))
                .or_default()
                .insert(node_hash, ());
        }
    }

    impl TreeHandle for MockTreeHandle {
        fn contains_hash(&self, model_id: &str, tree_kind: TreeKind, node_hash: u64) -> bool {
            self.known
                .get(&(model_id.to_string(), tree_kind))
                .is_some_and(|entry| entry.contains_key(&node_hash))
        }
    }

    fn empty_handle() -> Arc<MockTreeHandle> {
        Arc::new(MockTreeHandle::default())
    }

    fn adapter_with_empty_handle(mesh: &MeshKV, node_name: &str) -> Arc<TreeSyncAdapter> {
        let ns = td_namespace(mesh);
        let tree: Arc<dyn TreeHandle> = empty_handle();
        TreeSyncAdapter::new(ns, tree, node_name.into())
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
        let adapter = adapter_with_empty_handle(&mesh, "node-a");

        adapter.on_local_insert("model-1", delta(1, "http://w1"));
        adapter.on_local_insert("model-1", delta(2, "http://w1"));
        adapter.on_local_insert("model-2", delta(3, "http://w2"));

        assert_eq!(adapter.pending_deltas.get("model-1").unwrap().len(), 2);
        assert_eq!(adapter.pending_deltas.get("model-2").unwrap().len(), 1);
    }

    #[tokio::test]
    async fn drain_batches_per_model_and_clears_buffer() {
        let mesh = MeshKV::new("node-a".into());
        let adapter = adapter_with_empty_handle(&mesh, "node-a");

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
        let adapter = adapter_with_empty_handle(&mesh, "node-a");

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
        let adapter = adapter_with_empty_handle(&mesh, "node-a");
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
        let adapter = adapter_with_empty_handle(&mesh, "node-a");
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
    async fn handle_incoming_batch_consults_tree() {
        // Inbound deltas must be classified via the injected
        // tree. Hashes the tree handle knows are "resolved"; those
        // it doesn't are "unknown". The tree handle itself is untouched
        // — `handle_incoming_batch` is read-only on membership.
        let mesh = MeshKV::new("node-a".into());
        let ns = td_namespace(&mesh);
        let tree = Arc::new(MockTreeHandle::default());
        tree.insert("model-1", TreeKind::String, 42);
        let adapter_tree: Arc<dyn TreeHandle> = tree.clone();
        let adapter = TreeSyncAdapter::new(ns, adapter_tree, "node-a".into());

        let batch = vec![
            TreeDelta {
                tree_kind: TreeKind::String,
                node_hash: 42, // known per tree handle
                worker_url: "http://w1".into(),
                epoch: 1,
            },
            TreeDelta {
                tree_kind: TreeKind::String,
                node_hash: 99, // unknown per tree handle
                worker_url: "http://w2".into(),
                epoch: 1,
            },
            TreeDelta {
                tree_kind: TreeKind::Token,
                node_hash: 42, // same hash but different kind — must not alias
                worker_url: "http://w3".into(),
                epoch: 1,
            },
        ];
        let bytes = Bytes::from(bincode::serialize(&batch).unwrap());
        adapter.handle_incoming_batch("model-1", &[bytes]);

        // Adapter neither mutates the tree handle nor caches membership.
        assert!(tree.contains_hash("model-1", TreeKind::String, 42));
        assert!(!tree.contains_hash("model-1", TreeKind::String, 99));
        assert!(!tree.contains_hash("model-1", TreeKind::Token, 42));
    }

    #[tokio::test]
    async fn handle_incoming_batch_ignores_malformed_payload() {
        // A corrupt batch must not propagate any state and must not
        // panic. The tree handle should never be queried since decode
        // fails before the per-delta loop.
        let mesh = MeshKV::new("node-a".into());
        let ns = td_namespace(&mesh);
        let tree: Arc<dyn TreeHandle> = Arc::new(MockTreeHandle::default());
        let adapter = TreeSyncAdapter::new(ns, tree, "node-a".into());

        adapter.handle_incoming_batch("model-1", &[Bytes::from_static(b"not-bincode")]);
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
        let tree: Arc<dyn TreeHandle> = empty_handle();
        let _ = TreeSyncAdapter::new(ns, tree, "node-a".into());
    }

    #[tokio::test]
    #[should_panic(expected = "node_name must not be empty")]
    async fn new_rejects_empty_node_name() {
        let mesh = MeshKV::new("node-a".into());
        let ns = td_namespace(&mesh);
        let tree: Arc<dyn TreeHandle> = empty_handle();
        let _ = TreeSyncAdapter::new(ns, tree, String::new());
    }

    #[tokio::test]
    #[should_panic(expected = "drain already registered for prefix 'td:'")]
    async fn start_is_fused() {
        // The mesh drain registry allows only one callback per
        // prefix, so a second start() on the same adapter must fail
        // loudly rather than register a phantom drain that the
        // OnceLock silently drops.
        let mesh = MeshKV::new("node-a".into());
        let adapter = adapter_with_empty_handle(&mesh, "node-a");
        adapter.start();
        adapter.start();
    }
}
