//! `td:` stream adapter: gateway ↔ mesh bridge for the distributed
//! prefix tree. Scope so far: tenant-delta fast path + inbound
//! hash resolution via [`TreeHandle`].
//!
//! - Outbound: `on_local_insert` buffers per-model `TreeDelta`s;
//!   the drain callback batches each model into one
//!   `td:{model_id}` stream entry per gossip round.
//! - Inbound: a spawned task subscribes to `td:`, decodes each
//!   batch, and asks the [`TreeHandle`] whether each delta's hash
//!   is locally known. Known → trace, unknown → debug (repair in
//!   a later slice).
//!
//! The adapter holds no tree-membership state. The tree owner
//! (`CacheAwarePolicy` in production) implements [`TreeHandle`];
//! dependency direction is adapter → policy.

use std::sync::{Arc, OnceLock};

use bytes::Bytes;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use smg_mesh::{DrainHandle, StreamNamespace};
use tracing::{debug, trace, warn};

use crate::policies::{TreeHandle, TreeKind};

const PREFIX: &str = "td:";

/// One prefix-tree change observed on a producing node. `node_hash`
/// is the blake3 8-byte id scoped by `(model_id, tree_kind, path)`;
/// `model_id` lives in the stream key, `tree_kind` is inside this
/// struct. Receivers resolve the hash via `TreeHandle`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeDelta {
    pub tree_kind: TreeKind,
    pub node_hash: u64,
    /// Worker URL that cached the prefix.
    pub worker_url: String,
    /// Cache-event epoch for intra-batch ordering on the receiver;
    /// the stream transport itself doesn't inspect it.
    pub epoch: u64,
}

/// Bridges the `td:` broadcast stream namespace to per-model
/// tenant buffers, querying a [`TreeHandle`] for inbound hash
/// resolution.
pub struct TreeSyncAdapter {
    tenant_deltas: Arc<StreamNamespace>,
    pending_deltas: DashMap<String, Vec<TreeDelta>>,
    /// Hash-membership oracle provided by the tree owner.
    tree: Arc<dyn TreeHandle>,
    node_name: String,
    /// Keeps the drain registration alive; dropping it unregisters
    /// from the mesh. `OnceLock` guards against a second `start`.
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

    /// Register the drain callback and start the inbound task. Call
    /// once per adapter — a second call panics via the mesh's
    /// one-drain-per-prefix invariant.
    pub fn start(self: &Arc<Self>) {
        // `Weak` avoids the `TreeSyncAdapter → DrainHandle →
        // DrainRegistry → drain closure → TreeSyncAdapter` strong
        // cycle that would leak the drain registration past adapter
        // drop. If upgrade fails, return an empty batch; the mesh
        // tears down the `DrainHandle` on its own `Drop`.
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

        // Same `Weak` pattern for the subscription task so a late
        // channel close can't strand the adapter alive. Exit on
        // first upgrade-None.
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
                        // Shared CRDT/stream subscription API — td:
                        // never emits tombstones today.
                        debug!(model_id, "unexpected td: tombstone event");
                    }
                }
            }
            debug!("TreeSyncAdapter tenant-delta subscription closed");
        });
    }

    /// Buffer a local tree insert for the next gossip round. Hot
    /// path — keep it cheap; the drain does the serialisation.
    /// `delta.node_hash` must be non-zero: 0 is
    /// `smg_mesh::tree_ops::GLOBAL_EVICTION_HASH` (producers remap
    /// 0→1 to keep the space disjoint).
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

    /// Collect each model's buffer into one `td:{model_id}` stream
    /// entry. Called once per gossip round. Iterate→collect→remove
    /// avoids the deadlock DashMap hits on iterate-and-mutate.
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
                    // Should be unreachable for this schema; drop
                    // the batch rather than re-enter next round.
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
                // Known locally; actual apply sink lands next slice.
                trace!(
                    model_id,
                    kind = ?delta.tree_kind,
                    hash = delta.node_hash,
                    worker_url = %delta.worker_url,
                    epoch = delta.epoch,
                    "resolved remote tenant delta against local tree handle",
                );
            } else {
                // Unknown — repair-request slice will turn this
                // into a `tree:req:` message later.
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

    /// Test-only [`TreeHandle`]: keyed by `(model_id, TreeKind)`
    /// with a set of known hashes.
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
        // Cleared model buckets must not emit empty batches.
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
        // End-to-end outbound: start → drain registration →
        // collect_round_batch pulls our entries.
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
        // Dropping the last strong Arc must actually drop the
        // adapter; a strong Arc in the drain closure would cycle
        // through DrainRegistry and leak it.
        let mesh = MeshKV::new("node-a".into());
        let adapter = adapter_with_empty_handle(&mesh, "node-a");
        adapter.start();

        let weak = Arc::downgrade(&adapter);
        drop(adapter);
        assert!(
            weak.upgrade().is_none(),
            "drain closure must not strongly hold the adapter",
        );

        // Drain is now a no-op; round produces no td: entries.
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
        // Deltas must be classified via the injected handle; kinds
        // must not alias (same hash, different kind ≠ match).
        // `handle_incoming_batch` is read-only on membership.
        let mesh = MeshKV::new("node-a".into());
        let ns = td_namespace(&mesh);
        let tree = Arc::new(MockTreeHandle::default());
        tree.insert("model-1", TreeKind::String, 42);
        let adapter_tree: Arc<dyn TreeHandle> = tree.clone();
        let adapter = TreeSyncAdapter::new(ns, adapter_tree, "node-a".into());

        let batch = vec![
            TreeDelta {
                tree_kind: TreeKind::String,
                node_hash: 42, // known
                worker_url: "http://w1".into(),
                epoch: 1,
            },
            TreeDelta {
                tree_kind: TreeKind::String,
                node_hash: 99, // unknown
                worker_url: "http://w2".into(),
                epoch: 1,
            },
            TreeDelta {
                tree_kind: TreeKind::Token,
                node_hash: 42, // same hash, different kind — must not alias
                worker_url: "http://w3".into(),
                epoch: 1,
            },
        ];
        let bytes = Bytes::from(bincode::serialize(&batch).unwrap());
        adapter.handle_incoming_batch("model-1", &[bytes]);

        assert!(tree.contains_hash("model-1", TreeKind::String, 42));
        assert!(!tree.contains_hash("model-1", TreeKind::String, 99));
        assert!(!tree.contains_hash("model-1", TreeKind::Token, 42));
    }

    #[tokio::test]
    async fn handle_incoming_batch_ignores_malformed_payload() {
        // Corrupt batch → no propagation, no panic.
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
        // Second start must panic at the mesh's
        // one-drain-per-prefix invariant.
        let mesh = MeshKV::new("node-a".into());
        let adapter = adapter_with_empty_handle(&mesh, "node-a");
        adapter.start();
        adapter.start();
    }
}
