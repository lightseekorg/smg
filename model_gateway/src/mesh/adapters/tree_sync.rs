//! `td:` stream adapter: gateway ↔ mesh bridge for the distributed
//! prefix tree.
//!
//! - Outbound: `on_local_insert` buffers per-model `TreeDelta`s;
//!   the drain callback batches each model into one
//!   `td:{model_id}` stream entry per gossip round.
//! - Inbound: a spawned task subscribes to `td:`, decodes each
//!   batch, and hands every delta to [`TreeHandle`] which either
//!   applies it (the matched prefix is already known locally and
//!   the worker becomes a tenant of that node) or returns false.
//!   On false, the adapter publishes a `tree:req:` repair request
//!   targeted at a random ALIVE peer; the response (`tree:page:`)
//!   is consumed in a later slice.
//!
//! The adapter holds no tree-membership state. The tree owner
//! (`CacheAwarePolicy` in production) implements [`TreeHandle`];
//! dependency direction is adapter → policy. Membership lookups
//! likewise go through a [`PeerList`] trait so the adapter doesn't
//! reach into the gossip controller directly.

use std::sync::{Arc, OnceLock};

use bytes::Bytes;
use dashmap::DashMap;
use rand::seq::IndexedRandom;
use serde::{Deserialize, Serialize};
use smg_mesh::{DrainHandle, StreamNamespace};
use tracing::{debug, trace, warn};
use uuid::Uuid;

use crate::policies::{TreeHandle, TreeKind};

const PREFIX: &str = "td:";
const REPAIR_REQUEST_PREFIX: &str = "tree:req:";

/// Live-membership view consumed by the adapter when picking a
/// peer to send a repair request to. Defined here (not in the
/// mesh crate) so adapter tests can supply a mock without spinning
/// up gossip; the production impl wraps the gossip controller's
/// alive-peer list and lands with the slice that wires the
/// adapter into `server.rs`.
pub trait PeerList: Send + Sync + std::fmt::Debug {
    /// Currently-ALIVE peer names, excluding the local node.
    /// Order is implementation-defined; the adapter picks one at
    /// random.
    fn alive_peers(&self) -> Vec<String>;
}

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

/// Why a node is asking for repair. Carried on the wire so the
/// responder can log/diagnose; doesn't change the response shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepairReason {
    /// Inbound `TreeDelta` referenced a hash this node didn't know.
    UnknownHash(u64),
}

/// Wire format for a `tree:req:{session_id}` message. Targeted at
/// one peer; the responder generates `tree:page:{session_id}:N`
/// fragments back to `requester_peer_id`. Slice 5d implements the
/// responder + page handler; slice 5c only emits these.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeRepairRequest {
    pub session_id: Uuid,
    pub requester_peer_id: String,
    pub target_peer_id: String,
    pub model_id: String,
    pub tree_kind: TreeKind,
    /// Resumable pagination cursor. `None` means "from the
    /// beginning"; slice 5d will set this on retry after a
    /// timeout to skip already-applied pages.
    pub cursor: Option<Vec<u8>>,
    pub reason: RepairReason,
}

/// Bridges the `td:` broadcast stream namespace to per-model
/// tenant buffers, querying a [`TreeHandle`] for inbound hash
/// resolution and a [`PeerList`] for repair-request targeting.
pub struct TreeSyncAdapter {
    tenant_deltas: Arc<StreamNamespace>,
    /// Targeted namespace for `tree:req:*` repair requests. The
    /// adapter only publishes here in slice 5c; subscription (the
    /// responder side) lands in slice 5d.
    tree_repair_requests: Arc<StreamNamespace>,
    pending_deltas: DashMap<String, Vec<TreeDelta>>,
    /// Hash-membership handle provided by the tree owner.
    tree: Arc<dyn TreeHandle>,
    /// Live-peer source for repair-request targeting.
    peers: Arc<dyn PeerList>,
    /// Outstanding repair sessions, keyed by (model_id, kind).
    /// Presence-only set: while a key is here, additional unknown
    /// hashes for the same (model, kind) are coalesced into the
    /// in-flight session. Slice 5d adds the response-side cleanup
    /// (and the per-session bookkeeping that needs reading).
    outstanding_repairs: DashMap<(String, TreeKind), ()>,
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
            .field("outstanding_repairs", &self.outstanding_repairs.len())
            .finish()
    }
}

impl TreeSyncAdapter {
    /// Build an adapter wrapping the `td:` broadcast namespace
    /// (tenant deltas), the `tree:req:` targeted namespace
    /// (repair requests), the local tree handle, peer list, and
    /// the local node name. Panics if either namespace prefix is
    /// wrong so a mis-wired caller fails loudly at startup instead
    /// of fanning entries into the wrong stream.
    pub fn new(
        tenant_deltas: Arc<StreamNamespace>,
        tree_repair_requests: Arc<StreamNamespace>,
        tree: Arc<dyn TreeHandle>,
        peers: Arc<dyn PeerList>,
        node_name: String,
    ) -> Arc<Self> {
        assert_eq!(
            tenant_deltas.prefix(),
            PREFIX,
            "TreeSyncAdapter requires a tenant-delta namespace scoped to `{PREFIX}`",
        );
        assert_eq!(
            tree_repair_requests.prefix(),
            REPAIR_REQUEST_PREFIX,
            "TreeSyncAdapter requires a repair-request namespace scoped to `{REPAIR_REQUEST_PREFIX}`",
        );
        assert!(
            !node_name.is_empty(),
            "TreeSyncAdapter node_name must not be empty",
        );
        Arc::new(Self {
            tenant_deltas,
            tree_repair_requests,
            pending_deltas: DashMap::new(),
            tree,
            peers,
            outstanding_repairs: DashMap::new(),
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
            if self.tree.apply_known_remote_insert(
                model_id,
                delta.tree_kind,
                delta.node_hash,
                &delta.worker_url,
            ) {
                trace!(
                    model_id,
                    kind = ?delta.tree_kind,
                    hash = delta.node_hash,
                    worker_url = %delta.worker_url,
                    epoch = delta.epoch,
                    "applied remote tenant delta against local tree",
                );
            } else {
                debug!(
                    model_id,
                    kind = ?delta.tree_kind,
                    hash = delta.node_hash,
                    worker_url = %delta.worker_url,
                    epoch = delta.epoch,
                    "unknown remote tenant delta hash, requesting repair",
                );
                self.request_repair_for_unknown_hash(model_id, delta.tree_kind, delta.node_hash);
            }
        }
    }

    /// Issue a `tree:req:{session_id}` repair request for an
    /// unknown hash, targeted at a random ALIVE peer. Coalesces
    /// duplicates: while a session is already in flight for the
    /// same `(model_id, tree_kind)`, subsequent unknown-hash
    /// callbacks are dropped — the in-flight request asks for the
    /// whole tree (`cursor=None`), so it'll cover the new hash
    /// when the response lands (slice 5d).
    fn request_repair_for_unknown_hash(&self, model_id: &str, tree_kind: TreeKind, node_hash: u64) {
        let key = (model_id.to_string(), tree_kind);
        if self.outstanding_repairs.contains_key(&key) {
            trace!(
                model_id,
                kind = ?tree_kind,
                hash = node_hash,
                "repair already in flight, coalescing unknown-hash trigger",
            );
            return;
        }

        let alive = self.peers.alive_peers();
        let Some(target) = alive.choose(&mut rand::rng()) else {
            warn!(
                model_id,
                kind = ?tree_kind,
                hash = node_hash,
                "no alive peers to request repair from; will retry on next unknown delta",
            );
            return;
        };

        let request = TreeRepairRequest {
            session_id: Uuid::now_v7(),
            requester_peer_id: self.node_name.clone(),
            target_peer_id: target.clone(),
            model_id: model_id.to_string(),
            tree_kind,
            cursor: None,
            reason: RepairReason::UnknownHash(node_hash),
        };

        let bytes = match bincode::serialize(&request) {
            Ok(bytes) => Bytes::from(bytes),
            Err(err) => {
                // Schema is fixed-shape — should be unreachable.
                warn!(model_id, %err, "failed to serialize tree repair request");
                return;
            }
        };

        let stream_key = format!("{REPAIR_REQUEST_PREFIX}{}", request.session_id);
        self.tree_repair_requests
            .publish_to(target, &stream_key, bytes);

        // Record the in-flight session AFTER publish so a publish
        // that panics on a programming error doesn't leave a
        // ghost entry blocking future requests.
        self.outstanding_repairs.insert(key, ());

        debug!(
            model_id,
            kind = ?tree_kind,
            hash = node_hash,
            target = %target,
            session_id = %request.session_id,
            "sent tree repair request",
        );
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

    fn req_namespace(mesh: &MeshKV) -> Arc<StreamNamespace> {
        mesh.configure_stream_prefix(
            REPAIR_REQUEST_PREFIX,
            StreamConfig {
                max_buffer_bytes: 64 * 1024,
                routing: StreamRouting::Targeted,
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

    /// Test-only [`TreeHandle`]: a set of known `(model, kind,
    /// hash)` tuples plus a log of every applied call so tests
    /// can assert on apply-side effects. Apply succeeds (returns
    /// `true`) iff the hash is in the known set.
    #[derive(Debug, Default)]
    struct MockTreeHandle {
        known: DashMap<(String, TreeKind, u64), ()>,
        applied: parking_lot::Mutex<Vec<(String, TreeKind, u64, String)>>,
    }

    impl MockTreeHandle {
        fn mark_known(&self, model_id: &str, tree_kind: TreeKind, node_hash: u64) {
            self.known
                .insert((model_id.to_string(), tree_kind, node_hash), ());
        }

        fn applied_calls(&self) -> Vec<(String, TreeKind, u64, String)> {
            self.applied.lock().clone()
        }
    }

    impl TreeHandle for MockTreeHandle {
        fn apply_known_remote_insert(
            &self,
            model_id: &str,
            tree_kind: TreeKind,
            node_hash: u64,
            worker_url: &str,
        ) -> bool {
            if !self
                .known
                .contains_key(&(model_id.to_string(), tree_kind, node_hash))
            {
                return false;
            }
            self.applied.lock().push((
                model_id.to_string(),
                tree_kind,
                node_hash,
                worker_url.to_string(),
            ));
            true
        }
    }

    fn empty_handle() -> Arc<MockTreeHandle> {
        Arc::new(MockTreeHandle::default())
    }

    /// Test-only [`PeerList`] returning a fixed list of peers.
    #[derive(Debug, Default)]
    struct MockPeerList {
        peers: parking_lot::Mutex<Vec<String>>,
    }

    impl MockPeerList {
        fn with(peers: &[&str]) -> Arc<Self> {
            Arc::new(Self {
                peers: parking_lot::Mutex::new(peers.iter().map(|s| (*s).into()).collect()),
            })
        }
    }

    impl PeerList for MockPeerList {
        fn alive_peers(&self) -> Vec<String> {
            self.peers.lock().clone()
        }
    }

    fn empty_peers() -> Arc<MockPeerList> {
        Arc::new(MockPeerList::default())
    }

    fn adapter_with_empty_handle(mesh: &MeshKV, node_name: &str) -> Arc<TreeSyncAdapter> {
        let td = td_namespace(mesh);
        let req = req_namespace(mesh);
        let tree: Arc<dyn TreeHandle> = empty_handle();
        let peers: Arc<dyn PeerList> = empty_peers();
        TreeSyncAdapter::new(td, req, tree, peers, node_name.into())
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
    async fn handle_incoming_batch_applies_known_and_skips_unknown() {
        // Known deltas hit `apply_known_remote_insert`; unknown
        // ones don't (and trigger repair, exercised in dedicated
        // tests below). Kinds must not alias: same hash on a
        // different `TreeKind` is a different (model, kind, hash)
        // tuple and stays unknown.
        let mesh = MeshKV::new("node-a".into());
        let td = td_namespace(&mesh);
        let req = req_namespace(&mesh);
        let tree = Arc::new(MockTreeHandle::default());
        tree.mark_known("model-1", TreeKind::String, 42);
        let adapter_tree: Arc<dyn TreeHandle> = tree.clone();
        let peers: Arc<dyn PeerList> = MockPeerList::with(&["node-b"]);
        let adapter = TreeSyncAdapter::new(td, req, adapter_tree, peers, "node-a".into());

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

        let applied = tree.applied_calls();
        assert_eq!(
            applied,
            vec![(
                "model-1".to_string(),
                TreeKind::String,
                42,
                "http://w1".to_string(),
            )],
            "only the (String, 42) delta applied; unknown String:99 and Token:42 did not",
        );
    }

    #[tokio::test]
    async fn handle_incoming_batch_ignores_malformed_payload() {
        // Corrupt batch → no propagation, no panic.
        let mesh = MeshKV::new("node-a".into());
        let adapter = adapter_with_empty_handle(&mesh, "node-a");

        adapter.handle_incoming_batch("model-1", &[Bytes::from_static(b"not-bincode")]);
    }

    #[tokio::test]
    #[should_panic(expected = "TreeSyncAdapter requires a tenant-delta namespace scoped to `td:`")]
    async fn new_rejects_wrong_td_prefix() {
        // Pass the repair-request namespace as the td: arg.
        let mesh = MeshKV::new("node-a".into());
        let req = req_namespace(&mesh);
        let req_again = mesh.configure_stream_prefix(
            "td-misnamed:",
            StreamConfig {
                max_buffer_bytes: 64 * 1024,
                routing: StreamRouting::Broadcast,
            },
        );
        let tree: Arc<dyn TreeHandle> = empty_handle();
        let peers: Arc<dyn PeerList> = empty_peers();
        let _ = TreeSyncAdapter::new(req_again, req, tree, peers, "node-a".into());
    }

    #[tokio::test]
    #[should_panic(
        expected = "TreeSyncAdapter requires a repair-request namespace scoped to `tree:req:`"
    )]
    async fn new_rejects_wrong_repair_prefix() {
        // Pass the td: namespace as the repair-request arg.
        let mesh = MeshKV::new("node-a".into());
        let td = td_namespace(&mesh);
        let td_again = mesh.configure_stream_prefix(
            "td-also:",
            StreamConfig {
                max_buffer_bytes: 64 * 1024,
                routing: StreamRouting::Broadcast,
            },
        );
        let tree: Arc<dyn TreeHandle> = empty_handle();
        let peers: Arc<dyn PeerList> = empty_peers();
        let _ = TreeSyncAdapter::new(td, td_again, tree, peers, "node-a".into());
    }

    #[tokio::test]
    #[should_panic(expected = "node_name must not be empty")]
    async fn new_rejects_empty_node_name() {
        let mesh = MeshKV::new("node-a".into());
        let td = td_namespace(&mesh);
        let req = req_namespace(&mesh);
        let tree: Arc<dyn TreeHandle> = empty_handle();
        let peers: Arc<dyn PeerList> = empty_peers();
        let _ = TreeSyncAdapter::new(td, req, tree, peers, String::new());
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

    #[tokio::test]
    async fn tree_repair_request_bincode_round_trip() {
        let req = TreeRepairRequest {
            session_id: Uuid::now_v7(),
            requester_peer_id: "node-a".into(),
            target_peer_id: "node-b".into(),
            model_id: "model-1".into(),
            tree_kind: TreeKind::Token,
            cursor: Some(vec![1, 2, 3]),
            reason: RepairReason::UnknownHash(42),
        };
        let bytes = bincode::serialize(&req).unwrap();
        let decoded: TreeRepairRequest = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded, req);
    }

    /// Drain the targeted-stream side buffer for repair requests
    /// so we can inspect what the adapter published.
    fn drain_repair_publishes(mesh: &MeshKV) -> Vec<(String, String, Bytes)> {
        let round = mesh.collect_round_batch();
        round
            .targeted_entries
            .into_iter()
            .filter(|(_, key, _)| key.starts_with(REPAIR_REQUEST_PREFIX))
            .collect()
    }

    #[tokio::test]
    async fn unknown_hash_publishes_repair_request_to_alive_peer() {
        let mesh = MeshKV::new("node-a".into());
        let td = td_namespace(&mesh);
        let req = req_namespace(&mesh);
        let tree: Arc<dyn TreeHandle> = Arc::new(MockTreeHandle::default());
        let peers: Arc<dyn PeerList> = MockPeerList::with(&["node-b"]);
        let adapter = TreeSyncAdapter::new(td, req, tree, peers, "node-a".into());

        // Single unknown delta → one targeted publish to node-b.
        let batch = vec![delta(99, "http://w1")];
        let bytes = Bytes::from(bincode::serialize(&batch).unwrap());
        adapter.handle_incoming_batch("model-1", &[bytes]);

        let publishes = drain_repair_publishes(&mesh);
        assert_eq!(publishes.len(), 1, "exactly one repair request");
        let (target, key, payload) = &publishes[0];
        assert_eq!(target, "node-b");
        assert!(key.starts_with(REPAIR_REQUEST_PREFIX));

        let request: TreeRepairRequest = bincode::deserialize(payload).unwrap();
        assert_eq!(request.requester_peer_id, "node-a");
        assert_eq!(request.target_peer_id, "node-b");
        assert_eq!(request.model_id, "model-1");
        assert_eq!(request.tree_kind, TreeKind::String);
        assert_eq!(request.reason, RepairReason::UnknownHash(99));
        assert!(request.cursor.is_none());

        // Outstanding-session bookkeeping records the in-flight key.
        assert!(adapter
            .outstanding_repairs
            .contains_key(&("model-1".to_string(), TreeKind::String)),);
    }

    #[tokio::test]
    async fn coalesces_duplicate_repair_for_same_model_kind() {
        // Two unknown hashes for the same (model, kind) within one
        // batch must produce only one repair request — the
        // in-flight session covers both.
        let mesh = MeshKV::new("node-a".into());
        let td = td_namespace(&mesh);
        let req = req_namespace(&mesh);
        let tree: Arc<dyn TreeHandle> = Arc::new(MockTreeHandle::default());
        let peers: Arc<dyn PeerList> = MockPeerList::with(&["node-b"]);
        let adapter = TreeSyncAdapter::new(td, req, tree, peers, "node-a".into());

        let batch = vec![delta(101, "http://w1"), delta(102, "http://w2")];
        let bytes = Bytes::from(bincode::serialize(&batch).unwrap());
        adapter.handle_incoming_batch("model-1", &[bytes]);

        let publishes = drain_repair_publishes(&mesh);
        assert_eq!(publishes.len(), 1, "duplicate request should coalesce");
    }

    #[tokio::test]
    async fn separate_kinds_get_separate_repair_sessions() {
        // String and Token unknown hashes for the same model are
        // distinct sessions — coalescing keys on (model, kind).
        let mesh = MeshKV::new("node-a".into());
        let td = td_namespace(&mesh);
        let req = req_namespace(&mesh);
        let tree: Arc<dyn TreeHandle> = Arc::new(MockTreeHandle::default());
        let peers: Arc<dyn PeerList> = MockPeerList::with(&["node-b"]);
        let adapter = TreeSyncAdapter::new(td, req, tree, peers, "node-a".into());

        let batch = vec![
            TreeDelta {
                tree_kind: TreeKind::String,
                node_hash: 1,
                worker_url: "http://w1".into(),
                epoch: 1,
            },
            TreeDelta {
                tree_kind: TreeKind::Token,
                node_hash: 1,
                worker_url: "http://w2".into(),
                epoch: 1,
            },
        ];
        let bytes = Bytes::from(bincode::serialize(&batch).unwrap());
        adapter.handle_incoming_batch("model-1", &[bytes]);

        let publishes = drain_repair_publishes(&mesh);
        assert_eq!(publishes.len(), 2, "one per (model, kind) pair");
    }

    #[tokio::test]
    async fn no_alive_peers_skips_repair_silently() {
        // Empty peer list → no publish, no panic, and no entry
        // recorded so a future delta retries once peers come up.
        let mesh = MeshKV::new("node-a".into());
        let adapter = adapter_with_empty_handle(&mesh, "node-a");

        let batch = vec![delta(99, "http://w1")];
        let bytes = Bytes::from(bincode::serialize(&batch).unwrap());
        adapter.handle_incoming_batch("model-1", &[bytes]);

        let publishes = drain_repair_publishes(&mesh);
        assert!(publishes.is_empty(), "no peers means nothing published");
        assert!(
            adapter.outstanding_repairs.is_empty(),
            "no session recorded so the next delta can retry",
        );
    }

    #[tokio::test]
    async fn known_hashes_do_not_trigger_repair() {
        let mesh = MeshKV::new("node-a".into());
        let td = td_namespace(&mesh);
        let req = req_namespace(&mesh);
        let tree = Arc::new(MockTreeHandle::default());
        tree.mark_known("model-1", TreeKind::String, 42);
        let adapter_tree: Arc<dyn TreeHandle> = tree.clone();
        let peers: Arc<dyn PeerList> = MockPeerList::with(&["node-b"]);
        let adapter = TreeSyncAdapter::new(td, req, adapter_tree, peers, "node-a".into());

        let batch = vec![delta(42, "http://w1")];
        let bytes = Bytes::from(bincode::serialize(&batch).unwrap());
        adapter.handle_incoming_batch("model-1", &[bytes]);

        let publishes = drain_repair_publishes(&mesh);
        assert!(publishes.is_empty(), "known hash skips repair entirely");
    }
}
