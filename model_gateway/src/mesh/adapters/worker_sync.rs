//! `worker:` CRDT adapter: gateway ↔ mesh bridge for `WorkerState`.
//!
//! Outbound: `on_worker_changed` bincode-serialises a `WorkerState`
//! and writes it under `worker:{worker_id}`. `on_worker_removed`
//! writes a tombstone. Both map 1:1 to the v1
//! `MeshSyncManager::sync_worker_state` / `remove_worker_state`
//! behaviour but target the typed CRDT namespace instead of the
//! untyped store.
//!
//! Inbound: `start` spawns a task that subscribes to the namespace
//! and routes each non-tombstone update through
//! `WorkerRegistry::on_remote_worker_state` — the same sink the v1
//! `WorkerStateSubscriber` wires up, so registry behaviour
//! (URL-dedupe, health promotion, `Registered` event fan-out) is
//! unchanged. Tombstones are logged at debug; the registry does not
//! yet expose a remote-remove hook, and wiring one belongs in a
//! later PR together with the v1 mirror teardown.
//!
//! The adapter writes through `CrdtNamespace::put`, which fires
//! local subscribers in addition to gossiping. A local write
//! therefore echoes back through `start`'s loop and lands in
//! `on_remote_worker_state`. That path is idempotent — the URL
//! lookup short-circuits to a health refresh — so the loop is
//! self-limiting and matches the v1 behaviour where a local
//! `sync_worker_state` also re-appears in the same node's store
//! before gossip fans out.

use std::sync::Arc;

use bytes::Bytes;
use smg_mesh::{CrdtNamespace, WorkerState, WorkerStateSubscriber};
use tracing::{debug, warn};

use crate::worker::WorkerRegistry;

const PREFIX: &str = "worker:";

/// Bridge between the `worker:` CRDT namespace and the gateway's
/// in-process `WorkerRegistry`.
pub struct WorkerSyncAdapter {
    workers: Arc<CrdtNamespace>,
    worker_registry: Arc<WorkerRegistry>,
}

impl std::fmt::Debug for WorkerSyncAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerSyncAdapter")
            .field("prefix", &self.workers.prefix())
            .finish_non_exhaustive()
    }
}

impl WorkerSyncAdapter {
    /// Build an adapter wrapping a `worker:`-scoped namespace and the
    /// gateway's worker registry. Panics if the namespace is not
    /// scoped to `worker:` so a mis-wired caller fails fast at
    /// startup rather than silently routing updates to the wrong
    /// prefix.
    pub fn new(workers: Arc<CrdtNamespace>, worker_registry: Arc<WorkerRegistry>) -> Arc<Self> {
        assert_eq!(
            workers.prefix(),
            PREFIX,
            "WorkerSyncAdapter requires a namespace scoped to `{PREFIX}`",
        );
        Arc::new(Self {
            workers,
            worker_registry,
        })
    }

    /// Start the inbound path. Subscribes first so no live event is
    /// lost, spawns the recv loop so it can start draining
    /// immediately, and then backfills from the calling thread — any
    /// entry already in the CRDT would otherwise have to wait for the
    /// next unrelated write before the registry saw it. Running the
    /// backfill outside the spawn keeps the live loop free to drain
    /// concurrently: `notify` uses `try_send` into a bounded mpsc, so
    /// a blocked recv while backfill is running could drop updates on
    /// a busy startup. `on_remote_worker_state` is idempotent on URL,
    /// so a key seen by both paths only refreshes health.
    pub fn start(self: &Arc<Self>) {
        let this = Arc::clone(self);
        let mut sub = self.workers.subscribe("");
        #[expect(
            clippy::disallowed_methods,
            reason = "subscription task ends automatically when the mesh KV drops and closes the channel; no handle needed"
        )]
        tokio::spawn(async move {
            while let Some((key, value)) = sub.receiver.recv().await {
                let Some(worker_id) = key.strip_prefix(PREFIX).filter(|s| !s.is_empty()) else {
                    warn!(key, "worker: subscription yielded unexpected key shape");
                    continue;
                };
                match value {
                    Some(fragments) => {
                        let total = fragments.iter().map(Bytes::len).sum();
                        let mut bytes = Vec::with_capacity(total);
                        for frag in fragments {
                            bytes.extend_from_slice(&frag);
                        }
                        this.apply_incoming(worker_id, &bytes);
                    }
                    None => debug!(
                        worker_id,
                        "remote worker tombstone (no-op pending registry remove-by-mesh hook)"
                    ),
                }
            }
            debug!("WorkerSyncAdapter subscription closed");
        });
        self.backfill_existing();
    }

    /// Replay every entry currently in the `worker:` namespace into
    /// the registry. Safe to run alongside the live subscription loop
    /// — the sink is idempotent on URL (health refresh short-circuit),
    /// so overlap with a concurrent live event is fine.
    fn backfill_existing(&self) {
        for key in self.workers.keys("") {
            let Some(worker_id) = key.strip_prefix(PREFIX).filter(|s| !s.is_empty()) else {
                warn!(key, "worker: backfill yielded unexpected key shape");
                continue;
            };
            if let Some(bytes) = self.workers.get(&key) {
                self.apply_incoming(worker_id, &bytes);
            }
        }
    }

    fn apply_incoming(&self, worker_id: &str, bytes: &[u8]) {
        match bincode::deserialize::<WorkerState>(bytes) {
            Ok(state) => self.worker_registry.on_remote_worker_state(&state),
            Err(err) => warn!(worker_id, %err, "failed to decode WorkerState"),
        }
    }

    /// Publish a worker update to the cluster. Callers pass the
    /// registry's current state; the adapter owns (de)serialisation
    /// and key formatting.
    pub fn on_worker_changed(&self, worker_id: &str, state: &WorkerState) {
        match bincode::serialize(state) {
            Ok(bytes) => self.workers.put(&format!("{PREFIX}{worker_id}"), bytes),
            Err(err) => warn!(worker_id, %err, "failed to serialize WorkerState"),
        }
    }

    /// Publish a tombstone for a worker, removing it from the CRDT.
    pub fn on_worker_removed(&self, worker_id: &str) {
        self.workers.delete(&format!("{PREFIX}{worker_id}"));
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use smg_mesh::{MergeStrategy, MeshKV};
    use tokio::time::sleep;

    use super::*;

    fn worker_namespace(mesh: &MeshKV) -> Arc<CrdtNamespace> {
        mesh.configure_crdt_prefix(PREFIX, MergeStrategy::LastWriterWins)
    }

    fn sample_state(worker_id: &str, url: &str) -> WorkerState {
        WorkerState {
            worker_id: worker_id.into(),
            model_id: "llama-3".into(),
            url: url.into(),
            health: true,
            load: 0.25,
            version: 1,
            spec: vec![],
        }
    }

    #[tokio::test]
    async fn on_worker_changed_writes_decodable_state() {
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = WorkerSyncAdapter::new(ns.clone(), registry);

        let state = sample_state("w1", "http://worker-a:8080");
        adapter.on_worker_changed("w1", &state);

        let raw = ns.get("worker:w1").expect("adapter wrote through to CRDT");
        let decoded: WorkerState = bincode::deserialize(&raw).unwrap();
        assert_eq!(decoded, state);
    }

    #[tokio::test]
    async fn on_worker_removed_tombstones_the_entry() {
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = WorkerSyncAdapter::new(ns.clone(), registry);

        let state = sample_state("w1", "http://worker-a:8080");
        adapter.on_worker_changed("w1", &state);
        assert!(ns.get("worker:w1").is_some());

        adapter.on_worker_removed("w1");
        assert!(
            ns.get("worker:w1").is_none(),
            "tombstone must hide the prior value from readers"
        );
    }

    #[tokio::test]
    async fn start_routes_remote_state_into_registry() {
        // Two adapters over one store mimic a remote node's write
        // (publisher) arriving at a local subscriber. The underlying
        // store is shared so the publisher's put fires the
        // subscriber adapter's subscription.
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);

        let registry = Arc::new(WorkerRegistry::new());
        let subscriber = WorkerSyncAdapter::new(ns.clone(), registry.clone());
        subscriber.start();

        let publisher_registry = Arc::new(WorkerRegistry::new());
        let publisher = WorkerSyncAdapter::new(ns, publisher_registry);
        publisher.on_worker_changed("w1", &sample_state("w1", "http://remote:8080"));

        // Subscription fanout is async; poll briefly.
        for _ in 0..20 {
            if registry.get_by_url("http://remote:8080").is_some() {
                return;
            }
            sleep(Duration::from_millis(10)).await;
        }
        panic!("registry did not see the remote worker");
    }

    #[tokio::test]
    async fn start_ignores_malformed_payload() {
        // A bad payload should not propagate into the registry and
        // must not kill the spawned task — a subsequent valid write
        // still lands.
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = WorkerSyncAdapter::new(ns.clone(), registry.clone());
        adapter.start();

        ns.put("worker:bogus", b"not-bincode".to_vec());
        sleep(Duration::from_millis(20)).await;
        assert!(registry.get_by_url("http://remote:8080").is_none());

        let good = sample_state("w1", "http://remote:8080");
        ns.put(
            "worker:w1",
            bincode::serialize(&good).expect("state serializes"),
        );
        for _ in 0..20 {
            if registry.get_by_url("http://remote:8080").is_some() {
                return;
            }
            sleep(Duration::from_millis(10)).await;
        }
        panic!("subscription task aborted after a bad payload");
    }

    #[tokio::test]
    async fn start_backfills_preexisting_entries() {
        // Rolling-restart scenario: the `worker:` namespace already
        // contains gossiped state before `start` runs. The adapter
        // must backfill the registry on spawn, not wait for the next
        // live event.
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);

        let seeded = sample_state("w-seeded", "http://seeded:8080");
        ns.put(
            "worker:w-seeded",
            bincode::serialize(&seeded).expect("state serializes"),
        );

        let registry = Arc::new(WorkerRegistry::new());
        let adapter = WorkerSyncAdapter::new(ns, registry.clone());
        adapter.start();

        for _ in 0..20 {
            if registry.get_by_url("http://seeded:8080").is_some() {
                return;
            }
            sleep(Duration::from_millis(10)).await;
        }
        panic!("registry did not see the pre-existing worker");
    }

    #[tokio::test]
    #[should_panic(expected = "WorkerSyncAdapter requires a namespace scoped to `worker:`")]
    async fn new_rejects_wrong_prefix() {
        let mesh = MeshKV::new("node-a".into());
        let ns = mesh.configure_crdt_prefix("policy:", MergeStrategy::LastWriterWins);
        let registry = Arc::new(WorkerRegistry::new());
        let _ = WorkerSyncAdapter::new(ns, registry);
    }
}
