//! Composition root for the gateway's mesh sync adapters.
//!
//! [`MeshAdapters::start`] is the single call server startup makes to bring
//! the CRDT bridge online: it registers each adapter's namespace with the
//! right merge strategy, constructs the adapters against the gateway's
//! registries, and starts their inbound sync loops. It must run before the
//! mesh server's gossip starts so every remote op merges through its
//! registered engine.

use std::sync::Arc;

use smg_mesh::{MergeStrategy, MeshKV};

use super::adapters::{RateLimitSyncAdapter, WorkerSyncAdapter};
use crate::worker::WorkerRegistry;

/// Owns the started mesh sync adapters. Mesh on means every adapter here is
/// constructed, its namespace registered, and its inbound loop running —
/// mesh off is represented by the absence of the whole struct.
pub struct MeshAdapters {
    worker: Arc<WorkerSyncAdapter>,
    rate_limit: Arc<RateLimitSyncAdapter>,
}

impl std::fmt::Debug for MeshAdapters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshAdapters")
            .field("worker", &self.worker)
            .field("rate_limit", &self.rate_limit)
            .finish()
    }
}

impl MeshAdapters {
    /// Register the `worker:` (last-writer-wins) and `rl:` (epoch-max-wins)
    /// CRDT namespaces, construct the adapters, and start their inbound sync
    /// loops. One call because the adapters' `start` methods are not
    /// idempotent (each call spawns another subscription task).
    ///
    /// MUST be called before `MeshServer::start` spawns gossip: a remote op
    /// arriving for an unregistered prefix would merge through the default
    /// last-writer-wins engine with the wrong semantics.
    ///
    /// # Panics
    ///
    /// Panics if either prefix is already configured (double call) or if
    /// `node_name` is empty or contains `':'` (the rate-limit shard-key
    /// separator).
    pub fn start(
        mesh_kv: &MeshKV,
        node_name: String,
        worker_registry: Arc<WorkerRegistry>,
    ) -> Arc<Self> {
        let worker_ns = mesh_kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
        let rl_ns = mesh_kv.configure_crdt_prefix("rl:", MergeStrategy::EpochMaxWins);
        let worker = WorkerSyncAdapter::new(worker_ns, worker_registry);
        let rate_limit = RateLimitSyncAdapter::new(rl_ns, node_name);
        worker.start();
        rate_limit.start();
        Arc::new(Self { worker, rate_limit })
    }

    /// Worker sync adapter — outbound publish seam and tests.
    pub fn worker(&self) -> &Arc<WorkerSyncAdapter> {
        &self.worker
    }

    /// Rate-limit sync adapter — counter sync and aggregate reads for the
    /// distributed rate-limit middleware.
    pub fn rate_limit(&self) -> &Arc<RateLimitSyncAdapter> {
        &self.rate_limit
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use smg_mesh::WorkerState;
    use tokio::time::sleep;

    use super::*;

    fn started(mesh: &MeshKV) -> Arc<MeshAdapters> {
        MeshAdapters::start(mesh, "node-a".into(), Arc::new(WorkerRegistry::new()))
    }

    #[tokio::test]
    async fn start_wires_worker_inbound_end_to_end() {
        let mesh = MeshKV::new("node-a".into());
        let registry = Arc::new(WorkerRegistry::new());
        let adapters = MeshAdapters::start(&mesh, "node-a".into(), registry.clone());

        // A put through the adapter echoes back through the namespace
        // subscription, exercising the registered prefix and the live
        // inbound loop end to end.
        let state = WorkerState {
            worker_id: "w1".into(),
            model_id: "llama-3".into(),
            url: "http://remote:8080".into(),
            health: true,
            load: 0.0,
            version: 1,
            spec: vec![],
        };
        adapters.worker().on_worker_changed("w1", &state);

        for _ in 0..20 {
            if registry.get_by_url("http://remote:8080").is_some() {
                return;
            }
            sleep(Duration::from_millis(10)).await;
        }
        panic!("inbound worker sync loop is not running");
    }

    #[tokio::test]
    async fn rl_namespace_uses_epoch_max_wins() {
        let mesh = MeshKV::new("node-a".into());
        let adapters = started(&mesh);

        // Under EpochMaxWins a lower-epoch write cannot rewind the shard;
        // under a mis-registered LWW engine the later write would win and
        // the aggregate would read 100.
        adapters.rate_limit().sync_counter("global", 2, 5);
        adapters.rate_limit().sync_counter("global", 1, 100);
        assert_eq!(adapters.rate_limit().get_aggregate("global"), 5);
    }

    #[tokio::test]
    #[should_panic(expected = "already configured")]
    async fn start_panics_on_second_call() {
        let mesh = MeshKV::new("node-a".into());
        let _adapters = started(&mesh);
        let _again = started(&mesh);
    }

    #[tokio::test]
    #[should_panic(expected = "must not contain ':'")]
    async fn start_panics_on_colon_node_name() {
        let mesh = MeshKV::new("node-a".into());
        let _ = MeshAdapters::start(&mesh, "node:a".into(), Arc::new(WorkerRegistry::new()));
    }
}
