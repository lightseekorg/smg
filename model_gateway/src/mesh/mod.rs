//! Gateway-side glue for the v2 mesh: adapters that bridge the
//! typed `MeshKV` namespaces to local registries, plus bootstrap
//! and shutdown wiring added in later steps.

use std::sync::Arc;

use smg_mesh::{
    gossip::NodeStatus, ClusterState, MergeStrategy, MeshKV, StreamConfig, StreamRouting,
};

use crate::{policies::PolicyRegistry, worker::WorkerRegistry};

pub mod adapters;

pub use adapters::{
    RateLimitSyncAdapter, TreeDelta, TreeSyncAdapter, TreeSyncConfig, WorkerSyncAdapter,
};

const WORKER_PREFIX: &str = "worker:";
const RATE_LIMIT_PREFIX: &str = "rl:";
const TENANT_DELTA_PREFIX: &str = "td:";
const REPAIR_REQUEST_PREFIX: &str = "tree:req:";
const REPAIR_PAGE_PREFIX: &str = "tree:page:";

const TENANT_DELTA_BUFFER_BYTES: usize = 1024 * 1024;
const REPAIR_REQUEST_BUFFER_BYTES: usize = 1024 * 1024;
const REPAIR_PAGE_BUFFER_BYTES: usize = 64 * 1024 * 1024;

/// Gateway-side v2 mesh adapters, kept alive for the process lifetime.
#[derive(Debug)]
pub struct MeshAdapters {
    pub worker_sync: Arc<WorkerSyncAdapter>,
    pub tree_sync: Arc<TreeSyncAdapter>,
    pub rate_limit_sync: Arc<RateLimitSyncAdapter>,
}

impl MeshAdapters {
    pub fn new(
        mesh_kv: Arc<MeshKV>,
        node_name: String,
        cluster_state: ClusterState,
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Self {
        Self::with_tree_sync_config(
            mesh_kv,
            node_name,
            cluster_state,
            worker_registry,
            policy_registry,
            TreeSyncConfig::default(),
        )
    }

    pub fn with_tree_sync_config(
        mesh_kv: Arc<MeshKV>,
        node_name: String,
        cluster_state: ClusterState,
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        tree_sync_config: TreeSyncConfig,
    ) -> Self {
        let workers = mesh_kv.configure_crdt_prefix(WORKER_PREFIX, MergeStrategy::LastWriterWins);
        let rate_limits =
            mesh_kv.configure_crdt_prefix(RATE_LIMIT_PREFIX, MergeStrategy::EpochMaxWins);
        let tenant_deltas = mesh_kv.configure_stream_prefix(
            TENANT_DELTA_PREFIX,
            StreamConfig {
                max_buffer_bytes: TENANT_DELTA_BUFFER_BYTES,
                routing: StreamRouting::Broadcast,
            },
        );
        let repair_requests = mesh_kv.configure_stream_prefix(
            REPAIR_REQUEST_PREFIX,
            StreamConfig {
                max_buffer_bytes: REPAIR_REQUEST_BUFFER_BYTES,
                routing: StreamRouting::Targeted,
            },
        );
        let repair_pages = mesh_kv.configure_stream_prefix(
            REPAIR_PAGE_PREFIX,
            StreamConfig {
                max_buffer_bytes: REPAIR_PAGE_BUFFER_BYTES,
                routing: StreamRouting::Targeted,
            },
        );

        let peers = Arc::new(ClusterPeerList {
            state: cluster_state,
            self_name: node_name.clone(),
        });

        Self {
            worker_sync: WorkerSyncAdapter::new(workers, worker_registry),
            tree_sync: TreeSyncAdapter::with_config(
                tenant_deltas,
                repair_requests,
                repair_pages,
                policy_registry,
                peers,
                node_name.clone(),
                tree_sync_config,
            ),
            rate_limit_sync: RateLimitSyncAdapter::new(rate_limits, node_name),
        }
    }

    pub fn start(&self) {
        self.worker_sync.start();
        self.tree_sync.start();
        self.rate_limit_sync.start();
    }
}

#[derive(Debug)]
struct ClusterPeerList {
    state: ClusterState,
    self_name: String,
}

impl adapters::PeerList for ClusterPeerList {
    fn alive_peers(&self) -> Vec<String> {
        self.state
            .read()
            .values()
            .filter(|node| node.name != self.self_name)
            .filter(|node| node.status == NodeStatus::Alive as i32)
            .map(|node| node.name.clone())
            .collect()
    }
}
