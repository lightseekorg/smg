//! Mesh Gossip Protocol and Distributed State Synchronization
//!
//! This crate provides mesh networking capabilities for distributed cluster state management:
//! - Gossip protocol for node discovery and failure detection
//! - CRDT-based state synchronization across cluster nodes
//! - Consistent hashing for request routing
//! - Partition detection and recovery

mod consistent_hash;
mod controller;
mod crdt;
mod flow_control;
mod incremental;
mod metrics;
mod mtls;
mod node_state_machine;
mod partition;
mod ping_server;
mod rate_limit_window;
mod service;
mod stores;
mod sync;
mod topology;
mod tree_ops;

#[cfg(test)]
mod test_utils;

// Re-export commonly used types
pub use crdt::SKey;
pub use metrics::init_mesh_metrics;
pub use partition::PartitionDetector;
pub use rate_limit_window::RateLimitWindow;
pub use service::{gossip, ClusterState, MeshServerBuilder, MeshServerConfig, MeshServerHandler};
pub use stores::{
    AppState, MembershipState, RateLimitConfig, StateStores, WorkerState,
    GLOBAL_RATE_LIMIT_COUNTER_KEY, GLOBAL_RATE_LIMIT_KEY,
};
pub use sync::{MeshSyncManager, OptionalMeshSyncManager};
pub use tree_ops::{TreeInsertOp, TreeOperation, TreeRemoveOp, TreeState};
