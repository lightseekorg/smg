//! Mesh Gossip Protocol and Distributed State Synchronization
//!
//! This crate provides mesh networking capabilities for distributed cluster state management:
//! - Gossip protocol for node discovery and failure detection
//! - CRDT-based state synchronization across cluster nodes
//! - Consistent hashing for request routing
//! - Partition detection and recovery

pub mod consistent_hash;
pub mod controller;
pub mod crdt;
pub mod flow_control;
pub mod incremental;
pub mod metrics;
pub mod mtls;
pub mod node_state_machine;
pub mod partition;
mod ping_server;
pub mod rate_limit_window;
pub mod service;
pub mod stores;
pub mod sync;
pub mod topology;
pub mod tree_ops;

#[cfg(test)]
mod test_utils;

// Re-export commonly used types
pub use crdt::{CRDTMap, CRDTPNCounter, SKey, SyncCRDTMap, SyncPNCounter};
pub use service::{
    broadcast_node_states, gossip, try_ping, ClusterState, MeshServerConfig, MeshServerHandler,
};
pub use stores::{
    tree_state_key, AppStore, MembershipState, MembershipStore, PolicyState, PolicyStore,
    RateLimitConfig, RateLimitStore, StateStores, StoreType, WorkerState, WorkerStore,
    GLOBAL_RATE_LIMIT_COUNTER_KEY, GLOBAL_RATE_LIMIT_KEY,
};
pub use sync::{MeshSyncManager, OptionalMeshSyncManager};
pub use tree_ops::{TreeInsertOp, TreeOperation, TreeRemoveOp, TreeState};
