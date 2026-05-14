//! Mesh Gossip Protocol and Distributed State Synchronization
//!
//! This crate provides mesh networking capabilities for distributed cluster state management:
//! - Gossip protocol for node discovery and failure detection
//! - CRDT-based state synchronization across cluster nodes
//! - Partition detection and recovery

mod chunk_assembler;
mod chunking;
mod collector;
mod consistent_hash;
mod controller;
mod crdt_kv;
mod flow_control;
mod hash;
pub mod kv;
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
mod types;

// Internal tests module with full access to private types
#[cfg(test)]
mod tests;

// Re-export commonly used types
pub use chunking::MAX_STREAM_CHUNK_BYTES;
pub use crdt_kv::{
    decode as decode_epoch_count, encode as encode_epoch_count, merge as merge_epoch_max_wins,
    CrdtOrMap, EpochCount, OperationLog, EPOCH_MAX_WINS_ENCODED_LEN,
};
pub use hash::{hash_node_path, hash_token_path, GLOBAL_EVICTION_HASH};
pub use kv::{
    CrdtNamespace, DrainHandle, MergeStrategy, MeshKV, StreamConfig, StreamDrainFn,
    StreamNamespace, StreamRouting, Subscription,
};
pub use metrics::init_mesh_metrics;
pub use mtls::{MTLSConfig, MTLSManager, OptionalMTLSManager};
pub use partition::PartitionDetector;
pub use service::{gossip, ClusterState, MeshServerBuilder, MeshServerConfig, MeshServerHandler};
pub use types::WorkerState;
