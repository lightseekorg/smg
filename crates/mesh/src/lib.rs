//! Mesh Gossip Protocol and Distributed State Synchronization
//!
//! This crate provides mesh networking capabilities for distributed cluster state management:
//! - Gossip protocol for node discovery and failure detection
//! - CRDT-based state synchronization across cluster nodes
//! - Consistent hashing for request routing
//! - Partition detection and recovery

mod chunk_assembler;
mod chunking;
mod crdt_kv;
mod flow_control;
mod gossip_controller;
mod gossip_service;
mod hash;
pub mod kv;
mod metrics;
mod mtls;
mod partition;
mod readiness_state_machine;
mod service;
mod sync_stream_messages;
mod topology;
mod types;

// Internal tests module with full access to private types
#[cfg(test)]
mod tests;

// Re-export commonly used types
// v2 API
pub use chunking::MAX_STREAM_CHUNK_BYTES;
pub use crdt_kv::{
    decode as decode_epoch_count, encode as encode_epoch_count, merge as merge_epoch_max_wins,
    CrdtOrMap, EpochCount, MergeStrategy, EPOCH_MAX_WINS_ENCODED_LEN,
};
pub use hash::{hash_node_path, hash_token_path, GLOBAL_EVICTION_HASH};
pub use kv::{
    CrdtNamespace, DrainHandle, MeshKV, StreamConfig, StreamDrainFn, StreamNamespace,
    StreamRouting, Subscription,
};
pub use metrics::init_mesh_metrics;
pub use mtls::{MTLSConfig, MTLSManager, OptionalMTLSManager};
pub use partition::PartitionDetector;
pub use service::{gossip, ClusterState, MeshServerBuilder, MeshServerConfig, MeshServerHandler};
pub use types::{
    MembershipState, RateLimitConfig, WorkerState, GLOBAL_RATE_LIMIT_COUNTER_KEY,
    GLOBAL_RATE_LIMIT_KEY,
};
