//! Gateway-side bridges between the v2 mesh KV and local registries.
//!
//! Each adapter scopes to one CRDT or stream prefix, serialises
//! state into the shared merge format, and routes remote updates
//! into the gateway-local component (worker registry, policy
//! registry, rate limiter, tree cache). Adapters are built against
//! the raw `smg_mesh` primitives so the gateway can migrate one
//! domain at a time without moving the v1 `MeshSyncManager` glue.

pub mod worker_sync;

pub use worker_sync::WorkerSyncAdapter;
