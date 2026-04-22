//! CRDT- and stream-namespace adapters that bridge `MeshKV` to
//! gateway-local state. Each adapter owns one prefix, serialises
//! domain types into the shared merge format, and routes remote
//! updates into the corresponding registry or cache.

pub mod worker_sync;

pub use worker_sync::WorkerSyncAdapter;
