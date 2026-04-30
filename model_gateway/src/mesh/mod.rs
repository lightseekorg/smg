//! Gateway-side glue for the v2 mesh: adapters that bridge the
//! typed `MeshKV` namespaces to local registries, plus bootstrap
//! and shutdown wiring added in later steps.

pub mod adapters;

pub use adapters::{RateLimitSyncAdapter, TreeDelta, TreeKind, TreeSyncAdapter, WorkerSyncAdapter};
