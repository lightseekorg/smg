//! Gateway-side glue for the v2 mesh: adapters that bridge the
//! typed `MeshKV` namespaces to local registries, plus bootstrap
//! and shutdown wiring added in later steps.

pub mod adapters;
pub mod global_rate_limit;
pub mod wiring;

pub use adapters::{RateLimitSyncAdapter, TreeDelta, TreeSyncAdapter, WorkerSyncAdapter};
pub use global_rate_limit::{GlobalRateLimiter, RateLimitConfig, RATE_LIMIT_CONFIG_KEY};
pub use wiring::MeshAdapters;
