//! Local-producer adapters for the four cross-region signals.
//!
//! Each adapter owns an `Arc<CrossRegionSyncService>` and the relevant data
//! source (worker registry, latency observer, …). Adapters either subscribe
//! to in-process events or run a periodic reconcile tick; in both cases the
//! end action is `CrossRegionSyncService::publish_signal` /
//! `CrossRegionSyncService::remove_signal`. Publication stages an envelope
//! in the sync service's outbox; mesh's drain callback ships staged entries
//! to peers on the next gossip round, and the subscriber wired in
//! [`crate::cross_region::sync_runtime`] applies inbound deliveries to
//! local `CrossRegionState`.

mod client_latency;
mod orchestrator;
mod region_readiness;
mod worker_health;
mod worker_load;

#[cfg(test)]
pub(super) mod test_support;

pub use client_latency::ClientLatencyAdapter;
pub use orchestrator::{CrossRegionProducers, ProducerCadences, ProducerHandles};
pub use region_readiness::RegionReadinessAdapter;
pub use worker_health::WorkerHealthAdapter;
pub use worker_load::WorkerLoadAdapter;
