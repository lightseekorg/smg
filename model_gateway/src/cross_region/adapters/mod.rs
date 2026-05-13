//! Local-producer adapters for the four cross-region signals.
//!
//! Each adapter owns an `Arc<CrossRegionSyncService>` and the relevant data
//! source (worker registry, latency observer, …). Adapters either subscribe
//! to in-process events or run a periodic reconcile tick; in both cases the
//! end action is `CrossRegionSyncService::publish_signal` /
//! `CrossRegionSyncService::remove_signal`.
//!
//! The pull-protocol HTTP endpoint that serves the producer log is **not**
//! built in this module; that's Phase 4 (the wire side of peer-to-peer).
//! For now adapters just feed the in-memory log.

mod client_latency;
mod orchestrator;
mod region_readiness;
mod worker_health;
mod worker_load;

pub use client_latency::ClientLatencyAdapter;
pub use orchestrator::{CrossRegionProducers, ProducerCadences, ProducerHandles};
pub use region_readiness::RegionReadinessAdapter;
pub use worker_health::WorkerHealthAdapter;
pub use worker_load::WorkerLoadAdapter;
