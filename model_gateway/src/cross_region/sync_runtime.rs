//! Cross-region sync runtime bundle owned by the gateway boot path.
//!
//! Holds the long-lived components of the cross-region sync plane:
//! - the producer orchestrator (with its in-memory log + materialized state),
//! - the producer task handles (readiness/health/load/latency periodic loops),
//! - and (when wired) the pull-server listener handle plus the pull-client
//!   orchestrator (both aborts-on-drop).
//!
//! Constructed once in `server::startup` when `cross_region.enabled` is true,
//! stashed on `AppState` so the producer + pull tasks stay alive for the
//! lifetime of the server. Dropping the bundle aborts every spawned task.

use std::sync::Arc;

use super::{
    adapters::{CrossRegionProducers, ProducerCadences, ProducerHandles},
    CrossRegionContext, CrossRegionResult, CrossRegionSyncService, PullClientOrchestrator,
    RegionPeerRegistry,
};
use crate::worker::WorkerRegistry;

/// Cross-region sync plane handles owned by the gateway for its lifetime.
///
/// The fields are private so consumers go through the typed accessors
/// (`sync()`, `producers()`, `peers()`); this keeps the boot path the only
/// place that constructs or replaces the bundle.
#[derive(Debug)]
pub struct CrossRegionSyncRuntime {
    producers: CrossRegionProducers,
    /// Configured peer registry. Cloned by the pull-server listener for its
    /// peer-identity allowlist and by the pull-client orchestrator to spawn
    /// one pull task per configured sync target.
    peers: RegionPeerRegistry,
    /// Producer task handles. Held to keep the spawned tasks alive — drop
    /// aborts them via `ProducerHandles::Drop`.
    _producer_handles: ProducerHandles,
    /// Pull-client orchestrator. `None` when the sync plane is disabled or
    /// no outbound pull tasks were spawned. Drop aborts the per-peer tasks.
    _pull_client: Option<PullClientOrchestrator>,
}

impl CrossRegionSyncRuntime {
    /// Build the producer orchestrator from runtime config and spawn its
    /// periodic and event-driven tasks. Retention windows come from
    /// `context.config.sync_retention()` so the operator-facing CLI knobs
    /// (`--cross-region-sync-plane-tombstone-retention-seconds` /
    /// `--cross-region-sync-plane-dead-replica-retention-seconds`) take
    /// effect end-to-end.
    pub fn start(
        context: &CrossRegionContext,
        worker_registry: Arc<WorkerRegistry>,
    ) -> CrossRegionResult<Self> {
        let producers = CrossRegionProducers::new_with_retention(
            context.config.region_id.clone(),
            context.config.server_name.clone(),
            context.config.sync_retention(),
        )?;
        let handles = producers.start(worker_registry, ProducerCadences::default());

        Ok(Self {
            producers,
            peers: context.peers.clone(),
            _producer_handles: handles,
            _pull_client: None,
        })
    }

    /// Attach a pull-client orchestrator built by the boot path. Takes `self`
    /// by value because the runtime is wrapped in `Arc` only after this step
    /// completes.
    pub fn with_pull_client(mut self, orchestrator: PullClientOrchestrator) -> Self {
        self._pull_client = Some(orchestrator);
        self
    }

    /// The shared sync service handle. Read consumers (candidate ranking,
    /// `/get_loads` projection) clone this; the pull server and pull client
    /// hold their own clones too.
    pub fn sync(&self) -> Arc<CrossRegionSyncService> {
        self.producers.sync.clone()
    }

    /// The full producer bundle, for adapters that publish on-demand (e.g.
    /// the request path recording a client-latency observation).
    pub fn producers(&self) -> &CrossRegionProducers {
        &self.producers
    }

    /// The configured peer registry. Cloned by the pull-server boot block
    /// (for the peer-identity allowlist) and the pull-client boot block
    /// (for the sync-target iteration).
    pub fn peers(&self) -> &RegionPeerRegistry {
        &self.peers
    }
}
