//! Cross-region sync runtime bundle owned by the gateway boot path.
//!
//! Holds the long-lived components of the mesh-backed cross-region sync plane:
//! - the producer orchestrator (one [`CrossRegionSyncService`] feeding mesh's
//!   `CrdtNamespace`),
//! - the producer task handles (readiness/health/load/latency periodic loops),
//! - the subscriber task handle that decodes inbound `cross_region:` events
//!   and drives [`apply_envelope_to_state`].
//!
//! Constructed once in `server::startup` when `cross_region.enabled` is true,
//! stashed on `AppState` so the producer + subscriber tasks stay alive for
//! the lifetime of the server. Dropping the bundle aborts every spawned task.

use std::sync::Arc;

use smg_mesh::{CrdtNamespace, MergeStrategy, MeshKV};
use tokio::task::JoinHandle;

use super::{
    adapters::{CrossRegionProducers, ProducerCadences, ProducerHandles},
    decode_envelope, mesh_path, CrossRegionContext, CrossRegionResult, CrossRegionSyncService,
    RegionPeerRegistry, SignalKey, CROSS_REGION_NAMESPACE_PREFIX,
};
use crate::worker::WorkerRegistry;

/// Cross-region sync plane handles owned by the gateway for its lifetime.
///
/// The fields are private so consumers go through the typed accessors
/// (`sync()`, `producers()`, `peers()`).
#[derive(Debug)]
pub struct CrossRegionSyncRuntime {
    producers: CrossRegionProducers,
    /// Configured peer registry. Currently unused by the runtime itself
    /// (mesh handles peer discovery) but exposed via `peers()` for boot
    /// wiring that wants to enforce the cross-region allowlist.
    peers: RegionPeerRegistry,
    /// Producer task handles. Held to keep the spawned tasks alive — drop
    /// aborts them via `ProducerHandles::Drop`.
    _producer_handles: ProducerHandles,
    /// Subscriber task. Reads `cross_region:` change events from the mesh
    /// namespace and applies them to `CrossRegionState`. Drop aborts.
    _subscriber: SubscriberHandle,
}

impl CrossRegionSyncRuntime {
    /// Build the producer orchestrator over the supplied mesh `CrdtNamespace`
    /// and spawn its periodic + event-driven tasks. Also spawns the
    /// subscriber task that drives inbound `cross_region:` events into
    /// `CrossRegionState`.
    ///
    /// `namespace` must be the namespace registered for
    /// [`CROSS_REGION_NAMESPACE_PREFIX`] on the gateway's mesh KV — i.e.
    /// `mesh_kv.configure_crdt_prefix(CROSS_REGION_NAMESPACE_PREFIX,
    /// MergeStrategy::LastWriterWins)`. The boot path is the only place
    /// that owns this registration.
    pub fn start(
        context: &CrossRegionContext,
        namespace: Arc<CrdtNamespace>,
        worker_registry: Arc<WorkerRegistry>,
    ) -> CrossRegionResult<Self> {
        let producers = CrossRegionProducers::new(
            context.config.region_id.clone(),
            context.config.server_name.clone(),
            namespace.clone(),
        )?;
        let handles = producers.start(worker_registry, ProducerCadences::default());
        let subscriber = spawn_subscriber(producers.sync.clone());

        Ok(Self {
            producers,
            peers: context.peers.clone(),
            _producer_handles: handles,
            _subscriber: subscriber,
        })
    }

    /// Convenience: register the `cross_region:` namespace on the supplied
    /// `MeshKV`, then call [`start`].
    pub fn start_with_mesh_kv(
        context: &CrossRegionContext,
        mesh_kv: &Arc<MeshKV>,
        worker_registry: Arc<WorkerRegistry>,
    ) -> CrossRegionResult<Self> {
        let namespace = mesh_kv
            .configure_crdt_prefix(CROSS_REGION_NAMESPACE_PREFIX, MergeStrategy::LastWriterWins);
        Self::start(context, namespace, worker_registry)
    }

    /// The shared sync service handle. Read consumers (candidate ranking,
    /// `/get_loads` projection) clone this.
    pub fn sync(&self) -> Arc<CrossRegionSyncService> {
        self.producers.sync.clone()
    }

    /// The full producer bundle, for adapters that publish on-demand (e.g.
    /// the request path recording a client-latency observation).
    pub fn producers(&self) -> &CrossRegionProducers {
        &self.producers
    }

    /// The configured peer registry. Exposed for diagnostics; the sync
    /// plane itself relies on mesh's transport-level mTLS allowlist for
    /// peer authorization.
    pub fn peers(&self) -> &RegionPeerRegistry {
        &self.peers
    }
}

/// Handle wrapping the subscriber task. Drop aborts.
#[derive(Debug)]
struct SubscriberHandle {
    task: JoinHandle<()>,
}

impl Drop for SubscriberHandle {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[expect(
    clippy::disallowed_methods,
    reason = "subscriber task is bounded by CrossRegionSyncRuntime which aborts on drop"
)]
fn spawn_subscriber(sync: Arc<CrossRegionSyncService>) -> SubscriberHandle {
    let namespace = sync.namespace();
    let state = sync.state();
    // Subscribe to every event under the cross_region: namespace.
    let mut subscription = namespace.subscribe("");
    let task = tokio::spawn(async move {
        while let Some((key, value)) = subscription.receiver.recv().await {
            // Mesh keys arrive with the namespace prefix; strip it before
            // parsing into a `SignalKey`.
            let signal_path = key
                .strip_prefix(CROSS_REGION_NAMESPACE_PREFIX)
                .unwrap_or(key.as_str());
            match value {
                Some(chunks) => match decode_envelope(&chunks) {
                    Ok(envelope) => {
                        crate::cross_region::apply_envelope_to_state(&mut state.write(), &envelope);
                    }
                    Err(error) => {
                        tracing::warn!(
                            key = %signal_path,
                            error = %error,
                            "dropping malformed cross-region envelope"
                        );
                    }
                },
                None => match SignalKey::from_path(signal_path) {
                    Some(signal_key) => {
                        state.write().remove_key(&signal_key);
                    }
                    None => {
                        tracing::warn!(
                            key = %signal_path,
                            "received tombstone for unrecognized cross-region key"
                        );
                    }
                },
            }
            // Suppress unused-variable warning on _ in the rare future case
            // where decode_envelope is extended to need the key.
            let _ = mesh_path; // ensure import is kept for sibling helpers
        }
    });
    SubscriberHandle { task }
}
