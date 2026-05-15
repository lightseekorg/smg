//! Cross-region sync runtime bundle owned by the gateway boot path.
//!
//! Holds the long-lived components of the mesh-backed cross-region sync plane:
//! - the producer orchestrator (one [`CrossRegionSyncService`] feeding a mesh
//!   broadcast [`StreamNamespace`]),
//! - the producer task handles (readiness/health/load/latency periodic loops),
//! - the subscriber task handle that decodes inbound `cross_region:` stream
//!   entries and drives [`apply_envelope_to_state`].
//!
//! Constructed once in `server::startup` when `cross_region.enabled` and
//! `cross_region.sync_plane.enabled` are both true and a mesh server is
//! running. Stashed on `AppState` so the producer + subscriber tasks stay
//! alive for the lifetime of the server. Dropping the bundle aborts every
//! spawned task and unregisters the drain callback.

use std::sync::Arc;

use smg_mesh::{MeshKV, StreamConfig, StreamNamespace, StreamRouting};
use tokio::task::JoinHandle;

use super::{
    adapters::{CrossRegionProducers, ProducerCadences, ProducerHandles},
    decode_envelope, CrossRegionContext, CrossRegionResult, CrossRegionSyncService,
    RegionPeerRegistry, CROSS_REGION_NAMESPACE_PREFIX,
};
use crate::worker::WorkerRegistry;

/// Per-round byte cap for the broadcast stream's targeted buffer. Broadcast
/// streams use the drain-callback path (no internal buffer), so this is
/// effectively unused, but the struct field is required by
/// [`StreamConfig`]. Set high enough to be a non-concern.
const CROSS_REGION_STREAM_BUFFER_BYTES: usize = 16 * 1024 * 1024;

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
    /// Subscriber task. Reads `cross_region:` stream entries from the mesh
    /// namespace and applies them to `CrossRegionState`. Drop aborts.
    _subscriber: SubscriberHandle,
}

impl CrossRegionSyncRuntime {
    /// Build the producer orchestrator over the supplied mesh
    /// [`StreamNamespace`] and spawn its periodic + event-driven tasks. Also
    /// spawns the subscriber task that drives inbound `cross_region:` entries
    /// into `CrossRegionState`.
    ///
    /// `namespace` must be the broadcast stream namespace registered for
    /// [`CROSS_REGION_NAMESPACE_PREFIX`] on the gateway's mesh KV. The boot
    /// path is the only place that owns this registration.
    pub fn start(
        context: &CrossRegionContext,
        namespace: Arc<StreamNamespace>,
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

    /// Convenience: register the `cross_region:` broadcast stream namespace
    /// on the supplied `MeshKV`, then call [`start`].
    pub fn start_with_mesh_kv(
        context: &CrossRegionContext,
        mesh_kv: &Arc<MeshKV>,
        worker_registry: Arc<WorkerRegistry>,
    ) -> CrossRegionResult<Self> {
        let namespace = mesh_kv.configure_stream_prefix(
            CROSS_REGION_NAMESPACE_PREFIX,
            StreamConfig {
                max_buffer_bytes: CROSS_REGION_STREAM_BUFFER_BYTES,
                routing: StreamRouting::Broadcast,
            },
        );
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
    // Subscribe to every entry delivered under the cross_region: namespace.
    let mut subscription = namespace.subscribe("");
    let task = tokio::spawn(async move {
        while let Some((key, value)) = subscription.receiver.recv().await {
            // Stream entries always arrive as `(key, Some(chunks))`. A
            // `None` payload would mean a CRDT tombstone — streams don't
            // emit those, so the arm exists only as a defensive guard.
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
                None => {
                    tracing::warn!(
                        key = %signal_path,
                        "unexpected tombstone delivery on cross-region stream namespace"
                    );
                }
            }
        }
    });
    SubscriberHandle { task }
}
