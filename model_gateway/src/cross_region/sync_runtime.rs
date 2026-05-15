//! Cross-region sync runtime bundle owned by the gateway boot path.
//!
//! Holds the long-lived components of the mesh-backed cross-region sync plane:
//! - the producer orchestrator (one [`CrossRegionSyncService`] feeding a mesh
//!   broadcast [`StreamNamespace`]),
//! - the producer task handles (readiness/health/load/latency periodic loops),
//! - the subscriber task handle that decodes inbound `cross_region:` stream
//!   entries and drives [`apply_envelope_to_state`],
//! - a periodic GC task that evicts materialized entries older than the
//!   consumer-side freshness window (no-tombstone sweep — see [`build_gc_loop`]).
//!
//! Constructed once in `server::startup` when `cross_region.enabled` and
//! `cross_region.sync_plane.enabled` are both true and a mesh server is
//! running. Stashed on `AppState` so the producer + subscriber tasks stay
//! alive for the lifetime of the server. Dropping the bundle aborts every
//! spawned task and unregisters the drain callback.

use std::{sync::Arc, time::Duration};

use parking_lot::RwLock;
use smg_mesh::{MeshKV, StreamConfig, StreamNamespace, StreamRouting};
use tokio::task::JoinHandle;

use super::{
    adapters::{CrossRegionProducers, ProducerCadences, ProducerHandles},
    decode_envelope, CrossRegionContext, CrossRegionResult, CrossRegionState,
    CrossRegionSyncService, RegionPeerRegistry, CROSS_REGION_NAMESPACE_PREFIX,
};
use crate::worker::WorkerRegistry;

/// Per-round byte cap for the broadcast stream's targeted buffer. Broadcast
/// streams use the drain-callback path (no internal buffer), so this is
/// effectively unused, but the struct field is required by
/// [`StreamConfig`]. Set high enough to be a non-concern.
const CROSS_REGION_STREAM_BUFFER_BYTES: usize = 16 * 1024 * 1024;

/// Multiplier applied to `signal_stale_after_seconds` to derive the GC
/// eviction age. Entries older than `signal_stale_after_seconds × this` are
/// considered abandoned (the producer has stopped re-emitting and the
/// projection has already filtered them out of cross-region rankings) and
/// dropped from materialized state to bound memory growth on worker churn.
const GC_AGE_MULTIPLIER: u64 = 4;

/// Cadence for the materialized-state GC sweep. Cheap operation (HashMap
/// retain pass over a few-thousand-entry working set) so we run it
/// frequently enough to keep stale entries from accumulating between
/// long-lived reconcile cadences.
const GC_INTERVAL: Duration = Duration::from_secs(30);

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
    /// Materialized-state GC task. Periodically evicts entries older than
    /// the freshness window so abandoned signals don't leak memory or
    /// clutter `worker_ids` / `regions` enumerations. Drop aborts.
    _gc: GcHandle,
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
        let gc_max_age_ms = gc_max_age_ms(context.config.sync_plane.signal_stale_after_seconds);
        let gc = spawn_gc_loop(producers.sync.state(), GC_INTERVAL, gc_max_age_ms);

        Ok(Self {
            producers,
            peers: context.peers.clone(),
            _producer_handles: handles,
            _subscriber: subscriber,
            _gc: gc,
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

/// Handle wrapping the GC sweep task. Drop aborts.
#[derive(Debug)]
struct GcHandle {
    task: JoinHandle<()>,
}

impl Drop for GcHandle {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Derive the GC eviction age (ms) from the consumer's freshness window in
/// seconds. Uses [`GC_AGE_MULTIPLIER`] so we keep a margin past the
/// projection-side filter, which avoids evicting entries that the producer
/// is about to re-emit on its reconcile tick.
fn gc_max_age_ms(signal_stale_after_seconds: u64) -> i64 {
    let millis = signal_stale_after_seconds
        .saturating_mul(GC_AGE_MULTIPLIER)
        .saturating_mul(1_000);
    i64::try_from(millis).unwrap_or(i64::MAX)
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

#[expect(
    clippy::disallowed_methods,
    reason = "GC task is bounded by CrossRegionSyncRuntime which aborts on drop"
)]
fn spawn_gc_loop(
    state: Arc<RwLock<CrossRegionState>>,
    interval: Duration,
    max_age_ms: i64,
) -> GcHandle {
    let task = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            ticker.tick().await;
            let dropped = state.write().gc_stale(now_ms(), max_age_ms);
            if dropped > 0 {
                tracing::debug!(
                    dropped,
                    max_age_ms,
                    "cross-region GC swept stale materialized entries"
                );
            }
        }
    });
    GcHandle { task }
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
