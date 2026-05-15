//! Cross-region signal sync service backed by mesh broadcast streams.
//!
//! Producers (the four adapters in `crate::cross_region::adapters`) call
//! [`CrossRegionSyncService::publish_signal`] / [`CrossRegionSyncService::remove_signal`].
//! Each `publish_signal` builds a typed envelope, encodes it, and stages the
//! bytes in an in-memory outbox keyed by [`SignalKey`] (latest-wins per key).
//! Once per gossip round, mesh invokes a drain callback we registered on the
//! shared [`StreamNamespace`]; the callback hands the pending entries to mesh
//! which fans them to every connected peer. Peers receive the bytes via the
//! same `Subscription` channel a local subscriber uses; the cross-region
//! runtime task decodes each delivery and applies it to
//! [`CrossRegionState`].
//!
//! Streams are documented as "ephemeral, lossy, application-regenerated
//! traffic" — at-most-once delivery, no retransmit on packet loss. The four
//! producer reconcile loops re-emit on a 5-20s cadence so any drop is
//! repaired by the next round, and the consumer-side `signal_stale_after_ms`
//! freshness window in [`crate::cross_region::view::RemoteRegionView`] gates
//! a missing-too-long signal out of cross-region ranking decisions.
//!
//! Tombstones are *not* shipped over the wire — broadcast streams have no
//! delete semantics. [`CrossRegionSyncService::remove_signal`] purges the
//! outbox and scrubs local state, but peers learn that a worker is gone by
//! the producer no longer including it on subsequent reconcile ticks; the
//! peer's freshness window then drops the entry from projections.
//!
//! The wall-clock-anchored `(version, actor)` envelope-level versioning is
//! kept as application metadata. [`apply_envelope_to_state`] enforces
//! per-key monotonicity via `should_replace((version, actor))` so an
//! out-of-order delivery never moves the materialized view backwards.

use std::{collections::HashMap, sync::Arc};

use bytes::Bytes;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use smg_mesh::{DrainHandle, StreamDrainFn, StreamNamespace};

use super::{
    ClientLatencySignal, CrossRegionError, CrossRegionResult, CrossRegionState, SignalEnvelope,
    SignalKey, SignalVersion, SmgReadinessSignal, WorkerHealthSignal, WorkerLoadSignal,
};

/// Mesh namespace prefix carrying cross-region signal envelopes. Must end
/// with `:` to match `MeshKV::configure_stream_prefix` conventions.
pub const CROSS_REGION_NAMESPACE_PREFIX: &str = "cross_region:";

/// Body-erased signal payload. Adapters construct one of these per publish
/// and hand it to [`CrossRegionSyncService::publish_signal`]. Tag-discriminated
/// serde so the on-the-wire envelope stays self-describing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
// PartialEq omitted because `WorkerLoadSignal` wraps `WorkerLoadInfo`, which
// is not `PartialEq`. None of the producer-side logic compares envelopes by
// equality; tests assert on individual fields.
pub enum SignalKind {
    SmgReadiness(SmgReadinessSignal),
    WorkerHealth(WorkerHealthSignal),
    WorkerLoad(Box<WorkerLoadSignal>),
    ClientLatency(ClientLatencySignal),
}

/// Cross-region signal sync service.
///
/// Owns the materialized [`CrossRegionState`] consumer-side view, an outbox
/// of pending envelopes staged for the next gossip round, and the drain
/// handle that keeps mesh's drain callback registered for the service's
/// lifetime. `region_id` and `server_name` are stamped onto every published
/// envelope as the key's region/replica segments and the envelope's `actor`
/// field, and validated against incoming key segments.
pub struct CrossRegionSyncService {
    region_id: String,
    server_name: String,
    state: Arc<RwLock<CrossRegionState>>,
    namespace: Arc<StreamNamespace>,
    /// Encoded envelopes staged for the next mesh round. Latest-wins per
    /// [`SignalKey`]; mesh drains this destructively each round.
    outbox: Arc<RwLock<HashMap<SignalKey, Bytes>>>,
    /// Per-key monotonic counter for the envelope-level `version` field.
    /// `apply_envelope_to_state` enforces `(version, actor)` ordering on
    /// inbound peer envelopes so the materialized view never moves
    /// backwards on an out-of-order delivery.
    latest_per_key: Arc<RwLock<HashMap<SignalKey, u64>>>,
    /// Keeps the drain callback registered for the service's lifetime.
    /// Drop unregisters it from mesh's drain registry.
    _drain_handle: DrainHandle,
}

impl std::fmt::Debug for CrossRegionSyncService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossRegionSyncService")
            .field("region_id", &self.region_id)
            .field("server_name", &self.server_name)
            .field("namespace", &self.namespace.prefix())
            .finish_non_exhaustive()
    }
}

impl CrossRegionSyncService {
    /// Build a sync service rooted at this replica's identity. Validates
    /// `region_id`/`server_name` charset so `/` in either can never reach
    /// a key path segment and break parsing. Registers the drain callback
    /// with mesh as a side-effect.
    pub fn new(
        region_id: String,
        server_name: String,
        namespace: Arc<StreamNamespace>,
    ) -> CrossRegionResult<Self> {
        validate_identity_segment("region_id", &region_id)?;
        validate_identity_segment("server_name", &server_name)?;
        let outbox: Arc<RwLock<HashMap<SignalKey, Bytes>>> = Arc::new(RwLock::new(HashMap::new()));
        let drain_handle = namespace.register_drain(build_drain_callback(outbox.clone()));
        Ok(Self {
            region_id,
            server_name,
            state: Arc::new(RwLock::new(CrossRegionState::new())),
            namespace,
            outbox,
            latest_per_key: Arc::new(RwLock::new(HashMap::new())),
            _drain_handle: drain_handle,
        })
    }

    /// This replica's region.
    pub fn region_id(&self) -> &str {
        &self.region_id
    }

    /// This replica's `server_name` (stamped as every envelope's `actor`).
    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    /// Shared materialized state. Consumers (candidate calculation,
    /// `/get_loads` projection) read through this Arc. Wrapped in `RwLock`
    /// because the mesh subscriber task writes from a separate task.
    pub fn state(&self) -> Arc<RwLock<CrossRegionState>> {
        self.state.clone()
    }

    /// Shared namespace handle. The cross-region subscriber task subscribes
    /// through this to receive peer updates.
    pub fn namespace(&self) -> Arc<StreamNamespace> {
        self.namespace.clone()
    }

    /// Publish a live signal. Validates the key/body against this replica's
    /// identity, serializes the envelope, stages it in the outbox for the
    /// next mesh gossip round, and mirrors into local state so the
    /// producer's self-view is consistent immediately.
    pub fn publish_signal(
        &self,
        key: SignalKey,
        signal: SignalKind,
        stale_after_ms: u32,
    ) -> CrossRegionResult<()> {
        self.validate_key(&key)?;
        validate_body_against_key(&key, &signal)?;
        let envelope = self.build_envelope(key.clone(), Some(signal), stale_after_ms, false);
        let bytes = encode_envelope(&envelope)?;
        self.outbox.write().insert(key, bytes);
        apply_envelope_to_state(&mut self.state.write(), &envelope);
        Ok(())
    }

    /// Drop a signal from the producer's outbox and scrub it from local
    /// state. Stream namespaces have no wire-level delete; peers learn that
    /// this replica's signal is gone by the producer not re-publishing it,
    /// after which the consumer's freshness window drops the entry from
    /// projections (see [`crate::cross_region::view::RemoteRegionView`]).
    pub fn remove_signal(&self, key: SignalKey) -> CrossRegionResult<()> {
        self.validate_key(&key)?;
        self.outbox.write().remove(&key);
        // Build a tombstone envelope so the local apply path honours the
        // `(version, actor)` ordering for the self-view.
        let envelope = self.build_envelope(key, None, 0, true);
        apply_envelope_to_state(&mut self.state.write(), &envelope);
        Ok(())
    }

    /// Snapshot the outbox without draining it. Exposed for tests and
    /// diagnostics; the production code path is mesh's drain callback, which
    /// removes entries on collection.
    #[doc(hidden)]
    pub fn outbox_snapshot(&self) -> Vec<SignalEnvelope<SignalKind>> {
        let outbox = self.outbox.read();
        outbox
            .values()
            .filter_map(|bytes| serde_json::from_slice(bytes).ok())
            .collect()
    }

    // ---- internals ----

    fn validate_key(&self, key: &SignalKey) -> CrossRegionResult<()> {
        if key.region_segment() != self.region_id {
            return Err(CrossRegionError::InvalidConfig {
                reason: format!(
                    "signal key region segment {:?} does not match local region {:?}",
                    key.region_segment(),
                    self.region_id,
                ),
            });
        }
        if key.server_name_segment() != self.server_name {
            return Err(CrossRegionError::InvalidConfig {
                reason: format!(
                    "signal key server_name segment {:?} does not match local server_name {:?}",
                    key.server_name_segment(),
                    self.server_name,
                ),
            });
        }
        Ok(())
    }

    fn build_envelope(
        &self,
        key: SignalKey,
        signal: Option<SignalKind>,
        stale_after_ms: u32,
        removed: bool,
    ) -> SignalEnvelope<SignalKind> {
        let now = now_ms();
        let prev = self.latest_per_key.read().get(&key).copied().unwrap_or(0);
        let now_u64 = u64::try_from(now.max(0)).unwrap_or(0);
        let version = now_u64.max(prev.saturating_add(1));
        self.latest_per_key.write().insert(key.clone(), version);
        SignalEnvelope {
            key,
            version,
            actor: self.server_name.clone(),
            generated_at_ms: now,
            stale_after_ms,
            removed,
            signal,
        }
    }
}

/// Mesh wire-key for a signal. The stream namespace prefix is required by
/// mesh's chunking layer so receivers can route the entry to the right
/// subscriber registry.
pub fn mesh_path(key: &SignalKey) -> String {
    format!("{}{}", CROSS_REGION_NAMESPACE_PREFIX, key.as_path())
}

fn encode_envelope(envelope: &SignalEnvelope<SignalKind>) -> CrossRegionResult<Bytes> {
    serde_json::to_vec(envelope)
        .map(Bytes::from)
        .map_err(|e| CrossRegionError::InvalidConfig {
            reason: format!("failed to encode signal envelope: {e}"),
        })
}

/// Closure that drains every staged envelope into mesh's per-round batch.
/// Mesh calls this exactly once per gossip round; the returned entries are
/// chunked and fanned to every connected peer.
fn build_drain_callback(outbox: Arc<RwLock<HashMap<SignalKey, Bytes>>>) -> StreamDrainFn {
    Box::new(move || {
        let mut staged = outbox.write();
        let mut out = Vec::with_capacity(staged.len());
        for (key, bytes) in staged.drain() {
            out.push((mesh_path(&key), bytes));
        }
        out
    })
}

/// Apply a decoded envelope to the materialized state. Used by the producer
/// self-mirror path on publish and by the mesh subscriber task on incoming
/// peer updates.
///
/// The application-level `(version, actor)` check on
/// `CrossRegionState::upsert_*` / `remove_key_with_version` guarantees the
/// materialized view never moves backwards even if mesh delivers an
/// out-of-order entry (streams are at-most-once, so reorderings are rare
/// but possible).
pub fn apply_envelope_to_state(
    state: &mut CrossRegionState,
    envelope: &SignalEnvelope<SignalKind>,
) {
    let version = SignalVersion {
        version: envelope.version,
        actor: envelope.actor.clone(),
        updated_at_ms: envelope.generated_at_ms,
    };
    if envelope.removed {
        state.remove_key_with_version(&envelope.key, &version);
        return;
    }
    match envelope.signal.as_ref() {
        Some(SignalKind::SmgReadiness(s)) => state.upsert_readiness(s.clone(), version),
        Some(SignalKind::WorkerHealth(s)) => state.upsert_worker_health(s.clone(), version),
        Some(SignalKind::WorkerLoad(s)) => state.upsert_worker_load(s.as_ref().clone(), version),
        Some(SignalKind::ClientLatency(s)) => state.upsert_client_latency(s.clone(), version),
        None => {}
    }
}

/// Defense-in-depth validation on the subscriber-side decode path. A
/// well-behaved peer never violates these invariants; an attacker or
/// misbehaving peer who somehow writes into the namespace gets their
/// envelope dropped here.
pub fn validate_remote_envelope(envelope: &SignalEnvelope<SignalKind>) -> CrossRegionResult<()> {
    if envelope.actor != envelope.key.server_name_segment() {
        return Err(CrossRegionError::InvalidConfig {
            reason: format!(
                "remote envelope actor {:?} does not match key server_name {:?}",
                envelope.actor,
                envelope.key.server_name_segment(),
            ),
        });
    }
    match (envelope.removed, envelope.signal.as_ref()) {
        (true, None) => Ok(()),
        (true, Some(_)) => Err(CrossRegionError::InvalidConfig {
            reason: "removed signal envelope must not carry a signal body".to_string(),
        }),
        (false, Some(signal)) => validate_body_against_key(&envelope.key, signal),
        (false, None) => Err(CrossRegionError::InvalidConfig {
            reason: "live signal envelope must carry a signal body".to_string(),
        }),
    }
}

/// Decode a subscriber-delivered byte payload into an envelope. Mesh
/// delivers values as `Vec<Bytes>` chunks; for our envelope sizes this is
/// always a single chunk, but the multi-chunk concatenation path is kept
/// for forward compatibility.
pub fn decode_envelope(chunks: &[Bytes]) -> CrossRegionResult<SignalEnvelope<SignalKind>> {
    let envelope: SignalEnvelope<SignalKind> = if chunks.len() == 1 {
        serde_json::from_slice(&chunks[0]).map_err(|e| CrossRegionError::InvalidConfig {
            reason: format!("failed to decode signal envelope: {e}"),
        })?
    } else {
        let mut buf = Vec::with_capacity(chunks.iter().map(|c| c.len()).sum());
        for chunk in chunks {
            buf.extend_from_slice(chunk);
        }
        serde_json::from_slice(&buf).map_err(|e| CrossRegionError::InvalidConfig {
            reason: format!("failed to decode signal envelope: {e}"),
        })?
    };
    validate_remote_envelope(&envelope)?;
    Ok(envelope)
}

/// Validate that a body's region/worker/server_name fields agree with the key.
/// Adapters should construct bodies consistently, but defense-in-depth keeps
/// this from accidentally diverging.
fn validate_body_against_key(key: &SignalKey, signal: &SignalKind) -> CrossRegionResult<()> {
    let matches = match (key, signal) {
        (
            SignalKey::SmgReadiness {
                region_id,
                server_name,
            },
            SignalKind::SmgReadiness(s),
        ) => s.region_id == *region_id && s.server_name == *server_name,
        (
            SignalKey::WorkerHealth {
                region_id,
                worker_id,
                server_name,
            },
            SignalKind::WorkerHealth(s),
        ) => {
            s.region_id == *region_id && s.worker_id == *worker_id && s.server_name == *server_name
        }
        (
            SignalKey::WorkerLoad {
                region_id,
                worker_id,
                server_name,
            },
            SignalKind::WorkerLoad(s),
        ) => {
            s.region_id == *region_id && s.worker_id == *worker_id && s.server_name == *server_name
        }
        (
            SignalKey::ClientLatency {
                client_region,
                target_region,
                server_name,
            },
            SignalKind::ClientLatency(s),
        ) => {
            s.client_region == *client_region
                && s.target_region == *target_region
                && s.server_name == *server_name
        }
        _ => false,
    };
    if matches {
        Ok(())
    } else {
        Err(CrossRegionError::InvalidConfig {
            reason: "signal body fields must match the envelope key segments".to_string(),
        })
    }
}

fn validate_identity_segment(field: &str, value: &str) -> CrossRegionResult<()> {
    if value.is_empty() {
        return Err(CrossRegionError::InvalidConfig {
            reason: format!("{field} must not be empty"),
        });
    }
    if !value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    {
        return Err(CrossRegionError::InvalidConfig {
            reason: format!(
                "{field} {value:?} must match [A-Za-z0-9._-]+ to be safe in key segments",
            ),
        });
    }
    Ok(())
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::WorkerStatus;
    use smg_mesh::{MeshKV, StreamConfig, StreamRouting};

    use super::*;

    const REGION: &str = "us-ashburn-1";
    const SERVER: &str = "smg-router-a";

    fn service() -> CrossRegionSyncService {
        let mesh_kv = Arc::new(MeshKV::new(SERVER.to_string()));
        let ns = mesh_kv.configure_stream_prefix(
            CROSS_REGION_NAMESPACE_PREFIX,
            StreamConfig {
                max_buffer_bytes: 16 * 1024 * 1024,
                routing: StreamRouting::Broadcast,
            },
        );
        CrossRegionSyncService::new(REGION.to_string(), SERVER.to_string(), ns)
            .expect("service should construct")
    }

    fn readiness_key() -> SignalKey {
        SignalKey::SmgReadiness {
            region_id: REGION.to_string(),
            server_name: SERVER.to_string(),
        }
    }

    fn readiness_body() -> SmgReadinessSignal {
        SmgReadinessSignal {
            region_id: REGION.to_string(),
            server_name: SERVER.to_string(),
            ready: true,
        }
    }

    fn make_namespace() -> Arc<StreamNamespace> {
        let mesh_kv = Arc::new(MeshKV::new(SERVER.to_string()));
        mesh_kv.configure_stream_prefix(
            CROSS_REGION_NAMESPACE_PREFIX,
            StreamConfig {
                max_buffer_bytes: 16 * 1024 * 1024,
                routing: StreamRouting::Broadcast,
            },
        )
    }

    #[test]
    fn new_rejects_empty_region_id() {
        let err = CrossRegionSyncService::new(String::new(), SERVER.to_string(), make_namespace())
            .expect_err("empty region_id must be rejected");
        assert!(matches!(err, CrossRegionError::InvalidConfig { .. }));
    }

    #[test]
    fn new_rejects_invalid_server_name() {
        let err = CrossRegionSyncService::new(
            REGION.to_string(),
            "smg/router".to_string(),
            make_namespace(),
        )
        .expect_err("invalid server_name must be rejected");
        assert!(matches!(err, CrossRegionError::InvalidConfig { .. }));
    }

    #[test]
    fn publish_stages_envelope_in_outbox_and_mirrors_state() {
        let svc = service();
        svc.publish_signal(
            readiness_key(),
            SignalKind::SmgReadiness(readiness_body()),
            30_000,
        )
        .unwrap();

        let staged = svc.outbox_snapshot();
        assert_eq!(staged.len(), 1);
        let envelope = &staged[0];
        assert_eq!(envelope.actor, SERVER);
        assert!(matches!(
            envelope.signal,
            Some(SignalKind::SmgReadiness(ref s)) if s.ready
        ));

        let state = svc.state();
        let state = state.read();
        assert!(
            state
                .readiness_replica(REGION, SERVER)
                .expect("present")
                .ready
        );
    }

    #[test]
    fn publish_signal_version_is_monotone_per_key() {
        let svc = service();
        svc.publish_signal(
            readiness_key(),
            SignalKind::SmgReadiness(readiness_body()),
            30_000,
        )
        .unwrap();
        let first_version = svc.outbox_snapshot()[0].version;

        svc.publish_signal(
            readiness_key(),
            SignalKind::SmgReadiness(SmgReadinessSignal {
                ready: false,
                ..readiness_body()
            }),
            30_000,
        )
        .unwrap();
        let second = &svc.outbox_snapshot()[0];
        assert!(second.version > first_version);
        // Same key collapses to a single outbox entry (latest wins).
        assert_eq!(svc.outbox_snapshot().len(), 1);
    }

    #[test]
    fn publish_rejects_wrong_region_in_key() {
        let svc = service();
        let key = SignalKey::SmgReadiness {
            region_id: "us-chicago-1".to_string(),
            server_name: SERVER.to_string(),
        };
        let body = SmgReadinessSignal {
            region_id: "us-chicago-1".to_string(),
            server_name: SERVER.to_string(),
            ready: true,
        };
        let err = svc
            .publish_signal(key, SignalKind::SmgReadiness(body), 30_000)
            .expect_err("wrong region must be rejected");
        assert!(matches!(err, CrossRegionError::InvalidConfig { .. }));
    }

    #[test]
    fn publish_rejects_wrong_server_name_in_key() {
        let svc = service();
        let key = SignalKey::SmgReadiness {
            region_id: REGION.to_string(),
            server_name: "other-server".to_string(),
        };
        let body = SmgReadinessSignal {
            region_id: REGION.to_string(),
            server_name: "other-server".to_string(),
            ready: true,
        };
        let err = svc
            .publish_signal(key, SignalKind::SmgReadiness(body), 30_000)
            .expect_err("wrong server_name must be rejected");
        assert!(matches!(err, CrossRegionError::InvalidConfig { .. }));
    }

    #[test]
    fn publish_rejects_body_field_mismatching_key() {
        let svc = service();
        let body = SmgReadinessSignal {
            region_id: "wrong-region".to_string(),
            server_name: SERVER.to_string(),
            ready: true,
        };
        let err = svc
            .publish_signal(readiness_key(), SignalKind::SmgReadiness(body), 30_000)
            .expect_err("body/key mismatch must be rejected");
        assert!(matches!(err, CrossRegionError::InvalidConfig { .. }));
    }

    #[test]
    fn remove_signal_drops_outbox_entry_and_local_state() {
        let svc = service();
        svc.publish_signal(
            readiness_key(),
            SignalKind::SmgReadiness(readiness_body()),
            30_000,
        )
        .unwrap();
        svc.remove_signal(readiness_key()).unwrap();

        assert!(svc.outbox_snapshot().is_empty());
        let state = svc.state();
        let state = state.read();
        assert!(state.readiness_replica(REGION, SERVER).is_none());
    }

    #[test]
    fn validate_remote_envelope_rejects_actor_key_mismatch() {
        let envelope = SignalEnvelope {
            key: readiness_key(),
            version: 1,
            actor: "different-actor".to_string(),
            generated_at_ms: 0,
            stale_after_ms: 0,
            removed: false,
            signal: Some(SignalKind::SmgReadiness(readiness_body())),
        };
        let err =
            validate_remote_envelope(&envelope).expect_err("actor/key mismatch must be rejected");
        assert!(matches!(err, CrossRegionError::InvalidConfig { .. }));
    }

    #[test]
    fn validate_remote_envelope_rejects_removed_with_body() {
        let envelope = SignalEnvelope {
            key: readiness_key(),
            version: 1,
            actor: SERVER.to_string(),
            generated_at_ms: 0,
            stale_after_ms: 0,
            removed: true,
            signal: Some(SignalKind::SmgReadiness(readiness_body())),
        };
        let err = validate_remote_envelope(&envelope).expect_err("removed+body must be rejected");
        assert!(matches!(err, CrossRegionError::InvalidConfig { .. }));
    }

    #[test]
    fn validate_remote_envelope_rejects_live_without_body() {
        let envelope = SignalEnvelope::<SignalKind> {
            key: readiness_key(),
            version: 1,
            actor: SERVER.to_string(),
            generated_at_ms: 0,
            stale_after_ms: 30_000,
            removed: false,
            signal: None,
        };
        let err =
            validate_remote_envelope(&envelope).expect_err("live without body must be rejected");
        assert!(matches!(err, CrossRegionError::InvalidConfig { .. }));
    }

    #[test]
    fn apply_envelope_to_state_round_trips_worker_health() {
        let mut state = CrossRegionState::new();
        let envelope = SignalEnvelope {
            key: SignalKey::WorkerHealth {
                region_id: "us-chicago-1".to_string(),
                worker_id: "w1".to_string(),
                server_name: "smg-router-peer".to_string(),
            },
            version: 5,
            actor: "smg-router-peer".to_string(),
            generated_at_ms: 1_700_000_000_000,
            stale_after_ms: 30_000,
            removed: false,
            signal: Some(SignalKind::WorkerHealth(WorkerHealthSignal {
                region_id: "us-chicago-1".to_string(),
                worker_id: "w1".to_string(),
                server_name: "smg-router-peer".to_string(),
                status: WorkerStatus::Ready,
            })),
        };
        apply_envelope_to_state(&mut state, &envelope);
        let observed = state
            .worker_health_replica("us-chicago-1", "w1", "smg-router-peer")
            .expect("worker health materialized");
        assert_eq!(observed.status, WorkerStatus::Ready);
    }
}
