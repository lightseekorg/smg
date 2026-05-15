//! Cross-region signal sync service backed by mesh gossip.
//!
//! Producers (the four adapters in `crate::cross_region::adapters`) call
//! [`CrossRegionSyncService::publish_signal`] / [`CrossRegionSyncService::remove_signal`]
//! to write a typed envelope into the configured mesh `CrdtNamespace`. Mesh
//! propagates the bytes via its existing gossip transport; peers receive
//! subscription events on the same namespace and drive
//! [`apply_envelope_to_state`] into the materialized [`CrossRegionState`].
//!
//! The wall-clock-anchored `(version, actor)` envelope-level versioning is
//! kept as application metadata (consumed by `RemoteRegionView` for
//! freshness). Apply ordering across replicas is owned by mesh's Lamport
//! clock + `LastWriterWins` CRDT merge — application-level
//! `should_replace((version, actor))` runs as a secondary check on
//! `CrossRegionState` so the materialized view never goes backwards.

use std::{collections::HashMap, sync::Arc};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use smg_mesh::CrdtNamespace;

use super::{
    ClientLatencySignal, CrossRegionError, CrossRegionResult, CrossRegionState, SignalEnvelope,
    SignalKey, SignalVersion, SmgReadinessSignal, WorkerHealthSignal, WorkerLoadSignal,
};

/// Mesh namespace prefix carrying cross-region signal envelopes. Must end
/// with `:` to match `MeshKV::configure_crdt_prefix` conventions.
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
/// Owns the materialized [`CrossRegionState`] consumer-side view and a handle
/// to the mesh `CrdtNamespace` that carries the wire envelopes. `region_id`
/// and `server_name` are stamped onto every published envelope as the key's
/// region/replica segments and the envelope's actor field, and validated
/// against incoming key segments.
pub struct CrossRegionSyncService {
    region_id: String,
    server_name: String,
    state: Arc<RwLock<CrossRegionState>>,
    namespace: Arc<CrdtNamespace>,
    /// Per-key monotonic counter for the envelope-level `version` field.
    /// Mesh's Lamport handles cross-replica apply ordering; this counter
    /// keeps the envelope's `(version, actor)` metadata strictly monotone
    /// per writer so `RemoteRegionView` freshness reads stay coherent.
    latest_per_key: Arc<RwLock<HashMap<SignalKey, u64>>>,
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
    /// a key path segment and break parsing.
    pub fn new(
        region_id: String,
        server_name: String,
        namespace: Arc<CrdtNamespace>,
    ) -> CrossRegionResult<Self> {
        validate_identity_segment("region_id", &region_id)?;
        validate_identity_segment("server_name", &server_name)?;
        Ok(Self {
            region_id,
            server_name,
            state: Arc::new(RwLock::new(CrossRegionState::new())),
            namespace,
            latest_per_key: Arc::new(RwLock::new(HashMap::new())),
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

    /// Shared namespace handle. The mesh subscriber task subscribes through
    /// this to receive peer updates.
    pub fn namespace(&self) -> Arc<CrdtNamespace> {
        self.namespace.clone()
    }

    /// Publish a live signal. Validates the key/body against this replica's
    /// identity, serializes the envelope, writes it through the mesh
    /// namespace, and mirrors into local state so the producer's self-view
    /// is consistent immediately (rather than waiting for the local
    /// subscriber callback to fire).
    pub fn publish_signal(
        &self,
        key: SignalKey,
        signal: SignalKind,
        stale_after_ms: u32,
    ) -> CrossRegionResult<()> {
        self.validate_key(&key)?;
        validate_body_against_key(&key, &signal)?;
        let envelope = self.build_envelope(key, Some(signal), stale_after_ms, false);
        self.put_through_mesh(&envelope)?;
        apply_envelope_to_state(&mut self.state.write(), &envelope);
        Ok(())
    }

    /// Publish a tombstone via the mesh namespace. Mesh's tombstone grace
    /// period handles retention; peers receive a `(key, None)` subscriber
    /// event when the tombstone propagates.
    pub fn remove_signal(&self, key: SignalKey) -> CrossRegionResult<()> {
        self.validate_key(&key)?;
        // Build a tombstone envelope for the producer's local apply path
        // so `remove_key_with_version` honours `(version, actor)` ordering
        // for the self-view.
        let envelope = self.build_envelope(key.clone(), None, 0, true);
        self.namespace.delete(&mesh_path(&envelope.key));
        apply_envelope_to_state(&mut self.state.write(), &envelope);
        Ok(())
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

    fn put_through_mesh(&self, envelope: &SignalEnvelope<SignalKind>) -> CrossRegionResult<()> {
        let path = mesh_path(&envelope.key);
        let bytes = serde_json::to_vec(envelope).map_err(|e| CrossRegionError::InvalidConfig {
            reason: format!("failed to encode signal envelope: {e}"),
        })?;
        self.namespace.put(&path, bytes);
        Ok(())
    }
}

/// Storage-side key path for the mesh `CrdtNamespace`. The namespace prefix
/// is required by `CrdtNamespace::put` to disambiguate which CRDT applies.
pub fn mesh_path(key: &SignalKey) -> String {
    format!("{}{}", CROSS_REGION_NAMESPACE_PREFIX, key.as_path())
}

/// Apply a decoded envelope to the materialized state. Used by the producer
/// self-mirror path on publish and by the mesh subscriber task on incoming
/// peer updates.
///
/// Mesh's CRDT has already decided which envelope wins for a given key at
/// the gossip layer; the application-level `(version, actor)` check on
/// `CrossRegionState::upsert_*` / `remove_key_with_version` runs here as a
/// secondary defense — it guarantees the materialized view never moves
/// backwards even if mesh delivers an out-of-order replay.
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
pub fn decode_envelope(chunks: &[bytes::Bytes]) -> CrossRegionResult<SignalEnvelope<SignalKind>> {
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
    use smg_mesh::{MergeStrategy, MeshKV};

    use super::*;

    const REGION: &str = "us-ashburn-1";
    const SERVER: &str = "smg-router-a";

    fn service() -> CrossRegionSyncService {
        let mesh_kv = Arc::new(MeshKV::new(SERVER.to_string()));
        let ns = mesh_kv
            .configure_crdt_prefix(CROSS_REGION_NAMESPACE_PREFIX, MergeStrategy::LastWriterWins);
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

    #[test]
    fn new_rejects_empty_region_id() {
        let mesh_kv = Arc::new(MeshKV::new(SERVER.to_string()));
        let ns = mesh_kv
            .configure_crdt_prefix(CROSS_REGION_NAMESPACE_PREFIX, MergeStrategy::LastWriterWins);
        let err = CrossRegionSyncService::new(String::new(), SERVER.to_string(), ns)
            .expect_err("empty region_id must be rejected");
        assert!(matches!(err, CrossRegionError::InvalidConfig { .. }));
    }

    #[test]
    fn new_rejects_invalid_server_name() {
        let mesh_kv = Arc::new(MeshKV::new(SERVER.to_string()));
        let ns = mesh_kv
            .configure_crdt_prefix(CROSS_REGION_NAMESPACE_PREFIX, MergeStrategy::LastWriterWins);
        let err = CrossRegionSyncService::new(REGION.to_string(), "smg/router".to_string(), ns)
            .expect_err("invalid server_name must be rejected");
        assert!(matches!(err, CrossRegionError::InvalidConfig { .. }));
    }

    #[test]
    fn publish_writes_through_namespace_and_mirrors_state() {
        let svc = service();
        svc.publish_signal(
            readiness_key(),
            SignalKind::SmgReadiness(readiness_body()),
            30_000,
        )
        .unwrap();

        let path = mesh_path(&readiness_key());
        let bytes = svc
            .namespace
            .get(&path)
            .expect("value present in namespace");
        let envelope: SignalEnvelope<SignalKind> = serde_json::from_slice(&bytes).expect("decode");
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
        let path = mesh_path(&readiness_key());
        let first: SignalEnvelope<SignalKind> =
            serde_json::from_slice(&svc.namespace.get(&path).unwrap()).unwrap();

        svc.publish_signal(
            readiness_key(),
            SignalKind::SmgReadiness(SmgReadinessSignal {
                ready: false,
                ..readiness_body()
            }),
            30_000,
        )
        .unwrap();
        let second: SignalEnvelope<SignalKind> =
            serde_json::from_slice(&svc.namespace.get(&path).unwrap()).unwrap();
        assert!(second.version > first.version);
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
    fn remove_signal_drops_key_from_namespace_and_state() {
        let svc = service();
        svc.publish_signal(
            readiness_key(),
            SignalKind::SmgReadiness(readiness_body()),
            30_000,
        )
        .unwrap();
        svc.remove_signal(readiness_key()).unwrap();

        let path = mesh_path(&readiness_key());
        assert!(svc.namespace.get(&path).is_none());
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
