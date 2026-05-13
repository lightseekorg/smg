//! Cross-region peer-to-peer signal sync service.
//!
//! Producer side: each adapter calls [`CrossRegionSyncService::publish_signal`]
//! to append an envelope to the in-memory log. The pull endpoint (Phase 4)
//! serves [`local_log_snapshot`] / [`local_log_delta`] over mTLS to peers.
//!
//! Consumer side: the per-peer poller calls [`apply_remote_envelopes`] to
//! merge pulled envelopes into the materialized [`CrossRegionState`] and
//! track observed peer-side max versions for restart-safe republication.

use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::{
    ClientLatencySignal, CrossRegionError, CrossRegionResult, CrossRegionState, SignalEnvelope,
    SignalKey, SignalVersion, SmgReadinessSignal, WorkerHealthSignal, WorkerLoadSignal,
};

/// Opaque per-producer log cursor handed out to consumers. Monotonic, never
/// reused; consumers pass the most recent value back via the pull endpoint's
/// `since=` parameter.
pub type Cursor = u64;

/// Returned by [`CrossRegionSyncService::local_log_delta`] when the requested
/// cursor falls outside the retained log range. The pull endpoint maps this
/// to a 409 so the consumer retries with no cursor for a full snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CursorStale;

/// Body-erased signal payload. Adapters construct one of these per publish
/// and hand it to [`CrossRegionSyncService::publish_signal`]. The wire format
/// is tag-discriminated JSON via serde so the pull endpoint's untyped
/// envelope stays self-describing.
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

/// Defaults for log retention. These are the producer-side limits;
/// the consumer-side `CrossRegionState` will get matching freshness gates
/// when the pull-protocol consumer is wired up.
#[derive(Debug, Clone, Copy)]
pub struct SyncRetention {
    /// Keep `removed: true` entries in the log this long so a peer recovering
    /// from a partition shorter than this still observes the deletion. Must
    /// exceed `max_tolerated_partition + max_clock_skew` (design §2b).
    pub tombstone_retention_ms: i64,
    /// Keep stale per-replica live entries this long. After this, a dead
    /// replica's keys are GC'd locally.
    pub dead_replica_retention_ms: i64,
}

impl Default for SyncRetention {
    fn default() -> Self {
        Self {
            // 24h — sized for typical multi-region operational outage budget.
            tombstone_retention_ms: 24 * 60 * 60 * 1_000,
            // 6h — covers routine replica churn.
            dead_replica_retention_ms: 6 * 60 * 60 * 1_000,
        }
    }
}

/// Cross-region signal sync service.
///
/// Owns the producer's append-only log and the consumer's materialized state.
/// `region_id` and `server_name` are stamped onto every published envelope as
/// the key's region/replica segments and the envelope's actor field.
pub struct CrossRegionSyncService {
    region_id: String,
    server_name: String,
    state: Arc<RwLock<CrossRegionState>>,
    log: Arc<RwLock<SignalLog>>,
    /// Per-key max version observed from peers' caches for keys this replica
    /// writes. Populated by [`apply_remote_envelopes`] when a pulled envelope
    /// carries our own `actor`. Used by [`publish_signal`] so a restarted
    /// producer can compute a `version` that exceeds whatever peers retained
    /// from before the restart (design §2c).
    observed_remote_max: Arc<RwLock<HashMap<SignalKey, u64>>>,
    retention: SyncRetention,
}

impl std::fmt::Debug for CrossRegionSyncService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossRegionSyncService")
            .field("region_id", &self.region_id)
            .field("server_name", &self.server_name)
            .field("retention", &self.retention)
            .finish_non_exhaustive()
    }
}

impl CrossRegionSyncService {
    /// Build a sync service rooted at this replica's identity. Validates
    /// `server_name` charset (design §2c step 4) so a `/` in the name can
    /// never reach a key segment and break parsing.
    pub fn new(region_id: String, server_name: String) -> CrossRegionResult<Self> {
        Self::new_with_retention(region_id, server_name, SyncRetention::default())
    }

    /// Variant that lets tests / config plumb custom retention windows.
    pub fn new_with_retention(
        region_id: String,
        server_name: String,
        retention: SyncRetention,
    ) -> CrossRegionResult<Self> {
        validate_identity_segment("region_id", &region_id)?;
        validate_identity_segment("server_name", &server_name)?;
        Ok(Self {
            region_id,
            server_name,
            state: Arc::new(RwLock::new(CrossRegionState::new())),
            log: Arc::new(RwLock::new(SignalLog::new())),
            observed_remote_max: Arc::new(RwLock::new(HashMap::new())),
            retention,
        })
    }

    /// This replica's region (the value adapters must put in the key's
    /// `region`/`client_region` segment).
    pub fn region_id(&self) -> &str {
        &self.region_id
    }

    /// This replica's `server_name` (the value adapters must put in the
    /// key's trailing segment and what every envelope's `actor` carries).
    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    /// Shared materialized state. Consumers (candidate calculation, the
    /// `/get_loads` projection) read through this Arc. Wrapped in `RwLock`
    /// because [`apply_remote_envelopes`] writes from a polling task.
    pub fn state(&self) -> Arc<RwLock<CrossRegionState>> {
        self.state.clone()
    }

    /// Publish a live signal. The service computes a wall-clock-anchored
    /// version (§2c), stamps the local `actor`, validates the key's
    /// region/server_name segments against this replica's identity, and
    /// appends the envelope to the log. Echoes into local state so adapters
    /// see a consistent self-view.
    pub fn publish_signal(
        &self,
        key: SignalKey,
        signal: SignalKind,
        stale_after_ms: u32,
    ) -> CrossRegionResult<()> {
        self.validate_key(&key)?;
        validate_body_against_key(&key, &signal)?;
        let envelope = self.build_envelope(key, Some(signal), stale_after_ms, false);
        self.append_and_mirror(envelope);
        Ok(())
    }

    /// Publish a tombstone for a key. The signal body becomes `None`,
    /// `stale_after_ms = 0`, and `removed = true`. Versioning is identical
    /// to [`publish_signal`] so a tombstone post-restart still outranks any
    /// live entry peers retained.
    pub fn remove_signal(&self, key: SignalKey) -> CrossRegionResult<()> {
        self.validate_key(&key)?;
        let envelope = self.build_envelope(key, None, 0, true);
        self.append_and_mirror(envelope);
        Ok(())
    }

    /// Serve a full snapshot of the local log. Returns the high-water cursor
    /// (= highest cursor present, or 0 if the log is empty) so the caller can
    /// fetch deltas with [`local_log_delta`] afterwards.
    pub fn local_log_snapshot(&self) -> (Vec<SignalEnvelope<SignalKind>>, Cursor) {
        let log = self.log.read();
        let entries = log
            .entries
            .iter()
            .map(|(_, env)| env.clone())
            .collect::<Vec<_>>();
        (entries, log.high_water())
    }

    /// Serve a cursor delta. Returns every envelope with cursor strictly
    /// greater than `since` plus the new high-water mark. `Err(CursorStale)`
    /// indicates the caller's cursor is older than the retained log range;
    /// the pull endpoint maps this to HTTP 409 and the client retries with
    /// no cursor.
    pub fn local_log_delta(
        &self,
        since: Cursor,
    ) -> Result<(Vec<SignalEnvelope<SignalKind>>, Cursor), CursorStale> {
        let log = self.log.read();
        if let Some((oldest, _)) = log.entries.front() {
            // The caller's cursor is stale iff the oldest retained cursor
            // is *strictly greater than* `since + 1` — that means we'd
            // miss at least one entry between (since, oldest).
            if *oldest > since.saturating_add(1) {
                return Err(CursorStale);
            }
        }
        let entries = log
            .entries
            .iter()
            .filter_map(|(c, env)| if *c > since { Some(env.clone()) } else { None })
            .collect::<Vec<_>>();
        Ok((entries, log.high_water()))
    }

    /// Apply envelopes pulled from a peer. Updates `CrossRegionState` for
    /// each typed signal and records the observed max version for any
    /// envelope whose actor matches this replica (so a restarted producer
    /// can compute a version that exceeds peer-cached pre-restart writes).
    pub fn apply_remote_envelopes(&self, _peer: &str, envelopes: &[SignalEnvelope<SignalKind>]) {
        let mut state = self.state.write();
        let mut observed = self.observed_remote_max.write();
        for env in envelopes {
            if env.actor == self.server_name {
                let entry = observed.entry(env.key.clone()).or_insert(0);
                if env.version > *entry {
                    *entry = env.version;
                }
            }
            apply_envelope_to_state(&mut state, env);
        }
    }

    /// Drop log entries whose age exceeds the retention bounds. Run by a
    /// periodic task in production; exposed here so tests and the pull
    /// endpoint can trigger it deterministically.
    pub fn gc_log(&self, now_ms: i64) {
        self.log.write().gc(now_ms, &self.retention);
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
        let prev = self
            .log
            .read()
            .latest_per_key
            .get(&key)
            .copied()
            .unwrap_or(0);
        let observed = self
            .observed_remote_max
            .read()
            .get(&key)
            .copied()
            .unwrap_or(0);
        // Wall-clock-anchored, multi-writer-safe version (design §2c).
        let now_u64 = u64::try_from(now.max(0)).unwrap_or(0);
        let version = now_u64
            .max(prev.saturating_add(1))
            .max(observed.saturating_add(1));
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

    fn append_and_mirror(&self, envelope: SignalEnvelope<SignalKind>) {
        // Append to log first (so any reader sees a consistent log+state).
        self.log.write().append(envelope.clone());
        // Mirror envelopes into state for the producer's self-view.
        apply_envelope_to_state(&mut self.state.write(), &envelope);
    }
}

/// Validate that a body's region/worker/server_name fields agree with the key.
/// Adapters should construct bodies consistently, but defense-in-depth keys
/// this from accidentally diverging.
fn validate_body_against_key(key: &SignalKey, signal: &SignalKind) -> CrossRegionResult<()> {
    let mismatch = |field: &str, key_val: &str, body_val: &str| CrossRegionError::InvalidConfig {
        reason: format!("signal body {field} {body_val:?} does not match key {field} {key_val:?}",),
    };
    match (key, signal) {
        (
            SignalKey::SmgReadiness {
                region_id,
                server_name,
            },
            SignalKind::SmgReadiness(body),
        ) => {
            if &body.region_id != region_id {
                return Err(mismatch("region_id", region_id, &body.region_id));
            }
            if &body.server_name != server_name {
                return Err(mismatch("server_name", server_name, &body.server_name));
            }
            Ok(())
        }
        (
            SignalKey::WorkerHealth {
                region_id,
                worker_id,
                server_name,
            },
            SignalKind::WorkerHealth(body),
        ) => {
            if &body.region_id != region_id {
                return Err(mismatch("region_id", region_id, &body.region_id));
            }
            if &body.worker_id != worker_id {
                return Err(mismatch("worker_id", worker_id, &body.worker_id));
            }
            if &body.server_name != server_name {
                return Err(mismatch("server_name", server_name, &body.server_name));
            }
            Ok(())
        }
        (
            SignalKey::WorkerLoad {
                region_id,
                worker_id,
                server_name,
            },
            SignalKind::WorkerLoad(body),
        ) => {
            if &body.region_id != region_id {
                return Err(mismatch("region_id", region_id, &body.region_id));
            }
            if &body.worker_id != worker_id {
                return Err(mismatch("worker_id", worker_id, &body.worker_id));
            }
            if &body.server_name != server_name {
                return Err(mismatch("server_name", server_name, &body.server_name));
            }
            Ok(())
        }
        (
            SignalKey::ClientLatency {
                client_region,
                target_region,
                server_name,
            },
            SignalKind::ClientLatency(body),
        ) => {
            if &body.client_region != client_region {
                return Err(mismatch(
                    "client_region",
                    client_region,
                    &body.client_region,
                ));
            }
            if &body.target_region != target_region {
                return Err(mismatch(
                    "target_region",
                    target_region,
                    &body.target_region,
                ));
            }
            if &body.server_name != server_name {
                return Err(mismatch("server_name", server_name, &body.server_name));
            }
            Ok(())
        }
        _ => Err(CrossRegionError::InvalidConfig {
            reason: "signal body kind does not match key kind".to_string(),
        }),
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

fn apply_envelope_to_state(state: &mut CrossRegionState, envelope: &SignalEnvelope<SignalKind>) {
    if envelope.removed {
        state.remove_key(&envelope.key);
        return;
    }
    let version = SignalVersion {
        version: envelope.version,
        updated_at_ms: envelope.generated_at_ms,
    };
    match envelope.signal.as_ref() {
        Some(SignalKind::SmgReadiness(s)) => state.upsert_readiness(s.clone(), version),
        Some(SignalKind::WorkerHealth(s)) => state.upsert_worker_health(s.clone(), version),
        Some(SignalKind::WorkerLoad(s)) => state.upsert_worker_load(s.as_ref().clone(), version),
        Some(SignalKind::ClientLatency(s)) => state.upsert_client_latency(s.clone(), version),
        None => {}
    }
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

#[derive(Debug)]
struct SignalLog {
    entries: VecDeque<(Cursor, SignalEnvelope<SignalKind>)>,
    /// Next cursor to hand out. Starts at 1 so 0 is reserved for
    /// "no cursor — give me a full snapshot".
    next_cursor: Cursor,
    /// Per-key max version seen by the local writer, used to compute the
    /// next version without scanning `entries`.
    latest_per_key: HashMap<SignalKey, u64>,
}

impl SignalLog {
    fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            next_cursor: 1,
            latest_per_key: HashMap::new(),
        }
    }

    fn append(&mut self, env: SignalEnvelope<SignalKind>) {
        let cursor = self.next_cursor;
        self.next_cursor = self.next_cursor.saturating_add(1);
        let key = env.key.clone();
        let version = env.version;
        self.entries.push_back((cursor, env));
        let entry = self.latest_per_key.entry(key).or_insert(0);
        if version > *entry {
            *entry = version;
        }
    }

    /// High-water cursor — the largest cursor present in the log, or 0 if
    /// empty. Returned by `local_log_snapshot` / `local_log_delta` and
    /// passed back as the consumer's next `since=` value.
    fn high_water(&self) -> Cursor {
        self.next_cursor.saturating_sub(1)
    }

    fn gc(&mut self, now_ms: i64, retention: &SyncRetention) {
        self.entries.retain(|(_, env)| {
            let age = now_ms.saturating_sub(env.generated_at_ms);
            let limit = if env.removed {
                retention.tombstone_retention_ms
            } else {
                retention.dead_replica_retention_ms
            };
            age <= limit
        });
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::WorkerStatus;

    use super::*;

    const REGION: &str = "us-ashburn-1";
    const SERVER: &str = "smg-router-a";

    fn service() -> CrossRegionSyncService {
        CrossRegionSyncService::new(REGION.to_string(), SERVER.to_string())
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
    fn new_rejects_invalid_server_name() {
        let err = CrossRegionSyncService::new(REGION.to_string(), "smg/router".to_string())
            .expect_err("slash in server_name must be rejected");
        assert!(matches!(err, CrossRegionError::InvalidConfig { .. }));
    }

    #[test]
    fn new_rejects_empty_region_id() {
        let err = CrossRegionSyncService::new(String::new(), SERVER.to_string())
            .expect_err("empty region_id must be rejected");
        assert!(matches!(err, CrossRegionError::InvalidConfig { .. }));
    }

    #[test]
    fn publish_signal_assigns_actor_and_increasing_version() {
        let svc = service();
        svc.publish_signal(
            readiness_key(),
            SignalKind::SmgReadiness(readiness_body()),
            30_000,
        )
        .expect("first publish should succeed");

        let (entries, cursor_after_first) = svc.local_log_snapshot();
        assert_eq!(entries.len(), 1);
        let first = &entries[0];
        assert_eq!(first.actor, SERVER);
        assert!(first.version > 0);
        assert_eq!(first.stale_after_ms, 30_000);
        assert!(!first.removed);
        assert!(matches!(first.signal, Some(SignalKind::SmgReadiness(_))));

        svc.publish_signal(
            readiness_key(),
            SignalKind::SmgReadiness(readiness_body()),
            30_000,
        )
        .expect("second publish should succeed");
        let (entries, cursor_after_second) = svc.local_log_snapshot();
        assert_eq!(entries.len(), 2);
        assert!(
            entries[1].version > entries[0].version,
            "second publish must outrank first: {} <= {}",
            entries[1].version,
            entries[0].version,
        );
        assert!(cursor_after_second > cursor_after_first);
    }

    #[test]
    fn remove_signal_emits_tombstone_with_higher_version() {
        let svc = service();
        svc.publish_signal(
            readiness_key(),
            SignalKind::SmgReadiness(readiness_body()),
            30_000,
        )
        .unwrap();
        svc.remove_signal(readiness_key()).expect("remove ok");

        let (entries, _) = svc.local_log_snapshot();
        assert_eq!(entries.len(), 2);
        let tombstone = &entries[1];
        assert!(tombstone.removed);
        assert_eq!(tombstone.stale_after_ms, 0);
        assert!(tombstone.signal.is_none());
        assert!(tombstone.version > entries[0].version);

        let state = svc.state();
        let state = state.read();
        assert!(
            state.readiness(REGION).is_none(),
            "tombstone should remove the materialized value immediately"
        );
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
            server_name: "smg-router-evil".to_string(),
        };
        let body = SmgReadinessSignal {
            region_id: REGION.to_string(),
            server_name: "smg-router-evil".to_string(),
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
        let key = readiness_key();
        // Key says SERVER but body claims something else — must fail.
        let body = SmgReadinessSignal {
            region_id: REGION.to_string(),
            server_name: "rogue".to_string(),
            ready: true,
        };
        let err = svc
            .publish_signal(key, SignalKind::SmgReadiness(body), 30_000)
            .expect_err("body/key server_name mismatch must be rejected");
        assert!(matches!(err, CrossRegionError::InvalidConfig { .. }));
    }

    #[test]
    fn worker_health_key_and_body_round_trip() {
        let svc = service();
        let key = SignalKey::WorkerHealth {
            region_id: REGION.to_string(),
            worker_id: "w-1".to_string(),
            server_name: SERVER.to_string(),
        };
        let body = WorkerHealthSignal {
            region_id: REGION.to_string(),
            worker_id: "w-1".to_string(),
            server_name: SERVER.to_string(),
            status: WorkerStatus::Ready,
        };
        svc.publish_signal(key, SignalKind::WorkerHealth(body), 60_000)
            .expect("publish ok");
        let (entries, _) = svc.local_log_snapshot();
        assert_eq!(entries.len(), 1);
        assert!(matches!(
            entries[0].signal,
            Some(SignalKind::WorkerHealth(_))
        ));
    }

    #[test]
    fn log_delta_returns_only_newer_entries() {
        let svc = service();
        let (_, c0) = svc.local_log_snapshot();
        svc.publish_signal(
            readiness_key(),
            SignalKind::SmgReadiness(readiness_body()),
            30_000,
        )
        .unwrap();
        let (delta1, c1) = svc.local_log_delta(c0).expect("cursor ok");
        assert_eq!(delta1.len(), 1);
        assert!(c1 > c0);

        // Another publish — delta from c1 returns just one entry.
        svc.publish_signal(
            readiness_key(),
            SignalKind::SmgReadiness(readiness_body()),
            30_000,
        )
        .unwrap();
        let (delta2, _) = svc.local_log_delta(c1).expect("cursor ok");
        assert_eq!(delta2.len(), 1);
    }

    #[test]
    fn log_delta_with_stale_cursor_returns_cursor_stale() {
        let svc = service();
        // Force a non-empty log so the stale-cursor branch fires.
        svc.publish_signal(
            readiness_key(),
            SignalKind::SmgReadiness(readiness_body()),
            30_000,
        )
        .unwrap();
        svc.publish_signal(
            readiness_key(),
            SignalKind::SmgReadiness(readiness_body()),
            30_000,
        )
        .unwrap();
        // GC dropping the front simulates a producer that has retained
        // only recent entries.
        {
            let mut log = svc.log.write();
            log.entries.pop_front();
        }
        // Cursor 0 is below the retained range now (front is cursor 2).
        let err = svc.local_log_delta(0).expect_err("stale cursor should 409");
        assert_eq!(err, CursorStale);
    }

    #[test]
    fn apply_remote_envelope_updates_state() {
        let svc = service();
        let peer_key = SignalKey::SmgReadiness {
            region_id: "us-chicago-1".to_string(),
            server_name: "smg-router-peer".to_string(),
        };
        let peer_envelope = SignalEnvelope {
            key: peer_key,
            version: 42,
            actor: "smg-router-peer".to_string(),
            generated_at_ms: 1_700_000_000_000,
            stale_after_ms: 30_000,
            removed: false,
            signal: Some(SignalKind::SmgReadiness(SmgReadinessSignal {
                region_id: "us-chicago-1".to_string(),
                server_name: "smg-router-peer".to_string(),
                ready: true,
            })),
        };
        svc.apply_remote_envelopes("us-chicago-1", &[peer_envelope]);
        let state = svc.state();
        let state = state.read();
        let signal = state
            .readiness("us-chicago-1")
            .expect("peer readiness must be materialized");
        assert!(signal.ready);
    }

    #[test]
    fn apply_remote_envelope_with_own_actor_seeds_observed_remote_max() {
        // Simulate post-restart: producer pulls peer's cache, sees its own
        // pre-restart write at a high version, then the next publish must
        // exceed that.
        let svc = service();
        let key = readiness_key();
        let pre_restart_envelope = SignalEnvelope {
            key: key.clone(),
            version: 999_999,
            actor: SERVER.to_string(),
            generated_at_ms: 1_700_000_000_000,
            stale_after_ms: 30_000,
            removed: false,
            signal: Some(SignalKind::SmgReadiness(readiness_body())),
        };
        svc.apply_remote_envelopes("peer-x", &[pre_restart_envelope]);

        svc.publish_signal(key, SignalKind::SmgReadiness(readiness_body()), 30_000)
            .unwrap();
        let (entries, _) = svc.local_log_snapshot();
        assert!(
            entries[0].version > 999_999,
            "post-restart publish must outrank pre-restart peer-observed version: {}",
            entries[0].version,
        );
    }

    #[test]
    fn publish_mirrors_into_local_state() {
        let svc = service();
        svc.publish_signal(
            readiness_key(),
            SignalKind::SmgReadiness(readiness_body()),
            30_000,
        )
        .unwrap();
        let state = svc.state();
        let state = state.read();
        assert!(state.readiness(REGION).is_some());
    }

    #[test]
    fn gc_drops_stale_live_entries_past_dead_replica_retention() {
        let svc = CrossRegionSyncService::new_with_retention(
            REGION.to_string(),
            SERVER.to_string(),
            SyncRetention {
                tombstone_retention_ms: 100,
                dead_replica_retention_ms: 50,
            },
        )
        .unwrap();
        svc.publish_signal(
            readiness_key(),
            SignalKind::SmgReadiness(readiness_body()),
            30_000,
        )
        .unwrap();
        let (entries, _) = svc.local_log_snapshot();
        assert_eq!(entries.len(), 1);
        // 60ms after publish: live retention (50ms) exceeded — GC should drop.
        let publish_ts = entries[0].generated_at_ms;
        svc.gc_log(publish_ts + 60);
        let (entries, _) = svc.local_log_snapshot();
        assert!(entries.is_empty(), "stale live entry should be GC'd");
    }

    #[test]
    fn gc_keeps_tombstones_longer_than_live_entries() {
        let svc = CrossRegionSyncService::new_with_retention(
            REGION.to_string(),
            SERVER.to_string(),
            SyncRetention {
                tombstone_retention_ms: 100,
                dead_replica_retention_ms: 50,
            },
        )
        .unwrap();
        svc.remove_signal(readiness_key()).unwrap();
        let (entries, _) = svc.local_log_snapshot();
        let tombstone_ts = entries[0].generated_at_ms;
        // 60ms: tombstone retention (100ms) NOT yet exceeded — kept.
        svc.gc_log(tombstone_ts + 60);
        let (entries, _) = svc.local_log_snapshot();
        assert_eq!(entries.len(), 1, "tombstone should outlive live retention");
        assert!(entries[0].removed);
    }
}
