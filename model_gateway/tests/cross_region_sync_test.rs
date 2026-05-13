//! Cross-region sync plane acceptance tests.
//!
//! Each test wires two `CrossRegionSyncService` instances directly (no HTTP)
//! and drives the producer-side log → consumer-side `apply_remote_envelopes`
//! path that the production pull-server/pull-client pair exercises end-to-end.
//! Skipping HTTP means we cover the *semantics* (apply ordering, tombstones,
//! cursor-stale → snapshot, freshness windowing, view projection) without
//! standing up mTLS certs or bound ports.
//!
//! Acceptance criteria from
//! `2026-05-13-cross-region-sync-implementation-plan.md` §"Phase G":
//!
//! 1. local publish on A round-trips to B's materialized state
//! 2. idempotent apply (`(version, actor)` equality is a no-op)
//! 3. older-version envelopes rejected after newer ones observed
//! 4. multi-replica signals in one region survive materialization
//! 5. per-replica tombstone removes only the addressed replica
//! 6. stale cursor → full-snapshot resync via cursor-0 path
//! 7. `RemoteRegionView` freshness window filters stale entries
//! 8. candidate ranking + `RemoteRegionView` project consistently

use openai_protocol::{
    model_type::Endpoint,
    worker::{WorkerLoadInfo, WorkerStatus},
};
use smg::{
    config::CrossRegionFailoverMode,
    cross_region::{
        CandidateCalculationInput, CandidateCalculator, ClientLatencySignal, CrossRegionBreaker,
        CrossRegionSyncService, CursorStale, FailoverPolicy, ModalityPolicy, RegionPeer,
        RegionPeerRegistry, RemoteRegionView, RoutingProfileContext, SignalKey, SignalKind,
        SmgReadinessSignal, SyncRetention, WorkerHealthSignal, WorkerLoadSignal,
    },
};

const REGION_A: &str = "us-ashburn-1";
const REGION_B: &str = "us-chicago-1";
const SERVER_A1: &str = "smg-router-a1";
const SERVER_A2: &str = "smg-router-a2";
const SERVER_B: &str = "smg-router-b";

#[expect(clippy::expect_used, reason = "test helper — fixture is known-valid")]
fn service(region: &str, server: &str) -> CrossRegionSyncService {
    CrossRegionSyncService::new(region.to_string(), server.to_string())
        .expect("service should construct")
}

#[expect(clippy::expect_used, reason = "test helper — fixture is known-valid")]
fn service_with_retention(
    region: &str,
    server: &str,
    retention: SyncRetention,
) -> CrossRegionSyncService {
    CrossRegionSyncService::new_with_retention(region.to_string(), server.to_string(), retention)
        .expect("service should construct")
}

/// Drive one pull cycle from A → B, returning B's new cursor. Mirrors the
/// pull-client's loop: on `CursorStale`, fall through to a snapshot request.
fn pull_and_apply(
    a: &CrossRegionSyncService,
    b: &CrossRegionSyncService,
    cursor: u64,
) -> (u64, bool) {
    if cursor == 0 {
        let (envs, next) = a.local_log_snapshot();
        b.apply_remote_envelopes(a.region_id(), &envs);
        return (next, false);
    }
    match a.local_log_delta(cursor) {
        Ok((envs, next)) => {
            b.apply_remote_envelopes(a.region_id(), &envs);
            (next, false)
        }
        Err(CursorStale) => {
            let (envs, next) = a.local_log_snapshot();
            b.apply_remote_envelopes(a.region_id(), &envs);
            (next, true)
        }
    }
}

fn readiness_key(server: &str) -> SignalKey {
    SignalKey::SmgReadiness {
        region_id: REGION_A.to_string(),
        server_name: server.to_string(),
    }
}

fn readiness_body(server: &str, ready: bool) -> SmgReadinessSignal {
    SmgReadinessSignal {
        region_id: REGION_A.to_string(),
        server_name: server.to_string(),
        ready,
    }
}

fn worker_health_key(server: &str, worker_id: &str) -> SignalKey {
    SignalKey::WorkerHealth {
        region_id: REGION_A.to_string(),
        worker_id: worker_id.to_string(),
        server_name: server.to_string(),
    }
}

fn worker_health_body(server: &str, worker_id: &str, status: WorkerStatus) -> WorkerHealthSignal {
    WorkerHealthSignal {
        region_id: REGION_A.to_string(),
        worker_id: worker_id.to_string(),
        server_name: server.to_string(),
        status,
    }
}

fn worker_load_key(server: &str, worker_id: &str) -> SignalKey {
    SignalKey::WorkerLoad {
        region_id: REGION_A.to_string(),
        worker_id: worker_id.to_string(),
        server_name: server.to_string(),
    }
}

fn worker_load_body(
    server: &str,
    worker_id: &str,
    model_id: &str,
    load: isize,
) -> WorkerLoadSignal {
    WorkerLoadSignal {
        region_id: REGION_A.to_string(),
        worker_id: worker_id.to_string(),
        server_name: server.to_string(),
        load: WorkerLoadInfo {
            worker: worker_id.to_string(),
            worker_type: None,
            load,
            details: None,
            region_id: Some(REGION_A.to_string()),
            worker_id: Some(worker_id.to_string()),
            model_id: Some(model_id.to_string()),
            status: Some(WorkerStatus::Ready),
            generated_at_ms: Some(0),
            version: Some(1),
            source: None,
            remote_workers: None,
        },
    }
}

/// Client-latency keys are owned by the *client* region (the observer).
/// A is the observer here; B is the latency target. This matches the
/// publisher-side ownership invariant the sync service enforces.
fn client_latency_key(server: &str) -> SignalKey {
    SignalKey::ClientLatency {
        client_region: REGION_A.to_string(),
        target_region: REGION_B.to_string(),
        server_name: server.to_string(),
    }
}

fn client_latency_body(server: &str, p50: u64, p95: u64) -> ClientLatencySignal {
    ClientLatencySignal {
        client_region: REGION_A.to_string(),
        target_region: REGION_B.to_string(),
        server_name: server.to_string(),
        p50_latency_ms: p50,
        p95_latency_ms: p95,
    }
}

// -------------------------------------------------------------------------
// 1. local publish on A round-trips to B's materialized state
// -------------------------------------------------------------------------

#[test]
fn local_publish_then_remote_apply_round_trip() {
    let a = service(REGION_A, SERVER_A1);
    let b = service(REGION_B, SERVER_B);

    a.publish_signal(
        readiness_key(SERVER_A1),
        SignalKind::SmgReadiness(readiness_body(SERVER_A1, true)),
        30_000,
    )
    .unwrap();
    a.publish_signal(
        worker_health_key(SERVER_A1, "w1"),
        SignalKind::WorkerHealth(worker_health_body(SERVER_A1, "w1", WorkerStatus::Ready)),
        30_000,
    )
    .unwrap();
    a.publish_signal(
        worker_load_key(SERVER_A1, "w1"),
        SignalKind::WorkerLoad(Box::new(worker_load_body(
            SERVER_A1,
            "w1",
            "cohere.command-r-plus",
            4,
        ))),
        30_000,
    )
    .unwrap();
    a.publish_signal(
        client_latency_key(SERVER_A1),
        SignalKind::ClientLatency(client_latency_body(SERVER_A1, 80, 250)),
        30_000,
    )
    .unwrap();

    let (cursor, _) = pull_and_apply(&a, &b, 0);
    assert!(cursor > 0, "cursor must advance after initial snapshot");

    let state = b.state();
    let state = state.read();
    assert!(
        state
            .readiness_replica(REGION_A, SERVER_A1)
            .expect("readiness materialized")
            .ready,
    );
    assert_eq!(
        state
            .worker_health_replica(REGION_A, "w1", SERVER_A1)
            .expect("worker health materialized")
            .status,
        WorkerStatus::Ready,
    );
    assert_eq!(
        state
            .worker_load_replica(REGION_A, "w1", SERVER_A1)
            .expect("worker load materialized")
            .load
            .load,
        4,
    );
    assert_eq!(
        state
            .client_latency_replica(REGION_A, REGION_B, SERVER_A1)
            .expect("client latency materialized")
            .p50_latency_ms,
        80,
    );
}

// -------------------------------------------------------------------------
// 2. idempotent apply (`(version, actor)` equality is a no-op)
// -------------------------------------------------------------------------

#[test]
fn idempotent_apply_same_envelope_no_op() {
    let a = service(REGION_A, SERVER_A1);
    let b = service(REGION_B, SERVER_B);

    a.publish_signal(
        readiness_key(SERVER_A1),
        SignalKind::SmgReadiness(readiness_body(SERVER_A1, true)),
        30_000,
    )
    .unwrap();
    let (envs, _) = a.local_log_snapshot();

    // Apply twice.
    b.apply_remote_envelopes(a.region_id(), &envs);
    b.apply_remote_envelopes(a.region_id(), &envs);

    // The second apply must be a no-op: state remains the same.
    let state = b.state();
    let state = state.read();
    let (signal, version) = state
        .readiness_replica_with_version(REGION_A, SERVER_A1)
        .expect("readiness materialized");
    assert!(signal.ready);
    // The version's actor field is the writing replica's server_name, set
    // exactly once. A second apply with identical (version, actor) is a
    // no-op so this value matches the publisher.
    assert_eq!(version.actor, SERVER_A1);
}

// -------------------------------------------------------------------------
// 3. older-version envelopes rejected after newer ones observed
// -------------------------------------------------------------------------

#[test]
fn older_version_rejected_after_newer_observed() {
    let a = service(REGION_A, SERVER_A1);
    let b = service(REGION_B, SERVER_B);

    // Publish ready=true twice — versions are monotone, so the second is
    // the "newer" one.
    a.publish_signal(
        readiness_key(SERVER_A1),
        SignalKind::SmgReadiness(readiness_body(SERVER_A1, true)),
        30_000,
    )
    .unwrap();
    a.publish_signal(
        readiness_key(SERVER_A1),
        SignalKind::SmgReadiness(readiness_body(SERVER_A1, false)),
        30_000,
    )
    .unwrap();

    let (envs, _) = a.local_log_snapshot();
    assert_eq!(envs.len(), 2, "two publish calls produce two envelopes");

    // Apply the *newer* envelope first.
    b.apply_remote_envelopes(a.region_id(), &envs[1..]);
    {
        let state = b.state();
        let state = state.read();
        assert!(!state.readiness_replica(REGION_A, SERVER_A1).unwrap().ready);
    }

    // Now apply the older envelope — must be rejected.
    b.apply_remote_envelopes(a.region_id(), &envs[..1]);
    let state = b.state();
    let state = state.read();
    assert!(
        !state.readiness_replica(REGION_A, SERVER_A1).unwrap().ready,
        "older-version apply must not overwrite newer-version state",
    );
}

// -------------------------------------------------------------------------
// 4. multi-replica signals in one region survive materialization
// -------------------------------------------------------------------------

#[test]
fn same_region_multi_replica_signals_do_not_overwrite() {
    // Two SMG replicas in REGION_A each publish their own readiness +
    // worker-load. B pulls from both. The materialized state on B must
    // keep both replica entries, and `RemoteRegionView` must aggregate them.
    let a1 = service(REGION_A, SERVER_A1);
    let a2 = service(REGION_A, SERVER_A2);
    let b = service(REGION_B, SERVER_B);

    a1.publish_signal(
        readiness_key(SERVER_A1),
        SignalKind::SmgReadiness(readiness_body(SERVER_A1, true)),
        30_000,
    )
    .unwrap();
    a2.publish_signal(
        readiness_key(SERVER_A2),
        SignalKind::SmgReadiness(readiness_body(SERVER_A2, false)),
        30_000,
    )
    .unwrap();

    a1.publish_signal(
        worker_load_key(SERVER_A1, "w1"),
        SignalKind::WorkerLoad(Box::new(worker_load_body(
            SERVER_A1,
            "w1",
            "cohere.command-r-plus",
            3,
        ))),
        30_000,
    )
    .unwrap();
    a2.publish_signal(
        worker_load_key(SERVER_A2, "w1"),
        SignalKind::WorkerLoad(Box::new(worker_load_body(
            SERVER_A2,
            "w1",
            "cohere.command-r-plus",
            5,
        ))),
        30_000,
    )
    .unwrap();

    let (envs1, _) = a1.local_log_snapshot();
    let (envs2, _) = a2.local_log_snapshot();
    b.apply_remote_envelopes(REGION_A, &envs1);
    b.apply_remote_envelopes(REGION_A, &envs2);

    let state = b.state();
    let state = state.read();
    assert!(state.readiness_replica(REGION_A, SERVER_A1).unwrap().ready);
    assert!(!state.readiness_replica(REGION_A, SERVER_A2).unwrap().ready);

    let view = RemoteRegionView::new(&state, now_ms(), 60_000);
    let projection = view.readiness(REGION_A).expect("fresh replicas observed");
    assert!(
        projection.ready,
        "any-fresh-ready aggregation: at least one replica reports ready",
    );

    let worker = view.worker(REGION_A, "w1").expect("worker observed");
    let load = worker.load_for_model("cohere.command-r-plus");
    assert_eq!(
        load.total,
        Some(8),
        "loads from two replicas sum without double-counting (3 + 5)",
    );
}

// -------------------------------------------------------------------------
// 5. per-replica tombstone removes only the addressed replica
// -------------------------------------------------------------------------

#[test]
fn single_replica_tombstone_does_not_remove_sibling_replica() {
    let a1 = service(REGION_A, SERVER_A1);
    let a2 = service(REGION_A, SERVER_A2);
    let b = service(REGION_B, SERVER_B);

    a1.publish_signal(
        worker_health_key(SERVER_A1, "w1"),
        SignalKind::WorkerHealth(worker_health_body(SERVER_A1, "w1", WorkerStatus::Ready)),
        30_000,
    )
    .unwrap();
    a2.publish_signal(
        worker_health_key(SERVER_A2, "w1"),
        SignalKind::WorkerHealth(worker_health_body(SERVER_A2, "w1", WorkerStatus::Ready)),
        30_000,
    )
    .unwrap();

    let (envs1, _) = a1.local_log_snapshot();
    let (envs2, _) = a2.local_log_snapshot();
    b.apply_remote_envelopes(REGION_A, &envs1);
    b.apply_remote_envelopes(REGION_A, &envs2);

    // A1 tombstones its own entry; A2's must survive.
    a1.remove_signal(worker_health_key(SERVER_A1, "w1"))
        .unwrap();
    let (tombstones, _) = a1.local_log_snapshot();
    b.apply_remote_envelopes(REGION_A, &tombstones);

    let state = b.state();
    let state = state.read();
    assert!(
        state
            .worker_health_replica(REGION_A, "w1", SERVER_A1)
            .is_none(),
        "A1's tombstone removes its own entry",
    );
    assert!(
        state
            .worker_health_replica(REGION_A, "w1", SERVER_A2)
            .is_some(),
        "A2's replica must survive A1's tombstone",
    );
}

// -------------------------------------------------------------------------
// 6. stale cursor → full-snapshot resync via cursor-0 path
// -------------------------------------------------------------------------

#[test]
fn stale_cursor_triggers_full_resync() {
    // CursorStale fires when an entry the consumer has not yet observed is
    // GC'd before they pull it. Setup:
    //   * Consumer's cursor = 1 (saw entry 1).
    //   * Producer publishes entry 2 (which the consumer hasn't seen).
    //   * Sleep so entry 2's retention window can close.
    //   * Producer publishes entry 3 (fresh).
    //   * GC drops entries 1 + 2, keeps entry 3.
    //   * Consumer delta(since=1): oldest is 3, gap to 2 is lost → 409.
    let a = service_with_retention(
        REGION_A,
        SERVER_A1,
        SyncRetention {
            tombstone_retention_ms: 10,
            dead_replica_retention_ms: 10,
        },
    );
    let b = service(REGION_B, SERVER_B);

    a.publish_signal(
        readiness_key(SERVER_A1),
        SignalKind::SmgReadiness(readiness_body(SERVER_A1, true)),
        30_000,
    )
    .unwrap();
    let (cursor_after_first_pull, resync_first) = pull_and_apply(&a, &b, 0);
    assert!(!resync_first, "initial pull is not a resync");

    // Entry 2 — consumer never observes this one directly; GC will drop it.
    a.publish_signal(
        readiness_key(SERVER_A1),
        SignalKind::SmgReadiness(readiness_body(SERVER_A1, false)),
        30_000,
    )
    .unwrap();

    // Sleep long enough for entries 1 + 2's retention windows to close,
    // then publish entry 3 fresh.
    std::thread::sleep(std::time::Duration::from_millis(50));
    a.publish_signal(
        readiness_key(SERVER_A1),
        SignalKind::SmgReadiness(readiness_body(SERVER_A1, true)),
        30_000,
    )
    .unwrap();

    let (envs_before_gc, _) = a.local_log_snapshot();
    assert_eq!(
        envs_before_gc.len(),
        3,
        "three publishes produce three entries"
    );
    let first_ts = envs_before_gc[0].generated_at_ms;
    // GC at t1 + 30ms: entries 1 + 2 are well past their 10ms window;
    // entry 3 (published after a 50ms sleep) is still fresh.
    a.gc_log(first_ts + 30);

    let (envs_after_gc, _) = a.local_log_snapshot();
    assert_eq!(
        envs_after_gc.len(),
        1,
        "GC drops entries 1 + 2 (windows closed); entry 3 is still inside its window",
    );

    let (cursor_after_resync, resynced) = pull_and_apply(&a, &b, cursor_after_first_pull);
    assert!(
        resynced,
        "stale cursor must trigger snapshot resync: a needed entry (entry 2) was GC'd"
    );
    assert!(
        cursor_after_resync > cursor_after_first_pull,
        "cursor advances past resync",
    );

    let state = b.state();
    let state = state.read();
    assert!(
        state.readiness_replica(REGION_A, SERVER_A1).unwrap().ready,
        "post-resync state must reflect the newest publish (entry 3, ready=true)",
    );
}

// -------------------------------------------------------------------------
// 7. RemoteRegionView freshness window filters stale entries
// -------------------------------------------------------------------------

#[test]
fn freshness_window_filters_stale_entries() {
    let a = service(REGION_A, SERVER_A1);
    let b = service(REGION_B, SERVER_B);

    a.publish_signal(
        readiness_key(SERVER_A1),
        SignalKind::SmgReadiness(readiness_body(SERVER_A1, true)),
        30_000,
    )
    .unwrap();
    let (envs, _) = a.local_log_snapshot();
    let publish_ts = envs[0].generated_at_ms;
    b.apply_remote_envelopes(a.region_id(), &envs);

    // Use a "now" that's older than the freshness window relative to the
    // publish timestamp.
    let state_handle = b.state();
    let state_guard = state_handle.read();
    let view = RemoteRegionView::new(
        &state_guard,
        publish_ts + 60_000,
        30_000, // 30s freshness window
    );
    assert!(
        view.readiness(REGION_A).is_none(),
        "stale entry (age > window) must not project",
    );
    assert!(
        view.has_readiness_replica(REGION_A),
        "the entry is still materialized; only the projection filters it",
    );

    // A "now" within the window does project.
    let fresh_view = RemoteRegionView::new(&state_guard, publish_ts + 5_000, 30_000);
    assert!(
        fresh_view.readiness(REGION_A).is_some(),
        "fresh entry must project under the same window",
    );
}

// -------------------------------------------------------------------------
// 8. candidate ranking + RemoteRegionView project consistently
// -------------------------------------------------------------------------

#[test]
fn remote_region_view_projects_consistently_with_candidate_ranking() {
    // Same materialized state feeds both `RemoteRegionView` directly and
    // `CandidateCalculator::build_candidates`. The view's projections must
    // match the candidate's published readiness/load/freshness.
    let a = service(REGION_A, SERVER_A1);
    let b = service(REGION_B, SERVER_B);

    a.publish_signal(
        readiness_key(SERVER_A1),
        SignalKind::SmgReadiness(readiness_body(SERVER_A1, true)),
        30_000,
    )
    .unwrap();
    a.publish_signal(
        worker_health_key(SERVER_A1, "w1"),
        SignalKind::WorkerHealth(worker_health_body(SERVER_A1, "w1", WorkerStatus::Ready)),
        30_000,
    )
    .unwrap();
    a.publish_signal(
        worker_load_key(SERVER_A1, "w1"),
        SignalKind::WorkerLoad(Box::new(worker_load_body(
            SERVER_A1,
            "w1",
            "cohere.command-r-plus",
            6,
        ))),
        30_000,
    )
    .unwrap();

    let (envs, _) = a.local_log_snapshot();
    let now = envs[0].generated_at_ms + 1_000;
    b.apply_remote_envelopes(a.region_id(), &envs);

    let state = b.state();
    let state = state.read();

    // RemoteRegionView projection.
    let view = RemoteRegionView::new(&state, now, 30_000);
    let view_readiness = view.readiness(REGION_A).expect("readiness projected");
    let view_worker = view.worker(REGION_A, "w1").expect("worker projected");
    let view_load = view_worker.load_for_model("cohere.command-r-plus");

    // Candidate ranking against the same state.
    let local_registry = smg::worker::WorkerRegistry::new();
    let peers = peer_registry(&[REGION_A]);
    let profile = profile_for(&[REGION_B, REGION_A], "cohere.command-r-plus");
    let breaker = CrossRegionBreaker::new();
    let calculator = CandidateCalculator::default();
    let input = CandidateCalculationInput {
        profile,
        local_region: REGION_B.to_string(),
        endpoint_type: Endpoint::Chat,
        local_worker_registry: &local_registry,
        remote_state: &state,
        peer_registry: &peers,
        breaker: &breaker,
        client_region: Some(REGION_B.to_string()),
        now_ms: now,
    };
    let output = calculator
        .build_candidates(input)
        .expect("calculator builds");
    let remote = output
        .candidates
        .iter()
        .find(|c| c.region_id == REGION_A)
        .expect("remote candidate produced");

    assert_eq!(
        remote.readiness, view_readiness.ready,
        "candidate readiness matches view readiness",
    );
    assert_eq!(
        remote.worker_load, view_load.total,
        "candidate load matches view sum",
    );
    assert!(
        remote.has_capacity,
        "remote worker is routable in both views"
    );
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

#[expect(clippy::expect_used, reason = "test helper — fixture is known-valid")]
fn peer_registry(target_regions: &[&str]) -> RegionPeerRegistry {
    let peers: Vec<RegionPeer> = target_regions
        .iter()
        .map(|region| {
            RegionPeer::new(
                region.to_string(),
                format!("https://smg-{region}.internal:8443"),
                format!("https://smg-{region}.internal:9443"),
                "oc1",
                "prod",
                None,
            )
            .expect("peer construction")
        })
        .collect();
    RegionPeerRegistry::new(peers).expect("peer registry")
}

#[expect(clippy::expect_used, reason = "test helper — fixture is known-valid")]
fn profile_for(allowed_regions: &[&str], model_id: &str) -> RoutingProfileContext {
    RoutingProfileContext::new(
        allowed_regions.iter().map(|r| (*r).to_string()).collect(),
        vec![model_id.to_string()],
        FailoverPolicy::new(CrossRegionFailoverMode::Manual, 1),
        ModalityPolicy::default(),
    )
    .expect("profile fixture is valid")
}
