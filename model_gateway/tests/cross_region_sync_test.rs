//! Cross-region sync plane acceptance tests over the mesh-backed sync service.
//!
//! Each test wires two `CrossRegionSyncService` instances (each with its own
//! in-process `MeshKV`) and drives the publish → remote-apply path by
//! snapshotting A's outbox (the entries mesh would ship next round) and
//! calling `apply_envelope_to_state` directly into B's materialized state.
//! That short-circuits mesh gossip (which we don't exercise in-process) but
//! covers everything the production subscriber task does once a peer's
//! envelope arrives: decode, validate, run the `(version, actor)` apply
//! check, and update `CrossRegionState`.
//!
//! Acceptance criteria mapped from the original plan:
//!   1. local publish on A round-trips to B's materialized state
//!   2. idempotent apply (`(version, actor)` equality is a no-op)
//!   3. older-version envelopes rejected after newer ones observed
//!   4. multi-replica signals in one region survive materialization
//!   5. removing a producer's outbox entry doesn't disturb sibling replicas
//!   6. `RemoteRegionView` freshness window filters stale entries
//!   7. candidate ranking + `RemoteRegionView` project consistently
//!
//! With broadcast streams there is no over-the-wire tombstone: peers learn
//! that a worker is gone by the producer no longer re-emitting and the
//! consumer's freshness window dropping the entry from projections.

use std::sync::Arc;

use openai_protocol::{
    model_type::Endpoint,
    worker::{WorkerLoadInfo, WorkerStatus},
};
use smg::{
    config::CrossRegionFailoverMode,
    cross_region::{
        apply_envelope_to_state, CandidateCalculationInput, CandidateCalculator,
        ClientLatencySignal, CrossRegionBreaker, CrossRegionSyncService, FailoverPolicy,
        ModalityPolicy, RegionPeer, RegionPeerRegistry, RemoteRegionView, RoutingProfileContext,
        SignalEnvelope, SignalKey, SignalKind, SmgReadinessSignal, WorkerHealthSignal,
        WorkerLoadSignal, CROSS_REGION_NAMESPACE_PREFIX,
    },
};
use smg_mesh::{MeshKV, StreamConfig, StreamRouting};

const REGION_A: &str = "us-ashburn-1";
const REGION_B: &str = "us-chicago-1";
const SERVER_A1: &str = "smg-router-a1";
const SERVER_A2: &str = "smg-router-a2";
const SERVER_B: &str = "smg-router-b";

#[expect(clippy::expect_used, reason = "test helper — fixture is known-valid")]
fn service(region: &str, server: &str) -> CrossRegionSyncService {
    let mesh_kv = Arc::new(MeshKV::new(server.to_string()));
    let namespace = mesh_kv.configure_stream_prefix(
        CROSS_REGION_NAMESPACE_PREFIX,
        StreamConfig {
            max_buffer_bytes: 16 * 1024 * 1024,
            routing: StreamRouting::Broadcast,
        },
    );
    CrossRegionSyncService::new(region.to_string(), server.to_string(), namespace)
        .expect("service should construct")
}

/// Snapshot every envelope currently staged in `producer`'s outbox (the
/// entries mesh would broadcast on its next gossip round), in stable key
/// order.
fn live_envelopes(producer: &CrossRegionSyncService) -> Vec<SignalEnvelope<SignalKind>> {
    let mut out = producer.outbox_snapshot();
    out.sort_by_key(|env| env.key.as_path());
    out
}

/// Apply every staged envelope from `producer`'s outbox into `consumer`'s
/// materialized state, mirroring what the production subscriber task does
/// once a peer's broadcast entry arrives.
fn ship(producer: &CrossRegionSyncService, consumer: &CrossRegionSyncService) {
    let state = consumer.state();
    let mut state = state.write();
    for envelope in live_envelopes(producer) {
        apply_envelope_to_state(&mut state, &envelope);
    }
}

/// Apply a tombstone for `key` to `consumer` — mirrors the subscriber's
/// behavior on a `(key, None)` event.
fn ship_tombstone(consumer: &CrossRegionSyncService, key: &SignalKey) {
    consumer.state().write().remove_key(key);
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

    ship(&a, &b);

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

    // Apply twice — mesh's CRDT collapses duplicate writes, but at the
    // application layer the `(version, actor)` check must still no-op.
    ship(&a, &b);
    ship(&a, &b);

    let state = b.state();
    let state = state.read();
    let (signal, version) = state
        .readiness_replica_with_version(REGION_A, SERVER_A1)
        .expect("readiness materialized");
    assert!(signal.ready);
    assert_eq!(version.actor, SERVER_A1);
}

// -------------------------------------------------------------------------
// 3. older-version envelopes rejected after newer ones observed
// -------------------------------------------------------------------------

#[test]
fn older_version_rejected_after_newer_observed() {
    let a = service(REGION_A, SERVER_A1);
    let b = service(REGION_B, SERVER_B);

    a.publish_signal(
        readiness_key(SERVER_A1),
        SignalKind::SmgReadiness(readiness_body(SERVER_A1, true)),
        30_000,
    )
    .unwrap();
    // Snapshot the first publish before it gets overwritten by the second.
    let first_envelope = live_envelopes(&a)
        .into_iter()
        .find(|e| matches!(e.signal, Some(SignalKind::SmgReadiness(_))))
        .expect("first readiness envelope present");

    a.publish_signal(
        readiness_key(SERVER_A1),
        SignalKind::SmgReadiness(readiness_body(SERVER_A1, false)),
        30_000,
    )
    .unwrap();

    // Ship the *newer* envelope first.
    ship(&a, &b);
    {
        let state = b.state();
        let state = state.read();
        assert!(!state.readiness_replica(REGION_A, SERVER_A1).unwrap().ready);
    }

    // Replay the older envelope directly — must be rejected.
    {
        let state = b.state();
        let mut state = state.write();
        apply_envelope_to_state(&mut state, &first_envelope);
    }
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

    ship(&a1, &b);
    ship(&a2, &b);

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

    ship(&a1, &b);
    ship(&a2, &b);

    // A1 tombstones its own entry; A2's must survive.
    a1.remove_signal(worker_health_key(SERVER_A1, "w1"))
        .unwrap();
    // Mesh delivers tombstones as `(key, None)` events — emulate that path
    // directly against B's state.
    ship_tombstone(&b, &worker_health_key(SERVER_A1, "w1"));

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
// 6. RemoteRegionView freshness window filters stale entries
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
    let envelopes = live_envelopes(&a);
    let publish_ts = envelopes
        .iter()
        .find(|e| matches!(e.signal, Some(SignalKind::SmgReadiness(_))))
        .expect("readiness envelope present")
        .generated_at_ms;
    ship(&a, &b);

    let state_handle = b.state();
    let state_guard = state_handle.read();
    let view = RemoteRegionView::new(&state_guard, publish_ts + 60_000, 30_000);
    assert!(
        view.readiness(REGION_A).is_none(),
        "stale entry (age > window) must not project",
    );
    assert!(
        view.has_readiness_replica(REGION_A),
        "the entry is still materialized; only the projection filters it",
    );

    let fresh_view = RemoteRegionView::new(&state_guard, publish_ts + 5_000, 30_000);
    assert!(
        fresh_view.readiness(REGION_A).is_some(),
        "fresh entry must project under the same window",
    );
}

// -------------------------------------------------------------------------
// 7. candidate ranking + RemoteRegionView project consistently
// -------------------------------------------------------------------------

#[test]
fn remote_region_view_projects_consistently_with_candidate_ranking() {
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

    let envelopes = live_envelopes(&a);
    let now = envelopes
        .iter()
        .map(|e| e.generated_at_ms)
        .min()
        .expect("at least one envelope")
        + 1_000;
    ship(&a, &b);

    let state = b.state();
    let state = state.read();

    let view = RemoteRegionView::new(&state, now, 30_000);
    let view_readiness = view.readiness(REGION_A).expect("readiness projected");
    let view_worker = view.worker(REGION_A, "w1").expect("worker projected");
    let view_load = view_worker.load_for_model("cohere.command-r-plus");

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
