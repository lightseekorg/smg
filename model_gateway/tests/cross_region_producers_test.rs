//! End-to-end test for the four cross-region local-signal producers, now
//! backed by an in-process mesh broadcast stream namespace. Drives a
//! `CrossRegionProducers` orchestrator end-to-end with a real
//! `WorkerRegistry`, exercises lifecycle events, records latency
//! observations, and asserts the resulting envelopes show up in the
//! producer's outbox (= the entries mesh would ship on the next gossip
//! round) with the right shape, version monotonicity, and worker-lifecycle
//! semantics.

use std::{sync::Arc, time::Duration};

use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use smg::{
    cross_region::{
        CrossRegionProducers, CrossRegionSyncService, ProducerCadences, SignalEnvelope, SignalKey,
        SignalKind, CROSS_REGION_NAMESPACE_PREFIX,
    },
    worker::{event::WorkerEvent, BasicWorkerBuilder, Worker, WorkerRegistry},
};
use smg_mesh::{MeshKV, StreamConfig, StreamRouting};

const REGION: &str = "us-ashburn-1";
const SERVER: &str = "smg-router-a";

fn make_registry() -> Arc<WorkerRegistry> {
    Arc::new(WorkerRegistry::new())
}

fn build_worker(url: &str, model: &str, status: WorkerStatus, load: usize) -> Arc<dyn Worker> {
    let worker = Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new(model))
            .status(status)
            .build(),
    );
    for _ in 0..load {
        worker.increment_load();
    }
    worker
}

#[expect(clippy::expect_used, reason = "test helper — fixture is known-valid")]
fn make_producers() -> CrossRegionProducers {
    let mesh_kv = Arc::new(MeshKV::new(SERVER.to_string()));
    let namespace = mesh_kv.configure_stream_prefix(
        CROSS_REGION_NAMESPACE_PREFIX,
        StreamConfig {
            max_buffer_bytes: 16 * 1024 * 1024,
            routing: StreamRouting::Broadcast,
        },
    );
    CrossRegionProducers::new(REGION.to_string(), SERVER.to_string(), namespace)
        .expect("producers should construct")
}

/// Snapshot every envelope currently staged in the producer's outbox (i.e.
/// the entries mesh would ship on the next gossip round). Tombstones do not
/// appear — `remove_signal` purges the outbox locally rather than emitting
/// a wire entry; peers learn that the signal is gone via the freshness
/// window after the producer stops re-emitting.
fn live_envelopes(sync: &CrossRegionSyncService) -> Vec<SignalEnvelope<SignalKind>> {
    let mut out = sync.outbox_snapshot();
    out.sort_by_key(|env| env.key.as_path());
    out
}

#[test]
fn producers_publish_per_replica_keys_for_all_four_signals() {
    let producers = make_producers();
    let registry = make_registry();

    let worker = build_worker(
        "http://w1:8000",
        "cohere.command-r-plus",
        WorkerStatus::Ready,
        7,
    );
    let worker_id = registry.register(worker.clone()).expect("register");

    // 1. Region readiness.
    producers
        .region_readiness
        .publish_ready(true)
        .expect("readiness publish");

    // 2. Worker health.
    producers
        .worker_health
        .publish_for(worker_id.as_str(), WorkerStatus::Ready)
        .expect("worker-health publish");

    // 3. Worker load (via reconcile to exercise the registry path).
    producers.worker_load.reconcile(&registry);

    // 4. Client latency.
    producers.client_latency.record_latency("us-chicago-1", 42);
    producers
        .client_latency
        .drain_and_publish()
        .expect("client-latency drain");

    let entries = live_envelopes(&producers.sync);
    assert_eq!(entries.len(), 4, "one envelope per signal kind");

    let mut kinds: Vec<&str> = entries
        .iter()
        .map(|e| match e.signal.as_ref().expect("signal body") {
            SignalKind::SmgReadiness(_) => "readiness",
            SignalKind::WorkerHealth(_) => "worker-health",
            SignalKind::WorkerLoad(_) => "worker-load",
            SignalKind::ClientLatency(_) => "client-latency",
        })
        .collect();
    kinds.sort_unstable();
    assert_eq!(
        kinds,
        vec![
            "client-latency",
            "readiness",
            "worker-health",
            "worker-load"
        ]
    );

    for env in &entries {
        assert_eq!(
            env.actor, SERVER,
            "every envelope's actor is local server_name"
        );
        assert_eq!(
            env.key.region_segment(),
            REGION,
            "every key's region matches local region: {:?}",
            env.key,
        );
        assert_eq!(
            env.key.server_name_segment(),
            SERVER,
            "every key's trailing server_name matches local replica: {:?}",
            env.key,
        );
    }

    for env in &entries {
        match (&env.key, env.signal.as_ref()) {
            (
                SignalKey::WorkerHealth {
                    worker_id: key_wid, ..
                },
                Some(SignalKind::WorkerHealth(body)),
            ) => {
                assert_eq!(key_wid, &body.worker_id);
                assert_eq!(key_wid, worker_id.as_str());
            }
            (
                SignalKey::WorkerLoad {
                    worker_id: key_wid, ..
                },
                Some(SignalKind::WorkerLoad(body)),
            ) => {
                assert_eq!(key_wid, &body.worker_id);
                assert_eq!(key_wid, worker_id.as_str());
            }
            _ => {}
        }
    }
}

#[test]
fn lifecycle_event_publishes_then_tombstones() {
    let producers = make_producers();
    let registry = make_registry();

    let worker = build_worker(
        "http://w1:8000",
        "cohere.command-r-plus",
        WorkerStatus::Pending,
        0,
    );
    let worker_id = registry.register(worker.clone()).expect("register");

    // Drive a Registered → StatusChanged → Removed transition through the
    // adapter (mirrors what the orchestrator's event subscriber does).
    producers
        .worker_health
        .handle_event(&WorkerEvent::Registered {
            worker_id: worker_id.clone(),
            worker: worker.clone(),
        })
        .expect("registered");
    let after_register = live_envelopes(&producers.sync);
    assert_eq!(after_register.len(), 1);
    let v_register = after_register[0].version;

    worker.set_status(WorkerStatus::Ready);
    producers
        .worker_health
        .handle_event(&WorkerEvent::StatusChanged {
            worker_id: worker_id.clone(),
            worker: worker.clone(),
            old_status: WorkerStatus::Pending,
            new_status: WorkerStatus::Ready,
        })
        .expect("status changed");
    let after_change = live_envelopes(&producers.sync);
    assert_eq!(after_change.len(), 1, "same key collapses to one envelope");
    assert!(
        after_change[0].version > v_register,
        "version monotonically increases on status change"
    );

    producers
        .worker_health
        .handle_event(&WorkerEvent::Removed {
            worker_id: worker_id.clone(),
            worker: worker.clone(),
        })
        .expect("removed");

    // Removal purges the outbox locally (peers age out via the freshness
    // window once the producer stops re-emitting) and drops the entry from
    // materialized state immediately.
    assert!(
        live_envelopes(&producers.sync).is_empty(),
        "remove_signal clears the outbox locally"
    );
    let state = producers.sync.state();
    let state = state.read();
    assert!(state
        .worker_health_replica(REGION, worker_id.as_str(), SERVER)
        .is_none());
}

#[test]
fn version_strictly_increases_per_key_across_publishes() {
    let producers = make_producers();

    producers.region_readiness.publish_ready(true).unwrap();
    producers
        .client_latency
        .publish_for("us-chicago-1", 30, 80)
        .unwrap();

    let after_first = live_envelopes(&producers.sync);
    assert_eq!(after_first.len(), 2);

    // Same key (us-chicago-1 latency), new sample → version must rise.
    let latency_v1 = after_first
        .iter()
        .find(|e| matches!(e.signal, Some(SignalKind::ClientLatency(_))))
        .expect("latency envelope present")
        .version;

    producers
        .client_latency
        .publish_for("us-chicago-1", 32, 84)
        .unwrap();
    let after_second = live_envelopes(&producers.sync);
    let latency_v2 = after_second
        .iter()
        .find(|e| matches!(e.signal, Some(SignalKind::ClientLatency(_))))
        .expect("latency envelope present")
        .version;
    assert!(latency_v2 > latency_v1);

    // Wrong region in the key should be rejected by the sync service and
    // never reach the outbox.
    let err = producers.sync.publish_signal(
        SignalKey::ClientLatency {
            client_region: "wrong-region".to_string(),
            target_region: "us-chicago-1".to_string(),
            server_name: SERVER.to_string(),
        },
        SignalKind::ClientLatency(smg::cross_region::ClientLatencySignal {
            client_region: "wrong-region".to_string(),
            target_region: "us-chicago-1".to_string(),
            server_name: SERVER.to_string(),
            p50_latency_ms: 30,
            p95_latency_ms: 80,
        }),
        30_000,
    );
    err.expect_err("wrong region must be rejected at the sync service");
    // Outbox footprint unchanged (still readiness + one latency key).
    assert_eq!(live_envelopes(&producers.sync).len(), 2);
}

#[tokio::test]
async fn orchestrator_start_publishes_via_periodic_tasks() {
    let cadences = ProducerCadences {
        readiness_reconcile_interval: Duration::from_millis(50),
        worker_health_reconcile_interval: Duration::from_millis(50),
        worker_load_refresh_interval: Duration::from_millis(50),
        client_latency_publish_interval: Duration::from_millis(50),
    };
    let producers = make_producers();
    let registry = make_registry();

    let worker = build_worker(
        "http://w1:8000",
        "cohere.command-r-plus",
        WorkerStatus::Ready,
        3,
    );
    registry.register(worker).expect("register");

    producers.client_latency.record_latency("us-chicago-1", 25);

    let _handles = producers.start(registry.clone(), cadences);

    tokio::time::sleep(Duration::from_millis(250)).await;

    let entries = live_envelopes(&producers.sync);

    let has_readiness = entries
        .iter()
        .any(|e| matches!(e.signal, Some(SignalKind::SmgReadiness(_))));
    let has_worker_load = entries
        .iter()
        .any(|e| matches!(e.signal, Some(SignalKind::WorkerLoad(_))));
    let has_worker_health = entries
        .iter()
        .any(|e| matches!(e.signal, Some(SignalKind::WorkerHealth(_))));
    let has_client_latency = entries
        .iter()
        .any(|e| matches!(e.signal, Some(SignalKind::ClientLatency(_))));

    assert!(has_readiness, "periodic readiness publish");
    assert!(has_worker_load, "periodic worker-load publish");
    assert!(
        has_worker_health,
        "worker-health reconcile loop publishes for the registered worker"
    );
    assert!(
        has_client_latency,
        "periodic client-latency drain publishes recorded sample"
    );
}

#[tokio::test]
async fn orchestrator_publishes_worker_health_on_registry_event() {
    let producers = make_producers();
    let registry = make_registry();

    let cadences = ProducerCadences {
        readiness_reconcile_interval: Duration::from_secs(3600),
        worker_health_reconcile_interval: Duration::from_secs(3600),
        worker_load_refresh_interval: Duration::from_secs(3600),
        client_latency_publish_interval: Duration::from_secs(3600),
    };
    let _handles = producers.start(registry.clone(), cadences);

    tokio::time::sleep(Duration::from_millis(20)).await;

    let worker = build_worker(
        "http://w1:8000",
        "cohere.command-r-plus",
        WorkerStatus::Ready,
        0,
    );
    registry.register(worker).expect("register");

    tokio::time::sleep(Duration::from_millis(50)).await;

    let entries = live_envelopes(&producers.sync);
    let worker_health_count = entries
        .iter()
        .filter(|e| matches!(e.signal, Some(SignalKind::WorkerHealth(_))))
        .count();
    assert!(
        worker_health_count >= 1,
        "event-driven path must publish at least one worker-health envelope; got {worker_health_count}",
    );
}
