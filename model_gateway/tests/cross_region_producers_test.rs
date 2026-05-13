//! End-to-end test for the four cross-region local-signal producers.
//!
//! Drives a `CrossRegionProducers` orchestrator end-to-end with a real
//! `WorkerRegistry`: registers workers, exercises lifecycle events, records
//! latency observations, and asserts the resulting envelopes show up in the
//! sync service's producer log with the right shape, version ordering, and
//! tombstone semantics. The HTTP pull endpoint that serves the log to peers
//! is out of scope here (Phase 4 wire-side).

use std::{sync::Arc, time::Duration};

use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use smg::{
    cross_region::{CrossRegionProducers, ProducerCadences, SignalKey, SignalKind},
    worker::{event::WorkerEvent, BasicWorkerBuilder, Worker, WorkerRegistry},
};

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

#[test]
fn producers_publish_per_replica_keys_for_all_four_signals() {
    let producers =
        CrossRegionProducers::new(REGION.to_string(), SERVER.to_string()).expect("construct");
    let registry = make_registry();

    // Local worker via the registry (drives worker-health Registered event
    // if we use a subscriber — here we drive adapters directly).
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

    let (entries, cursor) = producers.sync.local_log_snapshot();
    assert!(cursor > 0, "log cursor advanced");
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

    // Each envelope's actor is this replica's server_name; key segments
    // match the local region + server_name + (worker_id where applicable).
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

    // Worker-load and worker-health both carry the same worker_id as the key.
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
    let producers =
        CrossRegionProducers::new(REGION.to_string(), SERVER.to_string()).expect("construct");
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
    producers
        .worker_health
        .handle_event(&WorkerEvent::Removed {
            worker_id: worker_id.clone(),
            worker: worker.clone(),
        })
        .expect("removed");

    let (entries, _) = producers.sync.local_log_snapshot();
    assert_eq!(entries.len(), 3);

    // Versions strictly increasing — every later envelope outranks every earlier.
    for window in entries.windows(2) {
        assert!(
            window[1].version > window[0].version,
            "envelope versions must be strictly increasing: {:?}",
            entries.iter().map(|e| e.version).collect::<Vec<_>>(),
        );
    }

    // Last envelope is a tombstone with no signal body and stale_after_ms = 0.
    let tombstone = entries.last().expect("at least one entry");
    assert!(tombstone.removed);
    assert!(tombstone.signal.is_none());
    assert_eq!(tombstone.stale_after_ms, 0);

    // Tombstones remove the materialized worker immediately.
    let state = producers.sync.state();
    let state = state.read();
    let health = state.worker_health_replica(REGION, worker_id.as_str(), SERVER);
    assert!(health.is_none());
}

#[test]
fn cursor_delta_streams_envelopes_in_order() {
    let producers =
        CrossRegionProducers::new(REGION.to_string(), SERVER.to_string()).expect("construct");

    let (_initial, c0) = producers.sync.local_log_snapshot();

    producers.region_readiness.publish_ready(true).unwrap();
    producers
        .client_latency
        .publish_for("us-chicago-1", 30, 80)
        .unwrap();

    let (delta, c1) = producers.sync.local_log_delta(c0).expect("cursor is fresh");
    assert_eq!(delta.len(), 2);
    assert!(c1 > c0);

    // Subsequent publish — delta from c1 returns just the new one.
    producers
        .client_latency
        .publish_for("us-phoenix-1", 5, 12)
        .unwrap();
    let (delta2, c2) = producers.sync.local_log_delta(c1).expect("cursor ok");
    assert_eq!(delta2.len(), 1);
    assert!(c2 > c1);

    // Wrong region in the key should be rejected, so the log doesn't grow.
    let len_before = producers.sync.local_log_snapshot().0.len();
    let err = producers
        .region_readiness
        .publish_ready(true)
        .and_then(|()| {
            // Build a deliberately wrong-region client-latency key by going
            // around the adapter, to confirm the sync service rejects.
            producers.sync.publish_signal(
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
            )
        });
    err.expect_err("wrong region must be rejected at the sync service");
    // The legitimate readiness publish above did succeed, so log grew by 1.
    assert_eq!(producers.sync.local_log_snapshot().0.len(), len_before + 1);
}

#[tokio::test]
async fn orchestrator_start_publishes_via_periodic_tasks() {
    // Use tight cadences so the test doesn't drag.
    let cadences = ProducerCadences {
        readiness_reconcile_interval: Duration::from_millis(50),
        worker_health_reconcile_interval: Duration::from_millis(50),
        worker_load_refresh_interval: Duration::from_millis(50),
        client_latency_publish_interval: Duration::from_millis(50),
    };
    let producers =
        CrossRegionProducers::new(REGION.to_string(), SERVER.to_string()).expect("construct");
    let registry = make_registry();

    // Register one worker before starting the tasks so reconcile sees it.
    let worker = build_worker(
        "http://w1:8000",
        "cohere.command-r-plus",
        WorkerStatus::Ready,
        3,
    );
    registry.register(worker).expect("register");

    // Record some latency so the client-latency drain has data to publish.
    producers.client_latency.record_latency("us-chicago-1", 25);

    let _handles = producers.start(registry.clone(), cadences);

    // Let the periodic loops tick at least twice each (cadence is 50ms; wait
    // 250ms for safety against scheduling jitter).
    tokio::time::sleep(Duration::from_millis(250)).await;

    let (entries, _) = producers.sync.local_log_snapshot();

    // We should see at least one envelope per periodic-driven signal kind.
    // (worker-health event loop also picks up the Registered broadcast that
    // was emitted before the loop started, but BroadcastStream is lossy on
    // missed events; the reconcile loop covers the gap.)
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
    // Drive the event-driven path explicitly: subscribe-event loop is started,
    // then a worker registers, and we expect a worker-health envelope.
    let producers =
        CrossRegionProducers::new(REGION.to_string(), SERVER.to_string()).expect("construct");
    let registry = make_registry();

    // Start with no workers so the reconcile loop on its own would publish
    // nothing — only the event-driven path produces the envelope.
    let cadences = ProducerCadences {
        readiness_reconcile_interval: Duration::from_secs(3600),
        worker_health_reconcile_interval: Duration::from_secs(3600),
        worker_load_refresh_interval: Duration::from_secs(3600),
        client_latency_publish_interval: Duration::from_secs(3600),
    };
    let _handles = producers.start(registry.clone(), cadences);

    // Let the event subscriber attach.
    tokio::time::sleep(Duration::from_millis(20)).await;

    let worker = build_worker(
        "http://w1:8000",
        "cohere.command-r-plus",
        WorkerStatus::Ready,
        0,
    );
    registry.register(worker).expect("register");

    // Let the event ride through the broadcast channel into the adapter.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let (entries, _) = producers.sync.local_log_snapshot();
    let worker_health_count = entries
        .iter()
        .filter(|e| matches!(e.signal, Some(SignalKind::WorkerHealth(_))))
        .count();
    assert!(
        worker_health_count >= 1,
        "event-driven path must publish at least one worker-health envelope; got {worker_health_count}",
    );
}
