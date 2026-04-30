//! Protocol-surface contract tests for worker load advisory metadata.

#[cfg(feature = "axum")]
use axum::{body::to_bytes, response::IntoResponse};
#[cfg(feature = "axum")]
use openai_protocol::worker::WorkerLoadsResult;
use openai_protocol::worker::{WorkerLoadInfo, WorkerLoadInfoSource, WorkerStatus};
use serde_json::json;

#[test]
fn old_worker_load_json_without_cross_region_fields_deserializes() {
    let load: WorkerLoadInfo = serde_json::from_value(json!({
        "worker": "http://127.0.0.1:8000",
        "load": 7
    }))
    .expect("old worker load JSON should deserialize");

    assert_eq!(load.worker, "http://127.0.0.1:8000");
    assert_eq!(load.load, 7);
    assert!(load.worker_type.is_none());
    assert!(load.details.is_none());
    assert!(load.region_id.is_none());
    assert!(load.worker_id.is_none());
    assert!(load.model_id.is_none());
    assert!(load.status.is_none());
    assert!(load.generated_at_ms.is_none());
    assert!(load.version.is_none());
    assert!(load.source.is_none());
    assert!(load.remote_workers.is_none());
}

#[test]
fn local_worker_load_serializes_existing_shape_when_advisory_fields_absent() {
    let load = WorkerLoadInfo {
        worker: "http://127.0.0.1:8000".to_string(),
        worker_type: None,
        load: 7,
        details: None,
        region_id: None,
        worker_id: None,
        model_id: None,
        status: None,
        generated_at_ms: None,
        version: None,
        source: None,
        remote_workers: None,
    };

    let serialized = serde_json::to_value(&load).expect("worker load should serialize");

    assert_eq!(
        serialized,
        json!({
            "worker": "http://127.0.0.1:8000",
            "load": 7
        })
    );
}

#[test]
fn remote_smg_aggregate_entry_round_trips_remote_workers() {
    let aggregate = json!({
        "worker": "region-peer/us-chicago-1",
        "load": 11,
        "region_id": "us-chicago-1",
        "generated_at_ms": 1_714_000_000_000i64,
        "version": 3,
        "source": "remote_smg",
        "remote_workers": [
            {
                "worker": "remote-worker-a",
                "load": 11,
                "region_id": "us-chicago-1",
                "worker_id": "remote-worker-a",
                "model_id": "cohere.command-r-plus",
                "status": "ready",
                "generated_at_ms": 1_714_000_000_000i64,
                "version": 3,
                "source": "local_worker"
            }
        ]
    });

    let load: WorkerLoadInfo =
        serde_json::from_value(aggregate.clone()).expect("remote aggregate should deserialize");

    assert_eq!(load.worker, "region-peer/us-chicago-1");
    assert_eq!(load.source, Some(WorkerLoadInfoSource::RemoteSmg));
    let remote_workers = load
        .remote_workers
        .as_ref()
        .expect("remote aggregate should carry advisory workers");
    assert_eq!(remote_workers.len(), 1);
    assert_eq!(
        remote_workers[0].source,
        Some(WorkerLoadInfoSource::LocalWorker)
    );
    assert_eq!(remote_workers[0].status, Some(WorkerStatus::Ready));

    let serialized = serde_json::to_value(&load).expect("remote aggregate should serialize");
    assert_eq!(serialized, aggregate);
}

#[test]
fn worker_load_info_source_serde_names_are_stable() {
    let local = serde_json::to_value(WorkerLoadInfoSource::LocalWorker)
        .expect("local source should serialize");
    let remote = serde_json::to_value(WorkerLoadInfoSource::RemoteSmg)
        .expect("remote source should serialize");

    assert_eq!(local, json!("local_worker"));
    assert_eq!(remote, json!("remote_smg"));
    assert_eq!(
        serde_json::from_value::<WorkerLoadInfoSource>(json!("local_worker"))
            .expect("local source should deserialize"),
        WorkerLoadInfoSource::LocalWorker
    );
    assert_eq!(
        serde_json::from_value::<WorkerLoadInfoSource>(json!("remote_smg"))
            .expect("remote source should deserialize"),
        WorkerLoadInfoSource::RemoteSmg
    );
}

#[cfg(feature = "axum")]
#[tokio::test]
async fn worker_loads_response_preserves_remote_smg_advisory_fields() {
    let result = WorkerLoadsResult {
        loads: vec![WorkerLoadInfo {
            worker: "region-peer/us-chicago-1".to_string(),
            worker_type: Some("decode".to_string()),
            load: 11,
            details: None,
            region_id: Some("us-chicago-1".to_string()),
            worker_id: None,
            model_id: None,
            status: None,
            generated_at_ms: Some(1_714_000_000_000),
            version: Some(3),
            source: Some(WorkerLoadInfoSource::RemoteSmg),
            remote_workers: Some(vec![WorkerLoadInfo {
                worker: "remote-worker-a".to_string(),
                worker_type: Some("prefill".to_string()),
                load: 11,
                details: None,
                region_id: Some("us-chicago-1".to_string()),
                worker_id: Some("remote-worker-a".to_string()),
                model_id: Some("cohere.command-r-plus".to_string()),
                status: Some(WorkerStatus::Ready),
                generated_at_ms: Some(1_714_000_000_000),
                version: Some(3),
                source: Some(WorkerLoadInfoSource::LocalWorker),
                remote_workers: None,
            }]),
        }],
        total_workers: 1,
        successful: 1,
        failed: 0,
    };

    let response = result.into_response();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body should be readable");
    let payload: serde_json::Value =
        serde_json::from_slice(&body).expect("response body should be JSON");
    let aggregate = &payload["workers"][0];
    let remote_worker = &aggregate["remote_workers"][0];

    assert_eq!(aggregate["worker"], "region-peer/us-chicago-1");
    assert_eq!(aggregate["region_id"], "us-chicago-1");
    assert_eq!(aggregate["source"], "remote_smg");
    assert_eq!(aggregate["generated_at_ms"], 1_714_000_000_000i64);
    assert_eq!(aggregate["version"], 3);
    assert!(aggregate.get("worker_type").is_none());
    assert_eq!(remote_worker["worker"], "remote-worker-a");
    assert_eq!(remote_worker["worker_id"], "remote-worker-a");
    assert_eq!(remote_worker["model_id"], "cohere.command-r-plus");
    assert_eq!(remote_worker["status"], "ready");
    assert_eq!(remote_worker["source"], "local_worker");
    assert!(remote_worker.get("worker_type").is_none());
}
