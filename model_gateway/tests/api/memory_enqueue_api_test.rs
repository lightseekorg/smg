use std::sync::Arc;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderValue, StatusCode},
};
use chrono::Duration;
use serde_json::json;
use smg::{
    config::RouterConfig,
    routers::{RouterFactory, RouterTrait},
};
use smg_data_connector::{
    ConversationId, ConversationMemoryType, NewConversation, NewConversationMemory,
};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType},
    recording_memory_writer::RecordingConversationMemoryWriter,
    test_app::create_test_app_with_context,
};

fn memory_header_value() -> HeaderValue {
    HeaderValue::from_static(
        r#"{"long_term_memory":{"enabled":true,"policy":"store_and_recall","subject_id":"subject_abc","embedding_model_id":"text-embedding-3-small","extraction_model_id":"gpt-4.1-mini"}}"#,
    )
}

fn openai_router_config(worker_url: String) -> RouterConfig {
    let mut cfg = RouterConfig::builder()
        .openai_mode(vec![worker_url])
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(8 * 1024 * 1024)
        .request_timeout_secs(60)
        .worker_startup_timeout_secs(5)
        .worker_startup_check_interval_secs(1)
        .log_level("warn")
        .max_concurrent_requests(16)
        .queue_timeout_secs(5)
        .build_unchecked();
    cfg.memory_runtime.enabled = true;
    cfg
}

fn regular_router_config() -> RouterConfig {
    let mut cfg = RouterConfig::builder()
        .regular_mode(vec![])
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(8 * 1024 * 1024)
        .request_timeout_secs(60)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        .log_level("warn")
        .max_concurrent_requests(16)
        .queue_timeout_secs(5)
        .build_unchecked();
    cfg.memory_runtime.enabled = true;
    cfg
}

#[expect(clippy::panic, reason = "test helper - failing fast is intentional")]
async fn build_test_app_with_memory_writer(
    config: RouterConfig,
    writer: RecordingConversationMemoryWriter,
) -> (axum::Router, Arc<smg::app_context::AppContext>) {
    let writer_obj: Arc<dyn smg_data_connector::ConversationMemoryWriter> = Arc::new(writer);
    let ctx = crate::common::create_test_context_with_memory_writer(config, writer_obj).await;
    let router: Arc<dyn RouterTrait> = match RouterFactory::create_router(&ctx).await {
        Ok(router) => Arc::from(router),
        Err(err) => panic!("router creation should succeed: {err}"),
    };
    let app = create_test_app_with_context(router, ctx.clone());
    (app, ctx)
}

#[expect(
    clippy::panic,
    reason = "test helper - missing rows should fail loudly"
)]
fn find_row(
    rows: &[NewConversationMemory],
    memory_type: ConversationMemoryType,
) -> &NewConversationMemory {
    match rows.iter().find(|row| row.memory_type == memory_type) {
        Some(row) => row,
        None => panic!("memory row type should exist"),
    }
}

#[tokio::test]
async fn responses_endpoint_enqueues_ltm_and_ondemand_rows() {
    let mut worker = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });
    let worker_url = worker.start().await.expect("worker should start");

    let recording_writer = RecordingConversationMemoryWriter::default();
    let (app, ctx) = build_test_app_with_memory_writer(
        openai_router_config(worker_url),
        recording_writer.clone(),
    )
    .await;

    let conversation_id = ConversationId::from("conv_enqueue_responses");
    ctx.conversation_storage
        .create_conversation(NewConversation {
            id: Some(conversation_id.clone()),
            metadata: None,
        })
        .await
        .expect("conversation should be created");

    let payload = json!({
        "model": "mock-model",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "remember this user fact"}]
        }],
        "conversation": conversation_id.0,
        "store": true,
        "stream": false
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header(CONTENT_TYPE, "application/json")
        .header("x-conversation-memory-config", memory_header_value())
        .body(Body::from(
            serde_json::to_string(&payload).expect("payload should serialize"),
        ))
        .expect("request should build");

    let resp = app.oneshot(req).await.expect("request should succeed");
    assert_eq!(resp.status(), StatusCode::OK);

    let rows = recording_writer.snapshot().await;
    assert_eq!(rows.len(), 2, "responses path should enqueue two rows");

    let ltm = find_row(&rows, ConversationMemoryType::Ltm);
    let ondemand = find_row(&rows, ConversationMemoryType::OnDemand);

    assert_eq!(ltm.conversation_id, conversation_id);
    assert!(
        ltm.response_id.is_some(),
        "responses enqueue should set response_id"
    );
    assert_eq!(ondemand.conversation_id, conversation_id);
    assert!(
        ondemand.response_id.is_some(),
        "ondemand row should retain response_id on responses path"
    );

    let delta = ltm.next_run_at - ondemand.next_run_at;
    assert_eq!(delta, Duration::seconds(30));

    let ondemand_content: serde_json::Value = serde_json::from_str(
        ondemand
            .content
            .as_deref()
            .expect("ondemand content must be present"),
    )
    .expect("ondemand content should be valid JSON");
    assert_eq!(
        ondemand_content["user_text"].as_str(),
        Some("remember this user fact")
    );
    assert!(
        ondemand_content["assistant_text"]
            .as_str()
            .is_some_and(|text| !text.is_empty()),
        "assistant text should be captured from response output"
    );

    worker.stop().await;
}

#[tokio::test]
async fn conversations_items_endpoint_enqueues_rows_without_response_id() {
    let recording_writer = RecordingConversationMemoryWriter::default();
    let (app, _ctx) =
        build_test_app_with_memory_writer(regular_router_config(), recording_writer.clone()).await;

    let create_req = Request::builder()
        .method("POST")
        .uri("/v1/conversations")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from("{}"))
        .expect("request should build");
    let create_resp = app
        .clone()
        .oneshot(create_req)
        .await
        .expect("request should succeed");
    assert_eq!(create_resp.status(), StatusCode::OK);
    let create_body = axum::body::to_bytes(create_resp.into_body(), usize::MAX)
        .await
        .expect("response body should be readable");
    let create_json: serde_json::Value =
        serde_json::from_slice(&create_body).expect("response should be valid JSON");
    let conversation_id = create_json["id"]
        .as_str()
        .expect("conversation id should exist");

    let items_payload = json!({
        "items": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "seeded user message"}]
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "seeded assistant response"}]
            }
        ]
    });

    let items_req = Request::builder()
        .method("POST")
        .uri(format!("/v1/conversations/{conversation_id}/items"))
        .header(CONTENT_TYPE, "application/json")
        .header("x-conversation-memory-config", memory_header_value())
        .body(Body::from(
            serde_json::to_string(&items_payload).expect("payload should serialize"),
        ))
        .expect("request should build");
    let items_resp = app
        .oneshot(items_req)
        .await
        .expect("request should succeed");
    assert_eq!(items_resp.status(), StatusCode::OK);

    let rows = recording_writer.snapshot().await;
    assert_eq!(
        rows.len(),
        2,
        "conversation item ingestion should enqueue two rows"
    );
    let ltm = find_row(&rows, ConversationMemoryType::Ltm);
    let ondemand = find_row(&rows, ConversationMemoryType::OnDemand);

    assert_eq!(ltm.conversation_id.0, conversation_id);
    assert!(ltm.response_id.is_none());
    assert_eq!(ondemand.conversation_id.0, conversation_id);
    assert!(ondemand.response_id.is_none());
    assert_eq!(
        ltm.next_run_at - ondemand.next_run_at,
        Duration::seconds(30)
    );

    let ondemand_content: serde_json::Value = serde_json::from_str(
        ondemand
            .content
            .as_deref()
            .expect("ondemand content must be present"),
    )
    .expect("ondemand content should be valid JSON");
    assert_eq!(
        ondemand_content["user_text"].as_str(),
        Some("seeded user message")
    );
    assert_eq!(
        ondemand_content["assistant_text"].as_str(),
        Some("seeded assistant response")
    );
}
