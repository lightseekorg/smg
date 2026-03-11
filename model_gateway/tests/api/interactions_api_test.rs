//! End-to-end tests for the Gemini Interactions API (/v1/interactions).

use std::sync::Arc;

use axum::http::StatusCode;
use openai_protocol::interactions::InteractionsRequest;
use smg::{config::RouterConfig, routers::RouterFactory};

use crate::common::{
    create_test_context, mock_gemini_server::MockGeminiServer,
    test_app::create_test_app_with_context, test_config::defaults,
};

// ============================================================================
// Helpers
// ============================================================================

/// Build a Gemini-mode RouterConfig pointing at the given worker URLs.
fn gemini_config(worker_urls: Vec<String>) -> RouterConfig {
    RouterConfig::builder()
        .gemini_mode(worker_urls)
        .random_policy()
        .host(defaults::HOST)
        .port(3900)
        .max_payload_size(defaults::MAX_PAYLOAD_SIZE)
        .request_timeout_secs(defaults::REQUEST_TIMEOUT_SECS)
        .worker_startup_timeout_secs(defaults::WORKER_STARTUP_TIMEOUT_SECS)
        .worker_startup_check_interval_secs(defaults::WORKER_STARTUP_CHECK_INTERVAL_SECS)
        .max_concurrent_requests(defaults::MAX_CONCURRENT_REQUESTS)
        .queue_timeout_secs(defaults::QUEUE_TIMEOUT_SECS)
        .build_unchecked()
}

/// Extract JSON body from an axum Response.
#[expect(clippy::expect_used, reason = "test helper — panicking is fine")]
async fn json_body(resp: axum::response::Response) -> serde_json::Value {
    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read body");
    serde_json::from_slice(&body_bytes).expect("parse JSON")
}

// ============================================================================
// Non-streaming: model-based request
// ============================================================================

#[tokio::test]
async fn test_interactions_non_streaming_with_model() {
    let mock = MockGeminiServer::new().await;
    let config = gemini_config(vec![mock.base_url()]);
    let ctx = create_test_context(config.clone()).await;
    let router = RouterFactory::create_gemini_router(&ctx)
        .await
        .expect("create gemini router");
    let router: Arc<dyn smg::routers::RouterTrait> = Arc::from(router);

    let request: InteractionsRequest = serde_json::from_value(serde_json::json!({
        "model": "gemini-2.5-flash",
        "input": "What is the capital of France?",
        "stream": false,
        "store": false
    }))
    .unwrap();

    let response = router
        .route_interactions(None, &request, request.model.as_deref())
        .await;

    assert_eq!(response.status(), StatusCode::OK);

    let body = json_body(response).await;
    assert_eq!(body["status"], "completed");
    assert_eq!(body["model"], "gemini-2.5-flash");
    assert_eq!(body["store"], false);
}

#[tokio::test]
async fn test_interactions_non_streaming_with_agent() {
    let mock = MockGeminiServer::new().await;
    let config = gemini_config(vec![mock.base_url()]);
    let ctx = create_test_context(config.clone()).await;
    let router = RouterFactory::create_gemini_router(&ctx)
        .await
        .expect("create gemini router");
    let router: Arc<dyn smg::routers::RouterTrait> = Arc::from(router);

    let request: InteractionsRequest = serde_json::from_value(serde_json::json!({
        "agent": "deep-research-pro-preview-12-2025",
        "input": "Research quantum computing",
        "stream": false,
        "background": true
    }))
    .unwrap();

    let response = router
        .route_interactions(None, &request, request.agent.as_deref())
        .await;

    assert_eq!(response.status(), StatusCode::OK);

    let body = json_body(response).await;
    assert!(body.get("id").is_some(), "response must include id");
    assert_eq!(body["status"], "completed");
    assert_eq!(body["agent"], "deep-research-pro-preview-12-2025");
}

// ============================================================================
// Response metadata patching
// ============================================================================

#[tokio::test]
async fn test_interactions_store_true_model_request_rejected() {
    // Phase 1: store=true for model requests returns 501 Not Implemented
    // because interaction persistence is not yet implemented.
    let mock = MockGeminiServer::new().await;
    let config = gemini_config(vec![mock.base_url()]);
    let ctx = create_test_context(config.clone()).await;
    let router = RouterFactory::create_gemini_router(&ctx)
        .await
        .expect("create gemini router");
    let router: Arc<dyn smg::routers::RouterTrait> = Arc::from(router);

    let request: InteractionsRequest = serde_json::from_value(serde_json::json!({
        "model": "gemini-2.5-flash",
        "input": "Hello",
        "stream": false,
        "store": true
    }))
    .unwrap();

    let response = router
        .route_interactions(None, &request, request.model.as_deref())
        .await;
    assert_eq!(
        response.status(),
        StatusCode::NOT_IMPLEMENTED,
        "store=true model requests should return 501"
    );
}

// ============================================================================
// Wildcard model selection: unknown models are forwarded to upstream
// ============================================================================

#[tokio::test]
async fn test_interactions_unknown_model_forwarded_to_upstream() {
    // Wildcard workers (no explicit model list) accept any model —
    // the upstream decides whether the model is valid.
    let mock = MockGeminiServer::new().await;
    let config = gemini_config(vec![mock.base_url()]);
    let ctx = create_test_context(config.clone()).await;

    // Remove the pre-registered worker and register a wildcard worker instead.
    ctx.worker_registry.remove_by_url(&mock.base_url());
    let worker: Arc<dyn smg::core::Worker> = Arc::new(
        smg::core::BasicWorkerBuilder::new(mock.base_url())
            .worker_type(smg::core::WorkerType::Regular)
            .runtime_type(smg::core::RuntimeType::External)
            .build(),
    );
    ctx.worker_registry.register(worker);

    let router = RouterFactory::create_gemini_router(&ctx)
        .await
        .expect("create gemini router");
    let router: Arc<dyn smg::routers::RouterTrait> = Arc::from(router);

    let request: InteractionsRequest = serde_json::from_value(serde_json::json!({
        "model": "nonexistent-model-xyz",
        "input": "Hello",
        "stream": false,
        "store": false
    }))
    .unwrap();

    let response = router
        .route_interactions(None, &request, request.model.as_deref())
        .await;

    // Wildcard worker forwards any model to upstream
    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    assert_eq!(body["model"], "nonexistent-model-xyz");
}

// ============================================================================
// previous_interaction_id returns 501
// ============================================================================

#[tokio::test]
async fn test_interactions_previous_interaction_id_not_implemented() {
    let mock = MockGeminiServer::new().await;
    let config = gemini_config(vec![mock.base_url()]);
    let ctx = create_test_context(config.clone()).await;
    let router = RouterFactory::create_gemini_router(&ctx)
        .await
        .expect("create gemini router");
    let router: Arc<dyn smg::routers::RouterTrait> = Arc::from(router);

    let request: InteractionsRequest = serde_json::from_value(serde_json::json!({
        "model": "gemini-2.5-flash",
        "input": "Continue the conversation",
        "stream": false,
        "store": false,
        "previous_interaction_id": "interaction_abc_123"
    }))
    .unwrap();

    let response = router
        .route_interactions(None, &request, request.model.as_deref())
        .await;

    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
}

// ============================================================================
// Full Axum app integration (through HTTP layer)
// ============================================================================

#[tokio::test]
async fn test_interactions_via_axum_app() {
    use axum::body::Body;
    use http::Request;
    use tower::ServiceExt;

    let mock = MockGeminiServer::new().await;
    let config = gemini_config(vec![mock.base_url()]);
    let ctx = create_test_context(config.clone()).await;
    let router = RouterFactory::create_gemini_router(&ctx)
        .await
        .expect("create gemini router");
    let router: Arc<dyn smg::routers::RouterTrait> = Arc::from(router);

    let app = create_test_app_with_context(router, ctx);

    let payload = serde_json::json!({
        "model": "gemini-2.5-flash",
        "input": "Hello from axum test",
        "stream": false,
        "store": false
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/interactions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = json_body(resp).await;
    assert_eq!(body["status"], "completed");
    assert_eq!(body["model"], "gemini-2.5-flash");
}

// ============================================================================
// Auth header forwarding (x-goog-api-key)
// ============================================================================

#[tokio::test]
async fn test_interactions_forwards_api_key_header() {
    // Note: In test environment, the upstream URL is localhost (not googleapis.com),
    // so apply_provider_headers treats it as Generic and sends Authorization: Bearer.
    // The mock server accepts both x-goog-api-key and Authorization headers.
    let mock = MockGeminiServer::new_with_auth(Some("test-key-123".to_string())).await;
    let config = gemini_config(vec![mock.base_url()]);
    let ctx = create_test_context(config.clone()).await;
    let router = RouterFactory::create_gemini_router(&ctx)
        .await
        .expect("create gemini router");
    let router: Arc<dyn smg::routers::RouterTrait> = Arc::from(router);

    let request: InteractionsRequest = serde_json::from_value(serde_json::json!({
        "model": "gemini-2.5-flash",
        "input": "Hello with auth",
        "stream": false,
        "store": false
    }))
    .unwrap();

    let mut headers = http::HeaderMap::new();
    headers.insert("x-goog-api-key", "test-key-123".parse().unwrap());

    let response = router
        .route_interactions(Some(&headers), &request, request.model.as_deref())
        .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    assert_eq!(body["status"], "completed");
}

#[tokio::test]
async fn test_interactions_rejects_wrong_api_key() {
    let mock = MockGeminiServer::new_with_auth(Some("correct-key".to_string())).await;
    let config = gemini_config(vec![mock.base_url()]);
    let ctx = create_test_context(config.clone()).await;
    let router = RouterFactory::create_gemini_router(&ctx)
        .await
        .expect("create gemini router");
    let router: Arc<dyn smg::routers::RouterTrait> = Arc::from(router);

    let request: InteractionsRequest = serde_json::from_value(serde_json::json!({
        "model": "gemini-2.5-flash",
        "input": "Hello with wrong auth",
        "stream": false,
        "store": false
    }))
    .unwrap();

    let mut headers = http::HeaderMap::new();
    headers.insert("x-goog-api-key", "wrong-key".parse().unwrap());

    let response = router
        .route_interactions(Some(&headers), &request, request.model.as_deref())
        .await;

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

// ============================================================================
// Multiple workers: request succeeds with multiple backends
// ============================================================================

#[tokio::test]
async fn test_interactions_multiple_workers() {
    let mock1 = MockGeminiServer::new().await;
    let mock2 = MockGeminiServer::new().await;
    let config = gemini_config(vec![mock1.base_url(), mock2.base_url()]);
    let ctx = create_test_context(config.clone()).await;
    let router = RouterFactory::create_gemini_router(&ctx)
        .await
        .expect("create gemini router");
    let router: Arc<dyn smg::routers::RouterTrait> = Arc::from(router);

    for i in 0..5 {
        let request: InteractionsRequest = serde_json::from_value(serde_json::json!({
            "model": "gemini-2.5-flash",
            "input": format!("Request number {i}"),
            "stream": false,
            "store": false
        }))
        .unwrap();

        let response = router
            .route_interactions(None, &request, request.model.as_deref())
            .await;

        assert_eq!(
            response.status(),
            StatusCode::OK,
            "request {i} should succeed"
        );
    }
}

// ============================================================================
// Payload transformations
// ============================================================================

#[tokio::test]
async fn test_interactions_model_id_override() {
    let mock = MockGeminiServer::new().await;
    let config = gemini_config(vec![mock.base_url()]);
    let ctx = create_test_context(config.clone()).await;
    let router = RouterFactory::create_gemini_router(&ctx)
        .await
        .expect("create gemini router");
    let router: Arc<dyn smg::routers::RouterTrait> = Arc::from(router);

    let request: InteractionsRequest = serde_json::from_value(serde_json::json!({
        "model": "gemini-2.5-flash",
        "input": "Hello",
        "stream": false,
        "store": false
    }))
    .unwrap();

    // Override model_id to gemini-2.5-pro
    let response = router
        .route_interactions(None, &request, Some("gemini-2.5-pro"))
        .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    // The mock echoes back the model from the payload, which should be overridden
    assert_eq!(body["model"], "gemini-2.5-pro");
}

// ============================================================================
// Validation: request body validation through axum app
// ============================================================================

#[tokio::test]
async fn test_interactions_validation_empty_input() {
    use axum::body::Body;
    use http::Request;
    use tower::ServiceExt;

    let mock = MockGeminiServer::new().await;
    let config = gemini_config(vec![mock.base_url()]);
    let ctx = create_test_context(config.clone()).await;
    let router = RouterFactory::create_gemini_router(&ctx)
        .await
        .expect("create gemini router");
    let router: Arc<dyn smg::routers::RouterTrait> = Arc::from(router);

    let app = create_test_app_with_context(router, ctx);

    let payload = serde_json::json!({
        "model": "gemini-2.5-flash",
        "input": "",
        "stream": false,
        "store": false
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/interactions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    // ValidatedJson returns 400 for validation errors
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}
