//! Integration tests for the Responses-API interceptor seam.
//!
//! Each test registers a `RecordingInterceptor`, drives a request through the
//! relevant router code path, and asserts that the corresponding lifecycle
//! hook(s) fire with the expected ctx fields populated. The tests are
//! deliberately storage-backend agnostic and avoid memory-feature awareness.

#[path = "common/mod.rs"]
pub mod common;

use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use axum::{http::StatusCode, routing::post, Json, Router};
use llm_tokenizer::registry::TokenizerRegistry;
use openai_protocol::{
    common::ConversationRef,
    responses::{ResponseInput, ResponsesRequest},
};
use serde_json::json;
use smg::{
    app_context::AppContext,
    config::RouterConfig,
    memory::MemoryExecutionContext,
    middleware::{TenantRequestMeta, TokenBucket},
    policies::PolicyRegistry,
    routers::{conversations::create_conversation_items_with_headers, openai::OpenAIRouter, RouterTrait},
    tenant::{RouteRequestMeta, TenantKey},
    worker::{WorkerMonitor, WorkerRegistry},
};
use smg_data_connector::{
    MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    NewConversation, NoOpConversationMemoryWriter,
};
use smg_extensions::{
    AfterPersistCtx, BeforeModelCtx, InterceptorRegistry, ResponsesInterceptor,
};
use smg_mcp::{McpConfig, McpOrchestrator};
use tokio::net::TcpListener;

use crate::common::test_app::register_external_worker;

// ============================================================================
// Recording interceptor
// ============================================================================

/// Records each phase invocation along with a snapshot of the loaded ctx
/// fields the tests care about.
#[derive(Default, Clone)]
pub struct RecordingInterceptor {
    pub before_calls: Arc<Mutex<Vec<RecordedBefore>>>,
    pub after_calls: Arc<Mutex<Vec<RecordedAfter>>>,
}

#[derive(Debug, Clone)]
pub struct RecordedBefore {
    pub has_conversation_id: bool,
    pub user_turns: u32,
}

#[derive(Debug, Clone)]
pub struct RecordedAfter {
    pub has_response_id: bool,
    pub has_response_json: bool,
    pub persisted_count: usize,
    pub has_conversation_id: bool,
}

#[async_trait]
impl ResponsesInterceptor for RecordingInterceptor {
    fn name(&self) -> &'static str {
        "recording"
    }

    async fn before_model(&self, ctx: &mut BeforeModelCtx<'_>) {
        // Use the documented destructure-with-rest pattern so future fields
        // don't break this test code.
        let BeforeModelCtx {
            conversation_id,
            turn_info,
            ..
        } = &*ctx;
        self.before_calls.lock().expect("mutex").push(RecordedBefore {
            has_conversation_id: conversation_id.is_some(),
            user_turns: turn_info.user_turns,
        });
    }

    async fn after_persist(&self, ctx: &AfterPersistCtx<'_>) {
        let AfterPersistCtx {
            response_id,
            response_json,
            persisted_item_ids,
            conversation_id,
            ..
        } = ctx;
        self.after_calls.lock().expect("mutex").push(RecordedAfter {
            has_response_id: response_id.is_some(),
            has_response_json: response_json.is_some(),
            persisted_count: persisted_item_ids.len(),
            has_conversation_id: conversation_id.is_some(),
        });
    }
}

fn registry_with(interceptor: Arc<dyn ResponsesInterceptor>) -> InterceptorRegistry {
    let mut builder = InterceptorRegistry::builder();
    builder.register(interceptor);
    builder.build()
}

// ============================================================================
// Test harness: build an AppContext with custom interceptors
// ============================================================================

/// Build an AppContext suitable for exercising the OpenAI router with a
/// caller-supplied interceptor registry. Mirrors `common::test_app::
/// create_test_app_context` but accepts an explicit `InterceptorRegistry`
/// so tests can observe hook firings.
async fn build_app_context_with_interceptors(
    interceptors: InterceptorRegistry,
) -> Arc<AppContext> {
    let router_config = RouterConfig::default();
    let client = reqwest::Client::new();

    let worker_job_queue = Arc::new(OnceLock::new());
    let workflow_engines = Arc::new(OnceLock::new());

    let mcp_orchestrator_lock: Arc<OnceLock<Arc<McpOrchestrator>>> = Arc::new(OnceLock::new());
    let empty_config = McpConfig {
        servers: vec![],
        pool: Default::default(),
        proxy: None,
        warmup: vec![],
        inventory: Default::default(),
        policy: Default::default(),
    };
    let mcp_orchestrator = McpOrchestrator::new(empty_config)
        .await
        .expect("mcp orchestrator");
    mcp_orchestrator_lock
        .set(Arc::new(mcp_orchestrator))
        .ok()
        .expect("mcp orchestrator set");

    let worker_registry = Arc::new(WorkerRegistry::new());
    let policy_registry = Arc::new(PolicyRegistry::new(router_config.policy.clone()));

    let response_storage = Arc::new(MemoryResponseStorage::new());
    let conversation_storage = Arc::new(MemoryConversationStorage::new());
    let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());
    let conversation_memory_writer = Arc::new(NoOpConversationMemoryWriter::new());

    let worker_monitor = Some(Arc::new(WorkerMonitor::new(
        worker_registry.clone(),
        policy_registry.clone(),
        client.clone(),
        router_config.load_monitor_interval_secs,
    )));

    let rate_limiter = match router_config.max_concurrent_requests {
        n if n <= 0 => None,
        n => {
            let rate_limit_tokens = router_config
                .rate_limit_tokens_per_second
                .filter(|&t| t > 0)
                .unwrap_or(n);
            Some(Arc::new(TokenBucket::new(
                n as usize,
                rate_limit_tokens as usize,
            )))
        }
    };

    Arc::new(
        AppContext::builder()
            .router_config(router_config)
            .client(client)
            .rate_limiter(rate_limiter)
            .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
            .reasoning_parser_factory(None)
            .tool_parser_factory(None)
            .worker_registry(worker_registry)
            .policy_registry(policy_registry)
            .response_storage(response_storage)
            .conversation_storage(conversation_storage)
            .conversation_item_storage(conversation_item_storage)
            .conversation_memory_writer(conversation_memory_writer)
            .worker_monitor(worker_monitor)
            .worker_job_queue(worker_job_queue)
            .workflow_engines(workflow_engines)
            .mcp_orchestrator(mcp_orchestrator_lock)
            .interceptors(interceptors)
            .build()
            .expect("AppContext build"),
    )
}

fn test_tenant_meta() -> TenantRequestMeta {
    RouteRequestMeta::new(TenantKey::from("test-tenant"))
}

/// Spin up a tiny axum mock that returns a non-streaming Responses payload.
/// Returns the upstream base URL plus a JoinHandle the caller can abort.
async fn spawn_non_streaming_mock() -> (String, tokio::task::JoinHandle<()>) {
    #[expect(clippy::disallowed_methods, reason = "test infrastructure")]
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");

    let app = Router::new().route(
        "/v1/responses",
        post(|Json(request): Json<serde_json::Value>| async move {
            let model = request
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("mock-model")
                .to_string();
            Json(json!({
                "id": "resp_mock_1",
                "object": "response",
                "created_at": 1_700_000_000,
                "status": "completed",
                "model": model,
                "output": [{
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": "hi",
                        "annotations": []
                    }]
                }],
                "metadata": {}
            }))
        }),
    );

    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    (format!("http://{addr}"), server)
}

// ============================================================================
// HTTP non-streaming
// ============================================================================

/// HTTP non-streaming: `POST /v1/responses` (via the OpenAI router) fires
/// before_model exactly once and after_persist exactly once.
#[tokio::test]
async fn http_non_streaming_fires_both_phases_once() {
    let recording = Arc::new(RecordingInterceptor::default());
    let registry = registry_with(recording.clone() as Arc<dyn ResponsesInterceptor>);

    let ctx = build_app_context_with_interceptors(registry).await;

    let (base_url, server) = spawn_non_streaming_mock().await;
    register_external_worker(&ctx, &base_url, Some(vec!["mock-model"]));

    let router = OpenAIRouter::new(&ctx).await.expect("router");

    // Pre-create a conversation so before_model has a conversation_id to record.
    let conv = ctx
        .conversation_storage
        .create_conversation(NewConversation {
            id: None,
            metadata: None,
        })
        .await
        .expect("create conversation");

    let request = ResponsesRequest {
        model: "mock-model".to_string(),
        input: ResponseInput::Text("hello".to_string()),
        store: Some(true),
        conversation: Some(ConversationRef::Id(conv.id.0.clone())),
        ..Default::default()
    };
    let tenant_meta = test_tenant_meta();
    let response = router
        .route_responses(None, &tenant_meta, &request, &request.model)
        .await;
    assert_eq!(response.status(), StatusCode::OK);

    // Drain the response body so the persist task runs to completion.
    let _ = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");

    let before = recording.before_calls.lock().expect("mutex");
    let after = recording.after_calls.lock().expect("mutex");

    assert_eq!(
        before.len(),
        1,
        "before_model should fire exactly once (got {:?})",
        *before
    );
    assert_eq!(
        after.len(),
        1,
        "after_persist should fire exactly once (got {:?})",
        *after
    );
    assert!(
        before[0].has_conversation_id,
        "before_model ctx should expose conversation_id"
    );
    assert!(
        after[0].has_conversation_id,
        "after_persist ctx should expose conversation_id"
    );
    assert!(
        after[0].has_response_id,
        "after_persist ctx should expose response_id"
    );
    assert!(
        after[0].has_response_json,
        "after_persist ctx should expose response_json"
    );

    server.abort();
}

// ============================================================================
// Items-only path
// ============================================================================

/// Items-only path: `POST /v1/conversations/{id}/items` fires only the
/// after_persist hook (no before_model), and the ctx exposes neither a
/// response_id nor a response_json.
#[tokio::test]
async fn items_only_path_fires_after_persist_with_no_response() {
    let recording = Arc::new(RecordingInterceptor::default());
    let registry = registry_with(recording.clone() as Arc<dyn ResponsesInterceptor>);

    let ctx = build_app_context_with_interceptors(registry).await;

    // Pre-create the conversation so the items-only endpoint accepts the request.
    let conv = ctx
        .conversation_storage
        .create_conversation(NewConversation {
            id: None,
            metadata: None,
        })
        .await
        .expect("create conversation");

    let body = json!({
        "items": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "first"}]
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "second"}]
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "third"}]
            }
        ]
    });

    let response = create_conversation_items_with_headers(
        &ctx.conversation_storage,
        &ctx.conversation_item_storage,
        &conv.id.0,
        body,
        MemoryExecutionContext::default(),
        ctx.interceptors.clone(),
        "req_test".to_string(),
        Some("test-tenant".to_string()),
        Default::default(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);

    let before = recording.before_calls.lock().expect("mutex");
    let after = recording.after_calls.lock().expect("mutex");

    assert!(
        before.is_empty(),
        "items-only path must not fire before_model (got {:?})",
        *before
    );
    assert_eq!(
        after.len(),
        1,
        "items-only path must fire after_persist exactly once (got {:?})",
        *after
    );
    assert!(
        !after[0].has_response_id,
        "items-only after_persist ctx must not carry a response_id"
    );
    assert!(
        !after[0].has_response_json,
        "items-only after_persist ctx must not carry a response_json"
    );
    assert_eq!(
        after[0].persisted_count, 3,
        "items-only after_persist ctx should report all persisted items"
    );
    assert!(
        after[0].has_conversation_id,
        "items-only after_persist ctx must expose conversation_id"
    );
}

// ============================================================================
// HTTP streaming
// ============================================================================

/// Spin up a tiny axum mock that returns an SSE Responses stream containing
/// a `response.completed` event. This is enough to drive the streaming
/// router through the post-stream persistence + after_persist hook.
async fn spawn_streaming_mock() -> (String, tokio::task::JoinHandle<()>) {
    use axum::response::Response;

    #[expect(clippy::disallowed_methods, reason = "test infrastructure")]
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");

    let sse_handler = post(|Json(_request): Json<serde_json::Value>| async move {
        let response_id = "resp_stream_int";
        let message_id = "msg_stream_int";
        let final_text = "stream-ok";

        let events = vec![
            (
                "response.created",
                json!({
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": 1_700_000_500,
                        "status": "in_progress",
                        "model": "",
                        "output": [],
                        "parallel_tool_calls": true,
                        "previous_response_id": null,
                        "reasoning": null,
                        "store": false,
                        "temperature": 1.0,
                        "text": {"format": {"type": "text"}},
                        "tool_choice": "auto",
                        "tools": [],
                        "top_p": 1.0,
                        "truncation": "disabled",
                        "usage": null,
                        "metadata": null
                    }
                }),
            ),
            (
                "response.output_item.added",
                json!({
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": []
                    }
                }),
            ),
            (
                "response.output_text.delta",
                json!({
                    "type": "response.output_text.delta",
                    "sequence_number": 2,
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": final_text,
                    "logprobs": []
                }),
            ),
            (
                "response.output_text.done",
                json!({
                    "type": "response.output_text.done",
                    "sequence_number": 3,
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": final_text,
                    "logprobs": []
                }),
            ),
            (
                "response.output_item.done",
                json!({
                    "type": "response.output_item.done",
                    "sequence_number": 4,
                    "output_index": 0,
                    "item": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{
                            "type": "output_text",
                            "text": final_text,
                            "annotations": [],
                            "logprobs": []
                        }]
                    }
                }),
            ),
            (
                "response.completed",
                json!({
                    "type": "response.completed",
                    "sequence_number": 5,
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": 1_700_000_500,
                        "status": "completed",
                        "model": "",
                        "output": [{
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{
                                "type": "output_text",
                                "text": final_text,
                                "annotations": [],
                                "logprobs": []
                            }]
                        }],
                        "parallel_tool_calls": true,
                        "previous_response_id": null,
                        "reasoning": null,
                        "store": false,
                        "temperature": 1.0,
                        "text": {"format": {"type": "text"}},
                        "tool_choice": "auto",
                        "tools": [],
                        "top_p": 1.0,
                        "truncation": "disabled",
                        "usage": {
                            "input_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 2
                        },
                        "metadata": null,
                        "instructions": null,
                        "user": null
                    }
                }),
            ),
        ];

        let payload = events
            .into_iter()
            .map(|(event, data)| format!("event: {event}\ndata: {data}\n\n"))
            .collect::<String>();

        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .body(axum::body::Body::from(payload))
            .expect("response")
    });

    let app = Router::new().route("/v1/responses", sse_handler);
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    (format!("http://{addr}"), server)
}

/// HTTP streaming: `POST /v1/responses` with `stream=true` fires
/// `before_model` once at request entry and `after_persist` once after the
/// stream completes and persistence is committed.
#[tokio::test]
async fn http_streaming_fires_both_phases_once() {
    use std::time::Duration;
    use tokio::time::sleep;

    let recording = Arc::new(RecordingInterceptor::default());
    let registry = registry_with(recording.clone() as Arc<dyn ResponsesInterceptor>);

    let ctx = build_app_context_with_interceptors(registry).await;

    let (base_url, server) = spawn_streaming_mock().await;
    register_external_worker(&ctx, &base_url, Some(vec!["mock-model"]));

    let router = OpenAIRouter::new(&ctx).await.expect("router");

    let conv = ctx
        .conversation_storage
        .create_conversation(NewConversation {
            id: None,
            metadata: None,
        })
        .await
        .expect("create conversation");

    let request = ResponsesRequest {
        model: "mock-model".to_string(),
        input: ResponseInput::Text("hello".to_string()),
        store: Some(true),
        stream: Some(true),
        conversation: Some(ConversationRef::Id(conv.id.0.clone())),
        ..Default::default()
    };
    let tenant_meta = test_tenant_meta();
    let response = router
        .route_responses(None, &tenant_meta, &request, &request.model)
        .await;
    assert_eq!(response.status(), StatusCode::OK);

    // Drain the SSE stream before checking interceptor records — the
    // streaming after_persist hook fires from a background task that runs
    // after the final `response.completed` event is observed.
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let body_text = String::from_utf8(body.to_vec()).expect("utf8");
    assert!(body_text.contains("response.completed"));

    // The persist + after_persist work runs on a detached task; poll until
    // the recording interceptor observes the call (with a generous bound).
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        if !recording.after_calls.lock().expect("mutex").is_empty() {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "after_persist hook never fired for streaming path"
        );
        sleep(Duration::from_millis(20)).await;
    }

    let before = recording.before_calls.lock().expect("mutex");
    let after = recording.after_calls.lock().expect("mutex");

    assert_eq!(
        before.len(),
        1,
        "streaming before_model should fire exactly once (got {:?})",
        *before
    );
    assert_eq!(
        after.len(),
        1,
        "streaming after_persist should fire exactly once (got {:?})",
        *after
    );
    assert!(before[0].has_conversation_id);
    assert!(after[0].has_response_id);
    assert!(after[0].has_response_json);
    assert!(after[0].has_conversation_id);

    server.abort();
}

// ============================================================================
// HTTP streaming with MCP tool calls (skipped: requires real MCP server)
// ============================================================================

/// Streaming with an MCP tool loop fires `after_persist` after the MCP
/// finalizer commits the post-tool response.
///
/// **Skipped:** triggering the MCP-streaming code path requires an actual
/// MCP server binding (registered through `McpOrchestrator`) plus an
/// upstream that emits compatible function-call SSE events. Spinning up
/// a real `rmcp` server in this integration test is out of scope; the
/// MCP-finalizer hook firing logic is structurally identical to the
/// regular streaming finalizer and is covered by the registry-level unit
/// tests in `smg-extensions` and the streaming hook test above.
#[tokio::test]
#[ignore = "requires MCP server fixture; covered indirectly by streaming + registry unit tests"]
async fn http_mcp_streaming_fires_after_persist() {
    // Intentionally empty - see doc comment.
}

// ============================================================================
// gRPC paths (unit tests live alongside the call sites)
// ============================================================================

/// gRPC harmony / regular hook firings are covered by unit tests at the
/// call sites:
///
/// - `model_gateway/src/routers/grpc/common/responses/utils.rs::tests`
///   (shared `persist_response_if_needed` after_persist)
/// - `model_gateway/src/routers/grpc/regular/responses/non_streaming.rs::tests`
///   (regular gRPC `before_model`)
/// - `model_gateway/src/routers/grpc/harmony/responses/non_streaming.rs::tests`
///   (harmony gRPC `before_model`)
///
/// Driving the full gRPC pipeline from an integration test requires a tonic
/// server harness plus a mock backend gRPC client; this is meaningfully
/// heavier than a unit test of the hook firing block itself, while
/// providing the same guarantee. The unit tests live in the `grpc::common`,
/// `grpc::regular`, and `grpc::harmony` modules so they can access
/// `pub(crate)` types like `persist_response_if_needed` and
/// `ResponsesContext`.
#[allow(dead_code)]
fn _grpc_coverage_note() {}
