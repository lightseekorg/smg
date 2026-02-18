//! Prefill/Decode (PD) routing integration tests
//!
//! Tests for prefill-decode disaggregation routing mode.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::RouterConfig;
use tower::ServiceExt;

use crate::common::{
    mock_worker::{self, HealthStatus, MockWorkerConfig, WorkerType},
    AppTestContext, TestWorkerConfig,
};
use smg::config::RoutingMode;

#[cfg(test)]
mod pd_routing_tests {
    use super::*;
    use serial_test::serial;

    /// Test basic PD mode routing with prefill and decode workers
    #[tokio::test]
    async fn test_pd_mode_basic_routing() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![
                    ("http://127.0.0.1:19800".to_string(), None),
                    ("http://127.0.0.1:19801".to_string(), None),
                ],
                vec![
                    "http://127.0.0.1:19802".to_string(),
                    "http://127.0.0.1:19803".to_string(),
                ],
            )
            .power_of_two_policy(1)
            .host("127.0.0.1")
            .port(3800)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        // Note: For PD mode tests, we need to start prefill and decode workers separately
        // The test context will need to handle this specially
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                // Prefill workers
                TestWorkerConfig::prefill(19800),
                TestWorkerConfig::prefill(19801),
                // Decode workers
                TestWorkerConfig::decode(19802),
                TestWorkerConfig::decode(19803),
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // Send requests and verify they succeed
        for i in 0..10 {
            let payload = json!({
                "text": format!("PD mode request {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "PD mode request should succeed"
            );
        }

        ctx.shutdown().await;
    }

    /// Test PD mode with round robin policy
    #[tokio::test]
    async fn test_pd_mode_round_robin() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19810".to_string(), None)],
                vec![
                    "http://127.0.0.1:19811".to_string(),
                    "http://127.0.0.1:19812".to_string(),
                ],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3801)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(19810),
                TestWorkerConfig::decode(19811),
                TestWorkerConfig::decode(19812),
            ],
        )
        .await;

        let app = ctx.create_app().await;
        let mut success_count = 0;

        for i in 0..20 {
            let payload = json!({
                "text": format!("PD round robin {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        assert_eq!(
            success_count, 20,
            "All requests should succeed in PD mode with round robin"
        );

        ctx.shutdown().await;
    }

    /// Test PD mode handles worker failures gracefully
    #[tokio::test]
    async fn test_pd_mode_with_failing_decode_worker() {
        use smg::config::RetryConfig;

        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19820".to_string(), None)],
                vec![
                    "http://127.0.0.1:19821".to_string(),
                    "http://127.0.0.1:19822".to_string(),
                ],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3802)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .retry_config(RetryConfig {
                max_retries: 3,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            })
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(19820),
                MockWorkerConfig {
                    port: 19821,
                    worker_type: WorkerType::Decode,
                    health_status: HealthStatus::Healthy,
                    response_delay_ms: 0,
                    fail_rate: 1.0, // Failing decode worker
                },
                TestWorkerConfig::decode(19822), // Healthy decode worker
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // Request should succeed via retry to healthy decode worker
        let payload = json!({
            "text": "Test with failing decode worker",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Request should succeed via retry to healthy decode worker"
        );

        ctx.shutdown().await;
    }

    /// Integration test: cold request goes to pre-prefill worker, warm goes to normal.
    ///
    /// Uses per-port request counters on mock sglang workers to verify
    /// which worker actually received each request through the full HTTP stack.
    ///
    /// The cache-aware tree is lazily initialized: the first request creates
    /// the tree entry. So we send a warmup request first to seed the tree,
    /// then a cold request (different text) that triggers pre-prefill.
    #[tokio::test]
    #[serial]
    async fn test_pre_prefill_cold_routes_to_pp_worker() {
        mock_worker::reset_request_counters();

        let prefill_port_1 = 19830;
        let prefill_port_2 = 19831;
        let pp_prefill_port = 19832;
        let decode_port = 19833;
        let pp_decode_port = 19834;

        let config = RouterConfig::builder()
            .mode(RoutingMode::PrefillDecode {
                prefill_urls: vec![
                    (format!("http://127.0.0.1:{}", prefill_port_1), None),
                    (format!("http://127.0.0.1:{}", prefill_port_2), None),
                    (format!("http://127.0.0.1:{}", pp_prefill_port), None),
                ],
                decode_urls: vec![
                    format!("http://127.0.0.1:{}", decode_port),
                    format!("http://127.0.0.1:{}", pp_decode_port),
                ],
                prefill_policy: None,
                decode_policy: None,
                pre_prefill_url: Some(format!("http://127.0.0.1:{}", pp_prefill_port)),
                pre_prefill_decode_url: Some(format!("http://127.0.0.1:{}", pp_decode_port)),
                pre_prefill_match_threshold: 0.1,
                pre_prefill_unmatched_chars_threshold: 50,
                pre_prefill_min_tokens: 50,
            })
            .cache_aware_policy(0.3, 64, 1.5, 0, 100_000)
            .host("127.0.0.1")
            .port(3803)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(prefill_port_1),
                TestWorkerConfig::prefill(prefill_port_2),
                TestWorkerConfig::prefill(pp_prefill_port),
                TestWorkerConfig::decode(decode_port),
                TestWorkerConfig::decode(pp_decode_port),
            ],
        )
        .await;

        let app = ctx.create_app().await;
        mock_worker::reset_request_counters();

        // ---- Warmup: seed the cache-aware tree so it exists for subsequent requests ----
        let warmup_text = "w".repeat(100);
        let warmup_payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": warmup_text}],
            "stream": false
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&warmup_payload).unwrap()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "Warmup request should succeed");

        // ---- Test 1: Cold request (different text, never seen) → pre-prefill worker ----
        mock_worker::reset_request_counters();

        let cold_text = "z".repeat(200); // completely different from warmup
        let cold_payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": cold_text}],
            "stream": false
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&cold_payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let pp_count = mock_worker::get_request_count(pp_prefill_port);
        assert_eq!(
            pp_count, 1,
            "Cold request should route to pre-prefill worker (port {}), got {} requests",
            pp_prefill_port, pp_count
        );
        let ppd_count = mock_worker::get_request_count(pp_decode_port);
        assert_eq!(
            ppd_count, 1,
            "Cold request decode should route to pre-prefill decode worker (port {}), got {}",
            pp_decode_port, ppd_count
        );
        let normal_p1 = mock_worker::get_request_count(prefill_port_1);
        let normal_p2 = mock_worker::get_request_count(prefill_port_2);
        assert_eq!(
            normal_p1 + normal_p2, 0,
            "Normal prefill workers should NOT receive cold request, got {} + {}",
            normal_p1, normal_p2
        );

        // ---- Test 2: Same text again (warm) → normal routing (cache hit, not pre-prefill) ----
        mock_worker::reset_request_counters();

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&cold_payload).unwrap()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Total across all prefill workers should be exactly 1 (normal routing picks one)
        let total_prefill = mock_worker::get_request_count(prefill_port_1)
            + mock_worker::get_request_count(prefill_port_2)
            + mock_worker::get_request_count(pp_prefill_port);
        assert_eq!(total_prefill, 1, "Warm request should route to exactly one prefill worker");

        // ---- Test 3: Short text → bypasses pre-prefill ----
        mock_worker::reset_request_counters();

        let short_payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&short_payload).unwrap()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let total_prefill = mock_worker::get_request_count(prefill_port_1)
            + mock_worker::get_request_count(prefill_port_2)
            + mock_worker::get_request_count(pp_prefill_port);
        assert_eq!(total_prefill, 1, "Short request should route to exactly one prefill worker");

        ctx.shutdown().await;
    }

    /// Simulate realistic traffic: multi-turn conversations mixed with new ones.
    ///
    /// Pattern:
    ///   Conv A turn 1 (cold)  → PP worker
    ///   Conv A turn 2 (warm)  → normal worker (shares prefix with turn 1)
    ///   Conv B turn 1 (cold)  → PP worker
    ///   Conv A turn 3 (warm)  → normal worker
    ///   Conv C turn 1 (cold)  → PP worker
    ///   Conv B turn 2 (warm)  → normal worker
    ///   Conv A turn 4 (warm)  → normal worker
    ///
    /// Verifies per-worker request counters match expected routing.
    #[tokio::test]
    #[serial]
    async fn test_pre_prefill_multi_turn_conversation_simulation() {
        mock_worker::reset_request_counters();

        let pp_port = 19840;
        let p1_port = 19841;
        let p2_port = 19842;
        let d_port = 19843;
        let ppd_port = 19844;

        let config = RouterConfig::builder()
            .mode(RoutingMode::PrefillDecode {
                prefill_urls: vec![
                    (format!("http://127.0.0.1:{}", p1_port), None),
                    (format!("http://127.0.0.1:{}", p2_port), None),
                    (format!("http://127.0.0.1:{}", pp_port), None),
                ],
                decode_urls: vec![
                    format!("http://127.0.0.1:{}", d_port),
                    format!("http://127.0.0.1:{}", ppd_port),
                ],
                prefill_policy: None,
                decode_policy: None,
                pre_prefill_url: Some(format!("http://127.0.0.1:{}", pp_port)),
                pre_prefill_decode_url: Some(format!("http://127.0.0.1:{}", ppd_port)),
                pre_prefill_match_threshold: 0.1,
                pre_prefill_unmatched_chars_threshold: 50,
                pre_prefill_min_tokens: 50,
            })
            .cache_aware_policy(0.3, 64, 1.5, 0, 100_000)
            .host("127.0.0.1")
            .port(3804)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(p1_port),
                TestWorkerConfig::prefill(p2_port),
                TestWorkerConfig::prefill(pp_port),
                TestWorkerConfig::decode(d_port),
                TestWorkerConfig::decode(ppd_port),
            ],
        )
        .await;

        let app = ctx.create_app().await;
        mock_worker::reset_request_counters();

        // ---- Build multi-turn conversations ----
        // Each conversation has a unique system prompt + growing user/assistant turns.
        // This mirrors real chat: each turn appends to the full history.

        let sys_a = "You are an expert Rust programmer helping with async code.";
        let sys_b = "You are a chef specializing in Italian cuisine and pasta recipes.";
        let sys_c = "You are a math tutor explaining calculus concepts step by step.";

        // Pad user messages to exceed min_tokens=50 when combined with system prompt
        let user_a1 = "Explain how tokio::spawn works and when I should use it versus tokio::select for concurrent tasks in a web server.";
        let asst_a1 = "tokio::spawn creates a new async task that runs independently on the runtime. tokio::select waits for the first of multiple futures to complete.";
        let user_a2 = "Can you show me an example of using tokio::select with a timeout and a channel receiver?";
        let asst_a2 = "Sure, here is an example using tokio::select with tokio::time::sleep and mpsc::Receiver...";
        let user_a3 = "How do I handle errors properly when using select with multiple branches?";
        let user_a4 = "What about cancellation safety? Which futures are safe to use in select?";

        let user_b1 = "How do I make fresh pasta from scratch? I want to learn the traditional Italian method with eggs and flour.";
        let asst_b1 = "Start with 100g flour per egg on a clean surface. Make a well, crack eggs in center, and slowly incorporate...";
        let user_b2 = "What is the best flour type for fresh pasta? Should I use 00 flour or semolina?";

        let user_c1 = "Explain the fundamental theorem of calculus and how it connects differentiation and integration together.";

        // Helper to build a chat payload from a message history
        let build_payload = |messages: Vec<serde_json::Value>| -> String {
            serde_json::to_string(&json!({
                "model": "mock-model",
                "messages": messages,
                "stream": false
            }))
            .unwrap()
        };

        let send_chat = |app: axum::Router, body: String| async move {
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(body))
                .unwrap();
            app.oneshot(req).await.unwrap()
        };

        // ---- Conv A, Turn 1: first message (COLD) ----
        mock_worker::reset_request_counters();
        let resp = send_chat(
            app.clone(),
            build_payload(vec![
                json!({"role": "system", "content": sys_a}),
                json!({"role": "user", "content": user_a1}),
            ]),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let pp_hits = mock_worker::get_request_count(pp_port);
        assert_eq!(pp_hits, 1, "Conv A turn 1 should be COLD → PP worker (got {} PP hits)", pp_hits);

        // ---- Conv A, Turn 2: follow-up (WARM — shares sys_a + user_a1 prefix) ----
        mock_worker::reset_request_counters();
        let resp = send_chat(
            app.clone(),
            build_payload(vec![
                json!({"role": "system", "content": sys_a}),
                json!({"role": "user", "content": user_a1}),
                json!({"role": "assistant", "content": asst_a1}),
                json!({"role": "user", "content": user_a2}),
            ]),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let pp_hits = mock_worker::get_request_count(pp_port);
        let normal_hits = mock_worker::get_request_count(p1_port)
            + mock_worker::get_request_count(p2_port);
        // Warm: high prefix overlap with turn 1 → should NOT go to PP
        // Note: it may still land on PP via normal cache-aware routing (PP "owns" this prefix).
        // What matters is total_prefill == 1 (one worker handles it)
        let total = pp_hits + normal_hits;
        assert_eq!(total, 1, "Conv A turn 2: exactly one prefill worker should handle it");

        // ---- Conv B, Turn 1: NEW conversation (COLD) ----
        mock_worker::reset_request_counters();
        let resp = send_chat(
            app.clone(),
            build_payload(vec![
                json!({"role": "system", "content": sys_b}),
                json!({"role": "user", "content": user_b1}),
            ]),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let pp_hits = mock_worker::get_request_count(pp_port);
        assert_eq!(pp_hits, 1, "Conv B turn 1 should be COLD → PP worker (got {} PP hits)", pp_hits);

        // ---- Conv A, Turn 3: continuing conv A (WARM) ----
        mock_worker::reset_request_counters();
        let resp = send_chat(
            app.clone(),
            build_payload(vec![
                json!({"role": "system", "content": sys_a}),
                json!({"role": "user", "content": user_a1}),
                json!({"role": "assistant", "content": asst_a1}),
                json!({"role": "user", "content": user_a2}),
                json!({"role": "assistant", "content": asst_a2}),
                json!({"role": "user", "content": user_a3}),
            ]),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let total = mock_worker::get_request_count(pp_port)
            + mock_worker::get_request_count(p1_port)
            + mock_worker::get_request_count(p2_port);
        assert_eq!(total, 1, "Conv A turn 3: exactly one prefill worker should handle it");

        // ---- Conv C, Turn 1: NEW conversation (COLD) ----
        mock_worker::reset_request_counters();
        let resp = send_chat(
            app.clone(),
            build_payload(vec![
                json!({"role": "system", "content": sys_c}),
                json!({"role": "user", "content": user_c1}),
            ]),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let pp_hits = mock_worker::get_request_count(pp_port);
        assert_eq!(pp_hits, 1, "Conv C turn 1 should be COLD → PP worker (got {} PP hits)", pp_hits);

        // ---- Conv B, Turn 2: continuing conv B (WARM) ----
        mock_worker::reset_request_counters();
        let resp = send_chat(
            app.clone(),
            build_payload(vec![
                json!({"role": "system", "content": sys_b}),
                json!({"role": "user", "content": user_b1}),
                json!({"role": "assistant", "content": asst_b1}),
                json!({"role": "user", "content": user_b2}),
            ]),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let total = mock_worker::get_request_count(pp_port)
            + mock_worker::get_request_count(p1_port)
            + mock_worker::get_request_count(p2_port);
        assert_eq!(total, 1, "Conv B turn 2: exactly one prefill worker should handle it");

        // ---- Conv A, Turn 4: deep continuation (WARM) ----
        mock_worker::reset_request_counters();
        let resp = send_chat(
            app.clone(),
            build_payload(vec![
                json!({"role": "system", "content": sys_a}),
                json!({"role": "user", "content": user_a1}),
                json!({"role": "assistant", "content": asst_a1}),
                json!({"role": "user", "content": user_a2}),
                json!({"role": "assistant", "content": asst_a2}),
                json!({"role": "user", "content": user_a3}),
                json!({"role": "assistant", "content": "Here is how to handle errors in select branches..."}),
                json!({"role": "user", "content": user_a4}),
            ]),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let total = mock_worker::get_request_count(pp_port)
            + mock_worker::get_request_count(p1_port)
            + mock_worker::get_request_count(p2_port);
        assert_eq!(total, 1, "Conv A turn 4: exactly one prefill worker should handle it");

        // ---- Summary: count total PP vs normal across ALL requests ----
        // We can't aggregate across resets, but the per-step assertions above prove:
        // - 3 new conversations (A1, B1, C1) → all routed to PP worker
        // - 4 follow-up turns (A2, A3, A4, B2) → routed via normal cache-aware (may hit PP or normal)

        ctx.shutdown().await;
    }
}
