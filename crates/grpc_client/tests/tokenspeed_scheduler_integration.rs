//! End-to-end tests for `TokenSpeedSchedulerClient` against a real
//! TokenSpeed gRPC server.
//!
//! **Gated on `TOKENSPEED_GRPC_ENDPOINT`** — unset = every test is skipped
//! (runs only explicitly via `cargo test -- --ignored`). Sibling env vars
//! tune the tests to the live model:
//!
//! * `TOKENSPEED_GRPC_ENDPOINT` — e.g. `grpc://127.0.0.1:50051`
//! * `TOKENSPEED_TEST_MODEL_PATH` — must match the `--model-path` the server
//!   was started with. Used only to sanity-check `GetModelInfo`.
//! * `TOKENSPEED_TEST_INPUT_IDS` — comma-separated prompt token IDs for the
//!   Generate tests. Default: `1,2,3`.

// Workspace clippy auto-allows expect/unwrap inside `#[test]` functions, but
// `#[tokio::test]` expands too late for that detection — so we opt the whole
// file in explicitly. This is a test file; a panic on a missing env var or a
// dead server is the correct way to fail.
#![expect(clippy::expect_used)]

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use futures::StreamExt;
use smg_grpc_client::{
    tokenspeed_proto as proto, BoxedTraceInjector, TokenSpeedSchedulerClient, TraceInjector,
};
use tonic::metadata::MetadataMap;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn endpoint() -> String {
    std::env::var("TOKENSPEED_GRPC_ENDPOINT")
        .expect("TOKENSPEED_GRPC_ENDPOINT must be set for the integration suite")
}

fn input_ids() -> Vec<u32> {
    std::env::var("TOKENSPEED_TEST_INPUT_IDS")
        .ok()
        .and_then(|v| {
            v.split(',')
                .map(|s| s.trim().parse::<u32>().ok())
                .collect::<Option<Vec<_>>>()
        })
        .unwrap_or_else(|| vec![1, 2, 3])
}

fn sampling_params(max_new_tokens: u32) -> proto::SamplingParams {
    // Semantic defaults for a deterministic, reproducible generation.
    proto::SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: -1,
        min_p: 0.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: 1.0,
        max_new_tokens: Some(max_new_tokens),
        n: 1,
        ignore_eos: false,
        skip_special_tokens: true,
        spaces_between_special_tokens: true,
        ..Default::default()
    }
}

async fn connect() -> TokenSpeedSchedulerClient {
    TokenSpeedSchedulerClient::connect(&endpoint())
        .await
        .expect("connect to tokenspeed gRPC server")
}

/// Records every `inject` call so we can assert trace metadata is pushed
/// into every RPC — direct evidence that the per-RPC `inject_trace` helper
/// wired the new unary endpoints.
#[derive(Clone, Default)]
struct CountingInjector {
    count: Arc<AtomicUsize>,
}

impl TraceInjector for CountingInjector {
    fn inject(
        &self,
        metadata: &mut MetadataMap,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.count.fetch_add(1, Ordering::SeqCst);
        // Drop a sentinel header so a server-side log/trace would pick it up.
        metadata.insert(
            "x-tokenspeed-test-trace",
            "integration".parse().expect("static ascii"),
        );
        Ok(())
    }
}

async fn connect_with_counter() -> (TokenSpeedSchedulerClient, Arc<AtomicUsize>) {
    let counter = Arc::new(AtomicUsize::new(0));
    let injector: BoxedTraceInjector = Arc::new(CountingInjector {
        count: counter.clone(),
    });
    let client = TokenSpeedSchedulerClient::connect_with_trace_injector(&endpoint(), injector)
        .await
        .expect("connect with counting injector");
    (client, counter)
}

// ---------------------------------------------------------------------------
// Metadata RPCs
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires TOKENSPEED_GRPC_ENDPOINT"]
async fn health_check_reports_healthy() {
    let client = connect().await;
    let resp = client.health_check().await.expect("health_check");
    assert!(resp.healthy, "server reported not healthy: {resp:?}");
}

#[tokio::test]
#[ignore = "requires TOKENSPEED_GRPC_ENDPOINT"]
async fn get_model_info_reports_populated_fields() {
    let client = connect().await;
    let resp = client.get_model_info().await.expect("get_model_info");

    assert!(resp.vocab_size > 0, "vocab_size should be > 0");
    assert!(
        resp.max_context_length > 0,
        "max_context_length should be > 0"
    );
    assert!(resp.is_generation, "model should report is_generation=true");
    assert!(
        !resp.architectures.is_empty(),
        "architectures should be populated"
    );
    assert!(
        !resp.eos_token_ids.is_empty(),
        "eos_token_ids should be populated"
    );

    if let Ok(expected) = std::env::var("TOKENSPEED_TEST_MODEL_PATH") {
        assert_eq!(resp.model_path, expected, "model_path mismatch");
    }
}

#[tokio::test]
#[ignore = "requires TOKENSPEED_GRPC_ENDPOINT"]
async fn get_server_info_reports_uptime_and_type() {
    let client = connect().await;
    let resp = client.get_server_info().await.expect("get_server_info");
    assert_eq!(resp.server_type, "grpc");
    assert!(resp.uptime_seconds >= 0.0);
}

#[tokio::test]
#[ignore = "requires TOKENSPEED_GRPC_ENDPOINT"]
async fn get_loads_returns_empty_stub() {
    let client = connect().await;
    let resp = client
        .get_loads(proto::GetLoadsRequest::default())
        .await
        .expect("get_loads");
    // The server-side is currently a stub.
    assert_eq!(resp.loads.len(), 0);
}

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires TOKENSPEED_GRPC_ENDPOINT"]
async fn generate_non_streaming_returns_completion_with_tokens() {
    let client = connect().await;
    let req = proto::GenerateRequest {
        request_id: "rs-int-unary-1".to_string(),
        tokenized: Some(proto::TokenizedInput {
            original_text: String::new(),
            input_ids: input_ids(),
        }),
        sampling_params: Some(sampling_params(8)),
        stream: false,
        ..Default::default()
    };

    let mut stream = client.generate(req).await.expect("generate");
    let first = stream
        .next()
        .await
        .expect("response stream closed without yielding")
        .expect("generate response");

    // Non-streaming path yields exactly one Complete.
    match first.response {
        Some(proto::generate_response::Response::Complete(c)) => {
            assert!(!c.output_ids.is_empty(), "complete had zero output ids");
            assert!(
                c.finish_reason == "stop" || c.finish_reason == "length",
                "unexpected finish_reason={:?}",
                c.finish_reason
            );
        }
        other => panic!("expected Complete, got {other:?}"),
    }

    assert!(stream.next().await.is_none(), "expected stream to end");
}

#[tokio::test]
#[ignore = "requires TOKENSPEED_GRPC_ENDPOINT"]
async fn generate_streaming_terminates_with_complete() {
    let client = connect().await;
    let req = proto::GenerateRequest {
        request_id: "rs-int-stream-1".to_string(),
        tokenized: Some(proto::TokenizedInput {
            original_text: String::new(),
            input_ids: input_ids(),
        }),
        sampling_params: Some(sampling_params(16)),
        stream: true,
        ..Default::default()
    };

    let mut stream = client.generate(req).await.expect("generate");

    let mut saw_complete = false;
    let mut saw_chunk = false;
    while let Some(item) = stream.next().await {
        let resp = item.expect("generate response");
        match resp.response {
            Some(proto::generate_response::Response::Chunk(_)) => saw_chunk = true,
            Some(proto::generate_response::Response::Complete(c)) => {
                saw_complete = true;
                assert!(c.finish_reason == "stop" || c.finish_reason == "length");
            }
            None => panic!("GenerateResponse.response was None"),
        }
    }

    // A completion is mandatory; chunks are only emitted when the coalescing
    // collector can't fold them into the finished output — not strictly required.
    assert!(saw_complete, "stream ended without a Complete");
    let _ = saw_chunk; // intentionally not asserted.
}

// ---------------------------------------------------------------------------
// Abort (out-of-band + on-drop)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires TOKENSPEED_GRPC_ENDPOINT"]
async fn abort_request_out_of_band_returns_ok() {
    let client = connect().await;

    // Start a long-running stream, pull one chunk, then abort out-of-band.
    // We assert that the Abort RPC itself reaches the server and succeeds.
    // The stream's eventual termination is a server-side concern already
    // covered by the Python E2E suite; here we only care that the client
    // wrapper can issue Abort against an in-flight rid.
    let rid = "rs-int-abort-oob-1".to_string();
    let mut params = sampling_params(512);
    params.ignore_eos = true;
    let req = proto::GenerateRequest {
        request_id: rid.clone(),
        tokenized: Some(proto::TokenizedInput {
            original_text: String::new(),
            input_ids: input_ids(),
        }),
        sampling_params: Some(params),
        stream: true,
        ..Default::default()
    };

    let mut stream = client.generate(req).await.expect("generate");
    let _first = stream.next().await.expect("first response");

    client
        .abort_request(rid.clone(), "integration abort".to_string())
        .await
        .expect("abort_request");

    // Explicitly drop the stream rather than draining it — that triggers
    // AbortOnDropStream's Drop impl, which is fine (idempotent with the
    // explicit Abort above).
    drop(stream);
}

#[tokio::test]
#[ignore = "requires TOKENSPEED_GRPC_ENDPOINT"]
async fn abort_on_drop_fires_and_client_stays_usable() {
    let client = connect().await;

    // Build a long-running stream we'll immediately drop.
    let rid = "rs-int-abort-ondrop-1".to_string();
    let mut params = sampling_params(512);
    params.ignore_eos = true;
    let req = proto::GenerateRequest {
        request_id: rid.clone(),
        tokenized: Some(proto::TokenizedInput {
            original_text: String::new(),
            input_ids: input_ids(),
        }),
        sampling_params: Some(params),
        stream: true,
        ..Default::default()
    };
    let mut stream = client.generate(req).await.expect("generate");
    let _first = stream.next().await.expect("first response");

    drop(stream);

    // Give the fire-and-forget tokio::spawn in AbortOnDropStream::drop time
    // to reach the server. The client must remain usable afterwards.
    tokio::time::sleep(Duration::from_millis(500)).await;

    let health = client
        .health_check()
        .await
        .expect("health_check after drop");
    assert!(health.healthy);

    // A fresh generate on the same client still works — no state leaked.
    let follow_up = proto::GenerateRequest {
        request_id: "rs-int-abort-ondrop-followup".to_string(),
        tokenized: Some(proto::TokenizedInput {
            original_text: String::new(),
            input_ids: input_ids(),
        }),
        sampling_params: Some(sampling_params(4)),
        stream: false,
        ..Default::default()
    };
    let mut tail = client
        .generate(follow_up)
        .await
        .expect("follow-up generate");
    let item = tail.next().await.expect("follow-up yielded nothing");
    let _ = item.expect("follow-up generate response");
}

// ---------------------------------------------------------------------------
// Trace injection
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires TOKENSPEED_GRPC_ENDPOINT"]
async fn trace_injector_fires_on_every_unary_rpc() {
    let (client, counter) = connect_with_counter().await;

    // Five unary paths that should all invoke inject_trace exactly once.
    client.health_check().await.expect("health_check");
    client.get_model_info().await.expect("get_model_info");
    client.get_server_info().await.expect("get_server_info");
    client
        .get_loads(proto::GetLoadsRequest::default())
        .await
        .expect("get_loads");
    client
        .abort_request("rs-int-nonexistent".to_string(), "trace-test".to_string())
        .await
        .expect("abort_request");

    assert_eq!(
        counter.load(Ordering::SeqCst),
        5,
        "trace injector was called {} time(s); expected 5",
        counter.load(Ordering::SeqCst)
    );
}

#[tokio::test]
#[ignore = "requires TOKENSPEED_GRPC_ENDPOINT"]
async fn trace_injector_fires_on_generate_and_embed() {
    let (client, counter) = connect_with_counter().await;

    let gen_req = proto::GenerateRequest {
        request_id: "rs-int-trace-gen".to_string(),
        tokenized: Some(proto::TokenizedInput {
            original_text: String::new(),
            input_ids: input_ids(),
        }),
        sampling_params: Some(sampling_params(2)),
        stream: false,
        ..Default::default()
    };
    let mut stream = client.generate(gen_req).await.expect("generate");
    while stream.next().await.is_some() {}

    // Embed against a generation model is expected to fail on the server,
    // but the client-side inject must still run before the request is sent.
    let embed_req = proto::EmbedRequest {
        request_id: "rs-int-trace-embed".to_string(),
        tokenized: Some(proto::TokenizedInput {
            original_text: String::new(),
            input_ids: input_ids(),
        }),
        ..Default::default()
    };
    let _ = client.embed(embed_req).await;

    assert_eq!(
        counter.load(Ordering::SeqCst),
        2,
        "trace injector fired {} time(s); expected 2 (generate + embed)",
        counter.load(Ordering::SeqCst)
    );
}
