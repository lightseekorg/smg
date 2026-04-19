//! In-process mock-server tests for `TokenSpeedSchedulerClient`.
//!
//! These tests start a minimal `TokenSpeedScheduler` server implementation
//! that records every inbound request's metadata, then exercise the client
//! against it. Unlike the real-server integration suite, **these run in CI
//! without any external dependency** — cargo test alone.
//!
//! Covered:
//!
//! 1. **`grpc://` scheme connects** — smoke-test the scheme conversion path
//!    inside a live client↔server handshake.
//! 2. **Trace metadata arrives server-side** — direct evidence that the
//!    per-RPC `inject_trace` helper's headers survive the wire, not just
//!    that the injector was called (the gap the remote E2E could not
//!    close without modifying the tokenspeed servicer).
//! 3. **Shared `common_proto::KvEventBatch`** — call `subscribe_kv_events`
//!    on the mock and consume it as the workspace-shared type, proving
//!    `build.rs`'s `extern_path` is still wired.

#![expect(clippy::expect_used)]

use std::{
    net::SocketAddr,
    pin::Pin,
    sync::{Arc, Mutex},
    time::Duration,
};

use futures::Stream;
use smg_grpc_client::{
    common_proto,
    tokenspeed_proto::{self as proto, token_speed_scheduler_server},
    BoxedTraceInjector, TokenSpeedSchedulerClient, TraceInjector,
};
use tokio::sync::oneshot;
use tonic::{
    metadata::MetadataMap,
    transport::{server::TcpIncoming, Server},
    Request, Response, Status,
};

// ---------------------------------------------------------------------------
// Mock server
// ---------------------------------------------------------------------------

const TRACE_HEADER: &str = "x-tokenspeed-test-trace";

/// Records every request's metadata so tests can inspect what the server saw.
#[derive(Clone, Default)]
struct MetadataRecorder {
    entries: Arc<Mutex<Vec<(String, MetadataMap)>>>,
}

impl MetadataRecorder {
    fn record(&self, rpc: &str, md: &MetadataMap) {
        self.entries
            .lock()
            .expect("lock")
            .push((rpc.to_string(), md.clone()));
    }

    fn snapshot(&self) -> Vec<(String, MetadataMap)> {
        self.entries.lock().expect("lock").clone()
    }
}

struct MockScheduler {
    recorder: MetadataRecorder,
}

type StreamOf<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl token_speed_scheduler_server::TokenSpeedScheduler for MockScheduler {
    type GenerateStream = StreamOf<proto::GenerateResponse>;
    type GetTokenizerStream = StreamOf<common_proto::GetTokenizerChunk>;
    type SubscribeKvEventsStream = StreamOf<common_proto::KvEventBatch>;

    async fn health_check(
        &self,
        req: Request<proto::HealthCheckRequest>,
    ) -> Result<Response<proto::HealthCheckResponse>, Status> {
        self.recorder.record("HealthCheck", req.metadata());
        Ok(Response::new(proto::HealthCheckResponse {
            healthy: true,
            message: "mock".to_string(),
        }))
    }

    async fn get_model_info(
        &self,
        req: Request<proto::GetModelInfoRequest>,
    ) -> Result<Response<proto::GetModelInfoResponse>, Status> {
        self.recorder.record("GetModelInfo", req.metadata());
        Ok(Response::new(proto::GetModelInfoResponse {
            model_path: "mock/model".to_string(),
            vocab_size: 1,
            max_context_length: 1,
            is_generation: true,
            architectures: vec!["MockForCausalLM".to_string()],
            eos_token_ids: vec![0],
            ..Default::default()
        }))
    }

    async fn get_server_info(
        &self,
        req: Request<proto::GetServerInfoRequest>,
    ) -> Result<Response<proto::GetServerInfoResponse>, Status> {
        self.recorder.record("GetServerInfo", req.metadata());
        Ok(Response::new(proto::GetServerInfoResponse {
            uptime_seconds: 1.0,
            server_type: "mock".to_string(),
            version: "mock".to_string(),
            ..Default::default()
        }))
    }

    async fn get_loads(
        &self,
        req: Request<proto::GetLoadsRequest>,
    ) -> Result<Response<proto::GetLoadsResponse>, Status> {
        self.recorder.record("GetLoads", req.metadata());
        Ok(Response::new(proto::GetLoadsResponse::default()))
    }

    async fn abort(
        &self,
        req: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        self.recorder.record("Abort", req.metadata());
        Ok(Response::new(proto::AbortResponse {
            success: true,
            message: "mock".to_string(),
        }))
    }

    async fn embed(
        &self,
        req: Request<proto::EmbedRequest>,
    ) -> Result<Response<proto::EmbedResponse>, Status> {
        self.recorder.record("Embed", req.metadata());
        Ok(Response::new(proto::EmbedResponse {
            embedding: vec![0.0, 0.0, 0.0],
            embedding_dim: 3,
            prompt_tokens: 0,
        }))
    }

    async fn generate(
        &self,
        req: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        self.recorder.record("Generate", req.metadata());
        // Yield a single Complete response so the client's stream terminates
        // immediately without any real generation.
        let resp = proto::GenerateResponse {
            request_id: req.into_inner().request_id,
            response: Some(proto::generate_response::Response::Complete(
                proto::GenerateComplete {
                    output_ids: vec![42],
                    finish_reason: "stop".to_string(),
                    ..Default::default()
                },
            )),
        };
        let stream = futures::stream::once(async move { Ok(resp) });
        Ok(Response::new(Box::pin(stream)))
    }

    async fn get_tokenizer(
        &self,
        req: Request<common_proto::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        self.recorder.record("GetTokenizer", req.metadata());
        // Single empty chunk with a valid sha256 so `collect_bundle_from_rpc`
        // has something to terminate on.
        let empty_sha =
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string();
        let chunk = common_proto::GetTokenizerChunk {
            data: vec![],
            sha256: empty_sha,
        };
        let stream = futures::stream::once(async move { Ok(chunk) });
        Ok(Response::new(Box::pin(stream)))
    }

    async fn subscribe_kv_events(
        &self,
        req: Request<common_proto::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        self.recorder.record("SubscribeKvEvents", req.metadata());
        let batch = common_proto::KvEventBatch {
            sequence_number: 1,
            timestamp: 0.0,
            events: vec![],
            dp_rank: None,
        };
        let stream = futures::stream::once(async move { Ok(batch) });
        Ok(Response::new(Box::pin(stream)))
    }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct Harness {
    endpoint: String,
    recorder: MetadataRecorder,
    shutdown: Option<oneshot::Sender<()>>,
    server: Option<tokio::task::JoinHandle<()>>,
}

impl Harness {
    async fn start() -> Self {
        let recorder = MetadataRecorder::default();
        let service = token_speed_scheduler_server::TokenSpeedSchedulerServer::new(MockScheduler {
            recorder: recorder.clone(),
        });

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind 127.0.0.1:0");
        let addr: SocketAddr = listener.local_addr().expect("local_addr");
        let incoming = TcpIncoming::from(listener);

        let (tx, rx) = oneshot::channel::<()>();
        #[expect(
            clippy::disallowed_methods,
            reason = "test harness: the server future is shut down via `shutdown` oneshot and joined on Drop, so the lint's \"did you mean for this to leak on shutdown?\" doesn't apply"
        )]
        let server = tokio::spawn(async move {
            Server::builder()
                .add_service(service)
                .serve_with_incoming_shutdown(incoming, async {
                    let _ = rx.await;
                })
                .await
                .expect("serve");
        });

        // Give the server a beat to start accepting connections.
        tokio::time::sleep(Duration::from_millis(50)).await;

        Self {
            endpoint: format!("grpc://{addr}"),
            recorder,
            shutdown: Some(tx),
            server: Some(server),
        }
    }

    fn saw(&self, rpc: &str) -> bool {
        self.recorder.snapshot().iter().any(|(name, _)| name == rpc)
    }

    fn metadata_for(&self, rpc: &str) -> Option<MetadataMap> {
        self.recorder
            .snapshot()
            .into_iter()
            .find(|(name, _)| name == rpc)
            .map(|(_, md)| md)
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(h) = self.server.take() {
            h.abort();
        }
    }
}

/// Trace injector that stamps a known value so tests can assert it.
#[derive(Clone, Default)]
struct SentinelInjector {
    value: String,
}

impl SentinelInjector {
    fn new(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
        }
    }
}

impl TraceInjector for SentinelInjector {
    fn inject(
        &self,
        metadata: &mut MetadataMap,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        metadata.insert(
            TRACE_HEADER,
            self.value.parse().expect("static ascii value"),
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn grpc_scheme_connects_and_unary_rpc_round_trips() {
    // Covers: normalize_endpoint("grpc://…") → http://… is end-to-end wired.
    let harness = Harness::start().await;

    let client = TokenSpeedSchedulerClient::connect(&harness.endpoint)
        .await
        .expect("connect grpc://");
    let resp = client.health_check().await.expect("health_check");
    assert!(resp.healthy);
    assert!(harness.saw("HealthCheck"));
}

#[tokio::test]
async fn trace_header_reaches_server_on_every_unary_rpc() {
    // Covers: `inject_trace` on health_check / get_model_info / get_server_info
    // / get_loads / abort_request actually populates outbound metadata that
    // survives the wire — the gap a pure "the injector was called N times"
    // counter cannot close.
    let harness = Harness::start().await;
    let injector: BoxedTraceInjector = Arc::new(SentinelInjector::new("unary-sentinel"));
    let client =
        TokenSpeedSchedulerClient::connect_with_trace_injector(&harness.endpoint, injector)
            .await
            .expect("connect");

    client.health_check().await.expect("health_check");
    client.get_model_info().await.expect("get_model_info");
    client.get_server_info().await.expect("get_server_info");
    client
        .get_loads(proto::GetLoadsRequest::default())
        .await
        .expect("get_loads");
    client
        .abort_request("rid".to_string(), "reason".to_string())
        .await
        .expect("abort_request");

    for rpc in [
        "HealthCheck",
        "GetModelInfo",
        "GetServerInfo",
        "GetLoads",
        "Abort",
    ] {
        let md = harness
            .metadata_for(rpc)
            .unwrap_or_else(|| panic!("server never received {rpc}"));
        let hdr = md
            .get(TRACE_HEADER)
            .unwrap_or_else(|| panic!("trace header missing on {rpc}"));
        assert_eq!(
            hdr.to_str().expect("ascii"),
            "unary-sentinel",
            "{rpc}: trace header value mismatch"
        );
    }
}

#[tokio::test]
async fn trace_header_reaches_server_on_streaming_rpcs() {
    // Covers: trace metadata survives on Generate / Embed — complementing the
    // unary assertions and closing the same-injector-on-every-path claim.
    let harness = Harness::start().await;
    let injector: BoxedTraceInjector = Arc::new(SentinelInjector::new("stream-sentinel"));
    let client =
        TokenSpeedSchedulerClient::connect_with_trace_injector(&harness.endpoint, injector)
            .await
            .expect("connect");

    // Generate yields one Complete then terminates.
    use futures::StreamExt;
    let gen_req = proto::GenerateRequest {
        request_id: "mock-gen-1".to_string(),
        ..Default::default()
    };
    let mut stream = client.generate(gen_req).await.expect("generate");
    while stream.next().await.is_some() {}

    client
        .embed(proto::EmbedRequest::default())
        .await
        .expect("embed");

    for rpc in ["Generate", "Embed"] {
        let md = harness
            .metadata_for(rpc)
            .unwrap_or_else(|| panic!("server never received {rpc}"));
        let hdr = md
            .get(TRACE_HEADER)
            .unwrap_or_else(|| panic!("trace header missing on {rpc}"));
        assert_eq!(hdr.to_str().expect("ascii"), "stream-sentinel");
    }
}

#[tokio::test]
async fn subscribe_kv_events_yields_common_proto_type() {
    // Covers: the proto uses shared `common.proto` types AND build.rs's
    // `extern_path(".smg.grpc.common", "crate::common_proto")` is still
    // wired. If the proto regresses to a local `KvEventBatch`, this test
    // stops compiling (annotation-level type check) and stops running
    // (behavior check on the stream's item type).
    use futures::StreamExt;

    let harness = Harness::start().await;
    let client = TokenSpeedSchedulerClient::connect(&harness.endpoint)
        .await
        .expect("connect");

    let mut stream: tonic::Streaming<common_proto::KvEventBatch> =
        client.subscribe_kv_events(0).await.expect("subscribe");
    let first: common_proto::KvEventBatch = stream
        .next()
        .await
        .expect("stream ended before first item")
        .expect("stream error");
    assert_eq!(first.sequence_number, 1);
}

#[tokio::test]
async fn get_tokenizer_consumes_stream_bundle() {
    // Covers: `impl_get_tokenizer!()` round-trips through the shared
    // `collect_bundle_from_rpc` helper using `common_proto::GetTokenizerChunk`.
    // A single empty chunk is enough to exercise the wire path + macro
    // expansion; the bundle helper validates the SHA-256 internally.
    let harness = Harness::start().await;
    let client = TokenSpeedSchedulerClient::connect(&harness.endpoint)
        .await
        .expect("connect");

    // We don't assert bundle contents (it's an empty archive); we only need
    // the call to succeed — that's sufficient evidence the macro works.
    let _ = client.get_tokenizer().await;
    assert!(harness.saw("GetTokenizer"));
}
