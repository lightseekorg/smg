//! gRPC client for TokenSpeed scheduler.
//!
//! Connects to the `TokenSpeedScheduler` gRPC service exposed by
//! TokenSpeed's standalone gRPC launcher (`tokenspeed.runtime.grpc.launch`).
//! This is the SMG Router's interface to TokenSpeed backends.

use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::Duration,
};

use tonic::{transport::Channel, Request, Streaming};
use tracing::{debug, warn};

use crate::{BoxedTraceInjector, NoopTraceInjector};

/// Include the generated protobuf code for `tokenspeed.grpc.scheduler`.
#[expect(clippy::allow_attributes)]
pub mod proto {
    #![allow(clippy::all, clippy::absolute_paths, unused_qualifications)]
    tonic::include_proto!("tokenspeed.grpc.scheduler");
}

/// A smart wrapper around `Streaming<GenerateResponse>` that automatically
/// sends abort when dropped (RAII cleanup on client disconnect).
pub struct AbortOnDropStream {
    inner: Streaming<proto::GenerateResponse>,
    request_id: String,
    client: TokenSpeedSchedulerClient,
    aborted: Arc<AtomicBool>,
}

impl AbortOnDropStream {
    pub fn new(
        stream: Streaming<proto::GenerateResponse>,
        request_id: String,
        client: TokenSpeedSchedulerClient,
    ) -> Self {
        debug!("Created AbortOnDropStream for request {}", request_id);
        Self {
            inner: stream,
            request_id,
            client,
            aborted: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Mark the request as completed to prevent abort on drop.
    pub fn mark_completed(&self) {
        self.aborted.store(true, Ordering::Release);
        debug!("Request {} marked as completed", self.request_id);
    }
}

impl Drop for AbortOnDropStream {
    fn drop(&mut self) {
        if self
            .aborted
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }

        let client = self.client.clone();
        let request_id = self.request_id.clone();

        #[expect(
            clippy::disallowed_methods,
            reason = "fire-and-forget abort on Drop is intentional"
        )]
        tokio::spawn(async move {
            debug!(
                "Stream dropped without completion for request {}, sending abort",
                request_id
            );
            let request_id_for_log = request_id.clone();
            if let Err(e) = client
                .abort_request(request_id, "Stream dropped".to_string())
                .await
            {
                warn!(
                    "Failed to send abort on drop for request {}: {}",
                    request_id_for_log, e
                );
            }
        });
    }
}

impl futures::Stream for AbortOnDropStream {
    type Item = Result<proto::GenerateResponse, tonic::Status>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(None) => {
                // Stream finished naturally — mark completed to prevent abort on drop.
                self.mark_completed();
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(e))) => {
                // Stream errored — mark completed to prevent abort on drop
                // (the server already knows about the error).
                self.mark_completed();
                Poll::Ready(Some(Err(e)))
            }
            other => other,
        }
    }
}

/// Map `grpc://` → `http://` and `grpcs://` → `https://`, pass everything
/// else through unchanged. Kept as a private helper so it can be unit-tested
/// without spinning up a TLS server.
fn normalize_endpoint(endpoint: &str) -> String {
    if let Some(addr) = endpoint.strip_prefix("grpcs://") {
        format!("https://{addr}")
    } else if let Some(addr) = endpoint.strip_prefix("grpc://") {
        format!("http://{addr}")
    } else {
        endpoint.to_string()
    }
}

/// gRPC client for the TokenSpeed scheduler.
#[derive(Clone)]
pub struct TokenSpeedSchedulerClient {
    client: proto::token_speed_scheduler_client::TokenSpeedSchedulerClient<Channel>,
    trace_injector: BoxedTraceInjector,
}

impl TokenSpeedSchedulerClient {
    /// Create a new client and connect to the TokenSpeed scheduler.
    pub async fn connect(endpoint: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::connect_with_trace_injector(endpoint, Arc::new(NoopTraceInjector)).await
    }

    /// Create a new client with a custom trace injector.
    pub async fn connect_with_trace_injector(
        endpoint: &str,
        trace_injector: BoxedTraceInjector,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Connecting to TokenSpeed scheduler at {}", endpoint);

        let http_endpoint = normalize_endpoint(endpoint);

        let channel = Channel::from_shared(http_endpoint)?
            .http2_keep_alive_interval(Duration::from_secs(30))
            .keep_alive_timeout(Duration::from_secs(10))
            .keep_alive_while_idle(true)
            .tcp_keepalive(Some(Duration::from_secs(60)))
            .tcp_nodelay(true)
            .http2_adaptive_window(true)
            .initial_stream_window_size(Some(16 * 1024 * 1024))
            .initial_connection_window_size(Some(32 * 1024 * 1024))
            .connect()
            .await?;

        let client = proto::token_speed_scheduler_client::TokenSpeedSchedulerClient::new(channel);

        Ok(Self {
            client,
            trace_injector,
        })
    }

    /// Set or replace the trace injector.
    #[must_use]
    pub fn with_trace_injector(mut self, trace_injector: BoxedTraceInjector) -> Self {
        self.trace_injector = trace_injector;
        self
    }

    /// Submit a generation request (returns auto-aborting streaming response).
    pub async fn generate(
        &self,
        req: proto::GenerateRequest,
    ) -> Result<AbortOnDropStream, tonic::Status> {
        let request_id = req.request_id.clone();
        let mut client = self.client.clone();
        let mut request = Request::new(req);
        self.inject_trace(&mut request);

        let response = client.generate(request).await?;

        Ok(AbortOnDropStream::new(
            response.into_inner(),
            request_id,
            self.clone(),
        ))
    }

    /// Submit an embedding request.
    pub async fn embed(
        &self,
        req: proto::EmbedRequest,
    ) -> Result<proto::EmbedResponse, tonic::Status> {
        let mut client = self.client.clone();
        let mut request = Request::new(req);
        self.inject_trace(&mut request);

        let response = client.embed(request).await?;
        Ok(response.into_inner())
    }

    /// Inject trace context into the request metadata.
    fn inject_trace(&self, request: &mut Request<impl Sized>) {
        if let Err(e) = self.trace_injector.inject(request.metadata_mut()) {
            warn!("Failed to inject trace context: {}", e);
        }
    }

    /// Perform health check.
    pub async fn health_check(&self) -> Result<proto::HealthCheckResponse, tonic::Status> {
        debug!("Sending health check request");
        let mut request = Request::new(proto::HealthCheckRequest {});
        self.inject_trace(&mut request);
        let mut client = self.client.clone();
        let response = client.health_check(request).await?;
        Ok(response.into_inner())
    }

    /// Abort a request.
    pub async fn abort_request(
        &self,
        request_id: String,
        reason: String,
    ) -> Result<(), tonic::Status> {
        let mut request = Request::new(proto::AbortRequest { request_id, reason });
        self.inject_trace(&mut request);
        let mut client = self.client.clone();
        client.abort(request).await?;
        Ok(())
    }

    /// Get model information.
    pub async fn get_model_info(&self) -> Result<proto::GetModelInfoResponse, tonic::Status> {
        let mut request = Request::new(proto::GetModelInfoRequest {});
        self.inject_trace(&mut request);
        let mut client = self.client.clone();
        let response = client.get_model_info(request).await?;
        Ok(response.into_inner())
    }

    /// Get server information.
    pub async fn get_server_info(&self) -> Result<proto::GetServerInfoResponse, tonic::Status> {
        let mut request = Request::new(proto::GetServerInfoRequest {});
        self.inject_trace(&mut request);
        let mut client = self.client.clone();
        let response = client.get_server_info(request).await?;
        Ok(response.into_inner())
    }

    /// Get load metrics.
    pub async fn get_loads(
        &self,
        req: proto::GetLoadsRequest,
    ) -> Result<proto::GetLoadsResponse, tonic::Status> {
        let mut request = Request::new(req);
        self.inject_trace(&mut request);
        let mut client = self.client.clone();
        let response = client.get_loads(request).await?;
        Ok(response.into_inner())
    }

    crate::impl_get_tokenizer!();
    crate::impl_subscribe_kv_events!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proto_types_compilation() {
        // Verify all key proto types compile and can be constructed.
        let _health_req = proto::HealthCheckRequest {};
        let _health_resp = proto::HealthCheckResponse::default();
        let _model_info_req = proto::GetModelInfoRequest {};
        let _server_info_req = proto::GetServerInfoRequest {};
        let _loads_req = proto::GetLoadsRequest::default();
    }

    #[test]
    fn test_generate_request_construction() {
        let sampling_params = proto::SamplingParams {
            temperature: 0.7,
            max_new_tokens: Some(128),
            top_p: 0.9,
            top_k: 50,
            stop: vec!["</s>".to_string()],
            ..Default::default()
        };

        let gen_req = proto::GenerateRequest {
            request_id: "ts-test-001".to_string(),
            tokenized: Some(proto::TokenizedInput {
                original_text: "Hello from TokenSpeed".to_string(),
                input_ids: vec![9906, 1917, 505],
            }),
            sampling_params: Some(sampling_params),
            return_logprob: true,
            top_logprobs_num: 3,
            stream: true,
            ..Default::default()
        };

        assert_eq!(gen_req.request_id, "ts-test-001");
        assert!(gen_req.stream);
        assert!(gen_req.return_logprob);
        assert_eq!(gen_req.top_logprobs_num, 3);
        let params = gen_req.sampling_params.unwrap();
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.max_new_tokens, Some(128));
    }

    #[test]
    fn test_abort_request_construction() {
        let abort_req = proto::AbortRequest {
            request_id: "ts-abort-001".to_string(),
            reason: "Client disconnected".to_string(),
        };
        assert_eq!(abort_req.request_id, "ts-abort-001");
        assert_eq!(abort_req.reason, "Client disconnected");
    }

    #[test]
    fn test_sampling_params_proto3_defaults() {
        // Proto3 defaults (all zeros) — callers must convert to semantic defaults.
        let params = proto::SamplingParams::default();
        assert_eq!(params.temperature, 0.0); // Semantic: 1.0
        assert_eq!(params.top_p, 0.0); // Semantic: 1.0
        assert_eq!(params.top_k, 0); // Semantic: -1
        assert_eq!(params.repetition_penalty, 0.0); // Semantic: 1.0
        assert_eq!(params.max_new_tokens, None);
        assert_eq!(params.stream_interval, None);
        assert!(!params.skip_special_tokens);
        assert!(!params.no_stop_trim);
    }

    #[test]
    fn test_embed_request_construction() {
        let embed_req = proto::EmbedRequest {
            request_id: "ts-embed-001".to_string(),
            tokenized: Some(proto::TokenizedInput {
                original_text: "Embed this text".to_string(),
                input_ids: vec![10, 20, 30],
            }),
            data_parallel_rank: 0,
            ..Default::default()
        };
        assert_eq!(embed_req.request_id, "ts-embed-001");
        let tokenized = embed_req.tokenized.unwrap();
        assert_eq!(tokenized.input_ids, vec![10, 20, 30]);
    }

    #[test]
    fn test_generate_stream_chunk_with_tokenspeed_fields() {
        // TokenSpeed-specific fields: spec_verify_ct, accept_draft_tokens.
        let chunk = proto::GenerateStreamChunk {
            token_ids: vec![100, 200],
            prompt_tokens: 5,
            completion_tokens: 2,
            cached_tokens: 3,
            spec_verify_ct: 4,
            accept_draft_tokens: 0.75,
            index: 0,
            ..Default::default()
        };
        assert_eq!(chunk.token_ids, vec![100, 200]);
        assert_eq!(chunk.spec_verify_ct, 4);
        assert_eq!(chunk.accept_draft_tokens, 0.75);
    }

    #[test]
    fn test_generate_complete_with_matched_stop() {
        // Test matched_stop oneof: string variant.
        let complete_str = proto::GenerateComplete {
            output_ids: vec![1, 2, 3],
            finish_reason: "stop".to_string(),
            matched_stop: Some(proto::generate_complete::MatchedStop::MatchedStopStr(
                "<eos>".to_string(),
            )),
            spec_verify_ct: 2,
            accept_draft_tokens: 0.9,
            ..Default::default()
        };
        assert_eq!(complete_str.finish_reason, "stop");
        assert_eq!(complete_str.spec_verify_ct, 2);
        match complete_str.matched_stop {
            Some(proto::generate_complete::MatchedStop::MatchedStopStr(s)) => {
                assert_eq!(s, "<eos>");
            }
            _ => panic!("Expected MatchedStopStr"),
        }

        // Test matched_stop oneof: token_id variant.
        let complete_token = proto::GenerateComplete {
            output_ids: vec![4, 5],
            finish_reason: "stop".to_string(),
            matched_stop: Some(proto::generate_complete::MatchedStop::MatchedTokenId(
                128009,
            )),
            ..Default::default()
        };
        match complete_token.matched_stop {
            Some(proto::generate_complete::MatchedStop::MatchedTokenId(id)) => {
                assert_eq!(id, 128009);
            }
            _ => panic!("Expected MatchedTokenId"),
        }
    }

    #[test]
    fn test_generate_response_oneof() {
        // Chunk variant.
        let chunk_resp = proto::GenerateResponse {
            request_id: "r1".to_string(),
            response: Some(proto::generate_response::Response::Chunk(
                proto::GenerateStreamChunk {
                    token_ids: vec![42],
                    ..Default::default()
                },
            )),
        };
        assert!(chunk_resp.response.is_some());

        // Complete variant.
        let complete_resp = proto::GenerateResponse {
            request_id: "r1".to_string(),
            response: Some(proto::generate_response::Response::Complete(
                proto::GenerateComplete {
                    output_ids: vec![42, 43],
                    finish_reason: "length".to_string(),
                    ..Default::default()
                },
            )),
        };
        assert!(complete_resp.response.is_some());
    }

    #[test]
    fn test_model_info_response_fields() {
        let resp = proto::GetModelInfoResponse {
            model_path: "Qwen/Qwen3-0.6B".to_string(),
            tokenizer_path: "Qwen/Qwen3-0.6B".to_string(),
            is_generation: true,
            vocab_size: 151936,
            max_context_length: 131072,
            supports_vision: false,
            model_type: "qwen3".to_string(),
            architectures: vec!["Qwen3ForCausalLM".to_string()],
            eos_token_ids: vec![151645, 151643],
            pad_token_id: 151643,
            bos_token_id: 151643,
            max_req_input_len: 131071,
            ..Default::default()
        };
        assert_eq!(resp.vocab_size, 151936);
        assert!(resp.is_generation);
        assert_eq!(resp.architectures, vec!["Qwen3ForCausalLM"]);
    }

    #[test]
    fn test_logprobs_types() {
        let output_lp = proto::OutputLogProbs {
            token_logprobs: vec![-0.5, -1.2, -0.1],
            token_ids: vec![100, 200, 300],
            top_logprobs: vec![proto::TopLogProbs {
                values: vec![-0.1, -0.5, -1.0],
                token_ids: vec![300, 100, 50],
            }],
        };
        assert_eq!(output_lp.token_logprobs.len(), 3);
        assert_eq!(output_lp.top_logprobs.len(), 1);
        assert_eq!(output_lp.top_logprobs[0].values.len(), 3);
    }

    #[test]
    fn test_disaggregated_params() {
        let dp = proto::DisaggregatedParams {
            bootstrap_host: "10.0.0.1".to_string(),
            bootstrap_port: 8080,
            bootstrap_room: 0,
        };
        assert_eq!(dp.bootstrap_host, "10.0.0.1");
        assert_eq!(dp.bootstrap_port, 8080);
    }

    #[tokio::test]
    async fn test_client_connect_invalid_endpoint() {
        let result = TokenSpeedSchedulerClient::connect("invalid://endpoint").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_constraint_oneof() {
        // Regex constraint.
        let params_regex = proto::SamplingParams {
            constraint: Some(proto::sampling_params::Constraint::Regex(
                r"\d+".to_string(),
            )),
            ..Default::default()
        };
        match params_regex.constraint {
            Some(proto::sampling_params::Constraint::Regex(r)) => assert_eq!(r, r"\d+"),
            _ => panic!("Expected regex constraint"),
        }

        // JSON schema constraint.
        let params_json = proto::SamplingParams {
            constraint: Some(proto::sampling_params::Constraint::JsonSchema(
                r#"{"type":"object"}"#.to_string(),
            )),
            ..Default::default()
        };
        match params_json.constraint {
            Some(proto::sampling_params::Constraint::JsonSchema(s)) => {
                assert!(s.contains("object"));
            }
            _ => panic!("Expected json_schema constraint"),
        }
    }

    #[test]
    fn normalize_endpoint_rewrites_all_four_schemes() {
        // `grpc://` is the canonical cleartext alias used across SMG.
        assert_eq!(
            normalize_endpoint("grpc://example.com:50051"),
            "http://example.com:50051"
        );
        // `grpcs://` must be mapped to `https://` so tonic actually negotiates
        // TLS — a regression here is silent (the client would fall back to an
        // unknown-scheme URI error or, worse, plaintext).
        assert_eq!(
            normalize_endpoint("grpcs://example.com:50051"),
            "https://example.com:50051"
        );
        // Already-normalized URIs pass through unchanged.
        assert_eq!(
            normalize_endpoint("http://example.com:50051"),
            "http://example.com:50051"
        );
        assert_eq!(
            normalize_endpoint("https://example.com:50051"),
            "https://example.com:50051"
        );
        // Non-URI inputs pass through so tonic gives a clear parse error.
        assert_eq!(normalize_endpoint("garbage"), "garbage");
    }

    #[test]
    fn normalize_endpoint_preserves_path_and_query() {
        assert_eq!(
            normalize_endpoint("grpcs://host:443/prefix?tok=1"),
            "https://host:443/prefix?tok=1"
        );
    }

    /// Compile-time check: `subscribe_kv_events()` returns a stream of the
    /// shared `common_proto::KvEventBatch`, not a TokenSpeed-local type. If
    /// the proto ever stops importing `common.proto` (or the build script
    /// loses its `extern_path` mapping), this function stops compiling.
    async fn _subscribe_kv_events_returns_common_proto_stream(
        client: TokenSpeedSchedulerClient,
    ) -> Result<Streaming<crate::common_proto::KvEventBatch>, tonic::Status> {
        client.subscribe_kv_events(0).await
    }

    /// Compile-time check: `get_tokenizer()` goes through the shared
    /// `StreamBundle` wrapper (which internally consumes
    /// `common_proto::GetTokenizerChunk`). A regression where the proto
    /// reintroduces a local `GetTokenizerChunk` would refuse to compile
    /// because `collect_bundle_from_rpc` is parameterized on the common type.
    async fn _get_tokenizer_returns_stream_bundle(
        client: TokenSpeedSchedulerClient,
    ) -> Result<crate::tokenizer_bundle::StreamBundle, Box<dyn std::error::Error + Send + Sync>>
    {
        client.get_tokenizer().await
    }
}
