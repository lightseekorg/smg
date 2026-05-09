//! gRPC client for the TokenSpeed scheduler service.
//!
//! TokenSpeed has a fully independent wire definition (see
//! ``proto/tokenspeed_scheduler.proto``) — distinct package
//! (``tokenspeed.grpc.scheduler``), distinct service, distinct messages with
//! intentionally trimmed field sets aimed at the top-tier LLM workloads
//! (Kimi K2, MiniMax M2, Qwen 3, gpt-oss, DeepSeek V4). Anything SGLang has
//! that doesn't apply here (PD-disaggregated serving, multimodal inputs,
//! LoRA hot-swap, hidden-state forwarding, embeddings, classifier outputs,
//! tokenizer streaming, KV-event subscription) is simply not on TokenSpeed's
//! wire.
//!
//! Request/response types are TokenSpeed-native end-to-end: the stream
//! yields ``tokenspeed_proto::GenerateResponse`` and ``build_*_request``
//! produces ``tokenspeed_proto::GenerateRequest``. The router dispatches
//! through dedicated ``ProtoGenerateRequest::TokenSpeed`` /
//! ``ProtoGenerateStreamChunk::TokenSpeed`` /
//! ``ProtoGenerateComplete::TokenSpeed`` arms — same shape as the other
//! per-backend variants.
//!
//! Sampling-params construction reuses the backend-neutral helpers in
//! ``crate::sampling_params`` (which currently return
//! ``sglang::SamplingParams``); the [`translate::sampling_params`]
//! field-mapper converts to TokenSpeed's shape at the seam. The unary RPC
//! responses (``GetModelInfo``, ``GetServerInfo``, ``GetLoads``) still
//! return SGLang-shaped types because their consumers ride the
//! ``ModelInfo`` / ``ServerInfo`` SGLang variants in the router; that's a
//! separate cleanup.

use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::Duration,
};

use openai_protocol::{
    chat::ChatCompletionRequest, completion::CompletionRequest, generate::GenerateRequest,
    messages::CreateMessageRequest, responses::ResponsesRequest,
};
use tonic::{transport::Channel, Request, Streaming};
use tracing::{debug, warn};

use crate::{sglang_scheduler::proto as sglang, BoxedTraceInjector, NoopTraceInjector};

#[expect(clippy::allow_attributes)]
pub mod tokenspeed_proto {
    #![allow(clippy::all, clippy::absolute_paths, unused_qualifications)]
    tonic::include_proto!("tokenspeed.grpc.scheduler");
}

/// Fire-and-forget abort sender used by [`AbortOnDropStream`]. The closure
/// captures the TokenSpeed client that owns the stream so ``Drop`` can
/// dispatch the abort RPC over the same connection without ``Drop`` itself
/// being async. Local to this module — SGLang's equivalent stream type
/// holds its own client field directly and doesn't need this indirection.
type AbortDispatcher = Arc<dyn Fn(String) + Send + Sync>;

/// Auto-aborting wrapper around the TokenSpeed generate stream.
///
/// Yields ``tokenspeed_proto::GenerateResponse`` directly (no translation
/// at the seam). Sends an Abort RPC on Drop unless ``mark_completed`` was
/// called first — same lifecycle contract as
/// ``sglang_scheduler::AbortOnDropStream``.
pub struct AbortOnDropStream {
    inner: Streaming<tokenspeed_proto::GenerateResponse>,
    request_id: String,
    abort_dispatcher: AbortDispatcher,
    aborted: Arc<AtomicBool>,
}

impl AbortOnDropStream {
    pub fn new(
        stream: Streaming<tokenspeed_proto::GenerateResponse>,
        request_id: String,
        abort_dispatcher: AbortDispatcher,
    ) -> Self {
        debug!(
            "Created TokenSpeed AbortOnDropStream for request {}",
            request_id
        );
        Self {
            inner: stream,
            request_id,
            abort_dispatcher,
            aborted: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn mark_completed(&self) {
        self.aborted.store(true, Ordering::Release);
        debug!("TokenSpeed request {} marked as completed", self.request_id);
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
        debug!(
            "TokenSpeed stream dropped without completion for request {}, sending abort",
            self.request_id
        );
        (self.abort_dispatcher)(self.request_id.clone());
    }
}

impl futures::Stream for AbortOnDropStream {
    type Item = Result<tokenspeed_proto::GenerateResponse, tonic::Status>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

/// gRPC client for the TokenSpeed scheduler.
#[derive(Clone)]
pub struct TokenSpeedSchedulerClient {
    client: tokenspeed_proto::token_speed_scheduler_client::TokenSpeedSchedulerClient<Channel>,
    trace_injector: BoxedTraceInjector,
}

impl TokenSpeedSchedulerClient {
    pub async fn connect(endpoint: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::connect_with_trace_injector(endpoint, Arc::new(NoopTraceInjector)).await
    }

    pub async fn connect_with_trace_injector(
        endpoint: &str,
        trace_injector: BoxedTraceInjector,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Connecting to TokenSpeed scheduler at {}", endpoint);

        let http_endpoint = if let Some(addr) = endpoint.strip_prefix("grpc://") {
            format!("http://{addr}")
        } else {
            endpoint.to_string()
        };

        // Same channel knobs as SglangSchedulerClient — independent of the
        // service being called and proven load-appropriate in prod.
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

        let client =
            tokenspeed_proto::token_speed_scheduler_client::TokenSpeedSchedulerClient::new(channel);

        Ok(Self {
            client,
            trace_injector,
        })
    }

    #[must_use]
    pub fn with_trace_injector(mut self, trace_injector: BoxedTraceInjector) -> Self {
        self.trace_injector = trace_injector;
        self
    }

    /// Submit a generation request.
    pub async fn generate(
        &self,
        req: tokenspeed_proto::GenerateRequest,
    ) -> Result<AbortOnDropStream, tonic::Status> {
        let request_id = req.request_id.clone();

        let mut client = self.client.clone();
        let mut request = Request::new(req);

        if let Err(e) = self.trace_injector.inject(request.metadata_mut()) {
            warn!("Failed to inject trace context: {}", e);
        }

        let response = client.generate(request).await?;

        Ok(AbortOnDropStream::new(
            response.into_inner(),
            request_id,
            tokenspeed_abort_dispatcher(self.clone()),
        ))
    }

    pub async fn health_check(&self) -> Result<sglang::HealthCheckResponse, tonic::Status> {
        debug!("Sending TokenSpeed health check request");
        let request = Request::new(tokenspeed_proto::HealthCheckRequest {});
        let mut client = self.client.clone();
        let response = client.health_check(request).await?;
        let r = response.into_inner();
        Ok(sglang::HealthCheckResponse {
            healthy: r.healthy,
            message: r.message,
        })
    }

    pub async fn abort_request(
        &self,
        request_id: String,
        reason: String,
    ) -> Result<(), tonic::Status> {
        debug!(
            "Sending TokenSpeed abort for {} (reason: {})",
            request_id, reason
        );
        let request = Request::new(tokenspeed_proto::AbortRequest {
            request_id: request_id.clone(),
            reason,
        });
        let mut client = self.client.clone();
        let response = client.abort(request).await?;
        debug!(
            "TokenSpeed abort response for {}: success={}, message={}",
            request_id,
            response.get_ref().success,
            response.get_ref().message
        );
        Ok(())
    }

    pub async fn get_model_info(&self) -> Result<sglang::GetModelInfoResponse, tonic::Status> {
        let request = Request::new(tokenspeed_proto::GetModelInfoRequest {});
        let mut client = self.client.clone();
        let response = client.get_model_info(request).await?;
        Ok(translate::model_info(response.into_inner()))
    }

    pub async fn get_server_info(&self) -> Result<sglang::GetServerInfoResponse, tonic::Status> {
        let request = Request::new(tokenspeed_proto::GetServerInfoRequest {});
        let mut client = self.client.clone();
        let response = client.get_server_info(request).await?;
        Ok(translate::server_info(response.into_inner()))
    }

    pub async fn get_loads(
        &self,
        include: Vec<String>,
    ) -> Result<sglang::GetLoadsResponse, tonic::Status> {
        let request = Request::new(tokenspeed_proto::GetLoadsRequest {
            dp_rank: None,
            include,
        });
        let mut client = self.client.clone();
        let response = client.get_loads(request).await?;
        Ok(translate::loads(response.into_inner()))
    }

    // ── Request builders ──────────────────────────────────────────────
    //
    // Produce ``tokenspeed_proto::GenerateRequest`` directly. Sampling-param
    // construction delegates to ``crate::sampling_params`` (which returns
    // ``sglang::SamplingParams`` because that proto is the most permissive
    // shape across our backends); ``translate::sampling_params`` field-maps
    // it to TokenSpeed's slimmer shape at the seam.

    #[expect(
        clippy::unused_self,
        reason = "receiver kept for API parity with the other engine clients"
    )]
    pub fn build_generate_request_from_chat(
        &self,
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<tokenspeed_proto::GenerateRequest, String> {
        let sglang_sampling = crate::sampling_params::build_grpc_sampling_params_from_chat(
            body,
            tool_call_constraint,
        )?;
        Ok(tokenspeed_proto::GenerateRequest {
            request_id,
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            sampling_params: Some(translate::sampling_params(sglang_sampling)),
            return_logprob: body.logprobs,
            logprob_start_len: Some(-1),
            top_logprobs_num: body.top_logprobs.unwrap_or(0) as i32,
            stream: body.stream,
            ..Default::default()
        })
    }

    #[expect(
        clippy::unused_self,
        reason = "receiver kept for API parity with the other engine clients"
    )]
    pub fn build_plain_generate_request(
        &self,
        request_id: String,
        body: &GenerateRequest,
        original_text: Option<String>,
        token_ids: Vec<u32>,
    ) -> Result<tokenspeed_proto::GenerateRequest, String> {
        let sglang_sampling = crate::sampling_params::build_sampling_params_from_plain(
            body.sampling_params.as_ref(),
        )?;
        Ok(tokenspeed_proto::GenerateRequest {
            request_id,
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                original_text: original_text.unwrap_or_default(),
                input_ids: token_ids,
            }),
            sampling_params: Some(translate::sampling_params(sglang_sampling)),
            return_logprob: body.return_logprob.unwrap_or(false),
            logprob_start_len: Some(body.logprob_start_len.unwrap_or(-1)),
            top_logprobs_num: body.top_logprobs_num.unwrap_or(0),
            token_ids_logprob: body.token_ids_logprob.clone().unwrap_or_default(),
            stream: body.stream,
        })
    }

    #[expect(
        clippy::unused_self,
        reason = "receiver kept for API parity with the other engine clients"
    )]
    pub fn build_generate_request_from_responses(
        &self,
        request_id: String,
        body: &ResponsesRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        constraint: Option<(String, String)>,
    ) -> Result<tokenspeed_proto::GenerateRequest, String> {
        let sglang_sampling =
            crate::sampling_params::build_grpc_sampling_params_from_responses(body, constraint)?;
        Ok(tokenspeed_proto::GenerateRequest {
            request_id,
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            sampling_params: Some(translate::sampling_params(sglang_sampling)),
            stream: body.stream.unwrap_or(false),
            ..Default::default()
        })
    }

    #[expect(
        clippy::unused_self,
        reason = "receiver kept for API parity with the other engine clients"
    )]
    pub fn build_generate_request_from_messages(
        &self,
        request_id: String,
        body: &CreateMessageRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<tokenspeed_proto::GenerateRequest, String> {
        let sglang_sampling = crate::sampling_params::build_grpc_sampling_params_from_messages(
            body,
            tool_call_constraint,
        )?;
        Ok(tokenspeed_proto::GenerateRequest {
            request_id,
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            sampling_params: Some(translate::sampling_params(sglang_sampling)),
            stream: body.stream.unwrap_or(false),
            ..Default::default()
        })
    }

    #[expect(
        clippy::unused_self,
        reason = "receiver kept for API parity with the other engine clients"
    )]
    pub fn build_generate_request_from_completion(
        &self,
        request_id: String,
        body: &CompletionRequest,
        original_text: String,
        token_ids: Vec<u32>,
    ) -> Result<tokenspeed_proto::GenerateRequest, String> {
        let sglang_sampling =
            crate::sampling_params::build_grpc_sampling_params_from_completion(body)?;
        Ok(tokenspeed_proto::GenerateRequest {
            request_id,
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                original_text,
                input_ids: token_ids,
            }),
            sampling_params: Some(translate::sampling_params(sglang_sampling)),
            return_logprob: body.logprobs.is_some(),
            logprob_start_len: Some(-1),
            top_logprobs_num: body.logprobs.unwrap_or(0) as i32,
            stream: body.stream,
            ..Default::default()
        })
    }
}

/// Spawn a fire-and-forget abort RPC against TokenSpeed when an
/// ``AbortOnDropStream`` is dropped without completion.
fn tokenspeed_abort_dispatcher(client: TokenSpeedSchedulerClient) -> AbortDispatcher {
    Arc::new(move |request_id: String| {
        let client = client.clone();
        let request_id_for_log = request_id.clone();
        #[expect(
            clippy::disallowed_methods,
            reason = "fire-and-forget abort on Drop is intentional"
        )]
        tokio::spawn(async move {
            if let Err(e) = client
                .abort_request(request_id, "Stream dropped".to_string())
                .await
            {
                warn!(
                    "Failed to send TokenSpeed abort on drop for request {}: {}",
                    request_id_for_log, e
                );
            }
        });
    })
}

// ── Sampling-params translation + unary RPC adapters ─────────────────
//
// `sampling_params` field-maps the sglang-shaped sampling params produced
// by `crate::sampling_params` into TokenSpeed's slimmer wire shape. The
// other adapters (model_info / server_info / loads) translate unary RPC
// responses into the SGLang-shaped types the router's metadata wrappers
// currently consume — that's a follow-up cleanup that can ride on top of
// the per-backend `ModelInfo` / `ServerInfo` enums.
mod translate {
    use super::{sglang, tokenspeed_proto as ts};

    pub(super) fn sampling_params(s: sglang::SamplingParams) -> ts::SamplingParams {
        // sglang's proto declares numeric scalars as non-optional, so the Rust
        // router has already substituted semantic defaults (e.g.
        // ``temperature=1.0``, ``top_p=1.0``, ``repetition_penalty=1.0``)
        // before getting here. tokenspeed's proto declares the same fields
        // as ``optional`` so the servicer can use ``HasField()`` to
        // distinguish presence — wrap the (already-defaulted) sglang values
        // in ``Some(...)`` to mark them as explicitly set on the wire. This
        // preserves the pre-fix behavior while letting future direct-to-
        // tokenspeed clients use ``None`` to mean "let the engine default
        // apply" (e.g. for health-probe / warmup paths that would otherwise
        // hit ``top_p must be in (0, 1], got 0.0``).
        ts::SamplingParams {
            temperature: Some(s.temperature),
            top_p: Some(s.top_p),
            top_k: Some(s.top_k),
            min_p: Some(s.min_p),
            frequency_penalty: Some(s.frequency_penalty),
            presence_penalty: Some(s.presence_penalty),
            repetition_penalty: Some(s.repetition_penalty),
            max_new_tokens: s.max_new_tokens,
            min_new_tokens: s.min_new_tokens,
            stop: s.stop,
            stop_token_ids: s.stop_token_ids,
            ignore_eos: s.ignore_eos,
            skip_special_tokens: s.skip_special_tokens,
            spaces_between_special_tokens: s.spaces_between_special_tokens,
            no_stop_trim: s.no_stop_trim,
            n: s.n,
            logit_bias: s.logit_bias,
            constraint: s.constraint.map(constraint),
            custom_params: s.custom_params,
        }
    }

    fn constraint(c: sglang::sampling_params::Constraint) -> ts::sampling_params::Constraint {
        match c {
            sglang::sampling_params::Constraint::Regex(r) => {
                ts::sampling_params::Constraint::Regex(r)
            }
            sglang::sampling_params::Constraint::JsonSchema(s) => {
                ts::sampling_params::Constraint::JsonSchema(s)
            }
            sglang::sampling_params::Constraint::EbnfGrammar(g) => {
                ts::sampling_params::Constraint::EbnfGrammar(g)
            }
            sglang::sampling_params::Constraint::StructuralTag(t) => {
                ts::sampling_params::Constraint::StructuralTag(t)
            }
        }
    }

    pub(super) fn model_info(r: ts::GetModelInfoResponse) -> sglang::GetModelInfoResponse {
        sglang::GetModelInfoResponse {
            model_path: r.model_path,
            tokenizer_path: r.tokenizer_path,
            // TokenSpeed only serves generative LLMs at this layer; classifier
            // / embedding models are out of scope. Hard-code accordingly.
            is_generation: true,
            preferred_sampling_params: r.preferred_sampling_params,
            weight_version: r.weight_version,
            served_model_name: r.served_model_name,
            max_context_length: r.max_context_length,
            vocab_size: r.vocab_size,
            supports_vision: false,
            model_type: r.model_type,
            eos_token_ids: r.eos_token_ids,
            pad_token_id: r.pad_token_id,
            bos_token_id: r.bos_token_id,
            max_req_input_len: r.max_req_input_len,
            architectures: r.architectures,
            id2label_json: String::new(),
            num_labels: 0,
            default_sampling_params_json: String::new(),
        }
    }

    pub(super) fn server_info(r: ts::GetServerInfoResponse) -> sglang::GetServerInfoResponse {
        sglang::GetServerInfoResponse {
            server_args: r.server_args,
            scheduler_info: r.scheduler_info,
            active_requests: r.active_requests,
            is_paused: r.is_paused,
            // TokenSpeed scheduler doesn't track this — router doesn't read
            // it for TokenSpeed either, so a fixed 0 is fine.
            last_receive_timestamp: 0.0,
            uptime_seconds: r.uptime_seconds,
            // sglang_version field on the SGLang struct is the runtime version;
            // for TokenSpeed we surface the TokenSpeed version through the same
            // slot since downstream metric labels keep the field name.
            sglang_version: r.tokenspeed_version,
            server_type: "grpc".to_string(),
            start_time: r.start_time,
            max_total_num_tokens: r.max_total_num_tokens,
        }
    }

    pub(super) fn loads(r: ts::GetLoadsResponse) -> sglang::GetLoadsResponse {
        sglang::GetLoadsResponse {
            timestamp: r.timestamp,
            version: r.version,
            dp_rank_count: r.dp_rank_count,
            loads: r.loads.into_iter().map(scheduler_load).collect(),
            aggregate: r.aggregate.map(aggregate_metrics),
        }
    }

    fn scheduler_load(s: ts::SchedulerLoad) -> sglang::SchedulerLoad {
        sglang::SchedulerLoad {
            dp_rank: s.dp_rank,
            num_running_reqs: s.num_running_reqs,
            num_waiting_reqs: s.num_waiting_reqs,
            num_total_reqs: s.num_total_reqs,
            num_used_tokens: s.num_used_tokens,
            max_total_num_tokens: s.max_total_num_tokens,
            token_usage: s.token_usage,
            gen_throughput: s.gen_throughput,
            cache_hit_rate: s.cache_hit_rate,
            utilization: s.utilization,
            max_running_requests: s.max_running_requests,
            memory: s.memory.map(|m| sglang::MemoryMetrics {
                weight_gb: m.weight_gb,
                kv_cache_gb: m.kv_cache_gb,
                graph_gb: m.graph_gb,
                token_capacity: m.token_capacity,
            }),
            // TokenSpeed's wire intentionally omits speculative / LoRA /
            // disaggregation metrics — fill the SGLang-shaped slots with
            // None so callers ignore them.
            speculative: None,
            lora: None,
            disaggregation: None,
            queues: s.queues.map(|q| sglang::QueueMetrics {
                waiting: q.waiting,
                grammar: q.grammar,
                paused: q.paused,
                retracted: q.retracted,
            }),
        }
    }

    fn aggregate_metrics(a: ts::AggregateMetrics) -> sglang::AggregateMetrics {
        sglang::AggregateMetrics {
            total_running_reqs: a.total_running_reqs,
            total_waiting_reqs: a.total_waiting_reqs,
            total_reqs: a.total_reqs,
            avg_token_usage: a.avg_token_usage,
            avg_throughput: a.avg_throughput,
            avg_utilization: a.avg_utilization,
        }
    }
}
