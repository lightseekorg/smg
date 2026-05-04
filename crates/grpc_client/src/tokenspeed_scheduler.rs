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
//! Internally this client still leverages SGLang's
//! ``build_grpc_sampling_params_from_*`` helpers because the source-of-truth
//! is an OpenAI request and most fields map identically. We translate from
//! the SGLang-shaped ``GenerateRequest`` into a TokenSpeed-shaped one at the
//! wire boundary, and translate the streamed response back so the router's
//! ``ProtoGenerateStreamChunk`` / ``ProtoGenerateComplete`` accessors can
//! operate on a familiar shape. When TokenSpeed needs a field SGLang lacks,
//! add it to the proto and extend the translator — not the router.

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

use crate::{
    sglang_scheduler::{proto as sglang, SglangSchedulerClient},
    BoxedTraceInjector, NoopTraceInjector,
};

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
/// Yields ``sglang::GenerateResponse`` (translated from the on-wire
/// ``tokenspeed_proto::GenerateResponse``) so the router-side
/// ``ProtoGenerateStreamChunk`` / ``ProtoGenerateComplete`` accessors can
/// keep operating on a single shape. Sends an Abort RPC on Drop unless
/// ``mark_completed`` was called first — same lifecycle contract as
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
    // Yield SGLang-shaped responses so the router's wrapper enums don't need
    // a TokenSpeed variant for every chunk-accessor.
    type Item = Result<sglang::GenerateResponse, tonic::Status>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(Some(Ok(ts_resp))) => {
                Poll::Ready(Some(Ok(translate::generate_response(ts_resp))))
            }
        }
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
    ///
    /// Accepts an SGLang-shaped request for symmetry with the router's
    /// existing dispatch path; the translation to TokenSpeed's slimmer wire
    /// shape (drops mm_inputs, disagg, LoRA, hidden-states, etc.) happens
    /// here at the wire boundary.
    pub async fn generate(
        &self,
        req: sglang::GenerateRequest,
    ) -> Result<AbortOnDropStream, tonic::Status> {
        let request_id = req.request_id.clone();
        let ts_req = translate::generate_request(req);

        let mut client = self.client.clone();
        let mut request = Request::new(ts_req);

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
    // These produce SGLang-shaped requests so the router's existing
    // ``ProtoGenerateRequest::Sglang`` plumbing is reused. The wire-side
    // translation to TokenSpeed shape happens inside ``generate()`` above.
    //
    // Sampling-param construction delegates to SglangSchedulerClient's
    // ``pub(crate)`` helpers — same OpenAI source, same semantics.

    #[expect(
        clippy::unused_self,
        reason = "receiver kept for API parity with SglangSchedulerClient"
    )]
    pub fn build_generate_request_from_chat(
        &self,
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<sglang::GenerateRequest, String> {
        let sampling_params = SglangSchedulerClient::build_grpc_sampling_params_from_chat(
            body,
            tool_call_constraint,
        )?;
        Ok(sglang::GenerateRequest {
            request_id,
            tokenized: Some(sglang::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling_params),
            return_logprob: body.logprobs,
            logprob_start_len: -1,
            top_logprobs_num: body.top_logprobs.unwrap_or(0) as i32,
            stream: body.stream,
            ..Default::default()
        })
    }

    #[expect(
        clippy::unused_self,
        reason = "receiver kept for API parity with SglangSchedulerClient"
    )]
    pub fn build_plain_generate_request(
        &self,
        request_id: String,
        body: &GenerateRequest,
        original_text: Option<String>,
        token_ids: Vec<u32>,
    ) -> Result<sglang::GenerateRequest, String> {
        let sampling_params =
            SglangSchedulerClient::build_sampling_params_from_plain(body.sampling_params.as_ref())?;
        Ok(sglang::GenerateRequest {
            request_id,
            tokenized: Some(sglang::TokenizedInput {
                original_text: original_text.unwrap_or_default(),
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling_params),
            return_logprob: body.return_logprob.unwrap_or(false),
            logprob_start_len: body.logprob_start_len.unwrap_or(-1),
            top_logprobs_num: body.top_logprobs_num.unwrap_or(0),
            token_ids_logprob: body.token_ids_logprob.clone().unwrap_or_default(),
            stream: body.stream,
            ..Default::default()
        })
    }

    #[expect(
        clippy::unused_self,
        reason = "receiver kept for API parity with SglangSchedulerClient"
    )]
    pub fn build_generate_request_from_responses(
        &self,
        request_id: String,
        body: &ResponsesRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        constraint: Option<(String, String)>,
    ) -> Result<sglang::GenerateRequest, String> {
        let sampling_params =
            SglangSchedulerClient::build_grpc_sampling_params_from_responses(body, constraint)?;
        Ok(sglang::GenerateRequest {
            request_id,
            tokenized: Some(sglang::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling_params),
            stream: body.stream.unwrap_or(false),
            ..Default::default()
        })
    }

    #[expect(
        clippy::unused_self,
        reason = "receiver kept for API parity with SglangSchedulerClient"
    )]
    pub fn build_generate_request_from_messages(
        &self,
        request_id: String,
        body: &CreateMessageRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<sglang::GenerateRequest, String> {
        let sampling_params = SglangSchedulerClient::build_grpc_sampling_params_from_messages(
            body,
            tool_call_constraint,
        )?;
        Ok(sglang::GenerateRequest {
            request_id,
            tokenized: Some(sglang::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling_params),
            stream: body.stream.unwrap_or(false),
            ..Default::default()
        })
    }

    #[expect(
        clippy::unused_self,
        reason = "receiver kept for API parity with SglangSchedulerClient"
    )]
    pub fn build_generate_request_from_completion(
        &self,
        request_id: String,
        body: &CompletionRequest,
        original_text: String,
        token_ids: Vec<u32>,
    ) -> Result<sglang::GenerateRequest, String> {
        let sampling_params =
            SglangSchedulerClient::build_grpc_sampling_params_from_completion(body)?;
        Ok(sglang::GenerateRequest {
            request_id,
            tokenized: Some(sglang::TokenizedInput {
                original_text,
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling_params),
            return_logprob: body.logprobs.is_some(),
            logprob_start_len: -1,
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

// ── Wire-boundary translation ─────────────────────────────────────────
//
// Maps SGLang-shaped types (used internally by the router) to TokenSpeed's
// slimmer wire types and back. Fields TokenSpeed doesn't carry on the wire
// (mm_inputs, disagg, LoRA, hidden states, embeddings, etc.) are dropped on
// the way out; fields TokenSpeed doesn't return are filled with defaults on
// the way in. When the protocols genuinely diverge — i.e. TokenSpeed needs
// a field SGLang doesn't have — extend this module rather than threading
// new variants through proto_wrapper.
mod translate {
    use super::{sglang, tokenspeed_proto as ts};

    pub(super) fn sampling_params(s: sglang::SamplingParams) -> ts::SamplingParams {
        ts::SamplingParams {
            temperature: s.temperature,
            top_p: s.top_p,
            top_k: s.top_k,
            min_p: s.min_p,
            frequency_penalty: s.frequency_penalty,
            presence_penalty: s.presence_penalty,
            repetition_penalty: s.repetition_penalty,
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

    pub(super) fn generate_request(r: sglang::GenerateRequest) -> ts::GenerateRequest {
        ts::GenerateRequest {
            request_id: r.request_id,
            tokenized: r.tokenized.map(|t| ts::TokenizedInput {
                input_ids: t.input_ids,
                original_text: t.original_text,
            }),
            sampling_params: r.sampling_params.map(sampling_params),
            return_logprob: r.return_logprob,
            // SGLang's wire-side `logprob_start_len` is non-optional `i32`
            // with `-1` as the "no input logprobs" sentinel; TokenSpeed's
            // proto makes the field `optional` so the servicer can tell
            // "unset" from "explicit 0". Always wrap in `Some(...)` so
            // existing SGLang-shaped callers preserve their sentinel
            // through to the Python side.
            logprob_start_len: Some(r.logprob_start_len),
            top_logprobs_num: r.top_logprobs_num,
            token_ids_logprob: r.token_ids_logprob,
            stream: r.stream,
            // Fields TokenSpeed has no concept of:
            //   r.mm_inputs, r.disaggregated_params, r.custom_logit_processor,
            //   r.timestamp, r.input_embeds, r.lora_id, r.data_parallel_rank,
            //   r.log_metrics, r.return_hidden_states
            // — silently dropped here. Routing multimodal / disagg / LoRA
            // requests to a TokenSpeed worker is a router-level config bug,
            // not something this layer should try to paper over.
        }
    }

    pub(super) fn generate_response(r: ts::GenerateResponse) -> sglang::GenerateResponse {
        let response = r.response.map(|resp| match resp {
            ts::generate_response::Response::Chunk(c) => {
                sglang::generate_response::Response::Chunk(stream_chunk(c))
            }
            ts::generate_response::Response::Complete(c) => {
                sglang::generate_response::Response::Complete(complete(c))
            }
        });
        sglang::GenerateResponse {
            request_id: r.request_id,
            response,
        }
    }

    fn stream_chunk(c: ts::GenerateStreamChunk) -> sglang::GenerateStreamChunk {
        sglang::GenerateStreamChunk {
            token_ids: c.token_ids,
            prompt_tokens: c.prompt_tokens,
            completion_tokens: c.completion_tokens,
            cached_tokens: c.cached_tokens,
            output_logprobs: c.output_logprobs.map(output_logprobs),
            // Fields not on TokenSpeed's wire — defaulted.
            hidden_states: vec![],
            input_logprobs: None,
            index: c.index,
        }
    }

    fn complete(c: ts::GenerateComplete) -> sglang::GenerateComplete {
        let matched_stop = c.matched_stop.map(|m| match m {
            ts::generate_complete::MatchedStop::MatchedTokenId(id) => {
                sglang::generate_complete::MatchedStop::MatchedTokenId(id)
            }
            ts::generate_complete::MatchedStop::MatchedStopStr(s) => {
                sglang::generate_complete::MatchedStop::MatchedStopStr(s)
            }
        });
        sglang::GenerateComplete {
            output_ids: c.output_ids,
            finish_reason: c.finish_reason,
            prompt_tokens: c.prompt_tokens,
            completion_tokens: c.completion_tokens,
            cached_tokens: c.cached_tokens,
            output_logprobs: c.output_logprobs.map(output_logprobs),
            // Not on TokenSpeed's wire.
            all_hidden_states: vec![],
            input_logprobs: None,
            matched_stop,
            index: c.index,
        }
    }

    fn output_logprobs(o: ts::OutputLogProbs) -> sglang::OutputLogProbs {
        sglang::OutputLogProbs {
            token_logprobs: o.token_logprobs,
            token_ids: o.token_ids,
            top_logprobs: o
                .top_logprobs
                .into_iter()
                .map(|t| sglang::TopLogProbs {
                    values: t.values,
                    token_ids: t.token_ids,
                })
                .collect(),
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
