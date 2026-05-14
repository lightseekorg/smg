//! gRPC client for the TokenSpeed scheduler service.
//!
//! Wire types are TokenSpeed-native end-to-end (`tokenspeed_proto::*`).
//! Sampling params come from the shared `crate::sampling_params` helpers
//! and are field-mapped to TokenSpeed's shape via [`translate::sampling_params`].
//! The unary RPC adapters (`translate::model_info` / `server_info` / `loads`)
//! still emit the legacy router-side shape pending dedicated
//! `ModelInfo::TokenSpeed` / `ServerInfo::TokenSpeed` arms in the router.

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

/// Fire-and-forget abort sender invoked from `Drop`.
type AbortDispatcher = Arc<dyn Fn(String) + Send + Sync>;

/// Auto-aborting wrapper around the TokenSpeed generate stream.
/// Sends Abort on Drop unless `mark_completed` ran first.
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

        // Channel knobs match the other engine clients.
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
        let tokenspeed_sampling = crate::sampling_params::build_grpc_sampling_params_from_chat(
            body,
            tool_call_constraint,
        )?;
        let mut sampling = translate::sampling_params(tokenspeed_sampling);
        sampling.temperature = body.temperature;
        sampling.top_p = body.top_p;
        sampling.top_k = body.top_k;
        sampling.min_p = body.min_p;
        sampling.frequency_penalty = body.frequency_penalty;
        sampling.presence_penalty = body.presence_penalty;
        sampling.repetition_penalty = body.repetition_penalty;
        Ok(tokenspeed_proto::GenerateRequest {
            request_id,
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling),
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
        let tokenspeed_sampling = crate::sampling_params::build_sampling_params_from_plain(
            body.sampling_params.as_ref(),
        )?;
        let mut sampling = translate::sampling_params(tokenspeed_sampling);
        let plain = body.sampling_params.as_ref();
        sampling.temperature = plain.and_then(|p| p.temperature);
        sampling.top_p = plain.and_then(|p| p.top_p);
        sampling.top_k = plain.and_then(|p| p.top_k);
        sampling.min_p = plain.and_then(|p| p.min_p);
        sampling.frequency_penalty = plain.and_then(|p| p.frequency_penalty);
        sampling.presence_penalty = plain.and_then(|p| p.presence_penalty);
        sampling.repetition_penalty = plain.and_then(|p| p.repetition_penalty);
        Ok(tokenspeed_proto::GenerateRequest {
            request_id,
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                original_text: original_text.unwrap_or_default(),
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling),
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
        let tokenspeed_sampling =
            crate::sampling_params::build_grpc_sampling_params_from_responses(body, constraint)?;
        let mut sampling = translate::sampling_params(tokenspeed_sampling);
        sampling.temperature = body.temperature;
        sampling.top_p = body.top_p;
        sampling.frequency_penalty = body.frequency_penalty;
        sampling.presence_penalty = body.presence_penalty;
        Ok(tokenspeed_proto::GenerateRequest {
            request_id,
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling),
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
        let tokenspeed_sampling = crate::sampling_params::build_grpc_sampling_params_from_messages(
            body,
            tool_call_constraint,
        )?;
        let mut sampling = translate::sampling_params(tokenspeed_sampling);
        sampling.temperature = body.temperature.map(|v| v as f32);
        sampling.top_p = body.top_p.map(|v| v as f32);
        sampling.top_k = body.top_k.map(|v| v as i32);
        sampling.min_p = None;
        sampling.frequency_penalty = None;
        sampling.presence_penalty = None;
        sampling.repetition_penalty = None;
        Ok(tokenspeed_proto::GenerateRequest {
            request_id,
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling),
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
        let tokenspeed_sampling =
            crate::sampling_params::build_grpc_sampling_params_from_completion(body)?;
        let mut sampling = translate::sampling_params(tokenspeed_sampling);
        sampling.temperature = body.temperature;
        sampling.top_p = body.top_p;
        sampling.top_k = body.top_k;
        sampling.min_p = body.min_p;
        sampling.frequency_penalty = body.frequency_penalty;
        sampling.presence_penalty = body.presence_penalty;
        sampling.repetition_penalty = body.repetition_penalty;
        Ok(tokenspeed_proto::GenerateRequest {
            request_id,
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                original_text,
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling),
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

// Sampling-params + unary RPC adapters: map between TokenSpeed's wire
// shape and the router-side shape the metadata wrappers consume.
mod translate {
    use super::{sglang, tokenspeed_proto as ts};

    pub(super) fn sampling_params(s: sglang::SamplingParams) -> ts::SamplingParams {
        // Source scalars are non-optional with semantic defaults already
        // applied; wrap in `Some(_)` so the servicer's `HasField()` checks
        // distinguish "set" from "unset" on the wire.
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
        // Surface TokenSpeed's `preferred_sampling_params` JSON in both label
        // fields the discovery path may consult, so worker-discovery can
        // expose model-published defaults (`temperature`, `top_p`, etc.) to
        // the router's default-injection stage.
        let preferred = r.preferred_sampling_params;
        sglang::GetModelInfoResponse {
            model_path: r.model_path,
            tokenizer_path: r.tokenizer_path,
            is_generation: true,
            preferred_sampling_params: preferred.clone(),
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
            default_sampling_params_json: preferred,
        }
    }

    pub(super) fn server_info(r: ts::GetServerInfoResponse) -> sglang::GetServerInfoResponse {
        sglang::GetServerInfoResponse {
            server_args: r.server_args,
            scheduler_info: r.scheduler_info,
            active_requests: r.active_requests,
            is_paused: r.is_paused,
            last_receive_timestamp: 0.0,
            uptime_seconds: r.uptime_seconds,
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
