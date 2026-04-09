use std::{
    collections::HashMap,
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
    sampling_params::SamplingParams as GenerateSamplingParams,
};
use tonic::{transport::Channel, Request, Streaming};
use tracing::{debug, warn};

use crate::{BoxedTraceInjector, NoopTraceInjector};

// Include the generated protobuf code
#[expect(clippy::allow_attributes)]
pub mod proto {
    #![allow(clippy::all, clippy::absolute_paths, unused_qualifications)]
    tonic::include_proto!("mlx.grpc.engine");
}

/// A smart wrapper around Streaming<GenerateResponse> that automatically
/// sends abort when dropped (e.g., due to client disconnection or early termination).
pub struct AbortOnDropStream {
    inner: Streaming<proto::GenerateResponse>,
    request_id: String,
    client: MlxEngineClient,
    aborted: Arc<AtomicBool>,
}

impl AbortOnDropStream {
    /// Create a new auto-aborting stream wrapper
    pub fn new(
        stream: Streaming<proto::GenerateResponse>,
        request_id: String,
        client: MlxEngineClient,
    ) -> Self {
        debug!("Created AbortOnDropStream for request {}", request_id);
        Self {
            inner: stream,
            request_id,
            client,
            aborted: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Manually mark the request as completed to prevent abort on drop.
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
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

/// gRPC client for MLX engine
#[derive(Clone)]
pub struct MlxEngineClient {
    client: proto::mlx_engine_client::MlxEngineClient<Channel>,
    trace_injector: BoxedTraceInjector,
}

impl MlxEngineClient {
    /// Create a new client and connect to the MLX server
    pub async fn connect(endpoint: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::connect_with_trace_injector(endpoint, Arc::new(NoopTraceInjector)).await
    }

    /// Create a new client with a custom trace injector
    pub async fn connect_with_trace_injector(
        endpoint: &str,
        trace_injector: BoxedTraceInjector,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Connecting to MLX gRPC server at {}", endpoint);

        let http_endpoint = if let Some(addr) = endpoint.strip_prefix("grpc://") {
            format!("http://{addr}")
        } else {
            endpoint.to_string()
        };

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

        let client = proto::mlx_engine_client::MlxEngineClient::new(channel);

        Ok(Self {
            client,
            trace_injector,
        })
    }

    /// Set or replace the trace injector
    #[must_use]
    pub fn with_trace_injector(mut self, trace_injector: BoxedTraceInjector) -> Self {
        self.trace_injector = trace_injector;
        self
    }

    /// Submit a generation request (returns auto-aborting streaming response)
    pub async fn generate(
        &self,
        req: proto::GenerateRequest,
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
            self.clone(),
        ))
    }

    /// Perform health check
    pub async fn health_check(&self) -> Result<proto::HealthCheckResponse, tonic::Status> {
        debug!("Sending health check request");
        let request = Request::new(proto::HealthCheckRequest {});

        let mut client = self.client.clone();
        let response = client.health_check(request).await?;
        debug!("Health check response received");
        Ok(response.into_inner())
    }

    /// Abort a request
    pub async fn abort_request(
        &self,
        request_id: String,
        _reason: String,
    ) -> Result<(), tonic::Status> {
        debug!("Sending abort request for {}", request_id);
        let request = Request::new(proto::AbortRequest {
            request_ids: vec![request_id.clone()],
        });

        let mut client = self.client.clone();
        let _response = client.abort(request).await?;
        debug!("Abort response received for {}", request_id);
        Ok(())
    }

    /// Get model information
    pub async fn get_model_info(&self) -> Result<proto::GetModelInfoResponse, tonic::Status> {
        debug!("Requesting model info");
        let request = Request::new(proto::GetModelInfoRequest {});

        let mut client = self.client.clone();
        let response = client.get_model_info(request).await?;
        debug!("Model info response received");
        Ok(response.into_inner())
    }

    /// Get server information
    pub async fn get_server_info(&self) -> Result<proto::GetServerInfoResponse, tonic::Status> {
        debug!("Requesting server info");
        let request = Request::new(proto::GetServerInfoRequest {});

        let mut client = self.client.clone();
        let response = client.get_server_info(request).await?;
        debug!("Server info response received");
        Ok(response.into_inner())
    }

    crate::impl_get_tokenizer!();

    // ── Request builders ────────────────────────────────────────────────

    // ── Unsupported feature validation ────────────────────────────────
    //
    // TODO(mlx): Gaps preventing feature parity with vLLM/SGLang:
    //
    // mlx-lm engine limitations:
    //   - Constrained decoding (json_schema, regex, grammar, structural_tag)
    //     — needs outlines/xgrammar integration in mlx-lm
    //   - Parallel samples (n > 1) — mlx-lm server doesn't expose this
    //   - response_format — same as constrained decoding
    //
    // Servicer limitations (fixable without mlx-lm changes):
    //   - TODO(mlx): String stop sequences — mlx-lm supports this via
    //     tokenizer.encode() → SequenceStateMachine. Fix by converting stop
    //     strings to token IDs in the preparation stage (which already has the
    //     Rust tokenizer) and passing them as stop_token_ids in the proto.
    //
    // Track upstream: https://github.com/ml-explore/mlx-lm

    fn reject_constraint(constraint: Option<&(String, String)>) -> Result<(), String> {
        if let Some((kind, _)) = constraint {
            return Err(format!(
                "MLX backend does not support structured output constraint: {kind}"
            ));
        }
        Ok(())
    }

    fn reject_n(n: Option<u32>) -> Result<(), String> {
        if n.is_some_and(|n| n > 1) {
            return Err("MLX backend does not support n > 1 (parallel samples)".to_string());
        }
        Ok(())
    }

    fn reject_stop_strings(has_stop_strings: bool) -> Result<(), String> {
        if has_stop_strings {
            return Err("MLX backend does not support string stop sequences".to_string());
        }
        Ok(())
    }

    fn reject_response_format(has_response_format: bool) -> Result<(), String> {
        if has_response_format {
            return Err(
                "MLX backend does not support response_format (structured outputs)".to_string(),
            );
        }
        Ok(())
    }

    fn reject_if_any_constraint(
        json_schema: Option<&String>,
        regex: Option<&String>,
        ebnf: Option<&String>,
    ) -> Result<(), String> {
        if json_schema.is_some() || regex.is_some() || ebnf.is_some() {
            return Err("MLX backend does not support structured output constraints".to_string());
        }
        Ok(())
    }

    // ── Public request builders ────────────────────────────────────────

    /// Build a GenerateRequest from OpenAI ChatCompletionRequest
    #[expect(
        clippy::unused_self,
        reason = "method receiver kept for consistent public API across gRPC backends"
    )]
    pub fn build_generate_request_from_chat(
        &self,
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        constraint: Option<(String, String)>,
    ) -> Result<proto::GenerateRequest, String> {
        Self::reject_constraint(constraint.as_ref())?;
        Self::reject_n(body.n)?;
        Self::reject_stop_strings(body.stop.as_ref().is_some_and(|s| !s.is_empty()))?;
        Self::reject_response_format(body.response_format.is_some())?;

        let sampling_params = Self::build_sampling_params_from_chat(body);
        Ok(Self::make_generate_request(
            request_id,
            processed_text,
            token_ids,
            sampling_params,
            body.stream,
        ))
    }

    /// Build a GenerateRequest from CompletionRequest (`/v1/completions`)
    #[expect(
        clippy::unused_self,
        reason = "method receiver kept for consistent public API"
    )]
    pub fn build_generate_request_from_completion(
        &self,
        request_id: String,
        body: &CompletionRequest,
        original_text: String,
        token_ids: Vec<u32>,
    ) -> Result<proto::GenerateRequest, String> {
        Self::reject_n(body.n)?;
        Self::reject_stop_strings(body.stop.as_ref().is_some_and(|s| !s.is_empty()))?;
        Self::reject_if_any_constraint(
            body.json_schema.as_ref(),
            body.regex.as_ref(),
            body.ebnf.as_ref(),
        )?;

        let sampling_params = Self::build_sampling_params_from_completion(body);
        Ok(Self::make_generate_request(
            request_id,
            original_text,
            token_ids,
            sampling_params,
            body.stream,
        ))
    }

    /// Build a GenerateRequest from CreateMessageRequest (Anthropic Messages API)
    #[expect(
        clippy::unused_self,
        reason = "method receiver kept for consistent public API across gRPC backends"
    )]
    pub fn build_generate_request_from_messages(
        &self,
        request_id: String,
        body: &CreateMessageRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        constraint: Option<(String, String)>,
    ) -> Result<proto::GenerateRequest, String> {
        Self::reject_constraint(constraint.as_ref())?;
        Self::reject_stop_strings(body.stop_sequences.as_ref().is_some_and(|s| !s.is_empty()))?;

        let sampling_params = Self::build_sampling_params_from_messages(body);
        Ok(Self::make_generate_request(
            request_id,
            processed_text,
            token_ids,
            sampling_params,
            body.stream.unwrap_or(false),
        ))
    }

    /// Build a basic GenerateRequest from the native GenerateRequest
    #[expect(
        clippy::unused_self,
        reason = "method receiver kept for consistent public API across gRPC backends"
    )]
    pub fn build_plain_generate_request(
        &self,
        request_id: String,
        body: &GenerateRequest,
        original_text: Option<String>,
        token_ids: Vec<u32>,
    ) -> Result<proto::GenerateRequest, String> {
        if let Some(ref sp) = body.sampling_params {
            Self::reject_n(sp.n)?;
            Self::reject_stop_strings(sp.stop.as_ref().is_some_and(|s| !s.is_empty()))?;
            Self::reject_if_any_constraint(
                sp.json_schema.as_ref(),
                sp.regex.as_ref(),
                sp.ebnf.as_ref(),
            )?;
        }

        let sampling_params = Self::build_sampling_params_from_plain(body.sampling_params.as_ref());
        Ok(Self::make_generate_request(
            request_id,
            original_text.unwrap_or_default(),
            token_ids,
            sampling_params,
            body.stream,
        ))
    }

    /// Build a GenerateRequest from ResponsesRequest (OpenAI Responses API)
    #[expect(
        clippy::unused_self,
        reason = "method receiver kept for consistent public API across gRPC backends"
    )]
    pub fn build_generate_request_from_responses(
        &self,
        request_id: String,
        body: &ResponsesRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        constraint: Option<(String, String)>,
    ) -> Result<proto::GenerateRequest, String> {
        Self::reject_constraint(constraint.as_ref())?;
        Self::reject_stop_strings(body.stop.as_ref().is_some_and(|s| !s.is_empty()))?;

        let sampling_params = Self::build_sampling_params_from_responses(body);
        Ok(Self::make_generate_request(
            request_id,
            processed_text,
            token_ids,
            sampling_params,
            body.stream.unwrap_or(false),
        ))
    }

    /// Shared helper to construct the proto GenerateRequest.
    fn make_generate_request(
        request_id: String,
        text: String,
        token_ids: Vec<u32>,
        sampling_params: proto::SamplingParams,
        stream: bool,
    ) -> proto::GenerateRequest {
        proto::GenerateRequest {
            request_id,
            input: Some(proto::generate_request::Input::Tokenized(
                proto::TokenizedInput {
                    original_text: text,
                    input_ids: token_ids,
                },
            )),
            sampling_params: Some(sampling_params),
            stream,
        }
    }

    // ── Private sampling param builders ─────────────────────────────────

    #[expect(deprecated, reason = "seed is legacy but still forwarded to backends")]
    fn build_sampling_params_from_chat(request: &ChatCompletionRequest) -> proto::SamplingParams {
        let logprobs = if request.logprobs {
            Some(request.top_logprobs.unwrap_or(1).min(20) as i32)
        } else {
            None
        };

        proto::SamplingParams {
            temperature: request.temperature,
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.map(|v| v.max(0) as u32).unwrap_or(0),
            min_p: request.min_p.unwrap_or(0.0),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
            max_tokens: request.max_completion_tokens,
            stop_token_ids: request.stop_token_ids.clone().unwrap_or_default(),
            ignore_eos: request.ignore_eos,
            logprobs,
            logit_bias: convert_logit_bias(request.logit_bias.as_ref()),
            seed: request.seed.and_then(|s| i32::try_from(s).ok()),
        }
    }

    fn build_sampling_params_from_completion(request: &CompletionRequest) -> proto::SamplingParams {
        let logprobs = request.logprobs.map(|v| v.min(5) as i32);

        proto::SamplingParams {
            temperature: request.temperature,
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.map(|v| v.max(0) as u32).unwrap_or(0),
            min_p: request.min_p.unwrap_or(0.0),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
            max_tokens: request.max_tokens,
            stop_token_ids: request.stop_token_ids.clone().unwrap_or_default(),
            ignore_eos: request.ignore_eos,
            logprobs,
            logit_bias: convert_logit_bias(request.logit_bias.as_ref()),
            seed: request.seed.and_then(|s| i32::try_from(s).ok()),
        }
    }

    fn build_sampling_params_from_messages(
        request: &CreateMessageRequest,
    ) -> proto::SamplingParams {
        proto::SamplingParams {
            temperature: Some(request.temperature.unwrap_or(1.0) as f32),
            top_p: request.top_p.unwrap_or(1.0) as f32,
            top_k: request.top_k.unwrap_or(0),
            max_tokens: Some(request.max_tokens),
            repetition_penalty: 1.0, // 1.0 = no penalty (0.0 would penalize everything)
            ..Default::default()
        }
    }

    fn build_sampling_params_from_plain(
        params: Option<&GenerateSamplingParams>,
    ) -> proto::SamplingParams {
        let mut sampling = proto::SamplingParams {
            temperature: Some(1.0),
            top_p: 1.0,
            repetition_penalty: 1.0, // 1.0 = no penalty
            ..Default::default()
        };

        let Some(p) = params else {
            return sampling;
        };

        if let Some(val) = p.temperature {
            sampling.temperature = Some(val);
        }
        if let Some(val) = p.top_p {
            sampling.top_p = val;
        }
        if let Some(val) = p.top_k {
            sampling.top_k = val.max(0) as u32;
        }
        if let Some(val) = p.frequency_penalty {
            sampling.frequency_penalty = val;
        }
        if let Some(val) = p.presence_penalty {
            sampling.presence_penalty = val;
        }
        if let Some(val) = p.repetition_penalty {
            sampling.repetition_penalty = val;
        }
        if let Some(val) = p.min_p {
            sampling.min_p = val;
        }
        if let Some(val) = p.ignore_eos {
            sampling.ignore_eos = val;
        }
        if let Some(stop_token_ids) = &p.stop_token_ids {
            sampling.stop_token_ids.clone_from(stop_token_ids);
        }
        if let Some(max_new_tokens) = p.max_new_tokens {
            sampling.max_tokens = Some(max_new_tokens);
        }
        if let Some(seed) = p.sampling_seed {
            sampling.seed = i32::try_from(seed).ok();
        }

        sampling
    }

    fn build_sampling_params_from_responses(request: &ResponsesRequest) -> proto::SamplingParams {
        proto::SamplingParams {
            temperature: request.temperature,
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.max(0) as u32,
            min_p: request.min_p,
            repetition_penalty: request.repetition_penalty,
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            logit_bias: Default::default(),
            max_tokens: request.max_output_tokens,
            stop_token_ids: vec![],
            ignore_eos: false,
            logprobs: request.top_logprobs.map(|v| v as i32),
            seed: None,
        }
    }
}

/// Convert OpenAI-style logit_bias (String keys) to proto logit_bias (i32 keys).
fn convert_logit_bias(bias: Option<&HashMap<String, f32>>) -> HashMap<i32, f32> {
    match bias {
        Some(map) => map
            .iter()
            .filter_map(|(k, v)| k.parse::<i32>().ok().map(|id| (id, *v)))
            .collect(),
        None => HashMap::new(),
    }
}
