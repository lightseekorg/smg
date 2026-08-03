// SPDX-License-Identifier: Apache-2.0
//
// ZMQ backend adapter (gateway glue): presents the vLLM engine surface (the same
// proto request/response types as `VllmEngineClient`) but speaks ZMQ directly to
// a same-host engine (vLLM EngineCore or TokenSpeed) via `engine-zmq-client`,
// bypassing the gRPC Python servicer.
//
// This bridges the gateway's proto request-execution pipeline to the raw ZMQ
// transport, so it lives with the router (which owns `GrpcClient`/`ProtoStream`),
// not in `smg-grpc-client` (pure gRPC) or `engine-zmq-client` (pure transport).
// It consumes the exact `vllm::GenerateRequest` the existing vLLM builders
// produce and emits `vllm::GenerateResponse` built from `EngineCoreOutput`, so
// the request-execution stage is reused unchanged.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use engine_zmq_client::{
    connect_handshake,
    connector::{EngineCoreClient, EngineCoreStream, TokenSpeedClient, TokenSpeedStream},
    protocol::{
        tokenspeed::{
            output::TokenSpeedOutput, request::TokenizedGenerateReqInput,
            sampling::SamplingParams as TokenSpeedSamplingParams,
        },
        vllm::{
            output::{EngineCoreFinishReason, EngineCoreOutput, StopReason},
            request::EngineCoreRequest,
            sampling::EngineCoreSamplingParams,
        },
        EngineLoad,
    },
    ConnectedEngine,
};
use futures::{stream::SelectAll, Stream, StreamExt};
use openai_protocol::worker::{SchedulerLoadSnapshot, WorkerLoadResponse};
use smg_grpc_client::vllm_proto as vllm;

use crate::worker::RuntimeType;

/// Loopback host for the same-host ZMQ transport (TCP handshake and local
/// binds). Shared with the worker-side socket derivation.
pub(crate) const ZMQ_LOOPBACK_HOST: &str = "127.0.0.1";

/// The engine protocol a ZMQ backend speaks. Both share the transport and
/// handshake; only the request/output struct shapes and the translation to/from
/// SMG proto differ.
#[derive(Clone)]
enum ZmqBackend {
    /// vLLM EngineCore.
    Vllm(Arc<EngineCoreClient>),
    /// TokenSpeed.
    TokenSpeed(Arc<TokenSpeedClient>),
}

/// Direct ZMQ connection to a same-host engine (vLLM EngineCore or TokenSpeed),
/// presented behind the vLLM gRPC client surface.
#[derive(Clone)]
pub struct ZmqEngineClient {
    backend: ZmqBackend,
    /// Model id advertised for metadata (the engine does not report it on the
    /// wire; it is configured at worker registration).
    model_id: String,
}

impl ZmqEngineClient {
    /// Bind the frontend sockets and complete the handshake with the engine(s),
    /// which must already be running and dialing `handshake_address`.
    ///
    /// `input_address`/`output_address` are the `ipc://` data-plane endpoints the
    /// engines connect to (chosen by SMG). `engine_count` is the number of DP
    /// ranks to await. `runtime` selects the wire protocol spoken over the shared
    /// transport (vLLM EngineCore vs TokenSpeed).
    pub async fn connect(
        handshake_address: &str,
        input_address: &str,
        output_address: &str,
        engine_count: usize,
        model_id: String,
        runtime: RuntimeType,
        timeout: Duration,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Single-engine scope for TokenSpeed: its wire carries no DP-rank routing
        // yet (`data_parallel_rank` is always `None`), so more than one engine
        // would silently send all traffic to engine 0. Reject it loudly until
        // DP>1 lands. The engine count is known here (the handshake awaits it).
        if matches!(runtime, RuntimeType::TokenSpeed) && engine_count > 1 {
            return Err(format!(
                "TokenSpeed ZMQ backend supports a single engine only (got \
                 engine_count={engine_count}); DP>1 is not yet supported"
            )
            .into());
        }
        // No silent fallback: any other runtime has no ZMQ engine adapter.
        // Reject before the handshake — no such engine ever dials in, so the
        // handshake would just block for the full timeout.
        if !matches!(
            runtime,
            RuntimeType::Vllm | RuntimeType::TokenSpeed | RuntimeType::Unspecified
        ) {
            return Err(format!(
                "ZMQ direct backend has no engine implementation for runtime \
                 {runtime}; only vllm and tokenspeed are supported"
            )
            .into());
        }

        let transport = connect_handshake(
            handshake_address,
            engine_count,
            ZMQ_LOOPBACK_HOST,
            Some(input_address),
            Some(output_address),
            timeout,
        )
        .await?;
        let backend = match runtime {
            RuntimeType::TokenSpeed => {
                ZmqBackend::TokenSpeed(Arc::new(TokenSpeedClient::new(transport)))
            }
            // vLLM EngineCore is the default ZMQ wire; an unspecified runtime maps
            // to it for backward compatibility (see `detect_backend`). All other
            // runtimes were rejected before the handshake.
            _ => ZmqBackend::Vllm(Arc::new(EngineCoreClient::new(transport))),
        };
        Ok(Self { backend, model_id })
    }

    /// The engine runtime behind this connection (the wire protocol chosen at
    /// connect time).
    pub fn runtime(&self) -> RuntimeType {
        match &self.backend {
            ZmqBackend::Vllm(_) => RuntimeType::Vllm,
            ZmqBackend::TokenSpeed(_) => RuntimeType::TokenSpeed,
        }
    }

    /// The engines connected on the shared transport (same handshake for both
    /// protocols).
    fn engines(&self) -> &[ConnectedEngine] {
        match &self.backend {
            ZmqBackend::Vllm(client) => client.engines(),
            ZmqBackend::TokenSpeed(client) => client.engines(),
        }
    }

    /// Submit a generate request and return a stream of vLLM-proto responses.
    /// The request is translated into the backend's wire protocol.
    ///
    /// Over gRPC the engine-side frontend (e.g. vLLM's AsyncLLM) fans `n` out
    /// itself and multiplexes the choices onto one stream. The raw ZMQ wire has
    /// no such frontend, so `n > 1` is fanned out HERE into `n` independent
    /// single-sample engine requests (see [`fan_out_requests`]); their outputs
    /// are merged back into one stream with each sub tagged via the proto
    /// `index` field, exactly like the gRPC contract.
    pub async fn generate(
        &self,
        req: vllm::GenerateRequest,
    ) -> Result<ZmqGenerateStream, tonic::Status> {
        let subs = fan_out_requests(req);
        // Sub-streams submitted before a mid-loop failure are dropped with the
        // error, which auto-aborts their engine-side requests.
        match &self.backend {
            ZmqBackend::Vllm(client) => {
                let mut streams = SelectAll::new();
                for (index, sub) in subs.into_iter().enumerate() {
                    let request =
                        translate_request(sub).map_err(tonic::Status::invalid_argument)?;
                    let stream = client.submit(request).await.map_err(zmq_status)?;
                    streams.push(VllmGenerateStream::new(stream, index as u32));
                }
                Ok(ZmqGenerateStream::Vllm(streams))
            }
            ZmqBackend::TokenSpeed(client) => {
                let mut streams = SelectAll::new();
                for (index, sub) in subs.into_iter().enumerate() {
                    let request = translate_request_tokenspeed(sub)
                        .map_err(tonic::Status::invalid_argument)?;
                    let stream = client.submit(request).await.map_err(zmq_status)?;
                    streams.push(TokenSpeedGenerateStream::new(stream, index as u32));
                }
                Ok(ZmqGenerateStream::TokenSpeed(streams))
            }
        }
    }

    /// Local liveness: false once the connection observed `ENGINE_CORE_DEAD` or
    /// a transport failure. No RPC (the raw ZMQ wire has no health RPC).
    pub fn is_alive(&self) -> bool {
        match &self.backend {
            ZmqBackend::Vllm(client) => client.is_alive(),
            ZmqBackend::TokenSpeed(client) => client.is_alive(),
        }
    }

    /// Health as an RPC-shaped response, derived from local liveness.
    pub fn health_check(&self) -> vllm::HealthCheckResponse {
        let alive = self.is_alive();
        vllm::HealthCheckResponse {
            healthy: alive,
            message: if alive {
                "ok".to_string()
            } else {
                "engine core dead".to_string()
            },
        }
    }

    /// Latest per-rank load for one engine index, if the backend carries it.
    /// vLLM piggybacks it on every batch; TokenSpeed does not (always `None`).
    fn engine_load(&self, engine_index: u32) -> Option<EngineLoad> {
        match &self.backend {
            ZmqBackend::Vllm(client) => client.engine_load(engine_index),
            ZmqBackend::TokenSpeed(client) => client.engine_load(engine_index),
        }
    }

    /// Per-rank load from the piggybacked `scheduler_stats` (SMG's DP routing
    /// signal), in the same shape as the gRPC `GetLoads` response. TokenSpeed
    /// carries no piggybacked load, so its response has no per-rank entries.
    pub fn get_loads(&self) -> WorkerLoadResponse {
        let loads: Vec<SchedulerLoadSnapshot> = self
            .engines()
            .iter()
            .filter_map(|engine| {
                let dp_rank = engine.engine_id.engine_index()?;
                let load = self.engine_load(dp_rank)?;
                Some(SchedulerLoadSnapshot {
                    dp_rank: i32::try_from(dp_rank).unwrap_or(i32::MAX),
                    num_running_reqs: i32::try_from(load.num_running).unwrap_or(i32::MAX),
                    num_waiting_reqs: i32::try_from(load.num_waiting).unwrap_or(i32::MAX),
                    token_usage: load.kv_cache_usage,
                    ..Default::default()
                })
            })
            .collect();
        WorkerLoadResponse {
            timestamp: String::new(),
            dp_rank_count: i32::try_from(loads.len()).unwrap_or(i32::MAX),
            loads,
        }
    }

    /// Model info derived from the handshake `EngineCoreReadyResponse` plus the
    /// configured model id (the engine does not report tokenizer/vocab metadata,
    /// so those come from worker config).
    pub fn get_model_info(&self) -> vllm::GetModelInfoResponse {
        let max_context_length = self
            .engines()
            .first()
            .map(|e| e.ready_response.max_model_len)
            .unwrap_or(0);
        vllm::GetModelInfoResponse {
            model_path: self.model_id.clone(),
            served_model_name: self.model_id.clone(),
            tokenizer_path: self.model_id.clone(),
            is_generation: true,
            max_context_length: u32::try_from(max_context_length).unwrap_or(u32::MAX),
            ..Default::default()
        }
    }

    /// Server info derived from the handshake response.
    pub fn get_server_info(&self) -> vllm::GetServerInfoResponse {
        let data_parallel_size = self
            .engines()
            .first()
            .map(|e| e.ready_response.data_parallel_size)
            .unwrap_or(1);
        let server_type = match &self.backend {
            ZmqBackend::Vllm(_) => "vllm",
            ZmqBackend::TokenSpeed(_) => "tokenspeed",
        };
        vllm::GetServerInfoResponse {
            data_parallel_size: i32::try_from(data_parallel_size).unwrap_or(i32::MAX),
            server_type: server_type.to_string(),
            ..Default::default()
        }
    }
}

/// Streaming generate output over ZMQ, presented as vLLM-proto
/// `GenerateResponse`. One sub-stream per fanned-out engine request (n>1);
/// they are polled together and yield as ready (interleaved), each tagging its
/// responses with its choice `index`. The merged stream ends when every sub
/// has delivered its terminal `Complete`. Dropping it before that aborts all
/// still-running engine-side sub-requests, so no explicit abort or
/// `mark_completed` is required. Which variant is active is fixed by the
/// backend protocol chosen at connect time.
pub enum ZmqGenerateStream {
    /// vLLM EngineCore outputs.
    Vllm(SelectAll<VllmGenerateStream>),
    /// TokenSpeed outputs.
    TokenSpeed(SelectAll<TokenSpeedGenerateStream>),
}

impl ZmqGenerateStream {
    /// Next vLLM-proto response, or `None` when the stream ends.
    pub async fn next(&mut self) -> Option<Result<vllm::GenerateResponse, tonic::Status>> {
        match self {
            Self::Vllm(streams) => streams.next().await,
            Self::TokenSpeed(streams) => streams.next().await,
        }
    }

    /// No-op: the ZMQ stream aborts natively on drop, so there is nothing to
    /// mark. Present for parity with the tonic abort-on-drop streams.
    #[expect(
        clippy::unused_self,
        reason = "kept for API parity with the tonic abort-on-drop streams"
    )]
    pub fn mark_completed(&mut self) {}
}

/// Accumulated per-request token counts shared by both stream mappers.
#[derive(Default)]
struct StreamState {
    output_ids: Vec<u32>,
    completion_tokens: u32,
    prompt_tokens: u32,
    cached_tokens: u32,
    /// Cumulative sampled-token logprobs across ticks. The proto contract is
    /// incremental per streaming chunk but cumulative on the terminal
    /// `Complete`, so accumulate here and drain into `Complete`.
    output_logprobs_val: Vec<f32>,
    output_logprobs_idx: Vec<u32>,
}

impl StreamState {
    /// Cumulative sampled-token logprobs for the terminal `Complete`, drained
    /// from the accumulated state (`None` when logprobs were not requested).
    fn take_complete_logprobs(&mut self) -> Option<vllm::OutputLogProbs> {
        (!self.output_logprobs_val.is_empty()).then(|| vllm::OutputLogProbs {
            token_logprobs: std::mem::take(&mut self.output_logprobs_val),
            token_ids: std::mem::take(&mut self.output_logprobs_idx),
            ..Default::default()
        })
    }
}

/// Streaming generate output for one vLLM EngineCore sub-request, mapping each
/// `EngineCoreOutput` to a vLLM-proto `GenerateResponse` (chunks until the
/// terminal output, then a complete), tagged with this sub's choice `index`.
pub struct VllmGenerateStream {
    inner: EngineCoreStream,
    state: StreamState,
    /// Choice index stamped on every chunk/complete (0 for n=1; the fan-out
    /// position for n>1) — the proto field the pipeline demuxes choices by.
    index: u32,
    /// Terminal `Complete` held back when the finish tick also carried new
    /// tokens: streaming frontends decode text/logprobs from chunks only, so
    /// the tick's delta goes out as a `Chunk` first.
    pending: Option<vllm::GenerateResponse>,
}

impl VllmGenerateStream {
    fn new(inner: EngineCoreStream, index: u32) -> Self {
        Self {
            inner,
            state: StreamState::default(),
            index,
            pending: None,
        }
    }

    fn map_output(
        &mut self,
        output: EngineCoreOutput,
    ) -> Result<vllm::GenerateResponse, tonic::Status> {
        let state = &mut self.state;
        if let Some(stats) = &output.prefill_stats {
            state.prompt_tokens = stats.num_prompt_tokens;
            state.cached_tokens = stats.num_cached_tokens;
        }
        let token_ids = output.new_token_ids;
        state.completion_tokens += token_ids.len() as u32;
        state.output_ids.extend(token_ids.iter().copied());

        // Sampled-token logprobs (entry 0 per position), if requested. Chunks
        // carry this tick's increment; the terminal `Complete` carries the
        // cumulative set, so accumulate into `state` and drain it on finish.
        let mut tick_logprobs_val = Vec::new();
        let mut tick_logprobs_idx = Vec::new();
        if let Some(logprobs) = &output.new_logprobs {
            let decoded = logprobs.as_direct().ok_or_else(|| {
                // The protocol layer resolves wire logprobs during decode, so
                // an unresolved payload here is a protocol bug — fail loudly.
                tonic::Status::internal("unresolved wire logprobs in engine output")
            })?;
            for position in &decoded.positions {
                if let Some(sampled) = position.entries.first() {
                    tick_logprobs_val.push(sampled.logprob);
                    tick_logprobs_idx.push(sampled.token_id);
                }
            }
        }
        let chunk_logprobs = (!tick_logprobs_val.is_empty()).then(|| vllm::OutputLogProbs {
            token_logprobs: tick_logprobs_val.clone(),
            token_ids: tick_logprobs_idx.clone(),
            ..Default::default()
        });
        state.output_logprobs_val.extend(tick_logprobs_val);
        state.output_logprobs_idx.extend(tick_logprobs_idx);

        let response = match output.finish_reason {
            Some(reason) => {
                let complete = vllm::GenerateResponse {
                    response: Some(vllm::generate_response::Response::Complete(
                        vllm::GenerateComplete {
                            output_ids: std::mem::take(&mut state.output_ids),
                            finish_reason: finish_reason_str(reason).to_string(),
                            prompt_tokens: state.prompt_tokens,
                            completion_tokens: state.completion_tokens,
                            cached_tokens: state.cached_tokens,
                            matched_stop: output.stop_reason.map(map_matched_stop),
                            output_logprobs: state.take_complete_logprobs(),
                            index: self.index,
                            ..Default::default()
                        },
                    )),
                };
                if token_ids.is_empty() {
                    return Ok(complete);
                }
                // The finish tick carried new tokens: emit them as a `Chunk`
                // first and hold the `Complete` for the next poll.
                let chunk = vllm::generate_response::Response::Chunk(vllm::GenerateStreamChunk {
                    token_ids,
                    prompt_tokens: state.prompt_tokens,
                    completion_tokens: state.completion_tokens,
                    cached_tokens: state.cached_tokens,
                    output_logprobs: chunk_logprobs,
                    index: self.index,
                    ..Default::default()
                });
                self.pending = Some(complete);
                chunk
            }
            None => vllm::generate_response::Response::Chunk(vllm::GenerateStreamChunk {
                token_ids,
                prompt_tokens: state.prompt_tokens,
                completion_tokens: state.completion_tokens,
                cached_tokens: state.cached_tokens,
                output_logprobs: chunk_logprobs,
                index: self.index,
                ..Default::default()
            }),
        };
        Ok(vllm::GenerateResponse {
            response: Some(response),
        })
    }
}

impl Stream for VllmGenerateStream {
    type Item = Result<vllm::GenerateResponse, tonic::Status>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::task::Poll;
        let this = self.get_mut();
        if let Some(pending) = this.pending.take() {
            return Poll::Ready(Some(Ok(pending)));
        }
        match std::pin::Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(output))) => Poll::Ready(Some(this.map_output(output))),
            Poll::Ready(Some(Err(error))) => Poll::Ready(Some(Err(zmq_status(error)))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Streaming generate output for one TokenSpeed sub-request, mapping each
/// `TokenSpeedOutput` to a vLLM-proto `GenerateResponse`, tagged with this
/// sub's choice `index`.
pub struct TokenSpeedGenerateStream {
    inner: TokenSpeedStream,
    state: StreamState,
    /// Choice index stamped on every chunk/complete (0 for n=1; the fan-out
    /// position for n>1) — the proto field the pipeline demuxes choices by.
    index: u32,
    /// Terminal `Complete` held back when the finish tick also carried new
    /// tokens: streaming frontends decode text/logprobs from chunks only, so
    /// the tick's delta goes out as a `Chunk` first.
    pending: Option<vllm::GenerateResponse>,
}

impl TokenSpeedGenerateStream {
    fn new(inner: TokenSpeedStream, index: u32) -> Self {
        Self {
            inner,
            state: StreamState::default(),
            index,
            pending: None,
        }
    }

    fn map_output(&mut self, output: TokenSpeedOutput) -> vllm::GenerateResponse {
        let state = &mut self.state;
        // TokenSpeed reports per-request token counts directly (cumulative for
        // completions), rather than vLLM's per-output prefill-stats deltas.
        if output.prompt_tokens > 0 {
            state.prompt_tokens = output.prompt_tokens;
        }
        if output.cached_tokens > 0 {
            state.cached_tokens = output.cached_tokens;
        }
        state.completion_tokens = output.completion_tokens;
        state.output_ids.extend(output.output_ids.iter().copied());

        // Sampled-token logprobs, if requested. The proto column is `float`, so
        // downcast the wire's `f64` values. Chunks carry this tick's increment;
        // the terminal `Complete` carries the cumulative set, so accumulate
        // into `state` and drain it on finish.
        let chunk_logprobs =
            (!output.output_logprobs_val.is_empty()).then(|| vllm::OutputLogProbs {
                token_logprobs: output
                    .output_logprobs_val
                    .iter()
                    .map(|&lp| lp as f32)
                    .collect(),
                token_ids: output.output_logprobs_idx.clone(),
                ..Default::default()
            });
        state
            .output_logprobs_val
            .extend(output.output_logprobs_val.iter().map(|&lp| lp as f32));
        state
            .output_logprobs_idx
            .extend(output.output_logprobs_idx.iter().copied());

        let response = match output.finish_reason {
            Some(reason) => {
                let complete = vllm::GenerateResponse {
                    response: Some(vllm::generate_response::Response::Complete(
                        vllm::GenerateComplete {
                            output_ids: std::mem::take(&mut state.output_ids),
                            finish_reason: normalize_finish_reason(&reason).to_string(),
                            prompt_tokens: state.prompt_tokens,
                            completion_tokens: state.completion_tokens,
                            cached_tokens: state.cached_tokens,
                            output_logprobs: state.take_complete_logprobs(),
                            index: self.index,
                            ..Default::default()
                        },
                    )),
                };
                if output.output_ids.is_empty() {
                    return complete;
                }
                // The finish tick carried new tokens: emit them as a `Chunk`
                // first and hold the `Complete` for the next poll.
                let chunk = vllm::generate_response::Response::Chunk(vllm::GenerateStreamChunk {
                    token_ids: output.output_ids,
                    prompt_tokens: state.prompt_tokens,
                    completion_tokens: state.completion_tokens,
                    cached_tokens: state.cached_tokens,
                    output_logprobs: chunk_logprobs,
                    index: self.index,
                    ..Default::default()
                });
                self.pending = Some(complete);
                chunk
            }
            None => vllm::generate_response::Response::Chunk(vllm::GenerateStreamChunk {
                token_ids: output.output_ids,
                prompt_tokens: state.prompt_tokens,
                completion_tokens: state.completion_tokens,
                cached_tokens: state.cached_tokens,
                output_logprobs: chunk_logprobs,
                index: self.index,
                ..Default::default()
            }),
        };
        vllm::GenerateResponse {
            response: Some(response),
        }
    }
}

impl Stream for TokenSpeedGenerateStream {
    type Item = Result<vllm::GenerateResponse, tonic::Status>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::task::Poll;
        let this = self.get_mut();
        if let Some(pending) = this.pending.take() {
            return Poll::Ready(Some(Ok(pending)));
        }
        match std::pin::Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(output))) => Poll::Ready(Some(Ok(this.map_output(output)))),
            Poll::Ready(Some(Err(error))) => Poll::Ready(Some(Err(zmq_status(error)))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Split an `n > 1` generate request into `n` independent single-sample wire
/// requests (an `n <= 1` request passes through untouched).
///
/// - Rids: sub `i` is `"{request_id}-{i}"` — engine-side rids must be unique,
///   and the pipeline's request id is already unique per request, so the
///   suffixed forms are too. Dropping the merged stream aborts every sub.
/// - Seeds: with no explicit seed each sub keeps `None` — the engine seeds each
///   rid independently, so the samples differ. An explicit seed becomes
///   `seed + i` per sub: a fixed seed with identical params would otherwise
///   make all n samples identical, and deriving distinct per-sample seeds from
///   the request seed is the established engine convention (each of the n
///   sequences gets its own sampler state for exactly this reason) while
///   staying deterministic for repeat runs.
/// - Usage: every sub reports the full `prompt_tokens` on its `Complete` (the
///   subs share one prompt), matching the gRPC engines' n>1 contract — the
///   pipeline de-duplicates (max per prompt), so nothing is counted n times.
fn fan_out_requests(req: vllm::GenerateRequest) -> Vec<vllm::GenerateRequest> {
    let n = req.sampling_params.as_ref().map_or(1, |sp| sp.n.max(1));
    if n <= 1 {
        return vec![req];
    }
    (0..n)
        .map(|i| {
            let mut sub = req.clone();
            sub.request_id = format!("{}-{i}", req.request_id);
            if let Some(sp) = sub.sampling_params.as_mut() {
                sp.n = 1;
                sp.seed = sp.seed.map(|seed| seed.wrapping_add(i as i32));
            }
            sub
        })
        .collect()
}

/// Translate a vLLM-proto generate request into a TokenSpeed
/// `TokenizedGenerateReqInput`. ZMQ mode requires pre-tokenized input (SMG
/// tokenizes upstream).
fn translate_request_tokenspeed(
    req: vllm::GenerateRequest,
) -> Result<TokenizedGenerateReqInput, String> {
    let input_ids = match req.input {
        Some(vllm::generate_request::Input::Tokenized(tokenized)) => tokenized.input_ids,
        Some(vllm::generate_request::Input::Text(_)) => {
            return Err("ZMQ mode requires pre-tokenized input (TokenizedInput)".to_string());
        }
        None => {
            return Err("ZMQ mode requires pre-tokenized input; no input provided".to_string());
        }
    };
    let stream = req.stream;
    // Single-engine TokenSpeed: a pinned DP rank other than 0 cannot be honored.
    if req.data_parallel_rank.is_some_and(|rank| rank != 0) {
        return Err(format!(
            "invalid data_parallel_rank {:?}: the TokenSpeed ZMQ backend is single-engine",
            req.data_parallel_rank
        ));
    }
    // TokenSpeed returns only the single sampled-token logprob per token
    // (`top_logprobs_num = 0` on its wire) and no prompt logprobs. The vLLM
    // sampling `logprobs` field is a count: the chat frontend maps a bare
    // `logprobs: true` to `1` and `top_logprobs = k` to `k`, so a count above 1
    // (or `-1` = "all") is a top-k request that cannot be honored — reject it
    // rather than silently return fewer. Counts of 0 or 1 are the plain
    // sampled-token logprob and are wired end-to-end. Note the flip side: at
    // this proto boundary a chat `top_logprobs: 1` is indistinguishable from a
    // bare `logprobs: true` (both arrive as count 1), so it is accepted and its
    // `top_logprobs` list simply stays empty.
    if let Some(sp) = req.sampling_params.as_ref() {
        if sp.logprobs.is_some_and(|n| !(0..=1).contains(&n)) {
            return Err(
                "top_logprobs are not supported over the TokenSpeed ZMQ backend".to_string(),
            );
        }
        if sp.prompt_logprobs.is_some() {
            return Err(
                "prompt logprobs are not supported over the TokenSpeed ZMQ backend".to_string(),
            );
        }
        // The response_format / forced-tool-choice constraint oneof is not
        // translated onto the TokenSpeed structured-output fields yet; dropping
        // it would return unconstrained text.
        if sp.constraint.is_some() {
            return Err(
                "structured output constraints are not supported over the ZMQ backend yet"
                    .to_string(),
            );
        }
        // The TokenSpeed wire has no per-sample demux; n>1 is fanned out into
        // single-sample sub-requests by `generate` before translation.
        // Stop strings are not forwarded: the direct ZMQ path sends token ids
        // only (the engine-side transport normalizes without a tokenizer) and
        // the gateway's stop decoder does not enforce them, so they would be
        // ignored and the request would run to max_tokens.
        if !sp.stop.is_empty() {
            return Err(
                "stop strings are not supported over the TokenSpeed ZMQ backend yet; \
                 use stop_token_ids"
                    .to_string(),
            );
        }
        // logit_bias is not translated onto the TokenSpeed wire either.
        if !sp.logit_bias.is_empty() {
            return Err("logit_bias is not supported over the TokenSpeed ZMQ backend".to_string());
        }
    }
    let return_logprob = req
        .sampling_params
        .as_ref()
        .is_some_and(|sp| sp.logprobs.is_some());
    Ok(TokenizedGenerateReqInput {
        rid: req.request_id,
        input_ids,
        sampling_params: req
            .sampling_params
            .map(translate_sampling_tokenspeed)
            .unwrap_or_else(|| {
                let mut params = TokenSpeedSamplingParams::default();
                params.normalize();
                params
            }),
        return_logprob,
        stream,
        // Every other field keeps its neutral default (the fields after
        // `stream` are not even emitted; the engine fills them from defaults).
        ..TokenizedGenerateReqInput::default()
    })
}

/// Map vLLM-proto sampling params onto TokenSpeed's native `SamplingParams`,
/// in the normalized form: the engine skips its decode-time re-derivation once
/// `is_normalized` is set, so [`TokenSpeedSamplingParams::normalize`] resolves
/// the derived fields (top_k sentinel, greedy collapse) before encoding.
fn translate_sampling_tokenspeed(sp: vllm::SamplingParams) -> TokenSpeedSamplingParams {
    let mut params = TokenSpeedSamplingParams {
        max_new_tokens: sp.max_tokens,
        stop_token_ids: (!sp.stop_token_ids.is_empty()).then_some(sp.stop_token_ids),
        temperature: f64::from(sp.temperature.unwrap_or(1.0)),
        top_p: f64::from(sp.top_p),
        // vLLM proto uses `0` for "all tokens"; TokenSpeed's API form is `-1`
        // (`normalize` resolves it to the engine's disabled sentinel).
        top_k: if sp.top_k == 0 {
            -1
        } else {
            i32::try_from(sp.top_k).unwrap_or(-1)
        },
        min_p: f64::from(sp.min_p),
        frequency_penalty: f64::from(sp.frequency_penalty),
        presence_penalty: f64::from(sp.presence_penalty),
        repetition_penalty: f64::from(sp.repetition_penalty),
        min_new_tokens: sp.min_tokens,
        ignore_eos: sp.ignore_eos,
        // A negative seed is a "no seed" sentinel; drop it rather than wrap
        // (the engine then derives a per-rid seed).
        seed: sp.seed.and_then(|seed| u64::try_from(seed).ok()),
        // Proto `0` means unspecified; TokenSpeed expects at least one sample.
        // The engine stores n without acting on it — n>1 is fanned out before
        // translation, so this is always 1 on the wire.
        n: sp.n.max(1),
        ..TokenSpeedSamplingParams::default()
    };
    params.normalize();
    params
}

/// Translate a vLLM-proto generate request into an `EngineCoreRequest`. ZMQ mode
/// requires pre-tokenized input (SMG tokenizes upstream).
fn translate_request(req: vllm::GenerateRequest) -> Result<EngineCoreRequest, String> {
    let prompt_token_ids = match req.input {
        Some(vllm::generate_request::Input::Tokenized(tokenized)) => Some(tokenized.input_ids),
        Some(vllm::generate_request::Input::Text(_)) => {
            return Err("ZMQ mode requires pre-tokenized input (TokenizedInput)".to_string());
        }
        None => {
            return Err("ZMQ mode requires pre-tokenized input; no input provided".to_string());
        }
    };
    let data_parallel_rank = req
        .data_parallel_rank
        .map(|rank| u32::try_from(rank).map_err(|_| format!("invalid data_parallel_rank: {rank}")))
        .transpose()?;
    if let Some(sp) = req.sampling_params.as_ref() {
        // The response_format / forced-tool-choice constraint oneof is not
        // translated onto the EngineCore wire yet; dropping it would return
        // unconstrained text.
        if sp.constraint.is_some() {
            return Err(
                "structured output constraints are not supported over the ZMQ backend yet"
                    .to_string(),
            );
        }
        // n is not carried on the EngineCore wire; n>1 is fanned out into
        // single-sample sub-requests by `generate` before translation.
        // The ZMQ renderer path has no prompt-logprob merge, so the engine's
        // prompt logprobs would be computed and then dropped.
        if sp.prompt_logprobs.is_some() {
            return Err("prompt logprobs are not supported over the ZMQ backend".to_string());
        }
    }
    Ok(EngineCoreRequest {
        request_id: req.request_id,
        prompt_token_ids,
        sampling_params: req.sampling_params.map(translate_sampling),
        arrival_time: now_secs(),
        data_parallel_rank,
        ..EngineCoreRequest::default()
    })
}

fn translate_sampling(sp: vllm::SamplingParams) -> EngineCoreSamplingParams {
    let logit_bias = if sp.logit_bias.is_empty() {
        None
    } else {
        Some(
            sp.logit_bias
                .into_iter()
                .filter_map(|(token, bias)| match u32::try_from(token) {
                    Ok(t) => Some((t, bias)),
                    Err(_) => {
                        // Don't fold negatives onto key 0 (which would silently
                        // drop all but the last); skip them with a warning.
                        tracing::warn!("dropping negative logit_bias token id {token}");
                        None
                    }
                })
                .collect::<HashMap<_, _>>(),
        )
    };
    EngineCoreSamplingParams {
        temperature: sp.temperature.unwrap_or(1.0),
        top_p: sp.top_p,
        top_k: sp.top_k,
        min_p: sp.min_p,
        frequency_penalty: sp.frequency_penalty,
        presence_penalty: sp.presence_penalty,
        repetition_penalty: sp.repetition_penalty,
        max_tokens: sp.max_tokens.unwrap_or(16),
        min_tokens: sp.min_tokens,
        stop_token_ids: sp.stop_token_ids,
        seed: sp.seed.map(i64::from),
        logprobs: sp.logprobs,
        // prompt_logprobs is rejected in `translate_request` (no renderer
        // support on the ZMQ path), so it is never forwarded.
        logit_bias,
        ..EngineCoreSamplingParams::default()
    }
}

fn map_matched_stop(reason: StopReason) -> vllm::generate_complete::MatchedStop {
    match reason {
        StopReason::TokenId(id) => vllm::generate_complete::MatchedStop::MatchedTokenId(id),
        StopReason::Text(text) => vllm::generate_complete::MatchedStop::MatchedStopStr(text),
    }
}

fn finish_reason_str(reason: EngineCoreFinishReason) -> &'static str {
    match reason {
        EngineCoreFinishReason::Stop | EngineCoreFinishReason::Repetition => "stop",
        EngineCoreFinishReason::Length => "length",
        EngineCoreFinishReason::Abort => "abort",
        EngineCoreFinishReason::Error => "error",
    }
}

/// Normalize a TokenSpeed wire finish-reason string into the canonical set the
/// gateway's response layer exact-matches (`stop`, `length`, `abort`, `error`) —
/// the same set the vLLM path emits via [`finish_reason_str`]. TokenSpeed emits
/// `stop`/`length`/`abort`; an unknown value falls back to `stop` with a warning
/// so a non-canonical string never mis-renders downstream.
fn normalize_finish_reason(reason: &str) -> &'static str {
    match reason {
        "stop" => "stop",
        "length" => "length",
        "abort" => "abort",
        "error" => "error",
        other => {
            tracing::warn!(
                finish_reason = other,
                "unknown TokenSpeed finish_reason; defaulting to \"stop\""
            );
            "stop"
        }
    }
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn zmq_status(error: engine_zmq_client::Error) -> tonic::Status {
    match error {
        engine_zmq_client::Error::EngineCoreDead => tonic::Status::unavailable(error.to_string()),
        other => tonic::Status::internal(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use engine_zmq_client::{
        mock_engine::{connect_to_frontend, default_ready_response, EngineInbound},
        protocol::vllm::{
            logprobs::{Logprobs, MaybeWireLogprobs, PositionLogprobs, TokenLogprob},
            output::{EngineCoreOutputs, RequestBatchOutputs},
        },
        EngineId,
    };

    use super::*;

    fn batch(
        request_id: &str,
        token: u32,
        logprob: Option<f32>,
        finish: Option<EngineCoreFinishReason>,
    ) -> EngineCoreOutputs {
        let finished = finish.map(|_| std::collections::BTreeSet::from([request_id.to_string()]));
        let new_logprobs = logprob.map(|lp| {
            MaybeWireLogprobs::Direct(Logprobs {
                positions: vec![PositionLogprobs {
                    entries: vec![TokenLogprob {
                        token_id: token,
                        logprob: lp,
                        rank: 1,
                    }],
                }],
            })
        });
        EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index: 0,
            outputs: vec![EngineCoreOutput {
                request_id: request_id.to_string(),
                new_token_ids: vec![token],
                new_logprobs,
                finish_reason: finish,
                ..Default::default()
            }],
            finished_requests: finished,
            ..Default::default()
        })
    }

    /// End-to-end over ipc://: the adapter translates a vLLM-proto request to
    /// EngineCore, and maps the engine's outputs back to vLLM-proto responses.
    #[tokio::test]
    async fn generate_e2e_translates_and_streams_vllm_proto() {
        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));

        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                RuntimeType::Vllm,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = client.expect("adapter connect");
        let engine = engine.expect("mock engine");

        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            let inbound = input.recv().await.unwrap();
            let request = match inbound {
                EngineInbound::Add(request) => request,
                other => panic!("expected Add, got {other:?}"),
            };
            assert_eq!(request.request_id, "r1");
            assert_eq!(request.prompt_token_ids, Some(vec![1, 2, 3]));
            assert_eq!(request.sampling_params.as_ref().unwrap().max_tokens, 2);
            output
                .send_outputs(&batch("r1", 10, Some(-0.5), None))
                .await
                .unwrap();
            output
                .send_outputs(&batch(
                    "r1",
                    11,
                    Some(-1.25),
                    Some(EngineCoreFinishReason::Length),
                ))
                .await
                .unwrap();
        });

        let req = vllm::GenerateRequest {
            request_id: "r1".to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2, 3],
                },
            )),
            sampling_params: Some(vllm::SamplingParams {
                max_tokens: Some(2),
                logprobs: Some(1),
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        };
        let mut stream = client.generate(req).await.expect("generate");

        let first = stream.next().await.expect("chunk item").expect("chunk ok");
        match first.response {
            Some(vllm::generate_response::Response::Chunk(chunk)) => {
                assert_eq!(chunk.token_ids, vec![10]);
                let logprobs = chunk.output_logprobs.expect("chunk logprobs");
                assert_eq!(logprobs.token_logprobs, vec![-0.5]);
                assert_eq!(logprobs.token_ids, vec![10]);
            }
            other => panic!("expected chunk, got {other:?}"),
        }
        // The finish tick carried a new token, so its delta is emitted as a
        // chunk before the (cumulative) terminal complete.
        let second = stream.next().await.expect("chunk item").expect("chunk ok");
        match second.response {
            Some(vllm::generate_response::Response::Chunk(chunk)) => {
                assert_eq!(chunk.token_ids, vec![11]);
                let logprobs = chunk.output_logprobs.expect("chunk logprobs");
                assert_eq!(logprobs.token_logprobs, vec![-1.25]);
                assert_eq!(logprobs.token_ids, vec![11]);
            }
            other => panic!("expected chunk, got {other:?}"),
        }
        let third = stream
            .next()
            .await
            .expect("complete item")
            .expect("complete ok");
        match third.response {
            Some(vllm::generate_response::Response::Complete(complete)) => {
                assert_eq!(complete.output_ids, vec![10, 11]);
                assert_eq!(complete.finish_reason, "length");
                assert_eq!(complete.completion_tokens, 2);
                let logprobs = complete.output_logprobs.expect("complete logprobs");
                assert_eq!(logprobs.token_logprobs, vec![-0.5, -1.25]);
                assert_eq!(logprobs.token_ids, vec![10, 11]);
            }
            other => panic!("expected complete, got {other:?}"),
        }
        assert!(stream.next().await.is_none());

        engine_task.await.unwrap();
    }

    /// End-to-end over ipc:// for a TokenSpeed backend: the adapter frames a
    /// tagged `TokenizedGenerateReqInput`, and maps `BatchTokenIDOutSlim`
    /// batches back to vLLM-proto responses. The mock engine speaks the shared
    /// transport with raw frames (it decodes/encodes the TokenSpeed structs
    /// directly).
    #[tokio::test]
    async fn generate_e2e_translates_and_streams_tokenspeed() {
        use engine_zmq_client::{
            codec::{decode_msgpack, encode_msgpack},
            protocol::tokenspeed::{
                output::BatchTokenIDOutSlim,
                request::{TokenSpeedRequestType, TokenizedGenerateReqInput},
            },
        };

        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));

        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                RuntimeType::TokenSpeed,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = client.expect("adapter connect");
        let engine = engine.expect("mock engine");

        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            let frames = input.recv_frames().await.unwrap();
            assert_eq!(
                TokenSpeedRequestType::from_frame(frames[0].as_ref()),
                Some(TokenSpeedRequestType::Add)
            );
            let request: TokenizedGenerateReqInput = decode_msgpack(frames[1].as_ref()).unwrap();
            assert_eq!(request.rid, "r1");
            assert_eq!(request.input_ids, vec![1, 2, 3]);
            assert_eq!(request.sampling_params.max_new_tokens, Some(2));
            // The adapter always emits the normalized sampling form.
            assert!(request.sampling_params.is_normalized);
            // A plain sampled-token logprob request (logprobs=1) sets the flag.
            assert!(request.return_logprob);

            let chunk = BatchTokenIDOutSlim {
                rids: vec!["r1".into()],
                output_ids: vec![vec![10]],
                finished_reasons: vec![String::new()],
                prompt_tokens: vec![3],
                completion_tokens: vec![1],
                cached_tokens: vec![0],
                output_token_logprobs_val: vec![vec![-0.5]],
                output_token_logprobs_idx: vec![vec![10]],
            };
            let done = BatchTokenIDOutSlim {
                rids: vec!["r1".into()],
                output_ids: vec![vec![11]],
                finished_reasons: vec!["length".into()],
                prompt_tokens: vec![3],
                completion_tokens: vec![2],
                cached_tokens: vec![0],
                output_token_logprobs_val: vec![vec![-1.25]],
                output_token_logprobs_idx: vec![vec![11]],
            };
            output
                .send_frames(vec![bytes::Bytes::from(encode_msgpack(&chunk).unwrap())])
                .await
                .unwrap();
            output
                .send_frames(vec![bytes::Bytes::from(encode_msgpack(&done).unwrap())])
                .await
                .unwrap();
        });

        let req = vllm::GenerateRequest {
            request_id: "r1".to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2, 3],
                },
            )),
            sampling_params: Some(vllm::SamplingParams {
                max_tokens: Some(2),
                // Plain sampled-token logprob (count 1); must be wired through.
                logprobs: Some(1),
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        };
        let mut stream = client.generate(req).await.expect("generate");

        let first = stream.next().await.expect("chunk item").expect("chunk ok");
        match first.response {
            Some(vllm::generate_response::Response::Chunk(chunk)) => {
                assert_eq!(chunk.token_ids, vec![10]);
                assert_eq!(chunk.prompt_tokens, 3);
                let logprobs = chunk.output_logprobs.expect("chunk logprobs");
                assert_eq!(logprobs.token_logprobs, vec![-0.5]);
                assert_eq!(logprobs.token_ids, vec![10]);
            }
            other => panic!("expected chunk, got {other:?}"),
        }
        // The finish tick carried a new token, so its delta is emitted as a
        // chunk before the (cumulative) terminal complete.
        let second = stream.next().await.expect("chunk item").expect("chunk ok");
        match second.response {
            Some(vllm::generate_response::Response::Chunk(chunk)) => {
                assert_eq!(chunk.token_ids, vec![11]);
                let logprobs = chunk.output_logprobs.expect("chunk logprobs");
                assert_eq!(logprobs.token_logprobs, vec![-1.25]);
                assert_eq!(logprobs.token_ids, vec![11]);
            }
            other => panic!("expected chunk, got {other:?}"),
        }
        let third = stream
            .next()
            .await
            .expect("complete item")
            .expect("complete ok");
        match third.response {
            Some(vllm::generate_response::Response::Complete(complete)) => {
                assert_eq!(complete.output_ids, vec![10, 11]);
                assert_eq!(complete.finish_reason, "length");
                assert_eq!(complete.completion_tokens, 2);
                // Chunks carry the tick's incremental logprobs; the terminal
                // `Complete` carries the cumulative set, parallel to `output_ids`
                // (the non-streaming renderer reads only the `Complete`).
                let logprobs = complete.output_logprobs.expect("complete logprobs");
                assert_eq!(logprobs.token_logprobs, vec![-0.5, -1.25]);
                assert_eq!(logprobs.token_ids, vec![10, 11]);
            }
            other => panic!("expected complete, got {other:?}"),
        }
        assert!(stream.next().await.is_none());

        engine_task.await.unwrap();
    }

    #[test]
    fn finish_reasons_map_to_vllm_strings() {
        assert_eq!(finish_reason_str(EngineCoreFinishReason::Length), "length");
        assert_eq!(
            finish_reason_str(EngineCoreFinishReason::Repetition),
            "stop"
        );
        assert_eq!(finish_reason_str(EngineCoreFinishReason::Abort), "abort");
    }

    #[test]
    fn tokenspeed_sampling_maps_top_k_sentinel_and_seed() {
        use engine_zmq_client::protocol::tokenspeed::sampling::TOP_K_DISABLED;

        // Proto top_k=0 ("all tokens") normalizes to the engine's disabled
        // sentinel; negative seed dropped; n floored to 1.
        let mapped = translate_sampling_tokenspeed(vllm::SamplingParams {
            top_k: 0,
            n: 0,
            seed: Some(-1),
            max_tokens: Some(8),
            ..Default::default()
        });
        assert_eq!(mapped.top_k, TOP_K_DISABLED);
        assert_eq!(mapped.n, 1);
        assert_eq!(mapped.seed, None);
        assert_eq!(mapped.max_new_tokens, Some(8));
        // The wire form is always normalized (the engine skips re-derivation).
        assert!(mapped.is_normalized);

        let mapped = translate_sampling_tokenspeed(vllm::SamplingParams {
            top_k: 40,
            seed: Some(7),
            ..Default::default()
        });
        assert_eq!(mapped.top_k, 40);
        assert_eq!(mapped.seed, Some(7));

        // A near-zero temperature collapses to greedy on the wire.
        let mapped = translate_sampling_tokenspeed(vllm::SamplingParams {
            temperature: Some(0.0),
            ..Default::default()
        });
        assert_eq!(mapped.temperature, 1.0);
        assert_eq!(mapped.top_k, 1);

        // Empty stop_token_ids ride as None (the normalized encoding).
        let mapped = translate_sampling_tokenspeed(vllm::SamplingParams::default());
        assert_eq!(mapped.stop_token_ids, None);
    }

    fn tokenized_req(sampling: vllm::SamplingParams) -> vllm::GenerateRequest {
        vllm::GenerateRequest {
            request_id: "r1".to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2, 3],
                },
            )),
            sampling_params: Some(sampling),
            stream: true,
            ..Default::default()
        }
    }

    #[test]
    fn tokenspeed_plain_logprobs_set_return_logprob() {
        // Counts 0 and 1 are the plain sampled-token case; both accepted.
        for count in [0, 1] {
            let req = translate_request_tokenspeed(tokenized_req(vllm::SamplingParams {
                logprobs: Some(count),
                ..Default::default()
            }))
            .expect("plain logprobs accepted");
            assert!(req.return_logprob);
        }

        // No logprobs -> the flag stays false.
        let req = translate_request_tokenspeed(tokenized_req(vllm::SamplingParams::default()))
            .expect("no logprobs accepted");
        assert!(!req.return_logprob);
    }

    #[test]
    fn tokenspeed_rejects_top_k_and_prompt_logprobs() {
        // Top-k (count > 1) cannot be honored.
        assert!(
            translate_request_tokenspeed(tokenized_req(vllm::SamplingParams {
                logprobs: Some(5),
                ..Default::default()
            }))
            .is_err()
        );
        // "all" (count -1) cannot be honored.
        assert!(
            translate_request_tokenspeed(tokenized_req(vllm::SamplingParams {
                logprobs: Some(-1),
                ..Default::default()
            }))
            .is_err()
        );
        // Prompt logprobs cannot be produced.
        assert!(
            translate_request_tokenspeed(tokenized_req(vllm::SamplingParams {
                prompt_logprobs: Some(1),
                ..Default::default()
            }))
            .is_err()
        );
    }

    #[test]
    fn tokenspeed_rejects_unsupported_sampling_features() {
        // Structured-output constraints have no TokenSpeed wire slot.
        let err = translate_request_tokenspeed(tokenized_req(vllm::SamplingParams {
            constraint: Some(vllm::sampling_params::Constraint::JsonObject(true)),
            ..Default::default()
        }))
        .expect_err("constraint rejected");
        assert!(err.contains("structured output"), "{err}");

        // Stop strings have no wire slot and are not enforced by the gateway.
        let err = translate_request_tokenspeed(tokenized_req(vllm::SamplingParams {
            stop: vec!["</s>".to_string()],
            ..Default::default()
        }))
        .expect_err("stop strings rejected");
        assert!(err.contains("stop_token_ids"), "{err}");

        // logit_bias has no wire slot.
        let err = translate_request_tokenspeed(tokenized_req(vllm::SamplingParams {
            logit_bias: HashMap::from([(7, 1.0)]),
            ..Default::default()
        }))
        .expect_err("logit_bias rejected");
        assert!(err.contains("logit_bias"), "{err}");
    }

    #[test]
    fn tokenspeed_rejects_nonzero_dp_rank() {
        // Single-engine backend: only rank 0 (or none) is valid.
        let mut req = tokenized_req(vllm::SamplingParams::default());
        req.data_parallel_rank = Some(1);
        assert!(translate_request_tokenspeed(req).is_err());

        let mut req = tokenized_req(vllm::SamplingParams::default());
        req.data_parallel_rank = Some(0);
        assert!(translate_request_tokenspeed(req).is_ok());
    }

    #[test]
    fn vllm_rejects_unsupported_sampling_features() {
        // Structured-output constraints are not translated onto the wire.
        let err = translate_request(tokenized_req(vllm::SamplingParams {
            constraint: Some(vllm::sampling_params::Constraint::JsonObject(true)),
            ..Default::default()
        }))
        .expect_err("constraint rejected");
        assert!(err.contains("structured output"), "{err}");

        // Prompt logprobs have no renderer merge on the ZMQ path.
        let err = translate_request(tokenized_req(vllm::SamplingParams {
            prompt_logprobs: Some(1),
            ..Default::default()
        }))
        .expect_err("prompt logprobs rejected");
        assert!(err.contains("prompt logprobs"), "{err}");
    }

    /// n=3 fans out into 3 single-sample wire requests with unique sub-rids.
    /// An explicit seed derives per-sub seeds (`seed + i`) so the samples
    /// differ deterministically; no seed stays `None` per sub (the engine
    /// seeds each rid independently).
    #[test]
    fn fan_out_splits_n_into_single_sample_requests() {
        let mut req = tokenized_req(vllm::SamplingParams {
            n: 3,
            seed: Some(7),
            ..Default::default()
        });
        req.request_id = "r1".to_string();

        let subs = fan_out_requests(req);
        assert_eq!(subs.len(), 3);
        let rids: Vec<&str> = subs.iter().map(|sub| sub.request_id.as_str()).collect();
        assert_eq!(rids, vec!["r1-0", "r1-1", "r1-2"]);
        for (i, sub) in subs.iter().enumerate() {
            let sp = sub.sampling_params.as_ref().unwrap();
            assert_eq!(sp.n, 1);
            assert_eq!(sp.seed, Some(7 + i as i32));
            // Everything else is shared verbatim.
            assert_eq!(
                sub.input,
                Some(vllm::generate_request::Input::Tokenized(
                    vllm::TokenizedInput {
                        original_text: String::new(),
                        input_ids: vec![1, 2, 3],
                    }
                ))
            );
        }

        // No explicit seed: every sub keeps None.
        let subs = fan_out_requests(tokenized_req(vllm::SamplingParams {
            n: 2,
            ..Default::default()
        }));
        assert!(subs
            .iter()
            .all(|sub| sub.sampling_params.as_ref().unwrap().seed.is_none()));

        // n<=1 passes through untouched (rid keeps its original form).
        let single = fan_out_requests(tokenized_req(vllm::SamplingParams::default()));
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].request_id, "r1");
    }

    /// n=2 over a vLLM EngineCore: two engine-side requests with distinct
    /// sub-rids, and the merged stream yields two `Complete`s tagged with the
    /// proto `index` (0 and 1) the pipeline demuxes choices by.
    #[tokio::test]
    async fn generate_e2e_fans_out_n2_vllm() {
        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));

        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                RuntimeType::Vllm,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = client.expect("adapter connect");
        let engine = engine.expect("mock engine");

        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            let mut rids = Vec::new();
            for _ in 0..2 {
                let request = match input.recv().await.unwrap() {
                    EngineInbound::Add(request) => request,
                    other => panic!("expected Add, got {other:?}"),
                };
                // Each sub is a single-sample request with a derived seed.
                let sp = request.sampling_params.as_ref().unwrap();
                assert_eq!(sp.max_tokens, 4);
                rids.push((request.request_id.clone(), sp.seed));
            }
            assert_eq!(
                rids,
                vec![("r1-0".to_string(), Some(5)), ("r1-1".to_string(), Some(6))],
                "sub-rids must be unique and seeds derived per sub"
            );
            output
                .send_outputs(&batch("r1-0", 10, None, Some(EngineCoreFinishReason::Stop)))
                .await
                .unwrap();
            output
                .send_outputs(&batch("r1-1", 11, None, Some(EngineCoreFinishReason::Stop)))
                .await
                .unwrap();
        });

        let mut req = tokenized_req(vllm::SamplingParams {
            n: 2,
            seed: Some(5),
            max_tokens: Some(4),
            ..Default::default()
        });
        req.request_id = "r1".to_string();
        let mut stream = client.generate(req).await.expect("generate");

        let mut completes = Vec::new();
        while let Some(item) = stream.next().await {
            match item.expect("stream item").response {
                Some(vllm::generate_response::Response::Complete(complete)) => {
                    completes.push(complete);
                }
                Some(vllm::generate_response::Response::Chunk(_)) | None => {}
            }
        }
        completes.sort_by_key(|complete| complete.index);
        assert_eq!(completes.len(), 2, "one Complete per fanned-out sub");
        assert_eq!(completes[0].index, 0);
        assert_eq!(completes[0].output_ids, vec![10]);
        assert_eq!(completes[1].index, 1);
        assert_eq!(completes[1].output_ids, vec![11]);

        engine_task.await.unwrap();
    }

    /// n=2 over TokenSpeed: two engine-side requests with distinct sub-rids
    /// (delivered in one wire batch), two indexed `Complete`s, and the shared
    /// prompt reported in full on each (the pipeline de-duplicates via max, so
    /// prompt tokens are not counted n times).
    #[tokio::test]
    async fn generate_e2e_fans_out_n2_tokenspeed() {
        use engine_zmq_client::{
            codec::{decode_msgpack, encode_msgpack},
            protocol::tokenspeed::{
                output::BatchTokenIDOutSlim,
                request::{TokenSpeedRequestType, TokenizedGenerateReqInput},
            },
        };

        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));

        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                RuntimeType::TokenSpeed,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = client.expect("adapter connect");
        let engine = engine.expect("mock engine");

        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            let mut rids = Vec::new();
            for _ in 0..2 {
                let frames = input.recv_frames().await.unwrap();
                assert_eq!(
                    TokenSpeedRequestType::from_frame(frames[0].as_ref()),
                    Some(TokenSpeedRequestType::Add)
                );
                let request: TokenizedGenerateReqInput =
                    decode_msgpack(frames[1].as_ref()).unwrap();
                assert_eq!(request.sampling_params.n, 1);
                rids.push((request.rid.clone(), request.sampling_params.seed));
            }
            assert_eq!(
                rids,
                vec![("r1-0".to_string(), Some(5)), ("r1-1".to_string(), Some(6))],
                "sub-rids must be unique and seeds derived per sub"
            );
            // Both subs finish in one wire batch (the batch demux fans them
            // back out to their sub-streams).
            let done = BatchTokenIDOutSlim {
                rids: vec!["r1-0".into(), "r1-1".into()],
                output_ids: vec![vec![10], vec![11]],
                finished_reasons: vec!["stop".into(), "stop".into()],
                prompt_tokens: vec![3, 3],
                completion_tokens: vec![1, 1],
                cached_tokens: vec![0, 0],
                output_token_logprobs_val: vec![vec![], vec![]],
                output_token_logprobs_idx: vec![vec![], vec![]],
            };
            output
                .send_frames(vec![bytes::Bytes::from(encode_msgpack(&done).unwrap())])
                .await
                .unwrap();
        });

        let mut req = tokenized_req(vllm::SamplingParams {
            n: 2,
            seed: Some(5),
            ..Default::default()
        });
        req.request_id = "r1".to_string();
        let mut stream = client.generate(req).await.expect("generate");

        let mut completes = Vec::new();
        while let Some(item) = stream.next().await {
            match item.expect("stream item").response {
                Some(vllm::generate_response::Response::Complete(complete)) => {
                    completes.push(complete);
                }
                Some(vllm::generate_response::Response::Chunk(_)) | None => {}
            }
        }
        completes.sort_by_key(|complete| complete.index);
        assert_eq!(completes.len(), 2, "one Complete per fanned-out sub");
        assert_eq!(completes[0].index, 0);
        assert_eq!(completes[0].output_ids, vec![10]);
        assert_eq!(completes[1].index, 1);
        assert_eq!(completes[1].output_ids, vec![11]);
        // Each sub reports the full shared prompt; the pipeline maxes, so
        // usage is not double-counted.
        assert!(completes.iter().all(|complete| complete.prompt_tokens == 3));

        engine_task.await.unwrap();
    }

    /// Dropping the merged stream before completion aborts EVERY fanned-out
    /// engine-side sub-request, not just one.
    #[tokio::test]
    async fn dropping_fanned_out_stream_aborts_all_subs() {
        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));

        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                RuntimeType::Vllm,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = client.expect("adapter connect");
        let engine = engine.expect("mock engine");
        let (mut engine_input, _engine_output) = engine.split();

        let stream = client
            .generate(tokenized_req(vllm::SamplingParams {
                n: 2,
                ..Default::default()
            }))
            .await
            .expect("generate");

        // Consume both Adds first so the drop-triggered aborts are the next
        // inbound messages.
        for _ in 0..2 {
            match engine_input.recv().await.unwrap() {
                EngineInbound::Add(_) => {}
                other => panic!("expected Add, got {other:?}"),
            }
        }

        drop(stream); // unfinished -> every sub auto-aborts

        let mut aborted = std::collections::BTreeSet::new();
        while aborted.len() < 2 {
            match engine_input.recv().await.unwrap() {
                EngineInbound::Abort(rids) => aborted.extend(rids),
                other => panic!("expected Abort, got {other:?}"),
            }
        }
        assert_eq!(
            aborted,
            std::collections::BTreeSet::from(["r1-0".to_string(), "r1-1".to_string()])
        );
    }

    #[test]
    fn tokenspeed_finish_reason_normalizes() {
        assert_eq!(normalize_finish_reason("stop"), "stop");
        assert_eq!(normalize_finish_reason("length"), "length");
        assert_eq!(normalize_finish_reason("abort"), "abort");
        assert_eq!(normalize_finish_reason("error"), "error");
        // An unknown wire value falls back to "stop".
        assert_eq!(normalize_finish_reason("garbage"), "stop");
    }
}
