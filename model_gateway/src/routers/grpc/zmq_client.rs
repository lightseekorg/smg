// SPDX-License-Identifier: Apache-2.0
//
// ZMQ backend adapter (gateway glue): presents the vLLM engine surface (the same
// proto request/response types as `VllmEngineClient`) but speaks ZMQ directly to
// a same-host vLLM EngineCore via `engine-zmq-client`, bypassing the gRPC Python
// servicer.
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
    connector::{EngineCoreClient, EngineCoreStream},
    protocol::vllm::{
        output::{EngineCoreFinishReason, EngineCoreOutput, StopReason},
        request::EngineCoreRequest,
        sampling::EngineCoreSamplingParams,
    },
};
use futures::StreamExt;
use openai_protocol::worker::{SchedulerLoadSnapshot, WorkerLoadResponse};
use smg_grpc_client::vllm_proto as vllm;

/// Direct ZMQ connection to a same-host vLLM EngineCore, presented behind the
/// vLLM gRPC client surface.
#[derive(Clone)]
pub struct ZmqEngineClient {
    inner: Arc<EngineCoreClient>,
    /// Model id advertised for metadata (EngineCore does not report it on the
    /// wire; it is configured at worker registration).
    model_id: String,
}

impl ZmqEngineClient {
    /// Bind the frontend sockets and complete the handshake with the engine(s),
    /// which must already be running and dialing `handshake_address`.
    ///
    /// `input_address`/`output_address` are the `ipc://` data-plane endpoints the
    /// engines connect to (chosen by SMG). `engine_count` is the number of DP
    /// ranks to await.
    pub async fn connect(
        handshake_address: &str,
        input_address: &str,
        output_address: &str,
        engine_count: usize,
        model_id: String,
        timeout: Duration,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let transport = connect_handshake(
            handshake_address,
            engine_count,
            "127.0.0.1",
            Some(input_address),
            Some(output_address),
            timeout,
        )
        .await?;
        Ok(Self {
            inner: Arc::new(EngineCoreClient::new(transport)),
            model_id,
        })
    }

    /// Submit a generate request and return a stream of vLLM-proto responses.
    pub async fn generate(
        &self,
        req: vllm::GenerateRequest,
    ) -> Result<ZmqGenerateStream, tonic::Status> {
        let core_request = translate_request(req).map_err(tonic::Status::invalid_argument)?;
        let stream = self.inner.submit(core_request).await.map_err(zmq_status)?;
        Ok(ZmqGenerateStream::new(stream))
    }

    /// Local liveness: false once the connection observed `ENGINE_CORE_DEAD` or
    /// a transport failure. No RPC (the raw ZMQ wire has no health RPC).
    pub fn is_alive(&self) -> bool {
        self.inner.is_alive()
    }

    /// Health as an RPC-shaped response, derived from local liveness.
    pub fn health_check(&self) -> vllm::HealthCheckResponse {
        let alive = self.inner.is_alive();
        vllm::HealthCheckResponse {
            healthy: alive,
            message: if alive {
                "ok".to_string()
            } else {
                "engine core dead".to_string()
            },
        }
    }

    /// Per-rank load from the piggybacked `scheduler_stats` (SMG's DP routing
    /// signal), in the same shape as the gRPC `GetLoads` response.
    pub fn get_loads(&self) -> WorkerLoadResponse {
        let loads: Vec<SchedulerLoadSnapshot> = self
            .inner
            .engines()
            .iter()
            .filter_map(|engine| {
                let dp_rank = engine.engine_id.engine_index()?;
                let stats = self.inner.scheduler_stats(dp_rank)?;
                Some(SchedulerLoadSnapshot {
                    dp_rank: i32::try_from(dp_rank).unwrap_or(i32::MAX),
                    num_running_reqs: i32::try_from(stats.num_running_reqs).unwrap_or(i32::MAX),
                    num_waiting_reqs: i32::try_from(stats.num_waiting_reqs).unwrap_or(i32::MAX),
                    token_usage: stats.kv_cache_usage,
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
    /// configured model id (EngineCore does not report tokenizer/vocab metadata,
    /// so those come from worker config).
    pub fn get_model_info(&self) -> vllm::GetModelInfoResponse {
        let max_context_length = self
            .inner
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
            .inner
            .engines()
            .first()
            .map(|e| e.ready_response.data_parallel_size)
            .unwrap_or(1);
        vllm::GetServerInfoResponse {
            data_parallel_size: i32::try_from(data_parallel_size).unwrap_or(i32::MAX),
            server_type: "vllm".to_string(),
            ..Default::default()
        }
    }
}

/// Streaming generate output, mapping each `EngineCoreOutput` to a vLLM-proto
/// `GenerateResponse` (chunks until the terminal output, then a complete). The
/// underlying [`EngineCoreStream`] auto-aborts on drop, so no explicit abort or
/// `mark_completed` is required.
pub struct ZmqGenerateStream {
    inner: EngineCoreStream,
    output_ids: Vec<u32>,
    completion_tokens: u32,
    prompt_tokens: u32,
    cached_tokens: u32,
}

impl ZmqGenerateStream {
    fn new(inner: EngineCoreStream) -> Self {
        Self {
            inner,
            output_ids: Vec::new(),
            completion_tokens: 0,
            prompt_tokens: 0,
            cached_tokens: 0,
        }
    }

    /// Next vLLM-proto response, or `None` when the stream ends.
    pub async fn next(&mut self) -> Option<Result<vllm::GenerateResponse, tonic::Status>> {
        match self.inner.next().await {
            Some(Ok(output)) => Some(Ok(self.map_output(output))),
            Some(Err(error)) => Some(Err(zmq_status(error))),
            None => None,
        }
    }

    /// No-op: the ZMQ stream aborts natively on drop, so there is nothing to
    /// mark. Present for parity with the tonic abort-on-drop streams.
    #[expect(
        clippy::unused_self,
        reason = "receiver kept for API parity with the tonic streams"
    )]
    pub fn mark_completed(&mut self) {}

    fn map_output(&mut self, output: EngineCoreOutput) -> vllm::GenerateResponse {
        if let Some(stats) = &output.prefill_stats {
            self.prompt_tokens = stats.num_prompt_tokens;
            self.cached_tokens = stats.num_cached_tokens;
        }
        self.completion_tokens += output.new_token_ids.len() as u32;
        self.output_ids.extend(output.new_token_ids.iter().copied());

        let response = match output.finish_reason {
            Some(reason) => vllm::generate_response::Response::Complete(vllm::GenerateComplete {
                output_ids: std::mem::take(&mut self.output_ids),
                finish_reason: finish_reason_str(reason).to_string(),
                prompt_tokens: self.prompt_tokens,
                completion_tokens: self.completion_tokens,
                cached_tokens: self.cached_tokens,
                matched_stop: output.stop_reason.map(map_matched_stop),
                ..Default::default()
            }),
            None => vllm::generate_response::Response::Chunk(vllm::GenerateStreamChunk {
                token_ids: output.new_token_ids,
                prompt_tokens: self.prompt_tokens,
                completion_tokens: self.completion_tokens,
                cached_tokens: self.cached_tokens,
                ..Default::default()
            }),
        };
        vllm::GenerateResponse {
            response: Some(response),
        }
    }
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
        prompt_logprobs: sp.prompt_logprobs,
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
        protocol::vllm::output::{EngineCoreOutputs, RequestBatchOutputs},
        EngineId,
    };

    use super::*;

    fn batch(
        request_id: &str,
        token: u32,
        finish: Option<EngineCoreFinishReason>,
    ) -> EngineCoreOutputs {
        let finished = finish.map(|_| std::collections::BTreeSet::from([request_id.to_string()]));
        EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index: 0,
            outputs: vec![EngineCoreOutput {
                request_id: request_id.to_string(),
                new_token_ids: vec![token],
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
            output.send_outputs(&batch("r1", 10, None)).await.unwrap();
            output
                .send_outputs(&batch("r1", 11, Some(EngineCoreFinishReason::Length)))
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
            }
            other => panic!("expected chunk, got {other:?}"),
        }
        let second = stream
            .next()
            .await
            .expect("complete item")
            .expect("complete ok");
        match second.response {
            Some(vllm::generate_response::Response::Complete(complete)) => {
                assert_eq!(complete.output_ids, vec![10, 11]);
                assert_eq!(complete.finish_reason, "length");
                assert_eq!(complete.completion_tokens, 2);
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
}
