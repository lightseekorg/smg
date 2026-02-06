//! Protocol buffer type wrappers for SGLang, vLLM, and TensorRT-LLM backends
//!
//! This module provides unified enums that wrap proto types from SGLang, vLLM, and TensorRT-LLM,
//! allowing the router to work with any backend transparently.

use futures_util::StreamExt;

use crate::grpc_client::{
    sglang_proto::{self as sglang, generate_complete::MatchedStop},
    sglang_scheduler::AbortOnDropStream as SglangStream,
    trtllm_proto as trtllm,
    trtllm_service::AbortOnDropStream as TrtllmStream,
    vllm_engine::AbortOnDropStream as VllmStream,
    vllm_proto as vllm,
};

// =====================
// Unified Logprobs Types
// =====================

/// Unified output logprobs (backend-agnostic)
#[derive(Clone, Debug)]
pub struct ProtoOutputLogProbs {
    pub token_logprobs: Vec<f32>,
    pub token_ids: Vec<i32>,
    pub top_logprobs: Vec<ProtoTopLogProbs>,
}

/// Unified top logprobs per position
#[derive(Clone, Debug)]
pub struct ProtoTopLogProbs {
    pub values: Vec<f32>,
    pub token_ids: Vec<i32>,
}

/// Unified input (prompt) logprobs
#[derive(Clone, Debug)]
pub struct ProtoInputLogProbs {
    pub token_logprobs: Vec<Option<f32>>, // First token is None
    pub token_ids: Vec<i32>,
    pub top_logprobs: Vec<ProtoTopLogProbs>,
}

/// Helper macro to convert output logprobs from proto types to unified type.
/// Both SGLang and vLLM have identical OutputLogProbs structure.
/// Note: Cloning is necessary as we convert from borrowed proto types to owned unified types.
/// OOM risk is mitigated by capping top_logprobs at 20 in sampling params.
macro_rules! convert_output_logprobs {
    ($lp:expr) => {
        ProtoOutputLogProbs {
            token_logprobs: $lp.token_logprobs.clone(),
            token_ids: $lp.token_ids.clone(),
            top_logprobs: $lp
                .top_logprobs
                .iter()
                .map(|t| ProtoTopLogProbs {
                    values: t.values.clone(),
                    token_ids: t.token_ids.clone(),
                })
                .collect(),
        }
    };
}

/// Helper macro to convert input logprobs from proto types to unified type.
macro_rules! convert_input_logprobs {
    ($lp:expr) => {
        ProtoInputLogProbs {
            token_logprobs: $lp.token_logprobs.iter().map(|t| t.value).collect(),
            token_ids: $lp.token_ids.clone(),
            top_logprobs: $lp
                .top_logprobs
                .iter()
                .map(|t| ProtoTopLogProbs {
                    values: t.values.clone(),
                    token_ids: t.token_ids.clone(),
                })
                .collect(),
        }
    };
}

/// Unified ProtoRequest
#[derive(Clone)]
pub enum ProtoRequest {
    Generate(ProtoGenerateRequest),
    Embed(ProtoEmbedRequest),
}

impl ProtoRequest {
    /// Get request ID from either variant
    pub fn request_id(&self) -> &str {
        match self {
            Self::Generate(req) => req.request_id(),
            Self::Embed(req) => req.request_id(),
        }
    }
}

/// Unified GenerateRequest that works with all backends
#[derive(Clone)]
pub enum ProtoGenerateRequest {
    Sglang(Box<sglang::GenerateRequest>),
    Vllm(Box<vllm::GenerateRequest>),
    Trtllm(Box<trtllm::GenerateRequest>),
}

impl ProtoGenerateRequest {
    /// Get SGLang variant (panics if not SGLang)
    pub fn as_sglang(&self) -> &sglang::GenerateRequest {
        match self {
            Self::Sglang(req) => req,
            _ => panic!("Expected SGLang GenerateRequest"),
        }
    }

    /// Get mutable SGLang variant (panics if not SGLang)
    pub fn as_sglang_mut(&mut self) -> &mut sglang::GenerateRequest {
        match self {
            Self::Sglang(req) => req,
            _ => panic!("Expected SGLang GenerateRequest"),
        }
    }

    /// Get vLLM variant (panics if not vLLM)
    pub fn as_vllm(&self) -> &vllm::GenerateRequest {
        match self {
            Self::Vllm(req) => req,
            _ => panic!("Expected vLLM GenerateRequest"),
        }
    }

    /// Get mutable vLLM variant (panics if not vLLM)
    pub fn as_vllm_mut(&mut self) -> &mut vllm::GenerateRequest {
        match self {
            Self::Vllm(req) => req,
            _ => panic!("Expected vLLM GenerateRequest"),
        }
    }

    /// Get TensorRT-LLM variant (panics if not TensorRT-LLM)
    pub fn as_trtllm(&self) -> &trtllm::GenerateRequest {
        match self {
            Self::Trtllm(req) => req,
            _ => panic!("Expected TensorRT-LLM GenerateRequest"),
        }
    }

    /// Get mutable TensorRT-LLM variant (panics if not TensorRT-LLM)
    pub fn as_trtllm_mut(&mut self) -> &mut trtllm::GenerateRequest {
        match self {
            Self::Trtllm(req) => req,
            _ => panic!("Expected TensorRT-LLM GenerateRequest"),
        }
    }

    /// Check if this is SGLang
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Check if this is vLLM
    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    /// Check if this is TensorRT-LLM
    pub fn is_trtllm(&self) -> bool {
        matches!(self, Self::Trtllm(_))
    }

    /// Set max_tokens for prefill-only execution (vLLM PD mode).
    /// The prefill request uses max_tokens=1 to trigger KV cache computation
    /// without generating unnecessary tokens.
    pub fn set_max_tokens_for_prefill(&mut self, max_tokens: u32) {
        match self {
            Self::Vllm(req) => {
                if let Some(ref mut params) = req.sampling_params {
                    params.max_tokens = Some(max_tokens);
                } else {
                    req.sampling_params = Some(vllm::SamplingParams {
                        max_tokens: Some(max_tokens),
                        ..Default::default()
                    });
                }
            }
            _ => {
                tracing::warn!("set_max_tokens_for_prefill called on non-vLLM request, ignoring");
            }
        }
    }

    /// Set stream mode on the request.
    pub fn set_stream(&mut self, stream: bool) {
        match self {
            Self::Vllm(req) => req.stream = stream,
            Self::Sglang(req) => req.stream = stream,
            Self::Trtllm(req) => req.streaming = stream,
        }
    }

    /// Clone the inner request (for passing to generate())
    pub fn clone_inner(&self) -> Self {
        self.clone()
    }

    /// Get request ID
    pub fn request_id(&self) -> &str {
        match self {
            Self::Sglang(req) => &req.request_id,
            Self::Vllm(req) => &req.request_id,
            Self::Trtllm(req) => &req.request_id,
        }
    }

    /// Set KV transfer parameters for Mooncake PD disaggregation (vLLM only).
    /// These parameters tell the decode worker where to fetch KV cache from the prefill worker.
    pub fn set_kv_transfer_params(&mut self, remote_host: String, remote_port: u32) {
        match self {
            Self::Vllm(req) => {
                req.kv_transfer_params = Some(vllm::KvTransferParams {
                    remote_host,
                    remote_port,
                });
            }
            _ => {
                tracing::warn!("set_kv_transfer_params called on non-vLLM request, ignoring");
            }
        }
    }
}

/// Unified GenerateResponse from stream
pub enum ProtoGenerateResponse {
    Sglang(Box<sglang::GenerateResponse>),
    Vllm(Box<vllm::GenerateResponse>),
    Trtllm(Box<trtllm::GenerateResponse>),
}

impl ProtoGenerateResponse {
    /// Get the response variant (chunk, complete, or error)
    ///
    /// Consumes self to avoid cloning large proto messages in hot streaming path
    pub fn into_response(self) -> ProtoResponseVariant {
        match self {
            Self::Sglang(resp) => match resp.response {
                Some(sglang::generate_response::Response::Chunk(chunk)) => {
                    ProtoResponseVariant::Chunk(ProtoGenerateStreamChunk::Sglang(chunk))
                }
                Some(sglang::generate_response::Response::Complete(complete)) => {
                    ProtoResponseVariant::Complete(ProtoGenerateComplete::Sglang(complete))
                }
                Some(sglang::generate_response::Response::Error(error)) => {
                    ProtoResponseVariant::Error(ProtoGenerateError::Sglang(error))
                }
                None => ProtoResponseVariant::None,
            },
            Self::Vllm(resp) => match resp.response {
                Some(vllm::generate_response::Response::Chunk(chunk)) => {
                    ProtoResponseVariant::Chunk(ProtoGenerateStreamChunk::Vllm(chunk))
                }
                Some(vllm::generate_response::Response::Complete(complete)) => {
                    ProtoResponseVariant::Complete(ProtoGenerateComplete::Vllm(complete))
                }
                // Note: vLLM proto no longer has Error variant in GenerateResponse
                None => ProtoResponseVariant::None,
            },
            Self::Trtllm(resp) => match resp.response {
                Some(trtllm::generate_response::Response::Chunk(chunk)) => {
                    ProtoResponseVariant::Chunk(ProtoGenerateStreamChunk::Trtllm(chunk))
                }
                Some(trtllm::generate_response::Response::Complete(complete)) => {
                    ProtoResponseVariant::Complete(ProtoGenerateComplete::Trtllm(complete))
                }
                Some(trtllm::generate_response::Response::Error(error)) => {
                    ProtoResponseVariant::Error(ProtoGenerateError::Trtllm(error))
                }
                None => ProtoResponseVariant::None,
            },
        }
    }
}

/// Response variant extracted from GenerateResponse
pub enum ProtoResponseVariant {
    Chunk(ProtoGenerateStreamChunk),
    Complete(ProtoGenerateComplete),
    Error(ProtoGenerateError),
    None,
}

/// Unified GenerateStreamChunk
#[derive(Clone)]
pub enum ProtoGenerateStreamChunk {
    Sglang(sglang::GenerateStreamChunk),
    Vllm(vllm::GenerateStreamChunk),
    Trtllm(trtllm::GenerateStreamChunk),
}

impl ProtoGenerateStreamChunk {
    /// Get SGLang variant (panics if not SGLang)
    pub fn as_sglang(&self) -> &sglang::GenerateStreamChunk {
        match self {
            Self::Sglang(chunk) => chunk,
            _ => panic!("Expected SGLang GenerateStreamChunk"),
        }
    }

    /// Get vLLM variant (panics if not vLLM)
    pub fn as_vllm(&self) -> &vllm::GenerateStreamChunk {
        match self {
            Self::Vllm(chunk) => chunk,
            _ => panic!("Expected vLLM GenerateStreamChunk"),
        }
    }

    /// Get TensorRT-LLM variant (panics if not TensorRT-LLM)
    pub fn as_trtllm(&self) -> &trtllm::GenerateStreamChunk {
        match self {
            Self::Trtllm(chunk) => chunk,
            _ => panic!("Expected TensorRT-LLM GenerateStreamChunk"),
        }
    }

    /// Check if this is SGLang
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Check if this is vLLM
    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    /// Check if this is TensorRT-LLM
    pub fn is_trtllm(&self) -> bool {
        matches!(self, Self::Trtllm(_))
    }

    /// Get token IDs from chunk (common field)
    pub fn token_ids(&self) -> &[u32] {
        match self {
            Self::Sglang(c) => &c.token_ids,
            Self::Vllm(c) => &c.token_ids,
            Self::Trtllm(c) => &c.token_ids,
        }
    }

    /// Get index (for n>1 support)
    /// Returns the index of this output when n>1 was requested (0-indexed)
    pub fn index(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.index,
            Self::Vllm(c) => c.index,
            Self::Trtllm(c) => c.sequence_index,
        }
    }

    /// Get output logprobs (SGLang, vLLM, and TensorRT-LLM)
    pub fn output_logprobs(&self) -> Option<ProtoOutputLogProbs> {
        match self {
            Self::Sglang(c) => c
                .output_logprobs
                .as_ref()
                .map(|lp| convert_output_logprobs!(lp)),
            Self::Vllm(c) => c
                .output_logprobs
                .as_ref()
                .map(|lp| convert_output_logprobs!(lp)),
            Self::Trtllm(c) => {
                if c.logprobs.is_empty() {
                    None
                } else {
                    Some(ProtoOutputLogProbs {
                        token_logprobs: c.logprobs.iter().map(|lp| lp.logprob).collect(),
                        token_ids: c.logprobs.iter().map(|lp| lp.token_id as i32).collect(),
                        top_logprobs: c
                            .logprobs
                            .iter()
                            .map(|lp| ProtoTopLogProbs {
                                values: lp.top_logprobs.iter().map(|t| t.logprob).collect(),
                                token_ids: lp
                                    .top_logprobs
                                    .iter()
                                    .map(|t| t.token_id as i32)
                                    .collect(),
                            })
                            .collect(),
                    })
                }
            }
        }
    }

    /// Get input logprobs (SGLang and vLLM only - streaming chunks don't have prompt logprobs)
    pub fn input_logprobs(&self) -> Option<ProtoInputLogProbs> {
        match self {
            Self::Sglang(c) => c
                .input_logprobs
                .as_ref()
                .map(|lp| convert_input_logprobs!(lp)),
            Self::Vllm(c) => c
                .input_logprobs
                .as_ref()
                .map(|lp| convert_input_logprobs!(lp)),
            // TRT-LLM streaming chunks don't have prompt_logprobs
            Self::Trtllm(_) => None,
        }
    }

    /// Get prompt tokens (cumulative)
    pub fn prompt_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.prompt_tokens,
            Self::Vllm(c) => c.prompt_tokens,
            Self::Trtllm(c) => c.prompt_tokens,
        }
    }

    /// Get completion tokens (cumulative)
    pub fn completion_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.completion_tokens,
            Self::Vllm(c) => c.completion_tokens,
            Self::Trtllm(c) => c.completion_tokens,
        }
    }

    /// Get cached tokens (cumulative)
    pub fn cached_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.cached_tokens,
            Self::Vllm(c) => c.cached_tokens,
            Self::Trtllm(c) => c.cached_tokens,
        }
    }
}

/// Unified GenerateComplete response
#[derive(Clone)]
pub enum ProtoGenerateComplete {
    Sglang(sglang::GenerateComplete),
    Vllm(vllm::GenerateComplete),
    Trtllm(trtllm::GenerateComplete),
}

impl ProtoGenerateComplete {
    /// Get SGLang variant (panics if not SGLang)
    pub fn as_sglang(&self) -> &sglang::GenerateComplete {
        match self {
            Self::Sglang(complete) => complete,
            _ => panic!("Expected SGLang GenerateComplete"),
        }
    }

    /// Get mutable SGLang variant (panics if not SGLang)
    pub fn as_sglang_mut(&mut self) -> &mut sglang::GenerateComplete {
        match self {
            Self::Sglang(complete) => complete,
            _ => panic!("Expected SGLang GenerateComplete"),
        }
    }

    /// Get vLLM variant (panics if not vLLM)
    pub fn as_vllm(&self) -> &vllm::GenerateComplete {
        match self {
            Self::Vllm(complete) => complete,
            _ => panic!("Expected vLLM GenerateComplete"),
        }
    }

    /// Get TensorRT-LLM variant (panics if not TensorRT-LLM)
    pub fn as_trtllm(&self) -> &trtllm::GenerateComplete {
        match self {
            Self::Trtllm(complete) => complete,
            _ => panic!("Expected TensorRT-LLM GenerateComplete"),
        }
    }

    /// Check if this is SGLang
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Check if this is vLLM
    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    /// Check if this is TensorRT-LLM
    pub fn is_trtllm(&self) -> bool {
        matches!(self, Self::Trtllm(_))
    }

    /// Get token IDs from either backend (output_ids in proto)
    pub fn token_ids(&self) -> &[u32] {
        match self {
            Self::Sglang(c) => &c.output_ids,
            Self::Vllm(c) => &c.output_ids,
            Self::Trtllm(c) => &c.output_token_ids,
        }
    }

    /// Get prompt tokens
    pub fn prompt_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.prompt_tokens,
            Self::Vllm(c) => c.prompt_tokens,
            Self::Trtllm(c) => c.prompt_tokens,
        }
    }

    /// Get completion tokens
    pub fn completion_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.completion_tokens,
            Self::Vllm(c) => c.completion_tokens,
            Self::Trtllm(c) => c.completion_tokens,
        }
    }

    /// Get finish reason
    pub fn finish_reason(&self) -> &str {
        match self {
            Self::Sglang(c) => &c.finish_reason,
            Self::Vllm(c) => &c.finish_reason,
            Self::Trtllm(c) => &c.finish_reason,
        }
    }

    /// Get index (for n>1 support)
    /// Returns the index of this output when n>1 was requested (0-indexed)
    pub fn index(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.index,
            Self::Vllm(c) => c.index,
            Self::Trtllm(c) => c.sequence_index,
        }
    }

    /// Get matched stop (SGLang only, returns oneof)
    /// vLLM and TensorRT-LLM don't have matched_stop, returns None
    pub fn matched_stop(&self) -> Option<&MatchedStop> {
        match self {
            Self::Sglang(c) => c.matched_stop.as_ref(),
            Self::Vllm(_) | Self::Trtllm(_) => None,
        }
    }

    /// Get matched stop as a JSON value
    ///
    /// Converts the proto MatchedStop oneof into a serde_json::Value:
    /// - MatchedTokenId → Number
    /// - MatchedStopStr → String
    /// - None → None
    pub fn matched_stop_json(&self) -> Option<serde_json::Value> {
        self.matched_stop().map(|m| match m {
            MatchedStop::MatchedTokenId(id) => {
                serde_json::Value::Number(serde_json::Number::from(*id))
            }
            MatchedStop::MatchedStopStr(s) => serde_json::Value::String(s.clone()),
        })
    }

    /// Get output IDs (decode tokens only)
    pub fn output_ids(&self) -> &[u32] {
        match self {
            Self::Sglang(c) => &c.output_ids,
            Self::Vllm(c) => &c.output_ids,
            Self::Trtllm(c) => &c.output_token_ids,
        }
    }

    /// Get cached tokens
    pub fn cached_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.cached_tokens,
            Self::Vllm(c) => c.cached_tokens,
            Self::Trtllm(c) => c.cached_tokens,
        }
    }

    /// Get input/prompt logprobs (SGLang, vLLM, and TensorRT-LLM)
    pub fn input_logprobs(&self) -> Option<ProtoInputLogProbs> {
        match self {
            Self::Sglang(c) => c.input_logprobs.as_ref().map(|lp| ProtoInputLogProbs {
                token_logprobs: lp.token_logprobs.iter().map(|t| t.value).collect(),
                token_ids: lp.token_ids.clone(),
                top_logprobs: lp
                    .top_logprobs
                    .iter()
                    .map(|t| ProtoTopLogProbs {
                        values: t.values.clone(),
                        token_ids: t.token_ids.clone(),
                    })
                    .collect(),
            }),
            Self::Vllm(c) => c.input_logprobs.as_ref().map(|lp| ProtoInputLogProbs {
                token_logprobs: lp.token_logprobs.iter().map(|t| t.value).collect(),
                token_ids: lp.token_ids.clone(),
                top_logprobs: lp
                    .top_logprobs
                    .iter()
                    .map(|t| ProtoTopLogProbs {
                        values: t.values.clone(),
                        token_ids: t.token_ids.clone(),
                    })
                    .collect(),
            }),
            Self::Trtllm(c) => {
                if c.prompt_logprobs.is_empty() {
                    None
                } else {
                    Some(ProtoInputLogProbs {
                        // First token has None logprob (no prior context)
                        token_logprobs: c
                            .prompt_logprobs
                            .iter()
                            .enumerate()
                            .map(|(i, lp)| if i == 0 { None } else { Some(lp.logprob) })
                            .collect(),
                        token_ids: c
                            .prompt_logprobs
                            .iter()
                            .map(|lp| lp.token_id as i32)
                            .collect(),
                        top_logprobs: c
                            .prompt_logprobs
                            .iter()
                            .map(|lp| ProtoTopLogProbs {
                                values: lp.top_logprobs.iter().map(|t| t.logprob).collect(),
                                token_ids: lp
                                    .top_logprobs
                                    .iter()
                                    .map(|t| t.token_id as i32)
                                    .collect(),
                            })
                            .collect(),
                    })
                }
            }
        }
    }

    /// Get output logprobs (SGLang, vLLM, and TensorRT-LLM)
    pub fn output_logprobs(&self) -> Option<ProtoOutputLogProbs> {
        match self {
            Self::Sglang(c) => c.output_logprobs.as_ref().map(|lp| ProtoOutputLogProbs {
                token_logprobs: lp.token_logprobs.clone(),
                token_ids: lp.token_ids.clone(),
                top_logprobs: lp
                    .top_logprobs
                    .iter()
                    .map(|t| ProtoTopLogProbs {
                        values: t.values.clone(),
                        token_ids: t.token_ids.clone(),
                    })
                    .collect(),
            }),
            Self::Vllm(c) => c.output_logprobs.as_ref().map(|lp| ProtoOutputLogProbs {
                token_logprobs: lp.token_logprobs.clone(),
                token_ids: lp.token_ids.clone(),
                top_logprobs: lp
                    .top_logprobs
                    .iter()
                    .map(|t| ProtoTopLogProbs {
                        values: t.values.clone(),
                        token_ids: t.token_ids.clone(),
                    })
                    .collect(),
            }),
            Self::Trtllm(c) => {
                if c.logprobs.is_empty() {
                    None
                } else {
                    Some(ProtoOutputLogProbs {
                        token_logprobs: c.logprobs.iter().map(|lp| lp.logprob).collect(),
                        token_ids: c.logprobs.iter().map(|lp| lp.token_id as i32).collect(),
                        top_logprobs: c
                            .logprobs
                            .iter()
                            .map(|lp| ProtoTopLogProbs {
                                values: lp.top_logprobs.iter().map(|t| t.logprob).collect(),
                                token_ids: lp
                                    .top_logprobs
                                    .iter()
                                    .map(|t| t.token_id as i32)
                                    .collect(),
                            })
                            .collect(),
                    })
                }
            }
        }
    }

    /// Get KV transfer parameters from prefill response (vLLM Mooncake PD only).
    /// Returns (remote_host, remote_port) if present.
    pub fn kv_transfer_params(&self) -> Option<(String, u32)> {
        match self {
            Self::Vllm(c) => c
                .kv_transfer_params
                .as_ref()
                .map(|params| (params.remote_host.clone(), params.remote_port)),
            Self::Sglang(_) | Self::Trtllm(_) => None,
        }
    }
}

/// Unified GenerateError
/// Note: vLLM proto no longer has GenerateError - errors are returned via gRPC status
#[derive(Clone)]
pub enum ProtoGenerateError {
    Sglang(sglang::GenerateError),
    Trtllm(trtllm::GenerateError),
}

impl ProtoGenerateError {
    /// Get error message
    pub fn message(&self) -> &str {
        match self {
            Self::Sglang(e) => &e.message,
            Self::Trtllm(e) => &e.message,
        }
    }
}

/// Unified stream wrapper
pub enum ProtoStream {
    Sglang(SglangStream),
    Vllm(VllmStream),
    Trtllm(TrtllmStream),
}

impl ProtoStream {
    /// Get next item from stream
    pub async fn next(&mut self) -> Option<Result<ProtoGenerateResponse, tonic::Status>> {
        match self {
            Self::Sglang(stream) => stream
                .next()
                .await
                .map(|result| result.map(|r| ProtoGenerateResponse::Sglang(Box::new(r)))),
            Self::Vllm(stream) => stream
                .next()
                .await
                .map(|result| result.map(|r| ProtoGenerateResponse::Vllm(Box::new(r)))),
            Self::Trtllm(stream) => stream
                .next()
                .await
                .map(|result| result.map(|r| ProtoGenerateResponse::Trtllm(Box::new(r)))),
        }
    }

    /// Mark stream as completed (no abort needed)
    pub fn mark_completed(&mut self) {
        match self {
            Self::Sglang(stream) => stream.mark_completed(),
            Self::Vllm(stream) => stream.mark_completed(),
            Self::Trtllm(stream) => stream.mark_completed(),
        }
    }
}

/// Unified EmbedRequest that works with both backends
#[derive(Clone)]
pub enum ProtoEmbedRequest {
    Sglang(Box<sglang::EmbedRequest>),
}

impl ProtoEmbedRequest {
    /// Get SGLang variant
    pub fn as_sglang(&self) -> &sglang::EmbedRequest {
        match self {
            Self::Sglang(req) => req,
        }
    }

    /// Get mutable SGLang variant
    pub fn as_sglang_mut(&mut self) -> &mut sglang::EmbedRequest {
        match self {
            Self::Sglang(req) => req,
        }
    }

    /// Check if this is SGLang
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Clone the inner request (for passing to embed())
    pub fn clone_inner(&self) -> Self {
        self.clone()
    }

    /// Get request ID
    pub fn request_id(&self) -> &str {
        match self {
            Self::Sglang(req) => &req.request_id,
        }
    }
}

/// Unified EmbedResponse
pub enum ProtoEmbedResponse {
    Sglang(sglang::EmbedResponse),
}

impl ProtoEmbedResponse {
    /// Get the response variant (complete or error)
    pub fn into_response(self) -> ProtoEmbedResponseVariant {
        match self {
            Self::Sglang(resp) => match resp.response {
                Some(sglang::embed_response::Response::Complete(complete)) => {
                    ProtoEmbedResponseVariant::Complete(ProtoEmbedComplete::Sglang(complete))
                }
                Some(sglang::embed_response::Response::Error(error)) => {
                    ProtoEmbedResponseVariant::Error(ProtoEmbedError::Sglang(error))
                }
                None => ProtoEmbedResponseVariant::None,
            },
        }
    }
}

/// Response variant extracted from EmbedResponse
pub enum ProtoEmbedResponseVariant {
    Complete(ProtoEmbedComplete),
    Error(ProtoEmbedError),
    None,
}

/// Unified EmbedComplete response
#[derive(Clone)]
pub enum ProtoEmbedComplete {
    Sglang(sglang::EmbedComplete),
}

impl ProtoEmbedComplete {
    /// Get embeddings
    pub fn embedding(&self) -> &[f32] {
        match self {
            Self::Sglang(c) => &c.embedding,
        }
    }

    /// Get prompt tokens
    pub fn prompt_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.prompt_tokens,
        }
    }

    /// Get cached tokens
    pub fn cached_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.cached_tokens,
        }
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.embedding_dim,
        }
    }
}

/// Unified EmbedError
#[derive(Clone)]
pub enum ProtoEmbedError {
    Sglang(sglang::EmbedError),
}

impl ProtoEmbedError {
    /// Get error message
    pub fn message(&self) -> &str {
        match self {
            Self::Sglang(e) => &e.message,
        }
    }

    /// Get error code
    pub fn code(&self) -> &str {
        match self {
            Self::Sglang(e) => &e.code,
        }
    }
}
