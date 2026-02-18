use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::Duration,
};

use llm_tokenizer::bundle::{
    collect_tokenizer_bundle, TokenizerBundle, TokenizerChunk, TokenizerFileChunk,
    TokenizerFileDescriptor, TokenizerMetadata,
};
use openai_protocol::{
    chat::ChatCompletionRequest, common::ResponseFormat, generate::GenerateRequest,
    responses::ResponsesRequest, sampling_params::SamplingParams as GenerateSamplingParams,
};
use tonic::{transport::Channel, Request, Streaming};
use tracing::{debug, warn};

use crate::{BoxedTraceInjector, NoopTraceInjector};

// Include the generated protobuf code
#[allow(clippy::all)]
pub mod proto {
    #![allow(clippy::all, unused_qualifications)]
    tonic::include_proto!("trtllm");
}

/// A smart wrapper around Streaming<GenerateResponse> that automatically
/// sends abort when dropped (e.g., due to client disconnection or early termination).
///
/// This leverages Rust's RAII pattern to ensure cleanup happens automatically,
/// regardless of how the stream is dropped (panic, early return, client disconnect, etc.).
pub struct AbortOnDropStream {
    inner: Streaming<proto::GenerateResponse>,
    request_id: String,
    client: TrtllmServiceClient,
    aborted: Arc<AtomicBool>,
}

impl AbortOnDropStream {
    /// Create a new auto-aborting stream wrapper
    pub fn new(
        stream: Streaming<proto::GenerateResponse>,
        request_id: String,
        client: TrtllmServiceClient,
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
    /// Call this when the request completes successfully to avoid unnecessary abort RPC.
    pub fn mark_completed(&self) {
        // Use Release ordering to ensure that this write is visible to other threads
        // that use Acquire on the same atomic variable
        self.aborted.store(true, Ordering::Release);
        debug!("Request {} marked as completed", self.request_id);
    }
}

impl Drop for AbortOnDropStream {
    fn drop(&mut self) {
        // Atomically check and set the aborted flag using compare_exchange.
        // If compare_exchange fails, it means the flag was already true (from mark_completed),
        // so we don't need to send abort. AcqRel is used for success to synchronize with
        // mark_completed's Release, and Acquire for failure to see writes from mark_completed.
        if self
            .aborted
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }

        let client = self.client.clone();
        let request_id = self.request_id.clone();

        // Spawn a background task to send abort (since Drop is sync but abort_request is async)
        tokio::spawn(async move {
            debug!(
                "Stream dropped without completion for request {}, sending abort",
                request_id
            );
            // Clone request_id for the error message since abort_request takes ownership
            let request_id_for_log = request_id.clone();
            if let Err(e) = client.abort_request(request_id).await {
                warn!(
                    "Failed to send abort on drop for request {}: {}",
                    request_id_for_log, e
                );
            }
        });
    }
}

// Implement Stream trait to make AbortOnDropStream work like the original Streaming
impl futures::Stream for AbortOnDropStream {
    type Item = Result<proto::GenerateResponse, tonic::Status>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Delegate to the inner stream
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

crate::impl_decode_tokenizer_chunk!();

/// gRPC client for TensorRT-LLM service
#[derive(Clone)]
pub struct TrtllmServiceClient {
    client: proto::trtllm_service_client::TrtllmServiceClient<Channel>,
    trace_injector: BoxedTraceInjector,
}

impl TrtllmServiceClient {
    /// Create a new client and connect to the TensorRT-LLM server
    pub async fn connect(endpoint: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::connect_with_trace_injector(endpoint, Arc::new(NoopTraceInjector)).await
    }

    /// Create a new client with a custom trace injector
    pub async fn connect_with_trace_injector(
        endpoint: &str,
        trace_injector: BoxedTraceInjector,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Connecting to TensorRT-LLM gRPC server at {}", endpoint);

        // Convert grpc:// to http:// for tonic
        let http_endpoint = if let Some(addr) = endpoint.strip_prefix("grpc://") {
            format!("http://{}", addr)
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
            .initial_stream_window_size(Some(16 * 1024 * 1024)) // 16MB
            .initial_connection_window_size(Some(32 * 1024 * 1024)) // 32MB
            .connect()
            .await?;

        let client = proto::trtllm_service_client::TrtllmServiceClient::new(channel);

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
    ///
    /// The returned stream automatically sends an abort request when dropped,
    /// ensuring proper cleanup even if the HTTP client disconnects or an error occurs.
    /// Call `mark_completed()` on the stream after successful completion to prevent
    /// unnecessary abort RPCs.
    pub async fn generate(
        &self,
        req: proto::GenerateRequest,
    ) -> Result<AbortOnDropStream, Box<dyn std::error::Error + Send + Sync>> {
        let request_id = req.request_id.clone();
        let mut client = self.client.clone();
        let mut request = Request::new(req);

        // Inject W3C trace context into gRPC metadata for distributed tracing
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
    pub async fn health_check(
        &self,
    ) -> Result<proto::HealthCheckResponse, Box<dyn std::error::Error + Send + Sync>> {
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
    ) -> Result<proto::AbortResponse, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Sending abort request for {}", request_id);
        let request = Request::new(proto::AbortRequest {
            request_id: request_id.clone(),
        });

        let mut client = self.client.clone();
        let response = client.abort(request).await?;
        debug!(
            "Abort response for {}: success={}, message={}",
            request_id,
            response.get_ref().success,
            response.get_ref().message
        );
        Ok(response.into_inner())
    }

    /// Get model information
    pub async fn get_model_info(
        &self,
    ) -> Result<proto::GetModelInfoResponse, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Requesting model info");
        let request = Request::new(proto::GetModelInfoRequest {});

        let mut client = self.client.clone();
        let response = client.get_model_info(request).await?;
        debug!("Model info response received");
        Ok(response.into_inner())
    }

    /// Get server information
    pub async fn get_server_info(
        &self,
    ) -> Result<proto::GetServerInfoResponse, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Requesting server info");
        let request = Request::new(proto::GetServerInfoRequest {});

        let mut client = self.client.clone();
        let response = client.get_server_info(request).await?;
        debug!("Server info response received");
        Ok(response.into_inner())
    }

    crate::impl_get_tokenizer!();

    /// Build a TensorRT-LLM GenerateRequest from OpenAI ChatCompletionRequest
    pub fn build_generate_request_from_chat(
        &self,
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        tool_call_constraint: Option<(String, String)>, // (constraint_type, constraint_value)
    ) -> Result<proto::GenerateRequest, String> {
        // Build sampling config
        let sampling_config = self.build_sampling_config_from_chat(body)?;

        // Build output config
        let output_config = self.build_output_config_from_chat(body);

        // Build guided decoding params if needed
        let guided_decoding = self.build_guided_decoding_from_chat(body, tool_call_constraint)?;

        // Stop words are injected by the router after building (via tokenization),
        // since TRT-LLM requires tokenized stop sequences (Vec<TokenSequence>).
        let stop_words = vec![];

        let max_tokens = body.max_completion_tokens.unwrap_or(2048);

        let grpc_request = proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: processed_text,
                input_token_ids: token_ids,
                query_token_ids: vec![],
            }),
            sampling_config: Some(sampling_config),
            output_config: Some(output_config),
            max_tokens,
            streaming: body.stream,
            end_id: None,
            pad_id: None,
            bad_words: vec![],
            stop_words,
            guided_decoding,
            embedding_bias: vec![],
            lora_config: None,
            prompt_tuning_config: None,
            multimodal_input: None,
            kv_cache_retention: None,
            disaggregated_params: None,
            lookahead_config: None,
            cache_salt_id: None,
            arrival_time: None,
        };

        Ok(grpc_request)
    }

    /// Build a basic GenerateRequest from the GenerateRequest spec
    pub fn build_plain_generate_request(
        &self,
        request_id: String,
        body: &GenerateRequest,
        original_text: Option<String>,
        token_ids: Vec<u32>,
    ) -> Result<proto::GenerateRequest, String> {
        let sampling_config =
            Self::build_sampling_config_from_plain(body.sampling_params.as_ref())?;
        let output_config = proto::OutputConfig {
            logprobs: if body.return_logprob.unwrap_or(false) {
                Some(body.top_logprobs_num.unwrap_or(0))
            } else {
                None
            },
            prompt_logprobs: None,
            return_context_logits: false,
            return_generation_logits: false,
            exclude_input_from_output: true,
            return_encoder_output: false,
            return_perf_metrics: false,
        };

        // Build guided decoding from plain sampling params
        let guided_decoding = if let Some(params) = &body.sampling_params {
            Self::build_guided_decoding_from_plain(params)?
        } else {
            None
        };

        let max_tokens = body
            .sampling_params
            .as_ref()
            .and_then(|p| p.max_new_tokens)
            .unwrap_or(2048);

        let grpc_request = proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: original_text.unwrap_or_default(),
                input_token_ids: token_ids,
                query_token_ids: vec![],
            }),
            sampling_config: Some(sampling_config),
            output_config: Some(output_config),
            max_tokens,
            streaming: body.stream,
            end_id: None,
            pad_id: None,
            bad_words: vec![],
            stop_words: vec![],
            guided_decoding,
            embedding_bias: vec![],
            lora_config: None,
            prompt_tuning_config: None,
            multimodal_input: None,
            kv_cache_retention: None,
            disaggregated_params: None,
            lookahead_config: None,
            cache_salt_id: None,
            arrival_time: None,
        };

        Ok(grpc_request)
    }

    /// Build a GenerateRequest from ResponsesRequest (OpenAI Responses API)
    pub fn build_generate_request_from_responses(
        &self,
        request_id: String,
        body: &ResponsesRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        _harmony_stop_ids: Option<Vec<u32>>,
        constraint: Option<(String, String)>,
    ) -> Result<proto::GenerateRequest, String> {
        let sampling_config = self.build_sampling_config_from_responses(body)?;
        let output_config = proto::OutputConfig {
            logprobs: body.top_logprobs.map(|v| v as i32),
            prompt_logprobs: None,
            return_context_logits: false,
            return_generation_logits: false,
            exclude_input_from_output: true,
            return_encoder_output: false,
            return_perf_metrics: false,
        };

        let guided_decoding = self.build_guided_decoding_from_responses(constraint)?;

        let max_tokens = body.max_output_tokens.unwrap_or(2048);

        let grpc_request = proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: processed_text,
                input_token_ids: token_ids,
                query_token_ids: vec![],
            }),
            sampling_config: Some(sampling_config),
            output_config: Some(output_config),
            max_tokens,
            streaming: body.stream.unwrap_or(false),
            end_id: None,
            pad_id: None,
            bad_words: vec![],
            stop_words: vec![],
            guided_decoding,
            embedding_bias: vec![],
            lora_config: None,
            prompt_tuning_config: None,
            multimodal_input: None,
            kv_cache_retention: None,
            disaggregated_params: None,
            lookahead_config: None,
            cache_salt_id: None,
            arrival_time: None,
        };

        Ok(grpc_request)
    }

    /// Build SamplingConfig from ChatCompletionRequest
    ///
    /// Uses sensible defaults when values are not specified:
    /// - temperature: 1.0 (neutral sampling)
    /// - top_p: 1.0 (no nucleus filtering)
    /// - repetition_penalty: 1.0 (no penalty)
    fn build_sampling_config_from_chat(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<proto::SamplingConfig, String> {
        Ok(proto::SamplingConfig {
            beam_width: 1,
            num_return_sequences: request.n.unwrap_or(1),
            top_k: request.top_k.map(|v| v.max(0)),
            top_p: Some(request.top_p.unwrap_or(1.0)),
            top_p_min: None,
            top_p_reset_ids: None,
            top_p_decay: None,
            seed: None,
            temperature: Some(request.temperature.unwrap_or(1.0)),
            min_tokens: None,
            beam_search_diversity_rate: None,
            repetition_penalty: Some(request.repetition_penalty.unwrap_or(1.0)),
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            prompt_ignore_length: None,
            length_penalty: None,
            early_stopping: None,
            no_repeat_ngram_size: None,
            min_p: request.min_p,
            beam_width_array: vec![],
        })
    }

    /// Build OutputConfig from ChatCompletionRequest
    fn build_output_config_from_chat(
        &self,
        request: &ChatCompletionRequest,
    ) -> proto::OutputConfig {
        proto::OutputConfig {
            logprobs: if request.logprobs {
                Some(request.top_logprobs.unwrap_or(0) as i32)
            } else {
                None
            },
            prompt_logprobs: None,
            return_context_logits: false,
            return_generation_logits: false,
            exclude_input_from_output: true,
            return_encoder_output: false,
            return_perf_metrics: false,
        }
    }

    /// Build GuidedDecodingParams from ChatCompletionRequest
    fn build_guided_decoding_from_chat(
        &self,
        request: &ChatCompletionRequest,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<Option<proto::GuidedDecodingParams>, String> {
        // Handle tool call constraint first
        if let Some((constraint_type, constraint_value)) = tool_call_constraint {
            let guide_type = match constraint_type.as_str() {
                "structural_tag" => proto::guided_decoding_params::GuideType::StructuralTag,
                "json_schema" => proto::guided_decoding_params::GuideType::JsonSchema,
                "ebnf" | "grammar" => proto::guided_decoding_params::GuideType::EbnfGrammar,
                "regex" => proto::guided_decoding_params::GuideType::Regex,
                _ => return Err(format!("Unknown constraint type: {}", constraint_type)),
            };
            return Ok(Some(proto::GuidedDecodingParams {
                guide_type: guide_type as i32,
                guide: constraint_value,
            }));
        }

        // Handle response_format
        match &request.response_format {
            Some(ResponseFormat::JsonObject) => {
                let schema = serde_json::json!({"type": "object"});
                let schema_str = serde_json::to_string(&schema)
                    .map_err(|e| format!("Failed to serialize JSON schema: {}", e))?;
                return Ok(Some(proto::GuidedDecodingParams {
                    guide_type: proto::guided_decoding_params::GuideType::JsonSchema as i32,
                    guide: schema_str,
                }));
            }
            Some(ResponseFormat::JsonSchema { json_schema }) => {
                let schema_str = serde_json::to_string(&json_schema.schema)
                    .map_err(|e| format!("Failed to serialize JSON schema: {}", e))?;
                return Ok(Some(proto::GuidedDecodingParams {
                    guide_type: proto::guided_decoding_params::GuideType::JsonSchema as i32,
                    guide: schema_str,
                }));
            }
            Some(ResponseFormat::Text) | None => {}
        }

        // Handle ebnf/regex from request
        if let Some(ebnf) = &request.ebnf {
            return Ok(Some(proto::GuidedDecodingParams {
                guide_type: proto::guided_decoding_params::GuideType::EbnfGrammar as i32,
                guide: ebnf.clone(),
            }));
        }

        if let Some(regex) = &request.regex {
            return Ok(Some(proto::GuidedDecodingParams {
                guide_type: proto::guided_decoding_params::GuideType::Regex as i32,
                guide: regex.clone(),
            }));
        }

        Ok(None)
    }

    /// Build SamplingConfig from ResponsesRequest
    /// Build SamplingConfig from ResponsesRequest
    fn build_sampling_config_from_responses(
        &self,
        request: &ResponsesRequest,
    ) -> Result<proto::SamplingConfig, String> {
        Ok(proto::SamplingConfig {
            beam_width: 1,
            num_return_sequences: 1,
            top_k: None,
            top_p: Some(request.top_p.unwrap_or(1.0)),
            top_p_min: None,
            top_p_reset_ids: None,
            top_p_decay: None,
            seed: None,
            temperature: Some(request.temperature.unwrap_or(1.0)),
            min_tokens: None,
            beam_search_diversity_rate: None,
            repetition_penalty: Some(1.0),
            presence_penalty: None,
            frequency_penalty: None,
            prompt_ignore_length: None,
            length_penalty: None,
            early_stopping: None,
            no_repeat_ngram_size: None,
            min_p: None,
            beam_width_array: vec![],
        })
    }

    /// Build GuidedDecodingParams from ResponsesRequest constraint
    fn build_guided_decoding_from_responses(
        &self,
        constraint: Option<(String, String)>,
    ) -> Result<Option<proto::GuidedDecodingParams>, String> {
        if let Some((constraint_type, constraint_value)) = constraint {
            let guide_type = match constraint_type.as_str() {
                "structural_tag" => proto::guided_decoding_params::GuideType::StructuralTag,
                "json_schema" => proto::guided_decoding_params::GuideType::JsonSchema,
                "ebnf" | "grammar" => proto::guided_decoding_params::GuideType::EbnfGrammar,
                "regex" => proto::guided_decoding_params::GuideType::Regex,
                _ => return Err(format!("Unknown constraint type: {}", constraint_type)),
            };
            Ok(Some(proto::GuidedDecodingParams {
                guide_type: guide_type as i32,
                guide: constraint_value,
            }))
        } else {
            Ok(None)
        }
    }

    fn build_sampling_config_from_plain(
        params: Option<&GenerateSamplingParams>,
    ) -> Result<proto::SamplingConfig, String> {
        let mut config = proto::SamplingConfig {
            beam_width: 1,
            num_return_sequences: 1,
            top_k: None,
            top_p: Some(1.0),
            top_p_min: None,
            top_p_reset_ids: None,
            top_p_decay: None,
            seed: None,
            temperature: Some(1.0),
            min_tokens: None,
            beam_search_diversity_rate: None,
            repetition_penalty: Some(1.0),
            presence_penalty: None,
            frequency_penalty: None,
            prompt_ignore_length: None,
            length_penalty: None,
            early_stopping: None,
            no_repeat_ngram_size: None,
            min_p: None,
            beam_width_array: vec![],
        };

        let Some(p) = params else {
            return Ok(config);
        };

        if let Some(val) = p.temperature {
            config.temperature = Some(val);
        }
        if let Some(val) = p.top_p {
            config.top_p = Some(val);
        }
        if let Some(val) = p.top_k {
            config.top_k = Some(val.max(0));
        }
        if let Some(val) = p.frequency_penalty {
            config.frequency_penalty = Some(val);
        }
        if let Some(val) = p.presence_penalty {
            config.presence_penalty = Some(val);
        }
        if let Some(val) = p.repetition_penalty {
            config.repetition_penalty = Some(val);
        }
        if let Some(val) = p.min_p {
            config.min_p = Some(val);
        }
        config.min_tokens = p.min_new_tokens;
        if let Some(n) = p.n {
            config.num_return_sequences = n;
        }

        Ok(config)
    }

    fn build_guided_decoding_from_plain(
        params: &GenerateSamplingParams,
    ) -> Result<Option<proto::GuidedDecodingParams>, String> {
        if let Some(json_schema) = &params.json_schema {
            return Ok(Some(proto::GuidedDecodingParams {
                guide_type: proto::guided_decoding_params::GuideType::JsonSchema as i32,
                guide: json_schema.clone(),
            }));
        }
        if let Some(regex) = &params.regex {
            return Ok(Some(proto::GuidedDecodingParams {
                guide_type: proto::guided_decoding_params::GuideType::Regex as i32,
                guide: regex.clone(),
            }));
        }
        if let Some(ebnf) = &params.ebnf {
            return Ok(Some(proto::GuidedDecodingParams {
                guide_type: proto::guided_decoding_params::GuideType::EbnfGrammar as i32,
                guide: ebnf.clone(),
            }));
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proto_types_compilation() {
        let _health_req = proto::HealthCheckRequest {};
    }

    #[test]
    fn test_generate_request_construction() {
        let sampling_config = proto::SamplingConfig {
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(50),
            beam_width: 1,
            num_return_sequences: 1,
            ..Default::default()
        };

        let output_config = proto::OutputConfig {
            logprobs: Some(5),
            exclude_input_from_output: true,
            ..Default::default()
        };

        let gen_req = proto::GenerateRequest {
            request_id: "test-req-123".to_string(),
            tokenized: Some(proto::TokenizedInput {
                original_text: "Hello world".to_string(),
                input_token_ids: vec![9906, 1917],
                query_token_ids: vec![],
            }),
            sampling_config: Some(sampling_config),
            output_config: Some(output_config),
            max_tokens: 128,
            streaming: false,
            ..Default::default()
        };

        assert_eq!(gen_req.request_id, "test-req-123");
        if let Some(ref tokenized) = gen_req.tokenized {
            assert_eq!(tokenized.original_text, "Hello world");
            assert_eq!(tokenized.input_token_ids, vec![9906, 1917]);
        }

        let config = gen_req.sampling_config.unwrap();
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.top_p, Some(0.9));
    }

    #[test]
    fn test_health_check_request() {
        let _health_req = proto::HealthCheckRequest {};
    }

    #[test]
    fn test_abort_request_construction() {
        let abort_req = proto::AbortRequest {
            request_id: "req-456".to_string(),
        };
        assert_eq!(abort_req.request_id, "req-456");
    }

    #[test]
    fn test_sampling_config_defaults() {
        let config = proto::SamplingConfig::default();
        assert_eq!(config.beam_width, 0);
        assert_eq!(config.temperature, None);
        assert_eq!(config.top_p, None);
        assert_eq!(config.top_k, None);
    }

    #[tokio::test]
    async fn test_client_connect_invalid_endpoint() {
        let result = TrtllmServiceClient::connect("invalid://endpoint").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_tokenized_input() {
        let tokenized = proto::TokenizedInput {
            original_text: "Hello world".to_string(),
            input_token_ids: vec![1, 15043, 1917, 2],
            query_token_ids: vec![],
        };

        assert_eq!(tokenized.original_text, "Hello world");
        assert_eq!(tokenized.input_token_ids, vec![1, 15043, 1917, 2]);
    }

    #[test]
    fn test_generate_stream_chunk() {
        let chunk = proto::GenerateStreamChunk {
            token_ids: vec![1234, 5678],
            sequence_index: 0,
            prompt_tokens: 5,
            completion_tokens: 2,
            cached_tokens: 3,
            logprobs: vec![],
        };

        assert_eq!(chunk.token_ids, vec![1234, 5678]);
        assert_eq!(chunk.prompt_tokens, 5);
        assert_eq!(chunk.completion_tokens, 2);
        assert_eq!(chunk.cached_tokens, 3);
    }

    #[test]
    fn test_guided_decoding_params() {
        let guided = proto::GuidedDecodingParams {
            guide_type: proto::guided_decoding_params::GuideType::JsonSchema as i32,
            guide: r#"{"type": "object"}"#.to_string(),
        };

        assert_eq!(
            guided.guide_type,
            proto::guided_decoding_params::GuideType::JsonSchema as i32
        );
        assert_eq!(guided.guide, r#"{"type": "object"}"#);
    }
}
