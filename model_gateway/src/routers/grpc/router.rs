use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use async_trait::async_trait;
use axum::{
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::StreamExt;
use openai_protocol::{
    chat::ChatCompletionRequest,
    classify::ClassifyRequest,
    common::{StringOrArray, Usage},
    completion::{
        CompletionChoice, CompletionRequest, CompletionResponse, CompletionStreamChoice,
        CompletionStreamResponse,
    },
    embedding::EmbeddingRequest,
    generate::{GenerateFinishReason, GenerateRequest},
    responses::{ResponsesGetParams, ResponsesRequest},
    sampling_params::SamplingParams,
};
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{debug, error};

use super::{
    common::responses::{
        build_sse_response,
        handlers::{cancel_response_impl, get_response_impl},
        utils::validate_worker_availability,
        ResponsesContext,
    },
    context::SharedComponents,
    harmony::{serve_harmony_responses, serve_harmony_responses_stream, HarmonyDetector},
    multimodal::MultimodalComponents,
    pipeline::RequestPipeline,
    regular::responses,
};
use crate::{
    app_context::AppContext,
    config::types::RetryConfig,
    core::{is_retryable_status, RetryExecutor, WorkerRegistry, UNKNOWN_MODEL_ID},
    observability::metrics::{metrics_labels, Metrics},
    routers::RouterTrait,
};

/// gRPC router implementation for SGLang
#[derive(Clone)]
pub struct GrpcRouter {
    worker_registry: Arc<WorkerRegistry>,
    pipeline: RequestPipeline,
    harmony_pipeline: RequestPipeline,
    embedding_pipeline: RequestPipeline,
    classify_pipeline: RequestPipeline,
    shared_components: Arc<SharedComponents>,
    responses_context: ResponsesContext,
    harmony_responses_context: ResponsesContext,
    retry_config: RetryConfig,
}

impl GrpcRouter {
    /// Create a new gRPC router
    pub fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        // Get tokenizer registry (no longer requires pre-loaded tokenizer)
        let tokenizer_registry = ctx.tokenizer_registry.clone();

        let reasoning_parser_factory = ctx
            .reasoning_parser_factory
            .as_ref()
            .ok_or_else(|| "gRPC router requires reasoning parser factory".to_string())?
            .clone();
        let tool_parser_factory = ctx
            .tool_parser_factory
            .as_ref()
            .ok_or_else(|| "gRPC router requires tool parser factory".to_string())?
            .clone();

        let worker_registry = ctx.worker_registry.clone();
        let _policy_registry = ctx.policy_registry.clone();

        // Create multimodal components (best-effort; non-fatal if initialization fails)
        let multimodal = match MultimodalComponents::new() {
            Ok(mc) => Some(Arc::new(mc)),
            Err(e) => {
                tracing::warn!("Multimodal components initialization failed (non-fatal): {e}");
                None
            }
        };

        // Create shared components for pipeline
        let shared_components = Arc::new(SharedComponents {
            tokenizer_registry: tokenizer_registry.clone(),
            tool_parser_factory: tool_parser_factory.clone(),
            reasoning_parser_factory: reasoning_parser_factory.clone(),
            multimodal,
        });

        // Create regular pipeline
        let pipeline = RequestPipeline::new_regular(
            worker_registry.clone(),
            _policy_registry.clone(),
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            ctx.configured_tool_parser.clone(),
            ctx.configured_reasoning_parser.clone(),
        );

        // Create Harmony pipelines
        let harmony_pipeline = RequestPipeline::new_harmony(
            worker_registry.clone(),
            _policy_registry.clone(),
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            ctx.configured_tool_parser.clone(),
            ctx.configured_reasoning_parser.clone(),
        );

        // Create Embedding pipeline
        let embedding_pipeline =
            RequestPipeline::new_embeddings(worker_registry.clone(), _policy_registry.clone());

        // Create Classify pipeline
        let classify_pipeline =
            RequestPipeline::new_classify(worker_registry.clone(), _policy_registry.clone());

        // Extract shared dependencies for responses contexts
        let mcp_orchestrator = ctx
            .mcp_orchestrator
            .get()
            .ok_or_else(|| "gRPC router requires MCP manager".to_string())?
            .clone();

        // Helper closure to create responses context with a given pipeline
        let create_responses_context = |pipeline: &RequestPipeline| {
            ResponsesContext::new(
                Arc::new(pipeline.clone()),
                shared_components.clone(),
                ctx.response_storage.clone(),
                ctx.conversation_storage.clone(),
                ctx.conversation_item_storage.clone(),
                mcp_orchestrator.clone(),
            )
        };

        // Create responses contexts for both pipelines
        let responses_context = create_responses_context(&pipeline);
        let harmony_responses_context = create_responses_context(&harmony_pipeline);

        Ok(GrpcRouter {
            worker_registry,
            pipeline,
            harmony_pipeline,
            embedding_pipeline,
            classify_pipeline,
            shared_components,
            responses_context,
            harmony_responses_context,
            retry_config: ctx.router_config.effective_retry_config(),
        })
    }

    /// Main route_chat implementation
    async fn route_chat_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Choose Harmony pipeline if workers indicate Harmony (checks architectures, hf_model_type)
        let is_harmony =
            HarmonyDetector::is_harmony_model_in_registry(&self.worker_registry, &body.model);

        debug!(
            "Processing chat completion request for model: {}, using_harmony={}",
            model_id.unwrap_or(UNKNOWN_MODEL_ID),
            is_harmony
        );

        let pipeline = if is_harmony {
            &self.harmony_pipeline
        } else {
            &self.pipeline
        };

        // Clone values needed for retry closure
        let request = Arc::new(body.clone());
        let headers_cloned = headers.cloned();
        let model_id_cloned = model_id.map(|s| s.to_string());
        let components = self.shared_components.clone();

        RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            // Operation: execute pipeline (creates fresh context each attempt)
            |_attempt| {
                let request = Arc::clone(&request);
                let headers = headers_cloned.clone();
                let model_id = model_id_cloned.clone();
                let components = Arc::clone(&components);
                async move {
                    pipeline
                        .execute_chat(request, headers, model_id, components)
                        .await
                }
            },
            // Should retry: check if status is retryable
            |res, _attempt| is_retryable_status(res.status()),
            // On backoff: record retry metrics
            |delay, attempt| {
                Metrics::record_worker_retry(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::ENDPOINT_CHAT,
                );
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            // On exhausted: record exhaustion
            || {
                Metrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::ENDPOINT_CHAT,
                );
            },
        )
        .await
    }

    /// Main route_generate implementation
    async fn route_generate_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!(
            "Processing generate request for model: {}",
            model_id.unwrap_or(UNKNOWN_MODEL_ID)
        );

        // Clone values needed for retry closure
        let request = Arc::new(body.clone());
        let headers_cloned = headers.cloned();
        let model_id_cloned = model_id.map(|s| s.to_string());
        let components = self.shared_components.clone();
        let pipeline = &self.pipeline;

        RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            // Operation: execute pipeline (creates fresh context each attempt)
            |_attempt| {
                let request = Arc::clone(&request);
                let headers = headers_cloned.clone();
                let model_id = model_id_cloned.clone();
                let components = Arc::clone(&components);
                async move {
                    pipeline
                        .execute_generate(request, headers, model_id, components)
                        .await
                }
            },
            // Should retry: check if status is retryable
            |res, _attempt| is_retryable_status(res.status()),
            // On backoff: record retry metrics
            |delay, attempt| {
                Metrics::record_worker_retry(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::ENDPOINT_GENERATE,
                );
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            // On exhausted: record exhaustion
            || {
                Metrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::ENDPOINT_GENERATE,
                );
            },
        )
        .await
    }

    /// Main route_responses implementation
    ///
    /// Routes to either Harmony or regular responses implementation based on model detection
    async fn route_responses_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        // 0. Fast worker validation (fail-fast before expensive operations)
        let requested_model: Option<&str> = model_id.or(Some(body.model.as_str()));

        if let Some(error_response) = requested_model
            .and_then(|model| validate_worker_availability(&self.worker_registry, model))
        {
            return error_response;
        }

        // Choose implementation based on Harmony model detection (checks worker metadata)
        let is_harmony =
            HarmonyDetector::is_harmony_model_in_registry(&self.worker_registry, &body.model);

        if is_harmony {
            debug!(
                "Processing Harmony responses request for model: {}, streaming: {}",
                model_id.unwrap_or(UNKNOWN_MODEL_ID),
                body.stream.unwrap_or(false)
            );
            let harmony_ctx = ResponsesContext::new(
                Arc::new(self.harmony_pipeline.clone()),
                self.shared_components.clone(),
                self.harmony_responses_context.response_storage.clone(),
                self.harmony_responses_context.conversation_storage.clone(),
                self.harmony_responses_context
                    .conversation_item_storage
                    .clone(),
                self.harmony_responses_context.mcp_orchestrator.clone(),
            );

            if body.stream.unwrap_or(false) {
                serve_harmony_responses_stream(&harmony_ctx, body.clone()).await
            } else {
                match serve_harmony_responses(&harmony_ctx, body.clone()).await {
                    Ok(response) => axum::Json(response).into_response(),
                    Err(error_response) => error_response,
                }
            }
        } else {
            responses::route_responses(
                &self.responses_context,
                Arc::new(body.clone()),
                headers.cloned(),
                model_id.map(|s| s.to_string()),
            )
            .await
        }
    }

    /// Main route_embeddings implementation
    async fn route_embeddings_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!(
            "Processing embedding request for model: {}",
            model_id.unwrap_or(UNKNOWN_MODEL_ID)
        );

        self.embedding_pipeline
            .execute_embeddings(
                Arc::new(body.clone()),
                headers.cloned(),
                model_id.map(|s| s.to_string()),
                self.shared_components.clone(),
            )
            .await
    }

    /// Main route_classify implementation
    async fn route_classify_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!(
            "Processing classify request for model: {}",
            model_id.unwrap_or(UNKNOWN_MODEL_ID)
        );

        self.classify_pipeline
            .execute_classify(
                Arc::new(body.clone()),
                headers.cloned(),
                model_id.map(|s| s.to_string()),
                self.shared_components.clone(),
            )
            .await
    }

    /// Build a GenerateRequest from a CompletionRequest, returning the resolved prompt text.
    fn build_generate_from_completion(
        body: &CompletionRequest,
        stream: bool,
    ) -> (GenerateRequest, String) {
        let prompt_text = match &body.prompt {
            StringOrArray::String(s) => s.clone(),
            StringOrArray::Array(arr) => arr.join(""),
        };

        let request = GenerateRequest {
            text: Some(prompt_text.clone()),
            model: Some(body.model.clone()),
            sampling_params: Some(SamplingParams {
                temperature: body.temperature,
                max_new_tokens: body.max_tokens,
                top_p: body.top_p,
                top_k: body.top_k,
                frequency_penalty: body.frequency_penalty,
                presence_penalty: body.presence_penalty,
                repetition_penalty: body.repetition_penalty,
                stop: body.stop.clone(),
                ignore_eos: Some(body.ignore_eos),
                n: body.n,
                min_p: body.min_p,
                min_new_tokens: body.min_tokens,
                regex: body.regex.clone(),
                ebnf: body.ebnf.clone(),
                json_schema: body.json_schema.clone(),
                stop_token_ids: body.stop_token_ids.clone(),
                no_stop_trim: Some(body.no_stop_trim),
                skip_special_tokens: Some(body.skip_special_tokens),
                sampling_seed: body.sampling_seed.or_else(|| body.seed.map(|s| s as u64)),
            }),
            stream,
            return_logprob: body.logprobs.map(|_| true),
            top_logprobs_num: body.logprobs.map(|l| l as i32),
            lora_path: body.lora_path.clone(),
            session_params: body.session_params.clone(),
            return_hidden_states: body.return_hidden_states,
            ..Default::default()
        };

        (request, prompt_text)
    }

    /// Main route_completion implementation
    async fn route_completion_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!(
            "Processing completion request for model: {}, stream={}",
            model_id.unwrap_or(UNKNOWN_MODEL_ID),
            body.stream
        );

        if body.stream {
            self.route_completion_streaming(headers, body, model_id)
                .await
        } else {
            self.route_completion_non_streaming(headers, body, model_id)
                .await
        }
    }

    /// Non-streaming /v1/completions: run through generate pipeline, wrap as CompletionResponse.
    async fn route_completion_non_streaming(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let (gen_request, prompt_text) = Self::build_generate_from_completion(body, false);

        let request = Arc::new(gen_request);
        let headers_cloned = headers.cloned();
        let model_id_cloned = model_id.map(|s| s.to_string());
        let components = self.shared_components.clone();
        let pipeline = &self.pipeline;
        let model = body.model.clone();
        let echo = body.echo;
        let suffix = body.suffix.clone();

        RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            |_attempt| {
                let request = Arc::clone(&request);
                let headers = headers_cloned.clone();
                let model_id = model_id_cloned.clone();
                let components = Arc::clone(&components);
                let model = model.clone();
                let prompt_text = prompt_text.clone();
                let suffix = suffix.clone();
                async move {
                    let gen_responses = match pipeline
                        .execute_generate_typed(request, headers, model_id, components)
                        .await
                    {
                        Ok(responses) => responses,
                        Err(response) => return response,
                    };

                    build_completion_response(
                        gen_responses,
                        &model,
                        &prompt_text,
                        echo,
                        suffix.as_deref(),
                    )
                }
            },
            |res, _attempt| is_retryable_status(res.status()),
            |delay, attempt| {
                Metrics::record_worker_retry(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::ENDPOINT_COMPLETIONS,
                );
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                Metrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::ENDPOINT_COMPLETIONS,
                );
            },
        )
        .await
    }

    /// Streaming /v1/completions: run through generate pipeline, transform SSE from
    /// SGLang generate format to OpenAI CompletionStreamResponse format.
    async fn route_completion_streaming(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let (gen_request, prompt_text) = Self::build_generate_from_completion(body, true);

        let request = Arc::new(gen_request);
        let headers_cloned = headers.cloned();
        let model_id_cloned = model_id.map(|s| s.to_string());
        let components = self.shared_components.clone();
        let pipeline = &self.pipeline;

        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            |_attempt| {
                let request = Arc::clone(&request);
                let headers = headers_cloned.clone();
                let model_id = model_id_cloned.clone();
                let components = Arc::clone(&components);
                async move {
                    pipeline
                        .execute_generate(request, headers, model_id, components)
                        .await
                }
            },
            |res, _attempt| is_retryable_status(res.status()),
            |delay, attempt| {
                Metrics::record_worker_retry(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::ENDPOINT_COMPLETIONS,
                );
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                Metrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::ENDPOINT_COMPLETIONS,
                );
            },
        )
        .await;

        if !response.status().is_success() {
            return response;
        }

        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let echo_prompt = if body.echo { Some(prompt_text) } else { None };

        transform_generate_sse_to_completion_sse(
            response,
            body.model.clone(),
            created,
            echo_prompt,
            body.suffix.clone(),
        )
    }
}

impl std::fmt::Debug for GrpcRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.worker_registry.stats();
        f.debug_struct("GrpcRouter")
            .field("workers_count", &stats.total_workers)
            .finish()
    }
}

#[async_trait]
impl RouterTrait for GrpcRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_generate_impl(headers, body, model_id).await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_chat_impl(headers, body, model_id).await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_completion_impl(headers, body, model_id).await
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_responses_impl(headers, body, model_id).await
    }

    async fn get_response(
        &self,
        _headers: Option<&HeaderMap>,
        response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        get_response_impl(&self.responses_context, response_id).await
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, response_id: &str) -> Response {
        cancel_response_impl(&self.responses_context, response_id).await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_embeddings_impl(headers, body, model_id).await
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_classify_impl(headers, body, model_id).await
    }

    fn router_type(&self) -> &'static str {
        "grpc"
    }
}

/// Build a non-streaming CompletionResponse from typed generate results.
fn build_completion_response(
    gen_responses: Vec<openai_protocol::generate::GenerateResponse>,
    model: &str,
    prompt_text: &str,
    echo: bool,
    suffix: Option<&str>,
) -> Response {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut total_prompt = 0u32;
    let mut total_completion = 0u32;

    let choices: Vec<CompletionChoice> = gen_responses
        .iter()
        .enumerate()
        .map(|(i, gen)| {
            total_prompt = total_prompt.max(gen.meta_info.prompt_tokens);
            total_completion += gen.meta_info.completion_tokens;

            let finish_reason = match &gen.meta_info.finish_reason {
                GenerateFinishReason::Length { .. } => "length",
                GenerateFinishReason::Stop { .. } => "stop",
                GenerateFinishReason::Other(_) => "stop",
            };

            let mut text = String::new();
            if echo {
                text.push_str(prompt_text);
            }
            text.push_str(&gen.text);
            if let Some(sfx) = suffix {
                text.push_str(sfx);
            }

            CompletionChoice {
                text,
                index: i as u32,
                logprobs: None,
                finish_reason: Some(finish_reason.to_string()),
                matched_stop: gen.meta_info.matched_stop.clone(),
            }
        })
        .collect();

    let request_id = gen_responses
        .first()
        .map(|g| g.meta_info.id.as_str())
        .unwrap_or("cmpl-unknown");

    let completion_response = CompletionResponse {
        id: request_id.to_string(),
        object: "text_completion".to_string(),
        created,
        model: model.to_string(),
        choices,
        usage: Some(Usage {
            prompt_tokens: total_prompt,
            completion_tokens: total_completion,
            total_tokens: total_prompt + total_completion,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }),
        system_fingerprint: None,
    };

    axum::Json(completion_response).into_response()
}

/// Transform an SSE response from SGLang generate format to OpenAI CompletionStreamResponse format.
///
/// SGLang emits accumulated text per chunk; this function computes deltas per index so
/// the output matches the OpenAI `/v1/completions` streaming contract.
fn transform_generate_sse_to_completion_sse(
    response: Response,
    model: String,
    created: u64,
    echo_prompt: Option<String>,
    suffix: Option<String>,
) -> Response {
    let (_, body) = response.into_parts();
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();

    #[expect(
        clippy::disallowed_methods,
        reason = "streaming transform is fire-and-forget; client disconnect terminates it"
    )]
    tokio::spawn(async move {
        let mut stream = body.into_data_stream();
        let mut prev_lens: HashMap<u32, usize> = HashMap::new();
        let mut request_id = String::from("cmpl-unknown");
        let mut echo_sent_for: HashSet<u32> = HashSet::new();
        let mut buf = String::new();

        while let Some(chunk_result) = stream.next().await {
            let bytes = match chunk_result {
                Ok(b) => b,
                Err(e) => {
                    error!("Completion stream read error: {e}");
                    break;
                }
            };

            buf.push_str(&String::from_utf8_lossy(&bytes));

            while let Some(boundary) = buf.find("\n\n") {
                let event = buf[..boundary].trim().to_string();
                buf = buf[boundary + 2..].to_string();

                if event.is_empty() {
                    continue;
                }

                let Some(data) = event.strip_prefix("data: ") else {
                    continue;
                };

                if data == "[DONE]" {
                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                    continue;
                }

                let Ok(gen_chunk) = serde_json::from_str::<Value>(data) else {
                    continue;
                };

                if gen_chunk.get("error").is_some() {
                    let _ = tx.send(Ok(Bytes::from(format!("data: {data}\n\n"))));
                    continue;
                }

                let index = gen_chunk["index"].as_u64().unwrap_or(0) as u32;
                let accumulated_text = gen_chunk["text"].as_str().unwrap_or("");

                if request_id == "cmpl-unknown" {
                    if let Some(id) = gen_chunk["meta_info"]["id"].as_str() {
                        request_id = if let Some(base) = id.rsplit_once('-') {
                            format!("cmpl-{}", base.0)
                        } else {
                            format!("cmpl-{id}")
                        };
                    }
                }

                let prev_len = prev_lens.entry(index).or_insert(0);
                let mut delta_owned = String::new();
                if echo_prompt.is_some() && echo_sent_for.insert(index) {
                    delta_owned.push_str(echo_prompt.as_deref().unwrap_or(""));
                }
                if accumulated_text.len() > *prev_len {
                    delta_owned.push_str(&accumulated_text[*prev_len..]);
                }
                *prev_len = accumulated_text.len();

                let finish_reason = match &gen_chunk["meta_info"]["finish_reason"] {
                    Value::Null => None,
                    Value::String(s) => match s.as_str() {
                        "length" | "stop" => Some(s.to_string()),
                        _ => Some("stop".to_string()),
                    },
                    obj => match obj["type"].as_str() {
                        Some("length") => Some("length".to_string()),
                        _ => Some("stop".to_string()),
                    },
                };

                if finish_reason.is_some() {
                    if let Some(ref sfx) = suffix {
                        delta_owned.push_str(sfx);
                    }
                }

                let chunk = CompletionStreamResponse {
                    id: request_id.clone(),
                    object: "text_completion".to_string(),
                    created,
                    choices: vec![CompletionStreamChoice {
                        text: delta_owned,
                        index,
                        logprobs: None,
                        finish_reason,
                    }],
                    model: model.clone(),
                    system_fingerprint: None,
                };

                if let Ok(sse_data) = serde_json::to_string(&chunk) {
                    if tx
                        .send(Ok(Bytes::from(format!("data: {sse_data}\n\n"))))
                        .is_err()
                    {
                        return;
                    }
                }
            }
        }
    });

    build_sse_response(rx)
}
