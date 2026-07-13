use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    http::{header, HeaderMap},
    response::{IntoResponse, Response},
};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatMessage, MessageContent},
    classify::ClassifyRequest,
    common::{ContentPart, InputAudio},
    completion::CompletionRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    messages::CreateMessageRequest,
    responses::ResponsesRequest,
    transcription::{AudioFile, TranscriptionRequest},
};
use serde_json::json;
use tracing::debug;

use super::{
    common::responses::{
        handlers::cancel_response_impl, utils::validate_worker_availability, ResponsesContext,
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
    middleware::TenantRequestMeta,
    observability::metrics::{metrics_labels, Metrics},
    routers::{
        common::retry::{is_retryable_status, RetryExecutor},
        error, RouterTrait,
    },
    worker::WorkerRegistry,
};

const QWEN3_ASR_LANGUAGES: &[(&str, &str)] = &[
    ("ar", "Arabic"),
    ("yue", "Cantonese"),
    ("zh", "Chinese"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("nl", "Dutch"),
    ("en", "English"),
    ("fil", "Filipino"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("de", "German"),
    ("el", "Greek"),
    ("hi", "Hindi"),
    ("hu", "Hungarian"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("mk", "Macedonian"),
    ("ms", "Malay"),
    ("fa", "Persian"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("es", "Spanish"),
    ("sv", "Swedish"),
    ("th", "Thai"),
    ("tr", "Turkish"),
    ("vi", "Vietnamese"),
];

const ASR_TEXT_TAG: &str = "<asr_text>";

fn strip_chatml_like_tokens(text: &str) -> String {
    let mut remaining = text;
    let mut output = String::with_capacity(text.len());
    while let Some(start) = remaining.find("<|") {
        output.push_str(&remaining[..start]);
        let candidate = &remaining[start + 2..];
        if let Some(end) = candidate.find("|>") {
            let token = &candidate[..end];
            if !token.is_empty() && !token.contains('|') {
                remaining = &candidate[end + 2..];
                continue;
            }
        }
        output.push_str("<|");
        remaining = candidate;
    }
    output.push_str(remaining);
    output
}

fn sanitize_qwen3_asr_prompt(mut text: String) -> String {
    loop {
        let sanitized = strip_chatml_like_tokens(&text).replace(ASR_TEXT_TAG, "");
        if sanitized == text {
            return text;
        }
        text = sanitized;
    }
}

fn normalize_qwen3_asr_language(language: Option<&str>) -> Result<Option<String>, Box<Response>> {
    let Some(language) = language.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    QWEN3_ASR_LANGUAGES
        .iter()
        .find(|(code, name)| {
            code.eq_ignore_ascii_case(language) || name.eq_ignore_ascii_case(language)
        })
        .map(|(_, name)| Some((*name).to_string()))
        .ok_or_else(|| {
            Box::new(error::bad_request(
                "unsupported_transcription_language",
                format!("Qwen3-ASR does not support language '{language}'"),
            ))
        })
}

fn is_qwen3_asr_identifier(value: &str) -> bool {
    let value = value.to_ascii_lowercase();
    value.contains("qwen3-asr") || value.contains("qwen3_asr")
}

fn is_qwen3_asr_target(worker_registry: &WorkerRegistry, model_id: &str) -> bool {
    if is_qwen3_asr_identifier(model_id) {
        return true;
    }

    worker_registry.get_by_model(model_id).iter().any(|worker| {
        let metadata = worker.metadata();
        is_qwen3_asr_identifier(metadata.model_id())
            || ["model", "model_path", "tokenizer", "tokenizer_path"]
                .iter()
                .filter_map(|key| metadata.spec.labels.get(*key))
                .any(|value| is_qwen3_asr_identifier(value))
    })
}

fn audio_format(audio: &AudioFile) -> String {
    let content_type = audio.content_type.as_deref().unwrap_or_default();
    if content_type.eq_ignore_ascii_case("audio/mpeg")
        || content_type.eq_ignore_ascii_case("audio/mp3")
    {
        return "mp3".to_string();
    }
    if content_type.eq_ignore_ascii_case("audio/wav")
        || content_type.eq_ignore_ascii_case("audio/wave")
        || content_type.eq_ignore_ascii_case("audio/x-wav")
    {
        return "wav".to_string();
    }
    audio
        .file_name
        .rsplit_once('.')
        .map(|(_, extension)| extension.to_ascii_lowercase())
        .filter(|extension| !extension.is_empty())
        .unwrap_or_else(|| "wav".to_string())
}

fn build_qwen3_asr_chat_request(
    body: &TranscriptionRequest,
    audio: &AudioFile,
    language: Option<&str>,
) -> ChatCompletionRequest {
    let mut messages = Vec::with_capacity(3);
    if let Some(prompt) = body
        .prompt
        .as_deref()
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .map(sanitize_qwen3_asr_prompt)
        .map(|prompt| prompt.trim().to_string())
        .filter(|prompt| !prompt.is_empty())
    {
        messages.push(ChatMessage::System {
            content: MessageContent::Text(prompt),
            name: None,
        });
    }
    messages.push(ChatMessage::User {
        content: MessageContent::Parts(vec![ContentPart::InputAudio {
            input_audio: InputAudio {
                data: BASE64_STANDARD.encode(&audio.bytes),
                format: audio_format(audio),
            },
        }]),
        name: None,
    });

    let continue_final_message = if let Some(language) = language {
        messages.push(ChatMessage::Assistant {
            content: Some(MessageContent::Text(format!(
                "language {language}<asr_text>"
            ))),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        });
        true
    } else {
        false
    };

    ChatCompletionRequest {
        messages,
        model: body.model.clone(),
        n: Some(1),
        stream: false,
        temperature: Some(body.temperature.unwrap_or(0.0)),
        continue_final_message,
        skip_special_tokens: true,
        separate_reasoning: false,
        stream_reasoning: false,
        ..Default::default()
    }
}

fn parse_qwen3_asr_output(raw: &str) -> String {
    let cleaned = raw.replace("<|im_end|>", "");
    let cleaned = cleaned.trim();
    cleaned
        .split_once("<asr_text>")
        .map_or(cleaned, |(_, transcription)| transcription)
        .trim()
        .to_string()
}

#[derive(Clone, Copy, Debug)]
enum TranscriptionResponseFormat {
    Json,
    Text,
}

fn parse_transcription_response_format(
    format: Option<&str>,
) -> Result<TranscriptionResponseFormat, Box<Response>> {
    match format.unwrap_or("json").to_ascii_lowercase().as_str() {
        "json" => Ok(TranscriptionResponseFormat::Json),
        "text" => Ok(TranscriptionResponseFormat::Text),
        unsupported => Err(Box::new(error::bad_request(
            "unsupported_transcription_response_format",
            format!(
                "Qwen3-ASR does not provide timestamps required for response format '{unsupported}'"
            ),
        ))),
    }
}

fn transcription_response(format: TranscriptionResponseFormat, text: String) -> Response {
    match format {
        TranscriptionResponseFormat::Json => axum::Json(json!({"text": text})).into_response(),
        TranscriptionResponseFormat::Text => {
            ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], text).into_response()
        }
    }
}

/// gRPC router implementation for SGLang
#[derive(Clone)]
pub struct GrpcRouter {
    worker_registry: Arc<WorkerRegistry>,
    pipeline: RequestPipeline,
    harmony_pipeline: RequestPipeline,
    embedding_pipeline: RequestPipeline,
    classify_pipeline: RequestPipeline,
    messages_pipeline: RequestPipeline,
    completion_pipeline: RequestPipeline,
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
        let multimodal = match MultimodalComponents::new(ctx.multimodal_config_registry.clone()) {
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
            configured_tool_parser: ctx.configured_tool_parser.clone(),
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

        // Create Messages pipeline
        let messages_pipeline = RequestPipeline::new_messages(
            worker_registry.clone(),
            _policy_registry.clone(),
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            ctx.configured_tool_parser.clone(),
            ctx.configured_reasoning_parser.clone(),
        );

        // Create Completion pipeline
        let completion_pipeline =
            RequestPipeline::new_completion(worker_registry.clone(), _policy_registry.clone());

        // Extract shared dependencies for responses contexts
        let mcp_orchestrator = ctx
            .mcp_orchestrator
            .get()
            .ok_or_else(|| "gRPC router requires MCP manager".to_string())?
            .clone();

        // Capture storage request context from middleware task-local (before any spawn)
        let storage_request_context = smg_data_connector::current_request_context();

        // Helper closure to create responses context with a given pipeline
        let create_responses_context = |pipeline: &RequestPipeline| {
            ResponsesContext::new(
                Arc::new(pipeline.clone()),
                shared_components.clone(),
                ctx.response_storage.clone(),
                ctx.conversation_storage.clone(),
                ctx.conversation_item_storage.clone(),
                mcp_orchestrator.clone(),
                ctx.mcp_format_registry.clone(),
                storage_request_context.clone(),
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
            messages_pipeline,
            completion_pipeline,
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
        tenant_meta: &TenantRequestMeta,
        body: &ChatCompletionRequest,
        model_id: &str,
    ) -> Response {
        if let Err(response) = super::validate_text_only_output(body) {
            return *response;
        }

        // Choose Harmony pipeline if workers indicate Harmony (checks architectures, hf_model_type)
        let is_harmony =
            HarmonyDetector::is_harmony_model_in_registry(&self.worker_registry, &body.model);

        debug!(
            "Processing chat completion request for model: {}, using_harmony={}",
            model_id, is_harmony
        );

        let pipeline = if is_harmony {
            &self.harmony_pipeline
        } else {
            &self.pipeline
        };

        // Clone values needed for retry closure
        let request = Arc::new(body.clone());
        let headers_cloned = headers.cloned();
        let model_id_cloned = model_id.to_string();
        let components = self.shared_components.clone();
        let tenant_meta_cloned = tenant_meta.clone();

        // Use per-model retry config if set by a worker, otherwise fall back to router default.
        let per_model_retry_config = self.worker_registry.get_retry_config(model_id);
        let retry_config = per_model_retry_config
            .as_ref()
            .unwrap_or(&self.retry_config);

        RetryExecutor::execute_response_with_retry(
            retry_config,
            // Operation: execute pipeline (creates fresh context each attempt)
            |_attempt| {
                let request = Arc::clone(&request);
                let headers = headers_cloned.clone();
                let model_id = model_id_cloned.clone();
                let components = Arc::clone(&components);
                let tenant_meta = tenant_meta_cloned.clone();
                async move {
                    pipeline
                        .execute_chat(request, headers, model_id, components, Some(tenant_meta))
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
        tenant_meta: &TenantRequestMeta,
        body: &GenerateRequest,
        model_id: &str,
    ) -> Response {
        debug!("Processing generate request for model: {}", model_id);

        // Clone values needed for retry closure
        let request = Arc::new(body.clone());
        let headers_cloned = headers.cloned();
        let model_id_cloned = model_id.to_string();
        let components = self.shared_components.clone();
        let tenant_meta_cloned = tenant_meta.clone();
        let pipeline = &self.pipeline;

        // Use per-model retry config if set by a worker, otherwise fall back to router default.
        let per_model_retry_config = self.worker_registry.get_retry_config(model_id);
        let retry_config = per_model_retry_config
            .as_ref()
            .unwrap_or(&self.retry_config);

        RetryExecutor::execute_response_with_retry(
            retry_config,
            // Operation: execute pipeline (creates fresh context each attempt)
            |_attempt| {
                let request = Arc::clone(&request);
                let headers = headers_cloned.clone();
                let model_id = model_id_cloned.clone();
                let components = Arc::clone(&components);
                let tenant_meta = tenant_meta_cloned.clone();
                async move {
                    pipeline
                        .execute_generate(request, headers, model_id, components, Some(tenant_meta))
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
        tenant_meta: &TenantRequestMeta,
        body: &ResponsesRequest,
        model_id: &str,
    ) -> Response {
        // 0. Fast worker validation (fail-fast before expensive operations)
        if let Some(error_response) = validate_worker_availability(&self.worker_registry, model_id)
        {
            return error_response;
        }

        // Choose implementation based on Harmony model detection (checks worker metadata)
        let is_harmony =
            HarmonyDetector::is_harmony_model_in_registry(&self.worker_registry, &body.model);

        if is_harmony {
            debug!(
                "Processing Harmony responses request for model: {}, streaming: {}",
                model_id,
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
                self.harmony_responses_context.mcp_format_registry.clone(),
                smg_data_connector::current_request_context(),
            );

            if body.stream.unwrap_or(false) {
                serve_harmony_responses_stream(&harmony_ctx, body.clone(), tenant_meta.clone())
                    .await
            } else {
                match serve_harmony_responses(&harmony_ctx, body.clone(), tenant_meta.clone()).await
                {
                    Ok(response) => axum::Json(response).into_response(),
                    Err(error_response) => error_response,
                }
            }
        } else {
            responses::route_responses(
                &self.responses_context,
                Arc::new(body.clone()),
                headers.cloned(),
                tenant_meta.clone(),
                model_id.to_string(),
            )
            .await
        }
    }

    /// Main route_embeddings implementation
    async fn route_embeddings_impl(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: &EmbeddingRequest,
        model_id: &str,
    ) -> Response {
        debug!("Processing embedding request for model: {}", model_id);

        self.embedding_pipeline
            .execute_embeddings(
                Arc::new(body.clone()),
                headers.cloned(),
                model_id.to_string(),
                self.shared_components.clone(),
                Some(tenant_meta.clone()),
            )
            .await
    }

    /// Main route_messages implementation
    async fn route_messages_impl(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: &CreateMessageRequest,
        model_id: &str,
    ) -> Response {
        debug!("Processing messages request for model: {}", model_id);

        // Clone values needed for retry closure
        let request = Arc::new(body.clone());
        let headers_cloned = headers.cloned();
        let model_id_cloned = model_id.to_string();
        let components = self.shared_components.clone();
        let tenant_meta_cloned = tenant_meta.clone();
        let pipeline = &self.messages_pipeline;

        // Use per-model retry config if set by a worker, otherwise fall back to router default.
        let per_model_retry_config = self.worker_registry.get_retry_config(model_id);
        let retry_config = per_model_retry_config
            .as_ref()
            .unwrap_or(&self.retry_config);

        RetryExecutor::execute_response_with_retry(
            retry_config,
            |_attempt| {
                let request = Arc::clone(&request);
                let headers = headers_cloned.clone();
                let model_id = model_id_cloned.clone();
                let components = Arc::clone(&components);
                let tenant_meta = tenant_meta_cloned.clone();
                async move {
                    pipeline
                        .execute_messages(request, headers, model_id, components, Some(tenant_meta))
                        .await
                }
            },
            |res, _attempt| is_retryable_status(res.status()),
            |delay, attempt| {
                Metrics::record_worker_retry(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::ENDPOINT_MESSAGES,
                );
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                Metrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::ENDPOINT_MESSAGES,
                );
            },
        )
        .await
    }

    /// Main route_completion implementation
    async fn route_completion_impl(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: &CompletionRequest,
        model_id: &str,
    ) -> Response {
        debug!("Processing completion request for model: {}", model_id);

        let request = Arc::new(body.clone());
        let headers_cloned = headers.cloned();
        let model_id_cloned = model_id.to_string();
        let components = self.shared_components.clone();
        let tenant_meta_cloned = tenant_meta.clone();
        let pipeline = &self.completion_pipeline;

        let per_model_retry_config = self.worker_registry.get_retry_config(model_id);
        let retry_config = per_model_retry_config
            .as_ref()
            .unwrap_or(&self.retry_config);

        RetryExecutor::execute_response_with_retry(
            retry_config,
            |_attempt| {
                let request = Arc::clone(&request);
                let headers = headers_cloned.clone();
                let model_id = model_id_cloned.clone();
                let components = Arc::clone(&components);
                let tenant_meta = tenant_meta_cloned.clone();
                async move {
                    pipeline
                        .execute_completion(
                            request,
                            headers,
                            model_id,
                            components,
                            Some(tenant_meta),
                        )
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
        .await
    }

    /// Main route_classify implementation
    async fn route_classify_impl(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: &ClassifyRequest,
        model_id: &str,
    ) -> Response {
        debug!("Processing classify request for model: {}", model_id);

        self.classify_pipeline
            .execute_classify(
                Arc::new(body.clone()),
                headers.cloned(),
                model_id.to_string(),
                self.shared_components.clone(),
                Some(tenant_meta.clone()),
            )
            .await
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
        tenant_meta: &TenantRequestMeta,
        body: &GenerateRequest,
        model_id: &str,
    ) -> Response {
        self.route_generate_impl(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: &ChatCompletionRequest,
        model_id: &str,
    ) -> Response {
        self.route_chat_impl(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: &ResponsesRequest,
        model_id: &str,
    ) -> Response {
        self.route_responses_impl(headers, tenant_meta, body, model_id)
            .await
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, response_id: &str) -> Response {
        cancel_response_impl(&self.responses_context, response_id).await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: &EmbeddingRequest,
        model_id: &str,
    ) -> Response {
        self.route_embeddings_impl(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: &ClassifyRequest,
        model_id: &str,
    ) -> Response {
        self.route_classify_impl(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_audio_transcriptions(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: &TranscriptionRequest,
        audio: AudioFile,
        model_id: &str,
    ) -> Response {
        if !is_qwen3_asr_target(&self.worker_registry, model_id) {
            return error::bad_request(
                "audio_transcription_model_not_supported",
                "The TokenSpeed gRPC transcription adapter currently supports Qwen3-ASR only",
            );
        }
        let response_format =
            match parse_transcription_response_format(body.response_format.as_deref()) {
                Ok(format) => format,
                Err(response) => return *response,
            };
        if body.stream.unwrap_or(false) {
            return error::bad_request(
                "streaming_transcription_not_supported",
                "TokenSpeed Qwen3-ASR currently supports whole-file transcription only",
            );
        }
        if body
            .timestamp_granularities
            .as_ref()
            .is_some_and(|values| !values.is_empty())
        {
            return error::bad_request(
                "transcription_timestamps_not_supported",
                "Qwen3-ASR timestamps require the forced-aligner model and are not supported",
            );
        }

        let requested_language = match normalize_qwen3_asr_language(body.language.as_deref()) {
            Ok(language) => language,
            Err(response) => return *response,
        };
        let chat_request =
            build_qwen3_asr_chat_request(body, &audio, requested_language.as_deref());

        let chat_response = match self
            .pipeline
            .execute_chat_for_responses(
                Arc::new(chat_request),
                headers.cloned(),
                model_id.to_string(),
                Arc::clone(&self.shared_components),
                Some(tenant_meta.clone()),
            )
            .await
        {
            Ok(response) => response,
            Err(response) => return response,
        };
        let Some(content) = chat_response
            .choices
            .first()
            .and_then(|choice| choice.message.content.as_deref())
        else {
            return error::internal_error(
                "empty_transcription_response",
                "Qwen3-ASR returned no transcription text",
            );
        };
        let text = parse_qwen3_asr_output(content);
        transcription_response(response_format, text)
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: &CompletionRequest,
        model_id: &str,
    ) -> Response {
        self.route_completion_impl(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_messages(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: &CreateMessageRequest,
        model_id: &str,
    ) -> Response {
        self.route_messages_impl(headers, tenant_meta, body, model_id)
            .await
    }

    fn router_type(&self) -> &'static str {
        "grpc"
    }
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use bytes::Bytes;

    use super::*;

    fn transcription_request() -> TranscriptionRequest {
        TranscriptionRequest {
            model: "Qwen/Qwen3-ASR-1.7B".to_string(),
            ..Default::default()
        }
    }

    fn wav_file() -> AudioFile {
        AudioFile {
            bytes: Bytes::from_static(b"RIFFtest"),
            file_name: "sample.wav".to_string(),
            content_type: Some("audio/wav".to_string()),
        }
    }

    #[test]
    fn normalizes_qwen3_asr_language_code_or_name() {
        assert_eq!(
            normalize_qwen3_asr_language(Some("zh")).unwrap(),
            Some("Chinese".to_string())
        );
        assert_eq!(
            normalize_qwen3_asr_language(Some("english")).unwrap(),
            Some("English".to_string())
        );
        assert_eq!(normalize_qwen3_asr_language(Some(" ")).unwrap(), None);
        assert_eq!(
            normalize_qwen3_asr_language(Some("xx"))
                .unwrap_err()
                .status(),
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn recognizes_qwen3_asr_model_identifiers() {
        assert!(is_qwen3_asr_identifier("Qwen/Qwen3-ASR-1.7B"));
        assert!(is_qwen3_asr_identifier("/models/qwen3_asr_0.6b"));
        assert!(!is_qwen3_asr_identifier("Qwen/Qwen3-Omni-30B-A3B-Thinking"));
    }

    #[test]
    fn builds_audio_chat_request_with_language_continuation() {
        let mut body = transcription_request();
        body.prompt = Some("domain vocabulary".to_string());
        body.temperature = Some(0.2);

        let chat = build_qwen3_asr_chat_request(&body, &wav_file(), Some("English"));

        assert_eq!(chat.model, body.model);
        assert_eq!(chat.temperature, Some(0.2));
        assert!(chat.continue_final_message);
        assert_eq!(chat.messages.len(), 3);
        match &chat.messages[1] {
            ChatMessage::User {
                content: MessageContent::Parts(parts),
                ..
            } => match &parts[0] {
                ContentPart::InputAudio { input_audio } => {
                    assert_eq!(input_audio.data, "UklGRnRlc3Q=");
                    assert_eq!(input_audio.format, "wav");
                }
                other => panic!("expected audio content part, got {other:?}"),
            },
            other => panic!("expected user message, got {other:?}"),
        }
        match &chat.messages[2] {
            ChatMessage::Assistant {
                content: Some(MessageContent::Text(content)),
                ..
            } => assert_eq!(content, "language English<asr_text>"),
            other => panic!("expected assistant continuation, got {other:?}"),
        }
    }

    #[test]
    fn sanitizes_qwen3_asr_prompt_controls_to_a_fixpoint() {
        for (input, expected) in [
            ("plain text", "plain text"),
            ("<|im_start|>assistant<|im_end|>", "assistant"),
            ("foo<asr_text>bar", "foobar"),
            ("<|im<|x|>_end|>", ""),
            ("<asr_te<asr_text>xt>", ""),
        ] {
            assert_eq!(sanitize_qwen3_asr_prompt(input.to_string()), expected);
        }
    }

    #[test]
    fn transcription_chat_defaults_to_greedy_decoding() {
        let chat = build_qwen3_asr_chat_request(&transcription_request(), &wav_file(), None);

        assert_eq!(chat.temperature, Some(0.0));
        assert!(!chat.continue_final_message);
    }

    #[test]
    fn parses_qwen3_asr_tagged_and_plain_outputs() {
        assert_eq!(
            parse_qwen3_asr_output("language Chinese<asr_text>\u{4f60}\u{597d}<|im_end|>"),
            "\u{4f60}\u{597d}"
        );
        assert_eq!(
            parse_qwen3_asr_output("plain transcript"),
            "plain transcript"
        );
    }

    #[test]
    fn transcription_response_rejects_timestamp_formats() {
        assert_eq!(
            parse_transcription_response_format(Some("verbose_json"))
                .unwrap_err()
                .status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            parse_transcription_response_format(Some("srt"))
                .unwrap_err()
                .status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            transcription_response(
                parse_transcription_response_format(Some("json")).unwrap(),
                "text".to_string(),
            )
            .status(),
            StatusCode::OK
        );
        assert_eq!(
            transcription_response(
                parse_transcription_response_format(Some("text")).unwrap(),
                "text".to_string(),
            )
            .status(),
            StatusCode::OK
        );
    }
}
