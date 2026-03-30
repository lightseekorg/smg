//! Streaming response processor for gRPC routers
//!
//! This module contains shared streaming logic for both Regular and PD router.

use std::{
    collections::{HashMap, HashSet},
    io,
    sync::Arc,
    time::Instant,
};

use axum::response::Response;
use bytes::Bytes;
use llm_tokenizer::{
    stop::{SequenceDecoderOutput, StopSequenceDecoder},
    traits::Tokenizer,
};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatCompletionStreamResponse},
    common::{
        FunctionCallDelta, FunctionCallResponse, ResponseFormat, StringOrArray, Tool, ToolCall,
        ToolCallDelta, ToolChoice, ToolChoiceValue, Usage,
    },
    generate::GenerateRequest,
};
use reasoning_parser::{ParserFactory as ReasoningParserFactory, ParserResult, ReasoningParser};
use serde_json::{json, Value};
use tokio::sync::{mpsc, mpsc::UnboundedSender};
use tool_parser::{ParserFactory as ToolParserFactory, StreamingParseResult, ToolParser};
use tracing::{debug, error, warn};

use crate::{
    observability::metrics::{metrics_labels, Metrics, StreamingMetricsParams},
    routers::grpc::{
        common::{response_formatting::CompletionTokenTracker, responses::build_sse_response},
        context,
        proto_wrapper::{ProtoResponseVariant, ProtoStream},
        utils,
    },
};

/// Shared streaming processor for both single and dual dispatch modes
#[derive(Clone)]
pub(crate) struct StreamingProcessor {
    tool_parser_factory: ToolParserFactory,
    reasoning_parser_factory: ReasoningParserFactory,
    configured_tool_parser: Option<String>,
    configured_reasoning_parser: Option<String>,
    backend_type: &'static str,
}

/// Context for generate endpoint streaming - groups config params to reduce function arguments
struct GenerateStreamContext {
    request_id: String,
    weight_version: String,
    return_logprob: bool,
    backend_type: &'static str,
    model: String,
}

impl StreamingProcessor {
    pub fn new(
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
        backend_type: &'static str,
    ) -> Self {
        Self {
            tool_parser_factory,
            reasoning_parser_factory,
            configured_tool_parser,
            configured_reasoning_parser,
            backend_type,
        }
    }

    /// Process streaming chat response and return SSE response
    ///
    /// This is the high-level entry point for streaming responses, handling:
    /// - Channel creation
    /// - Background task spawning
    /// - SSE response building
    ///
    /// Note: Caller should attach load guards to the returned response using
    /// `WorkerLoadGuard::attach_to_response()` for proper RAII lifecycle management.
    pub fn process_streaming_response(
        self: Arc<Self>,
        execution_result: context::ExecutionResult,
        chat_request: Arc<ChatCompletionRequest>,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
    ) -> Response {
        use bytes::Bytes;
        use tokio::sync::mpsc;

        let effective_skip_special_tokens = if chat_request.tools.is_some() {
            match &chat_request.tool_choice {
                Some(ToolChoice::Value(ToolChoiceValue::None)) => chat_request.skip_special_tokens,
                Some(_) => false,
                None => false,
            }
        } else {
            chat_request.skip_special_tokens
        };

        let stop_params = (
            chat_request.stop.clone(),
            chat_request.stop_token_ids.clone(),
            effective_skip_special_tokens,
            chat_request.no_stop_trim,
        );

        // Create SSE channel
        let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

        // Spawn background task based on execution mode
        match execution_result {
            context::ExecutionResult::Single { stream } => {
                let processor = self.clone();
                let dispatch_clone = dispatch.clone();
                let tokenizer_clone = tokenizer.clone();
                #[expect(
                    clippy::disallowed_methods,
                    reason = "streaming task is fire-and-forget; client disconnect terminates it"
                )]
                tokio::spawn(async move {
                    let result = processor
                        .process_streaming_chunks(
                            stream,
                            dispatch_clone,
                            tokenizer_clone,
                            stop_params,
                            chat_request,
                            &tx,
                        )
                        .await;

                    if let Err(e) = result {
                        utils::send_error_sse(&tx, &e, "internal_error");
                    }

                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                });
            }
            context::ExecutionResult::Dual { prefill, decode } => {
                let processor = self.clone();
                let tokenizer_clone = tokenizer.clone();
                #[expect(
                    clippy::disallowed_methods,
                    reason = "streaming task is fire-and-forget; client disconnect terminates it"
                )]
                tokio::spawn(async move {
                    let result = processor
                        .process_dual_streaming_chunks(
                            prefill,
                            *decode,
                            dispatch,
                            tokenizer_clone,
                            stop_params,
                            chat_request,
                            &tx,
                        )
                        .await;

                    if let Err(e) = result {
                        utils::send_error_sse(&tx, &e, "internal_error");
                    }

                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                });
            }
            context::ExecutionResult::Embedding { .. } => {
                utils::send_error_sse(
                    &tx,
                    "Embeddings not supported in streaming mode",
                    "invalid_request_error",
                );
                let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
            }
        }

        // Return SSE response
        build_sse_response(rx)
    }

    /// Process streaming chunks from a single stream (Regular mode)
    pub async fn process_streaming_chunks(
        &self,
        mut grpc_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool),
        original_request: Arc<ChatCompletionRequest>,
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        // Metrics timing
        let start_time = Instant::now();
        let mut first_token_time: Option<Instant> = None;

        // Extract request parameters
        let separate_reasoning = original_request.separate_reasoning;
        let tool_choice = &original_request.tool_choice;
        let tools = &original_request.tools;
        let history_tool_calls_count = utils::get_history_tool_calls_count(&original_request);
        let stream_options = &original_request.stream_options;

        // Phase 1: Initialize state tracking (per-index for n>1 support)
        let mut is_firsts: HashMap<u32, bool> = HashMap::new();
        let mut stream_buffers: HashMap<u32, String> = HashMap::new();
        let mut finish_reasons: HashMap<u32, String> = HashMap::new();
        let mut matched_stops: HashMap<u32, Option<Value>> = HashMap::new();
        let mut prompt_tokens: HashMap<u32, u32> = HashMap::new();
        let mut completion_tokens = CompletionTokenTracker::new();
        let mut streamed_output_tokens: HashMap<u32, usize> = HashMap::new();
        let mut complete_output_ids: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut cached_tokens: HashMap<u32, u32> = HashMap::new();

        // Parser state (lazy initialization per index)
        type PooledReasoningParser = Arc<tokio::sync::Mutex<Box<dyn ReasoningParser>>>;
        let mut reasoning_parsers: HashMap<u32, PooledReasoningParser> = HashMap::new();

        type PooledToolParser = Arc<tokio::sync::Mutex<Box<dyn ToolParser>>>;
        let mut tool_parsers: HashMap<u32, PooledToolParser> = HashMap::new();
        let mut has_tool_calls: HashMap<u32, bool> = HashMap::new();
        let mut emitted_tool_call_indices: HashMap<u32, HashSet<usize>> = HashMap::new();
        let mut repaired_specific_function_args: HashMap<u32, String> = HashMap::new();

        // Per-index stop decoders (each index needs its own state for n>1 support)
        let mut stop_decoders: HashMap<u32, StopSequenceDecoder> = HashMap::new();

        // Reusable SSE formatting buffer to avoid allocations per chunk
        let mut sse_buffer = Vec::with_capacity(512);

        // Use dispatch metadata for consistent response fields
        let request_id = &dispatch.request_id;
        let model = &dispatch.model;
        let created = dispatch.created;
        let system_fingerprint = dispatch.weight_version.as_deref();

        // Check parser availability once upfront (log warning only once per request)
        let model_reasoning_parser_available = utils::check_reasoning_parser_availability(
            &self.reasoning_parser_factory,
            self.configured_reasoning_parser.as_deref(),
            model,
        );
        let reasoning_parser_available = separate_reasoning && model_reasoning_parser_available;
        let strip_reasoning_in_tool_stream =
            !separate_reasoning && tools.is_some() && model_reasoning_parser_available;

        // Check if JSON schema constraint was used (specific function or required mode)
        let used_json_schema = match tool_choice {
            Some(ToolChoice::Function { .. }) => true,
            Some(ToolChoice::Value(ToolChoiceValue::Required)) => true,
            Some(ToolChoice::AllowedTools { mode, .. }) => mode == "required",
            _ => false,
        };
        let defer_empty_auto_tool_calls =
            Self::should_defer_empty_auto_tool_calls(&original_request);

        // Check if this is the specific function case (LLM generates parameters only, no name field)
        let is_specific_function = matches!(tool_choice, Some(ToolChoice::Function { .. }));

        let tool_parser_available = tools.is_some()
            && utils::check_tool_parser_availability(
                &self.tool_parser_factory,
                self.configured_tool_parser.as_deref(),
                model,
            );

        if separate_reasoning && !reasoning_parser_available {
            debug!(
                "No reasoning parser found for model '{}', skipping reasoning parsing",
                model
            );
        }

        if tools.is_some() && !tool_parser_available {
            debug!(
                "No tool parser found for model '{}', skipping tool call parsing",
                model
            );
        }

        // Phase 2: Main streaming loop
        while let Some(response) = grpc_stream.next().await {
            let gen_response = response.map_err(|e| format!("Stream error: {}", e.message()))?;

            match gen_response.into_response() {
                ProtoResponseVariant::Chunk(chunk) => {
                    // Track TTFT immediately on first chunk received from backend
                    if first_token_time.is_none() {
                        first_token_time = Some(Instant::now());
                    }

                    let index = chunk.index();
                    *streamed_output_tokens.entry(index).or_insert(0) += chunk.token_ids().len();

                    completion_tokens.record_chunk(&chunk);

                    // Get or create stop decoder for this index
                    let stop_decoder = stop_decoders.entry(index).or_insert_with(|| {
                        let (ref stop, ref stop_token_ids, skip_special_tokens, no_stop_trim) =
                            stop_params;
                        utils::create_stop_decoder(
                            &tokenizer,
                            stop.as_ref(),
                            stop_token_ids.as_ref(),
                            skip_special_tokens,
                            no_stop_trim,
                        )
                    });

                    // Process tokens through stop decoder
                    let (chunk_text, _should_stop) =
                        Self::process_chunk_tokens(stop_decoder, chunk.token_ids());

                    if chunk_text.is_empty() {
                        continue;
                    }

                    // Process logprobs if present
                    let choice_logprobs = chunk.output_logprobs().map(|ref proto_logprobs| {
                        utils::convert_proto_to_openai_logprobs(proto_logprobs, &tokenizer)
                    });

                    // Initialize stream buffer if first time
                    let stream_buffer = stream_buffers.entry(index).or_default();

                    // Send first chunk with role
                    if is_firsts.get(&index).copied().unwrap_or(true) {
                        let first_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                            .created(created)
                            .add_choice_role(index, "assistant")
                            .maybe_system_fingerprint(system_fingerprint)
                            .build();
                        Self::format_sse_chunk_into(&mut sse_buffer, &first_chunk);
                        tx.send(Ok(Bytes::from(sse_buffer.clone())))
                            .map_err(|_| "Failed to send first chunk".to_string())?;
                        is_firsts.insert(index, false);
                    }

                    // Calculate delta
                    let mut delta = chunk_text;
                    stream_buffer.push_str(&delta);

                    // Reasoning content handling
                    let in_reasoning = if reasoning_parser_available
                        || strip_reasoning_in_tool_stream
                    {
                        let (normal_text, reasoning_chunk, in_reasoning) = self
                            .process_reasoning_stream(
                                &delta,
                                index,
                                &mut reasoning_parsers,
                                request_id,
                                model,
                                created,
                                system_fingerprint,
                            )
                            .await;
                        if separate_reasoning {
                            if let Some(chunk) = reasoning_chunk {
                                Self::format_sse_chunk_into(&mut sse_buffer, &chunk);
                                tx.send(Ok(Bytes::from(sse_buffer.clone())))
                                    .map_err(|_| "Failed to send reasoning chunk".to_string())?;
                            }
                        }
                        delta = normal_text;
                        in_reasoning
                    } else {
                        false
                    };

                    if delta.is_empty() && in_reasoning {
                        continue;
                    }

                    // Tool call handling
                    let tool_choice_enabled =
                        !matches!(tool_choice, Some(ToolChoice::Value(ToolChoiceValue::None)));

                    if let Some(tools_ref) = tools.as_ref() {
                        if !in_reasoning
                            && tool_choice_enabled
                            && (tool_parser_available || used_json_schema)
                        {
                            let tool_chunks = if is_specific_function {
                                // Handle specific function case - emit tool call deltas with arguments
                                Self::process_specific_function_stream(
                                    &delta,
                                    index,
                                    &mut has_tool_calls,
                                    &mut emitted_tool_call_indices,
                                    tool_choice.as_ref(),
                                    request_id,
                                    model,
                                    created,
                                    system_fingerprint,
                                    history_tool_calls_count,
                                )
                            } else {
                                // Use incremental parser for regular/required modes
                                self.process_tool_calls_stream(
                                    &delta,
                                    index,
                                    &mut tool_parsers,
                                    &mut has_tool_calls,
                                    &mut emitted_tool_call_indices,
                                    tools_ref,
                                    request_id,
                                    model,
                                    created,
                                    system_fingerprint,
                                    history_tool_calls_count,
                                    used_json_schema,
                                    defer_empty_auto_tool_calls,
                                )
                                .await
                            };

                            for chunk in tool_chunks {
                                Self::format_sse_chunk_into(&mut sse_buffer, &chunk);
                                tx.send(Ok(Bytes::from(sse_buffer.clone())))
                                    .map_err(|_| "Failed to send tool call chunk".to_string())?;
                            }

                            // Always skip regular content when tool parsing is active
                            // Parser either emitted chunks or buffered content
                            continue;
                        }
                    }

                    // Regular content emission
                    if !delta.is_empty() {
                        let content_chunk =
                            ChatCompletionStreamResponse::builder(request_id, model)
                                .created(created)
                                .add_choice_content_with_logprobs(
                                    index,
                                    "assistant",
                                    delta,
                                    choice_logprobs,
                                )
                                .maybe_system_fingerprint(system_fingerprint)
                                .build();
                        Self::format_sse_chunk_into(&mut sse_buffer, &content_chunk);
                        tx.send(Ok(Bytes::from(sse_buffer.clone())))
                            .map_err(|_| "Failed to send content chunk".to_string())?;
                    }
                }
                ProtoResponseVariant::Complete(complete) => {
                    let index = complete.index();

                    // Some backends only surface the final token in Complete.output_ids.
                    // Process the unseen suffix before flushing any held stop-decoder text.
                    if let Some(decoder) = stop_decoders.get_mut(&index) {
                        let unseen_output_ids = Self::unseen_completion_tokens(
                            complete.output_ids(),
                            streamed_output_tokens.get(&index).copied().unwrap_or(0),
                        );

                        let (suffix_text, _should_stop) =
                            Self::process_chunk_tokens(decoder, unseen_output_ids);

                        let flushed_text = match decoder.flush() {
                            SequenceDecoderOutput::Text(text)
                            | SequenceDecoderOutput::StoppedWithText(text) => text,
                            SequenceDecoderOutput::Stopped | SequenceDecoderOutput::Held => {
                                String::new()
                            }
                        };

                        let final_text = format!("{suffix_text}{flushed_text}");

                        if !final_text.is_empty() {
                            let stream_buffer = stream_buffers.entry(index).or_default();
                            stream_buffer.push_str(&final_text);
                            let mut flushed_delta = final_text;
                            let mut emitted_as_tool_chunk = false;
                            let mut in_reasoning = false;
                            let tool_choice_enabled = !matches!(
                                tool_choice,
                                Some(ToolChoice::Value(ToolChoiceValue::None))
                            );

                            if reasoning_parser_available || strip_reasoning_in_tool_stream {
                                let (normal_text, reasoning_chunk, parser_in_reasoning) = self
                                    .process_reasoning_stream(
                                        &flushed_delta,
                                        index,
                                        &mut reasoning_parsers,
                                        request_id,
                                        model,
                                        created,
                                        system_fingerprint,
                                    )
                                    .await;
                                if separate_reasoning {
                                    if let Some(chunk) = reasoning_chunk {
                                        let sse_chunk =
                                            serde_json::to_string(&chunk).map_err(|e| {
                                                format!(
                                                    "Failed to serialize flushed reasoning chunk: {e}"
                                                )
                                            })?;
                                        tx.send(Ok(Bytes::from(format!("data: {sse_chunk}\n\n"))))
                                            .map_err(|_| {
                                                "Failed to send flushed reasoning chunk".to_string()
                                            })?;
                                    }
                                }
                                flushed_delta = normal_text;
                                in_reasoning = parser_in_reasoning;
                            }

                            if let Some(tools_ref) = tools.as_ref() {
                                if !in_reasoning
                                    && tool_choice_enabled
                                    && (tool_parser_available || used_json_schema)
                                {
                                    let tool_chunks = if is_specific_function {
                                        Self::process_specific_function_stream(
                                            &flushed_delta,
                                            index,
                                            &mut has_tool_calls,
                                            &mut emitted_tool_call_indices,
                                            tool_choice.as_ref(),
                                            request_id,
                                            model,
                                            created,
                                            system_fingerprint,
                                            history_tool_calls_count,
                                        )
                                    } else {
                                        self.process_tool_calls_stream(
                                            &flushed_delta,
                                            index,
                                            &mut tool_parsers,
                                            &mut has_tool_calls,
                                            &mut emitted_tool_call_indices,
                                            tools_ref,
                                            request_id,
                                            model,
                                            created,
                                            system_fingerprint,
                                            history_tool_calls_count,
                                            used_json_schema,
                                            defer_empty_auto_tool_calls,
                                        )
                                        .await
                                    };
                                    let has_tool_chunks = !tool_chunks.is_empty();

                                    for chunk in tool_chunks {
                                        let sse_chunk =
                                            serde_json::to_string(&chunk).map_err(|e| {
                                                format!(
                                                    "Failed to serialize flushed tool chunk: {e}"
                                                )
                                            })?;
                                        tx.send(Ok(Bytes::from(format!("data: {sse_chunk}\n\n"))))
                                            .map_err(|_| {
                                                "Failed to send flushed tool chunk".to_string()
                                            })?;
                                    }

                                    emitted_as_tool_chunk = has_tool_chunks;
                                }
                            }

                            if !in_reasoning && !emitted_as_tool_chunk && !flushed_delta.is_empty()
                            {
                                let content_chunk =
                                    ChatCompletionStreamResponse::builder(request_id, model)
                                        .created(created)
                                        .add_choice_content(index, "assistant", flushed_delta)
                                        .maybe_system_fingerprint(system_fingerprint)
                                        .build();

                                let sse_chunk =
                                    serde_json::to_string(&content_chunk).map_err(|e| {
                                        format!("Failed to serialize content chunk: {e}")
                                    })?;
                                tx.send(Ok(Bytes::from(format!("data: {sse_chunk}\n\n"))))
                                    .map_err(|_| "Failed to send flushed content".to_string())?;
                            }
                        }
                    }

                    let authoritative_suffix = {
                        let streamed_text =
                            stream_buffers.get(&index).map(String::as_str).unwrap_or("");
                        Self::missing_completion_suffix(
                            &tokenizer,
                            &stop_params,
                            complete.output_ids(),
                            streamed_text,
                        )?
                    };

                    // Reconcile against a fresh full decode of Complete.output_ids().
                    // This covers cases where incremental stop-decoder state drops a trailing suffix
                    // that still appears when the same ids are decoded authoritatively end-to-end.
                    if let Some(authoritative_suffix) = authoritative_suffix {
                        let stream_buffer = stream_buffers.entry(index).or_default();
                        stream_buffer.push_str(&authoritative_suffix);
                        let mut flushed_delta = authoritative_suffix;
                        let mut emitted_as_tool_chunk = false;
                        let mut in_reasoning = false;
                        let tool_choice_enabled =
                            !matches!(tool_choice, Some(ToolChoice::Value(ToolChoiceValue::None)));

                        if reasoning_parser_available || strip_reasoning_in_tool_stream {
                            let (normal_text, reasoning_chunk, parser_in_reasoning) = self
                                .process_reasoning_stream(
                                    &flushed_delta,
                                    index,
                                    &mut reasoning_parsers,
                                    request_id,
                                    model,
                                    created,
                                    system_fingerprint,
                                )
                                .await;
                            if separate_reasoning {
                                if let Some(chunk) = reasoning_chunk {
                                    let sse_chunk = serde_json::to_string(&chunk).map_err(|e| {
                                        format!(
                                            "Failed to serialize authoritative reasoning chunk: {e}"
                                        )
                                    })?;
                                    tx.send(Ok(Bytes::from(format!("data: {sse_chunk}\n\n"))))
                                        .map_err(|_| {
                                            "Failed to send authoritative reasoning chunk"
                                                .to_string()
                                        })?;
                                }
                            }
                            flushed_delta = normal_text;
                            in_reasoning = parser_in_reasoning;
                        }

                        if let Some(tools_ref) = tools.as_ref() {
                            if !in_reasoning
                                && tool_choice_enabled
                                && (tool_parser_available || used_json_schema)
                            {
                                let tool_chunks = if is_specific_function {
                                    Self::process_specific_function_stream(
                                        &flushed_delta,
                                        index,
                                        &mut has_tool_calls,
                                        &mut emitted_tool_call_indices,
                                        tool_choice.as_ref(),
                                        request_id,
                                        model,
                                        created,
                                        system_fingerprint,
                                        history_tool_calls_count,
                                    )
                                } else {
                                    self.process_tool_calls_stream(
                                        &flushed_delta,
                                        index,
                                        &mut tool_parsers,
                                        &mut has_tool_calls,
                                        &mut emitted_tool_call_indices,
                                        tools_ref,
                                        request_id,
                                        model,
                                        created,
                                        system_fingerprint,
                                        history_tool_calls_count,
                                        used_json_schema,
                                        defer_empty_auto_tool_calls,
                                    )
                                    .await
                                };
                                let has_tool_chunks = !tool_chunks.is_empty();

                                for chunk in tool_chunks {
                                    let sse_chunk = serde_json::to_string(&chunk).map_err(|e| {
                                        format!("Failed to serialize authoritative tool chunk: {e}")
                                    })?;
                                    tx.send(Ok(Bytes::from(format!("data: {sse_chunk}\n\n"))))
                                        .map_err(|_| {
                                            "Failed to send authoritative tool chunk".to_string()
                                        })?;
                                }

                                emitted_as_tool_chunk = has_tool_chunks;
                            }
                        }

                        if !in_reasoning && !emitted_as_tool_chunk && !flushed_delta.is_empty() {
                            let content_chunk =
                                ChatCompletionStreamResponse::builder(request_id, model)
                                    .created(created)
                                    .add_choice_content(index, "assistant", flushed_delta)
                                    .maybe_system_fingerprint(system_fingerprint)
                                    .build();

                            let sse_chunk = serde_json::to_string(&content_chunk).map_err(|e| {
                                format!("Failed to serialize authoritative content chunk: {e}")
                            })?;
                            tx.send(Ok(Bytes::from(format!("data: {sse_chunk}\n\n"))))
                                .map_err(|_| {
                                    "Failed to send authoritative content chunk".to_string()
                                })?;
                        }
                    }

                    // Keep the authoritative output ids so tool fallback parsing can
                    // reconstruct the full completion even when incremental text chunks
                    // missed special tool-call tokens.
                    complete_output_ids.insert(index, complete.output_ids().to_vec());

                    // Store metadata
                    prompt_tokens.insert(index, complete.prompt_tokens());

                    completion_tokens.record_complete(&complete);

                    cached_tokens.insert(index, complete.cached_tokens());
                    finish_reasons.insert(index, complete.finish_reason().to_string());

                    matched_stops.insert(index, complete.matched_stop_json());

                    // Don't break - continue reading all Complete messages for n>1
                }
                ProtoResponseVariant::Error(error) => {
                    return Err(error.message().to_string());
                }
                ProtoResponseVariant::None => continue,
            }
        }

        // Phase 3: Check unstreamed tool args
        for (index, parser) in &tool_parsers {
            let parser_guard = parser.lock().await;
            if let Some(unstreamed_items) = parser_guard.get_unstreamed_tool_args() {
                for tool_call_item in unstreamed_items {
                    if defer_empty_auto_tool_calls {
                        if let Some(name) = tool_call_item.name.as_deref() {
                            if Self::is_empty_tool_args(&tool_call_item.parameters)
                                || Self::tool_args_missing_required_fields(
                                    tools.as_deref().unwrap_or(&[]),
                                    name,
                                    &tool_call_item.parameters,
                                )
                            {
                                continue;
                            }
                        }
                    }

                    let tool_call_delta = ToolCallDelta {
                        index: tool_call_item.tool_index as u32,
                        id: None,
                        tool_type: None,
                        function: Some(FunctionCallDelta {
                            name: None,
                            arguments: if tool_call_item.parameters.is_empty() {
                                None
                            } else {
                                Some(tool_call_item.parameters)
                            },
                        }),
                    };

                    let tool_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                        .created(created)
                        .add_choice_tool_call_delta(*index, tool_call_delta)
                        .maybe_system_fingerprint(system_fingerprint)
                        .build();

                    let sse_chunk = serde_json::to_string(&tool_chunk)
                        .map_err(|e| format!("Failed to serialize tool chunk: {e}"))?;
                    tx.send(Ok(Bytes::from(format!("data: {sse_chunk}\n\n"))))
                        .map_err(|_| "Failed to send unstreamed tool args".to_string())?;
                }
            }

            let authoritative_full_text = complete_output_ids
                .get(index)
                .and_then(|output_ids| {
                    Self::decode_complete_output_text(&tokenizer, &stop_params, output_ids)
                        .ok()
                });

            if let Some(full_text) = authoritative_full_text
                .as_deref()
                .or_else(|| stream_buffers.get(index).map(String::as_str))
            {
                if let Ok((_normal_text, parsed_tool_calls)) =
                    parser_guard.parse_complete(full_text).await
                {
                    if parsed_tool_calls.is_empty()
                        && !has_tool_calls.get(index).copied().unwrap_or(false)
                    {
                        let preview: String = full_text.chars().take(800).collect();
                        warn!(
                            request_id,
                            index = *index,
                            full_text_len = full_text.len(),
                            full_text_preview = %preview,
                            "streaming fallback parse_complete found no tool calls"
                        );
                    }

                    for (tool_index, tool_call) in parsed_tool_calls.into_iter().enumerate() {
                        let already_emitted = emitted_tool_call_indices
                            .get(index)
                            .is_some_and(|indices| indices.contains(&tool_index));
                        if already_emitted {
                            continue;
                        }

                        if defer_empty_auto_tool_calls {
                            let should_defer =
                                Self::is_empty_tool_args(&tool_call.function.arguments)
                                    || Self::tool_args_missing_required_fields(
                                        tools.as_deref().unwrap_or(&[]),
                                        &tool_call.function.name,
                                        &tool_call.function.arguments,
                                    );
                            if should_defer {
                                continue;
                            }
                        }

                        has_tool_calls.insert(*index, true);
                        emitted_tool_call_indices
                            .entry(*index)
                            .or_default()
                            .insert(tool_index);

                        let tool_call_delta = ToolCallDelta {
                            index: tool_index as u32,
                            id: Some(utils::generate_tool_call_id(
                                model,
                                &tool_call.function.name,
                                tool_index,
                                history_tool_calls_count,
                            )),
                            tool_type: Some("function".to_string()),
                            function: Some(FunctionCallDelta {
                                name: Some(tool_call.function.name),
                                arguments: Some(tool_call.function.arguments),
                            }),
                        };

                        let tool_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                            .created(created)
                            .add_choice_tool_call_delta(*index, tool_call_delta)
                            .maybe_system_fingerprint(system_fingerprint)
                            .build();

                        let sse_chunk = serde_json::to_string(&tool_chunk)
                            .map_err(|e| format!("Failed to serialize fallback tool chunk: {e}"))?;
                        tx.send(Ok(Bytes::from(format!("data: {sse_chunk}\n\n"))))
                            .map_err(|_| "Failed to send fallback tool chunk".to_string())?;
                    }
                }
            }
        }

        match utils::deterministic_auto_tool_repair(&original_request).await {
            Ok(Some(repaired_tool_calls)) => {
                let repair_index = finish_reasons
                    .keys()
                    .copied()
                    .next()
                    .or_else(|| stream_buffers.keys().copied().next())
                    .unwrap_or(0);

                if is_specific_function {
                    if let Some(arguments) = repaired_tool_calls
                        .into_iter()
                        .next()
                        .and_then(|tool_call| tool_call.function.arguments)
                    {
                        repaired_specific_function_args.insert(repair_index, arguments);
                    }
                } else {
                    let already_emitted_count = emitted_tool_call_indices
                        .get(&repair_index)
                        .map_or(0, HashSet::len);

                    for (tool_index, tool_call) in repaired_tool_calls
                        .into_iter()
                        .enumerate()
                        .skip(already_emitted_count)
                    {
                        has_tool_calls.insert(repair_index, true);
                        emitted_tool_call_indices
                            .entry(repair_index)
                            .or_default()
                            .insert(tool_index);

                        let tool_call_delta = ToolCallDelta {
                            index: tool_index as u32,
                            id: Some(utils::generate_tool_call_id(
                                model,
                                &tool_call.function.name,
                                tool_index,
                                history_tool_calls_count,
                            )),
                            tool_type: Some("function".to_string()),
                            function: Some(FunctionCallDelta {
                                name: Some(tool_call.function.name),
                                arguments: tool_call.function.arguments,
                            }),
                        };

                        let tool_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                            .created(created)
                            .add_choice_tool_call_delta(repair_index, tool_call_delta)
                            .maybe_system_fingerprint(system_fingerprint)
                            .build();

                        let sse_chunk = serde_json::to_string(&tool_chunk).map_err(|e| {
                            format!("Failed to serialize deterministic repair tool chunk: {e}")
                        })?;
                        tx.send(Ok(Bytes::from(format!("data: {sse_chunk}\n\n"))))
                            .map_err(|_| {
                                "Failed to send deterministic repair tool chunk".to_string()
                            })?;
                    }
                }
            }
            Ok(None) => {}
            Err(e) => {
                warn!(request_id, "Deterministic auto tool repair failed: {}", e);
            }
        }

        if is_specific_function {
            for (index, full_text) in &mut stream_buffers {
                if !has_tool_calls.get(index).copied().unwrap_or(false) {
                    continue;
                }

                let Some(ToolChoice::Function { function, .. }) = tool_choice.as_ref() else {
                    continue;
                };

                let mut final_args = repaired_specific_function_args
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| full_text.clone());

                if let Some(repair_suffix) = Self::json_repair_suffix(&final_args) {
                    final_args.push_str(&repair_suffix);
                }

                let mut repaired_tool_calls = Some(vec![ToolCall {
                    id: String::new(),
                    tool_type: "function".to_string(),
                    function: FunctionCallResponse {
                        name: function.name.clone(),
                        arguments: Some(final_args),
                    },
                }]);
                let mut ignored_content = String::new();
                utils::repair_tool_calls_and_content(
                    &original_request,
                    &mut repaired_tool_calls,
                    &mut ignored_content,
                );

                let Some(final_args) = repaired_tool_calls
                    .as_ref()
                    .and_then(|tool_calls| tool_calls.first())
                    .and_then(|tool_call| tool_call.function.arguments.clone())
                else {
                    continue;
                };

                *full_text = final_args.clone();

                let tool_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                    .created(created)
                    .add_choice_tool_args(*index, final_args)
                    .maybe_system_fingerprint(system_fingerprint)
                    .build();

                let sse_chunk = serde_json::to_string(&tool_chunk).map_err(|e| {
                    format!("Failed to serialize specific-function repair chunk: {e}")
                })?;
                tx.send(Ok(Bytes::from(format!("data: {sse_chunk}\n\n"))))
                    .map_err(|_| "Failed to send specific-function repair chunk".to_string())?;
            }
        }

        if Self::is_structured_json_response(&original_request) {
            for (index, full_text) in &mut stream_buffers {
                if has_tool_calls.get(index).copied().unwrap_or(false) {
                    continue;
                }

                let Some(repair_suffix) = Self::json_repair_suffix(full_text) else {
                    continue;
                };

                full_text.push_str(&repair_suffix);

                let content_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                    .created(created)
                    .add_choice_content(*index, "assistant", repair_suffix)
                    .maybe_system_fingerprint(system_fingerprint)
                    .build();

                let sse_chunk = serde_json::to_string(&content_chunk)
                    .map_err(|e| format!("Failed to serialize JSON repair chunk: {e}"))?;
                tx.send(Ok(Bytes::from(format!("data: {sse_chunk}\n\n"))))
                    .map_err(|_| "Failed to send JSON repair chunk".to_string())?;
            }
        }

        // Phase 4: Finish reason chunks
        for (index, finish_reason) in &finish_reasons {
            let final_finish_reason =
                if has_tool_calls.get(index).copied().unwrap_or(false) && finish_reason == "stop" {
                    "tool_calls".to_string()
                } else {
                    finish_reason.clone()
                };

            let matched_stop_value = matched_stops.get(index).and_then(|v| v.clone());

            let finish_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                .created(created)
                .add_choice_finish_reason(*index, final_finish_reason, matched_stop_value)
                .maybe_system_fingerprint(system_fingerprint)
                .build();

            let sse_chunk = serde_json::to_string(&finish_chunk)
                .map_err(|e| format!("Failed to serialize finish chunk: {e}"))?;
            tx.send(Ok(Bytes::from(format!("data: {sse_chunk}\n\n"))))
                .map_err(|_| "Failed to send finish chunk".to_string())?;
        }

        // Phase 5: Usage chunk
        if let Some(stream_opts) = stream_options {
            if stream_opts.include_usage.unwrap_or(false) {
                let total_prompt: u32 = prompt_tokens.values().sum();
                let total_completion: u32 = completion_tokens.total();
                let total_cached: u32 = cached_tokens.values().sum();

                let usage_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                    .created(created)
                    .usage(
                        Usage::from_counts(total_prompt, total_completion)
                            .with_cached_tokens(total_cached),
                    )
                    .maybe_system_fingerprint(system_fingerprint)
                    .build();

                let sse_chunk = serde_json::to_string(&usage_chunk)
                    .map_err(|e| format!("Failed to serialize usage chunk: {e}"))?;
                tx.send(Ok(Bytes::from(format!("data: {sse_chunk}\n\n"))))
                    .map_err(|_| "Failed to send usage chunk".to_string())?;
            }
        }

        // Mark stream as completed successfully to prevent abort on drop
        grpc_stream.mark_completed();

        // Record streaming metrics
        let total_prompt: u32 = prompt_tokens.values().sum();
        let total_completion: u32 = completion_tokens.total();
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: self.backend_type,
            model_id: model,
            endpoint: metrics_labels::ENDPOINT_CHAT,
            ttft: first_token_time.map(|t| t.duration_since(start_time)),
            generation_duration: start_time.elapsed(),
            input_tokens: Some(total_prompt as u64),
            output_tokens: total_completion as u64,
        });

        Ok(())
    }

    /// Process dual streaming chunks (prefill + decode) - PD mode
    #[expect(clippy::too_many_arguments)]
    pub async fn process_dual_streaming_chunks(
        &self,
        mut prefill_stream: ProtoStream,
        decode_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool),
        original_request: Arc<ChatCompletionRequest>,
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        // Phase 1.5: Collect input_logprobs from prefill stream if requested
        if original_request.logprobs {
            while let Some(response) = prefill_stream.next().await {
                let gen_response =
                    response.map_err(|e| format!("Prefill stream error: {}", e.message()))?;
                match gen_response.into_response() {
                    ProtoResponseVariant::Complete(_complete) => {
                        // Input logprobs collected but not yet used in streaming
                        // (OpenAI spec doesn't require prompt logprobs in streaming responses)
                        break;
                    }
                    ProtoResponseVariant::Error(error) => {
                        return Err(format!("Prefill error: {}", error.message()));
                    }
                    _ => continue,
                }
            }
        }

        // Phase 2-5: Process decode stream (same as single mode)
        // Note: decode_stream will be marked completed inside process_streaming_chunks
        let result = self
            .process_streaming_chunks(
                decode_stream,
                dispatch,
                tokenizer,
                stop_params,
                original_request,
                tx,
            )
            .await;

        // Mark prefill stream as completed AFTER decode completes successfully
        // This ensures that if client disconnects during decode, BOTH streams send abort
        if result.is_ok() {
            prefill_stream.mark_completed();
        }

        result
    }

    /// Process streaming generate response and return SSE response
    ///
    /// Simpler than chat - no tool/reasoning parsing, just text accumulation
    ///
    /// Note: Caller should attach load guards to the returned response using
    /// `WorkerLoadGuard::attach_to_response()` for proper RAII lifecycle management.
    pub fn process_streaming_generate(
        self: Arc<Self>,
        execution_result: context::ExecutionResult,
        generate_request: Arc<GenerateRequest>,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
    ) -> Response {
        // Create SSE channel
        let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

        // Build context once, clone for spawned task
        let ctx = GenerateStreamContext {
            request_id: dispatch.request_id.clone(),
            weight_version: dispatch
                .weight_version
                .clone()
                .unwrap_or_else(|| "default".to_string()),
            return_logprob: generate_request.return_logprob.unwrap_or(false),
            backend_type: self.backend_type,
            model: dispatch.model.clone(),
        };

        // Spawn background task based on execution mode
        match execution_result {
            context::ExecutionResult::Single { stream } => {
                let tokenizer = tokenizer.clone();
                #[expect(
                    clippy::disallowed_methods,
                    reason = "streaming task is fire-and-forget; client disconnect terminates it"
                )]
                tokio::spawn(async move {
                    let result =
                        Self::process_generate_streaming(tokenizer, stream, ctx, &tx).await;

                    if let Err(e) = result {
                        utils::send_error_sse(&tx, &e, "internal_error");
                    }

                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                });
            }
            context::ExecutionResult::Dual { prefill, decode } => {
                // For PD mode, need to handle prefill stream for input_logprobs
                let tokenizer = tokenizer.clone();
                #[expect(
                    clippy::disallowed_methods,
                    reason = "streaming task is fire-and-forget; client disconnect terminates it"
                )]
                tokio::spawn(async move {
                    let result = Self::process_generate_streaming_dual(
                        tokenizer, prefill, *decode, ctx, &tx,
                    )
                    .await;

                    if let Err(e) = result {
                        utils::send_error_sse(&tx, &e, "internal_error");
                    }

                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                });
            }
            context::ExecutionResult::Embedding { .. } => {
                utils::send_error_sse(
                    &tx,
                    "Embeddings not supported in streaming generate",
                    "invalid_request_error",
                );
                let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
            }
        }

        // Return SSE response
        build_sse_response(rx)
    }

    /// Process streaming chunks for generate endpoint (no tool/reasoning parsing)
    /// TODO: add streaming logprob support
    async fn process_generate_streaming(
        tokenizer: Arc<dyn Tokenizer>,
        mut stream: ProtoStream,
        ctx: GenerateStreamContext,
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        let start_time = Instant::now();
        let mut first_token_time: Option<Instant> = None;

        // Track state per index for n>1 case
        let mut accumulated_texts: HashMap<u32, String> = HashMap::new();
        let mut completion_tokens_map: HashMap<u32, u32> = HashMap::new();

        while let Some(response) = stream.next().await {
            let gen_response = response.map_err(|e| format!("Stream error: {}", e.message()))?;

            match gen_response.into_response() {
                ProtoResponseVariant::Chunk(chunk) => {
                    // Track TTFT immediately on first chunk received from backend
                    if first_token_time.is_none() {
                        first_token_time = Some(Instant::now());
                    }

                    let index = chunk.index();

                    // Both backends send delta token_ids, so accumulate for both
                    let completion_tokens = completion_tokens_map.entry(index).or_insert(0);
                    *completion_tokens += chunk.token_ids().len() as u32;
                    let current_completion_tokens = *completion_tokens;

                    // Decode tokens to text (skip_special_tokens=true to handle newlines correctly)
                    let chunk_text = tokenizer
                        .decode(chunk.token_ids(), true)
                        .unwrap_or_default();

                    // Accumulate text for this index
                    let accumulated_text = accumulated_texts.entry(index).or_default();
                    accumulated_text.push_str(&chunk_text);

                    // Generate unique ID per index
                    let index_id = format!("{}-{}", ctx.request_id, index);

                    // Build streaming response chunk (SGLang format)
                    let chunk_response = serde_json::json!({
                        "text": accumulated_text.clone(),
                        "output_ids": chunk.token_ids(),
                        "meta_info": {
                            "id": index_id,
                            "finish_reason": null,
                            "prompt_tokens": chunk.prompt_tokens(),
                            "weight_version": &ctx.weight_version,
                            "completion_tokens": current_completion_tokens,
                            "cached_tokens": chunk.cached_tokens()
                        },
                        "index": index
                    });

                    let sse_data = serde_json::to_string(&chunk_response)
                        .map_err(|e| format!("Failed to serialize generate chunk: {e}"))?;
                    tx.send(Ok(Bytes::from(format!("data: {sse_data}\n\n"))))
                        .map_err(|_| "Failed to send chunk".to_string())?;
                }
                ProtoResponseVariant::Complete(complete) => {
                    let index = complete.index();
                    let accumulated_text =
                        accumulated_texts.get(&index).cloned().unwrap_or_default();
                    let completion_tokens = *completion_tokens_map.get(&index).unwrap_or(&0);
                    let index_id = format!("{}-{}", ctx.request_id, index);
                    let e2e_latency = start_time.elapsed().as_secs_f64();

                    // Send final chunk with finish_reason
                    let finish_response = serde_json::json!({
                        "text": accumulated_text,
                        "output_ids": complete.output_ids()[complete.output_ids().len().saturating_sub(1)..].to_vec(),
                        "meta_info": {
                            "id": index_id,
                            "finish_reason": complete.finish_reason(),
                            "prompt_tokens": complete.prompt_tokens(),
                            "weight_version": &ctx.weight_version,
                            "completion_tokens": completion_tokens,
                            "cached_tokens": complete.cached_tokens(),
                            "e2e_latency": e2e_latency
                        },
                        "index": index
                    });

                    let sse_data = serde_json::to_string(&finish_response)
                        .map_err(|e| format!("Failed to serialize generate finish: {e}"))?;
                    tx.send(Ok(Bytes::from(format!("data: {sse_data}\n\n"))))
                        .map_err(|_| "Failed to send finish chunk".to_string())?;

                    // Continue to process all completions if n>1
                }
                ProtoResponseVariant::Error(error) => {
                    return Err(error.message().to_string());
                }
                ProtoResponseVariant::None => continue,
            }
        }

        // Mark stream as completed successfully to prevent abort on drop
        stream.mark_completed();

        // Record streaming metrics
        let total_completion: u32 = completion_tokens_map.values().sum();
        Self::record_generate_metrics(start_time, first_token_time, total_completion, &ctx);

        Ok(())
    }

    /// Process dual streaming for generate endpoint (PD mode with logprobs support)
    async fn process_generate_streaming_dual(
        tokenizer: Arc<dyn Tokenizer>,
        mut prefill_stream: ProtoStream,
        decode_stream: ProtoStream,
        ctx: GenerateStreamContext,
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        // Collect input_logprobs from prefill stream if requested
        let input_token_logprobs = if ctx.return_logprob {
            let mut input_logprobs = None;
            while let Some(response) = prefill_stream.next().await {
                let gen_response =
                    response.map_err(|e| format!("Prefill stream error: {}", e.message()))?;
                match gen_response.into_response() {
                    ProtoResponseVariant::Complete(complete) => {
                        // Extract input_logprobs from prefill Complete message (convert proto to SGLang format)
                        input_logprobs = complete
                            .input_logprobs()
                            .as_ref()
                            .map(utils::convert_generate_input_logprobs);
                        break;
                    }
                    ProtoResponseVariant::Error(error) => {
                        return Err(format!("Prefill error: {}", error.message()));
                    }
                    _ => continue,
                }
            }
            input_logprobs
        } else {
            None
        };

        // Process decode stream with input_logprobs prepended
        // Note: decode_stream will be marked completed inside the function
        let result = Self::process_generate_streaming_with_input_logprobs(
            tokenizer,
            decode_stream,
            ctx,
            input_token_logprobs,
            tx,
        )
        .await;

        // Mark prefill stream as completed AFTER decode completes successfully
        // This ensures that if client disconnects during decode, BOTH streams send abort
        if result.is_ok() {
            prefill_stream.mark_completed();
        }

        result
    }

    /// Process generate streaming with optional input_logprobs
    async fn process_generate_streaming_with_input_logprobs(
        tokenizer: Arc<dyn Tokenizer>,
        mut stream: ProtoStream,
        ctx: GenerateStreamContext,
        input_token_logprobs: Option<Vec<Vec<Option<f64>>>>,
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        let start_time = Instant::now();
        let mut first_token_time: Option<Instant> = None;

        // Track state per index for n>1 case
        let mut accumulated_texts: HashMap<u32, String> = HashMap::new();
        let mut accumulated_output_logprobs: HashMap<u32, Option<Vec<Vec<Option<f64>>>>> =
            HashMap::new();
        let mut completion_tokens_map: HashMap<u32, u32> = HashMap::new();

        while let Some(response) = stream.next().await {
            let gen_response = response.map_err(|e| format!("Stream error: {}", e.message()))?;

            match gen_response.into_response() {
                ProtoResponseVariant::Chunk(chunk) => {
                    // Track TTFT immediately on first chunk received from backend
                    if first_token_time.is_none() {
                        first_token_time = Some(Instant::now());
                    }

                    let index = chunk.index();

                    // Both backends send delta token_ids, so accumulate for both
                    let completion_tokens = completion_tokens_map.entry(index).or_insert(0);
                    *completion_tokens += chunk.token_ids().len() as u32;
                    let current_completion_tokens = *completion_tokens;

                    // Decode tokens to text
                    let chunk_text = tokenizer
                        .decode(chunk.token_ids(), true)
                        .unwrap_or_default();

                    // Accumulate text for this index
                    let accumulated_text = accumulated_texts.entry(index).or_default();
                    accumulated_text.push_str(&chunk_text);

                    // Handle output logprobs based on backend behavior:
                    // - SGLang sends cumulative logprobs (replace is correct)
                    // - vLLM sends delta logprobs (need to extend/accumulate)
                    if let Some(ref output_logprobs) = chunk.output_logprobs() {
                        let converted = utils::convert_generate_output_logprobs(output_logprobs);
                        if chunk.is_vllm() {
                            // vLLM sends delta - extend existing logprobs
                            if let Some(v) = accumulated_output_logprobs
                                .entry(index)
                                .or_insert_with(|| Some(Vec::new()))
                                .as_mut()
                            {
                                v.extend(converted);
                            }
                        } else {
                            // SGLang sends cumulative - replace
                            accumulated_output_logprobs.insert(index, Some(converted));
                        }
                    }

                    // Generate unique ID per index
                    let index_id = format!("{}-{}", ctx.request_id, index);

                    // Build streaming response chunk with accumulated logprobs
                    let current_output_logprobs = accumulated_output_logprobs
                        .get(&index)
                        .and_then(|o| o.as_ref());

                    let chunk_response = json!({
                        "text": accumulated_text.clone(),
                        "output_ids": chunk.token_ids(),
                        "meta_info": {
                            "id": index_id,
                            "finish_reason": null,
                            "prompt_tokens": chunk.prompt_tokens(),
                            "weight_version": &ctx.weight_version,
                            "input_token_logprobs": input_token_logprobs.as_ref(),
                            "output_token_logprobs": current_output_logprobs,
                            "completion_tokens": current_completion_tokens,
                            "cached_tokens": chunk.cached_tokens()
                        },
                        "index": index
                    });

                    let sse_data = serde_json::to_string(&chunk_response)
                        .map_err(|e| format!("Failed to serialize generate chunk: {e}"))?;
                    tx.send(Ok(Bytes::from(format!("data: {sse_data}\n\n"))))
                        .map_err(|_| "Failed to send chunk".to_string())?;
                }
                ProtoResponseVariant::Complete(complete) => {
                    let index = complete.index();
                    let accumulated_text =
                        accumulated_texts.get(&index).cloned().unwrap_or_default();

                    // Use accumulated count (we tracked deltas from both backends)
                    let completion_tokens = *completion_tokens_map.get(&index).unwrap_or(&0);

                    let final_output_logprobs = accumulated_output_logprobs
                        .get(&index)
                        .and_then(|o| o.as_ref());
                    let index_id = format!("{}-{}", ctx.request_id, index);
                    let e2e_latency = start_time.elapsed().as_secs_f64();

                    // Parse finish_reason
                    let finish_reason = utils::parse_finish_reason(
                        complete.finish_reason(),
                        complete.completion_tokens(),
                    );

                    // Send final chunk with finish_reason
                    let finish_response = json!({
                        "text": accumulated_text,
                        "output_ids": complete.output_ids()[complete.output_ids().len().saturating_sub(1)..].to_vec(),
                        "meta_info": {
                            "id": index_id,
                            "finish_reason": finish_reason,
                            "prompt_tokens": complete.prompt_tokens(),
                            "weight_version": &ctx.weight_version,
                            "input_token_logprobs": input_token_logprobs.as_ref(),
                            "output_token_logprobs": final_output_logprobs,
                            "completion_tokens": completion_tokens,
                            "cached_tokens": complete.cached_tokens(),
                            "e2e_latency": e2e_latency
                        },
                        "index": index
                    });

                    let sse_data = serde_json::to_string(&finish_response)
                        .map_err(|e| format!("Failed to serialize generate finish: {e}"))?;
                    tx.send(Ok(Bytes::from(format!("data: {sse_data}\n\n"))))
                        .map_err(|_| "Failed to send finish chunk".to_string())?;

                    // Continue to process all completions if n>1
                }
                ProtoResponseVariant::Error(error) => {
                    return Err(error.message().to_string());
                }
                ProtoResponseVariant::None => continue,
            }
        }

        // Mark stream as completed successfully to prevent abort on drop
        stream.mark_completed();

        // Record streaming metrics
        let total_completion: u32 = completion_tokens_map.values().sum();
        Self::record_generate_metrics(start_time, first_token_time, total_completion, &ctx);

        Ok(())
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /// Record streaming metrics for generate endpoint
    fn record_generate_metrics(
        start_time: Instant,
        first_token_time: Option<Instant>,
        total_completion: u32,
        ctx: &GenerateStreamContext,
    ) {
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: ctx.backend_type,
            model_id: &ctx.model,
            endpoint: metrics_labels::ENDPOINT_GENERATE,
            ttft: first_token_time.map(|t| t.duration_since(start_time)),
            generation_duration: start_time.elapsed(),
            input_tokens: None, // generate endpoint doesn't expose prompt tokens in streaming
            output_tokens: total_completion as u64,
        });
    }

    /// Process a chunk of tokens through the stop decoder
    fn process_chunk_tokens(
        stop_decoder: &mut StopSequenceDecoder,
        token_ids: &[u32],
    ) -> (String, bool) {
        let mut chunk_text = String::new();

        for &token_id in token_ids {
            match stop_decoder.process_token(token_id).unwrap_or_else(|e| {
                debug!(
                    "Error processing token {}: {}. Treating as Held.",
                    token_id, e
                );
                SequenceDecoderOutput::Held
            }) {
                SequenceDecoderOutput::Text(text) => {
                    chunk_text.push_str(&text);
                }
                SequenceDecoderOutput::StoppedWithText(text) => {
                    chunk_text.push_str(&text);
                    return (chunk_text, true);
                }
                SequenceDecoderOutput::Stopped => {
                    return (chunk_text, true);
                }
                SequenceDecoderOutput::Held => {}
            }
        }
        (chunk_text, false)
    }

    #[inline]
    fn unseen_completion_tokens(output_ids: &[u32], streamed_count: usize) -> &[u32] {
        if streamed_count >= output_ids.len() {
            &[]
        } else {
            &output_ids[streamed_count..]
        }
    }

    fn decode_complete_output_text(
        tokenizer: &Arc<dyn Tokenizer>,
        stop_params: &(Option<StringOrArray>, Option<Vec<u32>>, bool, bool),
        output_ids: &[u32],
    ) -> Result<String, String> {
        let (stop, stop_token_ids, skip_special_tokens, no_stop_trim) = stop_params;
        let mut stop_decoder = utils::create_stop_decoder(
            tokenizer,
            stop.as_ref(),
            stop_token_ids.as_ref(),
            *skip_special_tokens,
            *no_stop_trim,
        );

        let outputs = stop_decoder
            .process_tokens(output_ids)
            .map_err(|e| format!("Failed to decode complete output ids: {e}"))?;

        let mut decoded_text = String::new();
        for output in outputs {
            match output {
                SequenceDecoderOutput::Text(text) => decoded_text.push_str(&text),
                SequenceDecoderOutput::StoppedWithText(text) => {
                    decoded_text.push_str(&text);
                    break;
                }
                SequenceDecoderOutput::Stopped => break,
                SequenceDecoderOutput::Held => {}
            }
        }

        if let SequenceDecoderOutput::Text(text) | SequenceDecoderOutput::StoppedWithText(text) =
            stop_decoder.flush()
        {
            decoded_text.push_str(&text);
        }

        Ok(decoded_text)
    }

    fn missing_completion_suffix(
        tokenizer: &Arc<dyn Tokenizer>,
        stop_params: &(Option<StringOrArray>, Option<Vec<u32>>, bool, bool),
        output_ids: &[u32],
        streamed_text: &str,
    ) -> Result<Option<String>, String> {
        let decoded_text = Self::decode_complete_output_text(tokenizer, stop_params, output_ids)?;
        Ok(decoded_text
            .strip_prefix(streamed_text)
            .filter(|suffix| !suffix.is_empty())
            .map(ToOwned::to_owned))
    }

    #[inline]
    fn is_structured_json_response(request: &ChatCompletionRequest) -> bool {
        matches!(
            request.response_format,
            Some(ResponseFormat::JsonObject { .. }) | Some(ResponseFormat::JsonSchema { .. })
        )
    }

    #[inline]
    fn is_auto_tool_choice(request: &ChatCompletionRequest) -> bool {
        request.tools.is_some()
            && matches!(
                request.tool_choice,
                Some(ToolChoice::Value(ToolChoiceValue::Auto)) | None
            )
    }

    #[inline]
    fn should_defer_empty_auto_tool_calls(request: &ChatCompletionRequest) -> bool {
        Self::is_auto_tool_choice(request)
    }

    #[inline]
    fn is_empty_tool_args(arguments: &str) -> bool {
        let trimmed = arguments.trim();
        trimmed.is_empty() || trimmed == "{}"
    }

    fn tool_args_missing_required_fields(
        tools: &[Tool],
        function_name: &str,
        arguments: &str,
    ) -> bool {
        let Ok(Value::Object(args)) = serde_json::from_str::<Value>(arguments) else {
            return false;
        };
        let Some(tool) = tools
            .iter()
            .find(|tool| tool.function.name == function_name)
        else {
            return true;
        };
        let Some(schema) = tool.function.parameters.as_object() else {
            return false;
        };
        let Some(required) = schema.get("required").and_then(Value::as_array) else {
            return false;
        };
        required
            .iter()
            .filter_map(Value::as_str)
            .any(|key| !args.contains_key(key))
    }

    fn json_repair_suffix(text: &str) -> Option<String> {
        if serde_json::from_str::<Value>(text).is_ok() {
            return None;
        }

        let mut stack = Vec::new();
        let mut in_string = false;
        let mut escaped = false;

        for ch in text.chars() {
            if in_string {
                if escaped {
                    escaped = false;
                    continue;
                }

                match ch {
                    '\\' => escaped = true,
                    '"' => in_string = false,
                    _ => {}
                }
                continue;
            }

            match ch {
                '"' => in_string = true,
                '{' | '[' => stack.push(ch),
                '}' => {
                    if stack.last() == Some(&'{') {
                        stack.pop();
                    } else {
                        return None;
                    }
                }
                ']' => {
                    if stack.last() == Some(&'[') {
                        stack.pop();
                    } else {
                        return None;
                    }
                }
                _ => {}
            }
        }

        if escaped {
            return None;
        }

        let mut suffix = String::new();
        if in_string {
            suffix.push('"');
        }

        for opener in stack.iter().rev() {
            suffix.push(match opener {
                '{' => '}',
                '[' => ']',
                _ => return None,
            });
        }

        if suffix.is_empty() {
            return None;
        }

        let repaired = format!("{text}{suffix}");
        serde_json::from_str::<Value>(&repaired).ok()?;
        Some(suffix)
    }

    /// Helper: Process reasoning content in streaming mode
    #[expect(clippy::too_many_arguments)]
    async fn process_reasoning_stream(
        &self,
        delta: &str,
        index: u32,
        reasoning_parsers: &mut HashMap<u32, Arc<tokio::sync::Mutex<Box<dyn ReasoningParser>>>>,
        request_id: &str,
        model: &str,
        created: u64,
        system_fingerprint: Option<&str>,
    ) -> (String, Option<ChatCompletionStreamResponse>, bool) {
        // Create fresh parser for this index (not pooled, to avoid state pollution)
        #[expect(
            clippy::expect_used,
            reason = "parser availability is checked upfront before streaming begins"
        )]
        reasoning_parsers.entry(index).or_insert_with(|| {
            let parser = utils::create_reasoning_parser(
                &self.reasoning_parser_factory,
                self.configured_reasoning_parser.as_deref(),
                model,
            )
            .expect("Parser should be available - checked upfront");
            Arc::new(tokio::sync::Mutex::new(parser))
        });

        if let Some(pooled_parser) = reasoning_parsers.get(&index) {
            let (parse_result, in_reasoning) = {
                let mut parser = pooled_parser.lock().await;
                let result = parser.parse_reasoning_streaming_incremental(delta);
                let in_reasoning = parser.is_in_reasoning();
                (result, in_reasoning)
            };

            match parse_result {
                Ok(ParserResult {
                    reasoning_text,
                    normal_text,
                }) => {
                    let chunk = if reasoning_text.is_empty() {
                        None
                    } else {
                        Some(
                            ChatCompletionStreamResponse::builder(request_id, model)
                                .created(created)
                                .add_choice_reasoning(index, reasoning_text)
                                .maybe_system_fingerprint(system_fingerprint)
                                .build(),
                        )
                    };
                    return (normal_text, chunk, in_reasoning);
                }
                Err(e) => {
                    warn!("Reasoning parsing error: {}", e);
                }
            }
        }

        (delta.to_string(), None, false)
    }

    /// Helper: Process specific function case - emit tool call deltas with arguments
    #[expect(clippy::too_many_arguments)]
    fn process_specific_function_stream(
        delta: &str,
        index: u32,
        has_tool_calls: &mut HashMap<u32, bool>,
        emitted_tool_call_indices: &mut HashMap<u32, HashSet<usize>>,
        tool_choice: Option<&ToolChoice>,
        request_id: &str,
        model: &str,
        created: u64,
        system_fingerprint: Option<&str>,
        history_tool_calls_count: usize,
    ) -> Vec<ChatCompletionStreamResponse> {
        let mut chunks = Vec::new();

        if let Some(ToolChoice::Function { function, .. }) = tool_choice {
            let is_first_call = !has_tool_calls.contains_key(&index);

            if is_first_call {
                // First chunk: send name and id
                has_tool_calls.insert(index, true);
                emitted_tool_call_indices
                    .entry(index)
                    .or_default()
                    .insert(0);

                let tool_call_id = utils::generate_tool_call_id(
                    model,
                    &function.name,
                    0,
                    history_tool_calls_count,
                );

                chunks.push(
                    ChatCompletionStreamResponse::builder(request_id, model)
                        .created(created)
                        .add_choice_tool_name(index, tool_call_id, function.name.clone())
                        .maybe_system_fingerprint(system_fingerprint)
                        .build(),
                );
            }

        }

        chunks
    }

    /// Helper: Process tool calls in streaming mode
    #[expect(clippy::too_many_arguments)]
    async fn process_tool_calls_stream(
        &self,
        delta: &str,
        index: u32,
        tool_parsers: &mut HashMap<u32, Arc<tokio::sync::Mutex<Box<dyn ToolParser>>>>,
        has_tool_calls: &mut HashMap<u32, bool>,
        emitted_tool_call_indices: &mut HashMap<u32, HashSet<usize>>,
        tools: &[Tool],
        request_id: &str,
        model: &str,
        created: u64,
        system_fingerprint: Option<&str>,
        history_tool_calls_count: usize,
        use_json_parser: bool,
        defer_empty_auto_tool_calls: bool,
    ) -> Vec<ChatCompletionStreamResponse> {
        let mut chunks = Vec::new();

        // Create fresh parser for this index (not pooled, to avoid state pollution)
        #[expect(
            clippy::expect_used,
            reason = "parser availability is checked upfront before streaming begins"
        )]
        tool_parsers.entry(index).or_insert_with(|| {
            let parser = if use_json_parser {
                utils::create_tool_parser(&self.tool_parser_factory, Some("json"), model)
                    .expect("JSON parser should be available")
            } else {
                utils::create_tool_parser(
                    &self.tool_parser_factory,
                    self.configured_tool_parser.as_deref(),
                    model,
                )
                .expect("Parser should be available - checked upfront")
            };
            Arc::new(tokio::sync::Mutex::new(parser))
        });

        if let Some(pooled_parser) = tool_parsers.get(&index) {
            let mut parser = pooled_parser.lock().await;

            match parser.parse_incremental(delta, tools).await {
                Ok(StreamingParseResult { normal_text, calls }) => {
                    // Emit normal text if present
                    if !normal_text.is_empty() {
                        chunks.push(
                            ChatCompletionStreamResponse::builder(request_id, model)
                                .created(created)
                                .add_choice_content(index, "assistant", normal_text)
                                .maybe_system_fingerprint(system_fingerprint)
                                .build(),
                        );
                    }

                    // Emit tool call chunks
                    for tool_call_item in calls {
                        if defer_empty_auto_tool_calls {
                            if let Some(name) = tool_call_item.name.as_deref() {
                                if Self::is_empty_tool_args(&tool_call_item.parameters)
                                    || (tool_call_item.parameters.trim_start().starts_with('{')
                                        && tool_call_item.parameters.trim_end().ends_with('}')
                                        && Self::tool_args_missing_required_fields(
                                            tools,
                                            name,
                                            &tool_call_item.parameters,
                                        ))
                                {
                                    continue;
                                }
                            }
                        }

                        has_tool_calls.insert(index, true);
                        if tool_call_item.name.is_some() {
                            emitted_tool_call_indices
                                .entry(index)
                                .or_default()
                                .insert(tool_call_item.tool_index);
                        }

                        let tool_call_id = if let Some(ref name) = tool_call_item.name {
                            Some(utils::generate_tool_call_id(
                                model,
                                name,
                                tool_call_item.tool_index,
                                history_tool_calls_count,
                            ))
                        } else {
                            None
                        };

                        let tool_call_delta = ToolCallDelta {
                            index: tool_call_item.tool_index as u32,
                            id: tool_call_id,
                            tool_type: if tool_call_item.name.is_some() {
                                Some("function".to_string())
                            } else {
                                None
                            },
                            function: Some(FunctionCallDelta {
                                name: tool_call_item.name,
                                arguments: if tool_call_item.parameters.is_empty() {
                                    None
                                } else {
                                    Some(tool_call_item.parameters)
                                },
                            }),
                        };

                        chunks.push(
                            ChatCompletionStreamResponse::builder(request_id, model)
                                .created(created)
                                .add_choice_tool_call_delta(index, tool_call_delta)
                                .maybe_system_fingerprint(system_fingerprint)
                                .build(),
                        );
                    }

                    return chunks;
                }
                Err(e) => {
                    error!("Tool call parsing error: {}", e);
                }
            }
        }

        chunks
    }

    /// Format a response as SSE chunk into a reusable buffer
    /// This avoids allocations by reusing the same buffer across multiple chunks
    #[inline]
    fn format_sse_chunk_into(buffer: &mut Vec<u8>, chunk: &ChatCompletionStreamResponse) {
        buffer.clear();
        buffer.extend_from_slice(b"data: ");
        if let Err(e) = serde_json::to_writer(&mut *buffer, chunk) {
            error!("Failed to serialize SSE chunk: {}", e);
            buffer.clear();
            buffer.extend_from_slice(b"data: ");
            let error_msg = json!({"error": "serialization_failed"}).to_string();
            buffer.extend_from_slice(error_msg.as_bytes());
        }
        buffer.extend_from_slice(b"\n\n");
    }
}

#[cfg(test)]
mod tests {
    use super::StreamingProcessor;

    #[test]
    fn unseen_completion_tokens_returns_only_unstreamed_suffix() {
        let output_ids = [10_u32, 11, 12, 13];
        assert_eq!(
            StreamingProcessor::unseen_completion_tokens(&output_ids, 3),
            &[13]
        );
    }

    #[test]
    fn unseen_completion_tokens_returns_empty_when_all_tokens_streamed() {
        let output_ids = [10_u32, 11, 12];
        assert!(StreamingProcessor::unseen_completion_tokens(&output_ids, 3).is_empty());
        assert!(StreamingProcessor::unseen_completion_tokens(&output_ids, 99).is_empty());
    }

    #[test]
    fn json_repair_suffix_closes_missing_object_brace() {
        let partial = r#"{"title":"Morning","items":["a","b"]"#;
        assert_eq!(
            StreamingProcessor::json_repair_suffix(partial).as_deref(),
            Some("}")
        );
    }

    #[test]
    fn json_repair_suffix_returns_none_for_valid_json() {
        let valid = r#"{"title":"Morning","items":["a","b"]}"#;
        assert!(StreamingProcessor::json_repair_suffix(valid).is_none());
    }
}
