//! Harmony streaming response processor

use std::{
    collections::{hash_map::Entry::Vacant, HashMap},
    io,
    sync::Arc,
    time::Instant,
};

use axum::response::Response;
use bytes::Bytes;
use openai_protocol::{
    chat::{
        ChatCompletionRequest, ChatCompletionStreamResponse, ChatMessageDelta, ChatStreamChoice,
    },
    common::{ChatLogProbs, FunctionCallDelta, ToolCall, ToolCallDelta, Usage},
    responses::{
        InputTokensDetails, OutputTokensDetails, ResponseStatus, ResponseUsage, ResponsesResponse,
        ResponsesUsage,
    },
};
use serde_json::json;
use tokio::sync::mpsc;
use tracing::{debug, error};

use super::{
    builder::convert_harmony_logprobs, processor::ResponsesIterationResult,
    types::HarmonyChannelDelta, HarmonyParserAdapter,
};
use crate::{
    observability::metrics::{metrics_labels, Metrics, StreamingMetricsParams},
    routers::{
        common::{
            agent_loop::{LoopEvent, StreamSink},
            responses_streaming::OutputItemType,
        },
        grpc::{
            common::{
                response_formatting::CompletionTokenTracker,
                responses::{build_sse_response, GrpcResponseStreamSink},
            },
            context,
            proto_wrapper::{ProtoResponseVariant, ProtoStream},
            utils,
        },
    },
};

/// Processor for streaming Harmony responses
///
/// Returns an SSE stream that parses Harmony tokens incrementally and
/// emits ChatCompletionChunk events for streaming responses.
pub(crate) struct HarmonyStreamingProcessor;

impl HarmonyStreamingProcessor {
    /// Create a new Harmony streaming processor
    pub fn new() -> Self {
        Self
    }

    /// Process a streaming Harmony Chat Completion response
    ///
    /// Returns an SSE response with streaming token updates.
    ///
    /// Note: Caller should attach load guards to the returned response using
    /// `WorkerLoadGuard::attach_to_response()` for proper RAII lifecycle management.
    #[expect(
        clippy::unused_self,
        reason = "takes Arc<Self> for API consistency with other streaming processors"
    )]
    #[expect(
        clippy::disallowed_methods,
        reason = "streaming tasks are fire-and-forget by design; client disconnect terminates them"
    )]
    pub fn process_streaming_chat_response(
        self: Arc<Self>,
        execution_result: context::ExecutionResult,
        chat_request: Arc<ChatCompletionRequest>,
        dispatch: context::DispatchMetadata,
    ) -> Response {
        // Create SSE channel
        let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

        // Spawn background task based on execution mode
        match execution_result {
            context::ExecutionResult::Single { stream } => {
                tokio::spawn(async move {
                    let result =
                        Self::process_single_stream(stream, dispatch, chat_request, &tx).await;

                    if let Err(e) = result {
                        error!("Harmony streaming error: {}", e);
                        utils::send_error_sse(&tx, &e, "internal_error");
                    }

                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                });
            }
            context::ExecutionResult::Dual { prefill, decode } => {
                tokio::spawn(async move {
                    let result =
                        Self::process_dual_stream(prefill, *decode, dispatch, chat_request, &tx)
                            .await;

                    if let Err(e) = result {
                        error!("Harmony dual streaming error: {}", e);
                        utils::send_error_sse(&tx, &e, "internal_error");
                    }

                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                });
            }
            context::ExecutionResult::Embedding { .. } => {
                error!("Harmony streaming not supported for embeddings");
                utils::send_error_sse(
                    &tx,
                    "Embeddings not supported in Harmony streaming",
                    "invalid_request_error",
                );
                let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
            }
        }

        // Return SSE response
        build_sse_response(rx)
    }

    /// Process streaming chunks from a single stream
    async fn process_single_stream(
        grpc_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        original_request: Arc<ChatCompletionRequest>,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        let mut prompt_tokens = HashMap::new();
        let mut cached_tokens = HashMap::new();
        Self::process_chat_decode_stream(
            grpc_stream,
            &dispatch,
            &original_request,
            tx,
            &mut prompt_tokens,
            &mut cached_tokens,
        )
        .await
    }

    /// Process streaming chunks from dual streams (prefill + decode)
    async fn process_dual_stream(
        mut prefill_stream: ProtoStream,
        decode_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        original_request: Arc<ChatCompletionRequest>,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        // Phase 1: Process prefill stream (collect metadata)
        let mut prompt_tokens: HashMap<u32, u32> = HashMap::new();
        let mut cached_tokens: HashMap<u32, u32> = HashMap::new();

        while let Some(result) = prefill_stream.next().await {
            let response = result.map_err(|e| format!("Prefill stream error: {}", e.message()))?;

            if let ProtoResponseVariant::Complete(complete_wrapper) = response.into_response() {
                prompt_tokens.insert(complete_wrapper.index(), complete_wrapper.prompt_tokens());
                cached_tokens.insert(complete_wrapper.index(), complete_wrapper.cached_tokens());
            }
        }

        // Phase 2: Decode (shared helper)
        Self::process_chat_decode_stream(
            decode_stream,
            &dispatch,
            &original_request,
            tx,
            &mut prompt_tokens,
            &mut cached_tokens,
        )
        .await?;

        // Mark prefill stream completed AFTER decode succeeds
        // This ensures that if client disconnects during decode, BOTH streams send abort
        prefill_stream.mark_completed();
        Ok(())
    }

    /// Process the decode phase of a Chat Completion stream.
    ///
    /// Shared between single-stream and dual-stream modes. The `prompt_tokens`
    /// and `cached_tokens` maps may be pre-populated from a prefill phase
    /// (dual stream) or empty (single stream). Values from `Complete` messages
    /// are inserted only if not already present.
    async fn process_chat_decode_stream(
        mut decode_stream: ProtoStream,
        dispatch: &context::DispatchMetadata,
        original_request: &ChatCompletionRequest,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        prompt_tokens: &mut HashMap<u32, u32>,
        cached_tokens: &mut HashMap<u32, u32>,
    ) -> Result<(), String> {
        // Timing for metrics
        let start_time = Instant::now();
        let mut first_token_time: Option<Instant> = None;

        // Per-index state management (for n>1 support)
        let mut parsers: HashMap<u32, HarmonyParserAdapter> = HashMap::new();
        let mut is_firsts: HashMap<u32, bool> = HashMap::new();
        let mut matched_stops: HashMap<u32, Option<serde_json::Value>> = HashMap::new();
        let mut completion_tokens = CompletionTokenTracker::new();

        let stream_options = &original_request.stream_options;

        // Process stream
        while let Some(result) = decode_stream.next().await {
            let response = result.map_err(|e| format!("Stream error: {}", e.message()))?;

            match response.into_response() {
                ProtoResponseVariant::Chunk(chunk_wrapper) => {
                    let index = chunk_wrapper.index();

                    // Track first token time for TTFT metric
                    if first_token_time.is_none() {
                        first_token_time = Some(Instant::now());
                    }

                    // Initialize parser for this index if needed
                    if let Vacant(e) = parsers.entry(index) {
                        e.insert(
                            HarmonyParserAdapter::new()
                                .map_err(|e| format!("Failed to create parser: {e}"))?,
                        );
                        is_firsts.insert(index, true);
                    }

                    completion_tokens.record_chunk(&chunk_wrapper);

                    // Convert logprobs if present and requested
                    let chunk_logprobs = if original_request.logprobs {
                        chunk_wrapper
                            .output_logprobs()
                            .map(|lp| convert_harmony_logprobs(&lp))
                    } else {
                        None
                    };

                    // Parse chunk via Harmony parser
                    let parser = parsers
                        .get_mut(&index)
                        .ok_or("Parser not found for index")?;

                    let delta_result = parser
                        .parse_chunk(chunk_wrapper.token_ids())
                        .map_err(|e| format!("Parse error: {e}"))?;

                    // Emit SSE event if there's a delta
                    if let Some(delta) = delta_result {
                        let is_first = is_firsts.get(&index).copied().unwrap_or(false);
                        Self::emit_chunk_delta(
                            &delta,
                            index,
                            is_first,
                            dispatch,
                            original_request,
                            tx,
                            chunk_logprobs,
                        )?;

                        if is_first {
                            is_firsts.insert(index, false);
                        }
                    }
                }
                ProtoResponseVariant::Complete(complete_wrapper) => {
                    let index = complete_wrapper.index();

                    // Store final metadata
                    matched_stops.insert(index, complete_wrapper.matched_stop_json());
                    prompt_tokens
                        .entry(index)
                        .or_insert_with(|| complete_wrapper.prompt_tokens());
                    completion_tokens.record_complete(&complete_wrapper);
                    cached_tokens
                        .entry(index)
                        .or_insert_with(|| complete_wrapper.cached_tokens());

                    // Finalize parser and emit final chunk
                    if let Some(parser) = parsers.get_mut(&index) {
                        let matched_stop = matched_stops.get(&index).and_then(|m| m.clone());

                        let final_output =
                            parser.finalize(complete_wrapper.finish_reason().to_string());

                        Self::emit_final_chunk(
                            index,
                            &final_output.finish_reason,
                            matched_stop.as_ref(),
                            dispatch,
                            original_request,
                            tx,
                        )?;
                    }
                }
                ProtoResponseVariant::None => {}
            }
        }

        // Mark stream as completed successfully to prevent abort on drop
        decode_stream.mark_completed();

        // Compute totals once for both usage chunk and metrics
        let total_prompt: u32 = prompt_tokens.values().sum();
        let total_completion: u32 = completion_tokens.total();
        let total_cached: u32 = cached_tokens.values().sum();

        // Emit final usage if requested
        if let Some(true) = stream_options.as_ref().and_then(|so| so.include_usage) {
            Self::emit_usage_chunk(
                total_prompt,
                total_completion,
                total_cached,
                dispatch,
                original_request,
                tx,
            )?;
        }

        // Record streaming metrics
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: metrics_labels::BACKEND_HARMONY,
            model_id: &original_request.model,
            endpoint: metrics_labels::ENDPOINT_CHAT,
            ttft: first_token_time.map(|t| t.duration_since(start_time)),
            generation_duration: start_time.elapsed(),
            input_tokens: Some(total_prompt as u64),
            output_tokens: total_completion as u64,
        });

        Ok(())
    }

    /// Emit a chunk delta from Harmony channels
    fn emit_chunk_delta(
        delta: &HarmonyChannelDelta,
        index: u32,
        is_first: bool,
        dispatch: &context::DispatchMetadata,
        original_request: &ChatCompletionRequest,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        logprobs: Option<ChatLogProbs>,
    ) -> Result<(), String> {
        // On first chunk, emit role announcement separately
        if is_first {
            let role_chunk = ChatCompletionStreamResponse::builder(
                &dispatch.request_id,
                &original_request.model,
            )
            .created(dispatch.created)
            .add_choice_role(index, "assistant")
            .maybe_system_fingerprint(dispatch.weight_version.as_deref())
            .build();

            let chunk_json = serde_json::to_string(&role_chunk)
                .map_err(|e| format!("JSON serialization error: {e}"))?;
            let sse_data = format!("data: {chunk_json}\n\n");

            tx.send(Ok(Bytes::from(sse_data)))
                .map_err(|_| "Failed to send role chunk".to_string())?;
        }

        // Emit content delta (role is always None for content chunks)
        let chat_delta = ChatMessageDelta {
            role: None,
            content: delta.final_delta.clone(),
            tool_calls: delta.commentary_delta.as_ref().map(|tc_delta| {
                vec![ToolCallDelta {
                    index: tc_delta.index as u32,
                    id: tc_delta.id.clone(),
                    tool_type: tc_delta.id.as_ref().map(|_| "function".to_string()),
                    function: tc_delta.function.as_ref().map(|f| FunctionCallDelta {
                        name: f.name.clone(),
                        arguments: f.arguments.clone(),
                    }),
                }]
            }),
            reasoning_content: delta.analysis_delta.clone(),
        };

        // Build and emit chunk
        let chunk =
            ChatCompletionStreamResponse::builder(&dispatch.request_id, &original_request.model)
                .created(dispatch.created)
                .add_choice(ChatStreamChoice {
                    index,
                    delta: chat_delta,
                    logprobs,
                    finish_reason: None,
                    matched_stop: None,
                })
                .maybe_system_fingerprint(dispatch.weight_version.as_deref())
                .build();

        let chunk_json =
            serde_json::to_string(&chunk).map_err(|e| format!("JSON serialization error: {e}"))?;
        let sse_data = format!("data: {chunk_json}\n\n");

        tx.send(Ok(Bytes::from(sse_data)))
            .map_err(|_| "Failed to send chunk".to_string())?;

        Ok(())
    }

    /// Emit final chunk with finish_reason
    fn emit_final_chunk(
        index: u32,
        finish_reason: &str,
        matched_stop: Option<&serde_json::Value>,
        dispatch: &context::DispatchMetadata,
        original_request: &ChatCompletionRequest,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        let chunk =
            ChatCompletionStreamResponse::builder(&dispatch.request_id, &original_request.model)
                .created(dispatch.created)
                .add_choice_finish_reason(index, finish_reason, matched_stop.cloned())
                .maybe_system_fingerprint(dispatch.weight_version.as_deref())
                .build();

        let chunk_json =
            serde_json::to_string(&chunk).map_err(|e| format!("JSON serialization error: {e}"))?;
        let sse_data = format!("data: {chunk_json}\n\n");

        tx.send(Ok(Bytes::from(sse_data)))
            .map_err(|_| "Failed to send final chunk".to_string())?;

        Ok(())
    }

    /// Emit usage chunk at the end
    fn emit_usage_chunk(
        prompt_tokens: u32,
        completion_tokens: u32,
        cached_tokens: u32,
        dispatch: &context::DispatchMetadata,
        original_request: &ChatCompletionRequest,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        let usage_chunk =
            ChatCompletionStreamResponse::builder(&dispatch.request_id, &original_request.model)
                .created(dispatch.created)
                .usage(
                    Usage::from_counts(prompt_tokens, completion_tokens)
                        .with_cached_tokens(cached_tokens),
                )
                .maybe_system_fingerprint(dispatch.weight_version.as_deref())
                .build();

        let chunk_json = serde_json::to_string(&usage_chunk)
            .map_err(|e| format!("JSON serialization error: {e}"))?;
        let sse_data = format!("data: {chunk_json}\n\n");

        tx.send(Ok(Bytes::from(sse_data)))
            .map_err(|_| "Failed to send usage chunk".to_string())?;

        Ok(())
    }

    /// Process streaming chunks for Responses API iteration.
    ///
    /// When MCP context is provided (session):
    /// - MCP tools with `ResponseFormat::WebSearchCall` → `web_search_call.*` events
    /// - Other MCP tools → `mcp_call.*` events
    /// - Other tools → `function_call.*` events
    ///
    /// When no MCP context is provided, all tool calls are treated as function calls.
    pub async fn process_responses_iteration_stream(
        execution_result: context::ExecutionResult,
        sink: &mut GrpcResponseStreamSink,
    ) -> Result<ResponsesIterationResult, String> {
        match execution_result {
            context::ExecutionResult::Single { stream } => {
                debug!("Processing Responses API single stream mode");
                Self::process_decode_stream(stream, sink, 0).await
            }
            context::ExecutionResult::Dual { prefill, decode } => {
                debug!("Processing Responses API dual stream mode");
                Self::process_responses_dual_stream(prefill, *decode, sink).await
            }
            context::ExecutionResult::Embedding { .. } => {
                Err("Embeddings not supported in Responses API streaming".to_string())
            }
        }
    }

    async fn process_responses_dual_stream(
        mut prefill_stream: ProtoStream,
        decode_stream: ProtoStream,
        sink: &mut GrpcResponseStreamSink,
    ) -> Result<ResponsesIterationResult, String> {
        // Phase 1: Drain prefill stream, collecting cached_tokens from Complete messages
        let mut prefill_cached_tokens_by_index: HashMap<u32, u32> = HashMap::new();
        while let Some(result) = prefill_stream.next().await {
            let response = result.map_err(|e| format!("Prefill stream error: {}", e.message()))?;
            if let ProtoResponseVariant::Complete(complete_wrapper) = response.into_response() {
                prefill_cached_tokens_by_index
                    .insert(complete_wrapper.index(), complete_wrapper.cached_tokens());
            }
        }
        let prefill_cached_tokens: u32 = prefill_cached_tokens_by_index.values().sum();

        // Phase 2: Process decode stream
        let result = Self::process_decode_stream(decode_stream, sink, prefill_cached_tokens).await;

        prefill_stream.mark_completed();
        result
    }

    /// Process decode stream for tool call events.
    async fn process_decode_stream(
        mut decode_stream: ProtoStream,
        sink: &mut GrpcResponseStreamSink,
        prefill_cached_tokens: u32,
    ) -> Result<ResponsesIterationResult, String> {
        let mut parser =
            HarmonyParserAdapter::new().map_err(|e| format!("Failed to create parser: {e}"))?;

        let mut has_analysis = false;
        let mut accumulated_final_text = String::new();
        let mut accumulated_tool_calls: Option<Vec<ToolCall>> = None;

        let mut accumulated_reasoning_text = String::new();
        let mut reasoning_closed = false;
        let mut message_output_index: Option<usize> = None;
        let mut message_item_id: Option<String> = None;
        let mut has_emitted_content_part_added = false;

        // Per-call_index → call_id mapping. Harmony only carries
        // `id` + `name` on a tool call's first chunk; subsequent
        // fragments only carry `arguments`. The sink owns family
        // classification and `output_index` / `item_id` allocation —
        // the processor only needs to know which `call_id` each
        // continuing fragment belongs to so it can address the
        // matching `LoopEvent`.
        let mut call_ids_by_index: HashMap<usize, String> = HashMap::new();

        // Metadata from Complete message; seed cached_tokens from prefill phase (dual-stream)
        let mut finish_reason: String;
        let mut finalized_analysis: Option<String> = None;
        let mut prompt_tokens: u32 = 0;
        let mut completion_tokens: u32 = 0;
        let mut cached_tokens: u32 = prefill_cached_tokens;
        let mut reasoning_token_count: u32 = 0;

        // Process stream
        let mut chunk_count = 0;
        while let Some(result) = decode_stream.next().await {
            chunk_count += 1;
            let response = result.map_err(|e| format!("Decode stream error: {}", e.message()))?;

            match response.into_response() {
                ProtoResponseVariant::Chunk(chunk_wrapper) => {
                    // Track token counts for vLLM (vLLM sends deltas)
                    // For SGLang, skip (SGLang sends cumulative values in Complete)
                    if chunk_wrapper.is_vllm() {
                        completion_tokens += chunk_wrapper.token_ids().len() as u32;
                    }

                    // Parse chunk via Harmony parser
                    let delta_result = parser
                        .parse_chunk(chunk_wrapper.token_ids())
                        .map_err(|e| format!("Parse error: {e}"))?;

                    // Emit SSE events if there's a delta
                    if let Some(delta) = delta_result {
                        // Analysis channel → stream reasoning body events.
                        if let Some(analysis_text) = &delta.analysis_delta {
                            if !analysis_text.is_empty() {
                                sink.emitter
                                    .process_reasoning_delta(analysis_text, &sink.tx)
                                    .map_err(|e| format!("Failed to emit reasoning delta: {e}"))?;
                                accumulated_reasoning_text.push_str(analysis_text);
                                has_analysis = true;
                            }
                        }

                        // Final channel → Message item (WITH text streaming)
                        if let Some(final_delta) = &delta.final_delta {
                            if !final_delta.is_empty() {
                                // Allocate message item if needed
                                if message_output_index.is_none() {
                                    let (output_index, item_id) =
                                        sink.emitter.allocate_output_index(OutputItemType::Message);
                                    message_output_index = Some(output_index);
                                    message_item_id = Some(item_id.clone());

                                    // Build message item structure
                                    let item = json!({
                                        "id": item_id,
                                        "type": "message",
                                        "role": "assistant",
                                        "content": []
                                    });

                                    // Emit output_item.added
                                    let event =
                                        sink.emitter.emit_output_item_added(output_index, &item);
                                    sink.emitter.send_event_best_effort(&event, &sink.tx);
                                }

                                let Some(output_index) = message_output_index else {
                                    continue;
                                };
                                let Some(item_id) = message_item_id.as_ref() else {
                                    continue;
                                };
                                let content_index = 0; // Single content part

                                // Emit content_part.added before first delta
                                if !has_emitted_content_part_added {
                                    let event = sink.emitter.emit_content_part_added(
                                        output_index,
                                        item_id,
                                        content_index,
                                    );
                                    sink.emitter.send_event_best_effort(&event, &sink.tx);
                                    has_emitted_content_part_added = true;
                                }

                                // Emit text delta
                                let event = sink.emitter.emit_text_delta(
                                    final_delta,
                                    output_index,
                                    item_id,
                                    content_index,
                                );
                                sink.emitter.send_event_best_effort(&event, &sink.tx);

                                accumulated_final_text.push_str(final_delta);
                            }
                        }

                        // Commentary channel → tool-call streaming.
                        // Forward each fragment to the sink as
                        // `LoopEvent::ToolCallEmission*`. The sink
                        // classifies family (function_call / mcp_call /
                        // hosted-builtin) by name lookup against its
                        // session-derived snapshot, allocates the
                        // `output_index` / `item_id`, and emits the
                        // family-correct wire events. The processor
                        // does **not** know or check policy.
                        if let Some(tc_delta) = &delta.commentary_delta {
                            let call_index = tc_delta.index;
                            if let Some(call_id) = &tc_delta.id {
                                // First chunk for this call_index.
                                let tool_name = tc_delta
                                    .function
                                    .as_ref()
                                    .and_then(|f| f.name.as_ref())
                                    .map(|n| n.as_str())
                                    .unwrap_or("");
                                call_ids_by_index.insert(call_index, call_id.clone());
                                sink.emit(LoopEvent::ToolCallEmissionStarted {
                                    call_id,
                                    item_id: call_id,
                                    name: tool_name,
                                });
                                if let Some(args) = tc_delta
                                    .function
                                    .as_ref()
                                    .and_then(|f| f.arguments.as_ref())
                                    .filter(|s| !s.is_empty())
                                {
                                    sink.emit(LoopEvent::ToolCallArgumentsFragment {
                                        call_id,
                                        fragment: args,
                                    });
                                }
                            } else if let Some(call_id) =
                                call_ids_by_index.get(&call_index).cloned()
                            {
                                // Subsequent fragment for an in-flight
                                // tool call.
                                if let Some(args) = tc_delta
                                    .function
                                    .as_ref()
                                    .and_then(|f| f.arguments.as_ref())
                                    .filter(|s| !s.is_empty())
                                {
                                    sink.emit(LoopEvent::ToolCallArgumentsFragment {
                                        call_id: &call_id,
                                        fragment: args,
                                    });
                                }
                            }
                        }
                    }
                }
                ProtoResponseVariant::Complete(complete_wrapper) => {
                    // Store final metadata
                    finish_reason = complete_wrapper.finish_reason().to_string();
                    prompt_tokens = complete_wrapper.prompt_tokens();
                    // Combine decode-stream cached_tokens with any prefill cached_tokens
                    cached_tokens = cached_tokens.saturating_add(complete_wrapper.cached_tokens());
                    // For vLLM, use accumulated count (we tracked deltas above)
                    // For SGLang, use complete value (already cumulative)
                    if !complete_wrapper.is_vllm() {
                        completion_tokens = complete_wrapper.completion_tokens();
                    }

                    // Finalize parser and get complete output
                    // Responses API: no user-specified stop sequences
                    let final_output = parser.finalize(finish_reason.clone());

                    // Store finalized output for later use, and flush any
                    // reasoning text that only became visible at parser
                    // finalize time.
                    if let Some(final_analysis) = final_output.analysis.as_ref() {
                        let missing = final_analysis
                            .strip_prefix(&accumulated_reasoning_text)
                            .unwrap_or(final_analysis.as_str());
                        if !missing.is_empty() {
                            sink.emitter
                                .process_reasoning_delta(missing, &sink.tx)
                                .map_err(|e| format!("Failed to emit finalized reasoning: {e}"))?;
                            accumulated_reasoning_text.push_str(missing);
                            has_analysis = true;
                        }
                    }
                    if !reasoning_closed && !accumulated_reasoning_text.is_empty() {
                        sink.emitter
                            .finish_reasoning_item(&sink.tx)
                            .map_err(|e| format!("Failed to finish reasoning item: {e}"))?;
                        reasoning_closed = true;
                    }

                    let missing_final_text = final_output
                        .final_text
                        .strip_prefix(&accumulated_final_text)
                        .unwrap_or(final_output.final_text.as_str());
                    if !missing_final_text.is_empty() {
                        if message_output_index.is_none() {
                            let (output_index, item_id) =
                                sink.emitter.allocate_output_index(OutputItemType::Message);
                            message_output_index = Some(output_index);
                            message_item_id = Some(item_id.clone());

                            let item = json!({
                                "id": item_id,
                                "type": "message",
                                "role": "assistant",
                                "content": []
                            });
                            let event = sink.emitter.emit_output_item_added(output_index, &item);
                            sink.emitter.send_event_best_effort(&event, &sink.tx);
                        }

                        if let (Some(output_index), Some(item_id)) =
                            (message_output_index, message_item_id.as_ref())
                        {
                            let content_index = 0;
                            if !has_emitted_content_part_added {
                                let event = sink.emitter.emit_content_part_added(
                                    output_index,
                                    item_id,
                                    content_index,
                                );
                                sink.emitter.send_event_best_effort(&event, &sink.tx);
                                has_emitted_content_part_added = true;
                            }
                            let event = sink.emitter.emit_text_delta(
                                missing_final_text,
                                output_index,
                                item_id,
                                content_index,
                            );
                            sink.emitter.send_event_best_effort(&event, &sink.tx);
                            accumulated_final_text.push_str(missing_final_text);
                        }
                    }

                    finalized_analysis = final_output.analysis;
                    accumulated_tool_calls = final_output.commentary;
                    reasoning_token_count = final_output.reasoning_token_count;

                    // The parser regenerates fresh UUID `call_…` ids
                    // on each `parse_chunk` (streaming-time) AND on
                    // `finalize` (post-stream), so the ids on
                    // `accumulated_tool_calls` do **not** match the
                    // `tc_delta.id` we already issued via
                    // `LoopEvent::ToolCallEmissionStarted` to the sink.
                    // Without this fix-up the sink's tracking misses
                    // on `EmissionDone` (no `*_arguments.done` fires)
                    // and on the post-execute `ToolCompleted` (no
                    // `mcp_call.completed` + closing `output_item.done`
                    // fires). Replace finalize-time ids with the
                    // streaming-time ids by call index — the parser's
                    // commentary always preserves emission order.
                    if let Some(ref mut tool_calls) = accumulated_tool_calls {
                        for (idx, tool_call) in tool_calls.iter_mut().enumerate() {
                            if let Some(stream_id) = call_ids_by_index.get(&idx) {
                                tool_call.id.clone_from(stream_id);
                            }
                        }
                    }

                    // Close out every in-flight tool call's wire
                    // lifecycle. Sink emits `*_arguments.done` for
                    // streaming families and (caller fc only) the
                    // closing `output_item.done(status=completed)`.
                    // For gateway tools sink leaves tracking alive
                    // so the post-execution `LoopEvent::ToolCompleted`
                    // can fire `mcp_call.completed/.failed` +
                    // `output_item.done(with output)` after the
                    // executor returns.
                    if let Some(ref tool_calls) = accumulated_tool_calls {
                        for tool_call in tool_calls {
                            let args_str = tool_call.function.arguments.as_deref().unwrap_or("");
                            sink.emit(LoopEvent::ToolCallEmissionDone {
                                call_id: &tool_call.id,
                                full_args: args_str,
                            });
                        }
                    }

                    // Close message item if we opened one
                    if let (Some(output_index), Some(item_id)) =
                        (message_output_index, message_item_id.as_ref())
                    {
                        let content_index = 0;

                        // Emit text_done
                        let event =
                            sink.emitter
                                .emit_text_done(output_index, item_id, content_index);
                        sink.emitter.send_event_best_effort(&event, &sink.tx);

                        // Emit content_part.done
                        let event = sink.emitter.emit_content_part_done(
                            output_index,
                            item_id,
                            content_index,
                        );
                        sink.emitter.send_event_best_effort(&event, &sink.tx);

                        // Emit output_item.done
                        let item = json!({
                            "id": item_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [{
                                "type": "output_text",
                                "text": accumulated_final_text.clone()
                            }]
                        });
                        let event = sink.emitter.emit_output_item_done(output_index, &item);

                        // Mark as completed before sending (so it's included in final output even if send fails)
                        sink.emitter.complete_output_item(output_index);

                        sink.emitter.send_event_best_effort(&event, &sink.tx);
                    }
                }
                ProtoResponseVariant::None => {}
            }
        }

        debug!(
            "Stream loop ended. Total chunks received: {}, has_analysis: {}, tool_calls: {}, final_text_len: {}",
            chunk_count,
            has_analysis,
            accumulated_tool_calls.as_ref().map(|tc| tc.len()).unwrap_or(0),
            accumulated_final_text.len()
        );

        // Extract tool calls from completed messages or incomplete commentary
        if chunk_count > 0 && accumulated_tool_calls.is_none() {
            let messages = parser.get_messages();

            // Try extracting from completed messages first
            let (analysis_opt, commentary_opt, final_text_extracted) =
                HarmonyParserAdapter::parse_messages(&messages);
            accumulated_tool_calls.clone_from(&commentary_opt);

            // If no tool calls found, check for incomplete commentary in parser state
            if accumulated_tool_calls.is_none() {
                accumulated_tool_calls = parser.extract_incomplete_commentary();
            }

            debug!(
                "Tool call extraction: completed_msgs={}, tool_calls={}, has_analysis={}, final_text_len={}",
                messages.len(),
                accumulated_tool_calls.as_ref().map(|tc| tc.len()).unwrap_or(0),
                analysis_opt.is_some(),
                final_text_extracted.len()
            );

            // Same id fix-up the Complete branch does, applied here
            // for the case where the stream ended without a Complete
            // message (post-loop `extract_incomplete_commentary`
            // path). Without this, the parser-generated finalize-time
            // call_id would not match the streaming-time id we
            // already announced via `LoopEvent::ToolCallEmissionStarted`,
            // so sink tracking would miss on `EmissionDone` /
            // `ToolCompleted` and the wire would skip
            // `*_arguments.done` + `mcp_call.completed` +
            // `output_item.done`.
            if let Some(ref mut tool_calls) = accumulated_tool_calls {
                for (idx, tool_call) in tool_calls.iter_mut().enumerate() {
                    if let Some(stream_id) = call_ids_by_index.get(&idx) {
                        tool_call.id.clone_from(stream_id);
                    }
                }
            }

            // Close out any tool calls the parser surfaced. Sink
            // takes a `LoopEvent::ToolCallEmissionDone` per call;
            // it knows from its own tracking whether the call is a
            // caller `function_call` (closes lifecycle now) or a
            // gateway tool (leaves tracking alive for the
            // post-execution `ToolCompleted` to finish). The
            // processor stays family-blind here.
            if let Some(ref tool_calls) = accumulated_tool_calls {
                for tool_call in tool_calls {
                    let args_str = tool_call.function.arguments.as_deref().unwrap_or("");
                    sink.emit(LoopEvent::ToolCallEmissionDone {
                        call_id: &tool_call.id,
                        full_args: args_str,
                    });
                }
            }
        }

        if !reasoning_closed && !accumulated_reasoning_text.is_empty() {
            sink.emitter
                .finish_reasoning_item(&sink.tx)
                .map_err(|e| format!("Failed to finish reasoning item: {e}"))?;
        }

        // Mark stream as completed successfully to prevent abort on drop
        decode_stream.mark_completed();

        // Return result based on whether tool calls were found
        if let Some(tool_calls) = accumulated_tool_calls {
            if !tool_calls.is_empty() {
                let analysis_content = if has_analysis {
                    finalized_analysis
                } else {
                    None
                };

                return Ok(ResponsesIterationResult::ToolCallsFound {
                    tool_calls,
                    analysis: analysis_content,
                    partial_text: accumulated_final_text,
                    usage: Usage::from_counts(prompt_tokens, completion_tokens)
                        .with_cached_tokens(cached_tokens)
                        .with_reasoning_tokens(reasoning_token_count),
                    request_id: sink.emitter.response_id.clone(),
                });
            }
        }

        // For streaming, we don't build the full ResponsesResponse here
        // The caller will build it from the SSE events
        // Return a placeholder Completed result (caller ignores these fields in streaming mode)
        Ok(ResponsesIterationResult::Completed {
            response: Box::new(
                ResponsesResponse::builder(&sink.emitter.response_id, "")
                    .status(ResponseStatus::Completed)
                    .usage(ResponsesUsage::Modern(ResponseUsage {
                        input_tokens: prompt_tokens,
                        output_tokens: completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                        input_tokens_details: if cached_tokens > 0 {
                            Some(InputTokensDetails { cached_tokens })
                        } else {
                            None
                        },
                        output_tokens_details: if reasoning_token_count > 0 {
                            Some(OutputTokensDetails {
                                reasoning_tokens: reasoning_token_count,
                            })
                        } else {
                            None
                        },
                    }))
                    .build(),
            ),
            usage: Usage::from_counts(prompt_tokens, completion_tokens)
                .with_cached_tokens(cached_tokens)
                .with_reasoning_tokens(reasoning_token_count),
        })
    }
}

impl Default for HarmonyStreamingProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {

    // The argument-streaming classification used to live on a local
    // helper (`streams_arguments`) because the harmony processor
    // emitted tool-call args inline through the legacy
    // `ResponseStreamEventEmitter` helpers. Both producer and
    // classifier moved to `ToolPresentation` /
    // `GrpcResponseStreamSink` after the agent-loop refactor; the
    // classification lock-down test moved with them
    // (`presentation::tests`). This module no longer needs to keep
    // a parallel anchor.
}
