//! Harmony streaming response processor

use std::{
    collections::{hash_map::Entry::Vacant, HashMap, HashSet},
    io,
    sync::Arc,
    time::Instant,
};

use axum::response::Response;
use bytes::Bytes;
use serde_json::json;
use tokio::sync::mpsc;
use tracing::{debug, error};

use super::{
    builder::convert_harmony_logprobs, processor::ResponsesIterationResult,
    types::HarmonyChannelDelta, HarmonyParserAdapter,
};
use crate::{
    mcp::{McpOrchestrator, ResponseFormat},
    observability::metrics::{metrics_labels, Metrics, StreamingMetricsParams},
    protocols::{
        chat::{
            ChatCompletionRequest, ChatCompletionStreamResponse, ChatMessageDelta, ChatStreamChoice,
        },
        common::{ChatLogProbs, FunctionCallDelta, ToolCall, ToolCallDelta, Usage},
        responses::{
            OutputTokensDetails, ResponseStatus, ResponseUsage, ResponsesResponse, ResponsesUsage,
        },
    },
    routers::{
        grpc::{
            common::{
                response_formatting::CompletionTokenTracker,
                responses::{
                    build_sse_response,
                    streaming::{
                        attach_mcp_server_label, OutputItemType, ResponseStreamEventEmitter,
                    },
                },
            },
            context,
            proto_wrapper::{ProtoResponseVariant, ProtoStream},
        },
        mcp_utils::resolve_tool_server_label,
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
                        let error_chunk = format!(
                            "data: {}\n\n",
                            json!({
                                "error": {
                                    "message": e,
                                    "type": "internal_error"
                                }
                            })
                        );
                        let _ = tx.send(Ok(Bytes::from(error_chunk)));
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
                        let error_chunk = format!(
                            "data: {}\n\n",
                            json!({
                                "error": {
                                    "message": e,
                                    "type": "internal_error"
                                }
                            })
                        );
                        let _ = tx.send(Ok(Bytes::from(error_chunk)));
                    }

                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                });
            }
            context::ExecutionResult::Embedding { .. } => {
                error!("Harmony streaming not supported for embeddings");
                let error_chunk = format!(
                    "data: {}\n\n",
                    json!({
                        "error": {
                            "message": "Embeddings not supported in Harmony streaming",
                            "type": "invalid_request_error"
                        }
                    })
                );
                let _ = tx.send(Ok(Bytes::from(error_chunk)));
            }
        }

        // Return SSE response
        build_sse_response(rx)
    }

    /// Process streaming chunks from a single stream
    async fn process_single_stream(
        mut grpc_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        original_request: Arc<ChatCompletionRequest>,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        // Timing for metrics
        let start_time = Instant::now();
        let mut first_token_time: Option<Instant> = None;

        // Per-index state management (for n>1 support)
        let mut parsers: HashMap<u32, HarmonyParserAdapter> = HashMap::new();
        let mut is_firsts: HashMap<u32, bool> = HashMap::new();
        let mut finish_reasons: HashMap<u32, Option<String>> = HashMap::new();
        let mut matched_stops: HashMap<u32, Option<serde_json::Value>> = HashMap::new();
        let mut prompt_tokens: HashMap<u32, u32> = HashMap::new();
        let mut completion_tokens = CompletionTokenTracker::new();

        let stream_options = &original_request.stream_options;

        // Process stream
        while let Some(result) = grpc_stream.next().await {
            let response = result.map_err(|e| format!("Stream error: {}", e))?;

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
                                .map_err(|e| format!("Failed to create parser: {}", e))?,
                        );
                        is_firsts.insert(index, true);
                    }

                    completion_tokens.record_chunk(&chunk_wrapper);

                    // Convert logprobs if present and requested
                    let chunk_logprobs = if original_request.logprobs {
                        chunk_wrapper.output_logprobs().and_then(|lp| {
                            convert_harmony_logprobs(&lp)
                                .map_err(|e| error!("Failed to convert streaming logprobs: {}", e))
                                .ok()
                        })
                    } else {
                        None
                    };

                    // Parse chunk via Harmony parser
                    let parser = parsers
                        .get_mut(&index)
                        .ok_or("Parser not found for index")?;

                    let delta_result = parser
                        .parse_chunk(chunk_wrapper.token_ids())
                        .map_err(|e| format!("Parse error: {}", e))?;

                    // Emit SSE event if there's a delta
                    if let Some(delta) = delta_result {
                        let is_first = is_firsts.get(&index).copied().unwrap_or(false);
                        Self::emit_chunk_delta(
                            &delta,
                            index,
                            is_first,
                            &dispatch,
                            &original_request,
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
                    finish_reasons
                        .insert(index, Some(complete_wrapper.finish_reason().to_string()));
                    matched_stops.insert(index, complete_wrapper.matched_stop_json());
                    prompt_tokens.insert(index, complete_wrapper.prompt_tokens());
                    completion_tokens.record_complete(&complete_wrapper);

                    // Finalize parser and emit final chunk
                    if let Some(parser) = parsers.get_mut(&index) {
                        let matched_stop = matched_stops.get(&index).and_then(|m| m.clone());
                        let final_output = parser
                            .finalize(
                                complete_wrapper.finish_reason().to_string(),
                                matched_stop.clone(),
                            )
                            .map_err(|e| format!("Finalize error: {}", e))?;

                        Self::emit_final_chunk(
                            index,
                            &final_output.finish_reason,
                            final_output.matched_stop.as_ref(),
                            &dispatch,
                            &original_request,
                            tx,
                        )?;
                    }
                }
                ProtoResponseVariant::Error(error_wrapper) => {
                    return Err(format!("Server error: {}", error_wrapper.message()));
                }
                ProtoResponseVariant::None => {}
            }
        }

        // Compute totals once for both usage chunk and metrics
        let total_prompt: u32 = prompt_tokens.values().sum();
        let total_completion: u32 = completion_tokens.total();

        // Emit final usage if requested
        if let Some(true) = stream_options.as_ref().and_then(|so| so.include_usage) {
            Self::emit_usage_chunk(
                total_prompt,
                total_completion,
                &dispatch,
                &original_request,
                tx,
            )?;
        }

        // Mark stream as completed successfully to prevent abort on drop
        grpc_stream.mark_completed();

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

    /// Process streaming chunks from dual streams (prefill + decode)
    async fn process_dual_stream(
        mut prefill_stream: ProtoStream,
        mut decode_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        original_request: Arc<ChatCompletionRequest>,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        // Timing for metrics
        let start_time = Instant::now();
        let mut first_token_time: Option<Instant> = None;

        // Phase 1: Process prefill stream (collect metadata)
        let mut prompt_tokens: HashMap<u32, u32> = HashMap::new();

        while let Some(result) = prefill_stream.next().await {
            let response = result.map_err(|e| format!("Prefill stream error: {}", e))?;

            if let ProtoResponseVariant::Complete(complete_wrapper) = response.into_response() {
                prompt_tokens.insert(complete_wrapper.index(), complete_wrapper.prompt_tokens());
            }
        }

        // Phase 2: Process decode stream (same as single stream)
        let mut parsers: HashMap<u32, HarmonyParserAdapter> = HashMap::new();
        let mut is_firsts: HashMap<u32, bool> = HashMap::new();
        let mut finish_reasons: HashMap<u32, Option<String>> = HashMap::new();
        let mut matched_stops: HashMap<u32, Option<serde_json::Value>> = HashMap::new();
        let mut completion_tokens = CompletionTokenTracker::new();

        let stream_options = &original_request.stream_options;

        while let Some(result) = decode_stream.next().await {
            let response = result.map_err(|e| format!("Decode stream error: {}", e))?;

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
                                .map_err(|e| format!("Failed to create parser: {}", e))?,
                        );
                        is_firsts.insert(index, true);
                    }

                    completion_tokens.record_chunk(&chunk_wrapper);

                    // Convert logprobs if present and requested
                    let chunk_logprobs = if original_request.logprobs {
                        chunk_wrapper.output_logprobs().and_then(|lp| {
                            convert_harmony_logprobs(&lp)
                                .map_err(|e| error!("Failed to convert streaming logprobs: {}", e))
                                .ok()
                        })
                    } else {
                        None
                    };

                    let parser = parsers
                        .get_mut(&index)
                        .ok_or("Parser not found for index")?;

                    let delta_result = parser
                        .parse_chunk(chunk_wrapper.token_ids())
                        .map_err(|e| format!("Parse error: {}", e))?;

                    if let Some(delta) = delta_result {
                        let is_first = is_firsts.get(&index).copied().unwrap_or(false);
                        Self::emit_chunk_delta(
                            &delta,
                            index,
                            is_first,
                            &dispatch,
                            &original_request,
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

                    finish_reasons
                        .insert(index, Some(complete_wrapper.finish_reason().to_string()));
                    matched_stops.insert(index, complete_wrapper.matched_stop_json());
                    completion_tokens.record_complete(&complete_wrapper);

                    if let Some(parser) = parsers.get_mut(&index) {
                        let matched_stop = matched_stops.get(&index).and_then(|m| m.clone());
                        let final_output = parser
                            .finalize(
                                complete_wrapper.finish_reason().to_string(),
                                matched_stop.clone(),
                            )
                            .map_err(|e| format!("Finalize error: {}", e))?;

                        Self::emit_final_chunk(
                            index,
                            &final_output.finish_reason,
                            final_output.matched_stop.as_ref(),
                            &dispatch,
                            &original_request,
                            tx,
                        )?;
                    }
                }
                ProtoResponseVariant::Error(error_wrapper) => {
                    return Err(format!("Server error: {}", error_wrapper.message()));
                }
                ProtoResponseVariant::None => {}
            }
        }

        decode_stream.mark_completed();

        // Mark prefill stream as completed AFTER decode completes successfully
        // This ensures that if client disconnects during decode, BOTH streams send abort
        prefill_stream.mark_completed();

        // Compute totals once for both usage chunk and metrics
        let total_prompt: u32 = prompt_tokens.values().sum();
        let total_completion: u32 = completion_tokens.total();

        // Emit final usage if requested
        if let Some(true) = stream_options.as_ref().and_then(|so| so.include_usage) {
            Self::emit_usage_chunk(
                total_prompt,
                total_completion,
                &dispatch,
                &original_request,
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
                .map_err(|e| format!("JSON serialization error: {}", e))?;
            let sse_data = format!("data: {}\n\n", chunk_json);

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

        let chunk_json = serde_json::to_string(&chunk)
            .map_err(|e| format!("JSON serialization error: {}", e))?;
        let sse_data = format!("data: {}\n\n", chunk_json);

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

        let chunk_json = serde_json::to_string(&chunk)
            .map_err(|e| format!("JSON serialization error: {}", e))?;
        let sse_data = format!("data: {}\n\n", chunk_json);

        tx.send(Ok(Bytes::from(sse_data)))
            .map_err(|_| "Failed to send final chunk".to_string())?;

        Ok(())
    }

    /// Emit usage chunk at the end
    fn emit_usage_chunk(
        prompt_tokens: u32,
        completion_tokens: u32,
        dispatch: &context::DispatchMetadata,
        original_request: &ChatCompletionRequest,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        let usage_chunk =
            ChatCompletionStreamResponse::builder(&dispatch.request_id, &original_request.model)
                .created(dispatch.created)
                .usage(Usage::from_counts(prompt_tokens, completion_tokens))
                .maybe_system_fingerprint(dispatch.weight_version.as_deref())
                .build();

        let chunk_json = serde_json::to_string(&usage_chunk)
            .map_err(|e| format!("JSON serialization error: {}", e))?;
        let sse_data = format!("data: {}\n\n", chunk_json);

        tx.send(Ok(Bytes::from(sse_data)))
            .map_err(|_| "Failed to send usage chunk".to_string())?;

        Ok(())
    }

    /// Process streaming chunks for Responses API iteration.
    ///
    /// When MCP context is provided (orchestrator, mcp_servers, mcp_tool_names):
    /// - MCP tools with `ResponseFormat::WebSearchCall` → `web_search_call.*` events
    /// - Other MCP tools → `mcp_call.*` events
    /// - Function tools (not in mcp_tool_names) → `function_call.*` events
    ///
    /// When no MCP context is provided, all tool calls are treated as function calls.
    pub async fn process_responses_iteration_stream(
        execution_result: context::ExecutionResult,
        emitter: &mut ResponseStreamEventEmitter,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        orchestrator: Option<&Arc<McpOrchestrator>>,
        mcp_servers: Option<&[(String, String)]>,
        mcp_tool_names: Option<&HashSet<String>>,
    ) -> Result<ResponsesIterationResult, String> {
        match execution_result {
            context::ExecutionResult::Single { stream } => {
                debug!("Processing Responses API single stream mode");
                Self::process_decode_stream(
                    stream,
                    emitter,
                    tx,
                    orchestrator,
                    mcp_servers,
                    mcp_tool_names,
                )
                .await
            }
            context::ExecutionResult::Dual { prefill, decode } => {
                debug!("Processing Responses API dual stream mode");
                Self::process_responses_dual_stream(
                    prefill,
                    *decode,
                    emitter,
                    tx,
                    orchestrator,
                    mcp_servers,
                    mcp_tool_names,
                )
                .await
            }
            context::ExecutionResult::Embedding { .. } => {
                Err("Embeddings not supported in Responses API streaming".to_string())
            }
        }
    }

    async fn process_responses_dual_stream(
        mut prefill_stream: ProtoStream,
        decode_stream: ProtoStream,
        emitter: &mut ResponseStreamEventEmitter,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        orchestrator: Option<&Arc<McpOrchestrator>>,
        mcp_servers: Option<&[(String, String)]>,
        mcp_tool_names: Option<&HashSet<String>>,
    ) -> Result<ResponsesIterationResult, String> {
        // Phase 1: Drain prefill stream
        while let Some(result) = prefill_stream.next().await {
            let _response = result.map_err(|e| format!("Prefill stream error: {}", e))?;
        }

        // Phase 2: Process decode stream
        let result = Self::process_decode_stream(
            decode_stream,
            emitter,
            tx,
            orchestrator,
            mcp_servers,
            mcp_tool_names,
        )
        .await;

        prefill_stream.mark_completed();
        result
    }

    /// Process decode stream for tool call events.
    async fn process_decode_stream(
        mut decode_stream: ProtoStream,
        emitter: &mut ResponseStreamEventEmitter,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        orchestrator: Option<&Arc<McpOrchestrator>>,
        mcp_servers: Option<&[(String, String)]>,
        mcp_tool_names: Option<&HashSet<String>>,
    ) -> Result<ResponsesIterationResult, String> {
        let empty_servers: &[(String, String)] = &[];
        let mcp_servers = mcp_servers.unwrap_or(empty_servers);
        let server_keys: Vec<String> = mcp_servers.iter().map(|(_, key)| key.clone()).collect();
        let mut parser =
            HarmonyParserAdapter::new().map_err(|e| format!("Failed to create parser: {}", e))?;

        let mut has_analysis = false;
        let mut accumulated_final_text = String::new();
        let mut accumulated_tool_calls: Option<Vec<ToolCall>> = None;

        let mut has_emitted_reasoning = false;
        let mut message_output_index: Option<usize> = None;
        let mut message_item_id: Option<String> = None;
        let mut has_emitted_content_part_added = false;

        // Tool call tracking: call_index -> (output_index, item_id, response_format)
        let mut tool_call_tracking: HashMap<usize, (usize, String, Option<ResponseFormat>)> =
            HashMap::new();

        // Metadata from Complete message
        let mut finish_reason = String::from("stop");
        let mut matched_stop: Option<serde_json::Value> = None;
        let mut prompt_tokens: u32 = 0;
        let mut completion_tokens: u32 = 0;
        let mut reasoning_token_count: u32 = 0;

        // Process stream
        let mut chunk_count = 0;
        while let Some(result) = decode_stream.next().await {
            chunk_count += 1;
            let response = result.map_err(|e| format!("Decode stream error: {}", e))?;

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
                        .map_err(|e| format!("Parse error: {}", e))?;

                    // Emit SSE events if there's a delta
                    if let Some(delta) = delta_result {
                        // Analysis channel → Reasoning item (wrapper events only, emitted once)
                        if let Some(_analysis_text) = &delta.analysis_delta {
                            if !has_emitted_reasoning {
                                // Emit reasoning item (added + done in one call)
                                // Note: reasoning_content will be provided at finalize
                                emitter
                                    .emit_reasoning_item(tx, None)
                                    .map_err(|e| format!("Failed to emit reasoning item: {}", e))?;

                                has_emitted_reasoning = true;
                                has_analysis = true;
                            }
                        }

                        // Final channel → Message item (WITH text streaming)
                        if let Some(final_delta) = &delta.final_delta {
                            if !final_delta.is_empty() {
                                // Allocate message item if needed
                                if message_output_index.is_none() {
                                    let (output_index, item_id) =
                                        emitter.allocate_output_index(OutputItemType::Message);
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
                                    let event = emitter.emit_output_item_added(output_index, &item);
                                    emitter.send_event_best_effort(&event, tx);
                                }

                                let output_index = message_output_index.unwrap();
                                let item_id = message_item_id.as_ref().unwrap();
                                let content_index = 0; // Single content part

                                // Emit content_part.added before first delta
                                if !has_emitted_content_part_added {
                                    let event = emitter.emit_content_part_added(
                                        output_index,
                                        item_id,
                                        content_index,
                                    );
                                    emitter.send_event_best_effort(&event, tx);
                                    has_emitted_content_part_added = true;
                                }

                                // Emit text delta
                                let event = emitter.emit_text_delta(
                                    final_delta,
                                    output_index,
                                    item_id,
                                    content_index,
                                );
                                emitter.send_event_best_effort(&event, tx);

                                accumulated_final_text.push_str(final_delta);
                            }
                        }

                        // Commentary channel → Tool call streaming
                        if let Some(tc_delta) = &delta.commentary_delta {
                            let call_index = tc_delta.index;

                            // New tool call (has id and name)
                            if let Some(call_id) = &tc_delta.id {
                                let tool_name = tc_delta
                                    .function
                                    .as_ref()
                                    .and_then(|f| f.name.as_ref())
                                    .map(|n| n.as_str())
                                    .unwrap_or("");

                                // Determine response_format based on MCP context
                                let response_format = if let Some(names) = mcp_tool_names {
                                    if names.contains(tool_name) {
                                        Some(
                                            orchestrator
                                                .and_then(|o| {
                                                    o.find_tool_by_name(tool_name, &server_keys)
                                                })
                                                .map(|e| e.response_format.clone())
                                                .unwrap_or(ResponseFormat::Passthrough),
                                        )
                                    } else {
                                        None // Function tool
                                    }
                                } else {
                                    None // No MCP context, treat as function tool
                                };

                                // Determine output item type and JSON type string
                                let output_item_type =
                                    ResponseStreamEventEmitter::output_item_type_for_format(
                                        response_format.as_ref(),
                                    );
                                let type_str = ResponseStreamEventEmitter::type_str_for_format(
                                    response_format.as_ref(),
                                );

                                let (output_index, item_id) =
                                    emitter.allocate_output_index(output_item_type);

                                tool_call_tracking.insert(
                                    call_index,
                                    (output_index, item_id.clone(), response_format.clone()),
                                );

                                // Build output_item.added event
                                let mut item = json!({
                                    "id": item_id,
                                    "type": type_str,
                                    "name": tool_name,
                                    "call_id": call_id,
                                    "arguments": "",
                                    "status": "in_progress"
                                });

                                let label = resolve_tool_server_label(
                                    orchestrator.map(|o| o.as_ref()),
                                    tool_name,
                                    mcp_servers,
                                    &server_keys,
                                );
                                attach_mcp_server_label(
                                    &mut item,
                                    Some(label.as_str()),
                                    response_format.as_ref(),
                                );

                                let event = emitter.emit_output_item_added(output_index, &item);
                                emitter.send_event_best_effort(&event, tx);

                                // Emit in_progress event for MCP tools
                                if let Some(ref fmt) = response_format {
                                    let event = emitter.emit_tool_call_in_progress(
                                        output_index,
                                        &item_id,
                                        fmt,
                                    );
                                    emitter.send_event_best_effort(&event, tx);

                                    // Emit searching/interpreting event for builtin tools
                                    if let Some(event) = emitter.emit_tool_call_searching(
                                        output_index,
                                        &item_id,
                                        fmt,
                                    ) {
                                        emitter.send_event_best_effort(&event, tx);
                                    }
                                }

                                // Emit initial arguments delta for mcp_call only (skip for builtin tools)
                                if matches!(
                                    response_format,
                                    Some(ResponseFormat::Passthrough) | None
                                ) {
                                    let event = match &response_format {
                                        Some(_) => emitter.emit_mcp_call_arguments_delta(
                                            output_index,
                                            &item_id,
                                            "",
                                        ),
                                        None => emitter.emit_function_call_arguments_delta(
                                            output_index,
                                            &item_id,
                                            "",
                                        ),
                                    };
                                    emitter.send_event_best_effort(&event, tx);
                                }
                            } else {
                                // Continuing tool call: emit arguments delta
                                if let Some((output_index, item_id, response_format)) =
                                    tool_call_tracking.get(&call_index)
                                {
                                    // Skip arguments streaming for builtin tools (only mcp_call streams arguments)
                                    if !matches!(
                                        response_format,
                                        Some(ResponseFormat::Passthrough) | None
                                    ) {
                                        continue;
                                    }

                                    if let Some(args) = tc_delta
                                        .function
                                        .as_ref()
                                        .and_then(|f| f.arguments.as_ref())
                                        .filter(|a| !a.is_empty())
                                    {
                                        let event = match response_format {
                                            Some(_) => emitter.emit_mcp_call_arguments_delta(
                                                *output_index,
                                                item_id,
                                                args,
                                            ),
                                            None => emitter.emit_function_call_arguments_delta(
                                                *output_index,
                                                item_id,
                                                args,
                                            ),
                                        };
                                        emitter.send_event_best_effort(&event, tx);
                                    }
                                }
                            }
                        }
                    }
                }
                ProtoResponseVariant::Complete(complete_wrapper) => {
                    // Store final metadata
                    finish_reason = complete_wrapper.finish_reason().to_string();
                    matched_stop = complete_wrapper.matched_stop_json();
                    prompt_tokens = complete_wrapper.prompt_tokens();
                    // For vLLM, use accumulated count (we tracked deltas above)
                    // For SGLang, use complete value (already cumulative)
                    if !complete_wrapper.is_vllm() {
                        completion_tokens = complete_wrapper.completion_tokens();
                    }

                    // Finalize parser and get complete output
                    let final_output = parser
                        .finalize(finish_reason.clone(), matched_stop.clone())
                        .map_err(|e| format!("Finalize error: {}", e))?;

                    // Store finalized tool calls and reasoning token count
                    accumulated_tool_calls = final_output.commentary.clone();
                    reasoning_token_count = final_output.reasoning_token_count;

                    // Complete all tool calls if we have commentary
                    if let Some(ref tool_calls) = accumulated_tool_calls {
                        for (call_idx, tool_call) in tool_calls.iter().enumerate() {
                            if let Some((output_index, item_id, response_format)) =
                                tool_call_tracking.get(&call_idx)
                            {
                                let tool_name = &tool_call.function.name;
                                let args_str =
                                    tool_call.function.arguments.as_deref().unwrap_or("");

                                // Emit arguments done (skip for builtin tools)
                                if matches!(
                                    response_format,
                                    Some(ResponseFormat::Passthrough) | None
                                ) {
                                    let event = match response_format {
                                        Some(_) => emitter.emit_mcp_call_arguments_done(
                                            *output_index,
                                            item_id,
                                            args_str,
                                        ),
                                        None => emitter.emit_function_call_arguments_done(
                                            *output_index,
                                            item_id,
                                            args_str,
                                        ),
                                    };
                                    emitter.send_event_best_effort(&event, tx);
                                }

                                // Emit completed event for MCP tools
                                if let Some(ref fmt) = response_format {
                                    let event = emitter.emit_tool_call_completed(
                                        *output_index,
                                        item_id,
                                        fmt,
                                    );
                                    emitter.send_event_best_effort(&event, tx);
                                }

                                // Determine type string for JSON
                                let type_str = ResponseStreamEventEmitter::type_str_for_format(
                                    response_format.as_ref(),
                                );

                                let mut item = json!({
                                    "id": item_id,
                                    "type": type_str,
                                    "name": tool_name,
                                    "call_id": &tool_call.id,
                                    "arguments": args_str,
                                    "status": "completed"
                                });

                                let label = resolve_tool_server_label(
                                    orchestrator.map(|o| o.as_ref()),
                                    tool_name,
                                    mcp_servers,
                                    &server_keys,
                                );
                                attach_mcp_server_label(
                                    &mut item,
                                    Some(label.as_str()),
                                    response_format.as_ref(),
                                );

                                let event = emitter.emit_output_item_done(*output_index, &item);
                                emitter.complete_output_item(*output_index);
                                emitter.send_event_best_effort(&event, tx);
                            }
                        }
                    }

                    // Close message item if we opened one
                    if let Some(output_index) = message_output_index {
                        let item_id = message_item_id.as_ref().unwrap();
                        let content_index = 0;

                        // Emit text_done
                        let event = emitter.emit_text_done(output_index, item_id, content_index);
                        emitter.send_event_best_effort(&event, tx);

                        // Emit content_part.done
                        let event =
                            emitter.emit_content_part_done(output_index, item_id, content_index);
                        emitter.send_event_best_effort(&event, tx);

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
                        let event = emitter.emit_output_item_done(output_index, &item);

                        // Mark as completed before sending (so it's included in final output even if send fails)
                        emitter.complete_output_item(output_index);

                        emitter.send_event_best_effort(&event, tx);
                    }
                }
                ProtoResponseVariant::Error(error_wrapper) => {
                    return Err(format!("Server error: {}", error_wrapper.message()));
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
            accumulated_tool_calls = commentary_opt.clone();

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

            // Complete any pending tool calls with data from completed messages
            if let Some(ref tool_calls) = accumulated_tool_calls {
                for (call_idx, tool_call) in tool_calls.iter().enumerate() {
                    if let Some((output_index, item_id, response_format)) =
                        tool_call_tracking.get(&call_idx)
                    {
                        let tool_name = &tool_call.function.name;
                        let args_str = tool_call.function.arguments.as_deref().unwrap_or("");

                        // Emit arguments done (skip for builtin tools)
                        if matches!(response_format, Some(ResponseFormat::Passthrough) | None) {
                            let event = match response_format {
                                Some(_) => emitter.emit_mcp_call_arguments_done(
                                    *output_index,
                                    item_id,
                                    args_str,
                                ),
                                None => emitter.emit_function_call_arguments_done(
                                    *output_index,
                                    item_id,
                                    args_str,
                                ),
                            };
                            emitter.send_event_best_effort(&event, tx);
                        }

                        // Emit completed event for MCP tools
                        if let Some(ref fmt) = response_format {
                            let event =
                                emitter.emit_tool_call_completed(*output_index, item_id, fmt);
                            emitter.send_event_best_effort(&event, tx);
                        }

                        let type_str = ResponseStreamEventEmitter::type_str_for_format(
                            response_format.as_ref(),
                        );

                        let mut item = json!({
                            "id": item_id,
                            "type": type_str,
                            "name": tool_name,
                            "call_id": &tool_call.id,
                            "arguments": args_str,
                            "status": "completed"
                        });

                        let label = resolve_tool_server_label(
                            orchestrator.map(|o| o.as_ref()),
                            tool_name,
                            mcp_servers,
                            &server_keys,
                        );
                        attach_mcp_server_label(
                            &mut item,
                            Some(label.as_str()),
                            response_format.as_ref(),
                        );

                        let event = emitter.emit_output_item_done(*output_index, &item);
                        emitter.complete_output_item(*output_index);
                        emitter.send_event_best_effort(&event, tx);
                    }
                }
            }
        }

        // Mark stream as completed successfully to prevent abort on drop
        decode_stream.mark_completed();

        // Return result based on whether tool calls were found
        if let Some(tool_calls) = accumulated_tool_calls {
            if !tool_calls.is_empty() {
                let analysis_content = if has_analysis {
                    // Get analysis from finalized parser output by calling finalize again
                    // This is safe because finalize can be called multiple times
                    let output = parser.finalize(finish_reason.clone(), matched_stop.clone())?;
                    output.analysis
                } else {
                    None
                };

                return Ok(ResponsesIterationResult::ToolCallsFound {
                    tool_calls,
                    analysis: analysis_content,
                    partial_text: accumulated_final_text,
                    usage: Usage::from_counts(prompt_tokens, completion_tokens)
                        .with_reasoning_tokens(reasoning_token_count),
                    request_id: emitter.response_id.clone(),
                });
            }
        }

        // For streaming, we don't build the full ResponsesResponse here
        // The caller will build it from the SSE events
        // Return a placeholder Completed result (caller ignores these fields in streaming mode)
        Ok(ResponsesIterationResult::Completed {
            response: Box::new(
                ResponsesResponse::builder(&emitter.response_id, "")
                    .status(ResponseStatus::Completed)
                    .usage(ResponsesUsage::Modern(ResponseUsage {
                        input_tokens: prompt_tokens,
                        output_tokens: completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                        input_tokens_details: None,
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
                .with_reasoning_tokens(reasoning_token_count),
        })
    }
}

impl Default for HarmonyStreamingProcessor {
    fn default() -> Self {
        Self::new()
    }
}
