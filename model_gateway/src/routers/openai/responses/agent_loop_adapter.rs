//! Shared-agent-loop adapters for the OpenAI Responses router.

use std::{
    collections::{HashMap, HashSet},
    io,
    sync::Arc,
};

use async_trait::async_trait;
use axum::http::{HeaderMap, StatusCode};
use bytes::Bytes;
use futures_util::StreamExt;
use openai_protocol::{
    common::{CompletionTokensDetails, PromptTokenUsageInfo, Usage},
    event_types::ResponseEvent,
    responses::{
        ResponseContentPart, ResponseInputOutputItem, ResponseOutputItem, ResponseStatus,
        ResponseTool, ResponsesRequest, ResponsesResponse,
    },
};
use serde_json::{to_value, Value};
use smg_mcp::McpToolSession;
use tokio::sync::mpsc;

use crate::{
    routers::{
        common::{
            agent_loop::{
                build_response_from_state, build_responses_iteration_request,
                normalize_output_item_id, AgentLoopAdapter, AgentLoopContext, AgentLoopError,
                AgentLoopState, ExecutedCall, IterationRequestFlavor, LoopEvent, LoopModelTurn,
                LoopToolCall, OutputFamily, PendingToolExecution, RenderMode, ResponseBuildHooks,
                StreamSink, ToolPresentation, ToolTransferDescriptor, ToolVisibility, UsageShape,
            },
            header_utils::ApiProvider,
            mcp_utils::collect_user_function_names,
            responses_streaming::{GatewayToolCompletion, ResponseStreamEventEmitter},
        },
        error,
        openai::{
            context::StreamingEventContext,
            provider::Provider,
            responses::{
                common::{parse_sse_block, ChunkProcessor},
                streaming::{forward_streaming_event, SseEventData},
                upstream_stream_parser::{
                    OpenAiUpstreamStreamParser, ParsedFunctionCall, StreamParserAction,
                },
            },
        },
    },
    worker::Endpoint,
};

#[derive(Clone)]
pub(crate) struct OpenAiUpstreamHandle {
    pub client: reqwest::Client,
    pub url: String,
    pub headers: Option<HeaderMap>,
    pub api_key: Option<String>,
    pub provider: Arc<dyn Provider>,
}

impl OpenAiUpstreamHandle {
    async fn post_json(&self, payload: &Value) -> Result<Value, AgentLoopError> {
        let mut request_builder = self.client.post(&self.url).json(payload);
        let provider = ApiProvider::from_url(&self.url);
        let auth_header =
            provider.extract_auth_header(self.headers.as_ref(), self.api_key.as_ref());
        request_builder = provider.apply_headers(request_builder, auth_header.as_ref());

        let response = request_builder.send().await.map_err(|e| {
            AgentLoopError::Upstream(format!("Failed to forward request to OpenAI: {e}"))
        })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            let body = error::sanitize_error_body(&body);
            // Forward the upstream status (400/429/5xx) instead of
            // collapsing into 502 so callers can react accordingly.
            let preserved_status = if status.is_client_error() || status.is_server_error() {
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY)
            } else {
                StatusCode::BAD_GATEWAY
            };
            let response = error::create_error(
                preserved_status,
                "upstream_error",
                format!("upstream error {status}: {body}"),
            );
            return Err(AgentLoopError::Response(Box::new(response)));
        }

        response.json::<Value>().await.map_err(|e| {
            AgentLoopError::Internal(format!("Failed to parse upstream response: {e}"))
        })
    }
}

pub(crate) struct OpenAiNonStreamingAdapter {
    upstream: OpenAiUpstreamHandle,
    iteration_request: ResponsesRequest,
    upstream_tools: Option<Vec<ResponseTool>>,
    original_tools: Option<Vec<ResponseTool>>,
    user_function_names: HashSet<String>,
    completed_response: Option<ResponsesResponse>,
}

impl OpenAiNonStreamingAdapter {
    pub(crate) fn new(
        iteration_request: &ResponsesRequest,
        echo_request: &ResponsesRequest,
        session: &McpToolSession<'_>,
        upstream: OpenAiUpstreamHandle,
    ) -> Self {
        Self {
            upstream,
            iteration_request: iteration_request.clone(),
            upstream_tools: build_openai_upstream_tools(iteration_request, session),
            original_tools: echo_request.tools.clone(),
            user_function_names: collect_user_function_names(echo_request),
            completed_response: None,
        }
    }
}

fn build_openai_upstream_tools(
    request: &ResponsesRequest,
    session: &McpToolSession<'_>,
) -> Option<Vec<ResponseTool>> {
    let original_tools = request.tools.clone().unwrap_or_default();
    if session.mcp_tools().is_empty() {
        return (!original_tools.is_empty()).then_some(original_tools);
    }

    let mut tools: Vec<ResponseTool> = original_tools
        .into_iter()
        .filter(|tool| matches!(tool, ResponseTool::Function(_)))
        .collect();
    tools.extend(session.build_response_tools());
    (!tools.is_empty()).then_some(tools)
}

fn build_iteration_payload(
    original_request: &ResponsesRequest,
    upstream_tools: Option<Vec<ResponseTool>>,
    state: &AgentLoopState,
    provider: &dyn Provider,
    stream: bool,
) -> Result<Value, AgentLoopError> {
    let typed_request = build_responses_iteration_request(
        original_request,
        state,
        IterationRequestFlavor::Responses {
            stream: Some(stream),
            tools: upstream_tools,
        },
    );
    let mut payload = to_value(&typed_request).map_err(|e| {
        AgentLoopError::Internal(format!("Failed to serialize iteration request: {e}"))
    })?;

    sanitize_openai_replay_input(&mut payload);

    provider
        .transform_request(&mut payload, Endpoint::Responses)
        .map_err(|e| AgentLoopError::InvalidRequest(format!("Provider transform error: {e}")))?;

    Ok(payload)
}

fn sanitize_openai_replay_input(payload: &mut Value) {
    let Some(input) = payload
        .as_object_mut()
        .and_then(|obj| obj.get_mut("input"))
        .and_then(Value::as_array_mut)
    else {
        return;
    };

    for item in input.iter_mut().filter_map(Value::as_object_mut) {
        if item.get("type").and_then(Value::as_str) == Some("reasoning") {
            // OpenAI accepts replayed reasoning items, but not the gateway's
            // output-side fields. In-loop calls use `store: false`, so a
            // retained `rs_*` id would make OpenAI look up a non-persisted item.
            // Keep `summary` / `encrypted_content` because those are valid
            // reasoning replay inputs.
            item.remove("id");
            item.remove("status");
            item.remove("content");
        }
    }
}

fn extract_pending_tool_calls(response: &ResponsesResponse) -> Vec<LoopToolCall> {
    response
        .output
        .iter()
        .filter_map(|item| match item {
            ResponseOutputItem::FunctionToolCall {
                id,
                call_id,
                name,
                arguments,
                ..
            } => Some(LoopToolCall {
                call_id: call_id.clone(),
                item_id: id.clone(),
                name: name.clone(),
                arguments: arguments.clone(),
                approval_request_id: None,
            }),
            _ => None,
        })
        .collect()
}

fn response_output_to_transcript(response: &ResponsesResponse) -> Vec<ResponseInputOutputItem> {
    response
        .output
        .iter()
        .filter(|item| !matches!(item, ResponseOutputItem::FunctionToolCall { .. }))
        .filter_map(|item| {
            to_value(item)
                .ok()
                .and_then(|value| serde_json::from_value::<ResponseInputOutputItem>(value).ok())
        })
        .collect()
}

fn extract_message_text(response: &ResponsesResponse) -> Option<String> {
    let mut parts = Vec::new();
    for item in &response.output {
        if let ResponseOutputItem::Message { role, content, .. } = item {
            if role != "assistant" {
                continue;
            }
            for part in content {
                if let ResponseContentPart::OutputText { text, .. } = part {
                    parts.push(text.clone());
                }
            }
        }
    }
    (!parts.is_empty()).then(|| parts.join("\n"))
}

fn extract_reasoning_text(response: &ResponsesResponse) -> Option<String> {
    let mut parts = Vec::new();
    for item in &response.output {
        if let ResponseOutputItem::Reasoning { content, .. } = item {
            for part in content {
                let openai_protocol::responses::ResponseReasoningContent::ReasoningText { text } =
                    part;
                parts.push(text.clone());
            }
        }
    }
    (!parts.is_empty()).then(|| parts.join("\n"))
}

fn extract_usage(response: &ResponsesResponse) -> Option<Usage> {
    response.usage.as_ref().map(|usage| {
        let usage = usage.to_response_usage();
        Usage {
            prompt_tokens: usage.input_tokens,
            completion_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
            prompt_tokens_details: usage.input_tokens_details.as_ref().map(|details| {
                PromptTokenUsageInfo {
                    cached_tokens: details.cached_tokens,
                }
            }),
            completion_tokens_details: usage.output_tokens_details.as_ref().map(|details| {
                CompletionTokensDetails {
                    reasoning_tokens: Some(details.reasoning_tokens),
                    accepted_prediction_tokens: None,
                    rejected_prediction_tokens: None,
                }
            }),
        }
    })
}

#[async_trait]
impl<S: StreamSink> AgentLoopAdapter<S> for OpenAiNonStreamingAdapter {
    type FinalResponse = ResponsesResponse;

    async fn call_upstream(
        &mut self,
        ctx: &AgentLoopContext<'_>,
        state: &mut AgentLoopState,
        _sink: &mut S,
    ) -> Result<(), AgentLoopError> {
        if ctx.session.is_none() {
            return Err(AgentLoopError::Internal(
                "OpenAI adapter missing MCP session".to_string(),
            ));
        }
        let payload = build_iteration_payload(
            &self.iteration_request,
            self.upstream_tools.clone(),
            state,
            self.upstream.provider.as_ref(),
            false,
        )?;
        let response_json = self.upstream.post_json(&payload).await?;
        let response: ResponsesResponse = serde_json::from_value(response_json).map_err(|e| {
            AgentLoopError::Internal(format!(
                "Failed to deserialize upstream ResponsesResponse: {e}"
            ))
        })?;

        let pending_tool_calls = extract_pending_tool_calls(&response);
        let has_tool_calls = !pending_tool_calls.is_empty();
        state.pending_tool_calls = pending_tool_calls;
        state.latest_turn = Some(LoopModelTurn {
            message_text: extract_message_text(&response),
            reasoning_text: extract_reasoning_text(&response),
            usage: extract_usage(&response),
            request_id: Some(response.id.clone()),
        });
        state
            .transcript
            .extend(response_output_to_transcript(&response));

        if has_tool_calls {
            self.completed_response = None;
        } else {
            self.completed_response = Some(response);
        }

        Ok(())
    }

    async fn render_final(
        mut self,
        ctx: &AgentLoopContext<'_>,
        state: AgentLoopState,
        mode: RenderMode,
        _sink: &mut S,
    ) -> Result<Self::FinalResponse, AgentLoopError> {
        let session = ctx.session.ok_or_else(|| {
            AgentLoopError::Internal("OpenAI render_final called without MCP session".to_string())
        })?;
        let hooks = ResponseBuildHooks {
            original_tools: self.original_tools.take(),
            user_function_names: self.user_function_names,
            response_id_override: None,
            usage_shape: UsageShape::Modern,
        };
        Ok(build_response_from_state(
            &state,
            self.completed_response.take(),
            mode,
            ctx.original_request,
            &hooks,
            session,
        ))
    }
}

struct RegisteredGatewayCall {
    output_index: usize,
    item_id: String,
    presentation: ToolPresentation,
    server_label: String,
}

pub(crate) struct OpenAiResponseStreamSink {
    tx: mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    emitter: ResponseStreamEventEmitter,
    disconnected: bool,
    buffered_mcp_list_tools: Vec<ResponseOutputItem>,
    registered_gateway_calls: HashMap<String, RegisteredGatewayCall>,
    tool_transfers: HashMap<String, ToolTransferDescriptor>,
    tool_server_labels: HashMap<String, String>,
}

impl OpenAiResponseStreamSink {
    pub(crate) fn new(
        tx: mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        emitter: ResponseStreamEventEmitter,
        tool_transfers: HashMap<String, ToolTransferDescriptor>,
        tool_server_labels: HashMap<String, String>,
    ) -> Self {
        Self {
            tx,
            emitter,
            disconnected: false,
            buffered_mcp_list_tools: Vec::new(),
            registered_gateway_calls: HashMap::new(),
            tool_transfers,
            tool_server_labels,
        }
    }

    pub(crate) fn next_output_index(&self) -> usize {
        self.emitter.next_output_index()
    }

    pub(crate) fn set_next_output_index(&mut self, next_output_index: usize) {
        self.emitter.advance_next_output_index_to(next_output_index);
    }

    fn transfer_descriptor(&self, name: &str) -> ToolTransferDescriptor {
        ToolTransferDescriptor::lookup(&self.tool_transfers, name)
    }

    fn server_label_for(&self, name: &str, family: OutputFamily) -> String {
        if matches!(family, OutputFamily::McpCall) {
            self.tool_server_labels
                .get(name)
                .cloned()
                .unwrap_or_default()
        } else {
            String::new()
        }
    }

    pub(crate) fn flush_buffered_mcp_list_tools(&mut self) {
        let items = std::mem::take(&mut self.buffered_mcp_list_tools);
        for item in items {
            self.emit_mcp_list_tools_item(&item);
        }
    }

    pub(crate) fn register_gateway_call(
        &mut self,
        call_id: String,
        output_index: usize,
        item_id: String,
        family: OutputFamily,
        server_label: String,
    ) {
        self.registered_gateway_calls.insert(
            call_id,
            RegisteredGatewayCall {
                output_index,
                item_id,
                presentation: ToolPresentation::from_family(family),
                server_label,
            },
        );
    }

    pub(crate) fn emit_final_response(&mut self, response: &ResponsesResponse) {
        if self.disconnected {
            return;
        }
        let event_type = if response.status == ResponseStatus::Incomplete {
            ResponseEvent::INCOMPLETE
        } else {
            ResponseEvent::COMPLETED
        };
        let payload = serde_json::json!({
            "type": event_type,
            "sequence_number": self.emitter.next_sequence(),
            "response": response,
        });
        self.send(&payload);
    }

    fn send(&mut self, payload: &Value) {
        if self.disconnected {
            return;
        }
        if !self.emitter.send_event_best_effort(payload, &self.tx) {
            self.disconnected = true;
        }
    }

    fn emit_mcp_list_tools_item(&mut self, item: &ResponseOutputItem) {
        if self.disconnected {
            return;
        }
        if self
            .emitter
            .emit_mcp_list_tools_item_lifecycle(item, &self.tx)
            .is_err()
        {
            self.disconnected = true;
        }
    }

    fn emit_gateway_execution_started(&mut self, call_id: &str) {
        let Some(tracking) = self.registered_gateway_calls.get(call_id) else {
            return;
        };
        let family = tracking.presentation.family;
        let output_index = tracking.output_index;
        let item_id = tracking.item_id.clone();
        if let Some(event) = self
            .emitter
            .emit_tool_call_searching(output_index, &item_id, family)
        {
            self.send(&event);
        }
    }

    fn emit_tool_completed(&mut self, executed: &ExecutedCall) {
        let Some(tracking) = self.registered_gateway_calls.remove(&executed.call_id) else {
            return;
        };

        if self
            .emitter
            .emit_gateway_tool_completed_lifecycle(
                GatewayToolCompletion {
                    output_index: tracking.output_index,
                    item_id: &tracking.item_id,
                    executed,
                    presentation: &tracking.presentation,
                    server_label: &tracking.server_label,
                    approval_request_id: None,
                },
                &self.tx,
            )
            .is_err()
        {
            self.disconnected = true;
        }
    }

    fn emit_approval_requested(&mut self, pending: &PendingToolExecution) {
        if self.disconnected {
            return;
        }
        if self
            .emitter
            .emit_approval_request_lifecycle(pending, &self.tx)
            .is_err()
        {
            self.disconnected = true;
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors LoopEvent::ApprovedToolReplay payload 1:1; bundling here would just recreate the event shape in another struct"
    )]
    fn emit_approved_tool_replay(
        &mut self,
        call_id: &str,
        item_id_hint: &str,
        name: &str,
        full_args: &str,
        family: OutputFamily,
        server_label: &str,
        approval_request_id: Option<&str>,
    ) {
        let call = LoopToolCall {
            call_id: call_id.to_string(),
            item_id: item_id_hint.to_string(),
            name: name.to_string(),
            arguments: full_args.to_string(),
            approval_request_id: approval_request_id.map(str::to_string),
        };
        let Ok(handle) =
            self.emitter
                .emit_approved_tool_replay_lifecycle(&call, family, server_label, &self.tx)
        else {
            self.disconnected = true;
            return;
        };

        self.registered_gateway_calls.insert(
            call_id.to_string(),
            RegisteredGatewayCall {
                output_index: handle.output_index,
                item_id: handle.item_id,
                presentation: handle.presentation,
                server_label: server_label.to_string(),
            },
        );
    }
}

impl StreamSink for OpenAiResponseStreamSink {
    fn emit(&mut self, event: LoopEvent<'_>) {
        match event {
            LoopEvent::McpListToolsItem { item } => self.buffered_mcp_list_tools.push(item.clone()),
            LoopEvent::ToolCallExecutionStarted { call_id, .. } => {
                self.emit_gateway_execution_started(call_id);
            }
            LoopEvent::ToolCompleted { executed } => self.emit_tool_completed(executed),
            LoopEvent::ApprovalRequested { pending } => self.emit_approval_requested(pending),
            LoopEvent::ApprovedToolReplay {
                call_id,
                item_id,
                name,
                full_args,
                family,
                server_label,
                approval_request_id,
            } => self.emit_approved_tool_replay(
                call_id,
                item_id,
                name,
                full_args,
                family,
                server_label,
                approval_request_id,
            ),
            LoopEvent::ResponseStarted
            | LoopEvent::ResponseFinished
            | LoopEvent::ResponseIncomplete { .. }
            | LoopEvent::ToolCallEmissionStarted { .. }
            | LoopEvent::ToolCallArgumentsFragment { .. }
            | LoopEvent::ToolCallEmissionDone { .. } => {}
        }
    }
}

pub(crate) struct OpenAiStreamingAdapter {
    upstream: OpenAiUpstreamHandle,
    iteration_request: ResponsesRequest,
    original_request: ResponsesRequest,
    upstream_tools: Option<Vec<ResponseTool>>,
    original_tools: Option<Vec<ResponseTool>>,
    user_function_names: HashSet<String>,
    completed_response: Option<ResponsesResponse>,
    response_id_override: Option<String>,
    previous_response_id: Option<String>,
}

impl OpenAiStreamingAdapter {
    pub(crate) fn new(
        iteration_request: &ResponsesRequest,
        echo_request: &ResponsesRequest,
        session: &McpToolSession<'_>,
        upstream: OpenAiUpstreamHandle,
    ) -> Self {
        Self {
            upstream,
            iteration_request: iteration_request.clone(),
            original_request: echo_request.clone(),
            upstream_tools: build_openai_upstream_tools(iteration_request, session),
            original_tools: echo_request.tools.clone(),
            user_function_names: collect_user_function_names(echo_request),
            completed_response: None,
            response_id_override: None,
            previous_response_id: echo_request.previous_response_id.clone(),
        }
    }

    fn register_gateway_calls(
        pending_calls: &[ParsedFunctionCall],
        sink: &mut OpenAiResponseStreamSink,
    ) {
        for call in pending_calls {
            let descriptor = sink.transfer_descriptor(&call.name);
            if matches!(descriptor.family, OutputFamily::Function)
                || matches!(descriptor.visibility, ToolVisibility::SuppressedForApproval)
            {
                continue;
            }
            let family = descriptor.family;
            let source_id = call.item_id.as_deref().unwrap_or(call.call_id.as_str());
            sink.register_gateway_call(
                call.call_id.clone(),
                call.effective_output_index(),
                normalize_output_item_id(family, source_id),
                family,
                sink.server_label_for(&call.name, family),
            );
        }
    }
}

#[async_trait]
impl AgentLoopAdapter<OpenAiResponseStreamSink> for OpenAiStreamingAdapter {
    type FinalResponse = ResponsesResponse;

    async fn call_upstream(
        &mut self,
        ctx: &AgentLoopContext<'_>,
        state: &mut AgentLoopState,
        sink: &mut OpenAiResponseStreamSink,
    ) -> Result<(), AgentLoopError> {
        if ctx.session.is_none() {
            return Err(AgentLoopError::Internal(
                "OpenAI streaming adapter missing MCP session".to_string(),
            ));
        }
        let payload = build_iteration_payload(
            &self.iteration_request,
            self.upstream_tools.clone(),
            state,
            self.upstream.provider.as_ref(),
            true,
        )?;

        let mut request_builder = self.upstream.client.post(&self.upstream.url).json(&payload);
        let provider = ApiProvider::from_url(&self.upstream.url);
        let auth_header = provider.extract_auth_header(
            self.upstream.headers.as_ref(),
            self.upstream.api_key.as_ref(),
        );
        request_builder = provider.apply_headers(request_builder, auth_header.as_ref());
        request_builder = request_builder.header("Accept", "text/event-stream");

        let response = request_builder.send().await.map_err(|e| {
            AgentLoopError::Upstream(format!(
                "Failed to forward streaming request to OpenAI: {e}"
            ))
        })?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let body = error::sanitize_error_body(&body);
            return Err(AgentLoopError::Upstream(format!(
                "upstream error {status}: {body}"
            )));
        }

        let mut upstream_stream = response.bytes_stream();
        let mut chunk_processor = ChunkProcessor::new();
        let mut seen_in_progress = false;
        let tool_transfers = sink.tool_transfers.clone();
        let tool_server_labels = sink.tool_server_labels.clone();
        let mut handler = OpenAiUpstreamStreamParser::with_starting_index_and_tool_transfers(
            sink.next_output_index(),
            tool_transfers.clone(),
        );
        if let Some(ref id) = self.response_id_override {
            handler.original_response_id = Some(id.clone());
        }
        let streaming_ctx = StreamingEventContext {
            original_request: &self.original_request,
            previous_response_id: self.previous_response_id.as_deref(),
            tool_transfers: &tool_transfers,
            tool_server_labels: &tool_server_labels,
        };

        while let Some(chunk_result) = upstream_stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    chunk_processor.push_chunk(&chunk);
                    while let Some(raw_block) = chunk_processor.next_block() {
                        let (event_name, data) = parse_sse_block(&raw_block);
                        if data.is_empty() {
                            continue;
                        }

                        let action = handler.process_event(event_name, data.as_ref());
                        match action {
                            StreamParserAction::Forward => {
                                let parsed = serde_json::from_str::<Value>(data.as_ref()).ok();
                                let should_skip = state.iteration > 1
                                    && parsed.as_ref().is_some_and(|value| {
                                        matches!(
                                            value.get("type").and_then(|t| t.as_str()),
                                            Some(ResponseEvent::CREATED)
                                                | Some(ResponseEvent::IN_PROGRESS)
                                        )
                                    });
                                let is_in_progress = !seen_in_progress
                                    && parsed.as_ref().is_some_and(|value| {
                                        value.get("type").and_then(|t| t.as_str())
                                            == Some(ResponseEvent::IN_PROGRESS)
                                    });
                                let tx = sink.tx.clone();
                                if !should_skip
                                    && !forward_streaming_event(
                                        SseEventData {
                                            raw_block: &raw_block,
                                            event_name,
                                            data: data.as_ref(),
                                            pre_parsed: parsed,
                                        },
                                        &mut handler,
                                        &tx,
                                        &streaming_ctx,
                                        &mut sink.emitter,
                                    )
                                {
                                    return Err(AgentLoopError::Upstream(
                                        "client disconnected during streaming".to_string(),
                                    ));
                                }
                                if is_in_progress {
                                    seen_in_progress = true;
                                    sink.flush_buffered_mcp_list_tools();
                                    handler.advance_next_output_index_to(sink.next_output_index());
                                }
                            }
                            StreamParserAction::Buffer | StreamParserAction::Drop => {}
                            StreamParserAction::ToolCallReady {
                                forward_triggering_event,
                            } => {
                                let tx = sink.tx.clone();
                                if forward_triggering_event
                                    && !forward_streaming_event(
                                        SseEventData {
                                            raw_block: &raw_block,
                                            event_name,
                                            data: data.as_ref(),
                                            pre_parsed: None,
                                        },
                                        &mut handler,
                                        &tx,
                                        &streaming_ctx,
                                        &mut sink.emitter,
                                    )
                                {
                                    return Err(AgentLoopError::Upstream(
                                        "client disconnected during streaming".to_string(),
                                    ));
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    return Err(AgentLoopError::Upstream(format!("Stream error: {e}")));
                }
            }
        }

        sink.set_next_output_index(handler.next_output_index());
        if self.response_id_override.is_none() {
            self.response_id_override = handler.original_response_id().map(ToOwned::to_owned);
        }

        let response = handler.snapshot_final_response().and_then(|response_json| {
            serde_json::from_value::<ResponsesResponse>(response_json).ok()
        });
        let pending_calls = handler.take_pending_calls();

        if let Some(response) = response {
            state.latest_turn = Some(LoopModelTurn {
                message_text: extract_message_text(&response),
                reasoning_text: extract_reasoning_text(&response),
                usage: extract_usage(&response),
                request_id: Some(response.id.clone()),
            });
            state
                .transcript
                .extend(response_output_to_transcript(&response));
            if pending_calls.is_empty() {
                self.completed_response = Some(response);
            } else {
                self.completed_response = None;
            }
        }

        Self::register_gateway_calls(&pending_calls, sink);
        state.pending_tool_calls = pending_calls
            .into_iter()
            .map(|call| LoopToolCall {
                call_id: call.call_id.clone(),
                item_id: call.item_id.unwrap_or_else(|| call.call_id.clone()),
                name: call.name,
                arguments: call.arguments_buffer,
                approval_request_id: None,
            })
            .collect();

        Ok(())
    }

    async fn render_final(
        mut self,
        ctx: &AgentLoopContext<'_>,
        state: AgentLoopState,
        mode: RenderMode,
        sink: &mut OpenAiResponseStreamSink,
    ) -> Result<Self::FinalResponse, AgentLoopError> {
        let session = ctx.session.ok_or_else(|| {
            AgentLoopError::Internal(
                "OpenAI streaming render_final without MCP session".to_string(),
            )
        })?;
        let hooks = ResponseBuildHooks {
            original_tools: self.original_tools.take(),
            user_function_names: self.user_function_names,
            response_id_override: None,
            usage_shape: UsageShape::Modern,
        };
        let mut response = build_response_from_state(
            &state,
            self.completed_response.take(),
            mode,
            ctx.original_request,
            &hooks,
            session,
        );
        if let Some(ref response_id) = self.response_id_override {
            response.id.clone_from(response_id);
        }
        sink.emit_final_response(&response);
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::{
        event_types::OutputItemEvent,
        responses::{
            ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseReasoningContent,
            SummaryTextContent,
        },
    };
    use serde_json::json;

    use super::*;
    use crate::routers::openai::provider::{OpenAIProvider, XAIProvider};

    fn parse_sse_payload(bytes: Bytes) -> Value {
        let text = std::str::from_utf8(&bytes).expect("sse is utf8");
        let data = text
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .expect("sse data line");
        serde_json::from_str(data).expect("valid event json")
    }

    fn drain_events(rx: &mut mpsc::UnboundedReceiver<Result<Bytes, io::Error>>) -> Vec<Value> {
        let mut events = Vec::new();
        while let Ok(Ok(bytes)) = rx.try_recv() {
            events.push(parse_sse_payload(bytes));
        }
        events
    }

    fn stream_sink() -> (
        OpenAiResponseStreamSink,
        mpsc::UnboundedReceiver<Result<Bytes, io::Error>>,
    ) {
        let (tx, rx) = mpsc::unbounded_channel();
        let emitter =
            ResponseStreamEventEmitter::new("resp_test".to_string(), "model".to_string(), 123);
        (
            OpenAiResponseStreamSink::new(tx, emitter, HashMap::new(), HashMap::new()),
            rx,
        )
    }

    #[test]
    fn iteration_payload_reapplies_provider_transform_to_replayed_items() {
        let mut state = AgentLoopState::new(
            ResponseInput::Items(vec![ResponseInputOutputItem::Message {
                id: "msg_user".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText {
                    text: "hello".to_string(),
                }],
                status: Some("completed".to_string()),
                phase: None,
            }]),
            Default::default(),
        );
        state.transcript.push(ResponseInputOutputItem::Message {
            id: "msg_assistant".to_string(),
            role: "assistant".to_string(),
            content: vec![ResponseContentPart::OutputText {
                text: "prior answer".to_string(),
                annotations: vec![],
                logprobs: None,
            }],
            status: Some("completed".to_string()),
            phase: None,
        });
        let request = ResponsesRequest {
            model: "grok-4".to_string(),
            input: ResponseInput::Items(Vec::new()),
            store: Some(true),
            previous_response_id: Some("resp_previous".to_string()),
            ..Default::default()
        };

        let payload = build_iteration_payload(&request, None, &state, &XAIProvider, false)
            .expect("iteration payload builds");
        let input = payload["input"].as_array().expect("input array");

        assert_eq!(payload["store"], json!(false));
        assert!(payload.get("previous_response_id").is_none());
        assert!(input[1].get("id").is_none());
        assert!(input[1].get("status").is_none());
        assert_eq!(input[1]["content"][0]["type"], json!("input_text"));
    }

    #[test]
    fn iteration_payload_preserves_reasoning_replay_but_strips_output_fields() {
        let state = AgentLoopState::new(
            ResponseInput::Items(vec![ResponseInputOutputItem::new_reasoning_encrypted(
                "rs_123".to_string(),
                vec![SummaryTextContent::SummaryText {
                    text: "summary survives".to_string(),
                }],
                vec![ResponseReasoningContent::ReasoningText {
                    text: "raw reasoning text is not valid OpenAI input replay".to_string(),
                }],
                "encrypted-payload".to_string(),
                Some("completed".to_string()),
            )]),
            Default::default(),
        );
        let request = ResponsesRequest {
            model: "gpt-5-mini".to_string(),
            input: ResponseInput::Items(Vec::new()),
            ..Default::default()
        };

        let payload = build_iteration_payload(&request, None, &state, &OpenAIProvider, false)
            .expect("iteration payload builds");
        let item = payload["input"][0]
            .as_object()
            .expect("reasoning item object");

        assert_eq!(item.get("type"), Some(&json!("reasoning")));
        assert!(item.get("id").is_none());
        assert!(item.get("status").is_none());
        assert!(item.get("content").is_none());
        assert_eq!(
            item.get("encrypted_content"),
            Some(&json!("encrypted-payload"))
        );
        assert_eq!(item["summary"][0]["text"], json!("summary survives"));
    }

    #[test]
    fn openai_stream_final_response_uses_incomplete_event() {
        let (mut sink, mut rx) = stream_sink();
        let response = ResponsesResponse::builder("resp_test", "model")
            .status(ResponseStatus::Incomplete)
            .incomplete_details(json!({ "reason": "max_output_tokens" }))
            .build();

        sink.emit_final_response(&response);

        let events = drain_events(&mut rx);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["type"], json!(ResponseEvent::INCOMPLETE));
        assert_eq!(events[0]["response"]["status"], json!("incomplete"));
        assert_eq!(
            events[0]["response"]["incomplete_details"]["reason"],
            json!("max_output_tokens")
        );
    }

    #[test]
    fn openai_stream_mcp_list_tools_items_omit_status() {
        let (mut sink, mut rx) = stream_sink();
        let item = ResponseOutputItem::McpListTools {
            id: "mcpl_test".to_string(),
            server_label: "deepwiki".to_string(),
            tools: vec![],
            error: None,
        };

        sink.emit(LoopEvent::McpListToolsItem { item: &item });
        sink.flush_buffered_mcp_list_tools();

        let events = drain_events(&mut rx);
        let added = events
            .iter()
            .find(|event| event["type"] == json!(OutputItemEvent::ADDED))
            .expect("mcp_list_tools added event");
        assert!(added["item"].get("status").is_none());
        let done = events
            .iter()
            .find(|event| event["type"] == json!(OutputItemEvent::DONE))
            .expect("mcp_list_tools done event");
        assert!(done["item"].get("status").is_none());
    }
}
