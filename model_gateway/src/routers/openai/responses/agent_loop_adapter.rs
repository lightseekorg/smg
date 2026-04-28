//! Shared-agent-loop adapters for the OpenAI Responses router.

use std::{
    collections::{HashMap, HashSet},
    io,
};

use async_trait::async_trait;
use axum::http::{HeaderMap, StatusCode};
use bytes::Bytes;
use futures_util::StreamExt;
use openai_protocol::{
    common::{CompletionTokensDetails, PromptTokenUsageInfo, Usage},
    event_types::{is_function_call_type, OutputItemEvent, ResponseEvent},
    responses::{
        ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseOutputItem,
        ResponsesRequest, ResponsesResponse,
    },
};
use serde_json::{to_value, Value};
use smg_mcp::McpToolSession;
use tokio::sync::mpsc;

use super::super::mcp::prepare_mcp_tools_as_functions;
use crate::routers::{
    common::{
        agent_loop::{
            build_response_from_state, normalize_output_item_id, AgentLoopAdapter,
            AgentLoopContext, AgentLoopError, AgentLoopState, ExecutedCall, LoopEvent,
            LoopModelTurn, LoopToolCall, OutputFamily, PendingToolExecution, RenderMode,
            ResponseBuildHooks, StreamSink, ToolPresentation, ToolTransferDescriptor,
            ToolVisibility, UsageShape,
        },
        header_utils::ApiProvider,
        mcp_utils::collect_user_function_names,
        responses_streaming::{OutputItemType, ResponseStreamEventEmitter},
    },
    error,
    openai::{
        context::StreamingEventContext,
        responses::{
            common::{parse_sse_block, ChunkProcessor},
            streaming::{forward_streaming_event, SseEventData},
            upstream_stream_parser::{
                OpenAiUpstreamStreamParser, ParsedFunctionCall, StreamParserAction,
            },
        },
    },
};

#[derive(Clone)]
pub(crate) struct OpenAiUpstreamHandle {
    pub client: reqwest::Client,
    pub url: String,
    pub headers: Option<HeaderMap>,
    pub api_key: Option<String>,
    pub base_payload: Value,
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
    original_tools: Option<Vec<openai_protocol::responses::ResponseTool>>,
    user_function_names: HashSet<String>,
    completed_response: Option<ResponsesResponse>,
}

impl OpenAiNonStreamingAdapter {
    pub(crate) fn new(request: &ResponsesRequest, upstream: OpenAiUpstreamHandle) -> Self {
        Self {
            upstream,
            original_tools: request.tools.clone(),
            user_function_names: collect_user_function_names(request),
            completed_response: None,
        }
    }
}

fn build_iteration_payload(
    base_payload: &Value,
    state: &AgentLoopState,
    session: &McpToolSession<'_>,
    stream: bool,
) -> Result<Value, AgentLoopError> {
    let mut payload = base_payload.clone();

    if state.tool_budget_exhausted {
        let obj = payload
            .as_object_mut()
            .ok_or_else(|| AgentLoopError::Internal("payload not an object".to_string()))?;
        obj.remove("tools");
        obj.remove("tool_choice");
    } else if !session.mcp_tools().is_empty() {
        prepare_mcp_tools_as_functions(&mut payload, session);
    }

    let obj = payload
        .as_object_mut()
        .ok_or_else(|| AgentLoopError::Internal("payload not an object".to_string()))?;

    let mut input_items = Vec::new();
    match &state.upstream_input {
        ResponseInput::Text(text) => {
            input_items.push(serde_json::json!({
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": text }]
            }));
        }
        ResponseInput::Items(items) => {
            let serialized = to_value(items).map_err(|e| {
                AgentLoopError::Internal(format!("Failed to serialize prepared input items: {e}"))
            })?;
            if let Some(arr) = serialized.as_array() {
                input_items.extend(arr.iter().cloned());
            }
        }
    }

    let serialized_transcript = to_value(&state.transcript).map_err(|e| {
        AgentLoopError::Internal(format!("Failed to serialize loop transcript: {e}"))
    })?;
    if let Some(arr) = serialized_transcript.as_array() {
        input_items.extend(arr.iter().cloned());
    }

    obj.insert("input".to_string(), Value::Array(input_items));
    obj.insert("stream".to_string(), Value::Bool(stream));
    obj.insert("store".to_string(), Value::Bool(false));
    obj.remove("previous_response_id");
    obj.remove("conversation");

    if state.iteration > 1 && obj.get("tools").is_some() {
        obj.insert("tool_choice".to_string(), Value::String("auto".to_string()));
    }

    Ok(payload)
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
        let session = ctx.session.ok_or_else(|| {
            AgentLoopError::Internal("OpenAI adapter missing MCP session".to_string())
        })?;
        let payload = build_iteration_payload(&self.upstream.base_payload, state, session, false)?;
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

fn should_forward_caller_function_done(
    data: &str,
    handler: &OpenAiUpstreamStreamParser,
    ctx: &StreamingEventContext<'_>,
) -> bool {
    let Ok(parsed) = serde_json::from_str::<Value>(data) else {
        return false;
    };
    if parsed.get("type").and_then(|value| value.as_str()) != Some(OutputItemEvent::DONE) {
        return false;
    }
    let Some(item) = parsed.get("item") else {
        return false;
    };
    let Some(item_type) = item.get("type").and_then(|value| value.as_str()) else {
        return false;
    };
    if !is_function_call_type(item_type) {
        return false;
    }
    let tool_name = item
        .get("name")
        .and_then(|value| value.as_str())
        .or_else(|| {
            let output_index = parsed
                .get("output_index")
                .and_then(|value| value.as_u64())
                .and_then(|value| usize::try_from(value).ok())?;
            handler
                .pending_calls
                .iter()
                .find(|call| call.output_index == output_index)
                .map(|call| call.name.as_str())
        });
    let Some(tool_name) = tool_name else {
        return false;
    };
    let descriptor = ctx
        .tool_transfers
        .get(tool_name)
        .copied()
        .unwrap_or_else(ToolTransferDescriptor::caller_function);
    matches!(descriptor.family, OutputFamily::Function)
        && matches!(descriptor.visibility, ToolVisibility::Visible)
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
        self.tool_transfers
            .get(name)
            .copied()
            .unwrap_or_else(ToolTransferDescriptor::caller_function)
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
        let payload = serde_json::json!({
            "type": ResponseEvent::COMPLETED,
            "sequence_number": self.emitter.next_sequence(),
            "response": response,
        });
        self.send(&payload);
    }

    fn send(&mut self, payload: &Value) {
        if self.disconnected {
            return;
        }
        if self.emitter.send_event(payload, &self.tx).is_err() {
            self.disconnected = true;
        }
    }

    fn emit_mcp_list_tools_item(&mut self, item: &ResponseOutputItem) {
        let ResponseOutputItem::McpListTools {
            id,
            server_label,
            tools,
            ..
        } = item
        else {
            return;
        };

        let tool_items: Vec<Value> = tools
            .iter()
            .filter_map(|tool| to_value(tool).ok())
            .collect();
        let (output_index, _allocated_id) = self
            .emitter
            .allocate_output_index(OutputItemType::McpListTools);
        let added_item = serde_json::json!({
            "id": id,
            "type": "mcp_list_tools",
            "server_label": server_label,
            "status": "in_progress",
            "tools": [],
        });
        let event = self
            .emitter
            .emit_output_item_added(output_index, &added_item);
        self.send(&event);
        let event = self
            .emitter
            .emit_mcp_list_tools_in_progress(output_index, id);
        self.send(&event);
        let event = self
            .emitter
            .emit_mcp_list_tools_completed(output_index, id, &tool_items);
        self.send(&event);
        let done_item = serde_json::json!({
            "id": id,
            "type": "mcp_list_tools",
            "server_label": server_label,
            "status": "completed",
            "tools": tool_items,
        });
        let event = self.emitter.emit_output_item_done(output_index, &done_item);
        self.send(&event);
        self.emitter.complete_output_item(output_index);
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

        let family = tracking.presentation.family;
        if executed.is_error && matches!(family, OutputFamily::McpCall) {
            let event = self.emitter.emit_mcp_call_failed(
                tracking.output_index,
                &tracking.item_id,
                &executed.output_string,
            );
            self.send(&event);
        } else if let Some(event) =
            self.emitter
                .emit_tool_call_completed(tracking.output_index, &tracking.item_id, family)
        {
            self.send(&event);
        }

        let mut final_item = tracking
            .presentation
            .render_final_item(executed, &tracking.item_id);
        if matches!(family, OutputFamily::McpCall) {
            if let Some(obj) = final_item.as_object_mut() {
                if obj
                    .get("server_label")
                    .and_then(|value| value.as_str())
                    .is_none_or(|label| label.is_empty())
                {
                    obj.insert(
                        "server_label".to_string(),
                        Value::String(tracking.server_label.clone()),
                    );
                }
            }
        }
        let event = self
            .emitter
            .emit_output_item_done(tracking.output_index, &final_item);
        self.send(&event);
        self.emitter.complete_output_item(tracking.output_index);
    }

    fn emit_approval_requested(&mut self, pending: &PendingToolExecution) {
        let (output_index, _allocated_item_id) = self
            .emitter
            .allocate_output_index(OutputItemType::McpApprovalRequest);
        let item_id = format!("mcpr_{}", pending.call.call_id);
        let item = serde_json::json!({
            "id": item_id.clone(),
            "type": "mcp_approval_request",
            "server_label": pending.server_label,
            "name": pending.call.name,
            "arguments": pending.call.arguments,
        });
        let event = self.emitter.emit_output_item_added(output_index, &item);
        self.send(&event);
        let done_item = serde_json::json!({
            "id": item_id,
            "type": "mcp_approval_request",
            "server_label": pending.server_label,
            "name": pending.call.name,
            "arguments": pending.call.arguments,
        });
        let event = self.emitter.emit_output_item_done(output_index, &done_item);
        self.send(&event);
        self.emitter.complete_output_item(output_index);
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
        let output_item_type =
            ResponseStreamEventEmitter::output_item_type_for_family(Some(family));
        let (output_index, allocated_item_id) =
            self.emitter.allocate_output_index(output_item_type);
        let presentation = ToolPresentation::from_family(family);
        let call = LoopToolCall {
            call_id: call_id.to_string(),
            item_id: item_id_hint.to_string(),
            name: name.to_string(),
            arguments: full_args.to_string(),
            approval_request_id: approval_request_id.map(str::to_string),
        };
        let item = presentation.render_initial_item(&call, server_label, &allocated_item_id);

        let event = self.emitter.emit_output_item_added(output_index, &item);
        self.send(&event);

        if let Some(event) =
            self.emitter
                .emit_tool_call_in_progress(output_index, &allocated_item_id, family)
        {
            self.send(&event);
        }

        if presentation.streams_arguments() {
            if let Some(event) = self.emitter.emit_tool_call_arguments_delta(
                output_index,
                &allocated_item_id,
                full_args,
                family,
            ) {
                self.send(&event);
            }
            if let Some(event) = self.emitter.emit_tool_call_arguments_done(
                output_index,
                &allocated_item_id,
                full_args,
                family,
            ) {
                self.send(&event);
            }
        }

        self.registered_gateway_calls.insert(
            call_id.to_string(),
            RegisteredGatewayCall {
                output_index,
                item_id: allocated_item_id,
                presentation,
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
    original_request: ResponsesRequest,
    original_tools: Option<Vec<openai_protocol::responses::ResponseTool>>,
    user_function_names: HashSet<String>,
    completed_response: Option<ResponsesResponse>,
    response_id_override: Option<String>,
    previous_response_id: Option<String>,
}

impl OpenAiStreamingAdapter {
    pub(crate) fn new(request: &ResponsesRequest, upstream: OpenAiUpstreamHandle) -> Self {
        Self {
            upstream,
            original_request: request.clone(),
            original_tools: request.tools.clone(),
            user_function_names: collect_user_function_names(request),
            completed_response: None,
            response_id_override: None,
            previous_response_id: request.previous_response_id.clone(),
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
        let session = ctx.session.ok_or_else(|| {
            AgentLoopError::Internal("OpenAI streaming adapter missing MCP session".to_string())
        })?;
        let payload = build_iteration_payload(&self.upstream.base_payload, state, session, true)?;

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
        let mut handler = OpenAiUpstreamStreamParser::with_starting_index(sink.next_output_index());
        if let Some(ref id) = self.response_id_override {
            handler.original_response_id = Some(id.clone());
        }
        let mut chunk_processor = ChunkProcessor::new();
        let mut seen_in_progress = false;
        let tool_transfers = sink.tool_transfers.clone();
        let tool_server_labels = sink.tool_server_labels.clone();
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
                                let should_forward_triggering_event = forward_triggering_event
                                    || should_forward_caller_function_done(
                                        data.as_ref(),
                                        &handler,
                                        &streaming_ctx,
                                    );
                                let tx = sink.tx.clone();
                                if should_forward_triggering_event
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
    use std::collections::HashMap;

    use serde_json::json;

    use super::*;

    fn streaming_ctx<'a>(
        request: &'a ResponsesRequest,
        tool_transfers: &'a HashMap<String, ToolTransferDescriptor>,
        tool_server_labels: &'a HashMap<String, String>,
    ) -> StreamingEventContext<'a> {
        StreamingEventContext {
            original_request: request,
            previous_response_id: None,
            tool_transfers,
            tool_server_labels,
        }
    }

    fn bootstrap_pending_call(
        handler: &mut OpenAiUpstreamStreamParser,
        output_index: usize,
        name: &str,
    ) {
        let event = json!({
            "type": OutputItemEvent::ADDED,
            "output_index": output_index,
            "item": {
                "type": "function_call",
                "id": "fc_test",
                "call_id": "call_test",
                "name": name,
            }
        });
        let _ = handler.process_event(Some(OutputItemEvent::ADDED), &event.to_string());
    }

    fn function_done_event(output_index: usize, name: Option<&str>) -> String {
        let mut item = json!({
            "type": "function_call",
            "id": "fc_test",
            "call_id": "call_test",
            "arguments": "{}",
            "status": "completed",
        });
        if let Some(name) = name {
            item["name"] = json!(name);
        }

        json!({
            "type": OutputItemEvent::DONE,
            "output_index": output_index,
            "item": item,
        })
        .to_string()
    }

    #[test]
    fn visible_caller_function_done_forwards() {
        let mut handler = OpenAiUpstreamStreamParser::with_starting_index(0);
        bootstrap_pending_call(&mut handler, 0, "caller_tool");
        let request = ResponsesRequest::default();
        let tool_transfers = HashMap::new();
        let tool_server_labels = HashMap::new();
        let ctx = streaming_ctx(&request, &tool_transfers, &tool_server_labels);

        assert!(should_forward_caller_function_done(
            &function_done_event(0, Some("caller_tool")),
            &handler,
            &ctx,
        ));
    }

    #[test]
    fn gateway_function_done_stays_suppressed() {
        let mut handler = OpenAiUpstreamStreamParser::with_starting_index(0);
        bootstrap_pending_call(&mut handler, 0, "gateway_tool");
        let request = ResponsesRequest::default();
        let mut tool_transfers = HashMap::new();
        tool_transfers.insert(
            "gateway_tool".to_string(),
            ToolTransferDescriptor::from_family_and_approval(OutputFamily::McpCall, false),
        );
        let tool_server_labels = HashMap::new();
        let ctx = streaming_ctx(&request, &tool_transfers, &tool_server_labels);

        assert!(!should_forward_caller_function_done(
            &function_done_event(0, Some("gateway_tool")),
            &handler,
            &ctx,
        ));
    }

    #[test]
    fn function_done_without_name_uses_pending_call_for_caller_function() {
        let mut handler = OpenAiUpstreamStreamParser::with_starting_index(0);
        bootstrap_pending_call(&mut handler, 0, "caller_tool");
        let request = ResponsesRequest::default();
        let tool_transfers = HashMap::new();
        let tool_server_labels = HashMap::new();
        let ctx = streaming_ctx(&request, &tool_transfers, &tool_server_labels);

        assert!(should_forward_caller_function_done(
            &function_done_event(0, None),
            &handler,
            &ctx,
        ));
    }

    #[test]
    fn function_done_without_name_uses_pending_call_for_gateway_tool() {
        let mut handler = OpenAiUpstreamStreamParser::with_starting_index(0);
        bootstrap_pending_call(&mut handler, 0, "gateway_tool");
        let request = ResponsesRequest::default();
        let mut tool_transfers = HashMap::new();
        tool_transfers.insert(
            "gateway_tool".to_string(),
            ToolTransferDescriptor::from_family_and_approval(OutputFamily::McpCall, false),
        );
        let tool_server_labels = HashMap::new();
        let ctx = streaming_ctx(&request, &tool_transfers, &tool_server_labels);

        assert!(!should_forward_caller_function_done(
            &function_done_event(0, None),
            &handler,
            &ctx,
        ));
    }
}
