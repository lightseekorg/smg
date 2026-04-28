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
    event_types::{
        CodeInterpreterCallEvent, FileSearchCallEvent, ImageGenerationCallEvent, McpEvent,
        OutputItemEvent, ResponseEvent, WebSearchCallEvent,
    },
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
            ResponseBuildHooks, StreamSink, ToolPresentation, UsageShape,
        },
        header_utils::ApiProvider,
        mcp_utils::collect_user_function_names,
    },
    error,
    openai::{
        context::StreamingEventContext,
        mcp::{FunctionCallInProgress, StreamAction, StreamingToolHandler},
        responses::{
            common::{parse_sse_block, ChunkProcessor},
            streaming::{forward_streaming_event, SseEventData},
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

fn response_contains_gateway_function_calls(
    response: &ResponsesResponse,
    session: &McpToolSession<'_>,
) -> bool {
    response.output.iter().any(|item| match item {
        ResponseOutputItem::FunctionToolCall { name, .. } => session.has_exposed_tool(name),
        _ => false,
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

        state.pending_tool_calls = extract_pending_tool_calls(&response);
        state.latest_turn = Some(LoopModelTurn {
            message_text: extract_message_text(&response),
            reasoning_text: extract_reasoning_text(&response),
            usage: extract_usage(&response),
            request_id: Some(response.id.clone()),
        });
        state
            .transcript
            .extend(response_output_to_transcript(&response));

        if response_contains_gateway_function_calls(&response, session) {
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
    sequence_number: u64,
    next_output_index: usize,
    disconnected: bool,
    buffered_mcp_list_tools: Vec<ResponseOutputItem>,
    registered_gateway_calls: HashMap<String, RegisteredGatewayCall>,
}

impl OpenAiResponseStreamSink {
    pub(crate) fn new(tx: mpsc::UnboundedSender<Result<Bytes, io::Error>>) -> Self {
        Self {
            tx,
            sequence_number: 0,
            next_output_index: 0,
            disconnected: false,
            buffered_mcp_list_tools: Vec::new(),
            registered_gateway_calls: HashMap::new(),
        }
    }

    pub(crate) fn next_output_index(&self) -> usize {
        self.next_output_index
    }

    pub(crate) fn set_next_output_index(&mut self, next_output_index: usize) {
        self.next_output_index = next_output_index;
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
            "sequence_number": self.sequence_number,
            "response": response,
        });
        self.sequence_number += 1;
        let block = format!("event: {}\ndata: {}\n\n", ResponseEvent::COMPLETED, payload);
        if self.tx.send(Ok(Bytes::from(block))).is_err() {
            self.disconnected = true;
        }
    }

    fn send_json_event(&mut self, event_name: &str, payload: Value) {
        if self.disconnected {
            return;
        }
        let block = format!("event: {event_name}\ndata: {payload}\n\n");
        if self.tx.send(Ok(Bytes::from(block))).is_err() {
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

        let output_index = self.next_output_index;
        self.next_output_index += 1;

        let tool_items: Vec<Value> = tools
            .iter()
            .filter_map(|tool| to_value(tool).ok())
            .collect();
        let added_item = serde_json::json!({
            "id": id,
            "type": "mcp_list_tools",
            "server_label": server_label,
            "status": "in_progress",
            "tools": [],
        });
        self.send_json_event(
            OutputItemEvent::ADDED,
            serde_json::json!({
                "type": OutputItemEvent::ADDED,
                "sequence_number": self.sequence_number,
                "output_index": output_index,
                "item": added_item,
            }),
        );
        self.sequence_number += 1;
        self.send_json_event(
            McpEvent::LIST_TOOLS_IN_PROGRESS,
            serde_json::json!({
                "type": McpEvent::LIST_TOOLS_IN_PROGRESS,
                "sequence_number": self.sequence_number,
                "output_index": output_index,
                "item_id": id,
            }),
        );
        self.sequence_number += 1;
        self.send_json_event(
            McpEvent::LIST_TOOLS_COMPLETED,
            serde_json::json!({
                "type": McpEvent::LIST_TOOLS_COMPLETED,
                "sequence_number": self.sequence_number,
                "output_index": output_index,
                "item_id": id,
            }),
        );
        self.sequence_number += 1;
        self.send_json_event(
            OutputItemEvent::DONE,
            serde_json::json!({
                "type": OutputItemEvent::DONE,
                "sequence_number": self.sequence_number,
                "output_index": output_index,
                "item": {
                    "id": id,
                    "type": "mcp_list_tools",
                    "server_label": server_label,
                    "status": "completed",
                    "tools": tool_items,
                },
            }),
        );
        self.sequence_number += 1;
    }

    fn emit_gateway_execution_started(&mut self, call_id: &str) {
        let Some(tracking) = self.registered_gateway_calls.get(call_id) else {
            return;
        };
        let event_type = match tracking.presentation.family {
            OutputFamily::WebSearchCall => Some(WebSearchCallEvent::SEARCHING),
            OutputFamily::CodeInterpreterCall => Some(CodeInterpreterCallEvent::INTERPRETING),
            OutputFamily::FileSearchCall => Some(FileSearchCallEvent::SEARCHING),
            OutputFamily::ImageGenerationCall => Some(ImageGenerationCallEvent::GENERATING),
            OutputFamily::McpCall | OutputFamily::Function => None,
        };
        let Some(event_type) = event_type else {
            return;
        };
        self.send_json_event(
            event_type,
            serde_json::json!({
                "type": event_type,
                "sequence_number": self.sequence_number,
                "output_index": tracking.output_index,
                "item_id": tracking.item_id,
            }),
        );
        self.sequence_number += 1;
    }

    fn emit_tool_completed(&mut self, executed: &ExecutedCall) {
        let Some(tracking) = self.registered_gateway_calls.remove(&executed.call_id) else {
            return;
        };

        let family = tracking.presentation.family;
        let event_type = if executed.is_error && matches!(family, OutputFamily::McpCall) {
            Some(McpEvent::CALL_FAILED)
        } else {
            match family {
                OutputFamily::WebSearchCall => Some(WebSearchCallEvent::COMPLETED),
                OutputFamily::CodeInterpreterCall => Some(CodeInterpreterCallEvent::COMPLETED),
                OutputFamily::FileSearchCall => Some(FileSearchCallEvent::COMPLETED),
                OutputFamily::ImageGenerationCall => Some(ImageGenerationCallEvent::COMPLETED),
                OutputFamily::McpCall => Some(McpEvent::CALL_COMPLETED),
                OutputFamily::Function => None,
            }
        };

        if let Some(event_type) = event_type {
            let mut payload = serde_json::json!({
                "type": event_type,
                "sequence_number": self.sequence_number,
                "output_index": tracking.output_index,
                "item_id": tracking.item_id,
            });
            if event_type == McpEvent::CALL_FAILED {
                payload["error"] = serde_json::json!(executed.output_string);
            }
            self.send_json_event(event_type, payload);
            self.sequence_number += 1;
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
        self.send_json_event(
            OutputItemEvent::DONE,
            serde_json::json!({
                "type": OutputItemEvent::DONE,
                "sequence_number": self.sequence_number,
                "output_index": tracking.output_index,
                "item": final_item,
            }),
        );
        self.sequence_number += 1;
    }

    fn emit_approval_requested(&mut self, pending: &PendingToolExecution) {
        let output_index = self.next_output_index;
        self.next_output_index += 1;
        let item_id = format!("mcpr_{}", pending.call.call_id);
        let item = serde_json::json!({
            "id": item_id,
            "type": "mcp_approval_request",
            "server_label": pending.server_label,
            "name": pending.call.name,
            "arguments": pending.call.arguments,
        });
        self.send_json_event(
            OutputItemEvent::ADDED,
            serde_json::json!({
                "type": OutputItemEvent::ADDED,
                "sequence_number": self.sequence_number,
                "output_index": output_index,
                "item": item,
            }),
        );
        self.sequence_number += 1;
        self.send_json_event(
            OutputItemEvent::DONE,
            serde_json::json!({
                "type": OutputItemEvent::DONE,
                "sequence_number": self.sequence_number,
                "output_index": output_index,
                "item": {
                    "id": item_id,
                    "type": "mcp_approval_request",
                    "server_label": pending.server_label,
                    "name": pending.call.name,
                    "arguments": pending.call.arguments,
                },
            }),
        );
        self.sequence_number += 1;
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
        let output_index = self.next_output_index;
        self.next_output_index += 1;
        let allocated_item_id = normalize_output_item_id(family, item_id_hint);
        let presentation = ToolPresentation::from_family(family);
        let call = LoopToolCall {
            call_id: call_id.to_string(),
            item_id: item_id_hint.to_string(),
            name: name.to_string(),
            arguments: full_args.to_string(),
            approval_request_id: approval_request_id.map(str::to_string),
        };
        let item = presentation.render_initial_item(&call, server_label, &allocated_item_id);

        self.send_json_event(
            OutputItemEvent::ADDED,
            serde_json::json!({
                "type": OutputItemEvent::ADDED,
                "sequence_number": self.sequence_number,
                "output_index": output_index,
                "item": item,
            }),
        );
        self.sequence_number += 1;

        let in_progress_event = match family {
            OutputFamily::McpCall => Some(McpEvent::CALL_IN_PROGRESS),
            OutputFamily::WebSearchCall => Some(WebSearchCallEvent::IN_PROGRESS),
            OutputFamily::CodeInterpreterCall => Some(CodeInterpreterCallEvent::IN_PROGRESS),
            OutputFamily::FileSearchCall => Some(FileSearchCallEvent::IN_PROGRESS),
            OutputFamily::ImageGenerationCall => Some(ImageGenerationCallEvent::IN_PROGRESS),
            OutputFamily::Function => None,
        };
        if let Some(event_type) = in_progress_event {
            self.send_json_event(
                event_type,
                serde_json::json!({
                    "type": event_type,
                    "sequence_number": self.sequence_number,
                    "output_index": output_index,
                    "item_id": allocated_item_id,
                }),
            );
            self.sequence_number += 1;
        }

        if matches!(family, OutputFamily::McpCall) {
            self.send_json_event(
                McpEvent::CALL_ARGUMENTS_DELTA,
                serde_json::json!({
                    "type": McpEvent::CALL_ARGUMENTS_DELTA,
                    "sequence_number": self.sequence_number,
                    "output_index": output_index,
                    "item_id": allocated_item_id,
                    "delta": full_args,
                    "obfuscation": null,
                }),
            );
            self.sequence_number += 1;
            self.send_json_event(
                McpEvent::CALL_ARGUMENTS_DONE,
                serde_json::json!({
                    "type": McpEvent::CALL_ARGUMENTS_DONE,
                    "sequence_number": self.sequence_number,
                    "output_index": output_index,
                    "item_id": allocated_item_id,
                    "arguments": full_args,
                }),
            );
            self.sequence_number += 1;
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
        pending_calls: &[FunctionCallInProgress],
        session: &McpToolSession<'_>,
        sink: &mut OpenAiResponseStreamSink,
    ) {
        for call in pending_calls {
            if !session.has_exposed_tool(&call.name) {
                continue;
            }
            let family = OutputFamily::from_mcp_format(&session.tool_response_format(&call.name));
            let source_id = call.item_id.as_deref().unwrap_or(call.call_id.as_str());
            sink.register_gateway_call(
                call.call_id.clone(),
                call.effective_output_index(),
                normalize_output_item_id(family, source_id),
                family,
                session.resolve_tool_server_label(&call.name),
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
        let mut handler = StreamingToolHandler::with_starting_index(sink.next_output_index());
        if let Some(ref id) = self.response_id_override {
            handler.original_response_id = Some(id.clone());
        }
        let mut chunk_processor = ChunkProcessor::new();
        let mut tool_calls_detected = false;
        let mut seen_in_progress = false;
        let streaming_ctx = StreamingEventContext {
            original_request: &self.original_request,
            previous_response_id: self.previous_response_id.as_deref(),
            session: Some(session),
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
                            StreamAction::Forward => {
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
                                        &mut sink.sequence_number,
                                    )
                                {
                                    return Err(AgentLoopError::Upstream(
                                        "client disconnected during streaming".to_string(),
                                    ));
                                }
                                if is_in_progress {
                                    seen_in_progress = true;
                                    sink.flush_buffered_mcp_list_tools();
                                }
                            }
                            StreamAction::Buffer | StreamAction::Drop => {}
                            StreamAction::ExecuteTools {
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
                                        &mut sink.sequence_number,
                                    )
                                {
                                    return Err(AgentLoopError::Upstream(
                                        "client disconnected during streaming".to_string(),
                                    ));
                                }
                                tool_calls_detected = true;
                                break;
                            }
                        }
                    }
                    if tool_calls_detected {
                        break;
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

        if let Some(response_json) = handler.snapshot_final_response() {
            if let Ok(response) = serde_json::from_value::<ResponsesResponse>(response_json) {
                state.latest_turn = Some(LoopModelTurn {
                    message_text: extract_message_text(&response),
                    reasoning_text: extract_reasoning_text(&response),
                    usage: extract_usage(&response),
                    request_id: Some(response.id.clone()),
                });
                state
                    .transcript
                    .extend(response_output_to_transcript(&response));
                if response_contains_gateway_function_calls(&response, session) {
                    self.completed_response = None;
                } else {
                    self.completed_response = Some(response);
                }
            }
        }

        let pending_calls = handler.take_pending_calls();
        Self::register_gateway_calls(&pending_calls, session, sink);
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
