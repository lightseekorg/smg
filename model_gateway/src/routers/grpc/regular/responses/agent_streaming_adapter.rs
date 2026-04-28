//! Streaming flavor of the Regular Responses agent-loop adapter.
//!
//! Drives `pipeline.execute_chat` with `stream=true` per iteration,
//! pumps the SSE body through `ResponseStreamEventEmitter::process_chunk`
//! (which already maps every Chat-stream variant onto the Responses
//! API event shapes), accumulates the per-iteration response so the
//! driver can decide what comes next, and emits the
//! `output_item.added` / `function_call_arguments.delta` /
//! `function_call_arguments.done` triples for any
//! caller-visible function tool calls before the loop exits.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use async_trait::async_trait;
use axum::http;
use bytes::Bytes;
use futures_util::StreamExt;
use openai_protocol::{
    chat::{
        ChatChoice, ChatCompletionMessage, ChatCompletionResponse, ChatCompletionStreamResponse,
    },
    common::{
        CompletionTokensDetails, FunctionCallResponse, ToolCall, ToolChoice, ToolChoiceValue,
        Usage, UsageInfo,
    },
    responses::{ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponsesRequest},
};
use serde_json::json;
use smg_mcp::McpToolSession;
use uuid::Uuid;

use super::conversions;
use crate::{
    middleware::TenantRequestMeta,
    routers::{
        common::agent_loop::{
            AgentLoopAdapter, AgentLoopContext, AgentLoopError, AgentLoopState, LoopModelTurn,
            LoopToolCall, RenderMode,
        },
        grpc::common::responses::{
            collect_user_function_names, persist_response_if_needed, GrpcResponseStreamSink,
            ResponsesContext,
        },
    },
};

#[derive(Clone)]
pub(crate) struct RegularStreamingUpstreamHandle {
    pub headers: Option<http::HeaderMap>,
    pub model_id: String,
    pub tenant_request_meta: TenantRequestMeta,
}

pub(crate) struct RegularStreamingAdapter<'a> {
    ctx: &'a ResponsesContext,
    upstream: RegularStreamingUpstreamHandle,
    user_function_names: HashSet<String>,
    mcp_chat_tools: Vec<openai_protocol::common::Tool>,
    /// Most recent iteration's accumulated chat response. Used by
    /// `render_final` to construct the persisted `ResponsesResponse`
    /// and by the driver to detect tool calls (mirroring the
    /// non-streaming adapter).
    cached_response: Option<ChatCompletionResponse>,
    /// Final-channel usage staged into the sink before the driver
    /// fires `LoopEvent::ResponseFinished`.
    last_usage: Option<UsageInfo>,
}

impl<'a> RegularStreamingAdapter<'a> {
    pub(crate) fn new(
        ctx: &'a ResponsesContext,
        upstream: RegularStreamingUpstreamHandle,
        request: &ResponsesRequest,
        session: &McpToolSession<'_>,
    ) -> Self {
        let user_function_names = collect_user_function_names(request);
        let mcp_chat_tools = session.build_chat_function_tools();
        Self {
            ctx,
            upstream,
            user_function_names,
            mcp_chat_tools,
            cached_response: None,
            last_usage: None,
        }
    }
}

fn build_iteration_request(
    original: &ResponsesRequest,
    state: &AgentLoopState,
) -> ResponsesRequest {
    let upstream_items = match &state.upstream_input {
        ResponseInput::Items(items) => items
            .iter()
            .map(openai_protocol::responses::normalize_input_item)
            .collect::<Vec<_>>(),
        ResponseInput::Text(text) => vec![ResponseInputOutputItem::Message {
            id: format!("msg_u_{}", Uuid::now_v7()),
            role: "user".to_string(),
            content: vec![ResponseContentPart::InputText { text: text.clone() }],
            status: Some("completed".to_string()),
            phase: None,
        }],
    };
    let mut combined: Vec<ResponseInputOutputItem> =
        Vec::with_capacity(upstream_items.len() + state.transcript.len());
    combined.extend(upstream_items);
    combined.extend(state.transcript.iter().cloned());

    let mut request = original.clone();
    request.input = ResponseInput::Items(combined);
    request.store = Some(false);
    request.previous_response_id = None;
    request.conversation = None;
    request.stream = Some(true);
    if state.tool_budget_exhausted {
        request.tools = None;
        request.tool_choice = None;
    } else if state.iteration > 1 {
        request.tool_choice = Some(openai_protocol::responses::ResponsesToolChoice::Options(
            openai_protocol::responses::ToolChoiceOptions::Auto,
        ));
    }
    request
}

#[async_trait]
impl<'a> AgentLoopAdapter<GrpcResponseStreamSink> for RegularStreamingAdapter<'a> {
    type FinalResponse = ();

    async fn call_upstream(
        &mut self,
        ctx: &AgentLoopContext<'_>,
        state: &mut AgentLoopState,
        sink: &mut GrpcResponseStreamSink,
    ) -> Result<(), AgentLoopError> {
        let request = build_iteration_request(ctx.original_request, state);
        let mut chat_request = conversions::responses_to_chat(&request).map_err(|e| {
            AgentLoopError::InvalidRequest(format!("Failed to convert request: {e}"))
        })?;

        let mut all_tools = chat_request.tools.take().unwrap_or_default();
        if !state.tool_budget_exhausted {
            all_tools.extend(self.mcp_chat_tools.iter().cloned());
        }
        chat_request.tools = (!all_tools.is_empty()).then_some(all_tools);
        chat_request.tool_choice = if state.tool_budget_exhausted {
            None
        } else if state.iteration <= 1 {
            chat_request
                .tool_choice
                .take()
                .or(Some(ToolChoice::Value(ToolChoiceValue::Auto)))
        } else {
            Some(ToolChoice::Value(ToolChoiceValue::Auto))
        };

        let chat_response = self
            .ctx
            .pipeline
            .execute_chat(
                Arc::new(chat_request),
                self.upstream.headers.clone(),
                self.upstream.model_id.clone(),
                self.ctx.components.clone(),
                Some(self.upstream.tenant_request_meta.clone()),
            )
            .await;

        let body = chat_response.into_body();
        let accumulated = consume_and_accumulate_stream(body, sink).await?;

        // Push raw fc identities to the loop's classification queue.
        // Wire emission already happened progressively inside
        // `sink.process_chat_chunk` as chat-stream chunks arrived;
        // the driver classifies gateway vs caller-fc + approval
        // gating from this list.
        state.pending_tool_calls = accumulated
            .choices
            .first()
            .and_then(|c| c.message.tool_calls.as_ref())
            .map(|calls| {
                calls
                    .iter()
                    .map(|tc| LoopToolCall {
                        call_id: tc.id.clone(),
                        item_id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        arguments: tc
                            .function
                            .arguments
                            .clone()
                            .unwrap_or_else(|| "{}".to_string()),
                        approval_request_id: None,
                    })
                    .collect()
            })
            .unwrap_or_default();
        if let Some(usage) = accumulated.usage.as_ref() {
            self.last_usage = Some(UsageInfo {
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
                reasoning_tokens: usage
                    .completion_tokens_details
                    .as_ref()
                    .and_then(|d| d.reasoning_tokens),
                prompt_tokens_details: None,
            });
        }
        state.latest_turn = Some(LoopModelTurn {
            usage: accumulated.usage.as_ref().map(|u| Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
                prompt_tokens_details: None,
                completion_tokens_details: u.completion_tokens_details.clone(),
            }),
            request_id: Some(accumulated.id.clone()),
            ..Default::default()
        });
        self.cached_response = Some(accumulated);
        Ok(())
    }

    async fn render_final(
        mut self,
        ctx: &AgentLoopContext<'_>,
        _state: AgentLoopState,
        mode: RenderMode,
        sink: &mut GrpcResponseStreamSink,
    ) -> Result<(), AgentLoopError> {
        // Stage usage so the sink's `ResponseFinished` /
        // `ResponseIncomplete` emit attaches it to the final
        // `response.completed`. The loop driver fires that event after
        // this method returns. Carry `reasoning_tokens` so the stored
        // record matches the SSE-side `usage_json` below.
        let usage_for_persist = self.last_usage.clone().map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            prompt_tokens_details: None,
            completion_tokens_details: u.reasoning_tokens.map(|n| CompletionTokensDetails {
                reasoning_tokens: Some(n),
                accepted_prediction_tokens: None,
                rejected_prediction_tokens: None,
            }),
        });
        let usage_json = self.last_usage.take().map(|u| {
            let mut obj = json!({
                "input_tokens": u.prompt_tokens,
                "output_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            });
            if let Some(reasoning) = u.reasoning_tokens {
                obj["output_tokens_details"] = json!({ "reasoning_tokens": reasoning });
            }
            obj
        });
        sink.set_final_usage(usage_json);

        // Persist the streamed response so a follow-up request whose
        // `previous_response_id` points at this turn can resolve the
        // chain. Built non-destructively so the sink's later
        // `emit_completed` (fired via `LoopEvent::ResponseFinished`
        // right after we return) sees the same accumulated state for
        // the SSE payload.
        let mut final_response = sink.emitter.finalize(usage_for_persist);
        final_response
            .previous_response_id
            .clone_from(&ctx.original_request.previous_response_id);
        final_response.conversation = ctx.original_request.conversation.as_ref().map(|c| {
            openai_protocol::common::ConversationRef::Object {
                id: c.as_id().to_string(),
            }
        });
        final_response.store = ctx.original_request.store.unwrap_or(true);
        if let RenderMode::Incomplete { reason, .. } = &mode {
            // Match the non-streaming `RenderMode::Incomplete` contract:
            // top-level `status` stays `Completed`, with the reason
            // attached to `incomplete_details`. Streaming previously
            // overrode `status` to `Incomplete` here, which made the
            // persisted record diverge from the non-streaming render
            // path and produced inconsistent reads on follow-up
            // `previous_response_id` resolutions.
            final_response.incomplete_details = Some(json!({ "reason": reason }));
        }
        persist_response_if_needed(
            self.ctx.conversation_storage.clone(),
            self.ctx.conversation_item_storage.clone(),
            self.ctx.response_storage.clone(),
            &final_response,
            ctx.original_request,
            self.ctx.request_context.clone(),
        )
        .await;

        let _ = self.user_function_names;
        Ok(())
    }
}

/// Pump the chat-stream body through the existing
/// `ResponseStreamEventEmitter::process_chunk` so the per-chunk SSE
/// translation stays a single seam, and accumulate a
/// `ChatCompletionResponse` so the loop driver can consult it for tool
/// calls.
async fn consume_and_accumulate_stream(
    body: axum::body::Body,
    sink: &mut GrpcResponseStreamSink,
) -> Result<ChatCompletionResponse, AgentLoopError> {
    let mut accumulator = ChatStreamAccumulator::new();
    let mut stream = body.into_data_stream();
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result
            .map_err(|e| AgentLoopError::Upstream(format!("Stream read error: {e}")))?;
        let event_str = String::from_utf8_lossy(&chunk);
        let event = event_str.trim();
        if event == "data: [DONE]" {
            break;
        }
        if let Some(json_str) = event.strip_prefix("data: ") {
            let json_str = json_str.trim();
            if let Ok(chat_chunk) = serde_json::from_str::<ChatCompletionStreamResponse>(json_str) {
                accumulator.process_chunk(&chat_chunk);
                // Sink owns chat-stream → wire-event translation,
                // including progressive tool_call args.delta dispatch
                // (driven by per-chunk `tool_calls` deltas + a
                // `LoopEvent::ToolCall*` sequence per call_index).
                sink.process_chat_chunk(&chat_chunk);
            } else {
                // Pass non-chunk events through verbatim so error
                // events still reach the client.
                let _ = sink.tx.send(Ok(Bytes::from(format!("{event}\n\n"))));
            }
        }
    }
    Ok(accumulator.finalize())
}

/// Emit the function-call SSE triple
/// (`output_item.added` → `function_call_arguments.delta` →
/// Lightweight chat-stream accumulator. Mirrors what the prior
/// `convert_and_accumulate_stream` did — just enough state to detect
/// tool calls and to surface a final `ChatCompletionResponse` shape
/// the loop driver can consult.
struct ChatStreamAccumulator {
    id: String,
    model: String,
    created: u64,
    content_buffer: String,
    reasoning_buffer: String,
    tool_calls: Vec<ToolCall>,
    tool_indexes: HashMap<u32, usize>,
    finish_reason: Option<String>,
    usage: Option<Usage>,
}

impl ChatStreamAccumulator {
    fn new() -> Self {
        Self {
            id: String::new(),
            model: String::new(),
            created: 0,
            content_buffer: String::new(),
            reasoning_buffer: String::new(),
            tool_calls: Vec::new(),
            tool_indexes: HashMap::new(),
            finish_reason: None,
            usage: None,
        }
    }

    fn process_chunk(&mut self, chunk: &ChatCompletionStreamResponse) {
        if self.id.is_empty() {
            self.id.clone_from(&chunk.id);
            self.model.clone_from(&chunk.model);
            self.created = chunk.created;
        }
        if let Some(choice) = chunk.choices.first() {
            if let Some(c) = &choice.delta.content {
                self.content_buffer.push_str(c);
            }
            if let Some(r) = &choice.delta.reasoning_content {
                self.reasoning_buffer.push_str(r);
            }
            if let Some(deltas) = &choice.delta.tool_calls {
                for delta in deltas {
                    let idx_into_calls = match self.tool_indexes.get(&delta.index).copied() {
                        Some(i) => i,
                        None => {
                            let new_idx = self.tool_calls.len();
                            self.tool_indexes.insert(delta.index, new_idx);
                            self.tool_calls.push(ToolCall {
                                id: delta.id.clone().unwrap_or_default(),
                                tool_type: "function".to_string(),
                                function: FunctionCallResponse {
                                    name: String::new(),
                                    arguments: Some(String::new()),
                                },
                            });
                            new_idx
                        }
                    };
                    if let Some(call) = self.tool_calls.get_mut(idx_into_calls) {
                        if let Some(id) = &delta.id {
                            if call.id.is_empty() {
                                call.id.clone_from(id);
                            }
                        }
                        if let Some(func) = &delta.function {
                            if let Some(name) = &func.name {
                                if !name.is_empty() && call.function.name.is_empty() {
                                    call.function.name.clone_from(name);
                                }
                            }
                            if let Some(args) = &func.arguments {
                                if let Some(buf) = call.function.arguments.as_mut() {
                                    buf.push_str(args);
                                }
                            }
                        }
                    }
                }
            }
            if let Some(reason) = &choice.finish_reason {
                self.finish_reason = Some(reason.clone());
            }
        }
        if let Some(usage) = &chunk.usage {
            self.usage = Some(usage.clone());
        }
    }

    fn finalize(self) -> ChatCompletionResponse {
        let message = ChatCompletionMessage {
            role: "assistant".to_string(),
            content: if self.content_buffer.is_empty() {
                None
            } else {
                Some(self.content_buffer)
            },
            tool_calls: if self.tool_calls.is_empty() {
                None
            } else {
                Some(self.tool_calls)
            },
            reasoning_content: if self.reasoning_buffer.is_empty() {
                None
            } else {
                Some(self.reasoning_buffer)
            },
        };
        ChatCompletionResponse {
            id: self.id,
            object: "chat.completion".to_string(),
            created: self.created,
            model: self.model,
            choices: vec![ChatChoice {
                index: 0,
                message,
                finish_reason: self.finish_reason,
                logprobs: None,
                matched_stop: None,
                hidden_states: None,
            }],
            usage: self.usage,
            system_fingerprint: None,
        }
    }
}
