//! Streaming flavor of the Harmony agent-loop adapter.
//!
//! Mirrors `HarmonyAdapter` but talks to
//! `pipeline::execute_harmony_responses_streaming` and feeds chunk-level
//! events into a `GrpcResponseStreamSink`. Kept as a separate type
//! because the trait's sink generic resolves to a concrete type at
//! impl-site — separate adapters are cleaner than dispatching streaming
//! vs. non-streaming inside one trait method.

use std::collections::HashSet;

use async_trait::async_trait;
use openai_protocol::{
    common::Usage,
    responses::{
        ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseReasoningContent,
        ResponseTool, ResponsesRequest, ResponsesToolChoice, ToolChoiceOptions,
    },
};
use serde_json::json;
use smg_mcp::McpToolSession;

use super::{
    common::strip_image_generation_from_request_tools,
    execution::convert_mcp_tools_to_response_tools,
};
use crate::{
    middleware::TenantRequestMeta,
    routers::{
        common::agent_loop::{
            AgentLoopAdapter, AgentLoopContext, AgentLoopError, AgentLoopState, LoopModelTurn,
            LoopToolCall, RenderMode,
        },
        grpc::{
            common::responses::{
                collect_user_function_names, persist_response_if_needed, GrpcResponseStreamSink,
                ResponsesContext,
            },
            harmony::{processor::ResponsesIterationResult, streaming::HarmonyStreamingProcessor},
        },
    },
};

pub(crate) struct HarmonyStreamingAdapter<'a> {
    ctx: &'a ResponsesContext,
    tenant_request_meta: TenantRequestMeta,
    original_tools: Option<Vec<ResponseTool>>,
    user_function_names: HashSet<String>,
    upstream_tools: Option<Vec<ResponseTool>>,
    /// Latest turn's usage. Streaming completion needs to surface
    /// `input_tokens_details.cached_tokens` whenever the prefill phase
    /// reports them, so the adapter retains the full Usage struct
    /// rather than the loop's smaller `LoopModelTurn::usage`.
    last_usage: Option<Usage>,
}

impl<'a> HarmonyStreamingAdapter<'a> {
    pub(crate) fn new(
        ctx: &'a ResponsesContext,
        tenant_request_meta: TenantRequestMeta,
        request: &ResponsesRequest,
        session: &McpToolSession<'_>,
    ) -> Self {
        let original_tools = request.tools.clone();
        let user_function_names = collect_user_function_names(request);

        let mut upstream = request.clone();
        let mcp_tools = session.mcp_tools();
        if !mcp_tools.is_empty() {
            let mcp_response_tools = convert_mcp_tools_to_response_tools(session);
            let mut all_tools = upstream.tools.take().unwrap_or_default();
            all_tools.extend(mcp_response_tools);
            upstream.tools = Some(all_tools);
        }
        strip_image_generation_from_request_tools(&mut upstream, session);

        Self {
            ctx,
            tenant_request_meta,
            original_tools,
            user_function_names,
            upstream_tools: upstream.tools,
            last_usage: None,
        }
    }
}

fn build_iteration_request(
    original: &ResponsesRequest,
    upstream_tools: Option<Vec<ResponseTool>>,
    state: &AgentLoopState,
) -> ResponsesRequest {
    use openai_protocol::responses::StringOrContentParts;
    let upstream_items = match &state.upstream_input {
        ResponseInput::Items(items) => items.clone(),
        ResponseInput::Text(text) => vec![ResponseInputOutputItem::SimpleInputMessage {
            content: StringOrContentParts::String(text.clone()),
            role: "user".to_string(),
            r#type: None,
            phase: None,
        }],
    };
    let mut combined: Vec<ResponseInputOutputItem> =
        Vec::with_capacity(upstream_items.len() + state.transcript.len());
    combined.extend(upstream_items);
    combined.extend(state.transcript.iter().cloned());

    let mut request = original.clone();
    request.input = ResponseInput::Items(combined);
    request.tools = if state.tool_budget_exhausted {
        None
    } else {
        upstream_tools
    };
    if state.tool_budget_exhausted {
        request.tool_choice = None;
    } else if state.iteration > 1 {
        request.tool_choice = Some(ResponsesToolChoice::Options(ToolChoiceOptions::Auto));
    }
    request.store = Some(false);
    request.previous_response_id = None;
    request.conversation = None;
    request
}

#[async_trait]
impl<'a> AgentLoopAdapter<GrpcResponseStreamSink> for HarmonyStreamingAdapter<'a> {
    type FinalResponse = ();

    async fn call_upstream(
        &mut self,
        ctx: &AgentLoopContext<'_>,
        state: &mut AgentLoopState,
        sink: &mut GrpcResponseStreamSink,
    ) -> Result<(), AgentLoopError> {
        let request =
            build_iteration_request(ctx.original_request, self.upstream_tools.clone(), state);

        let (execution_result, _load_guards) = self
            .ctx
            .pipeline
            .execute_harmony_responses_streaming(
                &request,
                self.ctx,
                Some(self.tenant_request_meta.clone()),
            )
            .await
            .map_err(|e| AgentLoopError::Response(Box::new(e)))?;

        // The streaming processor parses the harmony token stream and
        // forwards chunk-level events to the sink. Tool-call events
        // ride on `LoopEvent::ToolCallEmissionStarted` /
        // `ArgumentsFragment` / `EmissionDone` (the sink classifies
        // family from its session-snapshot); text events still go
        // direct via `sink.emitter`. Either way the processor stays
        // policy-blind — no `has_exposed_tool` / response_format
        // lookup runs in this layer.
        let iteration_result =
            HarmonyStreamingProcessor::process_responses_iteration_stream(execution_result, sink)
                .await
                .map_err(AgentLoopError::Internal)?;

        match iteration_result {
            ResponsesIterationResult::ToolCallsFound {
                tool_calls,
                analysis,
                partial_text,
                usage,
                request_id,
            } => {
                if let Some(text) = analysis.as_ref() {
                    if !text.is_empty() {
                        state
                            .transcript
                            .push(ResponseInputOutputItem::new_reasoning(
                                format!("reasoning_{request_id}"),
                                vec![],
                                vec![ResponseReasoningContent::ReasoningText {
                                    text: text.clone(),
                                }],
                                Some("completed".to_string()),
                            ));
                    }
                }
                if !partial_text.is_empty() {
                    state.transcript.push(ResponseInputOutputItem::Message {
                        id: format!("msg_{request_id}"),
                        role: "assistant".to_string(),
                        content: vec![ResponseContentPart::OutputText {
                            text: partial_text.clone(),
                            annotations: vec![],
                            logprobs: None,
                        }],
                        status: Some("completed".to_string()),
                        phase: None,
                    });
                }

                // The harmony processor already fired matching
                // `ToolCallEmissionStarted/Fragment/Done` events on
                // the sink as tokens streamed and at parser-finalize.
                // Adapter only pushes raw fc identities into the
                // driver's classification queue.
                state
                    .pending_tool_calls
                    .extend(tool_calls.into_iter().map(|tc| LoopToolCall {
                        call_id: tc.id.clone(),
                        item_id: tc.id,
                        name: tc.function.name,
                        // Don't substitute "{}" for missing args — the
                        // shared executor's malformed-args path already
                        // surfaces an empty / unparsable payload as a
                        // clean tool-error item. Coercing to a valid
                        // empty object would silently execute a tool
                        // with empty arguments instead of failing
                        // loudly when the harmony parser truncates.
                        arguments: tc.function.arguments.unwrap_or_default(),
                        approval_request_id: None,
                    }));
                self.last_usage = Some(usage.clone());

                state.latest_turn = Some(LoopModelTurn {
                    message_text: if partial_text.is_empty() {
                        None
                    } else {
                        Some(partial_text)
                    },
                    reasoning_text: analysis,
                    usage: Some(usage),
                    request_id: Some(request_id),
                });
            }
            ResponsesIterationResult::Completed { response: _, usage } => {
                self.last_usage = Some(usage.clone());
                state.latest_turn = Some(LoopModelTurn {
                    usage: Some(usage),
                    ..Default::default()
                });
            }
        }

        Ok(())
    }

    async fn render_final(
        mut self,
        ctx: &AgentLoopContext<'_>,
        _state: AgentLoopState,
        mode: RenderMode,
        sink: &mut GrpcResponseStreamSink,
    ) -> Result<(), AgentLoopError> {
        let usage_for_persist = self.last_usage.clone();
        let usage_json = self.last_usage.take().map(|u| {
            let mut obj = json!({
                "input_tokens": u.prompt_tokens,
                "output_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            });
            if let Some(details) = &u.prompt_tokens_details {
                if details.cached_tokens > 0 {
                    obj["input_tokens_details"] = json!({ "cached_tokens": details.cached_tokens });
                }
            }
            if let Some(details) = &u.completion_tokens_details {
                if let Some(reasoning) = details.reasoning_tokens {
                    obj["output_tokens_details"] = json!({ "reasoning_tokens": reasoning });
                }
            }
            obj
        });
        sink.set_final_usage(usage_json);

        // Persist the streamed response so a subsequent request that
        // sets `previous_response_id` to this turn's id can resolve
        // it. The emitter's `finalize` reads its accumulated
        // output_items non-destructively, so the matching
        // `emit_completed` (fired via `LoopEvent::ResponseFinished`
        // right after `render_final` returns) still sees the same
        // state for the SSE payload.
        let mut final_response = sink.emitter.finalize(usage_for_persist);
        // Echo prev/conv from the user-provided request shape to
        // match what the non-streaming render path produces. The
        // stream emitter's per-event response object already carries
        // these (see the conversation echo normalization in
        // `emit_completed`), but the persisted record needs them too
        // so the `previous_response_id` lookup chain resolves.
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
            // attached to `incomplete_details`.
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

        // The `response.completed` event is fired by the sink in
        // response to `LoopEvent::ResponseFinished` /
        // `LoopEvent::ResponseIncomplete`, which the driver emits
        // automatically. `RenderMode::ApprovalInterrupt` flows the same
        // way — the emitter has already pushed the approval item by
        // the time we land here. Drop unused params to keep the trait
        // shape consistent with the non-streaming path.
        let _ = self.original_tools;
        let _ = self.user_function_names;
        Ok(())
    }
}
