//! Streaming flavor of the Harmony agent-loop adapter.
//!
//! Uses `pipeline::execute_harmony_responses_streaming` and feeds
//! chunk-level events into a `GrpcResponseStreamSink`.

use async_trait::async_trait;
use openai_protocol::responses::{
    ResponseContentPart, ResponseInputOutputItem, ResponseReasoningContent, ResponseTool,
    ResponsesRequest,
};
use serde_json::{json, Value};
use smg_mcp::McpToolSession;

use super::{
    common::strip_image_generation_from_request_tools,
    execution::convert_mcp_tools_to_response_tools,
};
use crate::{
    middleware::TenantRequestMeta,
    routers::{
        common::agent_loop::{
            build_responses_iteration_request, AgentLoopAdapter, AgentLoopContext, AgentLoopError,
            AgentLoopState, IterationRequestFlavor, LoopModelTurn, LoopToolCall, RenderMode,
        },
        grpc::{
            common::responses::{
                finalize_streamed_response_for_persist, GrpcResponseStreamSink, ResponsesContext,
                StreamingPersistHandles,
            },
            harmony::{processor::ResponsesIterationResult, streaming::HarmonyStreamingProcessor},
        },
    },
};

pub(crate) struct HarmonyStreamingAdapter<'a> {
    ctx: &'a ResponsesContext,
    tenant_request_meta: TenantRequestMeta,
    upstream_tools: Option<Vec<ResponseTool>>,
}

impl<'a> HarmonyStreamingAdapter<'a> {
    pub(crate) fn new(
        ctx: &'a ResponsesContext,
        tenant_request_meta: TenantRequestMeta,
        request: &ResponsesRequest,
        session: &McpToolSession<'_>,
    ) -> Self {
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
            upstream_tools: upstream.tools,
        }
    }
}

fn usage_to_stream_json(usage: &openai_protocol::common::Usage) -> Value {
    let mut obj = json!({
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    });
    if let Some(details) = &usage.prompt_tokens_details {
        if details.cached_tokens > 0 {
            obj["input_tokens_details"] = json!({ "cached_tokens": details.cached_tokens });
        }
    }
    if let Some(details) = &usage.completion_tokens_details {
        if let Some(reasoning) = details.reasoning_tokens {
            obj["output_tokens_details"] = json!({ "reasoning_tokens": reasoning });
        }
    }
    obj
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
        let request = build_responses_iteration_request(
            ctx.original_request,
            state,
            IterationRequestFlavor::Responses {
                stream: None,
                tools: self.upstream_tools.clone(),
            },
        );

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
                state.latest_turn = Some(LoopModelTurn {
                    usage: Some(usage),
                    ..Default::default()
                });
            }
        }

        Ok(())
    }

    async fn render_final(
        self,
        ctx: &AgentLoopContext<'_>,
        state: AgentLoopState,
        mode: RenderMode,
        sink: &mut GrpcResponseStreamSink,
    ) -> Result<(), AgentLoopError> {
        let usage_for_persist = state
            .latest_turn
            .as_ref()
            .and_then(|turn| turn.usage.clone());
        let usage_json = usage_for_persist.as_ref().map(usage_to_stream_json);
        sink.set_final_usage(usage_json);

        finalize_streamed_response_for_persist(
            sink,
            usage_for_persist,
            &mode,
            ctx.original_request,
            StreamingPersistHandles {
                conversation_storage: self.ctx.conversation_storage.clone(),
                conversation_item_storage: self.ctx.conversation_item_storage.clone(),
                response_storage: self.ctx.response_storage.clone(),
                request_context: self.ctx.request_context.clone(),
            },
        )
        .await;

        // The sink emits the terminal SSE event after the driver sends the
        // terminal loop event.
        Ok(())
    }
}
