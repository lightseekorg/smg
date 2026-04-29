//! `AgentLoopAdapter` impl for the gRPC harmony Responses surface.
//!
//! Harmony's upstream is a structured-generation pipeline that emits
//! parsed analysis / final / commentary channels per iteration. The
//! adapter:
//!
//! - rebuilds the `ResponsesRequest` for each iteration from
//!   `state.upstream_input + state.transcript`
//! - calls `pipeline.execute_harmony_responses` (non-streaming) or
//!   `serve_harmony_responses_stream` (streaming) — see the streaming
//!   wrapper below — and translates the outcome into
//!   `LoopModelTurn` + `state.pending_*`
//! - on `Completed` stashes the pipeline's pre-built `ResponsesResponse`
//!   so `render_final` can preserve harmony's existing output shape
//! - on `render_final` injects MCP metadata using the same shared
//!   `inject_mcp_output_items` contract the driver already records
//!   into `state.mcp_output_items`

use std::collections::HashSet;

use async_trait::async_trait;
use openai_protocol::responses::{
    ResponseContentPart, ResponseInputOutputItem, ResponseReasoningContent, ResponseTool,
    ResponsesRequest, ResponsesResponse,
};
use smg_mcp::McpToolSession;

use super::{
    common::strip_image_generation_from_request_tools,
    execution::convert_mcp_tools_to_response_tools,
};
use crate::{
    middleware::TenantRequestMeta,
    routers::{
        common::agent_loop::{
            build_response_from_state, build_responses_iteration_request, AgentLoopAdapter,
            AgentLoopContext, AgentLoopError, AgentLoopState, IterationInputOptions,
            IterationRequestOptions, LoopModelTurn, LoopToolCall, RenderMode, ResponseBuildHooks,
            StreamSink, UsageShape,
        },
        grpc::common::responses::{collect_user_function_names, ResponsesContext},
    },
};

/// Adapter state owned for the lifetime of one Responses request.
pub(crate) struct HarmonyAdapter<'a> {
    ctx: &'a ResponsesContext,
    tenant_request_meta: TenantRequestMeta,
    /// Tools the *client* sent (pre-MCP-injection). Restored onto the
    /// final response so internal MCP/builtin function tools never
    /// leak back to the caller.
    original_tools: Option<Vec<ResponseTool>>,
    user_function_names: HashSet<String>,
    /// Tools merged into the upstream request — original tools plus
    /// session-derived MCP function tools, with `image_generation`
    /// stripped when an MCP server has taken ownership of it.
    upstream_tools: Option<Vec<ResponseTool>>,
    /// Lazily filled by `Completed` iterations so `render_final` can
    /// reuse harmony's already-built response shape (output items,
    /// usage, etc.) instead of reconstructing it from `LoopModelTurn`.
    completed_response: Option<Box<ResponsesResponse>>,
}

impl<'a> HarmonyAdapter<'a> {
    pub(crate) fn new(
        ctx: &'a ResponsesContext,
        tenant_request_meta: TenantRequestMeta,
        request: &ResponsesRequest,
        session: &McpToolSession<'_>,
    ) -> Self {
        let original_tools = request.tools.clone();
        let user_function_names = collect_user_function_names(request);

        // Merge MCP tools and apply harmony's image_generation strip
        // before stashing the upstream tool list. We do it here once
        // instead of on each iteration so the driver's transcript-based
        // request rebuild stays a pure splice.
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
            completed_response: None,
        }
    }
}

#[async_trait]
impl<'a, S: StreamSink> AgentLoopAdapter<S> for HarmonyAdapter<'a> {
    type FinalResponse = ResponsesResponse;

    async fn call_upstream(
        &mut self,
        _ctx: &AgentLoopContext<'_>,
        state: &mut AgentLoopState,
        _sink: &mut S,
    ) -> Result<(), AgentLoopError> {
        let request = build_responses_iteration_request(
            // The adapter clones the user's `ResponsesRequest` into
            // `upstream_tools` storage at construction; the driver
            // exposes the original via ctx.original_request so we
            // do not have to keep a second copy here.
            _ctx.original_request,
            state,
            IterationRequestOptions::with_tool_override(
                IterationInputOptions::preserved_message(),
                None,
                self.upstream_tools.clone(),
            ),
        );

        let iteration_result = self
            .ctx
            .pipeline
            .execute_harmony_responses(&request, self.ctx, Some(self.tenant_request_meta.clone()))
            .await
            .map_err(|e| AgentLoopError::Response(Box::new(e)))?;

        use crate::routers::grpc::harmony::processor::ResponsesIterationResult;
        match iteration_result {
            ResponsesIterationResult::ToolCallsFound {
                tool_calls,
                analysis,
                partial_text,
                usage,
                request_id,
            } => {
                let session_opt = _ctx.session;

                // Stage assistant-side replay items the next iteration
                // (or the final renderer) needs to see in the transcript.
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

                // Push raw fc identities; the driver classifies into
                // gateway / caller fc and handles routing. NS path
                // doesn't fire wire emission events here (NoopSink
                // ignores them anyway); ST is handled by the
                // streaming adapter.
                let _ = session_opt;
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
            ResponsesIterationResult::Completed { response, usage } => {
                state.latest_turn = Some(LoopModelTurn {
                    usage: Some(usage),
                    ..Default::default()
                });
                self.completed_response = Some(response);
            }
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
            AgentLoopError::Internal(
                "harmony render_final called without an MCP session".to_string(),
            )
        })?;

        // `Completed` iterations stash harmony's pipeline-rendered
        // response (with channels, output items already shaped); every
        // other termination (function tool exit, max_tool_calls budget
        // hit, approval interrupt) lets the unified builder synthesize
        // from `latest_turn` + `pending_user_tool_calls`.
        let pre_built = self.completed_response.take().map(|r| *r);
        let hooks = ResponseBuildHooks {
            original_tools: self.original_tools.take(),
            user_function_names: self.user_function_names.clone(),
            response_id_override: None,
            usage_shape: UsageShape::Modern,
        };
        Ok(build_response_from_state(
            &state,
            pre_built,
            mode,
            ctx.original_request,
            &hooks,
            session,
        ))
    }
}
