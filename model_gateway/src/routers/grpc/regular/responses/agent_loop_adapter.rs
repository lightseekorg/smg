//! `AgentLoopAdapter` impls for the gRPC regular Responses surface.
//!
//! Regular's upstream is a chat-completion worker, so the adapter has
//! more translation work than harmony: every iteration converts the
//! Responses-native transcript to chat messages (via
//! `conversions::responses_to_chat`) and feeds the parsed
//! `LoopModelTurn` primitives back into loop state. The adapter never
//! shapes the final response itself — `render_final` is a thin
//! delegate over `build_response_from_state`, the driver-owned
//! constructor.
//!
//! Two adapters live here:
//!
//! - `RegularAdapter` — non-streaming, runs against
//!   `pipeline.execute_chat_for_responses` and returns
//!   `ResponsesResponse`.
//! - `RegularStreamingAdapter` — streaming, drives
//!   `pipeline.execute_chat` and routes chunk-level deltas through
//!   the gRPC sink's `ResponseStreamEventEmitter`.
//!
//! Both rebuild iteration requests from `state.upstream_input +
//! state.transcript` so the agent loop's transcript is the single
//! source of truth — there is no per-surface "current_request" mutated
//! between iterations.

use std::{collections::HashSet, sync::Arc};

use async_trait::async_trait;
use axum::http;
use openai_protocol::{
    chat::ChatCompletionResponse,
    common::{ToolChoice, ToolChoiceValue},
    responses::{ResponsesRequest, ResponsesResponse},
};
use smg_mcp::McpToolSession;

use super::{common::append_assistant_prefix_to_transcript, conversions};
use crate::{
    middleware::TenantRequestMeta,
    routers::{
        common::agent_loop::{
            build_response_from_state, build_responses_iteration_request, AgentLoopAdapter,
            AgentLoopContext, AgentLoopError, AgentLoopState, IterationRequestFlavor,
            LoopModelTurn, LoopToolCall, RenderMode, ResponseBuildHooks, StreamSink, UsageShape,
        },
        grpc::common::responses::{collect_user_function_names, ResponsesContext},
    },
};

/// Per-request handle the chat pipeline call needs but the loop's
/// canonical state does not own. The driver passes only loop state to
/// the adapter; everything provider-specific (headers, tenant meta,
/// caller-assigned response_id) lives here.
#[derive(Clone)]
pub(crate) struct RegularUpstreamHandle {
    pub headers: Option<http::HeaderMap>,
    pub model_id: String,
    pub response_id: Option<String>,
    pub tenant_request_meta: TenantRequestMeta,
}

pub(crate) struct RegularAdapter<'a> {
    ctx: &'a ResponsesContext,
    upstream: RegularUpstreamHandle,
    user_function_names: HashSet<String>,
    /// MCP tools rendered into the chat-tools array shape and cloned
    /// into each iteration's chat request.
    mcp_chat_tools: Vec<openai_protocol::common::Tool>,
}

impl<'a> RegularAdapter<'a> {
    pub(crate) fn new(
        ctx: &'a ResponsesContext,
        upstream: RegularUpstreamHandle,
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
        }
    }
}

fn extract_tool_calls(response: &ChatCompletionResponse) -> Vec<LoopToolCall> {
    response
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
        .unwrap_or_default()
}

#[async_trait]
impl<'a, S: StreamSink> AgentLoopAdapter<S> for RegularAdapter<'a> {
    type FinalResponse = ResponsesResponse;

    async fn call_upstream(
        &mut self,
        ctx: &AgentLoopContext<'_>,
        state: &mut AgentLoopState,
        _sink: &mut S,
    ) -> Result<(), AgentLoopError> {
        let request = build_responses_iteration_request(
            ctx.original_request,
            state,
            IterationRequestFlavor::RegularChat { stream: None },
        );

        let mut chat_request = conversions::responses_to_chat(&request).map_err(|e| {
            AgentLoopError::InvalidRequest(format!("Failed to convert request: {e}"))
        })?;

        // Merge MCP tools and pin tool_choice for continuations
        // (mirrors the old `prepare_chat_tools_and_choice`).
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
            .execute_chat_for_responses(
                Arc::new(chat_request),
                self.upstream.headers.clone(),
                self.upstream.model_id.clone(),
                self.ctx.components.clone(),
                Some(self.upstream.tenant_request_meta.clone()),
            )
            .await
            .map_err(|e| AgentLoopError::Response(Box::new(e)))?;

        // Hand off to the driver as a uniform `LoopModelTurn` —
        // message text, reasoning text, usage, tool_calls. The driver
        // (and the unified builder) read only from this state, so the
        // adapter never has to shape the response itself again.
        let first_choice = chat_response.choices.first();
        let message_text = first_choice.and_then(|c| c.message.content.clone());
        let reasoning_text = first_choice.and_then(|c| c.message.reasoning_content.clone());

        state.pending_tool_calls = extract_tool_calls(&chat_response);
        if !state.pending_tool_calls.is_empty() {
            append_assistant_prefix_to_transcript(
                state,
                &chat_response.id,
                reasoning_text.as_deref(),
                message_text.as_deref(),
            );
        }
        state.latest_turn = Some(LoopModelTurn {
            message_text: message_text.filter(|s| !s.is_empty()),
            reasoning_text: reasoning_text.filter(|s| !s.is_empty()),
            usage: chat_response
                .usage
                .as_ref()
                .map(|u| openai_protocol::common::Usage {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                    total_tokens: u.total_tokens,
                    prompt_tokens_details: None,
                    completion_tokens_details: u.completion_tokens_details.clone(),
                }),
            request_id: Some(chat_response.id.clone()),
        });

        Ok(())
    }

    async fn render_final(
        self,
        ctx: &AgentLoopContext<'_>,
        state: AgentLoopState,
        mode: RenderMode,
        _sink: &mut S,
    ) -> Result<Self::FinalResponse, AgentLoopError> {
        let session = ctx.session.ok_or_else(|| {
            AgentLoopError::Internal(
                "regular render_final called without an MCP session".to_string(),
            )
        })?;
        let hooks = ResponseBuildHooks {
            original_tools: ctx.original_request.tools.clone(),
            user_function_names: self.user_function_names,
            response_id_override: self.upstream.response_id.clone(),
            usage_shape: UsageShape::Modern,
        };
        Ok(build_response_from_state(
            &state,
            None,
            mode,
            ctx.original_request,
            &hooks,
            session,
        ))
    }
}
