//! Driver-owned construction of the terminal `ResponsesResponse`.
//!
//! The driver hands the builder `(state, mode, hooks)`, and the builder
//! emits items directly from loop primitives:
//!
//! * `latest_turn.message_text` / `reasoning_text` → assistant
//!   message + reasoning items
//! * `pending_user_tool_calls` → caller-declared `function_call`
//!   items (the only place a `function_call` ever leaves this builder)
//! * `mode`-specific items (`approval_items` for `ApprovalInterrupt`)
//! * `state.mcp_output_items` / `state.emitted_mcp_list_tools_items`
//!   → MCP wire items via `inject_mcp_output_items`
//!
//! Surface knobs (`tools` echo, response-id source, usage shape) live
//! on [`ResponseBuildHooks`] so each adapter only contributes what is
//! genuinely surface-specific.

use std::{
    collections::HashSet,
    time::{SystemTime, UNIX_EPOCH},
};

use openai_protocol::{
    common::{ConversationRef, Usage},
    responses::{
        InputTokensDetails, OutputTokensDetails, ResponseContentPart, ResponseOutputItem,
        ResponseStatus, ResponseTool, ResponseUsage, ResponsesRequest, ResponsesResponse,
        ResponsesUsage, SummaryTextContent,
    },
};
use serde_json::json;
use smg_mcp::McpToolSession;
use uuid::Uuid;

use super::{
    driver::RenderMode,
    state::{AgentLoopState, LoopUserToolCall},
};
use crate::routers::common::mcp_utils::inject_mcp_output_items;

fn unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Surface-specific knobs the unified builder needs but cannot derive
/// from loop state alone. Each adapter assembles one of these from its
/// own request-scoped fields and hands it to
/// [`build_response_from_state`].
pub(crate) struct ResponseBuildHooks {
    /// Tools the *client* sent before any session injection. Restored
    /// onto the response so internal MCP / hosted-builtin function
    /// tools never leak back to the caller.
    pub original_tools: Option<Vec<ResponseTool>>,
    /// Names of caller-declared function tools — the only fc names
    /// that should round-trip as `function_call` output items. The
    /// builder does not consult this directly; it is forwarded to
    /// [`inject_mcp_output_items`] for MCP visibility filtering.
    pub user_function_names: HashSet<String>,
    /// Per-request response id assigned outside the loop (e.g. via the
    /// caller's `x-response-id` header on the regular surface). When
    /// `None`, the builder falls back to `latest_turn.request_id` and
    /// finally to a freshly-minted v7 UUID.
    pub response_id_override: Option<String>,
    /// How the surface wants `usage` shaped. Responses API surfaces use the
    /// modern `input_tokens` / `output_tokens` schema.
    pub usage_shape: UsageShape,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum UsageShape {
    /// Modern Responses usage with `input_tokens_details` and
    /// `output_tokens_details.reasoning_tokens`.
    Modern,
}

/// Build the terminal `ResponsesResponse` from loop state + render
/// mode. `pre_built_base` is a non-empty harmony-pipeline-rendered
/// response from a `Completed` iteration; when present the builder
/// uses it as the starting point instead of synthesizing a fresh one
/// from `latest_turn` (harmony already shaped channels / output items
/// in a way the from-primitives path cannot reproduce). All other
/// surfaces — including harmony's interrupt / incomplete paths — pass
/// `None` and let the builder construct from primitives.
pub(crate) fn build_response_from_state(
    state: &AgentLoopState,
    pre_built_base: Option<ResponsesResponse>,
    mode: RenderMode,
    request: &ResponsesRequest,
    hooks: &ResponseBuildHooks,
    session: &McpToolSession<'_>,
) -> ResponsesResponse {
    let mut response =
        pre_built_base.unwrap_or_else(|| build_base_from_state(state, request, hooks));

    // Echo the request's tools, scrubbing internal-MCP server entries
    // so SMG-side implementation details never leak to the caller.
    response.tools = hooks
        .original_tools
        .clone()
        .unwrap_or_default()
        .into_iter()
        .filter(|tool| {
            let value = serde_json::to_value(tool).unwrap_or(serde_json::Value::Null);
            !session.should_hide_tool_json(&value, &hooks.user_function_names)
        })
        .collect();

    match mode {
        RenderMode::Normal => {}
        RenderMode::ApprovalInterrupt(approval_items) => {
            response.status = ResponseStatus::Completed;
            response.output.extend(approval_items);
        }
        RenderMode::Incomplete { reason } => {
            response.status = ResponseStatus::Incomplete;
            response.incomplete_details = Some(json!({ "reason": reason }));
        }
    }

    if response.completed_at.is_none()
        && matches!(
            response.status,
            ResponseStatus::Completed
                | ResponseStatus::Incomplete
                | ResponseStatus::Failed
                | ResponseStatus::Cancelled
        )
    {
        response.completed_at = Some(unix_timestamp());
    }

    inject_mcp_output_items(
        &mut response.output,
        state.emitted_mcp_list_tools_items.clone(),
        state.mcp_output_items.clone(),
        &hooks.user_function_names,
        session,
    );

    // Echo request-level fields onto the response. `store` is always
    // overridden because in-loop sub-calls force `store: false`.
    response
        .previous_response_id
        .clone_from(&request.previous_response_id);
    response.conversation = request
        .conversation
        .as_ref()
        .map(|c| ConversationRef::Object {
            id: c.as_id().to_string(),
        });
    response.store = request.store.unwrap_or(true);
    if response.model.is_empty() {
        response.model.clone_from(&request.model);
    }
    if response.instructions.is_none() {
        response.instructions.clone_from(&request.instructions);
    }
    if response.metadata.is_empty() {
        if let Some(metadata) = &request.metadata {
            response.metadata.clone_from(metadata);
        }
    }
    if response.safety_identifier.is_none() {
        response
            .safety_identifier
            .clone_from(&request.safety_identifier);
    }

    response
}

/// Synthesize the response base from loop primitives. Used when the
/// adapter has no pre-built response (regular surface always; harmony
/// surface only on the function-tool / max-budget / approval-interrupt
/// paths).
fn build_base_from_state(
    state: &AgentLoopState,
    request: &ResponsesRequest,
    hooks: &ResponseBuildHooks,
) -> ResponsesResponse {
    let latest = state.latest_turn.as_ref();
    let usage = latest
        .and_then(|t| t.usage.clone())
        .unwrap_or_else(|| Usage::from_counts(0, 0));
    let request_id = hooks
        .response_id_override
        .clone()
        .or_else(|| latest.and_then(|t| t.request_id.clone()))
        .unwrap_or_else(|| format!("resp_{}", Uuid::now_v7()));
    let analysis = latest.and_then(|t| t.reasoning_text.clone());
    let partial_text = latest
        .and_then(|t| t.message_text.clone())
        .unwrap_or_default();

    let mut output: Vec<ResponseOutputItem> = Vec::new();
    if let Some(analysis_text) = analysis {
        output.push(ResponseOutputItem::new_reasoning(
            format!("reasoning_{request_id}"),
            vec![SummaryTextContent::SummaryText {
                text: analysis_text,
            }],
            vec![],
            None,
        ));
    }
    if !partial_text.is_empty() {
        output.push(ResponseOutputItem::Message {
            id: format!("msg_{request_id}"),
            role: "assistant".to_string(),
            content: vec![ResponseContentPart::OutputText {
                text: partial_text,
                annotations: vec![],
                logprobs: None,
            }],
            status: "completed".to_string(),
            phase: None,
        });
    }
    // Caller-declared fc round-trip back to the caller as
    // `function_call` items. Gateway/MCP calls never reach here —
    // they are represented by `mcp_call` (executed) /
    // `mcp_approval_request` (gated) items spliced in by the caller of
    // this function.
    for tool_call in &state.pending_user_tool_calls {
        output.push(make_function_call_item(tool_call));
    }

    let created_at = unix_timestamp();

    ResponsesResponse::builder(&request_id, &request.model)
        .copy_from_request(request)
        .created_at(created_at)
        .completed_at(created_at)
        .status(ResponseStatus::Completed)
        .output(output)
        .usage(shape_usage(&usage, hooks.usage_shape))
        .build()
}

fn make_function_call_item(call: &LoopUserToolCall) -> ResponseOutputItem {
    ResponseOutputItem::FunctionToolCall {
        id: call.item_id.clone(),
        call_id: call.call_id.clone(),
        name: call.name.clone(),
        arguments: call.arguments.clone(),
        output: None,
        status: "completed".to_string(),
    }
}

fn shape_usage(usage: &Usage, shape: UsageShape) -> ResponsesUsage {
    match shape {
        UsageShape::Modern => ResponsesUsage::Modern(ResponseUsage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            input_tokens_details: usage
                .prompt_tokens_details
                .as_ref()
                .map(InputTokensDetails::from),
            output_tokens_details: usage.completion_tokens_details.as_ref().and_then(|d| {
                d.reasoning_tokens.map(|tokens| OutputTokensDetails {
                    reasoning_tokens: tokens,
                })
            }),
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use openai_protocol::responses::{ResponseInput, ResponsesRequest};
    use smg_mcp::{McpOrchestrator, McpToolSession};

    use super::*;
    use crate::routers::common::agent_loop::state::AgentLoopState;

    fn fresh_request() -> ResponsesRequest {
        ResponsesRequest {
            model: "model_x".to_string(),
            input: ResponseInput::Items(Vec::new()),
            ..Default::default()
        }
    }

    fn fresh_state() -> AgentLoopState {
        AgentLoopState::new(ResponseInput::Items(Vec::new()), HashSet::new())
    }

    fn hooks() -> ResponseBuildHooks {
        ResponseBuildHooks {
            original_tools: None,
            user_function_names: HashSet::new(),
            response_id_override: Some("resp_test".to_string()),
            usage_shape: UsageShape::Modern,
        }
    }

    #[test]
    fn incomplete_mode_sets_status_incomplete_and_attaches_details() {
        let orchestrator = McpOrchestrator::new_test();
        let session = McpToolSession::new(&orchestrator, vec![], "test-request");
        let state = fresh_state();
        let request = fresh_request();
        let response = build_response_from_state(
            &state,
            None,
            RenderMode::Incomplete {
                reason: "max_tool_calls".to_string(),
            },
            &request,
            &hooks(),
            &session,
        );
        assert_eq!(response.status, ResponseStatus::Incomplete);
        assert_eq!(
            response
                .incomplete_details
                .as_ref()
                .and_then(|d| d.get("reason"))
                .and_then(|v| v.as_str()),
            Some("max_tool_calls")
        );
    }
}
