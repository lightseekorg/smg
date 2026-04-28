//! Mutable loop state and the action the driver takes next.
//!
//! Every iteration's mutations live on `AgentLoopState`. There is no
//! parallel "runtime" struct holding loop state in a second place —
//! adapters thread their own per-request data (ctx, params) through
//! their adapter struct, but anything that changes between iterations
//! belongs here.

use std::collections::HashSet;

use openai_protocol::{
    common::Usage,
    responses::{ResponseInputOutputItem, ResponseOutputItem},
};

use super::presentation::ToolPresentation;

/// Control signal for the driver. After every step the current step
/// sets `state.next_action` explicitly; the driver matches on it.
/// There is no implicit "if has_mcp_tools then loop else exit" branch
/// outside this enum.
#[derive(Debug, Clone)]
pub(crate) enum NextAction {
    /// Build the next upstream request from the canonical transcript
    /// and execute one model turn.
    CallLlm,
    /// Execute the gateway-owned (MCP / builtin) tools queued in
    /// `state.pending_gateway_tool_calls`. User-managed function-tool
    /// calls do not produce this action — gRPC adapters surface
    /// those through their own per-surface accumulators and rely on
    /// `state.stop_after_tool_execution` to short-circuit to `Finish`.
    ExecuteTools(Vec<PlannedToolExecution>),
    /// The next gated MCP call in the most recent turn requires user
    /// approval. Render the approval-interrupt response with a single
    /// `mcp_approval_request` item and return immediately. Per the
    /// OpenAI Responses contract a turn surfaces one approval boundary
    /// at a time — if the model emitted multiple gated calls, the
    /// remaining calls reissue on the continuation turn (the model
    /// re-emits them after seeing the first call's result).
    InterruptForApproval(PendingToolExecution),
    /// Render the final response from current state (terminal or
    /// incomplete) and return.
    Finish,
}

/// Tool call extracted from a model turn.
///
/// Carries **identity** — `call_id`, `item_id`, `name`, `arguments` —
/// plus the optional `mcpr_*` approval-request id when the call is an
/// approval continuation. Anything else session-derived (server
/// label, wire-side presentation) is enriched by the driver at
/// partition time and rides on [`PlannedToolExecution`] /
/// [`PendingToolExecution`] / event payloads.
#[derive(Debug, Clone)]
pub(crate) struct LoopToolCall {
    /// Wire `call_id` (the model-emitted correlation id, `call_*`).
    pub call_id: String,
    /// Wire `id` (item id, `fc_*`). Falls back to `call_id` when the
    /// upstream omitted an explicit item id.
    pub item_id: String,
    pub name: String,
    pub arguments: String,
    /// `mcpr_<call_id>` approval-request id when this call originated
    /// from an `mcp_approval_response` continuation. The transformer
    /// echoes it back into the rendered `mcp_call.approval_request_id`
    /// field so clients can correlate the eventual completion with the
    /// approval prompt the gateway emitted on the prior turn.
    pub approval_request_id: Option<String>,
}

/// User-managed function tool call returned to the caller as part of
/// the final response. Populated by every adapter that detects a
/// caller-declared `Function` tool call in the model's turn — the
/// loop driver does not execute these (gateway tools are the only
/// kind it dispatches), but it still owns the typed channel so the
/// final-render paths read from one place. gRPC adapters and the
/// future OpenAI adapter both write into
/// [`AgentLoopState::pending_user_tool_calls`]; render_final pulls
/// items out of that channel to splice into the response output.
#[derive(Debug, Clone)]
pub(crate) struct LoopUserToolCall {
    pub call_id: String,
    pub item_id: String,
    pub name: String,
    pub arguments: String,
}

/// One scheduled MCP tool execution. The `call` field carries the
/// raw model-emitted identity; the other fields are session-derived
/// enrichment the driver adds at partition time. Sinks read all
/// rendering-relevant data from this struct instead of going back to
/// the session, so the streaming side never holds a `&McpToolSession`.
#[derive(Debug, Clone)]
pub(crate) struct PlannedToolExecution {
    pub call: LoopToolCall,
    pub server_label: String,
    pub presentation: ToolPresentation,
}

/// One MCP tool call that hit an approval boundary instead of
/// executing. Carries `server_label` because the rendered
/// `mcp_approval_request` item needs it; `presentation` is **not**
/// stored because the approval-request item shape is family-agnostic
/// (always `mcp_approval_request`, never a hosted-builtin family —
/// hosted builtins are `require_approval: never` by definition).
#[derive(Debug, Clone)]
pub(crate) struct PendingToolExecution {
    pub call: LoopToolCall,
    pub server_label: String,
}

/// Result of one completed (or errored) MCP execution.
#[derive(Debug, Clone)]
pub(crate) struct ExecutedCall {
    pub call_id: String,
    pub item_id: String,
    pub name: String,
    pub arguments: String,
    /// Stringified tool output as the gateway will surface it back to
    /// the upstream model on the next turn.
    pub output_string: String,
    /// Surface-rendered output item (`mcp_call`, `web_search_call`,
    /// `code_interpreter_call`, etc.) for client-visible injection.
    pub transformed_item: Option<ResponseOutputItem>,
    pub is_error: bool,
    /// Forwarded from [`LoopToolCall::approval_request_id`] for calls
    /// that originated from an approval continuation. The presentation
    /// layer stamps it onto the rendered `mcp_call.approval_request_id`
    /// field so the wire payload echoes the approval prompt.
    pub approval_request_id: Option<String>,
}

/// Result of one model turn. Surface adapters fill this in from
/// their upstream response shape; the driver consumes only this
/// struct.
#[derive(Debug, Clone, Default)]
pub(crate) struct LoopModelTurn {
    /// Final-channel text (assistant message body).
    pub message_text: Option<String>,
    /// Analysis-channel text (reasoning_content).
    pub reasoning_text: Option<String>,
    pub usage: Option<Usage>,
    /// Upstream-assigned request id, used by some renderers to derive
    /// stable item ids.
    pub request_id: Option<String>,
}

/// Mutable per-request loop state.
pub(crate) struct AgentLoopState {
    pub iteration: usize,
    pub total_gateway_tool_calls: usize,

    /// LLM-consumable transcript items appended this request — the
    /// `function_call` / `function_call_output` / assistant-message
    /// pairs synthesized as the loop runs. Surface adapters splice
    /// this onto `prepared.upstream_input` to build each next turn's
    /// upstream payload.
    pub transcript: Vec<ResponseInputOutputItem>,

    /// Original input the client sent, captured before the loop starts
    /// so each next-turn upstream payload can rebuild from it.
    pub upstream_input: openai_protocol::responses::ResponseInput,

    /// Client-visible MCP output items collected this request — the
    /// `mcp_call` / `web_search_call` / `code_interpreter_call` /
    /// `file_search_call` items injected at render time. Stays in
    /// loop state across iterations; rendered into the final response
    /// during `Finish` (and approval-interrupt / incomplete renders).
    pub mcp_output_items: Vec<ResponseOutputItem>,

    /// Server labels for which a `mcp_list_tools` item has already been
    /// emitted (either in a prior turn loaded via `previous_response_id`
    /// or earlier in this request's stream). Dedupe is by label per
    /// the design doc's contract.
    pub emitted_mcp_server_labels: HashSet<String>,

    /// `mcp_list_tools` items already pushed to a streaming client this
    /// request. Final `response.completed` must reuse these exact items
    /// (same ids, same tool list) per the streaming contract.
    pub emitted_mcp_list_tools_items: Vec<ResponseOutputItem>,

    /// Output of the most recent CallLlm. Populated by the adapter's
    /// `call_upstream`, consumed by the driver to decide the next
    /// action.
    pub latest_turn: Option<LoopModelTurn>,

    /// Raw tool calls the adapter saw in the most recent model turn,
    /// not yet classified or routed. The adapter pushes every fc
    /// here without consulting the session — classification (gateway
    /// vs caller-declared user fc) and approval gating happen in the
    /// driver's `decide_after_call_llm`.
    pub pending_tool_calls: Vec<LoopToolCall>,

    /// Driver-classified gateway tool calls awaiting `ExecuteTools`.
    /// Used as carry-over storage when a turn produces a mix of
    /// approval-gated and immediate calls — the immediate side runs
    /// first, the gated side stays here until `decide_after_execute_tools`
    /// surfaces them as `mcp_approval_request` items. Adapters do
    /// **not** write to this field.
    pub pending_gateway_tool_calls: Vec<LoopToolCall>,

    /// Driver-populated record of caller-declared function tool calls
    /// from the most recent turn(s). Wire-side these have already
    /// been emitted by the streaming sink (via `ToolCallEmissionStarted`
    /// / `Fragment` / `Done`); this field exists so non-streaming
    /// `render_final` paths can splice the corresponding `function_call`
    /// output items into the final response. Adapters do **not** write
    /// to this field.
    pub pending_user_tool_calls: Vec<LoopUserToolCall>,

    /// True when the most recent turn produced any user-managed
    /// function tool call. After `ExecuteTools` finishes the gateway
    /// portion the loop transitions to `Finish` instead of `CallLlm`,
    /// so the user-managed siblings reach the caller.
    pub stop_after_tool_execution: bool,

    /// Set when the driver wants to terminate with `incomplete_details`
    /// instead of `Normal`.
    pub incomplete: Option<IncompleteTermination>,

    /// True after `max_tool_calls` has been reached. The next model turn is
    /// forced to run without tools so it can produce a final answer from the
    /// tool results already in the transcript, matching OpenAI Responses.
    pub tool_budget_exhausted: bool,

    next_action: NextAction,
}

#[derive(Debug, Clone)]
pub(crate) struct IncompleteTermination {
    pub reason: String,
}

impl AgentLoopState {
    pub(crate) fn new(
        upstream_input: openai_protocol::responses::ResponseInput,
        emitted_mcp_server_labels: HashSet<String>,
    ) -> Self {
        Self {
            iteration: 0,
            total_gateway_tool_calls: 0,
            transcript: Vec::new(),
            upstream_input,
            mcp_output_items: Vec::new(),
            emitted_mcp_server_labels,
            emitted_mcp_list_tools_items: Vec::new(),
            latest_turn: None,
            pending_tool_calls: Vec::new(),
            pending_gateway_tool_calls: Vec::new(),
            pending_user_tool_calls: Vec::new(),
            stop_after_tool_execution: false,
            incomplete: None,
            tool_budget_exhausted: false,
            next_action: NextAction::CallLlm,
        }
    }

    pub(crate) fn take_next_action(&mut self) -> NextAction {
        std::mem::replace(&mut self.next_action, NextAction::Finish)
    }

    pub(crate) fn set_next_action(&mut self, action: NextAction) {
        self.next_action = action;
    }

    /// Append an executed MCP call's transcript pair (assistant
    /// `function_call` + `function_call_output`) and stash the
    /// transformed item for final-render injection.
    pub(crate) fn record_executed_call(&mut self, executed: ExecutedCall) {
        self.transcript
            .push(ResponseInputOutputItem::FunctionToolCall {
                id: executed.item_id.clone(),
                call_id: executed.call_id.clone(),
                name: executed.name.clone(),
                arguments: executed.arguments.clone(),
                output: None,
                status: Some("completed".to_string()),
            });
        self.transcript
            .push(ResponseInputOutputItem::FunctionCallOutput {
                id: None,
                call_id: executed.call_id.clone(),
                output: executed.output_string.clone(),
                status: Some("completed".to_string()),
            });
        if let Some(item) = executed.transformed_item {
            self.mcp_output_items.push(item);
        }
    }
}
