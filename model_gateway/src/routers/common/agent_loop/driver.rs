//! Loop driver — `run_agent_loop` plus the `AgentLoopAdapter` trait
//! that surface adapters (openai / grpc-regular / grpc-harmony) plug
//! into.
//!
//! The driver owns control flow. It alternates between asking the
//! adapter to call the model, executing gateway-owned tools through
//! `McpToolSession`, and handing off to the adapter's renderer at one
//! of three terminal modes (`Normal`, `ApprovalInterrupt`,
//! `Incomplete`). Surface adapters are forbidden from making loop
//! control decisions of their own.

use async_trait::async_trait;
use openai_protocol::responses::{ResponseInputOutputItem, ResponseOutputItem, ResponsesRequest};
use smg_mcp::McpToolSession;

use super::{
    error::AgentLoopError,
    events::{LoopEvent, StreamSink},
    prepared::PreparedLoopInput,
    state::{
        AgentLoopState, IncompleteTermination, LoopToolCall, LoopUserToolCall, NextAction,
        PendingToolExecution, PlannedToolExecution,
    },
    tooling::{execute_planned_tool, remaining_tool_call_budget},
};
use crate::{observability::metrics::Metrics, routers::common::mcp_utils::DEFAULT_MAX_ITERATIONS};

/// Terminal render mode the driver hands to `render_final`. The
/// adapter shapes the actual response body around this signal.
#[derive(Debug, Clone)]
pub(crate) enum RenderMode {
    /// Loop completed normally — last turn produced no executable
    /// gateway tool calls, or only user-managed function calls.
    Normal,
    /// One MCP call requires user approval. The driver attaches the
    /// rendered `mcp_approval_request` output item and the adapter
    /// splices it into the rendered response in turn order.
    ApprovalInterrupt(Vec<ResponseOutputItem>),
    /// Loop terminated early due to an iteration/output limit.
    Incomplete { reason: String },
}

/// Per-request immutable context. Holds the prepared input, the
/// optional MCP session, and the original client request so adapters
/// can read fields they do not get to mutate.
pub(crate) struct AgentLoopContext<'a> {
    pub prepared: &'a PreparedLoopInput,
    pub session: Option<&'a McpToolSession<'a>>,
    pub original_request: &'a ResponsesRequest,
    pub max_tool_calls: Option<usize>,
}

/// Surface-specific plug surface. Each impl owns the upstream wire
/// format it talks (chat completions / harmony pipeline / OpenAI
/// Responses passthrough) and the final response body shape it
/// returns. The sink type is a generic so the same adapter struct can
/// run non-streaming (with `NoopSink`) and streaming (with the
/// surface's `GrpcResponseStreamSink`) requests without forking the
/// adapter type — that would otherwise duplicate every adapter just
/// to flip one associated type. The trait stays narrow either way:
/// every loop-control concern lives inside `run_agent_loop`.
#[async_trait]
pub(crate) trait AgentLoopAdapter<S: StreamSink>: Send {
    type FinalResponse: Send;

    /// Run one model turn. Adapter consults `state` for the rebuilt
    /// transcript, calls its upstream, fills `state.latest_turn` /
    /// `state.pending_gateway_tool_calls` /
    /// `state.pending_user_tool_calls`, and emits chunk-level events
    /// through `sink`. Adapter does not decide what comes next — only
    /// the driver does.
    async fn call_upstream(
        &mut self,
        ctx: &AgentLoopContext<'_>,
        state: &mut AgentLoopState,
        sink: &mut S,
    ) -> Result<(), AgentLoopError>;

    /// Build the terminal response. The driver passes consumed loop
    /// state plus the render mode so all three terminations
    /// (`Normal` / `ApprovalInterrupt` / `Incomplete`) flow through
    /// one shared render boundary per the design doc.
    async fn render_final(
        self,
        ctx: &AgentLoopContext<'_>,
        state: AgentLoopState,
        mode: RenderMode,
        sink: &mut S,
    ) -> Result<Self::FinalResponse, AgentLoopError>;
}

/// Drive a Responses request through the shared agent loop until it
/// reaches a terminal render mode. Surface adapters call this in place
/// of any per-surface `execute_tool_loop` / `execute_with_mcp_loop`
/// logic.
pub(crate) async fn run_agent_loop<S: StreamSink, A: AgentLoopAdapter<S>>(
    mut adapter: A,
    ctx: AgentLoopContext<'_>,
    mut state: AgentLoopState,
    mut sink: S,
) -> Result<A::FinalResponse, AgentLoopError> {
    // Resolve approval continuation from the loop's control vocabulary
    // before starting the model loop. The driver owns the entire
    // approval state machine: an approve continuation builds enriched
    // `PlannedToolExecution` items directly (server label + presentation
    // looked up from session here, not by the adapter) and sets
    // `next_action = ExecuteTools(...)` so the loop skips the opening
    // `CallLlm`. The model never sees a fake "approved by user"
    // message — only the real tool result on the turn after
    // execution. A deny synthesizes a `function_call +
    // function_call_output("Tool call denied by user[: <reason>]")`
    // pair in transcript and falls through to `CallLlm` so the model
    // can react. Orphan responses still 400.
    prime_pending_from_approval(&ctx.prepared.control_items, ctx.session, &mut state)?;

    sink.emit(LoopEvent::ResponseStarted);

    // Emit `mcp_list_tools` items for every connected MCP server label
    // that has not already been listed in a prior turn. The dedupe is
    // keyed on `state.emitted_mcp_server_labels` per the design doc;
    // streaming sinks emit the four-event sequence (added /
    // in_progress / completed / done), non-streaming sinks ignore.
    if let Some(session) = ctx.session {
        for binding in session.mcp_servers() {
            if state.emitted_mcp_server_labels.contains(&binding.label) {
                continue;
            }
            if session.is_internal_server_label(&binding.label) {
                continue;
            }
            let item = session.build_mcp_list_tools_item(&binding.label, &binding.server_key);
            sink.emit(LoopEvent::McpListToolsItem { item: &item });
            state.emitted_mcp_list_tools_items.push(item);
            state
                .emitted_mcp_server_labels
                .insert(binding.label.clone());
        }
    }

    loop {
        match state.take_next_action() {
            NextAction::CallLlm => {
                state.iteration += 1;
                if state.iteration > effective_iteration_cap() {
                    state.incomplete = Some(IncompleteTermination {
                        reason: "max_iterations".to_string(),
                    });
                    state.set_next_action(NextAction::Finish);
                    continue;
                }

                adapter.call_upstream(&ctx, &mut state, &mut sink).await?;

                // Decide post-CallLlm transition entirely from state —
                // no per-surface branching is allowed here. Session
                // is the source of truth for per-call policy
                // (server_label / presentation / requires_approval);
                // the driver consults it here rather than relying on
                // adapter-pre-filled fields on `LoopToolCall`.
                let next = decide_after_call_llm(&mut state, ctx.session);
                state.set_next_action(next);
            }

            NextAction::ExecuteTools(plan) => {
                let session = ctx.session.ok_or_else(|| {
                    AgentLoopError::Internal(
                        "ExecuteTools selected but no MCP session attached to context".to_string(),
                    )
                })?;
                run_execute_tools::<S>(&mut state, session, plan, &ctx, &mut sink).await?;
                let next = decide_after_execute_tools(&mut state, ctx.session);
                state.set_next_action(next);
            }

            NextAction::InterruptForApproval(pending) => {
                let approval_items =
                    build_mcp_approval_request_items(std::slice::from_ref(&pending));
                sink.emit(LoopEvent::ApprovalRequested { pending: &pending });
                // Run `render_final` *before* `ResponseFinished` so
                // streaming adapters get a chance to stage final
                // usage / metadata via `sink.set_final_usage` (or the
                // equivalent) before the sink builds the
                // `response.completed` payload.
                let final_response = adapter
                    .render_final(
                        &ctx,
                        state,
                        RenderMode::ApprovalInterrupt(approval_items),
                        &mut sink,
                    )
                    .await?;
                sink.emit(LoopEvent::ResponseFinished);
                return Ok(final_response);
            }

            NextAction::Finish => {
                let incomplete = state.incomplete.take();
                let (mode, finish_reason) = match incomplete {
                    Some(IncompleteTermination { reason }) => {
                        let mode = RenderMode::Incomplete {
                            reason: reason.clone(),
                        };
                        (mode, Some(reason))
                    }
                    None => (RenderMode::Normal, None),
                };
                // Same ordering as `InterruptForApproval`: stage usage
                // through `render_final` first, then fire the
                // termination event so `response.completed` carries the
                // final-turn usage on the wire.
                let final_response = adapter.render_final(&ctx, state, mode, &mut sink).await?;
                match finish_reason {
                    Some(reason) => sink.emit(LoopEvent::ResponseIncomplete {
                        reason: reason.as_str(),
                    }),
                    None => sink.emit(LoopEvent::ResponseFinished),
                }
                return Ok(final_response);
            }
        }
    }
}

/// Resolve approval continuation by pairing each `mcp_approval_response`
/// in the loop's control vocabulary with its matching
/// `mcp_approval_request`, then translating the user's decision into a
/// concrete state-machine transition that the rest of the loop drives
/// through normally:
///
/// - **approve = true.** The originally-paused call (carrying its
///   stored `name` / `arguments` / `server_label`) is validated against
///   the current MCP session, enriched into `PlannedToolExecution`, and
///   the next action is primed to `ExecuteTools(...)` so the loop
///   **skips** the opening `CallLlm`. The tool runs through the normal
///   execute path; only the real tool output reaches the model on the
///   following turn. At no point does the model see a synthesized
///   "approved by user" message — that would leak control-layer state
///   into the prompt.
///
/// - **approve = false.** A `function_call + function_call_output`
///   pair carrying `"Tool call denied by user[: <reason>]"` is
///   appended to `state.transcript`. The default `CallLlm` action
///   then runs so the model can react to the denial in its next
///   reply. No execution happens.
///
/// Returns `Err(AgentLoopError::InvalidRequest)` when a response has
/// no matching request — orphan stitching errors are unrecoverable.
fn prime_pending_from_approval(
    control_items: &[ResponseInputOutputItem],
    session: Option<&McpToolSession<'_>>,
    state: &mut AgentLoopState,
) -> Result<(), AgentLoopError> {
    use std::collections::{HashMap, HashSet};

    use super::{presentation::ToolPresentation, state::LoopToolCall};

    let mut requests: HashMap<&str, (&str, &str, &str)> = HashMap::new();
    let mut responses: Vec<(&str, bool, Option<&str>)> = Vec::new();
    let mut seen_responses: HashSet<&str> = HashSet::new();
    for item in control_items {
        match item {
            ResponseInputOutputItem::McpApprovalRequest {
                id,
                server_label,
                name,
                arguments,
            } => {
                requests.insert(
                    id.as_str(),
                    (server_label.as_str(), name.as_str(), arguments.as_str()),
                );
            }
            // The chain can carry the same approval pair across
            // multiple turns once it has been settled; only the
            // first occurrence in iteration order should drive a
            // plan. Later duplicates would otherwise build a
            // second `PlannedToolExecution` for the same call_id.
            ResponseInputOutputItem::McpApprovalResponse {
                approval_request_id,
                approve,
                reason,
                ..
            } if seen_responses.insert(approval_request_id.as_str()) => {
                responses.push((approval_request_id.as_str(), *approve, reason.as_deref()));
            }
            _ => {}
        }
    }

    if responses.is_empty() {
        return Ok(());
    }

    // Approvals already settled on a previous turn surface as
    // `FunctionCallOutput` entries in the lowered transcript:
    // `transcript_lower` projects historical `mcp_call` items into
    // `FunctionToolCall` + `FunctionCallOutput` pairs, and the deny
    // branch below pushes the same pair on the current turn. Skipping
    // approval responses whose call_id already has output prevents
    // `prime_pending_from_approval` from re-executing non-idempotent
    // MCP tools when a long `previous_response_id` chain replays past
    // approval pairs.
    // Own the keys so the loop body below can still push new
    // FunctionToolCall / FunctionCallOutput entries to
    // `state.transcript` (deny branch).
    let resolved_call_ids: HashSet<String> = state
        .transcript
        .iter()
        .filter_map(|item| match item {
            ResponseInputOutputItem::FunctionCallOutput { call_id, .. } => Some(call_id.clone()),
            _ => None,
        })
        .collect();

    let mut planned: Vec<PlannedToolExecution> = Vec::new();
    // Per-pass dedup so two responses whose `approval_request_id`s
    // normalize to the same derived `call_id` cannot both execute.
    let mut processed_call_ids: HashSet<String> = HashSet::new();
    for (approval_request_id, approve, reason) in responses {
        let derived_call_id = approval_request_id
            .strip_prefix("mcpr_")
            .unwrap_or(approval_request_id);
        if resolved_call_ids.contains(derived_call_id)
            || !processed_call_ids.insert(derived_call_id.to_string())
        {
            continue;
        }

        let Some((server_label, tool_name, arguments)) = requests.get(approval_request_id).copied()
        else {
            return Err(AgentLoopError::InvalidRequest(format!(
                "approval_continuation_invalid: `mcp_approval_response.approval_request_id` \
                 = '{approval_request_id}' does not match any prior `mcp_approval_request` in \
                 the request's history or input. Include the matching request item in \
                 `previous_response_id` history, `conversation` items, or hand-stitched \
                 `input`, or omit the orphan response."
            )));
        };

        // The approval-request id carries the prefix `mcpr_`; strip it
        // so the synthesized `function_call_output.call_id` matches
        // the original `call_id` the model emitted. The driver
        // re-derives `mcpr_<call_id>` when rendering the approval
        // request item, so this round-trips losslessly. Reuses
        // `derived_call_id` from the resolved-skip filter above.
        let call_id = derived_call_id.to_string();

        if approve {
            let Some(s) = session else {
                return Err(AgentLoopError::InvalidRequest(format!(
                    "approval_continuation_invalid_tool: tool '{tool_name}' from server \
                     '{server_label}' cannot be approved because the current request has no \
                     MCP session. Re-issue the request with a `tools[]` array that enables the \
                     matching MCP server/tool."
                )));
            };

            // Validate the current session still exposes the tool the
            // approval prompt referenced. Servers can be removed or
            // renamed between the original turn and the continuation;
            // executing a stale binding would silently misroute the
            // call. Per the design doc this is `invalid_request_error`.
            if !s.has_exposed_tool(tool_name)
                || s.resolve_tool_server_label(tool_name) != server_label
            {
                return Err(AgentLoopError::InvalidRequest(format!(
                    "approval_continuation_invalid_tool: tool '{tool_name}' from \
                     server '{server_label}' is not exposed by the current MCP session. \
                     The server may have been disabled or renamed; re-issue the request \
                     with a `tools[]` array that re-enables it."
                )));
            }

            // Build a fully-enriched `PlannedToolExecution` directly:
            // session lookups happen here in the driver. The adapter
            // never participates — its only job is to surface the
            // model's raw fc emission. The continuation path bypasses
            // `decide_after_call_llm` entirely (we set
            // `next_action = ExecuteTools(...)` below) so policy is
            // never re-checked for this call_id.
            let presentation =
                ToolPresentation::from_mcp_format(&s.tool_response_format(tool_name));
            planned.push(PlannedToolExecution {
                call: LoopToolCall {
                    call_id: call_id.clone(),
                    item_id: call_id.clone(),
                    name: tool_name.to_string(),
                    arguments: arguments.to_string(),
                    approval_request_id: Some(approval_request_id.to_string()),
                },
                server_label: server_label.to_string(),
                presentation,
            });
        } else {
            let summary = match reason.filter(|s| !s.is_empty()) {
                Some(reason_text) => format!("Tool call denied by user: {reason_text}"),
                None => "Tool call denied by user.".to_string(),
            };
            state
                .transcript
                .push(ResponseInputOutputItem::FunctionToolCall {
                    id: call_id.clone(),
                    call_id: call_id.clone(),
                    name: tool_name.to_string(),
                    arguments: arguments.to_string(),
                    output: None,
                    status: Some("completed".to_string()),
                });
            state
                .transcript
                .push(ResponseInputOutputItem::FunctionCallOutput {
                    id: None,
                    call_id,
                    output: summary,
                    status: Some("completed".to_string()),
                });
        }
    }

    if !planned.is_empty() {
        state.set_next_action(NextAction::ExecuteTools(planned));
    }

    Ok(())
}

/// Outer iteration ceiling. `DEFAULT_MAX_ITERATIONS` already gates how
/// many gateway tool calls can run; the loop iteration counter exists
/// only to defend against pathological spinning where the adapter
/// returns no tool calls and no completion (which should never
/// happen but is cheap to guard).
fn effective_iteration_cap() -> usize {
    DEFAULT_MAX_ITERATIONS + 8
}

/// Pure post-CallLlm transition. Inputs are loop state set by the
/// adapter's `call_upstream`; output is the next action.
///
/// Adapters push **raw** [`LoopToolCall`] identities (call_id,
/// item_id, name, arguments) into `state.pending_tool_calls` without
/// consulting the session for any policy. This function performs the
/// only session-touching classification:
///
/// 1. **Gateway vs caller fc**: `session.has_exposed_tool(name)`
///    splits the batch. Caller fc go to `state.pending_user_tool_calls`
///    (driver-populated record for `render_final`); the adapter has
///    already emitted their wire lifecycle through the sink during
///    turn streaming.
/// 2. **Gateway gating**: `session.tool_requires_approval(name)`
///    splits gateway calls into immediate (→ `ExecuteTools`) and
///    gated (→ `InterruptForApproval`). Mixed turns run the
///    immediate side first; gated calls carry over to
///    `decide_after_execute_tools`.
fn decide_after_call_llm(
    state: &mut AgentLoopState,
    session: Option<&McpToolSession<'_>>,
) -> NextAction {
    if state.tool_budget_exhausted {
        state.pending_tool_calls.clear();
        state.pending_gateway_tool_calls.clear();
        return NextAction::Finish;
    }

    let raw = std::mem::take(&mut state.pending_tool_calls);
    if raw.is_empty() && state.pending_gateway_tool_calls.is_empty() {
        return NextAction::Finish;
    }

    // Step 1: gateway vs caller-fc classification.
    let mut gateway_calls = std::mem::take(&mut state.pending_gateway_tool_calls);
    let mut user_calls: Vec<LoopToolCall> = Vec::new();
    for call in raw {
        let is_gateway = session
            .map(|s| s.has_exposed_tool(&call.name))
            .unwrap_or(false);
        if is_gateway {
            gateway_calls.push(call);
        } else {
            user_calls.push(call);
        }
    }

    // Caller fc never reaches the executor. Record them for
    // `render_final` (NS path needs to splice `function_call` items
    // into output) and flip the after-execute carrier so the loop
    // hands back to the caller after this iteration's gateway batch.
    if !user_calls.is_empty() {
        state.stop_after_tool_execution = true;
        state
            .pending_user_tool_calls
            .extend(user_calls.into_iter().map(|c| LoopUserToolCall {
                call_id: c.call_id,
                item_id: c.item_id,
                name: c.name,
                arguments: c.arguments,
            }));
    }

    if gateway_calls.is_empty() {
        // Only caller fc this turn — nothing to dispatch, finish so
        // render_final can splice the user fc items.
        return NextAction::Finish;
    }

    // Step 2: gateway approval split.
    let (gated, immediate): (Vec<_>, Vec<_>) = gateway_calls.into_iter().partition(|call| {
        session
            .map(|s| s.tool_requires_approval(&call.name))
            .unwrap_or(false)
    });

    if !immediate.is_empty() {
        // Restage the gated calls onto state so the next iteration's
        // `decide_after_execute_tools` surfaces them as approval
        // requests — running them now would skip the gate.
        state.pending_gateway_tool_calls = gated;
        let plan = immediate
            .into_iter()
            .map(|call| enrich_planned(call, session))
            .collect();
        return NextAction::ExecuteTools(plan);
    }

    // Single approval boundary per turn (OpenAI Responses contract):
    // surface only the first gated call. Any remaining gated calls
    // re-issue on the continuation turn — the model re-emits them
    // after seeing the first call's result.
    let mut gated_iter = gated.into_iter();
    match gated_iter.next() {
        Some(first) => {
            let _drop_extra_gated = gated_iter; // explicit: rest discarded by design.
            NextAction::InterruptForApproval(enrich_pending(first, session))
        }
        // Unreachable: caller already exited via `gateway_calls.is_empty()`
        // and the immediate-vs-gated partition above; if we reach here the
        // partition flipped without the caller noticing. Finish gracefully
        // rather than panic.
        None => NextAction::Finish,
    }
}

/// Enrich a raw `LoopToolCall` with session-derived rendering
/// metadata for the immediate-execute path. Pulled out so the
/// approval-continuation path reuses the exact same lookup rules.
fn enrich_planned(
    call: LoopToolCall,
    session: Option<&McpToolSession<'_>>,
) -> PlannedToolExecution {
    let server_label = session
        .map(|s| s.resolve_tool_server_label(&call.name))
        .unwrap_or_default();
    let presentation = session
        .map(|s| {
            super::presentation::ToolPresentation::from_mcp_format(
                &s.tool_response_format(&call.name),
            )
        })
        .unwrap_or_default();
    PlannedToolExecution {
        call,
        server_label,
        presentation,
    }
}

/// Enrich a raw `LoopToolCall` with `server_label` for the
/// approval-interrupt path. Approval-request items are family-
/// agnostic on the wire (always `mcp_approval_request`), so unlike
/// `enrich_planned` this helper does not need `presentation`.
fn enrich_pending(
    call: LoopToolCall,
    session: Option<&McpToolSession<'_>>,
) -> PendingToolExecution {
    let server_label = session
        .map(|s| s.resolve_tool_server_label(&call.name))
        .unwrap_or_default();
    PendingToolExecution { call, server_label }
}

/// Pure post-ExecuteTools transition. Three branches:
///
/// 1. The just-run batch left gated calls staged on
///    `pending_gateway_tool_calls` (e.g. a turn produced one immediate
///    + one approval-required call). Surface those as
///    `mcp_approval_request` items now — the immediate side already
///    executed, the gated side never will on this request. Enrichment
///    via session here mirrors `decide_after_call_llm`.
/// 2. The most recent turn carried a user-managed function call
///    alongside the gateway calls. Hand back to the caller; running
///    another model turn would discard the function call.
/// 3. Otherwise, loop back to `CallLlm` so the model can react to the
///    tool outputs we just appended to transcript.
fn decide_after_execute_tools(
    state: &mut AgentLoopState,
    session: Option<&McpToolSession<'_>>,
) -> NextAction {
    if state.tool_budget_exhausted {
        state.pending_gateway_tool_calls.clear();
        return NextAction::CallLlm;
    }

    if !state.pending_gateway_tool_calls.is_empty() {
        // Single approval boundary: surface only the first gated call.
        // Remaining gated calls re-issue on the continuation turn.
        let mut iter = std::mem::take(&mut state.pending_gateway_tool_calls).into_iter();
        if let Some(first) = iter.next() {
            let _drop_extra_gated = iter;
            return NextAction::InterruptForApproval(enrich_pending(first, session));
        }
    }
    if state.stop_after_tool_execution {
        return NextAction::Finish;
    }
    NextAction::CallLlm
}

async fn run_execute_tools<S: StreamSink>(
    state: &mut AgentLoopState,
    session: &McpToolSession<'_>,
    plan: Vec<PlannedToolExecution>,
    ctx: &AgentLoopContext<'_>,
    sink: &mut S,
) -> Result<(), AgentLoopError> {
    let model_id = ctx.original_request.model.as_str();
    Metrics::record_mcp_tool_iteration(model_id);

    // Materialize the budget once — `effective_tool_call_limit` is
    // already captured inside `remaining_tool_call_budget`.
    let mut remaining = remaining_tool_call_budget(state, ctx.max_tool_calls);

    for plan_item in plan {
        if remaining == 0 {
            // OpenAI Responses treats `max_tool_calls` as an execution cap,
            // not an incomplete terminal state: stop dispatching further tools
            // and ask the model for a final answer using the results already
            // appended to the transcript. Buffered streaming lifecycles for
            // skipped calls never flush, so clients don't see phantom items.
            state.pending_gateway_tool_calls.clear();
            state.tool_budget_exhausted = true;
            return Ok(());
        }

        // Approval continuations skip the model turn, so no adapter
        // stream exists to emit the opening tool lifecycle. Replay it
        // here, after `ResponseStarted` / `mcp_list_tools` have already
        // been emitted and only when the call is actually going to run.
        if let Some(approval_request_id) = plan_item.call.approval_request_id.as_deref() {
            sink.emit(LoopEvent::ApprovedToolReplay {
                call_id: &plan_item.call.call_id,
                item_id: &plan_item.call.item_id,
                name: &plan_item.call.name,
                full_args: &plan_item.call.arguments,
                family: plan_item.presentation.family,
                server_label: &plan_item.server_label,
                approval_request_id: Some(approval_request_id),
            });
        }

        sink.emit(LoopEvent::ToolCallExecutionStarted {
            call_id: &plan_item.call.call_id,
            full_args: &plan_item.call.arguments,
        });

        // The wire-side announce events (`output_item.added` +
        // `mcp_call.in_progress` + `mcp_call_arguments.delta/done`)
        // were emitted during turn streaming via
        // `ToolCallEmissionStarted` / `Fragment` / `Done` from the
        // adapter. Here we just need the closing half: tool actually
        // runs, then the sink emits `mcp_call.completed/.failed` +
        // `output_item.done(with output)` via `ToolCompleted`.
        let executed = execute_planned_tool(session, plan_item, ctx).await?;
        sink.emit(LoopEvent::ToolCompleted {
            executed: &executed,
        });
        state.total_gateway_tool_calls += 1;
        state.record_executed_call(*executed);
        remaining = remaining.saturating_sub(1);
        if remaining == 0 {
            state.tool_budget_exhausted = true;
        }
    }

    Ok(())
}

/// Build user-visible `mcp_approval_request` items, one per pending
/// gated tool call. The id format `mcpr_<call_id>` round-trips
/// losslessly with the `prime_pending_from_approval` strip step so
/// continuation requests can correlate the response back.
fn build_mcp_approval_request_items(pending: &[PendingToolExecution]) -> Vec<ResponseOutputItem> {
    pending
        .iter()
        .map(|p| ResponseOutputItem::McpApprovalRequest {
            id: format!("mcpr_{}", p.call.call_id),
            server_label: p.server_label.clone(),
            name: p.call.name.clone(),
            arguments: p.call.arguments.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    //! State-machine tests for the approval transitions. These exercise
    //! the driver's pure decision functions
    //! (`prime_pending_from_approval`, `decide_after_call_llm`,
    //! `decide_after_execute_tools`) without spinning up an
    //! `McpToolSession` — every input is constructed by hand so the
    //! assertions cover the control-layer contract directly.
    //!
    //! A mock session lives only in the integration tests
    //! (`crates/mcp/src/core/session.rs::tests`) where the policy
    //! query (`tool_requires_approval`) is exercised against a real
    //! orchestrator. The driver-level tests here just take
    //! `requires_approval: bool` at face value.
    use std::{borrow::Cow, sync::Arc};

    use openai_protocol::responses::ResponseInputOutputItem;
    use smg_mcp::{McpOrchestrator, McpServerBinding, McpToolSession, Tool, ToolEntry};

    use super::*;
    use crate::routers::common::agent_loop::state::{AgentLoopState, LoopToolCall, NextAction};

    fn loop_call(call_id: &str, name: &str) -> LoopToolCall {
        LoopToolCall {
            call_id: call_id.to_string(),
            item_id: call_id.to_string(),
            name: name.to_string(),
            arguments: "{}".to_string(),
            approval_request_id: None,
        }
    }

    fn fresh_state() -> AgentLoopState {
        use std::collections::HashSet;
        AgentLoopState::new(
            openai_protocol::responses::ResponseInput::Items(Vec::new()),
            HashSet::new(),
        )
    }

    fn create_test_tool(name: &str) -> Tool {
        Tool {
            name: Cow::Owned(name.to_string()),
            title: None,
            description: Some(Cow::Owned(format!("Test tool: {name}"))),
            input_schema: Arc::new(serde_json::Map::new()),
            output_schema: None,
            annotations: None,
            icons: None,
        }
    }

    fn session_with_tool<'a>(
        orchestrator: &'a McpOrchestrator,
        server_label: &str,
        server_key: &str,
        tool_name: &str,
    ) -> McpToolSession<'a> {
        let tool = create_test_tool(tool_name);
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(server_key, tool));
        McpToolSession::new(
            orchestrator,
            vec![McpServerBinding {
                label: server_label.to_string(),
                server_key: server_key.to_string(),
                allowed_tools: None,
            }],
            "test-request",
        )
    }

    #[test]
    fn decide_after_call_llm_routes_immediate_when_no_session_gates_anything() {
        // Without a session, `tool_requires_approval` returns false for
        // every name — so every pending call goes straight to
        // `ExecuteTools`. This isolates the partition logic from the
        // policy lookup.
        let mut state = fresh_state();
        state
            .pending_gateway_tool_calls
            .push(loop_call("call_a", "ask_question"));
        state
            .pending_gateway_tool_calls
            .push(loop_call("call_b", "search_docs"));

        match decide_after_call_llm(&mut state, None) {
            NextAction::ExecuteTools(plans) => {
                assert_eq!(plans.len(), 2);
                assert!(state.pending_gateway_tool_calls.is_empty());
            }
            other => panic!("expected ExecuteTools, got {other:?}"),
        }
    }

    #[test]
    fn decide_after_call_llm_finishes_when_no_pending() {
        let mut state = fresh_state();
        match decide_after_call_llm(&mut state, None) {
            NextAction::Finish => {}
            other => panic!("expected Finish, got {other:?}"),
        }
    }

    #[test]
    fn decide_after_execute_tools_budget_exhaustion_suppresses_gated_carryover() {
        let mut state = fresh_state();
        state.tool_budget_exhausted = true;
        state
            .pending_gateway_tool_calls
            .push(loop_call("call_gated", "ask_question"));

        match decide_after_execute_tools(&mut state, None) {
            NextAction::CallLlm => {}
            other => panic!("expected CallLlm, got {other:?}"),
        }
        assert!(
            state.pending_gateway_tool_calls.is_empty(),
            "budget-exhausted turns must not surface approval requests for skipped calls"
        );
    }

    #[test]
    fn prime_pending_from_approval_approve_true_skips_call_llm() {
        let mut state = fresh_state();
        let control = vec![
            ResponseInputOutputItem::McpApprovalRequest {
                id: "mcpr_call_a".to_string(),
                server_label: "deepwiki".to_string(),
                name: "ask_question".to_string(),
                arguments: r#"{"q":"hi"}"#.to_string(),
            },
            ResponseInputOutputItem::McpApprovalResponse {
                id: None,
                approval_request_id: "mcpr_call_a".to_string(),
                approve: true,
                reason: None,
            },
        ];

        let orchestrator = McpOrchestrator::new_test();
        let session = session_with_tool(&orchestrator, "deepwiki", "server1", "ask_question");
        prime_pending_from_approval(&control, Some(&session), &mut state)
            .expect("approve=true must succeed with current session");

        // Driver primed the action: ExecuteTools(1), with the original
        // call's name + arguments preserved.
        match state.take_next_action() {
            NextAction::ExecuteTools(plans) => {
                assert_eq!(plans.len(), 1);
                let call = &plans[0].call;
                assert_eq!(call.call_id, "call_a");
                assert_eq!(call.name, "ask_question");
                assert_eq!(call.arguments, r#"{"q":"hi"}"#);
            }
            other => panic!("expected primed ExecuteTools, got {other:?}"),
        }

        // Crucially: NO transcript items synthesized. The model never
        // sees a fake "approved by user" message — only the real
        // tool result, which lands after ExecuteTools runs.
        assert!(
            state.transcript.is_empty(),
            "transcript should be untouched on approve=true (saw {:?})",
            state.transcript
        );
    }

    #[test]
    fn prime_pending_from_approval_approve_false_synthesizes_denial() {
        let mut state = fresh_state();
        let control = vec![
            ResponseInputOutputItem::McpApprovalRequest {
                id: "mcpr_call_b".to_string(),
                server_label: "deepwiki".to_string(),
                name: "search_docs".to_string(),
                arguments: "{}".to_string(),
            },
            ResponseInputOutputItem::McpApprovalResponse {
                id: None,
                approval_request_id: "mcpr_call_b".to_string(),
                approve: false,
                reason: Some("user said no".to_string()),
            },
        ];

        prime_pending_from_approval(&control, None, &mut state)
            .expect("approve=false must succeed");

        // Default action: CallLlm so the model can react to the denial.
        // Nothing is queued for execution.
        assert!(
            state.pending_gateway_tool_calls.is_empty(),
            "deny must not schedule execution"
        );
        assert!(matches!(state.take_next_action(), NextAction::CallLlm));

        // Transcript carries the synthesized fc + fco("denied: user said no").
        assert_eq!(state.transcript.len(), 2);
        match &state.transcript[1] {
            ResponseInputOutputItem::FunctionCallOutput { output, .. } => {
                assert_eq!(output, "Tool call denied by user: user said no");
            }
            other => panic!("expected FunctionCallOutput, got {other:?}"),
        }
    }

    #[test]
    fn prime_pending_from_approval_orphan_response_errors() {
        let mut state = fresh_state();
        let control = vec![ResponseInputOutputItem::McpApprovalResponse {
            id: None,
            approval_request_id: "mcpr_unknown".to_string(),
            approve: true,
            reason: None,
        }];
        let err = prime_pending_from_approval(&control, None, &mut state)
            .expect_err("orphan response must reject");
        match err {
            AgentLoopError::InvalidRequest(msg) => {
                assert!(msg.contains("approval_continuation_invalid"));
                assert!(msg.contains("mcpr_unknown"));
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn prime_pending_from_approval_approve_true_requires_session() {
        let mut state = fresh_state();
        let control = vec![
            ResponseInputOutputItem::McpApprovalRequest {
                id: "mcpr_call_a".to_string(),
                server_label: "deepwiki".to_string(),
                name: "ask_question".to_string(),
                arguments: r#"{"q":"hi"}"#.to_string(),
            },
            ResponseInputOutputItem::McpApprovalResponse {
                id: None,
                approval_request_id: "mcpr_call_a".to_string(),
                approve: true,
                reason: None,
            },
        ];

        let err = prime_pending_from_approval(&control, None, &mut state)
            .expect_err("approve=true without session must reject as invalid request");
        match err {
            AgentLoopError::InvalidRequest(msg) => {
                assert!(msg.contains("approval_continuation_invalid_tool"));
                assert!(msg.contains("no MCP session"));
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn prime_pending_from_approval_mixed_approve_deny() {
        let mut state = fresh_state();
        let control = vec![
            ResponseInputOutputItem::McpApprovalRequest {
                id: "mcpr_a".to_string(),
                server_label: "s1".to_string(),
                name: "tool_a".to_string(),
                arguments: "{}".to_string(),
            },
            ResponseInputOutputItem::McpApprovalRequest {
                id: "mcpr_b".to_string(),
                server_label: "s2".to_string(),
                name: "tool_b".to_string(),
                arguments: "{}".to_string(),
            },
            ResponseInputOutputItem::McpApprovalResponse {
                id: None,
                approval_request_id: "mcpr_a".to_string(),
                approve: true,
                reason: None,
            },
            ResponseInputOutputItem::McpApprovalResponse {
                id: None,
                approval_request_id: "mcpr_b".to_string(),
                approve: false,
                reason: None,
            },
        ];

        let orchestrator = McpOrchestrator::new_test();
        let session = session_with_tool(&orchestrator, "s1", "server1", "tool_a");
        prime_pending_from_approval(&control, Some(&session), &mut state)
            .expect("mix must succeed");

        // Approved → primed for execution. Denied → transcript pair only.
        match state.take_next_action() {
            NextAction::ExecuteTools(plans) => {
                assert_eq!(plans.len(), 1);
                assert_eq!(plans[0].call.call_id, "a");
            }
            other => panic!("expected ExecuteTools, got {other:?}"),
        }
        assert_eq!(state.transcript.len(), 2);
    }

    /// `approve = true` against a session that no longer exposes the
    /// referenced tool must surface as `invalid_request_error`. The
    /// stale binding is recoverable on the client side (re-send with a
    /// matching `tools[]`); silently dispatching against a disabled
    /// server is not. Review P1.3.
    #[test]
    fn prime_pending_from_approval_rejects_unexposed_tool() {
        let orchestrator = McpOrchestrator::new_test();
        let session = McpToolSession::new(&orchestrator, vec![], "test-request");
        let mut state = fresh_state();
        let control = vec![
            ResponseInputOutputItem::McpApprovalRequest {
                id: "mcpr_call_z".to_string(),
                server_label: "deepwiki".to_string(),
                name: "ask_question".to_string(),
                arguments: r#"{"q":"hi"}"#.to_string(),
            },
            ResponseInputOutputItem::McpApprovalResponse {
                id: None,
                approval_request_id: "mcpr_call_z".to_string(),
                approve: true,
                reason: None,
            },
        ];

        let err = prime_pending_from_approval(&control, Some(&session), &mut state)
            .expect_err("disabled tool must reject");
        match err {
            AgentLoopError::InvalidRequest(msg) => {
                assert!(
                    msg.contains("approval_continuation_invalid_tool"),
                    "expected invalid_tool error, got {msg}"
                );
                assert!(msg.contains("ask_question"));
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }
}
