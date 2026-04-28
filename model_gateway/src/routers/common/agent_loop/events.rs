//! Semantic streaming events.
//!
//! The driver emits `LoopEvent` values; each surface's `StreamSink`
//! impl maps them to wire-level SSE chunks. Non-streaming runs use
//! `NoopSink` and ignore everything.
//!
//! Per the design doc the sink may *only* translate events. It is not
//! allowed to make loop control decisions (whether to call the model
//! again, whether to interrupt for approval, etc.). Those belong to
//! `decide_next_action` and the driver.

use openai_protocol::responses::ResponseOutputItem;

use super::{
    presentation::OutputFamily,
    state::{ExecutedCall, PendingToolExecution},
};

/// Events the loop driver hands to a `StreamSink`. Borrowed from
/// loop state so a sink can read what it needs without forcing a
/// clone on every dispatch.
#[derive(Debug)]
pub(crate) enum LoopEvent<'a> {
    /// First event of a streaming response. Sinks emit
    /// `response.created` + `response.in_progress` here.
    ResponseStarted,

    /// Visible MCP server inventory item. Sinks emit the four-event
    /// sequence (`output_item.added` → `mcp_list_tools.in_progress` →
    /// `mcp_list_tools.completed` → `output_item.done`) using a stable
    /// `item_id` per the streaming contract.
    McpListToolsItem { item: &'a ResponseOutputItem },

    /// Adapter saw a tool call's first chunk in the model token
    /// stream — `name` is now known. Sinks classify the tool by
    /// looking the name up in their family-snapshot, allocate an
    /// `output_index` / `item_id`, and emit `output_item.added` for
    /// the family plus the family's `*_in_progress` event (when one
    /// exists — caller-declared `function_call` has none, hosted
    /// builtins do). Argument streaming for families that support
    /// it begins with an initial empty `*_arguments.delta`.
    ///
    /// The adapter does **not** classify or know the family; this
    /// event is the single source of "a tool call has started" for
    /// every wire family.
    ToolCallEmissionStarted {
        call_id: &'a str,
        item_id: &'a str,
        name: &'a str,
    },

    /// Adapter saw a subsequent argument fragment for a tool call
    /// already opened by `ToolCallEmissionStarted`. Sinks emit
    /// `*_arguments.delta` for the family if it streams arguments
    /// (mcp_call / function_call); hosted builtins skip these
    /// silently because their progress rides on structured
    /// `*_in_progress` / `*_searching` events instead.
    ToolCallArgumentsFragment { call_id: &'a str, fragment: &'a str },

    /// Adapter saw the model finish emitting this tool call's
    /// arguments. `full_args` is the complete buffer for callers
    /// that need to record it. Sinks emit `*_arguments.done` for
    /// argument-streaming families. For caller-declared
    /// `function_call` (which has no execute phase), sinks **also**
    /// emit `output_item.done(status=completed)` — this closes the
    /// wire lifecycle. For gateway tools the matching `output_item.done`
    /// is fired later by `ToolCompleted` after execution.
    ToolCallEmissionDone {
        call_id: &'a str,
        full_args: &'a str,
    },

    /// Gateway-owned tool execution is about to start. Sinks that buffered the
    /// model-emitted open-half lifecycle flush it here so only calls that will
    /// actually execute become visible on the stream.
    ToolCallExecutionStarted {
        call_id: &'a str,
        full_args: &'a str,
    },

    /// Tool execution completed. Sinks emit the family-specific
    /// completion event plus `output_item.done`. Only fired for
    /// gateway tools (`mcp_call` / hosted builtin) — caller-declared
    /// `function_call` already closed in `ToolCallEmissionDone`.
    ToolCompleted { executed: &'a ExecutedCall },

    /// Compressed open-half lifecycle for an approval-continuation
    /// call. The model is *not* invoked on continuation — the driver
    /// primes execution directly — so there is no live model-stream
    /// to pump `EmissionStarted` / `Fragment` / `Done` through. Sinks
    /// translate this single event into the four open-half wire
    /// events (`output_item.added` → `*.in_progress` →
    /// `*_arguments.delta` → `*_arguments.done`) using the already-
    /// known full arguments. The matching `output_item.done` /
    /// `*.completed` fires later via [`ToolCompleted`].
    ApprovedToolReplay {
        call_id: &'a str,
        item_id: &'a str,
        name: &'a str,
        full_args: &'a str,
        family: OutputFamily,
        server_label: &'a str,
        approval_request_id: Option<&'a str>,
    },

    /// Approval boundary reached. Sinks emit a single
    /// `output_item.added` / `output_item.done` pair for the pending
    /// gated call, then finalize the response immediately after. The
    /// driver surfaces only the first gated call per turn — additional
    /// gated calls reissue on continuation per the OpenAI Responses
    /// contract (one boundary at a time).
    ApprovalRequested { pending: &'a PendingToolExecution },

    /// Final-channel `response.completed` (or incomplete-equivalent)
    /// event. Closes the SSE stream.
    ResponseFinished,

    /// Replaces `ResponseFinished` when the loop terminates with
    /// `incomplete_details`. Sinks still emit `response.completed`,
    /// but with the incomplete reason attached.
    ResponseIncomplete { reason: &'a str },
}

/// Surface-specific event-to-wire translator. Streaming surfaces delegate to
/// the shared `ResponseStreamEventEmitter` for OpenAI-compatible SSE.
pub(crate) trait StreamSink: Send {
    fn emit(&mut self, event: LoopEvent<'_>);
}

/// No-op sink for non-streaming requests. The driver still calls
/// `emit` at every event boundary — keeping streaming and
/// non-streaming on one code path is the entire point — but every
/// dispatch is a no-op here.
pub(crate) struct NoopSink;

impl StreamSink for NoopSink {
    fn emit(&mut self, _: LoopEvent<'_>) {}
}
