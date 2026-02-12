//! StreamRequestExecutionWithTool step (streaming with MCP tool interception).
//!
//! Transition: StreamRequestWithTool → ResponseDone
//!
//! This step combines streaming event forwarding and tool execution into a
//! single self-contained loop. It replaces the previous two-step approach
//! (StreamRequestExecution + StreamToolExecution) with an internal loop that
//! handles the full tool-call cycle without returning to the driver.

use axum::response::Response;

use crate::routers::gemini::{
    context::RequestContext,
    state::{RequestState, StepResult},
};

/// Stream events from the upstream worker with MCP tool call interception.
///
/// Runs an internal loop:
/// 1. Make a streaming request to upstream.
/// 2. Forward/transform/buffer SSE events to the client.
/// 3. If tool calls are detected: execute them, send results as SSE events,
///    build a resume payload, and loop back to step 1.
/// 4. If no tool calls: send the final completion event and break.
///
/// ## Reads
/// - `ctx.processing.payload` — the JSON body (updated on each tool-loop iteration).
/// - `ctx.processing.upstream_url` — the worker endpoint.
/// - `ctx.streaming.sse_tx` — the SSE channel sender.
/// - `ctx.streaming.is_first_iteration` — controls dedup of lifecycle events.
/// - `ctx.streaming.sequence_number`, `ctx.streaming.next_output_index` — event numbering.
/// - `ctx.input.headers` — forwarded headers.
/// - `ctx.components.mcp_orchestrator` — for tool execution.
///
/// ## Writes
/// - `ctx.streaming.sequence_number` — incremented per event sent.
/// - `ctx.streaming.next_output_index` — updated across iterations.
/// - `ctx.streaming.is_first_iteration` — set to `false` after first iteration.
/// - `ctx.processing.payload` — replaced with resume payload on each tool-loop iteration.
/// - `ctx.state` → `ResponseDone`.
pub(crate) async fn stream_request_execution_with_tool(
    ctx: &mut RequestContext,
) -> Result<StepResult, Response> {
    // TODO: implement streaming with tool interception
    //
    // loop {
    //     // ── Stream events from upstream ────────────────────────────
    //     //
    //     // 1. POST ctx.processing.payload to ctx.processing.upstream_url
    //     //    with Accept: text/event-stream. Forward headers.
    //     // 2. On error: send SSE error event via ctx.streaming.sse_tx, return Err.
    //     // 3. Create StreamingToolHandler::with_starting_index(ctx.streaming.next_output_index).
    //     //    If not first iteration, set handler.original_response_id from preserved ID.
    //     // 4. Read upstream SSE byte stream via ChunkProcessor. For each event:
    //     //    a. Feed to handler.process_event(event_name, data):
    //     //       - Forward: Apply event transformations:
    //     //           * On subsequent iterations (is_first_iteration == false), skip
    //     //             interaction.start / interaction.in_progress to avoid duplicates.
    //     //           * Rewrite function_call events → mcp_server_tool_call with correct IDs.
    //     //           * Remap output_index for sequential numbering.
    //     //           * Update ctx.streaming.sequence_number.
    //     //         Send the transformed event to ctx.streaming.sse_tx.
    //     //         After first interaction.in_progress, send mcp_list_tools events (once).
    //     //       - Buffer: Accumulate tool call arguments silently.
    //     //       - ExecuteTools: Forward the final event, then break inner stream loop.
    //     // 5. Update ctx.streaming.next_output_index from handler.
    //     // 6. Preserve response ID from first iteration for subsequent ones.
    //     //
    //     // ── Check for tool calls ──────────────────────────────────
    //     //
    //     // 7. If no tool calls detected:
    //     //    a. Send synthetic interaction.completed event with the full response.
    //     //    b. Persist interaction if store is true.
    //     //    c. Send "data: [DONE]\n\n".
    //     //    d. Break out of the loop.
    //     //
    //     // ── Execute tool calls and resume ─────────────────────────
    //     //
    //     // 8. Extract pending tool calls from handler.
    //     // 9. Increment tool_loop_state.iteration and total_calls.
    //     // 10. Check against max_tool_calls / DEFAULT_MAX_ITERATIONS.
    //     //     If exceeded: send SSE error, break.
    //     // 11. For each pending tool call:
    //     //     a. Look up response format from MCP session.
    //     //     b. Send in-progress SSE event via ctx.streaming.sse_tx.
    //     //     c. Execute tool via mcp_session.execute_tool().
    //     //     d. Send tool result SSE events (output_item.added, content.delta,
    //     //        output_item.done).
    //     //     e. Record in tool_loop_state.conversation_history and mcp_call_items.
    //     //     f. Increment ctx.streaming.sequence_number.
    //     // 12. Build resume payload from tool_loop_state.
    //     // 13. Set ctx.processing.payload = Some(resume_payload).
    //     // 14. Set ctx.streaming.is_first_iteration = false.
    //     // 15. Continue loop (next iteration makes another streaming request).
    // }

    ctx.state = RequestState::ResponseDone;
    Ok(StepResult::Continue)
}
