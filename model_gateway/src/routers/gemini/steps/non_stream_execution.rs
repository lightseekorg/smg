//! NonStreamRequestExecution step.
//!
//! Transition: NonStreamRequest → NonStreamRequest (tool loop)
//!                               | ProcessResponse (no tool calls)

use axum::response::Response;

use crate::routers::gemini::{
    context::RequestContext,
    state::{RequestState, StepResult},
};

/// POST the payload to the upstream worker and handle the response.
///
/// If the response contains function calls that correspond to MCP tools,
/// execute them, append results to the conversation history, rebuild the
/// payload, and **stay in `NonStreamRequest`** (the driver will call
/// this step again). Otherwise, advance to `ProcessResponse`.
///
/// ## Reads
/// - `ctx.processing.payload` — the JSON body to send.
/// - `ctx.processing.upstream_url` — the worker endpoint.
/// - `ctx.processing.worker` — for circuit breaker recording.
/// - `ctx.input.headers` — forwarded headers.
///
/// ## Writes
/// - `ctx.processing.upstream_response` — the parsed response JSON.
/// - `ctx.processing.payload` — replaced with a resume payload on tool-loop re-entry.
/// - `ctx.state` → `NonStreamRequest` or `ProcessResponse`.
pub(crate) async fn non_stream_request_execution(
    ctx: &mut RequestContext,
) -> Result<StepResult, Response> {
    // TODO: implement non-streaming request execution
    //
    // 1. Build reqwest POST to ctx.processing.upstream_url with ctx.processing.payload
    //    as JSON body. Forward relevant headers from ctx.input.headers.
    // 2. Send the request.
    //    - On connection error: record circuit breaker failure, return Err(bad_gateway).
    // 3. Check HTTP status.
    //    - On non-success: record circuit breaker failure, return Err with upstream status + body.
    // 4. Parse response body as serde_json::Value.
    //    - On parse error: record circuit breaker failure, return Err(parse_error).
    // 5. Record circuit breaker success.
    // 6. Store parsed response in ctx.processing.upstream_response.
    // 7. Scan response outputs for items with type "function_call".
    // 8. For each function_call, check if it maps to an MCP tool via the session.
    // 9. If MCP tool calls found:
    //    a. Execute each tool via mcp_session.execute_tool().
    //    b. Append function_call + function_call_output items to
    //       tool_loop_state.conversation_history.
    //    c. Increment tool_loop_state.iteration and total_calls.
    //    d. Check against max_tool_calls / DEFAULT_MAX_ITERATIONS limit.
    //       If exceeded, return Err(max_tool_calls error).
    //    e. Build resume payload: original input + conversation history + function results.
    //    f. Set ctx.processing.payload = Some(resume_payload).
    //    g. Set ctx.state = NonStreamRequest (loop back).
    //    h. Return Ok(StepResult::Continue).
    // 10. If no MCP tool calls:
    //    a. Set ctx.state = ProcessResponse.
    //    b. Return Ok(StepResult::Continue).

    ctx.state = RequestState::ProcessResponse;
    Ok(StepResult::Continue)
}
