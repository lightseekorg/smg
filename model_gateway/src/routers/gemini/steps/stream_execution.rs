//! StreamRequestExecution step (no MCP tools — simple passthrough).
//!
//! Terminal step: returns `StepResult::Response` when streaming is complete.

use axum::response::Response;

use crate::routers::gemini::{context::RequestContext, state::StepResult};

/// Open a streaming connection to the upstream worker and forward SSE events
/// directly to the client without tool interception.
///
/// This is the simple passthrough path for streaming requests that do not
/// contain MCP tools. Events are forwarded as-is with minimal transformation
/// (metadata patching only).
///
/// ## Reads
/// - `ctx.processing.payload` — the JSON body to stream.
/// - `ctx.processing.upstream_url` — the worker endpoint.
/// - `ctx.streaming.sse_tx` — the SSE channel sender.
/// - `ctx.input.headers` — forwarded headers.
///
/// ## Returns
/// `Ok(StepResult::Response(Response::default()))` — signals completion.
pub(crate) async fn stream_request_execution(
    _ctx: &mut RequestContext,
) -> Result<StepResult, Response> {
    // TODO: implement simple streaming passthrough
    //
    // 1. POST ctx.processing.payload to ctx.processing.upstream_url
    //    with Accept: text/event-stream. Forward headers from ctx.input.headers.
    // 2. On connection error: send SSE error event via ctx.streaming.sse_tx, return Err.
    // 3. On non-success HTTP status: send SSE error event, return Err.
    // 4. Record circuit breaker success.
    // 5. Read upstream SSE byte stream chunk by chunk.
    // 6. For each SSE event:
    //    a. Apply minimal transformations (patch store, previous_interaction_id).
    //    b. Accumulate response for persistence if store is true.
    //    c. Forward event to ctx.streaming.sse_tx.
    // 7. After stream completes:
    //    a. Persist interaction if store is true.
    //    b. Send "data: [DONE]\n\n".
    // 8. Return Ok(StepResult::Response(Response::default())).

    Ok(StepResult::Response(Response::default()))
}
