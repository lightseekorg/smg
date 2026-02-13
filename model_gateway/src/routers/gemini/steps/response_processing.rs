//! ResponseProcessing step (non-streaming).
//!
//! Terminal step: returns `StepResult::Response` directly.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};

use crate::routers::gemini::{context::RequestContext, state::StepResult};

/// Finalize the non-streaming response and return it to the client.
///
/// This step produces the terminal HTTP `Response` via `StepResult::Response`.
/// The driver returns it directly — no `final_response` field on the context.
///
/// ## Reads
/// - `ctx.upstream_response` — the last upstream response JSON.
/// - `ctx.original_request` — for metadata patching.
///
/// ## Returns
/// `Ok(StepResult::Response(response))` — the final HTTP response.
pub(crate) async fn response_processing(ctx: &mut RequestContext) -> Result<StepResult, Response> {
    // TODO: implement response processing
    //
    // 1. Take ctx.upstream_response (the last response from upstream).
    // 2. If an MCP session exists:
    //    a. Inject MCP metadata into the response:
    //       - mcp_list_tools items (one per MCP server, listing available tools).
    //       - mcp_call items (the executed tool call/result pairs).
    //    b. Restore original tools in the response: convert function tools back
    //       to McpServer / built-in tool format matching the original request.
    // 3. Patch response with request metadata:
    //    - Set store, previous_interaction_id, model from the original request.
    // 4. Persist the interaction to ctx.shared.interaction_storage if store is true.
    // 5. Return Ok(StepResult::Response((StatusCode::OK, Json(response_json)).into_response())).

    // Placeholder: return the upstream response as-is (or an empty 200).
    let response_json = ctx
        .processing
        .upstream_response
        .take()
        .unwrap_or(serde_json::json!({}));
    Ok(StepResult::Response(
        (StatusCode::OK, Json(response_json)).into_response(),
    ))
}
