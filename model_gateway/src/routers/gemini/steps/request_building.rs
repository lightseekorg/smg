//! RequestBuilding step.
//!
//! Transition: BuildRequest → NonStreamRequest | StreamRequestWithTool | StreamRequest

use axum::response::Response;

use crate::routers::gemini::{
    context::RequestContext,
    state::{RequestState, StepResult},
};

/// Build the upstream request payload and create an MCP session if needed.
///
/// ## Reads
/// - `ctx.input.original_request` — the client's `InteractionsRequest`.
/// - `ctx.components.mcp_orchestrator` — for MCP tool session creation.
///
/// ## Writes
/// - `ctx.processing.payload` — the JSON payload to POST upstream.
/// - `ctx.state`:
///   - `NonStreamRequest` — non-streaming request.
///   - `StreamRequestWithTool` — streaming request with MCP tools.
///   - `StreamRequest` — streaming request without MCP tools.
///
/// ## MCP handling
/// When the request contains `InteractionsTool::McpServer` tools, this step:
/// 1. Validates MCP server connectivity via `ensure_request_mcp_client()`.
/// 2. Creates an `McpToolSession` for the request lifetime.
/// 3. Converts MCP tool definitions to function tool definitions in the payload
///    (via `prepare_mcp_tools_as_functions`), so the upstream worker sees only
///    function tools.
///
/// These MCP-related fields would be stored on `RequestContext` once the
/// MCP context fields are added in a later phase.
pub(crate) async fn request_building(ctx: &mut RequestContext) -> Result<StepResult, Response> {
    // TODO: implement request building
    //
    // 1. Serialize ctx.input.original_request to serde_json::Value (the payload).
    // 2. Check if tools contains any InteractionsTool::McpServer entries.
    // 3. If MCP tools present:
    //    a. Call ensure_request_mcp_client() to validate servers → Vec<(label, key)>.
    //    b. Create McpToolSession::new(orchestrator, mcp_servers, request_id).
    //    c. Call prepare_mcp_tools_as_functions(&mut payload, &session) to convert
    //       MCP tool defs → function tool defs in the payload.
    //    d. Store session and tool_loop_state on ctx (fields to be added in MCP phase).
    // 4. Set ctx.processing.payload = Some(payload).
    // 5. Determine has_mcp_tools from step 2.
    // 6. Branch on stream / tools:
    //    - stream && has_mcp_tools: ctx.state = StreamRequestWithTool.
    //    - stream && !has_mcp_tools: ctx.state = StreamRequest.
    //    - !stream:                  ctx.state = NonStreamRequest.

    let has_mcp_tools = false; // TODO: detect from tools list

    if ctx.input.original_request.stream {
        if has_mcp_tools {
            ctx.state = RequestState::StreamRequestWithTool;
        } else {
            ctx.state = RequestState::StreamRequest;
        }
    } else {
        ctx.state = RequestState::NonStreamRequest;
    }
    Ok(StepResult::Continue)
}
