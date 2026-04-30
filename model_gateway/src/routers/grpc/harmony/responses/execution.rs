//! MCP tool execution logic for Harmony Responses

use axum::response::Response;
use openai_protocol::{
    common::ToolCall,
    responses::{ResponseOutputItem, ResponseTool},
};
use serde_json::{from_str, json, Value};
use smg_mcp::{McpToolSession, ToolExecutionInput};
use tracing::{debug, error};

use super::common::McpCallTracking;
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::common::mcp_utils::prepare_hosted_dispatch_args,
};

/// Tool execution result
///
/// Contains the result of executing a single MCP tool.
/// Used for building conversation history (raw output) and status tracking.
pub(crate) struct ToolResult {
    /// Tool call ID (for matching with request)
    pub call_id: String,

    /// Tool output (JSON value) - used for conversation history
    pub output: Value,

    /// Whether this is an error result
    pub is_error: bool,

    /// Correctly-typed output item for the Responses API, produced by
    /// `ResponseTransformer::transform` (via
    /// `ToolExecutionOutput::to_response_item`). Carries the per-tool
    /// shape — e.g. `McpCall { output }`, `WebSearchCall { .. }`,
    /// `ImageGenerationCall { result }` — so downstream code can
    /// serialize the authoritative item rather than re-deriving fields
    /// from `output`.
    pub output_item: ResponseOutputItem,
}

/// Execute MCP tools and collect results
///
/// Executes tool calls via the unified MCP orchestrator batch API.
/// Tool execution errors are returned as error results to the model
/// (allows model to handle gracefully).
///
/// `request_user` is the request-level `user` identifier (OpenAI Responses
/// API `user` field), forwarded into hosted-tool dispatch args so the MCP
/// server can attribute usage. Plain MCP function tools (Passthrough format)
/// are not affected.
///
/// Vector of tool results (one per tool call)
pub(super) async fn execute_mcp_tools(
    session: &McpToolSession<'_>,
    tool_calls: &[ToolCall],
    tracking: &mut McpCallTracking,
    model_id: &str,
    request_tools: &[ResponseTool],
    request_user: Option<&str>,
) -> Result<Vec<ToolResult>, Response> {
    // Convert tool calls to execution inputs, merging caller-declared
    // hosted-tool configuration from `request_tools` into dispatch args.
    // For non-hosted-tool calls (Passthrough format), no override lookup runs.
    // Non-object model payloads coerce to `{}` so the override merge actually
    // applies rather than silently dropping the caller's declared config.
    let inputs: Vec<ToolExecutionInput> = tool_calls
        .iter()
        .map(|tc| {
            let args_str = tc.function.arguments.as_deref().unwrap_or("{}");
            let mut args: Value = match from_str::<Value>(args_str) {
                Ok(Value::Object(map)) => Value::Object(map),
                Ok(_) => {
                    debug!(
                        function = "execute_mcp_tools",
                        tool_name = %tc.function.name,
                        call_id = %tc.id,
                        "Tool arguments parsed to non-object JSON; coercing to empty object"
                    );
                    json!({})
                }
                Err(e) => {
                    error!(
                        function = "execute_mcp_tools",
                        tool_name = %tc.function.name,
                        call_id = %tc.id,
                        error = %e,
                        "Failed to parse tool arguments JSON, using empty object"
                    );
                    json!({})
                }
            };
            let response_format = session.tool_response_format(&tc.function.name);
            prepare_hosted_dispatch_args(&mut args, &response_format, request_tools, request_user);
            ToolExecutionInput {
                call_id: tc.id.clone(),
                tool_name: tc.function.name.clone(),
                arguments: args,
            }
        })
        .collect();

    debug!(
        tool_count = inputs.len(),
        "Executing MCP tools via unified API"
    );

    // Execute all tools via unified batch API
    let outputs = session.execute_tools(inputs).await;

    // Convert outputs to ToolResults and record metrics/tracking
    let results: Vec<ToolResult> = outputs
        .into_iter()
        .map(|output| {
            // Transform to correctly-typed ResponseOutputItem
            let output_item = output.to_response_item();

            // Record this call in tracking
            tracking.record_call(output_item.clone());

            // Record MCP tool metrics
            Metrics::record_mcp_tool_duration(model_id, &output.tool_name, output.duration);
            Metrics::record_mcp_tool_call(
                model_id,
                &output.tool_name,
                if output.is_error {
                    metrics_labels::RESULT_ERROR
                } else {
                    metrics_labels::RESULT_SUCCESS
                },
            );

            ToolResult {
                call_id: output.call_id,
                output: output.output,
                output_item,
                is_error: output.is_error,
            }
        })
        .collect();

    Ok(results)
}

pub(crate) fn convert_mcp_tools_to_response_tools(
    session: &McpToolSession<'_>,
) -> Vec<ResponseTool> {
    session.build_response_tools()
}
