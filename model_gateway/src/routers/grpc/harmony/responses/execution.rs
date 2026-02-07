//! MCP tool execution logic for Harmony Responses

use axum::response::Response;
use serde_json::{from_str, json, Value};
use tracing::{debug, error};

use super::common::McpCallTracking;
use crate::{
    mcp::{McpToolSession, ToolEntry, ToolExecutionInput},
    observability::metrics::{metrics_labels, Metrics},
    protocols::{
        common::{Function, ToolCall},
        responses::{ResponseTool, ResponseToolType},
    },
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
}

/// Execute MCP tools and collect results
///
/// Executes tool calls via the unified MCP orchestrator batch API.
/// Tool execution errors are returned as error results to the model
/// (allows model to handle gracefully).
///
/// Vector of tool results (one per tool call)
pub(super) async fn execute_mcp_tools(
    session: &McpToolSession<'_>,
    tool_calls: &[ToolCall],
    tracking: &mut McpCallTracking,
    model_id: &str,
) -> Result<Vec<ToolResult>, Response> {
    // Convert tool calls to execution inputs
    let inputs: Vec<ToolExecutionInput> = tool_calls
        .iter()
        .map(|tc| {
            let args_str = tc.function.arguments.as_deref().unwrap_or("{}");
            let args: Value = from_str(args_str).unwrap_or_else(|e| {
                error!(
                    function = "execute_mcp_tools",
                    tool_name = %tc.function.name,
                    call_id = %tc.id,
                    error = %e,
                    "Failed to parse tool arguments JSON, using empty object"
                );
                json!({})
            });
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
                is_error: output.is_error,
            }
        })
        .collect();

    Ok(results)
}

/// Convert MCP tools to Responses API tool format
///
/// Converts MCP ToolEntry (from inventory) to ResponseTool format so the model
/// knows about available MCP tools when making tool calls.
pub(crate) fn convert_mcp_tools_to_response_tools(mcp_tools: &[ToolEntry]) -> Vec<ResponseTool> {
    mcp_tools
        .iter()
        .map(|entry| ResponseTool {
            r#type: ResponseToolType::Mcp,
            function: Some(Function {
                name: entry.tool.name.to_string(),
                description: entry.tool.description.as_ref().map(|d| d.to_string()),
                parameters: Value::Object((*entry.tool.input_schema).clone()),
                strict: None,
            }),
            server_url: None, // MCP tools from inventory don't have individual server URLs
            authorization: None,
            headers: None,
            server_label: Some(entry.server_key().to_string()),
            server_description: entry.tool.description.as_ref().map(|d| d.to_string()),
            require_approval: None,
            allowed_tools: None,
        })
        .collect()
}
