//! MCP tool execution logic for Harmony Responses

use std::sync::Arc;

use axum::response::Response;
use serde_json::{from_str, json, Value};
use tracing::{debug, error};

use super::common::McpCallTracking;
use crate::{
    mcp::{McpOrchestrator, ToolCallContext, ToolEntry, ToolExecutionInput},
    observability::metrics::{metrics_labels, Metrics},
    protocols::{
        common::{Function, ToolCall},
        responses::{ResponseOutputItem, ResponseTool, ResponseToolType},
    },
    routers::grpc::common::responses::utils::partition_outcomes,
};

pub(crate) struct ToolResult {
    pub call_id: String,
    pub output: Value,
    pub is_error: bool,
}

pub(crate) struct McpExecutionResult {
    pub executed: Vec<ToolResult>,
    pub pending_approvals: Vec<ResponseOutputItem>,
}

pub(super) async fn execute_mcp_tools(
    mcp_orchestrator: &Arc<McpOrchestrator>,
    tool_calls: &[ToolCall],
    tracking: &mut McpCallTracking,
    model_id: &str,
    tool_ctx: &ToolCallContext<'_>,
) -> Result<McpExecutionResult, Response> {
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

    let outcomes = mcp_orchestrator.execute_tools(inputs, tool_ctx).await;

    let (executed_outputs, pending_approvals) = partition_outcomes(outcomes, model_id);

    let mut executed = Vec::with_capacity(executed_outputs.len());
    for output in executed_outputs {
        let output_item = output.to_response_item();
        tracking.record_call(output_item);

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

        executed.push(ToolResult {
            call_id: output.call_id,
            output: output.output,
            is_error: output.is_error,
        });
    }

    Ok(McpExecutionResult {
        executed,
        pending_approvals,
    })
}

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
            server_url: None,
            authorization: None,
            headers: None,
            server_label: Some(entry.server_key().to_string()),
            server_description: entry.tool.description.as_ref().map(|d| d.to_string()),
            require_approval: None,
            allowed_tools: None,
        })
        .collect()
}
