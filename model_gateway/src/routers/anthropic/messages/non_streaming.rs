//! Non-streaming MCP tool loop for Anthropic Messages API

use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use tracing::{info, warn};

use super::tools::{execute_mcp_tool_calls, rebuild_response_with_mcp_blocks, McpToolCall};
use crate::{
    mcp::McpToolSession,
    observability::metrics::Metrics,
    protocols::messages::{CreateMessageRequest, InputContent, InputMessage, Role, StopReason},
    routers::{anthropic::context::MessagesContext, error, mcp_utils::DEFAULT_MAX_ITERATIONS},
};

/// Execute the MCP tool loop for non-streaming Messages API requests.
pub(crate) async fn execute_tool_loop(
    messages_ctx: &MessagesContext,
    mut request: CreateMessageRequest,
    headers: Option<HeaderMap>,
    model_id: &str,
    mcp_servers: Vec<(String, String)>,
) -> Response {
    let request_id = format!("msg_{}", uuid::Uuid::new_v4());
    let session = McpToolSession::new(&messages_ctx.mcp_orchestrator, mcp_servers, &request_id);

    let mut all_mcp_calls: Vec<McpToolCall> = Vec::new();

    let mut message = match messages_ctx
        .pipeline
        .execute_for_messages(request.clone(), headers.clone(), model_id)
        .await
    {
        Ok(m) => m,
        Err(response) => return response,
    };

    for iteration in 0..DEFAULT_MAX_ITERATIONS {
        Metrics::record_mcp_tool_iteration(model_id);

        let tool_calls = super::tools::extract_tool_calls(&message.content);
        if tool_calls.is_empty() {
            let final_message = rebuild_response_with_mcp_blocks(message, &all_mcp_calls);
            return (StatusCode::OK, Json(final_message)).into_response();
        }

        if message.stop_reason != Some(StopReason::ToolUse) {
            warn!(
                iteration = iteration,
                tool_count = tool_calls.len(),
                stop_reason = ?message.stop_reason,
                "Tool use blocks present but stop_reason is not tool_use; tool calls dropped"
            );
            return error::bad_gateway(
                "inconsistent_tool_state",
                format!(
                    "Model returned {} tool_use block(s) but stop_reason is {:?}; \
                     tool calls will not be executed",
                    tool_calls.len(),
                    message.stop_reason
                ),
            );
        }

        info!(
            iteration = iteration,
            tool_count = tool_calls.len(),
            "MCP tool loop: executing tool calls"
        );

        let (new_calls, assistant_blocks, tool_result_blocks) =
            execute_mcp_tool_calls(&message.content, &tool_calls, &session, model_id).await;

        all_mcp_calls.extend(new_calls);

        request.messages.push(InputMessage {
            role: Role::Assistant,
            content: InputContent::Blocks(assistant_blocks),
        });
        request.messages.push(InputMessage {
            role: Role::User,
            content: InputContent::Blocks(tool_result_blocks),
        });

        message = match messages_ctx
            .pipeline
            .execute_for_messages(request.clone(), headers.clone(), model_id)
            .await
        {
            Ok(m) => m,
            Err(response) => return response,
        };
    }

    // The last pipeline call produced a message — check if it completed naturally
    let tool_calls = super::tools::extract_tool_calls(&message.content);
    if tool_calls.is_empty() {
        let final_message = rebuild_response_with_mcp_blocks(message, &all_mcp_calls);
        return (StatusCode::OK, Json(final_message)).into_response();
    }

    error::bad_gateway(
        "mcp_max_iterations",
        format!(
            "MCP tool loop exceeded maximum iterations ({})",
            DEFAULT_MAX_ITERATIONS
        ),
    )
}
