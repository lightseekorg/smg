//! Utility functions for /v1/responses endpoint

use std::sync::Arc;

use axum::response::Response;
use serde_json::to_value;
use tracing::{debug, error, warn};

use crate::{
    core::WorkerRegistry,
    data_connector::{ConversationItemStorage, ConversationStorage, ResponseStorage},
    mcp::{ApprovalResponseInput, McpOrchestrator, ToolExecutionOutcome},
    protocols::{
        common::Tool,
        responses::{
            ResponseInput, ResponseInputOutputItem, ResponseTool, ResponseToolType,
            ResponsesRequest, ResponsesResponse,
        },
    },
    routers::{
        error,
        mcp_utils::{ensure_request_mcp_client, McpConnectionResult},
        persistence_utils::persist_conversation_items,
    },
};

/// Validate MCP connections for tools declared in request.
///
/// Returns server keys and per-server approval modes.
pub(crate) async fn ensure_mcp_connection(
    mcp_orchestrator: &Arc<McpOrchestrator>,
    tools: Option<&[ResponseTool]>,
) -> Result<McpConnectionResult, Response> {
    let has_explicit_mcp_tools = tools
        .map(|t| {
            t.iter()
                .any(|tool| matches!(tool.r#type, ResponseToolType::Mcp))
        })
        .unwrap_or(false);

    let has_builtin_tools = tools
        .map(|t| {
            t.iter().any(|tool| {
                matches!(
                    tool.r#type,
                    ResponseToolType::WebSearchPreview | ResponseToolType::CodeInterpreter
                )
            })
        })
        .unwrap_or(false);

    if !has_explicit_mcp_tools && !has_builtin_tools {
        return Ok(McpConnectionResult::empty());
    }

    if let Some(tools) = tools {
        match ensure_request_mcp_client(mcp_orchestrator, tools).await {
            Some(conn_result) => return Ok(conn_result),
            None => {
                if has_explicit_mcp_tools {
                    error!(
                        function = "ensure_mcp_connection",
                        "Failed to connect to MCP servers"
                    );
                    return Err(error::failed_dependency(
                        "connect_mcp_server_failed",
                        "Failed to connect to MCP servers. Check server_url and authorization.",
                    ));
                }
                debug!(
                    function = "ensure_mcp_connection",
                    "No MCP routing configured for builtin tools, passing through to model"
                );
                return Ok(McpConnectionResult::empty());
            }
        }
    }

    Ok(McpConnectionResult::empty())
}

/// Validate that workers are available for the requested model
pub(crate) fn validate_worker_availability(
    worker_registry: &Arc<WorkerRegistry>,
    model: &str,
) -> Option<Response> {
    let available_models = worker_registry.get_models();

    if !available_models.contains(&model.to_string()) {
        return Some(error::service_unavailable(
            "no_available_workers",
            format!(
                "No workers available for model '{}'. Available models: {}",
                model,
                available_models.join(", ")
            ),
        ));
    }

    None
}

/// Extract tools with schemas from ResponseTools.
pub(crate) fn extract_tools_from_response_tools(
    response_tools: Option<&[ResponseTool]>,
    include_mcp: bool,
) -> Vec<Tool> {
    let Some(tools) = response_tools else {
        return Vec::new();
    };

    tools
        .iter()
        .filter_map(|rt| {
            match rt.r#type {
                // Function tools: Schema in request
                ResponseToolType::Function => rt.function.as_ref().map(|f| Tool {
                    tool_type: "function".to_string(),
                    function: f.clone(),
                }),
                // MCP tools: Schema populated by convert_mcp_tools_to_response_tools()
                // Only include if requested (Harmony case)
                ResponseToolType::Mcp if include_mcp => rt.function.as_ref().map(|f| Tool {
                    tool_type: "function".to_string(),
                    function: f.clone(),
                }),
                // Hosted tools: No schema available, skip
                _ => None,
            }
        })
        .collect()
}

/// Persist response to storage if store=true.
/// Partition tool execution outcomes into executed results and pending approval output items.
///
/// Records metrics for pending approvals. Returns (executed, pending_approval_items).
pub(crate) fn partition_outcomes(
    outcomes: Vec<ToolExecutionOutcome>,
    model_id: &str,
) -> (
    Vec<crate::mcp::ToolExecutionOutput>,
    Vec<crate::protocols::responses::ResponseOutputItem>,
) {
    use crate::{
        mcp::PendingApprovalOutput,
        observability::metrics::{metrics_labels, Metrics},
    };

    let mut executed = Vec::new();
    let mut pending = Vec::new();

    for outcome in outcomes {
        match outcome {
            ToolExecutionOutcome::Executed(output) => executed.push(output),
            ToolExecutionOutcome::PendingApproval(PendingApprovalOutput {
                tool_name,
                server_label,
                arguments_str,
                elicitation_id,
                ..
            }) => {
                Metrics::record_mcp_tool_call(model_id, &tool_name, metrics_labels::RESULT_PENDING);
                pending.push(
                    crate::protocols::responses::ResponseOutputItem::McpApprovalRequest {
                        id: elicitation_id,
                        server_label,
                        name: tool_name,
                        arguments: arguments_str,
                    },
                );
            }
        }
    }

    (executed, pending)
}

pub(crate) async fn persist_response_if_needed(
    conversation_storage: Arc<dyn ConversationStorage>,
    conversation_item_storage: Arc<dyn ConversationItemStorage>,
    response_storage: Arc<dyn ResponseStorage>,
    response: &ResponsesResponse,
    original_request: &ResponsesRequest,
) {
    if !original_request.store.unwrap_or(true) {
        return;
    }

    if let Ok(response_json) = to_value(response) {
        if let Err(e) = persist_conversation_items(
            conversation_storage,
            conversation_item_storage,
            response_storage,
            &response_json,
            original_request,
        )
        .await
        {
            warn!("Failed to persist response: {}", e);
        } else {
            debug!("Persisted response: {}", response.id);
        }
    }
}

/// Stream approval results as SSE events.
///
/// Used when client sends mcp_approval_response items in a streaming request.
/// Shared between regular and harmony streaming paths.
pub(crate) async fn stream_approval_results(
    ctx: &super::ResponsesContext,
    response_id: &str,
    modified_request: &ResponsesRequest,
    original_request: &ResponsesRequest,
    output_items: Vec<crate::protocols::responses::ResponseOutputItem>,
) -> Response {
    use std::time::{SystemTime, UNIX_EPOCH};

    use tokio::sync::mpsc;

    use super::streaming::{OutputItemType, ResponseStreamEventEmitter};
    use crate::protocols::responses::ResponseStatus;

    let model = modified_request.model.clone();
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let (tx, rx) = mpsc::unbounded_channel();
    let mut emitter =
        ResponseStreamEventEmitter::new(response_id.to_string(), model.clone(), created_at);
    emitter.set_original_request(modified_request.clone());

    let response = ResponsesResponse::builder(response_id, &model)
        .copy_from_request(modified_request)
        .status(ResponseStatus::Completed)
        .output(output_items.clone())
        .build();

    let ctx_clone = ctx.clone();
    let original_request_clone = original_request.clone();

    tokio::spawn(async move {
        let event = emitter.emit_created();
        if emitter.send_event(&event, &tx).is_err() {
            return;
        }
        let event = emitter.emit_in_progress();
        if emitter.send_event(&event, &tx).is_err() {
            return;
        }

        for result in output_items {
            let (output_index, _item_id) = emitter.allocate_output_index(OutputItemType::McpCall);
            let item_json = to_value(&result).unwrap_or_default();

            let event = emitter.emit_output_item_added(output_index, &item_json);
            if emitter.send_event(&event, &tx).is_err() {
                return;
            }

            let event = emitter.emit_output_item_done(output_index, &item_json);
            if emitter.send_event(&event, &tx).is_err() {
                return;
            }
        }

        let event = emitter.emit_completed(None);
        let _ = emitter.send_event(&event, &tx);

        persist_response_if_needed(
            ctx_clone.conversation_storage.clone(),
            ctx_clone.conversation_item_storage.clone(),
            ctx_clone.response_storage.clone(),
            &response,
            &original_request_clone,
        )
        .await;
    });

    super::build_sse_response(rx)
}

/// Process any mcp_approval_response items from input. Returns output items if approvals
/// were found and executed, None otherwise.
pub(crate) async fn process_pending_approvals(
    orchestrator: &McpOrchestrator,
    input: &ResponseInput,
    server_label: &str,
) -> Option<Vec<crate::protocols::responses::ResponseOutputItem>> {
    let items = match input {
        ResponseInput::Items(items) => items,
        ResponseInput::Text(_) => return None,
    };

    let approvals: Vec<ApprovalResponseInput> = items
        .iter()
        .filter_map(|item| {
            if let ResponseInputOutputItem::McpApprovalResponse {
                approval_request_id,
                approve,
                reason,
                ..
            } = item
            {
                Some(ApprovalResponseInput {
                    approval_request_id: approval_request_id.clone(),
                    approve: *approve,
                    reason: reason.clone(),
                })
            } else {
                None
            }
        })
        .collect();

    if approvals.is_empty() {
        return None;
    }

    let outcomes = orchestrator
        .process_approval_responses(approvals, server_label)
        .await;

    let output_items: Vec<_> = outcomes
        .into_iter()
        .filter_map(|outcome| match outcome {
            ToolExecutionOutcome::Executed(output) => Some(output.to_response_item()),
            ToolExecutionOutcome::PendingApproval(_) => None, // shouldn't happen for approved tools
        })
        .collect();

    if output_items.is_empty() {
        return None;
    }

    Some(output_items)
}
