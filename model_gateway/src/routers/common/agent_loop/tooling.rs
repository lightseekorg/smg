//! Loop-level helpers for partitioning, budgeting, and executing
//! gateway-owned tool calls.

use openai_protocol::responses::{
    CodeInterpreterCallStatus, FileSearchCallStatus, ImageGenerationCallStatus, ResponseOutputItem,
    WebSearchAction, WebSearchCallStatus,
};
use smg_mcp::{McpToolSession, ToolExecutionInput};

use super::{
    error::AgentLoopError,
    presentation::{OutputFamily, ToolPresentation},
    state::{AgentLoopState, ExecutedCall, PlannedToolExecution},
};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::common::mcp_utils::DEFAULT_MAX_ITERATIONS,
};

/// Effective per-request gateway-tool budget. The user's
/// `max_tool_calls`, when set, is clamped by SMG's safety limit so a
/// caller cannot opt out of the gateway-side ceiling. This is a public
/// behavior contract per the design doc.
pub(crate) fn effective_tool_call_limit(max_tool_calls: Option<usize>) -> usize {
    match max_tool_calls {
        Some(user_max) => user_max.min(DEFAULT_MAX_ITERATIONS),
        None => DEFAULT_MAX_ITERATIONS,
    }
}

/// Remaining budget the loop is allowed to spend on more gateway tool
/// calls, after `state.total_gateway_tool_calls` have already run.
pub(crate) fn remaining_tool_call_budget(
    state: &AgentLoopState,
    max_tool_calls: Option<usize>,
) -> usize {
    effective_tool_call_limit(max_tool_calls).saturating_sub(state.total_gateway_tool_calls)
}

/// Execute one planned MCP tool call against the request-scoped
/// session. The driver has already gated approval-required calls out
/// of this batch via `decide_after_call_llm`, so every entry that
/// reaches this function is meant to dispatch unconditionally — the
/// orchestrator never re-checks policy on its own. Returns the
/// resulting [`ExecutedCall`] (boxed because it carries a transformed
/// `ResponseOutputItem` and the stringified output, both potentially
/// large per call).
pub(crate) async fn execute_planned_tool(
    session: &McpToolSession<'_>,
    plan: PlannedToolExecution,
    model_id: &str,
) -> Result<Box<ExecutedCall>, AgentLoopError> {
    let PlannedToolExecution {
        call,
        server_label,
        presentation,
    } = plan;

    // Reject malformed JSON / non-object arguments BEFORE dispatch.
    // The driver routes these through the same `ToolCompleted` path as
    // a normal failed execution, so streaming (`mcp_call.failed` +
    // `output_item.done(status=failed)`) and non-streaming (`mcp_call`
    // with `status: failed`, `error: <reason>`) surfaces are
    // wire-identical to a tool error returned by the server itself.
    // This replaces the prior coerce-to-`{}` behaviour, which silently
    // executed gateway tools with empty arguments.
    let arguments = match serde_json::from_str::<serde_json::Value>(&call.arguments) {
        Ok(serde_json::Value::Object(map)) => serde_json::Value::Object(map),
        Ok(_) | Err(_) => {
            let reason = format!(
                "Invalid tool arguments: expected JSON object, got {}",
                truncate_for_message(&call.arguments)
            );
            Metrics::record_mcp_tool_call(model_id, &call.name, metrics_labels::RESULT_ERROR);
            return Ok(Box::new(ExecutedCall {
                approval_request_id: call.approval_request_id.clone(),
                transformed_item: Some(synthesize_error_item(
                    &call,
                    &server_label,
                    &presentation,
                    &reason,
                )),
                output_string: reason,
                is_error: true,
                call_id: call.call_id,
                item_id: call.item_id,
                name: call.name,
                arguments: call.arguments,
            }));
        }
    };

    let tool_output = session
        .execute_tool(ToolExecutionInput {
            call_id: call.call_id.clone(),
            tool_name: call.name.clone(),
            arguments,
        })
        .await;

    Metrics::record_mcp_tool_duration(model_id, &tool_output.tool_name, tool_output.duration);
    Metrics::record_mcp_tool_call(
        model_id,
        &tool_output.tool_name,
        if tool_output.is_error {
            metrics_labels::RESULT_ERROR
        } else {
            metrics_labels::RESULT_SUCCESS
        },
    );

    let output_string = tool_output.output.to_string();
    let mut transformed_item = tool_output.to_response_item();
    backfill_approval_request_id(&mut transformed_item, call.approval_request_id.as_deref());
    Ok(Box::new(ExecutedCall {
        call_id: call.call_id,
        item_id: call.item_id,
        name: call.name,
        arguments: call.arguments,
        output_string,
        transformed_item: Some(transformed_item),
        is_error: tool_output.is_error,
        approval_request_id: call.approval_request_id,
    }))
}

fn backfill_approval_request_id(item: &mut ResponseOutputItem, approval_request_id: Option<&str>) {
    let Some(id) = approval_request_id else {
        return;
    };
    if let ResponseOutputItem::McpCall {
        approval_request_id,
        ..
    } = item
    {
        *approval_request_id = Some(id.to_string());
    }
}

/// Build a family-aware error output item for a malformed-arguments
/// rejection. The presentation layer's `render_final_item` would also
/// produce a wire-correct shape from a `transformed_item: None`
/// `ExecutedCall`, but pre-rendering here keeps the persisted
/// non-streaming response and the streaming `output_item.done` payload
/// aligned with the executor-error path: transformed item present,
/// family preserved, and failed status set before the sink overlays any
/// wire-only error fields.
fn synthesize_error_item(
    call: &super::state::LoopToolCall,
    server_label: &str,
    presentation: &ToolPresentation,
    reason: &str,
) -> ResponseOutputItem {
    match presentation.family {
        OutputFamily::McpCall => ResponseOutputItem::McpCall {
            id: call.item_id.clone(),
            status: "failed".to_string(),
            approval_request_id: call.approval_request_id.clone(),
            arguments: call.arguments.clone(),
            error: Some(reason.to_string()),
            name: call.name.clone(),
            output: String::new(),
            server_label: server_label.to_string(),
        },
        OutputFamily::WebSearchCall => ResponseOutputItem::WebSearchCall {
            id: call.item_id.clone(),
            status: WebSearchCallStatus::Failed,
            action: WebSearchAction::Search {
                query: None,
                queries: Vec::new(),
                sources: Vec::new(),
            },
            results: None,
        },
        OutputFamily::CodeInterpreterCall => ResponseOutputItem::CodeInterpreterCall {
            id: call.item_id.clone(),
            status: CodeInterpreterCallStatus::Failed,
            container_id: String::new(),
            code: None,
            outputs: None,
        },
        OutputFamily::FileSearchCall => ResponseOutputItem::FileSearchCall {
            id: call.item_id.clone(),
            status: FileSearchCallStatus::Failed,
            queries: Vec::new(),
            results: None,
        },
        OutputFamily::ImageGenerationCall => ResponseOutputItem::ImageGenerationCall {
            id: call.item_id.clone(),
            action: None,
            background: None,
            output_format: None,
            quality: None,
            result: String::new(),
            revised_prompt: None,
            size: None,
            status: ImageGenerationCallStatus::Failed,
        },
        OutputFamily::Function => ResponseOutputItem::FunctionToolCall {
            id: call.item_id.clone(),
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
            output: None,
            status: "failed".to_string(),
        },
    }
}

fn truncate_for_message(value: &str) -> String {
    const MAX: usize = 80;
    if value.len() <= MAX {
        value.to_string()
    } else {
        format!("{}…", &value[..MAX])
    }
}

#[cfg(test)]
mod tests {
    use smg_mcp::{McpOrchestrator, McpToolSession};

    use super::*;
    use crate::routers::common::agent_loop::state::LoopToolCall;

    fn passthrough_plan(arguments: &str) -> PlannedToolExecution {
        PlannedToolExecution {
            call: LoopToolCall {
                call_id: "call_x".to_string(),
                item_id: "fc_x".to_string(),
                name: "search".to_string(),
                arguments: arguments.to_string(),
                approval_request_id: None,
            },
            server_label: "deepwiki".to_string(),
            presentation: ToolPresentation::default(),
        }
    }

    /// Malformed argument JSON must become a tool-error result the
    /// driver can surface through the normal `ToolCompleted` path —
    /// not a silent coerce-to-`{}` execution. Review P1.4.
    #[tokio::test]
    async fn malformed_arguments_returns_tool_error_without_executing() {
        let orchestrator = McpOrchestrator::new_test();
        let session = McpToolSession::new(&orchestrator, vec![], "test-request");
        let plan = passthrough_plan("not-valid-json");
        let executed = execute_planned_tool(&session, plan, "model-x")
            .await
            .expect("malformed args must produce an Ok(ExecutedCall) with is_error");
        assert!(executed.is_error, "malformed args must mark is_error");
        assert!(
            executed.output_string.contains("Invalid tool arguments"),
            "error message must mention invalid arguments, got {:?}",
            executed.output_string
        );
        // Transformed item is pre-built so streaming + non-streaming
        // paths render identical wire payloads.
        let item = executed
            .transformed_item
            .as_ref()
            .expect("malformed-args path must populate transformed_item");
        let v = serde_json::to_value(item).unwrap();
        assert_eq!(v["status"], "failed");
    }

    /// Non-object JSON (e.g. an array) is also malformed for tool
    /// arguments — same rejection path as syntactically broken JSON.
    #[tokio::test]
    async fn non_object_arguments_returns_tool_error() {
        let orchestrator = McpOrchestrator::new_test();
        let session = McpToolSession::new(&orchestrator, vec![], "test-request");
        let plan = passthrough_plan("[1,2,3]");
        let executed = execute_planned_tool(&session, plan, "model-x")
            .await
            .expect("non-object args must surface as a tool error");
        assert!(executed.is_error);
        assert!(executed.output_string.contains("Invalid tool arguments"));
    }

    /// Malformed hosted-builtin calls must keep their hosted output
    /// family. Streaming clients already saw `web_search_call.*`; the
    /// final item must not suddenly become an `mcp_call`.
    #[tokio::test]
    async fn malformed_hosted_builtin_arguments_keep_family() {
        let orchestrator = McpOrchestrator::new_test();
        let session = McpToolSession::new(&orchestrator, vec![], "test-request");
        let mut plan = passthrough_plan("not-json");
        plan.presentation = ToolPresentation::from_family(OutputFamily::WebSearchCall);

        let executed = execute_planned_tool(&session, plan, "model-x")
            .await
            .expect("malformed hosted builtin args must surface as tool error");

        assert!(executed.is_error);
        let item = executed.transformed_item.as_ref().expect("missing item");
        let v = serde_json::to_value(item).unwrap();
        assert_eq!(v["type"], "web_search_call");
        assert_eq!(v["status"], "failed");
    }

    /// Approved continuation metadata belongs in the common execution
    /// result, not only in the streaming sink, so non-streaming and
    /// persisted responses echo the original `mcpr_*` id too.
    #[tokio::test]
    async fn approval_request_id_is_backfilled_on_executed_mcp_item() {
        let orchestrator = McpOrchestrator::new_test();
        let session = McpToolSession::new(&orchestrator, vec![], "test-request");
        let mut plan = passthrough_plan("{}");
        plan.call.approval_request_id = Some("mcpr_call_x".to_string());

        let executed = execute_planned_tool(&session, plan, "model-x")
            .await
            .expect("execution fallback should still render an item");

        let item = executed.transformed_item.as_ref().expect("missing item");
        let v = serde_json::to_value(item).unwrap();
        assert_eq!(v["type"], "mcp_call");
        assert_eq!(v["approval_request_id"], "mcpr_call_x");
    }
}
