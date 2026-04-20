//! Helpers for non-streaming MCP approval continuations on the Responses API.
//!
//! This module keeps two concerns together:
//! 1. Build the sanitized input we forward upstream.
//! 2. Parse the current turn's `mcp_approval_response` into a concrete
//!    continuation we can execute locally.

use openai_protocol::responses::{
    approval_request_id_to_call_id, normalize_input_item, ResponseInput, ResponseInputOutputItem,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ApprovalContinuation {
    pub approval_request_id: String,
    pub call_id: String,
    pub tool_name: String,
    /// Captured from the stored `mcp_approval_request.server_label` so replay
    /// can reject a continuation that rebinds the same tool name to a
    /// different MCP backend between turns.
    pub server_label: String,
    pub arguments: String,
}

pub(crate) struct PreparedResponsesInput {
    pub upstream_input: ResponseInput,
    pub approval_continuation: Option<ApprovalContinuation>,
}

pub(crate) fn prepare_responses_input(
    merged_input: &ResponseInput,
    pending_mcp_approval_requests: &[ResponseInputOutputItem],
    is_streaming: bool,
) -> Result<PreparedResponsesInput, String> {
    let approval_continuation =
        extract_approval_continuation(merged_input, pending_mcp_approval_requests, is_streaming)?;

    Ok(PreparedResponsesInput {
        upstream_input: sanitize_input_for_upstream(merged_input),
        approval_continuation,
    })
}

pub(crate) fn sanitize_input_for_upstream(input: &ResponseInput) -> ResponseInput {
    match input {
        ResponseInput::Text(text) => ResponseInput::Text(text.clone()),
        ResponseInput::Items(items) => ResponseInput::Items(
            items
                .iter()
                .map(normalize_input_item)
                .filter(|item| {
                    !matches!(
                        item,
                        ResponseInputOutputItem::McpApprovalRequest { .. }
                            | ResponseInputOutputItem::McpApprovalResponse { .. }
                    )
                })
                .collect(),
        ),
    }
}

fn extract_approval_continuation(
    merged_input: &ResponseInput,
    pending_mcp_approval_requests: &[ResponseInputOutputItem],
    is_streaming: bool,
) -> Result<Option<ApprovalContinuation>, String> {
    let approval_responses = normalized_input_items(merged_input)
        .into_iter()
        .filter_map(|item| match item {
            ResponseInputOutputItem::McpApprovalResponse {
                approval_request_id,
                approve,
                ..
            } => Some((approval_request_id, approve)),
            _ => None,
        })
        .collect::<Vec<_>>();

    if approval_responses.is_empty() {
        return Ok(None);
    }

    // Defensive check: protocol validation already rejects streaming
    // approval-only continuations at the HTTP boundary.
    if is_streaming {
        return Err(
            "mcp_approval_response is not supported for streaming responses requests".to_string(),
        );
    }

    // Defensive check: protocol validation already enforces a single
    // `mcp_approval_response` item per request.
    if approval_responses.len() > 1 {
        return Err("Only one mcp_approval_response item is supported per request".to_string());
    }

    let (approval_request_id, approve) = &approval_responses[0];
    if !approve {
        return Err(
            "mcp_approval_response.approve=false is not supported for previous_response_id continuations"
                .to_string(),
        );
    }

    let Some((tool_name, server_label, arguments)) = pending_mcp_approval_requests
        .iter()
        .rev()
        .find_map(|item| match item {
            ResponseInputOutputItem::McpApprovalRequest {
                id,
                name,
                server_label,
                arguments,
            } if id == approval_request_id => {
                Some((name.clone(), server_label.clone(), arguments.clone()))
            }
            _ => None,
        })
    else {
        return Err(format!(
            "mcp_approval_response.approval_request_id '{approval_request_id}' does not match any pending mcp_approval_request"
        ));
    };

    Ok(Some(ApprovalContinuation {
        approval_request_id: approval_request_id.clone(),
        call_id: approval_request_id_to_call_id(approval_request_id),
        tool_name,
        server_label,
        arguments,
    }))
}

fn normalized_input_items(input: &ResponseInput) -> Vec<ResponseInputOutputItem> {
    match input {
        ResponseInput::Text(_) => Vec::new(),
        ResponseInput::Items(items) => items.iter().map(normalize_input_item).collect(),
    }
}
