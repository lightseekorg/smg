//! Input history loading for the Responses API.
//!
//! Loads conversation history and/or previous response chains into the request
//! input before forwarding to the upstream provider.

use std::collections::HashSet;

use axum::response::Response;
use openai_protocol::{
    event_types::ItemType,
    responses::{
        approval_request_id_to_call_id, mcp_item_id_to_prefixed_id, normalize_input_item,
        ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponsesRequest,
    },
};
use serde_json::Value;
use smg_data_connector::{ConversationId, ListParams, ResponseId, ResponseStorageError, SortOrder};
use tracing::{debug, warn};

use super::super::context::ResponsesComponents;
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::{
        common::{
            header_utils::ConversationMemoryConfig, persistence_utils::split_stored_message_content,
        },
        error,
    },
};

const MAX_CONVERSATION_HISTORY_ITEMS: usize = 100;

pub(crate) struct LoadedInputHistory {
    pub previous_response_id: Option<String>,
    pub existing_mcp_list_tools_labels: Vec<String>,
    pub pending_mcp_approval_requests: Vec<ResponseInputOutputItem>,
}

/// Load conversation history and/or previous response chain into request input.
///
/// Mutates `request_body.input` with the loaded items.
/// Returns `Ok(LoadedInputHistory)` on success, or `Err(response)` on validation failure.
pub(crate) async fn load_input_history(
    components: &ResponsesComponents,
    conversation: Option<&str>,
    request_body: &mut ResponsesRequest,
    model: &str,
) -> Result<LoadedInputHistory, Response> {
    let previous_response_id = request_body
        .previous_response_id
        .take()
        .filter(|id| !id.is_empty());
    let mut existing_mcp_list_tools_labels = HashSet::new();
    let mut pending_mcp_approval_requests = Vec::new();

    // Load items from previous response chain if specified
    let mut chain_items: Option<Vec<ResponseInputOutputItem>> = None;
    if let Some(prev_id_str) = &previous_response_id {
        let prev_id = ResponseId::from(prev_id_str.as_str());
        match components
            .response_storage
            .get_response_chain(&prev_id, None)
            .await
        {
            Ok(chain) if !chain.responses.is_empty() => {
                existing_mcp_list_tools_labels.extend(chain.responses.iter().flat_map(|stored| {
                    extract_mcp_list_tools_labels(
                        stored.raw_response.get("output").unwrap_or(&Value::Null),
                    )
                }));

                let consumed_mcp_approval_request_ids = chain
                    .responses
                    .iter()
                    .flat_map(|stored| {
                        extract_consumed_mcp_approval_request_ids_from_array(
                            stored
                                .raw_response
                                .get("output")
                                .unwrap_or(&Value::Array(vec![])),
                        )
                    })
                    .collect::<HashSet<_>>();

                pending_mcp_approval_requests = chain
                    .responses
                    .iter()
                    .flat_map(|stored| {
                        extract_mcp_approval_requests_from_array(
                            stored
                                .raw_response
                                .get("output")
                                .unwrap_or(&Value::Array(vec![])),
                        )
                    })
                    .filter(|item| match item {
                        ResponseInputOutputItem::McpApprovalRequest { id, .. } => {
                            !consumed_mcp_approval_request_ids.contains(id)
                        }
                        _ => true,
                    })
                    .collect();

                let items: Vec<ResponseInputOutputItem> = chain
                    .responses
                    .iter()
                    .flat_map(|stored| {
                        deserialize_upstream_input_items(&stored.input)
                            .into_iter()
                            .chain(deserialize_upstream_output_items_from_array(
                                stored
                                    .raw_response
                                    .get("output")
                                    .unwrap_or(&Value::Array(vec![])),
                            ))
                    })
                    .collect();
                chain_items = Some(items);
            }
            Ok(_) | Err(ResponseStorageError::ResponseNotFound(_)) => {
                Metrics::record_router_error(
                    metrics_labels::ROUTER_OPENAI,
                    metrics_labels::BACKEND_EXTERNAL,
                    metrics_labels::CONNECTION_HTTP,
                    model,
                    metrics_labels::ENDPOINT_RESPONSES,
                    metrics_labels::ERROR_VALIDATION,
                );
                return Err(error::bad_request(
                    "previous_response_not_found",
                    format!("Previous response with id '{prev_id_str}' not found."),
                ));
            }
            Err(e) => {
                warn!(
                    "Failed to load previous response chain for {}: {}",
                    prev_id_str, e
                );
                Metrics::record_router_error(
                    metrics_labels::ROUTER_OPENAI,
                    metrics_labels::BACKEND_EXTERNAL,
                    metrics_labels::CONNECTION_HTTP,
                    model,
                    metrics_labels::ENDPOINT_RESPONSES,
                    metrics_labels::ERROR_INTERNAL,
                );
                return Err(error::internal_error(
                    "load_previous_response_chain_failed",
                    format!("Failed to load previous response chain for {prev_id_str}: {e}"),
                ));
            }
        }
    }

    // Load conversation history if specified
    if let Some(conv_id_str) = conversation {
        let conv_id = ConversationId::from(conv_id_str);

        if let Ok(None) = components
            .conversation_storage
            .get_conversation(&conv_id)
            .await
        {
            Metrics::record_router_error(
                metrics_labels::ROUTER_OPENAI,
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::CONNECTION_HTTP,
                model,
                metrics_labels::ENDPOINT_RESPONSES,
                metrics_labels::ERROR_VALIDATION,
            );
            return Err(error::not_found(
                "not_found",
                format!("No conversation found with id '{}'", conv_id.0),
            ));
        }

        let params = ListParams {
            limit: MAX_CONVERSATION_HISTORY_ITEMS,
            order: SortOrder::Asc,
            after: None,
        };

        match components
            .conversation_item_storage
            .list_items(&conv_id, params)
            .await
        {
            Ok(stored_items) => {
                let mut items: Vec<ResponseInputOutputItem> = Vec::new();
                for item in stored_items {
                    match item.item_type.as_str() {
                        "message" => {
                            // Stored content may be either the raw content array
                            // (legacy shape) or an object `{content: [...], phase: ...}`
                            // when the message carried a phase label (P3).
                            let (content_value, stored_phase) =
                                split_stored_message_content(item.content);
                            match serde_json::from_value::<Vec<ResponseContentPart>>(content_value)
                            {
                                Ok(content_parts) => {
                                    items.push(ResponseInputOutputItem::Message {
                                        id: item.id.0.clone(),
                                        role: item
                                            .role
                                            .clone()
                                            .unwrap_or_else(|| "user".to_string()),
                                        content: content_parts,
                                        status: item.status.clone(),
                                        phase: stored_phase,
                                    });
                                }
                                Err(e) => {
                                    tracing::error!("Failed to deserialize message content: {}", e);
                                }
                            }
                        }
                        ItemType::FUNCTION_CALL => {
                            match serde_json::from_value::<ResponseInputOutputItem>(item.content) {
                                Ok(func_call) => items.push(func_call),
                                Err(e) => {
                                    tracing::error!("Failed to deserialize function_call: {}", e);
                                }
                            }
                        }
                        ItemType::FUNCTION_CALL_OUTPUT => {
                            tracing::debug!(
                                item_id = %item.id.0,
                                "Loading function_call_output from DB"
                            );
                            match serde_json::from_value::<ResponseInputOutputItem>(item.content) {
                                Ok(func_output) => {
                                    tracing::debug!(
                                        "Successfully deserialized function_call_output"
                                    );
                                    items.push(func_output);
                                }
                                Err(e) => {
                                    tracing::error!(
                                        "Failed to deserialize function_call_output: {}",
                                        e
                                    );
                                }
                            }
                        }
                        "reasoning" => {}
                        _ => {
                            warn!("Unknown item type in conversation: {}", item.item_type);
                        }
                    }
                }

                append_current_input(&mut items, &request_body.input, conv_id_str);
                request_body.input = ResponseInput::Items(items);
            }
            Err(e) => {
                warn!("Failed to load conversation history: {}", e);
            }
        }
    }

    // Apply previous response chain items if loaded.
    // Note: conversation and previous_response_id are mutually exclusive
    // (enforced by the caller in route_responses), so this branch and the
    // conversation branch above never both modify request_body.input.
    if let Some(mut items) = chain_items {
        let id_suffix = previous_response_id.as_deref().unwrap_or("new");
        append_current_input(&mut items, &request_body.input, id_suffix);
        request_body.input = ResponseInput::Items(items);
    }

    Ok(LoadedInputHistory {
        previous_response_id,
        existing_mcp_list_tools_labels: existing_mcp_list_tools_labels.into_iter().collect(),
        pending_mcp_approval_requests,
    })
}

fn extract_consumed_mcp_approval_request_ids_from_array(array: &Value) -> Vec<String> {
    let Some(arr) = array.as_array() else {
        return Vec::new();
    };

    arr.iter()
        .filter_map(|item| {
            (item.get("type").and_then(|value| value.as_str()) == Some(ItemType::MCP_CALL))
                .then(|| {
                    item.get("approval_request_id")
                        .and_then(|value| value.as_str())
                })
                .flatten()
                .map(ToOwned::to_owned)
        })
        .collect()
}

fn extract_mcp_approval_requests_from_array(array: &Value) -> Vec<ResponseInputOutputItem> {
    let Some(arr) = array.as_array() else {
        return Vec::new();
    };
    arr.iter()
        .filter(|item| {
            item.get("type").and_then(|value| value.as_str()) == Some("mcp_approval_request")
        })
        .filter_map(|item| {
            serde_json::from_value::<ResponseInputOutputItem>(item.clone())
                .map_err(|e| warn!("Failed to deserialize mcp_approval_request for replay: {e}"))
                .ok()
        })
        .collect()
}

fn deserialize_upstream_input_items(input: &Value) -> Vec<ResponseInputOutputItem> {
    match input {
        Value::String(text) => vec![ResponseInputOutputItem::new_user_text(text.clone())],
        Value::Array(arr) => arr
            .iter()
            .flat_map(upstream_input_items_from_value)
            .collect(),
        _ => Vec::new(),
    }
}

fn deserialize_upstream_output_items_from_array(array: &Value) -> Vec<ResponseInputOutputItem> {
    array
        .as_array()
        .map(|arr| {
            arr.iter()
                .flat_map(upstream_output_items_from_value)
                .collect()
        })
        .unwrap_or_default()
}

fn upstream_input_items_from_value(item: &Value) -> Vec<ResponseInputOutputItem> {
    let parsed = match serde_json::from_value::<ResponseInputOutputItem>(item.clone()) {
        Ok(parsed) => parsed,
        Err(e) => {
            warn!(
                error = %e,
                item_type = replay_item_type(item),
                item_id = replay_item_id(item),
                "Failed to deserialize input item for upstream replay"
            );
            return Vec::new();
        }
    };

    match normalize_input_item(&parsed) {
        ResponseInputOutputItem::McpApprovalRequest { .. }
        | ResponseInputOutputItem::McpApprovalResponse { .. } => Vec::new(),
        item => vec![item],
    }
}

fn upstream_output_items_from_value(item: &Value) -> Vec<ResponseInputOutputItem> {
    match item.get("type").and_then(|value| value.as_str()) {
        Some(ItemType::MCP_LIST_TOOLS) | Some("mcp_approval_request") => Vec::new(),
        Some(ItemType::MCP_CALL) => mcp_call_output_to_upstream_items(item),
        _ => upstream_input_items_from_value(item),
    }
}

fn mcp_call_output_to_upstream_items(item: &Value) -> Vec<ResponseInputOutputItem> {
    let Some(id) = item.get("id").and_then(|value| value.as_str()) else {
        warn!(
            item_type = replay_item_type(item),
            item_id = replay_item_id(item),
            "Skipping mcp_call without id during upstream replay"
        );
        return Vec::new();
    };
    let Some(name) = item.get("name").and_then(|value| value.as_str()) else {
        warn!(
            item_type = replay_item_type(item),
            item_id = replay_item_id(item),
            "Skipping mcp_call without name during upstream replay"
        );
        return Vec::new();
    };
    let Some(arguments) = item.get("arguments").and_then(|value| value.as_str()) else {
        warn!(
            item_type = replay_item_type(item),
            item_id = replay_item_id(item),
            "Skipping mcp_call without arguments during upstream replay"
        );
        return Vec::new();
    };

    let status = normalize_replayed_tool_status(
        item.get("status")
            .and_then(|value| value.as_str())
            .unwrap_or("completed"),
    );
    let output = mcp_call_output_string(item);

    let call_id = item
        .get("approval_request_id")
        .and_then(|value| value.as_str())
        .map(approval_request_id_to_call_id)
        .unwrap_or_else(|| mcp_item_id_to_prefixed_id(id, "call_"));

    vec![
        ResponseInputOutputItem::FunctionToolCall {
            id: mcp_item_id_to_prefixed_id(id, "fc_"),
            call_id: call_id.clone(),
            name: name.to_string(),
            arguments: arguments.to_string(),
            output: None,
            status: Some(status.clone()),
        },
        ResponseInputOutputItem::FunctionCallOutput {
            id: None,
            call_id,
            output,
            status: Some(status),
        },
    ]
}

fn mcp_call_output_string(item: &Value) -> String {
    let output = item.get("output").cloned().unwrap_or(Value::Null);
    let error = item.get("error").cloned().unwrap_or(Value::Null);

    if error.is_null() {
        return match output {
            Value::String(text) => text,
            other => other.to_string(),
        };
    }

    json_output_with_error(output, error).to_string()
}

fn normalize_replayed_tool_status(status: &str) -> String {
    match status {
        // Persisted MCP calls should already be terminal. If a non-terminal
        // status somehow lands in storage, normalize it back to `completed`
        // so upstream replay does not resurrect an in-progress tool call.
        "completed" | "failed" | "incomplete" | "cancelled" => status.to_string(),
        _ => "completed".to_string(),
    }
}

fn json_output_with_error(output: Value, error: Value) -> Value {
    serde_json::json!({
        "output": output,
        "error": error,
    })
}

fn replay_item_type(item: &Value) -> &str {
    item.get("type")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown")
}

fn replay_item_id(item: &Value) -> &str {
    item.get("id")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown")
}

fn extract_mcp_list_tools_labels(array: &Value) -> Vec<String> {
    array
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    (item.get("type").and_then(|t| t.as_str()) == Some(ItemType::MCP_LIST_TOOLS))
                        .then(|| item.get("server_label").and_then(|v| v.as_str()))
                        .flatten()
                        .map(ToOwned::to_owned)
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Append current request input to items list, creating a user message if needed
fn append_current_input(
    items: &mut Vec<ResponseInputOutputItem>,
    input: &ResponseInput,
    id_suffix: &str,
) {
    match input {
        ResponseInput::Text(text) => {
            items.push(ResponseInputOutputItem::Message {
                id: format!("msg_u_{id_suffix}"),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText { text: text.clone() }],
                status: Some("completed".to_string()),
                phase: None,
            });
        }
        ResponseInput::Items(current_items) => {
            items.extend(current_items.iter().map(normalize_input_item));
        }
    }
}

/// Memory hook entrypoint for Responses API.
///
/// This is intentionally a no-op in this PR: it confirms header parsing is
/// connected to request flow and logs activation state for follow-up retrieval work.
pub(crate) fn inject_memory_context(
    config: &ConversationMemoryConfig,
    _request_body: &mut ResponsesRequest,
) {
    if config.long_term_memory.enabled {
        debug!(
            has_subject_id = config.long_term_memory.subject_id.is_some(),
            has_embedding_model = config.long_term_memory.embedding_model_id.is_some(),
            has_extraction_model = config.long_term_memory.extraction_model_id.is_some(),
            "LTM recall requested - retrieval not yet implemented"
        );
    }

    if config.short_term_memory.enabled {
        debug!(
            has_condenser_model = config.short_term_memory.condenser_model_id.is_some(),
            "STM recall requested - retrieval not yet implemented"
        );
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::responses::{ResponseInput, ResponseInputOutputItem, ResponsesRequest};
    use serde_json::{json, Value};

    use super::{
        inject_memory_context, mcp_call_output_string, mcp_call_output_to_upstream_items,
        normalize_replayed_tool_status,
    };
    use crate::routers::common::header_utils::{
        ConversationMemoryConfig, LongTermMemoryConfig, ShortTermMemoryConfig,
    };

    #[test]
    fn inject_memory_context_is_no_op_for_now() {
        let config = ConversationMemoryConfig {
            long_term_memory: LongTermMemoryConfig {
                enabled: true,
                policy: None,
                subject_id: Some("subj-1".to_string()),
                embedding_model_id: Some("embed-1".to_string()),
                extraction_model_id: Some("extract-1".to_string()),
            },
            short_term_memory: ShortTermMemoryConfig {
                enabled: true,
                condenser_model_id: Some("condense-1".to_string()),
            },
        };
        let mut request = ResponsesRequest {
            input: ResponseInput::Text("hello".to_string()),
            ..Default::default()
        };

        inject_memory_context(&config, &mut request);

        match request.input {
            ResponseInput::Text(text) => assert_eq!(text, "hello"),
            ResponseInput::Items(_) => {
                panic!("request input should remain unchanged for no-op hook")
            }
        }
    }

    #[test]
    fn mcp_call_output_string_wraps_error_with_output() {
        let item = json!({
            "type": "mcp_call",
            "output": "partial output",
            "error": "boom",
        });

        let output = mcp_call_output_string(&item);
        let parsed: Value = serde_json::from_str(&output).expect("wrapped output should be json");
        assert_eq!(
            parsed,
            json!({
                "output": "partial output",
                "error": "boom",
            })
        );
    }

    #[test]
    fn mcp_call_output_string_preserves_output_when_error_missing() {
        let item = json!({
            "type": "mcp_call",
            "output": "plain output",
        });

        assert_eq!(mcp_call_output_string(&item), "plain output");
    }

    #[test]
    fn normalize_replayed_tool_status_falls_back_to_completed_for_non_terminal_values() {
        assert_eq!(normalize_replayed_tool_status("in_progress"), "completed");
        assert_eq!(normalize_replayed_tool_status("queued"), "completed");
    }

    #[test]
    fn mcp_call_output_to_upstream_items_keeps_terminal_status_and_normalizes_non_terminal() {
        let terminal = json!({
            "type": "mcp_call",
            "id": "mcp_123",
            "name": "tool",
            "arguments": "{}",
            "output": "ok",
            "status": "failed",
        });
        let terminal_items = mcp_call_output_to_upstream_items(&terminal);
        assert_eq!(terminal_items.len(), 2);
        match &terminal_items[0] {
            ResponseInputOutputItem::FunctionToolCall { status, .. } => {
                assert_eq!(status.as_deref(), Some("failed"));
            }
            other => panic!("expected function_call, got {other:?}"),
        }
        match &terminal_items[1] {
            ResponseInputOutputItem::FunctionCallOutput { status, .. } => {
                assert_eq!(status.as_deref(), Some("failed"));
            }
            other => panic!("expected function_call_output, got {other:?}"),
        }

        let non_terminal = json!({
            "type": "mcp_call",
            "id": "mcp_456",
            "name": "tool",
            "arguments": "{}",
            "output": "ok",
            "status": "in_progress",
        });
        let non_terminal_items = mcp_call_output_to_upstream_items(&non_terminal);
        match &non_terminal_items[0] {
            ResponseInputOutputItem::FunctionToolCall { status, .. } => {
                assert_eq!(status.as_deref(), Some("completed"));
            }
            other => panic!("expected function_call, got {other:?}"),
        }
        match &non_terminal_items[1] {
            ResponseInputOutputItem::FunctionCallOutput { status, .. } => {
                assert_eq!(status.as_deref(), Some("completed"));
            }
            other => panic!("expected function_call_output, got {other:?}"),
        }
    }
}
