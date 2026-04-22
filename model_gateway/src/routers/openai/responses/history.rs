//! Input history loading and normalization for the Responses API.
//!
//! Loading is split into two phases so that stitched `input` and
//! server-replayed history go through the same normalization boundary:
//!
//! - [`load_input_history`] fetches server-managed transcript sources
//!   (`conversation`, `previous_response_id`) and appends the incoming client
//!   `input` to them. It is source-acquisition only.
//! - [`prepare_agent_loop_input`] normalizes the combined transcript into an
//!   upstream-consumable shape and records control signals (the set of
//!   `mcp_list_tools` server labels already present in the transcript).

use std::collections::HashSet;

use axum::response::Response;
use openai_protocol::{
    event_types::ItemType,
    responses::{
        normalize_input_item, ResponseContentPart, ResponseInput, ResponseInputOutputItem,
        ResponsesRequest,
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
}

/// Result of normalizing a Responses request transcript for the agent loop.
///
/// - `upstream_input` is the transcript in LLM-consumable form: message,
///   reasoning, function_call, function_call_output, and approval items.
/// - `existing_mcp_list_tools_labels` records server labels whose
///   `mcp_list_tools` has already been surfaced in the transcript, so the
///   loop can avoid re-emitting them.
pub(crate) struct PreparedAgentLoopInput {
    pub upstream_input: ResponseInput,
    pub existing_mcp_list_tools_labels: Vec<String>,
}

/// Load conversation history and/or previous response chain into request input.
///
/// Mutates `request_body.input` with the loaded items. This function does
/// source acquisition only — normalization (expanding `mcp_call` into
/// function-call pairs, collecting `mcp_list_tools` labels, etc.) is the job
/// of [`prepare_agent_loop_input`].
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
                let items: Vec<ResponseInputOutputItem> = chain
                    .responses
                    .iter()
                    .flat_map(|stored| {
                        deserialize_items_from_array(&stored.input)
                            .into_iter()
                            .chain(deserialize_items_from_array(
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
    })
}

/// Normalize a combined transcript into an upstream-consumable shape.
///
/// Called by the router after [`load_input_history`]. Both server-managed
/// replay (via `previous_response_id` / `conversation`) and client-stitched
/// `input` go through this single boundary.
///
/// Rules applied here:
/// - `mcp_call` items are expanded into a `function_call` + `function_call_output`
///   pair so the upstream model sees a replayable tool execution rather than
///   a Responses-only wrapper that could be re-executed.
/// - `mcp_list_tools` items are stripped from the upstream transcript — they
///   are client-visible metadata, not LLM context — but their `server_label`
///   is recorded so the loop can dedupe re-emission.
/// - `SimpleInputMessage` is normalized to a full `Message` so downstream
///   code sees a single shape.
pub(crate) fn prepare_agent_loop_input(input: &ResponseInput) -> PreparedAgentLoopInput {
    let items = match input {
        ResponseInput::Text(_) => {
            return PreparedAgentLoopInput {
                upstream_input: input.clone(),
                existing_mcp_list_tools_labels: Vec::new(),
            };
        }
        ResponseInput::Items(items) => items,
    };

    let mut upstream_items = Vec::with_capacity(items.len());
    let mut seen_labels: HashSet<String> = HashSet::new();
    let mut existing_mcp_list_tools_labels = Vec::new();

    for item in items {
        match normalize_input_item(item) {
            ResponseInputOutputItem::McpListTools { server_label, .. } => {
                if seen_labels.insert(server_label.clone()) {
                    existing_mcp_list_tools_labels.push(server_label);
                }
            }
            ResponseInputOutputItem::McpCall {
                id,
                approval_request_id,
                arguments,
                name,
                output,
                error,
                ..
            } => {
                let call_id = approval_request_id
                    .as_deref()
                    .map(approval_request_to_call_id)
                    .unwrap_or_else(|| mcp_item_id_to_prefixed_id(&id, "call_"));
                // A failed replayed call carries its diagnostic in `error`
                // (with `output` often empty); surface it as the upstream
                // function_call_output so the model sees the failure reason
                // instead of a silently "successful" empty result.
                let tool_output = error.unwrap_or(output);
                upstream_items.push(ResponseInputOutputItem::FunctionToolCall {
                    id: mcp_item_id_to_prefixed_id(&id, "fc_"),
                    call_id: call_id.clone(),
                    name,
                    arguments,
                    output: None,
                    status: Some("completed".to_string()),
                });
                upstream_items.push(ResponseInputOutputItem::FunctionCallOutput {
                    id: None,
                    call_id,
                    output: tool_output,
                    status: Some("completed".to_string()),
                });
            }
            normalized => upstream_items.push(normalized),
        }
    }

    PreparedAgentLoopInput {
        upstream_input: ResponseInput::Items(upstream_items),
        existing_mcp_list_tools_labels,
    }
}

/// Derive a `call_*` id from an `mcpr_*` approval-request id, preserving the
/// suffix so the same correlation id flows through function-call replay.
fn approval_request_to_call_id(approval_request_id: &str) -> String {
    mcp_item_id_to_prefixed_id(approval_request_id, "call_")
}

/// Rewrite an MCP item id (`mcp_*` or `mcpr_*`) to an id with `prefix`,
/// preserving the stable suffix so replayed items keep their correlation.
fn mcp_item_id_to_prefixed_id(item_id: &str, prefix: &str) -> String {
    let suffix = item_id
        .strip_prefix("mcp_")
        .or_else(|| item_id.strip_prefix("mcpr_"))
        .unwrap_or(item_id);
    format!("{prefix}{suffix}")
}

/// Deserialize ResponseInputOutputItems from a JSON array value
fn deserialize_items_from_array(array: &Value) -> Vec<ResponseInputOutputItem> {
    array
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    serde_json::from_value::<ResponseInputOutputItem>(item.clone())
                        .map_err(|e| warn!("Failed to deserialize item: {}. Item: {}", e, item))
                        .ok()
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
    use openai_protocol::responses::{
        ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponsesRequest,
    };

    use super::{inject_memory_context, prepare_agent_loop_input};
    use crate::routers::common::header_utils::{
        ConversationMemoryConfig, LongTermMemoryConfig, ShortTermMemoryConfig,
    };

    /// Text input should pass through unchanged with no MCP labels recorded.
    #[test]
    fn prepare_agent_loop_input_passes_text_through() {
        let prepared = prepare_agent_loop_input(&ResponseInput::Text("hello".to_string()));
        match prepared.upstream_input {
            ResponseInput::Text(text) => assert_eq!(text, "hello"),
            ResponseInput::Items(_) => panic!("expected text input to remain text"),
        }
        assert!(prepared.existing_mcp_list_tools_labels.is_empty());
    }

    /// `mcp_call` must expand into a `function_call` + `function_call_output`
    /// pair so the upstream transcript stays LLM-consumable. The correlation
    /// id derived from `approval_request_id` (when present) or the `mcp_` id
    /// must match on both items.
    #[test]
    fn prepare_agent_loop_input_expands_mcp_call_to_function_pair() {
        let prepared = prepare_agent_loop_input(&ResponseInput::Items(vec![
            ResponseInputOutputItem::Message {
                id: "msg_1".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText {
                    text: "find something".to_string(),
                }],
                status: None,
                phase: None,
            },
            ResponseInputOutputItem::McpCall {
                id: "mcp_abc".to_string(),
                server_label: "deepwiki".to_string(),
                name: "ask_question".to_string(),
                arguments: "{\"q\":\"x\"}".to_string(),
                output: "answer".to_string(),
                status: "completed".to_string(),
                approval_request_id: None,
                error: None,
            },
        ]));

        let ResponseInput::Items(items) = prepared.upstream_input else {
            panic!("expected items upstream input");
        };
        assert_eq!(items.len(), 3, "expected message + function pair");

        match (&items[1], &items[2]) {
            (
                ResponseInputOutputItem::FunctionToolCall {
                    call_id: call_a,
                    name,
                    arguments,
                    ..
                },
                ResponseInputOutputItem::FunctionCallOutput {
                    call_id: call_b,
                    output,
                    ..
                },
            ) => {
                assert_eq!(call_a, call_b, "call_id must pair the two items");
                assert_eq!(call_a, "call_abc");
                assert_eq!(name, "ask_question");
                assert_eq!(arguments, "{\"q\":\"x\"}");
                assert_eq!(output, "answer");
            }
            other => panic!("unexpected normalized pair: {other:?}"),
        }
    }

    /// A failed replayed `mcp_call` carries its diagnostic in `error` with
    /// `output` often empty; the normalized `function_call_output` must
    /// surface that diagnostic so the upstream model sees the failure reason
    /// rather than a silently "successful" empty tool result.
    #[test]
    fn prepare_agent_loop_input_folds_mcp_call_error_into_upstream_output() {
        let prepared = prepare_agent_loop_input(&ResponseInput::Items(vec![
            ResponseInputOutputItem::Message {
                id: "msg_1".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText {
                    text: "find something".to_string(),
                }],
                status: None,
                phase: None,
            },
            ResponseInputOutputItem::McpCall {
                id: "mcp_abc".to_string(),
                server_label: "deepwiki".to_string(),
                name: "ask_question".to_string(),
                arguments: "{}".to_string(),
                output: String::new(),
                status: "failed".to_string(),
                approval_request_id: None,
                error: Some("upstream 5xx".to_string()),
            },
        ]));

        let ResponseInput::Items(items) = prepared.upstream_input else {
            panic!("expected items upstream input");
        };
        match &items[2] {
            ResponseInputOutputItem::FunctionCallOutput { output, .. } => {
                assert_eq!(output, "upstream 5xx");
            }
            other => panic!("expected FunctionCallOutput, got {other:?}"),
        }
    }

    /// When an `mcp_call` was resumed from approval, its correlation id comes
    /// from `approval_request_id` (`mcpr_*`) so later turns can still match
    /// the originating approval request against the executed call.
    #[test]
    fn prepare_agent_loop_input_reuses_approval_request_id_for_call_id() {
        let prepared = prepare_agent_loop_input(&ResponseInput::Items(vec![
            ResponseInputOutputItem::Message {
                id: "msg_1".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText {
                    text: "continue".to_string(),
                }],
                status: None,
                phase: None,
            },
            ResponseInputOutputItem::McpCall {
                id: "mcp_resumed".to_string(),
                server_label: "docs".to_string(),
                name: "search".to_string(),
                arguments: "{}".to_string(),
                output: "ok".to_string(),
                status: "completed".to_string(),
                approval_request_id: Some("mcpr_xyz".to_string()),
                error: None,
            },
        ]));

        let ResponseInput::Items(items) = prepared.upstream_input else {
            panic!("expected items upstream input");
        };
        match &items[1] {
            ResponseInputOutputItem::FunctionToolCall { call_id, .. } => {
                assert_eq!(call_id, "call_xyz");
            }
            other => panic!("expected FunctionToolCall, got {other:?}"),
        }
    }

    /// `mcp_list_tools` items are client-visible metadata, not LLM context.
    /// They must be stripped from the upstream transcript and their
    /// `server_label` recorded for dedupe on output assembly.
    #[test]
    fn prepare_agent_loop_input_collects_list_tools_labels() {
        let prepared = prepare_agent_loop_input(&ResponseInput::Items(vec![
            ResponseInputOutputItem::Message {
                id: "msg_1".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText {
                    text: "hi".to_string(),
                }],
                status: None,
                phase: None,
            },
            ResponseInputOutputItem::McpListTools {
                id: "mcpl_1".to_string(),
                server_label: "deepwiki".to_string(),
                tools: vec![],
            },
            ResponseInputOutputItem::McpListTools {
                id: "mcpl_2".to_string(),
                server_label: "deepwiki".to_string(),
                tools: vec![],
            },
            ResponseInputOutputItem::McpListTools {
                id: "mcpl_3".to_string(),
                server_label: "docs".to_string(),
                tools: vec![],
            },
        ]));

        let ResponseInput::Items(items) = prepared.upstream_input else {
            panic!("expected items upstream input");
        };
        assert_eq!(
            items.len(),
            1,
            "mcp_list_tools items must not be sent upstream"
        );
        assert_eq!(
            prepared.existing_mcp_list_tools_labels,
            vec!["deepwiki".to_string(), "docs".to_string()],
            "labels should be deduped in first-seen order",
        );
    }

    /// Approval items are passed through to the upstream transcript so OpenAI
    /// can match an `mcp_approval_response` against its originating
    /// `mcp_approval_request` and resume execution.
    #[test]
    fn prepare_agent_loop_input_preserves_approval_items_in_upstream() {
        let prepared = prepare_agent_loop_input(&ResponseInput::Items(vec![
            ResponseInputOutputItem::Message {
                id: "msg_1".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText {
                    text: "hi".to_string(),
                }],
                status: None,
                phase: None,
            },
            ResponseInputOutputItem::McpApprovalRequest {
                id: "mcpr_1".to_string(),
                server_label: "docs".to_string(),
                name: "search".to_string(),
                arguments: "{}".to_string(),
            },
            ResponseInputOutputItem::McpApprovalResponse {
                id: None,
                approval_request_id: "mcpr_1".to_string(),
                approve: true,
                reason: None,
            },
        ]));

        let ResponseInput::Items(items) = prepared.upstream_input else {
            panic!("expected items upstream input");
        };
        assert!(items
            .iter()
            .any(|item| matches!(item, ResponseInputOutputItem::McpApprovalRequest { .. })),);
        assert!(items
            .iter()
            .any(|item| matches!(item, ResponseInputOutputItem::McpApprovalResponse { .. })),);
    }

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
}
