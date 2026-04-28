//! Shared history hydration for Responses API surfaces.

use std::{collections::HashSet, sync::Arc};

use axum::response::Response;
use openai_protocol::{
    event_types::ItemType,
    responses::{ResponseContentPart, ResponseInputOutputItem, ResponsesRequest},
};
use serde_json::{from_value, Value};
use smg_data_connector::{
    ConversationId, ConversationItemStorage, ConversationStorage, ListParams, ResponseId,
    ResponseStorage, ResponseStorageError, SortOrder,
};
use tracing::{debug, warn};

use crate::routers::{
    common::{
        persistence_utils::split_stored_message_content,
        transcript_lower::{extract_control_items, lower_transcript},
    },
    error,
};

const MAX_CONVERSATION_HISTORY_ITEMS: usize = 100;

/// History items + per-chain dedupe metadata for one `/v1/responses`
/// request. `items` are already lowered to the core item set; callers
/// should splice them in front of the current request input.
pub(crate) struct LoadedHistory {
    pub items: Vec<ResponseInputOutputItem>,
    pub existing_mcp_list_tools_labels: HashSet<String>,
    pub control_items: Vec<ResponseInputOutputItem>,
}

impl LoadedHistory {
    pub(crate) fn empty() -> Self {
        Self {
            items: Vec::new(),
            existing_mcp_list_tools_labels: HashSet::new(),
            control_items: Vec::new(),
        }
    }
}

/// Resolve the request's history from `previous_response_id` or
/// `conversation` (whichever is set), lower it, and return the result.
pub(crate) async fn load_request_history(
    response_storage: &Arc<dyn ResponseStorage>,
    conversation_storage: &Arc<dyn ConversationStorage>,
    conversation_item_storage: &Arc<dyn ConversationItemStorage>,
    request: &ResponsesRequest,
) -> Result<LoadedHistory, Response> {
    if let Some(prev_id) = request
        .previous_response_id
        .as_deref()
        .filter(|s| !s.is_empty())
    {
        return load_previous_response_history(response_storage, prev_id).await;
    }
    if let Some(conv_ref) = request.conversation.as_ref() {
        return load_conversation_history(
            conversation_storage,
            conversation_item_storage,
            conv_ref.as_id(),
        )
        .await;
    }
    Ok(LoadedHistory::empty())
}

async fn load_previous_response_history(
    response_storage: &Arc<dyn ResponseStorage>,
    prev_id_str: &str,
) -> Result<LoadedHistory, Response> {
    let prev_id = ResponseId::from(prev_id_str);

    let chain = match response_storage.get_response_chain(&prev_id, None).await {
        Ok(chain) => chain,
        Err(ResponseStorageError::ResponseNotFound(_)) => {
            return Err(error::bad_request(
                "previous_response_not_found",
                format!("Previous response with id '{prev_id_str}' not found."),
            ));
        }
        Err(e) => {
            return Err(error::internal_error(
                "load_previous_response_chain_failed",
                format!("Failed to load previous response chain for {prev_id_str}: {e}"),
            ));
        }
    };

    if chain.responses.is_empty() {
        return Err(error::bad_request(
            "previous_response_not_found",
            format!("Previous response with id '{prev_id_str}' not found."),
        ));
    }

    let mut items = Vec::new();
    let mut existing_mcp_list_tools_labels = HashSet::new();

    for stored in &chain.responses {
        existing_mcp_list_tools_labels.extend(extract_mcp_list_tools_labels(
            stored.raw_response.get("output").unwrap_or(&Value::Null),
        ));

        extend_with_deserialized_items(&mut items, &stored.input, "input");
        extend_with_deserialized_items(
            &mut items,
            stored.raw_response.get("output").unwrap_or(&Value::Null),
            "output",
        );
    }

    let control_items = extract_control_items(&items);
    let items = lower_transcript(items);
    debug!(
        previous_response_id = %prev_id_str,
        history_items_count = items.len(),
        mcp_list_tools_labels_count = existing_mcp_list_tools_labels.len(),
        control_items_count = control_items.len(),
        "Loaded previous response history"
    );

    Ok(LoadedHistory {
        items,
        existing_mcp_list_tools_labels,
        control_items,
    })
}

async fn load_conversation_history(
    conversation_storage: &Arc<dyn ConversationStorage>,
    conversation_item_storage: &Arc<dyn ConversationItemStorage>,
    conv_id_str: &str,
) -> Result<LoadedHistory, Response> {
    let conv_id = ConversationId::from(conv_id_str);

    let conversation = conversation_storage
        .get_conversation(&conv_id)
        .await
        .map_err(|e| {
            error::internal_error(
                "check_conversation_failed",
                format!("Failed to check conversation: {e}"),
            )
        })?;

    if conversation.is_none() {
        return Err(error::not_found(
            "conversation_not_found",
            format!(
                "Conversation '{conv_id_str}' not found. Please create the conversation first using the conversations API."
            ),
        ));
    }

    let params = ListParams {
        limit: MAX_CONVERSATION_HISTORY_ITEMS,
        order: SortOrder::Asc,
        after: None,
    };

    let stored_items = match conversation_item_storage.list_items(&conv_id, params).await {
        Ok(items) => items,
        Err(e) => {
            warn!("Failed to load conversation history: {}", e);
            return Ok(LoadedHistory::empty());
        }
    };

    let mut items: Vec<ResponseInputOutputItem> = Vec::new();
    let mut existing_mcp_list_tools_labels = HashSet::new();

    for item in stored_items {
        match item.item_type.as_str() {
            "message" => {
                let (content_value, stored_phase) = split_stored_message_content(item.content);
                match from_value::<Vec<ResponseContentPart>>(content_value) {
                    Ok(content_parts) => {
                        items.push(ResponseInputOutputItem::Message {
                            id: item.id.0.clone(),
                            role: item.role.clone().unwrap_or_else(|| "user".to_string()),
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
            "reasoning"
            | ItemType::FUNCTION_CALL
            | ItemType::FUNCTION_CALL_OUTPUT
            | "mcp_call"
            | "mcp_list_tools"
            | "mcp_approval_request"
            | "mcp_approval_response" => {
                if item.item_type == "mcp_list_tools" {
                    if let Some(label) = item.content.get("server_label").and_then(|v| v.as_str()) {
                        existing_mcp_list_tools_labels.insert(label.to_string());
                    }
                }
                match from_value::<ResponseInputOutputItem>(item.content) {
                    Ok(parsed) => items.push(parsed),
                    Err(e) => {
                        warn!(
                            "Failed to deserialize conversation item type='{}': {}",
                            item.item_type, e
                        );
                    }
                }
            }
            other => {
                warn!("Unknown item type in conversation: {}", other);
            }
        }
    }

    let control_items = extract_control_items(&items);
    let items = lower_transcript(items);
    debug!(
        conversation = %conv_id_str,
        history_items_count = items.len(),
        mcp_list_tools_labels_count = existing_mcp_list_tools_labels.len(),
        control_items_count = control_items.len(),
        "Loaded conversation history"
    );

    Ok(LoadedHistory {
        items,
        existing_mcp_list_tools_labels,
        control_items,
    })
}

fn extend_with_deserialized_items(
    out: &mut Vec<ResponseInputOutputItem>,
    value: &Value,
    kind: &str,
) {
    let Some(arr) = value.as_array() else {
        return;
    };
    for item in arr {
        match from_value::<ResponseInputOutputItem>(item.clone()) {
            Ok(parsed) => out.push(parsed),
            Err(e) => {
                let item_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("<no-id>");
                let item_type = item
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("<no-type>");
                warn!(
                    "Failed to deserialize stored {} item (id={}, type={}): {}",
                    kind, item_id, item_type, e,
                );
            }
        }
    }
}

fn extract_mcp_list_tools_labels(output: &Value) -> Vec<String> {
    output
        .as_array()
        .map(|items| {
            items
                .iter()
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
