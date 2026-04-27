//! Shared history hydration for /v1/responses gRPC routers.
//!
//! Regular and harmony both need the same canonical-input contract on
//! the way into a model turn:
//!
//! 1. resolve `previous_response_id` **or** `conversation` (mutually
//!    exclusive — validated upstream by `ResponsesRequest::validate`)
//!    into the prior history items
//! 2. lower the high-level item types (`mcp_call`, `mcp_list_tools`,
//!    image-gen, …) to the core item set every backend speaks
//! 3. extract the set of MCP server labels that already have a
//!    `mcp_list_tools` item upstream so the agent loop can skip
//!    re-emitting them
//!
//! All three steps live behind one entry point ([`load_request_history`])
//! so adapters never see — and can never forget — any of them.

use std::collections::HashSet;

use axum::response::Response;
use openai_protocol::{
    event_types::ItemType,
    responses::{ResponseContentPart, ResponseInputOutputItem, ResponsesRequest},
};
use serde_json::{from_value, Value};
use smg_data_connector::{ConversationId, ListParams, ResponseId, ResponseStorageError, SortOrder};
use tracing::{debug, error, warn};

use crate::routers::{
    common::{
        mcp_utils::extract_mcp_list_tools_labels,
        persistence_utils::split_stored_message_content,
        transcript_lower::{extract_control_items, lower_transcript},
    },
    error,
    grpc::common::responses::ResponsesContext,
};

const MAX_CONVERSATION_HISTORY_ITEMS: usize = 100;

/// History items + per-chain dedupe metadata for one /v1/responses
/// request. `items` are already lowered to the core item set; callers
/// should splice them straight in front of the request's input.
/// `control_items` retains the loop's control vocabulary
/// (`mcp_list_tools`, `mcp_approval_request`, `mcp_approval_response`)
/// in source order so the driver entry can validate continuation
/// semantics without re-scanning the lowered transcript.
pub(crate) struct LoadedHistory {
    pub items: Vec<ResponseInputOutputItem>,
    pub existing_mcp_list_tools_labels: HashSet<String>,
    pub control_items: Vec<ResponseInputOutputItem>,
}

impl LoadedHistory {
    fn empty() -> Self {
        Self {
            items: Vec::new(),
            existing_mcp_list_tools_labels: HashSet::new(),
            control_items: Vec::new(),
        }
    }
}

/// Resolve the request's history from `previous_response_id` or
/// `conversation` (whichever is set), lower it, and return the result.
///
/// Returns an empty `LoadedHistory` if neither field is set. Returns an
/// axum `Response` error when the chain / conversation cannot be
/// loaded — callers bubble it up as the HTTP error directly.
pub(crate) async fn load_request_history(
    ctx: &ResponsesContext,
    request: &ResponsesRequest,
) -> Result<LoadedHistory, Response> {
    if let Some(prev_id) = request
        .previous_response_id
        .as_deref()
        .filter(|s| !s.is_empty())
    {
        return load_previous_response_history(ctx, prev_id).await;
    }
    if let Some(conv_ref) = request.conversation.as_ref() {
        return load_conversation_history(ctx, conv_ref.as_id()).await;
    }
    Ok(LoadedHistory::empty())
}

/// Load the response chain for `previous_response_id`, flatten its
/// stored `input` + `output` arrays, lower the result, and collect
/// already-emitted `mcp_list_tools` labels.
pub(crate) async fn load_previous_response_history(
    ctx: &ResponsesContext,
    prev_id_str: &str,
) -> Result<LoadedHistory, Response> {
    let prev_id = ResponseId::from(prev_id_str);

    let chain = match ctx
        .response_storage
        .get_response_chain(&prev_id, None)
        .await
    {
        Ok(chain) => chain,
        Err(ResponseStorageError::ResponseNotFound(_)) => {
            return Err(error::bad_request(
                "previous_response_not_found",
                format!("Previous response with id '{prev_id_str}' not found."),
            ));
        }
        Err(e) => {
            error!(
                function = "load_previous_response_history",
                prev_id = %prev_id_str,
                error = %e,
                "Failed to load previous response chain from storage"
            );
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

    // Pull control items out *before* lowering — the lowering pass
    // drops them, but the driver entry needs them to validate
    // continuation semantics (matched approval pairs, replayed
    // listings, etc.).
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

/// Load conversation items for `conversation_id`, lower them, and
/// collect any `mcp_list_tools` labels already present.
async fn load_conversation_history(
    ctx: &ResponsesContext,
    conv_id_str: &str,
) -> Result<LoadedHistory, Response> {
    let conv_id = ConversationId::from(conv_id_str);

    let conversation = ctx
        .conversation_storage
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

    let stored_items = match ctx
        .conversation_item_storage
        .list_items(&conv_id, params)
        .await
    {
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
                // Conversation storage round-trips these as raw JSON.
                // Deserialize back into the input-item enum and let the
                // lowering pass below project the ones that need it.
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
                warn!(
                    "Failed to deserialize stored {} item: {}. Item: {}",
                    kind, e, item
                );
            }
        }
    }
}
