//! Shared history hydration for Responses API surfaces.

use std::{collections::HashSet, sync::Arc};

use axum::response::Response;
use openai_protocol::{
    event_types::ItemType,
    responses::{
        self, ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponsesRequest,
    },
};
use serde_json::{from_value, Value};
use smg_data_connector::{
    ConversationId, ConversationItemStorage, ConversationStorage, ListParams, ResponseId,
    ResponseStorage, ResponseStorageError, SortOrder,
};
use tracing::{debug, warn};

use crate::routers::{
    common::{
        agent_loop::PreparedLoopInput,
        persistence_utils::split_stored_message_content,
        transcript_lower::{
            extract_control_items, extract_mcp_list_tools_server_labels, lower_transcript,
        },
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

/// A Responses request after history/conversation hydration and current-input assembly.
pub(crate) struct PreparedRequestHistory {
    pub request: ResponsesRequest,
    pub prepared: PreparedLoopInput,
    pub existing_mcp_list_tools_labels: HashSet<String>,
}

enum HistorySource<'a> {
    PreviousResponse(&'a str),
    Conversation(&'a str),
    Conflict,
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

/// Resolve stored history and merge it with the current request input.
pub(crate) async fn prepare_request_history(
    response_storage: &Arc<dyn ResponseStorage>,
    conversation_storage: &Arc<dyn ConversationStorage>,
    conversation_item_storage: &Arc<dyn ConversationItemStorage>,
    request: &ResponsesRequest,
) -> Result<PreparedRequestHistory, Response> {
    let loaded = load_request_history(
        response_storage,
        conversation_storage,
        conversation_item_storage,
        request,
    )
    .await?;

    Ok(assemble_request_history(request, loaded))
}

fn assemble_request_history(
    request: &ResponsesRequest,
    loaded: LoadedHistory,
) -> PreparedRequestHistory {
    let LoadedHistory {
        items,
        mut existing_mcp_list_tools_labels,
        mut control_items,
    } = loaded;

    let mut modified_request = request.clone();
    let mut combined = items;
    append_current_input(
        &mut combined,
        &mut existing_mcp_list_tools_labels,
        &mut control_items,
        &modified_request.input,
    );

    modified_request.previous_response_id = None;
    modified_request.input = ResponseInput::Items(lower_transcript(combined));
    modified_request.conversation = None;

    PreparedRequestHistory {
        prepared: PreparedLoopInput::new(modified_request.input.clone(), control_items),
        request: modified_request,
        existing_mcp_list_tools_labels,
    }
}

fn append_current_input(
    combined: &mut Vec<ResponseInputOutputItem>,
    existing_mcp_list_tools_labels: &mut HashSet<String>,
    control_items: &mut Vec<ResponseInputOutputItem>,
    input: &ResponseInput,
) {
    match input {
        ResponseInput::Items(items) => {
            existing_mcp_list_tools_labels.extend(extract_mcp_list_tools_server_labels(items));
            control_items.extend(extract_control_items(items));
            combined.extend(items.iter().map(responses::normalize_input_item));
        }
        ResponseInput::Text(text) => combined.push(current_text_item(text)),
    }
}

fn current_text_item(text: &str) -> ResponseInputOutputItem {
    ResponseInputOutputItem::Message {
        id: format!("msg_u_{}", uuid::Uuid::now_v7()),
        role: "user".to_string(),
        content: vec![ResponseContentPart::InputText {
            text: text.to_string(),
        }],
        status: Some("completed".to_string()),
        phase: None,
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
    match request_history_source(request) {
        Some(HistorySource::PreviousResponse(prev_id)) => {
            load_previous_response_history(response_storage, prev_id).await
        }
        Some(HistorySource::Conversation(conversation_id)) => {
            load_conversation_history(
                conversation_storage,
                conversation_item_storage,
                conversation_id,
            )
            .await
        }
        Some(HistorySource::Conflict) => Err(error::bad_request(
            "invalid_request",
            "Cannot specify both 'conversation' and 'previous_response_id'".to_string(),
        )),
        None => Ok(LoadedHistory::empty()),
    }
}

fn request_history_source(request: &ResponsesRequest) -> Option<HistorySource<'_>> {
    let previous_response_id = request
        .previous_response_id
        .as_deref()
        .filter(|s| !s.is_empty());
    let conversation = request.conversation.as_ref().filter(|c| !c.is_empty());

    if previous_response_id.is_some() && conversation.is_some() {
        return Some(HistorySource::Conflict);
    }

    previous_response_id
        .map(HistorySource::PreviousResponse)
        .or_else(|| conversation.map(|c| HistorySource::Conversation(c.as_id())))
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
            | ItemType::WEB_SEARCH_CALL
            | ItemType::CODE_INTERPRETER_CALL
            | ItemType::FILE_SEARCH_CALL
            | ItemType::IMAGE_GENERATION_CALL
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

#[cfg(test)]
mod tests {
    use openai_protocol::common::ConversationRef;

    use super::*;

    fn text_request(text: &str) -> ResponsesRequest {
        ResponsesRequest {
            input: ResponseInput::Text(text.to_string()),
            previous_response_id: Some("resp_prev".to_string()),
            ..Default::default()
        }
    }

    fn items_request(items: Vec<ResponseInputOutputItem>) -> ResponsesRequest {
        ResponsesRequest {
            input: ResponseInput::Items(items),
            ..Default::default()
        }
    }

    #[test]
    fn history_source_rejects_previous_response_and_conversation_together() {
        let request = ResponsesRequest {
            input: ResponseInput::Text("hello".to_string()),
            previous_response_id: Some("resp_prev".to_string()),
            conversation: Some(ConversationRef::Id("conv_1".to_string())),
            ..Default::default()
        };

        assert!(matches!(
            request_history_source(&request),
            Some(HistorySource::Conflict)
        ));
    }

    #[test]
    fn history_source_treats_empty_conversation_as_unset() {
        let request = ResponsesRequest {
            input: ResponseInput::Text("hello".to_string()),
            conversation: Some(ConversationRef::Id(String::new())),
            ..Default::default()
        };

        assert!(request_history_source(&request).is_none());
    }

    #[test]
    fn history_source_uses_conversation_when_previous_response_is_empty() {
        let request = ResponsesRequest {
            input: ResponseInput::Text("hello".to_string()),
            previous_response_id: Some(String::new()),
            conversation: Some(ConversationRef::Object {
                id: "conv_1".to_string(),
            }),
            ..Default::default()
        };

        let source = request_history_source(&request);

        match source {
            Some(HistorySource::Conversation(id)) => assert_eq!(id, "conv_1"),
            _ => panic!("expected conversation history source"),
        }
    }

    #[test]
    fn assemble_text_input_builds_message_and_clears_history_source() {
        let prepared = assemble_request_history(&text_request("hello"), LoadedHistory::empty());

        assert!(prepared.request.previous_response_id.is_none());
        assert!(prepared.request.conversation.is_none());

        let ResponseInput::Items(items) = &prepared.prepared.upstream_input else {
            panic!("prepared input should be itemized");
        };
        assert_eq!(items.len(), 1);
        match &items[0] {
            ResponseInputOutputItem::Message { role, content, .. } => {
                assert_eq!(role, "user");
                assert!(matches!(
                    content.as_slice(),
                    [ResponseContentPart::InputText { text }] if text == "hello"
                ));
            }
            other => panic!("expected structured message, got {other:?}"),
        }
    }

    #[test]
    fn assemble_item_input_normalizes_simple_messages() {
        let prepared = assemble_request_history(
            &items_request(vec![ResponseInputOutputItem::SimpleInputMessage {
                content: responses::StringOrContentParts::String("hello".to_string()),
                role: "user".to_string(),
                r#type: None,
                phase: None,
            }]),
            LoadedHistory::empty(),
        );

        let ResponseInput::Items(items) = &prepared.prepared.upstream_input else {
            panic!("prepared input should be itemized");
        };
        assert_eq!(items.len(), 1);
        match &items[0] {
            ResponseInputOutputItem::Message { role, content, .. } => {
                assert_eq!(role, "user");
                assert!(matches!(
                    content.as_slice(),
                    [ResponseContentPart::InputText { text }] if text == "hello"
                ));
            }
            other => panic!("expected normalized message, got {other:?}"),
        }
    }
}
