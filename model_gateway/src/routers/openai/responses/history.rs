//! Input history loading for the Responses API.
//!
//! Loads conversation history and/or previous response chains into the request
//! input before forwarding to the upstream provider.

use axum::response::Response;
use openai_protocol::{
    event_types::ItemType,
    responses::{ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponsesRequest},
};
use serde_json::Value;
use smg_data_connector::{ConversationId, ListParams, ResponseId, ResponseStorageError, SortOrder};
use tracing::{debug, warn};

use super::super::context::ResponsesComponents;
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::{error, header_utils::ConversationMemoryConfig},
};

const MAX_CONVERSATION_HISTORY_ITEMS: usize = 100;

/// Load conversation history and/or previous response chain into request input.
///
/// Mutates `request_body.input` with the loaded items.
/// Returns `Ok(original_previous_response_id)` on success, or `Err(response)` on validation failure.
pub(crate) async fn load_input_history(
    components: &ResponsesComponents,
    conversation: Option<&str>,
    request_body: &mut ResponsesRequest,
    model: &str,
) -> Result<Option<String>, Response> {
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
                            match serde_json::from_value::<Vec<ResponseContentPart>>(item.content) {
                                Ok(content_parts) => {
                                    items.push(ResponseInputOutputItem::Message {
                                        id: item.id.0.clone(),
                                        role: item
                                            .role
                                            .clone()
                                            .unwrap_or_else(|| "user".to_string()),
                                        content: content_parts,
                                        status: item.status.clone(),
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

    Ok(previous_response_id)
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
            });
        }
        ResponseInput::Items(current_items) => {
            for item in current_items {
                items.push(openai_protocol::responses::normalize_input_item(item));
            }
        }
    }
}

// ── Memory context injection ──────────────────────────────────────────────────

/// Inject retrieved memories into the request body before dispatching upstream.
///
/// Reads [`ConversationMemoryConfig`] parsed from the incoming headers and
/// prepends a system message containing the relevant memories so the model
/// has context for the current turn.
///
/// # Current status
///
/// This is a **stub**. The config is read and logged so that the call-site
/// contract is established, but no retrieval is performed yet.
///
/// # TODO
/// - Wire a `ConversationMemoryReader` through the storage layer.
/// - On `ltm_enabled`: retrieve long-term memories for `subject_id` and
///   prepend as a system message.
/// - On `stm_enabled`: retrieve the short-term memory summary (STMO output)
///   and prepend as a system message.
pub(crate) async fn inject_memory_context(
    config: &ConversationMemoryConfig,
    _request_body: &mut ResponsesRequest,
) {
    if config.ltm_enabled {
        // TODO: retrieve long-term memories and prepend as system message
        debug!(
            subject_id = config.ltm_subject_id.as_deref().unwrap_or("<none>"),
            embedding_model = config.ltm_embedding_model_id.as_deref().unwrap_or("<none>"),
            extraction_model = config
                .ltm_extraction_model_id
                .as_deref()
                .unwrap_or("<none>"),
            "LTM recall requested — retrieval not yet implemented"
        );
    }

    if config.stm_enabled {
        // TODO: retrieve STMO condensation summary and prepend as system message
        debug!(
            condenser_model = config.stm_condenser_model_id.as_deref().unwrap_or("<none>"),
            "STMO recall requested — retrieval not yet implemented"
        );
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::responses::{ResponseInput, ResponsesRequest};

    use super::inject_memory_context;
    use crate::routers::header_utils::ConversationMemoryConfig;

    #[tokio::test]
    async fn inject_memory_context_is_no_op_for_now() {
        let config = ConversationMemoryConfig {
            ltm_enabled: true,
            ltm_subject_id: Some("subj-1".to_string()),
            ltm_embedding_model_id: Some("embed-1".to_string()),
            ltm_extraction_model_id: Some("extract-1".to_string()),
            stm_enabled: true,
            stm_condenser_model_id: Some("condense-1".to_string()),
        };
        let mut request = ResponsesRequest {
            input: ResponseInput::Text("hello".to_string()),
            ..Default::default()
        };

        inject_memory_context(&config, &mut request).await;

        assert!(matches!(request.input, ResponseInput::Text(_)));
    }
}
