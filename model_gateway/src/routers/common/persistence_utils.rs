//! Utilities for persisting responses and conversation items across router implementations.

use std::sync::Arc;

use chrono::Utc;
use openai_protocol::responses::{
    generate_id, MessagePhase, ResponseInput, ResponseInputOutputItem, ResponsesRequest,
    StringOrContentParts,
};
use serde_json::{json, Value};
use smg_data_connector::{
    with_request_context, ConversationId, ConversationItem, ConversationItemId,
    ConversationItemStorage, ConversationMemoryStatus, ConversationMemoryType,
    ConversationMemoryWriter, ConversationStorage, NewConversationItem, NewConversationMemory,
    RequestContext as StorageRequestContext, ResponseId, ResponseStorage, StoredResponse,
};
use tracing::{debug, info, warn};

use crate::memory::MemoryExecutionContext;

// ============================================================================
// Constants
// ============================================================================

/// Field mappings for item types that store data in content
pub const ITEM_TYPE_FIELDS: &[(&str, &[&str])] = &[
    (
        "mcp_call",
        &[
            "name",
            "arguments",
            "output",
            "server_label",
            "approval_request_id",
            "error",
        ],
    ),
    // T11: `error` is optional per spec (openai-responses-api-spec.md §McpListTools
    // L253-255) and symmetric with `mcp_call.error` above; keep it in the
    // persistence whitelist so failed list-tools items round-trip through
    // conversation storage without losing the failure reason.
    ("mcp_list_tools", &["tools", "server_label", "error"]),
    ("function_call", &["call_id", "name", "arguments", "output"]),
    ("function_call_output", &["call_id", "output"]),
];

// ============================================================================
// JSON Serialization
// ============================================================================

/// Convert a ConversationItem to JSON, extracting specified fields based on item type
/// or including content as-is for standard message types.
pub fn item_to_json(item: &ConversationItem) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("id".to_string(), json!(item.id.0));
    obj.insert("type".to_string(), json!(item.item_type));

    if let Some(role) = &item.role {
        obj.insert("role".to_string(), json!(role));
    }

    // Find field mappings for this item type
    let fields = ITEM_TYPE_FIELDS
        .iter()
        .find(|(t, _)| *t == item.item_type)
        .map(|(_, fields)| *fields);

    if let Some(fields) = fields {
        // Extract specific fields from content
        if let Some(content_obj) = item.content.as_object() {
            for field in fields {
                if let Some(value) = content_obj.get(*field) {
                    obj.insert((*field).to_string(), value.clone());
                }
            }
        }
    } else if item.item_type == "message" {
        // Message items may store either a bare content array (legacy) or an
        // object `{content: [...], phase: "..."}` when the message carried a
        // phase label (P3). Either way, expose `content` and hoist `phase`
        // back to the message level so round-trip is transparent to callers.
        let (content_value, phase) = split_stored_message_content(item.content.clone());
        obj.insert("content".to_string(), content_value);
        if let Some(phase) = phase {
            if let Ok(phase_value) = serde_json::to_value(phase) {
                obj.insert("phase".to_string(), phase_value);
            }
        }
    } else {
        // Default: include content as-is
        obj.insert("content".to_string(), item.content.clone());
    }

    if let Some(status) = &item.status {
        obj.insert("status".to_string(), json!(status));
    }

    Value::Object(obj)
}

/// Split stored message content into (content_parts_value, phase).
///
/// Legacy messages were stored with their `content` field directly (an array of
/// content parts). When a message carries a `phase` label (P3), persistence
/// wraps it as `{"content": [...], "phase": "..."}`. This helper accepts both
/// shapes so callers can round-trip phase without breaking existing rows.
pub fn split_stored_message_content(raw: Value) -> (Value, Option<MessagePhase>) {
    if let Value::Object(mut map) = raw {
        // Only treat objects with an explicit `content` key as the wrapped
        // shape; any other object is either malformed or a future extension
        // and is returned unchanged so the caller's error path surfaces it.
        if map.contains_key("content") {
            let content = map.remove("content").unwrap_or(Value::Array(Vec::new()));
            let phase = map
                .get("phase")
                .cloned()
                .and_then(|v| serde_json::from_value::<MessagePhase>(v).ok());
            return (content, phase);
        }
        return (Value::Object(map), None);
    }
    (raw, None)
}

// ============================================================================
// Item Creation Helper
// ============================================================================

/// Create a conversation item and optionally link it to a conversation.
/// Sets default "completed" status if not provided.
pub async fn create_and_link_item(
    item_storage: &Arc<dyn ConversationItemStorage>,
    conv_id_opt: Option<&ConversationId>,
    mut new_item: NewConversationItem,
) -> Result<(), String> {
    if new_item.status.is_none() {
        new_item.status = Some("completed".to_string());
    }

    let created = item_storage
        .create_item(new_item)
        .await
        .map_err(|e| format!("Failed to create item: {e}"))?;

    if let Some(conv_id) = conv_id_opt {
        item_storage
            .link_item(conv_id, &created.id, Utc::now())
            .await
            .map_err(|e| format!("Failed to link item: {e}"))?;

        debug!(
            conversation_id = %conv_id.0,
            item_id = %created.id.0,
            item_type = %created.item_type,
            "Persisted conversation item and link"
        );
    } else {
        debug!(
            item_id = %created.id.0,
            item_type = %created.item_type,
            "Persisted conversation item (no conversation link)"
        );
    }

    Ok(())
}

// ============================================================================
// Response Persistence
// ============================================================================

/// Extract a string field from JSON, returning owned String
fn get_string(json: &Value, key: &str) -> Option<String> {
    json.get(key).and_then(|v| v.as_str()).map(String::from)
}

/// Build a StoredResponse from response JSON and original request
pub fn build_stored_response(
    response_json: &Value,
    original_body: &ResponsesRequest,
) -> StoredResponse {
    let mut stored = StoredResponse::new(None);

    // Initialize empty array - will be populated by persist_conversation_items
    stored.input = Value::Array(vec![]);

    stored.model = get_string(response_json, "model").or_else(|| Some(original_body.model.clone()));

    stored.safety_identifier.clone_from(&original_body.user);
    // `StoredResponse.conversation_id` is `Option<String>`; flatten the
    // request's `Option<ConversationRef>` union down to its underlying id.
    stored.conversation_id = original_body
        .conversation
        .as_ref()
        .map(|c| c.as_id().to_string());

    stored.previous_response_id = get_string(response_json, "previous_response_id")
        .map(|s| ResponseId::from(s.as_str()))
        .or_else(|| {
            original_body
                .previous_response_id
                .as_deref()
                .map(ResponseId::from)
        });

    if let Some(id_str) = get_string(response_json, "id") {
        stored.id = ResponseId::from(id_str.as_str());
    }

    stored.raw_response = response_json.clone();
    stored
}

/// Extract and normalize input items from ResponseInput
fn extract_input_items(input: &ResponseInput) -> Result<Vec<Value>, String> {
    let items = match input {
        ResponseInput::Text(text) => {
            // Convert simple text to message item
            vec![json!({
                "id": generate_id("msg"),
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
                "status": "completed"
            })]
        }
        ResponseInput::Items(items) => {
            // Process all item types and ensure IDs
            items
                .iter()
                .map(|item| {
                    match item {
                        ResponseInputOutputItem::SimpleInputMessage {
                            content,
                            role,
                            phase,
                            ..
                        } => {
                            // Convert SimpleInputMessage to standard message format with ID
                            let content_json = match content {
                                StringOrContentParts::String(s) => {
                                    json!([{"type": "input_text", "text": s}])
                                }
                                StringOrContentParts::Array(parts) => {
                                    serde_json::to_value(parts)
                                        .map_err(|e| format!("Failed to serialize content: {e}"))?
                                }
                            };

                            let mut msg = json!({
                                "id": generate_id("msg"),
                                "type": "message",
                                "role": role,
                                "content": content_json,
                                "status": "completed"
                            });
                            // Preserve phase so it round-trips through
                            // conversation storage (P3).
                            if let Some(phase) = phase {
                                if let (Some(obj), Ok(phase_val)) =
                                    (msg.as_object_mut(), serde_json::to_value(phase))
                                {
                                    obj.insert("phase".to_string(), phase_val);
                                }
                            }
                            Ok(msg)
                        }
                        _ => {
                            // For other item types, serialize and ensure ID
                            let mut value = serde_json::to_value(item)
                                .map_err(|e| format!("Failed to serialize item: {e}"))?;

                            // Ensure ID exists - generate if missing
                            if let Some(obj) = value.as_object_mut() {
                                if !obj.contains_key("id")
                                    || obj
                                        .get("id")
                                        .and_then(|v| v.as_str())
                                        .map(|s| s.is_empty())
                                        .unwrap_or(true)
                                {
                                    // Generate ID with appropriate prefix based on type
                                    let item_type =
                                        obj.get("type").and_then(|v| v.as_str()).unwrap_or("item");
                                    let prefix = match item_type {
                                        "function_call" | "function_call_output" => "fc",
                                        "message" => "msg",
                                        _ => "item",
                                    };
                                    obj.insert("id".to_string(), json!(generate_id(prefix)));
                                }
                            }

                            Ok(value)
                        }
                    }
                })
                .collect::<Result<Vec<_>, String>>()?
        }
    };

    Ok(items)
}

/// Convert a JSON item to NewConversationItem
///
/// For input items: function_call/function_call_output store whole item as content
/// For output items: message extracts content field, others store whole item
fn item_to_new_conversation_item(
    item_value: &Value,
    response_id: Option<String>,
    is_input: bool,
) -> NewConversationItem {
    let item_type = item_value
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("message");

    // Determine if we should store the whole item or just the content field
    let store_whole_item = if is_input {
        item_type == "function_call" || item_type == "function_call_output"
    } else {
        item_type != "message"
    };

    let content = if store_whole_item {
        item_value.clone()
    } else if item_type == "message" && item_value.get("phase").is_some_and(|v| !v.is_null()) {
        // Message carries a phase label: wrap the content array alongside
        // `phase` so multi-turn retrieval preserves it (P3). `item_to_json`
        // and the history load paths both recognize this shape.
        let content_value = item_value.get("content").cloned().unwrap_or(json!([]));
        let phase_value = item_value.get("phase").cloned().unwrap_or(Value::Null);
        json!({ "content": content_value, "phase": phase_value })
    } else {
        item_value.get("content").cloned().unwrap_or(json!([]))
    };

    NewConversationItem {
        id: item_value
            .get("id")
            .and_then(|v| v.as_str())
            .map(ConversationItemId::from),
        response_id,
        item_type: item_type.to_string(),
        role: item_value
            .get("role")
            .and_then(|v| v.as_str())
            .map(String::from),
        content,
        status: item_value
            .get("status")
            .and_then(|v| v.as_str())
            .map(String::from),
    }
}

/// Link all input and output items to a conversation
async fn link_items_to_conversation(
    item_storage: &Arc<dyn ConversationItemStorage>,
    conv_id: &ConversationId,
    input_items: &[Value],
    output_items: &[Value],
    response_id: &str,
) -> Result<(), String> {
    let response_id_opt = Some(response_id.to_string());

    for item in input_items {
        let new_item = item_to_new_conversation_item(item, response_id_opt.clone(), true);
        create_and_link_item(item_storage, Some(conv_id), new_item).await?;
    }

    for item in output_items {
        let new_item = item_to_new_conversation_item(item, response_id_opt.clone(), false);
        create_and_link_item(item_storage, Some(conv_id), new_item).await?;
    }

    Ok(())
}

#[expect(
    clippy::manual_is_multiple_of,
    reason = "usize::is_multiple_of is not stable; % remainder check is the portable equivalent"
)]
fn should_enqueue_stmo(user_turns: usize) -> bool {
    user_turns >= 4 && (user_turns - 1) % 3 == 0
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConversationTurnInfo {
    pub user_turns: usize,
    pub total_items: usize,
}

pub fn count_conversation_turn_info(input: &ResponseInput) -> ConversationTurnInfo {
    match input {
        ResponseInput::Text(_) => ConversationTurnInfo {
            user_turns: 1,
            total_items: 1,
        },
        ResponseInput::Items(items) => {
            let user_turns = items
                .iter()
                .filter(|item| match item {
                    ResponseInputOutputItem::SimpleInputMessage { role, .. } => {
                        role.eq_ignore_ascii_case("user")
                    }
                    ResponseInputOutputItem::Message { role, .. } => {
                        role.eq_ignore_ascii_case("user")
                    }
                    _ => false,
                })
                .count();
            ConversationTurnInfo {
                user_turns,
                total_items: items.len(),
            }
        }
    }
}

async fn maybe_schedule_stmo_after_persist(
    conversation_memory_writer: &Arc<dyn ConversationMemoryWriter>,
    memory_execution_context: &MemoryExecutionContext,
    conversation_id: &ConversationId,
    response_id: &ResponseId,
    user_turns: usize,
    total_items: usize,
) -> Result<bool, String> {
    if !memory_execution_context.stm_enabled {
        return Ok(false);
    }

    if !should_enqueue_stmo(user_turns) {
        return Ok(false);
    }

    let mut memory_config = json!({
        "last_index": user_turns,
        "target_item_end": total_items,
    });

    if let Some(model_id) = &memory_execution_context.stm_condenser_model_id {
        memory_config["condenser_model"] = json!(model_id);
    }

    let memory_config = serde_json::to_string(&memory_config)
        .map_err(|e| format!("Failed to serialize STMO memory config: {e}"))?;

    let input = NewConversationMemory {
        conversation_id: conversation_id.clone(),
        conversation_version: None,
        response_id: Some(response_id.clone()),
        memory_type: ConversationMemoryType::Stmo,
        status: ConversationMemoryStatus::Ready,
        attempt: 0,
        owner_id: None,
        next_run_at: Utc::now(),
        lease_until: None,
        content: None,
        memory_config: Some(memory_config),
        scope_id: None,
        error_msg: None,
    };

    conversation_memory_writer
        .create_memory(input)
        .await
        .map_err(|e| format!("Failed to enqueue STMO memory: {e}"))?;

    Ok(true)
}

async fn handle_stmo_after_persist(
    conversation_memory_writer: &Arc<dyn ConversationMemoryWriter>,
    memory_execution_context: &MemoryExecutionContext,
    conversation_id: &ConversationId,
    response_id: &ResponseId,
    conversation_turn_info: Option<ConversationTurnInfo>,
    output_item_count: usize,
) {
    if !memory_execution_context.stm_enabled {
        return;
    }

    let Some(turn_info) = conversation_turn_info else {
        debug!(
            conversation_id = %conversation_id.0,
            response_id = %response_id.0,
            "STMO skipped: missing conversation turn info"
        );
        return;
    };

    let user_turns = turn_info.user_turns;
    let total_items = turn_info.total_items + output_item_count;

    match maybe_schedule_stmo_after_persist(
        conversation_memory_writer,
        memory_execution_context,
        conversation_id,
        response_id,
        user_turns,
        total_items,
    )
    .await
    {
        Ok(true) => {
            info!(
                conversation_id = %conversation_id.0,
                response_id = %response_id.0,
                user_turns,
                total_items,
                "Enqueued STMO memory condensation job"
            );
        }
        Ok(false) => {
            debug!(
                conversation_id = %conversation_id.0,
                response_id = %response_id.0,
                user_turns,
                total_items,
                "STMO not enqueued for this response boundary"
            );
        }
        Err(e) => {
            warn!(
                conversation_id = %conversation_id.0,
                response_id = %response_id.0,
                error = %e,
                "Failed to enqueue STMO memory job; continuing without failing response"
            );
        }
    }
}

/// Persist conversation items to storage
///
/// This function:
/// 1. Extracts and normalizes input items from the request
/// 2. Extracts output items from the response
/// 3. Stores ALL items in response storage (always)
/// 4. If conversation provided, also links items to conversation
#[expect(
    clippy::too_many_arguments,
    reason = "persistence entrypoint assembles all storage handles and request context in one call"
)]
pub async fn persist_conversation_items(
    conversation_storage: Arc<dyn ConversationStorage>,
    item_storage: Arc<dyn ConversationItemStorage>,
    conversation_memory_writer: Arc<dyn ConversationMemoryWriter>,
    response_storage: Arc<dyn ResponseStorage>,
    response_json: &Value,
    original_body: &ResponsesRequest,
    request_context: Option<StorageRequestContext>,
    memory_execution_context: MemoryExecutionContext,
    conversation_turn_info: Option<ConversationTurnInfo>,
) -> Result<(), String> {
    let inner = persist_conversation_items_inner(
        conversation_storage,
        item_storage,
        conversation_memory_writer,
        response_storage,
        response_json,
        original_body,
        memory_execution_context,
        conversation_turn_info,
    );
    match request_context {
        Some(ctx) => with_request_context(ctx, inner).await,
        None => inner.await,
    }
}

async fn persist_conversation_items_inner(
    conversation_storage: Arc<dyn ConversationStorage>,
    item_storage: Arc<dyn ConversationItemStorage>,
    conversation_memory_writer: Arc<dyn ConversationMemoryWriter>,
    response_storage: Arc<dyn ResponseStorage>,
    response_json: &Value,
    original_body: &ResponsesRequest,
    memory_execution_context: MemoryExecutionContext,
    conversation_turn_info: Option<ConversationTurnInfo>,
) -> Result<(), String> {
    // Respect store=false: skip persistence entirely (matches official API behavior)
    if !original_body.store.unwrap_or(true) {
        return Ok(());
    }

    // Extract response ID
    let response_id_str = response_json
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "Response missing id field".to_string())?;
    let response_id = ResponseId::from(response_id_str);

    // Parse and normalize input items from request
    let input_items = extract_input_items(&original_body.input)?;

    // Parse output items from response
    let output_items = response_json
        .get("output")
        .and_then(|v| v.as_array())
        .cloned()
        .ok_or_else(|| "No output array in response".to_string())?;

    // Build and store response
    let mut stored_response = build_stored_response(response_json, original_body);
    stored_response.id = response_id.clone();
    stored_response.input = Value::Array(input_items.clone());

    response_storage
        .store_response(stored_response)
        .await
        .map_err(|e| format!("Failed to store response: {e}"))?;

    // Check if conversation is provided and validate it exists
    let conv_id_opt = if let Some(conv_ref) = &original_body.conversation {
        let conv_id = ConversationId::from(conv_ref.as_id());
        match conversation_storage.get_conversation(&conv_id).await {
            Ok(Some(_)) => Some(conv_id),
            Ok(None) => {
                warn!(conversation_id = %conv_id.0, "Conversation not found, skipping item linking");
                None
            }
            Err(e) => return Err(format!("Failed to get conversation: {e}")),
        }
    } else {
        None
    };

    // If conversation exists, link items to it
    if let Some(conv_id) = conv_id_opt {
        link_items_to_conversation(
            &item_storage,
            &conv_id,
            &input_items,
            &output_items,
            response_id_str,
        )
        .await?;

        handle_stmo_after_persist(
            &conversation_memory_writer,
            &memory_execution_context,
            &conv_id,
            &response_id,
            conversation_turn_info,
            output_items.len(),
        )
        .await;

        info!(
            conversation_id = %conv_id.0,
            response_id = %response_id.0,
            input_count = input_items.len(),
            output_count = output_items.len(),
            "Persisted response and linked items to conversation"
        );
    } else {
        info!(
            response_id = %response_id.0,
            input_count = input_items.len(),
            output_count = output_items.len(),
            "Persisted response without conversation linking"
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stmo_fires_at_boundary_turns() {
        // Fires at 4, 7, 10 — not before or between
        assert!(!should_enqueue_stmo(3));
        assert!(should_enqueue_stmo(4));
        assert!(!should_enqueue_stmo(5));
        assert!(!should_enqueue_stmo(6));
        assert!(should_enqueue_stmo(7));
        assert!(should_enqueue_stmo(10));
    }

    #[test]
    fn count_user_turns_case_insensitive() {
        let input = ResponseInput::Items(vec![
            ResponseInputOutputItem::SimpleInputMessage {
                content: StringOrContentParts::String("u1".to_string()),
                role: "user".to_string(),
                r#type: None,
                phase: None,
            },
            ResponseInputOutputItem::SimpleInputMessage {
                content: StringOrContentParts::String("a1".to_string()),
                role: "assistant".to_string(),
                r#type: None,
                phase: None,
            },
            ResponseInputOutputItem::SimpleInputMessage {
                content: StringOrContentParts::String("u2".to_string()),
                role: "User".to_string(),
                r#type: None,
                phase: None,
            },
            ResponseInputOutputItem::FunctionToolCall {
                id: "fc_1".to_string(),
                call_id: "call_1".to_string(),
                name: "tool".to_string(),
                arguments: "{}".to_string(),
                output: None,
                status: None,
            },
        ]);
        let info = count_conversation_turn_info(&input);
        assert_eq!(info.user_turns, 2);
        assert_eq!(info.total_items, 4);
    }

    #[test]
    fn total_items_exceeds_user_turns() {
        let input = ResponseInput::Items(vec![
            ResponseInputOutputItem::SimpleInputMessage {
                content: StringOrContentParts::String("u1".to_string()),
                role: "user".to_string(),
                r#type: None,
                phase: None,
            },
            ResponseInputOutputItem::SimpleInputMessage {
                content: StringOrContentParts::String("a1".to_string()),
                role: "assistant".to_string(),
                r#type: None,
                phase: None,
            },
            ResponseInputOutputItem::SimpleInputMessage {
                content: StringOrContentParts::String("u2".to_string()),
                role: "user".to_string(),
                r#type: None,
                phase: None,
            },
            ResponseInputOutputItem::SimpleInputMessage {
                content: StringOrContentParts::String("a2".to_string()),
                role: "assistant".to_string(),
                r#type: None,
                phase: None,
            },
            ResponseInputOutputItem::SimpleInputMessage {
                content: StringOrContentParts::String("u3".to_string()),
                role: "user".to_string(),
                r#type: None,
                phase: None,
            },
            ResponseInputOutputItem::SimpleInputMessage {
                content: StringOrContentParts::String("a3".to_string()),
                role: "assistant".to_string(),
                r#type: None,
                phase: None,
            },
            ResponseInputOutputItem::SimpleInputMessage {
                content: StringOrContentParts::String("u4".to_string()),
                role: "user".to_string(),
                r#type: None,
                phase: None,
            },
        ]);

        let info = count_conversation_turn_info(&input);
        let target_item_end = info.total_items + 2; // response output contributes after load

        assert_eq!(info.user_turns, 4);
        assert_eq!(info.total_items, 7);
        assert_eq!(target_item_end, 9); // total > user turns
        assert!(should_enqueue_stmo(info.user_turns));
    }
}
