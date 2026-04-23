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
    ("mcp_list_tools", &["tools", "server_label"]),
    ("function_call", &["call_id", "name", "arguments", "output"]),
    ("function_call_output", &["call_id", "output"]),
];

const ITEM_KEY_TYPE: &str = "type";
const ITEM_TYPE_MESSAGE: &str = "message";
const ITEM_KEY_ROLE: &str = "role";
const ROLE_USER: &str = "user";

const STMO_FIRST_TURN: usize = 4;
const STMO_TURN_INTERVAL: usize = 3;

const STMO_CFG_KEY_CONDENSER_MODEL: &str = "condenser_model";
const STMO_CFG_KEY_LAST_INDEX: &str = "last_index";
const STMO_CFG_KEY_TARGET_ITEM_END: &str = "target_item_end";

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

/// Persist conversation items to storage
///
/// This function:
/// 1. Extracts and normalizes input items from the request
/// 2. Extracts output items from the response
/// 3. Stores ALL items in response storage (always)
/// 4. If conversation provided, also links items to conversation
#[expect(
    clippy::too_many_arguments,
    reason = "threads storage handles plus request payload/context through a shared persistence entrypoint"
)]
pub async fn persist_conversation_items(
    conversation_storage: Arc<dyn ConversationStorage>,
    item_storage: Arc<dyn ConversationItemStorage>,
    response_storage: Arc<dyn ResponseStorage>,
    conversation_memory_writer: Arc<dyn ConversationMemoryWriter>,
    memory_execution_context: MemoryExecutionContext,
    response_json: &Value,
    original_body: &ResponsesRequest,
    request_context: Option<StorageRequestContext>,
) -> Result<(), String> {
    let inner = persist_conversation_items_inner(
        conversation_storage,
        item_storage,
        response_storage,
        conversation_memory_writer,
        memory_execution_context,
        response_json,
        original_body,
    );
    match request_context {
        Some(ctx) => with_request_context(ctx, inner).await,
        None => inner.await,
    }
}

async fn persist_conversation_items_inner(
    conversation_storage: Arc<dyn ConversationStorage>,
    item_storage: Arc<dyn ConversationItemStorage>,
    response_storage: Arc<dyn ResponseStorage>,
    conversation_memory_writer: Arc<dyn ConversationMemoryWriter>,
    memory_execution_context: MemoryExecutionContext,
    response_json: &Value,
    original_body: &ResponsesRequest,
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
        enqueue_stmo_if_needed(
            &conversation_memory_writer,
            &memory_execution_context,
            &conv_id,
            &response_id,
            &input_items,
            &output_items,
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

/// Counts user-message turns from a heterogeneous list of input items.
///
/// A turn is counted only when an item has:
/// - `type == ITEM_TYPE_MESSAGE`
/// - `role == ROLE_USER`
///
/// Malformed objects, missing fields, non-string fields, and non-object JSON values are
/// handled safely by `get(...).and_then(Value::as_str)` and are treated as non-matching
/// items (i.e., they do not contribute to the count).
fn count_user_turns_in_input_items(items: &[Value]) -> usize {
    items
        .iter()
        .filter(|item| {
            item.get(ITEM_KEY_TYPE).and_then(Value::as_str) == Some(ITEM_TYPE_MESSAGE)
                && item.get(ITEM_KEY_ROLE).and_then(Value::as_str) == Some(ROLE_USER)
        })
        .count()
}

/// Returns true when STMO should be enqueued for the current user-turn count.
///
/// Trigger pattern: first at turn 4, then every 3 turns after that (4, 7, 10, 13, ...).
fn should_enqueue_stmo_for_current_turn(user_turns: usize) -> bool {
    user_turns >= STMO_FIRST_TURN && (user_turns - 1).is_multiple_of(STMO_TURN_INTERVAL)
}

/// Enqueues an STMO job when the current request is eligible.
///
/// Eligibility gates:
/// - STM is enabled in `memory_execution_context`
/// - current user turn count (from `input_items`) is on an STMO trigger boundary
/// - `stm_condenser_model_id` is present
///
/// On success, this creates a `NewConversationMemory` row with type `Stmo`, status `Ready`,
/// and a minimal `memory_config` payload (`condenser_model`, `last_index`, `target_item_end`).
///
/// This function is best-effort: enqueue failures are logged and swallowed.
async fn enqueue_stmo_if_needed(
    conversation_memory_writer: &Arc<dyn ConversationMemoryWriter>,
    memory_execution_context: &MemoryExecutionContext,
    conversation_id: &ConversationId,
    response_id: &ResponseId,
    input_items: &[Value],
    output_items: &[Value],
) {
    if !memory_execution_context.stm_enabled {
        return;
    }

    // Turn counting intentionally uses `input_items`, which is expected to include
    // accumulated conversation input at this stage of the responses pipeline.
    let user_turns = count_user_turns_in_input_items(input_items);
    if !should_enqueue_stmo_for_current_turn(user_turns) {
        return;
    }

    let Some(condenser_model) = memory_execution_context.stm_condenser_model_id.as_deref() else {
        warn!(
            conversation_id = %conversation_id.0,
            response_id = %response_id.0,
            "STM enabled but condenser model id is missing; skipping STMO enqueue"
        );
        return;
    };

    let target_item_end = input_items.len() + output_items.len();
    // STMO worker config semantics:
    // - `last_index`: latest observed user-turn count at enqueue time.
    // - `target_item_end`: exclusive end index for items included in this run.
    let job_config = json!({
        STMO_CFG_KEY_CONDENSER_MODEL: condenser_model,
        STMO_CFG_KEY_LAST_INDEX: user_turns,
        STMO_CFG_KEY_TARGET_ITEM_END: target_item_end,
    })
    .to_string();

    let row = NewConversationMemory {
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
        memory_config: Some(job_config),
        scope_id: None,
        error_msg: None,
    };

    if let Err(err) = conversation_memory_writer.create_memory(row).await {
        warn!(
            conversation_id = %conversation_id.0,
            response_id = %response_id.0,
            error = %err,
            "Failed to enqueue STMO job (best-effort; request flow continues)"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use serde_json::Value;
    use smg_data_connector::{ConversationMemoryId, ConversationMemoryResult};

    use super::*;

    struct RecordingConversationMemoryWriter {
        rows: Mutex<Vec<NewConversationMemory>>,
    }

    impl RecordingConversationMemoryWriter {
        fn new() -> Self {
            Self {
                rows: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl ConversationMemoryWriter for RecordingConversationMemoryWriter {
        async fn create_memory(
            &self,
            input: NewConversationMemory,
        ) -> ConversationMemoryResult<ConversationMemoryId> {
            self.rows.lock().expect("rows mutex poisoned").push(input);
            Ok(ConversationMemoryId::from("mem_test"))
        }
    }

    fn user_message_item() -> Value {
        json!({
            ITEM_KEY_TYPE: ITEM_TYPE_MESSAGE,
            ITEM_KEY_ROLE: ROLE_USER,
        })
    }

    #[test]
    fn stmo_turn_boundary_matches_expected_sequence() {
        let cases = [
            (1, false),
            (2, false),
            (3, false),
            (4, true),
            (5, false),
            (6, false),
            (7, true),
            (8, false),
            (9, false),
            (10, true),
            (11, false),
            (12, false),
            (13, true),
        ];

        for (turn, expected) in cases {
            assert_eq!(
                should_enqueue_stmo_for_current_turn(turn),
                expected,
                "turn={turn}"
            );
        }
    }

    #[tokio::test]
    async fn enqueue_stmo_if_needed_enqueues_expected_row_on_boundary() {
        let writer = Arc::new(RecordingConversationMemoryWriter::new());
        let writer_dyn: Arc<dyn ConversationMemoryWriter> = writer.clone();

        let memory_execution_context = MemoryExecutionContext {
            stm_enabled: true,
            stm_condenser_model_id: Some("condense-1".to_string()),
            ..MemoryExecutionContext::default()
        };

        let input_items = vec![
            user_message_item(),
            user_message_item(),
            user_message_item(),
            user_message_item(),
        ];
        let output_items =
            vec![json!({ ITEM_KEY_TYPE: ITEM_TYPE_MESSAGE, ITEM_KEY_ROLE: "assistant" })];
        let conversation_id = ConversationId::from("conv_test");
        let response_id = ResponseId::from("resp_test");

        enqueue_stmo_if_needed(
            &writer_dyn,
            &memory_execution_context,
            &conversation_id,
            &response_id,
            &input_items,
            &output_items,
        )
        .await;

        let rows = writer.rows.lock().expect("rows mutex poisoned");
        assert_eq!(rows.len(), 1, "should enqueue exactly one STMO row");

        let row = &rows[0];
        assert_eq!(row.conversation_id, conversation_id);
        assert_eq!(row.response_id, Some(response_id));
        assert_eq!(row.memory_type, ConversationMemoryType::Stmo);
        assert_eq!(row.status, ConversationMemoryStatus::Ready);

        let config = row
            .memory_config
            .as_deref()
            .expect("memory_config should be set");
        let config_json: Value =
            serde_json::from_str(config).expect("memory_config must be valid JSON");

        assert_eq!(
            config_json
                .get(STMO_CFG_KEY_CONDENSER_MODEL)
                .and_then(Value::as_str),
            Some("condense-1")
        );
        assert_eq!(
            config_json
                .get(STMO_CFG_KEY_LAST_INDEX)
                .and_then(Value::as_u64),
            Some(4)
        );
        assert_eq!(
            config_json
                .get(STMO_CFG_KEY_TARGET_ITEM_END)
                .and_then(Value::as_u64),
            Some(5)
        );
    }
}
