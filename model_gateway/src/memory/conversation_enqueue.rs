use std::sync::Arc;

use chrono::Utc;
use serde_json::Value;
use smg_data_connector::{ConversationId, ConversationMemoryWriter, ResponseId};
use tracing::warn;

use super::{build_enqueue_plan, EnqueueInputs, MemoryExecutionContext};

fn push_trimmed(text: &str, out: &mut Vec<String>) {
    let trimmed = text.trim();
    if !trimmed.is_empty() {
        out.push(trimmed.to_string());
    }
}

fn collect_text_fragments(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::String(text) => push_trimmed(text, out),
        Value::Array(values) => {
            for item in values {
                collect_text_fragments(item, out);
            }
        }
        Value::Object(map) => {
            if let Some(text) = map.get("text").and_then(Value::as_str) {
                push_trimmed(text, out);
            } else if let Some(content) = map.get("content") {
                collect_text_fragments(content, out);
            }
        }
        _ => {}
    }
}

fn join_text_fragments(fragments: Vec<String>) -> Option<String> {
    if fragments.is_empty() {
        None
    } else {
        Some(fragments.join("\n"))
    }
}

pub(crate) fn extract_role_message_text_from_items(items: &[Value], role: &str) -> Option<String> {
    let mut fragments = Vec::new();

    for item in items {
        if item.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        if item.get("role").and_then(Value::as_str) != Some(role) {
            continue;
        }
        if let Some(content) = item.get("content") {
            collect_text_fragments(content, &mut fragments);
        }
    }

    join_text_fragments(fragments)
}

pub(crate) async fn enqueue_conversation_memory_rows(
    conversation_memory_writer: &Arc<dyn ConversationMemoryWriter>,
    memory_execution_context: &MemoryExecutionContext,
    conversation_id: &ConversationId,
    response_id: Option<ResponseId>,
    user_text: Option<String>,
    assistant_text: Option<String>,
) {
    let plan = match build_enqueue_plan(EnqueueInputs {
        now: Utc::now(),
        memory_execution_context: memory_execution_context.clone(),
        conversation_id: conversation_id.clone(),
        response_id,
        user_text,
        assistant_text,
    }) {
        Ok(Some(plan)) => plan,
        Ok(None) => return,
        Err(reason) => {
            warn!(
                conversation_id = %conversation_id.0,
                ?reason,
                "Skipping conversation memory enqueue due to invalid durable memory configuration"
            );
            return;
        }
    };

    if let Err(err) = conversation_memory_writer.create_memories(plan.rows).await {
        warn!(
            conversation_id = %conversation_id.0,
            error = %err,
            "Failed to enqueue conversation memory rows"
        );
    }
}

pub(crate) async fn enqueue_conversation_memory_rows_from_turn_items(
    conversation_memory_writer: &Arc<dyn ConversationMemoryWriter>,
    memory_execution_context: &MemoryExecutionContext,
    conversation_id: &ConversationId,
    response_id: Option<ResponseId>,
    input_items: &[Value],
    output_items: &[Value],
) {
    enqueue_conversation_memory_rows(
        conversation_memory_writer,
        memory_execution_context,
        conversation_id,
        response_id,
        extract_role_message_text_from_items(input_items, "user"),
        extract_role_message_text_from_items(output_items, "assistant"),
    )
    .await;
}

pub(crate) async fn enqueue_conversation_memory_rows_from_items(
    conversation_memory_writer: &Arc<dyn ConversationMemoryWriter>,
    memory_execution_context: &MemoryExecutionContext,
    conversation_id: &ConversationId,
    items: &[Value],
) {
    enqueue_conversation_memory_rows(
        conversation_memory_writer,
        memory_execution_context,
        conversation_id,
        None,
        extract_role_message_text_from_items(items, "user"),
        extract_role_message_text_from_items(items, "assistant"),
    )
    .await;
}
