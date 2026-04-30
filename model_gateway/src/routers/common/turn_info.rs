//! Basic turn counting for `BeforeModelCtx` / `AfterPersistCtx`.
//!
//! Per spec §`compute_turn_info` scope: PR 1 implements only generic counting.
//! The five STMO-specific special cases from PR #1400 (raw-count correction,
//! chain-overlap guard, store=false short-circuit, no-history-loaded skip,
//! MCP-streaming skip) are NOT implemented here. If future memory work needs
//! them, they live inside the memory crate's scheduler, not in core.

use serde_json::Value;
use smg_data_connector::{ConversationId, ConversationItemStorage, ListParams, SortOrder};
use smg_extensions::ConversationTurnInfo;
use tracing::warn;

/// Compute basic turn-counting telemetry for a request.
///
/// Counts user-role messages from history + incoming input as `user_turns`,
/// total items as `total_items`, and exposes the raw stored-item count
/// (pre-filter) as `raw_stored_item_count` for callers that need it.
pub async fn compute_turn_info(
    history: &dyn ConversationItemStorage,
    conversation_id: Option<&ConversationId>,
    incoming_input: Option<&Value>,
) -> ConversationTurnInfo {
    let Some(conv_id) = conversation_id else {
        return ConversationTurnInfo::new(
            count_user_turns_in_input(incoming_input),
            count_total_items_in_input(incoming_input),
            None,
        );
    };

    let params = ListParams {
        limit: 1024,
        order: SortOrder::Asc,
        after: None,
    };

    let stored = match history.list_items(conv_id, params).await {
        Ok(items) => items,
        Err(e) => {
            warn!(error = %e, "failed to list conversation items for turn counting");
            return ConversationTurnInfo::default();
        }
    };

    let raw_stored = stored.len() as u32;
    let mut user_turns: u32 = stored
        .iter()
        .filter(|it| it.role.as_deref() == Some("user"))
        .count() as u32;
    let mut total_items: u32 = stored.len() as u32;

    user_turns += count_user_turns_in_input(incoming_input);
    total_items += count_total_items_in_input(incoming_input);

    ConversationTurnInfo::new(user_turns, total_items, Some(raw_stored))
}

fn count_user_turns_in_input(input: Option<&Value>) -> u32 {
    let Some(arr) = input.and_then(|v| v.as_array()) else {
        return 0;
    };
    arr.iter()
        .filter(|item| {
            item.get("role").and_then(|r| r.as_str()) == Some("user")
        })
        .count() as u32
}

fn count_total_items_in_input(input: Option<&Value>) -> u32 {
    input
        .and_then(|v| v.as_array())
        .map(|a| a.len() as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use serde_json::json;
    use smg_data_connector::{
        ConversationId, MemoryConversationItemStorage, NewConversationItem,
    };

    #[tokio::test]
    async fn empty_input_no_conversation_returns_zeros() {
        let history = MemoryConversationItemStorage::new();
        let info = compute_turn_info(&history, None, None).await;
        assert_eq!(info.user_turns, 0);
        assert_eq!(info.total_items, 0);
        assert_eq!(info.raw_stored_item_count, None);
    }

    #[tokio::test]
    async fn input_only_user_turns() {
        let history = MemoryConversationItemStorage::new();
        let input = json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"}
        ]);
        let info = compute_turn_info(&history, None, Some(&input)).await;
        assert_eq!(info.user_turns, 2);
        assert_eq!(info.total_items, 3);
    }

    #[tokio::test]
    async fn history_plus_input_aggregates() {
        let history = MemoryConversationItemStorage::new();
        let conv_id: ConversationId = "conv_test".into();
        let item = history
            .create_item(NewConversationItem {
                id: None,
                response_id: None,
                item_type: "message".into(),
                role: Some("user".into()),
                content: json!([]),
                status: Some("completed".into()),
            })
            .await
            .unwrap();
        history
            .link_item(&conv_id, &item.id, Utc::now())
            .await
            .unwrap();

        let input = json!([{"role": "user", "content": "next"}]);
        let info = compute_turn_info(&history, Some(&conv_id), Some(&input)).await;

        assert_eq!(info.user_turns, 2);
        assert_eq!(info.total_items, 2);
        assert_eq!(info.raw_stored_item_count, Some(1));
    }
}
