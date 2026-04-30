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

/// Test-only invocation counter for `compute_turn_info`.
///
/// Used by unit tests to verify that callers correctly skip the turn-info
/// compute when their interceptor registry is empty, avoiding unnecessary
/// work on the hot path.
#[cfg(test)]
pub(crate) mod test_instrumentation {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(crate) static COMPUTE_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

    pub(crate) fn current_count() -> usize {
        COMPUTE_CALL_COUNT.load(Ordering::SeqCst)
    }

    pub(crate) fn reset() {
        COMPUTE_CALL_COUNT.store(0, Ordering::SeqCst);
    }
}

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
    #[cfg(test)]
    test_instrumentation::COMPUTE_CALL_COUNT
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

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
    use std::sync::Arc;

    use super::*;
    use chrono::Utc;
    use serde_json::json;
    use serial_test::serial;
    use smg_data_connector::{
        ConversationId, ConversationItemStorage, ConversationStorage,
        MemoryConversationItemStorage, MemoryConversationStorage, NewConversation,
        NewConversationItem,
    };
    use smg_extensions::InterceptorRegistry;

    use crate::{
        memory::MemoryExecutionContext,
        routers::conversations::create_conversation_items_with_headers,
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

    /// Verifies the items-only path skips `compute_turn_info` when the
    /// interceptor registry is empty. This guards against accidentally
    /// re-introducing the work cost on the no-interceptor hot path.
    ///
    /// Both arms of the test run inline against a single counter, so the
    /// `#[serial]` attribute prevents interleaving with other tests in
    /// this module that also exercise `compute_turn_info`.
    #[tokio::test]
    #[serial]
    async fn items_only_skips_compute_turn_info_when_registry_is_empty() {
        let conversation_storage: Arc<dyn ConversationStorage> =
            Arc::new(MemoryConversationStorage::new());
        let conversation_item_storage: Arc<dyn ConversationItemStorage> =
            Arc::new(MemoryConversationItemStorage::new());

        let conv = conversation_storage
            .create_conversation(NewConversation {
                id: None,
                metadata: None,
            })
            .await
            .expect("create conversation");

        let body = json!({
            "items": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}]
                }
            ]
        });

        // Empty registry → compute_turn_info must not be called.
        test_instrumentation::reset();
        let baseline = test_instrumentation::current_count();

        let resp_empty = create_conversation_items_with_headers(
            &conversation_storage,
            &conversation_item_storage,
            &conv.id.0,
            body.clone(),
            MemoryExecutionContext::default(),
            InterceptorRegistry::default(),
            "req_test_empty".to_string(),
            Some("test-tenant".to_string()),
            Default::default(),
        )
        .await;
        assert_eq!(resp_empty.status(), http::StatusCode::OK);

        assert_eq!(
            test_instrumentation::current_count(),
            baseline,
            "compute_turn_info must not be invoked when the registry is empty"
        );

        // Sanity: with a non-empty registry, compute_turn_info IS called.
        struct Noop;
        #[async_trait::async_trait]
        impl smg_extensions::ResponsesInterceptor for Noop {
            fn name(&self) -> &'static str {
                "noop"
            }
        }

        let mut builder = InterceptorRegistry::builder();
        builder.register(Arc::new(Noop));
        let registry = builder.build();

        let resp_nonempty = create_conversation_items_with_headers(
            &conversation_storage,
            &conversation_item_storage,
            &conv.id.0,
            body,
            MemoryExecutionContext::default(),
            registry,
            "req_test_nonempty".to_string(),
            Some("test-tenant".to_string()),
            Default::default(),
        )
        .await;
        assert_eq!(resp_nonempty.status(), http::StatusCode::OK);

        assert!(
            test_instrumentation::current_count() > baseline,
            "compute_turn_info must be invoked at least once when a registered interceptor exists (got count={}, baseline={})",
            test_instrumentation::current_count(),
            baseline
        );
    }
}
