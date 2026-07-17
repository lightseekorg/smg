//! PostgreSQL integration coverage for JSON columns.
//!
//! Run with a disposable database:
//! ```text
//! SMG_TEST_POSTGRES_URL=postgres://postgres:postgres@localhost/smg_test \
//!   cargo test -p data-connector --test postgres_json -- --ignored
//! ```

use std::env;

use chrono::Utc;
use data_connector::{
    create_storage, HistoryBackend, ListParams, NewConversation, NewConversationItem,
    PostgresConfig, SchemaConfig, SortOrder, StorageFactoryConfig, StoredResponse,
};
use serde_json::json;

#[tokio::test]
#[ignore = "requires a PostgreSQL instance configured through SMG_TEST_POSTGRES_URL"]
async fn json_columns_round_trip() {
    let db_url = env::var("SMG_TEST_POSTGRES_URL")
        .expect("SMG_TEST_POSTGRES_URL must point to a disposable PostgreSQL database");
    let schema = SchemaConfig {
        auto_migrate: true,
        ..Default::default()
    };
    let postgres = PostgresConfig {
        db_url,
        pool_max: 4,
        schema: Some(schema),
    };
    let backend = HistoryBackend::Postgres;
    let storage = create_storage(StorageFactoryConfig {
        backend: &backend,
        oracle: None,
        postgres: Some(&postgres),
        redis: None,
        hook: None,
    })
    .await
    .expect("PostgreSQL storage should initialize");

    let first_input = json!([{
        "role": "user",
        "content": [{"type": "input_text", "text": "remember PG-HISTORY-TEST"}]
    }]);
    let first_raw_response = json!({
        "id": "response-one",
        "output": [{"type": "message", "content": [{"text": "stored"}]}]
    });
    let mut first = StoredResponse::new(None);
    first.input = first_input.clone();
    first.raw_response = first_raw_response.clone();
    let first_id = storage
        .response_storage
        .store_response(first)
        .await
        .expect("first response should be stored");

    let loaded = storage
        .response_storage
        .get_response(&first_id)
        .await
        .expect("first response should be read")
        .expect("first response should exist");
    assert_eq!(loaded.input, first_input);
    assert_eq!(loaded.raw_response, first_raw_response);

    let second_input = json!([{"role": "user", "content": "what was the token?"}]);
    let second_raw_response = json!({"id": "response-two", "status": "completed"});
    let mut second = StoredResponse::new(Some(first_id.clone()));
    second.input = second_input.clone();
    second.raw_response = second_raw_response.clone();
    let second_id = storage
        .response_storage
        .store_response(second)
        .await
        .expect("second response should be stored");

    let chain = storage
        .response_storage
        .get_response_chain(&second_id, None)
        .await
        .expect("response chain should be read");
    assert_eq!(chain.responses.len(), 2);
    assert_eq!(chain.responses[0].id, first_id);
    assert_eq!(chain.responses[0].input, first_input);
    assert_eq!(chain.responses[0].raw_response, first_raw_response);
    assert_eq!(chain.responses[1].id, second_id);
    assert_eq!(
        chain.responses[1].previous_response_id,
        Some(first_id.clone())
    );
    assert_eq!(chain.responses[1].input, second_input);
    assert_eq!(chain.responses[1].raw_response, second_raw_response);

    let metadata = json!({"tenant": "acme", "nested": {"retention": 7}})
        .as_object()
        .expect("metadata fixture should be an object")
        .clone();
    let conversation = storage
        .conversation_storage
        .create_conversation(NewConversation {
            id: None,
            metadata: Some(metadata.clone()),
        })
        .await
        .expect("conversation should be stored");
    let loaded_conversation = storage
        .conversation_storage
        .get_conversation(&conversation.id)
        .await
        .expect("conversation should be read")
        .expect("conversation should exist");
    assert_eq!(loaded_conversation.metadata, Some(metadata));

    let updated_metadata = json!({"tenant": "acme", "updated": true})
        .as_object()
        .expect("metadata fixture should be an object")
        .clone();
    let updated = storage
        .conversation_storage
        .update_conversation(&conversation.id, Some(updated_metadata.clone()))
        .await
        .expect("conversation should be updated")
        .expect("conversation should exist");
    assert_eq!(updated.metadata, Some(updated_metadata));
    let loaded_updated = storage
        .conversation_storage
        .get_conversation(&conversation.id)
        .await
        .expect("updated conversation should be read")
        .expect("conversation should exist");
    assert_eq!(loaded_updated.metadata, updated.metadata);

    let cleared = storage
        .conversation_storage
        .update_conversation(&conversation.id, None)
        .await
        .expect("conversation metadata should be cleared")
        .expect("conversation should exist");
    assert_eq!(cleared.metadata, None);
    let loaded_cleared = storage
        .conversation_storage
        .get_conversation(&conversation.id)
        .await
        .expect("conversation should be read after clearing metadata")
        .expect("conversation should exist");
    assert_eq!(loaded_cleared.metadata, None);

    let item_content = json!([{
        "type": "input_text",
        "text": "nested content",
        "annotations": [{"kind": "test", "value": 1}]
    }]);
    let item = storage
        .conversation_item_storage
        .create_item(NewConversationItem {
            id: None,
            response_id: Some(second_id.0.clone()),
            item_type: "message".to_string(),
            role: Some("user".to_string()),
            content: item_content.clone(),
            status: Some("completed".to_string()),
        })
        .await
        .expect("conversation item should be stored");
    let loaded_item = storage
        .conversation_item_storage
        .get_item(&item.id)
        .await
        .expect("conversation item should be read")
        .expect("conversation item should exist");
    assert_eq!(loaded_item.content, item_content);

    storage
        .conversation_item_storage
        .link_item(&conversation.id, &item.id, Utc::now())
        .await
        .expect("conversation item should be linked");
    let listed = storage
        .conversation_item_storage
        .list_items(
            &conversation.id,
            ListParams {
                limit: 10,
                order: SortOrder::Asc,
                after: None,
            },
        )
        .await
        .expect("conversation items should be listed");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, item.id);
    assert_eq!(listed[0].content, item_content);
}
