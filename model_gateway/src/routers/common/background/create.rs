//! Background create path: resolve input snapshot + enqueue.

use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::{
    event_types::ItemType,
    responses::{ResponseContentPart, ResponseInputOutputItem, ResponsesRequest},
};
use serde_json::{json, Value};
use smg_data_connector::{
    BackgroundRepositoryError, BackgroundResponseRepository, ConversationId, ConversationItem,
    ConversationItemStorage, ConversationStorage, EnqueueRequest,
    RequestContext as StorageRequestContext, ResponseId, ResponseStorage, StoredResponse,
};
use tracing::warn;
use uuid::Uuid;

use crate::{
    config::BackgroundConfig,
    routers::{common::persistence_utils::split_stored_message_content, error},
};

const MAX_SNAPSHOT_ITEMS: usize = 100;

/// Storage handles the background create path needs. Passed in from the
/// caller so the handler doesn't reach into `AppContext` directly — every
/// entry point (HTTP regular, OpenAI, gRPC) assembles these from whatever
/// shape its context already has.
pub struct BackgroundCreateDeps<'a> {
    pub repository: Option<&'a Arc<dyn BackgroundResponseRepository>>,
    pub response_storage: &'a dyn ResponseStorage,
    pub conversation_storage: &'a dyn ConversationStorage,
    pub conversation_item_storage: &'a dyn ConversationItemStorage,
    pub background_config: &'a BackgroundConfig,
    /// Forwarded to `enqueue` so the worker replays the caller's tenant /
    /// principal identity when it later writes the finalized response.
    pub request_context: Option<&'a StorageRequestContext>,
}

/// Handle `POST /v1/responses` with `background=true`.
///
/// Returns a JSON `Response` object with `status: "queued"`, or an HTTP error
/// response when validation / snapshot resolution / enqueue fails.
pub async fn handle_background_create(
    deps: BackgroundCreateDeps<'_>,
    request: &ResponsesRequest,
    model_id: &str,
) -> Response {
    let Some(repository) = deps.repository else {
        return error::bad_request(
            "background_not_supported",
            "Background mode is not supported on this history_backend. \
             Use memory, postgres, or oracle.",
        );
    };

    let snapshot = match resolve_snapshot(&deps, request).await {
        Ok(s) => s,
        Err(resp) => return resp,
    };

    let response_id = ResponseId::from(format!("resp_{}", Uuid::now_v7()).as_str());
    let now_unix = chrono::Utc::now().timestamp();
    let initial_raw = initial_queued_response(&response_id, model_id, now_unix, request);
    let request_json = serde_json::to_value(request).unwrap_or(Value::Null);

    let mut enqueue_req = EnqueueRequest::new(
        response_id.clone(),
        model_id.to_string(),
        request_json,
        Value::Array(snapshot),
        initial_raw.clone(),
        false,
        request.priority,
    );
    enqueue_req.conversation_id = request.conversation.as_ref().map(|c| c.as_id().to_string());
    enqueue_req
        .safety_identifier
        .clone_from(&request.safety_identifier);
    enqueue_req.previous_response_id = request
        .previous_response_id
        .as_deref()
        .map(ResponseId::from);

    let max_depth = u64::from(deps.background_config.max_queue_depth);
    let request_context = deps.request_context.cloned();
    match repository
        .enqueue(enqueue_req, request_context, Some(max_depth))
        .await
    {
        Ok(_) => Json(initial_raw).into_response(),
        Err(BackgroundRepositoryError::QueueFull { current, limit }) => error::create_error(
            StatusCode::TOO_MANY_REQUESTS,
            "queue_full",
            format!("Background queue is at capacity ({current}/{limit})."),
        ),
        Err(BackgroundRepositoryError::InvalidTransition(msg)) => {
            error::create_error(StatusCode::CONFLICT, "invalid_transition", msg)
        }
        Err(e) => error::internal_error("background_enqueue_failed", e.to_string()),
    }
}

/// Build the execution-time input snapshot by resolving `previous_response_id`
/// or `conversation` and appending the request's own `input` items.
async fn resolve_snapshot(
    deps: &BackgroundCreateDeps<'_>,
    request: &ResponsesRequest,
) -> Result<Vec<Value>, Response> {
    let mut items: Vec<Value> = Vec::new();

    if let Some(prev_id_str) = request.previous_response_id.as_deref() {
        append_prev_chain_items(deps.response_storage, prev_id_str, &mut items).await?;
    } else if let Some(conv_ref) = request.conversation.as_ref() {
        append_conversation_items(
            deps.conversation_storage,
            deps.conversation_item_storage,
            conv_ref.as_id(),
            &mut items,
        )
        .await?;
    }

    let request_input_json = serde_json::to_value(&request.input).unwrap_or(Value::Null);
    if let Some(arr) = request_input_json.as_array() {
        items.extend(arr.iter().cloned());
    } else if !request_input_json.is_null() {
        // ResponseInput::String — wrap as a single user message the worker
        // can execute. Keeping a primitive shape here avoids divergence from
        // what the rest of the pipeline stores on `StoredResponse.input`.
        items.push(json!({
            "type": "message",
            "role": "user",
            "content": request_input_json,
        }));
    }

    if items.len() > MAX_SNAPSHOT_ITEMS {
        return Err(error::create_error(
            StatusCode::CONFLICT,
            "resolved_snapshot_too_large",
            format!(
                "Resolved snapshot has {} items, exceeds the cap of {}.",
                items.len(),
                MAX_SNAPSHOT_ITEMS
            ),
        ));
    }

    Ok(items)
}

async fn append_prev_chain_items(
    storage: &dyn ResponseStorage,
    prev_id_str: &str,
    items: &mut Vec<Value>,
) -> Result<(), Response> {
    let prev_id = ResponseId::from(prev_id_str);
    let chain = match storage.get_response_chain(&prev_id, None).await {
        Ok(chain) => chain,
        Err(e) => {
            return Err(error::internal_error(
                "load_previous_response_chain_failed",
                format!("Failed to load previous response chain for {prev_id_str}: {e}"),
            ));
        }
    };
    if chain.responses.is_empty() {
        return Err(error::not_found(
            "previous_response_not_found",
            format!("Previous response with id '{prev_id_str}' not found."),
        ));
    }

    for stored in &chain.responses {
        if let Err(boxed) = check_prev_response_usable(stored, prev_id_str) {
            return Err(*boxed);
        }
        append_stored_response_items(stored, items);
    }
    Ok(())
}

fn check_prev_response_usable(
    stored: &StoredResponse,
    prev_id_str: &str,
) -> Result<(), Box<Response>> {
    let status = stored
        .raw_response
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("completed");
    match status {
        "queued" | "in_progress" => Err(Box::new(error::create_error(
            StatusCode::CONFLICT,
            "previous_response_not_ready",
            format!(
                "Previous response '{prev_id_str}' is still {status}; \
                 cannot chain until it reaches a terminal state."
            ),
        ))),
        "failed" | "cancelled" => Err(Box::new(error::create_error(
            StatusCode::CONFLICT,
            "previous_response_not_usable",
            format!(
                "Previous response '{prev_id_str}' is {status}; \
                 only completed or incomplete responses can be chained."
            ),
        ))),
        _ => Ok(()),
    }
}

fn append_stored_response_items(stored: &StoredResponse, items: &mut Vec<Value>) {
    if let Some(arr) = stored.input.as_array() {
        items.extend(arr.iter().cloned());
    }
    if let Some(out_arr) = stored.raw_response.get("output").and_then(Value::as_array) {
        items.extend(out_arr.iter().cloned());
    }
}

async fn append_conversation_items(
    conv_storage: &dyn ConversationStorage,
    item_storage: &dyn ConversationItemStorage,
    conv_id_str: &str,
    items: &mut Vec<Value>,
) -> Result<(), Response> {
    let conv_id = ConversationId::from(conv_id_str.to_string());
    match conv_storage.get_conversation(&conv_id).await {
        Ok(Some(_)) => {}
        Ok(None) => {
            return Err(error::not_found(
                "conversation_not_found",
                format!("Conversation '{conv_id_str}' not found."),
            ));
        }
        Err(e) => {
            return Err(error::internal_error(
                "load_conversation_failed",
                format!("Failed to load conversation '{conv_id_str}': {e}"),
            ));
        }
    }

    // Fetch one past the cap so oversize conversations surface as 409
    // instead of silently truncating. The cap applies to *replayable* items
    // — `conversation_item_to_snapshot_value` drops reasoning and unknown
    // types, so a raw row count can overstate the snapshot size. Fetching
    // `MAX + 1` gives the check enough headroom when every row is
    // replayable; this bound is a heuristic for the common case, not a
    // guarantee of strict rejection for pathologically reasoning-heavy
    // conversations larger than the fetch window.
    let list_params = smg_data_connector::ListParams {
        limit: MAX_SNAPSHOT_ITEMS + 1,
        order: smg_data_connector::SortOrder::Asc,
        after: None,
    };
    let conv_items = match item_storage.list_items(&conv_id, list_params).await {
        Ok(items) => items,
        Err(e) => {
            return Err(error::internal_error(
                "load_conversation_items_failed",
                format!("Failed to load conversation items for '{conv_id_str}': {e}"),
            ));
        }
    };

    let mut converted = Vec::with_capacity(conv_items.len());
    for ci in conv_items {
        match conversation_item_to_snapshot_value(ci, conv_id_str) {
            Ok(Some(value)) => converted.push(value),
            Ok(None) => {}
            Err(boxed) => return Err(*boxed),
        }
    }

    if converted.len() > MAX_SNAPSHOT_ITEMS {
        return Err(error::create_error(
            StatusCode::CONFLICT,
            "conversation_too_large",
            format!(
                "Conversation '{conv_id_str}' resolves to more than \
                 {MAX_SNAPSHOT_ITEMS} replayable items; background snapshots \
                 cannot exceed this cap."
            ),
        ));
    }

    items.extend(converted);
    Ok(())
}

/// Convert a stored `ConversationItem` row into the `ResponseInputOutputItem`
/// wire shape the worker consumes.
///
/// `Ok(None)` marks an item that is intentionally omitted from the snapshot
/// (reasoning rows, unknown types). `Err` is boxed to keep the success
/// variant small.
fn conversation_item_to_snapshot_value(
    ci: ConversationItem,
    conv_id_str: &str,
) -> Result<Option<Value>, Box<Response>> {
    let converted: Option<ResponseInputOutputItem> = match ci.item_type.as_str() {
        "message" => {
            let (content_value, stored_phase) = split_stored_message_content(ci.content);
            let content_parts: Vec<ResponseContentPart> =
                match serde_json::from_value(content_value) {
                    Ok(parts) => parts,
                    Err(e) => {
                        return Err(Box::new(error::internal_error(
                            "deserialize_conversation_item_failed",
                            format!(
                                "Failed to deserialize message content for conversation \
                                 '{conv_id_str}' item '{}': {e}",
                                ci.id.0
                            ),
                        )));
                    }
                };
            Some(ResponseInputOutputItem::Message {
                id: ci.id.0,
                role: ci.role.unwrap_or_else(|| "user".to_string()),
                content: content_parts,
                status: ci.status,
                phase: stored_phase,
            })
        }
        ItemType::FUNCTION_CALL | ItemType::FUNCTION_CALL_OUTPUT => {
            match serde_json::from_value::<ResponseInputOutputItem>(ci.content) {
                Ok(item) => Some(item),
                Err(e) => {
                    return Err(Box::new(error::internal_error(
                        "deserialize_conversation_item_failed",
                        format!(
                            "Failed to deserialize {} content for conversation \
                             '{conv_id_str}' item '{}': {e}",
                            ci.item_type, ci.id.0
                        ),
                    )));
                }
            }
        }
        "reasoning" => None,
        other => {
            warn!(
                "Dropping unknown conversation item type '{other}' in background snapshot \
                 for conversation '{conv_id_str}'"
            );
            None
        }
    };

    match converted {
        Some(item) => match serde_json::to_value(&item) {
            Ok(v) => Ok(Some(v)),
            Err(e) => Err(Box::new(error::internal_error(
                "serialize_conversation_item_failed",
                format!("Failed to serialize conversation item for '{conv_id_str}': {e}"),
            ))),
        },
        None => Ok(None),
    }
}

fn initial_queued_response(
    response_id: &ResponseId,
    model_id: &str,
    created_at_unix: i64,
    request: &ResponsesRequest,
) -> Value {
    let mut obj = json!({
        "id": response_id.0,
        "object": "response",
        "created_at": created_at_unix,
        "status": "queued",
        "background": true,
        "model": model_id,
        "output": [],
    });

    if let Some(conv) = request.conversation.as_ref() {
        obj["conversation"] = Value::String(conv.as_id().to_string());
    }
    if let Some(prev_id) = request.previous_response_id.as_ref() {
        obj["previous_response_id"] = Value::String(prev_id.clone());
    }
    obj
}

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;
    use openai_protocol::responses::ResponseInput;
    use smg_data_connector::{
        MemoryBackgroundRepository, MemoryConversationItemStorage, MemoryConversationStorage,
        MemoryResponseStorage,
    };

    use super::*;

    struct Harness {
        bg: Arc<dyn BackgroundResponseRepository>,
        response_storage: Arc<MemoryResponseStorage>,
        conversation_storage: Arc<MemoryConversationStorage>,
        conversation_item_storage: Arc<MemoryConversationItemStorage>,
        config: BackgroundConfig,
    }

    impl Harness {
        fn new(max_queue_depth: u32) -> Self {
            let rs = Arc::new(MemoryResponseStorage::new());
            let bg: Arc<dyn BackgroundResponseRepository> =
                Arc::new(MemoryBackgroundRepository::new(Arc::clone(&rs)));
            let config = BackgroundConfig {
                max_queue_depth,
                ..Default::default()
            };
            Self {
                bg,
                response_storage: rs,
                conversation_storage: Arc::new(MemoryConversationStorage::new()),
                conversation_item_storage: Arc::new(MemoryConversationItemStorage::new()),
                config,
            }
        }

        fn deps_with_repo(&self) -> BackgroundCreateDeps<'_> {
            BackgroundCreateDeps {
                repository: Some(&self.bg),
                response_storage: self.response_storage.as_ref(),
                conversation_storage: self.conversation_storage.as_ref(),
                conversation_item_storage: self.conversation_item_storage.as_ref(),
                background_config: &self.config,
                request_context: None,
            }
        }

        fn deps_without_repo(&self) -> BackgroundCreateDeps<'_> {
            BackgroundCreateDeps {
                repository: None,
                response_storage: self.response_storage.as_ref(),
                conversation_storage: self.conversation_storage.as_ref(),
                conversation_item_storage: self.conversation_item_storage.as_ref(),
                background_config: &self.config,
                request_context: None,
            }
        }
    }

    fn bg_req() -> ResponsesRequest {
        ResponsesRequest {
            background: Some(true),
            store: Some(true),
            input: ResponseInput::Text("hello".to_string()),
            ..Default::default()
        }
    }

    async fn body_json(resp: Response) -> Value {
        let (_parts, body) = resp.into_parts();
        let bytes = to_bytes(body, 1024 * 1024).await.unwrap();
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    }

    #[tokio::test]
    async fn returns_bad_request_when_repository_missing() {
        let h = Harness::new(10);
        let resp = handle_background_create(h.deps_without_repo(), &bg_req(), "gpt-5.1").await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert_eq!(body["error"]["code"], "background_not_supported");
    }

    #[tokio::test]
    async fn happy_path_returns_queued_response() {
        let h = Harness::new(10);
        let resp = handle_background_create(h.deps_with_repo(), &bg_req(), "gpt-5.1").await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["status"], "queued");
        assert_eq!(body["background"], true);
        assert_eq!(body["model"], "gpt-5.1");
        assert!(body["id"].as_str().unwrap().starts_with("resp_"));
    }

    #[tokio::test]
    async fn returns_too_many_requests_when_queue_at_cap() {
        let h = Harness::new(1);
        let first = handle_background_create(h.deps_with_repo(), &bg_req(), "gpt-5.1").await;
        assert_eq!(first.status(), StatusCode::OK);
        let second = handle_background_create(h.deps_with_repo(), &bg_req(), "gpt-5.1").await;
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
        let body = body_json(second).await;
        assert_eq!(body["error"]["code"], "queue_full");
    }

    #[tokio::test]
    async fn returns_not_found_when_previous_response_missing() {
        let h = Harness::new(10);
        let mut req = bg_req();
        req.previous_response_id = Some("resp_missing".to_string());
        let resp = handle_background_create(h.deps_with_repo(), &req, "gpt-5.1").await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = body_json(resp).await;
        assert_eq!(body["error"]["code"], "previous_response_not_found");
    }

    #[tokio::test]
    async fn returns_conflict_when_previous_response_still_queued() {
        let h = Harness::new(10);
        // First background create leaves r1 in status=queued in the mirrored
        // response storage. Chaining to it must fail with `not_ready`.
        let mut first = bg_req();
        let resp = handle_background_create(h.deps_with_repo(), &first, "gpt-5.1").await;
        assert_eq!(resp.status(), StatusCode::OK);
        let first_body = body_json(resp).await;
        let first_id = first_body["id"].as_str().unwrap().to_string();

        first = bg_req();
        first.previous_response_id = Some(first_id);
        let resp = handle_background_create(h.deps_with_repo(), &first, "gpt-5.1").await;
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let body = body_json(resp).await;
        assert_eq!(body["error"]["code"], "previous_response_not_ready");
    }

    #[tokio::test]
    async fn returns_conflict_when_previous_response_cancelled() {
        let h = Harness::new(10);
        // Seed the response storage with a cancelled response manually.
        use smg_data_connector::ResponseStorage;
        let mut prior = StoredResponse::new(None);
        prior.id = ResponseId::from("resp_cancelled");
        prior.raw_response = json!({"id": "resp_cancelled", "status": "cancelled"});
        h.response_storage.store_response(prior).await.unwrap();

        let mut req = bg_req();
        req.previous_response_id = Some("resp_cancelled".to_string());
        let resp = handle_background_create(h.deps_with_repo(), &req, "gpt-5.1").await;
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let body = body_json(resp).await;
        assert_eq!(body["error"]["code"], "previous_response_not_usable");
    }

    async fn seed_conversation(h: &Harness, conv_id_str: &str, item_count: usize) {
        use smg_data_connector::{
            ConversationItemStorage, ConversationStorage, NewConversation, NewConversationItem,
        };
        let conv_id = ConversationId::from(conv_id_str.to_string());
        h.conversation_storage
            .create_conversation(NewConversation {
                id: Some(conv_id.clone()),
                metadata: None,
            })
            .await
            .unwrap();
        for i in 0..item_count {
            let item = h
                .conversation_item_storage
                .create_item(NewConversationItem {
                    id: None,
                    response_id: None,
                    item_type: "message".to_string(),
                    role: Some("user".to_string()),
                    content: json!([{"type": "input_text", "text": format!("turn {i}")}]),
                    status: Some("completed".to_string()),
                })
                .await
                .unwrap();
            h.conversation_item_storage
                .link_item(&conv_id, &item.id, chrono::Utc::now())
                .await
                .unwrap();
        }
    }

    #[tokio::test]
    async fn conversation_snapshot_uses_response_input_output_shape() {
        // Snapshot must carry `ResponseInputOutputItem` (`type`/`content`),
        // not raw `ConversationItem` storage rows (`item_type`/`created_at`).
        let h = Harness::new(10);
        seed_conversation(&h, "conv_snapshot", 1).await;

        let mut req = bg_req();
        req.conversation = Some(openai_protocol::common::ConversationRef::Id(
            "conv_snapshot".to_string(),
        ));
        let resp = handle_background_create(h.deps_with_repo(), &req, "gpt-5.1").await;
        assert_eq!(resp.status(), StatusCode::OK);

        let job =
            h.bg.claim_next(
                "test-worker",
                chrono::Utc::now(),
                std::time::Duration::from_secs(30),
            )
            .await
            .unwrap()
            .expect("enqueued job is claimable");
        let input = job.input.as_array().expect("snapshot is an array");
        let first = &input[0];
        assert_eq!(
            first["type"], "message",
            "conversation item should use ResponseInputOutputItem shape"
        );
        assert_eq!(first["role"], "user");
        assert_eq!(
            first["content"][0]["type"], "input_text",
            "content parts should be preserved"
        );
        assert!(
            first.get("item_type").is_none(),
            "ConversationItem shape (item_type) must not leak into the snapshot"
        );
    }

    #[tokio::test]
    async fn conversation_too_large_returns_conflict() {
        let h = Harness::new(10);
        seed_conversation(&h, "conv_overflow", MAX_SNAPSHOT_ITEMS + 1).await;

        let mut req = bg_req();
        req.conversation = Some(openai_protocol::common::ConversationRef::Id(
            "conv_overflow".to_string(),
        ));
        let resp = handle_background_create(h.deps_with_repo(), &req, "gpt-5.1").await;
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let body = body_json(resp).await;
        assert_eq!(body["error"]["code"], "conversation_too_large");
    }

    #[tokio::test]
    async fn conversation_with_reasoning_rows_below_cap_is_accepted() {
        // The cap counts replayable items, not raw storage rows. Seeding a
        // conversation with more storage rows than the cap but where
        // reasoning items take the excess must still succeed.
        use smg_data_connector::{ConversationItemStorage, NewConversationItem};
        let h = Harness::new(10);
        seed_conversation(&h, "conv_mixed", MAX_SNAPSHOT_ITEMS - 1).await;
        let conv_id = ConversationId::from("conv_mixed".to_string());
        // Append two reasoning rows — they are dropped by the converter,
        // pushing raw row count to MAX_SNAPSHOT_ITEMS + 1 while the
        // replayable count stays at MAX_SNAPSHOT_ITEMS - 1.
        for _ in 0..2 {
            let item = h
                .conversation_item_storage
                .create_item(NewConversationItem {
                    id: None,
                    response_id: None,
                    item_type: "reasoning".to_string(),
                    role: None,
                    content: json!({"summary": []}),
                    status: None,
                })
                .await
                .unwrap();
            h.conversation_item_storage
                .link_item(&conv_id, &item.id, chrono::Utc::now())
                .await
                .unwrap();
        }

        let mut req = bg_req();
        req.conversation = Some(openai_protocol::common::ConversationRef::Id(
            "conv_mixed".to_string(),
        ));
        let resp = handle_background_create(h.deps_with_repo(), &req, "gpt-5.1").await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn priority_is_propagated_to_enqueue() {
        let h = Harness::new(10);
        let mut req = bg_req();
        req.priority = 7;
        let resp = handle_background_create(h.deps_with_repo(), &req, "gpt-5.1").await;
        assert_eq!(resp.status(), StatusCode::OK);

        let job =
            h.bg.claim_next(
                "test-worker",
                chrono::Utc::now(),
                std::time::Duration::from_secs(30),
            )
            .await
            .unwrap()
            .expect("enqueued job is claimable");
        assert_eq!(job.priority, 7);
    }
}
