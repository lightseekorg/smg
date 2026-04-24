//! Background create path: resolve input snapshot + enqueue.
//!
//! Called from each router entry point (HTTP regular, OpenAI, gRPC regular)
//! when `background=true`. Non-streaming only in BGM-PR-04.

use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::responses::ResponsesRequest;
use serde_json::{json, Value};
use smg_data_connector::{
    BackgroundRepositoryError, BackgroundResponseRepository, ConversationId,
    ConversationItemStorage, ConversationStorage, EnqueueRequest, ResponseId, ResponseStorage,
    StoredResponse,
};
use uuid::Uuid;

use crate::{config::BackgroundConfig, routers::error};

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

    if request.store != Some(true) {
        return error::bad_request(
            "background_requires_store",
            "Background mode requires 'store' to be true.",
        );
    }

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
        0,
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
    match repository.enqueue(enqueue_req, None, Some(max_depth)).await {
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
            "previous_response_too_large",
            format!(
                "Resolved snapshot has {} items, exceeds Phase 1 cap of {}.",
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

    let list_params = smg_data_connector::ListParams {
        limit: MAX_SNAPSHOT_ITEMS,
        order: smg_data_connector::SortOrder::Asc,
        after: None,
    };
    match item_storage.list_items(&conv_id, list_params).await {
        Ok(conv_items) => {
            for ci in conv_items.into_iter().take(MAX_SNAPSHOT_ITEMS) {
                if let Ok(v) = serde_json::to_value(&ci) {
                    items.push(v);
                }
            }
            Ok(())
        }
        Err(e) => Err(error::internal_error(
            "load_conversation_items_failed",
            format!("Failed to load conversation items for '{conv_id_str}': {e}"),
        )),
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
            }
        }

        fn deps_without_repo(&self) -> BackgroundCreateDeps<'_> {
            BackgroundCreateDeps {
                repository: None,
                response_storage: self.response_storage.as_ref(),
                conversation_storage: self.conversation_storage.as_ref(),
                conversation_item_storage: self.conversation_item_storage.as_ref(),
                background_config: &self.config,
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
    async fn returns_bad_request_when_store_not_true() {
        let h = Harness::new(10);
        let mut req = bg_req();
        req.store = Some(false);
        let resp = handle_background_create(h.deps_with_repo(), &req, "gpt-5.1").await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert_eq!(body["error"]["code"], "background_requires_store");
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
}
