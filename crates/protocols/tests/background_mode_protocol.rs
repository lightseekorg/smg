//! Protocol-surface contract tests for background-mode responses (BGM-PR-01, narrow scope).
//!
//! Covers only the additive changes this PR makes:
//!
//! - `ResponseStatus::Incomplete` serializes as `"incomplete"` and round-trips.
//! - `reasoning` items (both input and output variants) round-trip
//!   `encrypted_content`.
//! - `ResponsesResponse` exposes `background`, `completed_at`, and
//!   `conversation` optional fields that serde round-trip.
//! - `ResponsesResponseBuilder::copy_from_request` propagates `background`
//!   and `conversation` from the request.
//! - `ResponsesResponse::is_incomplete()` reports the new status.
//!
//! Intentionally out of scope (deferred to later PRs):
//! - Strict typing of `incomplete_details` (still `Option<Value>`).
//! - Any change to the `validate_responses_cross_parameters` validator.

use openai_protocol::responses::{
    ResponseInputOutputItem, ResponseOutputItem, ResponseReasoningContent, ResponseStatus,
    ResponsesRequest, ResponsesResponse, SummaryTextContent,
};
use serde_json::json;

// ---------------------------------------------------------------------------
// ResponseStatus::Incomplete
// ---------------------------------------------------------------------------

#[test]
fn response_status_incomplete_serializes_snake_case() {
    let s = serde_json::to_string(&ResponseStatus::Incomplete).expect("serialize");
    assert_eq!(s, "\"incomplete\"");

    let back: ResponseStatus = serde_json::from_str("\"incomplete\"").expect("deserialize");
    assert_eq!(back, ResponseStatus::Incomplete);
}

// ---------------------------------------------------------------------------
// Reasoning encrypted_content round-trip
// ---------------------------------------------------------------------------

#[test]
fn reasoning_output_item_round_trips_encrypted_content() {
    let item = ResponseOutputItem::new_reasoning_encrypted(
        "r_1".to_string(),
        vec![SummaryTextContent::SummaryText {
            text: "thought summary".to_string(),
        }],
        vec![ResponseReasoningContent::ReasoningText {
            text: "inner thought".to_string(),
        }],
        "opaque-ciphertext-xyz".to_string(),
        Some("completed".to_string()),
    );

    let v = serde_json::to_value(&item).expect("serialize");
    assert_eq!(v["encrypted_content"], "opaque-ciphertext-xyz");

    let back: ResponseOutputItem = serde_json::from_value(v).expect("deserialize");
    match back {
        ResponseOutputItem::Reasoning {
            encrypted_content, ..
        } => assert_eq!(encrypted_content.as_deref(), Some("opaque-ciphertext-xyz")),
        _ => panic!("expected Reasoning variant"),
    }
}

#[test]
fn reasoning_input_item_deserializes_encrypted_content() {
    let item: ResponseInputOutputItem = serde_json::from_value(json!({
        "type": "reasoning",
        "id": "r_1",
        "summary": [],
        "encrypted_content": "ct-abc",
    }))
    .expect("deserialize");
    match item {
        ResponseInputOutputItem::Reasoning {
            encrypted_content, ..
        } => assert_eq!(encrypted_content.as_deref(), Some("ct-abc")),
        _ => panic!("expected Reasoning variant"),
    }
}

// ---------------------------------------------------------------------------
// ResponsesResponse: background, completed_at, conversation
// ---------------------------------------------------------------------------

#[test]
fn responses_response_round_trips_new_fields() {
    let resp = ResponsesResponse::builder("resp_xyz", "gpt-5.4")
        .status(ResponseStatus::Completed)
        .background(true)
        .completed_at(1_700_000_000)
        .conversation("conv_123")
        .build();

    let v = serde_json::to_value(&resp).expect("serialize");
    assert_eq!(v["background"], true);
    assert_eq!(v["completed_at"], 1_700_000_000);
    assert_eq!(v["conversation"], "conv_123");

    let back: ResponsesResponse = serde_json::from_value(v).expect("deserialize");
    assert_eq!(back.background, Some(true));
    assert_eq!(back.completed_at, Some(1_700_000_000));
    assert_eq!(back.conversation.as_deref(), Some("conv_123"));
}

#[test]
fn responses_response_new_fields_absent_when_unset() {
    // ResponsesResponse uses `#[serde_with::skip_serializing_none]`, so Option
    // fields left as None are omitted from the wire output rather than emitted
    // as `null`. This test locks that contract.
    let resp = ResponsesResponse::builder("resp_xyz", "gpt-5.4")
        .status(ResponseStatus::InProgress)
        .build();

    let v = serde_json::to_value(&resp).expect("serialize");
    let obj = v.as_object().expect("object");
    assert!(
        !obj.contains_key("background"),
        "background must be omitted"
    );
    assert!(
        !obj.contains_key("completed_at"),
        "completed_at must be omitted"
    );
    assert!(
        !obj.contains_key("conversation"),
        "conversation must be omitted"
    );
}

#[test]
fn responses_response_deserializes_without_new_fields() {
    // Existing wire payloads that predate this PR must still deserialize —
    // the three new fields are `#[serde(default)]` on the struct.
    let v = json!({
        "id": "resp_legacy",
        "object": "response",
        "created_at": 1_699_000_000i64,
        "status": "completed",
        "model": "gpt-4",
    });
    let resp: ResponsesResponse = serde_json::from_value(v).expect("deserialize legacy payload");
    assert_eq!(resp.background, None);
    assert_eq!(resp.completed_at, None);
    assert_eq!(resp.conversation, None);
    assert_eq!(resp.status, ResponseStatus::Completed);
}

// ---------------------------------------------------------------------------
// Builder propagation + helpers
// ---------------------------------------------------------------------------

#[test]
fn copy_from_request_propagates_background_and_conversation() {
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": "hello",
        "background": true,
        "conversation": "conv_abc",
    }))
    .expect("deserialize");
    let resp = ResponsesResponse::builder("resp_xyz", "gpt-5.4")
        .copy_from_request(&request)
        .build();
    assert_eq!(resp.background, Some(true));
    assert_eq!(resp.conversation.as_deref(), Some("conv_abc"));
}

#[test]
fn copy_from_request_accepts_conversation_object_form() {
    // P6: `conversation` accepts the spec's object form
    // `{ id: string }` (ResponseConversationParam). The ResponsesResponse
    // builder flattens either wire shape down to the underlying id.
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": "hello",
        "conversation": { "id": "conv_obj" },
    }))
    .expect("deserialize object form");
    let resp = ResponsesResponse::builder("resp_xyz", "gpt-5.4")
        .copy_from_request(&request)
        .build();
    assert_eq!(resp.conversation.as_deref(), Some("conv_obj"));
}

#[test]
fn is_incomplete_helper() {
    let resp = ResponsesResponse::builder("resp_xyz", "gpt-5.4")
        .status(ResponseStatus::Incomplete)
        .build();
    assert!(resp.is_incomplete());
    assert!(!resp.is_complete());
    assert!(!resp.is_failed());
}
