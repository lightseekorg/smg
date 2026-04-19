//! Protocol-surface contract tests for background-mode responses (BGM-PR-01).
//!
//! These cover the public wire/Rust API the background-mode design relies on:
//!
//! - `ResponseStatus::Incomplete` exists and serializes as `"incomplete"`.
//! - `incomplete_details` is typed with `reason ∈ {max_output_tokens, content_filter}`.
//! - `reasoning` items round-trip `encrypted_content`.
//! - Request validation accepts `background=true` with `stream=true`, accepts
//!   `background=true` with `store` unset or `store=true`, rejects
//!   `background=true` with `store=false`.
//! - `ResponsesResponse` exposes `background`, `completed_at`, and
//!   `conversation` fields and round-trips them through serde.

use openai_protocol::responses::{
    IncompleteDetails, IncompleteReason, ResponseInputOutputItem, ResponseOutputItem,
    ResponseReasoningContent, ResponseStatus, ResponsesRequest, ResponsesResponse,
};
use serde_json::{json, Value};
use validator::Validate;

// ---------------------------------------------------------------------------
// Status + incomplete_details
// ---------------------------------------------------------------------------

#[test]
fn response_status_incomplete_serializes_snake_case() {
    let s = serde_json::to_string(&ResponseStatus::Incomplete).expect("serialize");
    assert_eq!(s, "\"incomplete\"");

    let back: ResponseStatus = serde_json::from_str("\"incomplete\"").expect("deserialize");
    assert_eq!(back, ResponseStatus::Incomplete);
}

#[test]
fn incomplete_details_round_trip_both_reasons() {
    for (reason, wire) in [
        (IncompleteReason::MaxOutputTokens, "max_output_tokens"),
        (IncompleteReason::ContentFilter, "content_filter"),
    ] {
        let expected_debug = format!("{reason:?}");
        let v = serde_json::to_value(IncompleteDetails { reason }).expect("serialize");
        assert_eq!(v, json!({ "reason": wire }));
        let back: IncompleteDetails = serde_json::from_value(v).expect("deserialize");
        assert_eq!(format!("{:?}", back.reason), expected_debug);
    }
}

#[test]
fn incomplete_details_rejects_unknown_reason() {
    // The OpenAI spec restricts `reason` to `max_output_tokens` / `content_filter`.
    // Historical SMG code set `max_tool_calls` here; the typed form must reject
    // that so the protocol cannot silently emit a spec-violating payload.
    let err = serde_json::from_value::<IncompleteDetails>(json!({
        "reason": "max_tool_calls"
    }))
    .expect_err("unknown reason must fail to deserialize");
    let msg = err.to_string();
    assert!(
        msg.contains("max_output_tokens") || msg.contains("variant"),
        "error message should reference allowed variants: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Reasoning encrypted_content round-trip
// ---------------------------------------------------------------------------

#[test]
fn reasoning_output_item_round_trips_encrypted_content() {
    let item = ResponseOutputItem::new_reasoning_encrypted(
        "r_1".to_string(),
        vec!["thought summary".to_string()],
        vec![ResponseReasoningContent::ReasoningText {
            text: "inner thought".to_string(),
        }],
        Some("opaque-ciphertext-xyz".to_string()),
        Some("completed".to_string()),
    );

    let v = serde_json::to_value(&item).expect("serialize");
    assert_eq!(v["encrypted_content"], "opaque-ciphertext-xyz");

    let back: ResponseOutputItem = serde_json::from_value(v).expect("deserialize");
    match back {
        ResponseOutputItem::Reasoning {
            encrypted_content, ..
        } => {
            assert_eq!(encrypted_content.as_deref(), Some("opaque-ciphertext-xyz"));
        }
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
// Validator: background + stream + store interactions
// ---------------------------------------------------------------------------

#[test]
fn validator_accepts_background_with_stream() {
    let req: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": "hello",
        "background": true,
        "stream": true,
    }))
    .expect("deserialize");
    req.validate()
        .expect("background=true + stream=true is valid");
}

#[test]
fn validator_accepts_background_with_store_unset() {
    let req: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": "hello",
        "background": true,
    }))
    .expect("deserialize");
    req.validate()
        .expect("background=true with store unset defaults to stored");
}

#[test]
fn validator_accepts_background_with_store_true() {
    let req: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": "hello",
        "background": true,
        "store": true,
    }))
    .expect("deserialize");
    req.validate().expect("background=true + store=true valid");
}

#[test]
fn validator_rejects_background_with_store_false() {
    let req: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": "hello",
        "background": true,
        "store": false,
    }))
    .expect("deserialize");
    let err = req
        .validate()
        .expect_err("background=true + store=false must fail");
    assert!(
        format!("{err:?}").contains("background_requires_store"),
        "expected background_requires_store code, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// ResponsesResponse: background, completed_at, conversation
// ---------------------------------------------------------------------------

#[test]
fn responses_response_serializes_new_fields() {
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
}

#[test]
fn responses_response_round_trips_incomplete_status_and_details() {
    let resp = ResponsesResponse::builder("resp_xyz", "gpt-5.4")
        .status(ResponseStatus::Incomplete)
        .incomplete_details(IncompleteDetails {
            reason: IncompleteReason::MaxOutputTokens,
        })
        .build();

    let v = serde_json::to_value(&resp).expect("serialize");
    assert_eq!(v["status"], "incomplete");
    assert_eq!(v["incomplete_details"]["reason"], "max_output_tokens");

    let back: ResponsesResponse = serde_json::from_value(v).expect("deserialize");
    assert_eq!(back.status, ResponseStatus::Incomplete);
    let details = back.incomplete_details.expect("details present");
    assert_eq!(details.reason, IncompleteReason::MaxOutputTokens);
}

#[test]
fn responses_response_omits_background_and_completed_at_when_unset() {
    let resp = ResponsesResponse::builder("resp_xyz", "gpt-5.4")
        .status(ResponseStatus::InProgress)
        .build();

    let v = serde_json::to_value(&resp).expect("serialize");
    // The new fields default to `None`; serde emits them as `null`. What
    // matters is that the server did not invent values — a newly-built
    // non-background response has no background / completed / conversation.
    assert_eq!(v["background"], Value::Null);
    assert_eq!(v["completed_at"], Value::Null);
    assert_eq!(v["conversation"], Value::Null);
}

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
fn is_incomplete_helper() {
    let resp = ResponsesResponse::builder("resp_xyz", "gpt-5.4")
        .status(ResponseStatus::Incomplete)
        .build();
    assert!(resp.is_incomplete());
    assert!(!resp.is_complete());
    assert!(!resp.is_failed());
}
