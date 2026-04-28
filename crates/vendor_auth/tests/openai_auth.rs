//! Unit tests for `OpenAiApiKeyAuth`.
//!
//! Covers tests:
//! - openai_api_key_sets_bearer_and_project
//! - openai_api_key_omits_project_when_unset
//!
//! See design doc § 6.6.

// Test files use unwrap/expect freely on infallible builders. The workspace
// `clippy::unwrap_used = "deny"` rule ordinarily forbids this, but
// `clippy.toml` sets `allow-unwrap-in-tests = true` for `#[test]` bodies.
// Helper functions outside `#[test]` are not auto-allowed; opt the whole
// test crate in explicitly.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use bytes::Bytes;
use http::Request;
use secrecy::SecretString;

use smg_vendor_auth::{OpenAiApiKeyAuth, OutboundAuth};

fn make_request() -> Request<Bytes> {
    Request::builder()
        .method("POST")
        .uri("https://api.openai.com/v1/containers")
        .header("content-type", "application/json")
        .body(Bytes::from_static(b"{\"name\":\"x\"}"))
        .unwrap()
}

#[tokio::test]
async fn openai_api_key_sets_bearer_and_project() {
    let auth =
        OpenAiApiKeyAuth::new(SecretString::from("sk-test-1234".to_owned())).with_project("proj_abc");
    let mut req = make_request();
    auth.apply(&mut req).await.expect("apply succeeds");

    let auth_hdr = req
        .headers()
        .get("authorization")
        .expect("authorization header is present")
        .to_str()
        .unwrap();
    assert_eq!(auth_hdr, "Bearer sk-test-1234");

    let project_hdr = req
        .headers()
        .get("openai-project")
        .expect("openai-project header is present")
        .to_str()
        .unwrap();
    assert_eq!(project_hdr, "proj_abc");
}

#[tokio::test]
async fn openai_api_key_omits_project_when_unset() {
    let auth = OpenAiApiKeyAuth::new(SecretString::from("sk-test".to_owned()));
    let mut req = make_request();
    auth.apply(&mut req).await.expect("apply succeeds");

    assert!(
        req.headers().get("openai-project").is_none(),
        "OpenAI-Project header must be absent when no project_id was set"
    );
    assert!(
        req.headers().get("authorization").is_some(),
        "authorization header still present"
    );
}
