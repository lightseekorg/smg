//! HTTP-level tests for [`OpenAiCompatContainersClient`].
//!
//! Uses `httpmock` to mount a local server, drives the client against it,
//! and asserts on the wire shape (request body + headers) and the
//! response-to-error mapping. Auth is provided by `OpenAiApiKeyAuth` (the
//! simplest [`OutboundAuth`] impl) for tests that don't need OCI signing.
//!
//! See design doc §7.5 for the test matrix and §13.1 for the wire-shape
//! reference.

// `clippy.toml` sets `allow-{unwrap,expect,panic}-in-tests = true` for `#[test]`
// bodies, but `-D warnings` promotes the workspace `warn` for these to deny.
// This file is a test target; opt out at file scope to mirror the pattern in
// `crates/vendor_auth/tests/oci_delegated_auth.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use bytes::Bytes;
use http::Request;
use httpmock::prelude::*;
use reqwest::Client;
use secrecy::SecretString;
use serde_json::json;
use smg_container_backend::{
    BackendError, Container, ContainerBackend, ContainerStatus, CreateContainerParams,
    ListOrder, ListQuery, MemoryLimit, OpenAiCompatContainersClient,
};
use smg_vendor_auth::{AuthError, OpenAiApiKeyAuth, OutboundAuth};
use url::Url;

fn client_for(server: &MockServer, auth: Arc<dyn OutboundAuth>) -> OpenAiCompatContainersClient {
    let base = Url::parse(&server.base_url()).expect("mock base url");
    OpenAiCompatContainersClient::new(auth, base, Client::new())
}

fn api_key_auth() -> Arc<dyn OutboundAuth> {
    Arc::new(OpenAiApiKeyAuth::new(SecretString::from("test-key".to_string())))
}

// =====================================================================
// Test 1 — create_round_trip
// Verifies wire shape per design doc §13.1 and BackendError::Backend
// reachability path is NOT exercised here (200 OK).
// =====================================================================
#[tokio::test]
async fn create_round_trip() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/containers")
            .header("authorization", "Bearer test-key")
            .header("content-type", "application/json")
            // Wire shape per §13.1: the body must serialize to this JSON.
            .json_body(json!({
                "name": "cntr-test-abc",
                "memory_limit": "4g",
                "file_ids": []
            }));
        then.status(200).header("content-type", "application/json").json_body(json!({
            "id": "cntr_xyz",
            "object": "container",
            "created_at": 1745611200,
            "status": "running",
            "name": "cntr-test-abc",
            "memory_limit": "4g",
            "file_ids": []
        }));
    });

    let client = client_for(&server, api_key_auth());
    let params = CreateContainerParams {
        name: Some("cntr-test-abc".into()),
        memory_limit: Some(MemoryLimit::Mem4G),
        ..Default::default()
    };
    let got: Container = client.create(params).await.expect("create succeeds");

    assert_eq!(got.id, "cntr_xyz");
    assert_eq!(got.object, "container");
    assert_eq!(got.status, ContainerStatus::Running);
    assert_eq!(got.memory_limit, Some(MemoryLimit::Mem4G));
    mock.assert();
}

// =====================================================================
// Test 2 — retrieve_404_maps_to_not_found
// Exercises BackendError::NotFound.
// =====================================================================
#[tokio::test]
async fn retrieve_404_maps_to_not_found() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(GET).path("/v1/containers/cntr_missing");
        then.status(404)
            .header("content-type", "application/json")
            .json_body(json!({"error": {"message": "not found"}}));
    });

    let client = client_for(&server, api_key_auth());
    let err = client.retrieve("cntr_missing").await.expect_err("404 surfaces as error");
    match err {
        BackendError::NotFound(body) => {
            assert!(body.contains("not found"), "body should contain server message: {body}");
        }
        other => panic!("expected NotFound, got {other:?}"),
    }
    mock.assert();
}

// =====================================================================
// Test 3 — rate_limit_passes_retry_after
// Exercises BackendError::RateLimited with retry-after parsing.
// =====================================================================
#[tokio::test]
async fn rate_limit_passes_retry_after() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/v1/containers");
        then.status(429)
            .header("retry-after", "30")
            .body("rate limited");
    });

    let client = client_for(&server, api_key_auth());
    let err = client
        .create(CreateContainerParams::default())
        .await
        .expect_err("429 surfaces as error");
    match err {
        BackendError::RateLimited { retry_after_secs } => {
            assert_eq!(retry_after_secs, Some(30));
        }
        other => panic!("expected RateLimited, got {other:?}"),
    }
    mock.assert();
}

// =====================================================================
// Test 4 — unauthorized_401_and_403_collapse
// Exercises BackendError::Unauthorized — both 401 and 403.
// =====================================================================
#[tokio::test]
async fn unauthorized_401_and_403_collapse() {
    // 401 path
    let server_401 = MockServer::start();
    let mock_401 = server_401.mock(|when, then| {
        when.method(GET).path("/v1/containers/cntr_a");
        then.status(401).body("unauthorized");
    });
    let client_401 = client_for(&server_401, api_key_auth());
    let err_401 = client_401.retrieve("cntr_a").await.expect_err("401");
    assert!(
        matches!(err_401, BackendError::Unauthorized),
        "expected Unauthorized for 401, got {err_401:?}"
    );
    mock_401.assert();

    // 403 path
    let server_403 = MockServer::start();
    let mock_403 = server_403.mock(|when, then| {
        when.method(GET).path("/v1/containers/cntr_b");
        then.status(403).body("forbidden");
    });
    let client_403 = client_for(&server_403, api_key_auth());
    let err_403 = client_403.retrieve("cntr_b").await.expect_err("403");
    assert!(
        matches!(err_403, BackendError::Unauthorized),
        "expected Unauthorized for 403, got {err_403:?}"
    );
    mock_403.assert();
}

// =====================================================================
// Test 5 — delete_succeeds_on_204
// Exercises Ok(()) on the delete path.
// =====================================================================
#[tokio::test]
async fn delete_succeeds_on_204() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(DELETE).path("/v1/containers/cntr_xyz");
        then.status(204);
    });

    let client = client_for(&server, api_key_auth());
    client.delete("cntr_xyz").await.expect("204 → Ok(())");
    mock.assert();
}

// =====================================================================
// Test 6 — list_paginates
// Verifies query-param wiring + Page envelope deserialization.
// =====================================================================
#[tokio::test]
async fn list_paginates() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(GET)
            .path("/v1/containers")
            .query_param("limit", "10")
            .query_param("after", "cntr_cursor")
            .query_param("order", "desc");
        then.status(200).header("content-type", "application/json").json_body(json!({
            "object": "list",
            "data": [{
                "id": "cntr_1",
                "object": "container",
                "created_at": 1,
                "status": "running",
                "file_ids": []
            }, {
                "id": "cntr_2",
                "object": "container",
                "created_at": 2,
                "status": "expired",
                "file_ids": []
            }],
            "first_id": "cntr_1",
            "last_id": "cntr_2",
            "has_more": true
        }));
    });

    let client = client_for(&server, api_key_auth());
    let page = client
        .list(ListQuery {
            limit: Some(10),
            after: Some("cntr_cursor".into()),
            before: None,
            order: Some(ListOrder::Desc),
        })
        .await
        .expect("list succeeds");

    assert_eq!(page.object, "list");
    assert_eq!(page.data.len(), 2);
    assert_eq!(page.first_id.as_deref(), Some("cntr_1"));
    assert_eq!(page.last_id.as_deref(), Some("cntr_2"));
    assert!(page.has_more);
    assert_eq!(page.data[0].status, ContainerStatus::Running);
    assert_eq!(page.data[1].status, ContainerStatus::Expired);
    mock.assert();
}

// =====================================================================
// Test 7 — auth_apply_called_once_per_request
// Wraps the real auth in a Counter to assert it's invoked exactly once
// per HTTP send. Guards against double-signing or missed signing.
// =====================================================================
#[derive(Debug)]
struct CountingAuth {
    inner: Arc<dyn OutboundAuth>,
    count: Arc<AtomicUsize>,
}

#[async_trait]
impl OutboundAuth for CountingAuth {
    async fn apply(&self, req: &mut Request<Bytes>) -> Result<(), AuthError> {
        self.count.fetch_add(1, Ordering::SeqCst);
        self.inner.apply(req).await
    }
}

#[tokio::test]
async fn auth_apply_called_once_per_request() {
    let server = MockServer::start();
    let _create_mock = server.mock(|when, then| {
        when.method(POST).path("/v1/containers");
        then.status(200).header("content-type", "application/json").json_body(json!({
            "id": "cntr_z",
            "object": "container",
            "created_at": 1,
            "status": "running",
            "file_ids": []
        }));
    });
    let _retrieve_mock = server.mock(|when, then| {
        when.method(GET).path("/v1/containers/cntr_z");
        then.status(200).header("content-type", "application/json").json_body(json!({
            "id": "cntr_z",
            "object": "container",
            "created_at": 1,
            "status": "running",
            "file_ids": []
        }));
    });

    let count = Arc::new(AtomicUsize::new(0));
    let counting: Arc<dyn OutboundAuth> = Arc::new(CountingAuth {
        inner: api_key_auth(),
        count: count.clone(),
    });
    let client = client_for(&server, counting);

    let _ = client.create(CreateContainerParams::default()).await.expect("create");
    assert_eq!(count.load(Ordering::SeqCst), 1, "create should sign once");

    let _ = client.retrieve("cntr_z").await.expect("retrieve");
    assert_eq!(count.load(Ordering::SeqCst), 2, "retrieve should sign once");
}
