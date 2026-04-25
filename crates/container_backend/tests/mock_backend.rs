//! Tests for the in-memory [`MockBackend`].
//!
//! Used downstream by SMG e2e (CB-3 / CB-4) to drive dispatch flows
//! without hitting the network. Only behavioural contracts that callers
//! depend on are tested here.

// `clippy.toml` sets `allow-{unwrap,expect,panic}-in-tests = true` for `#[test]`
// bodies, but `-D warnings` promotes the workspace `warn` for these to deny.
// This file is a test target; opt out at file scope.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use smg_container_backend::{
    BackendError, ContainerBackend, ContainerStatus, CreateContainerParams, MemoryLimit,
    MockBackend,
};

// =====================================================================
// Test 8 — mock_backend_isolated_state
// Two MockBackend instances must not share state.
// =====================================================================
#[tokio::test]
async fn mock_backend_isolated_state() {
    let a = MockBackend::new();
    let b = MockBackend::new();

    let c = a
        .create(CreateContainerParams {
            name: Some("from-a".into()),
            memory_limit: Some(MemoryLimit::Mem1G),
            ..Default::default()
        })
        .await
        .expect("a.create");

    // Container present in a:
    let got = a.retrieve(&c.id).await.expect("a.retrieve");
    assert_eq!(got.id, c.id);
    assert_eq!(got.status, ContainerStatus::Running);

    // Same id NOT present in b:
    let err = b
        .retrieve(&c.id)
        .await
        .expect_err("b should not see a's container");
    assert!(matches!(err, BackendError::NotFound(_)));

    // b's create produces a fresh id (mock_seq is per-instance):
    let from_b = b.create(CreateContainerParams::default()).await.expect("b.create");
    assert_eq!(from_b.id, "cntr_mock_1");
    assert_eq!(c.id, "cntr_mock_1"); // same prefix, fresh seq per backend
}

#[tokio::test]
async fn mock_backend_round_trip() {
    let backend = MockBackend::new();

    let created = backend
        .create(CreateContainerParams {
            name: Some("test".into()),
            memory_limit: Some(MemoryLimit::Mem4G),
            file_ids: vec!["file_1".into(), "file_2".into()],
            ..Default::default()
        })
        .await
        .expect("create");

    assert!(created.id.starts_with("cntr_mock_"));
    assert_eq!(created.status, ContainerStatus::Running);
    assert_eq!(created.memory_limit, Some(MemoryLimit::Mem4G));
    assert_eq!(created.file_ids, vec!["file_1", "file_2"]);

    let listed = backend
        .list(Default::default())
        .await
        .expect("list");
    assert_eq!(listed.data.len(), 1);
    assert!(!listed.has_more);

    backend.delete(&created.id).await.expect("delete");
    let err = backend.retrieve(&created.id).await.expect_err("gone");
    assert!(matches!(err, BackendError::NotFound(_)));

    // delete is idempotent
    backend.delete(&created.id).await.expect("delete again");
}
