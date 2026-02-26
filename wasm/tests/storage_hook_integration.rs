//! Integration tests for WasmStorageHook using a compiled WASM guest component.
//!
//! These tests load the pre-built `wasm-guest-storage-hook` example and verify
//! the full host↔guest round-trip: Rust types → WIT types → WASM execution →
//! WIT types → Rust types.
//!
//! Run with: `cargo test -p smg-wasm --features storage-hooks`

#![cfg(feature = "storage-hooks")]

use std::{collections::HashMap, path::Path};

use serde_json::json;
use smg_data_connector::{
    context::RequestContext,
    hooks::{BeforeHookResult, ExtraColumns, StorageHook, StorageOperation},
};
use smg_wasm::WasmStorageHook;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/storage_hook_guest.wasm"
);

/// Load the pre-built WASM guest fixture.
///
/// Returns `None` when the fixture has not been compiled (e.g. in CI where the
/// WASM guest targets are not built). Tests should early-return with `Ok(())`
/// so that `cargo test` passes without the fixture present.
///
/// Build the fixture locally with: `./wasm/tests/fixtures/build_fixtures.sh`
fn load_hook() -> Result<Option<WasmStorageHook>, Box<dyn std::error::Error>> {
    let path = Path::new(FIXTURE_PATH);
    if !path.exists() {
        // Fixture not built — skip silently. Build with:
        // ./wasm/tests/fixtures/build_fixtures.sh
        return Ok(None);
    }
    let wasm_bytes = std::fs::read(path)?;
    Ok(Some(WasmStorageHook::new(&wasm_bytes)?))
}

/// Early-return from a test when the WASM fixture is not available.
macro_rules! require_hook {
    () => {
        match load_hook()? {
            Some(hook) => hook,
            None => return Ok(()),
        }
    };
}

// ── StoreResponse tests ─────────────────────────────────────────────────

#[tokio::test]
async fn store_response_with_tenant_id_continues_with_extra_columns() -> TestResult {
    let hook = require_hook!();

    let mut ctx_data = HashMap::new();
    ctx_data.insert("tenant_id".into(), "acme-corp".into());
    ctx_data.insert("user_id".into(), "user_42".into());
    let ctx = RequestContext::with_data(ctx_data);

    let payload = json!({"id": "resp_123"});
    let result = hook
        .before(StorageOperation::StoreResponse, Some(&ctx), &payload)
        .await?;

    match result {
        BeforeHookResult::Continue(extra) => {
            assert_eq!(
                extra.get("TENANT_ID").and_then(|v| v.as_str()),
                Some("acme-corp"),
                "TENANT_ID should come from context"
            );
            assert_eq!(
                extra.get("STORED_BY").and_then(|v| v.as_str()),
                Some("user_42"),
                "STORED_BY should come from user_id in context"
            );
        }
        BeforeHookResult::Reject(reason) => {
            panic!("expected Continue, got Reject: {reason}");
        }
    }
    Ok(())
}

#[tokio::test]
async fn store_response_without_tenant_id_is_rejected() -> TestResult {
    let hook = require_hook!();

    let payload = json!({"id": "resp_123"});
    let result = hook
        .before(StorageOperation::StoreResponse, None, &payload)
        .await?;

    match result {
        BeforeHookResult::Reject(reason) => {
            assert!(
                reason.contains("tenant_id"),
                "rejection should mention tenant_id, got: {reason}"
            );
        }
        BeforeHookResult::Continue(_) => {
            panic!("expected Reject without tenant_id");
        }
    }
    Ok(())
}

// ── CreateConversation tests ────────────────────────────────────────────

#[tokio::test]
async fn create_conversation_adds_created_by_from_context() -> TestResult {
    let hook = require_hook!();

    let mut ctx_data = HashMap::new();
    ctx_data.insert("tenant_id".into(), "acme-corp".into());
    ctx_data.insert("user_id".into(), "admin".into());
    let ctx = RequestContext::with_data(ctx_data);

    let payload = json!({});
    let result = hook
        .before(StorageOperation::CreateConversation, Some(&ctx), &payload)
        .await?;

    match result {
        BeforeHookResult::Continue(extra) => {
            assert_eq!(
                extra.get("TENANT_ID").and_then(|v| v.as_str()),
                Some("acme-corp"),
            );
            assert_eq!(
                extra.get("CREATED_BY").and_then(|v| v.as_str()),
                Some("admin"),
            );
        }
        BeforeHookResult::Reject(reason) => {
            panic!("expected Continue, got Reject: {reason}");
        }
    }
    Ok(())
}

// ── Passthrough operations ──────────────────────────────────────────────

#[tokio::test]
async fn get_response_passes_through_without_extra_columns() -> TestResult {
    let hook = require_hook!();

    let payload = json!({"id": "resp_123"});
    let result = hook
        .before(StorageOperation::GetResponse, None, &payload)
        .await?;

    match result {
        BeforeHookResult::Continue(extra) => {
            assert!(extra.is_empty(), "read ops should not add extra columns");
        }
        BeforeHookResult::Reject(reason) => {
            panic!("read ops should not reject: {reason}");
        }
    }
    Ok(())
}

// ── After hook ──────────────────────────────────────────────────────────

#[tokio::test]
async fn after_hook_passes_through_extra_columns() -> TestResult {
    let hook = require_hook!();

    let mut extra = ExtraColumns::new();
    extra.insert("TENANT_ID".into(), json!("acme-corp"));

    let payload = json!({"id": "resp_123"});
    let result_json = json!({"stored": true});

    let updated = hook
        .after(
            StorageOperation::StoreResponse,
            None,
            &payload,
            &result_json,
            &extra,
        )
        .await?;

    assert_eq!(
        updated.get("TENANT_ID").and_then(|v| v.as_str()),
        Some("acme-corp"),
        "after() should pass through extra columns"
    );
    Ok(())
}

// ── Schema config alignment tests ─────────────────────────────────────

#[tokio::test]
async fn wasm_hook_extra_columns_match_schema_config_declarations() -> TestResult {
    let hook = require_hook!();

    let mut ctx_data = HashMap::new();
    ctx_data.insert("tenant_id".into(), "acme-corp".into());
    ctx_data.insert("user_id".into(), "user_42".into());
    let ctx = RequestContext::with_data(ctx_data);

    let payload = json!({"id": "resp_456"});
    let result = hook
        .before(StorageOperation::StoreResponse, Some(&ctx), &payload)
        .await?;

    match result {
        BeforeHookResult::Continue(extra) => {
            // A typical SchemaConfig would declare these extra columns for
            // multi-tenant response storage.  Verify the hook output keys
            // align with what the schema config expects.
            let extra_keys: std::collections::HashSet<&String> = extra.keys().collect();
            let expected_keys: std::collections::HashSet<String> = ["TENANT_ID", "STORED_BY"]
                .iter()
                .map(|s| (*s).to_string())
                .collect();
            let expected_refs: std::collections::HashSet<&String> = expected_keys.iter().collect();

            assert_eq!(
                extra_keys, expected_refs,
                "hook extra columns keys should match schema config declarations"
            );
        }
        BeforeHookResult::Reject(reason) => {
            panic!("expected Continue, got Reject: {reason}");
        }
    }
    Ok(())
}

// ── Multi-operation tests ─────────────────────────────────────────────

#[tokio::test]
async fn wasm_hook_works_across_multiple_operation_types() -> TestResult {
    let hook = require_hook!();

    // 1. StoreResponse with tenant_id → Continue with TENANT_ID + STORED_BY
    let mut store_ctx_data = HashMap::new();
    store_ctx_data.insert("tenant_id".into(), "acme-corp".into());
    store_ctx_data.insert("user_id".into(), "user_42".into());
    let store_ctx = RequestContext::with_data(store_ctx_data);

    let store_payload = json!({"id": "resp_001"});
    let store_result = hook
        .before(
            StorageOperation::StoreResponse,
            Some(&store_ctx),
            &store_payload,
        )
        .await?;

    match &store_result {
        BeforeHookResult::Continue(extra) => {
            assert!(
                extra.contains_key("TENANT_ID"),
                "StoreResponse should include TENANT_ID"
            );
            assert!(
                extra.contains_key("STORED_BY"),
                "StoreResponse should include STORED_BY"
            );
        }
        BeforeHookResult::Reject(reason) => {
            panic!("StoreResponse expected Continue, got Reject: {reason}");
        }
    }

    // 2. CreateConversation with user_id → Continue with CREATED_BY
    let mut conv_ctx_data = HashMap::new();
    conv_ctx_data.insert("tenant_id".into(), "acme-corp".into());
    conv_ctx_data.insert("user_id".into(), "admin".into());
    let conv_ctx = RequestContext::with_data(conv_ctx_data);

    let conv_payload = json!({});
    let conv_result = hook
        .before(
            StorageOperation::CreateConversation,
            Some(&conv_ctx),
            &conv_payload,
        )
        .await?;

    match &conv_result {
        BeforeHookResult::Continue(extra) => {
            assert!(
                extra.contains_key("CREATED_BY"),
                "CreateConversation should include CREATED_BY"
            );
        }
        BeforeHookResult::Reject(reason) => {
            panic!("CreateConversation expected Continue, got Reject: {reason}");
        }
    }

    // 3. GetResponse with no special context → Continue with empty extra columns
    let get_payload = json!({"id": "resp_001"});
    let get_result = hook
        .before(StorageOperation::GetResponse, None, &get_payload)
        .await?;

    match &get_result {
        BeforeHookResult::Continue(extra) => {
            assert!(
                extra.is_empty(),
                "GetResponse should return empty extra columns"
            );
        }
        BeforeHookResult::Reject(reason) => {
            panic!("GetResponse expected Continue, got Reject: {reason}");
        }
    }

    Ok(())
}

// ── Before→after round-trip tests ─────────────────────────────────────

#[tokio::test]
async fn wasm_hook_after_receives_before_extra_columns() -> TestResult {
    let hook = require_hook!();

    // Run before() to get extra columns
    let mut ctx_data = HashMap::new();
    ctx_data.insert("tenant_id".into(), "acme-corp".into());
    ctx_data.insert("user_id".into(), "user_42".into());
    let ctx = RequestContext::with_data(ctx_data);

    let payload = json!({"id": "resp_789"});
    let before_result = hook
        .before(StorageOperation::StoreResponse, Some(&ctx), &payload)
        .await?;

    let before_extra = match before_result {
        BeforeHookResult::Continue(extra) => extra,
        BeforeHookResult::Reject(reason) => {
            panic!("expected Continue from before(), got Reject: {reason}");
        }
    };

    // Pass before() extra columns into after()
    let result_json = json!({"stored": true});
    let after_extra = hook
        .after(
            StorageOperation::StoreResponse,
            Some(&ctx),
            &payload,
            &result_json,
            &before_extra,
        )
        .await?;

    // Verify after() returns the extra columns unchanged
    assert_eq!(
        after_extra.get("TENANT_ID").and_then(|v| v.as_str()),
        Some("acme-corp"),
        "after() should preserve TENANT_ID from before()"
    );
    assert_eq!(
        after_extra.get("STORED_BY").and_then(|v| v.as_str()),
        Some("user_42"),
        "after() should preserve STORED_BY from before()"
    );
    assert_eq!(
        before_extra.len(),
        after_extra.len(),
        "after() should return the same number of extra columns as before()"
    );

    Ok(())
}

// ── Rejection tests ───────────────────────────────────────────────────

#[tokio::test]
async fn wasm_hook_rejection_includes_reason() -> TestResult {
    let hook = require_hook!();

    // Call StoreResponse WITHOUT tenant_id in context
    let payload = json!({"id": "resp_no_tenant"});
    let result = hook
        .before(StorageOperation::StoreResponse, None, &payload)
        .await?;

    match result {
        BeforeHookResult::Reject(reason) => {
            assert!(!reason.is_empty(), "rejection reason should not be empty");
            assert!(
                reason.contains("tenant_id"),
                "rejection reason should mention tenant_id, got: {reason}"
            );
        }
        BeforeHookResult::Continue(_) => {
            panic!("expected Reject without tenant_id, got Continue");
        }
    }
    Ok(())
}
