//! WASM Guest Storage Hook Example for Shepherd Model Gateway
//!
//! Demonstrates a storage hook that:
//! - Adds a `TENANT_ID` extra column from the request context on writes
//! - Adds a `CREATED_BY` extra column on conversation creation
//! - Rejects `StoreResponse` if the `tenant_id` context entry is missing
//! - Passes through all other operations unchanged

wit_bindgen::generate!({
    path: "../../../crates/wasm/src/interface/storage",
    world: "storage-hook",
});

use exports::smg::storage::{
    storage_hook_after::Guest as AfterGuest,
    storage_hook_before::Guest as BeforeGuest,
};
use smg::storage::storage_hook_types::{
    BeforeResult, ContextEntry, ExtraColumn, Operation,
};

struct StorageHookImpl;

// ── Helpers ──────────────────────────────────────────────────────────────

fn find_context_value(context: &[ContextEntry], key: &str) -> Option<String> {
    context
        .iter()
        .find(|e| e.key == key)
        .map(|e| e.value.clone())
}

// ── Before hook ──────────────────────────────────────────────────────────

impl BeforeGuest for StorageHookImpl {
    fn before(op: Operation, context: Vec<ContextEntry>, _payload: String) -> BeforeResult {
        let tenant_id = find_context_value(&context, "tenant_id");

        match op {
            // For StoreResponse, require a tenant_id in the context
            Operation::StoreResponse => match tenant_id {
                Some(tid) => {
                    let mut extra = vec![ExtraColumn {
                        name: "TENANT_ID".to_string(),
                        value: tid,
                    }];
                    // Optionally add the user who stored the response
                    if let Some(user) = find_context_value(&context, "user_id") {
                        extra.push(ExtraColumn {
                            name: "STORED_BY".to_string(),
                            value: user,
                        });
                    }
                    BeforeResult::DoContinue(extra)
                }
                None => BeforeResult::Reject(
                    "tenant_id is required in request context for StoreResponse".to_string(),
                ),
            },

            // For CreateConversation, add CREATED_BY if user_id is in context
            Operation::CreateConversation => {
                let mut extra = Vec::new();
                if let Some(tid) = tenant_id {
                    extra.push(ExtraColumn {
                        name: "TENANT_ID".to_string(),
                        value: tid,
                    });
                }
                if let Some(user) = find_context_value(&context, "user_id") {
                    extra.push(ExtraColumn {
                        name: "CREATED_BY".to_string(),
                        value: user,
                    });
                }
                BeforeResult::DoContinue(extra)
            }

            // For CreateItem, pass through tenant_id if available
            Operation::CreateItem | Operation::LinkItem => {
                let extra = match tenant_id {
                    Some(tid) => vec![ExtraColumn {
                        name: "TENANT_ID".to_string(),
                        value: tid,
                    }],
                    None => Vec::new(),
                };
                BeforeResult::DoContinue(extra)
            }

            // All other operations: continue without extra columns
            _ => BeforeResult::DoContinue(Vec::new()),
        }
    }
}

// ── After hook ───────────────────────────────────────────────────────────

impl AfterGuest for StorageHookImpl {
    fn after(
        _op: Operation,
        _context: Vec<ContextEntry>,
        _payload: String,
        _result_json: String,
        extra: Vec<ExtraColumn>,
    ) -> Vec<ExtraColumn> {
        // Pass through the extra columns unchanged.
        // A real implementation might log the operation, update metrics,
        // or enrich the extra columns with post-operation data.
        extra
    }
}

export!(StorageHookImpl);
