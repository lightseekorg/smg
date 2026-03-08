//! Passthrough WASM Storage Hook for testing.
//!
//! Always continues without rejection. Adds a marker extra column
//! (`HOOK_ACTIVE = "true"`) on write operations so tests can verify
//! the hook pipeline is running.

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

struct PassthroughHook;

impl BeforeGuest for PassthroughHook {
    fn before(op: Operation, _context: Vec<ContextEntry>, _payload: String) -> BeforeResult {
        let extra = match op {
            // Add marker extra column on write operations
            Operation::StoreResponse
            | Operation::CreateConversation
            | Operation::UpdateConversation
            | Operation::CreateItem
            | Operation::LinkItem => {
                vec![ExtraColumn {
                    name: "HOOK_ACTIVE".to_string(),
                    value: "true".to_string(),
                }]
            }
            // Read/delete operations: pass through without extra columns
            _ => Vec::new(),
        };
        BeforeResult::DoContinue(extra)
    }
}

impl AfterGuest for PassthroughHook {
    fn after(
        _op: Operation,
        _context: Vec<ContextEntry>,
        _payload: String,
        _result_json: String,
        extra: Vec<ExtraColumn>,
    ) -> Vec<ExtraColumn> {
        extra
    }
}

export!(PassthroughHook);
