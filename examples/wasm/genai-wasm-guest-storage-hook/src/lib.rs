//! WASM Guest Storage Hook Example for Shepherd Model Gateway
//!
//! Customized to support Oracle extra column `GENERATIVE_AI_PROJECT_ID`.
//!
//! Behavior:
//! - For `CreateConversation`, if the request context contains
//!   `generative_ai_project_id` (populated by the HTTP layer from the
//!   `OpenAI-Project` header), this hook writes an extra
//!   column `GENERATIVE_AI_PROJECT_ID` with that value.
//! - For `StoreResponse`, if the request context contains
//!   `generative_ai_project_id`, write it to the same
//!   `GENERATIVE_AI_PROJECT_ID` extra column.
//! - Existing multi-tenant / audit examples for `TENANT_ID` and
//!   `CREATED_BY` have been removed to keep this hook focused on
//!   storage context propagation only.

wit_bindgen::generate!({
    path: "../../../crates/wasm/src/interface/storage",
    world: "storage-hook",
});

use exports::smg::storage::{
    storage_hook_after::Guest as AfterGuest,
    storage_hook_before::Guest as BeforeGuest,
};
use smg::storage::storage_hook_types::{BeforeResult, ContextEntry, ExtraColumn, Operation};

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
        let generative_ai_project_id =
            find_context_value(&context, "generative_ai_project_id");

        match op {
            // For StoreResponse, map context key `generative_ai_project_id`
            // into Oracle extra column `GENERATIVE_AI_PROJECT_ID` when present.
            Operation::StoreResponse => {
                let mut extra = Vec::new();
                if let Some(project_id) = generative_ai_project_id.clone() {
                    extra.push(ExtraColumn {
                        name: "GENERATIVE_AI_PROJECT_ID".to_string(),
                        value: project_id,
                    });
                }
                BeforeResult::DoContinue(extra)
            }

            // For CreateConversation, map context key `generative_ai_project_id`
            // into Oracle extra column `GENERATIVE_AI_PROJECT_ID`.
            Operation::CreateConversation => {
                let mut extra = Vec::new();
                if let Some(project_id) = generative_ai_project_id {
                    // Column name must match SchemaConfig.extra_columns key
                    // in crates/data_connector/oracle_schema_config.yaml
                    extra.push(ExtraColumn {
                        name: "GENERATIVE_AI_PROJECT_ID".to_string(),
                        value: project_id,
                    });
                }
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
