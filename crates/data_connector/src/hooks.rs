//! Storage hook trait and supporting types.
//!
//! Hooks let teams inject custom logic (audit, tenancy, field population,
//! validation) before and after storage operations without forking the
//! codebase.  A hook receives a [`StorageOperation`] discriminant plus a
//! JSON-serialised payload and returns either a continuation signal (with
//! optional [`HookWrites`]) or a rejection.
//!
//! The hook trait is intentionally coarse-grained: a single `before`/`after`
//! pair dispatched by operation enum, rather than one method per storage
//! operation.  This keeps the trait small, avoids a combinatorial explosion
//! of default methods, and maps directly to WASM guest exports (Phase 2b).

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;

use crate::context::RequestContext;

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

/// Key-value bag for extra columns that hooks can read/write.
///
/// On writes, the hook populates values (e.g. `"EXPIRES_AT" → "2099-01-01"`)
/// and the backend persists them to the extra columns declared in schema config.
/// On reads, the backend fills the bag from stored extra column values so the
/// hook can inspect them.
pub type ExtraColumns = HashMap<String, Value>;

/// A single insert row targeting a configured extra table.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExtraTableWrite {
    /// Logical extra-table name from `SchemaConfig.extra_tables`.
    pub table: String,
    /// Column values for a single inserted row.
    pub row: HashMap<String, Value>,
}

/// Hook-produced write data forwarded to storage backends.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HookWrites {
    /// Extra columns to persist on the main storage table.
    pub extra_columns: ExtraColumns,
    /// Additional rows to insert into configured side tables.
    pub extra_table_writes: Vec<ExtraTableWrite>,
}

impl From<ExtraColumns> for HookWrites {
    fn from(extra_columns: ExtraColumns) -> Self {
        Self {
            extra_columns,
            extra_table_writes: Vec::new(),
        }
    }
}

/// Identifies which storage operation is being hooked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageOperation {
    // ── ConversationStorage ──────────────────────────────────────────────
    CreateConversation,
    GetConversation,
    UpdateConversation,
    DeleteConversation,

    // ── ConversationItemStorage ──────────────────────────────────────────
    CreateItem,
    LinkItem,
    LinkItems,
    ListItems,
    GetItem,
    IsItemLinked,
    DeleteItem,

    // ── ResponseStorage ──────────────────────────────────────────────────
    StoreResponse,
    GetResponse,
    DeleteResponse,
    GetResponseChain,
    ListIdentifierResponses,
    DeleteIdentifierResponses,
}

/// Result from a before-hook. Controls whether the backend operation proceeds.
#[derive(Debug)]
pub enum BeforeHookResult {
    /// Proceed with the operation.  The [`HookWrites`] payload is forwarded to
    /// the backend so it can persist hook-provided values alongside core data.
    Continue(HookWrites),

    /// Abort the operation and return an error to the caller.
    Reject(String),
}

impl Default for BeforeHookResult {
    fn default() -> Self {
        Self::Continue(HookWrites::default())
    }
}

/// Errors returned by hook implementations.
#[derive(Debug, thiserror::Error)]
pub enum HookError {
    /// The hook explicitly rejected the operation.
    #[error("hook rejected: {0}")]
    Rejected(String),

    /// An internal hook error (logged, operation continues by default).
    #[error("hook error: {0}")]
    Internal(String),
}

// ────────────────────────────────────────────────────────────────────────────
// Trait
// ────────────────────────────────────────────────────────────────────────────

/// Trait for storage operation hooks.
///
/// Implementors intercept storage operations to inject custom logic such as
/// audit logging, field population, PII redaction, or multi-tenancy filtering.
///
/// # Error Handling
///
/// - `before()` returning `Ok(BeforeHookResult::Reject(_))` aborts the operation.
/// - `before()` returning `Err(_)` logs a warning and **continues** (non-fatal).
/// - `after()` returning `Err(_)` logs a warning and **continues** (non-fatal).
///
/// This ensures hooks cannot accidentally break storage operations unless they
/// explicitly intend to via `Reject`.
#[async_trait]
pub trait StorageHook: Send + Sync + 'static {
    /// Called before a storage operation executes.
    ///
    /// `payload` is a JSON-serialised representation of the operation arguments
    /// (e.g. a `NewConversation` for `CreateConversation`).
    async fn before(
        &self,
        operation: StorageOperation,
        context: Option<&RequestContext>,
        payload: &Value,
    ) -> Result<BeforeHookResult, HookError>;

    /// Called after a storage operation completes successfully.
    ///
    /// `payload` is the same JSON from `before`.  `result` is the
    /// JSON-serialised operation result.  `writes` contains the hook writes
    /// from `before`.  The returned [`HookWrites`] can be used by the caller
    /// (e.g. to surface hook-produced data in API responses).
    ///
    /// Note: only `extra_columns` from the returned `HookWrites` are used;
    /// `extra_table_writes` are executed exclusively during the before phase.
    async fn after(
        &self,
        operation: StorageOperation,
        context: Option<&RequestContext>,
        payload: &Value,
        result: &Value,
        writes: &HookWrites,
    ) -> Result<HookWrites, HookError>;
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn before_hook_result_default_is_continue_with_empty_extra() {
        let result = BeforeHookResult::default();
        match result {
            BeforeHookResult::Continue(writes) => {
                assert!(writes.extra_columns.is_empty());
                assert!(writes.extra_table_writes.is_empty());
            }
            BeforeHookResult::Reject(_) => panic!("expected Continue"),
        }
    }

    #[test]
    fn storage_operation_is_copy() {
        let op = StorageOperation::CreateConversation;
        let op2 = op; // Copy
        assert_eq!(op, op2);
    }

    #[test]
    fn hook_error_display() {
        let err = HookError::Rejected("bad input".to_string());
        assert_eq!(err.to_string(), "hook rejected: bad input");

        let err = HookError::Internal("timeout".to_string());
        assert_eq!(err.to_string(), "hook error: timeout");
    }
}
