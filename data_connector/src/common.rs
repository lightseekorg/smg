use std::collections::HashMap;

use serde_json::Value;

use crate::{
    core::ConversationMetadata,
    hooks::ExtraColumns,
    schema::{SchemaConfig, TableConfig},
};

/// Logical column names for the responses table, in canonical SELECT order.
///
/// Shared between Oracle and Postgres backends to build dynamic SELECT queries.
/// The order here doesn't affect correctness (both backends read by name, not
/// position), but having a single source prevents accidental divergence.
pub(super) const RESPONSE_COLUMNS: &[&str] = &[
    "id",
    "conversation_id",
    "previous_response_id",
    "input",
    "instructions",
    "output",
    "tool_calls",
    "metadata",
    "created_at",
    "safety_identifier",
    "model",
    "raw_response",
];

/// Build the `SELECT col1, col2, ... FROM table` base query for responses.
///
/// Used by Oracle and Postgres to pre-build the SELECT prefix at construction
/// time, avoiding repeated string formatting on every query. Respects
/// `skip_columns` (omits them). Extra columns are write-side enrichment only
/// and are NOT included in SELECT.
pub(super) fn build_response_select_base(schema: &SchemaConfig) -> String {
    let s = &schema.responses;
    let table = s.qualified_table(schema.owner.as_deref());
    let cols: Vec<&str> = RESPONSE_COLUMNS
        .iter()
        .filter(|&&logical| !s.is_skipped(logical))
        .map(|&logical| s.col(logical))
        .collect();
    format!("SELECT {} FROM {table}", cols.join(", "))
}

/// Generate DDL column definitions for extra columns.
///
/// Returns e.g. `["TENANT_ID VARCHAR(128)", "EXPIRES_AT TIMESTAMP"]`.
/// Sorted for deterministic DDL generation.
pub(super) fn extra_column_defs(tc: &TableConfig) -> Vec<String> {
    let names = sorted_extra_column_names(tc);
    names
        .iter()
        .filter_map(|name| {
            tc.extra_columns
                .get(*name)
                .map(|def| format!("{name} {}", def.sql_type))
        })
        .collect()
}

/// Get extra column names sorted alphabetically for deterministic SQL generation.
pub(super) fn sorted_extra_column_names(tc: &TableConfig) -> Vec<&str> {
    let mut names: Vec<&str> = tc.extra_columns.keys().map(String::as_str).collect();
    names.sort_unstable();
    names
}

/// Convert a `serde_json::Value` to a SQL-bindable string representation.
///
/// Used by backends to bind extra column values (from hooks or defaults)
/// as text parameters in SQL statements.
pub(super) fn value_to_sql_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Resolve extra column values for a write operation.
///
/// For each extra column defined in the table config, resolves the value from:
/// 1. Hook-provided `ExtraColumns` (highest priority)
/// 2. `ColumnDef.default_value` from schema config
/// 3. `None` (if neither provides a non-null value)
///
/// Returns `(column_name, resolved_value)` pairs in sorted order.
pub(super) fn resolve_extra_column_values<'a>(
    tc: &'a TableConfig,
    hook_extra: &ExtraColumns,
) -> Vec<(&'a str, Option<String>)> {
    sorted_extra_column_names(tc)
        .into_iter()
        .map(|name| {
            let val = hook_extra
                .get(name)
                .filter(|v| !v.is_null())
                .map(value_to_sql_string)
                .or_else(|| {
                    tc.extra_columns
                        .get(name)
                        .and_then(|def| def.default_value.as_ref())
                        .filter(|v| !v.is_null())
                        .map(value_to_sql_string)
                });
            (name, val)
        })
        .collect()
}

/// Parse raw JSON string into `ConversationMetadata` (`JsonMap<String, Value>`).
///
/// Shared across Postgres, Redis, and Oracle conversation storage backends.
/// Returns `Ok(None)` for `None`, empty strings, and the literal `"null"`.
pub(super) fn parse_conversation_metadata(
    raw: Option<String>,
) -> Result<Option<ConversationMetadata>, String> {
    match raw {
        Some(s) if !s.is_empty() => {
            let s = s.trim();
            if s.is_empty() || s.eq_ignore_ascii_case("null") {
                return Ok(None);
            }
            serde_json::from_str::<ConversationMetadata>(s)
                .map(Some)
                .map_err(|e| e.to_string())
        }
        _ => Ok(None),
    }
}

pub(super) fn parse_tool_calls(raw: Option<String>) -> Result<Vec<Value>, String> {
    match raw {
        Some(s) if !s.is_empty() => serde_json::from_str(&s).map_err(|e| e.to_string()),
        _ => Ok(Vec::new()),
    }
}

pub(super) fn parse_metadata(raw: Option<String>) -> Result<HashMap<String, Value>, String> {
    match raw {
        Some(s) if !s.is_empty() => serde_json::from_str(&s).map_err(|e| e.to_string()),
        _ => Ok(HashMap::new()),
    }
}

pub(super) fn parse_raw_response(raw: Option<String>) -> Result<Value, String> {
    match raw {
        Some(s) if !s.is_empty() => serde_json::from_str(&s).map_err(|e| e.to_string()),
        _ => Ok(Value::Null),
    }
}

pub(super) fn parse_json_value(raw: Option<String>) -> Result<Value, String> {
    match raw {
        Some(s) if !s.is_empty() => serde_json::from_str(&s).map_err(|e| e.to_string()),
        _ => Ok(Value::Array(vec![])),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn parse_tool_calls_handles_empty_input() {
        assert!(parse_tool_calls(None).unwrap().is_empty());
        assert!(parse_tool_calls(Some(String::new())).unwrap().is_empty());
    }

    #[test]
    fn parse_tool_calls_round_trips() {
        let payload = json!([{ "type": "test", "value": 1 }]).to_string();
        let parsed = parse_tool_calls(Some(payload)).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["type"], "test");
        assert_eq!(parsed[0]["value"], 1);
    }

    #[test]
    fn parse_metadata_defaults_to_empty_map() {
        assert!(parse_metadata(None).unwrap().is_empty());
    }

    #[test]
    fn parse_metadata_round_trips() {
        let payload = json!({"key": "value", "nested": {"bool": true}}).to_string();
        let parsed = parse_metadata(Some(payload)).unwrap();
        assert_eq!(parsed.get("key").unwrap(), "value");
        assert_eq!(parsed["nested"]["bool"], true);
    }

    #[test]
    fn parse_raw_response_handles_null() {
        assert_eq!(parse_raw_response(None).unwrap(), Value::Null);
    }

    #[test]
    fn parse_raw_response_round_trips() {
        let payload = json!({"id": "abc"}).to_string();
        let parsed = parse_raw_response(Some(payload)).unwrap();
        assert_eq!(parsed["id"], "abc");
    }

    #[test]
    fn parse_conversation_metadata_none_returns_ok_none() {
        assert!(parse_conversation_metadata(None).unwrap().is_none());
    }

    #[test]
    fn parse_conversation_metadata_empty_string_returns_ok_none() {
        assert!(parse_conversation_metadata(Some(String::new()))
            .unwrap()
            .is_none());
    }

    #[test]
    fn parse_conversation_metadata_null_string_returns_ok_none() {
        assert!(parse_conversation_metadata(Some("null".to_string()))
            .unwrap()
            .is_none());
        // Also test case-insensitive
        assert!(parse_conversation_metadata(Some("NULL".to_string()))
            .unwrap()
            .is_none());
        assert!(parse_conversation_metadata(Some("Null".to_string()))
            .unwrap()
            .is_none());
    }

    #[test]
    fn parse_conversation_metadata_valid_json_object() {
        let payload = json!({"key": "value", "count": 42}).to_string();
        let parsed = parse_conversation_metadata(Some(payload))
            .unwrap()
            .expect("should be Some");
        assert_eq!(parsed.get("key").expect("key should exist"), "value");
        assert_eq!(
            parsed
                .get("count")
                .expect("count should exist")
                .as_i64()
                .expect("should be i64"),
            42
        );
    }

    #[test]
    fn parse_conversation_metadata_invalid_json_returns_err() {
        let result = parse_conversation_metadata(Some("not json".to_string()));
        assert!(result.is_err());
    }

    // ── Extra column helpers ─────────────────────────────────────────────

    #[test]
    fn extra_column_defs_empty_by_default() {
        let tc = TableConfig::with_table("t");
        assert!(extra_column_defs(&tc).is_empty());
    }

    #[test]
    fn extra_column_defs_generates_sql() {
        let mut tc = TableConfig::with_table("t");
        tc.extra_columns.insert(
            "TENANT_ID".to_string(),
            crate::schema::ColumnDef {
                sql_type: "VARCHAR(128)".to_string(),
                default_value: None,
            },
        );
        tc.extra_columns.insert(
            "EXPIRES_AT".to_string(),
            crate::schema::ColumnDef {
                sql_type: "TIMESTAMP".to_string(),
                default_value: None,
            },
        );
        let defs = extra_column_defs(&tc);
        assert_eq!(defs.len(), 2);
        // Sorted alphabetically
        assert_eq!(defs[0], "EXPIRES_AT TIMESTAMP");
        assert_eq!(defs[1], "TENANT_ID VARCHAR(128)");
    }

    #[test]
    fn sorted_extra_column_names_returns_sorted() {
        let mut tc = TableConfig::with_table("t");
        tc.extra_columns.insert(
            "z_col".to_string(),
            crate::schema::ColumnDef {
                sql_type: "TEXT".to_string(),
                default_value: None,
            },
        );
        tc.extra_columns.insert(
            "a_col".to_string(),
            crate::schema::ColumnDef {
                sql_type: "TEXT".to_string(),
                default_value: None,
            },
        );
        let names = sorted_extra_column_names(&tc);
        assert_eq!(names, vec!["a_col", "z_col"]);
    }

    // ── resolve_extra_column_values ──────────────────────────────────────

    #[test]
    fn resolve_extra_values_prefers_hook_over_default() {
        use crate::hooks::ExtraColumns;

        let mut tc = TableConfig::with_table("t");
        tc.extra_columns.insert(
            "TENANT_ID".to_string(),
            crate::schema::ColumnDef {
                sql_type: "VARCHAR(128)".to_string(),
                default_value: Some(json!("default_tenant")),
            },
        );

        let mut hook = ExtraColumns::new();
        hook.insert("TENANT_ID".to_string(), json!("hook_tenant"));

        let resolved = resolve_extra_column_values(&tc, &hook);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].0, "TENANT_ID");
        assert_eq!(resolved[0].1, Some("hook_tenant".to_string()));
    }

    #[test]
    fn resolve_extra_values_falls_back_to_default() {
        use crate::hooks::ExtraColumns;

        let mut tc = TableConfig::with_table("t");
        tc.extra_columns.insert(
            "TENANT_ID".to_string(),
            crate::schema::ColumnDef {
                sql_type: "VARCHAR(128)".to_string(),
                default_value: Some(json!("default_tenant")),
            },
        );

        let hook = ExtraColumns::new(); // empty
        let resolved = resolve_extra_column_values(&tc, &hook);
        assert_eq!(resolved[0].1, Some("default_tenant".to_string()));
    }

    #[test]
    fn resolve_extra_values_returns_none_when_no_value() {
        use crate::hooks::ExtraColumns;

        let mut tc = TableConfig::with_table("t");
        tc.extra_columns.insert(
            "TENANT_ID".to_string(),
            crate::schema::ColumnDef {
                sql_type: "VARCHAR(128)".to_string(),
                default_value: None,
            },
        );

        let hook = ExtraColumns::new(); // empty
        let resolved = resolve_extra_column_values(&tc, &hook);
        assert_eq!(resolved[0].1, None);
    }

    #[test]
    fn resolve_extra_values_skips_null_hook_values() {
        use crate::hooks::ExtraColumns;

        let mut tc = TableConfig::with_table("t");
        tc.extra_columns.insert(
            "TENANT_ID".to_string(),
            crate::schema::ColumnDef {
                sql_type: "VARCHAR(128)".to_string(),
                default_value: Some(json!("fallback")),
            },
        );

        let mut hook = ExtraColumns::new();
        hook.insert("TENANT_ID".to_string(), Value::Null);

        let resolved = resolve_extra_column_values(&tc, &hook);
        assert_eq!(resolved[0].1, Some("fallback".to_string()));
    }

    // ── build_response_select_base with skip/extra ──────────────────────

    #[test]
    fn select_base_skips_columns() {
        let mut cfg = SchemaConfig::default();
        cfg.responses
            .skip_columns
            .insert("raw_response".to_string());
        cfg.responses
            .skip_columns
            .insert("safety_identifier".to_string());
        let sql = build_response_select_base(&cfg);
        assert!(!sql.contains("raw_response"));
        assert!(!sql.contains("safety_identifier"));
        assert!(sql.contains("id")); // not skipped
    }

    #[test]
    fn select_base_excludes_extra_columns() {
        let mut cfg = SchemaConfig::default();
        cfg.responses.extra_columns.insert(
            "tenant_id".to_string(),
            crate::schema::ColumnDef {
                sql_type: "TEXT".to_string(),
                default_value: None,
            },
        );
        let sql = build_response_select_base(&cfg);
        // Extra columns are write-side only — NOT included in SELECT
        assert!(!sql.contains("tenant_id"));
        assert!(sql.contains("id")); // core cols still there
    }
}
