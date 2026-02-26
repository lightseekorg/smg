//! Oracle-specific schema migrations.
//!
//! Each migration is a function that generates Oracle DDL from [`SchemaConfig`],
//! so it respects custom table/column names. PL/SQL exception handling ensures
//! idempotency (safe to re-run if a previous attempt partially completed).

use crate::{schema::SchemaConfig, versioning::Migration};

/// Oracle migration list. Append new migrations here.
pub(crate) static ORACLE_MIGRATIONS: [Migration; 2] = [
    Migration {
        version: 1,
        description: "Add safety_identifier column to responses",
        up: oracle_v1_up,
    },
    Migration {
        version: 2,
        description: "Remove legacy user_id column from responses",
        up: oracle_v2_up,
    },
];

fn oracle_v1_up(schema: &SchemaConfig) -> Vec<String> {
    let s = &schema.responses;
    if s.is_skipped("safety_identifier") {
        return vec![];
    }
    let table = s.qualified_table(schema.owner.as_deref());
    let col = s.col("safety_identifier");
    // PL/SQL block: ORA-01430 = "column already exists" (idempotent)
    vec![format!(
        "BEGIN EXECUTE IMMEDIATE 'ALTER TABLE {table} ADD ({col} VARCHAR2(128))'; \
         EXCEPTION WHEN OTHERS THEN IF SQLCODE != -1430 THEN RAISE; END IF; END;"
    )]
}

fn oracle_v2_up(schema: &SchemaConfig) -> Vec<String> {
    let s = &schema.responses;
    // Don't drop USER_ID if a configured column maps to that name
    // or if it's defined as an extra column.
    if s.columns
        .values()
        .any(|v| v.eq_ignore_ascii_case("USER_ID"))
        || s.extra_columns
            .keys()
            .any(|k| k.eq_ignore_ascii_case("USER_ID"))
    {
        return vec![];
    }
    let table = s.qualified_table(schema.owner.as_deref());
    // PL/SQL block: ORA-00904 = "invalid identifier" (column doesn't exist)
    vec![format!(
        "BEGIN EXECUTE IMMEDIATE 'ALTER TABLE {table} DROP (USER_ID)'; \
         EXCEPTION WHEN OTHERS THEN IF SQLCODE != -904 THEN RAISE; END IF; END;"
    )]
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::TableConfig;

    #[test]
    fn oracle_migrations_are_sequential() {
        for (i, m) in ORACLE_MIGRATIONS.iter().enumerate() {
            assert_eq!(m.version, (i + 1) as u32, "migration {i} has wrong version");
        }
    }

    #[test]
    fn oracle_v1_up_generates_plsql_add_column() {
        let schema = SchemaConfig::default();
        let stmts = oracle_v1_up(&schema);
        assert_eq!(stmts.len(), 1);
        assert!(stmts[0].contains("ADD"), "got: {}", stmts[0]);
        assert!(stmts[0].contains("SQLCODE"), "got: {}", stmts[0]);
    }

    #[test]
    fn oracle_v1_up_skipped_returns_empty() {
        let schema = SchemaConfig {
            responses: TableConfig {
                skip_columns: ["safety_identifier".to_string()].into_iter().collect(),
                ..TableConfig::with_table("responses")
            },
            ..Default::default()
        };
        let stmts = oracle_v1_up(&schema);
        assert!(stmts.is_empty());
    }

    #[test]
    fn oracle_v2_up_generates_plsql_drop_column() {
        let schema = SchemaConfig::default();
        let stmts = oracle_v2_up(&schema);
        assert_eq!(stmts.len(), 1);
        assert!(stmts[0].contains("DROP"), "got: {}", stmts[0]);
        assert!(stmts[0].contains("USER_ID"), "got: {}", stmts[0]);
    }

    #[test]
    fn oracle_v2_up_skipped_when_column_maps_to_user_id() {
        let mut schema = SchemaConfig::default();
        schema
            .responses
            .columns
            .insert("safety_identifier".to_string(), "USER_ID".to_string());
        let stmts = oracle_v2_up(&schema);
        assert!(stmts.is_empty(), "should skip drop when USER_ID is mapped");
    }

    #[test]
    fn oracle_v2_up_skipped_when_extra_column_is_user_id() {
        let mut schema = SchemaConfig::default();
        schema.responses.extra_columns.insert(
            "USER_ID".to_string(),
            crate::schema::ColumnDef {
                sql_type: "VARCHAR2(128)".to_string(),
                default_value: None,
            },
        );
        let stmts = oracle_v2_up(&schema);
        assert!(
            stmts.is_empty(),
            "should skip drop when USER_ID is an extra column"
        );
    }
}
