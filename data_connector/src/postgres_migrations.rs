//! Postgres-specific schema migrations.
//!
//! Each migration is a function that generates Postgres DDL from [`SchemaConfig`],
//! so it respects custom table/column names. `IF NOT EXISTS` / `IF EXISTS`
//! clauses ensure idempotency.

use crate::{schema::SchemaConfig, versioning::Migration};

/// Postgres migration list. Append new migrations here.
pub(crate) static POSTGRES_MIGRATIONS: [Migration; 2] = [
    Migration {
        version: 1,
        description: "Add safety_identifier column to responses",
        up: pg_v1_up,
    },
    Migration {
        version: 2,
        description: "Remove legacy user_id column from responses",
        up: pg_v2_up,
    },
];

fn pg_v1_up(schema: &SchemaConfig) -> Vec<String> {
    let s = &schema.responses;
    if s.is_skipped("safety_identifier") {
        return vec![];
    }
    let table = s.qualified_table(schema.owner.as_deref());
    let col = s.col("safety_identifier");
    vec![format!(
        "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} VARCHAR(128)"
    )]
}

fn pg_v2_up(schema: &SchemaConfig) -> Vec<String> {
    let s = &schema.responses;
    // Don't drop user_id if a configured column maps to that name
    // or if it's defined as an extra column.
    if s.columns
        .values()
        .any(|v| v.eq_ignore_ascii_case("user_id"))
        || s.extra_columns
            .keys()
            .any(|k| k.eq_ignore_ascii_case("user_id"))
    {
        return vec![];
    }
    let table = s.qualified_table(schema.owner.as_deref());
    vec![format!("ALTER TABLE {table} DROP COLUMN IF EXISTS user_id")]
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::TableConfig;

    #[test]
    fn postgres_migrations_are_sequential() {
        for (i, m) in POSTGRES_MIGRATIONS.iter().enumerate() {
            assert_eq!(m.version, (i + 1) as u32, "migration {i} has wrong version");
        }
    }

    #[test]
    fn pg_v1_up_generates_add_column_if_not_exists() {
        let schema = SchemaConfig::default();
        let stmts = pg_v1_up(&schema);
        assert_eq!(stmts.len(), 1);
        assert!(stmts[0].contains("IF NOT EXISTS"), "got: {}", stmts[0]);
    }

    #[test]
    fn pg_v1_up_skipped_returns_empty() {
        let schema = SchemaConfig {
            responses: TableConfig {
                skip_columns: ["safety_identifier".to_string()].into_iter().collect(),
                ..TableConfig::with_table("responses")
            },
            ..Default::default()
        };
        let stmts = pg_v1_up(&schema);
        assert!(stmts.is_empty());
    }

    #[test]
    fn pg_v2_up_generates_drop_column_if_exists() {
        let schema = SchemaConfig::default();
        let stmts = pg_v2_up(&schema);
        assert_eq!(stmts.len(), 1);
        assert!(stmts[0].contains("IF EXISTS"), "got: {}", stmts[0]);
    }

    #[test]
    fn pg_v2_up_skipped_when_column_maps_to_user_id() {
        let mut schema = SchemaConfig::default();
        schema
            .responses
            .columns
            .insert("safety_identifier".to_string(), "user_id".to_string());
        let stmts = pg_v2_up(&schema);
        assert!(stmts.is_empty(), "should skip drop when user_id is mapped");
    }

    #[test]
    fn pg_v2_up_skipped_when_extra_column_is_user_id() {
        let mut schema = SchemaConfig::default();
        schema.responses.extra_columns.insert(
            "user_id".to_string(),
            crate::schema::ColumnDef {
                sql_type: "VARCHAR(128)".to_string(),
                default_value: None,
            },
        );
        let stmts = pg_v2_up(&schema);
        assert!(
            stmts.is_empty(),
            "should skip drop when user_id is an extra column"
        );
    }
}
