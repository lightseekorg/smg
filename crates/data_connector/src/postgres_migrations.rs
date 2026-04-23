//! Postgres-specific schema migrations.
//!
//! Each migration is a function that generates Postgres DDL from [`SchemaConfig`],
//! so it respects custom table/column names. `IF NOT EXISTS` / `IF EXISTS`
//! clauses ensure idempotency.
#![cfg_attr(not(test), allow(dead_code))]

use crate::{schema::SchemaConfig, versioning::Migration};

const POSTGRES_V1: Migration = Migration {
    version: 1,
    description: "Add safety_identifier column to responses",
    up: pg_v1_up,
};
const POSTGRES_V2: Migration = Migration {
    version: 2,
    description: "Remove legacy user_id column from responses",
    up: pg_v2_up,
};
const POSTGRES_V3: Migration = Migration {
    version: 3,
    description: "Drop redundant output, metadata, instructions, tool_calls columns from responses",
    up: pg_v3_up,
};
const POSTGRES_V4: Migration = Migration {
    version: 4,
    description: "Create skills table",
    up: pg_v4_up,
};
const POSTGRES_V5: Migration = Migration {
    version: 5,
    description: "Create skill_versions table",
    up: pg_v5_up,
};
const POSTGRES_V6: Migration = Migration {
    version: 6,
    description: "Create tenant_aliases table",
    up: pg_v6_up,
};
const POSTGRES_V7: Migration = Migration {
    version: 7,
    description: "Create bundle_tokens table",
    up: pg_v7_up,
};
const POSTGRES_V8: Migration = Migration {
    version: 8,
    description: "Create continuation_cookies table",
    up: pg_v8_up,
};
const POSTGRES_V9: Migration = Migration {
    version: 9,
    description: "Extend responses with background-mode columns",
    up: pg_v9_up,
};
const POSTGRES_V10: Migration = Migration {
    version: 10,
    description: "Create background_queue table",
    up: pg_v10_up,
};
const POSTGRES_V11: Migration = Migration {
    version: 11,
    description: "Create response_stream_chunks table",
    up: pg_v11_up,
};

/// Core history-backend migrations required by the SQL response/conversation
/// storage path during normal gateway startup.
pub(crate) static POSTGRES_HISTORY_MIGRATIONS: [Migration; 3] =
    [POSTGRES_V1, POSTGRES_V2, POSTGRES_V3];

/// Postgres migration list. Append new migrations here.
pub(crate) static POSTGRES_MIGRATIONS: [Migration; 11] = [
    POSTGRES_V1,
    POSTGRES_V2,
    POSTGRES_V3,
    POSTGRES_V4,
    POSTGRES_V5,
    POSTGRES_V6,
    POSTGRES_V7,
    POSTGRES_V8,
    POSTGRES_V9,
    POSTGRES_V10,
    POSTGRES_V11,
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

/// Drop the four redundant columns (output, metadata, instructions, tool_calls)
/// that are now fully covered by `raw_response`.
fn pg_v3_up(schema: &SchemaConfig) -> Vec<String> {
    let s = &schema.responses;
    let table = s.qualified_table(schema.owner.as_deref());

    // Resolve each redundant field to its physical column name, then drop it.
    // Skip if another field maps to the same physical name or it's an extra column.
    let redundant = ["output", "metadata", "instructions", "tool_calls"];

    let cols_to_drop: Vec<_> = redundant
        .iter()
        .filter_map(|&field| {
            let col = s.col(field);
            let mapped_by_non_redundant_field = s.columns.iter().any(|(k, v)| {
                !k.eq_ignore_ascii_case(field)
                    && !redundant.iter().any(|r| k.eq_ignore_ascii_case(r))
                    && v.eq_ignore_ascii_case(col)
            });
            let used_as_extra = s.extra_columns.keys().any(|k| k.eq_ignore_ascii_case(col));
            if mapped_by_non_redundant_field || used_as_extra {
                None
            } else {
                Some(format!("DROP COLUMN IF EXISTS {col}"))
            }
        })
        .collect();

    if cols_to_drop.is_empty() {
        return vec![];
    }

    vec![format!("ALTER TABLE {table} {}", cols_to_drop.join(", "))]
}

fn pg_v4_up(schema: &SchemaConfig) -> Vec<String> {
    let table = pg_qualified_table(schema, "skills");
    vec![
        format!(
            "CREATE TABLE IF NOT EXISTS {table} (\
             skill_id VARCHAR(64) PRIMARY KEY, \
             tenant_id VARCHAR(64) NOT NULL, \
             name VARCHAR(64) NOT NULL, \
             short_description TEXT, \
             description TEXT, \
             source VARCHAR(64) NOT NULL DEFAULT 'custom', \
             has_code_files BOOLEAN NOT NULL DEFAULT false, \
             latest_version VARCHAR(64), \
             default_version VARCHAR(64), \
             created_at TIMESTAMPTZ NOT NULL, \
             updated_at TIMESTAMPTZ NOT NULL)"
        ),
        format!("CREATE INDEX IF NOT EXISTS idx_skills_tenant_name ON {table}(tenant_id, name)"),
    ]
}

fn pg_v5_up(schema: &SchemaConfig) -> Vec<String> {
    let skills_table = pg_qualified_table(schema, "skills");
    let table = pg_qualified_table(schema, "skill_versions");
    vec![
        format!(
            "CREATE TABLE IF NOT EXISTS {table} (\
             skill_id VARCHAR(64) NOT NULL REFERENCES {skills_table}(skill_id) ON DELETE CASCADE, \
             version VARCHAR(64) NOT NULL, \
             version_number INTEGER NOT NULL, \
             name VARCHAR(64) NOT NULL, \
             short_description TEXT, \
             description TEXT NOT NULL, \
             interface JSONB, \
             dependencies JSONB, \
             policy JSONB, \
             deprecated BOOLEAN NOT NULL DEFAULT false, \
             file_manifest JSONB NOT NULL, \
             instruction_token_counts JSONB NOT NULL, \
             created_at TIMESTAMPTZ NOT NULL, \
             PRIMARY KEY (skill_id, version))"
        ),
        format!(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_skill_version_number ON {table}(skill_id, version_number)"
        ),
    ]
}

fn pg_v6_up(schema: &SchemaConfig) -> Vec<String> {
    let table = pg_qualified_table(schema, "tenant_aliases");
    vec![
        format!(
            "CREATE TABLE IF NOT EXISTS {table} (\
             alias_tenant_id VARCHAR(64) PRIMARY KEY, \
             canonical_tenant_id VARCHAR(64) NOT NULL, \
             created_at TIMESTAMPTZ NOT NULL, \
             expires_at TIMESTAMPTZ)"
        ),
        format!(
            "CREATE INDEX IF NOT EXISTS idx_tenant_aliases_canonical ON {table}(canonical_tenant_id)"
        ),
    ]
}

fn pg_v7_up(schema: &SchemaConfig) -> Vec<String> {
    let table = pg_qualified_table(schema, "bundle_tokens");
    vec![
        format!(
            "CREATE TABLE IF NOT EXISTS {table} (\
             token_hash VARCHAR(64) PRIMARY KEY, \
             tenant_id VARCHAR(64) NOT NULL, \
             exec_id VARCHAR(64) NOT NULL, \
             skill_id VARCHAR(64) NOT NULL, \
             skill_version VARCHAR(64) NOT NULL, \
             created_at TIMESTAMPTZ NOT NULL, \
             expires_at TIMESTAMPTZ NOT NULL)"
        ),
        format!("CREATE INDEX IF NOT EXISTS idx_bundle_tokens_exec_id ON {table}(exec_id)"),
        format!("CREATE INDEX IF NOT EXISTS idx_bundle_tokens_expires_at ON {table}(expires_at)"),
    ]
}

fn pg_v8_up(schema: &SchemaConfig) -> Vec<String> {
    let table = pg_qualified_table(schema, "continuation_cookies");
    vec![
        format!(
            "CREATE TABLE IF NOT EXISTS {table} (\
             cookie_hash VARCHAR(64) PRIMARY KEY, \
             tenant_id VARCHAR(64) NOT NULL, \
             exec_id VARCHAR(64) NOT NULL, \
             request_id VARCHAR(64) NOT NULL, \
             created_at TIMESTAMPTZ NOT NULL, \
             expires_at TIMESTAMPTZ NOT NULL)"
        ),
        format!("CREATE INDEX IF NOT EXISTS idx_continuation_cookies_exec_id ON {table}(exec_id)"),
        format!(
            "CREATE INDEX IF NOT EXISTS idx_continuation_cookies_expires_at ON {table}(expires_at)"
        ),
    ]
}

fn pg_qualified_table(schema: &SchemaConfig, table: &str) -> String {
    match schema.owner.as_deref() {
        Some(owner) => format!("{owner}.{table}"),
        None => table.to_string(),
    }
}

/// Extend `responses` with background-mode columns + backfill `started_at` /
/// `completed_at` from `created_at` on historical rows (legacy rows were
/// persisted only after synchronous completion, so conflating the three
/// timestamps is semantically correct per the design).
fn pg_v9_up(schema: &SchemaConfig) -> Vec<String> {
    let s = &schema.responses;
    let table = s.qualified_table(schema.owner.as_deref());
    let created_at_col = s.col("created_at");

    // Inline CHECK on `status` — DB-level guard against stringly-typed state
    // drift. Values mirror the `ResponseStatus` enum. `ADD COLUMN IF NOT EXISTS`
    // applies the CHECK on first add and is a no-op on re-run; safe.
    let status_col = s.col("status");
    let status_ty = format!(
        "TEXT NOT NULL DEFAULT 'completed' \
         CHECK ({status_col} IN ('queued', 'in_progress', 'completed', \
                                  'failed', 'cancelled', 'incomplete'))"
    );
    // JSONB (not JSON) for request_json / request_context_json — gives us
    // indexing, operator support, and faster reads at a small write cost.
    let bg_columns: &[(&str, &str)] = &[
        ("status", status_ty.as_str()),
        ("background", "BOOLEAN NOT NULL DEFAULT false"),
        ("stream_enabled", "BOOLEAN NOT NULL DEFAULT false"),
        ("cancel_requested", "BOOLEAN NOT NULL DEFAULT false"),
        ("request_json", "JSONB"),
        ("request_context_json", "JSONB"),
        ("started_at", "TIMESTAMPTZ"),
        ("completed_at", "TIMESTAMPTZ"),
        ("next_stream_sequence", "BIGINT NOT NULL DEFAULT 0"),
    ];

    let mut stmts: Vec<String> = bg_columns
        .iter()
        .filter(|(field, _)| !s.is_skipped(field))
        .map(|(field, ty)| {
            let col = s.col(field);
            format!("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {ty}")
        })
        .collect();

    // Backfill started_at / completed_at from created_at for historical rows.
    // Guarded on created_at being present — deployments that skip created_at
    // would otherwise generate UPDATEs against a non-existent column.
    let created_at_present = !s.is_skipped("created_at");
    if created_at_present && !s.is_skipped("started_at") {
        let col = s.col("started_at");
        stmts.push(format!(
            "UPDATE {table} SET {col} = {created_at_col} WHERE {col} IS NULL"
        ));
    }
    if created_at_present && !s.is_skipped("completed_at") {
        let col = s.col("completed_at");
        stmts.push(format!(
            "UPDATE {table} SET {col} = {created_at_col} WHERE {col} IS NULL"
        ));
    }

    stmts
}

/// Create the `background_queue` work-queue table.
fn pg_v10_up(schema: &SchemaConfig) -> Vec<String> {
    let q = &schema.background_queue;
    let r = &schema.responses;
    let queue_table = q.qualified_table(schema.owner.as_deref());
    let resp_table = r.qualified_table(schema.owner.as_deref());
    let resp_id_col = r.col("id");

    let response_id = q.col("response_id");
    let priority = q.col("priority");
    let retry_attempt = q.col("retry_attempt");
    let next_attempt_at = q.col("next_attempt_at");
    let lease_expires_at = q.col("lease_expires_at");
    let worker_id = q.col("worker_id");
    let created_at = q.col("created_at");

    vec![
        format!(
            "CREATE TABLE IF NOT EXISTS {queue_table} (\
                {response_id} VARCHAR(64) PRIMARY KEY \
                    REFERENCES {resp_table}({resp_id_col}) ON DELETE CASCADE, \
                {priority} INT NOT NULL, \
                {retry_attempt} INT NOT NULL DEFAULT 0, \
                {next_attempt_at} TIMESTAMPTZ NOT NULL, \
                {lease_expires_at} TIMESTAMPTZ, \
                {worker_id} TEXT, \
                {created_at} TIMESTAMPTZ NOT NULL DEFAULT now(), \
                CONSTRAINT bg_queue_lease_pairing_chk \
                    CHECK (({worker_id} IS NULL) = ({lease_expires_at} IS NULL))\
            )"
        ),
        // Claim index: partial index over unclaimed rows, ordered to match the
        // claim query (priority asc, next_attempt_at asc, created_at asc).
        //
        // Fixed names (not derived from the table identifier) so custom table
        // names can't collide via Postgres' 63-byte identifier truncation, and
        // so the matching Oracle names stay ≤30 chars.
        format!(
            "CREATE INDEX IF NOT EXISTS bg_queue_claim_idx ON {queue_table} \
                ({priority}, {next_attempt_at}, {created_at}) \
                WHERE {lease_expires_at} IS NULL"
        ),
        // Lease-sweep index for the janitor that requeues expired leases.
        format!(
            "CREATE INDEX IF NOT EXISTS bg_queue_sweep_idx ON {queue_table} \
                ({lease_expires_at}) \
                WHERE {lease_expires_at} IS NOT NULL"
        ),
    ]
}

/// Create the `response_stream_chunks` per-response SSE log table.
fn pg_v11_up(schema: &SchemaConfig) -> Vec<String> {
    let c = &schema.response_stream_chunks;
    let r = &schema.responses;
    let chunks_table = c.qualified_table(schema.owner.as_deref());
    let resp_table = r.qualified_table(schema.owner.as_deref());
    let resp_id_col = r.col("id");

    let response_id = c.col("response_id");
    let sequence = c.col("sequence");
    let event_type = c.col("event_type");
    let data = c.col("data");
    let created_at = c.col("created_at");

    vec![
        format!(
            "CREATE TABLE IF NOT EXISTS {chunks_table} (\
                {response_id} VARCHAR(64) NOT NULL \
                    REFERENCES {resp_table}({resp_id_col}) ON DELETE CASCADE, \
                {sequence} BIGINT NOT NULL, \
                {event_type} TEXT NOT NULL, \
                {data} JSONB NOT NULL, \
                {created_at} TIMESTAMPTZ NOT NULL DEFAULT now(), \
                PRIMARY KEY ({response_id}, {sequence})\
            )"
        ),
        // Cleanup index for the retention-window janitor.
        // Named `stream_chunks_cleanup_idx` rather than
        // `response_stream_chunks_cleanup_idx` so the corresponding Oracle
        // identifier stays ≤30 chars (pre-12.2 limit).
        format!(
            "CREATE INDEX IF NOT EXISTS stream_chunks_cleanup_idx \
                ON {chunks_table} ({created_at})"
        ),
    ]
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::TableConfig;

    #[test]
    fn postgres_history_migrations_cover_only_core_history_schema() {
        let versions: Vec<u32> = POSTGRES_HISTORY_MIGRATIONS
            .iter()
            .map(|migration| migration.version)
            .collect();
        assert_eq!(versions, vec![1, 2, 3]);
    }

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

    #[test]
    fn pg_v3_up_generates_one_drop_statement() {
        let schema = SchemaConfig::default();
        let stmts = pg_v3_up(&schema);
        assert_eq!(stmts.len(), 1);
        let stmt = &stmts[0];
        assert!(stmt.contains("DROP COLUMN IF EXISTS output"));
        assert!(stmt.contains("DROP COLUMN IF EXISTS metadata"));
        assert!(stmt.contains("DROP COLUMN IF EXISTS instructions"));
        assert!(stmt.contains("DROP COLUMN IF EXISTS tool_calls"));
    }

    #[test]
    fn pg_v3_up_skips_when_output_is_used_by_another_field() {
        let mut schema = SchemaConfig::default();
        // Another field maps to physical column "output"
        schema
            .responses
            .columns
            .insert("safety_identifier".to_string(), "output".to_string());
        let stmts = pg_v3_up(&schema);
        assert_eq!(stmts.len(), 1);
        assert!(
            !stmts[0].contains("EXISTS output"),
            "should skip output when another field maps to it: {stmts:?}"
        );
        assert!(stmts[0].contains("metadata"));
    }

    #[test]
    fn pg_v3_up_skips_extra_column_named_metadata() {
        let mut schema = SchemaConfig::default();
        schema.responses.extra_columns.insert(
            "metadata".to_string(),
            crate::schema::ColumnDef {
                sql_type: "JSON".to_string(),
                default_value: None,
            },
        );
        let stmts = pg_v3_up(&schema);
        assert_eq!(stmts.len(), 1);
        assert!(
            !stmts[0].contains("metadata"),
            "should skip metadata when it's an extra column: {stmts:?}"
        );
        assert!(stmts[0].contains("output"));
    }

    #[test]
    fn pg_v3_up_drops_mapped_physical_column_name() {
        let mut schema = SchemaConfig::default();
        schema
            .responses
            .columns
            .insert("output".to_string(), "resp_output".to_string());
        let stmts = pg_v3_up(&schema);
        assert_eq!(stmts.len(), 1);
        assert!(
            stmts[0].contains("resp_output"),
            "should drop mapped physical column: {stmts:?}"
        );
        assert!(
            !stmts[0].contains("EXISTS output"),
            "should not use logical name: {stmts:?}"
        );
    }

    #[test]
    fn pg_v4_up_creates_skills_table_and_index() {
        let schema = SchemaConfig::default();
        let stmts = pg_v4_up(&schema);
        assert_eq!(stmts.len(), 2);
        assert!(stmts[0].contains("CREATE TABLE IF NOT EXISTS skills"));
        assert!(stmts[0].contains("skill_id VARCHAR(64) PRIMARY KEY"));
        assert!(stmts[0].contains("source VARCHAR(64) NOT NULL DEFAULT 'custom'"));
        assert!(stmts[1].contains("idx_skills_tenant_name"));
    }

    #[test]
    fn pg_v5_up_creates_skill_versions_table_and_index() {
        let schema = SchemaConfig::default();
        let stmts = pg_v5_up(&schema);
        assert_eq!(stmts.len(), 2);
        assert!(stmts[0].contains("CREATE TABLE IF NOT EXISTS skill_versions"));
        assert!(stmts[0].contains("REFERENCES skills(skill_id) ON DELETE CASCADE"));
        assert!(stmts[0].contains("file_manifest JSONB NOT NULL"));
        assert!(stmts[0].contains("instruction_token_counts JSONB NOT NULL"));
        assert!(stmts[1].contains("idx_skill_version_number"));
    }

    #[test]
    fn pg_v6_up_creates_tenant_aliases_table() {
        let schema = SchemaConfig::default();
        let stmts = pg_v6_up(&schema);
        assert_eq!(stmts.len(), 2);
        assert!(stmts[0].contains("CREATE TABLE IF NOT EXISTS tenant_aliases"));
        assert!(stmts[1].contains("idx_tenant_aliases_canonical"));
    }

    #[test]
    fn pg_v7_up_creates_bundle_tokens_table() {
        let schema = SchemaConfig::default();
        let stmts = pg_v7_up(&schema);
        assert_eq!(stmts.len(), 3);
        assert!(stmts[0].contains("CREATE TABLE IF NOT EXISTS bundle_tokens"));
        assert!(stmts[0].contains("token_hash VARCHAR(64) PRIMARY KEY"));
        assert!(stmts[1].contains("idx_bundle_tokens_exec_id"));
        assert!(stmts[2].contains("idx_bundle_tokens_expires_at"));
    }

    #[test]
    fn pg_v8_up_creates_continuation_cookies_table() {
        let schema = SchemaConfig::default();
        let stmts = pg_v8_up(&schema);
        assert_eq!(stmts.len(), 3);
        assert!(stmts[0].contains("CREATE TABLE IF NOT EXISTS continuation_cookies"));
        assert!(stmts[0].contains("cookie_hash VARCHAR(64) PRIMARY KEY"));
        assert!(stmts[1].contains("idx_continuation_cookies_exec_id"));
        assert!(stmts[2].contains("idx_continuation_cookies_expires_at"));
    }

    // ── v9: extend responses with background-mode columns ─────────────────

    #[test]
    fn pg_v9_up_status_column_has_check_constraint() {
        let schema = SchemaConfig::default();
        let stmts = pg_v9_up(&schema);
        let status_stmt = stmts
            .iter()
            .find(|s| s.contains("ADD COLUMN IF NOT EXISTS status"))
            .expect("status ADD COLUMN missing");
        assert!(
            status_stmt.contains("CHECK (status IN"),
            "status must have a CHECK constraint: {status_stmt}"
        );
        for val in [
            "'queued'",
            "'in_progress'",
            "'completed'",
            "'failed'",
            "'cancelled'",
            "'incomplete'",
        ] {
            assert!(
                status_stmt.contains(val),
                "CHECK must enumerate {val}: {status_stmt}"
            );
        }
    }

    #[test]
    fn pg_v9_up_adds_all_nine_columns_and_backfill() {
        let schema = SchemaConfig::default();
        let stmts = pg_v9_up(&schema);
        // 9 ADD COLUMN + 2 UPDATE backfills
        assert_eq!(stmts.len(), 11, "got: {stmts:?}");
        for col in [
            "status",
            "background",
            "stream_enabled",
            "cancel_requested",
            "request_json",
            "request_context_json",
            "started_at",
            "completed_at",
            "next_stream_sequence",
        ] {
            assert!(
                stmts
                    .iter()
                    .any(|s| s.contains(&format!("COLUMN IF NOT EXISTS {col}"))),
                "missing ADD COLUMN for {col}: {stmts:?}"
            );
        }
        // Backfill UPDATEs
        assert!(stmts
            .iter()
            .any(|s| s.contains("SET started_at = created_at")));
        assert!(stmts
            .iter()
            .any(|s| s.contains("SET completed_at = created_at")));
    }

    #[test]
    fn pg_v9_up_honors_skip_columns() {
        let schema = SchemaConfig {
            responses: TableConfig {
                skip_columns: ["background".to_string(), "started_at".to_string()]
                    .into_iter()
                    .collect(),
                ..TableConfig::with_table("responses")
            },
            ..Default::default()
        };
        let stmts = pg_v9_up(&schema);
        assert!(
            !stmts
                .iter()
                .any(|s| s.contains("ADD COLUMN IF NOT EXISTS background")),
            "should skip background column: {stmts:?}"
        );
        // Skipping started_at must also skip its backfill UPDATE.
        assert!(
            !stmts.iter().any(|s| s.contains("SET started_at")),
            "should skip started_at backfill: {stmts:?}"
        );
        // completed_at is NOT skipped so its backfill must still be there.
        assert!(stmts
            .iter()
            .any(|s| s.contains("SET completed_at = created_at")));
    }

    // ── v10: create background_queue ───────────────────────────────────────

    #[test]
    fn pg_v10_up_creates_table_and_two_indexes() {
        let schema = SchemaConfig::default();
        let stmts = pg_v10_up(&schema);
        assert_eq!(stmts.len(), 3, "got: {stmts:?}");
        assert!(stmts[0].contains("CREATE TABLE IF NOT EXISTS background_queue"));
        assert!(
            stmts[0].contains("ON DELETE CASCADE"),
            "FK must cascade: {stmts:?}"
        );
        assert!(stmts[1].contains("bg_queue_claim_idx"));
        assert!(
            stmts[1].contains("WHERE lease_expires_at IS NULL"),
            "claim index must be partial over unclaimed rows: {stmts:?}"
        );
        assert!(stmts[2].contains("bg_queue_sweep_idx"));
        assert!(stmts[2].contains("WHERE lease_expires_at IS NOT NULL"));
    }

    #[test]
    fn pg_v10_up_enforces_lease_pairing_invariant() {
        // worker_id and lease_expires_at must be both NULL or both set.
        let schema = SchemaConfig::default();
        let stmts = pg_v10_up(&schema);
        assert!(
            stmts[0].contains("CONSTRAINT bg_queue_lease_pairing_chk")
                && stmts[0].contains("CHECK ((worker_id IS NULL) = (lease_expires_at IS NULL))"),
            "lease pairing CHECK constraint missing: {stmts:?}"
        );
    }

    #[test]
    fn pg_v10_up_uses_fixed_object_names_independent_of_table_identifier() {
        // If an operator customizes `background_queue.table` to a long name,
        // Postgres truncates identifiers to 63 bytes — derived index names
        // like `{table}_claim_idx` and `{table}_sweep_idx` would collide, and
        // `CREATE INDEX IF NOT EXISTS` would silently skip the second one.
        // Fix: fixed non-derived names.
        let mut schema = SchemaConfig::default();
        schema.background_queue.table =
            "custom_background_queue_name_that_is_really_quite_long".to_string();
        let stmts = pg_v10_up(&schema);
        assert!(
            stmts[0].contains("CONSTRAINT bg_queue_lease_pairing_chk"),
            "constraint name must be fixed: {stmts:?}"
        );
        assert!(stmts[1].contains("bg_queue_claim_idx"));
        assert!(stmts[2].contains("bg_queue_sweep_idx"));
    }

    // ── v11: create response_stream_chunks ─────────────────────────────────

    #[test]
    fn pg_v11_up_creates_table_with_composite_pk_and_cleanup_index() {
        let schema = SchemaConfig::default();
        let stmts = pg_v11_up(&schema);
        assert_eq!(stmts.len(), 2, "got: {stmts:?}");
        assert!(stmts[0].contains("CREATE TABLE IF NOT EXISTS response_stream_chunks"));
        assert!(
            stmts[0].contains("PRIMARY KEY (response_id, sequence)"),
            "composite PK on (response_id, sequence): {stmts:?}"
        );
        assert!(stmts[0].contains("ON DELETE CASCADE"));
        assert!(stmts[1].contains("stream_chunks_cleanup_idx"));
    }
}
