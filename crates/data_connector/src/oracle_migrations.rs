//! Oracle-specific schema migrations.
//!
//! Each migration is a function that generates Oracle DDL from [`SchemaConfig`],
//! so it respects custom table/column names. PL/SQL exception handling ensures
//! idempotency (safe to re-run if a previous attempt partially completed).

use crate::{schema::SchemaConfig, versioning::Migration};

/// Oracle migration list. Append new migrations here.
pub(crate) static ORACLE_MIGRATIONS: [Migration; 11] = [
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
    Migration {
        version: 3,
        description:
            "Drop redundant output, metadata, instructions, tool_calls columns from responses",
        up: oracle_v3_up,
    },
    Migration {
        version: 4,
        description: "Create skills table",
        up: oracle_v4_up,
    },
    Migration {
        version: 5,
        description: "Create skill_versions table",
        up: oracle_v5_up,
    },
    Migration {
        version: 6,
        description: "Create tenant_aliases table",
        up: oracle_v6_up,
    },
    Migration {
        version: 7,
        description: "Create bundle_tokens table",
        up: oracle_v7_up,
    },
    Migration {
        version: 8,
        description: "Create continuation_cookies table",
        up: oracle_v8_up,
    },
    Migration {
        version: 9,
        description: "Extend responses with background-mode columns",
        up: oracle_v9_up,
    },
    Migration {
        version: 10,
        description: "Create background_queue table",
        up: oracle_v10_up,
    },
    Migration {
        version: 11,
        description: "Create response_stream_chunks table",
        up: oracle_v11_up,
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

/// Drop the four redundant columns (output, metadata, instructions, tool_calls)
/// that are now fully covered by `raw_response`.
fn oracle_v3_up(schema: &SchemaConfig) -> Vec<String> {
    let s = &schema.responses;
    let table = s.qualified_table(schema.owner.as_deref());

    // Resolve each redundant field to its physical column name (uppercased for Oracle).
    // Skip if another field maps to the same physical name or it's an extra column.
    // Drop one column per statement so a missing column doesn't block dropping others.
    let redundant = ["output", "metadata", "instructions", "tool_calls"];

    redundant
        .iter()
        .filter_map(|&field| {
            let col = s.col(field).to_uppercase();
            let mapped_by_non_redundant_field = s.columns.iter().any(|(k, v)| {
                !k.eq_ignore_ascii_case(field)
                    && !redundant.iter().any(|r| k.eq_ignore_ascii_case(r))
                    && v.eq_ignore_ascii_case(&col)
            });
            let used_as_extra = s.extra_columns.keys().any(|k| k.eq_ignore_ascii_case(&col));
            if mapped_by_non_redundant_field || used_as_extra {
                None
            } else {
                // PL/SQL block: ORA-00904 = "invalid identifier" (column doesn't exist)
                Some(format!(
                    "BEGIN EXECUTE IMMEDIATE 'ALTER TABLE {table} DROP ({col})'; \
                     EXCEPTION WHEN OTHERS THEN IF SQLCODE != -904 THEN RAISE; END IF; END;"
                ))
            }
        })
        .collect()
}

fn oracle_idempotent_ddl(ddl: &str) -> String {
    let escaped = ddl.replace('\'', "''");
    format!(
        "BEGIN EXECUTE IMMEDIATE '{escaped}'; \
         EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF; END;"
    )
}

fn oracle_v4_up(schema: &SchemaConfig) -> Vec<String> {
    let table = oracle_qualified_name(schema, "SKILLS");
    let index = oracle_qualified_name(schema, "IDX_SKILLS_TENANT_NAME");
    vec![
        oracle_idempotent_ddl(&format!(
            "CREATE TABLE {table} (\
             SKILL_ID VARCHAR2(64) PRIMARY KEY, \
            TENANT_ID VARCHAR2(64) NOT NULL, \
            NAME VARCHAR2(64) NOT NULL, \
            SHORT_DESCRIPTION CLOB, \
            DESCRIPTION CLOB, \
            SOURCE VARCHAR2(64) DEFAULT 'custom' NOT NULL, \
            HAS_CODE_FILES NUMBER(1) DEFAULT 0 NOT NULL, \
            LATEST_VERSION VARCHAR2(64), \
            DEFAULT_VERSION VARCHAR2(64), \
            CREATED_AT TIMESTAMP WITH TIME ZONE NOT NULL, \
             UPDATED_AT TIMESTAMP WITH TIME ZONE NOT NULL)"
        )),
        oracle_idempotent_ddl(&format!(
            "CREATE INDEX {index} ON {table} (TENANT_ID, NAME)"
        )),
    ]
}

fn oracle_v5_up(schema: &SchemaConfig) -> Vec<String> {
    let skills_table = oracle_qualified_name(schema, "SKILLS");
    let table = oracle_qualified_name(schema, "SKILL_VERSIONS");
    let index = oracle_qualified_name(schema, "IDX_SKILL_VERSION_NUMBER");
    vec![
        oracle_idempotent_ddl(&format!(
            "CREATE TABLE {table} (\
             SKILL_ID VARCHAR2(64) NOT NULL, \
             VERSION VARCHAR2(64) NOT NULL, \
             VERSION_NUMBER NUMBER(10) NOT NULL, \
             NAME VARCHAR2(64) NOT NULL, \
             SHORT_DESCRIPTION CLOB, \
             DESCRIPTION CLOB NOT NULL, \
             INTERFACE CLOB CHECK (INTERFACE IS JSON), \
             DEPENDENCIES CLOB CHECK (DEPENDENCIES IS JSON), \
             POLICY CLOB CHECK (POLICY IS JSON), \
             DEPRECATED NUMBER(1) DEFAULT 0 NOT NULL, \
             FILE_MANIFEST CLOB NOT NULL CHECK (FILE_MANIFEST IS JSON), \
             INSTRUCTION_TOKEN_COUNTS CLOB NOT NULL CHECK (INSTRUCTION_TOKEN_COUNTS IS JSON), \
             CREATED_AT TIMESTAMP WITH TIME ZONE NOT NULL, \
             CONSTRAINT PK_SKILL_VERSIONS PRIMARY KEY (SKILL_ID, VERSION), \
             CONSTRAINT FK_SKILL_VERSIONS_SKILL FOREIGN KEY (SKILL_ID) REFERENCES {skills_table}(SKILL_ID) ON DELETE CASCADE)"
        )),
        oracle_idempotent_ddl(&format!(
            "CREATE UNIQUE INDEX {index} ON {table} (SKILL_ID, VERSION_NUMBER)"
        )),
    ]
}

fn oracle_v6_up(schema: &SchemaConfig) -> Vec<String> {
    let table = oracle_qualified_name(schema, "TENANT_ALIASES");
    let index = oracle_qualified_name(schema, "IDX_TENANT_ALIASES_CANONICAL");
    vec![
        oracle_idempotent_ddl(&format!(
            "CREATE TABLE {table} (\
             ALIAS_TENANT_ID VARCHAR2(64) PRIMARY KEY, \
             CANONICAL_TENANT_ID VARCHAR2(64) NOT NULL, \
             CREATED_AT TIMESTAMP WITH TIME ZONE NOT NULL, \
             EXPIRES_AT TIMESTAMP WITH TIME ZONE)"
        )),
        oracle_idempotent_ddl(&format!(
            "CREATE INDEX {index} ON {table} (CANONICAL_TENANT_ID)"
        )),
    ]
}

fn oracle_v7_up(schema: &SchemaConfig) -> Vec<String> {
    let table = oracle_qualified_name(schema, "BUNDLE_TOKENS");
    let exec_index = oracle_qualified_name(schema, "IDX_BUNDLE_TOKENS_EXEC_ID");
    let expires_index = oracle_qualified_name(schema, "IDX_BUNDLE_TOKENS_EXPIRES_AT");
    vec![
        oracle_idempotent_ddl(&format!(
            "CREATE TABLE {table} (\
             TOKEN_HASH VARCHAR2(64) PRIMARY KEY, \
             TENANT_ID VARCHAR2(64) NOT NULL, \
             EXEC_ID VARCHAR2(64) NOT NULL, \
             SKILL_ID VARCHAR2(64) NOT NULL, \
             SKILL_VERSION VARCHAR2(64) NOT NULL, \
             CREATED_AT TIMESTAMP WITH TIME ZONE NOT NULL, \
             EXPIRES_AT TIMESTAMP WITH TIME ZONE NOT NULL)"
        )),
        oracle_idempotent_ddl(&format!("CREATE INDEX {exec_index} ON {table} (EXEC_ID)")),
        oracle_idempotent_ddl(&format!(
            "CREATE INDEX {expires_index} ON {table} (EXPIRES_AT)"
        )),
    ]
}

fn oracle_v8_up(schema: &SchemaConfig) -> Vec<String> {
    let table = oracle_qualified_name(schema, "CONTINUATION_COOKIES");
    let exec_index = oracle_qualified_name(schema, "IDX_CONTINUATION_COOKIES_EXEC");
    let expires_index = oracle_qualified_name(schema, "IDX_CONTINUATION_COOKIES_EXP");
    vec![
        oracle_idempotent_ddl(&format!(
            "CREATE TABLE {table} (\
             COOKIE_HASH VARCHAR2(64) PRIMARY KEY, \
             TENANT_ID VARCHAR2(64) NOT NULL, \
             EXEC_ID VARCHAR2(64) NOT NULL, \
             REQUEST_ID VARCHAR2(64) NOT NULL, \
             CREATED_AT TIMESTAMP WITH TIME ZONE NOT NULL, \
             EXPIRES_AT TIMESTAMP WITH TIME ZONE NOT NULL)"
        )),
        oracle_idempotent_ddl(&format!("CREATE INDEX {exec_index} ON {table} (EXEC_ID)")),
        oracle_idempotent_ddl(&format!(
            "CREATE INDEX {expires_index} ON {table} (EXPIRES_AT)"
        )),
    ]
}

fn oracle_qualified_name(schema: &SchemaConfig, object_name: &str) -> String {
    match &schema.owner {
        Some(owner) => format!("{owner}.{object_name}"),
        None => object_name.to_string(),
    }
}

/// Extend `responses` with background-mode columns + backfill
/// `started_at` / `completed_at` from `created_at` on historical rows.
///
/// Oracle quirks: no native BOOLEAN (use `NUMBER(1)`), no `JSONB`
/// (use `CLOB` with an `IS JSON` check), no `TIMESTAMPTZ` (use
/// `TIMESTAMP WITH TIME ZONE`). Each `ALTER TABLE` is PL/SQL-wrapped so a
/// previously-added column (ORA-01430) is a no-op.
fn oracle_v9_up(schema: &SchemaConfig) -> Vec<String> {
    let s = &schema.responses;
    let table = s.qualified_table(schema.owner.as_deref());
    let created_at_col = s.col("created_at").to_uppercase();

    // CLOB + `IS JSON` check constraint ensures only valid JSON lands in
    // request_json / request_context_json (Oracle < 21c has no native JSON type).
    let request_json_col = s.col("request_json").to_uppercase();
    let request_context_json_col = s.col("request_context_json").to_uppercase();
    let request_json_ty = format!("CLOB CHECK ({request_json_col} IS JSON)");
    let request_context_json_ty = format!("CLOB CHECK ({request_context_json_col} IS JSON)");

    let bg_columns: &[(&str, &str)] = &[
        ("status", "VARCHAR2(32) DEFAULT 'completed' NOT NULL"),
        ("background", "NUMBER(1) DEFAULT 0 NOT NULL"),
        ("stream_enabled", "NUMBER(1) DEFAULT 0 NOT NULL"),
        ("cancel_requested", "NUMBER(1) DEFAULT 0 NOT NULL"),
        ("request_json", request_json_ty.as_str()),
        ("request_context_json", request_context_json_ty.as_str()),
        ("started_at", "TIMESTAMP WITH TIME ZONE"),
        ("completed_at", "TIMESTAMP WITH TIME ZONE"),
        ("next_stream_sequence", "NUMBER(19) DEFAULT 0 NOT NULL"),
    ];

    let mut stmts: Vec<String> = bg_columns
        .iter()
        .filter(|(field, _)| !s.is_skipped(field))
        .map(|(field, ty)| {
            let col = s.col(field).to_uppercase();
            // ORA-01430 = "column being added already exists in table"
            format!(
                "BEGIN EXECUTE IMMEDIATE 'ALTER TABLE {table} ADD ({col} {ty})'; \
                 EXCEPTION WHEN OTHERS THEN IF SQLCODE != -1430 THEN RAISE; END IF; END;"
            )
        })
        .collect();

    // Backfill started_at / completed_at from created_at for historical rows.
    // Guarded on created_at being present — deployments that skip created_at
    // would otherwise generate UPDATEs against a non-existent column.
    let created_at_present = !s.is_skipped("created_at");
    if created_at_present && !s.is_skipped("started_at") {
        let col = s.col("started_at").to_uppercase();
        stmts.push(format!(
            "UPDATE {table} SET {col} = {created_at_col} WHERE {col} IS NULL"
        ));
    }
    if created_at_present && !s.is_skipped("completed_at") {
        let col = s.col("completed_at").to_uppercase();
        stmts.push(format!(
            "UPDATE {table} SET {col} = {created_at_col} WHERE {col} IS NULL"
        ));
    }

    stmts
}

/// Create the `background_queue` work-queue table.
///
/// Oracle lacks Postgres-style partial indexes (`CREATE INDEX ... WHERE ...`),
/// so the claim index is a plain composite. The lease-sweep index on
/// `lease_expires_at` naturally excludes NULL rows in Oracle single-column
/// B-tree indexes (which is what we want — we only sweep claimed rows).
fn oracle_v10_up(schema: &SchemaConfig) -> Vec<String> {
    let q = &schema.background_queue;
    let r = &schema.responses;
    let queue_table = q.qualified_table(schema.owner.as_deref());
    let resp_table = r.qualified_table(schema.owner.as_deref());
    let resp_id_col = r.col("id").to_uppercase();

    let response_id = q.col("response_id").to_uppercase();
    let priority = q.col("priority").to_uppercase();
    let retry_attempt = q.col("retry_attempt").to_uppercase();
    let next_attempt_at = q.col("next_attempt_at").to_uppercase();
    let lease_expires_at = q.col("lease_expires_at").to_uppercase();
    let worker_id = q.col("worker_id").to_uppercase();
    let created_at = q.col("created_at").to_uppercase();
    let queue_table_name = q.table.to_uppercase();

    vec![
        // ORA-00955 = "name is already used by an existing object"
        format!(
            "BEGIN EXECUTE IMMEDIATE 'CREATE TABLE {queue_table} (\
                {response_id} VARCHAR2(64) PRIMARY KEY, \
                {priority} NUMBER(10) NOT NULL, \
                {retry_attempt} NUMBER(10) DEFAULT 0 NOT NULL, \
                {next_attempt_at} TIMESTAMP WITH TIME ZONE NOT NULL, \
                {lease_expires_at} TIMESTAMP WITH TIME ZONE, \
                {worker_id} VARCHAR2(256), \
                {created_at} TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL, \
                CONSTRAINT {queue_table_name}_FK FOREIGN KEY ({response_id}) \
                    REFERENCES {resp_table}({resp_id_col}) ON DELETE CASCADE\
            )'; EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF; END;"
        ),
        // Claim index — plain composite (Oracle has no partial indexes).
        format!(
            "BEGIN EXECUTE IMMEDIATE 'CREATE INDEX {queue_table_name}_CLAIM_IDX \
                ON {queue_table} ({priority}, {next_attempt_at}, {created_at})'; \
             EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF; END;"
        ),
        // Lease-sweep index — Oracle single-column B-tree indexes exclude
        // NULL rows by default, which matches the design's intent.
        format!(
            "BEGIN EXECUTE IMMEDIATE 'CREATE INDEX {queue_table_name}_LEASE_SWEEP_IDX \
                ON {queue_table} ({lease_expires_at})'; \
             EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF; END;"
        ),
    ]
}

/// Create the `response_stream_chunks` per-response SSE log table.
fn oracle_v11_up(schema: &SchemaConfig) -> Vec<String> {
    let c = &schema.response_stream_chunks;
    let r = &schema.responses;
    let chunks_table = c.qualified_table(schema.owner.as_deref());
    let resp_table = r.qualified_table(schema.owner.as_deref());
    let resp_id_col = r.col("id").to_uppercase();

    let response_id = c.col("response_id").to_uppercase();
    let sequence = c.col("sequence").to_uppercase();
    let event_type = c.col("event_type").to_uppercase();
    let data = c.col("data").to_uppercase();
    let created_at = c.col("created_at").to_uppercase();
    let chunks_table_name = c.table.to_uppercase();

    vec![
        format!(
            "BEGIN EXECUTE IMMEDIATE 'CREATE TABLE {chunks_table} (\
                {response_id} VARCHAR2(64) NOT NULL, \
                {sequence} NUMBER(19) NOT NULL, \
                {event_type} VARCHAR2(128) NOT NULL, \
                {data} CLOB CHECK ({data} IS JSON) NOT NULL, \
                {created_at} TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL, \
                CONSTRAINT {chunks_table_name}_PK PRIMARY KEY ({response_id}, {sequence}), \
                CONSTRAINT {chunks_table_name}_FK FOREIGN KEY ({response_id}) \
                    REFERENCES {resp_table}({resp_id_col}) ON DELETE CASCADE\
            )'; EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF; END;"
        ),
        // Cleanup index for the retention-window janitor.
        format!(
            "BEGIN EXECUTE IMMEDIATE 'CREATE INDEX {chunks_table_name}_CLEANUP_IDX \
                ON {chunks_table} ({created_at})'; \
             EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF; END;"
        ),
    ]
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

    // ── v9: extend responses with background-mode columns ─────────────────

    #[test]
    fn oracle_v9_up_adds_nine_columns_with_plsql_wrappers() {
        let schema = SchemaConfig::default();
        let stmts = oracle_v9_up(&schema);
        // 9 ADD + 2 UPDATE
        assert_eq!(stmts.len(), 11, "got: {stmts:?}");
        for col in [
            "STATUS",
            "BACKGROUND",
            "STREAM_ENABLED",
            "CANCEL_REQUESTED",
            "REQUEST_JSON",
            "REQUEST_CONTEXT_JSON",
            "STARTED_AT",
            "COMPLETED_AT",
            "NEXT_STREAM_SEQUENCE",
        ] {
            assert!(
                stmts
                    .iter()
                    .any(|s| s.contains(col) && s.contains("SQLCODE != -1430")),
                "missing PL/SQL-wrapped ADD for {col}: {stmts:?}"
            );
        }
        assert!(
            stmts.iter().any(|s| s.contains("NUMBER(1)")),
            "Oracle should use NUMBER(1) for booleans: {stmts:?}"
        );
        assert!(
            stmts.iter().any(|s| s.contains("TIMESTAMP WITH TIME ZONE")),
            "Oracle should use TIMESTAMP WITH TIME ZONE: {stmts:?}"
        );
    }

    // ── v10: create background_queue ───────────────────────────────────────

    #[test]
    fn oracle_v10_up_creates_table_and_two_indexes() {
        let schema = SchemaConfig::default();
        let stmts = oracle_v10_up(&schema);
        assert_eq!(stmts.len(), 3, "got: {stmts:?}");
        // Table name is lowercase (from qualified_table); columns/indexes are
        // uppercased to match existing Oracle migration convention.
        assert!(stmts[0].contains("CREATE TABLE background_queue"));
        assert!(stmts[0].contains("ON DELETE CASCADE"));
        assert!(
            stmts[0].contains("SQLCODE != -955"),
            "idempotency via ORA-00955: {stmts:?}"
        );
        assert!(stmts[1].contains("BACKGROUND_QUEUE_CLAIM_IDX"));
        // Oracle has no partial indexes — no WHERE clause on CREATE INDEX.
        assert!(
            !stmts[1].contains("WHERE"),
            "Oracle claim index must be plain composite (no WHERE): {stmts:?}"
        );
        assert!(stmts[2].contains("BACKGROUND_QUEUE_LEASE_SWEEP_IDX"));
    }

    // ── v11: create response_stream_chunks ─────────────────────────────────

    #[test]
    fn oracle_v11_up_creates_table_with_composite_pk() {
        let schema = SchemaConfig::default();
        let stmts = oracle_v11_up(&schema);
        assert_eq!(stmts.len(), 2, "got: {stmts:?}");
        assert!(stmts[0].contains("CREATE TABLE response_stream_chunks"));
        assert!(
            stmts[0].contains("PRIMARY KEY (RESPONSE_ID, SEQUENCE)"),
            "composite PK on (response_id, sequence): {stmts:?}"
        );
        assert!(stmts[0].contains("ON DELETE CASCADE"));
        assert!(stmts[1].contains("RESPONSE_STREAM_CHUNKS_CLEANUP_IDX"));
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

    #[test]
    fn oracle_v3_up_generates_per_column_plsql_drops() {
        let schema = SchemaConfig::default();
        let stmts = oracle_v3_up(&schema);
        assert_eq!(stmts.len(), 4);
        assert!(stmts[0].contains("OUTPUT"), "got: {}", stmts[0]);
        assert!(stmts[1].contains("METADATA"), "got: {}", stmts[1]);
        assert!(stmts[2].contains("INSTRUCTIONS"), "got: {}", stmts[2]);
        assert!(stmts[3].contains("TOOL_CALLS"), "got: {}", stmts[3]);
        for stmt in &stmts {
            assert!(stmt.contains("SQLCODE"), "got: {stmt}");
        }
    }

    #[test]
    fn oracle_v3_up_skips_when_output_is_used_by_another_field() {
        let mut schema = SchemaConfig::default();
        schema
            .responses
            .columns
            .insert("safety_identifier".to_string(), "OUTPUT".to_string());
        let stmts = oracle_v3_up(&schema);
        assert_eq!(stmts.len(), 3, "expected 3 statements (OUTPUT skipped)");
        for stmt in &stmts {
            assert!(
                !stmt.contains("DROP (OUTPUT)"),
                "should skip OUTPUT when mapped: {stmt}"
            );
        }
    }

    #[test]
    fn oracle_v4_up_creates_skills_table_and_index() {
        let schema = SchemaConfig::default();
        let stmts = oracle_v4_up(&schema);
        assert_eq!(stmts.len(), 2);
        assert!(stmts[0].contains("CREATE TABLE SKILLS"));
        assert!(stmts[0].contains("SKILL_ID VARCHAR2(64) PRIMARY KEY"));
        assert!(stmts[0].contains("SOURCE VARCHAR2(64) DEFAULT ''custom'' NOT NULL"));
        assert!(stmts[0].contains("CREATED_AT TIMESTAMP WITH TIME ZONE NOT NULL"));
        assert!(stmts[0].contains("UPDATED_AT TIMESTAMP WITH TIME ZONE NOT NULL"));
        assert!(stmts[1].contains("IDX_SKILLS_TENANT_NAME"));
    }

    #[test]
    fn oracle_v5_up_creates_skill_versions_table_and_index() {
        let schema = SchemaConfig::default();
        let stmts = oracle_v5_up(&schema);
        assert_eq!(stmts.len(), 2);
        assert!(stmts[0].contains("CREATE TABLE SKILL_VERSIONS"));
        assert!(stmts[0].contains("CHECK (FILE_MANIFEST IS JSON)"));
        assert!(stmts[0].contains("REFERENCES SKILLS(SKILL_ID) ON DELETE CASCADE"));
        assert!(stmts[0].contains("CREATED_AT TIMESTAMP WITH TIME ZONE NOT NULL"));
        assert!(stmts[1].contains("IDX_SKILL_VERSION_NUMBER"));
    }

    #[test]
    fn oracle_v6_up_creates_tenant_aliases_table() {
        let schema = SchemaConfig::default();
        let stmts = oracle_v6_up(&schema);
        assert_eq!(stmts.len(), 2);
        assert!(stmts[0].contains("CREATE TABLE TENANT_ALIASES"));
        assert!(stmts[0].contains("CREATED_AT TIMESTAMP WITH TIME ZONE NOT NULL"));
        assert!(stmts[0].contains("EXPIRES_AT TIMESTAMP WITH TIME ZONE"));
        assert!(stmts[1].contains("IDX_TENANT_ALIASES_CANONICAL"));
    }

    #[test]
    fn oracle_v7_up_creates_bundle_tokens_table() {
        let schema = SchemaConfig::default();
        let stmts = oracle_v7_up(&schema);
        assert_eq!(stmts.len(), 3);
        assert!(stmts[0].contains("CREATE TABLE BUNDLE_TOKENS"));
        assert!(stmts[0].contains("TOKEN_HASH VARCHAR2(64) PRIMARY KEY"));
        assert!(stmts[0].contains("CREATED_AT TIMESTAMP WITH TIME ZONE NOT NULL"));
        assert!(stmts[0].contains("EXPIRES_AT TIMESTAMP WITH TIME ZONE NOT NULL"));
        assert!(stmts[1].contains("IDX_BUNDLE_TOKENS_EXEC_ID"));
        assert!(stmts[2].contains("IDX_BUNDLE_TOKENS_EXPIRES_AT"));
    }

    #[test]
    fn oracle_v8_up_creates_continuation_cookies_table() {
        let schema = SchemaConfig::default();
        let stmts = oracle_v8_up(&schema);
        assert_eq!(stmts.len(), 3);
        assert!(stmts[0].contains("CREATE TABLE CONTINUATION_COOKIES"));
        assert!(stmts[0].contains("COOKIE_HASH VARCHAR2(64) PRIMARY KEY"));
        assert!(stmts[0].contains("CREATED_AT TIMESTAMP WITH TIME ZONE NOT NULL"));
        assert!(stmts[0].contains("EXPIRES_AT TIMESTAMP WITH TIME ZONE NOT NULL"));
        assert!(stmts[1].contains("IDX_CONTINUATION_COOKIES_EXEC"));
        assert!(stmts[2].contains("IDX_CONTINUATION_COOKIES_EXP"));
    }

    #[test]
    fn oracle_idempotent_ddl_escapes_single_quotes() {
        let stmt = oracle_idempotent_ddl("CREATE TABLE T (SOURCE VARCHAR2(64) DEFAULT 'custom')");
        assert!(stmt.contains("DEFAULT ''custom''"));
        assert!(stmt.contains("SQLCODE != -955"));
    }

    #[test]
    fn oracle_v4_to_v8_qualify_objects_with_owner() {
        let schema = SchemaConfig {
            owner: Some("OWNER".to_string()),
            ..Default::default()
        };

        let v4 = oracle_v4_up(&schema);
        assert!(v4[0].contains("CREATE TABLE OWNER.SKILLS"));
        assert!(v4[1].contains("CREATE INDEX OWNER.IDX_SKILLS_TENANT_NAME ON OWNER.SKILLS"));

        let v5 = oracle_v5_up(&schema);
        assert!(v5[0].contains("CREATE TABLE OWNER.SKILL_VERSIONS"));
        assert!(v5[0].contains("REFERENCES OWNER.SKILLS(SKILL_ID) ON DELETE CASCADE"));
        assert!(v5[1].contains(
            "CREATE UNIQUE INDEX OWNER.IDX_SKILL_VERSION_NUMBER ON OWNER.SKILL_VERSIONS"
        ));

        let v6 = oracle_v6_up(&schema);
        assert!(v6[0].contains("CREATE TABLE OWNER.TENANT_ALIASES"));
        assert!(v6[1]
            .contains("CREATE INDEX OWNER.IDX_TENANT_ALIASES_CANONICAL ON OWNER.TENANT_ALIASES"));

        let v7 = oracle_v7_up(&schema);
        assert!(v7[0].contains("CREATE TABLE OWNER.BUNDLE_TOKENS"));
        assert!(
            v7[1].contains("CREATE INDEX OWNER.IDX_BUNDLE_TOKENS_EXEC_ID ON OWNER.BUNDLE_TOKENS")
        );
        assert!(v7[2]
            .contains("CREATE INDEX OWNER.IDX_BUNDLE_TOKENS_EXPIRES_AT ON OWNER.BUNDLE_TOKENS"));

        let v8 = oracle_v8_up(&schema);
        assert!(v8[0].contains("CREATE TABLE OWNER.CONTINUATION_COOKIES"));
        assert!(v8[1].contains(
            "CREATE INDEX OWNER.IDX_CONTINUATION_COOKIES_EXEC ON OWNER.CONTINUATION_COOKIES"
        ));
        assert!(v8[2].contains(
            "CREATE INDEX OWNER.IDX_CONTINUATION_COOKIES_EXP ON OWNER.CONTINUATION_COOKIES"
        ));
    }
}
