//! Oracle-specific schema migrations.
//!
//! Each migration is a function that generates Oracle DDL from [`SchemaConfig`],
//! so it respects custom table/column names. PL/SQL exception handling ensures
//! idempotency (safe to re-run if a previous attempt partially completed).

use crate::{schema::SchemaConfig, versioning::Migration};

/// Oracle migration list. Append new migrations here.
pub(crate) static ORACLE_MIGRATIONS: [Migration; 8] = [
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

fn oracle_v4_up(schema: &SchemaConfig) -> Vec<String> {
    let table = oracle_qualified_name(schema, "SKILLS");
    let index = oracle_qualified_name(schema, "IDX_SKILLS_TENANT_NAME");
    vec![
        format!(
            "BEGIN EXECUTE IMMEDIATE 'CREATE TABLE {table} (\
             ID VARCHAR2(64) PRIMARY KEY, \
             TENANT_ID VARCHAR2(64) NOT NULL, \
             NAME VARCHAR2(64) NOT NULL, \
             SHORT_DESCRIPTION CLOB, \
             DESCRIPTION CLOB, \
             SOURCE VARCHAR2(64) DEFAULT ''custom'' NOT NULL, \
             HAS_CODE_FILES NUMBER(1) DEFAULT 0 NOT NULL, \
             LATEST_VERSION VARCHAR2(64), \
             DEFAULT_VERSION VARCHAR2(64), \
             CREATED_AT TIMESTAMP WITH TIME ZONE NOT NULL, \
             UPDATED_AT TIMESTAMP WITH TIME ZONE NOT NULL)'; \
             EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF; END;"
        ),
        format!(
            "BEGIN EXECUTE IMMEDIATE 'CREATE INDEX {index} ON {table} (TENANT_ID, NAME)'; \
             EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF; END;"
        ),
    ]
}

fn oracle_v5_up(schema: &SchemaConfig) -> Vec<String> {
    let skills_table = oracle_qualified_name(schema, "SKILLS");
    let table = oracle_qualified_name(schema, "SKILL_VERSIONS");
    let index = oracle_qualified_name(schema, "IDX_SKILL_VERSION_NUMBER");
    vec![
        format!(
            "BEGIN EXECUTE IMMEDIATE 'CREATE TABLE {table} (\
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
             CONSTRAINT FK_SKILL_VERSIONS_SKILL FOREIGN KEY (SKILL_ID) REFERENCES {skills_table}(ID) ON DELETE CASCADE)'; \
             EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF; END;"
        ),
        format!(
            "BEGIN EXECUTE IMMEDIATE 'CREATE UNIQUE INDEX {index} ON {table} (SKILL_ID, VERSION_NUMBER)'; \
             EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF; END;"
        ),
    ]
}

fn oracle_v6_up(schema: &SchemaConfig) -> Vec<String> {
    let table = oracle_qualified_name(schema, "TENANT_ALIASES");
    let index = oracle_qualified_name(schema, "IDX_TENANT_ALIASES_CANONICAL");
    vec![
        format!(
            "BEGIN EXECUTE IMMEDIATE 'CREATE TABLE {table} (\
             ALIAS_TENANT_ID VARCHAR2(64) PRIMARY KEY, \
             CANONICAL_TENANT_ID VARCHAR2(64) NOT NULL, \
             CREATED_AT TIMESTAMP WITH TIME ZONE NOT NULL, \
             EXPIRES_AT TIMESTAMP WITH TIME ZONE)'; \
             EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF; END;"
        ),
        format!(
            "BEGIN EXECUTE IMMEDIATE 'CREATE INDEX {index} ON {table} (CANONICAL_TENANT_ID)'; \
             EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF; END;"
        ),
    ]
}

fn oracle_v7_up(schema: &SchemaConfig) -> Vec<String> {
    let table = oracle_qualified_name(schema, "BUNDLE_TOKENS");
    let exec_index = oracle_qualified_name(schema, "IDX_BUNDLE_TOKENS_EXEC_ID");
    let expires_index = oracle_qualified_name(schema, "IDX_BUNDLE_TOKENS_EXPIRES_AT");
    vec![
        format!(
            "BEGIN EXECUTE IMMEDIATE 'CREATE TABLE {table} (\
             TOKEN VARCHAR2(64) PRIMARY KEY, \
             TENANT_ID VARCHAR2(64) NOT NULL, \
             EXEC_ID VARCHAR2(64) NOT NULL, \
             SKILL_ID VARCHAR2(64) NOT NULL, \
             SKILL_VERSION VARCHAR2(64) NOT NULL, \
             CREATED_AT TIMESTAMP WITH TIME ZONE NOT NULL, \
             EXPIRES_AT TIMESTAMP WITH TIME ZONE NOT NULL)'; \
             EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF; END;"
        ),
        format!(
            "BEGIN EXECUTE IMMEDIATE 'CREATE INDEX {exec_index} ON {table} (EXEC_ID)'; \
             EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF; END;"
        ),
        format!(
            "BEGIN EXECUTE IMMEDIATE 'CREATE INDEX {expires_index} ON {table} (EXPIRES_AT)'; \
             EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF; END;"
        ),
    ]
}

fn oracle_v8_up(schema: &SchemaConfig) -> Vec<String> {
    let table = oracle_qualified_name(schema, "CONTINUATION_COOKIES");
    let exec_index = oracle_qualified_name(schema, "IDX_CONTINUATION_COOKIES_EXEC");
    let expires_index = oracle_qualified_name(schema, "IDX_CONTINUATION_COOKIES_EXP");
    vec![
        format!(
            "BEGIN EXECUTE IMMEDIATE 'CREATE TABLE {table} (\
             COOKIE VARCHAR2(64) PRIMARY KEY, \
             TENANT_ID VARCHAR2(64) NOT NULL, \
             EXEC_ID VARCHAR2(64) NOT NULL, \
             REQUEST_ID VARCHAR2(64) NOT NULL, \
             CREATED_AT TIMESTAMP WITH TIME ZONE NOT NULL, \
             EXPIRES_AT TIMESTAMP WITH TIME ZONE NOT NULL)'; \
             EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF; END;"
        ),
        format!(
            "BEGIN EXECUTE IMMEDIATE 'CREATE INDEX {exec_index} ON {table} (EXEC_ID)'; \
             EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF; END;"
        ),
        format!(
            "BEGIN EXECUTE IMMEDIATE 'CREATE INDEX {expires_index} ON {table} (EXPIRES_AT)'; \
             EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF; END;"
        ),
    ]
}

fn oracle_qualified_name(schema: &SchemaConfig, object_name: &str) -> String {
    match &schema.owner {
        Some(owner) => format!("{owner}.{object_name}"),
        None => object_name.to_string(),
    }
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
        assert!(stmts[0].contains("REFERENCES SKILLS(ID) ON DELETE CASCADE"));
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
        assert!(stmts[0].contains("CREATED_AT TIMESTAMP WITH TIME ZONE NOT NULL"));
        assert!(stmts[0].contains("EXPIRES_AT TIMESTAMP WITH TIME ZONE NOT NULL"));
        assert!(stmts[1].contains("IDX_CONTINUATION_COOKIES_EXEC"));
        assert!(stmts[2].contains("IDX_CONTINUATION_COOKIES_EXP"));
    }
}
