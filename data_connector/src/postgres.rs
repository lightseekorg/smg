//! Postgres storage implementation using PostgresStore helper
//!
//! Structure:
//! 1. PostgresStore helper and common utilities
//! 2. PostgresConversationStorage
//! 3. PostgresConversationItemStorage
//! 4. PostgresResponseStorage

use std::{str::FromStr, sync::Arc};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use deadpool_postgres::{Manager, ManagerConfig, Pool, RecyclingMethod};
use serde_json::Value;
use tokio_postgres::{NoTls, Row};

use crate::{
    common::{
        build_response_select_base, parse_json_value, parse_metadata, parse_raw_response,
        parse_tool_calls,
    },
    config::PostgresConfig,
    core::{
        make_item_id, Conversation, ConversationId, ConversationItem, ConversationItemId,
        ConversationItemResult, ConversationItemStorage, ConversationItemStorageError,
        ConversationMetadata, ConversationResult, ConversationStorage, ConversationStorageError,
        ListParams, NewConversation, NewConversationItem, ResponseId, ResponseResult,
        ResponseStorage, ResponseStorageError, SortOrder, StoredResponse,
    },
    schema::SchemaConfig,
};

pub(crate) struct PostgresStore {
    pool: Pool,
    pub(crate) schema: Arc<SchemaConfig>,
}

impl PostgresStore {
    pub fn new(config: PostgresConfig) -> Result<Self, String> {
        let schema = config.schema.clone().unwrap_or_default();
        schema.validate()?;
        let schema = Arc::new(schema);

        let pg_config = tokio_postgres::Config::from_str(config.db_url.as_str())
            .map_err(|e| format!("Invalid PostgreSQL connection URL: {e}"))?;
        let mgr_config = ManagerConfig {
            recycling_method: RecyclingMethod::Fast,
        };
        let mgr = Manager::from_config(pg_config, NoTls, mgr_config);
        let pool = Pool::builder(mgr)
            .max_size(config.pool_max)
            .build()
            .map_err(|e| format!("Failed to build PostgreSQL connection pool: {e}"))?;

        Ok(Self { pool, schema })
    }
}

impl Clone for PostgresStore {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            schema: self.schema.clone(),
        }
    }
}

pub(super) struct PostgresConversationStorage {
    store: PostgresStore,
}

impl PostgresConversationStorage {
    pub async fn new(store: PostgresStore) -> Result<Self, ConversationStorageError> {
        let s = &store.schema.conversations;
        let table = s.qualified_table(store.schema.owner.as_deref());

        let col_defs = [
            format!("{} VARCHAR(64) PRIMARY KEY", s.col("id")),
            format!("{} TIMESTAMPTZ", s.col("created_at")),
            format!("{} JSON", s.col("metadata")),
        ];

        let ddl = format!(
            "CREATE TABLE IF NOT EXISTS {table} ({});",
            col_defs.join(", ")
        );

        let client = store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        client
            .batch_execute(&ddl)
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        Ok(Self { store })
    }

    fn parse_metadata(
        metadata: Option<String>,
    ) -> Result<Option<ConversationMetadata>, ConversationStorageError> {
        crate::common::parse_conversation_metadata(metadata)
            .map_err(ConversationStorageError::StorageError)
    }
}

#[async_trait]
impl ConversationStorage for PostgresConversationStorage {
    async fn create_conversation(
        &self,
        input: NewConversation,
    ) -> Result<Conversation, ConversationStorageError> {
        let conversation = Conversation::new(input);
        let id_str = conversation.id.0.as_str();
        let created_at: DateTime<Utc> = conversation.created_at;
        let metadata_json = conversation
            .metadata
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;

        let s = &self.store.schema.conversations;
        let table = s.qualified_table(self.store.schema.owner.as_deref());
        let col_id = s.col("id");
        let col_created = s.col("created_at");
        let col_meta = s.col("metadata");

        let sql = format!(
            "INSERT INTO {table} ({col_id}, {col_created}, {col_meta}) VALUES ($1, $2, $3)"
        );

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        client
            .execute(&sql, &[&id_str, &created_at, &metadata_json])
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        Ok(conversation)
    }

    async fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> Result<Option<Conversation>, ConversationStorageError> {
        let s = &self.store.schema.conversations;
        let table = s.qualified_table(self.store.schema.owner.as_deref());
        let col_id = s.col("id");
        let col_created = s.col("created_at");
        let col_meta = s.col("metadata");

        let sql =
            format!("SELECT {col_id}, {col_created}, {col_meta} FROM {table} WHERE {col_id} = $1");

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        let rows = client
            .query(&sql, &[&id.0.as_str()])
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        if rows.is_empty() {
            return Ok(None);
        }
        let row = &rows[0];
        let id_str: String = row.get(col_id);
        let created_at: DateTime<Utc> = row.get(col_created);
        let metadata_json: Option<String> = row.get(col_meta);
        let metadata = Self::parse_metadata(metadata_json)?;
        Ok(Some(Conversation::with_parts(
            ConversationId(id_str),
            created_at,
            metadata,
        )))
    }

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> Result<Option<Conversation>, ConversationStorageError> {
        let metadata_json = metadata.as_ref().map(serde_json::to_string).transpose()?;

        let s = &self.store.schema.conversations;
        let table = s.qualified_table(self.store.schema.owner.as_deref());
        let col_id = s.col("id");
        let col_meta = s.col("metadata");
        let col_created = s.col("created_at");

        let sql = format!(
            "UPDATE {table} SET {col_meta} = $1 WHERE {col_id} = $2 RETURNING {col_created}"
        );

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        let rows = client
            .query(&sql, &[&metadata_json, &id.0.as_str()])
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        if rows.is_empty() {
            return Ok(None);
        }
        let row = &rows[0];
        let created_at: DateTime<Utc> = row.get(col_created);
        Ok(Some(Conversation::with_parts(
            ConversationId(id.0.clone()),
            created_at,
            metadata,
        )))
    }

    async fn delete_conversation(&self, id: &ConversationId) -> ConversationResult<bool> {
        let s = &self.store.schema.conversations;
        let table = s.qualified_table(self.store.schema.owner.as_deref());
        let col_id = s.col("id");

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        let rows_deleted = client
            .execute(
                &format!("DELETE FROM {table} WHERE {col_id} = $1"),
                &[&id.0.as_str()],
            )
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        Ok(rows_deleted > 0)
    }
}

pub(super) struct PostgresConversationItemStorage {
    store: PostgresStore,
}

impl PostgresConversationItemStorage {
    pub async fn new(store: PostgresStore) -> Result<Self, ConversationItemStorageError> {
        let schema = &store.schema;
        let si = &schema.conversation_items;
        let sl = &schema.conversation_item_links;
        let items_table = si.qualified_table(schema.owner.as_deref());
        let links_table = sl.qualified_table(schema.owner.as_deref());

        // ── conversation_items DDL ──
        let item_col_defs = [
            format!("{} VARCHAR(64) PRIMARY KEY", si.col("id")),
            format!("{} VARCHAR(64)", si.col("response_id")),
            format!("{} VARCHAR(32) NOT NULL", si.col("item_type")),
            format!("{} VARCHAR(32)", si.col("role")),
            format!("{} JSON", si.col("content")),
            format!("{} VARCHAR(32)", si.col("status")),
            format!("{} TIMESTAMPTZ", si.col("created_at")),
        ];

        // ── conversation_item_links DDL ──
        let col_conv_id = sl.col("conversation_id");
        let col_item_id = sl.col("item_id");
        let col_added_at = sl.col("added_at");

        let mut link_col_defs = vec![
            format!("{col_conv_id} VARCHAR(64)"),
            format!("{col_item_id} VARCHAR(64) NOT NULL"),
            format!("{col_added_at} TIMESTAMPTZ"),
        ];
        link_col_defs.push(format!(
            "CONSTRAINT pk_conv_item_link PRIMARY KEY ({col_conv_id}, {col_item_id})"
        ));

        let ddl = format!(
            "CREATE TABLE IF NOT EXISTS {items_table} ({});\n\
             CREATE TABLE IF NOT EXISTS {links_table} ({});\n\
             CREATE INDEX IF NOT EXISTS conv_item_links_conv_idx ON {links_table} ({col_conv_id}, {col_added_at});",
            item_col_defs.join(", "),
            link_col_defs.join(", "),
        );

        let client = store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        client
            .batch_execute(&ddl)
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        Ok(Self { store })
    }
}

#[async_trait]
impl ConversationItemStorage for PostgresConversationItemStorage {
    async fn create_item(
        &self,
        item: NewConversationItem,
    ) -> Result<ConversationItem, ConversationItemStorageError> {
        let NewConversationItem {
            id: opt_id,
            response_id,
            item_type,
            role,
            content,
            status,
        } = item;
        let id = opt_id.unwrap_or_else(|| make_item_id(&item_type));
        let created_at = Utc::now();
        let content_json = serde_json::to_string(&content)?;

        let si = &self.store.schema.conversation_items;
        let table = si.qualified_table(self.store.schema.owner.as_deref());

        let col_id = si.col("id");
        let col_response_id = si.col("response_id");
        let col_item_type = si.col("item_type");
        let col_role = si.col("role");
        let col_content = si.col("content");
        let col_status = si.col("status");
        let col_created_at = si.col("created_at");

        let sql = format!(
            "INSERT INTO {table} ({col_id}, {col_response_id}, {col_item_type}, {col_role}, {col_content}, {col_status}, {col_created_at}) \
             VALUES ($1, $2, $3, $4, $5, $6, $7)"
        );

        let params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = vec![
            &id.0,
            &response_id,
            &item_type,
            &role,
            &content_json,
            &status,
            &created_at,
        ];

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        client
            .execute(&sql, &params)
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        Ok(ConversationItem {
            id,
            response_id,
            item_type,
            role,
            content,
            status,
            created_at,
        })
    }

    async fn link_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
        added_at: DateTime<Utc>,
    ) -> ConversationItemResult<()> {
        let sl = &self.store.schema.conversation_item_links;
        let table = sl.qualified_table(self.store.schema.owner.as_deref());
        let col_conv = sl.col("conversation_id");
        let col_item = sl.col("item_id");
        let col_added = sl.col("added_at");

        let sql = format!(
            "INSERT INTO {table} ({col_conv}, {col_item}, {col_added}) VALUES ($1, $2, $3)"
        );

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        client
            .execute(
                &sql,
                &[&conversation_id.0.as_str(), &item_id.0.as_str(), &added_at],
            )
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        Ok(())
    }

    async fn list_items(
        &self,
        conversation_id: &ConversationId,
        params: ListParams,
    ) -> ConversationItemResult<Vec<ConversationItem>> {
        let schema = &self.store.schema;
        let si = &schema.conversation_items;
        let sl = &schema.conversation_item_links;
        let items_table = si.qualified_table(schema.owner.as_deref());
        let links_table = sl.qualified_table(schema.owner.as_deref());

        let l_conv_id = sl.col("conversation_id");
        let l_item_id = sl.col("item_id");
        let l_added_at = sl.col("added_at");
        let i_id = si.col("id");

        let cid = conversation_id.0.as_str();
        let limit: i64 = params.limit as i64;
        let order_desc = matches!(params.order, SortOrder::Desc);

        let after_key: Option<(DateTime<Utc>, String)> = if let Some(ref aid) = params.after {
            let client = self
                .store
                .pool
                .get()
                .await
                .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
            let cursor_sql = format!(
                "SELECT {l_added_at} FROM {links_table} WHERE {l_conv_id} = $1 AND {l_item_id} = $2"
            );
            let rows = client
                .query(&cursor_sql, &[&cid, &aid.as_str()])
                .await
                .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
            if rows.is_empty() {
                None
            } else {
                let row = &rows[0];
                let ts: DateTime<Utc> = row.get(0);
                Some((ts, aid.clone()))
            }
        } else {
            None
        };

        // Build select columns from items table (prefixed with i.)
        let mut select_cols = Vec::new();
        for field in &[
            "id",
            "response_id",
            "item_type",
            "role",
            "content",
            "status",
            "created_at",
        ] {
            select_cols.push(format!("i.{}", si.col(field)));
        }

        let mut sql = format!(
            "SELECT {} FROM {links_table} l JOIN {items_table} i ON i.{i_id} = l.{l_item_id} \
             WHERE l.{l_conv_id} = $1",
            select_cols.join(", "),
        );

        if let Some((_ts, _iid)) = &after_key {
            if order_desc {
                sql.push_str(&format!(
                    " AND (l.{l_added_at} < $2 OR (l.{l_added_at} = $2 AND l.{l_item_id} < $3))"
                ));
            } else {
                sql.push_str(&format!(
                    " AND (l.{l_added_at} > $2 OR (l.{l_added_at} = $2 AND l.{l_item_id} > $3))"
                ));
            }
        }
        if order_desc {
            sql.push_str(&format!(
                " ORDER BY l.{l_added_at} DESC, l.{l_item_id} DESC"
            ));
        } else {
            sql.push_str(&format!(" ORDER BY l.{l_added_at} ASC, l.{l_item_id} ASC"));
        }
        if after_key.is_some() {
            sql.push_str(" LIMIT $4");
        } else {
            sql.push_str(" LIMIT $2");
        }

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        let rows = if let Some((ts, iid)) = &after_key {
            client
                .query(&sql, &[&cid, ts, iid, &limit])
                .await
                .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?
        } else {
            client
                .query(&sql, &[&cid, &limit])
                .await
                .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?
        };

        let col_id = si.col("id");
        let col_resp_id = si.col("response_id");
        let col_item_type = si.col("item_type");
        let col_role = si.col("role");
        let col_content = si.col("content");
        let col_status = si.col("status");
        let col_created_at = si.col("created_at");

        let mut out = Vec::new();
        for row in rows {
            let id: String = row.get(col_id);
            let resp_id: Option<String> = row.get(col_resp_id);
            let item_type: String = row.get(col_item_type);
            let role: Option<String> = row.get(col_role);
            let content_raw: Option<String> = row.get(col_content);
            let status: Option<String> = row.get(col_status);
            let created_at: DateTime<Utc> = row.get(col_created_at);

            let content = match content_raw {
                Some(s) => serde_json::from_str(&s).map_err(ConversationItemStorageError::from)?,
                None => Value::Null,
            };
            out.push(ConversationItem {
                id: ConversationItemId(id),
                response_id: resp_id,
                item_type,
                role,
                content,
                status,
                created_at,
            });
        }
        Ok(out)
    }

    async fn get_item(
        &self,
        item_id: &ConversationItemId,
    ) -> Result<Option<ConversationItem>, ConversationItemStorageError> {
        let si = &self.store.schema.conversation_items;
        let table = si.qualified_table(self.store.schema.owner.as_deref());
        let col_id = si.col("id");

        let mut select_cols = Vec::new();
        for field in &[
            "id",
            "response_id",
            "item_type",
            "role",
            "content",
            "status",
            "created_at",
        ] {
            select_cols.push(si.col(field).to_string());
        }

        let sql = format!(
            "SELECT {} FROM {table} WHERE {col_id} = $1",
            select_cols.join(", "),
        );

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        let rows = client
            .query(&sql, &[&item_id.0.as_str()])
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        if rows.is_empty() {
            Ok(None)
        } else {
            let row = &rows[0];
            let id: String = row.get(si.col("id"));
            let response_id: Option<String> = row.get(si.col("response_id"));
            let item_type: String = row.get(si.col("item_type"));
            let role: Option<String> = row.get(si.col("role"));
            let content_raw: Option<String> = row.get(si.col("content"));
            let status: Option<String> = row.get(si.col("status"));
            let created_at: DateTime<Utc> = row.get(si.col("created_at"));

            let content = match content_raw {
                Some(s) => serde_json::from_str(&s)
                    .map_err(ConversationItemStorageError::SerializationError)?,
                None => Value::Null,
            };

            Ok(Some(ConversationItem {
                id: ConversationItemId(id),
                response_id,
                item_type,
                role,
                content,
                status,
                created_at,
            }))
        }
    }

    async fn is_item_linked(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<bool> {
        let sl = &self.store.schema.conversation_item_links;
        let table = sl.qualified_table(self.store.schema.owner.as_deref());
        let col_conv = sl.col("conversation_id");
        let col_item = sl.col("item_id");

        let sql = format!("SELECT COUNT(*) FROM {table} WHERE {col_conv} = $1 AND {col_item} = $2");

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        let row = client
            .query_one(&sql, &[&conversation_id.0.as_str(), &item_id.0.as_str()])
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        let count: i64 = row.get(0);
        Ok(count > 0)
    }

    async fn delete_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<()> {
        let sl = &self.store.schema.conversation_item_links;
        let table = sl.qualified_table(self.store.schema.owner.as_deref());
        let col_conv = sl.col("conversation_id");
        let col_item = sl.col("item_id");

        let sql = format!("DELETE FROM {table} WHERE {col_conv} = $1 AND {col_item} = $2");

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        client
            .execute(&sql, &[&conversation_id.0.as_str(), &item_id.0.as_str()])
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        Ok(())
    }
}

pub(super) struct PostgresResponseStorage {
    store: PostgresStore,
    select_base: String,
}

impl PostgresResponseStorage {
    pub async fn new(store: PostgresStore) -> Result<Self, ResponseStorageError> {
        let schema = &store.schema;
        let s = &schema.responses;
        let table = s.qualified_table(schema.owner.as_deref());

        let col_defs = vec![
            format!("{} VARCHAR(64) PRIMARY KEY", s.col("id")),
            format!("{} VARCHAR(64)", s.col("conversation_id")),
            format!("{} VARCHAR(64)", s.col("previous_response_id")),
            format!("{} JSON", s.col("input")),
            format!("{} TEXT", s.col("instructions")),
            format!("{} JSON", s.col("output")),
            format!("{} JSON", s.col("tool_calls")),
            format!("{} JSON", s.col("metadata")),
            format!("{} TIMESTAMPTZ", s.col("created_at")),
            format!("{} VARCHAR(128)", s.col("safety_identifier")),
            format!("{} VARCHAR(128)", s.col("model")),
            format!("{} JSON", s.col("raw_response")),
        ];

        let mut ddl = format!(
            "CREATE TABLE IF NOT EXISTS {table} ({});",
            col_defs.join(", "),
        );
        ddl.push_str(&format!(
            "\nCREATE INDEX IF NOT EXISTS responses_safety_idx ON {table} ({});",
            s.col("safety_identifier")
        ));

        let client = store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        client
            .batch_execute(&ddl)
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        let select_base = build_response_select_base(&store.schema);
        Ok(Self { store, select_base })
    }

    pub fn build_response_from_row(
        row: &Row,
        schema: &SchemaConfig,
    ) -> Result<StoredResponse, String> {
        let s = &schema.responses;

        let id: String = row.get(s.col("id"));
        let conversation_id: Option<String> = row.get(s.col("conversation_id"));
        let previous: Option<String> = row.get(s.col("previous_response_id"));
        let input_json: Option<String> = row.get(s.col("input"));
        let instructions: Option<String> = row.get(s.col("instructions"));
        let output_json: Option<String> = row.get(s.col("output"));
        let tool_calls_json: Option<String> = row.get(s.col("tool_calls"));
        let metadata_json: Option<String> = row.get(s.col("metadata"));
        let created_at: DateTime<Utc> = row.get(s.col("created_at"));
        let safety_identifier: Option<String> = row.get(s.col("safety_identifier"));
        let model: Option<String> = row.get(s.col("model"));
        let raw_response_json: Option<String> = row.get(s.col("raw_response"));

        let previous_response_id = previous.map(ResponseId);
        let tool_calls = parse_tool_calls(tool_calls_json)?;
        let metadata = parse_metadata(metadata_json)?;
        let raw_response = parse_raw_response(raw_response_json)?;
        let input = parse_json_value(input_json)?;
        let output = parse_json_value(output_json)?;

        Ok(StoredResponse {
            id: ResponseId(id),
            previous_response_id,
            input,
            instructions,
            output,
            tool_calls,
            metadata,
            created_at,
            safety_identifier,
            model,
            conversation_id,
            raw_response,
        })
    }
}

#[async_trait]
impl ResponseStorage for PostgresResponseStorage {
    async fn store_response(
        &self,
        response: StoredResponse,
    ) -> Result<ResponseId, ResponseStorageError> {
        let StoredResponse {
            id: response_id,
            previous_response_id,
            input,
            instructions,
            output,
            tool_calls,
            metadata,
            created_at,
            safety_identifier,
            model,
            conversation_id,
            raw_response,
        } = response;
        let previous_id = previous_response_id.map(|r| r.0);
        let tool_calls_value = serde_json::to_value(&tool_calls)?;
        let metadata_value = serde_json::to_value(&metadata)?;

        let s = &self.store.schema.responses;
        let table = s.qualified_table(self.store.schema.owner.as_deref());

        let col_id = s.col("id");
        let col_prev = s.col("previous_response_id");
        let col_input = s.col("input");
        let col_instructions = s.col("instructions");
        let col_output = s.col("output");
        let col_tool_calls = s.col("tool_calls");
        let col_metadata = s.col("metadata");
        let col_created_at = s.col("created_at");
        let col_safety = s.col("safety_identifier");
        let col_model = s.col("model");
        let col_conv = s.col("conversation_id");
        let col_raw = s.col("raw_response");

        let sql = format!(
            "INSERT INTO {table} ({col_id}, {col_prev}, {col_input}, {col_instructions}, {col_output}, \
             {col_tool_calls}, {col_metadata}, {col_created_at}, {col_safety}, {col_model}, {col_conv}, {col_raw}) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)"
        );

        let params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = vec![
            &response_id.0,
            &previous_id,
            &input,
            &instructions,
            &output,
            &tool_calls_value,
            &metadata_value,
            &created_at,
            &safety_identifier,
            &model,
            &conversation_id,
            &raw_response,
        ];

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        let insert_count = client
            .execute(&sql, &params)
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        tracing::debug!(rows_affected = insert_count, "Response stored in Postgres");
        Ok(response_id)
    }

    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> Result<Option<StoredResponse>, ResponseStorageError> {
        let col_id = self.store.schema.responses.col("id");
        let sql = format!("{} WHERE {col_id} = $1", self.select_base);

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        let rows = client
            .query(&sql, &[&response_id.0.as_str()])
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        if rows.is_empty() {
            return Ok(None);
        }
        Self::build_response_from_row(&rows[0], &self.store.schema)
            .map(Some)
            .map_err(|err| ResponseStorageError::StorageError(err.to_string()))
    }

    async fn delete_response(&self, response_id: &ResponseId) -> ResponseResult<()> {
        let s = &self.store.schema.responses;
        let table = s.qualified_table(self.store.schema.owner.as_deref());
        let col_id = s.col("id");

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        client
            .execute(
                &format!("DELETE FROM {table} WHERE {col_id} = $1"),
                &[&response_id.0.as_str()],
            )
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        Ok(())
    }

    async fn list_identifier_responses(
        &self,
        identifier: &str,
        limit: Option<usize>,
    ) -> ResponseResult<Vec<StoredResponse>> {
        let s = &self.store.schema.responses;
        let col_safety = s.col("safety_identifier");
        let col_created = s.col("created_at");

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        let rows = if let Some(l) = limit {
            let l_i64: i64 = l as i64;
            let sql = format!(
                "{} WHERE {col_safety} = $1 ORDER BY {col_created} DESC LIMIT $2",
                self.select_base,
            );
            client
                .query(&sql, &[&identifier, &l_i64])
                .await
                .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?
        } else {
            let sql = format!(
                "{} WHERE {col_safety} = $1 ORDER BY {col_created} DESC",
                self.select_base,
            );
            client
                .query(&sql, &[&identifier])
                .await
                .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?
        };

        let schema = &self.store.schema;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let resp = Self::build_response_from_row(&row, schema)
                .map_err(ResponseStorageError::StorageError)?;
            out.push(resp);
        }

        Ok(out)
    }

    async fn delete_identifier_responses(&self, identifier: &str) -> ResponseResult<usize> {
        let s = &self.store.schema.responses;
        let table = s.qualified_table(self.store.schema.owner.as_deref());
        let col_safety = s.col("safety_identifier");

        let sql = format!("DELETE FROM {table} WHERE {col_safety} = $1");

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        let rows_deleted = client
            .execute(&sql, &[&identifier])
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        Ok(rows_deleted as usize)
    }
}
