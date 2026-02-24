//! Oracle storage implementation using OracleStore helper.
//!
//! Structure:
//! 1. OracleStore helper and common utilities
//! 2. OracleConversationStorage
//! 3. OracleConversationItemStorage
//! 4. OracleResponseStorage

use std::{path::Path, sync::Arc, time::Duration};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use deadpool::managed::{Manager, Metrics, Pool, RecycleError, RecycleResult};
use oracle::{Connection, Connector, Row};
use serde_json::Value;

use super::core::{
    make_item_id, Conversation, ConversationId, ConversationItem, ConversationItemId,
    ConversationItemStorage, ConversationItemStorageError, ConversationMetadata,
    ConversationStorage, ConversationStorageError, ListParams, NewConversation,
    NewConversationItem, ResponseChain, ResponseId, ResponseStorage, ResponseStorageError,
    SortOrder, StoredResponse,
};
use crate::{
    common::{parse_json_value, parse_metadata, parse_raw_response, parse_tool_calls},
    config::OracleConfig,
};
// ============================================================================
// PART 1: OracleStore Helper + Common Utilities
// ============================================================================

/// Schema initializer function signature for Oracle storage backends.
pub(crate) type SchemaInitFn = fn(&Connection) -> Result<(), String>;

/// Shared Oracle connection pool infrastructure.
///
/// This helper eliminates ~540 LOC of duplication across storage implementations.
/// It handles connection pooling, error mapping, and client configuration.
pub(crate) struct OracleStore {
    pool: Pool<OracleConnectionManager>,
}

impl OracleStore {
    /// Create a connection pool and initialize all schemas.
    ///
    /// Accepts a list of schema initializers that run on a single connection
    /// before the pool is created, ensuring all tables exist.
    pub fn new(config: &OracleConfig, init_schemas: &[SchemaInitFn]) -> Result<Self, String> {
        // Configure Oracle client (wallet env vars, etc.)
        configure_oracle_env(config)?;

        // Initialize schemas using a single connection
        let conn = connect_oracle(
            config.external_auth,
            &config.username,
            &config.password,
            &config.connect_descriptor,
        )
        .map_err(map_oracle_error)?;

        for init_schema in init_schemas {
            init_schema(&conn)?;
        }
        drop(conn);

        // Create connection pool
        let config_arc = Arc::new(config.clone());
        let manager = OracleConnectionManager {
            params: Arc::new(OracleConnectParams::from_config(&config_arc)),
        };

        let mut builder = Pool::builder(manager)
            .max_size(config.pool_max)
            .runtime(deadpool::Runtime::Tokio1);

        if config.pool_timeout_secs > 0 {
            builder = builder.wait_timeout(Some(Duration::from_secs(config.pool_timeout_secs)));
        }

        let pool = builder
            .build()
            .map_err(|e| format!("Failed to build Oracle pool: {e}"))?;

        Ok(Self { pool })
    }

    /// Execute function with a connection from the pool
    pub async fn execute<F, T>(&self, func: F) -> Result<T, String>
    where
        F: FnOnce(&Connection) -> Result<T, String> + Send + 'static,
        T: Send + 'static,
    {
        let connection = self
            .pool
            .get()
            .await
            .map_err(|e| format!("Failed to get Oracle connection: {e}"))?;

        tokio::task::spawn_blocking(move || {
            let result = func(&connection);
            drop(connection);
            result
        })
        .await
        .map_err(|e| format!("Task execution failed: {e}"))?
    }
}

impl Clone for OracleStore {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
        }
    }
}

// Error mapping helper
pub(crate) fn map_oracle_error(err: oracle::Error) -> String {
    if let Some(db_err) = err.db_error() {
        format!(
            "Oracle error (code {}): {}",
            db_err.code(),
            db_err.message()
        )
    } else {
        err.to_string()
    }
}

/// Validate Oracle wallet path and set `TNS_ADMIN` environment variable.
///
/// # Thread-safety note
///
/// `std::env::set_var` is not thread-safe and is marked unsafe in Rust 2024 edition.
/// This function is called once during `OracleStore::new()` initialization, before the
/// connection pool is created and before any worker threads are spawned.  When migrating
/// to edition 2024, this call will need to be wrapped in `unsafe` (requires removing the
/// workspace `unsafe_code = "deny"` lint or adding a targeted `#[expect]`), or the
/// environment variable should be set by the process launcher instead.
pub(crate) fn configure_oracle_env(config: &OracleConfig) -> Result<(), String> {
    if let Some(wallet_path) = &config.wallet_path {
        let path = Path::new(wallet_path);

        if !path.is_dir() {
            return Err(format!(
                "Oracle wallet path '{wallet_path}' is not a directory"
            ));
        }

        if !path.join("tnsnames.ora").exists() && !path.join("sqlnet.ora").exists() {
            return Err(format!(
                "Oracle wallet path '{wallet_path}' is missing tnsnames.ora or sqlnet.ora"
            ));
        }

        std::env::set_var("TNS_ADMIN", wallet_path);
    }
    Ok(())
}

// Connection parameters
#[derive(Clone)]
pub(crate) struct OracleConnectParams {
    pub username: String,
    pub password: String,
    pub connect_descriptor: String,
    pub external_auth: bool,
}

impl OracleConnectParams {
    pub fn from_config(config: &OracleConfig) -> Self {
        Self {
            username: config.username.clone(),
            password: config.password.clone(),
            connect_descriptor: config.connect_descriptor.clone(),
            external_auth: config.external_auth,
        }
    }
}

impl std::fmt::Debug for OracleConnectParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OracleConnectParams")
            .field("username", &self.username)
            .field("connect_descriptor", &self.connect_descriptor)
            .field("external_auth", &self.external_auth)
            .finish()
    }
}

// Connection manager (same for all stores)
#[derive(Clone)]
struct OracleConnectionManager {
    params: Arc<OracleConnectParams>,
}

impl std::fmt::Debug for OracleConnectionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OracleConnectionManager")
            .field("username", &self.params.username)
            .field("connect_descriptor", &self.params.connect_descriptor)
            .finish()
    }
}

#[async_trait]
impl Manager for OracleConnectionManager {
    type Type = Connection;
    type Error = oracle::Error;

    fn create(
        &self,
    ) -> impl std::future::Future<Output = Result<Connection, oracle::Error>> + Send {
        let params = self.params.clone();
        async move {
            let mut conn = connect_oracle(
                params.external_auth,
                &params.username,
                &params.password,
                &params.connect_descriptor,
            )?;
            conn.set_autocommit(true);
            Ok(conn)
        }
    }

    #[expect(clippy::manual_async_fn)]
    fn recycle(
        &self,
        conn: &mut Connection,
        _: &Metrics,
    ) -> impl std::future::Future<Output = RecycleResult<Self::Error>> + Send {
        async move { conn.ping().map_err(RecycleError::Backend) }
    }
}

fn connect_oracle(
    external_auth: bool,
    username: &str,
    password: &str,
    connect_descriptor: &str,
) -> Result<Connection, oracle::Error> {
    if external_auth {
        Connector::new("", "", connect_descriptor)
            .external_auth(true)
            .connect()
    } else {
        Connection::connect(username, password, connect_descriptor)
    }
}

// ============================================================================
// PART 2: OracleConversationStorage
// ============================================================================

#[derive(Clone)]
pub(super) struct OracleConversationStorage {
    store: OracleStore,
}

impl OracleConversationStorage {
    pub fn new(store: OracleStore) -> Self {
        Self { store }
    }

    pub(crate) fn init_schema(conn: &Connection) -> Result<(), String> {
        let exists: i64 = conn
            .query_row_as(
                // Downstream expects tables to exist in ADMIN schema.
                "SELECT COUNT(*) FROM all_tables WHERE owner = 'ADMIN' AND table_name = 'CONVERSATIONS'",
                &[],
            )
            .map_err(map_oracle_error)?;

        if exists == 0 {
            return Err("CONVERSATIONS table does not exist. Please create the table.".to_string());
        }

        Ok(())
    }

    fn parse_metadata(
        raw: Option<String>,
    ) -> Result<Option<ConversationMetadata>, ConversationStorageError> {
        match raw {
            Some(json) if !json.is_empty() => {
                let value: Value = serde_json::from_str(&json)?;
                match value {
                    Value::Object(map) => Ok(Some(map)),
                    Value::Null => Ok(None),
                    other => Err(ConversationStorageError::StorageError(format!(
                        "conversation metadata expected object, got {other}"
                    ))),
                }
            }
            _ => Ok(None),
        }
    }
}

#[async_trait]
impl ConversationStorage for OracleConversationStorage {
    async fn create_conversation(
        &self,
        input: NewConversation,
    ) -> Result<Conversation, ConversationStorageError> {
        let conversation = Conversation::new(input.clone());
        let id_str = conversation.id.0.clone();
        // Read conversation_store_id from task-local.
        let conversation_store_id = super::core::CONVERSATION_STORE_ID
            .try_with(|id| id.clone())
            .ok()
            .flatten();
        let created_at = conversation.created_at;
        let metadata_json = conversation
            .metadata
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        let expires_at = created_at + chrono::Duration::hours(24);

        self.store
            .execute(move |conn| {
                conn.execute(
                    "INSERT INTO ADMIN.\"CONVERSATIONS\" (\"CONVERSATION_ID\", \"CONVERSATION_STORE_ID\", \"GENERATIVE_AI_PROJECT_ID\", \"CREATED_AT\", \"METADATA\", \"ITEMS\", \"UPDATED_AT\", \"EXPIRES_AT\", \"VERSION\", \"SHORT_TERM_MEMORY\") VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10)",
                    &[&id_str, &conversation_store_id, &None::<String>, &created_at, &metadata_json, &"[]", &created_at, &expires_at, &0, &None::<String>],
                )
                .map(|_| ())
                .map_err(map_oracle_error)
            })
            .await
            .map_err(ConversationStorageError::StorageError)?;

        Ok(conversation)
    }

    async fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> Result<Option<Conversation>, ConversationStorageError> {
        let lookup = id.0.clone();
        self.store
            .execute(move |conn| {
                let mut stmt = conn
                    .statement(
                        "SELECT \"CONVERSATION_ID\", \"CREATED_AT\", \"METADATA\", \"GENERATIVE_AI_PROJECT_ID\", \"UPDATED_AT\", \"VERSION\", \"SHORT_TERM_MEMORY\", \"EXPIRES_AT\" FROM ADMIN.\"CONVERSATIONS\" WHERE \"CONVERSATION_ID\" = :1",
                    )
                    .build()
                    .map_err(map_oracle_error)?;
                let mut rows = stmt.query(&[&lookup]).map_err(map_oracle_error)?;

                if let Some(row_res) = rows.next() {
                    let row = row_res.map_err(map_oracle_error)?;
                    let id: String = row.get(0).map_err(map_oracle_error)?;
                    let created_at: DateTime<Utc> = row.get(1).map_err(map_oracle_error)?;
                    let metadata_raw: Option<String> = row.get(2).map_err(map_oracle_error)?;
                    let metadata = Self::parse_metadata(metadata_raw).map_err(|e| e.to_string())?;
                    Ok(Some(Conversation::with_parts(
                        ConversationId(id),
                        created_at,
                        metadata,
                    )))
                } else {
                    Ok(None)
                }
            })
            .await
            .map_err(ConversationStorageError::StorageError)
    }

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> Result<Option<Conversation>, ConversationStorageError> {
        let id_str = id.0.clone();
        let metadata_json = metadata.as_ref().map(serde_json::to_string).transpose()?;
        let now = Utc::now();
        let conversation_id = id.clone();

        self.store
            .execute(move |conn| {
                let res = conn
                    .execute(
                        "UPDATE ADMIN.\"CONVERSATIONS\" SET \"METADATA\" = :1, \"UPDATED_AT\" = :2, \"GENERATIVE_AI_PROJECT_ID\" = :3, \"SHORT_TERM_MEMORY\" = :4 WHERE \"CONVERSATION_ID\" = :5",
                        &[&metadata_json, &now, &None::<String>, &None::<String>, &id_str],
                    )
                    .map_err(map_oracle_error)?;

                if res.row_count().map_err(map_oracle_error)? == 0 {
                    return Ok(None);
                }

                let created_at: DateTime<Utc> = conn
                    .query_row_as(
                        "SELECT \"CREATED_AT\" FROM ADMIN.\"CONVERSATIONS\" WHERE \"CONVERSATION_ID\" = :1",
                        &[&id_str],
                    )
                    .map_err(map_oracle_error)?;

                Ok(Some(Conversation::with_parts(conversation_id, created_at, metadata)))
            })
            .await
            .map_err(ConversationStorageError::StorageError)
    }

    async fn delete_conversation(
        &self,
        id: &ConversationId,
    ) -> Result<bool, ConversationStorageError> {
        let id_str = id.0.clone();
        let res = self
            .store
            .execute(move |conn| {
                conn.execute(
                    "DELETE FROM ADMIN.\"CONVERSATIONS\" WHERE \"CONVERSATION_ID\" = :1",
                    &[&id_str],
                )
                .map_err(map_oracle_error)
            })
            .await
            .map_err(ConversationStorageError::StorageError)?;

        Ok(res
            .row_count()
            .map_err(|e| ConversationStorageError::StorageError(map_oracle_error(e)))?
            > 0)
    }
}

// ============================================================================
// PART 3: OracleConversationItemStorage
// ============================================================================

#[derive(Clone)]
pub(super) struct OracleConversationItemStorage {
    store: OracleStore,
}

impl OracleConversationItemStorage {
    pub fn new(store: OracleStore) -> Self {
        Self { store }
    }

    pub(crate) fn init_schema(conn: &Connection) -> Result<(), String> {
        // Items are embedded in ADMIN.CONVERSATIONS.ITEMS.
        let exists: i64 = conn
            .query_row_as(
                "SELECT COUNT(*) FROM all_tables WHERE owner = 'ADMIN' AND table_name = 'CONVERSATIONS'",
                &[],
            )
            .map_err(map_oracle_error)?;

        if exists == 0 {
            return Err("CONVERSATIONS table does not exist. Please create the table.".to_string());
        }

        Ok(())
    }
}

#[async_trait]
impl ConversationItemStorage for OracleConversationItemStorage {
    async fn create_item(
        &self,
        item: NewConversationItem,
    ) -> Result<ConversationItem, ConversationItemStorageError> {
        // Get conversation_id from task-local context.
        let conversation_id = super::core::CURRENT_CONVERSATION_ID
            .try_with(|id| id.clone())
            .ok()
            .flatten()
            .ok_or_else(|| {
                ConversationItemStorageError::StorageError(
                    "conversation_id context required".to_string(),
                )
            })?;

        let id = item
            .id
            .clone()
            .unwrap_or_else(|| make_item_id(&item.item_type));
        let created_at = Utc::now();

        let conversation_item = ConversationItem {
            id: id.clone(),
            response_id: item.response_id.clone(),
            item_type: item.item_type.clone(),
            role: item.role.clone(),
            content: item.content,
            status: item.status.clone(),
            created_at,
        };

        let cid = conversation_id.0.clone();
        let item_json = serde_json::to_value(&conversation_item)
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;

        self.store
            .execute(move |conn| {
                // First, get the current items.
                let current_items_json: Option<String> = conn
                    .query_row_as(
                        "SELECT \"ITEMS\" FROM ADMIN.\"CONVERSATIONS\" WHERE \"CONVERSATION_ID\" = :1",
                        &[&cid],
                    )
                    .map_err(map_oracle_error)
                    .map_err(|e| format!("Failed to get conversation: {e}"))?;

                let mut items_array: Vec<Value> = if let Some(json_str) = current_items_json {
                    serde_json::from_str(&json_str).map_err(|e| e.to_string())?
                } else {
                    Vec::new()
                };

                items_array.push(item_json);

                let updated_items_json =
                    serde_json::to_string(&items_array).map_err(|e| e.to_string())?;

                conn.execute(
                    "UPDATE ADMIN.\"CONVERSATIONS\" SET \"ITEMS\" = :1, \"UPDATED_AT\" = :2, \"GENERATIVE_AI_PROJECT_ID\" = :3, \"SHORT_TERM_MEMORY\" = :4, \"VERSION\" = :5 WHERE \"CONVERSATION_ID\" = :6",
                    &[&updated_items_json, &Utc::now(), &None::<String>, &None::<String>, &0, &cid],
                )
                .map_err(map_oracle_error)?;

                Ok(())
            })
            .await
            .map_err(ConversationItemStorageError::StorageError)?;

        Ok(conversation_item)
    }

    async fn link_item(
        &self,
        _conversation_id: &ConversationId,
        _item_id: &ConversationItemId,
        _added_at: DateTime<Utc>,
    ) -> Result<(), ConversationItemStorageError> {
        // Items are embedded, so linking is implicit.
        Ok(())
    }

    async fn list_items(
        &self,
        conversation_id: &ConversationId,
        params: ListParams,
    ) -> Result<Vec<ConversationItem>, ConversationItemStorageError> {
        let cid = conversation_id.0.clone();
        self.store
            .execute(move |conn| {
                let items_json: Option<String> = conn
                    .query_row_as(
                        "SELECT \"ITEMS\" FROM ADMIN.\"CONVERSATIONS\" WHERE \"CONVERSATION_ID\" = :1",
                        &[&cid],
                    )
                    .map_err(map_oracle_error)
                    .ok();

                let items_array: Vec<Value> = if let Some(json_str) = items_json {
                    serde_json::from_str(&json_str).map_err(|e| e.to_string())?
                } else {
                    Vec::new()
                };

                let mut conversation_items: Vec<ConversationItem> = Vec::new();
                for item_value in items_array {
                    let item: ConversationItem =
                        serde_json::from_value(item_value).map_err(|e| e.to_string())?;
                    conversation_items.push(item);
                }

                // Apply sorting and pagination
                let order_desc = matches!(params.order, SortOrder::Desc);
                if order_desc {
                    conversation_items.sort_by(|a, b| b.created_at.cmp(&a.created_at));
                } else {
                    conversation_items.sort_by(|a, b| a.created_at.cmp(&b.created_at));
                }

                let mut result_items = Vec::new();
                let mut skip = false;
                if let Some(ref after_id) = params.after {
                    for item in conversation_items {
                        if skip {
                            result_items.push(item);
                        } else if item.id.0 == *after_id {
                            skip = true;
                        }
                        if result_items.len() >= params.limit {
                            break;
                        }
                    }
                } else {
                    for item in conversation_items {
                        result_items.push(item);
                        if result_items.len() >= params.limit {
                            break;
                        }
                    }
                }

                Ok(result_items)
            })
            .await
            .map_err(ConversationItemStorageError::StorageError)
    }

    async fn get_item(
        &self,
        item_id: &ConversationItemId,
    ) -> Result<Option<ConversationItem>, ConversationItemStorageError> {
        let iid = item_id.0.clone();

        let conv_id = super::core::CURRENT_CONVERSATION_ID
            .try_with(|id| id.clone())
            .ok()
            .flatten()
            .ok_or_else(|| {
                ConversationItemStorageError::StorageError(
                    "conversation_id context required".to_string(),
                )
            })?;

        let cid = conv_id.0;

        self.store
            .execute(move |conn| {
                let items_json: Option<String> = conn
                    .query_row_as(
                        "SELECT \"ITEMS\" FROM ADMIN.\"CONVERSATIONS\" WHERE \"CONVERSATION_ID\" = :1",
                        &[&cid],
                    )
                    .map_err(map_oracle_error)
                    .ok();

                if let Some(json_str) = items_json {
                    let items_array: Vec<Value> =
                        serde_json::from_str(&json_str).map_err(|e| e.to_string())?;
                    for item_value in items_array {
                        let item: ConversationItem = serde_json::from_value(item_value)
                            .map_err(|e| e.to_string())?;
                        if item.id.0 == iid {
                            return Ok(Some(item));
                        }
                    }
                }

                Ok(None)
            })
            .await
            .map_err(ConversationItemStorageError::StorageError)
    }

    async fn is_item_linked(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> Result<bool, ConversationItemStorageError> {
        let cid = conversation_id.0.clone();
        let iid = item_id.0.clone();

        self.store
            .execute(move |conn| {
                let items_json: Option<String> = conn
                    .query_row_as(
                        "SELECT \"ITEMS\" FROM ADMIN.\"CONVERSATIONS\" WHERE \"CONVERSATION_ID\" = :1",
                        &[&cid],
                    )
                    .map_err(map_oracle_error)
                    .ok();

                if let Some(json_str) = items_json {
                    let items_array: Vec<Value> =
                        serde_json::from_str(&json_str).map_err(|e| e.to_string())?;
                    for item_value in items_array {
                        let item: ConversationItem =
                            serde_json::from_value(item_value).map_err(|e| e.to_string())?;
                        if item.id.0 == iid {
                            return Ok(true);
                        }
                    }
                }

                Ok(false)
            })
            .await
            .map_err(ConversationItemStorageError::StorageError)
    }

    async fn delete_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> Result<(), ConversationItemStorageError> {
        let cid = conversation_id.0.clone();
        let iid = item_id.0.clone();

        self.store
            .execute(move |conn| {
                let current_items_json: Option<String> = conn
                    .query_row_as(
                        "SELECT \"ITEMS\" FROM ADMIN.\"CONVERSATIONS\" WHERE \"CONVERSATION_ID\" = :1",
                        &[&cid],
                    )
                    .map_err(map_oracle_error)
                    .map_err(|e| format!("Failed to get conversation: {e}"))?;

                let mut items_array: Vec<Value> = if let Some(json_str) = current_items_json {
                    serde_json::from_str(&json_str).map_err(|e| e.to_string())?
                } else {
                    Vec::new()
                };

                items_array.retain(|item_value| {
                    if let Ok(item) = serde_json::from_value::<ConversationItem>(item_value.clone()) {
                        item.id.0 != iid
                    } else {
                        true
                    }
                });

                let updated_items_json =
                    serde_json::to_string(&items_array).map_err(|e| e.to_string())?;

                conn.execute(
                    "UPDATE ADMIN.\"CONVERSATIONS\" SET \"ITEMS\" = :1, \"UPDATED_AT\" = :2, \"GENERATIVE_AI_PROJECT_ID\" = :3, \"SHORT_TERM_MEMORY\" = :4, \"VERSION\" = :5 WHERE \"CONVERSATION_ID\" = :6",
                    &[&updated_items_json, &Utc::now(), &None::<String>, &None::<String>, &0, &cid],
                )
                .map_err(map_oracle_error)?;
                Ok(())
            })
            .await
            .map_err(ConversationItemStorageError::StorageError)
    }
}

// ============================================================================
// PART 4: OracleResponseStorage
// ============================================================================

const SELECT_BASE: &str = "SELECT \"RESPONSE_ID\", \"CONVERSATION_STORE_ID\", \"CONVERSATION_ID\", \"PREVIOUS_RESPONSE_ID\", \
    \"INPUT_ITEMS\", \"RESPONSE_OBJECT\", \"MODEL\", \"CREATED_AT\", \"EXPIRES_AT\", \"SUBJECT_ID\", \"INPUT_EMBEDDING\", \"OUTPUT_EMBEDDING\", \"GENERATIVE_AI_PROJECT_ID\" FROM ADMIN.\"RESPONSES\"";

#[derive(Clone)]
pub(super) struct OracleResponseStorage {
    store: OracleStore,
}

impl OracleResponseStorage {
    pub fn new(store: OracleStore) -> Self {
        Self { store }
    }

    pub(crate) fn init_schema(conn: &Connection) -> Result<(), String> {
        // Downstream expects tables to exist in ADMIN schema.
        let exists: i64 = conn
            .query_row_as(
                "SELECT COUNT(*) FROM all_tables WHERE owner = 'ADMIN' AND table_name = 'RESPONSES'",
                &[],
            )
            .map_err(map_oracle_error)?;

        if exists == 0 {
            return Err("RESPONSES table does not exist. Please create the table.".to_string());
        }

        Ok(())
    }

    fn build_response_from_row(row: &Row) -> Result<StoredResponse, String> {
        let id: String = row.get(0).map_err(map_oracle_error)?;
        let conversation_store_id: Option<String> = row.get(1).map_err(map_oracle_error)?;
        let conversation_id: Option<String> = row.get(2).map_err(map_oracle_error)?;
        let previous: Option<String> = row.get(3).map_err(map_oracle_error)?;
        let input_json: Option<String> = row.get(4).map_err(map_oracle_error)?;
        let output_json: Option<String> = row.get(5).map_err(map_oracle_error)?;
        let model: Option<String> = row.get(6).map_err(map_oracle_error)?;
        let created_at: DateTime<Utc> = row.get(7).map_err(map_oracle_error)?;
        let _expires_at: Option<DateTime<Utc>> = row.get(8).map_err(map_oracle_error)?;
        let _subject_id: Option<String> = row.get(9).map_err(map_oracle_error)?;
        let _generative_ai_project_id: Option<String> = row.get(12).map_err(map_oracle_error)?;

        let previous_response_id = previous.map(ResponseId);
        let input = parse_json_value(input_json)?;
        let output = parse_json_value(output_json.clone())?;

        // Set defaults for fields not in the ADMIN.RESPONSES schema.
        let instructions = None;
        let tool_calls = parse_tool_calls(None)?;
        let metadata = parse_metadata(None)?;
        let raw_response = parse_raw_response(output_json.clone())?;
        let safety_identifier = None;

        let mut response = StoredResponse {
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
        };

        if let Some(store_id) = conversation_store_id {
            response.metadata.insert(
                "oci:conversation_store_id".to_string(),
                serde_json::json!(store_id),
            );
        }

        Ok(response)
    }
}

#[async_trait]
impl ResponseStorage for OracleResponseStorage {
    async fn store_response(
        &self,
        response: StoredResponse,
    ) -> Result<ResponseId, ResponseStorageError> {
        let response_id = response.id.clone();
        let response_id_str = response_id.0.clone();
        let previous_id = response.previous_response_id.map(|r| r.0);
        let json_input = serde_json::to_string(&response.input)?;
        let json_output = serde_json::to_string(&response.output)?;
        let model = response.model.clone();
        let created_at = response.created_at;
        let conversation_id = response.conversation_id.clone();
        let conversation_store_id = response
            .metadata
            .get("oci:conversation_store_id")
            .and_then(|v| v.as_str())
            .map(String::from);
        let expires_at = created_at + chrono::Duration::hours(24);

        self.store
            .execute(move |conn| {
                // Work around rust-oracle option binding.
                let conversation_store_id_ref = conversation_store_id.as_deref();
                conn.execute(
                    "INSERT INTO ADMIN.\"RESPONSES\" (\"RESPONSE_ID\", \"CONVERSATION_STORE_ID\", \"CONVERSATION_ID\", \"PREVIOUS_RESPONSE_ID\", \
                        \"INPUT_ITEMS\", \"RESPONSE_OBJECT\", \"MODEL\", \"CREATED_AT\", \"EXPIRES_AT\", \"SUBJECT_ID\", \"GENERATIVE_AI_PROJECT_ID\") \
                     VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11)",
                    &[
                        &response_id_str,
                        &conversation_store_id_ref,
                        &conversation_id,
                        &previous_id,
                        &json_input,
                        &json_output,
                        &model,
                        &created_at,
                        &expires_at,
                        &None::<String>,
                        &None::<String>,
                    ],
                )
                .map(|_| ())
                .map_err(map_oracle_error)
            })
            .await
            .map_err(ResponseStorageError::StorageError)?;

        Ok(response_id)
    }

    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> Result<Option<StoredResponse>, ResponseStorageError> {
        let id = response_id.0.clone();
        self.store
            .execute(move |conn| {
                let mut stmt = conn
                    .statement(&format!("{SELECT_BASE} WHERE \"RESPONSE_ID\" = :1"))
                    .build()
                    .map_err(map_oracle_error)?;
                let mut rows = stmt.query(&[&id]).map_err(map_oracle_error)?;
                match rows.next() {
                    Some(row) => {
                        let row = row.map_err(map_oracle_error)?;
                        Self::build_response_from_row(&row).map(Some)
                    }
                    None => Ok(None),
                }
            })
            .await
            .map_err(ResponseStorageError::StorageError)
    }

    async fn delete_response(&self, response_id: &ResponseId) -> Result<(), ResponseStorageError> {
        let id = response_id.0.clone();
        self.store
            .execute(move |conn| {
                conn.execute(
                    "DELETE FROM ADMIN.\"RESPONSES\" WHERE \"RESPONSE_ID\" = :1",
                    &[&id],
                )
                .map(|_| ())
                .map_err(map_oracle_error)
            })
            .await
            .map_err(ResponseStorageError::StorageError)
    }

    async fn get_response_chain(
        &self,
        response_id: &ResponseId,
        max_depth: Option<usize>,
    ) -> Result<ResponseChain, ResponseStorageError> {
        let mut chain = ResponseChain::new();
        let mut current_id = Some(response_id.clone());
        let mut visited = 0usize;

        while let Some(ref lookup_id) = current_id {
            if let Some(limit) = max_depth {
                if visited >= limit {
                    break;
                }
            }

            let fetched = self.get_response(lookup_id).await?;
            match fetched {
                Some(response) => {
                    current_id.clone_from(&response.previous_response_id);
                    chain.responses.push(response);
                    visited += 1;
                }
                None => break,
            }
        }

        chain.responses.reverse();
        Ok(chain)
    }

    async fn list_identifier_responses(
        &self,
        _identifier: &str,
        _limit: Option<usize>,
    ) -> Result<Vec<StoredResponse>, ResponseStorageError> {
        Err(ResponseStorageError::StorageError(
            "list_identifier_responses not supported: RESPONSES table does not have safety_identifier column"
                .to_string(),
        ))
    }

    async fn delete_identifier_responses(
        &self,
        _identifier: &str,
    ) -> Result<usize, ResponseStorageError> {
        Err(ResponseStorageError::StorageError(
            "delete_identifier_responses not supported: RESPONSES table does not have safety_identifier column"
                .to_string(),
        ))
    }
}
