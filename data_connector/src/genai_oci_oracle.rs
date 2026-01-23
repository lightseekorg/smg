//! Genai OCI Oracle storage implementation using GenaiOciOracleStore helper.
//!
//! Structure:
//! 1. GenaiOciOracleStore helper and common utilities
//! 2. GenaiOciOracleConversationStorage
//! 3. GenaiOciOracleConversationItemStorage
//! 4. GenaiOciOracleResponseStorage

use std::{path::Path, sync::Arc, time::Duration};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use deadpool::managed::{Manager, Metrics, Pool, RecycleError, RecycleResult};
use oracle::{Connection, Row};
use serde_json::Value;

use super::{
    common::{parse_json_value, parse_metadata, parse_raw_response, parse_tool_calls},
    core::{
        make_item_id, Conversation, ConversationId, ConversationItem, ConversationItemId,
        ConversationItemStorage, ConversationItemStorageError, ConversationMetadata,
        ConversationStorage, ConversationStorageError, ListParams, NewConversation,
        NewConversationItem, ResponseChain, ResponseId, ResponseStorage, ResponseStorageError,
        SortOrder, StoredResponse,
    },
};
use crate::config::OracleConfig;
// ============================================================================
// PART 1: GenaiOciOracleStore Helper + Common Utilities
// ============================================================================

/// Shared Oracle connection pool infrastructure
///
/// This helper eliminates ~540 LOC of duplication across storage implementations.
/// It handles connection pooling, error mapping, and client configuration.
pub(crate) struct GenaiOciOracleStore {
    pool: Pool<GenaiOciOracleConnectionManager>,
}

impl GenaiOciOracleStore {
    /// Create pool with custom schema initialization
    ///
    /// The `init_schema` function receives a connection and should:
    /// - Check if tables/indexes exist
    /// - Create them if needed
    /// - Return Ok(()) on success or Err(message) on failure
    pub fn new(
        config: &OracleConfig,
        init_schema: impl FnOnce(&Connection) -> Result<(), String>,
    ) -> Result<Self, String> {
        // Configure Oracle client (wallet, etc.)
        configure_genai_oci_oracle_client(config)?;

        // Initialize schema using the provided function
        let conn = Connection::connect(
            &config.username,
            &config.password,
            &config.connect_descriptor,
        )
        .map_err(map_genai_oci_oracle_error)?;

        init_schema(&conn)?;
        drop(conn);

        // Create connection pool
        let config_arc = Arc::new(config.clone());
        let manager = GenaiOciOracleConnectionManager {
            params: Arc::new(GenaiOciOracleConnectParams::from_config(&config_arc)),
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

impl Clone for GenaiOciOracleStore {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
        }
    }
}

// Error mapping helper
pub(crate) fn map_genai_oci_oracle_error(err: oracle::Error) -> String {
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

// Client configuration helper
fn configure_genai_oci_oracle_client(config: &OracleConfig) -> Result<(), String> {
    if let Some(wallet_path) = &config.wallet_path {
        let path = Path::new(wallet_path);

        if !path.is_dir() {
            return Err(format!(
                "Oracle wallet path '{}' is not a directory",
                wallet_path
            ));
        }

        if !path.join("tnsnames.ora").exists() && !path.join("sqlnet.ora").exists() {
            return Err(format!(
                "Oracle wallet path '{}' is missing tnsnames.ora or sqlnet.ora",
                wallet_path
            ));
        }

        // Update sqlnet.ora to replace placeholder directory with actual path
        let sqlnet_path = path.join("sqlnet.ora");
        if sqlnet_path.exists() {
            let content = std::fs::read_to_string(&sqlnet_path)
                .map_err(|e| format!("Failed to read sqlnet.ora: {}", e))?;

            // Replace placeholder "?" with actual wallet directory
            let updated_content =
                content.replace("DIRECTORY=\"?\"", &format!("DIRECTORY=\"{}\"", wallet_path));

            if updated_content != content {
                std::fs::write(&sqlnet_path, updated_content)
                    .map_err(|e| format!("Failed to update sqlnet.ora: {}", e))?;
            }
        }

        std::env::set_var("TNS_ADMIN", wallet_path);
    }
    Ok(())
}

// Connection parameters
#[derive(Clone)]
pub(crate) struct GenaiOciOracleConnectParams {
    pub username: String,
    pub password: String,
    pub connect_descriptor: String,
}

impl GenaiOciOracleConnectParams {
    pub fn from_config(config: &OracleConfig) -> Self {
        Self {
            username: config.username.clone(),
            password: config.password.clone(),
            connect_descriptor: config.connect_descriptor.clone(),
        }
    }
}

impl std::fmt::Debug for GenaiOciOracleConnectParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenaiOciOracleConnectParams")
            .field("username", &self.username)
            .field("connect_descriptor", &self.connect_descriptor)
            .finish()
    }
}

// Connection manager (same for all stores)
#[derive(Clone)]
struct GenaiOciOracleConnectionManager {
    params: Arc<GenaiOciOracleConnectParams>,
}

impl std::fmt::Debug for GenaiOciOracleConnectionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenaiOciOracleConnectionManager")
            .field("username", &self.params.username)
            .field("connect_descriptor", &self.params.connect_descriptor)
            .finish()
    }
}

#[async_trait]
impl Manager for GenaiOciOracleConnectionManager {
    type Type = Connection;
    type Error = oracle::Error;

    fn create(
        &self,
    ) -> impl std::future::Future<Output = Result<Connection, oracle::Error>> + Send {
        let params = self.params.clone();
        async move {
            let mut conn = Connection::connect(
                &params.username,
                &params.password,
                &params.connect_descriptor,
            )?;
            conn.set_autocommit(true);
            Ok(conn)
        }
    }

    #[allow(clippy::manual_async_fn)]
    fn recycle(
        &self,
        conn: &mut Connection,
        _: &Metrics,
    ) -> impl std::future::Future<Output = RecycleResult<Self::Error>> + Send {
        async move { conn.ping().map_err(RecycleError::Backend) }
    }
}

// ============================================================================
// PART 2: GenaiOciOracleConversationStorage
// ============================================================================

#[derive(Clone)]
pub(super) struct GenaiOciOracleConversationStorage {
    store: GenaiOciOracleStore,
}

impl GenaiOciOracleConversationStorage {
    pub fn new(config: OracleConfig) -> Result<Self, ConversationStorageError> {
        let store = GenaiOciOracleStore::new(&config, |conn| {
            // Check if table exists
            let exists: i64 = conn
                .query_row_as(
                    "SELECT COUNT(*) FROM user_tables WHERE table_name = 'CONVERSATIONS'",
                    &[],
                )
                .map_err(map_genai_oci_oracle_error)?;

            // Return error if table doesn't exist
            if exists == 0 {
                return Err(
                    "CONVERSATIONS table does not exist. Please create the table.".to_string(),
                );
            }

            Ok(())
        })
        .map_err(ConversationStorageError::StorageError)?;

        Ok(Self { store })
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
impl ConversationStorage for GenaiOciOracleConversationStorage {
    async fn create_conversation(
        &self,
        input: NewConversation,
    ) -> Result<Conversation, ConversationStorageError> {
        let conversation = Conversation::new(input.clone());
        let id_str = conversation.id.0.clone();
        let conversation_store_id = input.conversation_store_id;
        let created_at = conversation.created_at;
        let metadata_json = conversation
            .metadata
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        let expires_at = created_at + chrono::Duration::hours(24); // Default 24 hour expiration

        self.store
            .execute(move |conn| {
                conn.execute(
                    "INSERT INTO \"CONVERSATIONS\" (\"CONVERSATION_ID\", \"CONVERSATION_STORE_ID\", \"CREATED_AT\", \"METADATA\", \"ITEMS\", \"EXPIRES_AT\") VALUES (:1, :2, :3, :4, :5, :6)",
                    &[&id_str, &conversation_store_id, &created_at, &metadata_json, &"[]", &expires_at],
                )
                .map(|_| ())
                .map_err(map_genai_oci_oracle_error)
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
                    .statement("SELECT \"CONVERSATION_ID\", \"CREATED_AT\", \"METADATA\" FROM \"CONVERSATIONS\" WHERE \"CONVERSATION_ID\" = :1")
                    .build()
                    .map_err(map_genai_oci_oracle_error)?;
                let mut rows = stmt.query(&[&lookup]).map_err(map_genai_oci_oracle_error)?;

                if let Some(row_res) = rows.next() {
                    let row = row_res.map_err(map_genai_oci_oracle_error)?;
                    let id: String = row.get(0).map_err(map_genai_oci_oracle_error)?;
                    let created_at: DateTime<Utc> = row.get(1).map_err(map_genai_oci_oracle_error)?;
                    let metadata_raw: Option<String> = row.get(2).map_err(map_genai_oci_oracle_error)?;
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
        let updated_at = Utc::now();
        let conversation_id = id.clone();

        self.store
            .execute(move |conn| {
                let affected = conn.execute(
                    "UPDATE \"CONVERSATIONS\" SET \"METADATA\" = :1, \"UPDATED_AT\" = :2 WHERE \"CONVERSATION_ID\" = :3",
                    &[&metadata_json, &updated_at, &id_str],
                )
                .map_err(map_genai_oci_oracle_error)?;

                if affected.row_count().map_err(map_genai_oci_oracle_error)? == 0 {
                    return Ok(None);
                }

                // Get the updated conversation
                let mut stmt = conn
                    .statement("SELECT \"CREATED_AT\" FROM \"CONVERSATIONS\" WHERE \"CONVERSATION_ID\" = :1")
                    .build()
                    .map_err(map_genai_oci_oracle_error)?;
                let mut rows = stmt.query(&[&id_str]).map_err(map_genai_oci_oracle_error)?;

                if let Some(row_res) = rows.next() {
                    let row = row_res.map_err(map_genai_oci_oracle_error)?;
                    let created_at: DateTime<Utc> = row.get(0).map_err(map_genai_oci_oracle_error)?;
                    Ok(Some(Conversation::with_parts(
                        conversation_id,
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

    async fn delete_conversation(
        &self,
        id: &ConversationId,
    ) -> Result<bool, ConversationStorageError> {
        let id_str = id.0.clone();
        let res = self
            .store
            .execute(move |conn| {
                conn.execute(
                    "DELETE FROM \"CONVERSATIONS\" WHERE \"CONVERSATION_ID\" = :1",
                    &[&id_str],
                )
                .map_err(map_genai_oci_oracle_error)
            })
            .await
            .map_err(ConversationStorageError::StorageError)?;

        Ok(res
            .row_count()
            .map_err(|e| ConversationStorageError::StorageError(map_genai_oci_oracle_error(e)))?
            > 0)
    }
}

// ============================================================================
// PART 3: GenaiOciOracleConversationItemStorage
// ============================================================================

#[derive(Clone)]
pub(super) struct GenaiOciOracleConversationItemStorage {
    store: GenaiOciOracleStore,
}

impl GenaiOciOracleConversationItemStorage {
    pub fn new(config: OracleConfig) -> Result<Self, ConversationItemStorageError> {
        // No schema initialization needed - items are stored in CONVERSATIONS table
        let store = GenaiOciOracleStore::new(&config, |_| Ok(()))
            .map_err(ConversationItemStorageError::StorageError)?;

        Ok(Self { store })
    }
}

#[async_trait]
impl ConversationItemStorage for GenaiOciOracleConversationItemStorage {
    async fn create_item(
        &self,
        item: NewConversationItem,
    ) -> Result<ConversationItem, ConversationItemStorageError> {
        let conversation_id = item.conversation_id.as_ref().ok_or_else(|| {
            ConversationItemStorageError::StorageError("conversation_id required".to_string())
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
                // First, get the current items and check if conversation exists
                let current_items_json: Option<String> = conn
                    .query_row_as(
                        "SELECT \"ITEMS\" FROM \"CONVERSATIONS\" WHERE \"CONVERSATION_ID\" = :1",
                        &[&cid],
                    )
                    .map_err(map_genai_oci_oracle_error)
                    .map_err(|e| format!("Failed to get conversation: {}", e))?;

                let mut items_array: Vec<Value> = if let Some(json_str) = current_items_json {
                    serde_json::from_str(&json_str).map_err(|e| e.to_string())?
                } else {
                    Vec::new()
                };

                // Add new item
                items_array.push(item_json);

                // Update the items column
                let updated_items_json = serde_json::to_string(&items_array)
                    .map_err(|e| e.to_string())?;

                conn.execute(
                    "UPDATE \"CONVERSATIONS\" SET \"ITEMS\" = :1, \"UPDATED_AT\" = :2 WHERE \"CONVERSATION_ID\" = :3",
                    &[&updated_items_json, &Utc::now(), &cid],
                )
                .map_err(map_genai_oci_oracle_error)?;

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
        // Items are now embedded, so linking is implicit
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
                        "SELECT \"ITEMS\" FROM \"CONVERSATIONS\" WHERE \"CONVERSATION_ID\" = :1",
                        &[&cid],
                    )
                    .map_err(map_genai_oci_oracle_error)
                    .ok(); // Convert to Option - None if no row found

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

                // Apply cursor-based pagination
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

        // Since items are embedded in conversations, we need to find which conversation contains this item
        // For now, we'll search across all conversations (this is inefficient but works for the migration)
        self.store
            .execute(move |conn| {
                let mut stmt = conn
                    .statement("SELECT \"ITEMS\" FROM \"CONVERSATIONS\"")
                    .build()
                    .map_err(map_genai_oci_oracle_error)?;
                let rows = stmt.query(&[]).map_err(map_genai_oci_oracle_error)?;

                for row_res in rows {
                    let row = row_res.map_err(map_genai_oci_oracle_error)?;
                    let items_json: Option<String> =
                        row.get(0).map_err(map_genai_oci_oracle_error)?;

                    if let Some(json_str) = items_json {
                        let items_array: Vec<Value> =
                            serde_json::from_str(&json_str).map_err(|e| e.to_string())?;

                        for item_value in items_array {
                            let item: ConversationItem = serde_json::from_value(item_value.clone())
                                .map_err(|e| e.to_string())?;
                            if item.id.0 == iid {
                                return Ok(Some(item));
                            }
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
                        "SELECT \"ITEMS\" FROM \"CONVERSATIONS\" WHERE \"CONVERSATION_ID\" = :1",
                        &[&cid],
                    )
                    .map_err(map_genai_oci_oracle_error)
                    .ok(); // Convert to Option - None if no row found

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
                // First, get the current items and check if conversation exists
                let current_items_json: Option<String> = conn
                    .query_row_as(
                        "SELECT \"ITEMS\" FROM \"CONVERSATIONS\" WHERE \"CONVERSATION_ID\" = :1",
                        &[&cid],
                    )
                    .map_err(map_genai_oci_oracle_error)
                    .map_err(|e| format!("Failed to get conversation: {}", e))?;

                let mut items_array: Vec<Value> = if let Some(json_str) = current_items_json {
                    serde_json::from_str(&json_str).map_err(|e| e.to_string())?
                } else {
                    Vec::new()
                };

                // Remove the item
                items_array.retain(|item_value| {
                    if let Ok(item) = serde_json::from_value::<ConversationItem>(item_value.clone()) {
                        item.id.0 != iid
                    } else {
                        true // Keep items that can't be parsed
                    }
                });

                // Update the items column
                let updated_items_json = serde_json::to_string(&items_array)
                    .map_err(|e| e.to_string())?;

                conn.execute(
                    "UPDATE \"CONVERSATIONS\" SET \"ITEMS\" = :1, \"UPDATED_AT\" = :2 WHERE \"CONVERSATION_ID\" = :3",
                    &[&updated_items_json, &Utc::now(), &cid],
                )
                .map_err(map_genai_oci_oracle_error)?;

                Ok(())
            })
            .await
            .map_err(ConversationItemStorageError::StorageError)
    }
}

// ============================================================================
// PART 4: GenaiOciOracleResponseStorage
// ============================================================================

const SELECT_BASE: &str = "SELECT \"RESPONSE_ID\", \"CONVERSATION_STORE_ID\", \"CONVERSATION_ID\", \"PREVIOUS_RESPONSE_ID\", \
    \"INPUT_ITEMS\", \"RESPONSE_OBJECT\", \"MODEL\", \"CREATED_AT\" FROM \"RESPONSES\"";

#[derive(Clone)]
pub(super) struct GenaiOciOracleResponseStorage {
    store: GenaiOciOracleStore,
}

impl GenaiOciOracleResponseStorage {
    pub fn new(config: OracleConfig) -> Result<Self, ResponseStorageError> {
        let store = GenaiOciOracleStore::new(&config, |conn| {
            // Check if table exists
            let exists: i64 = conn
                .query_row_as(
                    "SELECT COUNT(*) FROM user_tables WHERE table_name = 'RESPONSES'",
                    &[],
                )
                .map_err(map_genai_oci_oracle_error)?;

            // Return error if table doesn't exist
            if exists == 0 {
                return Err("RESPONSES table does not exist. Please create the table.".to_string());
            }

            Ok(())
        })
        .map_err(ResponseStorageError::StorageError)?;

        Ok(Self { store })
    }

    fn build_response_from_row(row: &Row) -> Result<StoredResponse, String> {
        let id: String = row.get(0).map_err(map_genai_oci_oracle_error)?;
        let conversation_store_id: Option<String> =
            row.get(1).map_err(map_genai_oci_oracle_error)?;
        let conversation_id: Option<String> = row.get(2).map_err(map_genai_oci_oracle_error)?;
        let previous: Option<String> = row.get(3).map_err(map_genai_oci_oracle_error)?;
        let input_json: Option<String> = row.get(4).map_err(map_genai_oci_oracle_error)?;
        let output_json: Option<String> = row.get(5).map_err(map_genai_oci_oracle_error)?;
        let model: Option<String> = row.get(6).map_err(map_genai_oci_oracle_error)?;
        let created_at: DateTime<Utc> = row.get(7).map_err(map_genai_oci_oracle_error)?;

        let previous_response_id = previous.map(ResponseId);
        let input = parse_json_value(input_json)?;
        let output = parse_json_value(output_json)?;

        // Set defaults for fields not in the new schema
        let instructions = None;
        let tool_calls = parse_tool_calls(None)?;
        let metadata = parse_metadata(None)?;
        let raw_response = parse_raw_response(None)?;
        let safety_identifier = None;

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
            conversation_store_id,
            raw_response,
        })
    }
}

#[async_trait]
impl ResponseStorage for GenaiOciOracleResponseStorage {
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
        let conversation_store_id = response.conversation_store_id;
        let expires_at = created_at + chrono::Duration::hours(24); // Default 24 hour expiration

        self.store
            .execute(move |conn| {
                conn.execute(
                    "INSERT INTO \"RESPONSES\" (\"RESPONSE_ID\", \"CONVERSATION_STORE_ID\", \"CONVERSATION_ID\", \"PREVIOUS_RESPONSE_ID\", \
                        \"INPUT_ITEMS\", \"RESPONSE_OBJECT\", \"MODEL\", \"CREATED_AT\", \"EXPIRES_AT\") \
                     VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9)",
                    &[
                        &response_id_str,
                        &conversation_store_id,
                        &conversation_id,
                        &previous_id,
                        &json_input,
                        &json_output,
                        &model,
                        &created_at,
                        &expires_at,
                    ],
                )
                .map(|_| ())
                .map_err(map_genai_oci_oracle_error)
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
                    .statement(&format!("{} WHERE \"RESPONSE_ID\" = :1", SELECT_BASE))
                    .build()
                    .map_err(map_genai_oci_oracle_error)?;
                let mut rows = stmt.query(&[&id]).map_err(map_genai_oci_oracle_error)?;
                match rows.next() {
                    Some(row) => {
                        let row = row.map_err(map_genai_oci_oracle_error)?;
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
                    "DELETE FROM \"RESPONSES\" WHERE \"RESPONSE_ID\" = :1",
                    &[&id],
                )
                .map(|_| ())
                .map_err(map_genai_oci_oracle_error)
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
                    current_id = response.previous_response_id.clone();
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
        identifier: &str,
        limit: Option<usize>,
    ) -> Result<Vec<StoredResponse>, ResponseStorageError> {
        let identifier = identifier.to_string();

        self.store
            .execute(move |conn| {
                let sql = if let Some(limit) = limit {
                    format!(
                        "SELECT * FROM ({} WHERE safety_identifier = :1 ORDER BY created_at DESC) WHERE ROWNUM <= {}",
                        SELECT_BASE, limit
                    )
                } else {
                    format!("{} WHERE safety_identifier = :1 ORDER BY created_at DESC", SELECT_BASE)
                };

                let mut stmt = conn.statement(&sql).build().map_err(map_genai_oci_oracle_error)?;
                let mut rows = stmt.query(&[&identifier]).map_err(map_genai_oci_oracle_error)?;
                let mut results = Vec::new();

                for row in &mut rows {
                    let row = row.map_err(map_genai_oci_oracle_error)?;
                    results.push(Self::build_response_from_row(&row)?);
                }

                Ok(results)
            })
            .await
            .map_err(ResponseStorageError::StorageError)
    }

    async fn delete_identifier_responses(
        &self,
        identifier: &str,
    ) -> Result<usize, ResponseStorageError> {
        let identifier = identifier.to_string();
        let affected = self
            .store
            .execute(move |conn| {
                conn.execute(
                    "DELETE FROM responses WHERE safety_identifier = :1",
                    &[&identifier],
                )
                .map_err(map_genai_oci_oracle_error)
            })
            .await
            .map_err(ResponseStorageError::StorageError)?;

        let deleted = affected
            .row_count()
            .map_err(|e| ResponseStorageError::StorageError(map_genai_oci_oracle_error(e)))?
            as usize;
        Ok(deleted)
    }
}

// Helper functions for response parsing
