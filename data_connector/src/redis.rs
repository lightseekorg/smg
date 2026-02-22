//! Redis storage implementation using RedisStore helper
//!
//! Structure:
//! 1. RedisStore helper and common utilities
//! 2. RedisConversationStorage
//! 3. RedisConversationItemStorage
//! 4. RedisResponseStorage

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use deadpool_redis::{Config, Pool, Runtime};
use redis::AsyncCommands;
use serde_json::Value;

use crate::{
    common::{parse_json_value, parse_metadata, parse_raw_response, parse_tool_calls},
    config::RedisConfig,
    core::{
        make_item_id, Conversation, ConversationId, ConversationItem, ConversationItemId,
        ConversationItemResult, ConversationItemStorage, ConversationItemStorageError,
        ConversationMetadata, ConversationResult, ConversationStorage, ConversationStorageError,
        ListParams, NewConversation, NewConversationItem, ResponseChain, ResponseId,
        ResponseResult, ResponseStorage, ResponseStorageError, SortOrder, StoredResponse,
    },
};

pub(crate) struct RedisStore {
    pool: Pool,
    retention_days: Option<u64>,
}

impl RedisStore {
    pub fn new(config: RedisConfig) -> Result<Self, String> {
        let mut cfg = Config::from_url(config.url);
        cfg.pool = Some(deadpool_redis::PoolConfig::new(config.pool_max));
        let pool = cfg
            .create_pool(Some(Runtime::Tokio1))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            pool,
            retention_days: config.retention_days,
        })
    }
}

impl Clone for RedisStore {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            retention_days: self.retention_days,
        }
    }
}

pub(super) struct RedisConversationStorage {
    store: RedisStore,
}

impl RedisConversationStorage {
    pub fn new(store: RedisStore) -> Self {
        Self { store }
    }

    fn conversation_key(id: &str) -> String {
        format!("conversation:{id}")
    }

    fn parse_metadata(
        metadata: Option<String>,
    ) -> Result<Option<ConversationMetadata>, ConversationStorageError> {
        crate::common::parse_conversation_metadata(metadata)
            .map_err(ConversationStorageError::StorageError)
    }
}

#[async_trait]
impl ConversationStorage for RedisConversationStorage {
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

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        let key = Self::conversation_key(id_str);

        let mut pipe = redis::pipe();
        pipe.hset(&key, "id", id_str);
        pipe.hset(&key, "created_at", created_at.to_rfc3339());
        if let Some(meta) = metadata_json {
            pipe.hset(&key, "metadata", meta);
        }

        // Expire after configured retention days (optional)
        if let Some(days) = self.store.retention_days {
            pipe.expire(&key, (days * 24 * 60 * 60) as i64);
        }

        pipe.query_async::<()>(&mut conn)
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;

        Ok(conversation)
    }

    async fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> Result<Option<Conversation>, ConversationStorageError> {
        let id_str = id.0.as_str();
        let key = Self::conversation_key(id_str);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;

        let exists: bool = conn
            .exists(&key)
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        if !exists {
            return Ok(None);
        }

        let (created_at_str, metadata_json): (String, Option<String>) = redis::pipe()
            .hget(&key, "created_at")
            .hget(&key, "metadata")
            .query_async(&mut conn)
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;

        let created_at = DateTime::parse_from_rfc3339(&created_at_str)
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?
            .with_timezone(&Utc);

        let metadata = Self::parse_metadata(metadata_json)?;

        Ok(Some(Conversation::with_parts(
            id.clone(),
            created_at,
            metadata,
        )))
    }

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> Result<Option<Conversation>, ConversationStorageError> {
        let id_str = id.0.as_str();
        let key = Self::conversation_key(id_str);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;

        let exists: bool = conn
            .exists(&key)
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        if !exists {
            return Ok(None);
        }

        let metadata_json = metadata.as_ref().map(serde_json::to_string).transpose()?;

        if let Some(meta) = metadata_json {
            conn.hset::<_, _, _, ()>(&key, "metadata", meta)
                .await
                .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        } else {
            conn.hdel::<_, _, ()>(&key, "metadata")
                .await
                .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        }

        // We need to fetch created_at to return the full object
        let created_at_str: String = conn
            .hget(&key, "created_at")
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        let created_at = DateTime::parse_from_rfc3339(&created_at_str)
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?
            .with_timezone(&Utc);

        Ok(Some(Conversation::with_parts(
            id.clone(),
            created_at,
            metadata,
        )))
    }

    async fn delete_conversation(&self, id: &ConversationId) -> ConversationResult<bool> {
        let id_str = id.0.as_str();
        let key = Self::conversation_key(id_str);
        // Also delete the items list for this conversation
        let items_key = format!("{key}:items");

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;

        let count: usize = redis::pipe()
            .del(&key)
            .del(&items_key)
            .query_async(&mut conn)
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;

        Ok(count > 0)
    }
}

pub(super) struct RedisConversationItemStorage {
    store: RedisStore,
}

impl RedisConversationItemStorage {
    pub fn new(store: RedisStore) -> Self {
        Self { store }
    }

    fn item_key(id: &str) -> String {
        format!("item:{id}")
    }

    fn conv_items_key(conv_id: &str) -> String {
        format!("conversation:{conv_id}:items")
    }

    /// Parse a Redis hash map into a `ConversationItem`, returning errors for
    /// corrupted data instead of silently substituting defaults.
    fn build_item_from_map(
        map: &std::collections::HashMap<String, String>,
        fallback_id: &str,
    ) -> Result<ConversationItem, ConversationItemStorageError> {
        let id = ConversationItemId(
            map.get("id")
                .cloned()
                .unwrap_or_else(|| fallback_id.to_string()),
        );
        let response_id = map.get("response_id").cloned();
        let item_type = map
            .get("item_type")
            .filter(|s| !s.is_empty())
            .cloned()
            .ok_or_else(|| {
                ConversationItemStorageError::StorageError(format!(
                    "item {fallback_id} missing item_type"
                ))
            })?;
        let role = map.get("role").cloned();
        let status = map.get("status").cloned();

        let content = match map.get("content") {
            Some(s) => {
                serde_json::from_str(s).map_err(ConversationItemStorageError::SerializationError)?
            }
            None => Value::Null,
        };

        let created_at_str = map.get("created_at").ok_or_else(|| {
            ConversationItemStorageError::StorageError(format!(
                "item {fallback_id} missing created_at"
            ))
        })?;
        let created_at = DateTime::parse_from_rfc3339(created_at_str)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| {
                ConversationItemStorageError::StorageError(format!(
                    "item {fallback_id} invalid created_at: {e}"
                ))
            })?;

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
}

#[async_trait]
impl ConversationItemStorage for RedisConversationItemStorage {
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

        let conversation_item = ConversationItem {
            id,
            response_id,
            item_type,
            role,
            content,
            status,
            created_at,
        };

        let id_str = conversation_item.id.0.as_str();
        let key = Self::item_key(id_str);

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;

        let mut pipe = redis::pipe();

        pipe.hset(&key, "id", id_str);
        if let Some(rid) = &conversation_item.response_id {
            pipe.hset(&key, "response_id", rid);
        }
        pipe.hset(&key, "item_type", &conversation_item.item_type);
        if let Some(r) = &conversation_item.role {
            pipe.hset(&key, "role", r);
        }
        pipe.hset(&key, "content", content_json);
        if let Some(s) = &conversation_item.status {
            pipe.hset(&key, "status", s);
        }
        pipe.hset(&key, "created_at", created_at.to_rfc3339());

        // Expire after configured retention days
        if let Some(days) = self.store.retention_days {
            pipe.expire(&key, (days * 24 * 60 * 60) as i64);
        }

        pipe.query_async::<()>(&mut conn)
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;

        Ok(conversation_item)
    }

    async fn link_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
        added_at: DateTime<Utc>,
    ) -> ConversationItemResult<()> {
        let cid = conversation_id.0.as_str();
        let iid = item_id.0.as_str();
        let key = Self::conv_items_key(cid);

        let score = added_at.timestamp_millis() as f64;

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        conn.zadd::<_, _, _, ()>(&key, iid, score)
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;

        Ok(())
    }

    async fn list_items(
        &self,
        conversation_id: &ConversationId,
        params: ListParams,
    ) -> ConversationItemResult<Vec<ConversationItem>> {
        let cid = conversation_id.0.as_str();
        let key = Self::conv_items_key(cid);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;

        let mut min = "-inf".to_string();
        let mut max = "+inf".to_string();
        // Track cursor score + id for post-filtering same-millisecond ties,
        // matching the composite (added_at, item_id) cursor of Postgres/Oracle.
        let mut cursor_score: Option<f64> = None;
        let mut cursor_id: Option<String> = None;

        if let Some(after_id) = &params.after {
            let score: Option<f64> = conn
                .zscore(&key, after_id)
                .await
                .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
            if let Some(s) = score {
                cursor_score = Some(s);
                cursor_id = Some(after_id.clone());
                // Use inclusive bound so we can post-filter ties by item_id.
                // Over-fetch slightly to account for items at the cursor's score.
                match params.order {
                    SortOrder::Asc => min = s.to_string(),
                    SortOrder::Desc => max = s.to_string(),
                }
            }
        }

        // Over-fetch to handle same-score ties that need filtering
        let fetch_limit = if cursor_score.is_some() {
            // Fetch extra to compensate for items we'll filter out at the cursor boundary
            (params.limit + 32) as isize
        } else {
            params.limit as isize
        };

        let item_ids: Vec<String> = match params.order {
            SortOrder::Asc => conn
                .zrangebyscore_limit(&key, min, max, 0, fetch_limit)
                .await
                .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?,
            SortOrder::Desc => conn
                .zrevrangebyscore_limit(&key, max, min, 0, fetch_limit)
                .await
                .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?,
        };

        // Post-filter: skip past the cursor item and all same-score predecessors.
        // Redis returns same-score members in lexicographic order (ASC) or
        // reverse-lex (DESC), so `skip_while` advances past items that appeared
        // on the previous page, then `skip(1)` drops the cursor item itself.
        let item_ids: Vec<String> = if let (Some(_), Some(ref c_id)) = (cursor_score, &cursor_id) {
            item_ids
                .into_iter()
                .skip_while(|id| id != c_id)
                .skip(1)
                .take(params.limit)
                .collect()
        } else {
            item_ids.into_iter().take(params.limit).collect()
        };

        if item_ids.is_empty() {
            return Ok(Vec::<ConversationItem>::new());
        }

        // Fetch all items in pipeline
        let mut pipe = redis::pipe();
        for iid in &item_ids {
            pipe.hgetall(Self::item_key(iid));
        }

        let results: Vec<std::collections::HashMap<String, String>> = pipe
            .query_async(&mut conn)
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;

        let mut items: Vec<ConversationItem> = Vec::with_capacity(results.len());
        for (i, map) in results.into_iter().enumerate() {
            if map.is_empty() {
                // Item might have been deleted or expired, skip
                continue;
            }

            items.push(Self::build_item_from_map(&map, &item_ids[i])?);
        }

        Ok(items)
    }

    async fn get_item(
        &self,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<Option<ConversationItem>> {
        let iid = item_id.0.as_str();
        let key = Self::item_key(iid);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;

        let map: std::collections::HashMap<String, String> = conn
            .hgetall(&key)
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;

        if map.is_empty() {
            return Ok(None);
        }

        Self::build_item_from_map(&map, iid).map(Some)
    }

    async fn is_item_linked(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<bool> {
        let cid = conversation_id.0.as_str();
        let iid = item_id.0.as_str();
        let key = Self::conv_items_key(cid);

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        let score: Option<f64> = conn
            .zscore(&key, iid)
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;

        Ok(score.is_some())
    }

    async fn delete_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<()> {
        let cid = conversation_id.0.as_str();
        let iid = item_id.0.as_str();
        let key = Self::conv_items_key(cid);

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        conn.zrem::<_, _, ()>(&key, iid)
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;

        Ok(())
    }
}

pub(super) struct RedisResponseStorage {
    store: RedisStore,
}

impl RedisResponseStorage {
    pub fn new(store: RedisStore) -> Self {
        Self { store }
    }

    fn response_key(id: &str) -> String {
        format!("response:{id}")
    }

    fn safety_key(identifier: &str) -> String {
        format!("safety:{identifier}:responses")
    }

    /// Build a `StoredResponse` from the Redis hash map returned by `HGETALL`.
    ///
    /// `fallback_id` is used when the map lacks an explicit `"id"` entry
    /// (e.g. when the key was derived from the sorted-set member).
    fn build_response_from_map(
        map: std::collections::HashMap<String, String>,
        fallback_id: &str,
    ) -> Result<StoredResponse, ResponseStorageError> {
        let id = ResponseId(
            map.get("id")
                .cloned()
                .unwrap_or_else(|| fallback_id.to_string()),
        );
        let previous_response_id = map
            .get("previous_response_id")
            .map(|s| ResponseId(s.clone()));
        let conversation_id = map.get("conversation_id").cloned();

        let input = parse_json_value(map.get("input").cloned())
            .map_err(ResponseStorageError::StorageError)?;
        let instructions = map.get("instructions").cloned();
        let output = parse_json_value(map.get("output").cloned())
            .map_err(ResponseStorageError::StorageError)?;
        let tool_calls = parse_tool_calls(map.get("tool_calls").cloned())
            .map_err(ResponseStorageError::StorageError)?;
        let metadata = parse_metadata(map.get("metadata").cloned())
            .map_err(ResponseStorageError::StorageError)?;

        let created_at_str = map.get("created_at").ok_or_else(|| {
            ResponseStorageError::StorageError(format!("response {fallback_id} missing created_at"))
        })?;
        let created_at = DateTime::parse_from_rfc3339(created_at_str)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| {
                ResponseStorageError::StorageError(format!(
                    "response {fallback_id} invalid created_at: {e}"
                ))
            })?;

        let safety_identifier = map.get("safety_identifier").cloned();
        let model = map.get("model").cloned();
        let raw_response = parse_raw_response(map.get("raw_response").cloned())
            .map_err(ResponseStorageError::StorageError)?;

        Ok(StoredResponse {
            id,
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
impl ResponseStorage for RedisResponseStorage {
    async fn store_response(
        &self,
        response: StoredResponse,
    ) -> Result<ResponseId, ResponseStorageError> {
        let response_id = response.id.clone();
        let response_id_str = response_id.0.as_str();
        let key = Self::response_key(response_id_str);

        let json_input = serde_json::to_string(&response.input)?;
        let json_output = serde_json::to_string(&response.output)?;
        let json_tool_calls = serde_json::to_string(&response.tool_calls)?;
        let json_metadata = serde_json::to_string(&response.metadata)?;
        let json_raw_response = serde_json::to_string(&response.raw_response)?;

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        let mut pipe = redis::pipe();

        pipe.hset(&key, "id", response_id_str);
        if let Some(prev) = &response.previous_response_id {
            pipe.hset(&key, "previous_response_id", &prev.0);
        }
        pipe.hset(&key, "input", json_input);
        if let Some(inst) = &response.instructions {
            pipe.hset(&key, "instructions", inst);
        }
        pipe.hset(&key, "output", json_output);
        pipe.hset(&key, "tool_calls", json_tool_calls);
        pipe.hset(&key, "metadata", json_metadata);
        pipe.hset(&key, "created_at", response.created_at.to_rfc3339());
        if let Some(safety) = &response.safety_identifier {
            pipe.hset(&key, "safety_identifier", safety);
        }
        if let Some(model) = &response.model {
            pipe.hset(&key, "model", model);
        }
        if let Some(cid) = &response.conversation_id {
            pipe.hset(&key, "conversation_id", cid);
        }
        pipe.hset(&key, "raw_response", json_raw_response);

        // Expire after configured retention days
        if let Some(days) = self.store.retention_days {
            pipe.expire(&key, (days * 24 * 60 * 60) as i64);
        }

        pipe.query_async::<()>(&mut conn)
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        // Index by safety identifier if present
        if let Some(safety) = &response.safety_identifier {
            let safety_key = Self::safety_key(safety);
            let score = response.created_at.timestamp_millis() as f64;
            conn.zadd::<_, _, _, ()>(safety_key, response_id_str, score)
                .await
                .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        }

        Ok(response_id)
    }

    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> Result<Option<StoredResponse>, ResponseStorageError> {
        let id = response_id.0.as_str();
        let key = Self::response_key(id);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        let map: std::collections::HashMap<String, String> = conn
            .hgetall(&key)
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        if map.is_empty() {
            return Ok(None);
        }

        Self::build_response_from_map(map, id).map(Some)
    }

    async fn delete_response(&self, response_id: &ResponseId) -> ResponseResult<()> {
        let id = response_id.0.as_str();
        let key = Self::response_key(id);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        // Atomic MULTI/EXEC: read the safety identifier and delete the hash
        // in a single transaction so no other client can modify the key
        // between the read and the delete.
        let (safety, ()): (Option<String>, ()) = redis::pipe()
            .atomic()
            .hget(&key, "safety_identifier")
            .del(&key)
            .query_async(&mut conn)
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        if let Some(s) = safety {
            conn.zrem::<_, _, ()>(Self::safety_key(&s), id)
                .await
                .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        }

        Ok(())
    }

    async fn get_response_chain(
        &self,
        response_id: &ResponseId,
        max_depth: Option<usize>,
    ) -> ResponseResult<ResponseChain> {
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
        identifier: &str,
        limit: Option<usize>,
    ) -> ResponseResult<Vec<StoredResponse>> {
        let key = Self::safety_key(identifier);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        // ZREVRANGE key 0 limit-1
        let stop = match limit {
            Some(l) => (l as isize) - 1,
            None => -1,
        };

        let response_ids: Vec<String> = conn
            .zrevrange(&key, 0, stop)
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        if response_ids.is_empty() {
            return Ok(Vec::<StoredResponse>::new());
        }

        let mut pipe = redis::pipe();
        for id in &response_ids {
            pipe.hgetall(Self::response_key(id));
        }

        let results: Vec<std::collections::HashMap<String, String>> = pipe
            .query_async(&mut conn)
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        let mut out: Vec<StoredResponse> = Vec::with_capacity(results.len());
        for (i, map) in results.into_iter().enumerate() {
            if map.is_empty() {
                continue;
            }

            out.push(Self::build_response_from_map(map, &response_ids[i])?);
        }

        Ok(out)
    }

    async fn delete_identifier_responses(&self, identifier: &str) -> ResponseResult<usize> {
        let key = Self::safety_key(identifier);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        // Get all IDs
        let response_ids: Vec<String> = conn
            .zrange(&key, 0, -1)
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        let count = response_ids.len();

        if count == 0 {
            return Ok(0);
        }

        let mut pipe = redis::pipe();
        for id in response_ids {
            pipe.del(Self::response_key(&id));
        }
        pipe.del(&key);

        pipe.query_async::<()>(&mut conn)
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        Ok(count)
    }
}
