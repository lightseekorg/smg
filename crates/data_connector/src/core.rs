// core.rs
//
// Core types for the data connector module.
// Contains all traits, data types, error types, and IDs for all storage backends.
//
// Structure:
// 1. Conversation types + trait
// 2. ConversationItem types + trait
// 3. Response types + trait

use std::{
    collections::{HashMap, HashSet},
    fmt::{Display, Formatter, Write},
};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value};

// ============================================================================
// Shared helpers
// ============================================================================

/// Generate a 50-character hex string from 25 cryptographically random bytes.
/// Used by both `ConversationId::new()` and `make_item_id()`.
fn random_hex_id() -> String {
    let mut rng = rand::rng();
    let mut bytes = [0u8; 25];
    rng.fill_bytes(&mut bytes);
    let mut hex_string = String::with_capacity(50);
    for b in &bytes {
        // Writing to a String is infallible; discard the always-Ok result.
        let _ = write!(hex_string, "{b:02x}");
    }
    hex_string
}

// ============================================================================
// PART 1: Conversation Storage
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ConversationId(pub String);

impl ConversationId {
    pub fn new() -> Self {
        Self(format!("conv_{}", random_hex_id()))
    }
}

impl Default for ConversationId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<String> for ConversationId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ConversationId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl Display for ConversationId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Metadata payload persisted with a conversation
pub type ConversationMetadata = JsonMap<String, Value>;

/// Input payload for creating a conversation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NewConversation {
    /// Optional conversation ID (if None, a random ID will be generated)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<ConversationId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ConversationMetadata>,
}

/// Stored conversation data structure
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Conversation {
    pub id: ConversationId,
    pub created_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ConversationMetadata>,
}

impl Conversation {
    pub fn new(new_conversation: NewConversation) -> Self {
        Self {
            id: new_conversation.id.unwrap_or_default(),
            created_at: Utc::now(),
            metadata: new_conversation.metadata,
        }
    }

    pub fn with_parts(
        id: ConversationId,
        created_at: DateTime<Utc>,
        metadata: Option<ConversationMetadata>,
    ) -> Self {
        Self {
            id,
            created_at,
            metadata,
        }
    }
}

/// Result alias for conversation storage operations
pub type ConversationResult<T> = Result<T, ConversationStorageError>;

/// Error type for conversation storage operations
#[derive(Debug, thiserror::Error)]
pub enum ConversationStorageError {
    #[error("Conversation not found: {0}")]
    ConversationNotFound(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Trait describing the CRUD interface for conversation storage backends
#[async_trait]
pub trait ConversationStorage: Send + Sync + 'static {
    async fn create_conversation(&self, input: NewConversation)
        -> ConversationResult<Conversation>;

    async fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> ConversationResult<Option<Conversation>>;

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> ConversationResult<Option<Conversation>>;

    async fn delete_conversation(&self, id: &ConversationId) -> ConversationResult<bool>;
}

// ============================================================================
// PART 2: ConversationItem Storage
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ConversationItemId(pub String);

impl Display for ConversationItemId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for ConversationItemId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ConversationItemId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationItem {
    pub id: ConversationItemId,
    pub response_id: Option<String>,
    pub item_type: String,
    pub role: Option<String>,
    pub content: Value,
    pub status: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewConversationItem {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<ConversationItemId>,
    pub response_id: Option<String>,
    pub item_type: String,
    pub role: Option<String>,
    pub content: Value,
    pub status: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SortOrder {
    Asc,
    Desc,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListParams {
    pub limit: usize,
    pub order: SortOrder,
    pub after: Option<String>, // item_id cursor
}

pub type ConversationItemResult<T> = Result<T, ConversationItemStorageError>;

#[derive(Debug, thiserror::Error)]
pub enum ConversationItemStorageError {
    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

#[async_trait]
pub trait ConversationItemStorage: Send + Sync + 'static {
    async fn create_item(
        &self,
        item: NewConversationItem,
    ) -> ConversationItemResult<ConversationItem>;

    async fn link_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
        added_at: DateTime<Utc>,
    ) -> ConversationItemResult<()>;

    /// Batch-link multiple items to a conversation in a single operation.
    /// Default implementation loops over `link_item`; backends may override
    /// with a more efficient batched approach.
    async fn link_items(
        &self,
        conversation_id: &ConversationId,
        items: &[(ConversationItemId, DateTime<Utc>)],
    ) -> ConversationItemResult<()> {
        for (item_id, added_at) in items {
            self.link_item(conversation_id, item_id, *added_at).await?;
        }
        Ok(())
    }

    async fn list_items(
        &self,
        conversation_id: &ConversationId,
        params: ListParams,
    ) -> ConversationItemResult<Vec<ConversationItem>>;

    /// Get a single item by ID
    async fn get_item(
        &self,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<Option<ConversationItem>>;

    /// Check if an item is linked to a conversation
    async fn is_item_linked(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<bool>;

    /// Delete an item link from a conversation (does not delete the item itself)
    async fn delete_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<()>;
}

/// Helper to build id prefix based on item_type
pub fn make_item_id(item_type: &str) -> ConversationItemId {
    let hex_string = random_hex_id();

    let prefix = match item_type {
        "message" => "msg",
        "reasoning" => "rs",
        "mcp_call" => "mcp",
        "mcp_list_tools" => "mcpl",
        "function_call" => "fc",
        other => {
            // Fallback: first 3 letters of type or "itm"
            let fallback: String = other.chars().take(3).collect();
            if fallback.is_empty() {
                return ConversationItemId(format!("itm_{hex_string}"));
            }
            return ConversationItemId(format!("{fallback}_{hex_string}"));
        }
    };
    ConversationItemId(format!("{prefix}_{hex_string}"))
}

// ============================================================================
// PART 3: Response Storage
// ============================================================================

/// Response identifier
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponseId(pub String);

impl ResponseId {
    pub fn new() -> Self {
        Self(ulid::Ulid::new().to_string())
    }
}

impl Display for ResponseId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Default for ResponseId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<String> for ResponseId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ResponseId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

/// Stored response data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredResponse {
    /// Unique response ID
    pub id: ResponseId,

    /// ID of the previous response in the chain (if any)
    pub previous_response_id: Option<ResponseId>,

    /// Input items as JSON array
    pub input: Value,

    /// When this response was created
    pub created_at: DateTime<Utc>,

    /// Safety identifier for content moderation
    pub safety_identifier: Option<String>,

    /// Model used for generation
    pub model: Option<String>,

    /// Conversation id if associated with a conversation
    #[serde(default)]
    pub conversation_id: Option<String>,

    /// Raw OpenAI response payload
    #[serde(default)]
    pub raw_response: Value,
}

impl StoredResponse {
    pub fn new(previous_response_id: Option<ResponseId>) -> Self {
        Self {
            id: ResponseId::new(),
            previous_response_id,
            input: Value::Array(vec![]),
            created_at: Utc::now(),
            safety_identifier: None,
            model: None,
            conversation_id: None,
            raw_response: Value::Null,
        }
    }
}

/// Response chain - a sequence of related responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseChain {
    /// The responses in chronological order
    pub responses: Vec<StoredResponse>,

    /// Metadata about the chain
    pub metadata: HashMap<String, Value>,
}

impl Default for ResponseChain {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseChain {
    pub fn new() -> Self {
        Self {
            responses: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Get the ID of the most recent response in the chain
    pub fn latest_response_id(&self) -> Option<&ResponseId> {
        self.responses.last().map(|r| &r.id)
    }

    /// Add a response to the chain
    pub fn add_response(&mut self, response: StoredResponse) {
        self.responses.push(response);
    }

    /// Build context from the chain for the next request
    pub fn build_context(&self, max_responses: Option<usize>) -> Vec<(Value, Value)> {
        let responses = if let Some(max) = max_responses {
            let start = self.responses.len().saturating_sub(max);
            &self.responses[start..]
        } else {
            &self.responses[..]
        };

        responses
            .iter()
            .map(|r| {
                let output = r
                    .raw_response
                    .get("output")
                    .cloned()
                    .unwrap_or(Value::Array(vec![]));
                (r.input.clone(), output)
            })
            .collect()
    }
}

/// Error type for response storage operations
#[derive(Debug, thiserror::Error)]
pub enum ResponseStorageError {
    #[error("Response not found: {0}")]
    ResponseNotFound(String),

    #[error("Invalid chain: {0}")]
    InvalidChain(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

pub type ResponseResult<T> = Result<T, ResponseStorageError>;

/// Trait for response storage
#[async_trait]
pub trait ResponseStorage: Send + Sync {
    /// Store a new response
    async fn store_response(&self, response: StoredResponse) -> ResponseResult<ResponseId>;

    /// Get a response by ID
    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> ResponseResult<Option<StoredResponse>>;

    /// Delete a response
    async fn delete_response(&self, response_id: &ResponseId) -> ResponseResult<()>;

    /// Get the chain of responses leading to a given response.
    ///
    /// Walks `previous_response_id` links from the given response backwards,
    /// collecting up to `max_depth` responses (or unlimited if `None`).
    /// Returns responses in chronological order (oldest first).
    ///
    /// The default implementation calls `self.get_response()` in a loop with
    /// cycle detection to prevent infinite loops from self-referencing chains.
    /// Backends that can walk the chain more efficiently (e.g. with a single
    /// lock or a recursive SQL query) should override this.
    async fn get_response_chain(
        &self,
        response_id: &ResponseId,
        max_depth: Option<usize>,
    ) -> ResponseResult<ResponseChain> {
        let mut chain = ResponseChain::new();
        let mut current_id = Some(response_id.clone());
        let mut seen = HashSet::new();

        while let Some(ref lookup_id) = current_id {
            if let Some(limit) = max_depth {
                if seen.len() >= limit {
                    break;
                }
            }

            // Cycle detection: error if we've already visited this ID.
            if !seen.insert(lookup_id.clone()) {
                return Err(ResponseStorageError::InvalidChain(format!(
                    "cycle detected at response {}",
                    lookup_id.0
                )));
            }

            let fetched = self.get_response(lookup_id).await?;
            match fetched {
                Some(response) => {
                    current_id.clone_from(&response.previous_response_id);
                    chain.responses.push(response);
                }
                None => break,
            }
        }

        chain.responses.reverse();
        Ok(chain)
    }

    /// List recent responses for a safety identifier
    async fn list_identifier_responses(
        &self,
        identifier: &str,
        limit: Option<usize>,
    ) -> ResponseResult<Vec<StoredResponse>>;

    /// Delete all responses for a safety identifier
    async fn delete_identifier_responses(&self, identifier: &str) -> ResponseResult<usize>;
}

// ============================================================================
// PART 4: ConversationMemory Insert Seam
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ConversationMemoryId(pub String);

impl From<String> for ConversationMemoryId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ConversationMemoryId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl Display for ConversationMemoryId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConversationMemoryType {
    #[serde(rename = "ONDEMAND")]
    OnDemand,
    #[serde(rename = "LTM")]
    Ltm,
    #[serde(rename = "STMO")]
    Stmo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConversationMemoryStatus {
    #[serde(rename = "READY")]
    Ready,
    #[serde(rename = "RUNNING")]
    Running,
    #[serde(rename = "SUCCESS")]
    Success,
    #[serde(rename = "FAILED")]
    Failed,
}

/// Insert-only payload for creating a new conversation memory row.
///
/// `NewConversationMemory` intentionally omits database-managed fields such as
/// `memory_id`, `created_at`, and `updated_at`. Callers should not set those
/// values directly; they are created or maintained by the database/write path.
/// When changing an existing record, use the appropriate update path rather
/// than reusing this struct.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NewConversationMemory {
    pub conversation_id: ConversationId,
    pub conversation_version: Option<i64>,
    pub response_id: Option<ResponseId>,
    pub memory_type: ConversationMemoryType,
    pub status: ConversationMemoryStatus,
    pub attempt: i64,
    pub owner_id: Option<String>,
    pub next_run_at: DateTime<Utc>,
    pub lease_until: Option<DateTime<Utc>>,
    pub content: Option<String>,
    pub memory_config: Option<String>,
    pub scope_id: Option<String>,
    pub error_msg: Option<String>,
}

pub type ConversationMemoryResult<T> = Result<T, ConversationMemoryStorageError>;

#[derive(Debug, thiserror::Error)]
pub enum ConversationMemoryStorageError {
    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

#[async_trait]
pub trait ConversationMemoryWriter: Send + Sync + 'static {
    async fn create_memory(
        &self,
        input: NewConversationMemory,
    ) -> ConversationMemoryResult<ConversationMemoryId>;
}

impl Default for StoredResponse {
    fn default() -> Self {
        Self::new(None)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    // ========================================================================
    // ConversationId tests
    // ========================================================================

    #[test]
    fn conversation_id_new_has_conv_prefix() {
        let id = ConversationId::new();
        assert!(
            id.0.starts_with("conv_"),
            "ConversationId should start with 'conv_', got: {id}"
        );
    }

    #[test]
    fn conversation_id_new_generates_unique_ids() {
        let ids: HashSet<String> = (0..100).map(|_| ConversationId::new().0).collect();
        assert_eq!(ids.len(), 100, "all 100 ConversationIds should be unique");
    }

    #[test]
    fn conversation_id_new_has_consistent_length() {
        // "conv_" (5 chars) + 50 hex chars = 55 total
        for _ in 0..10 {
            let id = ConversationId::new();
            assert_eq!(
                id.0.len(),
                55,
                "ConversationId should be 55 chars (conv_ + 50 hex), got {} chars: {id}",
                id.0.len()
            );
        }
    }

    #[test]
    fn conversation_id_default_works_same_as_new() {
        let id = ConversationId::default();
        assert!(
            id.0.starts_with("conv_"),
            "Default ConversationId should start with 'conv_', got: {id}"
        );
        assert_eq!(id.0.len(), 55, "Default ConversationId should be 55 chars");
    }

    #[test]
    fn conversation_id_from_string() {
        let id = ConversationId::from("my_custom_id".to_string());
        assert_eq!(id.0, "my_custom_id");
    }

    #[test]
    fn conversation_id_from_str() {
        let id = ConversationId::from("my_custom_id");
        assert_eq!(id.0, "my_custom_id");
    }

    #[test]
    fn conversation_id_display() {
        let id = ConversationId::from("conv_abc123");
        assert_eq!(format!("{id}"), "conv_abc123");
    }

    // ========================================================================
    // ConversationItemId tests
    // ========================================================================

    #[test]
    fn conversation_item_id_from_string() {
        let id = ConversationItemId::from("item_123".to_string());
        assert_eq!(id.0, "item_123");
    }

    #[test]
    fn conversation_item_id_from_str() {
        let id = ConversationItemId::from("item_456");
        assert_eq!(id.0, "item_456");
    }

    #[test]
    fn conversation_item_id_display() {
        let id = ConversationItemId::from("msg_abc");
        assert_eq!(format!("{id}"), "msg_abc");
    }

    // ========================================================================
    // ResponseId tests
    // ========================================================================

    #[test]
    fn response_id_new_generates_valid_ulid() {
        let id = ResponseId::new();
        // ULID strings are 26 characters, uppercase alphanumeric (Crockford Base32)
        assert_eq!(
            id.0.len(),
            26,
            "ULID string should be 26 chars, got {} chars: {}",
            id.0.len(),
            id.0
        );
        assert!(
            id.0.chars().all(|c| c.is_ascii_alphanumeric()),
            "ULID should contain only alphanumeric characters, got: {}",
            id.0
        );
    }

    #[test]
    fn response_id_new_generates_unique_ids() {
        let ids: HashSet<String> = (0..100).map(|_| ResponseId::new().0).collect();
        assert_eq!(ids.len(), 100, "all 100 ResponseIds should be unique");
    }

    #[test]
    fn response_id_default_works_same_as_new() {
        let id = ResponseId::default();
        assert_eq!(id.0.len(), 26, "Default ResponseId should be 26-char ULID");
    }

    #[test]
    fn response_id_from_string() {
        let id = ResponseId::from("resp_custom".to_string());
        assert_eq!(id.0, "resp_custom");
    }

    #[test]
    fn response_id_from_str() {
        let id = ResponseId::from("resp_custom");
        assert_eq!(id.0, "resp_custom");
    }

    // ========================================================================
    // make_item_id() tests
    // ========================================================================

    #[test]
    fn make_item_id_message_prefix() {
        let id = make_item_id("message");
        assert!(
            id.0.starts_with("msg_"),
            "message type should produce 'msg_' prefix, got: {id}"
        );
    }

    #[test]
    fn make_item_id_reasoning_prefix() {
        let id = make_item_id("reasoning");
        assert!(
            id.0.starts_with("rs_"),
            "reasoning type should produce 'rs_' prefix, got: {id}"
        );
    }

    #[test]
    fn make_item_id_mcp_call_prefix() {
        let id = make_item_id("mcp_call");
        assert!(
            id.0.starts_with("mcp_"),
            "mcp_call type should produce 'mcp_' prefix, got: {id}"
        );
    }

    #[test]
    fn make_item_id_mcp_list_tools_prefix() {
        let id = make_item_id("mcp_list_tools");
        assert!(
            id.0.starts_with("mcpl_"),
            "mcp_list_tools type should produce 'mcpl_' prefix, got: {id}"
        );
    }

    #[test]
    fn make_item_id_function_call_prefix() {
        let id = make_item_id("function_call");
        assert!(
            id.0.starts_with("fc_"),
            "function_call type should produce 'fc_' prefix, got: {id}"
        );
    }

    #[test]
    fn make_item_id_unknown_type_uses_first_3_chars() {
        let id = make_item_id("custom_type");
        assert!(
            id.0.starts_with("cus_"),
            "unknown type 'custom_type' should produce 'cus_' prefix, got: {id}"
        );
    }

    #[test]
    fn make_item_id_empty_type_uses_itm() {
        let id = make_item_id("");
        assert!(
            id.0.starts_with("itm_"),
            "empty type string should produce 'itm_' prefix, got: {id}"
        );
    }

    #[test]
    fn make_item_id_correct_length() {
        // Each known prefix: prefix + "_" + 50 hex chars
        let test_cases = vec![
            ("message", "msg_"),
            ("reasoning", "rs_"),
            ("mcp_call", "mcp_"),
            ("mcp_list_tools", "mcpl_"),
            ("function_call", "fc_"),
        ];

        for (item_type, prefix) in test_cases {
            let id = make_item_id(item_type);
            let expected_len = prefix.len() + 50;
            assert_eq!(
                id.0.len(),
                expected_len,
                "make_item_id(\"{item_type}\") should be {expected_len} chars ('{prefix}' + 50 hex), got {} chars: {id}",
                id.0.len()
            );
        }

        // Unknown type: first 3 chars + "_" + 50 hex = 54 chars
        let id = make_item_id("custom_type");
        assert_eq!(
            id.0.len(),
            54,
            "unknown type should be 54 chars (3 char prefix + '_' + 50 hex), got {} chars: {id}",
            id.0.len()
        );

        // Empty type: "itm_" + 50 hex = 54 chars
        let id = make_item_id("");
        assert_eq!(
            id.0.len(),
            54,
            "empty type should be 54 chars ('itm_' + 50 hex), got {} chars: {id}",
            id.0.len()
        );
    }

    // ========================================================================
    // Conversation tests
    // ========================================================================

    #[test]
    fn conversation_new_generates_id_if_none_provided() {
        let conv = Conversation::new(NewConversation {
            id: None,
            metadata: None,
        });
        assert!(
            conv.id.0.starts_with("conv_"),
            "should generate a ConversationId when none provided, got: {}",
            conv.id
        );
    }

    #[test]
    fn conversation_new_uses_provided_id() {
        let custom_id = ConversationId::from("my_conv_id");
        let conv = Conversation::new(NewConversation {
            id: Some(custom_id.clone()),
            metadata: None,
        });
        assert_eq!(conv.id, custom_id, "should use the provided ConversationId");
    }

    #[test]
    fn conversation_with_parts_preserves_all_fields() {
        let id = ConversationId::from("test_id");
        let created_at = Utc::now();
        let mut metadata = ConversationMetadata::new();
        metadata.insert("key".to_string(), Value::String("value".to_string()));

        let conv = Conversation::with_parts(id.clone(), created_at, Some(metadata.clone()));

        assert_eq!(conv.id, id);
        assert_eq!(conv.created_at, created_at);
        assert_eq!(conv.metadata, Some(metadata));
    }

    // ========================================================================
    // StoredResponse tests
    // ========================================================================

    #[test]
    fn stored_response_new_none_has_no_previous() {
        let resp = StoredResponse::new(None);
        assert!(
            resp.previous_response_id.is_none(),
            "new(None) should have no previous_response_id"
        );
    }

    #[test]
    fn stored_response_new_some_has_correct_previous() {
        let prev_id = ResponseId::from("prev_123");
        let resp = StoredResponse::new(Some(prev_id.clone()));
        assert_eq!(
            resp.previous_response_id,
            Some(prev_id),
            "new(Some(id)) should set previous_response_id"
        );
    }

    #[test]
    fn stored_response_default_works() {
        let resp = StoredResponse::default();
        assert!(
            resp.previous_response_id.is_none(),
            "default() should have no previous_response_id"
        );
        assert_eq!(
            resp.input,
            Value::Array(vec![]),
            "default input should be empty array"
        );
        assert_eq!(
            resp.raw_response,
            Value::Null,
            "default raw_response should be Null"
        );
    }

    // ========================================================================
    // ResponseChain tests
    // ========================================================================

    #[test]
    fn response_chain_new_creates_empty_chain() {
        let chain = ResponseChain::new();
        assert!(
            chain.responses.is_empty(),
            "new chain should have no responses"
        );
        assert!(
            chain.metadata.is_empty(),
            "new chain should have no metadata"
        );
    }

    #[test]
    fn response_chain_add_response_appends() {
        let mut chain = ResponseChain::new();
        let r1 = StoredResponse::new(None);
        let r2 = StoredResponse::new(None);
        let r1_id = r1.id.clone();
        let r2_id = r2.id.clone();

        chain.add_response(r1);
        assert_eq!(chain.responses.len(), 1, "chain should have 1 response");

        chain.add_response(r2);
        assert_eq!(chain.responses.len(), 2, "chain should have 2 responses");
        assert_eq!(chain.responses[0].id, r1_id, "first response should be r1");
        assert_eq!(chain.responses[1].id, r2_id, "second response should be r2");
    }

    #[test]
    fn response_chain_latest_response_id_returns_last() {
        let mut chain = ResponseChain::new();
        let r1 = StoredResponse::new(None);
        let r2 = StoredResponse::new(None);
        let r2_id = r2.id.clone();

        chain.add_response(r1);
        chain.add_response(r2);

        assert_eq!(
            chain.latest_response_id(),
            Some(&r2_id),
            "latest_response_id should return the last response's ID"
        );
    }

    #[test]
    fn response_chain_latest_response_id_returns_none_for_empty() {
        let chain = ResponseChain::new();
        assert_eq!(
            chain.latest_response_id(),
            None,
            "latest_response_id should return None for empty chain"
        );
    }

    #[test]
    fn response_chain_build_context_returns_input_output_pairs() {
        use serde_json::json;

        let mut chain = ResponseChain::new();

        let mut r1 = StoredResponse::new(None);
        r1.input = Value::String("input1".to_string());
        r1.raw_response = json!({"output": "output1"});

        let mut r2 = StoredResponse::new(None);
        r2.input = Value::String("input2".to_string());
        r2.raw_response = json!({"output": "output2"});

        chain.add_response(r1);
        chain.add_response(r2);

        let context = chain.build_context(None);
        assert_eq!(context.len(), 2, "should return 2 pairs");
        assert_eq!(context[0].0, Value::String("input1".to_string()));
        assert_eq!(context[0].1, Value::String("output1".to_string()));
        assert_eq!(context[1].0, Value::String("input2".to_string()));
        assert_eq!(context[1].1, Value::String("output2".to_string()));
    }

    #[test]
    fn response_chain_build_context_with_max_responses_limits_output() {
        use serde_json::json;

        let mut chain = ResponseChain::new();

        for i in 0..5 {
            let mut resp = StoredResponse::new(None);
            resp.input = Value::String(format!("input{i}"));
            resp.raw_response = json!({"output": format!("output{i}")});
            chain.add_response(resp);
        }

        let context = chain.build_context(Some(2));
        assert_eq!(context.len(), 2, "should return only 2 most recent pairs");
        // Should be the last 2 responses (index 3 and 4)
        assert_eq!(context[0].0, Value::String("input3".to_string()));
        assert_eq!(context[0].1, Value::String("output3".to_string()));
        assert_eq!(context[1].0, Value::String("input4".to_string()));
        assert_eq!(context[1].1, Value::String("output4".to_string()));
    }

    // ========================================================================
    // ConversationMemory seam tests
    // ========================================================================

    #[test]
    fn conversation_memory_status_serializes_to_expected_uppercase_values() {
        assert_eq!(
            serde_json::to_string(&ConversationMemoryStatus::Ready).unwrap(),
            "\"READY\""
        );
        assert_eq!(
            serde_json::to_string(&ConversationMemoryStatus::Running).unwrap(),
            "\"RUNNING\""
        );
        assert_eq!(
            serde_json::to_string(&ConversationMemoryStatus::Success).unwrap(),
            "\"SUCCESS\""
        );
        assert_eq!(
            serde_json::to_string(&ConversationMemoryStatus::Failed).unwrap(),
            "\"FAILED\""
        );
    }

    #[test]
    fn conversation_memory_type_serializes_to_expected_uppercase_values() {
        assert_eq!(
            serde_json::to_string(&ConversationMemoryType::OnDemand).unwrap(),
            "\"ONDEMAND\""
        );
        assert_eq!(
            serde_json::to_string(&ConversationMemoryType::Ltm).unwrap(),
            "\"LTM\""
        );
        assert_eq!(
            serde_json::to_string(&ConversationMemoryType::Stmo).unwrap(),
            "\"STMO\""
        );
    }

    #[test]
    fn new_conversation_memory_keeps_insert_only_fields() {
        let input = NewConversationMemory {
            conversation_id: ConversationId::from("conv_123"),
            conversation_version: Some(7),
            response_id: Some(ResponseId::from("resp_123")),
            memory_type: ConversationMemoryType::Ltm,
            status: ConversationMemoryStatus::Ready,
            attempt: 0,
            owner_id: None,
            next_run_at: Utc::now(),
            lease_until: None,
            content: None,
            memory_config: None,
            scope_id: None,
            error_msg: None,
        };

        assert_eq!(input.attempt, 0);
        assert_eq!(input.conversation_id, ConversationId::from("conv_123"));
        assert_eq!(input.response_id, Some(ResponseId::from("resp_123")));
        assert!(input.lease_until.is_none());
    }
}
