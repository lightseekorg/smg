//! Data connector module for response storage and conversation storage.
//!
//! Provides storage backends for:
//! - Conversations
//! - Conversation items
//! - Responses
//!
//! Supported backends:
//! - Memory (default)
//! - None (no-op)
//! - Oracle ATP
//! - Postgres
//! - Redis

mod common;
pub mod config;
mod core;
mod factory;
mod memory;
mod noop;
mod oracle;
mod postgres;
mod redis;

// Re-export config types
// Re-export core types and traits
pub use core::{
    Conversation, ConversationId, ConversationItem, ConversationItemId, ConversationItemStorage,
    ConversationStorage, ListParams, NewConversation, NewConversationItem, ResponseId,
    ResponseStorage, SortOrder, StoredResponse,
};

pub use config::{HistoryBackend, OracleConfig, PostgresConfig, RedisConfig};
// Re-export factory
pub use factory::{create_storage, StorageFactoryConfig};
// Re-export memory implementations for testing
pub use memory::{MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage};
