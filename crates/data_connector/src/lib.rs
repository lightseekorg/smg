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

pub mod background;
mod common;
pub mod config;
pub mod context;
mod core;
mod factory;
mod hooked;
pub mod hooks;
mod memory;
mod memory_background;
mod noop;
mod oracle;
mod oracle_migrations;
mod postgres;
mod postgres_migrations;
mod redis;
pub mod schema;
pub(crate) mod versioning;

// Re-export config types
// Re-export core types and traits
pub use core::{
    Conversation, ConversationId, ConversationItem, ConversationItemId, ConversationItemStorage,
    ConversationMemoryId, ConversationMemoryResult, ConversationMemoryStatus,
    ConversationMemoryStorageError, ConversationMemoryType, ConversationMemoryWriter,
    ConversationStorage, ListParams, NewConversation, NewConversationItem, NewConversationMemory,
    ResponseId, ResponseStorage, ResponseStorageError, SortOrder, StoredResponse,
};

pub use background::{
    BackgroundRepositoryError, BackgroundRepositoryResult, BackgroundResponseRepository,
    DeleteResult, EnqueueRequest, FinalizeRequest, FinalizeResult, FinalizeStatus, LeasedJob,
    QueuedResponse, ResumeEventBatch, StoredCancelResult, StoredStreamEvent,
};
pub use config::{HistoryBackend, OracleConfig, PostgresConfig, RedisConfig};
// Re-export hook infrastructure
pub use context::{
    current_extra_columns, current_request_context, with_extra_columns, with_request_context,
    RequestContext,
};
// Re-export factory
pub use factory::{
    backend_supports_memory_writer, create_storage, StorageBundle, StorageFactoryConfig,
};
pub use hooks::{BeforeHookResult, ExtraColumns, HookError, StorageHook, StorageOperation};
// Re-export memory implementations for testing
pub use memory::{MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage};
pub use memory_background::MemoryBackgroundRepository;
pub use noop::NoOpConversationMemoryWriter;
// Re-export schema config types
pub use schema::{ColumnDef, SchemaConfig, TableConfig};
