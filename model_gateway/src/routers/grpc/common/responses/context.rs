//! Shared context for /v1/responses endpoint handlers
//!
//! This context is used by both regular and harmony response implementations.

use std::sync::Arc;

use smg_data_connector::{
    ConversationItemStorage, ConversationMemoryWriter, ConversationStorage,
    RequestContext as StorageRequestContext, ResponseStorage,
};
use smg_mcp::McpOrchestrator;

use crate::{
    memory::MemoryExecutionContext,
    routers::grpc::{context::SharedComponents, pipeline::RequestPipeline},
};

/// Bundled storage handles for persistence operations.
///
/// Groups the four storage backends that every persistence call needs so they
/// can be passed as a single unit rather than four individual arguments.
/// Mirrors the pattern introduced in the LTM pipeline (PR #1357) so the two
/// code paths stay consistent and future merges remain clean.
#[derive(Clone)]
pub(crate) struct PersistenceHandles {
    pub response_storage: Arc<dyn ResponseStorage>,
    pub conversation_storage: Arc<dyn ConversationStorage>,
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,
    pub conversation_memory_writer: Arc<dyn ConversationMemoryWriter>,
}

/// Context for /v1/responses endpoint
///
/// Used by both regular and harmony implementations.
/// All fields are Arc/shared references, so cloning this context is cheap.
#[derive(Clone)]
pub(crate) struct ResponsesContext {
    /// Chat pipeline for executing requests
    pub pipeline: Arc<RequestPipeline>,

    /// Shared components (tokenizer, parsers)
    pub components: Arc<SharedComponents>,

    /// Bundled storage handles for persistence operations.
    pub persistence: PersistenceHandles,

    /// MCP orchestrator for tool support
    pub mcp_orchestrator: Arc<McpOrchestrator>,

    /// Storage hook request context extracted from HTTP headers by middleware.
    pub request_context: Option<StorageRequestContext>,

    /// Maximum conversation history items to load into request context.
    pub max_conversation_history_items: usize,

    /// Memory execution context derived from per-request headers.
    ///
    /// Controls whether LTM store/recall and STM condensation are active for
    /// this request.  Built from `x-conversation-memory-config` + the runtime
    /// feature flag at the gRPC entry point and threaded down to the persistence
    /// layer so it can gate memory side-effects without re-parsing headers.
    pub memory_execution_context: MemoryExecutionContext,
}

impl ResponsesContext {
    /// Create a new responses context.
    pub fn new(
        pipeline: Arc<RequestPipeline>,
        components: Arc<SharedComponents>,
        persistence: PersistenceHandles,
        mcp_orchestrator: Arc<McpOrchestrator>,
        request_context: Option<StorageRequestContext>,
        max_conversation_history_items: usize,
        memory_execution_context: MemoryExecutionContext,
    ) -> Self {
        Self {
            pipeline,
            components,
            persistence,
            mcp_orchestrator,
            request_context,
            max_conversation_history_items,
            memory_execution_context,
        }
    }
}
