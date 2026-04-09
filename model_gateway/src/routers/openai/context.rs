//! Request context types for OpenAI router pipeline.

use std::sync::Arc;

use axum::http::HeaderMap;
use openai_protocol::{chat::ChatCompletionRequest, responses::ResponsesRequest};
use serde_json::Value;
use smg_data_connector::{
    ConversationItemStorage, ConversationMemoryWriter, ConversationStorage,
    RequestContext as StorageRequestContext, ResponseStorage,
};
use smg_mcp::{McpOrchestrator, McpToolSession};

use super::provider::Provider;
use crate::{config::RouterConfig, memory::MemoryExecutionContext, middleware, worker::Worker};

/// Mutable per-request context used throughout OpenAI router handling.
pub struct RequestContext {
    /// Parsed request input and request metadata.
    pub input: RequestInput,
    /// Shared components available to handlers for this request.
    pub components: ComponentRefs,
    /// Mutable processing state populated as the pipeline advances.
    pub state: ProcessingState,
    /// Storage hook request context extracted from HTTP headers by middleware.
    pub storage_request_context: Option<StorageRequestContext>,
    pub memory_execution_context: MemoryExecutionContext,
}

/// Immutable request input metadata captured at context creation.
pub struct RequestInput {
    /// Request payload variant.
    pub request_type: RequestType,
    /// Original HTTP headers when available.
    pub headers: Option<HeaderMap>,
    /// Optional model id override.
    pub model_id: Option<String>,
}

/// Supported OpenAI-compatible request shapes.
pub enum RequestType {
    /// Chat Completions request payload.
    Chat(Arc<ChatCompletionRequest>),
    /// Responses API request payload.
    Responses(Arc<ResponsesRequest>),
}

#[derive(Clone)]
/// Components shared by all OpenAI router request types.
pub struct SharedComponents {
    /// HTTP client used for worker/proxy calls.
    pub client: reqwest::Client,
    /// Router runtime configuration.
    pub router_config: Arc<RouterConfig>,
}

/// Components required by Responses-specific flows.
pub struct ResponsesComponents {
    /// Shared components also used by non-responses flows.
    pub shared: Arc<SharedComponents>,
    /// MCP orchestrator used for tool session management.
    pub mcp_orchestrator: Arc<McpOrchestrator>,
    /// Response persistence backend.
    pub response_storage: Arc<dyn ResponseStorage>,
    /// Conversation persistence backend.
    pub conversation_storage: Arc<dyn ConversationStorage>,
    /// Conversation item persistence backend.
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,
    /// Optional memory writer for LTM store operations.
    pub conversation_memory_writer: Option<Arc<dyn ConversationMemoryWriter>>,
}

/// Union of component sets available to a request context.
pub enum ComponentRefs {
    /// Shared-only components (chat and other non-responses paths).
    Shared(Arc<SharedComponents>),
    /// Full responses components with storage handles.
    Responses(Arc<ResponsesComponents>),
}

impl ComponentRefs {
    /// Access HTTP client for outbound worker/proxy calls.
    pub fn client(&self) -> &reqwest::Client {
        match self {
            ComponentRefs::Shared(s) => &s.client,
            ComponentRefs::Responses(r) => &r.shared.client,
        }
    }

    /// Access router configuration shared by both request context variants.
    pub fn router_config(&self) -> &Arc<RouterConfig> {
        match self {
            ComponentRefs::Shared(s) => &s.router_config,
            ComponentRefs::Responses(r) => &r.shared.router_config,
        }
    }

    /// Access MCP orchestrator when responses components are present.
    pub fn mcp_orchestrator(&self) -> Option<&Arc<McpOrchestrator>> {
        match self {
            ComponentRefs::Shared(_) => None,
            ComponentRefs::Responses(r) => Some(&r.mcp_orchestrator),
        }
    }

    /// Access response storage when responses components are present.
    pub fn response_storage(&self) -> Option<&Arc<dyn ResponseStorage>> {
        match self {
            ComponentRefs::Shared(_) => None,
            ComponentRefs::Responses(r) => Some(&r.response_storage),
        }
    }

    /// Access conversation storage when responses components are present.
    pub fn conversation_storage(&self) -> Option<&Arc<dyn ConversationStorage>> {
        match self {
            ComponentRefs::Shared(_) => None,
            ComponentRefs::Responses(r) => Some(&r.conversation_storage),
        }
    }

    /// Access conversation item storage when responses components are present.
    pub fn conversation_item_storage(&self) -> Option<&Arc<dyn ConversationItemStorage>> {
        match self {
            ComponentRefs::Shared(_) => None,
            ComponentRefs::Responses(r) => Some(&r.conversation_item_storage),
        }
    }

    /// Access optional conversation memory writer when response storage is available.
    pub fn conversation_memory_writer(&self) -> Option<&Arc<dyn ConversationMemoryWriter>> {
        match self {
            ComponentRefs::Shared(_) => None,
            ComponentRefs::Responses(r) => r.conversation_memory_writer.as_ref(),
        }
    }
}

#[derive(Default)]
/// Mutable state accumulated while processing a request.
pub struct ProcessingState {
    /// Selected worker/provider pair once routing is resolved.
    pub worker: Option<WorkerSelection>,
    /// Prepared outbound payload and target URL once rendered.
    pub payload: Option<PayloadState>,
}

/// Selected worker and provider binding for request execution.
pub struct WorkerSelection {
    /// Destination worker chosen by routing policy.
    pub worker: Arc<dyn Worker>,
    /// Provider strategy used to render/transform payloads.
    pub provider: Arc<dyn Provider>,
}

/// Prepared outbound request payload and metadata.
pub struct PayloadState {
    /// Serialized outbound JSON payload.
    pub json: Value,
    /// Destination worker URL.
    pub url: String,
    /// Prior response id when continuing a conversation.
    pub previous_response_id: Option<String>,
}

impl RequestContext {
    /// Build request context for Responses API calls and initialize memory execution context.
    pub fn for_responses(
        request: Arc<ResponsesRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: ComponentRefs,
    ) -> Self {
        let empty_headers = HeaderMap::new();
        let memory_execution_context = middleware::build_memory_execution_context(
            components.router_config(),
            headers.as_ref().unwrap_or(&empty_headers),
        );

        Self {
            input: RequestInput {
                request_type: RequestType::Responses(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
            storage_request_context: None,
            memory_execution_context,
        }
    }

    /// Build request context for Chat Completions calls and initialize memory execution context.
    pub fn for_chat(
        request: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: ComponentRefs,
    ) -> Self {
        let empty_headers = HeaderMap::new();
        let memory_execution_context = middleware::build_memory_execution_context(
            components.router_config(),
            headers.as_ref().unwrap_or(&empty_headers),
        );

        Self {
            input: RequestInput {
                request_type: RequestType::Chat(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
            storage_request_context: None,
            memory_execution_context,
        }
    }
}

impl RequestContext {
    /// Recompute memory execution context from current headers and router runtime settings.
    pub fn refresh_memory_execution_context(&mut self) {
        let empty_headers = HeaderMap::new();
        let headers = self.headers().unwrap_or(&empty_headers);
        self.memory_execution_context =
            middleware::build_memory_execution_context(self.components.router_config(), headers);
    }

    /// Return responses request reference when this context contains responses input.
    pub fn responses_request(&self) -> Option<&ResponsesRequest> {
        match &self.input.request_type {
            RequestType::Responses(req) => Some(req.as_ref()),
            RequestType::Chat(_) => None,
        }
    }

    /// Clone and return responses request when this context contains responses input.
    pub fn responses_request_arc(&self) -> Option<Arc<ResponsesRequest>> {
        match &self.input.request_type {
            RequestType::Responses(req) => Some(Arc::clone(req)),
            RequestType::Chat(_) => None,
        }
    }

    /// Whether the underlying request asks for streaming output.
    pub fn is_streaming(&self) -> bool {
        match &self.input.request_type {
            RequestType::Chat(req) => req.stream,
            RequestType::Responses(req) => req.stream.unwrap_or(false),
        }
    }

    /// Borrow request headers when present.
    pub fn headers(&self) -> Option<&HeaderMap> {
        self.input.headers.as_ref()
    }

    /// Borrow model id override when present.
    pub fn model_id(&self) -> Option<&str> {
        self.input.model_id.as_deref()
    }

    /// Borrow selected worker when routing has completed.
    pub fn worker(&self) -> Option<&Arc<dyn Worker>> {
        self.state.worker.as_ref().map(|w| &w.worker)
    }

    /// Borrow selected provider when routing has completed.
    pub fn provider(&self) -> Option<&dyn Provider> {
        self.state.worker.as_ref().map(|w| w.provider.as_ref())
    }

    /// Borrow prepared payload state when available.
    pub fn payload(&self) -> Option<&PayloadState> {
        self.state.payload.as_ref()
    }

    /// Take ownership of prepared payload state, leaving `None` behind.
    pub fn take_payload(&mut self) -> Option<PayloadState> {
        self.state.payload.take()
    }
}

/// Storage handles packaged for streaming response lifecycle.
pub struct StorageHandles {
    /// Response persistence backend.
    pub response: Arc<dyn ResponseStorage>,
    /// Conversation persistence backend.
    pub conversation: Arc<dyn ConversationStorage>,
    /// Conversation item persistence backend.
    pub conversation_item: Arc<dyn ConversationItemStorage>,
    /// Optional conversation memory writer.
    pub conversation_memory_writer: Option<Arc<dyn ConversationMemoryWriter>>,
    /// Optional hook context propagated from inbound request headers.
    pub request_context: Option<StorageRequestContext>,
    /// Effective memory execution context for this request.
    pub memory_execution_context: MemoryExecutionContext,
}

/// Owned request state used by streaming handlers.
pub struct OwnedStreamingContext {
    /// Destination URL for worker call.
    pub url: String,
    /// Serialized outbound payload.
    pub payload: Value,
    /// Original responses request body.
    pub original_body: ResponsesRequest,
    /// Prior response id when request continues an existing thread.
    pub previous_response_id: Option<String>,
    /// Storage handles required for persistence updates during streaming.
    pub storage: StorageHandles,
}

impl RequestContext {
    /// Convert request context into owned streaming context once payload is prepared.
    pub fn into_streaming_context(mut self) -> Result<OwnedStreamingContext, &'static str> {
        let payload_state = self.take_payload().ok_or("Payload not prepared")?;
        let original_body = self
            .responses_request()
            .ok_or("Expected responses request")?
            .clone();
        let response = self
            .components
            .response_storage()
            .ok_or("Response storage required")?
            .clone();
        let conversation = self
            .components
            .conversation_storage()
            .ok_or("Conversation storage required")?
            .clone();
        let conversation_item = self
            .components
            .conversation_item_storage()
            .ok_or("Conversation item storage required")?
            .clone();

        Ok(OwnedStreamingContext {
            url: payload_state.url,
            payload: payload_state.json,
            original_body,
            previous_response_id: payload_state.previous_response_id,
            storage: StorageHandles {
                response,
                conversation,
                conversation_item,
                conversation_memory_writer: self.components.conversation_memory_writer().cloned(),
                request_context: self.storage_request_context,
                memory_execution_context: self.memory_execution_context,
            },
        })
    }
}

/// Lightweight borrowed context passed to per-event streaming emitters.
pub struct StreamingEventContext<'a> {
    /// Original responses request.
    pub original_request: &'a ResponsesRequest,
    /// Prior response id if the request continues a thread.
    pub previous_response_id: Option<&'a str>,
    /// Optional MCP session borrowed during streaming callbacks.
    pub session: Option<&'a McpToolSession<'a>>,
}

/// Backward-compatible alias for streaming request context.
pub type StreamingRequest = OwnedStreamingContext;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use axum::http::{HeaderMap, HeaderValue};
    use openai_protocol::{chat::ChatCompletionRequest, responses::ResponsesRequest};
    use serde_json::json;
    use smg_data_connector::{
        ConversationMemoryId, ConversationMemoryResult, ConversationMemoryWriter,
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
        NewConversationMemory,
    };

    use super::{
        ComponentRefs, PayloadState, RequestContext, ResponsesComponents, SharedComponents,
    };
    use crate::{
        config::RouterConfig,
        memory::{MemoryRuntimeConfig, MEMORY_LTM_STORE_ENABLED_HEADER},
    };

    struct DummyMemoryWriter;

    #[async_trait]
    impl ConversationMemoryWriter for DummyMemoryWriter {
        async fn create_memory(
            &self,
            _input: NewConversationMemory,
        ) -> ConversationMemoryResult<ConversationMemoryId> {
            Ok(ConversationMemoryId("mem_test".to_string()))
        }
    }

    fn shared_components_with_memory_enabled() -> ComponentRefs {
        let router_config = RouterConfig::builder()
            .memory_runtime_config(MemoryRuntimeConfig {
                ltm_enabled: true,
                ltm_store_enabled: true,
            })
            .build_unchecked();

        ComponentRefs::Shared(Arc::new(SharedComponents {
            client: reqwest::Client::new(),
            router_config: Arc::new(router_config),
        }))
    }

    #[test]
    fn for_responses_initializes_memory_context_from_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(
            MEMORY_LTM_STORE_ENABLED_HEADER,
            HeaderValue::from_static("enabled"),
        );

        let ctx = RequestContext::for_responses(
            Arc::new(ResponsesRequest::default()),
            Some(headers),
            None,
            shared_components_with_memory_enabled(),
        );

        assert!(ctx.memory_execution_context.store_ltm_active);
    }

    #[test]
    fn for_chat_initializes_memory_context_from_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(
            MEMORY_LTM_STORE_ENABLED_HEADER,
            HeaderValue::from_static("enabled"),
        );

        let ctx = RequestContext::for_chat(
            Arc::new(ChatCompletionRequest::default()),
            Some(headers),
            None,
            shared_components_with_memory_enabled(),
        );

        assert!(ctx.memory_execution_context.store_ltm_active);
    }

    #[tokio::test]
    async fn responses_streaming_context_preserves_memory_writer_and_execution_context() {
        let mut headers = HeaderMap::new();
        headers.insert(
            MEMORY_LTM_STORE_ENABLED_HEADER,
            HeaderValue::from_static("enabled"),
        );

        let shared = Arc::new(SharedComponents {
            client: reqwest::Client::new(),
            router_config: Arc::new(
                RouterConfig::builder()
                    .memory_runtime_config(MemoryRuntimeConfig {
                        ltm_enabled: true,
                        ltm_store_enabled: true,
                    })
                    .build_unchecked(),
            ),
        });

        let components = ComponentRefs::Responses(Arc::new(ResponsesComponents {
            shared,
            mcp_orchestrator: Arc::new(
                smg_mcp::McpOrchestrator::new(smg_mcp::McpConfig {
                    servers: vec![],
                    pool: Default::default(),
                    proxy: None,
                    warmup: vec![],
                    inventory: Default::default(),
                    policy: Default::default(),
                })
                .await
                .expect("mcp orchestrator should initialize"),
            ),
            response_storage: Arc::new(MemoryResponseStorage::new()),
            conversation_storage: Arc::new(MemoryConversationStorage::new()),
            conversation_item_storage: Arc::new(MemoryConversationItemStorage::new()),
            conversation_memory_writer: Some(Arc::new(DummyMemoryWriter)),
        }));

        let mut ctx = RequestContext::for_responses(
            Arc::new(ResponsesRequest::default()),
            Some(headers),
            None,
            components,
        );
        ctx.state.payload = Some(PayloadState {
            json: json!({"input":"hello"}),
            url: "http://worker.local/v1/responses".to_string(),
            previous_response_id: None,
        });

        let streaming = ctx
            .into_streaming_context()
            .expect("streaming context should be created");
        assert!(streaming.storage.conversation_memory_writer.is_some());
        assert!(streaming.storage.memory_execution_context.store_ltm_active);
    }
}
