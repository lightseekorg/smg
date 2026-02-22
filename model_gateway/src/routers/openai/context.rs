//! Request context types for OpenAI router pipeline.

use std::sync::Arc;

use axum::http::HeaderMap;
use openai_protocol::{chat::ChatCompletionRequest, responses::ResponsesRequest};
use serde_json::Value;
use smg_data_connector::{ConversationItemStorage, ConversationStorage, ResponseStorage};
use smg_mcp::{McpOrchestrator, McpToolSession};

use super::provider::Provider;
use crate::core::Worker;

pub struct RequestContext {
    pub input: RequestInput,
    pub components: ComponentRefs,
    pub state: ProcessingState,
}

pub struct RequestInput {
    pub request_type: RequestType,
    pub headers: Option<HeaderMap>,
    pub model_id: Option<String>,
}

pub enum RequestType {
    Chat(Arc<ChatCompletionRequest>),
    Responses(Arc<ResponsesRequest>),
}

#[derive(Clone)]
pub struct SharedComponents {
    pub client: reqwest::Client,
}

pub struct ResponsesComponents {
    pub shared: SharedComponents,
    pub mcp_orchestrator: Arc<McpOrchestrator>,
    pub response_storage: Arc<dyn ResponseStorage>,
    pub conversation_storage: Arc<dyn ConversationStorage>,
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,
}

pub enum ComponentRefs {
    Shared(Arc<SharedComponents>),
    Responses(Arc<ResponsesComponents>),
}

impl ComponentRefs {
    pub fn client(&self) -> &reqwest::Client {
        match self {
            ComponentRefs::Shared(s) => &s.client,
            ComponentRefs::Responses(r) => &r.shared.client,
        }
    }

    pub fn mcp_orchestrator(&self) -> Option<&Arc<McpOrchestrator>> {
        match self {
            ComponentRefs::Shared(_) => None,
            ComponentRefs::Responses(r) => Some(&r.mcp_orchestrator),
        }
    }

    pub fn response_storage(&self) -> Option<&Arc<dyn ResponseStorage>> {
        match self {
            ComponentRefs::Shared(_) => None,
            ComponentRefs::Responses(r) => Some(&r.response_storage),
        }
    }

    pub fn conversation_storage(&self) -> Option<&Arc<dyn ConversationStorage>> {
        match self {
            ComponentRefs::Shared(_) => None,
            ComponentRefs::Responses(r) => Some(&r.conversation_storage),
        }
    }

    pub fn conversation_item_storage(&self) -> Option<&Arc<dyn ConversationItemStorage>> {
        match self {
            ComponentRefs::Shared(_) => None,
            ComponentRefs::Responses(r) => Some(&r.conversation_item_storage),
        }
    }
}

#[derive(Default)]
pub struct ProcessingState {
    pub worker: Option<WorkerSelection>,
    pub payload: Option<PayloadState>,
}

pub struct WorkerSelection {
    pub worker: Arc<dyn Worker>,
    pub provider: Arc<dyn Provider>,
}

pub struct PayloadState {
    pub json: Value,
    pub url: String,
    pub previous_response_id: Option<String>,
}

impl RequestContext {
    pub fn for_responses(
        request: Arc<ResponsesRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: ComponentRefs,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Responses(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    pub fn for_chat(
        request: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: ComponentRefs,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Chat(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        }
    }
}

impl RequestContext {
    pub fn responses_request(&self) -> Option<&ResponsesRequest> {
        match &self.input.request_type {
            RequestType::Responses(req) => Some(req.as_ref()),
            RequestType::Chat(_) => None,
        }
    }

    pub fn responses_request_arc(&self) -> Option<Arc<ResponsesRequest>> {
        match &self.input.request_type {
            RequestType::Responses(req) => Some(Arc::clone(req)),
            RequestType::Chat(_) => None,
        }
    }

    pub fn is_streaming(&self) -> bool {
        match &self.input.request_type {
            RequestType::Chat(req) => req.stream,
            RequestType::Responses(req) => req.stream.unwrap_or(false),
        }
    }

    pub fn headers(&self) -> Option<&HeaderMap> {
        self.input.headers.as_ref()
    }

    pub fn model_id(&self) -> Option<&str> {
        self.input.model_id.as_deref()
    }

    pub fn worker(&self) -> Option<&Arc<dyn Worker>> {
        self.state.worker.as_ref().map(|w| &w.worker)
    }

    pub fn provider(&self) -> Option<&dyn Provider> {
        self.state.worker.as_ref().map(|w| w.provider.as_ref())
    }

    pub fn payload(&self) -> Option<&PayloadState> {
        self.state.payload.as_ref()
    }

    pub fn take_payload(&mut self) -> Option<PayloadState> {
        self.state.payload.take()
    }
}

pub struct StorageHandles {
    pub response: Arc<dyn ResponseStorage>,
    pub conversation: Arc<dyn ConversationStorage>,
    pub conversation_item: Arc<dyn ConversationItemStorage>,
}

pub struct OwnedStreamingContext {
    pub url: String,
    pub payload: Value,
    pub original_body: ResponsesRequest,
    pub previous_response_id: Option<String>,
    pub storage: StorageHandles,
}

impl RequestContext {
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
            },
        })
    }
}

pub struct StreamingEventContext<'a> {
    pub original_request: &'a ResponsesRequest,
    pub previous_response_id: Option<&'a str>,
    pub session: Option<&'a McpToolSession>,
}

pub type StreamingRequest = OwnedStreamingContext;
