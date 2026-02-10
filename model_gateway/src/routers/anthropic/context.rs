//! Request context for Anthropic router pipeline
//!
//! This module provides a centralized request context that flows through
//! all pipeline stages, carrying request data, shared components, and
//! accumulated processing state.

use std::{
    sync::{Arc, RwLock as StdRwLock},
    time::{Duration, Instant},
};

use axum::http::HeaderMap;

use super::pipeline::MessagesPipeline;
use crate::{
    core::Worker,
    mcp::McpOrchestrator,
    protocols::messages::{CreateMessageRequest, Message},
};

// ============================================================================
// Request Input
// ============================================================================

#[derive(Debug)]
pub(crate) struct RequestInput {
    pub request: CreateMessageRequest,
    pub headers: Option<HeaderMap>,
    pub model_id: String,
}

impl RequestInput {
    pub fn new(request: CreateMessageRequest, headers: Option<HeaderMap>, model_id: &str) -> Self {
        Self {
            request,
            headers,
            model_id: model_id.to_string(),
        }
    }
}

// ============================================================================
// Shared Components
// ============================================================================

pub(crate) struct SharedComponents {
    pub http_client: reqwest::Client,
    pub request_timeout: Duration,
}

impl std::fmt::Debug for SharedComponents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedComponents")
            .field("http_client", &"<reqwest::Client>")
            .field("request_timeout", &self.request_timeout)
            .finish()
    }
}

// ============================================================================
// Messages Context
// ============================================================================

#[derive(Clone)]
pub(crate) struct MessagesContext {
    pub pipeline: Arc<MessagesPipeline>,
    pub mcp_orchestrator: Arc<McpOrchestrator>,
    /// MCP servers `(label, server_key)` connected for this request.
    pub requested_servers: Arc<StdRwLock<Vec<(String, String)>>>,
}

impl MessagesContext {
    pub fn new(pipeline: Arc<MessagesPipeline>, mcp_orchestrator: Arc<McpOrchestrator>) -> Self {
        Self {
            pipeline,
            mcp_orchestrator,
            requested_servers: Arc::new(StdRwLock::new(Vec::new())),
        }
    }
}

// ============================================================================
// Processing State
// ============================================================================

#[derive(Debug)]
pub(crate) struct HttpRequestState {
    pub url: String,
    pub headers: HeaderMap,
}

#[derive(Debug, Default)]
pub(crate) struct ResponseState {
    pub worker_response: Option<reqwest::Response>,
    pub status_code: Option<u16>,
}

#[derive(Debug, Default)]
pub(crate) struct ProcessingState {
    pub worker: Option<Arc<dyn Worker>>,
    pub http_request: Option<HttpRequestState>,
    pub response: ResponseState,
    pub parsed_message: Option<Message>,
}

// ============================================================================
// Request Context
// ============================================================================

/// Central context object that flows through all pipeline stages
pub(crate) struct RequestContext {
    pub input: RequestInput,
    pub state: ProcessingState,
    /// Timestamp when request processing started (for metrics)
    pub start_time: Instant,
}

impl RequestContext {
    /// Create a new request context
    pub fn new(request: CreateMessageRequest, headers: Option<HeaderMap>, model_id: &str) -> Self {
        Self {
            input: RequestInput::new(request, headers, model_id),
            state: ProcessingState::default(),
            start_time: Instant::now(),
        }
    }

    pub fn is_streaming(&self) -> bool {
        self.input.request.stream.unwrap_or(false)
    }
}

impl std::fmt::Debug for RequestContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestContext")
            .field("model_id", &self.input.model_id)
            .field("is_streaming", &self.is_streaming())
            .field("has_worker", &self.state.worker.is_some())
            .finish()
    }
}
