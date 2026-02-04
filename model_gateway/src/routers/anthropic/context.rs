//! Request context for Anthropic router pipeline
//!
//! This module provides a centralized request context that flows through
//! all pipeline stages, carrying request data, shared components, and
//! accumulated processing state.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use axum::http::HeaderMap;

use crate::{
    core::{Worker, WorkerRegistry},
    protocols::messages::CreateMessageRequest,
};

// ============================================================================
// Request Input
// ============================================================================

#[derive(Debug)]
pub(crate) struct RequestInput {
    pub request: Arc<CreateMessageRequest>,
    pub headers: Option<HeaderMap>,
    pub model_id: String,
}

impl RequestInput {
    pub fn new(
        request: Arc<CreateMessageRequest>,
        headers: Option<HeaderMap>,
        model_id: &str,
    ) -> Self {
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
    pub worker_registry: Arc<WorkerRegistry>,
    pub request_timeout: Duration,
}

impl SharedComponents {
    /// Create new shared components
    pub fn new(
        http_client: reqwest::Client,
        worker_registry: Arc<WorkerRegistry>,
        request_timeout: Duration,
    ) -> Self {
        Self {
            http_client,
            worker_registry,
            request_timeout,
        }
    }
}

impl std::fmt::Debug for SharedComponents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedComponents")
            .field("http_client", &"<reqwest::Client>")
            .field("worker_registry", &"<WorkerRegistry>")
            .field("request_timeout", &self.request_timeout)
            .finish()
    }
}

// ============================================================================
// Processing State
// ============================================================================

#[derive(Debug, Clone)]
pub(crate) struct ValidationOutput {
    pub is_streaming: bool,
    pub max_tokens: u32,
}

#[derive(Debug)]
pub(crate) struct HttpRequestState {
    pub url: String,
    pub headers: HeaderMap,
}

#[derive(Debug, Clone)]
pub(crate) struct DispatchMetadata {
    pub created: Instant,
    pub streaming: bool,
}

impl DispatchMetadata {
    /// Create new dispatch metadata
    pub fn new(streaming: bool) -> Self {
        Self {
            created: Instant::now(),
            streaming,
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct ResponseState {
    pub worker_response: Option<reqwest::Response>,
    pub status_code: Option<u16>,
}

#[derive(Debug, Default)]
pub(crate) struct ProcessingState {
    pub validation: Option<ValidationOutput>,
    pub worker: Option<Arc<dyn Worker>>,
    pub http_request: Option<HttpRequestState>,
    pub dispatch: Option<DispatchMetadata>,
    pub response: ResponseState,
}

// ============================================================================
// Request Context
// ============================================================================

/// Central context object that flows through all pipeline stages
pub(crate) struct RequestContext {
    pub input: RequestInput,
    pub state: ProcessingState,
}

impl RequestContext {
    /// Create a new request context
    pub fn new(
        request: Arc<CreateMessageRequest>,
        headers: Option<HeaderMap>,
        model_id: &str,
    ) -> Self {
        Self {
            input: RequestInput::new(request, headers, model_id),
            state: ProcessingState::default(),
        }
    }

    pub fn is_streaming(&self) -> bool {
        self.input.request.stream.unwrap_or(false)
    }

    pub fn start_time(&self) -> Option<Instant> {
        self.state.dispatch.as_ref().map(|d| d.created)
    }
}

impl std::fmt::Debug for RequestContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestContext")
            .field("model_id", &self.input.model_id)
            .field("is_streaming", &self.is_streaming())
            .field("has_worker", &self.state.worker.is_some())
            .field("has_dispatch", &self.state.dispatch.is_some())
            .finish()
    }
}
