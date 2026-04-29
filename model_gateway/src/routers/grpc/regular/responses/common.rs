//! Surface-side helpers for regular Responses.
//!
//! - `ResponsesCallContext` — request-scoped knobs the handler bundles
//!   for both modes.

use axum::http;

use crate::middleware::TenantRequestMeta;

/// Per-request parameters for chat pipeline execution. Bundles values
/// that are always threaded together through the regular responses
/// call chain.
pub(super) struct ResponsesCallContext {
    pub headers: Option<http::HeaderMap>,
    pub model_id: String,
    pub response_id: Option<String>,
    pub tenant_request_meta: TenantRequestMeta,
}
