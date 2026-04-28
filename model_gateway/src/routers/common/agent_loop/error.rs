//! Loop-level error type.
//!
//! Surface adapters and the driver speak this enum so the loop core does
//! not need to import `axum::Response`. Each surface adapter converts to
//! its own HTTP error shape at the outermost boundary.

use axum::response::Response;

use crate::routers::error;

#[derive(Debug)]
pub(crate) enum AgentLoopError {
    /// The request itself is malformed — invalid stitched continuation,
    /// missing approval context, etc. Maps to 400.
    InvalidRequest(String),
    /// Upstream model worker rejected the call or returned a bad response.
    /// Maps to 502 / propagates the upstream HTTP status when available.
    Upstream(String),
    /// Gateway-internal failure (serialization, missing state, etc.).
    /// Maps to 500.
    Internal(String),
    /// Pre-built HTTP response from a deeper layer (pipeline error already
    /// carrying a proper status). The driver returns it verbatim instead
    /// of re-classifying. Boxed because `Response` is much larger than
    /// the other variants — keeps `Result<_, AgentLoopError>` compact.
    Response(Box<Response>),
}

impl AgentLoopError {
    /// Convert to an axum `Response` for the outermost handler. Pre-built
    /// responses pass through unchanged.
    pub(crate) fn into_response(self) -> Response {
        match self {
            AgentLoopError::InvalidRequest(msg) => error::bad_request("invalid_request", msg),
            AgentLoopError::Upstream(msg) => error::bad_gateway("upstream_error", msg),
            AgentLoopError::Internal(msg) => error::internal_error("internal_error", msg),
            AgentLoopError::Response(resp) => *resp,
        }
    }

    /// Human-readable error message suitable for surfacing to the
    /// client (e.g. inside an SSE `error` event payload). For the
    /// pre-built `Response` variant we cannot peek at the JSON body
    /// without consuming it, so we fall back to the status code's
    /// reason phrase — adapters that need the body should handle
    /// `Response` separately before reaching this method.
    pub(crate) fn message(&self) -> String {
        match self {
            AgentLoopError::InvalidRequest(msg)
            | AgentLoopError::Upstream(msg)
            | AgentLoopError::Internal(msg) => msg.clone(),
            AgentLoopError::Response(resp) => resp
                .status()
                .canonical_reason()
                .unwrap_or("error")
                .to_string(),
        }
    }
}

impl From<Response> for AgentLoopError {
    fn from(resp: Response) -> Self {
        AgentLoopError::Response(Box::new(resp))
    }
}
