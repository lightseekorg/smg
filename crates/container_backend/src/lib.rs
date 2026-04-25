//! `smg-container-backend` — vendor-agnostic container client for SMG.
//!
//! This crate exposes the [`ContainerBackend`] trait and a single concrete
//! v1 impl, [`OpenAiCompatContainersClient`], that serves both:
//!
//! 1. The public OpenAI `/v1/containers` endpoint, and
//! 2. OCI's OpenAI-compat container endpoint
//!
//! per design doc decision **D5** — the wire surface is identical; the only
//! difference is which [`OutboundAuth`](smg_vendor_auth::OutboundAuth) is
//! injected (and, for OCI, the base URL).
//!
//! See `.claude/plans/container-backend-design.md` §7 for the full contract.
//!
//! # Layout
//!
//! - [`types`] — wire types ([`Container`], [`MemoryLimit`], …).
//! - [`openai_compat`] — the [`OpenAiCompatContainersClient`] HTTP client.
//! - [`mock`] — an in-memory [`MockBackend`] for hermetic tests
//!   (used by SMG e2e in CB-3 / CB-4).

pub mod mock;
pub mod openai_compat;
pub mod types;

pub use mock::MockBackend;
pub use openai_compat::OpenAiCompatContainersClient;
pub use types::{
    Container, ContainerStatus, CreateContainerParams, DomainSecret, ExpiresAfter, ListOrder,
    ListQuery, MemoryLimit, NetworkPolicy, Page,
};

use async_trait::async_trait;

/// Errors returned by [`ContainerBackend`] implementations.
///
/// HTTP-level mapping (see [`OpenAiCompatContainersClient`]):
/// - `401` or `403` → [`BackendError::Unauthorized`]
/// - `404` → [`BackendError::NotFound`]
/// - `429` → [`BackendError::RateLimited`] (parses `retry-after` if present)
/// - other non-2xx → [`BackendError::Backend`]
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// The backend returned 404 — the container id (or list cursor) does not
    /// exist or is no longer addressable. Carries the response body for
    /// caller-side logging.
    #[error("not found: {0}")]
    NotFound(String),

    /// The backend returned 401 or 403 — the inbound credential was rejected.
    /// Both codes collapse here; callers don't need to distinguish bad-auth
    /// from forbidden-resource at this layer.
    #[error("unauthorized")]
    Unauthorized,

    /// The backend returned 429. `retry_after_secs` is parsed from the
    /// `retry-after` header when present.
    #[error("rate limited (retry-after: {retry_after_secs:?}s)")]
    RateLimited {
        /// Number of seconds the client should wait before retrying.
        retry_after_secs: Option<u64>,
    },

    /// Any other non-2xx response. `status` is the HTTP status code; `message`
    /// is the response body (lossily UTF-8 decoded).
    #[error("backend error {status}: {message}")]
    Backend { status: u16, message: String },

    /// `reqwest`-level transport error (DNS, TCP, TLS, body read).
    #[error("transport: {0}")]
    Transport(#[from] reqwest::Error),

    /// The injected [`OutboundAuth`](smg_vendor_auth::OutboundAuth) failed to
    /// apply credentials to the request.
    #[error("auth: {0}")]
    Auth(#[from] smg_vendor_auth::AuthError),

    /// Request body serialization or response body deserialization failed.
    #[error("serialization: {0}")]
    Serialization(#[from] serde_json::Error),

    /// The HTTP request did not complete before the configured deadline.
    /// (Reserved — `reqwest::Error` already covers timeouts in v1; this
    /// variant exists so callers can distinguish a deadline violation from a
    /// transport failure once we wire an external deadline.)
    #[error("timeout")]
    Timeout,
}

/// A vendor-agnostic container backend.
///
/// v1 impls:
/// - [`OpenAiCompatContainersClient`] — real HTTP, both OpenAI public and OCI.
/// - [`MockBackend`] — in-memory, for hermetic tests.
///
/// Future impls (GCP / Azure / AWS) slot under the same trait.
#[async_trait]
pub trait ContainerBackend: Send + Sync + std::fmt::Debug {
    /// Create a new container.
    async fn create(&self, params: CreateContainerParams) -> Result<Container, BackendError>;
    /// Retrieve the current state of an existing container.
    async fn retrieve(&self, id: &str) -> Result<Container, BackendError>;
    /// Delete a container. Idempotent at the trait level (callers may treat
    /// `404` as success if they wish — the trait returns `NotFound`).
    async fn delete(&self, id: &str) -> Result<(), BackendError>;
    /// List containers. Pagination follows the OpenAI list-envelope shape
    /// ([`Page`]).
    async fn list(&self, q: ListQuery) -> Result<Page<Container>, BackendError>;
}
