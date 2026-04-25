//! `smg-vendor-auth` — outbound (vendor) authentication for SMG.
//!
//! This crate is the OUTBOUND counterpart to the existing inbound `smg-auth`
//! crate. It exposes an [`OutboundAuth`] trait with two v1 implementations:
//!
//! - [`OpenAiApiKeyAuth`] — Bearer-token + optional `OpenAI-Project` header
//!   for OpenAI public and any OpenAI-compat endpoint.
//! - [`OciDelegatedAuth`] — `opc-obo-token` + `OpenAI-Project` header plus an
//!   OCI HTTP Signature v1 over the request, for OCI's OpenAI-compat
//!   container endpoints.
//!
//! See `.claude/plans/container-backend-design.md` for the design contract.
//!
//! # Locked decisions (per design doc § 2)
//!
//! - **D7**: crate name is `vendor_auth` (disambiguates against the existing
//!   inbound `smg-auth`).
//! - **D8**: OCI source under `oci/` is copied verbatim from `oci-rust-sdk`
//!   under UPL-1.0; see `src/oci/NOTICE.md`.
//! - **D10**: v1 ships instance-principals only; resource-principal v2
//!   support is deferred.

pub mod oci;
pub mod oci_delegated;
pub mod oci_provider_factory;
pub mod openai;
pub mod signer_adapter;

use async_trait::async_trait;
use bytes::Bytes;
use http::Request;

pub use oci_delegated::OciDelegatedAuth;
pub use openai::OpenAiApiKeyAuth;

/// Errors returned by [`OutboundAuth::apply`] implementations.
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    /// OCI HTTP signing failed (translation, signing, or header conversion).
    #[error("oci signer: {0}")]
    Signer(#[from] anyhow::Error),
    /// The credential was malformed (e.g. empty token).
    #[error("invalid credential: {0}")]
    InvalidCredential(&'static str),
    /// A header value could not be constructed (typically because the input
    /// contained characters disallowed in HTTP header values).
    #[error("invalid header value: {0}")]
    InvalidHeader(#[from] http::header::InvalidHeaderValue),
}

/// Apply outbound credentials to a fully-constructed HTTP request.
///
/// Implementations may add headers, sign the request, or both.
///
/// # Contract
///
/// - The request URI, method, and body must be set before `apply` is called.
/// - Implementations may read `req.body()` for hashing (the OCI signer needs
///   SHA-256).
/// - Not idempotent: calling twice on the same request adds duplicate
///   headers; callers must call exactly once per outbound request.
#[async_trait]
pub trait OutboundAuth: Send + Sync + std::fmt::Debug {
    async fn apply(&self, req: &mut Request<Bytes>) -> Result<(), AuthError>;
}
