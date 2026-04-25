//! `OciDelegatedAuth` — outbound auth for OCI's OpenAI-compat endpoints.
//!
//! Mirrors Java's OBO branch:
//! - `AbstractContainerToolProcessor.java:122-125` — picks up an instance-
//!   principal-backed `S2SAuthenticationDetailsProvider`, builds the OCI
//!   OpenAI client, attaches OBO + project headers.
//! - `:128-146` — `resolveContainerRequestHeaders`, OBO case, returns
//!   `OpenAI-Project` + `opc-obo-token`.
//!
//! On `apply`, sets:
//! - `opc-obo-token: <obo>` — the inbound OBO token, forwarded as-is.
//! - `OpenAI-Project: <project_id>` — required by OCI's OpenAI-compat
//!   gateway; functions as compartment selector. See Java
//!   `RequestValidator.java:1824` ("`OpenAI-Project or opc-compartment-id
//!   must be provided`").
//! - `Authorization: Signature ...` — OCI HTTP Signature v1 over
//!   `(request-target)`, `host`, `date`, plus body headers for body methods.
//!   See [`crate::signer_adapter::sign_request`].
//!
//! ## Wire-shape contract (Java reference, line-by-line)
//!
//! Java (`AbstractContainerToolProcessor.java:122-146`):
//! ```java
//! // OBO branch
//! var provider = S2SAuthenticationDetailsProvider.useInstancePrincipals();   // ← we receive provider via constructor
//! var client = OciOpenAi.createClient(provider, compartmentId, endpoint);    // ← signer middleware injected
//! return new ImmutableMap.Builder<String, String>()
//!     .put(OPENAI_PROJECT_HEADER, projectId)                                 // ← we set "openai-project"
//!     .put(OPC_OBO_TOKEN, oboToken)                                          // ← we set "opc-obo-token"
//!     .build();
//! ```
//!
//! Rust mapping in this file:
//! 1. `apply()` receives a fully-constructed `http::Request<Bytes>`.
//! 2. We `insert("opc-obo-token", ...)` and `insert("openai-project", ...)`
//!    (Java step: header builder).
//! 3. We call `signer_adapter::sign_request(req, &auth_provider)` to invoke
//!    the OCI signer (Java step: middleware injected by `OciOpenAi.createClient`).
//!
//! ## R1 — opc-obo-token in signed-headers list
//!
//! See design doc §11 R1 + §13 wire-example note. Upstream
//! `oci-common::signer::get_required_headers` builds its `headers="..."` list
//! from a fixed set (`date`, `(request-target)`, `host` + body headers) and
//! does NOT include `opc-obo-token`. Whether OCI's gateway accepts an OBO
//! that wasn't signed-over is verified by integration test in CB-4. If it
//! must be signed, follow-up CB-1.A adds an additive entry-point with
//! `extra_headers_to_sign: &[&str]`.

use async_trait::async_trait;
use bytes::Bytes;
use http::Request;
use secrecy::{ExposeSecret, SecretString};
use std::sync::Arc;

use crate::oci::authentication_provider::AuthenticationProvider;
use crate::{AuthError, OutboundAuth};

/// OCI delegated outbound auth.
///
/// Holds an inbound OBO token (forwarded), the project id (= OCI compartment),
/// and the OCI auth provider used by the signer.
#[derive(Debug, Clone)]
pub struct OciDelegatedAuth {
    /// Inbound OBO token (already minted by upstream). Forwarded as
    /// `opc-obo-token`. Wrapped in `SecretString` so the value is redacted in
    /// any `Debug` / `Display` output (see R7).
    obo_token: SecretString,

    /// Required by OCI's OpenAI-compat container endpoints; maps to
    /// compartment.
    project_id: String,

    /// OCI auth provider (instance principals in v1; resource principals v2
    /// in v2). Boxed because the upstream signer takes a
    /// `&Box<dyn AuthenticationProvider>`.
    auth_provider: Arc<Box<dyn AuthenticationProvider + Send + Sync>>,
}

impl OciDelegatedAuth {
    pub fn new(
        obo_token: impl Into<SecretString>,
        project_id: impl Into<String>,
        auth_provider: Arc<Box<dyn AuthenticationProvider + Send + Sync>>,
    ) -> Self {
        Self {
            obo_token: obo_token.into(),
            project_id: project_id.into(),
            auth_provider,
        }
    }
}

#[async_trait]
impl OutboundAuth for OciDelegatedAuth {
    async fn apply(&self, req: &mut Request<Bytes>) -> Result<(), AuthError> {
        // 1. Add OCI-specific application headers BEFORE signing.
        //
        //    The upstream signer signs over a fixed list (`date`,
        //    `(request-target)`, `host` + body headers). It does NOT include
        //    `opc-obo-token` in that list — see R1 in the design doc. We
        //    still set the header here so it's transmitted; whether it must
        //    also be in the signed list is a CB-1.A follow-up.
        let obo_val: http::HeaderValue = self
            .obo_token
            .expose_secret()
            .parse()
            .map_err(|_| AuthError::InvalidCredential("obo_token contains invalid header chars"))?;
        req.headers_mut().insert("opc-obo-token", obo_val);

        let pid_val: http::HeaderValue = self.project_id.parse().map_err(|_| {
            AuthError::InvalidCredential("project_id contains invalid header chars")
        })?;
        req.headers_mut().insert("openai-project", pid_val);

        // 2. Sign with the OCI signer. This is the only place we call into
        //    the upstream-copied signer; the adapter
        //    `signer_adapter::sign_request` lives in our crate.
        crate::signer_adapter::sign_request(req, &self.auth_provider)?;
        Ok(())
    }
}
