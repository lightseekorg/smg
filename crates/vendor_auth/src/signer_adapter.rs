//! Adapter from `http::Request<Bytes>` to the OCI HTTP signer.
//!
//! The OCI signer (`crate::oci::signer::get_required_headers`) is a pure
//! function whose public types live in the `reqwest` crate
//! (`reqwest::header::HeaderMap`, `reqwest::Method`). SMG's outbound HTTP
//! surface is `http::Request<Bytes>`. This adapter translates between the two
//! at the boundary so the rest of SMG never sees `reqwest::header`.
//!
//! ## OCI HTTP Signing — wire shape
//!
//! Reference: <https://docs.oracle.com/en-us/iaas/Content/API/Concepts/signingrequests.htm>
//!
//! The signer adds these headers to the request:
//!
//! - `date` — RFC 2822 timestamp (added if not present).
//! - `host` — derived from URL (added if not present).
//! - For body methods (POST/PUT/PATCH) — `content-type`, `content-length`,
//!   `x-content-sha256` (base64 of SHA-256 of the request body).
//! - `Authorization: Signature version="1",keyId="<id>",algorithm="rsa-sha256",headers="<list>",signature="<base64>"`
//!
//! For instance-principal callers the `keyId` is `ST$<security-token>`. See
//! `crate::oci::instance_principals_provider::InstancePrincipalAuthProvider::key_id`
//! (verbatim copy of upstream — OCI SDK convention since 2023).
//!
//! ## Headers signed (default)
//!
//! Per `crate::oci::signer::get_required_headers_from_key_and_id` (signer.rs:48-77):
//! `date`, `(request-target)`, `host` always; `content-type`, `content-length`,
//! `x-content-sha256` for body methods. **Note**: `opc-obo-token` is NOT in the
//! upstream default list; if OCI's gateway requires it, see follow-up CB-1.A
//! (R1 in design doc).

use crate::AuthError;
use crate::oci::authentication_provider::AuthenticationProvider;
use crate::oci::signer::get_required_headers;
use bytes::Bytes;
use http::Request;
use openssl::pkey::Private;
use openssl::rsa::Rsa;
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;
use url::Url;

/// Sign an outbound HTTP request using an OCI authentication provider.
///
/// Mirrors Java's `OciOpenAi.createClient(provider, …)` injection plus the
/// OCI signer middleware in `oci-java-sdk` (see Java reference
/// `AbstractContainerToolProcessor.java:122-125`).
///
/// Side effects:
/// - Adds `date` if absent.
/// - Adds `host` if absent.
/// - For body methods (POST/PUT/PATCH): adds `content-type`,
///   `content-length`, `x-content-sha256` if absent.
/// - Always overwrites `authorization` with the OCI Signature v1 header.
///
/// The function does NOT consume `req`; on success the headers are mutated
/// in place.
pub fn sign_request(
    req: &mut Request<Bytes>,
    auth_provider: &Arc<Box<dyn AuthenticationProvider + Send + Sync>>,
) -> Result<(), AuthError> {
    // 1. Translate Method, URL, body, headers
    let method = method_to_reqwest(req.method())?;
    let url = Url::parse(&req.uri().to_string())
        .map_err(|e| AuthError::Signer(anyhow::anyhow!("invalid uri: {e}")))?;
    let body = std::str::from_utf8(req.body())
        .map_err(|e| AuthError::Signer(anyhow::anyhow!("non-utf8 body: {e}")))?;
    let original_headers = http_to_reqwest_headers(req.headers())?;

    // 2. Pull query params out of the URL (signer wants them as a HashMap)
    let query_params: HashMap<String, String> = url
        .query_pairs()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();

    // 3. Wrap the Send+Sync provider in a forwarding adapter so we can hand
    //    the upstream signer the `&Box<dyn AuthenticationProvider>` it
    //    expects (no Send+Sync bound) without unsafe trait-object widening.
    //    The forwarding wrapper's lifetime is bounded by this stack frame.
    let forwarder: Box<dyn AuthenticationProvider> =
        Box::new(SyncProviderForwarder { inner: auth_provider.clone() });

    let signed_headers: reqwest::header::HeaderMap = get_required_headers(
        method,
        body,
        original_headers,
        url,
        &forwarder,
        query_params,
        /* exclude_body */ false,
    );

    // 4. Write back the headers added/overwritten by the signer. Use `insert`
    //    so any existing values for the same name are replaced (e.g. the
    //    `authorization` header).
    for (name, value) in &signed_headers {
        let h_name = http::HeaderName::from_bytes(name.as_str().as_bytes())
            .map_err(|e| AuthError::Signer(anyhow::anyhow!("invalid signed-header name: {e}")))?;
        let h_val = http::HeaderValue::from_bytes(value.as_bytes()).map_err(|e| {
            AuthError::Signer(anyhow::anyhow!("invalid signed-header value: {e}"))
        })?;
        req.headers_mut().insert(h_name, h_val);
    }
    Ok(())
}

fn method_to_reqwest(m: &http::Method) -> Result<reqwest::Method, AuthError> {
    reqwest::Method::from_bytes(m.as_str().as_bytes())
        .map_err(|e| AuthError::Signer(anyhow::anyhow!("invalid method: {e}")))
}

fn http_to_reqwest_headers(
    h: &http::HeaderMap,
) -> Result<reqwest::header::HeaderMap, AuthError> {
    let mut out = reqwest::header::HeaderMap::new();
    for (n, v) in h {
        let name = reqwest::header::HeaderName::from_bytes(n.as_str().as_bytes()).map_err(|e| {
            AuthError::Signer(anyhow::anyhow!("invalid outbound header name: {e}"))
        })?;
        let val = reqwest::header::HeaderValue::from_bytes(v.as_bytes()).map_err(|e| {
            AuthError::Signer(anyhow::anyhow!("invalid outbound header value: {e}"))
        })?;
        out.insert(name, val);
    }
    Ok(out)
}

/// Forwarding wrapper that satisfies `dyn AuthenticationProvider` by
/// delegating every method to a stored `Arc<Box<dyn AuthenticationProvider +
/// Send + Sync>>`. Used to bridge the auto-trait bound mismatch between SMG's
/// `Send + Sync` provider type and the upstream signer's bound-less trait
/// object — without an unsafe transmute.
#[derive(Debug, Clone)]
struct SyncProviderForwarder {
    inner: Arc<Box<dyn AuthenticationProvider + Send + Sync>>,
}

impl AuthenticationProvider for SyncProviderForwarder {
    fn tenancy_id(&self) -> Result<String, Box<dyn Error>> {
        self.inner.tenancy_id()
    }
    fn user_id(&self) -> Result<String, Box<dyn Error>> {
        self.inner.user_id()
    }
    fn fingerprint(&self) -> Result<String, Box<dyn Error>> {
        self.inner.fingerprint()
    }
    fn private_key(&self) -> Result<Rsa<Private>, Box<dyn Error>> {
        self.inner.private_key()
    }
    fn key_id(&self) -> Result<String, Box<dyn Error>> {
        self.inner.key_id()
    }
    fn region_id(&self) -> Result<Option<String>, Box<dyn Error>> {
        self.inner.region_id()
    }
}
