//! [`OpenAiCompatContainersClient`] — HTTP client for the OpenAI Containers
//! REST surface.
//!
//! Per design doc D5: a single client serves both the public OpenAI endpoint
//! and OCI's OpenAI-compat endpoint. The wire shape is identical; the only
//! difference is which [`OutboundAuth`] is injected and what the base URL is.
//!
//! # Auth boundary
//!
//! For each request we:
//!
//! 1. Build a fully-formed `http::Request<Bytes>` with method, URI, body, and
//!    `content-type: application/json` set.
//! 2. Call [`OutboundAuth::apply`] on the request — this is where vendor-auth
//!    signs (OCI) or attaches a bearer header (OpenAI).
//! 3. Translate the now-signed `http::Request` into a `reqwest::RequestBuilder`
//!    and send.
//!
//! Step (1) before (2) is load-bearing for OCI: the signer hashes the body
//! and content-type into `x-content-sha256` / `Authorization: Signature`. If
//! we set headers after `apply`, the signature would be over a stale view of
//! the request.
//!
//! # Error mapping
//!
//! | HTTP status     | `BackendError` variant                      |
//! |-----------------|---------------------------------------------|
//! | 2xx             | (success — return parsed body)              |
//! | 401, 403        | [`BackendError::Unauthorized`]              |
//! | 404             | [`BackendError::NotFound`]                  |
//! | 429             | [`BackendError::RateLimited`] + `retry-after` |
//! | other 4xx, 5xx  | [`BackendError::Backend`]                   |
//!
//! Mirrors Java `AbstractContainerToolProcessor.java:280-292` 4xx-mapping
//! intent (Java collapses all 4xx to `IllegalArgumentException` upstream;
//! we keep the status-class distinction so retry logic can branch on it).

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use http::{Method, Request};
use smg_vendor_auth::OutboundAuth;
use url::Url;

use crate::types::{Container, CreateContainerParams, ListOrder, ListQuery, Page};
use crate::{BackendError, ContainerBackend};

/// HTTP client for the OpenAI-shaped Containers endpoint.
///
/// Constructed with an [`OutboundAuth`] (selects API-key vs OCI-OBO mode), a
/// base URL (selects which vendor / region), and a shared `reqwest::Client`.
///
/// `Clone` is cheap — the auth and HTTP client are reference-counted.
#[derive(Debug, Clone)]
pub struct OpenAiCompatContainersClient {
    auth: Arc<dyn OutboundAuth>,
    base_url: Url,
    http: reqwest::Client,
}

impl OpenAiCompatContainersClient {
    /// Construct a new client.
    ///
    /// `base_url` must be the API root (e.g. `https://api.openai.com/` for
    /// public OpenAI, or the OCI region's OpenAI-compat root). The client
    /// appends `/v1/containers...` regardless of any path prefix on
    /// `base_url`. This matches the wire shape in design doc §13.
    pub fn new(auth: Arc<dyn OutboundAuth>, base_url: Url, http: reqwest::Client) -> Self {
        Self {
            auth,
            base_url,
            http,
        }
    }

    /// Build the absolute URL for a `/v1/containers{suffix}` endpoint.
    ///
    /// Both OpenAI public (`https://api.openai.com`) and OCI OpenAI-compat
    /// (e.g. `https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com`)
    /// expose the surface at the absolute path `/v1/containers...`, so
    /// overwriting the path on the base URL is correct.
    fn endpoint(&self, suffix: &str) -> Url {
        let mut u = self.base_url.clone();
        u.set_path(&format!("/v1/containers{suffix}"));
        // Clear any inherited query string from the base URL — query params
        // for our requests are added by the caller (e.g. list pagination).
        u.set_query(None);
        u
    }

    /// Sign and send an HTTP request, returning the raw response.
    ///
    /// The caller owns method, URL, and body. `content-type: application/json`
    /// is set here, before [`OutboundAuth::apply`], so the OCI signer hashes
    /// the correct value.
    async fn send_signed(
        &self,
        method: Method,
        url: Url,
        body: Bytes,
    ) -> Result<reqwest::Response, BackendError> {
        let mut req = Request::builder()
            .method(method.clone())
            .uri(url.as_str())
            .header("content-type", "application/json")
            .body(body)
            .map_err(|e| BackendError::Backend {
                status: 0,
                message: format!("invalid http request: {e}"),
            })?;

        // Auth boundary — vendor_auth signs (OCI) or attaches bearer (OpenAI).
        self.auth.apply(&mut req).await?;

        // Translate http::Request<Bytes> -> reqwest::RequestBuilder.
        // Note: `reqwest::Method` is a re-export of `http::Method`, so the
        // `parts.method` value is reused as-is.
        let (parts, body) = req.into_parts();
        let mut rb = self.http.request(parts.method, url.as_str());
        for (name, value) in &parts.headers {
            rb = rb.header(name.as_str(), value.as_bytes());
        }
        rb = rb.body(body);
        let resp = rb.send().await?;
        Ok(resp)
    }

    /// Decode a JSON response body or map a non-2xx status to a [`BackendError`].
    async fn handle<T: serde::de::DeserializeOwned>(
        resp: reqwest::Response,
    ) -> Result<T, BackendError> {
        let status = resp.status();
        if status.is_success() {
            let bytes = resp.bytes().await?;
            return Ok(serde_json::from_slice::<T>(&bytes)?);
        }

        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());

        let bytes = resp.bytes().await.unwrap_or_default();
        let body = String::from_utf8_lossy(&bytes).to_string();

        Err(match status.as_u16() {
            401 | 403 => BackendError::Unauthorized,
            404 => BackendError::NotFound(body),
            429 => BackendError::RateLimited {
                retry_after_secs: retry_after,
            },
            s => BackendError::Backend { status: s, message: body },
        })
    }
}

#[async_trait]
impl ContainerBackend for OpenAiCompatContainersClient {
    async fn create(&self, params: CreateContainerParams) -> Result<Container, BackendError> {
        let body = serde_json::to_vec(&params)?;
        let resp = self
            .send_signed(Method::POST, self.endpoint(""), Bytes::from(body))
            .await?;
        Self::handle(resp).await
    }

    async fn retrieve(&self, id: &str) -> Result<Container, BackendError> {
        let resp = self
            .send_signed(
                Method::GET,
                self.endpoint(&format!("/{id}")),
                Bytes::new(),
            )
            .await?;
        Self::handle(resp).await
    }

    async fn delete(&self, id: &str) -> Result<(), BackendError> {
        let resp = self
            .send_signed(
                Method::DELETE,
                self.endpoint(&format!("/{id}")),
                Bytes::new(),
            )
            .await?;
        if resp.status().is_success() {
            return Ok(());
        }
        // Drain body and route through the same error mapper.
        let err = Self::handle::<serde_json::Value>(resp)
            .await
            .err()
            .unwrap_or(BackendError::Backend {
                status: 0,
                message: String::from("delete failed without status"),
            });
        Err(err)
    }

    async fn list(&self, q: ListQuery) -> Result<Page<Container>, BackendError> {
        let mut url = self.endpoint("");
        {
            let mut qp = url.query_pairs_mut();
            if let Some(l) = q.limit {
                qp.append_pair("limit", &l.to_string());
            }
            if let Some(a) = &q.after {
                qp.append_pair("after", a);
            }
            if let Some(b) = &q.before {
                qp.append_pair("before", b);
            }
            if let Some(o) = q.order {
                qp.append_pair(
                    "order",
                    match o {
                        ListOrder::Asc => "asc",
                        ListOrder::Desc => "desc",
                    },
                );
            }
        }
        let resp = self.send_signed(Method::GET, url, Bytes::new()).await?;
        Self::handle(resp).await
    }
}
