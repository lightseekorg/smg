//! Bearer-token auth middleware backed by a precomputed SHA-256 hash.
//!
//! The hash is compared in constant time so the comparison cost does not
//! leak the configured key length.

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

#[derive(Clone)]
pub struct AuthConfig {
    /// Precomputed SHA-256 hash of the API key, used for constant-time comparison
    /// that doesn't leak key length via timing.
    api_key_hash: Option<[u8; 32]>,
}

impl AuthConfig {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key_hash: api_key.map(|k| Sha256::digest(k.as_bytes()).into()),
        }
    }
}

/// Middleware to validate Bearer token against configured API key.
/// Only active when router has an API key configured.
pub async fn auth_middleware(
    State(auth_config): State<AuthConfig>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if let Some(expected_hash) = &auth_config.api_key_hash {
        let token = request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|h| h.to_str().ok())
            .and_then(|h| h.strip_prefix("Bearer "));

        let authorized = token.is_some_and(|t| {
            Sha256::digest(t.as_bytes())
                .as_slice()
                .ct_eq(expected_hash)
                .unwrap_u8()
                == 1
        });
        if !authorized {
            return StatusCode::UNAUTHORIZED.into_response();
        }
    }

    next.run(request).await
}
