//! Utility functions for the Gemini Interactions router.

use axum::http::{HeaderMap, HeaderValue};

/// Extract auth header for Gemini Interactions API requests.
///
/// Gemini uses `x-goog-api-key` instead of `Authorization`.
///
/// Precedence:
/// 1. `x-goog-api-key` header (native Gemini auth).
/// 2. `Authorization: Bearer <token>` header (strip prefix, use raw key).
/// 3. Worker's configured API key (fallback).
pub(crate) fn extract_gemini_auth_header(
    headers: Option<&HeaderMap>,
    worker_api_key: Option<&String>,
) -> Option<HeaderValue> {
    if let Some(h) = headers {
        // 1. Prefer x-goog-api-key
        if let Some(v) = h.get("x-goog-api-key").and_then(|v| {
            v.to_str()
                .ok()
                .filter(|s| !s.trim().is_empty())
                .map(|_| v.clone())
        }) {
            return Some(v);
        }

        // 2. Fall back to Authorization: Bearer <token>
        if let Some(token) = h
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| {
                s.split_once(' ')
                    .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
                    .map(|(_, token)| token.trim())
            })
            .filter(|t| !t.is_empty())
        {
            if let Ok(v) = HeaderValue::from_str(token) {
                return Some(v);
            }
        }
    }

    // 3. Worker's configured API key
    worker_api_key.and_then(|k| HeaderValue::from_str(k).ok())
}
