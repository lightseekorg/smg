//! Storage request-context middleware.
//!
//! Maps configured request headers onto a [`StorageRequestContext`] so the
//! storage layer can read tenant/user/etc. fields without re-parsing headers.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use smg_data_connector::{with_request_context, RequestContext as StorageRequestContext};

use std::collections::HashMap;

use crate::server::AppState;

fn extract_header_str(headers: &http::HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

/// Build a storage request-context from the configured header→key mapping.
///
/// Exposed (`pub(crate)`) so non-middleware entry points — e.g. the WebSocket
/// Responses upgrade, which runs its handler outside the middleware's
/// task-local scope — can re-establish the same scope from the request headers.
pub(crate) fn build_storage_request_context(
    headers_config: &HashMap<String, String>,
    headers: &http::HeaderMap,
) -> Option<StorageRequestContext> {
    let mut ctx = StorageRequestContext::new();

    for (header_name, context_key) in headers_config {
        let header_name = header_name.trim();
        let context_key = context_key.trim();

        if header_name.is_empty() || context_key.is_empty() {
            continue;
        }

        if let Some(value) = extract_header_str(headers, header_name) {
            ctx.set(context_key, value);
        }
    }

    (!ctx.data().is_empty()).then_some(ctx)
}

pub async fn storage_context_middleware(
    State(state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if state
        .context
        .router_config
        .storage_context_headers
        .is_empty()
    {
        return next.run(request).await;
    }

    match build_storage_request_context(
        &state.context.router_config.storage_context_headers,
        request.headers(),
    ) {
        Some(ctx) => with_request_context(ctx, next.run(request)).await,
        None => next.run(request).await,
    }
}

#[cfg(test)]
mod tests {
    use axum::http::{HeaderMap, HeaderValue};

    use super::*;

    #[test]
    fn build_storage_request_context_maps_configured_headers() {
        let headers_config = HashMap::from([
            ("x-tenant-id".to_string(), "tenant_id".to_string()),
            ("x-user-id".to_string(), "user_id".to_string()),
        ]);

        let mut headers = HeaderMap::new();
        headers.insert("x-tenant-id", HeaderValue::from_static("tenant-abc"));
        headers.insert("x-user-id", HeaderValue::from_static("user-123"));

        let ctx = build_storage_request_context(&headers_config, &headers).unwrap();

        assert_eq!(ctx.get("tenant_id"), Some("tenant-abc"));
        assert_eq!(ctx.get("user_id"), Some("user-123"));
    }

    #[test]
    fn build_storage_request_context_ignores_empty_entries_and_missing_headers() {
        let headers_config = HashMap::from([
            (" ".to_string(), "tenant_id".to_string()),
            ("x-empty-key".to_string(), " ".to_string()),
            ("x-present".to_string(), "present_key".to_string()),
        ]);

        let mut headers = HeaderMap::new();
        headers.insert("x-present", HeaderValue::from_static("  keep-me  "));

        let ctx = build_storage_request_context(&headers_config, &headers).unwrap();

        assert_eq!(ctx.get("present_key"), Some("keep-me"));
        assert_eq!(ctx.data().len(), 1);
    }
}
