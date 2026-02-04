//! Anthropic Models API implementation

use axum::{
    body::Body,
    extract::Request,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use tracing::{debug, warn};

use super::{utils::should_propagate_header, AnthropicRouter};
use crate::core::model_card::ProviderType;

const MAX_RESPONSE_SIZE: usize = 10 * 1024 * 1024;

pub async fn handle_list_models(router: &AnthropicRouter, req: Request<Body>) -> Response {
    debug!("Handling list models request");

    let headers = req.headers();

    let all_workers = router.context().worker_registry.get_workers_filtered(
        None, // model_id
        None, // worker_type
        None, // connection_mode
        None, // runtime_type
        true, // healthy_only
    );

    // SECURITY: In multi-provider setups, filter by Anthropic provider to prevent credential leakage
    // This ensures Anthropic credentials (X-API-Key, Authorization) are never sent to
    // non-Anthropic workers (e.g., OpenAI, xAI workers)
    let mut first_provider = None;
    let has_multiple_providers = all_workers.iter().any(|w| {
        if let Some(p) = w.default_provider() {
            match first_provider {
                None => {
                    first_provider = Some(p);
                    false
                }
                Some(first) => first != p,
            }
        } else {
            false
        }
    });

    let healthy_workers: Vec<_> = if has_multiple_providers {
        // Multi-provider setup: only use explicitly Anthropic workers
        all_workers
            .into_iter()
            .filter(|w| matches!(w.default_provider(), Some(ProviderType::Anthropic)))
            .collect()
    } else {
        // Single-provider or no-provider setup: use all workers
        all_workers
    };

    if healthy_workers.is_empty() {
        warn!("No healthy Anthropic workers available for /v1/models request");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "No healthy Anthropic workers available",
        )
            .into_response();
    }

    let worker = &healthy_workers[0];

    let url = format!("{}/v1/models", worker.url());

    debug!("Forwarding list models request to: {}", url);

    let mut req_builder = router.http_client().get(&url);

    for (key, value) in headers {
        if should_propagate_header(key.as_str()) {
            req_builder = req_builder.header(key, value);
        }
    }

    match req_builder.send().await {
        Ok(response) => {
            let status = response.status();

            // SECURITY: Check content-length header first to reject obviously oversized responses
            // without buffering any data
            if let Some(content_length) = response.content_length() {
                if content_length > MAX_RESPONSE_SIZE as u64 {
                    warn!(
                        "Response content-length too large: {} bytes (max {})",
                        content_length, MAX_RESPONSE_SIZE
                    );
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Response body exceeds maximum size",
                    )
                        .into_response();
                }
            }

            // SECURITY: Read body incrementally to avoid buffering unbounded data
            // when content-length is unknown (e.g., chunked transfer encoding)
            let mut stream = response.bytes_stream();
            let mut body_bytes = Vec::new();
            let mut total_size: usize = 0;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        total_size += chunk.len();
                        if total_size > MAX_RESPONSE_SIZE {
                            warn!(
                                "Response body too large: {} bytes (max {})",
                                total_size, MAX_RESPONSE_SIZE
                            );
                            return (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                "Response body exceeds maximum size",
                            )
                                .into_response();
                        }
                        body_bytes.extend_from_slice(&chunk);
                    }
                    Err(e) => {
                        warn!("Failed to read response body chunk: {}", e);
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to read response: {}", e),
                        )
                            .into_response();
                    }
                }
            }

            let body = String::from_utf8_lossy(&body_bytes).to_string();
            (status, body).into_response()
        }
        Err(e) => {
            warn!("Failed to forward list models request: {}", e);
            (
                StatusCode::BAD_GATEWAY,
                format!("Failed to forward request: {}", e),
            )
                .into_response()
        }
    }
}
