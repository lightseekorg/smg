//! Worker selection and HTTP transport layer for Anthropic router
//!
//! Contains the core primitives for selecting workers, building requests,
//! sending them, and processing responses. These functions are composed
//! by the streaming and non-streaming processors.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use openai_protocol::messages::{CreateMessageRequest, Message};
use tracing::{debug, error, info, warn};

use super::utils::{read_response_body_limited, should_propagate_header, ReadBodyResult};
use crate::{
    core::{ProviderType, Worker, WorkerRegistry},
    observability::metrics::{bool_to_static_str, metrics_labels, Metrics},
    routers::error,
};

/// Maximum error response body size to prevent DoS (1 MB)
const MAX_ERROR_RESPONSE_SIZE: usize = 1024 * 1024;

/// Select the best worker for the given model.
#[allow(clippy::result_large_err)]
pub(crate) fn select_worker(
    worker_registry: &WorkerRegistry,
    model_id: &str,
) -> Result<Arc<dyn Worker>, Response> {
    debug!(model = %model_id, "Selecting worker for request");

    match find_best_worker_for_model(worker_registry, model_id) {
        Some(w) => {
            debug!(
                model = %model_id,
                worker_url = %w.url(),
                worker_load = %w.load(),
                "Selected worker for request"
            );
            Ok(w)
        }
        None => {
            warn!(model = %model_id, "No healthy workers available for model");
            Err(error::service_unavailable(
                "no_workers",
                format!("No healthy workers available for model '{}'", model_id),
            ))
        }
    }
}

/// Build the target URL and propagated headers for a worker request.
pub(crate) fn build_request(
    worker: &dyn Worker,
    headers: Option<&HeaderMap>,
) -> (String, HeaderMap) {
    let url = format!("{}/v1/messages", worker.url());
    let mut propagated = HeaderMap::new();
    if let Some(input_headers) = headers {
        for (key, value) in input_headers {
            if should_propagate_header(key.as_str()) {
                propagated.insert(key.clone(), value.clone());
            }
        }
    }
    debug!(url = %url, header_count = %propagated.len(), "Request built");
    (url, propagated)
}

/// Send the HTTP request to the worker.
pub(crate) async fn send_request(
    http_client: &reqwest::Client,
    url: &str,
    headers: &HeaderMap,
    request: &CreateMessageRequest,
    timeout: Duration,
) -> Result<reqwest::Response, Response> {
    debug!(url = %url, "Sending request to worker");

    let mut builder = http_client.post(url).json(request).timeout(timeout);
    for (key, value) in headers {
        builder = builder.header(key, value);
    }

    match builder.send().await {
        Ok(response) => {
            debug!(url = %url, status = %response.status(), "Received response from worker");
            Ok(response)
        }
        Err(e) => {
            warn!(url = %url, error = %e, "Request to worker failed");

            if e.is_timeout() {
                Err(error::gateway_timeout(
                    "timeout",
                    format!("Request timeout: {}", e),
                ))
            } else if e.is_connect() {
                Err(error::bad_gateway(
                    "connection_failed",
                    format!("Connection failed: {}", e),
                ))
            } else {
                Err(error::bad_gateway(
                    "request_failed",
                    format!("Request failed: {}", e),
                ))
            }
        }
    }
}

/// Parse a successful non-streaming JSON response.
pub(crate) async fn parse_response(
    response: reqwest::Response,
    model_id: &str,
    start_time: Instant,
) -> Result<Message, Response> {
    match response.json::<Message>().await {
        Ok(message) => {
            debug!(
                model = %model_id,
                message_id = %message.id,
                "Successfully parsed response"
            );
            record_success_metrics(model_id, &message, start_time);
            info!(
                model = %model_id,
                message_id = %message.id,
                input_tokens = %message.usage.input_tokens,
                output_tokens = %message.usage.output_tokens,
                "Completed non-streaming request"
            );
            Ok(message)
        }
        Err(e) => {
            error!(model = %model_id, error = %e, "Failed to parse response");
            Metrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::CONNECTION_HTTP,
                model_id,
                "messages",
                start_time.elapsed(),
            );
            Metrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::CONNECTION_HTTP,
                model_id,
                "messages",
                "parse_error",
            );
            Err(error::bad_gateway(
                "parse_error",
                "Invalid response from backend",
            ))
        }
    }
}

/// Handle a non-success response: read body (limited) and record metrics.
pub(crate) async fn handle_error_response(
    response: reqwest::Response,
    model_id: &str,
    start_time: Instant,
) -> Response {
    let status = response.status();

    let body = match read_response_body_limited(response, MAX_ERROR_RESPONSE_SIZE).await {
        ReadBodyResult::Ok(b) if b.is_empty() => format!("Backend returned error: {}", status),
        ReadBodyResult::Ok(b) => b,
        ReadBodyResult::TooLarge => {
            warn!(
                model = %model_id,
                max_size = %MAX_ERROR_RESPONSE_SIZE,
                "Error response body too large"
            );
            format!("Backend returned error: {} (response too large)", status)
        }
        ReadBodyResult::Error(e) => {
            warn!(model = %model_id, error = %e, "Failed to read error response body");
            format!("Backend returned error: {}", status)
        }
    };

    warn!(
        model = %model_id,
        status = %status,
        body_preview = %body.chars().take(200).collect::<String>(),
        "Backend error"
    );

    Metrics::record_router_duration(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model_id,
        "messages",
        start_time.elapsed(),
    );
    Metrics::record_router_error(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model_id,
        "messages",
        metrics_labels::ERROR_BACKEND,
    );

    (status, body).into_response()
}

// ============================================================================
// Metrics Helpers
// ============================================================================

pub(crate) fn record_router_request(model_id: &str, streaming: bool) {
    Metrics::record_router_request(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model_id,
        "messages",
        bool_to_static_str(streaming),
    );
}

// ============================================================================
// Worker Filtering
// ============================================================================

/// Get healthy Anthropic workers from the registry.
///
/// SECURITY: In multi-provider setups, this filters by Anthropic provider to prevent
/// credential leakage. Anthropic credentials (X-API-Key, Authorization) are never sent
/// to non-Anthropic workers (e.g., OpenAI, xAI).
///
/// In single-provider setups (all workers have the same or no provider), returns all healthy workers.
/// In multi-provider setups, returns only healthy workers with `ProviderType::Anthropic`.
pub(crate) fn get_healthy_anthropic_workers(
    worker_registry: &WorkerRegistry,
) -> Vec<Arc<dyn Worker>> {
    let workers = worker_registry.get_workers_filtered(
        None, // model_id
        None, // worker_type
        None, // connection_mode
        None, // runtime_type
        true, // healthy_only
    );

    filter_by_anthropic_provider(workers)
}

/// Find the best worker for a given model.
///
/// Returns the least-loaded healthy Anthropic worker that supports the given model.
fn find_best_worker_for_model(
    worker_registry: &WorkerRegistry,
    model_id: &str,
) -> Option<Arc<dyn Worker>> {
    get_healthy_anthropic_workers(worker_registry)
        .into_iter()
        .filter(|w| w.supports_model(model_id))
        .min_by_key(|w| w.load())
}

/// Filter workers to only include Anthropic workers in multi-provider setups.
///
/// Treats `None` (no provider configured) as a distinct provider value so that
/// a mix of `Some(...)` and `None` workers is recognized as multi-provider,
/// preventing Anthropic credentials from leaking to untagged workers.
fn filter_by_anthropic_provider(workers: Vec<Arc<dyn Worker>>) -> Vec<Arc<dyn Worker>> {
    // Early-exit pattern to detect multiple providers without HashSet allocation.
    // Track Option<ProviderType> so None vs Some(...) counts as different.
    let mut first_provider: Option<Option<ProviderType>> = None;
    let has_multiple_providers = workers.iter().any(|w| {
        let provider = w.default_provider().cloned();
        match first_provider {
            None => {
                first_provider = Some(provider);
                false
            }
            Some(ref first) => *first != provider,
        }
    });

    if has_multiple_providers {
        // Multi-provider setup: only use explicitly Anthropic workers
        workers
            .into_iter()
            .filter(|w| matches!(w.default_provider(), Some(ProviderType::Anthropic)))
            .collect()
    } else {
        // Single-provider or no-provider setup: use all workers
        workers
    }
}

fn record_success_metrics(model_id: &str, message: &Message, start_time: Instant) {
    Metrics::record_router_duration(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model_id,
        "messages",
        start_time.elapsed(),
    );
    Metrics::record_router_tokens(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        model_id,
        "messages",
        metrics_labels::TOKEN_INPUT,
        message.usage.input_tokens as u64,
    );
    Metrics::record_router_tokens(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        model_id,
        "messages",
        metrics_labels::TOKEN_OUTPUT,
        message.usage.output_tokens as u64,
    );
}
