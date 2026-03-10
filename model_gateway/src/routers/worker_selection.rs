//! Shared worker selection for all routers.
//!
//! Public APIs:
//! - [`select_worker`] — full orchestration: filter → least-loaded → refresh-on-miss → error
//! - [`get_candidates`] — filter workers by [`WorkerQuery`] criteria (for model listing)
//! - [`refresh_external_models`] — proactively refresh model lists from external workers

use std::sync::Arc;

use axum::{http::HeaderValue, response::Response};
use futures_util::future::join_all;

use crate::{
    core::{
        ConnectionMode, ModelCard, ProviderType, RuntimeType, Worker, WorkerRegistry, WorkerType,
    },
    routers::{error, header_utils::apply_provider_headers},
};

// ============================================================================
// Core Types
// ============================================================================

/// Criteria for filtering workers from the registry.
///
/// All fields are optional — `None` means "no filter on this dimension."
/// Implements `Default` so callers can use struct update syntax:
///
/// ```ignore
/// let query = WorkerQuery {
///     runtime_type: Some(RuntimeType::External),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Default)]
pub struct WorkerQuery<'a> {
    /// Model ID for indexed O(1) lookup. `None` = any model.
    pub model_id: Option<&'a str>,
    /// Filter by worker type (Regular, Prefill, Decode).
    pub worker_type: Option<WorkerType>,
    /// Filter by connection mode (Http, Grpc).
    pub connection_mode: Option<ConnectionMode>,
    /// Filter by runtime type (External, Sglang, Vllm, Trtllm).
    pub runtime_type: Option<RuntimeType>,
    /// Provider-based security filtering for multi-provider setups.
    /// When set, prevents credentials from leaking to workers of a different provider.
    pub provider: Option<ProviderType>,
}

/// The result of worker selection.
pub enum SelectedWorker {
    /// Single worker selected.
    Single(Arc<dyn Worker>),
    /// Prefill-Decode worker pair for disaggregated serving.
    PrefillDecode {
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        runtime_type: RuntimeType,
    },
}

// ============================================================================
// Public API
// ============================================================================

/// Get available worker candidates matching the query.
///
/// Combines three filtering stages:
/// 1. **Registry lookup** — indexed by model/type/connection/runtime (O(1) for model)
/// 2. **Availability** — only workers where `is_available()` (healthy + circuit breaker)
/// 3. **Provider security** — in multi-provider setups, restricts to matching provider
pub fn get_candidates(registry: &WorkerRegistry, query: &WorkerQuery) -> Vec<Arc<dyn Worker>> {
    let workers = registry.get_workers_filtered(
        query.model_id,
        query.worker_type,
        query.connection_mode,
        query.runtime_type,
        false, // we filter availability ourselves for consistent behavior
    );

    let candidates: Vec<_> = workers.into_iter().filter(|w| w.is_available()).collect();

    match &query.provider {
        Some(provider) => filter_by_provider(candidates, provider.clone()),
        None => candidates,
    }
}

/// Select the best worker for a model with refresh-on-miss.
///
/// Full orchestration used by all external API routers (OpenAI, Anthropic,
/// Gemini, Realtime):
///
/// 1. Filter candidates via [`get_candidates`] with the given query.
/// 2. Pick the least-loaded worker that supports `model_id`.
/// 3. On miss, refresh external model lists by calling `/v1/models` on all
///    external workers, then retry.
/// 4. Return an error distinguishing "model not found" from "all workers
///    circuit-broken".
pub async fn select_worker(
    registry: &WorkerRegistry,
    client: &reqwest::Client,
    query: &WorkerQuery<'_>,
    model_id: &str,
    auth_header: Option<&HeaderValue>,
) -> Result<Arc<dyn Worker>, Response> {
    if let Some(worker) = find_best_worker(registry, query, model_id) {
        return Ok(worker);
    }

    tracing::debug!(
        model = model_id,
        "No worker found, refreshing external worker models"
    );
    refresh_external_models(registry, client, auth_header).await;

    find_best_worker(registry, query, model_id).ok_or_else(|| {
        if any_worker_supports_model(registry, query, model_id) {
            error::service_unavailable(
                "service_unavailable",
                format!("All workers for model '{model_id}' are temporarily unavailable"),
            )
        } else {
            error::model_not_found(model_id)
        }
    })
}

// ============================================================================
// Internal Helpers
// ============================================================================

/// Find the least-loaded available worker that supports the given model.
fn find_best_worker(
    registry: &WorkerRegistry,
    query: &WorkerQuery,
    model_id: &str,
) -> Option<Arc<dyn Worker>> {
    get_candidates(registry, query)
        .into_iter()
        .filter(|w| w.supports_model(model_id))
        .min_by_key(|w| w.load())
}

/// Check if any healthy worker supports the model (regardless of circuit breaker).
///
/// Used to distinguish "model not found" from "all workers circuit-broken".
fn any_worker_supports_model(
    registry: &WorkerRegistry,
    query: &WorkerQuery,
    model_id: &str,
) -> bool {
    let workers = registry.get_workers_filtered(
        query.model_id,
        query.worker_type,
        query.connection_mode,
        query.runtime_type,
        true, // healthy only — model exists even if circuit-broken
    );
    let candidates = match &query.provider {
        Some(p) => filter_by_provider(workers, p.clone()),
        None => workers,
    };
    candidates.iter().any(|w| w.supports_model(model_id))
}

// ============================================================================
// Provider Security Filtering
// ============================================================================

/// In multi-provider setups, filter to only workers matching the target provider.
///
/// In single-provider (or no-provider) setups, returns all workers unchanged.
/// This prevents credentials from leaking to workers of a different provider.
fn filter_by_provider(
    workers: Vec<Arc<dyn Worker>>,
    target: ProviderType,
) -> Vec<Arc<dyn Worker>> {
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
        workers
            .into_iter()
            .filter(|w| matches!(w.default_provider(), Some(p) if *p == target))
            .collect()
    } else {
        workers
    }
}

// ============================================================================
// External Model Refresh
// ============================================================================

/// Refresh model lists for all healthy external workers by calling their
/// `/v1/models` endpoints in parallel.
///
/// Used by `select_worker` on cache miss and by the `/v1/models` endpoint
/// to proactively refresh before listing.
pub async fn refresh_external_models(
    registry: &WorkerRegistry,
    client: &reqwest::Client,
    auth_header: Option<&HeaderValue>,
) {
    let external_workers =
        registry.get_workers_filtered(None, None, None, Some(RuntimeType::External), true);

    if external_workers.is_empty() {
        return;
    }

    tracing::debug!(
        "Refreshing models for {} external workers",
        external_workers.len()
    );

    let futures: Vec<_> = external_workers
        .iter()
        .map(|w| refresh_worker_models(client, w, auth_header))
        .collect();

    join_all(futures).await;
}

/// Refresh a single worker's model list by calling its `/v1/models` endpoint.
async fn refresh_worker_models(
    client: &reqwest::Client,
    worker: &Arc<dyn Worker>,
    auth_header: Option<&HeaderValue>,
) -> bool {
    let url = format!("{}/v1/models", worker.url());
    let mut backend_req = client.get(&url);
    if let Some(auth) = auth_header {
        backend_req = apply_provider_headers(backend_req, &url, Some(auth));
    }

    match backend_req.send().await {
        Ok(response) if response.status().is_success() => {
            match response.json::<serde_json::Value>().await {
                Ok(json_response) => {
                    if let Some(data) = json_response.get("data").and_then(|d| d.as_array()) {
                        let model_cards: Vec<ModelCard> = data
                            .iter()
                            .filter_map(|m| m.get("id").and_then(|id| id.as_str()))
                            .map(ModelCard::new)
                            .collect();

                        if !model_cards.is_empty() {
                            tracing::info!(
                                "Model refresh: found {} models from {}",
                                model_cards.len(),
                                url
                            );
                            worker.set_models(model_cards);
                            return true;
                        }
                    }
                    false
                }
                Err(e) => {
                    tracing::warn!("Failed to parse models response: {}", e);
                    false
                }
            }
        }
        Ok(response) => {
            tracing::debug!(
                "Model refresh returned non-success status {} from {}",
                response.status(),
                url
            );
            false
        }
        Err(e) => {
            tracing::warn!("Failed to fetch models from backend: {}", e);
            false
        }
    }
}
