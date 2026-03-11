//! Shared worker selection for all routers.
//!
//! Single public API: [`WorkerSelector::select_worker`].

use std::sync::Arc;

use axum::{
    http::{HeaderMap, HeaderValue},
    response::Response,
};
use futures_util::future::join_all;
use openai_protocol::models::ListModelsResponse;

use crate::{
    core::{ConnectionMode, ProviderType, RuntimeType, Worker, WorkerRegistry, WorkerType},
    routers::{
        error,
        header_utils::{apply_provider_headers, extract_auth_header},
    },
};

// ============================================================================
// Public Types
// ============================================================================

/// Holds references to shared infrastructure needed for worker selection.
///
/// Created once per router (or per-request where lifetimes differ) and
/// reused across calls.
pub struct WorkerSelector<'a> {
    pub registry: &'a WorkerRegistry,
    pub client: &'a reqwest::Client,
}

/// Input for [`WorkerSelector::select_worker`].
///
/// Combines the model to resolve with optional registry filters and
/// the caller's HTTP headers (used for auth passthrough during
/// upstream model refresh).
#[derive(Debug, Default)]
pub struct SelectWorkerRequest<'a> {
    /// Model ID to select a worker for (required).
    pub model_id: &'a str,

    /// Caller's HTTP headers — used to extract the auth token for
    /// upstream `/v1/models` refresh on cache miss.
    pub headers: Option<&'a HeaderMap>,

    /// Provider-based security filtering for multi-provider setups.
    /// When set, prevents credentials from leaking to workers of a
    /// different provider (e.g. Anthropic key to OpenAI worker).
    pub provider: Option<ProviderType>,

    /// Filter by worker type (Regular, Prefill, Decode). `None` = any.
    pub worker_type: Option<WorkerType>,

    /// Filter by connection mode (Http, Grpc). `None` = any.
    pub connection_mode: Option<ConnectionMode>,

    /// Filter by runtime type (External, Sglang, Vllm, Trtllm). `None` = any.
    pub runtime_type: Option<RuntimeType>,
}

// ============================================================================
// Public API
// ============================================================================

impl<'a> WorkerSelector<'a> {
    pub fn new(registry: &'a WorkerRegistry, client: &'a reqwest::Client) -> Self {
        Self { registry, client }
    }

    /// Select the best worker for a model with refresh-on-miss.
    ///
    /// 1. Filter available workers by the request criteria.
    /// 2. Pick the least-loaded worker that supports the model.
    /// 3. On miss, refresh external model lists by calling `/v1/models`
    ///    on all external workers (vendor-aware parsing), then retry.
    /// 4. Return an error distinguishing "model not found" from "all
    ///    workers circuit-broken".
    pub async fn select_worker(
        &self,
        req: &SelectWorkerRequest<'_>,
    ) -> Result<Arc<dyn Worker>, Response> {
        if let Some(worker) = self.find_best_worker(req) {
            return Ok(worker);
        }

        tracing::debug!(
            model = req.model_id,
            "No worker found, refreshing external worker models"
        );

        let auth = extract_auth_header(req.headers, None);
        self.refresh_external_models(auth.as_ref()).await;

        self.find_best_worker(req).ok_or_else(|| {
            if self.any_worker_supports_model(req) {
                error::service_unavailable(
                    "service_unavailable",
                    format!(
                        "All workers for model '{}' are temporarily unavailable",
                        req.model_id
                    ),
                )
            } else {
                error::model_not_found(req.model_id)
            }
        })
    }
}

// ============================================================================
// Internal Helpers
// ============================================================================

impl WorkerSelector<'_> {
    /// Get available worker candidates matching the request filters.
    fn get_candidates(&self, req: &SelectWorkerRequest<'_>) -> Vec<Arc<dyn Worker>> {
        let workers = self.registry.get_workers_filtered(
            None, // model_id index lookup not used here — we filter via supports_model
            req.worker_type,
            req.connection_mode,
            req.runtime_type,
            false, // we filter availability ourselves for consistent behavior
        );

        let candidates: Vec<_> = workers.into_iter().filter(|w| w.is_available()).collect();

        match &req.provider {
            Some(provider) => filter_by_provider(candidates, provider.clone()),
            None => candidates,
        }
    }

    /// Find the least-loaded available worker that supports the model.
    fn find_best_worker(&self, req: &SelectWorkerRequest<'_>) -> Option<Arc<dyn Worker>> {
        self.get_candidates(req)
            .into_iter()
            .filter(|w| w.supports_model(req.model_id))
            .min_by_key(|w| w.load())
    }

    /// Check if any healthy worker supports the model (regardless of circuit breaker).
    ///
    /// Used to distinguish "model not found" from "all workers circuit-broken".
    fn any_worker_supports_model(&self, req: &SelectWorkerRequest<'_>) -> bool {
        let workers = self.registry.get_workers_filtered(
            None,
            req.worker_type,
            req.connection_mode,
            req.runtime_type,
            true, // healthy only — model exists even if circuit-broken
        );
        let candidates = match &req.provider {
            Some(p) => filter_by_provider(workers, p.clone()),
            None => workers,
        };
        candidates.iter().any(|w| w.supports_model(req.model_id))
    }
}

// ============================================================================
// Provider Security Filtering
// ============================================================================

/// In multi-provider setups, filter to only workers matching the target provider.
///
/// In single-provider (or no-provider) setups, returns all workers unchanged.
/// This prevents credentials from leaking to workers of a different provider.
fn filter_by_provider(workers: Vec<Arc<dyn Worker>>, target: ProviderType) -> Vec<Arc<dyn Worker>> {
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

impl WorkerSelector<'_> {
    /// Refresh model lists for all healthy external workers by calling their
    /// `/v1/models` endpoints in parallel.
    ///
    /// Uses [`ListModelsResponse::parse_upstream`] with provider detection from
    /// the worker URL so that both OpenAI and Anthropic response formats are
    /// handled correctly, and the resulting `ModelCard`s carry provider info.
    async fn refresh_external_models(&self, auth_header: Option<&HeaderValue>) {
        let external_workers =
            self.registry
                .get_workers_filtered(None, None, None, Some(RuntimeType::External), true);

        if external_workers.is_empty() {
            return;
        }

        tracing::debug!(
            "Refreshing models for {} external workers",
            external_workers.len()
        );

        let futures: Vec<_> = external_workers
            .iter()
            .map(|w| refresh_worker_models(self.client, w, auth_header))
            .collect();

        join_all(futures).await;
    }
}

/// Refresh a single worker's model list by calling its `/v1/models` endpoint.
///
/// Parses the response using [`ListModelsResponse::parse_upstream`] which handles
/// both OpenAI (`{"data": [{"id": "..."}]}`) and Anthropic (`{"data": [{"id": "...",
/// "type": "model"}]}`) formats, and tags each `ModelCard` with the provider
/// inferred from the worker URL.
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
                Ok(json) => {
                    let provider = ProviderType::from_url(&url);
                    let model_cards = ListModelsResponse::parse_upstream(&json, provider);

                    if !model_cards.is_empty() {
                        tracing::info!(
                            "Model refresh: found {} models from {}",
                            model_cards.len(),
                            url
                        );
                        worker.set_models(model_cards);
                        return true;
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
