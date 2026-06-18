//! Shared worker selection for all routers.
//!
//! Single public API: [`WorkerSelector::select_worker`].

use std::{sync::Arc, time::Duration};

use axum::{
    http::{HeaderMap, HeaderValue},
    response::Response,
};
use futures_util::future::join_all;
use openai_protocol::models::ListModelsResponse;

use crate::{
    routers::{
        common::header_utils::{apply_provider_headers, extract_auth_header},
        error,
    },
    worker::{
        ConnectionMode, ProviderType, RuntimeType, Worker, WorkerRegistry, WorkerType,
        UNKNOWN_MODEL_ID,
    },
};

/// Holds references to shared infrastructure needed for worker selection.
///
/// Created once per router (or per-request where lifetimes differ) and
/// reused across calls.
pub struct WorkerSelector<'a> {
    registry: &'a WorkerRegistry,
    client: &'a reqwest::Client,
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
        self.refresh_external_models(auth.as_ref(), req.provider.as_ref())
            .await;

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

    /// Available workers passing every request filter, ready for load-based
    /// selection.
    ///
    /// When the model is known, this uses the registry's bounded,
    /// wildcard-safe per-model lookup (`get_candidates_for_model`) instead
    /// of scanning the whole fleet — the O(total fleet) scan was the hot-path
    /// cost this addresses. If that bounded set turns up no available
    /// candidates, it falls back to the full scan so behavior is never worse
    /// (the bounded index can briefly lag a concurrent registration, and we
    /// must never regress a servable model into "no worker found").
    fn get_candidates(&self, req: &SelectWorkerRequest<'_>) -> Vec<Arc<dyn Worker>> {
        if req.model_id != UNKNOWN_MODEL_ID {
            let bounded =
                Self::filter_candidates(self.registry.get_candidates_for_model(req.model_id), req);
            if !bounded.is_empty() {
                return bounded;
            }
        }

        // Unknown model (caller wants any worker) or empty bounded set:
        // fall back to the full scan, preserving the original behavior.
        Self::filter_candidates(
            self.registry.get_workers_filtered(
                None, // full scan — wildcard-safe via supports_model below
                req.worker_type,
                req.connection_mode,
                req.runtime_type,
                false, // we filter availability ourselves for consistent behavior
            ),
            req,
        )
    }

    /// Apply the request's worker_type/connection_mode/runtime_type,
    /// availability, and provider filters to a candidate set.
    ///
    /// `get_workers_filtered` already applies the type/mode/runtime filters,
    /// so re-applying them to that path is a cheap no-op; the bounded
    /// per-model path relies on them here.
    fn filter_candidates(
        workers: Vec<Arc<dyn Worker>>,
        req: &SelectWorkerRequest<'_>,
    ) -> Vec<Arc<dyn Worker>> {
        let candidates: Vec<_> = workers
            .into_iter()
            .filter(|w| Self::matches_filters(w, req) && w.is_available())
            .collect();

        match &req.provider {
            Some(provider) => filter_by_provider(candidates, provider),
            None => candidates,
        }
    }

    /// Per-worker worker_type/connection_mode/runtime_type filter, mirroring
    /// `WorkerRegistry::get_workers_filtered`. Applied to the bounded
    /// per-model candidate set (which is not pre-filtered).
    fn matches_filters(worker: &Arc<dyn Worker>, req: &SelectWorkerRequest<'_>) -> bool {
        if let Some(ref wtype) = req.worker_type {
            if *worker.worker_type() != *wtype {
                return false;
            }
        }
        if let Some(ref conn) = req.connection_mode {
            if worker.connection_mode() != conn {
                return false;
            }
        }
        if let Some(ref rt) = req.runtime_type {
            if worker.metadata().spec.runtime_type != *rt {
                return false;
            }
        }
        true
    }

    fn find_best_worker(&self, req: &SelectWorkerRequest<'_>) -> Option<Arc<dyn Worker>> {
        self.get_candidates(req)
            .into_iter()
            .filter(|w| w.supports_model(req.model_id))
            .min_by_key(|w| w.load())
    }

    /// Check if any healthy worker supports the model (regardless of circuit breaker).
    /// Used to distinguish "model not found" from "all workers circuit-broken".
    fn any_worker_supports_model(&self, req: &SelectWorkerRequest<'_>) -> bool {
        if req.model_id != UNKNOWN_MODEL_ID {
            let bounded = Self::healthy_supporting_candidates(
                self.registry.get_candidates_for_model(req.model_id),
                req,
            );
            if bounded.iter().any(|w| w.supports_model(req.model_id)) {
                return true;
            }
            // Bounded set yielded no supporting worker — fall through to the
            // full scan so we never falsely report "model not found" if the
            // per-model index briefly lags a registration.
        }

        let workers = self.registry.get_workers_filtered(
            None,
            req.worker_type,
            req.connection_mode,
            req.runtime_type,
            true, // healthy only — model exists even if circuit-broken
        );
        Self::healthy_supporting_candidates(workers, req)
            .iter()
            .any(|w| w.supports_model(req.model_id))
    }

    /// Filter a candidate set to healthy workers passing the request's
    /// type/mode/runtime and provider filters (no circuit-breaker check —
    /// the model exists even if every worker is circuit-broken).
    fn healthy_supporting_candidates(
        workers: Vec<Arc<dyn Worker>>,
        req: &SelectWorkerRequest<'_>,
    ) -> Vec<Arc<dyn Worker>> {
        let candidates: Vec<_> = workers
            .into_iter()
            .filter(|w| Self::matches_filters(w, req) && w.is_healthy())
            .collect();
        match &req.provider {
            Some(p) => filter_by_provider(candidates, p),
            None => candidates,
        }
    }

    /// Refresh model lists for healthy external workers in parallel.
    ///
    /// When `provider` is set, only workers matching that provider are refreshed
    /// to prevent credential leakage across providers. Each worker falls back to
    /// its own configured API key when the caller provides no auth.
    async fn refresh_external_models(
        &self,
        auth_header: Option<&HeaderValue>,
        provider: Option<&ProviderType>,
    ) {
        let mut external_workers =
            self.registry
                .get_workers_filtered(None, None, None, Some(RuntimeType::External), true);

        // Only refresh workers matching the request's provider to avoid sending
        // e.g. an OpenAI key to Anthropic workers during model discovery.
        if let Some(p) = provider {
            external_workers.retain(|w| matches!(w.default_provider(), Some(wp) if wp == p));
        }

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

        // Timeout prevents a slow/unresponsive worker from blocking all
        // requests that trigger refresh-on-miss.
        const REFRESH_TIMEOUT: Duration = Duration::from_secs(5);
        let _ = tokio::time::timeout(REFRESH_TIMEOUT, join_all(futures)).await;
    }
}

/// In multi-provider setups, filter to only workers matching the target provider.
/// In single-provider (or no-provider) setups, returns all workers unchanged.
fn filter_by_provider(
    workers: Vec<Arc<dyn Worker>>,
    target: &ProviderType,
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
            .filter(|w| matches!(w.default_provider(), Some(p) if p == target))
            .collect()
    } else {
        workers
    }
}

/// Refresh a single worker's model list by calling its `/v1/models` endpoint.
///
/// Auth headers are adapted per-vendor via [`apply_provider_headers`] (e.g.
/// Anthropic uses `x-api-key`, OpenAI uses `Authorization: Bearer`). The
/// response is parsed via [`ListModelsResponse::parse_upstream`].
async fn refresh_worker_models(
    client: &reqwest::Client,
    worker: &Arc<dyn Worker>,
    auth_header: Option<&HeaderValue>,
) -> bool {
    let url = format!("{}/v1/models", worker.url());
    let mut backend_req = client.get(&url);

    // Use caller's auth if provided, otherwise fall back to worker's configured API key.
    // This matches how auth is handled in request routing (e.g. openai/router.rs).
    let worker_auth = auth_header.cloned().or_else(|| {
        worker
            .api_key()
            .and_then(|k| HeaderValue::from_str(&format!("Bearer {k}")).ok())
    });
    if let Some(ref auth) = worker_auth {
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

#[cfg(test)]
mod tests {
    use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};

    use super::*;
    use crate::worker::BasicWorkerBuilder;

    fn ready_worker(builder: BasicWorkerBuilder) -> Arc<dyn Worker> {
        // disable_health_check makes the worker start Ready (and thus
        // is_available, since a fresh circuit breaker permits execution).
        let worker: Arc<dyn Worker> = Arc::new(
            builder
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        assert!(worker.is_available(), "test worker must be routable");
        worker
    }

    fn selector_request(model_id: &str) -> SelectWorkerRequest<'_> {
        SelectWorkerRequest {
            model_id,
            ..Default::default()
        }
    }

    fn client() -> reqwest::Client {
        reqwest::Client::new()
    }

    /// The core regression: a wildcard worker (no models declared) must be
    /// selectable for an arbitrary model via the bounded candidate path,
    /// even though it is not in `get_by_model(arbitrary_model)`.
    #[tokio::test]
    async fn wildcard_worker_selected_for_arbitrary_model_via_bounded_path() {
        let registry = WorkerRegistry::new();
        registry
            .register(ready_worker(BasicWorkerBuilder::new(
                "http://wildcard:8080",
            )))
            .unwrap();

        // Sanity: not in the per-model index for this arbitrary model.
        assert!(registry.get_by_model("totally-made-up-model").is_empty());

        let client = client();
        let selector = WorkerSelector::new(&registry, &client);
        let chosen = selector
            .select_worker(&selector_request("totally-made-up-model"))
            .await
            .expect("wildcard worker should serve any model");
        assert_eq!(chosen.url(), "http://wildcard:8080");
    }

    /// Per-model bounding returns the right worker for a normal model and
    /// excludes workers serving other models.
    #[tokio::test]
    async fn bounded_path_selects_correct_model_and_excludes_others() {
        let registry = WorkerRegistry::new();
        registry
            .register(ready_worker(
                BasicWorkerBuilder::new("http://a:8080").model(ModelCard::new("model-a")),
            ))
            .unwrap();
        registry
            .register(ready_worker(
                BasicWorkerBuilder::new("http://b:8080").model(ModelCard::new("model-b")),
            ))
            .unwrap();

        let client = client();
        let selector = WorkerSelector::new(&registry, &client);

        let chosen = selector
            .select_worker(&selector_request("model-a"))
            .await
            .expect("model-a worker exists");
        assert_eq!(
            chosen.url(),
            "http://a:8080",
            "must not pick model-b worker"
        );
    }

    /// Empty-bounded-set fallback: a worker reachable only by an *alias*
    /// (not its indexed id, and not a wildcard) is found via the full-scan
    /// fallback so we never regress a servable model into model_not_found.
    #[tokio::test]
    async fn empty_bounded_set_falls_back_to_full_scan() {
        let registry = WorkerRegistry::new();
        // Indexed under id "gpt-4"; supports "gpt-4-latest" only via alias.
        registry
            .register(ready_worker(
                BasicWorkerBuilder::new("http://aliased:8080")
                    .model(ModelCard::new("gpt-4").with_alias("gpt-4-latest")),
            ))
            .unwrap();

        // The bounded lookup keys on the literal id and finds nothing for the
        // alias (no wildcard workers either) — proving the fallback is what
        // surfaces the worker.
        assert!(
            registry.get_candidates_for_model("gpt-4-latest").is_empty(),
            "alias is not an index key, so the bounded set is empty"
        );

        let client = client();
        let selector = WorkerSelector::new(&registry, &client);
        let chosen = selector
            .select_worker(&selector_request("gpt-4-latest"))
            .await
            .expect("alias-only worker must be found via fallback");
        assert_eq!(chosen.url(), "http://aliased:8080");
    }

    /// A genuinely unknown model with no wildcard workers yields
    /// model_not_found (the fallback does not invent workers).
    #[tokio::test]
    async fn unknown_model_without_wildcard_is_not_found() {
        let registry = WorkerRegistry::new();
        registry
            .register(ready_worker(
                BasicWorkerBuilder::new("http://a:8080").model(ModelCard::new("model-a")),
            ))
            .unwrap();

        let client = client();
        let selector = WorkerSelector::new(&registry, &client);
        let result = selector
            .select_worker(&selector_request("nonexistent"))
            .await;
        assert!(result.is_err(), "no worker and no wildcard → error");
    }

    /// With UNKNOWN_MODEL_ID the common/HTTP path takes the full-scan branch
    /// (no per-model index lookup). A wildcard worker — which `supports_model`
    /// accepts for the sentinel — is selectable, matching the pre-bounding
    /// behavior. (A *specific* worker does not `supports_model` the sentinel,
    /// so this path only resolves to wildcards, unchanged by this PR.)
    #[tokio::test]
    async fn unknown_model_id_selects_wildcard_via_full_scan() {
        let registry = WorkerRegistry::new();
        registry
            .register(ready_worker(BasicWorkerBuilder::new(
                "http://wildcard:8080",
            )))
            .unwrap();

        let client = client();
        let selector = WorkerSelector::new(&registry, &client);
        let chosen = selector
            .select_worker(&selector_request(UNKNOWN_MODEL_ID))
            .await
            .expect("wildcard worker for unknown model id");
        assert_eq!(chosen.url(), "http://wildcard:8080");
    }

    /// The bounded path still honors the worker_type filter.
    #[tokio::test]
    async fn bounded_path_honors_worker_type_filter() {
        let registry = WorkerRegistry::new();
        registry
            .register(ready_worker(
                BasicWorkerBuilder::new("http://regular:8080")
                    .model(ModelCard::new("m"))
                    .worker_type(WorkerType::Regular),
            ))
            .unwrap();
        registry
            .register(ready_worker(
                BasicWorkerBuilder::new("http://prefill:8080")
                    .model(ModelCard::new("m"))
                    .worker_type(WorkerType::Prefill),
            ))
            .unwrap();

        let client = client();
        let selector = WorkerSelector::new(&registry, &client);
        let chosen = selector
            .select_worker(&SelectWorkerRequest {
                model_id: "m",
                worker_type: Some(WorkerType::Prefill),
                ..Default::default()
            })
            .await
            .expect("prefill worker exists for model m");
        assert_eq!(chosen.url(), "http://prefill:8080");
    }
}
