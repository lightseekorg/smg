//! Router Manager for coordinating multiple routers and workers
//!
//! Provides centralized management based on enable_igw flag:
//! - Single Router Mode (enable_igw=false): Router owns workers directly
//! - Multi-Router Mode (enable_igw=true): RouterManager coordinates everything

use std::{collections::HashSet, sync::Arc};

use arc_swap::ArcSwap;
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use dashmap::DashMap;
use futures::future::select_all;
use openai_protocol::{
    chat::ChatCompletionRequest,
    classify::ClassifyRequest,
    completion::CompletionRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    interactions::InteractionsRequest,
    messages::CreateMessageRequest,
    model_card::ModelCard,
    models::ListModelsResponse,
    realtime_session::{
        RealtimeClientSecretCreateRequest, RealtimeSessionCreateRequest,
        RealtimeTranscriptionSessionCreateRequest,
    },
    rerank::RerankRequest,
    responses::ResponsesRequest,
    transcription::TranscriptionRequest,
};
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::{
    app_context::AppContext,
    config::RoutingMode,
    routers::{
        common::header_utils::apply_provider_headers,
        factory::{router_ids, RouterId},
        AudioFile, RouterFactory, RouterTrait,
    },
    server::ServerConfig,
    worker::{ConnectionMode, ProviderType, RuntimeType, WorkerRegistry, WorkerType},
};

pub struct RouterManager {
    worker_registry: Arc<WorkerRegistry>,
    client: reqwest::Client,
    gateway_api_key: Option<String>,
    routers: Arc<DashMap<RouterId, Arc<dyn RouterTrait>>>,
    routers_snapshot: ArcSwap<Vec<Arc<dyn RouterTrait>>>,
    default_router: Arc<std::sync::RwLock<Option<RouterId>>>,
    enable_igw: bool,
}

impl RouterManager {
    pub fn new(worker_registry: Arc<WorkerRegistry>, client: reqwest::Client) -> Self {
        Self {
            worker_registry,
            client,
            gateway_api_key: None,
            routers: Arc::new(DashMap::new()),
            routers_snapshot: ArcSwap::from_pointee(Vec::new()),
            default_router: Arc::new(std::sync::RwLock::new(None)),
            enable_igw: false, // Will be set properly in from_config
        }
    }

    /// Register a router if creation succeeded, log either way.
    fn try_register(
        &self,
        id: RouterId,
        label: &str,
        result: Result<Box<dyn RouterTrait>, String>,
    ) {
        match result {
            Ok(router) => {
                info!("Created {label} router");
                self.register_router(id, Arc::from(router));
            }
            Err(e) => {
                warn!("Failed to create {label} router: {e}");
            }
        }
    }

    pub async fn from_config(
        config: &ServerConfig,
        app_context: &Arc<AppContext>,
    ) -> Result<Arc<Self>, String> {
        let mut manager = Self::new(
            app_context.worker_registry.clone(),
            app_context.client.clone(),
        );
        manager.enable_igw = config.router_config.enable_igw;
        manager
            .gateway_api_key
            .clone_from(&config.router_config.api_key);
        let manager = Arc::new(manager);

        if config.router_config.enable_igw {
            info!("Initializing RouterManager in multi-router mode (IGW)");

            let routers =
                RouterFactory::create_igw_routers(&config.router_config.policy, app_context).await;

            for (id, label, result) in routers {
                manager.try_register(id, label, result);
            }

            info!(
                "RouterManager initialized with {} routers for multi-router mode",
                manager.router_count(),
            );
        } else {
            info!("Initializing RouterManager in single-router mode");

            let single_router = Arc::from(RouterFactory::create_router(app_context).await?);
            let router_id = Self::determine_router_id(
                &config.router_config.mode,
                config.router_config.connection_mode,
            );

            info!("Created single router with ID: {}", router_id.as_str());
            manager.register_router(router_id.clone(), single_router);
            manager.set_default_router(router_id);
        }

        if manager.router_count() == 0 {
            return Err("No routers could be initialized".to_string());
        }

        Ok(manager)
    }

    pub fn determine_router_id(
        routing_mode: &RoutingMode,
        connection_mode: ConnectionMode,
    ) -> RouterId {
        match (connection_mode, routing_mode) {
            (ConnectionMode::Http, RoutingMode::Regular { .. }) => router_ids::HTTP_REGULAR,
            (ConnectionMode::Http, RoutingMode::PrefillDecode { .. }) => router_ids::HTTP_PD,
            (ConnectionMode::Http, RoutingMode::OpenAI { .. }) => router_ids::HTTP_OPENAI,
            (ConnectionMode::Http, RoutingMode::Anthropic { .. }) => router_ids::HTTP_ANTHROPIC,
            (ConnectionMode::Grpc, RoutingMode::Regular { .. }) => router_ids::GRPC_REGULAR,
            (ConnectionMode::Grpc, RoutingMode::PrefillDecode { .. }) => router_ids::GRPC_PD,
            (ConnectionMode::Http, RoutingMode::Gemini { .. }) => router_ids::HTTP_GEMINI,
            (ConnectionMode::Grpc, RoutingMode::OpenAI { .. }) => router_ids::GRPC_REGULAR,
            (ConnectionMode::Grpc, RoutingMode::Anthropic { .. }) => router_ids::GRPC_REGULAR,
            (ConnectionMode::Grpc, RoutingMode::Gemini { .. }) => router_ids::GRPC_REGULAR,
        }
    }

    pub fn register_router(&self, id: RouterId, router: Arc<dyn RouterTrait>) {
        self.routers.insert(id.clone(), router);

        // Update the lock-free snapshot for fast per-request iteration
        let new_snapshot: Vec<_> = self.routers.iter().map(|e| e.value().clone()).collect();
        self.routers_snapshot.store(Arc::new(new_snapshot));

        let mut default_router = self
            .default_router
            .write()
            .unwrap_or_else(|e| e.into_inner());
        if default_router.is_none() {
            *default_router = Some(id.clone());
            info!("Set default router to {}", id.as_str());
        }
    }

    pub fn set_default_router(&self, id: RouterId) {
        let mut default_router = self
            .default_router
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *default_router = Some(id);
    }

    pub fn router_count(&self) -> usize {
        self.routers.len()
    }

    pub fn get_router_for_model(&self, model_id: &str) -> Option<Arc<dyn RouterTrait>> {
        let workers = self.worker_registry.get_by_model(model_id);

        // Find the best router ID based on worker capabilities
        // Priority: external (provider-specific) > grpc-pd > http-pd > grpc-regular > http-regular
        let best_router_id = workers
            .iter()
            .map(|w| {
                let is_pd = matches!(w.worker_type(), WorkerType::Prefill | WorkerType::Decode);
                let is_grpc = matches!(w.connection_mode(), ConnectionMode::Grpc);
                let is_external = matches!(w.metadata().spec.runtime_type, RuntimeType::External);

                if is_external {
                    // Route external workers to the correct provider-specific router
                    let router_id = match w.provider_for_model(model_id) {
                        Some(ProviderType::Gemini) => &router_ids::HTTP_GEMINI,
                        Some(ProviderType::Anthropic) => &router_ids::HTTP_ANTHROPIC,
                        _ => &router_ids::HTTP_OPENAI,
                    };
                    return (4, router_id);
                }

                match (is_grpc, is_pd) {
                    (true, true) => (3, &router_ids::GRPC_PD),
                    (false, true) => (2, &router_ids::HTTP_PD),
                    (true, false) => (1, &router_ids::GRPC_REGULAR),
                    (false, false) => (0, &router_ids::HTTP_REGULAR),
                }
            })
            .max_by_key(|(score, _)| *score)
            .map(|(_, id)| id);

        if let Some(router_id) = best_router_id {
            if let Some(router) = self.routers.get(router_id) {
                return Some(router.clone());
            }
        }

        // Fallback to default router
        let default_router = self
            .default_router
            .read()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(ref default_id) = *default_router {
            self.routers.get(default_id).map(|r| r.clone())
        } else {
            None
        }
    }

    pub fn select_router_for_request(
        &self,
        headers: Option<&HeaderMap>,
        model_id: Option<&str>,
    ) -> Option<Arc<dyn RouterTrait>> {
        // In single-router mode (enable_igw=false), always use the default router
        if !self.enable_igw {
            let default_router = self
                .default_router
                .read()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(ref default_id) = *default_router {
                debug!(
                    "Single-router mode: using default router {} for model {:?}",
                    default_id.as_str(),
                    model_id
                );
                return self.routers.get(default_id).map(|r| r.clone());
            }
        }

        let prefer_pd = headers
            .and_then(|h| {
                h.get("x-prefer-pd")
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s == "true" || s == "1")
            })
            .unwrap_or(false);

        let (num_regular_workers, num_pd_workers) = self.worker_registry.get_worker_distribution();
        let mut best_router = None;
        let mut best_score = -1.0;

        // Extract router validity check into a closure to reduce redundancy
        let is_router_valid =
            |is_pd: bool| (is_pd && num_pd_workers > 0) || (!is_pd && num_regular_workers > 0);

        if let Some(model) = model_id {
            // Efficient Single Lookup for Specific Model
            if let Some(router) = self.get_router_for_model(model) {
                if is_router_valid(router.is_pd_mode()) {
                    return Some(router);
                }
            }
        } else {
            // ZERO-ALLOCATION Snapshot Iteration (Hot Path Optimization)
            // Atomic load avoids heap allocations and DashMap shard locks per-request
            let routers_snapshot = self.routers_snapshot.load();
            for router in routers_snapshot.iter() {
                let mut score = 1.0;

                let is_pd = router.is_pd_mode();
                if prefer_pd && is_pd {
                    score += 2.0;
                } else if !prefer_pd && !is_pd {
                    score += 1.0;
                }
                // TODO: Once routers expose worker stats, we can evaluate:
                // - Average worker priority vs priority_threshold
                // - Average worker cost vs max_cost
                // - Current load and health status

                if score > best_score && is_router_valid(is_pd) {
                    best_score = score;
                    best_router = Some(Arc::clone(router));
                }
            }
        }

        best_router
    }

    /// Build a response from self-hosted registry models (excludes external workers).
    fn registry_models_response(&self) -> Response {
        let cards: Vec<_> = self
            .worker_registry
            .get_all()
            .iter()
            .filter(|w| !matches!(w.metadata().spec.runtime_type, RuntimeType::External))
            .flat_map(|w| w.models())
            .collect();
        if cards.is_empty() {
            (StatusCode::SERVICE_UNAVAILABLE, "No models available").into_response()
        } else {
            let resp = ListModelsResponse::from_model_cards(cards);
            (StatusCode::OK, Json(resp)).into_response()
        }
    }

    /// Fan out to all healthy external upstreams concurrently with the caller's
    /// bearer token and return the first successful model inventory. Returns an
    /// empty vec on total failure.
    async fn fetch_upstream_models(&self, bearer_token: &str) -> Vec<ModelCard> {
        let unique_urls: Vec<_> = self
            .worker_registry
            .get_workers_filtered(None, None, None, Some(RuntimeType::External), true)
            .iter()
            .map(|w| w.url().to_string())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        if unique_urls.is_empty() {
            return Vec::new();
        }

        debug!(
            "Trying {} upstream(s) for model discovery",
            unique_urls.len()
        );

        let auth = match HeaderValue::from_str(&format!("Bearer {bearer_token}")) {
            Ok(v) => Some(v),
            Err(e) => {
                warn!("Bearer token contains invalid header characters: {e}");
                return Vec::new();
            }
        };

        // Fan out concurrently; return first non-empty result.
        let mut pending: Vec<_> = unique_urls
            .into_iter()
            .map(|url| {
                Box::pin(Self::fetch_models_from(
                    self.client.clone(),
                    url,
                    auth.clone(),
                ))
            })
            .collect();

        while !pending.is_empty() {
            let (cards, _index, remaining) = select_all(pending).await;
            if !cards.is_empty() {
                return cards;
            }
            pending = remaining;
        }

        Vec::new()
    }

    /// Fetch models from a single upstream endpoint.
    async fn fetch_models_from(
        client: reqwest::Client,
        base_url: String,
        auth: Option<HeaderValue>,
    ) -> Vec<ModelCard> {
        let url = format!("{base_url}/v1/models");
        let req = apply_provider_headers(client.get(&url), &url, auth.as_ref());

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                debug!("Failed to reach upstream {url}: {e}");
                return Vec::new();
            }
        };

        if !resp.status().is_success() {
            debug!(
                "Upstream {url} returned {} for model discovery",
                resp.status()
            );
            return Vec::new();
        }

        match resp.json::<Value>().await {
            Ok(json) => ListModelsResponse::parse_upstream(&json, ProviderType::from_url(&url)),
            Err(e) => {
                warn!("Failed to parse upstream models from {url}: {e}");
                Vec::new()
            }
        }
    }
}

#[async_trait]
impl RouterTrait for RouterManager {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        let router = self.select_router_for_request(None, None);
        if let Some(router) = router {
            router.health_generate(_req).await
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "No routers with healthy workers available",
            )
                .into_response()
        }
    }

    async fn get_server_info(&self, req: Request<Body>) -> Response {
        let router = self.select_router_for_request(None, None);
        if let Some(router) = router {
            router.get_server_info(req).await
        } else {
            (StatusCode::SERVICE_UNAVAILABLE, "No routers available").into_response()
        }
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        // Extract token from Authorization header (case-insensitive "Bearer " prefix
        // per RFC 7235) or Anthropic-style x-api-key header.
        let bearer_token = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|h| h.to_str().ok())
            .and_then(|h| {
                let lower = h.to_ascii_lowercase();
                lower.starts_with("bearer ").then(|| h[7..].to_string())
            })
            .or_else(|| {
                req.headers()
                    .get("x-api-key")
                    .and_then(|h| h.to_str().ok())
                    .map(String::from)
            });

        // Short-circuit: if the token matches the gateway's own API key, skip
        // upstream fan-out and return registry models directly.
        if let Some(ref token) = bearer_token {
            let is_gateway_key = self.gateway_api_key.as_ref().is_some_and(|gw| gw == token);
            if is_gateway_key {
                return self.registry_models_response();
            }
        }

        // If the caller sent a provider token, try to discover models from
        // upstream providers. This enables BYOK (bring your own key) flows.
        if let Some(ref token) = bearer_token {
            let upstream_cards = self.fetch_upstream_models(token).await;
            if !upstream_cards.is_empty() {
                let resp = ListModelsResponse::from_model_cards(upstream_cards);
                return (StatusCode::OK, Json(resp)).into_response();
            }
            // All upstreams failed or returned nothing — fall through to registry.
        }

        self.registry_models_response()
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Route to default router or first available router
        let router_id = {
            let default_router = self
                .default_router
                .read()
                .unwrap_or_else(|e| e.into_inner());
            default_router.clone()
        };

        let router = if let Some(id) = router_id {
            self.routers.get(&id).map(|r| r.clone())
        } else {
            // If no default, use first available router
            self.routers.iter().next().map(|r| r.value().clone())
        };

        if let Some(router) = router {
            router.get_model_info(req).await
        } else {
            (StatusCode::SERVICE_UNAVAILABLE, "No routers available").into_response()
        }
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: &str,
    ) -> Response {
        let router = self.select_router_for_request(headers, Some(model_id));

        if let Some(router) = router {
            router.route_generate(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for this request",
            )
                .into_response()
        }
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: &str,
    ) -> Response {
        let router = self.select_router_for_request(headers, Some(model_id));

        if let Some(router) = router {
            router.route_chat(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: &str,
    ) -> Response {
        let router = self.select_router_for_request(headers, Some(model_id));

        if let Some(router) = router {
            router.route_completion(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_messages(
        &self,
        headers: Option<&HeaderMap>,
        body: &CreateMessageRequest,
        model_id: &str,
    ) -> Response {
        let router = self.select_router_for_request(headers, Some(model_id));

        if let Some(router) = router {
            router.route_messages(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: &str,
    ) -> Response {
        let router = self.select_router_for_request(headers, Some(model_id));

        if let Some(router) = router {
            router.route_responses(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available to handle responses request",
            )
                .into_response()
        }
    }

    async fn route_interactions(
        &self,
        headers: Option<&HeaderMap>,
        body: &InteractionsRequest,
        model_id: Option<&str>,
    ) -> Response {
        let selected_model = model_id.or(body.model.as_deref()).or(body.agent.as_deref());
        let router = self.select_router_for_request(headers, selected_model);

        if let Some(router) = router {
            router
                .route_interactions(headers, body, selected_model)
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available to handle interactions request",
            )
                .into_response()
        }
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.cancel_response(headers, response_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("No router available to cancel response '{response_id}'"),
            )
                .into_response()
        }
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: &str,
    ) -> Response {
        let router = self.select_router_for_request(headers, Some(model_id));

        if let Some(router) = router {
            router.route_embeddings(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: &str,
    ) -> Response {
        let router = self.select_router_for_request(headers, Some(model_id));

        if let Some(router) = router {
            router.route_classify(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_audio_transcriptions(
        &self,
        headers: Option<&HeaderMap>,
        body: &TranscriptionRequest,
        audio: AudioFile,
        model_id: &str,
    ) -> Response {
        let router = self.select_router_for_request(headers, Some(model_id));

        if let Some(router) = router {
            router
                .route_audio_transcriptions(headers, body, audio, model_id)
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: &str,
    ) -> Response {
        let router = self.select_router_for_request(headers, Some(model_id));

        if let Some(router) = router {
            router.route_rerank(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for rerank request",
            )
                .into_response()
        }
    }

    async fn route_realtime_session(
        &self,
        headers: Option<&HeaderMap>,
        body: &RealtimeSessionCreateRequest,
    ) -> Response {
        let model = body.model.as_deref();
        let router = self.select_router_for_request(headers, model);
        if let Some(router) = router {
            router.route_realtime_session(headers, body).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for realtime session request",
            )
                .into_response()
        }
    }

    async fn route_realtime_client_secret(
        &self,
        headers: Option<&HeaderMap>,
        body: &RealtimeClientSecretCreateRequest,
    ) -> Response {
        let model = body.session.model.as_deref();
        let router = self.select_router_for_request(headers, model);
        if let Some(router) = router {
            router.route_realtime_client_secret(headers, body).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for realtime client secret request",
            )
                .into_response()
        }
    }

    async fn route_realtime_transcription_session(
        &self,
        headers: Option<&HeaderMap>,
        body: &RealtimeTranscriptionSessionCreateRequest,
    ) -> Response {
        let model = body.model.as_deref();
        let router = self.select_router_for_request(headers, model);
        if let Some(router) = router {
            router
                .route_realtime_transcription_session(headers, body)
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for realtime transcription request",
            )
                .into_response()
        }
    }

    async fn route_realtime_ws(&self, req: Request<Body>, model: &str) -> Response {
        let router = self.select_router_for_request(None, Some(model));
        if let Some(router) = router {
            router.route_realtime_ws(req, model).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for realtime WebSocket request",
            )
                .into_response()
        }
    }

    async fn route_realtime_webrtc(&self, req: Request<Body>, model: &str) -> Response {
        let router = self.select_router_for_request(None, Some(model));
        if let Some(router) = router {
            router.route_realtime_webrtc(req, model).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for realtime WebRTC request",
            )
                .into_response()
        }
    }

    fn router_type(&self) -> &'static str {
        "manager"
    }
}

impl std::fmt::Debug for RouterManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let default_router = self
            .default_router
            .read()
            .unwrap_or_else(|e| e.into_inner());
        f.debug_struct("RouterManager")
            .field("routers_count", &self.routers.len())
            .field("workers_count", &self.worker_registry.get_all().len())
            .field("default_router", &*default_router)
            .finish()
    }
}
