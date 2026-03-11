use std::{
    any::Any,
    sync::{atomic::AtomicBool, Arc},
    time::Instant,
};

use axum::{body::Body, extract::Request, http::HeaderMap, response::Response};
use openai_protocol::{
    chat::ChatCompletionRequest,
    realtime_session::{
        RealtimeClientSecretCreateRequest, RealtimeSessionCreateRequest,
        RealtimeTranscriptionSessionCreateRequest,
    },
    responses::{ResponseInput, ResponseInputOutputItem, ResponsesGetParams, ResponsesRequest},
};
use serde_json::to_value;

use super::{
    chat::{self, RouterContext},
    context::{
        ComponentRefs, PayloadState, RequestContext, ResponsesComponents, SharedComponents,
        WorkerSelection,
    },
    health,
    provider::ProviderRegistry,
    responses::{handle_non_streaming_response, handle_streaming_response},
};
use crate::{
    app_context::AppContext,
    config::types::RetryConfig,
    core::{Endpoint, ProviderType, Worker, WorkerRegistry},
    observability::metrics::{bool_to_static_str, metrics_labels, Metrics},
    routers::{
        error,
        header_utils::extract_auth_header,
        openai::realtime::{rest::forward_realtime_rest, ws::handle_realtime_ws, RealtimeRegistry},
        worker_selection::{SelectWorkerRequest, WorkerSelector},
    },
};

/// Resolve the provider implementation for a given worker and model.
///
/// Checks (in order): worker's per-model provider, model name heuristic,
/// then falls back to the default provider.
pub(super) fn resolve_provider(
    registry: &ProviderRegistry,
    worker: &dyn Worker,
    model: &str,
) -> Arc<dyn super::provider::Provider> {
    if let Some(pt) = worker.provider_for_model(model) {
        return registry.get_arc(pt);
    }
    if let Some(pt) = ProviderType::from_model_name(model) {
        return registry.get_arc(&pt);
    }
    registry.default_provider_arc()
}

pub struct OpenAIRouter {
    worker_registry: Arc<WorkerRegistry>,
    provider_registry: ProviderRegistry,
    healthy: AtomicBool,
    shared_components: Arc<SharedComponents>,
    responses_components: Arc<ResponsesComponents>,
    retry_config: RetryConfig,
    realtime_registry: Arc<RealtimeRegistry>,
}

impl std::fmt::Debug for OpenAIRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let registry_stats = self.worker_registry.stats();
        f.debug_struct("OpenAIRouter")
            .field("registered_workers", &registry_stats.total_workers)
            .field("registered_models", &registry_stats.total_models)
            .field("healthy_workers", &registry_stats.healthy_workers)
            .field("healthy", &self.healthy)
            .finish()
    }
}

impl OpenAIRouter {
    #[expect(
        clippy::unused_async,
        reason = "async for API consistency with other router constructors"
    )]
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        let worker_registry = ctx.worker_registry.clone();
        let mcp_orchestrator = ctx
            .mcp_orchestrator
            .get()
            .ok_or_else(|| "MCP manager not initialized in AppContext".to_string())?
            .clone();

        let shared_components = Arc::new(SharedComponents {
            client: ctx.client.clone(),
        });

        let responses_components = Arc::new(ResponsesComponents {
            shared: SharedComponents {
                client: ctx.client.clone(),
            },
            mcp_orchestrator: mcp_orchestrator.clone(),
            response_storage: ctx.response_storage.clone(),
            conversation_storage: ctx.conversation_storage.clone(),
            conversation_item_storage: ctx.conversation_item_storage.clone(),
        });

        Ok(Self {
            worker_registry,
            provider_registry: ProviderRegistry::new(),
            healthy: AtomicBool::new(true),
            shared_components,
            responses_components,
            retry_config: ctx.router_config.effective_retry_config(),
            realtime_registry: ctx.realtime_registry.clone(),
        })
    }

    async fn select_worker(
        &self,
        model_id: &str,
        headers: Option<&HeaderMap>,
    ) -> Result<Arc<dyn Worker>, Response> {
        WorkerSelector::new(&self.worker_registry, &self.shared_components.client)
            .select_worker(&SelectWorkerRequest {
                model_id,
                headers,
                provider: Some(ProviderType::OpenAI),
                ..Default::default()
            })
            .await
    }

    fn responses_components(&self) -> Arc<ResponsesComponents> {
        Arc::clone(&self.responses_components)
    }
}

#[async_trait::async_trait]
impl crate::routers::RouterTrait for OpenAIRouter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        health::health_generate(&self.worker_registry)
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        health::get_server_info(&self.worker_registry)
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let deps = RouterContext {
            worker_registry: &self.worker_registry,
            provider_registry: &self.provider_registry,
            shared_components: &self.shared_components,
            client: &self.shared_components.client,
            retry_config: &self.retry_config,
        };
        chat::route_chat(&deps, headers, body, model_id).await
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        let start = Instant::now();
        let model = model_id.unwrap_or(body.model.as_str());
        let streaming = body.stream.unwrap_or(false);

        Metrics::record_router_request(
            metrics_labels::ROUTER_OPENAI,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model,
            metrics_labels::ENDPOINT_RESPONSES,
            bool_to_static_str(streaming),
        );

        let worker = match self.select_worker(model, headers).await {
            Ok(w) => w,
            Err(response) => {
                Metrics::record_router_error(
                    metrics_labels::ROUTER_OPENAI,
                    metrics_labels::BACKEND_EXTERNAL,
                    metrics_labels::CONNECTION_HTTP,
                    model,
                    metrics_labels::ENDPOINT_RESPONSES,
                    metrics_labels::ERROR_NO_WORKERS,
                );
                return response;
            }
        };

        // Validate mutual exclusivity of conversation and previous_response_id
        // Treat empty strings as unset to match other metadata paths
        let has_conversation = body.conversation.as_ref().is_some_and(|s| !s.is_empty());
        let has_previous_response = body
            .previous_response_id
            .as_ref()
            .is_some_and(|s| !s.is_empty());
        if has_conversation && has_previous_response {
            Metrics::record_router_error(
                metrics_labels::ROUTER_OPENAI,
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::CONNECTION_HTTP,
                model,
                metrics_labels::ENDPOINT_RESPONSES,
                metrics_labels::ERROR_VALIDATION,
            );
            return error::bad_request(
                "invalid_request",
                "Cannot specify both 'conversation' and 'previous_response_id'".to_string(),
            );
        }

        let mut request_body = body.clone();
        if let Some(model) = model_id {
            request_body.model = model.to_string();
        }
        request_body.conversation = None;

        let original_previous_response_id = match super::responses::history::load_input_history(
            &self.responses_components,
            body,
            &mut request_body,
            model,
        )
        .await
        {
            Ok(id) => id,
            Err(response) => return response,
        };

        request_body.store = Some(false);
        if let ResponseInput::Items(ref mut items) = request_body.input {
            items.retain(|item| !matches!(item, ResponseInputOutputItem::Reasoning { .. }));
        }

        let mut payload = match to_value(&request_body) {
            Ok(v) => v,
            Err(e) => {
                Metrics::record_router_error(
                    metrics_labels::ROUTER_OPENAI,
                    metrics_labels::BACKEND_EXTERNAL,
                    metrics_labels::CONNECTION_HTTP,
                    model,
                    metrics_labels::ENDPOINT_RESPONSES,
                    metrics_labels::ERROR_VALIDATION,
                );
                return error::bad_request(
                    "invalid_request",
                    format!("Failed to serialize request: {e}"),
                );
            }
        };

        let provider = resolve_provider(&self.provider_registry, worker.as_ref(), model);
        if let Err(e) = provider.transform_request(&mut payload, Endpoint::Responses) {
            Metrics::record_router_error(
                metrics_labels::ROUTER_OPENAI,
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::CONNECTION_HTTP,
                model,
                metrics_labels::ENDPOINT_RESPONSES,
                metrics_labels::ERROR_VALIDATION,
            );
            return error::bad_request("invalid_request", format!("Provider transform error: {e}"));
        }

        let mut ctx = RequestContext::for_responses(
            Arc::new(body.clone()),
            headers.cloned(),
            model_id.map(String::from),
            ComponentRefs::Responses(self.responses_components()),
        );

        ctx.state.worker = Some(WorkerSelection {
            worker: Arc::clone(&worker),
            provider: Arc::clone(&provider),
        });

        ctx.state.payload = Some(PayloadState {
            json: payload,
            url: format!("{}/v1/responses", worker.url()),
            previous_response_id: original_previous_response_id,
        });

        let response = if ctx.is_streaming() {
            handle_streaming_response(ctx).await
        } else {
            handle_non_streaming_response(ctx).await
        };

        if response.status().is_success() {
            Metrics::record_router_duration(
                metrics_labels::ROUTER_OPENAI,
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::CONNECTION_HTTP,
                model,
                metrics_labels::ENDPOINT_RESPONSES,
                start.elapsed(),
            );
        }

        response
    }

    async fn get_response(
        &self,
        _headers: Option<&HeaderMap>,
        response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        super::responses::get_response(&self.responses_components, response_id).await
    }

    async fn list_response_input_items(
        &self,
        _headers: Option<&HeaderMap>,
        response_id: &str,
    ) -> Response {
        super::responses::list_response_input_items(&self.responses_components, response_id).await
    }

    async fn route_realtime_session(
        &self,
        headers: Option<&HeaderMap>,
        body: &RealtimeSessionCreateRequest,
    ) -> Response {
        // TODO(Phase 3): Inject MCP tool definitions into body.tools
        let model = body.model.as_deref().unwrap_or_default();
        let worker = self.select_worker(model, headers).await;
        forward_realtime_rest(
            &self.shared_components.client,
            worker,
            headers,
            body,
            model,
            "/v1/realtime/sessions",
            metrics_labels::ENDPOINT_REALTIME_SESSIONS,
        )
        .await
    }

    async fn route_realtime_client_secret(
        &self,
        headers: Option<&HeaderMap>,
        body: &RealtimeClientSecretCreateRequest,
    ) -> Response {
        // TODO(Phase 3): Inject MCP tool definitions into body.session.tools
        let model = body.session.model.as_deref().unwrap_or_default();
        let worker = self.select_worker(model, headers).await;
        forward_realtime_rest(
            &self.shared_components.client,
            worker,
            headers,
            body,
            model,
            "/v1/realtime/client_secrets",
            metrics_labels::ENDPOINT_REALTIME_CLIENT_SECRETS,
        )
        .await
    }

    async fn route_realtime_transcription_session(
        &self,
        headers: Option<&HeaderMap>,
        body: &RealtimeTranscriptionSessionCreateRequest,
    ) -> Response {
        let model = body.model.as_deref().unwrap_or_default();
        let worker = self.select_worker(model, headers).await;
        forward_realtime_rest(
            &self.shared_components.client,
            worker,
            headers,
            body,
            model,
            "/v1/realtime/transcription_sessions",
            metrics_labels::ENDPOINT_REALTIME_TRANSCRIPTION,
        )
        .await
    }

    async fn route_realtime_ws(&self, req: Request<Body>, model: &str) -> Response {
        let (parts, _body) = req.into_parts();

        Metrics::record_router_request(
            metrics_labels::ROUTER_OPENAI,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_WEBSOCKET,
            model,
            metrics_labels::ENDPOINT_REALTIME,
            "false",
        );

        let auth_header = extract_auth_header(Some(&parts.headers), None);
        let worker = self.select_worker(model, Some(&parts.headers)).await;

        handle_realtime_ws(
            parts,
            model.to_owned(),
            worker,
            auth_header,
            Arc::clone(&self.realtime_registry),
        )
        .await
    }

    fn router_type(&self) -> &'static str {
        "openai"
    }
}
