use std::{
    future::Future,
    io,
    path::PathBuf,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::{
    extract::{multipart::MultipartError, Extension, Multipart, Path, Query, Request, State},
    http::{header::InvalidHeaderName, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use axum_server::{
    accept::Accept,
    tls_rustls::{RustlsAcceptor, RustlsConfig},
};
use llm_tokenizer::TokenizerRegistry;
use openai_protocol::{
    chat::ChatCompletionRequest,
    classify::ClassifyRequest,
    completion::CompletionRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    interactions::InteractionsRequest,
    messages::CreateMessageRequest,
    parser::{ParseFunctionCallRequest, SeparateReasoningRequest},
    realtime_session::{
        RealtimeClientSecretCreateRequest, RealtimeSessionCreateRequest,
        RealtimeTranscriptionSessionCreateRequest,
    },
    rerank::{RerankRequest, V1RerankReqInput},
    responses::ResponsesRequest,
    skills::{
        SkillGetQuery, SkillPatchRequest, SkillVersionPatchRequest, SkillVersionsListQuery,
        SkillsListQuery,
    },
    tokenize::{AddTokenizerRequest, DetokenizeRequest, TokenizeRequest},
    transcription::TranscriptionRequest,
    validated::ValidatedJson,
    worker::{
        WorkerLoadInfo, WorkerLoadInfoSource, WorkerLoadsResult, WorkerSpec, WorkerUpdateRequest,
    },
};
use rustls::crypto::ring;
use serde::Deserialize;
use serde_json::{json, Value};
use smg_mesh::{
    MTLSConfig, MTLSManager, MeshServerBuilder, MeshServerConfig, MeshServerHandler,
    SpiffeIdentity, WorkerStateSubscriber,
};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    signal, spawn,
    sync::mpsc,
};
use tokio_rustls::server::TlsStream;
use tower::Layer;
use tracing::{debug, error, info, warn, Level};
use wfaas::LoggingSubscriber;

use crate::{
    app_context::AppContext,
    config::{RouterConfig, RoutingMode},
    cross_region::{
        config::seconds_to_millis_saturating, headers::REQUEST_MODE_HEADER,
        validate_settled_local_execution, AuthenticatedPeerIdentity, CrossRegionContext,
        CrossRegionError, CrossRegionHeaders, CrossRegionRuntimeConfig, CrossRegionState,
        CrossRegionSyncRuntime, RegionPeer, RegionPeerRegistry, RemoteRegionView,
        SettledRequestContext, UnresolvedRequestContext,
    },
    middleware::{self, AuthConfig, QueuedRequest},
    observability::{
        logging::{self, LoggingConfig},
        metrics::{self, PrometheusConfig},
        metrics_server,
        metrics_ws::{collectors, registry::WatchRegistry},
        otel_trace,
    },
    routers::{
        conversations,
        mesh::{
            get_app_config, get_cluster_status, get_global_rate_limit, get_global_rate_limit_stats,
            get_mesh_health, get_policy_state, get_policy_states, get_worker_state,
            get_worker_states, set_global_rate_limit, trigger_graceful_shutdown, update_app_config,
        },
        openai::realtime::ws::RealtimeQueryParams,
        parse, responses as response_handlers,
        router_manager::RouterManager,
        skills, tokenize, AudioFile, RouterTrait,
    },
    service_discovery::{start_service_discovery, ServiceDiscoveryConfig},
    wasm::route::{add_wasm_module, list_wasm_modules, remove_wasm_module},
    worker::{
        manager::{WorkerManager, WorkerManagerConfig},
        worker::WorkerType,
    },
    workflow::{
        job_queue::{JobQueue, JobQueueConfig},
        Job, TokenizerConfigRequest, WorkflowEngines,
    },
};
#[derive(Clone)]
pub struct AppState {
    pub router: Arc<dyn RouterTrait>,
    pub context: Arc<AppContext>,
    pub concurrency_queue_tx: Option<mpsc::Sender<QueuedRequest>>,
    pub router_manager: Option<Arc<RouterManager>>,
    pub mesh_handler: Option<Arc<MeshServerHandler>>,
    /// Cross-region sync plane runtime. `None` when cross_region is disabled.
    /// `Arc<_>` so cloning `AppState` does not duplicate `Drop` semantics on
    /// the spawned producer tasks (drop only fires when the last Arc dies).
    pub cross_region_sync: Option<Arc<CrossRegionSyncRuntime>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RequestListener {
    Local,
    Forwarding {
        peer_identity: Option<AuthenticatedPeerIdentity>,
    },
}

impl RequestListener {
    /// Return true when this request arrived through the forwarding listener.
    fn is_forwarding(&self) -> bool {
        matches!(self, Self::Forwarding { .. })
    }

    /// Consume the listener context and return the authenticated forwarding peer.
    fn into_forwarding_peer(self) -> Option<AuthenticatedPeerIdentity> {
        match self {
            Self::Local => None,
            Self::Forwarding { peer_identity } => peer_identity,
        }
    }
}

/// Request-forwarding TLS acceptor that injects the authenticated peer identity.
#[derive(Debug, Clone)]
struct ForwardingPeerIdentityAcceptor {
    inner: RustlsAcceptor,
}

impl ForwardingPeerIdentityAcceptor {
    /// Create a TLS acceptor for the request-forwarding listener.
    fn new(config: RustlsConfig) -> Self {
        Self {
            inner: RustlsAcceptor::new(config),
        }
    }
}

impl<I, S> Accept<I, S> for ForwardingPeerIdentityAcceptor
where
    I: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    S: Send + 'static,
    <Extension<AuthenticatedPeerIdentity> as Layer<S>>::Service: Send + 'static,
{
    type Stream = TlsStream<I>;
    type Service = <Extension<AuthenticatedPeerIdentity> as Layer<S>>::Service;
    type Future = Pin<Box<dyn Future<Output = io::Result<(Self::Stream, Self::Service)>> + Send>>;

    fn accept(&self, stream: I, service: S) -> Self::Future {
        let accept = self.inner.accept(stream, service);
        Box::pin(async move {
            let (stream, service) = accept.await?;
            let peer_identity = authenticated_identity_from_tls_stream(&stream)?;
            Ok((stream, Extension(peer_identity).layer(service)))
        })
    }
}

/// Extract and validate the peer's SMG SPIFFE identity from a TLS stream.
fn authenticated_identity_from_tls_stream<I>(
    stream: &TlsStream<I>,
) -> io::Result<AuthenticatedPeerIdentity> {
    let (_, session) = stream.get_ref();
    let peer_certs = session.peer_certificates().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::PermissionDenied,
            "request-forwarding mTLS peer certificate is required",
        )
    })?;
    let leaf_cert = peer_certs.first().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::PermissionDenied,
            "request-forwarding mTLS peer certificate chain is empty",
        )
    })?;
    let spiffe_identity =
        SpiffeIdentity::from_certificate_der(leaf_cert.as_ref()).map_err(|error| {
            io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("invalid request-forwarding SPIFFE identity: {error}"),
            )
        })?;

    AuthenticatedPeerIdentity::from_spiffe_identity(spiffe_identity).map_err(|error| {
        io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("invalid authenticated forwarding peer identity: {error}"),
        )
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum InferenceRequestDispatch {
    Local,
    Unresolved(UnresolvedRequestContext),
    Settled {
        context: SettledRequestContext,
        peer_identity: AuthenticatedPeerIdentity,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InferenceDispatchError {
    status: StatusCode,
    message: String,
}

impl InferenceDispatchError {
    /// Build an inference dispatch error with the response status and message.
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    /// Return the HTTP status for this dispatch error.
    fn status(&self) -> StatusCode {
        self.status
    }
}

impl From<CrossRegionError> for InferenceDispatchError {
    /// Convert cross-region parsing errors into dispatch errors.
    fn from(error: CrossRegionError) -> Self {
        Self::new(error.http_status(), error.to_string())
    }
}

impl IntoResponse for InferenceDispatchError {
    /// Convert the dispatch error into a plain HTTP response.
    fn into_response(self) -> Response {
        (self.status(), self.message).into_response()
    }
}

/// Classify an inference request before it enters local worker routing.
fn classify_inference_request(
    headers: &HeaderMap,
    listener: RequestListener,
    platform_max_retry: u32,
) -> Result<InferenceRequestDispatch, InferenceDispatchError> {
    if !headers.contains_key(REQUEST_MODE_HEADER) {
        if listener.is_forwarding() {
            return Err(InferenceDispatchError::new(
                StatusCode::FORBIDDEN,
                "forwarded inference requests must use SETTLED request mode",
            ));
        }
        return Ok(InferenceRequestDispatch::Local);
    }

    match CrossRegionHeaders::parse(headers, platform_max_retry)? {
        CrossRegionHeaders::Unresolved(context) => {
            if listener.is_forwarding() {
                return Err(InferenceDispatchError::new(
                    StatusCode::FORBIDDEN,
                    "forwarded inference requests must use SETTLED request mode",
                ));
            }
            Ok(InferenceRequestDispatch::Unresolved(context))
        }
        CrossRegionHeaders::Settled(context) => {
            if !listener.is_forwarding() {
                return Err(InferenceDispatchError::new(
                    StatusCode::FORBIDDEN,
                    "SETTLED cross-region requests must arrive on the trusted request-forwarding listener",
                ));
            }
            let peer_identity = listener.into_forwarding_peer().ok_or_else(|| {
                InferenceDispatchError::new(
                    StatusCode::FORBIDDEN,
                    "SETTLED cross-region requests require authenticated forwarding peer identity",
                )
            })?;
            Ok(InferenceRequestDispatch::Settled {
                context,
                peer_identity,
            })
        }
    }
}

/// Return the configured local region id required for settled execution.
fn local_cross_region_id(config: &RouterConfig) -> Result<&str, CrossRegionError> {
    config
        .cross_region
        .region_id
        .as_deref()
        .filter(|region| !region.trim().is_empty())
        .ok_or_else(|| CrossRegionError::InvalidConfig {
            reason: "cross_region.region_id is required for settled execution".to_string(),
        })
}

/// Build the inbound peer registry used to verify authenticated source regions.
fn settled_peer_registry(config: &RouterConfig) -> Result<RegionPeerRegistry, CrossRegionError> {
    let peers = config
        .cross_region
        .peers
        .iter()
        .map(RegionPeer::from_config)
        .collect::<Result<Vec<_>, _>>()?;
    RegionPeerRegistry::new(peers)
}

/// Build the mTLS manager used by cross-region request-forwarding and sync listeners.
fn cross_region_mtls_manager(config: &RouterConfig) -> Result<MTLSManager, CrossRegionError> {
    let Some(runtime_config) = CrossRegionRuntimeConfig::from_router_config(&config.cross_region)?
    else {
        return Err(CrossRegionError::InvalidConfig {
            reason: "cross_region must be enabled for mTLS listener config".to_string(),
        });
    };
    let mtls = runtime_config.mtls;
    Ok(MTLSManager::new(MTLSConfig {
        ca_cert_path: PathBuf::from(mtls.ca_cert_path),
        server_cert_path: PathBuf::from(mtls.server_cert_path),
        server_key_path: PathBuf::from(mtls.server_key_path),
        client_cert_path: PathBuf::from(mtls.client_cert_path),
        client_key_path: PathBuf::from(mtls.client_key_path),
        require_client_cert: true,
        ..MTLSConfig::default()
    }))
}

/// Validate settled request metadata before local router execution.
fn validate_settled_dispatch(
    state: &AppState,
    context: &SettledRequestContext,
    peer_identity: &AuthenticatedPeerIdentity,
    request_model_id: &str,
) -> Result<(), InferenceDispatchError> {
    let router_config = &state.context.router_config;
    let local_region_id = local_cross_region_id(router_config)?;
    let peer_registry = settled_peer_registry(router_config)?;
    validate_settled_local_execution(
        context,
        local_region_id,
        peer_identity,
        &peer_registry,
        request_model_id,
    )?;
    Ok(())
}

async fn parse_function_call(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ParseFunctionCallRequest>,
) -> Response {
    parse::parse_function_call(&state.context, &req).await
}

async fn parse_reasoning(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SeparateReasoningRequest>,
) -> Response {
    parse::parse_reasoning(&state.context, &req).await
}

async fn sink_handler() -> Response {
    StatusCode::NOT_FOUND.into_response()
}

async fn liveness() -> Response {
    (StatusCode::OK, "OK").into_response()
}

async fn readiness(State(state): State<Arc<AppState>>) -> Response {
    let workers = state.context.worker_registry.get_all();
    let healthy_workers: Vec<_> = workers.iter().filter(|w| w.is_healthy()).collect();

    let is_ready = if state.context.router_config.enable_igw {
        !healthy_workers.is_empty()
    } else {
        match &state.context.router_config.mode {
            RoutingMode::PrefillDecode { .. } => {
                let has_prefill = healthy_workers
                    .iter()
                    .any(|w| matches!(w.worker_type(), WorkerType::Prefill));
                let has_decode = healthy_workers
                    .iter()
                    .any(|w| matches!(w.worker_type(), WorkerType::Decode));
                has_prefill && has_decode
            }
            RoutingMode::Regular { .. } => !healthy_workers.is_empty(),
            RoutingMode::OpenAI { .. } => !healthy_workers.is_empty(),
            RoutingMode::Anthropic { .. } => !healthy_workers.is_empty(),
            RoutingMode::Gemini { .. } => !healthy_workers.is_empty(),
        }
    };

    if is_ready {
        (
            StatusCode::OK,
            Json(json!({
                "status": "ready",
                "healthy_workers": healthy_workers.len(),
                "total_workers": workers.len()
            })),
        )
            .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "not ready",
                "reason": "insufficient healthy workers"
            })),
        )
            .into_response()
    }
}

async fn health(_state: State<Arc<AppState>>) -> Response {
    liveness().await
}

async fn health_generate(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.health_generate(req).await
}

async fn engine_metrics(State(state): State<Arc<AppState>>) -> Response {
    WorkerManager::get_engine_metrics(&state.context.worker_registry, &state.context.client)
        .await
        .into_response()
}

async fn get_server_info(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_server_info(req).await
}

async fn v1_models(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_models(req).await
}

async fn get_model_info(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_model_info(req).await
}

async fn generate(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    Json(body): Json<GenerateRequest>,
) -> Response {
    state
        .router
        .route_generate(Some(&headers), &tenant_meta, &body, &body.model)
        .await
}

async fn v1_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    route_chat_completion_for_listener(state, headers, tenant_meta, body, RequestListener::Local)
        .await
}

async fn v1_chat_completions_forwarded(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    peer_identity: Option<Extension<AuthenticatedPeerIdentity>>,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    route_chat_completion_for_listener(
        state,
        headers,
        tenant_meta,
        body,
        RequestListener::Forwarding {
            peer_identity: peer_identity.map(|Extension(identity)| identity),
        },
    )
    .await
}

/// Dispatch chat-completions by cross-region request mode before local routing.
async fn route_chat_completion_for_listener(
    state: Arc<AppState>,
    headers: HeaderMap,
    tenant_meta: middleware::TenantRequestMeta,
    body: ChatCompletionRequest,
    listener: RequestListener,
) -> Response {
    let max_retry = state
        .context
        .router_config
        .cross_region
        .request_plane
        .max_platform_retries;
    match classify_inference_request(&headers, listener, max_retry) {
        Ok(InferenceRequestDispatch::Local) => {
            state
                .router
                .route_chat(Some(&headers), &tenant_meta, &body, &body.model)
                .await
        }
        Ok(InferenceRequestDispatch::Settled {
            context,
            peer_identity,
        }) => {
            if let Err(error) =
                validate_settled_dispatch(&state, &context, &peer_identity, &body.model)
            {
                return error.into_response();
            }
            debug!(
                route_id = %context.route.route_id,
                entry_region = %context.common.entry_region,
                target_region = %context.route.target_region,
                committed_model = %context.route.committed_model,
                "executing settled cross-region chat request locally"
            );
            state
                .router
                .route_chat(Some(&headers), &tenant_meta, &body, &body.model)
                .await
        }
        Ok(InferenceRequestDispatch::Unresolved(context)) => unresolved_request_plane_response(
            &state,
            "/v1/chat/completions",
            context.common.opc_request_id.as_str(),
        ),
        Err(error) => error.into_response(),
    }
}

async fn v1_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    ValidatedJson(body): ValidatedJson<CompletionRequest>,
) -> Response {
    state
        .router
        .route_completion(Some(&headers), &tenant_meta, &body, &body.model)
        .await
}

async fn rerank(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    ValidatedJson(body): ValidatedJson<RerankRequest>,
) -> Response {
    state
        .router
        .route_rerank(Some(&headers), &tenant_meta, &body, &body.model)
        .await
}

async fn v1_rerank(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    Json(body): Json<V1RerankReqInput>,
) -> Response {
    let rerank_body: RerankRequest = body.into();
    state
        .router
        .route_rerank(
            Some(&headers),
            &tenant_meta,
            &rerank_body,
            &rerank_body.model,
        )
        .await
}

async fn v1_responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    ValidatedJson(body): ValidatedJson<ResponsesRequest>,
) -> Response {
    route_responses_for_listener(state, headers, tenant_meta, body, RequestListener::Local).await
}

async fn v1_responses_forwarded(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    peer_identity: Option<Extension<AuthenticatedPeerIdentity>>,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    ValidatedJson(body): ValidatedJson<ResponsesRequest>,
) -> Response {
    route_responses_for_listener(
        state,
        headers,
        tenant_meta,
        body,
        RequestListener::Forwarding {
            peer_identity: peer_identity.map(|Extension(identity)| identity),
        },
    )
    .await
}

/// Dispatch responses requests by cross-region request mode before local routing.
async fn route_responses_for_listener(
    state: Arc<AppState>,
    headers: HeaderMap,
    tenant_meta: middleware::TenantRequestMeta,
    body: ResponsesRequest,
    listener: RequestListener,
) -> Response {
    let max_retry = state
        .context
        .router_config
        .cross_region
        .request_plane
        .max_platform_retries;
    match classify_inference_request(&headers, listener, max_retry) {
        Ok(InferenceRequestDispatch::Local) => {
            state
                .router
                .route_responses(Some(&headers), &tenant_meta, &body, &body.model)
                .await
        }
        Ok(InferenceRequestDispatch::Settled {
            context,
            peer_identity,
        }) => {
            if let Err(error) =
                validate_settled_dispatch(&state, &context, &peer_identity, &body.model)
            {
                return error.into_response();
            }
            debug!(
                route_id = %context.route.route_id,
                entry_region = %context.common.entry_region,
                target_region = %context.route.target_region,
                committed_model = %context.route.committed_model,
                "executing settled cross-region responses request locally"
            );
            state
                .router
                .route_responses(Some(&headers), &tenant_meta, &body, &body.model)
                .await
        }
        Ok(InferenceRequestDispatch::Unresolved(context)) => unresolved_request_plane_response(
            &state,
            "/v1/responses",
            context.common.opc_request_id.as_str(),
        ),
        Err(error) => error.into_response(),
    }
}

/// Return the explicit request-plane stub response for unresolved requests.
fn unresolved_request_plane_response(
    state: &AppState,
    path: &'static str,
    opc_request_id: &str,
) -> Response {
    if !state.context.router_config.cross_region.enabled
        || !state
            .context
            .router_config
            .cross_region
            .request_plane
            .enabled
    {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "cross-region request plane is disabled",
        )
            .into_response();
    }

    (
        StatusCode::NOT_IMPLEMENTED,
        format!(
            "cross-region request plane dispatch for {path} is not implemented yet; opc_request_id={opc_request_id}"
        ),
    )
        .into_response()
}

async fn v1_interactions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    ValidatedJson(body): ValidatedJson<InteractionsRequest>,
) -> Response {
    let model_id = body.model.as_deref().or(body.agent.as_deref());
    state
        .router
        .route_interactions(Some(&headers), &tenant_meta, &body, model_id)
        .await
}

async fn v1_embeddings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    Json(body): Json<EmbeddingRequest>,
) -> Response {
    state
        .router
        .route_embeddings(Some(&headers), &tenant_meta, &body, &body.model)
        .await
}

async fn v1_messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    ValidatedJson(body): ValidatedJson<CreateMessageRequest>,
) -> Response {
    state
        .router
        .route_messages(Some(&headers), &tenant_meta, &body, &body.model)
        .await
}

async fn v1_classify(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    Json(body): Json<ClassifyRequest>,
) -> Response {
    state
        .router
        .route_classify(Some(&headers), &tenant_meta, &body, &body.model)
        .await
}

async fn v1_audio_transcriptions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    mut multipart: Multipart,
) -> Response {
    let mut file_bytes: Option<bytes::Bytes> = None;
    let mut file_name: Option<String> = None;
    let mut file_content_type: Option<String> = None;
    let mut req = TranscriptionRequest::default();
    let mut timestamp_granularities: Vec<String> = Vec::new();

    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Failed to read multipart field: {e}"),
                )
                    .into_response();
            }
        };

        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                file_name = field.file_name().map(str::to_string);
                file_content_type = field.content_type().map(str::to_string);
                match field.bytes().await {
                    Ok(b) => file_bytes = Some(b),
                    Err(e) => {
                        return (
                            StatusCode::BAD_REQUEST,
                            format!("Failed to read audio file bytes: {e}"),
                        )
                            .into_response();
                    }
                }
            }
            "model" => match field.text().await {
                Ok(t) => req.model = t,
                Err(e) => return bad_text_field("model", e),
            },
            "language" => match field.text().await {
                Ok(t) => req.language = Some(t),
                Err(e) => return bad_text_field("language", e),
            },
            "prompt" => match field.text().await {
                Ok(t) => req.prompt = Some(t),
                Err(e) => return bad_text_field("prompt", e),
            },
            "response_format" => match field.text().await {
                Ok(t) => req.response_format = Some(t),
                Err(e) => return bad_text_field("response_format", e),
            },
            "temperature" => match field.text().await {
                Ok(t) => match t.trim().parse::<f32>() {
                    Ok(v) if v.is_finite() && (0.0..=1.0).contains(&v) => {
                        req.temperature = Some(v);
                    }
                    Ok(v) => {
                        return (
                            StatusCode::BAD_REQUEST,
                            format!(
                                "Invalid 'temperature' value: {v} (must be a finite number in [0.0, 1.0])"
                            ),
                        )
                            .into_response();
                    }
                    Err(e) => {
                        return (
                            StatusCode::BAD_REQUEST,
                            format!("Invalid 'temperature' value: {e}"),
                        )
                            .into_response();
                    }
                },
                Err(e) => return bad_text_field("temperature", e),
            },
            "timestamp_granularities" | "timestamp_granularities[]" => match field.text().await {
                Ok(t) => timestamp_granularities.push(t),
                Err(e) => return bad_text_field("timestamp_granularities", e),
            },
            "stream" => match field.text().await {
                Ok(t) => match t.as_str() {
                    "true" | "True" | "TRUE" | "1" => req.stream = Some(true),
                    "false" | "False" | "FALSE" | "0" => req.stream = Some(false),
                    other => {
                        return (
                            StatusCode::BAD_REQUEST,
                            format!("Invalid 'stream' value: '{other}' (expected true/false/1/0)"),
                        )
                            .into_response();
                    }
                },
                Err(e) => return bad_text_field("stream", e),
            },
            _ => {
                // Unknown field; drain to free resources but otherwise ignore.
                let _ = field.bytes().await;
            }
        }
    }

    // Reject blank/whitespace-only `model` before it reaches worker selection.
    if req.model.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, "Missing required 'model' field").into_response();
    }
    req.model = req.model.trim().to_string();
    let bytes = match file_bytes {
        Some(b) if !b.is_empty() => b,
        Some(_) => {
            return (StatusCode::BAD_REQUEST, "Uploaded 'file' part is empty").into_response();
        }
        None => {
            return (StatusCode::BAD_REQUEST, "Missing required 'file' part").into_response();
        }
    };

    if !timestamp_granularities.is_empty() {
        req.timestamp_granularities = Some(timestamp_granularities);
    }

    let audio = AudioFile {
        bytes,
        file_name: file_name.unwrap_or_else(|| "audio".to_string()),
        content_type: file_content_type,
    };

    state
        .router
        .route_audio_transcriptions(Some(&headers), &tenant_meta, &req, audio, &req.model)
        .await
}

fn bad_text_field(field: &str, e: MultipartError) -> Response {
    (
        StatusCode::BAD_REQUEST,
        format!("Failed to read '{field}' field: {e}"),
    )
        .into_response()
}

async fn v1_responses_get(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
) -> Response {
    response_handlers::get_response(&state.context.response_storage, &response_id).await
}

async fn v1_responses_cancel(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    state
        .router
        .cancel_response(Some(&headers), &response_id)
        .await
}

async fn v1_responses_delete(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
) -> Response {
    response_handlers::delete_response(&state.context.response_storage, &response_id).await
}

async fn v1_responses_list_input_items(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
) -> Response {
    response_handlers::list_response_input_items(&state.context.response_storage, &response_id)
        .await
}

async fn v1_conversations_create(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    conversations::create_conversation(&state.context.conversation_storage, body).await
}

async fn v1_conversations_get(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
) -> Response {
    conversations::get_conversation(&state.context.conversation_storage, &conversation_id).await
}

async fn v1_conversations_update(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    conversations::update_conversation(&state.context.conversation_storage, &conversation_id, body)
        .await
}

async fn v1_conversations_delete(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
) -> Response {
    conversations::delete_conversation(&state.context.conversation_storage, &conversation_id).await
}

#[derive(Deserialize, Default)]
struct ListItemsQuery {
    limit: Option<usize>,
    order: Option<String>,
    after: Option<String>,
}

async fn v1_conversations_list_items(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    Query(ListItemsQuery {
        limit,
        order,
        after,
    }): Query<ListItemsQuery>,
) -> Response {
    conversations::list_conversation_items(
        &state.context.conversation_storage,
        &state.context.conversation_item_storage,
        &conversation_id,
        limit,
        order.as_deref(),
        after.as_deref(),
    )
    .await
}

#[derive(Deserialize, Default)]
struct GetItemQuery {
    /// Additional fields to include in response (not yet implemented)
    include: Option<Vec<String>>,
}

async fn v1_conversations_create_items(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let memory_execution_context =
        middleware::build_memory_execution_context(&state.context.router_config, &headers);

    conversations::create_conversation_items_with_headers(
        &state.context.conversation_storage,
        &state.context.conversation_item_storage,
        &conversation_id,
        body,
        memory_execution_context,
    )
    .await
}

async fn v1_conversations_get_item(
    State(state): State<Arc<AppState>>,
    Path((conversation_id, item_id)): Path<(String, String)>,
    Query(query): Query<GetItemQuery>,
) -> Response {
    conversations::get_conversation_item(
        &state.context.conversation_storage,
        &state.context.conversation_item_storage,
        &conversation_id,
        &item_id,
        query.include,
    )
    .await
}

async fn v1_conversations_delete_item(
    State(state): State<Arc<AppState>>,
    Path((conversation_id, item_id)): Path<(String, String)>,
) -> Response {
    conversations::delete_conversation_item(
        &state.context.conversation_storage,
        &state.context.conversation_item_storage,
        &conversation_id,
        &item_id,
    )
    .await
}

async fn v1_realtime_webrtc(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RealtimeQueryParams>,
    req: Request,
) -> Response {
    // Model may come from query param (application/sdp) or session body
    // (multipart/form-data). Let the handler validate per content type.
    let model = params.model.unwrap_or_default();
    state.router.route_realtime_webrtc(req, &model).await
}

async fn v1_realtime_ws(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RealtimeQueryParams>,
    req: Request,
) -> Response {
    let model = match params.model {
        Some(m) if !m.trim().is_empty() => m,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                "Missing required 'model' query parameter",
            )
                .into_response();
        }
    };
    state.router.route_realtime_ws(req, &model).await
}

async fn v1_realtime_session(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<RealtimeSessionCreateRequest>,
) -> Response {
    state
        .router
        .route_realtime_session(Some(&headers), &body)
        .await
}

async fn v1_realtime_client_secret(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<RealtimeClientSecretCreateRequest>,
) -> Response {
    state
        .router
        .route_realtime_client_secret(Some(&headers), &body)
        .await
}

async fn v1_realtime_transcription_session(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<RealtimeTranscriptionSessionCreateRequest>,
) -> Response {
    state
        .router
        .route_realtime_transcription_session(Some(&headers), &body)
        .await
}

async fn flush_cache(State(state): State<Arc<AppState>>, _req: Request) -> Response {
    WorkerManager::flush_cache_all(&state.context.worker_registry, &state.context.client)
        .await
        .into_response()
}

async fn get_loads(State(state): State<Arc<AppState>>, _req: Request) -> Response {
    let mut loads =
        WorkerManager::get_all_worker_loads(&state.context.worker_registry, &state.context.client)
            .await;
    append_remote_worker_loads(&state, &mut loads);
    loads.into_response()
}

fn append_remote_worker_loads(state: &AppState, loads: &mut WorkerLoadsResult) {
    let Some(sync_runtime) = state.cross_region_sync.as_ref() else {
        return;
    };
    let Some(local_region) = state
        .context
        .router_config
        .cross_region
        .region_id
        .as_deref()
    else {
        return;
    };
    let sync = sync_runtime.sync();
    let remote_state = sync.state();
    let remote_state = remote_state.read();
    let max_age_ms = seconds_to_millis_saturating(
        state
            .context
            .router_config
            .cross_region
            .sync_plane
            .signal_stale_after_seconds,
    );
    // `total_workers`/`successful`/`failed` describe the local worker query
    // (see `WorkerManager::get_all_worker_loads`). Remote-region projections
    // are additive content under `loads.loads` and must not perturb those
    // counts.
    append_projected_remote_worker_loads(
        loads,
        project_remote_worker_loads(&remote_state, local_region, max_age_ms, now_ms()),
    );
}

fn append_projected_remote_worker_loads(
    loads: &mut WorkerLoadsResult,
    remote_loads: Vec<WorkerLoadInfo>,
) {
    loads.loads.extend(remote_loads);
}

/// Pure projection of `CrossRegionState` into `/get_loads`-shaped envelopes,
/// one per remote region with at least one fresh worker-load entry.
///
/// Each outer envelope is tagged `WorkerLoadInfoSource::RemoteSmg` and
/// carries a `region-peer/{region_id}` placeholder URL plus the per-worker
/// observations nested under `remote_workers`. The inner observations are
/// also tagged `RemoteSmg` — they were materialized via the cross-region
/// sync plane, not via the local worker registry.
fn project_remote_worker_loads(
    remote_state: &CrossRegionState,
    local_region: &str,
    max_age_ms: i64,
    now_ms: i64,
) -> Vec<WorkerLoadInfo> {
    let view = RemoteRegionView::new(remote_state, now_ms, max_age_ms);
    let mut envelopes = Vec::new();

    for region_id in view.regions() {
        if region_id == local_region {
            continue;
        }
        let mut remote_workers = Vec::new();
        let mut aggregate_load = 0isize;
        let mut aggregate_generated_at_ms = None;
        let mut aggregate_version = None;

        for worker_id in view.worker_ids(region_id) {
            let Some(worker) = view.worker(region_id, worker_id) else {
                continue;
            };
            for entry in worker.fresh_load_entries() {
                aggregate_load = aggregate_load.saturating_add(entry.total_load);
                aggregate_generated_at_ms = Some(
                    aggregate_generated_at_ms
                        .unwrap_or(i64::MIN)
                        .max(entry.generated_at_ms),
                );
                aggregate_version = Some(aggregate_version.unwrap_or(0).max(entry.version));
                remote_workers.push(WorkerLoadInfo {
                    worker: entry.worker_id.clone(),
                    worker_type: None,
                    load: entry.total_load,
                    details: None,
                    region_id: Some(entry.region_id),
                    worker_id: Some(entry.worker_id),
                    model_id: entry.model_id,
                    status: entry.status,
                    generated_at_ms: Some(entry.generated_at_ms),
                    version: Some(entry.version),
                    // Inner entries describe remote workers observed via the
                    // sync plane — not the local worker registry.
                    source: Some(WorkerLoadInfoSource::RemoteSmg),
                    remote_workers: None,
                });
            }
        }

        if remote_workers.is_empty() {
            continue;
        }
        remote_workers.sort_by(|a, b| {
            a.worker
                .cmp(&b.worker)
                .then_with(|| a.model_id.cmp(&b.model_id))
        });
        envelopes.push(WorkerLoadInfo {
            worker: format!("region-peer/{region_id}"),
            worker_type: None,
            load: aggregate_load,
            details: None,
            region_id: Some(region_id.to_string()),
            worker_id: None,
            model_id: None,
            status: None,
            generated_at_ms: aggregate_generated_at_ms,
            version: aggregate_version,
            source: Some(WorkerLoadInfoSource::RemoteSmg),
            remote_workers: Some(remote_workers),
        });
    }

    envelopes
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

async fn create_worker(
    State(state): State<Arc<AppState>>,
    Json(config): Json<WorkerSpec>,
) -> Response {
    match state.context.worker_service.create_worker(config).await {
        Ok(result) => result.into_response(),
        Err(err) => err.into_response(),
    }
}

async fn list_workers_rest(State(state): State<Arc<AppState>>) -> Response {
    state.context.worker_service.list_workers().into_response()
}

async fn get_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id_raw): Path<String>,
) -> Response {
    match state.context.worker_service.get_worker(&worker_id_raw) {
        Ok(result) => result.into_response(),
        Err(err) => err.into_response(),
    }
}

async fn delete_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id_raw): Path<String>,
) -> Response {
    match state
        .context
        .worker_service
        .delete_worker(&worker_id_raw)
        .await
    {
        Ok(result) => result.into_response(),
        Err(err) => err.into_response(),
    }
}

async fn update_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id_raw): Path<String>,
    Json(update): Json<WorkerUpdateRequest>,
) -> Response {
    match state
        .context
        .worker_service
        .update_worker(&worker_id_raw, update)
        .await
    {
        Ok(result) => result.into_response(),
        Err(err) => err.into_response(),
    }
}

async fn replace_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id_raw): Path<String>,
    Json(config): Json<WorkerSpec>,
) -> Response {
    match state
        .context
        .worker_service
        .replace_worker(&worker_id_raw, config)
        .await
    {
        Ok(result) => result.into_response(),
        Err(err) => err.into_response(),
    }
}

// ============================================================================
// Tokenize / Detokenize Handlers
// ============================================================================

async fn v1_tokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TokenizeRequest>,
) -> Response {
    tokenize::tokenize(&state.context.tokenizer_registry, request).await
}

async fn v1_detokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DetokenizeRequest>,
) -> Response {
    tokenize::detokenize(&state.context.tokenizer_registry, request).await
}

async fn v1_tokenizers_add(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AddTokenizerRequest>,
) -> Response {
    tokenize::add_tokenizer(&state.context, request).await
}

async fn v1_tokenizers_list(State(state): State<Arc<AppState>>) -> Response {
    tokenize::list_tokenizers(&state.context.tokenizer_registry).await
}

async fn v1_tokenizers_get(
    State(state): State<Arc<AppState>>,
    Path(tokenizer_id): Path<String>,
) -> Response {
    tokenize::get_tokenizer_info(&state.context, &tokenizer_id).await
}

async fn v1_tokenizers_status(
    State(state): State<Arc<AppState>>,
    Path(tokenizer_id): Path<String>,
) -> Response {
    tokenize::get_tokenizer_status(&state.context, &tokenizer_id).await
}

async fn v1_tokenizers_remove(
    State(state): State<Arc<AppState>>,
    Path(tokenizer_id): Path<String>,
) -> Response {
    tokenize::remove_tokenizer(&state.context, &tokenizer_id).await
}

async fn v1_skills_create(State(state): State<Arc<AppState>>, multipart: Multipart) -> Response {
    skills::create_skill(State(state), multipart).await
}

async fn v1_skills_list(
    State(state): State<Arc<AppState>>,
    query: Query<SkillsListQuery>,
    headers: HeaderMap,
) -> Response {
    skills::list_skills(State(state), query, headers).await
}

async fn v1_skills_get(
    State(state): State<Arc<AppState>>,
    Path(skill_id): Path<String>,
    query: Query<SkillGetQuery>,
    headers: HeaderMap,
) -> Response {
    skills::get_skill(State(state), Path(skill_id), query, headers).await
}

async fn v1_skills_patch(
    State(state): State<Arc<AppState>>,
    Path(skill_id): Path<String>,
    query: Query<SkillGetQuery>,
    ValidatedJson(body): ValidatedJson<SkillPatchRequest>,
) -> Response {
    skills::patch_skill(State(state), Path(skill_id), query, Json(body)).await
}

async fn v1_skills_create_version(
    State(state): State<Arc<AppState>>,
    Path(skill_id): Path<String>,
    multipart: Multipart,
) -> Response {
    skills::create_skill_version(State(state), Path(skill_id), multipart).await
}

async fn v1_skills_list_versions(
    State(state): State<Arc<AppState>>,
    Path(skill_id): Path<String>,
    query: Query<SkillVersionsListQuery>,
    headers: HeaderMap,
) -> Response {
    skills::list_skill_versions(State(state), Path(skill_id), query, headers).await
}

async fn v1_skills_get_version(
    State(state): State<Arc<AppState>>,
    Path((skill_id, version)): Path<(String, String)>,
    query: Query<SkillGetQuery>,
    headers: HeaderMap,
) -> Response {
    skills::get_skill_version(State(state), Path((skill_id, version)), query, headers).await
}

async fn v1_skills_patch_version(
    State(state): State<Arc<AppState>>,
    Path((skill_id, version)): Path<(String, String)>,
    query: Query<SkillGetQuery>,
    ValidatedJson(body): ValidatedJson<SkillVersionPatchRequest>,
) -> Response {
    skills::patch_skill_version(State(state), Path((skill_id, version)), query, Json(body)).await
}

async fn v1_skills_delete(
    State(state): State<Arc<AppState>>,
    Path(skill_id): Path<String>,
    query: Query<SkillGetQuery>,
) -> Response {
    skills::delete_skill(State(state), Path(skill_id), query).await
}

async fn v1_skills_delete_version(
    State(state): State<Arc<AppState>>,
    Path((skill_id, version)): Path<(String, String)>,
    query: Query<SkillGetQuery>,
) -> Response {
    skills::delete_skill_version(State(state), Path((skill_id, version)), query).await
}

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub router_config: RouterConfig,
    pub max_payload_size: usize,
    pub log_dir: Option<String>,
    pub log_level: Option<String>,
    pub log_json: bool,
    pub service_discovery_config: Option<ServiceDiscoveryConfig>,
    pub prometheus_config: Option<PrometheusConfig>,
    pub request_timeout_secs: u64,
    pub request_id_headers: Option<Vec<String>>,
    pub shutdown_grace_period_secs: u64,
    /// Control plane authentication configuration
    pub control_plane_auth: Option<smg_auth::ControlPlaneAuthConfig>,
    pub mesh_server_config: Option<MeshServerConfig>,
    /// Bind address for WebRTC UDP sockets.
    /// `None` means use the default (0.0.0.0, auto-detect candidate IP).
    pub webrtc_bind_addr: Option<std::net::IpAddr>,
    /// STUN server for ICE candidate gathering (host:port).
    /// `None` means use the default (stun.l.google.com:19302).
    pub webrtc_stun_server: Option<String>,
}

pub fn build_app(
    app_state: Arc<AppState>,
    auth_config: AuthConfig,
    control_plane_auth_state: Option<smg_auth::ControlPlaneAuthState>,
    max_payload_size: usize,
    request_id_headers: Vec<String>,
    cors_allowed_origins: Vec<String>,
) -> Result<Router, InvalidHeaderName> {
    // Pending (upgrade not completed): 30s TTL
    // Disconnected: 60 min TTL
    app_state.context.realtime_registry.start_reaper(
        Duration::from_secs(3600),
        Duration::from_secs(30),
        Duration::from_secs(60),
    );

    let tenant_resolution_state =
        middleware::TenantResolutionState::new(&app_state.context.router_config)?
            .with_tenant_alias_store(
                app_state
                    .context
                    .skill_service
                    .as_ref()
                    .and_then(|skill_service| skill_service.tenant_alias_store()),
            );

    let protected_routes = Router::new()
        .route("/v1/responses", post(v1_responses))
        .route("/v1/responses/{response_id}", get(v1_responses_get))
        .route(
            "/v1/responses/{response_id}/cancel",
            post(v1_responses_cancel),
        )
        .route("/v1/responses/{response_id}", delete(v1_responses_delete))
        .route(
            "/v1/responses/{response_id}/input_items",
            get(v1_responses_list_input_items),
        )
        .route("/v1/conversations", post(v1_conversations_create))
        .route(
            "/v1/conversations/{conversation_id}",
            get(v1_conversations_get)
                .post(v1_conversations_update)
                .delete(v1_conversations_delete),
        )
        .route(
            "/v1/conversations/{conversation_id}/items",
            get(v1_conversations_list_items).post(v1_conversations_create_items),
        )
        .route(
            "/v1/conversations/{conversation_id}/items/{item_id}",
            get(v1_conversations_get_item).delete(v1_conversations_delete_item),
        )
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::storage_context_middleware,
        ))
        .route("/generate", post(generate))
        .route("/v1/chat/completions", post(v1_chat_completions))
        .route("/v1/completions", post(v1_completions))
        .route("/rerank", post(rerank))
        .route("/v1/rerank", post(v1_rerank))
        .route("/v1/embeddings", post(v1_embeddings))
        .route("/v1/messages", post(v1_messages))
        .route("/v1/interactions", post(v1_interactions))
        .route("/v1/classify", post(v1_classify))
        // Tokenize / Detokenize endpoints
        .route("/v1/tokenize", post(v1_tokenize))
        .route("/v1/detokenize", post(v1_detokenize))
        // Realtime REST endpoints (same middleware as other protected routes)
        .route("/v1/realtime/sessions", post(v1_realtime_session))
        .route(
            "/v1/realtime/client_secrets",
            post(v1_realtime_client_secret),
        )
        .route(
            "/v1/realtime/transcription_sessions",
            post(v1_realtime_transcription_session),
        )
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::concurrency_limit_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            tenant_resolution_state.clone(),
            middleware::route_request_meta_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            auth_config.clone(),
            middleware::auth_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::wasm_middleware,
        ));

    // WebSocket and WebRTC routes: auth + concurrency but NO WASM middleware.
    // WASM OnResponse reconstructs the response from status/headers/body,
    // dropping the response extensions that carry the WebSocket upgrade future.
    let realtime_routes = Router::new()
        .route("/v1/realtime", get(v1_realtime_ws))
        .route("/v1/realtime/calls", post(v1_realtime_webrtc))
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::concurrency_limit_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            tenant_resolution_state.clone(),
            middleware::route_request_meta_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            auth_config.clone(),
            middleware::auth_middleware,
        ));

    // Multipart upload routes: auth + concurrency but NO WASM middleware.
    // The WASM OnRequest phase buffers the full body into a `Vec<u8>` subject
    // to the WASM manager's `max_body_size` (10MB default). Audio uploads
    // routinely exceed that, so WASM middleware would reject them with 400
    // before reaching the handler.
    let multipart_upload_routes = Router::new()
        .route("/v1/audio/transcriptions", post(v1_audio_transcriptions))
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::concurrency_limit_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            tenant_resolution_state,
            middleware::route_request_meta_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            auth_config.clone(),
            middleware::auth_middleware,
        ));

    let public_routes = Router::new()
        .route("/liveness", get(liveness))
        .route("/readiness", get(readiness))
        .route("/health", get(health))
        .route("/health_generate", get(health_generate))
        .route("/engine_metrics", get(engine_metrics))
        .route("/v1/models", get(v1_models))
        .route("/get_model_info", get(get_model_info))
        .route("/get_server_info", get(get_server_info));

    // Build admin routes with control plane auth if configured, otherwise use simple API key auth
    let mut admin_routes = Router::new()
        .route("/flush_cache", post(flush_cache))
        .route("/get_loads", get(get_loads))
        .route("/parse/function_call", post(parse_function_call))
        .route("/parse/reasoning", post(parse_reasoning))
        .route("/wasm", post(add_wasm_module))
        .route("/wasm/{module_uuid}", delete(remove_wasm_module))
        .route("/wasm", get(list_wasm_modules))
        // Tokenizer management endpoints
        .route(
            "/v1/tokenizers",
            post(v1_tokenizers_add).get(v1_tokenizers_list),
        )
        .route(
            "/v1/tokenizers/{tokenizer_id}",
            get(v1_tokenizers_get).delete(v1_tokenizers_remove),
        )
        .route(
            "/v1/tokenizers/{tokenizer_id}/status",
            get(v1_tokenizers_status),
        );

    if app_state.context.router_config.skills_enabled
        && app_state
            .context
            .router_config
            .skills
            .as_ref()
            .is_some_and(|skills_config| skills_config.admin.enabled)
        && app_state.context.skill_service.is_some()
    {
        admin_routes = admin_routes
            .route("/v1/skills", post(v1_skills_create).get(v1_skills_list))
            .route(
                "/v1/skills/{skill_id}",
                get(v1_skills_get)
                    .patch(v1_skills_patch)
                    .delete(v1_skills_delete),
            )
            .route(
                "/v1/skills/{skill_id}/versions",
                post(v1_skills_create_version).get(v1_skills_list_versions),
            )
            .route(
                "/v1/skills/{skill_id}/versions/{version}",
                get(v1_skills_get_version)
                    .patch(v1_skills_patch_version)
                    .delete(v1_skills_delete_version),
            );
    }

    // Build worker routes
    let worker_routes = Router::new()
        .route("/workers", post(create_worker).get(list_workers_rest))
        .route(
            "/workers/{worker_id}",
            get(get_worker)
                .put(replace_worker)
                .patch(update_worker)
                .delete(delete_worker),
        );

    // Apply authentication middleware to control plane routes
    let apply_control_plane_auth = |routes: Router<Arc<AppState>>| {
        if let Some(ref cp_state) = control_plane_auth_state {
            routes.route_layer(axum::middleware::from_fn_with_state(
                cp_state.clone(),
                smg_auth::control_plane_auth_middleware,
            ))
        } else {
            routes.route_layer(axum::middleware::from_fn_with_state(
                auth_config.clone(),
                middleware::auth_middleware,
            ))
        }
    };
    let admin_routes = apply_control_plane_auth(admin_routes);
    let worker_routes = apply_control_plane_auth(worker_routes);

    // HA management routes
    let mesh_routes = Router::new()
        .route("/ha/status", get(get_cluster_status))
        .route("/ha/health", get(get_mesh_health))
        .route("/ha/workers", get(get_worker_states))
        .route("/ha/workers/{worker_id}", get(get_worker_state))
        .route("/ha/policies", get(get_policy_states))
        .route("/ha/policies/{model_id}", get(get_policy_state))
        .route("/ha/config/{key}", get(get_app_config))
        .route("/ha/config", post(update_app_config))
        .route("/ha/rate-limit", post(set_global_rate_limit))
        .route("/ha/rate-limit", get(get_global_rate_limit))
        .route("/ha/rate-limit/stats", get(get_global_rate_limit_stats))
        .route("/ha/shutdown", post(trigger_graceful_shutdown))
        .route_layer(axum::middleware::from_fn_with_state(
            auth_config.clone(),
            middleware::auth_middleware,
        ));

    Ok(Router::new()
        .merge(protected_routes)
        .merge(realtime_routes)
        .merge(multipart_upload_routes)
        .merge(public_routes)
        .merge(admin_routes)
        .merge(worker_routes)
        .merge(mesh_routes)
        .layer(axum::extract::DefaultBodyLimit::max(max_payload_size))
        .layer(tower_http::limit::RequestBodyLimitLayer::new(
            max_payload_size,
        ))
        .layer(middleware::create_logging_layer())
        .layer(middleware::HttpMetricsLayer::new(
            app_state.context.inflight_tracker.clone(),
        ))
        .layer(middleware::RequestIdLayer::new(request_id_headers))
        .layer(create_cors_layer(cors_allowed_origins))
        .fallback(sink_handler)
        .with_state(app_state))
}

/// Build the request-forwarding listener app with only forwarded inference paths.
pub fn build_request_forwarding_app(
    app_state: Arc<AppState>,
    auth_config: AuthConfig,
    max_payload_size: usize,
    request_id_headers: Vec<String>,
    cors_allowed_origins: Vec<String>,
) -> Result<Router, InvalidHeaderName> {
    let tenant_resolution_state =
        middleware::TenantResolutionState::new(&app_state.context.router_config)?
            .with_tenant_alias_store(
                app_state
                    .context
                    .skill_service
                    .as_ref()
                    .and_then(|skill_service| skill_service.tenant_alias_store()),
            );

    let responses_routes = Router::new()
        .route("/v1/responses", post(v1_responses_forwarded))
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::storage_context_middleware,
        ));

    let forwarding_routes = Router::new()
        .merge(responses_routes)
        .route("/v1/chat/completions", post(v1_chat_completions_forwarded))
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::concurrency_limit_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            tenant_resolution_state,
            middleware::route_request_meta_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            auth_config,
            middleware::auth_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::wasm_middleware,
        ));

    Ok(Router::new()
        .merge(forwarding_routes)
        .layer(axum::extract::DefaultBodyLimit::max(max_payload_size))
        .layer(tower_http::limit::RequestBodyLimitLayer::new(
            max_payload_size,
        ))
        .layer(middleware::create_logging_layer())
        .layer(middleware::HttpMetricsLayer::new(
            app_state.context.inflight_tracker.clone(),
        ))
        .layer(middleware::RequestIdLayer::new(request_id_headers))
        .layer(create_cors_layer(cors_allowed_origins))
        .fallback(sink_handler)
        .with_state(app_state))
}

pub async fn startup(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    static LOGGING_INITIALIZED: AtomicBool = AtomicBool::new(false);

    if let Some(trace_config) = &config.router_config.trace_config {
        otel_trace::otel_tracing_init(
            trace_config.enable_trace,
            Some(&trace_config.otlp_traces_endpoint),
        )?;
    }

    let _log_guard = if LOGGING_INITIALIZED.swap(true, Ordering::SeqCst) {
        None
    } else {
        Some(logging::init_logging(
            LoggingConfig {
                level: config
                    .log_level
                    .as_deref()
                    .and_then(|s| match s.to_uppercase().parse::<Level>() {
                        Ok(l) => Some(l),
                        Err(_) => {
                            warn!("Invalid log level string: '{s}'. Defaulting to INFO.");
                            None
                        }
                    })
                    .unwrap_or(Level::INFO),
                json_format: config.log_json,
                log_dir: config.log_dir.clone(),
                colorize: true,
                log_file_name: "smg".to_string(),
                log_targets: None,
            },
            config.router_config.trace_config.clone(),
        ))
    };

    // Start metrics server and collectors.
    // Metrics server binds the port now; collectors start after AppContext is built.
    let (prometheus_handle, watch_registry) =
        if let Some(prometheus_config) = &config.prometheus_config {
            let handle = metrics::start_prometheus(prometheus_config.clone());
            let registry = Arc::new(WatchRegistry::new());
            let _server_handle = metrics_server::start_metrics_server(
                handle.clone(),
                prometheus_config.host.clone(),
                prometheus_config.port,
                registry.clone(),
                metrics_server::DEFAULT_MAX_WS_CONNECTIONS,
            )
            .await;
            (Some(handle), Some(registry))
        } else {
            (None, None)
        };

    // Initialize mesh server if configured, it will return a handler for mesh management
    let mesh_handler = if let Some(mesh_server_config) = &config.mesh_server_config {
        // Create mesh server builder and build with stores
        let (mesh_server, handler) = MeshServerBuilder::from(mesh_server_config).build();

        // Start rate limit window reset task (managed by handler)
        handler.start_rate_limit_task(1); // Reset every 1 second

        #[expect(
            clippy::disallowed_methods,
            reason = "mesh server runs for the lifetime of the process; shutdown is handled by the mesh handler"
        )]
        spawn(async move {
            if let Err(e) = mesh_server.start().await {
                tracing::error!("Mesh server failed: {}", e);
            }
        });

        Some(Arc::new(handler))
    } else {
        None
    };

    info!(
        "Starting router on {}:{} | mode: {:?} | policy: {:?} | max_payload: {}MB",
        config.host,
        config.port,
        config.router_config.mode,
        config.router_config.policy,
        config.max_payload_size / (1024 * 1024)
    );

    let app_context = Arc::new(
        AppContext::from_config(
            config.router_config.clone(),
            config.request_timeout_secs,
            config.webrtc_bind_addr,
            config.webrtc_stun_server.clone(),
        )
        .await?,
    );

    if config.prometheus_config.is_some() {
        app_context.inflight_tracker.start_sampler(20);
    }

    // Start WS metrics collectors now that AppContext is available.
    let _collector_handles = match (&prometheus_handle, &watch_registry) {
        (Some(handle), Some(registry)) => Some(collectors::start_collectors(
            app_context.clone(),
            registry.clone(),
            collectors::CollectorConfig::default(),
            handle.clone(),
        )),
        _ => None,
    };

    let weak_context = Arc::downgrade(&app_context);
    let worker_job_queue = JobQueue::new(JobQueueConfig::default(), weak_context);
    #[expect(
        clippy::expect_used,
        reason = "OnceLock initialization during startup; double-init is a fatal bug"
    )]
    app_context
        .worker_job_queue
        .set(worker_job_queue)
        .expect("JobQueue should only be initialized once");

    // Initialize typed workflow engines
    let engines = WorkflowEngines::new(&config.router_config);

    // Subscribe logging to all workflow engines
    engines.subscribe_all(Arc::new(LoggingSubscriber)).await;

    #[expect(
        clippy::expect_used,
        reason = "OnceLock initialization during startup; double-init is a fatal bug"
    )]
    app_context
        .workflow_engines
        .set(engines)
        .expect("WorkflowEngines should only be initialized once");
    debug!(
        "Workflow engines initialized (health check timeout: {}s)",
        config.router_config.health_check.timeout_secs
    );

    // Submit startup tokenizer job if tokenizer path is configured
    // This runs before worker initialization to ensure tokenizer is available
    if config.router_config.disable_tokenizer_autoload {
        info!("Tokenizer autoload disabled via config; skipping startup tokenizer load");
    } else if let Some(tokenizer_source) = config
        .router_config
        .tokenizer_path
        .as_ref()
        .or(config.router_config.model_path.as_ref())
    {
        info!("Loading startup tokenizer from: {}", tokenizer_source);

        #[expect(
            clippy::expect_used,
            reason = "JobQueue was just initialized above; absence is unreachable"
        )]
        let job_queue = app_context
            .worker_job_queue
            .get()
            .expect("JobQueue should be initialized");

        let tokenizer_config = TokenizerConfigRequest {
            id: TokenizerRegistry::generate_id(),
            name: tokenizer_source.clone(),
            source: tokenizer_source.clone(),
            chat_template_path: config.router_config.chat_template.clone(),
            cache_config: config.router_config.tokenizer_cache.to_option(),
            fail_on_duplicate: false,
        };

        let job = Job::AddTokenizer {
            config: Box::new(tokenizer_config),
        };

        job_queue
            .submit(job)
            .await
            .map_err(|e| format!("Failed to submit startup tokenizer job: {e}"))?;

        info!("Startup tokenizer job submitted (will complete in background)");
    }

    info!(
        "Initializing workers for routing mode: {:?}",
        config.router_config.mode
    );

    // Submit worker initialization job to queue
    #[expect(
        clippy::expect_used,
        reason = "JobQueue was initialized above; absence is unreachable"
    )]
    let job_queue = app_context
        .worker_job_queue
        .get()
        .expect("JobQueue should be initialized");
    let job = Job::InitializeWorkersFromConfig {
        router_config: Box::new(config.router_config.clone()),
    };
    job_queue
        .submit(job)
        .await
        .map_err(|e| format!("Failed to submit worker initialization job: {e}"))?;

    info!("Worker initialization job submitted (will complete in background)");

    if let Some(mcp_config) = &config.router_config.mcp_config {
        info!("Found {} MCP server(s) in config", mcp_config.servers.len());
        let mcp_job = Job::InitializeMcpServers {
            mcp_config: Box::new(mcp_config.clone()),
        };
        job_queue
            .submit(mcp_job)
            .await
            .map_err(|e| format!("Failed to submit MCP initialization job: {e}"))?;
    } else {
        info!("No MCP config provided, skipping MCP server initialization");
    }

    // Note: MCP orchestrator handles background refresh internally via refresh channel
    // configured by inventory.refresh_interval in mcp.yaml

    let worker_stats = app_context.worker_registry.stats();
    info!(
        "Workers initialized: {} total, {} healthy",
        worker_stats.total_workers, worker_stats.healthy_workers
    );

    let router_manager = RouterManager::from_config(&config, &app_context).await?;
    let router: Arc<dyn RouterTrait> = router_manager.clone();

    // WorkerManager owns the background health check loop. Its handle must
    // outlive the server to keep the task alive — bind it here so its Drop
    // (which aborts the task) runs at server shutdown.
    let _worker_manager = if config.router_config.health_check.disable_health_check {
        info!("Global health checks disabled via CLI/config; skipping WorkerManager");
        None
    } else {
        let manager = WorkerManager::start(
            app_context.worker_registry.clone(),
            WorkerManagerConfig {
                default_check_interval_secs: config.router_config.health_check.check_interval_secs,
                remove_unhealthy: config.router_config.health_check.remove_unhealthy_workers,
            },
            app_context.worker_job_queue.get().cloned(),
        );
        debug!(
            "Started WorkerManager health check loop with {}s default interval",
            config.router_config.health_check.check_interval_secs
        );
        Some(manager)
    };

    // WorkerMonitor subscribes to registry events. Starting its event
    // loop here (after the synchronous worker population in
    // RouterManager::from_config above) means the bootstrap reconcile
    // captures every worker that exists at this point and the event
    // task picks up everything registered afterwards.
    if let Some(ref worker_monitor) = app_context.worker_monitor {
        worker_monitor.start_event_loop();
        debug!("Started WorkerMonitor event loop");
    }

    let (limiter, processor) = middleware::ConcurrencyLimiter::new(
        app_context.rate_limiter.clone(),
        config.router_config.queue_size,
        Duration::from_secs(config.router_config.queue_timeout_secs),
    );

    if app_context.rate_limiter.is_none() {
        info!("Rate limiting is disabled (max_concurrent_requests = -1)");
    }

    match processor {
        Some(proc) => {
            #[expect(
                clippy::disallowed_methods,
                reason = "request queue processor runs for the lifetime of the server"
            )]
            spawn(proc.run());
            debug!(
                "Started request queue (size: {}, timeout: {}s)",
                config.router_config.queue_size, config.router_config.queue_timeout_secs
            );
        }
        None => {
            debug!(
                "Rate limiting enabled (max_concurrent_requests = {}, queue disabled)",
                config.router_config.max_concurrent_requests
            );
        }
    }

    // Set mesh sync manager to worker registry and policy registry if mesh is enabled
    // This allows these components to sync state across mesh nodes when mesh is enabled,
    // but they work independently without mesh when mesh is disabled.
    // Using thread-safe set_mesh_sync method that works with Arc-wrapped registries
    if let Some(ref handle) = mesh_handler {
        app_context
            .worker_registry
            .set_mesh_sync(Some(handle.sync_manager.clone()));
        handle
            .sync_manager
            .register_worker_state_subscriber(app_context.worker_registry.clone());
        // Replay workers already in the CRDT store — they arrived between
        // mesh server start and subscriber registration above.
        for state in handle.sync_manager.get_all_worker_states() {
            app_context.worker_registry.on_remote_worker_state(&state);
        }
        info!("Mesh sync manager set on worker registry");

        handle
            .sync_manager
            .register_tree_state_subscriber(app_context.policy_registry.clone());
        app_context
            .policy_registry
            .set_mesh_sync(Some(handle.sync_manager.clone()));
        info!("Mesh sync manager set on policy registry");
    }

    // Get mesh cluster state and port before moving mesh_handler into app_state
    let mesh_cluster_state = mesh_handler.as_ref().map(|h| h.state.clone());
    let mesh_port = config
        .mesh_server_config
        .as_ref()
        .map(|c| c.advertise_addr.port());

    // Start the cross-region signal sync runtime. Publishes producer signals
    // through the shared mesh broadcast stream (`cross_region:` prefix) and
    // spawns the subscriber that applies inbound envelopes into the
    // materialized `CrossRegionState`. Requires both `sync_plane.enabled`
    // and an active mesh handler — starting cross-region sync without mesh
    // is a misconfiguration we fail boot for rather than silently no-op.
    let cross_region_sync = match CrossRegionContext::from_router_config(
        &config.router_config.cross_region,
    ) {
        Ok(Some(context)) => {
            if context.config.sync_plane.enabled {
                let handler = mesh_handler.as_ref().ok_or_else(|| {
                    format!(
                        "Cross-region sync plane is enabled for region {} but mesh server is not running; \
                         cross-region signals require mesh transport",
                        context.config.region_id,
                    )
                })?;
                let runtime = CrossRegionSyncRuntime::start_with_mesh_kv(
                    &context,
                    handler.mesh_kv(),
                    app_context.worker_registry.clone(),
                )
                .map_err(|error| format!("Failed to start cross-region sync runtime: {error}"))?;
                info!(
                    region = %context.config.region_id,
                    server = %context.config.server_name,
                    "Cross-region signal sync runtime started over mesh",
                );
                Some(Arc::new(runtime))
            } else {
                info!(
                    region = %context.config.region_id,
                    "Cross-region sync plane disabled; skipping runtime",
                );
                None
            }
        }
        Ok(None) => None,
        Err(error) => {
            return Err(format!("Invalid cross-region runtime config: {error}").into());
        }
    };

    let app_state = Arc::new(AppState {
        router,
        context: app_context.clone(),
        concurrency_queue_tx: limiter.queue_tx.clone(),
        router_manager: Some(router_manager),
        mesh_handler,
        cross_region_sync,
    });
    if let Some(service_discovery_config) = config.service_discovery_config {
        if service_discovery_config.enabled {
            let app_context_arc = Arc::clone(&app_state.context);

            match start_service_discovery(
                service_discovery_config,
                app_context_arc,
                mesh_cluster_state,
                mesh_port,
            )
            .await
            {
                Ok(handle) => {
                    info!("Service discovery started");
                    #[expect(
                        clippy::disallowed_methods,
                        reason = "service discovery runs for the lifetime of the server"
                    )]
                    spawn(async move {
                        if let Err(e) = handle.await {
                            error!("Service discovery task failed: {:?}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to start service discovery: {e}");
                    warn!("Continuing without service discovery");
                }
            }
        }
    }

    info!(
        "Router ready | workers: {:?}",
        WorkerManager::get_worker_urls(&app_state.context.worker_registry)
    );

    let request_id_headers = config.request_id_headers.clone().unwrap_or_else(|| {
        vec![
            "x-request-id".to_string(),
            "x-correlation-id".to_string(),
            "x-trace-id".to_string(),
            "request-id".to_string(),
        ]
    });

    let auth_config = AuthConfig::new(config.router_config.api_key.clone());

    // Initialize control plane authentication if configured
    let control_plane_auth_state =
        smg_auth::ControlPlaneAuthState::try_init(config.control_plane_auth.as_ref()).await;

    let forwarding_app = if config.router_config.cross_region.enabled
        && config.router_config.cross_region.request_plane.enabled
    {
        Some(build_request_forwarding_app(
            app_state.clone(),
            auth_config.clone(),
            config.max_payload_size,
            request_id_headers.clone(),
            config.router_config.cors_allowed_origins.clone(),
        )?)
    } else {
        None
    };
    let forwarding_mtls_manager = if forwarding_app.is_some() {
        Some(
            cross_region_mtls_manager(&config.router_config)
                .map_err(|error| format!("Invalid request-forwarding mTLS config: {error}"))?,
        )
    } else {
        None
    };

    let app = build_app(
        app_state,
        auth_config,
        control_plane_auth_state,
        config.max_payload_size,
        request_id_headers,
        config.router_config.cors_allowed_origins.clone(),
    )?;

    // TcpListener::bind accepts &str and handles IPv4/IPv6 via ToSocketAddrs
    let bind_addr = format!("{}:{}", config.host, config.port);
    info!("Starting server on {}", bind_addr);

    // Parse address and set up graceful shutdown (common to both TLS and non-TLS)
    let addr: std::net::SocketAddr = bind_addr
        .parse()
        .map_err(|e| format!("Invalid address: {e}"))?;

    let handle = axum_server::Handle::new();
    let handle_clone = handle.clone();
    let forwarding_handle = if let Some(forwarding_app) = forwarding_app {
        let forwarding_mtls_manager = forwarding_mtls_manager.ok_or_else(|| {
            "request-forwarding mTLS manager missing for enabled forwarding listener".to_string()
        })?;
        let forwarding_bind_addr = format!(
            "{}:{}",
            config.host, config.router_config.cross_region.request_plane.listen_port
        );
        let forwarding_addr: std::net::SocketAddr = forwarding_bind_addr
            .parse()
            .map_err(|e| format!("Invalid request-forwarding address: {e}"))?;
        let listener = std::net::TcpListener::bind(forwarding_addr)
            .map_err(|e| format!("Failed to bind request-forwarding listener: {e}"))?;
        listener
            .set_nonblocking(true)
            .map_err(|e| format!("Failed to set request-forwarding listener nonblocking: {e}"))?;
        let forwarding_handle = axum_server::Handle::new();
        let server_handle = forwarding_handle.clone();
        let mtls_server_config =
            forwarding_mtls_manager
                .load_server_config()
                .await
                .map_err(|error| {
                    format!("Failed to load request-forwarding mTLS server config: {error}")
                })?;
        let forwarding_acceptor =
            ForwardingPeerIdentityAcceptor::new(RustlsConfig::from_config(mtls_server_config));
        let forwarding_server = axum_server::from_tcp(listener)
            .map_err(|e| format!("Failed to create request-forwarding listener: {e}"))?
            .acceptor(forwarding_acceptor)
            .handle(server_handle)
            .serve(forwarding_app.into_make_service_with_connect_info::<std::net::SocketAddr>());
        info!(
            "Starting cross-region request-forwarding listener on {}",
            forwarding_bind_addr
        );
        #[expect(
            clippy::disallowed_methods,
            reason = "request-forwarding listener runs for the lifetime of the server"
        )]
        spawn(async move {
            if let Err(error) = forwarding_server.await {
                error!("Cross-region request-forwarding listener failed: {}", error);
            }
        });
        Some(forwarding_handle)
    } else {
        None
    };

    let inflight_tracker = app_context.inflight_tracker.clone();
    let drain_timeout = Duration::from_secs(config.shutdown_grace_period_secs);
    #[expect(
        clippy::disallowed_methods,
        reason = "shutdown signal handler must outlive the server to trigger graceful shutdown"
    )]
    spawn(async move {
        shutdown_signal().await;

        // Phase 1: Gate — stop accepting new connections, mark as draining
        info!(
            in_flight = inflight_tracker.len(),
            "Beginning graceful shutdown: gating new connections"
        );
        inflight_tracker.begin_drain();
        handle_clone.graceful_shutdown(Some(drain_timeout));
        if let Some(handle) = forwarding_handle {
            handle.graceful_shutdown(Some(drain_timeout));
        }

        // Phase 2: Drain — wait for in-flight requests to complete
        // Re-check after gating to catch requests that arrived between the
        // snapshot and graceful_shutdown stopping the accept loop.
        if !inflight_tracker.is_empty() {
            let drained = inflight_tracker.wait_for_drain(drain_timeout).await;
            if drained {
                info!("All in-flight requests drained");
            } else {
                warn!(
                    remaining = inflight_tracker.len(),
                    timeout_secs = drain_timeout.as_secs(),
                    "Drain timed out, forcing shutdown with requests still in-flight"
                );
            }
        }
        // Phase 3: Teardown proceeds after axum server stops (in the main task)
    });

    let server_result = if let (Some(cert), Some(key)) = (
        &config.router_config.server_cert,
        &config.router_config.server_key,
    ) {
        info!("TLS enabled");
        ring::default_provider()
            .install_default()
            .map_err(|e| format!("Failed to install rustls ring provider: {e:?}"))?;

        let tls_config = RustlsConfig::from_pem(cert.clone(), key.clone())
            .await
            .map_err(|e| format!("Failed to create TLS config: {e}"))?;

        axum_server::bind_rustls(addr, tls_config)
            .handle(handle)
            .serve(app.into_make_service_with_connect_info::<std::net::SocketAddr>())
            .await
    } else {
        axum_server::bind(addr)
            .handle(handle)
            .serve(app.into_make_service_with_connect_info::<std::net::SocketAddr>())
            .await
    };

    // Graceful Shutdown

    info!("HTTP server stopped. Starting component cleanup...");

    // This triggers background task cancellation, waits for tools, and denies approvals
    if let Some(orchestrator) = app_context.mcp_orchestrator.get() {
        orchestrator.shutdown().await;
    }

    info!("Cleanup complete. Process exiting.");

    // Return original server error if any, otherwise Ok
    server_result.map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

#[expect(
    clippy::expect_used,
    reason = "signal handler installation is infallible on supported platforms; failure is fatal"
)]
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {
            info!("Received Ctrl+C, starting graceful shutdown");
        },
        () = terminate => {
            info!("Received terminate signal, starting graceful shutdown");
        },
    }
}

fn create_cors_layer(allowed_origins: Vec<String>) -> tower_http::cors::CorsLayer {
    use tower_http::cors::Any;

    let cors = if allowed_origins.is_empty() {
        tower_http::cors::CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
            .expose_headers(Any)
    } else {
        let origins: Vec<http::HeaderValue> = allowed_origins
            .into_iter()
            .filter_map(|origin| origin.parse().ok())
            .collect();

        tower_http::cors::CorsLayer::new()
            .allow_origin(origins)
            .allow_methods([
                http::Method::GET,
                http::Method::POST,
                http::Method::PATCH,
                http::Method::DELETE,
                http::Method::OPTIONS,
            ])
            .allow_headers([http::header::CONTENT_TYPE, http::header::AUTHORIZATION])
            .expose_headers([http::header::HeaderName::from_static("x-request-id")])
    };

    cors.max_age(Duration::from_secs(3600))
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, OnceLock,
        },
    };

    use async_trait::async_trait;
    use axum::{
        body::{to_bytes, Body},
        http::Request as HttpRequest,
    };
    use http::HeaderValue;
    use llm_tokenizer::registry::TokenizerRegistry;
    use openai_protocol::{
        chat::ChatCompletionRequest, responses::ResponsesRequest, validated::ValidatedJson,
    };
    use smg_data_connector::{
        current_request_context, MemoryConversationItemStorage, MemoryConversationStorage,
        MemoryResponseStorage, NoOpConversationMemoryWriter,
    };
    use tower::ServiceExt;

    use super::*;
    use crate::{
        config::CrossRegionPeerConfig,
        cross_region::headers::{
            ALLOWED_MODELS_HEADER, ALLOWED_REGIONS_HEADER, ATTEMPT_HEADER, COMMITTED_MODEL_HEADER,
            CONTRACT_VERSION_HEADER, ENTRY_REGION_HEADER, FAILOVER_MODE_HEADER, MAX_RETRY_HEADER,
            OPC_REQUEST_ID_HEADER, REQUEST_MODE_HEADER, REQUEST_MODE_SETTLED,
            REQUEST_MODE_UNRESOLVED, ROUTE_ID_HEADER, SOURCE_SERVICE_HEADER, TARGET_REGION_HEADER,
        },
        policies::PolicyRegistry,
        tenant::{canonical_tenant_key, TenantIdentity},
        worker::WorkerRegistry,
    };

    #[derive(Debug, Default)]
    struct DispatchSpyRouter {
        chat_calls: AtomicUsize,
        responses_calls: AtomicUsize,
        response_storage_context_hits: AtomicUsize,
    }

    #[async_trait]
    impl RouterTrait for DispatchSpyRouter {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        async fn route_chat(
            &self,
            _headers: Option<&HeaderMap>,
            _tenant_meta: &middleware::TenantRequestMeta,
            _body: &ChatCompletionRequest,
            _model_id: &str,
        ) -> Response {
            self.chat_calls.fetch_add(1, Ordering::SeqCst);
            (StatusCode::OK, "local chat").into_response()
        }

        async fn route_responses(
            &self,
            _headers: Option<&HeaderMap>,
            _tenant_meta: &middleware::TenantRequestMeta,
            _body: &ResponsesRequest,
            _model_id: &str,
        ) -> Response {
            self.responses_calls.fetch_add(1, Ordering::SeqCst);
            if current_request_context()
                .and_then(|context| {
                    context
                        .get("tenant_id")
                        .map(|tenant_id| tenant_id == "tenant-abc")
                })
                .unwrap_or(false)
            {
                self.response_storage_context_hits
                    .fetch_add(1, Ordering::SeqCst);
            }
            (StatusCode::OK, "local responses").into_response()
        }

        fn router_type(&self) -> &'static str {
            "dispatch-spy"
        }
    }

    /// Build a minimal app state for request-mode dispatch tests.
    fn test_state(router: Arc<DispatchSpyRouter>) -> Arc<AppState> {
        test_state_with_router_config(router, |_| {})
    }

    /// Build a minimal app state after applying test-specific router config.
    fn test_state_with_router_config(
        router: Arc<DispatchSpyRouter>,
        configure: impl FnOnce(&mut RouterConfig),
    ) -> Arc<AppState> {
        let mut router_config = RouterConfig::default();
        router_config.cross_region.enabled = true;
        router_config.cross_region.region_id = Some("us-ashburn-1".to_string());
        router_config.cross_region.request_plane.enabled = true;
        router_config.cross_region.peers = vec![CrossRegionPeerConfig {
            region_id: Some("us-chicago-1".to_string()),
            request_url: Some("https://smg-region-agent.us-chicago-1.internal:8443".to_string()),
            sync_url: Some("https://smg-region-agent.us-chicago-1.internal:9443".to_string()),
            realm: Some("oc1".to_string()),
            environment: Some("prod".to_string()),
            ..CrossRegionPeerConfig::default()
        }];
        configure(&mut router_config);

        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(router_config.policy.clone()));

        let context = Arc::new(
            AppContext::builder()
                .router_config(router_config)
                .client(reqwest::Client::new())
                .rate_limiter(None)
                .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
                .reasoning_parser_factory(None)
                .tool_parser_factory(None)
                .worker_registry(worker_registry)
                .policy_registry(policy_registry)
                .response_storage(Arc::new(MemoryResponseStorage::new()))
                .conversation_storage(Arc::new(MemoryConversationStorage::new()))
                .conversation_item_storage(Arc::new(MemoryConversationItemStorage::new()))
                .conversation_memory_writer(Arc::new(NoOpConversationMemoryWriter::new()))
                .worker_monitor(None)
                .worker_job_queue(Arc::new(OnceLock::new()))
                .workflow_engines(Arc::new(OnceLock::new()))
                .mcp_orchestrator(Arc::new(OnceLock::new()))
                .build()
                .expect("test app context should build"),
        );

        let router_trait: Arc<dyn RouterTrait> = router.clone();
        Arc::new(AppState {
            router: router_trait,
            context,
            concurrency_queue_tx: None,
            router_manager: None,
            mesh_handler: None,
            cross_region_sync: None,
        })
    }

    /// Build tenant metadata for direct handler calls.
    fn tenant_meta() -> middleware::TenantRequestMeta {
        middleware::TenantRequestMeta::new(canonical_tenant_key(TenantIdentity::Anonymous))
    }

    /// Build a minimal valid chat-completions request.
    fn chat_request() -> ChatCompletionRequest {
        chat_request_for_model("cohere.command-r-plus")
    }

    /// Build a minimal chat-completions request for a specific model.
    fn chat_request_for_model(model: &str) -> ChatCompletionRequest {
        serde_json::from_value(json!({
            "model": model,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .expect("chat request fixture should deserialize")
    }

    /// Build a minimal valid responses request.
    fn responses_request() -> ResponsesRequest {
        responses_request_for_model("cohere.command-r-plus")
    }

    /// Build a minimal responses request for a specific model.
    fn responses_request_for_model(model: &str) -> ResponsesRequest {
        serde_json::from_value(json!({
            "model": model,
            "input": "hello"
        }))
        .expect("responses request fixture should deserialize")
    }

    /// Add a static header value.
    fn insert(headers: &mut HeaderMap, name: &'static str, value: &'static str) {
        headers.insert(name, HeaderValue::from_static(value));
    }

    /// Build an authenticated peer identity for forwarding-listener tests.
    fn peer_identity(region_id: &str) -> AuthenticatedPeerIdentity {
        AuthenticatedPeerIdentity::new(
            region_id,
            format!(
                "spiffe://oraclecorp.com/oci/oc1/prod/region/{region_id}/service/smg-region-agent"
            ),
        )
        .expect("peer identity should build")
    }

    /// Build a forwarding listener context with the configured source peer.
    fn forwarding_listener() -> RequestListener {
        RequestListener::Forwarding {
            peer_identity: Some(peer_identity("us-chicago-1")),
        }
    }

    /// Build a forwarding listener context without authenticated mTLS identity.
    fn forwarding_listener_without_peer() -> RequestListener {
        RequestListener::Forwarding {
            peer_identity: None,
        }
    }

    /// Build valid unresolved cross-region headers.
    fn unresolved_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        insert(&mut headers, CONTRACT_VERSION_HEADER, "1");
        insert(&mut headers, REQUEST_MODE_HEADER, REQUEST_MODE_UNRESOLVED);
        insert(&mut headers, ENTRY_REGION_HEADER, "us-ashburn-1");
        insert(&mut headers, SOURCE_SERVICE_HEADER, "dp-api");
        insert(&mut headers, OPC_REQUEST_ID_HEADER, "opc-request-1");
        insert(
            &mut headers,
            ALLOWED_REGIONS_HEADER,
            "us-ashburn-1, us-chicago-1",
        );
        insert(&mut headers, ALLOWED_MODELS_HEADER, "cohere.command-r-plus");
        insert(&mut headers, FAILOVER_MODE_HEADER, "AUTO");
        insert(&mut headers, MAX_RETRY_HEADER, "3");
        headers
    }

    /// Build valid settled cross-region headers.
    fn settled_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        insert(&mut headers, CONTRACT_VERSION_HEADER, "1");
        insert(&mut headers, REQUEST_MODE_HEADER, REQUEST_MODE_SETTLED);
        insert(&mut headers, ENTRY_REGION_HEADER, "us-chicago-1");
        insert(&mut headers, SOURCE_SERVICE_HEADER, "smg");
        insert(&mut headers, OPC_REQUEST_ID_HEADER, "opc-request-1");
        insert(&mut headers, TARGET_REGION_HEADER, "us-ashburn-1");
        insert(
            &mut headers,
            COMMITTED_MODEL_HEADER,
            "cohere.command-r-plus",
        );
        insert(&mut headers, ROUTE_ID_HEADER, "route-1");
        insert(&mut headers, ATTEMPT_HEADER, "1");
        headers
    }

    /// Read a response body as UTF-8 text for assertions.
    async fn body_text(response: Response) -> String {
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should read");
        String::from_utf8(bytes.to_vec()).expect("response body should be UTF-8")
    }

    /// Build an HTTP request for exercising the forwarding Axum app.
    fn forwarded_responses_http_request(headers: HeaderMap) -> HttpRequest<Body> {
        let mut request = HttpRequest::builder()
            .method(http::Method::POST)
            .uri("/v1/responses")
            .header(http::header::CONTENT_TYPE, "application/json")
            .header("x-tenant-id", "tenant-abc")
            .body(Body::from(
                r#"{"model":"cohere.command-r-plus","input":"hello"}"#,
            ))
            .expect("forwarded responses request should build");

        for (name, value) in &headers {
            request.headers_mut().insert(name.clone(), value.clone());
        }

        request
    }

    #[test]
    fn request_mode_classifier_handles_ordinary_unresolved_and_settled() {
        assert!(matches!(
            classify_inference_request(&HeaderMap::new(), RequestListener::Local, 5),
            Ok(InferenceRequestDispatch::Local)
        ));
        assert!(matches!(
            classify_inference_request(&unresolved_headers(), RequestListener::Local, 5),
            Ok(InferenceRequestDispatch::Unresolved(_))
        ));
        assert!(matches!(
            classify_inference_request(&settled_headers(), forwarding_listener(), 5),
            Ok(InferenceRequestDispatch::Settled { .. })
        ));
    }

    #[test]
    fn settled_request_is_rejected_on_local_listener() {
        let error = classify_inference_request(&settled_headers(), RequestListener::Local, 5)
            .expect_err("settled request must require forwarding listener");

        assert_eq!(error.status(), StatusCode::FORBIDDEN);
    }

    #[test]
    fn forwarding_listener_rejects_missing_and_unresolved_request_mode() {
        let missing_mode =
            classify_inference_request(&HeaderMap::new(), forwarding_listener_without_peer(), 5)
                .expect_err("forwarding listener must require settled mode");
        let unresolved =
            classify_inference_request(&unresolved_headers(), forwarding_listener(), 5)
                .expect_err("forwarding listener must reject unresolved mode");

        assert_eq!(missing_mode.status(), StatusCode::FORBIDDEN);
        assert_eq!(unresolved.status(), StatusCode::FORBIDDEN);
    }

    #[test]
    fn forwarding_listener_rejects_settled_without_authenticated_peer_identity() {
        let error =
            classify_inference_request(&settled_headers(), forwarding_listener_without_peer(), 5)
                .expect_err("settled request should require authenticated peer identity");

        assert_eq!(error.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn ordinary_chat_request_uses_existing_local_router_path() {
        let router = Arc::new(DispatchSpyRouter::default());
        let state = test_state(router.clone());

        let response = v1_chat_completions(
            State(state),
            HeaderMap::new(),
            Extension(tenant_meta()),
            ValidatedJson(chat_request()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_text(response).await, "local chat");
        assert_eq!(router.chat_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn unresolved_chat_request_enters_cross_region_request_plane_stub() {
        let router = Arc::new(DispatchSpyRouter::default());
        let state = test_state(router.clone());

        let response = v1_chat_completions(
            State(state),
            unresolved_headers(),
            Extension(tenant_meta()),
            ValidatedJson(chat_request()),
        )
        .await;
        let status = response.status();
        let body = body_text(response).await;

        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert!(body.contains("cross-region request plane"));
        assert_eq!(router.chat_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn settled_chat_request_on_local_listener_is_rejected_before_local_router() {
        let router = Arc::new(DispatchSpyRouter::default());
        let state = test_state(router.clone());

        let response = v1_chat_completions(
            State(state),
            settled_headers(),
            Extension(tenant_meta()),
            ValidatedJson(chat_request()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert_eq!(router.chat_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn settled_chat_request_on_forwarding_listener_executes_locally() {
        let router = Arc::new(DispatchSpyRouter::default());
        let state = test_state(router.clone());

        let response = v1_chat_completions_forwarded(
            State(state),
            settled_headers(),
            Some(Extension(peer_identity("us-chicago-1"))),
            Extension(tenant_meta()),
            ValidatedJson(chat_request()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_text(response).await, "local chat");
        assert_eq!(router.chat_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn settled_chat_request_without_forwarding_identity_is_rejected_before_local_router() {
        let router = Arc::new(DispatchSpyRouter::default());
        let state = test_state(router.clone());

        let response = v1_chat_completions_forwarded(
            State(state),
            settled_headers(),
            None,
            Extension(tenant_meta()),
            ValidatedJson(chat_request()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert_eq!(router.chat_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn settled_chat_request_with_target_mismatch_is_rejected_before_local_router() {
        let router = Arc::new(DispatchSpyRouter::default());
        let state = test_state(router.clone());
        let mut headers = settled_headers();
        insert(&mut headers, TARGET_REGION_HEADER, "us-phoenix-1");

        let response = v1_chat_completions_forwarded(
            State(state),
            headers,
            Some(Extension(peer_identity("us-chicago-1"))),
            Extension(tenant_meta()),
            ValidatedJson(chat_request()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(router.chat_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn settled_chat_request_with_peer_region_mismatch_is_rejected() {
        let router = Arc::new(DispatchSpyRouter::default());
        let state = test_state(router.clone());

        let response = v1_chat_completions_forwarded(
            State(state),
            settled_headers(),
            Some(Extension(peer_identity("us-phoenix-1"))),
            Extension(tenant_meta()),
            ValidatedJson(chat_request()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert_eq!(router.chat_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn settled_chat_request_delegates_model_supportability_to_local_router() {
        let router = Arc::new(DispatchSpyRouter::default());
        let state = test_state(router.clone());
        let mut headers = settled_headers();
        insert(&mut headers, COMMITTED_MODEL_HEADER, "dynamic-model");

        let response = v1_chat_completions_forwarded(
            State(state),
            headers,
            Some(Extension(peer_identity("us-chicago-1"))),
            Extension(tenant_meta()),
            ValidatedJson(chat_request_for_model("dynamic-model")),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(router.chat_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn settled_chat_request_with_model_mismatch_is_rejected() {
        let router = Arc::new(DispatchSpyRouter::default());
        let state = test_state(router.clone());

        let response = v1_chat_completions_forwarded(
            State(state),
            settled_headers(),
            Some(Extension(peer_identity("us-chicago-1"))),
            Extension(tenant_meta()),
            ValidatedJson(chat_request_for_model("different-model")),
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(router.chat_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn missing_mode_chat_request_on_forwarding_listener_is_rejected() {
        let router = Arc::new(DispatchSpyRouter::default());
        let state = test_state(router.clone());

        let response = v1_chat_completions_forwarded(
            State(state),
            HeaderMap::new(),
            None,
            Extension(tenant_meta()),
            ValidatedJson(chat_request()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert_eq!(router.chat_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn ordinary_responses_request_uses_existing_local_router_path() {
        let router = Arc::new(DispatchSpyRouter::default());
        let state = test_state(router.clone());

        let response = v1_responses(
            State(state),
            HeaderMap::new(),
            Extension(tenant_meta()),
            ValidatedJson(responses_request()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_text(response).await, "local responses");
        assert_eq!(router.responses_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn unresolved_responses_request_enters_cross_region_request_plane_stub() {
        let router = Arc::new(DispatchSpyRouter::default());
        let state = test_state(router.clone());

        let response = v1_responses(
            State(state),
            unresolved_headers(),
            Extension(tenant_meta()),
            ValidatedJson(responses_request()),
        )
        .await;
        let status = response.status();
        let body = body_text(response).await;

        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert!(body.contains("cross-region request plane"));
        assert_eq!(router.responses_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn settled_responses_request_on_forwarding_listener_executes_locally() {
        let router = Arc::new(DispatchSpyRouter::default());
        let state = test_state(router.clone());

        let response = v1_responses_forwarded(
            State(state),
            settled_headers(),
            Some(Extension(peer_identity("us-chicago-1"))),
            Extension(tenant_meta()),
            ValidatedJson(responses_request()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_text(response).await, "local responses");
        assert_eq!(router.responses_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn missing_mode_responses_request_on_forwarding_listener_is_rejected() {
        let router = Arc::new(DispatchSpyRouter::default());
        let state = test_state(router.clone());

        let response = v1_responses_forwarded(
            State(state),
            HeaderMap::new(),
            None,
            Extension(tenant_meta()),
            ValidatedJson(responses_request()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert_eq!(router.responses_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn forwarded_responses_route_preserves_storage_context() {
        let router = Arc::new(DispatchSpyRouter::default());
        let state = test_state_with_router_config(router.clone(), |router_config| {
            router_config.storage_context_headers =
                HashMap::from([("x-tenant-id".to_string(), "tenant_id".to_string())]);
        });
        let app = build_request_forwarding_app(
            state,
            AuthConfig::new(None),
            1024 * 1024,
            Vec::new(),
            Vec::new(),
        )
        .expect("forwarding app should build");
        let mut request = forwarded_responses_http_request(settled_headers());
        request
            .extensions_mut()
            .insert(peer_identity("us-chicago-1"));

        let response = app
            .oneshot(request)
            .await
            .expect("forwarded request should complete");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_text(response).await, "local responses");
        assert_eq!(router.responses_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            router.response_storage_context_hits.load(Ordering::SeqCst),
            1
        );
    }

    #[tokio::test]
    async fn forwarding_app_rejects_settled_request_without_authenticated_peer_identity() {
        let router = Arc::new(DispatchSpyRouter::default());
        let state = test_state(router.clone());
        let app = build_request_forwarding_app(
            state,
            AuthConfig::new(None),
            1024 * 1024,
            Vec::new(),
            Vec::new(),
        )
        .expect("forwarding app should build");
        let request = forwarded_responses_http_request(settled_headers());

        let response = app
            .oneshot(request)
            .await
            .expect("forwarded request should complete");

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert_eq!(router.responses_calls.load(Ordering::SeqCst), 0);
    }

    mod remote_worker_loads_projection {
        //! Regression tests for `project_remote_worker_loads` (the pure
        //! projection inside `/get_loads`).
        //!
        //! Two earlier bugs the tests guard against:
        //!   1. Inner `remote_workers` entries were tagged `LocalWorker`
        //!      when they were observed via the cross-region sync plane.
        //!   2. `total_workers` / `successful` in the response were being
        //!      incremented by the number of projection *entries*, which
        //!      breaks the LOCAL-worker semantic of those counters.
        //!
        //! Both are fixed in the function under test; these tests pin the
        //! behavior so a future change can't silently regress them.

        use openai_protocol::worker::{WorkerLoadInfo, WorkerStatus};

        use super::super::{append_projected_remote_worker_loads, project_remote_worker_loads};
        use crate::cross_region::{CrossRegionState, SignalVersion, WorkerLoadSignal};

        const NOW_MS: i64 = 10_000_000;
        const WINDOW_MS: i64 = 30_000;
        const LOCAL_REGION: &str = "us-ashburn-1";

        fn version(version: u64, actor: &str) -> SignalVersion {
            SignalVersion {
                version,
                actor: actor.to_string(),
                updated_at_ms: NOW_MS - 1_000,
            }
        }

        fn upsert_load(
            state: &mut CrossRegionState,
            region: &str,
            worker: &str,
            server: &str,
            model_id: &str,
            load: isize,
            v: SignalVersion,
        ) {
            state.upsert_worker_load(
                WorkerLoadSignal {
                    region_id: region.to_string(),
                    worker_id: worker.to_string(),
                    server_name: server.to_string(),
                    load: WorkerLoadInfo {
                        worker: worker.to_string(),
                        worker_type: None,
                        load,
                        details: None,
                        region_id: Some(region.to_string()),
                        worker_id: Some(worker.to_string()),
                        model_id: Some(model_id.to_string()),
                        status: Some(WorkerStatus::Ready),
                        generated_at_ms: Some(NOW_MS - 1_000),
                        version: Some(v.version),
                        source: None,
                        remote_workers: None,
                    },
                },
                v,
            );
        }

        #[test]
        fn skips_local_region_and_groups_per_remote_region() {
            let mut state = CrossRegionState::new();
            // Local region's own entry — must be filtered out.
            upsert_load(
                &mut state,
                LOCAL_REGION,
                "local-w",
                "smg-local",
                "cohere.command-r-plus",
                1,
                version(1, "smg-local"),
            );
            // Remote region entry that should appear.
            upsert_load(
                &mut state,
                "us-chicago-1",
                "w1",
                "smg-chi-a",
                "cohere.command-r-plus",
                3,
                version(1, "smg-chi-a"),
            );

            let envelopes = project_remote_worker_loads(&state, LOCAL_REGION, WINDOW_MS, NOW_MS);

            assert_eq!(envelopes.len(), 1, "only the remote region should project");
            let envelope = &envelopes[0];
            assert_eq!(envelope.region_id.as_deref(), Some("us-chicago-1"));
            assert_eq!(envelope.worker, "region-peer/us-chicago-1");
            assert_eq!(envelope.load, 3);
        }

        #[test]
        fn inner_remote_workers_are_tagged_remote_smg_not_local() {
            // Regression for bug 1: inner entries described workers observed
            // via cross-region sync; they must NOT be `LocalWorker`.
            use openai_protocol::worker::WorkerLoadInfoSource;

            let mut state = CrossRegionState::new();
            upsert_load(
                &mut state,
                "us-chicago-1",
                "w1",
                "smg-chi-a",
                "cohere.command-r-plus",
                5,
                version(1, "smg-chi-a"),
            );

            let envelopes = project_remote_worker_loads(&state, LOCAL_REGION, WINDOW_MS, NOW_MS);

            assert_eq!(envelopes.len(), 1);
            let envelope = &envelopes[0];
            assert_eq!(envelope.source, Some(WorkerLoadInfoSource::RemoteSmg));
            let remote_workers = envelope
                .remote_workers
                .as_ref()
                .expect("envelope must carry per-worker observations");
            assert!(
                !remote_workers.is_empty(),
                "the envelope must include the projected worker",
            );
            for inner in remote_workers {
                assert_eq!(
                    inner.source,
                    Some(WorkerLoadInfoSource::RemoteSmg),
                    "inner observations must be tagged RemoteSmg (regression for bug 1)",
                );
            }
        }

        #[test]
        fn appending_remote_projection_does_not_change_local_worker_counts() {
            use openai_protocol::worker::WorkerLoadInfoSource;

            let mut loads = openai_protocol::worker::WorkerLoadsResult {
                loads: vec![WorkerLoadInfo {
                    worker: "http://local-worker:8000".to_string(),
                    worker_type: None,
                    load: 2,
                    details: None,
                    region_id: Some(LOCAL_REGION.to_string()),
                    worker_id: Some("local-w".to_string()),
                    model_id: Some("cohere.command-r-plus".to_string()),
                    status: Some(WorkerStatus::Ready),
                    generated_at_ms: Some(NOW_MS),
                    version: Some(1),
                    source: Some(WorkerLoadInfoSource::LocalWorker),
                    remote_workers: None,
                }],
                total_workers: 2,
                successful: 1,
                failed: 1,
            };
            let remote_projection = vec![WorkerLoadInfo {
                worker: "region-peer/us-chicago-1".to_string(),
                worker_type: None,
                load: 5,
                details: None,
                region_id: Some("us-chicago-1".to_string()),
                worker_id: None,
                model_id: None,
                status: None,
                generated_at_ms: Some(NOW_MS),
                version: Some(1),
                source: Some(WorkerLoadInfoSource::RemoteSmg),
                remote_workers: Some(vec![WorkerLoadInfo {
                    worker: "w1".to_string(),
                    worker_type: None,
                    load: 5,
                    details: None,
                    region_id: Some("us-chicago-1".to_string()),
                    worker_id: Some("w1".to_string()),
                    model_id: Some("cohere.command-r-plus".to_string()),
                    status: Some(WorkerStatus::Ready),
                    generated_at_ms: Some(NOW_MS),
                    version: Some(1),
                    source: Some(WorkerLoadInfoSource::RemoteSmg),
                    remote_workers: None,
                }]),
            }];

            append_projected_remote_worker_loads(&mut loads, remote_projection);

            assert_eq!(loads.loads.len(), 2);
            assert_eq!(loads.total_workers, 2);
            assert_eq!(loads.successful, 1);
            assert_eq!(loads.failed, 1);
            assert_eq!(loads.loads[1].source, Some(WorkerLoadInfoSource::RemoteSmg));
        }

        #[test]
        fn aggregates_load_across_replicas_for_same_worker() {
            // Two SMG replicas in us-chicago-1 each observe worker w1.
            // Phase F's `fresh_load_entries` sums their loads.
            let mut state = CrossRegionState::new();
            upsert_load(
                &mut state,
                "us-chicago-1",
                "w1",
                "smg-chi-a",
                "cohere.command-r-plus",
                4,
                version(1, "smg-chi-a"),
            );
            upsert_load(
                &mut state,
                "us-chicago-1",
                "w1",
                "smg-chi-b",
                "cohere.command-r-plus",
                3,
                version(1, "smg-chi-b"),
            );

            let envelopes = project_remote_worker_loads(&state, LOCAL_REGION, WINDOW_MS, NOW_MS);
            assert_eq!(envelopes.len(), 1);
            let envelope = &envelopes[0];
            assert_eq!(envelope.load, 7, "sum of replica loads");
            let remote_workers = envelope.remote_workers.as_ref().unwrap();
            assert_eq!(remote_workers.len(), 1, "one (worker, model) entry");
            assert_eq!(remote_workers[0].load, 7);
        }

        #[test]
        fn empty_state_returns_no_envelopes() {
            let envelopes = project_remote_worker_loads(
                &CrossRegionState::new(),
                LOCAL_REGION,
                WINDOW_MS,
                NOW_MS,
            );
            assert!(envelopes.is_empty());
        }

        #[test]
        fn stale_loads_are_filtered_out() {
            let mut state = CrossRegionState::new();
            // Entry older than the freshness window.
            let stale = SignalVersion {
                version: 1,
                actor: "smg-chi-a".to_string(),
                updated_at_ms: NOW_MS - WINDOW_MS - 1_000,
            };
            upsert_load(
                &mut state,
                "us-chicago-1",
                "w1",
                "smg-chi-a",
                "cohere.command-r-plus",
                5,
                stale,
            );

            let envelopes = project_remote_worker_loads(&state, LOCAL_REGION, WINDOW_MS, NOW_MS);
            assert!(
                envelopes.is_empty(),
                "stale entries must not produce projection envelopes",
            );
        }
    }
}
