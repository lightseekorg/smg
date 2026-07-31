use std::{sync::Arc, time::Instant};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::StreamExt;
use memchr::memmem;
use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{GenerationRequest, InputIds, StringOrArray},
    completion::CompletionRequest,
    generate::GenerateRequest,
    rerank::RerankRequest,
};
use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, warn};

use crate::{
    config::types::RetryConfig,
    middleware::TenantRequestMeta,
    observability::{
        events::{self, Event},
        metrics::{bool_to_static_str, metrics_labels, Metrics},
        otel_trace::inject_trace_context_http,
    },
    policies::{LoadBalancingPolicy, PolicyRegistry, SelectWorkerInfo, WorkerLeg},
    routers::{
        common::{
            header_utils,
            retry::{is_retryable_status, RetryExecutor},
            sse::SseEncoder,
        },
        error,
        grpc::utils::{error_type_from_status, route_to_endpoint},
        RouterTrait, PD_PREFILL_QUEUE_FULL, PD_PREFILL_QUEUE_TIMEOUT,
    },
    worker::{
        acquire_prefill, HashRing, PrefillAcquireError, PrefillAdmission,
        PrefillAdmissionRejection, PrefillCandidateError, PrefillLoadGuard,
        PrefillSelectionContext, Worker, WorkerLoadGuard, WorkerRegistry, WorkerType,
        UNKNOWN_MODEL_ID,
    },
};

type PdWorkerPair = (Arc<dyn Worker>, Arc<dyn Worker>);
type PdWorkerPools = (Vec<Arc<dyn Worker>>, Vec<Arc<dyn Worker>>);

/// One admitted PD request: the chosen pair and the load it holds.
struct AdmittedPdPair {
    prefill: Arc<dyn Worker>,
    decode: Arc<dyn Worker>,
    prefill_guard: PrefillLoadGuard,
    decode_guard: WorkerLoadGuard,
}

pub struct PDRouter {
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub client: Client,
    pub retry_config: RetryConfig,
    pub api_key: Option<String>,
    prefill_admission: Option<Arc<PrefillAdmission>>,
}

impl std::fmt::Debug for PDRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PDRouter")
            .field("worker_registry", &self.worker_registry)
            .field("policy_registry", &self.policy_registry)
            .field("client", &self.client)
            .field("retry_config", &self.retry_config)
            .field("api_key_configured", &self.api_key.is_some())
            .field(
                "prefill_admission_enabled",
                &self.prefill_admission.is_some(),
            )
            .finish()
    }
}

#[derive(Clone)]
struct PDRequestContext<'a> {
    route: &'static str,
    bootstrap_batch_size: Option<usize>,
    is_stream: bool,
    return_logprob: bool,
    request_text: Option<String>,
    model_id: &'a str,
    headers: Option<HeaderMap>,
}

#[derive(Debug)]
enum PDSelectionError {
    Unavailable(String),
    QueueFull,
    QueueTimeout,
}

#[derive(Debug)]
enum PDCandidateError {
    Unavailable(String),
    PrefillAtCapacity,
}

impl std::fmt::Display for PDSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(message) => f.write_str(message),
            Self::QueueFull => f.write_str("Prefill admission queue is full"),
            Self::QueueTimeout => f.write_str("Timed out waiting for Prefill admission"),
        }
    }
}

impl From<PrefillAdmissionRejection> for PDSelectionError {
    fn from(value: PrefillAdmissionRejection) -> Self {
        match value {
            PrefillAdmissionRejection::QueueFull => Self::QueueFull,
            PrefillAdmissionRejection::QueueTimeout => Self::QueueTimeout,
            PrefillAdmissionRejection::Unavailable => {
                Self::Unavailable("No available prefill and decode worker pair".to_string())
            }
        }
    }
}

impl From<PDCandidateError> for PDSelectionError {
    fn from(value: PDCandidateError) -> Self {
        match value {
            PDCandidateError::Unavailable(message) => Self::Unavailable(message),
            PDCandidateError::PrefillAtCapacity => {
                Self::Unavailable("No available prefill workers".to_string())
            }
        }
    }
}

impl From<PrefillAcquireError<PDCandidateError>> for PDSelectionError {
    fn from(value: PrefillAcquireError<PDCandidateError>) -> Self {
        match value {
            PrefillAcquireError::Candidate(error) => error.into(),
            PrefillAcquireError::Rejected(rejection) => rejection.into(),
        }
    }
}

impl PrefillCandidateError for PDCandidateError {
    fn is_at_capacity(&self) -> bool {
        matches!(self, Self::PrefillAtCapacity)
    }
}

impl PDRouter {
    async fn proxy_to_first_prefill_worker(
        &self,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let workers = self.worker_registry.get_prefill_workers();
        let first_worker_url = workers.first().map(|w| w.url().to_string());

        if let Some(worker_url) = first_worker_url {
            self.proxy_to_worker(worker_url, endpoint, headers).await
        } else {
            error::service_unavailable("no_prefill_servers", "No prefill servers available")
        }
    }

    async fn proxy_to_worker(
        &self,
        worker_url: String,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let url = format!("{worker_url}/{endpoint}");
        let mut request_builder = self.client.get(&url);

        if let Some(headers) = headers {
            for (name, value) in headers {
                request_builder = request_builder.header(name, value);
            }
        }

        match request_builder.send().await {
            Ok(res) if res.status().is_success() => {
                let response_headers = header_utils::preserve_response_headers(res.headers());

                match res.bytes().await {
                    Ok(body) => {
                        let mut response = Response::new(Body::from(body));
                        *response.status_mut() = StatusCode::OK;
                        *response.headers_mut() = response_headers;
                        response
                    }
                    Err(e) => {
                        error!("Failed to read response body: {}", e);
                        error::internal_error(
                            "read_response_body_failed",
                            format!("Failed to read response body: {e}"),
                        )
                    }
                }
            }
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                // Use the status code to determine which error function to use
                match status {
                    StatusCode::BAD_REQUEST => error::bad_request(
                        "server_bad_request",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::NOT_FOUND => error::not_found(
                        "server_not_found",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::INTERNAL_SERVER_ERROR => error::internal_error(
                        "server_internal_error",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::SERVICE_UNAVAILABLE => error::service_unavailable(
                        "server_unavailable",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::BAD_GATEWAY => error::bad_gateway(
                        "server_bad_gateway",
                        format!("Server returned status: {}", res.status()),
                    ),
                    _ => error::internal_error(
                        "server_error",
                        format!("Server returned status: {}", res.status()),
                    ),
                }
            }
            Err(e) => {
                error!("Failed to proxy request server: {}", e);
                error::internal_error(
                    "proxy_request_failed",
                    format!("Failed to proxy request: {e}"),
                )
            }
        }
    }

    #[expect(
        clippy::unused_async,
        reason = "async for API consistency with other router constructors"
    )]
    pub async fn new(ctx: &Arc<crate::app_context::AppContext>) -> Result<Self, String> {
        Ok(PDRouter {
            worker_registry: Arc::clone(&ctx.worker_registry),
            policy_registry: Arc::clone(&ctx.policy_registry),
            client: ctx.client.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            api_key: ctx.router_config.api_key.clone(),
            prefill_admission: ctx.prefill_admission.clone(),
        })
    }

    fn handle_server_selection_error(error: PDSelectionError) -> Response {
        error!("Failed to select PD pair error={}", error);
        match error {
            PDSelectionError::Unavailable(message) => error::service_unavailable(
                "server_selection_failed",
                format!("No available servers: {message}"),
            ),
            PDSelectionError::QueueFull => {
                error::too_many_requests(PD_PREFILL_QUEUE_FULL, "Prefill admission queue is full")
            }
            PDSelectionError::QueueTimeout => error::too_many_requests(
                PD_PREFILL_QUEUE_TIMEOUT,
                "Timed out waiting for Prefill admission",
            ),
        }
    }

    fn should_retry(response: &Response) -> bool {
        is_retryable_status(response.status())
            && !matches!(
                error::extract_error_code_from_response(response),
                PD_PREFILL_QUEUE_FULL | PD_PREFILL_QUEUE_TIMEOUT
            )
    }

    fn handle_serialization_error(error: impl std::fmt::Display) -> Response {
        error!("Failed to serialize request error={}", error);
        error::internal_error("serialization_failed", "Failed to serialize request")
    }

    fn get_generate_bootstrap_batch_size(req: &GenerateRequest) -> Option<usize> {
        // GenerateRequest doesn't support batch via arrays, only via input_ids
        if let Some(InputIds::Batch(batches)) = &req.input_ids {
            if !batches.is_empty() {
                return Some(batches.len());
            }
        }
        None
    }

    fn get_chat_bootstrap_batch_size(req: &ChatCompletionRequest) -> Option<usize> {
        if let Some(n) = req.n {
            if n > 1 {
                return Some(n as usize);
            }
        }
        None
    }

    fn get_completion_bootstrap_batch_size(req: &CompletionRequest) -> Option<usize> {
        if let StringOrArray::Array(arr) = &req.prompt {
            if !arr.is_empty() {
                return Some(arr.len());
            }
        }
        None
    }

    // Static key strings to avoid per-request allocations
    const BOOTSTRAP_HOST_KEY: &'static str = "bootstrap_host";
    const BOOTSTRAP_PORT_KEY: &'static str = "bootstrap_port";
    const BOOTSTRAP_ROOM_KEY: &'static str = "bootstrap_room";

    fn inject_bootstrap_into_value(
        mut original: Value,
        prefill_worker: &dyn Worker,
        batch_size: Option<usize>,
    ) -> Result<Value, String> {
        let obj = original
            .as_object_mut()
            .ok_or_else(|| "Request must be a JSON object".to_string())?;

        if let Some(n) = batch_size {
            let mut hosts = Vec::with_capacity(n);
            let mut ports = Vec::with_capacity(n);
            let mut rooms = Vec::with_capacity(n);
            for _ in 0..n {
                hosts.push(prefill_worker.bootstrap_host());
                ports.push(prefill_worker.bootstrap_port());
                rooms.push(super::pd_types::generate_room_id());
            }
            obj.insert(
                Self::BOOTSTRAP_HOST_KEY.to_string(),
                Value::Array(hosts.into_iter().map(Value::from).collect()),
            );
            obj.insert(
                Self::BOOTSTRAP_PORT_KEY.to_string(),
                Value::Array(
                    ports
                        .into_iter()
                        .map(|p| match p {
                            Some(v) => Value::from(v),
                            None => Value::Null,
                        })
                        .collect(),
                ),
            );
            obj.insert(
                Self::BOOTSTRAP_ROOM_KEY.to_string(),
                Value::Array(rooms.into_iter().map(Value::from).collect()),
            );
        } else {
            obj.insert(
                Self::BOOTSTRAP_HOST_KEY.to_string(),
                Value::from(prefill_worker.bootstrap_host()),
            );
            obj.insert(
                Self::BOOTSTRAP_PORT_KEY.to_string(),
                match prefill_worker.bootstrap_port() {
                    Some(v) => Value::from(v),
                    None => Value::Null,
                },
            );
            obj.insert(
                Self::BOOTSTRAP_ROOM_KEY.to_string(),
                Value::from(super::pd_types::generate_room_id()),
            );
        }
        Ok(original)
    }

    fn inject_dp_rank_to_json(json_val: &mut Value, rank: isize, rank_key: &str) {
        if let Some(obj) = json_val.as_object_mut() {
            obj.insert(rank_key.to_string(), Value::Number(rank.into()));
        }
    }

    fn select_dp_ranks(
        &self,
        prefill: &dyn Worker,
        decode: &dyn Worker,
        request_text: Option<&str>,
    ) -> (Option<isize>, Option<isize>) {
        let mut prefill_rank = prefill.dp_rank().map(|rank| rank as isize);
        let mut decode_rank = decode.dp_rank().map(|rank| rank as isize);

        if prefill_rank.is_some() && decode_rank.is_some() {
            return (prefill_rank, decode_rank);
        }
        let Some(dp_rank_policy) = self.policy_registry.get_dp_rank_policy() else {
            return (prefill_rank, decode_rank);
        };

        let estimated_cost = request_text.map_or(1, |text| {
            let word_count = text.split_whitespace().count();
            ((word_count as f64 * 1.3).ceil() as isize).max(1)
        });
        if prefill_rank.is_none() {
            prefill_rank = dp_rank_policy.select_dp_rank(prefill, estimated_cost);
        }
        if decode_rank.is_none() {
            decode_rank = dp_rank_policy.select_dp_rank(decode, estimated_cost);
        }

        (prefill_rank, decode_rank)
    }

    async fn execute_dual_dispatch<T: Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        original_request: &T,
        mut context: PDRequestContext<'_>,
    ) -> Response {
        let start_time = Instant::now();

        let route = context.route;
        // Resolve once, here, so every registry, policy and metrics lookup
        // below is keyed by the canonical model ID. Only `get_by_model`
        // understands aliases; retry configs, hash rings and policies do not.
        let canonical_model = self.worker_registry.resolve_model_alias(context.model_id);
        let model = canonical_model.as_deref().unwrap_or(context.model_id);
        context.model_id = model;
        let endpoint = route_to_endpoint(route);

        // Record request start (Layer 2)
        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_PD,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint,
            bool_to_static_str(context.is_stream),
        );
        // Clone request once outside the retry loop, then use Arc to share across attempts
        // This avoids O(retries) clones by sharing the same data
        let shared_request = Arc::new(original_request.clone());

        // Use per-model retry config if set by a worker, otherwise fall back to router default.
        let per_model_retry_config = self.worker_registry.get_retry_config(model);
        let retry_config = per_model_retry_config
            .as_ref()
            .unwrap_or(&self.retry_config);

        let response = RetryExecutor::execute_response_with_retry(
            retry_config,
            {
                move |attempt: u32| {
                    // Clone Arc (cheap reference count increment) instead of cloning the entire request
                    let shared_request = Arc::clone(&shared_request);
                    let context = context.clone();
                    async move {
                        let admitted = match self
                            .select_pd_pair_with_admission(
                                context.request_text.as_deref(),
                                context.model_id,
                                context.headers.as_ref(),
                            )
                            .await
                        {
                            Ok(pair) => pair,
                            Err(e) => {
                                return Self::handle_server_selection_error(e);
                            }
                        };
                        let (prefill, decode) =
                            (Arc::clone(&admitted.prefill), Arc::clone(&admitted.decode));

                        debug!(
                            "PD retry attempt {} using prefill={} decode={}",
                            attempt,
                            prefill.url(),
                            decode.url()
                        );

                        let mut json_request = match serde_json::to_value(shared_request.as_ref()) {
                            Ok(v) => v,
                            Err(e) => return Self::handle_serialization_error(e),
                        };
                        // The prefill and decode workers only know the
                        // canonical name, so forward that, not the alias the
                        // client sent.
                        super::set_request_model(&mut json_request, model);

                        json_request = match Self::inject_bootstrap_into_value(
                            json_request,
                            prefill.as_ref(),
                            context.bootstrap_batch_size,
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                Metrics::record_pd_bootstrap_failure();
                                return Self::handle_serialization_error(e);
                            }
                        };

                        let mut prefill_json_request = json_request.clone();
                        let mut decode_json_request = json_request;

                        let (prefill_rank, decode_rank) = self.select_dp_ranks(
                            prefill.as_ref(),
                            decode.as_ref(),
                            context.request_text.as_deref(),
                        );

                        if let Some(p_rank) = prefill_rank {
                            Self::inject_dp_rank_to_json(
                                &mut prefill_json_request,
                                p_rank,
                                "routed_dp_rank",
                            );
                            Self::inject_dp_rank_to_json(
                                &mut decode_json_request,
                                p_rank,
                                "disagg_prefill_dp_rank",
                            );
                        }
                        if let Some(d_rank) = decode_rank {
                            Self::inject_dp_rank_to_json(
                                &mut decode_json_request,
                                d_rank,
                                "routed_dp_rank",
                            );
                        }
                        if prefill_rank.is_some() || decode_rank.is_some() {
                            debug!(
                                "PD selected DP ranks prefill={:?} decode={:?}",
                                prefill_rank, decode_rank
                            );
                        }

                        let response = self
                            .execute_dual_dispatch_internal(
                                headers,
                                prefill_json_request,
                                decode_json_request,
                                context,
                                admitted,
                            )
                            .await;

                        let status = response.status();
                        prefill.record_outcome(status.as_u16());
                        decode.record_outcome(status.as_u16());

                        // Record worker errors for server errors (5xx)
                        if status.is_server_error() {
                            let error_type = error_type_from_status(status);
                            Metrics::record_worker_error(
                                metrics_labels::WORKER_PREFILL,
                                metrics_labels::CONNECTION_HTTP,
                                error_type,
                            );
                            Metrics::record_worker_error(
                                metrics_labels::WORKER_DECODE,
                                metrics_labels::CONNECTION_HTTP,
                                error_type,
                            );
                        }

                        response
                    }
                }
            },
            |res, _attempt| Self::should_retry(res),
            |delay, attempt| {
                // Layer 3 worker metrics (PD mode uses both prefill and decode workers)
                Metrics::record_worker_retry(metrics_labels::WORKER_PREFILL, endpoint);
                Metrics::record_worker_retry(metrics_labels::WORKER_DECODE, endpoint);
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_PREFILL, endpoint);
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_DECODE, endpoint);
            },
        )
        .await;

        // Record Layer 2 metrics
        let duration = start_time.elapsed();
        if response.status().is_success() {
            Metrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                duration,
            );
        } else if !Self::should_retry(&response) {
            Metrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                error_type_from_status(response.status()),
            );
        }

        response
    }

    async fn handle_decode_error_response(
        &self,
        res: reqwest::Response,
        context: &PDRequestContext<'_>,
        decode: Arc<dyn Worker>,
        decode_guard: WorkerLoadGuard,
    ) -> Response {
        let status = res.status();

        if context.is_stream {
            // Handle streaming error response
            let response_headers = header_utils::preserve_response_headers(res.headers());
            let error_payload = match res.bytes().await {
                Ok(error_body) => {
                    if let Ok(error_json) = serde_json::from_slice::<Value>(&error_body) {
                        json!({ "message": error_json, "status": status.as_u16() })
                    } else {
                        json!({ "message": String::from_utf8_lossy(&error_body).to_string(), "status": status.as_u16() })
                    }
                }
                Err(e) => {
                    json!({ "message": format!("Decode server error: {}", e), "status": status.as_u16() })
                }
            };
            drop(decode_guard);

            let sse_data = format!(
                "data: {}\n\n",
                serde_json::to_string(&json!({ "error": error_payload })).unwrap_or_default()
            );
            let error_stream = tokio_stream::once(Ok(Bytes::from(sse_data)));

            let decode_url = decode.url().to_string();
            self.create_streaming_response(
                error_stream,
                status,
                None,
                context.return_logprob,
                Some(decode_url),
                Some(response_headers),
                None,
            )
        } else {
            // Handle non-streaming error response
            let error_body = res.bytes().await;
            drop(decode_guard);
            match error_body {
                Ok(error_body) => {
                    // Try to parse error message from body, fallback to status-based error
                    let error_message = if let Ok(error_json) =
                        serde_json::from_slice::<Value>(&error_body)
                    {
                        if let Some(msg) = error_json
                            .get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                        {
                            msg.to_string()
                        } else if let Some(msg) = error_json.get("message").and_then(|m| m.as_str())
                        {
                            msg.to_string()
                        } else {
                            String::from_utf8_lossy(&error_body).to_string()
                        }
                    } else {
                        String::from_utf8_lossy(&error_body).to_string()
                    };

                    let status_code = StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    match status_code {
                        StatusCode::BAD_REQUEST => {
                            error::bad_request("decode_bad_request", error_message)
                        }
                        StatusCode::NOT_FOUND => {
                            error::not_found("decode_not_found", error_message)
                        }
                        StatusCode::INTERNAL_SERVER_ERROR => {
                            error::internal_error("decode_internal_error", error_message)
                        }
                        StatusCode::SERVICE_UNAVAILABLE => {
                            error::service_unavailable("decode_unavailable", error_message)
                        }
                        StatusCode::BAD_GATEWAY => {
                            error::bad_gateway("decode_bad_gateway", error_message)
                        }
                        _ => error::internal_error("decode_error", error_message),
                    }
                }
                Err(e) => {
                    let error_message = format!("Decode server error: {e}");
                    let status_code = StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    match status_code {
                        StatusCode::BAD_REQUEST => {
                            error::bad_request("decode_read_failed", error_message)
                        }
                        StatusCode::NOT_FOUND => {
                            error::not_found("decode_read_failed", error_message)
                        }
                        StatusCode::INTERNAL_SERVER_ERROR => {
                            error::internal_error("decode_read_failed", error_message)
                        }
                        StatusCode::SERVICE_UNAVAILABLE => {
                            error::service_unavailable("decode_read_failed", error_message)
                        }
                        StatusCode::BAD_GATEWAY => {
                            error::bad_gateway("decode_read_failed", error_message)
                        }
                        _ => error::internal_error("decode_read_failed", error_message),
                    }
                }
            }
        }
    }

    // Internal method that performs the actual dual dispatch (without retry logic)
    async fn execute_dual_dispatch_internal(
        &self,
        headers: Option<&HeaderMap>,
        prefill_json_request: Value,
        decode_json_request: Value,
        context: PDRequestContext<'_>,
        admitted: AdmittedPdPair,
    ) -> Response {
        let AdmittedPdPair {
            prefill,
            decode,
            prefill_guard,
            decode_guard,
        } = admitted;
        let mut headers_with_trace = headers.cloned().unwrap_or_default();
        inject_trace_context_http(&mut headers_with_trace);
        let headers = Some(&headers_with_trace);

        // Build both requests
        let prefill_request = self.build_post_with_headers(
            &self.client,
            prefill.as_ref(),
            context.route,
            &prefill_json_request,
            headers,
            false,
        );
        let decode_request = self.build_post_with_headers(
            &self.client,
            decode.as_ref(),
            context.route,
            &decode_json_request,
            headers,
            false,
        );

        // Send both requests concurrently and wait for both
        // Note: Using borrowed references avoids heap allocation
        events::RequestPDSentEvent {
            prefill_url: prefill.url(),
            decode_url: decode.url(),
        }
        .emit();

        // Wait only for both response heads here. Decode can reject the request
        // before Prefill reaches EOF, and that rejection must cancel Prefill
        // instead of waiting for a bootstrap that Decode will never consume.
        let runtime = prefill.metadata().spec.runtime_type.as_str();
        let dispatch_start = Instant::now();
        let prefill_fut = async {
            let response = prefill_request.send().await.map_err(|e| {
                error!("PD prefill transport error: {e}");
                error::bad_gateway(
                    "PD disaggregation request failed",
                    format!("Prefill transport error: {e}"),
                )
            })?;
            Ok::<_, Response>((dispatch_start.elapsed(), response))
        };
        let decode_fut = async {
            let response = decode_request.send().await.map_err(|e| {
                error!("PD decode transport error: {e}");
                error::bad_gateway(
                    "PD disaggregation request failed",
                    format!("Decode transport error: {e}"),
                )
            })?;
            Ok::<_, Response>((dispatch_start.elapsed(), response))
        };
        let pd_result = tokio::try_join!(prefill_fut, decode_fut);

        events::RequestReceivedEvent {}.emit();

        let ((prefill_head_elapsed, prefill_response), (decode_head_elapsed, decode_response)) =
            match pd_result {
                Ok(pair) => pair,
                Err(response) => return response,
            };

        // Process decode response
        let status = StatusCode::from_u16(decode_response.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        debug!("Decode response status: {}", status);

        if !status.is_success() {
            error!(
                "Decode server returned error status decode_url={} status={}",
                decode.url(),
                status
            );

            drop(prefill_response);
            drop(prefill_guard);
            return self
                .handle_decode_error_response(decode_response, &context, decode, decode_guard)
                .await;
        }

        // Honest PD TTFT: dispatch to the decode response head — the first
        // user-visible decode output, since the gateway forwards the decode body
        // unbuffered. Complements the decode-only `smg_router_ttft_seconds`,
        // which PD never narrows to a single leg.
        Metrics::record_pd_ttft(
            metrics_labels::BACKEND_PD,
            context.model_id,
            runtime,
            decode_head_elapsed,
        );

        let prefill_drain_start = Instant::now();
        let prefill_body = match self
            .process_prefill_response(prefill_response, prefill.url(), context.return_logprob)
            .await
        {
            Ok((_, body)) => body,
            Err(response) => return response,
        };
        Metrics::record_pd_prefill_duration(
            metrics_labels::BACKEND_PD,
            context.model_id,
            runtime,
            prefill_head_elapsed + prefill_drain_start.elapsed(),
        );
        drop(prefill_guard);

        if context.is_stream {
            // Streaming response
            let prefill_logprobs = if context.return_logprob {
                prefill_body
                    .as_ref()
                    .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                    .and_then(|json| json.pointer("/meta_info/input_token_logprobs").cloned())
            } else {
                None
            };

            let response_headers =
                header_utils::preserve_response_headers(decode_response.headers());

            self.create_streaming_response(
                decode_response.bytes_stream(),
                status,
                prefill_logprobs,
                context.return_logprob,
                None,
                Some(response_headers),
                Some(decode_guard),
            )
        } else {
            // Non-streaming response
            if context.return_logprob {
                self.process_non_streaming_response(
                    decode_response,
                    status,
                    context.return_logprob,
                    prefill_body,
                )
                .await
            } else {
                // Direct passthrough when no logprobs needed
                let response_headers =
                    header_utils::preserve_response_headers(decode_response.headers());

                match decode_response.bytes().await {
                    Ok(decode_body) => {
                        let mut response = Response::new(Body::from(decode_body));
                        *response.status_mut() = status;
                        *response.headers_mut() = response_headers;
                        response
                    }
                    Err(e) => {
                        error!("Failed to read decode response: {}", e);
                        error::internal_error("read_response_failed", "Failed to read response")
                    }
                }
            }
        }
    }

    fn policies_need_request_text(&self) -> bool {
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();
        prefill_policy.needs_request_text() || decode_policy.needs_request_text()
    }

    async fn select_pd_pair_with_admission(
        &self,
        request_text: Option<&str>,
        model_id: &str,
        headers: Option<&HeaderMap>,
    ) -> Result<AdmittedPdPair, PDSelectionError> {
        debug!("Selecting admitted PD pair: model_id={:?}", model_id);

        let ((prefill, decode), prefill_guard) = acquire_prefill(
            self.prefill_admission.as_deref(),
            headers,
            |(prefill, _): &PdWorkerPair| prefill,
            |capacity| self.select_pd_candidate(request_text, model_id, headers, capacity),
        )
        .await?;
        let decode_guard = WorkerLoadGuard::new(Arc::clone(&decode), headers);

        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();
        Self::record_pd_selection(model_id, &prefill_policy, &decode_policy);

        Ok(AdmittedPdPair {
            prefill,
            decode,
            prefill_guard,
            decode_guard,
        })
    }

    fn select_pd_candidate(
        &self,
        request_text: Option<&str>,
        model_id: &str,
        headers: Option<&HeaderMap>,
        capacity: Option<&PrefillSelectionContext<'_>>,
    ) -> Result<PdWorkerPair, PDCandidateError> {
        let (prefill_workers, decode_workers) = self.pd_worker_pools(model_id);
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();
        let hash_ring = self.worker_registry.get_hash_ring(model_id);

        let mut available_prefill = Self::available_workers(&prefill_workers, "prefill")
            .map_err(PDCandidateError::Unavailable)?;
        let available_decode = Self::available_workers(&decode_workers, "decode")
            .map_err(PDCandidateError::Unavailable)?;
        let target_header = if prefill_policy.name() == "consistent_hashing" {
            header_utils::extract_target_worker(headers)
        } else {
            None
        };
        let targeted = target_header.is_some();

        if let Some(capacity) = capacity {
            if !targeted {
                available_prefill.retain(|worker| capacity.has_capacity(worker));
                if available_prefill.is_empty() {
                    return Err(PDCandidateError::PrefillAtCapacity);
                }
            }
        }

        if let Some(target_header) = target_header {
            let target = target_header
                .parse::<usize>()
                .ok()
                .and_then(|index| available_prefill.get(index))
                .ok_or_else(|| {
                    PDCandidateError::Unavailable(
                        "Target Prefill worker is unavailable".to_string(),
                    )
                })?;
            if capacity.is_some_and(|capacity| !capacity.has_capacity(target)) {
                return Err(PDCandidateError::PrefillAtCapacity);
            }
        }

        let prefill = self
            .pick_worker_by_policy_arc(
                &available_prefill,
                &prefill_policy,
                request_text,
                headers,
                hash_ring.clone(),
                "prefill",
                WorkerLeg::Prefill,
            )
            .map_err(PDCandidateError::Unavailable)?;

        let decode = self
            .pick_worker_by_policy_arc(
                &available_decode,
                &decode_policy,
                request_text,
                headers,
                hash_ring,
                "decode",
                WorkerLeg::Decode,
            )
            .map_err(PDCandidateError::Unavailable)?;

        Ok((prefill, decode))
    }

    fn pd_worker_pools(&self, model_id: &str) -> PdWorkerPools {
        let is_unknown_model = model_id == UNKNOWN_MODEL_ID;
        let by_model = self.worker_registry.get_by_model(model_id);
        let mut prefill_workers: Vec<_> = by_model
            .iter()
            .filter(|worker| matches!(worker.worker_type(), WorkerType::Prefill))
            .cloned()
            .collect();
        let mut decode_workers: Vec<_> = by_model
            .iter()
            .filter(|worker| matches!(worker.worker_type(), WorkerType::Decode))
            .cloned()
            .collect();

        if is_unknown_model {
            if prefill_workers.is_empty() {
                prefill_workers = self.worker_registry.get_prefill_workers().to_vec();
            }
            if decode_workers.is_empty() {
                decode_workers = self.worker_registry.get_decode_workers().to_vec();
            }
        }

        (prefill_workers, decode_workers)
    }

    fn record_pd_selection(
        model: &str,
        prefill_policy: &Arc<dyn LoadBalancingPolicy>,
        decode_policy: &Arc<dyn LoadBalancingPolicy>,
    ) {
        Metrics::record_worker_selection(
            metrics_labels::WORKER_PREFILL,
            metrics_labels::CONNECTION_HTTP,
            model,
            prefill_policy.name(),
        );
        Metrics::record_worker_selection(
            metrics_labels::WORKER_DECODE,
            metrics_labels::CONNECTION_HTTP,
            model,
            decode_policy.name(),
        );
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "HTTP PD worker pick threads policy + request context + leg"
    )]
    fn pick_worker_by_policy_arc(
        &self,
        workers: &[Arc<dyn Worker>],
        policy: &Arc<dyn LoadBalancingPolicy>,
        request_text: Option<&str>,
        headers: Option<&HeaderMap>,
        hash_ring: Option<Arc<HashRing>>,
        worker_type: &str,
        leg: WorkerLeg,
    ) -> Result<Arc<dyn Worker>, String> {
        let selected_idx = self
            .policy_registry
            .select_worker(
                policy,
                workers,
                &SelectWorkerInfo {
                    request_text,
                    tokens: None,
                    headers,
                    hash_ring,
                    leg,
                },
            )
            .ok_or_else(|| {
                format!(
                    "Policy {} failed to select a {} worker",
                    policy.name(),
                    worker_type
                )
            })?;

        workers.get(selected_idx).cloned().ok_or_else(|| {
            format!(
                "Policy {} returned an invalid {} worker index",
                policy.name(),
                worker_type
            )
        })
    }

    fn available_workers(
        workers: &[Arc<dyn Worker>],
        worker_type: &str,
    ) -> Result<Vec<Arc<dyn Worker>>, String> {
        if workers.is_empty() {
            return Err(format!(
                "No {worker_type} workers available. Please check if {worker_type} servers are configured and healthy."
            ));
        }

        let available_workers: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        if available_workers.is_empty() {
            return Err(format!(
                "No available {worker_type} workers (all circuits open or unhealthy)"
            ));
        }

        Ok(available_workers)
    }

    #[expect(clippy::too_many_arguments)]
    #[expect(
        clippy::unused_self,
        reason = "method on PDRouter for consistent API; may use self in future"
    )]
    fn create_streaming_response(
        &self,
        stream: impl futures_util::Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
        status: StatusCode,
        prefill_logprobs: Option<Value>,
        return_logprob: bool,
        decode_url: Option<String>,
        headers: Option<HeaderMap>,
        decode_guard: Option<WorkerLoadGuard>,
    ) -> Response {
        let (tx, rx) = mpsc::unbounded_channel();

        #[expect(
            clippy::disallowed_methods,
            reason = "fire-and-forget stream relay; gateway shutdown need not wait for decode stream forwarding"
        )]
        tokio::spawn(async move {
            let _decode_guard = decode_guard;
            futures_util::pin_mut!(stream);
            // Reusable SSE encoder for the logprob-merge re-encode path.
            let mut encoder = SseEncoder::new();
            // Whether the next chunk begins at an SSE line boundary (i.e. the
            // previous chunk ended with an EOL); used to anchor the [DONE]
            // sentinel detection when the match sits at the start of a chunk.
            let mut at_line_start = true;
            loop {
                let chunk_result = tokio::select! {
                    biased;
                    () = tx.closed() => break,
                    chunk_result = stream.next() => chunk_result,
                };
                let Some(chunk_result) = chunk_result else {
                    break;
                };
                match chunk_result {
                    Ok(chunk) => {
                        let is_done = Self::chunk_contains_done_event(&chunk, at_line_start);
                        if let Some(&last) = chunk.last() {
                            at_line_start = last == b'\n' || last == b'\r';
                        }

                        let result = if return_logprob && prefill_logprobs.is_some() {
                            Self::merge_streaming_logprobs(
                                prefill_logprobs.as_ref(),
                                &chunk,
                                &mut encoder,
                            )
                            .unwrap_or(chunk)
                        } else {
                            chunk
                        };

                        if tx.send(Ok(result)).is_err() {
                            break;
                        }

                        if is_done {
                            break;
                        }
                    }
                    Err(e) => {
                        if let Some(ref url) = decode_url {
                            error!("Stream error from decode server {}: {}", url, e);
                        }
                        let _ = tx.send(Err(format!("Stream error: {e}")));
                        break;
                    }
                }
            }
        });

        let stream = UnboundedReceiverStream::new(rx);
        let body = Body::from_stream(stream);

        let mut response = Response::new(body);
        *response.status_mut() = status;

        let mut response_headers = headers.unwrap_or_default();
        response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        *response.headers_mut() = response_headers;

        response
    }

    /// Build a non-streaming PD response with `Content-Type: application/json`.
    ///
    /// Axum's `(StatusCode, Bytes).into_response()` defaults to
    /// `application/octet-stream`, which breaks OpenAI-style JSON clients.
    fn non_stream_pd_json_response(status: StatusCode, body: Bytes) -> Response {
        let mut response = Response::new(Body::from(body));
        *response.status_mut() = status;
        response
            .headers_mut()
            .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        response
    }

    // Helper to process non-streaming decode response with logprob merging
    async fn process_non_streaming_response(
        &self,
        res: reqwest::Response,
        status: StatusCode,
        return_logprob: bool,
        prefill_body: Option<Bytes>,
    ) -> Response {
        let response = res.bytes().await;
        let decode_body = match response {
            Ok(decode_body) => decode_body,
            Err(e) => {
                error!("Failed to read decode response: {}", e);
                return error::internal_error("read_response_failed", "Failed to read response");
            }
        };

        if !return_logprob {
            return Self::non_stream_pd_json_response(status, decode_body);
        }

        let Some(prefill_body) = prefill_body else {
            return Self::non_stream_pd_json_response(status, decode_body);
        };

        // Merge logprobs from prefill and decode
        let (Ok(prefill_json), Ok(mut decode_json)) = (
            serde_json::from_slice::<Value>(&prefill_body),
            serde_json::from_slice::<Value>(&decode_body),
        ) else {
            warn!("Failed to parse responses for logprob merging");
            return Self::non_stream_pd_json_response(status, decode_body);
        };

        Self::merge_logprobs_in_json(&prefill_json, &mut decode_json);

        // Return merged response
        match serde_json::to_vec(&decode_json) {
            Ok(body) => Self::non_stream_pd_json_response(status, Bytes::from(body)),
            Err(e) => {
                error!("Failed to serialize merged response: {}", e);
                Self::non_stream_pd_json_response(status, decode_body)
            }
        }
    }

    // Helper to process prefill response and extract body if needed for logprobs
    async fn process_prefill_response(
        &self,
        prefill_response: reqwest::Response,
        prefill_url: &str,
        return_logprob: bool,
    ) -> Result<(StatusCode, Option<Bytes>), Response> {
        let prefill_status = StatusCode::from_u16(prefill_response.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        // Check if prefill succeeded
        if !prefill_status.is_success() {
            // Get error body from prefill
            let error_msg = prefill_response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown prefill error".to_string());

            error!(
                "Prefill server returned error status prefill_url={} status={} body={}",
                prefill_url, prefill_status, error_msg
            );

            // Map prefill_status to appropriate error function
            let error_response = match prefill_status {
                StatusCode::BAD_REQUEST => error::bad_request(
                    "prefill_bad_request",
                    format!("Prefill server error ({prefill_status}): {error_msg}"),
                ),
                StatusCode::NOT_FOUND => error::not_found(
                    "prefill_not_found",
                    format!("Prefill server error ({prefill_status}): {error_msg}"),
                ),
                StatusCode::INTERNAL_SERVER_ERROR => error::internal_error(
                    "prefill_internal_error",
                    format!("Prefill server error ({prefill_status}): {error_msg}"),
                ),
                StatusCode::SERVICE_UNAVAILABLE => error::service_unavailable(
                    "prefill_unavailable",
                    format!("Prefill server error ({prefill_status}): {error_msg}"),
                ),
                StatusCode::BAD_GATEWAY => error::bad_gateway(
                    "prefill_bad_gateway",
                    format!("Prefill server error ({prefill_status}): {error_msg}"),
                ),
                _ => error::internal_error(
                    "prefill_error",
                    format!("Prefill server error ({prefill_status}): {error_msg}"),
                ),
            };
            return Err(error_response);
        }

        // Read prefill body if needed for logprob merging
        let prefill_body = if return_logprob {
            match prefill_response.bytes().await {
                Ok(body) => Some(body),
                Err(e) => {
                    warn!("Failed to read prefill response body for logprobs: {}", e);
                    None
                }
            }
        } else {
            // For non-logprob requests, just consume the response without storing
            debug!("Consuming prefill response body (non-logprob request)");
            match prefill_response.bytes().await {
                Ok(_) => debug!("Prefill response consumed successfully"),
                Err(e) => warn!("Error consuming prefill response: {}", e),
            }
            None
        };

        Ok((prefill_status, prefill_body))
    }

    #[expect(
        clippy::unused_self,
        reason = "method on PDRouter for consistent API; may use self.api_key in future"
    )]
    fn build_post_with_headers(
        &self,
        client: &Client,
        worker: &dyn Worker,
        route: &'static str,
        json_request: &Value,
        headers: Option<&HeaderMap>,
        connection_close: bool,
    ) -> reqwest::RequestBuilder {
        let endpoint_url = worker.endpoint_url(route);
        let mut request = client.post(endpoint_url).json(json_request);
        if connection_close {
            request = request.header("Connection", "close");
        }
        if let Some(headers) = headers {
            for (name, value) in headers {
                if header_utils::should_forward_request_header(name.as_str()) {
                    if let Ok(val) = value.to_str() {
                        request = request.header(name, val);
                    }
                }
            }
        }
        request
    }

    // Helper to merge logprobs from prefill and decode responses
    // Optimized to avoid double cloning by taking ownership of decode array
    fn merge_logprobs_in_json(prefill_json: &Value, decode_json: &mut Value) -> bool {
        if let (Some(prefill_meta), Some(decode_meta)) = (
            prefill_json.get("meta_info"),
            decode_json.get_mut("meta_info"),
        ) {
            if let (Some(prefill_logprobs), Some(decode_logprobs)) = (
                prefill_meta.get("input_token_logprobs"),
                decode_meta.get_mut("input_token_logprobs"),
            ) {
                if let Some(prefill_arr) = prefill_logprobs.as_array() {
                    // Take ownership of decode array to avoid cloning it
                    let decode_arr = std::mem::take(decode_logprobs);
                    if let Value::Array(decode_vec) = decode_arr {
                        // Pre-allocate merged array with exact capacity
                        let mut merged = Vec::with_capacity(prefill_arr.len() + decode_vec.len());
                        merged.extend(prefill_arr.iter().cloned());
                        merged.extend(decode_vec);
                        decode_meta["input_token_logprobs"] = Value::Array(merged);
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Line-anchored detection of the SSE `data: [DONE]` terminal event in a
    /// raw upstream chunk: a match must start at a line boundary and be
    /// immediately followed by a complete empty-line event delimiter within
    /// the same chunk. Payload text that merely contains those bytes never
    /// qualifies — real EOL bytes cannot occur inside a `data:` payload
    /// (JSON escapes them). Requiring the full delimiter also rejects
    /// multi-line events like `data: [DONE]\ndata: x\n\n`, whose joined data
    /// is not exactly `[DONE]`.
    ///
    /// `at_line_start` says whether `chunk` begins at a line boundary. A
    /// sentinel or delimiter split across chunks is never treated as
    /// terminal — every byte is still forwarded and the relay then ends via
    /// upstream EOF, so deferring is always safe while a false positive
    /// kills a live stream.
    fn chunk_contains_done_event(chunk: &[u8], at_line_start: bool) -> bool {
        const DONE_EVENT: &[u8] = b"data: [DONE]";
        // Length of the EOL sequence at `bytes[pos..]`: 2 for \r\n, 1 for a
        // bare \r or \n, 0 if none.
        fn eol_len_at(bytes: &[u8], pos: usize) -> usize {
            match bytes.get(pos) {
                Some(b'\r') => 1 + usize::from(bytes.get(pos + 1) == Some(&b'\n')),
                Some(b'\n') => 1,
                _ => 0,
            }
        }
        let mut from = 0;
        while let Some(pos) = memmem::find(&chunk[from..], DONE_EVENT) {
            let start = from + pos;
            let anchored = match start.checked_sub(1) {
                None => at_line_start,
                Some(prev) => chunk[prev] == b'\n' || chunk[prev] == b'\r',
            };
            if anchored {
                let line_end = start + DONE_EVENT.len();
                let eol1 = eol_len_at(chunk, line_end);
                if eol1 > 0 && eol_len_at(chunk, line_end + eol1) > 0 {
                    return true;
                }
            }
            from = start + 1;
        }
        false
    }

    // Simple helper to merge logprobs in streaming responses
    // Optimized to reduce allocations in the merge path
    fn merge_streaming_logprobs(
        prefill_logprobs: Option<&Value>,
        decode_chunk: &[u8],
        encoder: &mut SseEncoder,
    ) -> Result<Bytes, ()> {
        // Skip non-data chunks
        let chunk_str = std::str::from_utf8(decode_chunk).map_err(|_| ())?;
        if !chunk_str.starts_with("data: ") {
            return Err(());
        }

        // Parse JSON from chunk. The `[DONE]` sentinel must be matched
        // exactly, not by substring: payloads that merely contain that text
        // still need their logprobs merged.
        let json_str = chunk_str.trim_start_matches("data: ").trim();
        if json_str == "[DONE]" {
            return Err(());
        }
        let mut decode_json: Value = serde_json::from_str(json_str).map_err(|_| ())?;

        // Merge prefill logprobs if available
        if let Some(p_logprobs) = prefill_logprobs {
            if let Some(meta) = decode_json.get_mut("meta_info") {
                if let Some(d_logprobs) = meta.get_mut("input_token_logprobs") {
                    if let Some(p_arr) = p_logprobs.as_array() {
                        // Take ownership of decode array to avoid cloning it
                        let decode_arr = std::mem::take(d_logprobs);
                        if let Value::Array(d_vec) = decode_arr {
                            // Pre-allocate merged array with exact capacity
                            let mut merged = Vec::with_capacity(p_arr.len() + d_vec.len());
                            merged.extend(p_arr.iter().cloned());
                            merged.extend(d_vec);
                            *d_logprobs = Value::Array(merged);
                        }
                    }
                }
            }
        }

        // Re-serialize via the shared encoder (reuses its buffer across chunks).
        encoder.encode_data(&decode_json).map_err(|_| ())
    }

    async fn send_health_generate(&self, url: &str) -> Result<StatusCode, reqwest::Error> {
        let response = self.client.get(url).send().await?;
        let status = response.status();
        let mut body = response.bytes_stream();
        while let Some(chunk) = body.next().await {
            let _ = chunk?;
        }
        Ok(status)
    }
}

#[async_trait]
impl RouterTrait for PDRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Note: This endpoint actually causes the model to generate tokens, so we only test one pair

        let AdmittedPdPair {
            prefill,
            decode,
            prefill_guard,
            decode_guard,
        } = match self
            .select_pd_pair_with_admission(None, UNKNOWN_MODEL_ID, None)
            .await
        {
            Ok(pair) => pair,
            Err(error) => return Self::handle_server_selection_error(error),
        };

        let prefill_url = prefill.endpoint_url("/health_generate");
        let decode_url = decode.endpoint_url("/health_generate");
        let (prefill_result, decode_result) = tokio::join!(
            async {
                let result = self.send_health_generate(&prefill_url).await;
                drop(prefill_guard);
                result
            },
            async {
                let result = self.send_health_generate(&decode_url).await;
                drop(decode_guard);
                result
            }
        );

        // Check results
        let mut errors = Vec::new();

        match prefill_result {
            Ok(status) if status.is_success() => {
                debug!(
                    "Health generate passed for prefill server: {}",
                    prefill.url()
                );
            }
            Ok(status) => {
                errors.push(format!(
                    "Prefill {} returned status {}",
                    prefill.url(),
                    status
                ));
            }
            Err(e) => {
                errors.push(format!("Prefill {} error: {}", prefill.url(), e));
            }
        }

        match decode_result {
            Ok(status) if status.is_success() => {
                debug!("Health generate passed for decode server: {}", decode.url());
            }
            Ok(status) => {
                errors.push(format!(
                    "Decode {} returned status {}",
                    decode.url(),
                    status
                ));
            }
            Err(e) => {
                errors.push(format!("Decode {} error: {}", decode.url(), e));
            }
        }

        if errors.is_empty() {
            (
                StatusCode::OK,
                format!(
                    "Health generate passed on selected pair: prefill={}, decode={}",
                    prefill.url(),
                    decode.url()
                ),
            )
                .into_response()
        } else {
            error::service_unavailable(
                "health_generate_failed",
                format!("Health generate failed: {errors:?}"),
            )
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        self.proxy_to_first_prefill_worker("get_server_info", None)
            .await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("get_model_info", Some(headers))
            .await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        body: &GenerateRequest,
        model_id: &str,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.return_logprob.unwrap_or(false);

        let request_text = if self.policies_need_request_text() {
            body.text.as_deref().map(|s| s.to_string())
        } else {
            None
        };

        let bootstrap_batch_size = Self::get_generate_bootstrap_batch_size(body);

        let context = PDRequestContext {
            route: "/generate",
            bootstrap_batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        body: &ChatCompletionRequest,
        model_id: &str,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs;

        let request_text = if self.policies_need_request_text() {
            Some(body.extract_text_for_routing())
        } else {
            None
        };

        // Calculate batch size
        let bootstrap_batch_size = Self::get_chat_bootstrap_batch_size(body);

        let context = PDRequestContext {
            route: "/v1/chat/completions",
            bootstrap_batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        body: &CompletionRequest,
        model_id: &str,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs.is_some();

        let request_text = if self.policies_need_request_text() {
            match &body.prompt {
                StringOrArray::String(s) => Some(s.clone()),
                StringOrArray::Array(v) => v.first().map(|s| s.to_string()),
            }
        } else {
            None
        };

        // Calculate batch size
        let bootstrap_batch_size = Self::get_completion_bootstrap_batch_size(body);

        let context = PDRequestContext {
            route: "/v1/completions",
            bootstrap_batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        body: &RerankRequest,
        model_id: &str,
    ) -> Response {
        // Extract text for cache-aware routing
        let req_text = if self.policies_need_request_text() {
            Some(body.query.clone())
        } else {
            None
        };

        let context = PDRequestContext {
            route: "/v1/rerank",
            bootstrap_batch_size: None,
            is_stream: false,
            return_logprob: false,
            request_text: req_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    fn router_type(&self) -> &'static str {
        "pd"
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, time::Duration};

    use openai_protocol::model_card::ModelCard;

    use super::*;
    use crate::{
        config::PolicyConfig,
        policies::MinimumTokensPolicy,
        worker::{BasicWorkerBuilder, WorkerLoadManager, WorkerType},
    };

    fn create_test_pd_router() -> PDRouter {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));

        PDRouter {
            worker_registry,
            policy_registry,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            api_key: Some("test_api_key".to_string()),
            prefill_admission: None,
        }
    }

    fn create_test_worker(url: String, worker_type: WorkerType, healthy: bool) -> Box<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .build();
        let status = if healthy {
            openai_protocol::worker::WorkerStatus::Ready
        } else {
            openai_protocol::worker::WorkerStatus::NotReady
        };
        worker.set_status(status);
        Box::new(worker)
    }

    #[test]
    fn test_done_event_detection() {
        // Production-incident payload: a delta whose arguments contained the
        // literal sentinel text; the old substring scan treated it as
        // terminal and silently killed the stream.
        let incident: &[u8] = b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"function\":{\"arguments\":\"// data: [DONE]\"}}]}}]}\n\n";
        // (chunk, chunk begins at a line boundary, expected, case)
        let cases: &[(&[u8], bool, bool, &str)] = &[
            (b"data: [DONE]\n\n", true, true, "standalone sentinel"),
            (
                b"data: {\"x\":1}\n\ndata: [DONE]\n\n",
                true,
                true,
                "sentinel after a data event",
            ),
            (b"data: [DONE]\r\n\r\n", true, true, "CRLF endings"),
            (
                b"\ndata: [DONE]\n\n",
                false,
                true,
                "line boundary inside the chunk",
            ),
            (incident, true, false, "sentinel text inside a JSON payload"),
            (
                b"data: [DONE]{\"x\":1}\n\n",
                true,
                false,
                "line continues with payload",
            ),
            (b"data: [DONE]\n\n", false, false, "chunk starts mid-line"),
            (
                b"data: [DONE]",
                true,
                false,
                "possibly a split payload line: defer",
            ),
            (
                b"data: [DONE]\n",
                true,
                false,
                "event delimiter incomplete: defer",
            ),
            (
                b"data: [DONE]\ndata: x\n\n",
                true,
                false,
                "one event, joined data is not [DONE]",
            ),
        ];
        for (chunk, at_line_start, expected, case) in cases {
            assert_eq!(
                PDRouter::chunk_contains_done_event(chunk, *at_line_start),
                *expected,
                "{case}"
            );
        }
    }

    #[test]
    fn test_merge_streaming_logprobs_sentinel_exact_match() {
        let mut encoder = SseEncoder::new();
        // The exact sentinel is skipped (caller forwards it verbatim)
        assert!(
            PDRouter::merge_streaming_logprobs(None, b"data: [DONE]\n\n", &mut encoder).is_err()
        );
        // A payload containing "[DONE]" as text is still processed
        assert!(PDRouter::merge_streaming_logprobs(
            None,
            b"data: {\"text\":\"[DONE]\",\"meta_info\":{}}\n\n",
            &mut encoder
        )
        .is_ok());
    }

    #[test]
    fn test_build_post_uses_dp_base_url_for_logical_worker() {
        let router = create_test_pd_router();
        let worker = BasicWorkerBuilder::new("http://127.0.0.1:30000")
            .worker_type(WorkerType::Decode)
            .dp_config(2, 4)
            .build();

        let request = router
            .build_post_with_headers(
                &router.client,
                &worker,
                "/generate",
                &json!({"text": "hello"}),
                None,
                false,
            )
            .build()
            .expect("request should build");

        assert_eq!(worker.url(), "http://127.0.0.1:30000@2");
        assert_eq!(
            worker.endpoint_url("/generate"),
            "http://127.0.0.1:30000/generate"
        );
        assert_eq!(request.url().as_str(), "http://127.0.0.1:30000/generate");
    }

    #[test]
    fn test_dp_aware_prefill_rank_is_not_overridden_by_minimum_tokens() {
        let router = create_test_pd_router();
        let prefill = BasicWorkerBuilder::new("http://prefill:8000")
            .worker_type(WorkerType::Prefill)
            .dp_config(2, 4)
            .build();
        let decode = BasicWorkerBuilder::new("http://decode:8000")
            .worker_type(WorkerType::Decode)
            .build();

        let load_manager = Arc::new(WorkerLoadManager::new());
        load_manager.update_dp_loads(&HashMap::from([
            (prefill.url().to_owned(), HashMap::from([(0, 0), (2, 100)])),
            (decode.url().to_owned(), HashMap::from([(3, 0)])),
        ]));
        router
            .policy_registry
            .set_dp_rank_policy(Arc::new(MinimumTokensPolicy::new(Some(load_manager))));

        assert_eq!(
            router.select_dp_ranks(&prefill, &decode, Some("one two")),
            (Some(2), Some(3))
        );
    }

    #[test]
    fn test_select_healthy_prefill_worker() {
        let router = create_test_pd_router();

        let healthy_worker =
            create_test_worker("http://healthy".to_string(), WorkerType::Prefill, true);
        let unhealthy_worker =
            create_test_worker("http://unhealthy".to_string(), WorkerType::Prefill, false);
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router
            .worker_registry
            .register_or_replace(Arc::from(unhealthy_worker));
        router
            .worker_registry
            .register_or_replace(Arc::from(healthy_worker));
        router
            .worker_registry
            .register_or_replace(Arc::from(decode_worker));

        let result = router.select_pd_candidate(None, UNKNOWN_MODEL_ID, None, None);

        assert!(result.is_ok());
        let (prefill, _decode) = result.unwrap();

        assert_eq!(prefill.url(), "http://healthy");
        assert!(prefill.is_healthy());
    }

    #[test]
    fn test_select_pd_pair_accepts_model_alias() {
        let router = create_test_pd_router();
        for (url, worker_type) in [
            ("http://prefill", WorkerType::Prefill),
            ("http://decode", WorkerType::Decode),
        ] {
            let worker = BasicWorkerBuilder::new(url)
                .worker_type(worker_type)
                .model(ModelCard::new("GLM-5.2").with_alias("GLM-5.2-Coding"))
                .build();
            worker.set_status(openai_protocol::worker::WorkerStatus::Ready);
            router.worker_registry.register(Arc::new(worker)).unwrap();
        }

        let (prefill, decode) = router
            .select_pd_candidate(None, "GLM-5.2-Coding", None, None)
            .expect("alias should select a PD pair");
        assert_eq!(prefill.url(), "http://prefill");
        assert_eq!(decode.url(), "http://decode");

        assert!(router
            .select_pd_candidate(None, "GLM-5.2-Unknown", None, None)
            .is_err());
    }

    #[test]
    fn test_empty_worker_lists() {
        let router = create_test_pd_router();

        let result = router.select_pd_candidate(None, UNKNOWN_MODEL_ID, None, None);

        assert!(matches!(
            result,
            Err(PDCandidateError::Unavailable(message))
                if message.contains("No prefill workers available")
        ));
    }

    #[tokio::test]
    async fn test_admission_skips_full_prefill_worker() {
        let mut router = create_test_pd_router();
        let full: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://full".to_string(),
            WorkerType::Prefill,
            true,
        ));
        let available: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://available".to_string(),
            WorkerType::Prefill,
            true,
        ));
        let decode: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://decode".to_string(),
            WorkerType::Decode,
            true,
        ));
        for worker in [&full, &available, &decode] {
            router
                .worker_registry
                .register_or_replace(Arc::clone(worker));
        }

        let admission = Arc::new(PrefillAdmission::new(1, 0, Duration::from_secs(1)));
        let occupied = admission
            .admit(None, {
                let full = Arc::clone(&full);
                move |capacity| capacity.select(Arc::clone(&full), ())
            })
            .await
            .unwrap();
        router.prefill_admission = Some(admission);

        let admitted = router
            .select_pd_pair_with_admission(None, UNKNOWN_MODEL_ID, None)
            .await
            .expect("selection should continue after the preferred worker is full");
        let AdmittedPdPair {
            prefill: selected,
            prefill_guard,
            decode_guard,
            ..
        } = admitted;

        assert_eq!(selected.url(), available.url());
        assert_eq!(full.load(), 1);
        assert_eq!(available.load(), 1);
        assert_eq!(decode.load(), 1);

        drop(prefill_guard);
        drop(decode_guard);
        drop(occupied);
        assert_eq!(full.load(), 0);
        assert_eq!(available.load(), 0);
        assert_eq!(decode.load(), 0);
    }

    #[tokio::test]
    async fn test_explicit_target_at_capacity_is_not_reassigned() {
        let mut router = create_test_pd_router();
        router.policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::ConsistentHashing));
        let available: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://available".to_string(),
            WorkerType::Prefill,
            true,
        ));
        let target: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://target".to_string(),
            WorkerType::Prefill,
            true,
        ));
        let decode: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://decode".to_string(),
            WorkerType::Decode,
            true,
        ));
        for worker in [&available, &target, &decode] {
            router
                .worker_registry
                .register_or_replace(Arc::clone(worker));
        }

        let admission = Arc::new(PrefillAdmission::new(1, 0, Duration::from_secs(1)));
        let occupied = admission
            .admit(None, {
                let target = Arc::clone(&target);
                move |capacity| capacity.select(Arc::clone(&target), ())
            })
            .await
            .unwrap();
        router.prefill_admission = Some(admission);

        let target_index = router
            .pd_worker_pools(UNKNOWN_MODEL_ID)
            .0
            .iter()
            .position(|worker| worker.url() == target.url())
            .unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-smg-target-worker",
            target_index.to_string().parse().unwrap(),
        );

        let error = match router
            .select_pd_pair_with_admission(None, UNKNOWN_MODEL_ID, Some(&headers))
            .await
        {
            Ok(_) => panic!("a full explicit target must not be replaced"),
            Err(error) => error,
        };

        assert!(matches!(error, PDSelectionError::QueueFull));
        assert_eq!(available.load(), 0);
        assert_eq!(target.load(), 1);
        assert_eq!(decode.load(), 0);
        drop(occupied);
    }

    #[test]
    fn test_local_queue_rejections_are_not_retried() {
        for error in [PDSelectionError::QueueFull, PDSelectionError::QueueTimeout] {
            let local = PDRouter::handle_server_selection_error(error);
            assert_eq!(local.status(), StatusCode::TOO_MANY_REQUESTS);
            assert!(!PDRouter::should_retry(&local));
        }

        let upstream = Response::builder()
            .status(StatusCode::TOO_MANY_REQUESTS)
            .body(Body::empty())
            .unwrap();
        assert!(PDRouter::should_retry(&upstream));
    }

    #[test]
    fn test_worker_load_metrics() {
        let prefill_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill,
            true,
        ));
        let decode_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://decode".to_string(),
            WorkerType::Decode,
            true,
        ));

        let _prefill_guard = WorkerLoadGuard::new(prefill_worker.clone(), None);
        let _decode_guard = WorkerLoadGuard::new(decode_worker.clone(), None);

        assert_eq!(prefill_worker.load(), 1);
        assert_eq!(decode_worker.load(), 1);

        drop(_prefill_guard);
        drop(_decode_guard);

        assert_eq!(prefill_worker.load(), 0);
        assert_eq!(decode_worker.load(), 0);
    }

    #[tokio::test]
    async fn test_streaming_decode_error_emits_valid_json_sse() {
        let router = create_test_pd_router();

        let decode: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://decode".to_string(),
            WorkerType::Decode,
            true,
        ));

        let upstream = http::Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(r#"{"error":"boom \"quoted\""}"#)
            .unwrap();
        let decode_response = reqwest::Response::from(upstream);

        let context = PDRequestContext {
            route: "/v1/chat/completions",
            bootstrap_batch_size: None,
            is_stream: true,
            return_logprob: false,
            request_text: None,
            model_id: UNKNOWN_MODEL_ID,
            headers: None,
        };

        let decode_guard = WorkerLoadGuard::new(decode.clone(), None);

        let response = router
            .handle_decode_error_response(decode_response, &context, decode, decode_guard)
            .await;

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let frame = std::str::from_utf8(&body).unwrap();

        let payload = frame
            .strip_prefix("data: ")
            .expect("SSE frame must start with `data: `")
            .trim_end();
        let parsed: Value =
            serde_json::from_str(payload).expect("bytes after `data: ` must be valid JSON");
        assert!(
            parsed.get("error").is_some(),
            "parsed SSE payload must contain an `error` field: {parsed}"
        );
    }

    #[tokio::test]
    async fn test_streaming_decode_load_tracks_upstream_lifetime() {
        use futures_util::StreamExt;
        use tokio::time::{timeout, Duration};

        let router = create_test_pd_router();
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router
            .worker_registry
            .register_or_replace(Arc::from(decode_worker));

        let decode_workers = router.worker_registry.get_decode_workers();
        let decode_ref = decode_workers[0].clone();

        assert_eq!(decode_ref.load(), 0);

        let (tx, rx) = mpsc::unbounded_channel();
        let stream = UnboundedReceiverStream::new(rx);
        let decode_guard = WorkerLoadGuard::new(decode_ref.clone(), None);

        let _response = router.create_streaming_response(
            stream.map(Ok),
            StatusCode::OK,
            None,
            false,
            None,
            None,
            Some(decode_guard),
        );

        assert_eq!(decode_ref.load(), 1);
        tx.send(Bytes::from("test data")).unwrap();
        tokio::task::yield_now().await;
        assert_eq!(decode_ref.load(), 1);

        drop(tx);
        timeout(Duration::from_secs(1), async {
            while decode_ref.load() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("decode load should be released when the upstream stream ends");
    }

    #[tokio::test]
    async fn test_streaming_decode_load_releases_on_client_disconnect() {
        use futures_util::StreamExt;
        use tokio::time::{timeout, Duration};

        let router = create_test_pd_router();
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router
            .worker_registry
            .register_or_replace(Arc::from(decode_worker));

        let decode_ref = router.worker_registry.get_decode_workers()[0].clone();
        let (_upstream_tx, upstream_rx) = mpsc::unbounded_channel::<Bytes>();
        let decode_guard = WorkerLoadGuard::new(decode_ref.clone(), None);

        let response = router.create_streaming_response(
            UnboundedReceiverStream::new(upstream_rx).map(Ok),
            StatusCode::OK,
            None,
            false,
            None,
            None,
            Some(decode_guard),
        );

        assert_eq!(decode_ref.load(), 1);
        drop(response);

        timeout(Duration::from_secs(1), async {
            while decode_ref.load() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("decode load should be released when the client disconnects");
    }
}
