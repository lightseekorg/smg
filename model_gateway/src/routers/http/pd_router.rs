use std::{sync::Arc, time::Instant};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use futures_util::StreamExt;
use memchr::memmem;
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatMessage},
    common::{InputIds, StringOrArray},
    completion::CompletionRequest,
    generate::GenerateRequest,
    rerank::RerankRequest,
};
use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, info, warn};

use super::pd_types::api_path;
use crate::{
    config::{types::RetryConfig, RoutingMode},
    core::{
        is_retryable_status, HashRing, RetryExecutor, Worker, WorkerLoadGuard, WorkerRegistry,
        WorkerType, UNKNOWN_MODEL_ID,
    },
    observability::{
        events::{self, Event},
        metrics::{bool_to_static_str, metrics_labels, Metrics},
        otel_trace::inject_trace_context_http,
    },
    policies::{CacheAwarePolicy, LoadBalancingPolicy, PolicyRegistry, SelectWorkerInfo},
    routers::{
        error,
        grpc::utils::{error_type_from_status, route_to_endpoint},
        header_utils, RouterTrait,
    },
};

#[derive(Debug)]
pub struct PDRouter {
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub client: Client,
    pub retry_config: RetryConfig,
    pub api_key: Option<String>,
    pub enable_igw: bool,
    /// Cache match ratio threshold below which a request is considered "cold"
    pub pre_prefill_match_threshold: f32,
    /// Minimum unmatched characters to trigger pre-prefill routing
    pub pre_prefill_unmatched_chars_threshold: usize,
    /// Minimum total tokens (chars as proxy) for pre-prefill eligibility
    pub pre_prefill_min_tokens: usize,
}

#[derive(Clone)]
struct PDRequestContext<'a> {
    route: &'static str,
    batch_size: Option<usize>,
    is_stream: bool,
    return_logprob: bool,
    request_text: Option<String>,
    model_id: Option<&'a str>,
    headers: Option<HeaderMap>,
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
        let url = format!("{}/{}", worker_url, endpoint);
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
                            format!("Failed to read response body: {}", e),
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
                    format!("Failed to proxy request: {}", e),
                )
            }
        }
    }

    pub async fn new(ctx: &Arc<crate::app_context::AppContext>) -> Result<Self, String> {
        // Extract pre-prefill thresholds from routing mode
        let (
            pre_prefill_match_threshold,
            pre_prefill_unmatched_chars_threshold,
            pre_prefill_min_tokens,
        ) = if let RoutingMode::PrefillDecode {
            pre_prefill_match_threshold,
            pre_prefill_unmatched_chars_threshold,
            pre_prefill_min_tokens,
            pre_prefill_urls,
            ..
        } = &ctx.router_config.mode
        {
            if !pre_prefill_urls.is_empty() {
                info!(
                    "Pre-prefill routing enabled: {} pre-prefill workers, match_threshold={}, unmatched_chars_threshold={}, min_tokens={}",
                    pre_prefill_urls.len(),
                    pre_prefill_match_threshold, pre_prefill_unmatched_chars_threshold, pre_prefill_min_tokens
                );

                let prefill_policy = ctx.policy_registry.get_prefill_policy();
                if prefill_policy.as_any().downcast_ref::<CacheAwarePolicy>().is_none() {
                    warn!(
                        "Pre-prefill routing is configured but prefill policy is '{}', not 'cache_aware'. \
                         Pre-prefill will be disabled at runtime.",
                        prefill_policy.name()
                    );
                }
            }

            (
                *pre_prefill_match_threshold,
                *pre_prefill_unmatched_chars_threshold,
                *pre_prefill_min_tokens,
            )
        } else {
            (0.0, 0, 0)
        };

        Ok(PDRouter {
            worker_registry: Arc::clone(&ctx.worker_registry),
            policy_registry: Arc::clone(&ctx.policy_registry),
            client: ctx.client.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            api_key: ctx.router_config.api_key.clone(),
            enable_igw: ctx.router_config.enable_igw,
            pre_prefill_match_threshold,
            pre_prefill_unmatched_chars_threshold,
            pre_prefill_min_tokens,
        })
    }

    fn handle_server_selection_error(error: String) -> Response {
        error!("Failed to select PD pair error={}", error);
        error::service_unavailable(
            "server_selection_failed",
            format!("No available servers: {}", error),
        )
    }

    fn handle_serialization_error(error: impl std::fmt::Display) -> Response {
        error!("Failed to serialize request error={}", error);
        error::internal_error("serialization_failed", "Failed to serialize request")
    }

    fn get_generate_batch_size(req: &GenerateRequest) -> Option<usize> {
        // GenerateRequest doesn't support batch via arrays, only via input_ids
        if let Some(InputIds::Batch(batches)) = &req.input_ids {
            if !batches.is_empty() {
                return Some(batches.len());
            }
        }
        None
    }

    fn get_chat_batch_size(req: &ChatCompletionRequest) -> Option<usize> {
        if let Some(n) = req.n {
            if n > 1 {
                return Some(n as usize);
            }
        }
        None
    }

    fn get_completion_batch_size(req: &CompletionRequest) -> Option<usize> {
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
            // Use static string keys to avoid per-request allocations
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
            // Use static string keys to avoid per-request allocations
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

    async fn execute_dual_dispatch<T: Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        original_request: &T,
        context: PDRequestContext<'_>,
    ) -> Response {
        let start_time = Instant::now();

        let route = context.route;
        let model = context.model_id.unwrap_or(UNKNOWN_MODEL_ID);
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
        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            {
                move |attempt: u32| {
                    // Clone Arc (cheap reference count increment) instead of cloning the entire request
                    let shared_request = Arc::clone(&shared_request);
                    let context = context.clone();
                    async move {
                        let (prefill, decode) = match self
                            .select_pd_pair(
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

                        json_request = match Self::inject_bootstrap_into_value(
                            json_request,
                            prefill.as_ref(),
                            context.batch_size,
                        ) {
                            Ok(v) => v,
                            Err(e) => return Self::handle_serialization_error(e),
                        };

                        let response = self
                            .execute_dual_dispatch_internal(
                                headers,
                                json_request,
                                context,
                                Arc::clone(&prefill),
                                Arc::clone(&decode),
                                start_time,
                            )
                            .await;

                        let status = response.status();
                        let not_error = status.is_success() || status.is_client_error();
                        prefill.record_outcome(not_error);
                        decode.record_outcome(not_error);

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
            |res, _attempt| is_retryable_status(res.status()),
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
        } else if !is_retryable_status(response.status()) {
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
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
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

            let sse_data = format!(
                "data: {{'error': {}}}",
                serde_json::to_string(&error_payload).unwrap_or_default()
            );
            let error_stream = tokio_stream::once(Ok(axum::body::Bytes::from(sse_data)));

            let decode_url = decode.url().to_string();
            self.create_streaming_response(
                error_stream,
                status,
                None,
                context.return_logprob,
                Some(decode_url),
                Some(response_headers),
                prefill,
                decode,
            )
        } else {
            // Handle non-streaming error response
            match res.bytes().await {
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
                    let error_message = format!("Decode server error: {}", e);
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
        json_request: Value,
        context: PDRequestContext<'_>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        _start_time: Instant,
    ) -> Response {
        // For non-streaming: use guard for automatic load management
        // For streaming: load will be managed in create_streaming_response
        let _prefill_guard =
            (!context.is_stream).then(|| WorkerLoadGuard::new(prefill.clone(), headers));
        let _decode_guard =
            (!context.is_stream).then(|| WorkerLoadGuard::new(decode.clone(), headers));

        let mut headers_with_trace = headers.cloned().unwrap_or_default();
        inject_trace_context_http(&mut headers_with_trace);
        let headers = Some(&headers_with_trace);

        // Build both requests
        let prefill_request = self.build_post_with_headers(
            &self.client,
            prefill.url(),
            context.route,
            &json_request,
            headers,
            false,
        );
        let decode_request = self.build_post_with_headers(
            &self.client,
            decode.url(),
            context.route,
            &json_request,
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

        let (prefill_result, decode_result) =
            tokio::join!(prefill_request.send(), decode_request.send());

        events::RequestReceivedEvent {}.emit();

        // Process decode response
        match decode_result {
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                debug!("Decode response status: {}", status);

                if !status.is_success() {
                    error!(
                        "Decode server returned error status decode_url={} status={}",
                        decode.url(),
                        status
                    );

                    return self
                        .handle_decode_error_response(res, &context, prefill, decode)
                        .await;
                }

                // Process prefill response
                let prefill_body = if context.return_logprob {
                    match self
                        .process_prefill_response(
                            prefill_result,
                            prefill.url(),
                            context.return_logprob,
                        )
                        .await
                    {
                        Ok((_, body)) => body,
                        Err(error_response) => return error_response,
                    }
                } else {
                    // Even if we don't need logprobs, we should check prefill status
                    match self
                        .process_prefill_response(prefill_result, prefill.url(), false)
                        .await
                    {
                        Ok((_, body)) => body,
                        Err(error_response) => return error_response,
                    }
                };

                if context.is_stream {
                    // Streaming response
                    let prefill_logprobs = if context.return_logprob {
                        prefill_body
                            .as_ref()
                            .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                            .and_then(|json| {
                                json.pointer("/meta_info/input_token_logprobs").cloned()
                            })
                    } else {
                        None
                    };

                    let response_headers = header_utils::preserve_response_headers(res.headers());

                    self.create_streaming_response(
                        res.bytes_stream(),
                        status,
                        prefill_logprobs,
                        context.return_logprob,
                        None,
                        Some(response_headers),
                        prefill,
                        decode,
                    )
                } else {
                    // Non-streaming response
                    if context.return_logprob {
                        self.process_non_streaming_response(
                            res,
                            status,
                            context.return_logprob,
                            prefill_body,
                        )
                        .await
                    } else {
                        // Direct passthrough when no logprobs needed
                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        match res.bytes().await {
                            Ok(decode_body) => {
                                let mut response = Response::new(Body::from(decode_body));
                                *response.status_mut() = status;
                                *response.headers_mut() = response_headers;
                                response
                            }
                            Err(e) => {
                                error!("Failed to read decode response: {}", e);
                                error::internal_error(
                                    "read_response_failed",
                                    "Failed to read response",
                                )
                            }
                        }
                    }
                }
            }
            Err(e) => {
                error!(
                    decode_url = %decode.url(),
                    error = %e,
                    "Decode request failed"
                );
                error::bad_gateway("decode_server_error", format!("Decode server error: {}", e))
            }
        }
    }

    fn policies_need_request_text(&self) -> bool {
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();
        prefill_policy.needs_request_text() || decode_policy.needs_request_text()
    }

    /// Returns true if pre-prefill routing is configured (any PrePrefill workers registered)
    fn has_pre_prefill(&self) -> bool {
        self.worker_registry
            .get_all()
            .iter()
            .any(|w| matches!(w.worker_type(), WorkerType::PrePrefill))
    }

    /// Extract concatenated text from all chat messages (User, System, Assistant, Developer).
    /// Returns None if no text content is found.
    pub fn extract_chat_request_text(messages: &[ChatMessage]) -> Option<String> {
        let mut result = String::new();
        for msg in messages {
            match msg {
                ChatMessage::User { content, .. }
                | ChatMessage::System { content, .. }
                | ChatMessage::Developer { content, .. } => {
                    let text = content.to_simple_string();
                    if !text.is_empty() {
                        result.push_str(&text);
                    }
                }
                ChatMessage::Assistant { content, .. } => {
                    if let Some(content) = content {
                        let text = content.to_simple_string();
                        if !text.is_empty() {
                            result.push_str(&text);
                        }
                    }
                }
                _ => {}
            }
        }
        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    /// Select prefill worker with pre-prefill routing logic.
    ///
    /// If the request is "cold" (low cache match ratio and enough unmatched content),
    /// route to the pre-prefill worker for asynchronous cache warming.
    /// Otherwise, use the normal prefill policy selection.
    ///
    /// Returns `(worker, is_pre_prefill)` where `is_pre_prefill` indicates
    /// whether the request was routed to the pre-prefill worker.
    fn select_prefill_worker(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&HeaderMap>,
        hash_ring: Option<Arc<HashRing>>,
    ) -> Result<(Arc<dyn Worker>, bool), String> {
        let prefill_policy = self.policy_registry.get_prefill_policy();

        // Check if pre-prefill routing should be attempted
        if let Some(text) = request_text {
            // Only consider pre-prefill for sufficiently long requests
            if text.len() >= self.pre_prefill_min_tokens {
                // Try to get cache match stats from the prefill policy
                if let Some(cache_aware) =
                    prefill_policy.as_any().downcast_ref::<CacheAwarePolicy>()
                {
                    if let Some((matched, total)) =
                        cache_aware.estimate_match_stats(prefill_workers, text)
                    {
                        let match_ratio = if total > 0 {
                            matched as f32 / total as f32
                        } else {
                            0.0
                        };
                        let unmatched = total.saturating_sub(matched);

                        debug!(
                            "Pre-prefill check: match_ratio={:.3}, unmatched={}, threshold={}, unmatched_threshold={}",
                            match_ratio, unmatched, self.pre_prefill_match_threshold, self.pre_prefill_unmatched_chars_threshold
                        );

                        // Route to pre-prefill if the request is "cold"
                        if match_ratio < self.pre_prefill_match_threshold
                            && unmatched >= self.pre_prefill_unmatched_chars_threshold
                        {
                            // Find an available PrePrefill worker by type
                            if let Some(worker) = prefill_workers
                                .iter()
                                .find(|w| {
                                    matches!(w.worker_type(), WorkerType::PrePrefill)
                                        && w.is_available()
                                })
                            {
                                info!(
                                    "Routing cold request to pre-prefill worker: url={}, match_ratio={:.3}, unmatched={}",
                                    worker.url(), match_ratio, unmatched
                                );
                                // Record the assignment so future similar requests hit cache
                                cache_aware.record_assignment(
                                    prefill_workers,
                                    text,
                                    worker.url(),
                                );
                                return Ok((worker.clone(), true));
                            }

                            debug!(
                                "No available PrePrefill worker, falling back to normal routing"
                            );
                        }
                    }
                }
            }
        }

        // Normal prefill worker selection
        let worker = Self::pick_worker_by_policy_arc(
            prefill_workers,
            &*prefill_policy,
            request_text,
            headers,
            hash_ring,
            "prefill",
        )?;
        Ok((worker, false))
    }

    /// Select the decode worker, preferring the pre-prefill decode worker
    /// when the request was routed to the pre-prefill worker.
    fn select_decode_worker(
        &self,
        decode_workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&HeaderMap>,
        hash_ring: Option<Arc<HashRing>>,
        is_pre_prefill: bool,
    ) -> Result<Arc<dyn Worker>, String> {
        // If this was a pre-prefill request, prefer a PrePrefillDecode worker
        if is_pre_prefill {
            if let Some(worker) = decode_workers
                .iter()
                .find(|w| {
                    matches!(w.worker_type(), WorkerType::PrePrefillDecode) && w.is_available()
                })
            {
                debug!(
                    "Using pre-prefill paired decode worker: {}",
                    worker.url()
                );
                return Ok(worker.clone());
            }
            debug!(
                "No available PrePrefillDecode worker, falling back to normal policy"
            );
        }

        // Normal decode worker selection (also used as fallback for pre-prefill)
        let decode_policy = self.policy_registry.get_decode_policy();
        Self::pick_worker_by_policy_arc(
            decode_workers,
            &*decode_policy,
            request_text,
            headers,
            hash_ring,
            "decode",
        )
    }

    async fn select_pd_pair(
        &self,
        request_text: Option<&str>,
        model_id: Option<&str>,
        headers: Option<&HeaderMap>,
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), String> {
        let effective_model_id = if !self.enable_igw { None } else { model_id };

        debug!(
            "Selecting PD pair: enable_igw={}, model_id={:?}, effective_model_id={:?}",
            self.enable_igw, model_id, effective_model_id
        );

        let prefill_workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model(model)
                .iter()
                .filter(|w| {
                    matches!(
                        w.worker_type(),
                        WorkerType::Prefill | WorkerType::PrePrefill
                    )
                })
                .cloned()
                .collect()
        } else {
            self.worker_registry.get_prefill_workers()
        };

        let decode_workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model(model)
                .iter()
                .filter(|w| {
                    matches!(
                        w.worker_type(),
                        WorkerType::Decode | WorkerType::PrePrefillDecode
                    )
                })
                .cloned()
                .collect()
        } else {
            self.worker_registry.get_decode_workers()
        };

        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        // Get cached hash ring for consistent hashing
        let hash_ring = self
            .worker_registry
            .get_hash_ring(effective_model_id.unwrap_or(UNKNOWN_MODEL_ID));

        // Use pre-prefill routing if configured
        let (prefill, is_pre_prefill) = if self.has_pre_prefill() {
            self.select_prefill_worker(
                &prefill_workers,
                request_text,
                headers,
                hash_ring.clone(),
            )?
        } else {
            let worker = Self::pick_worker_by_policy_arc(
                &prefill_workers,
                &*prefill_policy,
                request_text,
                headers,
                hash_ring.clone(),
                "prefill",
            )?;
            (worker, false)
        };

        let decode = if self.has_pre_prefill() {
            self.select_decode_worker(
                &decode_workers,
                request_text,
                headers,
                hash_ring,
                is_pre_prefill,
            )?
        } else {
            Self::pick_worker_by_policy_arc(
                &decode_workers,
                &*decode_policy,
                request_text,
                headers,
                hash_ring,
                "decode",
            )?
        };

        // Record worker selection metrics (Layer 3)
        let model = model_id.unwrap_or(UNKNOWN_MODEL_ID);
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

        Ok((prefill, decode))
    }

    fn pick_worker_by_policy_arc(
        workers: &[Arc<dyn Worker>],
        policy: &dyn LoadBalancingPolicy,
        request_text: Option<&str>,
        headers: Option<&HeaderMap>,
        hash_ring: Option<Arc<HashRing>>,
        worker_type: &str,
    ) -> Result<Arc<dyn Worker>, String> {
        if workers.is_empty() {
            return Err(format!(
                "No {} workers available. Please check if {} servers are configured and healthy.",
                worker_type, worker_type
            ));
        }

        let available_workers: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        if available_workers.is_empty() {
            return Err(format!(
                "No available {} workers (all circuits open or unhealthy)",
                worker_type
            ));
        }

        let selected_idx = policy
            .select_worker(
                &available_workers,
                &SelectWorkerInfo {
                    request_text,
                    tokens: None, // HTTP doesn't have tokens, use gRPC for PrefixHash
                    headers,
                    hash_ring,
                },
            )
            .ok_or_else(|| {
                format!(
                    "Policy {} failed to select a {} worker",
                    policy.name(),
                    worker_type
                )
            })?;

        Ok(available_workers[selected_idx].clone())
    }

    #[allow(clippy::too_many_arguments)]
    fn create_streaming_response(
        &self,
        stream: impl futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
        status: StatusCode,
        prefill_logprobs: Option<Value>,
        return_logprob: bool,
        decode_url: Option<String>,
        headers: Option<HeaderMap>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
    ) -> Response {
        use crate::core::AttachedBody;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            futures_util::pin_mut!(stream);
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        let is_done = memmem::find(&chunk, b"data: [DONE]").is_some();

                        let result = if return_logprob && prefill_logprobs.is_some() {
                            Self::merge_streaming_logprobs(prefill_logprobs.clone(), &chunk)
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
                        let _ = tx.send(Err(format!("Stream error: {}", e)));
                        break;
                    }
                }
            }
        });

        let stream = UnboundedReceiverStream::new(rx);
        let body = Body::from_stream(stream);

        let guards = vec![
            WorkerLoadGuard::new(prefill, headers.as_ref()),
            WorkerLoadGuard::new(decode, headers.as_ref()),
        ];

        let mut response = Response::new(body);
        *response.status_mut() = status;

        let mut response_headers = headers.unwrap_or_default();
        response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        *response.headers_mut() = response_headers;

        AttachedBody::wrap_response(response, guards)
    }

    // Helper to process non-streaming decode response with logprob merging
    async fn process_non_streaming_response(
        &self,
        res: reqwest::Response,
        status: StatusCode,
        return_logprob: bool,
        prefill_body: Option<bytes::Bytes>,
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
            return (status, decode_body).into_response();
        }

        let Some(prefill_body) = prefill_body else {
            return (status, decode_body).into_response();
        };

        // Merge logprobs from prefill and decode
        let (Ok(prefill_json), Ok(mut decode_json)) = (
            serde_json::from_slice::<Value>(&prefill_body),
            serde_json::from_slice::<Value>(&decode_body),
        ) else {
            warn!("Failed to parse responses for logprob merging");
            return (status, decode_body).into_response();
        };

        Self::merge_logprobs_in_json(&prefill_json, &mut decode_json);

        // Return merged response
        match serde_json::to_vec(&decode_json) {
            Ok(body) => (status, body).into_response(),
            Err(e) => {
                error!("Failed to serialize merged response: {}", e);
                (status, decode_body).into_response()
            }
        }
    }

    // Helper to process prefill response and extract body if needed for logprobs
    async fn process_prefill_response(
        &self,
        prefill_result: Result<reqwest::Response, reqwest::Error>,
        prefill_url: &str,
        return_logprob: bool,
    ) -> Result<(StatusCode, Option<bytes::Bytes>), Response> {
        // Check prefill result first - it's critical for disaggregated mode
        let prefill_response = match prefill_result {
            Ok(response) => response,
            Err(e) => {
                error!(
                    "Prefill server failed (CRITICAL) prefill_url={} error={}. Decode will timeout without prefill KV cache.",
                    prefill_url,
                    e
                );

                // Return error immediately - don't wait for decode to timeout
                return Err(error::bad_gateway(
                    "prefill_server_error",
                    format!(
                        "Prefill server error: {}. This will cause decode timeout.",
                        e
                    ),
                ));
            }
        };

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
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::NOT_FOUND => error::not_found(
                    "prefill_not_found",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::INTERNAL_SERVER_ERROR => error::internal_error(
                    "prefill_internal_error",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::SERVICE_UNAVAILABLE => error::service_unavailable(
                    "prefill_unavailable",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::BAD_GATEWAY => error::bad_gateway(
                    "prefill_bad_gateway",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                _ => error::internal_error(
                    "prefill_error",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
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

    fn build_post_with_headers(
        &self,
        client: &Client,
        url: &str,
        route: &'static str,
        json_request: &Value,
        headers: Option<&HeaderMap>,
        connection_close: bool,
    ) -> reqwest::RequestBuilder {
        let mut request = client.post(api_path(url, route)).json(json_request);
        if connection_close {
            request = request.header("Connection", "close");
        }
        if let Some(headers) = headers {
            for (name, value) in headers.iter() {
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

    // Simple helper to merge logprobs in streaming responses
    // Optimized to reduce allocations in the merge path
    fn merge_streaming_logprobs(
        prefill_logprobs: Option<Value>,
        decode_chunk: &[u8],
    ) -> Result<bytes::Bytes, ()> {
        // Skip non-data chunks
        let chunk_str = std::str::from_utf8(decode_chunk).map_err(|_| ())?;
        if !chunk_str.starts_with("data: ") || chunk_str.contains("[DONE]") {
            return Err(());
        }

        // Parse JSON from chunk
        let json_str = chunk_str.trim_start_matches("data: ").trim();
        let mut decode_json: Value = serde_json::from_str(json_str).map_err(|_| ())?;

        // Merge prefill logprobs if available
        if let Some(ref p_logprobs) = prefill_logprobs {
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

        // Re-serialize
        let merged_str = format!(
            "data: {}\n\n",
            serde_json::to_string(&decode_json).unwrap_or_default()
        );
        Ok(bytes::Bytes::from(merged_str))
    }
}

#[async_trait]
impl RouterTrait for PDRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Note: This endpoint actually causes the model to generate tokens, so we only test one pair

        // Select a random worker pair using the policy
        let (prefill, decode) = match self.select_pd_pair(None, None, None).await {
            Ok(pair) => pair,
            Err(e) => {
                return error::service_unavailable(
                    "no_healthy_worker_pair",
                    format!("No healthy worker pair available: {}", e),
                );
            }
        };

        let prefill_url = format!("{}/health_generate", prefill.url());
        let (prefill_result, decode_result) = tokio::join!(
            self.client.get(&prefill_url).send(),
            self.client
                .get(format!("{}/health_generate", decode.url()))
                .send()
        );

        // Check results
        let mut errors = Vec::new();

        match prefill_result {
            Ok(res) if res.status().is_success() => {
                debug!(
                    "Health generate passed for prefill server: {}",
                    prefill.url()
                );
            }
            Ok(res) => {
                errors.push(format!(
                    "Prefill {} returned status {}",
                    prefill.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Prefill {} error: {}", prefill.url(), e));
            }
        }

        match decode_result {
            Ok(res) if res.status().is_success() => {
                debug!("Health generate passed for decode server: {}", decode.url());
            }
            Ok(res) => {
                errors.push(format!(
                    "Decode {} returned status {}",
                    decode.url(),
                    res.status()
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
                format!("Health generate failed: {:?}", errors),
            )
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // Get info from the first decode server to match sglang's server info format
        // Note: We use decode workers for server info to match expected format
        self.proxy_to_first_prefill_worker("get_server_info", None)
            .await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("v1/models", Some(headers))
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
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.return_logprob.unwrap_or(false);

        let request_text = if self.policies_need_request_text() || self.has_pre_prefill() {
            body.text.as_deref().map(|s| s.to_string())
        } else {
            None
        };

        let batch_size = Self::get_generate_batch_size(body);

        let context = PDRequestContext {
            route: "/generate",
            batch_size,
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
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs;

        let request_text = if self.policies_need_request_text() || self.has_pre_prefill() {
            Self::extract_chat_request_text(&body.messages)
        } else {
            None
        };

        // Calculate batch size
        let batch_size = Self::get_chat_batch_size(body);

        let context = PDRequestContext {
            route: "/v1/chat/completions",
            batch_size,
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
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs.is_some();

        let request_text = if self.policies_need_request_text() || self.has_pre_prefill() {
            match &body.prompt {
                StringOrArray::String(s) => Some(s.clone()),
                StringOrArray::Array(v) => v.first().map(|s| s.to_string()),
            }
        } else {
            None
        };

        // Calculate batch size
        let batch_size = Self::get_completion_batch_size(body);

        let context = PDRequestContext {
            route: "/v1/completions",
            batch_size,
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
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Extract text for cache-aware routing
        let req_text = if self.policies_need_request_text() {
            Some(body.query.clone())
        } else {
            None
        };

        let context = PDRequestContext {
            route: "/v1/rerank",
            batch_size: None,
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
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};
    use crate::policies::CacheAwarePolicy;
    use openai_protocol::chat::MessageContent;

    fn create_test_pd_router() -> PDRouter {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::RoundRobin));

        PDRouter {
            worker_registry,
            policy_registry,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            api_key: Some("test_api_key".to_string()),
            enable_igw: false,
            pre_prefill_match_threshold: 0.1,
            pre_prefill_unmatched_chars_threshold: 10000,
            pre_prefill_min_tokens: 10000,
        }
    }

    fn create_test_worker(url: String, worker_type: WorkerType, healthy: bool) -> Box<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .build();
        worker.set_healthy(healthy);
        Box::new(worker)
    }

    #[tokio::test]
    async fn test_select_healthy_prefill_worker() {
        let router = create_test_pd_router();

        let healthy_worker =
            create_test_worker("http://healthy".to_string(), WorkerType::Prefill, true);
        let unhealthy_worker =
            create_test_worker("http://unhealthy".to_string(), WorkerType::Prefill, false);
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(unhealthy_worker));
        router.worker_registry.register(Arc::from(healthy_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        let result = router.select_pd_pair(None, None, None).await;

        assert!(result.is_ok());
        let (prefill, _decode) = result.unwrap();

        assert_eq!(prefill.url(), "http://healthy");
        assert!(prefill.is_healthy());
    }

    #[tokio::test]
    async fn test_empty_worker_lists() {
        let router = create_test_pd_router();

        let result = router.select_pd_pair(None, None, None).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No prefill workers available"));
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
    async fn test_streaming_load_tracking() {
        use futures_util::StreamExt;
        use tokio::time::{sleep, Duration};

        let router = create_test_pd_router();

        let prefill_worker =
            create_test_worker("http://prefill".to_string(), WorkerType::Prefill, true);
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(prefill_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        let prefill_workers = router.worker_registry.get_prefill_workers();
        let decode_workers = router.worker_registry.get_decode_workers();

        let prefill_ref = prefill_workers[0].clone();
        let decode_ref = decode_workers[0].clone();

        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stream = UnboundedReceiverStream::new(rx);

        {
            let response = router.create_streaming_response(
                stream.map(Ok),
                StatusCode::OK,
                None,
                false,
                None,
                None,
                prefill_ref.clone(),
                decode_ref.clone(),
            );

            // Guards are now attached to response body, so load should be 1
            assert_eq!(prefill_ref.load(), 1);
            assert_eq!(decode_ref.load(), 1);

            tx.send(bytes::Bytes::from("test data")).unwrap();

            sleep(Duration::from_millis(10)).await;

            // Load still 1 while response body exists
            assert_eq!(prefill_ref.load(), 1);
            assert_eq!(decode_ref.load(), 1);

            drop(tx);

            // Response (and its body with guards) dropped here
            drop(response);
        }

        // Guards dropped when response dropped
        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);
    }

    // ==================== Pre-Prefill Routing Tests ====================

    /// Helper: create a PDRouter with pre-prefill enabled and cache-aware prefill policy.
    /// Workers must be registered and policy initialized separately.
    fn create_pre_prefill_router(
        worker_registry: Arc<WorkerRegistry>,
        min_tokens: usize,
        unmatched_threshold: usize,
    ) -> PDRouter {
        use crate::policies::PolicyFactory;

        let cache_aware_config = crate::config::PolicyConfig::CacheAware {
            cache_threshold: 0.3,
            balance_abs_threshold: 64,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 0,
            max_tree_size: 100_000,
        };
        let policy_registry = Arc::new(PolicyRegistry::new(cache_aware_config.clone()));
        // Set the prefill policy explicitly so get_prefill_policy() returns cache-aware
        let prefill_policy = PolicyFactory::create_from_config(&cache_aware_config);
        policy_registry.set_prefill_policy(prefill_policy);

        PDRouter {
            worker_registry,
            policy_registry,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            api_key: None,
            enable_igw: false,
            pre_prefill_match_threshold: 0.1,
            pre_prefill_unmatched_chars_threshold: unmatched_threshold,
            pre_prefill_min_tokens: min_tokens,
        }
    }

    // ---------- extract_chat_request_text ----------

    #[test]
    fn test_extract_chat_request_text_user_messages() {
        let messages = vec![
            ChatMessage::User {
                content: MessageContent::Text("Hello world".to_string()),
                name: None,
            },
            ChatMessage::User {
                content: MessageContent::Text(" how are you".to_string()),
                name: None,
            },
        ];
        let result = PDRouter::extract_chat_request_text(&messages);
        assert_eq!(result, Some("Hello world how are you".to_string()));
    }

    #[test]
    fn test_extract_chat_request_text_system_and_user() {
        let messages = vec![
            ChatMessage::System {
                content: MessageContent::Text("You are a helpful assistant.".to_string()),
                name: None,
            },
            ChatMessage::User {
                content: MessageContent::Text("Tell me a story.".to_string()),
                name: None,
            },
        ];
        let result = PDRouter::extract_chat_request_text(&messages);
        assert_eq!(
            result,
            Some("You are a helpful assistant.Tell me a story.".to_string())
        );
    }

    #[test]
    fn test_extract_chat_request_text_with_assistant() {
        let messages = vec![
            ChatMessage::User {
                content: MessageContent::Text("Hi".to_string()),
                name: None,
            },
            ChatMessage::Assistant {
                content: Some(MessageContent::Text("Hello!".to_string())),
                name: None,
                tool_calls: None,
                reasoning_content: None,
            },
            ChatMessage::User {
                content: MessageContent::Text("More".to_string()),
                name: None,
            },
        ];
        let result = PDRouter::extract_chat_request_text(&messages);
        assert_eq!(result, Some("HiHello!More".to_string()));
    }

    #[test]
    fn test_extract_chat_request_text_empty() {
        let messages: Vec<ChatMessage> = vec![];
        let result = PDRouter::extract_chat_request_text(&messages);
        assert_eq!(result, None);
    }

    // ---------- has_pre_prefill ----------

    #[test]
    fn test_has_pre_prefill_disabled() {
        let router = create_test_pd_router();
        assert!(!router.has_pre_prefill());
    }

    #[test]
    fn test_has_pre_prefill_enabled() {
        let wr = Arc::new(WorkerRegistry::new());
        // Register a PrePrefill worker so has_pre_prefill() returns true
        let pp = create_test_worker("http://pp:8000".to_string(), WorkerType::PrePrefill, true);
        wr.register(Arc::from(pp));
        let router = create_pre_prefill_router(wr, 10, 10);
        assert!(router.has_pre_prefill());
    }

    // ---------- select_pd_pair: normal path (no pre-prefill configured) ----------

    #[tokio::test]
    async fn test_select_pd_pair_without_pre_prefill() {
        let router = create_test_pd_router();

        let prefill_worker =
            create_test_worker("http://prefill:8000".to_string(), WorkerType::Prefill, true);
        let decode_worker =
            create_test_worker("http://decode:8000".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(prefill_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        let result = router.select_pd_pair(Some("test text"), None, None).await;
        assert!(result.is_ok());
        let (prefill, decode) = result.unwrap();
        assert_eq!(prefill.url(), "http://prefill:8000");
        assert_eq!(decode.url(), "http://decode:8000");
    }

    // ---------- select_pd_pair: short request bypasses pre-prefill ----------

    #[tokio::test]
    async fn test_short_request_bypasses_pre_prefill() {
        let wr = Arc::new(WorkerRegistry::new());

        // Register a PrePrefill worker so has_pre_prefill() returns true
        let pp = create_test_worker("http://pp:8000".to_string(), WorkerType::PrePrefill, true);
        wr.register(Arc::from(pp));

        // min_tokens=100 so "short" (5 chars) is below threshold
        let router = create_pre_prefill_router(
            wr.clone(),
            100,  // min_tokens - high bar, "short" won't reach it
            10,   // unmatched_chars_threshold
        );

        // Also register normal prefill and decode workers
        let prefill = create_test_worker("http://prefill:8000".to_string(), WorkerType::Prefill, true);
        let decode = create_test_worker("http://decode:8000".to_string(), WorkerType::Decode, true);

        wr.register(Arc::from(prefill));
        wr.register(Arc::from(decode));

        let prefill_policy = router.policy_registry.get_prefill_policy();
        if let Some(ca) = prefill_policy.as_any().downcast_ref::<CacheAwarePolicy>() {
            ca.init_workers(&wr.get_prefill_workers());
        }

        // "short" is only 5 chars, below min_tokens=100 -> normal routing
        // The pre-prefill special path should NOT be triggered, but normal policy
        // may select any available prefill-type worker
        let result = router.select_pd_pair(Some("short"), None, None).await;
        assert!(result.is_ok());
        let (p, d) = result.unwrap();
        assert!(
            p.url() == "http://prefill:8000" || p.url() == "http://pp:8000",
            "Expected a prefill worker, got: {}", p.url()
        );
        assert_eq!(d.url(), "http://decode:8000");
    }

    // ---------- select_pd_pair: cold request routes to pre-prefill worker ----------

    #[tokio::test]
    async fn test_cold_request_routes_to_pre_prefill() {
        let wr = Arc::new(WorkerRegistry::new());

        let prefill = create_test_worker("http://prefill:8000".to_string(), WorkerType::Prefill, true);
        let pp = create_test_worker("http://pp:8000".to_string(), WorkerType::PrePrefill, true);
        let decode = create_test_worker("http://decode:8000".to_string(), WorkerType::Decode, true);
        let ppd = create_test_worker("http://ppd:8000".to_string(), WorkerType::PrePrefillDecode, true);

        wr.register(Arc::from(prefill));
        wr.register(Arc::from(pp));
        wr.register(Arc::from(decode));
        wr.register(Arc::from(ppd));

        // min_tokens=10, unmatched_threshold=10 -> easy to trigger with a 20-char cold string
        let router = create_pre_prefill_router(
            wr.clone(),
            10,  // min_tokens
            10,  // unmatched_chars_threshold
        );

        // Init cache-aware policy with prefill workers
        let prefill_policy = router.policy_registry.get_prefill_policy();
        let ca = prefill_policy
            .as_any()
            .downcast_ref::<CacheAwarePolicy>()
            .expect("prefill policy must be CacheAwarePolicy for pre-prefill to work");
        ca.init_workers(&wr.get_prefill_workers());

        // Verify the decision chain step by step:
        let cold_text = "x".repeat(40); // 40 chars, well above min_tokens=10

        // Step 1: text.len() >= min_tokens? YES (40 >= 10)
        assert!(cold_text.len() >= router.pre_prefill_min_tokens);

        // Step 2: estimate_match_stats returns Some with matched=0 (cold)
        let stats = ca.estimate_match_stats(&wr.get_prefill_workers(), &cold_text);
        assert!(stats.is_some(), "estimate_match_stats should return Some after init_workers");
        let (matched, total) = stats.unwrap();
        assert_eq!(matched, 0, "no prior cache entries -> matched should be 0");
        assert_eq!(total, 40, "total should equal input length");

        // Step 3: match_ratio < threshold? YES (0.0 < 0.1)
        let match_ratio = matched as f32 / total as f32;
        assert!(match_ratio < router.pre_prefill_match_threshold);

        // Step 4: unmatched >= unmatched_threshold? YES (40 >= 10)
        let unmatched = total - matched;
        assert!(unmatched >= router.pre_prefill_unmatched_chars_threshold);

        // Now call select_pd_pair and verify the final routing decision
        let result = router.select_pd_pair(Some(&cold_text), None, None).await;
        assert!(result.is_ok());
        let (prefill_selected, decode_selected) = result.unwrap();

        // Cold request should be routed to the pre-prefill worker
        assert_eq!(prefill_selected.url(), "http://pp:8000",
            "cold request should route to pre-prefill worker");
        // And paired with the pre-prefill decode worker
        assert_eq!(decode_selected.url(), "http://ppd:8000",
            "pre-prefill should pair with its dedicated decode worker");
    }

    // ---------- select_pd_pair: warm request uses normal routing ----------

    #[tokio::test]
    async fn test_warm_request_uses_normal_routing() {
        let wr = Arc::new(WorkerRegistry::new());

        let prefill = create_test_worker("http://prefill:8000".to_string(), WorkerType::Prefill, true);
        let pp = create_test_worker("http://pp:8000".to_string(), WorkerType::PrePrefill, true);
        let decode = create_test_worker("http://decode:8000".to_string(), WorkerType::Decode, true);
        let ppd = create_test_worker("http://ppd:8000".to_string(), WorkerType::PrePrefillDecode, true);

        wr.register(Arc::from(prefill));
        wr.register(Arc::from(pp));
        wr.register(Arc::from(decode));
        wr.register(Arc::from(ppd));

        let router = create_pre_prefill_router(
            wr.clone(),
            10, // min_tokens
            10, // unmatched_chars_threshold
        );

        // Init cache-aware and pre-populate the tree so request is "warm"
        let prefill_policy = router.policy_registry.get_prefill_policy();
        let ca = prefill_policy
            .as_any()
            .downcast_ref::<CacheAwarePolicy>()
            .expect("prefill policy must be CacheAwarePolicy");
        ca.init_workers(&wr.get_prefill_workers());

        // Pre-populate: record a previous assignment for this exact text to http://prefill:8000
        let warm_text = "this is a warm request with enough characters to exceed thresholds easily";
        ca.record_assignment(&wr.get_prefill_workers(), warm_text, "http://prefill:8000");

        // Verify the tree now has high match for this text
        let stats = ca.estimate_match_stats(&wr.get_prefill_workers(), warm_text);
        assert!(stats.is_some());
        let (matched, total) = stats.unwrap();
        assert_eq!(matched, total, "exact text should fully match");
        let match_ratio = matched as f32 / total as f32;
        // match_ratio should be 1.0, well above threshold of 0.1
        assert!(match_ratio >= router.pre_prefill_match_threshold,
            "match_ratio {} should be >= threshold {} for warm request",
            match_ratio, router.pre_prefill_match_threshold);

        // Same text again -> high match ratio -> should NOT route to pre-prefill
        let result = router.select_pd_pair(Some(warm_text), None, None).await;
        assert!(result.is_ok());
        let (prefill_selected, _) = result.unwrap();

        // Cache-aware routing should return http://prefill:8000 (the recorded tenant)
        assert_eq!(prefill_selected.url(), "http://prefill:8000",
            "warm request should route to the cached worker, not pre-prefill");
    }

    // ---------- select_pd_pair: pre-prefill worker unavailable -> fallback ----------

    #[tokio::test]
    async fn test_pre_prefill_worker_unavailable_falls_back() {
        let wr = Arc::new(WorkerRegistry::new());

        let prefill = create_test_worker("http://prefill:8000".to_string(), WorkerType::Prefill, true);
        // Pre-prefill worker is registered but UNHEALTHY
        let pp = create_test_worker("http://pp:8000".to_string(), WorkerType::PrePrefill, false);
        let decode = create_test_worker("http://decode:8000".to_string(), WorkerType::Decode, true);

        wr.register(Arc::from(prefill));
        wr.register(Arc::from(pp));
        wr.register(Arc::from(decode));

        let router = create_pre_prefill_router(
            wr.clone(),
            10, // min_tokens
            10, // unmatched_chars_threshold
        );

        let prefill_policy = router.policy_registry.get_prefill_policy();
        if let Some(ca) = prefill_policy.as_any().downcast_ref::<CacheAwarePolicy>() {
            ca.init_workers(&wr.get_prefill_workers());
        }

        let cold_text = "x".repeat(100); // cold, long enough
        let result = router.select_pd_pair(Some(&cold_text), None, None).await;
        assert!(result.is_ok());
        let (prefill_selected, _) = result.unwrap();

        // Pre-prefill worker is unhealthy, so should fallback to normal prefill worker
        assert_eq!(prefill_selected.url(), "http://prefill:8000");
    }

    // ---------- select_pd_pair: no request text -> normal routing ----------

    #[tokio::test]
    async fn test_no_request_text_uses_normal_routing() {
        let wr = Arc::new(WorkerRegistry::new());

        // Register a PrePrefill worker so has_pre_prefill() returns true
        let pp = create_test_worker("http://pp:8000".to_string(), WorkerType::PrePrefill, true);
        wr.register(Arc::from(pp));

        // Also register normal workers and a PrePrefillDecode worker
        let prefill = create_test_worker("http://prefill:8000".to_string(), WorkerType::Prefill, true);
        let decode = create_test_worker("http://decode:8000".to_string(), WorkerType::Decode, true);
        let ppd = create_test_worker("http://ppd:8000".to_string(), WorkerType::PrePrefillDecode, true);

        wr.register(Arc::from(prefill));
        wr.register(Arc::from(decode));
        wr.register(Arc::from(ppd));

        let router = create_pre_prefill_router(
            wr.clone(),
            10,
            10,
        );

        let prefill_policy = router.policy_registry.get_prefill_policy();
        if let Some(ca) = prefill_policy.as_any().downcast_ref::<CacheAwarePolicy>() {
            ca.init_workers(&wr.get_prefill_workers());
        }

        // No request text -> can't estimate match -> normal routing
        // The pre-prefill decode worker should NOT be selected since is_pre_prefill=false
        let result = router.select_pd_pair(None, None, None).await;
        assert!(result.is_ok());
        let (prefill_selected, decode_selected) = result.unwrap();
        // Any prefill-type worker is valid (normal policy selection)
        assert!(
            prefill_selected.url() == "http://prefill:8000" || prefill_selected.url() == "http://pp:8000",
            "Expected a prefill worker, got: {}", prefill_selected.url()
        );
        // Decode should use normal policy (not the pre-prefill paired decode)
        assert!(
            decode_selected.url() == "http://decode:8000" || decode_selected.url() == "http://ppd:8000",
            "Expected a decode worker, got: {}", decode_selected.url()
        );
    }
}
