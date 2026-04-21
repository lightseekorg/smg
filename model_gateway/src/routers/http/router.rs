use std::{sync::Arc, time::Instant};

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures_util::{stream, StreamExt};
use openai_protocol::{
    chat::ChatCompletionRequest,
    classify::ClassifyRequest,
    common::GenerationRequest,
    completion::CompletionRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    rerank::{RerankRequest, RerankResponse, RerankResult},
    responses::ResponsesRequest,
    transcription::TranscriptionRequest,
};
use reqwest::{
    multipart::{Form, Part},
    Client,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::{ReceiverStream, UnboundedReceiverStream};
use tracing::error;

use crate::{
    app_context::AppContext,
    config::types::RetryConfig,
    observability::{
        events::{self, Event},
        metrics::{bool_to_static_str, metrics_labels, Metrics},
        otel_trace::inject_trace_context_http,
    },
    policies::{PolicyRegistry, SelectWorkerInfo},
    routers::{
        common::{
            header_utils,
            retry::{is_retryable_status, RetryExecutor},
        },
        error::{self, extract_error_code_from_response},
        grpc::utils::{error_type_from_status, route_to_endpoint},
        AudioFile, RouterTrait,
    },
    worker::{AttachedBody, ConnectionMode, Worker, WorkerLoadGuard, WorkerRegistry, WorkerType},
};

/// Regular router that uses injected load balancing policies
pub struct Router {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    client: Client,
    retry_config: RetryConfig,
}

impl std::fmt::Debug for Router {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Router")
            .field("worker_registry", &self.worker_registry)
            .field("policy_registry", &self.policy_registry)
            .field("client", &self.client)
            .field("retry_config", &self.retry_config)
            .finish()
    }
}

impl Router {
    /// Create a new router with injected policy and client
    #[expect(
        clippy::unused_async,
        reason = "async for API consistency with other router constructors"
    )]
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        Ok(Router {
            worker_registry: ctx.worker_registry.clone(),
            policy_registry: ctx.policy_registry.clone(),
            client: ctx.client.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
        })
    }

    fn select_first_worker(&self) -> Result<String, String> {
        let workers = self.worker_registry.get_all();
        let healthy_workers: Vec<_> = workers.iter().filter(|w| w.is_healthy()).collect();
        if healthy_workers.is_empty() {
            Err("No workers are available".to_string())
        } else {
            Ok(healthy_workers[0].url().to_string())
        }
    }

    async fn proxy_get_request(&self, req: Request<Body>, endpoint: &str) -> Response {
        let headers = header_utils::copy_request_headers(&req);

        match self.select_first_worker() {
            Ok(worker_url) => {
                let mut request_builder = self.client.get(format!("{worker_url}/{endpoint}"));
                for (name, value) in headers {
                    if header_utils::should_forward_request_header(&name) {
                        request_builder = request_builder.header(name, value);
                    }
                }

                match request_builder.send().await {
                    Ok(res) => {
                        let status = StatusCode::from_u16(res.status().as_u16())
                            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                        // Preserve headers from backend
                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        match res.bytes().await {
                            Ok(body) => {
                                let mut response = Response::new(Body::from(body));
                                *response.status_mut() = status;
                                *response.headers_mut() = response_headers;
                                response
                            }
                            Err(e) => error::internal_error(
                                "read_response_failed",
                                format!("Failed to read response: {e}"),
                            ),
                        }
                    }
                    Err(e) => convert_reqwest_error(e),
                }
            }
            Err(e) => error::service_unavailable("no_workers", e),
        }
    }

    /// Select worker considering circuit breaker state.
    /// Filters to workers serving the specified model. When model is "unknown"
    /// (generate endpoint without model), considers all HTTP workers.
    fn select_worker_for_model(
        &self,
        model_id: &str,
        text: Option<&str>,
        headers: Option<&HeaderMap>,
    ) -> Option<Arc<dyn Worker>> {
        // UNKNOWN_MODEL_ID means caller didn't specify a model — find any available worker
        let model_filter = if model_id == crate::worker::UNKNOWN_MODEL_ID {
            None
        } else {
            Some(model_id)
        };
        let workers = self.worker_registry.get_workers_filtered(
            model_filter,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Http),
            None,  // any runtime type
            false, // get all workers, we'll filter by is_available() next
        );

        let available: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();
        if available.is_empty() {
            return None;
        }

        // Get the appropriate policy for this model
        let policy = self.policy_registry.get_policy_or_default(model_id);

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self.worker_registry.get_hash_ring(model_id);

        let idx = policy.select_worker(
            &available,
            &SelectWorkerInfo {
                request_text: text,
                tokens: None, // HTTP doesn't have tokens, use gRPC for PrefixHash
                headers,
                hash_ring,
            },
        )?;

        // Record worker selection metric (Layer 3)
        Metrics::record_worker_selection(
            metrics_labels::WORKER_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            model_id,
            policy.name(),
        );

        Some(available[idx].clone())
    }

    pub async fn route_typed_request<T: GenerationRequest + serde::Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        model_id: &str,
    ) -> Response {
        let start = Instant::now();
        let is_stream = typed_req.is_stream();
        let text = typed_req.extract_text_for_routing();
        let model = model_id;
        let endpoint = route_to_endpoint(route);

        // Record request start (Layer 2)
        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint,
            bool_to_static_str(is_stream),
        );

        // Use per-model retry config if set by a worker, otherwise fall back to router default.
        let per_model_retry_config = self.worker_registry.get_retry_config(model_id);
        let retry_config = per_model_retry_config
            .as_ref()
            .unwrap_or(&self.retry_config);

        let response = RetryExecutor::execute_response_with_retry(
            retry_config,
            // operation per attempt
            |_: u32| async {
                let res = self
                    .route_typed_request_once(headers, typed_req, route, model_id, is_stream, &text)
                    .await;

                // Need to be outside `route_typed_request_once` because that function has multiple return paths
                Metrics::record_router_upstream_response(
                    metrics_labels::ROUTER_HTTP,
                    res.status().as_u16(),
                    extract_error_code_from_response(&res),
                );

                res
            },
            // should_retry predicate
            |res, _attempt| is_retryable_status(res.status()),
            // on_backoff hook
            |delay, attempt| {
                // Layer 3 worker metrics
                Metrics::record_worker_retry(metrics_labels::WORKER_REGULAR, endpoint);
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            // on_exhausted hook
            || {
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_REGULAR, endpoint);
            },
        )
        .await;

        if response.status().is_success() {
            let duration = start.elapsed();
            Metrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                duration,
            );
        } else if !is_retryable_status(response.status()) {
            Metrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                error_type_from_status(response.status()),
            );
        }

        response
    }

    async fn route_typed_request_once<T: GenerationRequest + serde::Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        model_id: &str,
        is_stream: bool,
        text: &str,
    ) -> Response {
        let worker = match self.select_worker_for_model(model_id, Some(text), headers) {
            Some(w) => w,
            None => {
                // Distinguish "no workers for this model" from "workers exist but unavailable"
                let model_filter = if model_id == crate::worker::UNKNOWN_MODEL_ID {
                    None
                } else {
                    Some(model_id)
                };
                let total = self.worker_registry.get_workers_filtered(
                    model_filter,
                    Some(WorkerType::Regular),
                    Some(ConnectionMode::Http),
                    None,
                    false,
                );
                return if total.is_empty() {
                    error::model_not_found(model_id)
                } else {
                    error::service_unavailable(
                        "no_available_workers",
                        "All workers are unavailable (circuit breaker open or unhealthy)",
                    )
                };
            }
        };

        let policy = self.policy_registry.get_policy_or_default(model_id);

        let load_guard = ["cache_aware", "manual"]
            .contains(&policy.name())
            .then(|| WorkerLoadGuard::new(worker.clone(), headers));

        // Note: Using borrowed reference avoids heap allocation
        events::RequestSentEvent { url: worker.url() }.emit();
        let mut headers_with_trace = headers.cloned().unwrap_or_default();
        inject_trace_context_http(&mut headers_with_trace);
        let headers = Some(&headers_with_trace);

        let response = self
            .send_typed_request(
                headers,
                typed_req,
                route,
                worker.as_ref(),
                is_stream,
                load_guard,
            )
            .await;

        events::RequestReceivedEvent {}.emit();

        let status = response.status();
        worker.record_outcome(status.as_u16());

        // Record worker errors for server errors (5xx)
        if status.is_server_error() {
            Metrics::record_worker_error(
                metrics_labels::WORKER_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                error_type_from_status(status),
            );
        }

        response
    }

    // Generic simple routing for GET/POST without JSON body
    async fn route_simple_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
        method: Method,
    ) -> Response {
        // TODO: currently the sglang worker is using in-memory state management, so this implementation has to fan out to all workers.
        // Eventually, we need to have router to manage the chat history with a proper database, will update this implementation accordingly.
        let workers = self.worker_registry.get_all();
        if workers.is_empty() {
            return error::service_unavailable("no_workers", "No available workers");
        }

        let filtered_headers: Vec<_> = headers
            .map(|hdrs| {
                hdrs.iter()
                    .filter(|(name, _)| header_utils::should_forward_request_header(name.as_str()))
                    .collect()
            })
            .unwrap_or_default();

        let futures: Vec<_> = workers
            .into_iter()
            .map(|worker| {
                let url = format!("{}/{}", worker.base_url(), endpoint);
                let client = self.client.clone();
                let method = method.clone();

                let headers = filtered_headers.clone();

                let api_key = worker.api_key().cloned();

                async move {
                    let mut request_builder = match method {
                        Method::GET => client.get(url),
                        Method::POST => client.post(url),
                        _ => {
                            return Err(error::method_not_allowed(
                                "unsupported_method",
                                "Unsupported method for simple routing",
                            ))
                        }
                    };

                    if let Some(key) = api_key {
                        let mut auth_header = String::with_capacity(7 + key.len());
                        auth_header.push_str("Bearer ");
                        auth_header.push_str(&key);
                        request_builder = request_builder.header("Authorization", auth_header);
                    }

                    for (name, value) in headers {
                        request_builder = request_builder.header(name.clone(), value.clone());
                    }

                    request_builder.send().await.map_err(convert_reqwest_error)
                }
            })
            .collect();

        // Now execute the collected futures concurrently
        let mut stream = stream::iter(futures).buffer_unordered(32);
        let mut last_response: Option<Response> = None;

        while let Some(result) = stream.next().await {
            match result {
                Ok(res) => {
                    let status = StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                    let response_headers = header_utils::preserve_response_headers(res.headers());

                    match res.bytes().await {
                        Ok(body) => {
                            let mut response = Response::new(Body::from(body));
                            *response.status_mut() = status;
                            *response.headers_mut() = response_headers;

                            if status.is_success() {
                                return response;
                            }
                            last_response = Some(response);
                        }
                        Err(e) => {
                            last_response = Some(error::internal_error(
                                "read_response_failed",
                                format!("Failed to read response: {e}"),
                            ));
                        }
                    }
                }
                Err(e) => {
                    last_response = Some(e);
                }
            }
        }

        last_response
            .unwrap_or_else(|| error::bad_gateway("no_worker_response", "No worker response"))
    }

    // Route a POST request with empty body to a specific endpoint
    async fn route_post_empty_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
    ) -> Response {
        self.route_simple_request(headers, endpoint, Method::POST)
            .await
    }

    /// Forward an audio transcription request to an audio-capable worker as
    /// `multipart/form-data`. Separate from `route_typed_request` because the
    /// endpoint is not JSON-bodied.
    async fn route_multipart_transcription(
        &self,
        headers: Option<&HeaderMap>,
        body: &TranscriptionRequest,
        audio: AudioFile,
        route: &'static str,
        model_id: &str,
    ) -> Response {
        let start = Instant::now();
        let is_stream = body.is_stream();
        let text = body.extract_text_for_routing();
        let endpoint = route_to_endpoint(route);

        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            model_id,
            endpoint,
            bool_to_static_str(is_stream),
        );

        // Finalize router metrics for an early error that never reached an
        // upstream worker (model_not_found, dp_aware_not_supported, no
        // available workers, build failure). Without this, pre-send failures
        // silently disappear from router_upstream_responses / router_error.
        let record_pre_send_error = |response: &Response| {
            let rstatus = response.status();
            Metrics::record_router_upstream_response(
                metrics_labels::ROUTER_HTTP,
                rstatus.as_u16(),
                extract_error_code_from_response(response),
            );
            if !is_retryable_status(rstatus) {
                Metrics::record_router_error(
                    metrics_labels::ROUTER_HTTP,
                    metrics_labels::BACKEND_REGULAR,
                    metrics_labels::CONNECTION_HTTP,
                    model_id,
                    endpoint,
                    error_type_from_status(rstatus),
                );
            }
        };

        // Multipart transcription can't route through `worker.prepare_request`,
        // which is the hook that injects `data_parallel_rank` for DP-aware
        // workers. Pre-filter DP-aware workers out of the candidate pool so
        // the policy can pick a non-DP worker when one exists; only fall back
        // to model_not_found / 400 when every candidate is DP-aware.
        let model_filter = if model_id == crate::worker::UNKNOWN_MODEL_ID {
            None
        } else {
            Some(model_id)
        };
        let all_workers = self.worker_registry.get_workers_filtered(
            model_filter,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Http),
            None,
            false,
        );
        if all_workers.is_empty() {
            let resp = error::model_not_found(model_id);
            record_pre_send_error(&resp);
            return resp;
        }
        let non_dp_workers: Vec<Arc<dyn Worker>> = all_workers
            .iter()
            .filter(|w| !w.is_dp_aware())
            .cloned()
            .collect();
        if non_dp_workers.is_empty() {
            let resp = error::bad_request(
                "dp_aware_not_supported",
                "/v1/audio/transcriptions does not yet support DP-aware workers",
            );
            record_pre_send_error(&resp);
            return resp;
        }
        let available: Vec<Arc<dyn Worker>> = non_dp_workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();
        if available.is_empty() {
            let resp = error::service_unavailable(
                "no_available_workers",
                "All workers are unavailable (circuit breaker open or unhealthy)",
            );
            record_pre_send_error(&resp);
            return resp;
        }

        let policy = self.policy_registry.get_policy_or_default(model_id);
        let hash_ring = self.worker_registry.get_hash_ring(model_id);
        let idx = match policy.select_worker(
            &available,
            &SelectWorkerInfo {
                request_text: Some(&text),
                tokens: None,
                headers,
                hash_ring,
            },
        ) {
            Some(i) => i,
            None => {
                let resp = error::service_unavailable(
                    "no_available_workers",
                    "Policy returned no eligible worker",
                );
                record_pre_send_error(&resp);
                return resp;
            }
        };
        Metrics::record_worker_selection(
            metrics_labels::WORKER_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            model_id,
            policy.name(),
        );
        let worker = available[idx].clone();

        let load_guard = ["cache_aware", "manual"]
            .contains(&policy.name())
            .then(|| WorkerLoadGuard::new(worker.clone(), headers));

        let mut headers_with_trace = headers.cloned().unwrap_or_default();
        inject_trace_context_http(&mut headers_with_trace);
        let headers = Some(&headers_with_trace);

        events::RequestSentEvent { url: worker.url() }.emit();

        let form = match build_transcription_form(body, audio) {
            Ok(f) => f,
            Err(e) => {
                let resp = error::bad_request("multipart_build_failed", e);
                record_pre_send_error(&resp);
                return resp;
            }
        };

        let endpoint_url = worker.endpoint_url(route);
        let mut request_builder = self.client.post(&endpoint_url).multipart(form);

        if let Some(key) = worker.api_key().cloned() {
            let mut auth_header = String::with_capacity(7 + key.len());
            auth_header.push_str("Bearer ");
            auth_header.push_str(&key);
            request_builder = request_builder.header("Authorization", auth_header);
        }

        if let Some(headers) = headers {
            for (name, value) in headers {
                // Skip Content-Type and Content-Length — reqwest sets the
                // correct multipart boundary itself.
                let name_str = name.as_str();
                if name_str.eq_ignore_ascii_case("content-type")
                    || name_str.eq_ignore_ascii_case("content-length")
                {
                    continue;
                }
                if header_utils::should_forward_request_header(name_str) {
                    request_builder = request_builder.header(name, value);
                }
            }
        }

        let res = match request_builder.send().await {
            Ok(res) => res,
            Err(e) => {
                error!(
                    "Failed to send multipart transcription request worker_url={} route={} error={}",
                    worker.url(),
                    route,
                    e
                );
                let err_resp = convert_reqwest_error(e);
                let err_status = err_resp.status();
                // Feed the synthetic status into the worker circuit breaker
                // and worker-error metric; transport failures (timeouts,
                // connect errors) must be visible to health tracking so the
                // same bad worker isn't picked repeatedly.
                worker.record_outcome(err_status.as_u16());
                if err_status.is_server_error() {
                    Metrics::record_worker_error(
                        metrics_labels::WORKER_REGULAR,
                        metrics_labels::CONNECTION_HTTP,
                        error_type_from_status(err_status),
                    );
                }
                Metrics::record_router_upstream_response(
                    metrics_labels::ROUTER_HTTP,
                    err_status.as_u16(),
                    extract_error_code_from_response(&err_resp),
                );
                // Mirror route_typed_request: a send failure must still bump
                // the terminal router_error counter, not just upstream_response.
                Metrics::record_router_error(
                    metrics_labels::ROUTER_HTTP,
                    metrics_labels::BACKEND_REGULAR,
                    metrics_labels::CONNECTION_HTTP,
                    model_id,
                    endpoint,
                    error_type_from_status(err_status),
                );
                return err_resp;
            }
        };

        let status = StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        Metrics::record_router_upstream_response(metrics_labels::ROUTER_HTTP, status.as_u16(), "");

        events::RequestReceivedEvent {}.emit();

        let response = if is_stream {
            // Preserve the upstream content-type verbatim. A `stream=true`
            // hint from the client doesn't guarantee the worker actually
            // streams — whisper backends may ignore it and return a normal
            // JSON body (success or 4xx error). Don't relabel non-SSE
            // responses as SSE; leave that judgment to whatever the worker
            // set.
            let response_headers = header_utils::preserve_response_headers(res.headers());
            let stream = res.bytes_stream();
            // Bounded channel applies backpressure: if the downstream client
            // is slow, the upstream relay awaits on `send` rather than piling
            // chunks in memory.
            const STREAM_RELAY_BUFFER: usize = 32;
            let (tx, rx) = mpsc::channel::<Result<bytes::Bytes, String>>(STREAM_RELAY_BUFFER);
            // Attribute worker-level and router-level outcomes to the actual
            // stream completion from inside the relay task: a mid-stream error
            // after a 2xx header, or a non-streaming 5xx header returned under
            // `stream=true`, must be visible to circuit-breaker + worker-error
            // + router-error metrics. Recording only at header time would mis-
            // classify those.
            let worker_for_stream = worker.clone();
            let stream_header_status = status;
            let stream_model_id = model_id.to_string();
            let stream_endpoint = endpoint;
            let stream_start = start;
            #[expect(
                clippy::disallowed_methods,
                reason = "fire-and-forget stream relay; gateway shutdown need not wait for individual stream forwarding"
            )]
            tokio::spawn(async move {
                let mut stream = stream;
                let mut stream_failed = false;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            if tx.send(Ok(bytes)).await.is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            stream_failed = true;
                            let _ = tx.send(Err(format!("Stream error: {e}"))).await;
                            break;
                        }
                    }
                }
                // Effective status = BAD_GATEWAY if the relay failed, else the
                // worker's header status. Covers both "5xx header returned
                // while stream=true" and "200 header then mid-stream break".
                let effective_status = if stream_failed {
                    StatusCode::BAD_GATEWAY
                } else {
                    stream_header_status
                };
                worker_for_stream.record_outcome(effective_status.as_u16());
                if effective_status.is_server_error() {
                    Metrics::record_worker_error(
                        metrics_labels::WORKER_REGULAR,
                        metrics_labels::CONNECTION_HTTP,
                        error_type_from_status(effective_status),
                    );
                }
                if effective_status.is_success() {
                    Metrics::record_router_duration(
                        metrics_labels::ROUTER_HTTP,
                        metrics_labels::BACKEND_REGULAR,
                        metrics_labels::CONNECTION_HTTP,
                        &stream_model_id,
                        stream_endpoint,
                        stream_start.elapsed(),
                    );
                } else {
                    Metrics::record_router_error(
                        metrics_labels::ROUTER_HTTP,
                        metrics_labels::BACKEND_REGULAR,
                        metrics_labels::CONNECTION_HTTP,
                        &stream_model_id,
                        stream_endpoint,
                        error_type_from_status(effective_status),
                    );
                }
            });
            let stream = ReceiverStream::new(rx);
            let body = Body::from_stream(stream);
            let mut response = Response::new(body);
            *response.status_mut() = status;
            *response.headers_mut() = response_headers;
            if let Some(guard) = load_guard {
                response = AttachedBody::wrap_response(response, guard);
            }
            response
        } else {
            let response_headers = header_utils::preserve_response_headers(res.headers());
            match res.bytes().await {
                Ok(body) => {
                    let mut response = Response::new(Body::from(body));
                    *response.status_mut() = status;
                    *response.headers_mut() = response_headers;
                    response
                }
                Err(e) => error::internal_error(
                    "read_response_body_failed",
                    format!("Failed to read response body: {e}"),
                ),
            }
        };

        // Non-streaming: classify metrics off the final response the client
        // will actually see. A body-read failure can rewrite a 2xx upstream
        // into a local 5xx, and we want the circuit breaker + metrics to see
        // that. Streaming outcomes are owned by the relay task above.
        if !is_stream {
            let final_status = response.status();
            worker.record_outcome(final_status.as_u16());
            if final_status.is_server_error() {
                Metrics::record_worker_error(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::CONNECTION_HTTP,
                    error_type_from_status(final_status),
                );
            }
            if final_status.is_success() {
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_HTTP,
                    metrics_labels::BACKEND_REGULAR,
                    metrics_labels::CONNECTION_HTTP,
                    model_id,
                    endpoint,
                    start.elapsed(),
                );
            } else {
                Metrics::record_router_error(
                    metrics_labels::ROUTER_HTTP,
                    metrics_labels::BACKEND_REGULAR,
                    metrics_labels::CONNECTION_HTTP,
                    model_id,
                    endpoint,
                    error_type_from_status(final_status),
                );
            }
        }

        response
    }

    // Send typed request directly without conversion
    async fn send_typed_request<T: serde::Serialize>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        worker: &dyn Worker,
        is_stream: bool,
        load_guard: Option<WorkerLoadGuard>,
    ) -> Response {
        let api_key = worker.api_key().cloned();
        let endpoint_url = worker.endpoint_url(route);

        let json_val = match serde_json::to_value(typed_req) {
            Ok(j) => j,
            Err(e) => {
                return error::bad_request(
                    "serialization_failed",
                    format!("Convert into serde_json::Value failed: {e}"),
                );
            }
        };

        let json_val = match worker.prepare_request(json_val) {
            Ok(prepared) => prepared,
            Err(e) => {
                return error::bad_request(
                    "request_preparation_failed",
                    format!("Failed to prepare request: {e}"),
                );
            }
        };

        let mut request_builder = self.client.post(&endpoint_url).json(&json_val);

        if let Some(key) = api_key {
            // Pre-allocate string with capacity to avoid reallocation
            let mut auth_header = String::with_capacity(7 + key.len());
            auth_header.push_str("Bearer ");
            auth_header.push_str(&key);
            request_builder = request_builder.header("Authorization", auth_header);
        }

        if let Some(headers) = headers {
            for (name, value) in headers {
                if header_utils::should_forward_request_header(name.as_str()) {
                    request_builder = request_builder.header(name, value);
                }
            }
        }

        let res = match request_builder.send().await {
            Ok(res) => res,
            Err(e) => {
                error!(
                    "Failed to send typed request worker_url={} route={} error={}",
                    worker.url(),
                    route,
                    e
                );

                return convert_reqwest_error(e);
            }
        };

        let status = StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        if is_stream {
            // Preserve headers for streaming response
            let mut response_headers = header_utils::preserve_response_headers(res.headers());
            // Ensure we set the correct content-type for SSE
            response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

            let stream = res.bytes_stream();
            let (tx, rx) = mpsc::unbounded_channel();

            // Spawn task to forward stream
            #[expect(
                clippy::disallowed_methods,
                reason = "fire-and-forget stream relay; gateway shutdown need not wait for individual stream forwarding"
            )]
            tokio::spawn(async move {
                let mut stream = stream;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            if tx.send(Ok(bytes)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
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
            *response.headers_mut() = response_headers;

            // Attach load guard to response body for proper RAII lifecycle
            // Guard is dropped when response body is consumed or client disconnects
            if let Some(guard) = load_guard {
                response = AttachedBody::wrap_response(response, guard);
            }
            response
        } else {
            // For non-streaming requests, preserve headers
            let response_headers = header_utils::preserve_response_headers(res.headers());

            let response = match res.bytes().await {
                Ok(body) => {
                    let mut response = Response::new(Body::from(body));
                    *response.status_mut() = status;
                    *response.headers_mut() = response_headers;
                    response
                }
                Err(e) => {
                    let error_msg = format!("Failed to get response body: {e}");
                    error::internal_error("read_response_body_failed", error_msg)
                }
            };

            // load_guard dropped here automatically after response body is read
            response
        }
    }

    async fn build_rerank_response(
        req: &RerankRequest,
        response: Response,
    ) -> anyhow::Result<Response> {
        let (_, response_body) = response.into_parts();
        let body_bytes = to_bytes(response_body, usize::MAX).await?;
        let rerank_results = serde_json::from_slice::<Vec<RerankResult>>(&body_bytes)?;
        let mut rerank_response =
            RerankResponse::new(rerank_results, req.model.clone(), req.rid.clone());
        // Sorting is handled by Python worker (serving_rerank.py)
        if let Some(top_k) = req.top_k {
            rerank_response.apply_top_k(top_k);
        }
        if !req.return_documents {
            rerank_response.drop_documents();
        }
        Ok(Json(rerank_response).into_response())
    }
}

fn build_transcription_form(body: &TranscriptionRequest, audio: AudioFile) -> Result<Form, String> {
    let AudioFile {
        bytes,
        file_name,
        content_type,
    } = audio;

    // Wrap the already-buffered Bytes in a reqwest Body (Arc refcount, no
    // additional copy) instead of Part::bytes, which would force a Vec copy.
    let file_len = bytes.len() as u64;
    let mut file_part =
        Part::stream_with_length(reqwest::Body::from(bytes), file_len).file_name(file_name);
    if let Some(ct) = content_type.as_deref() {
        file_part = file_part
            .mime_str(ct)
            .map_err(|e| format!("Invalid audio content-type '{ct}': {e}"))?;
    }

    let mut form = Form::new()
        .part("file", file_part)
        .text("model", body.model.clone());

    if let Some(ref language) = body.language {
        form = form.text("language", language.clone());
    }
    if let Some(ref prompt) = body.prompt {
        form = form.text("prompt", prompt.clone());
    }
    if let Some(ref fmt) = body.response_format {
        form = form.text("response_format", fmt.clone());
    }
    if let Some(temp) = body.temperature {
        form = form.text("temperature", temp.to_string());
    }
    if let Some(ref grans) = body.timestamp_granularities {
        for g in grans {
            form = form.text("timestamp_granularities[]", g.clone());
        }
    }
    if let Some(stream) = body.stream {
        form = form.text("stream", stream.to_string());
    }

    Ok(form)
}

fn convert_reqwest_error(e: reqwest::Error) -> Response {
    let url = e
        .url()
        .map(|u| u.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let message = format!("{e}. URL: {url}");

    // TODO improve error status code
    let (status, code) = if let Some(upstream_status) = e.status() {
        (upstream_status, "call_upstream_status_error")
    } else if e.is_builder() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_builder_error",
        )
    } else if e.is_request() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_request_error",
        )
    } else if e.is_redirect() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_redirect_error",
        )
    } else if e.is_body() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_body_error",
        )
    } else if e.is_decode() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_decode_error",
        )
    } else if e.is_timeout() {
        (StatusCode::GATEWAY_TIMEOUT, "call_upstream_timeout")
    } else if e.is_connect() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_connection_failed",
        )
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_request_failed",
        )
    };

    error::create_error(status, code, message)
}

use async_trait::async_trait;

#[async_trait]
impl RouterTrait for Router {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "health_generate").await
    }

    async fn get_server_info(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "get_server_info").await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "get_model_info").await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: &str,
    ) -> Response {
        self.route_typed_request(headers, body, "/generate", model_id)
            .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: &str,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/chat/completions", model_id)
            .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: &str,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/completions", model_id)
            .await
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: &str,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/responses", model_id)
            .await
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let endpoint = format!("v1/responses/{response_id}/cancel");
        self.route_post_empty_request(headers, &endpoint).await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: &str,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/embeddings", model_id)
            .await
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: &str,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/classify", model_id)
            .await
    }

    async fn route_audio_transcriptions(
        &self,
        headers: Option<&HeaderMap>,
        body: &TranscriptionRequest,
        audio: AudioFile,
        model_id: &str,
    ) -> Response {
        self.route_multipart_transcription(
            headers,
            body,
            audio,
            "/v1/audio/transcriptions",
            model_id,
        )
        .await
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: &str,
    ) -> Response {
        let response = self
            .route_typed_request(headers, body, "/v1/rerank", model_id)
            .await;
        if response.status().is_success() {
            match Self::build_rerank_response(body, response).await {
                Ok(rerank_response) => rerank_response,
                Err(e) => {
                    error!("Failed to build rerank response: {}", e);
                    return error::internal_error(
                        "rerank_response_build_failed",
                        "Failed to build rerank response",
                    );
                }
            }
        } else {
            response
        }
    }

    fn router_type(&self) -> &'static str {
        "regular"
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::{config::types::PolicyConfig, worker::BasicWorkerBuilder};

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    fn create_test_regular_router() -> Router {
        // Create registries
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));

        // Register test workers
        let worker1 = BasicWorkerBuilder::new("http://worker1:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();
        let worker2 = BasicWorkerBuilder::new("http://worker2:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();
        worker_registry.register_or_replace(Arc::new(worker1));
        worker_registry.register_or_replace(Arc::new(worker2));

        Router {
            worker_registry,
            policy_registry,
            client: Client::new(),
            retry_config: RetryConfig::default(),
        }
    }

    fn create_test_unhealthy_router() -> Router {
        let router = create_test_regular_router();
        let workers = router.worker_registry.get_all();
        workers[0].set_status(openai_protocol::worker::WorkerStatus::NotReady);
        router
    }

    #[test]
    fn test_router_get_worker_urls_regular() {
        let router = create_test_regular_router();
        let workers = router.worker_registry.get_all();
        let urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();

        assert_eq!(urls.len(), 2);
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));
    }

    #[test]
    fn test_select_first_worker_regular() {
        let router = create_test_regular_router();
        let result = router.select_first_worker();

        assert!(result.is_ok());
        let url = result.unwrap();
        // DashMap doesn't guarantee order, so just check we get one of the workers
        assert!(url == "http://worker1:8080" || url == "http://worker2:8080");
    }

    #[test]
    fn test_select_first_worker_with_unhealthy_worker() {
        let router = create_test_unhealthy_router();
        let result = router.select_first_worker();

        assert!(result.is_ok());
        let url = result.unwrap();

        let worker = router.worker_registry.get_by_url(&url).unwrap();
        assert!(worker.is_healthy());
    }
}
