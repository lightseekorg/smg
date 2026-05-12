use std::{any::Any, sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::Response,
};
use openai_protocol::chat::ChatCompletionRequest;
use reqwest::Url;
use serde_json::to_vec;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::error;

use super::{
    context::RouterContext,
    converse_stream::forward_bedrock_converse_stream_as_sse,
    errors,
    request_map::map_chat_request,
    response_map::map_non_stream_response,
    signing::{self, AwsSigner},
};
use crate::{
    app_context::AppContext,
    config::types::RetryConfig,
    middleware::TenantRequestMeta,
    observability::metrics::{bool_to_static_str, metrics_labels, Metrics},
    routers::{
        common::{
            retry::{is_retryable_status, RetryExecutor},
            worker_selection::{SelectWorkerRequest, WorkerSelector},
        },
        RouterTrait,
    },
    worker::ProviderType,
};

pub struct BedrockRouter {
    ctx: RouterContext,
    retry_config: RetryConfig,
    request_timeout: Duration,
}

impl BedrockRouter {
    pub fn new(context: Arc<AppContext>) -> Self {
        let bedrock = context.router_config.bedrock.clone();
        let region = bedrock
            .resolved_signing_region()
            .unwrap_or_else(|| "us-east-1".to_string());
        let service = if bedrock.service.is_empty() {
            "bedrock".to_string()
        } else {
            bedrock.service.clone()
        };
        let signer = AwsSigner::new(region, service);
        let retry_config = context.router_config.effective_retry_config();
        let request_timeout = Duration::from_secs(context.router_config.request_timeout_secs);

        Self {
            ctx: RouterContext::new(
                context.worker_registry.clone(),
                context.client.clone(),
                bedrock,
                signer,
            ),
            retry_config,
            request_timeout,
        }
    }

    fn resolve_model(&self, incoming_model: &str) -> String {
        self.ctx
            .bedrock
            .model_map
            .get(incoming_model)
            .cloned()
            .unwrap_or_else(|| incoming_model.to_string())
    }
}

impl std::fmt::Debug for BedrockRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BedrockRouter").finish()
    }
}

#[async_trait]
impl RouterTrait for BedrockRouter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        body: &ChatCompletionRequest,
        model_id: &str,
    ) -> Response {
        let stream = body.stream;
        Metrics::record_router_request(
            metrics_labels::ROUTER_OPENAI,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model_id,
            metrics_labels::ENDPOINT_CHAT,
            bool_to_static_str(stream),
        );

        let bedrock_model = self.resolve_model(model_id);
        let selector = WorkerSelector::new(&self.ctx.worker_registry, &self.ctx.http_client);
        let worker = match selector
            .select_worker(&SelectWorkerRequest {
                model_id: &bedrock_model,
                headers,
                provider: Some(ProviderType::Bedrock),
                ..Default::default()
            })
            .await
        {
            Ok(w) => w,
            Err(resp) => return resp,
        };
        let payload = map_chat_request(body);
        let payload_bytes = match to_vec(&payload) {
            Ok(p) => Arc::new(p),
            Err(e) => return errors::map_bad_mapping_error(e),
        };

        let stream_path = if stream {
            "converse-stream"
        } else {
            "converse"
        };
        let encoded_model = signing::urlencoding_simple(&bedrock_model);
        let endpoint = format!("{}/model/{}/{}", worker.url(), encoded_model, stream_path);
        let parsed_url = match Url::parse(&endpoint) {
            Ok(u) => u,
            Err(e) => return errors::map_bad_mapping_error(e),
        };

        let client = self.ctx.http_client.clone();
        let signer = self.ctx.signer.clone();
        let worker_for_retry = Arc::clone(&worker);
        let openai_model_id = model_id.to_string();
        let request_timeout = self.request_timeout;

        RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            |_attempt| {
                let client = client.clone();
                let body = Arc::clone(&payload_bytes);
                let url = parsed_url.clone();
                let endpoint = endpoint.clone();
                let worker = Arc::clone(&worker_for_retry);
                let signer = signer.clone();
                let openai_model_id = openai_model_id.clone();
                async move {
                    let signed = match signer.sign("POST", &url, &body).await {
                        Ok(s) => s,
                        Err(e) => return errors::map_signing_error(e),
                    };

                    let mut req = client
                        .post(&endpoint)
                        .header("Authorization", signed.authorization)
                        .header("X-Amz-Date", signed.amz_date)
                        .header("X-Amz-Content-Sha256", signed.payload_hash)
                        .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
                        .body((*body).clone());
                    if stream {
                        req = req.header(
                            "Accept",
                            HeaderValue::from_static("application/vnd.amazon.eventstream"),
                        );
                    } else {
                        req = req.timeout(request_timeout);
                    }
                    if let Some(token) = signed.security_token {
                        req = req.header("X-Amz-Security-Token", token);
                    }

                    let resp = match req.send().await {
                        Ok(r) => r,
                        Err(e) => {
                            worker.record_outcome(503);
                            return errors::map_send_error(e);
                        }
                    };
                    let status = StatusCode::from_u16(resp.status().as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    worker.record_outcome(status.as_u16());

                    if !status.is_success() {
                        let bytes = match resp.bytes().await {
                            Ok(b) => b,
                            Err(e) => return errors::map_send_error(e),
                        };
                        return errors::map_upstream_error(status, &bytes);
                    }

                    if stream {
                        let byte_stream = resp.bytes_stream();
                        let (tx, rx) = mpsc::unbounded_channel();
                        #[expect(
                            clippy::disallowed_methods,
                            reason = "fire-and-forget Bedrock stream translation; same pattern as openai chat streaming"
                        )]
                        tokio::spawn(async move {
                            forward_bedrock_converse_stream_as_sse(
                                byte_stream,
                                openai_model_id,
                                tx,
                            )
                            .await;
                        });
                        let mut response =
                            Response::new(Body::from_stream(UnboundedReceiverStream::new(rx)));
                        *response.status_mut() = StatusCode::OK;
                        response.headers_mut().insert(
                            CONTENT_TYPE,
                            HeaderValue::from_static("text/event-stream"),
                        );
                        response
                    } else {
                        let bytes = match resp.bytes().await {
                            Ok(b) => b,
                            Err(e) => return errors::map_send_error(e),
                        };
                        match map_non_stream_response(&bytes, &openai_model_id) {
                            Ok(mapped) => match to_vec(&mapped) {
                                Ok(serialized) => {
                                    let mut response = Response::new(Body::from(serialized));
                                    *response.status_mut() = StatusCode::OK;
                                    response.headers_mut().insert(
                                        CONTENT_TYPE,
                                        HeaderValue::from_static("application/json"),
                                    );
                                    response
                                }
                                Err(e) => errors::map_bad_mapping_error(e),
                            },
                            Err(e) => errors::map_bad_mapping_error(e),
                        }
                    }
                }
            },
            |res, _| is_retryable_status(res.status()),
            |delay, attempt| {
                Metrics::record_worker_retry(
                    metrics_labels::BACKEND_EXTERNAL,
                    metrics_labels::ENDPOINT_CHAT,
                );
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                Metrics::record_worker_retries_exhausted(
                    metrics_labels::BACKEND_EXTERNAL,
                    metrics_labels::ENDPOINT_CHAT,
                );
            },
        )
        .await
    }

    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: &openai_protocol::responses::ResponsesRequest,
        _model_id: &str,
    ) -> Response {
        error!("Responses endpoint is not implemented in Bedrock router yet");
        errors::unsupported_endpoint()
    }

    fn router_type(&self) -> &'static str {
        "bedrock"
    }
}
