//! Responses API routing orchestration.
//!
//! Mirrors the delegation pattern in `chat.rs`: the `RouterTrait` method in
//! `router.rs` packs borrowed references into [`ResponsesRouterContext`] and
//! delegates to [`route_responses`].

use std::{net::IpAddr, sync::Arc, time::Instant};

use axum::{http::HeaderMap, response::Response};
use openai_protocol::{
    model_card::ModelCard,
    responses::{ResponseInput, ResponseInputOutputItem, ResponsesRequest},
    worker::{HealthCheckConfig, RuntimeType, WorkerModels, WorkerSpec},
};
use serde_json::to_value;
use tracing::debug;
use url::Url;

use super::{
    super::{
        context::{
            ComponentRefs, PayloadState, RequestContext, ResponsesComponents,
            ResponsesPayloadState, WorkerSelection,
        },
        provider::ProviderRegistry,
        router::resolve_provider,
    },
    handle_non_streaming_response, handle_streaming_response,
};
use crate::{
    middleware::TenantRequestMeta,
    observability::metrics::{bool_to_static_str, metrics_labels, Metrics},
    routers::{
        common::{
            header_utils::{
                extract_conversation_memory_config, extract_model_provider,
                extract_provider_endpoint,
            },
            worker_selection::{SelectWorkerRequest, WorkerSelector},
        },
        error,
    },
    worker::{BasicWorkerBuilder, Endpoint, ProviderType, Worker, WorkerRegistry},
};

struct EndpointOverride {
    worker_base_url: String,
    target_url: String,
    provider: ProviderType,
}

struct ProviderHeaderMapping {
    header_value: &'static str,
    provider: ProviderType,
}

fn provider_header_mappings() -> [ProviderHeaderMapping; 5] {
    [
        ProviderHeaderMapping {
            header_value: ProviderType::OpenAI.as_str(),
            provider: ProviderType::OpenAI,
        },
        ProviderHeaderMapping {
            header_value: "gpt-oss",
            provider: ProviderType::OpenAI,
        },
        ProviderHeaderMapping {
            header_value: ProviderType::XAI.as_str(),
            provider: ProviderType::XAI,
        },
        ProviderHeaderMapping {
            header_value: "google",
            provider: ProviderType::Gemini,
        },
        ProviderHeaderMapping {
            header_value: ProviderType::Anthropic.as_str(),
            provider: ProviderType::Anthropic,
        },
    ]
}

fn supported_provider_header_values() -> String {
    provider_header_mappings()
        .into_iter()
        .map(|mapping| mapping.header_value)
        .collect::<Vec<_>>()
        .join(", ")
}

fn normalize_provider_endpoint(raw: &str) -> Option<(String, String)> {
    let mut parsed = Url::parse(raw.trim()).ok()?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return None;
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return None;
    }
    let host = parsed.host_str()?;
    if host.eq_ignore_ascii_case("localhost") {
        return None;
    }
    let ip_host = host
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host);
    if let Ok(ip) = ip_host.parse::<IpAddr>() {
        let blocked = match ip {
            IpAddr::V4(ip) => ip.is_loopback() || ip.is_private() || ip.is_link_local(),
            IpAddr::V6(ip) => {
                ip.is_loopback() || ip.is_unique_local() || ip.is_unicast_link_local()
            }
        };
        if blocked {
            return None;
        }
    }
    if parsed.path() == "/" {
        return None;
    }

    parsed.set_fragment(None);
    let target_url = parsed.to_string();
    let worker_base_url = parsed.origin().ascii_serialization();

    Some((worker_base_url, target_url))
}

#[derive(Debug)]
enum EndpointOverrideError {
    InvalidProvider,
    InvalidEndpoint,
}

fn endpoint_override_error_response(error: EndpointOverrideError) -> Response {
    match error {
        EndpointOverrideError::InvalidProvider => error::bad_request(
            "invalid_request",
            format!(
                "x-model-provider must be one of {}",
                supported_provider_header_values()
            ),
        ),
        EndpointOverrideError::InvalidEndpoint => error::bad_request(
            "invalid_request",
            "x-provider-endpoint must be an absolute public http(s) URL with a non-empty path and no credentials".to_string(),
        ),
    }
}

fn provider_from_header(
    headers: Option<&HeaderMap>,
) -> Result<ProviderType, EndpointOverrideError> {
    let Some(raw_provider) = extract_model_provider(headers) else {
        return Ok(ProviderType::OpenAI);
    };

    let provider = raw_provider.trim();
    if provider.is_empty() {
        return Err(EndpointOverrideError::InvalidProvider);
    }

    provider_header_mappings()
        .into_iter()
        .find(|mapping| provider.eq_ignore_ascii_case(mapping.header_value))
        .map(|mapping| mapping.provider)
        .ok_or(EndpointOverrideError::InvalidProvider)
}

fn endpoint_override_from_headers(
    headers: Option<&HeaderMap>,
) -> Result<Option<EndpointOverride>, EndpointOverrideError> {
    let Some(raw_endpoint) = extract_provider_endpoint(headers) else {
        if extract_model_provider(headers).is_some() {
            provider_from_header(headers)?;
        }
        return Ok(None);
    };

    let provider = provider_from_header(headers)?;
    let (worker_base_url, target_url) =
        normalize_provider_endpoint(raw_endpoint).ok_or(EndpointOverrideError::InvalidEndpoint)?;

    Ok(Some(EndpointOverride {
        worker_base_url,
        target_url,
        provider,
    }))
}

fn build_request_scoped_worker(
    base_url: &str,
    model: &str,
    provider: ProviderType,
    http_client: &reqwest::Client,
) -> Arc<dyn Worker> {
    let mut spec = WorkerSpec::new(base_url.to_string());
    spec.runtime_type = RuntimeType::External;
    spec.provider = Some(provider);
    spec.models = WorkerModels::from(vec![ModelCard::new(model)]);

    // Reuse the router's shared client handle for both streaming and non-streaming overrides.
    Arc::new(
        BasicWorkerBuilder::from_spec(spec)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .http_client(http_client.clone())
            .build(),
    )
}

/// Shared context passed to responses routing functions.
pub(in crate::routers::openai) struct ResponsesRouterContext<'a> {
    pub worker_registry: &'a WorkerRegistry,
    pub provider_registry: &'a ProviderRegistry,
    pub responses_components: &'a Arc<ResponsesComponents>,
}

/// Route a responses API request to the appropriate upstream worker.
pub(in crate::routers::openai) async fn route_responses(
    deps: &ResponsesRouterContext<'_>,
    headers: Option<&HeaderMap>,
    tenant_meta: &TenantRequestMeta,
    body: &ResponsesRequest,
    model_id: &str,
) -> Response {
    let start = Instant::now();
    let model = model_id;
    let streaming = body.stream.unwrap_or(false);

    Metrics::record_router_request(
        metrics_labels::ROUTER_OPENAI,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model,
        metrics_labels::ENDPOINT_RESPONSES,
        bool_to_static_str(streaming),
    );

    let endpoint_override = match endpoint_override_from_headers(headers) {
        Ok(override_url) => override_url,
        Err(err) => {
            debug!(
                model,
                error = ?err,
                "Rejecting Responses endpoint override"
            );
            Metrics::record_router_error(
                metrics_labels::ROUTER_OPENAI,
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::CONNECTION_HTTP,
                model,
                metrics_labels::ENDPOINT_RESPONSES,
                metrics_labels::ERROR_VALIDATION,
            );
            return endpoint_override_error_response(err);
        }
    };

    let worker = if let Some(endpoint_override) = endpoint_override.as_ref() {
        debug!(
            model,
            provider = ?endpoint_override.provider,
            upstream_base_url = %endpoint_override.worker_base_url,
            streaming,
            "Using request-scoped Responses endpoint override"
        );
        build_request_scoped_worker(
            &endpoint_override.worker_base_url,
            model,
            endpoint_override.provider.clone(),
            &deps.responses_components.shared.client,
        )
    } else {
        match WorkerSelector::new(
            deps.worker_registry,
            &deps.responses_components.shared.client,
        )
        .select_worker(&SelectWorkerRequest {
            model_id: model,
            headers,
            provider: Some(ProviderType::OpenAI),
            ..Default::default()
        })
        .await
        {
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
        }
    };

    // Validate mutual exclusivity of conversation and previous_response_id
    // Treat empty strings as unset to match other metadata paths. The
    // `conversation` field is a `Option<ConversationRef>` union (bare string
    // or `{ id }` object); `ConversationRef::is_empty` covers both shapes.
    let conversation = body.conversation.as_ref().filter(|c| !c.is_empty());
    let has_previous_response = body
        .previous_response_id
        .as_ref()
        .is_some_and(|s| !s.is_empty());
    if conversation.is_some() && has_previous_response {
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
    request_body.model = model_id.to_string();
    request_body.conversation = None;

    let loaded_history = match super::history::load_input_history(
        deps.responses_components,
        conversation.map(|c| c.as_id()),
        &mut request_body,
        model,
    )
    .await
    {
        Ok(id) => id,
        Err(response) => return response,
    };

    if let Some(memory_config) = extract_conversation_memory_config(headers) {
        super::history::inject_memory_context(&memory_config, &mut request_body);
    }

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

    let provider = resolve_provider(deps.provider_registry, worker.as_ref(), model);
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

    let request_headers = headers.cloned().map(|mut headers| {
        if endpoint_override.is_some() {
            headers.remove("x-provider-endpoint");
            headers.remove("x-model-provider");
        }
        headers
    });

    let mut ctx = RequestContext::for_responses(
        Arc::new(body.clone()),
        request_headers,
        Some(model_id.to_string()),
        ComponentRefs::Responses(Arc::clone(deps.responses_components)),
    );
    ctx.storage_request_context = smg_data_connector::current_request_context();
    ctx.tenant_request_meta = Some(tenant_meta.clone());

    ctx.state.worker = Some(WorkerSelection {
        worker: Arc::clone(&worker),
        provider: Arc::clone(&provider),
    });

    ctx.state.payload = Some(PayloadState {
        json: payload,
        url: endpoint_override
            .as_ref()
            .map(|endpoint_override| endpoint_override.target_url.clone())
            .unwrap_or_else(|| format!("{}/v1/responses", worker.url())),
    });
    ctx.state.responses_payload = Some(ResponsesPayloadState {
        previous_response_id: loaded_history.previous_response_id,
        existing_mcp_list_tools_labels: loaded_history.existing_mcp_list_tools_labels,
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

#[cfg(test)]
mod tests {
    //! R1 wire-contract tests: the OpenAI-compat Responses router forwards
    //! the caller's request body to the upstream provider by serialising the
    //! `ResponsesRequest` value (see [`route_responses`] around the
    //! `to_value(&request_body)` site). These tests lock the shape that the
    //! post-P1 content-part variants produce, so any future change to the
    //! serde layer surfaces here before it reaches an upstream.
    use axum::http::{HeaderMap, HeaderValue};
    use openai_protocol::{
        common::Detail,
        responses::{
            Annotation, FileDetail, ResponseContentPart, ResponseInput, ResponseInputOutputItem,
            ResponsesRequest,
        },
    };
    use serde_json::{json, to_value};

    use super::{
        endpoint_override_from_headers, normalize_provider_endpoint, EndpointOverrideError,
    };

    fn headers_with_model_provider(provider: &'static str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("x-model-provider", HeaderValue::from_static(provider));
        headers
    }

    #[test]
    fn endpoint_override_without_endpoint_accepts_valid_provider_hint() {
        let headers = headers_with_model_provider("openai");
        let result = endpoint_override_from_headers(Some(&headers));

        assert!(matches!(result, Ok(None)));
    }

    #[test]
    fn endpoint_override_without_endpoint_rejects_invalid_provider_hint() {
        let headers = headers_with_model_provider("gemiin");
        let result = endpoint_override_from_headers(Some(&headers));

        assert!(matches!(
            result,
            Err(EndpointOverrideError::InvalidProvider)
        ));
    }

    #[test]
    fn endpoint_override_without_endpoint_or_provider_is_absent() {
        let headers = HeaderMap::new();
        let result = endpoint_override_from_headers(Some(&headers));

        assert!(matches!(result, Ok(None)));
    }

    #[test]
    fn endpoint_override_normalization_accepts_public_url_and_strips_fragment() {
        let Some((worker_base_url, target_url)) =
            normalize_provider_endpoint("https://api.example.com/v1/responses?alt=sse#secret")
        else {
            panic!("public endpoint should be accepted");
        };

        assert_eq!(worker_base_url, "https://api.example.com");
        assert_eq!(target_url, "https://api.example.com/v1/responses?alt=sse");
    }

    #[test]
    fn endpoint_override_normalization_rejects_userinfo() {
        assert!(
            normalize_provider_endpoint("https://user:pass@api.example.com/v1/responses").is_none()
        );
        assert!(normalize_provider_endpoint("https://user@api.example.com/v1/responses").is_none());
    }

    #[test]
    fn endpoint_override_normalization_rejects_local_and_private_hosts() {
        for endpoint in [
            "https://localhost/v1/responses",
            "https://LOCALHOST/v1/responses",
            "http://127.0.0.1/v1/responses",
            "http://[::1]/v1/responses",
            "http://10.0.0.1/v1/responses",
            "http://172.16.0.1/v1/responses",
            "http://192.168.1.1/v1/responses",
            "http://169.254.169.254/latest/meta-data",
            "http://[fc00::1]/v1/responses",
            "http://[fe80::1]/v1/responses",
        ] {
            assert!(
                normalize_provider_endpoint(endpoint).is_none(),
                "{endpoint} should be rejected"
            );
        }
    }

    fn build_request_with_mixed_content() -> ResponsesRequest {
        ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Items(vec![ResponseInputOutputItem::Message {
                id: "msg_r1".to_string(),
                role: "user".to_string(),
                content: vec![
                    ResponseContentPart::InputText {
                        text: "what is in this image and this file?".to_string(),
                    },
                    ResponseContentPart::InputImage {
                        detail: Some(Detail::Auto),
                        file_id: Some("file-img".to_string()),
                        image_url: Some("https://example.com/dog.jpg".to_string()),
                    },
                    ResponseContentPart::InputFile {
                        detail: Some(FileDetail::High),
                        file_data: Some("JVBERi0xLjQK".to_string()),
                        file_id: Some("file-pdf".to_string()),
                        file_url: Some("https://example.com/report.pdf".to_string()),
                        filename: Some("report.pdf".to_string()),
                    },
                    ResponseContentPart::Refusal {
                        refusal: "I cannot process that request.".to_string(),
                    },
                ],
                status: Some("completed".to_string()),
                phase: None,
            }]),
            ..Default::default()
        }
    }

    /// Exercises the exact `to_value(&request_body)` step `route_responses`
    /// uses to build the upstream payload — see `route.rs` handler body.
    fn serialize_like_router(req: &ResponsesRequest) -> serde_json::Value {
        to_value(req).expect("router serializes ResponsesRequest without error")
    }

    #[test]
    fn router_serialization_preserves_input_image_fields() {
        let req = build_request_with_mixed_content();
        let payload = serialize_like_router(&req);
        let content = &payload["input"][0]["content"];

        assert_eq!(content[1]["type"], json!("input_image"));
        assert_eq!(content[1]["detail"], json!("auto"));
        assert_eq!(content[1]["file_id"], json!("file-img"));
        assert_eq!(
            content[1]["image_url"],
            json!("https://example.com/dog.jpg")
        );
    }

    #[test]
    fn router_serialization_preserves_input_file_fields() {
        let req = build_request_with_mixed_content();
        let payload = serialize_like_router(&req);
        let content = &payload["input"][0]["content"];

        assert_eq!(content[2]["type"], json!("input_file"));
        assert_eq!(content[2]["detail"], json!("high"));
        assert_eq!(content[2]["file_data"], json!("JVBERi0xLjQK"));
        assert_eq!(content[2]["file_id"], json!("file-pdf"));
        assert_eq!(
            content[2]["file_url"],
            json!("https://example.com/report.pdf")
        );
        assert_eq!(content[2]["filename"], json!("report.pdf"));
    }

    #[test]
    fn router_serialization_preserves_refusal() {
        let req = build_request_with_mixed_content();
        let payload = serialize_like_router(&req);
        let content = &payload["input"][0]["content"];

        assert_eq!(content[3]["type"], json!("refusal"));
        assert_eq!(
            content[3]["refusal"],
            json!("I cannot process that request.")
        );
    }

    #[test]
    fn router_serialization_omits_empty_input_image_fields() {
        // `file_id` / `image_url` / `detail` are all optional; the wire
        // payload must not carry `null`s when the caller leaves them unset
        // (the #[serde(skip_serializing_if = "Option::is_none")] attributes
        // on ResponseContentPart guarantee this).
        let req = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Items(vec![ResponseInputOutputItem::Message {
                id: "msg_sparse".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputImage {
                    detail: None,
                    file_id: Some("file-only".to_string()),
                    image_url: None,
                }],
                status: None,
                phase: None,
            }]),
            ..Default::default()
        };
        let payload = serialize_like_router(&req);
        let image = &payload["input"][0]["content"][0];

        assert_eq!(image["type"], json!("input_image"));
        assert_eq!(image["file_id"], json!("file-only"));
        assert!(
            image.get("detail").is_none(),
            "detail should be omitted when None"
        );
        assert!(
            image.get("image_url").is_none(),
            "image_url should be omitted when None"
        );
    }

    #[test]
    fn router_serialization_round_trips_typed_annotations_on_output_text() {
        // Assistant turns replayed from storage carry `OutputText` with typed
        // annotations. R1 must preserve the annotation union end-to-end.
        let req = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Items(vec![ResponseInputOutputItem::Message {
                id: "msg_prior".to_string(),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: "Here are three citations.".to_string(),
                    annotations: vec![
                        Annotation::FileCitation {
                            file_id: "file-1".to_string(),
                            filename: "spec.pdf".to_string(),
                            index: 0,
                        },
                        Annotation::UrlCitation {
                            url: "https://example.com".to_string(),
                            title: "Example".to_string(),
                            start_index: 10,
                            end_index: 24,
                        },
                        Annotation::FilePath {
                            file_id: "file-2".to_string(),
                            index: 2,
                        },
                    ],
                    logprobs: None,
                }],
                status: Some("completed".to_string()),
                phase: None,
            }]),
            ..Default::default()
        };

        let payload = serialize_like_router(&req);
        let annotations = &payload["input"][0]["content"][0]["annotations"];
        assert_eq!(annotations[0]["type"], json!("file_citation"));
        assert_eq!(annotations[0]["filename"], json!("spec.pdf"));
        assert_eq!(annotations[1]["type"], json!("url_citation"));
        assert_eq!(annotations[1]["url"], json!("https://example.com"));
        assert_eq!(annotations[1]["start_index"], json!(10));
        assert_eq!(annotations[2]["type"], json!("file_path"));
        assert_eq!(annotations[2]["index"], json!(2));
    }
}
