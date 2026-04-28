//! Responses API routing orchestration.
//!
//! Mirrors the delegation pattern in `chat.rs`: the `RouterTrait` method in
//! `router.rs` packs borrowed references into [`ResponsesRouterContext`] and
//! delegates to [`route_responses`].

use std::{sync::Arc, time::Instant};

use axum::{http::HeaderMap, response::Response};
use openai_protocol::{
    model_card::ModelCard,
    responses::{ResponseInput, ResponseInputOutputItem, ResponsesRequest},
    worker::{RuntimeType, WorkerModels, WorkerSpec},
};
use serde_json::to_value;
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
    worker::{BasicWorkerBuilder, Endpoint, ProviderType, WorkerRegistry},
};

fn parse_provider_hint(headers: Option<&HeaderMap>, model: &str) -> ProviderType {
    if let Some(raw) = extract_model_provider(headers) {
        let normalized = raw.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "openai" | "gpt" | "gpt-oss" => return ProviderType::OpenAI,
            "xai" | "grok" => return ProviderType::XAI,
            "gemini" | "google" => return ProviderType::Gemini,
            "anthropic" | "claude" => return ProviderType::Anthropic,
            _ => {
                tracing::warn!(
                    model,
                    header_value = raw,
                    "Invalid x-model-provider hint; falling back to model-name inference"
                );
            }
        }
    }

    ProviderType::from_model_name(model).unwrap_or(ProviderType::OpenAI)
}

fn normalize_override_base_url(override_url: &str) -> Option<String> {
    let mut parsed = Url::parse(override_url).ok()?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return None;
    }
    parsed.set_path("");
    parsed.set_query(None);
    parsed.set_fragment(None);
    Some(parsed.to_string().trim_end_matches('/').to_string())
}

fn build_ephemeral_external_worker(
    base_url: &str,
    model: &str,
    provider: ProviderType,
    disable_health_check: bool,
) -> Arc<dyn crate::worker::Worker> {
    let mut spec = WorkerSpec::new(base_url.to_string());
    spec.runtime_type = RuntimeType::External;
    spec.provider = Some(provider);
    spec.health.disable_health_check = Some(disable_health_check);
    spec.models = WorkerModels::from(vec![ModelCard::new(model)]);
    Arc::new(BasicWorkerBuilder::from_spec(spec).build())
}

fn parse_endpoint_override(headers: Option<&HeaderMap>) -> Option<String> {
    extract_provider_endpoint(headers).and_then(|endpoint| {
        let trimmed = endpoint.trim();
        Url::parse(trimmed)
            .ok()
            .filter(|u| matches!(u.scheme(), "http" | "https"))
            .map(|_| trimmed.to_string())
    })
}

fn endpoint_override_for_provider(
    headers: Option<&HeaderMap>,
    provider_hint: &ProviderType,
    _model: &str,
) -> Option<String> {
    if provider_hint == &ProviderType::Gemini {
        return parse_endpoint_override(headers);
    }

    None
}

async fn select_worker_with_metrics(
    deps: &ResponsesRouterContext<'_>,
    headers: Option<&HeaderMap>,
    model: &str,
    provider_hint: &ProviderType,
) -> Result<Arc<dyn crate::worker::Worker>, Response> {
    match WorkerSelector::new(
        deps.worker_registry,
        &deps.responses_components.shared.client,
    )
    .select_worker(&SelectWorkerRequest {
        model_id: model,
        headers,
        provider: Some(provider_hint.clone()),
        ..Default::default()
    })
    .await
    {
        Ok(worker) => Ok(worker),
        Err(response) => {
            Metrics::record_router_error(
                metrics_labels::ROUTER_OPENAI,
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::CONNECTION_HTTP,
                model,
                metrics_labels::ENDPOINT_RESPONSES,
                metrics_labels::ERROR_NO_WORKERS,
            );
            Err(response)
        }
    }
}

#[expect(
    clippy::result_large_err,
    reason = "routing helpers use shared axum Response error contracts across router code"
)]
fn validate_endpoint_worker_provider(
    worker: &Arc<dyn crate::worker::Worker>,
    provider_hint: &ProviderType,
    model: &str,
    endpoint: &str,
    base_url: &str,
) -> Result<(), Response> {
    match worker.default_provider() {
        Some(existing_provider) if existing_provider == provider_hint => Ok(()),
        Some(existing_provider) => {
            tracing::warn!(
                model,
                endpoint_override = %endpoint,
                endpoint_base_url = %base_url,
                existing_provider = ?existing_provider,
                requested_provider = ?provider_hint,
                "responses: endpoint override resolved to provider-mismatched worker"
            );
            Err(error::bad_request(
                "invalid_request",
                "Endpoint override worker provider does not match requested model provider"
                    .to_string(),
            ))
        }
        None => {
            tracing::warn!(
                model,
                endpoint_override = %endpoint,
                endpoint_base_url = %base_url,
                requested_provider = ?provider_hint,
                "responses: endpoint override resolved to worker without provider metadata"
            );
            Err(error::bad_request(
                "invalid_request",
                "Endpoint override worker has no provider metadata; explicit provider match is required"
                    .to_string(),
            ))
        }
    }
}

#[expect(
    clippy::result_large_err,
    reason = "routing helpers use shared axum Response error contracts across router code"
)]
fn resolve_or_register_gemini_endpoint_worker(
    deps: &ResponsesRouterContext<'_>,
    endpoint: &str,
    base_url: &str,
    model: &str,
    provider_hint: &ProviderType,
) -> Result<Arc<dyn crate::worker::Worker>, Response> {
    if let Some(found) = deps.worker_registry.get_by_url(base_url) {
        validate_endpoint_worker_provider(&found, provider_hint, model, endpoint, base_url)?;
        return Ok(found);
    }

    if provider_hint == &ProviderType::Gemini {
        tracing::warn!(
            model,
            endpoint_override = %endpoint,
            endpoint_base_url = %base_url,
            provider_hint = ?provider_hint,
            "responses: endpoint override worker not found; registering ephemeral external worker"
        );

        let ephemeral =
            build_ephemeral_external_worker(base_url, model, provider_hint.clone(), true);
        let _ = deps.worker_registry.register_or_replace(ephemeral.clone());

        Ok(ephemeral)
    } else {
        tracing::warn!(
            model,
            endpoint_override = %endpoint,
            endpoint_base_url = %base_url,
            provider_hint = ?provider_hint,
            "responses: endpoint override worker not found; ephemeral registration is Gemini-only"
        );
        Err(error::bad_request(
            "invalid_request",
            "Endpoint override worker not found; dynamic worker registration is only supported for Gemini"
                .to_string(),
        ))
    }
}

async fn resolve_responses_worker(
    deps: &ResponsesRouterContext<'_>,
    headers: Option<&HeaderMap>,
    model: &str,
    provider_hint: &ProviderType,
    endpoint_override: Option<&String>,
) -> Result<Arc<dyn crate::worker::Worker>, Response> {
    if provider_hint != &ProviderType::Gemini {
        return select_worker_with_metrics(deps, headers, model, provider_hint).await;
    }

    if let Some(endpoint) = endpoint_override {
        let Some(base_url) = normalize_override_base_url(endpoint) else {
            tracing::warn!(
                model,
                endpoint_override = %endpoint,
                "responses: invalid endpoint override; falling back to normal worker selection"
            );
            return select_worker_with_metrics(deps, headers, model, provider_hint).await;
        };

        return resolve_or_register_gemini_endpoint_worker(
            deps,
            endpoint,
            &base_url,
            model,
            provider_hint,
        );
    }

    select_worker_with_metrics(deps, headers, model, provider_hint).await
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
    let provider_hint = parse_provider_hint(headers, model);

    Metrics::record_router_request(
        metrics_labels::ROUTER_OPENAI,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model,
        metrics_labels::ENDPOINT_RESPONSES,
        bool_to_static_str(streaming),
    );

    let endpoint_override = endpoint_override_for_provider(headers, &provider_hint, model);
    let worker = match resolve_responses_worker(
        deps,
        headers,
        model,
        &provider_hint,
        endpoint_override.as_ref(),
    )
    .await
    {
        Ok(worker) => worker,
        Err(response) => {
            tracing::warn!(
                model,
                provider_hint = ?provider_hint,
                endpoint_override = ?endpoint_override,
                status = %response.status(),
                "responses: worker resolution failed"
            );
            return response;
        }
    };

    // Validate mutual exclusivity of conversation and previous_response_id
    // Treat empty strings as unset to match other metadata paths
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

    let mut ctx = RequestContext::for_responses(
        Arc::new(body.clone()),
        headers.cloned(),
        Some(model_id.to_string()),
        ComponentRefs::Responses(Arc::clone(deps.responses_components)),
    );
    ctx.storage_request_context = smg_data_connector::current_request_context();
    ctx.tenant_request_meta = Some(tenant_meta.clone());

    ctx.state.worker = Some(WorkerSelection {
        worker: Arc::clone(&worker),
        provider: Arc::clone(&provider),
    });

    let upstream_url = match provider.upstream_url(worker.url(), endpoint_override.as_deref()) {
        Ok(url) => url,
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
                format!("Provider upstream URL error: {e}"),
            );
        }
    };

    ctx.state.payload = Some(PayloadState {
        json: payload,
        url: upstream_url,
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
    use axum::http::HeaderMap;
    use openai_protocol::{
        common::Detail,
        responses::{
            Annotation, FileDetail, ResponseContentPart, ResponseInput, ResponseInputOutputItem,
            ResponsesRequest,
        },
    };
    use serde_json::{json, to_value};

    use super::*;

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

    #[test]
    fn endpoint_override_used_for_gemini_only() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-provider-endpoint",
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
                .parse()
                .expect("valid header"),
        );

        let gemini_override = endpoint_override_for_provider(
            Some(&headers),
            &ProviderType::Gemini,
            "gemini-2.5-flash",
        );
        let openai_override =
            endpoint_override_for_provider(Some(&headers), &ProviderType::OpenAI, "openai.gpt-4.1");

        assert!(
            gemini_override.is_some(),
            "Gemini must honor endpoint override"
        );
        assert!(
            openai_override.is_none(),
            "Non-Gemini providers must ignore endpoint override"
        );
    }

    #[test]
    fn endpoint_override_invalid_url_is_rejected_even_for_gemini() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-provider-endpoint",
            "notaurl".parse().expect("valid header"),
        );

        let gemini_override = endpoint_override_for_provider(
            Some(&headers),
            &ProviderType::Gemini,
            "gemini-2.5-flash",
        );

        assert!(
            gemini_override.is_none(),
            "Invalid URL must not be used as endpoint override"
        );
    }

    #[test]
    fn ephemeral_google_worker_keeps_health_checks_enabled() {
        let worker = build_ephemeral_external_worker(
            "https://generativelanguage.googleapis.com",
            "google.gemini-2.5-flash",
            ProviderType::Gemini,
            true,
        );

        assert_eq!(
            worker.metadata().spec.health.disable_health_check,
            Some(true),
            "Google ephemeral worker should keep health checks enabled"
        );
    }
}
