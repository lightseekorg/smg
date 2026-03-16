//! Responses API routing orchestration.
//!
//! Mirrors the delegation pattern in `chat.rs`: the `RouterTrait` method in
//! `router.rs` packs borrowed references into [`ResponsesRouterContext`] and
//! delegates to [`route_responses`].

use std::{sync::Arc, time::Instant};

use axum::{http::HeaderMap, response::Response};
use openai_protocol::responses::{ResponseInput, ResponseInputOutputItem, ResponsesRequest};
use serde_json::to_value;

use super::{
    super::{
        context::{
            ComponentRefs, PayloadState, RequestContext, ResponsesComponents, WorkerSelection,
        },
        provider::ProviderRegistry,
        router::resolve_provider,
    },
    handle_non_streaming_response, handle_streaming_response,
};
use crate::{
    core::{Endpoint, ProviderType, WorkerRegistry},
    observability::metrics::{bool_to_static_str, metrics_labels, Metrics},
    routers::{
        error,
        worker_selection::{SelectWorkerRequest, WorkerSelector},
    },
};

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

    let worker = match WorkerSelector::new(
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
    };

    // Validate mutual exclusivity of conversation and previous_response_id
    // Treat empty strings as unset to match other metadata paths
    let conversation = body.conversation.as_ref().filter(|s| !s.is_empty());
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
    if let Some(model) = model_id {
        request_body.model = model.to_string();
    }
    request_body.conversation = None;

    let original_previous_response_id = match super::history::load_input_history(
        deps.responses_components,
        conversation.map(String::as_str),
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
        model_id.map(String::from),
        ComponentRefs::Responses(Arc::clone(deps.responses_components)),
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
