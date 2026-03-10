//! Anthropic Models API implementation

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::models::ListModelsResponse;
use tracing::{debug, warn};

use super::worker::get_healthy_anthropic_workers;
use crate::routers::error;

pub fn handle_list_models(
    router: &super::AnthropicRouter,
    _req: axum::extract::Request<axum::body::Body>,
) -> Response {
    debug!("Handling list models request");

    // SECURITY: Filter by Anthropic provider in multi-provider setups
    let healthy_workers = get_healthy_anthropic_workers(&router.context().worker_registry);

    if healthy_workers.is_empty() {
        warn!("No healthy Anthropic workers available for /v1/models request");
        return error::service_unavailable("no_workers", "No healthy Anthropic workers available");
    }

    let cards = healthy_workers.iter().flat_map(|w| w.models());
    let resp = ListModelsResponse::from_model_cards(cards);

    (StatusCode::OK, Json(resp)).into_response()
}
