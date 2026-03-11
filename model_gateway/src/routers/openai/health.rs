//! Health check and server info endpoints for the OpenAI router.

use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

use crate::{
    core::{RuntimeType, Worker, WorkerRegistry},
    routers::error,
};

fn external_workers(registry: &WorkerRegistry) -> Vec<Arc<dyn Worker>> {
    registry
        .get_all()
        .into_iter()
        .filter(|w| w.metadata().spec.runtime_type == RuntimeType::External)
        .collect()
}

pub(super) fn health_generate(registry: &WorkerRegistry) -> Response {
    let workers = external_workers(registry);
    if workers.is_empty() {
        return error::service_unavailable("service_unavailable", "No external workers registered");
    }

    let (healthy, unhealthy): (Vec<_>, Vec<_>) = workers.iter().partition(|w| w.is_healthy());

    if unhealthy.is_empty() {
        (
            StatusCode::OK,
            format!("OK - {} workers healthy", healthy.len()),
        )
            .into_response()
    } else {
        let unhealthy_info: Vec<_> = unhealthy
            .iter()
            .map(|w| format!("{} ({})", w.model_id(), w.url()))
            .collect();
        error::service_unavailable(
            "service_unavailable",
            format!(
                "{}/{} workers unhealthy: {}",
                unhealthy.len(),
                workers.len(),
                unhealthy_info.join(", ")
            ),
        )
    }
}

pub(super) fn get_server_info(registry: &WorkerRegistry) -> Response {
    let stats = registry.stats();
    let workers = external_workers(registry);
    let worker_urls: Vec<_> = workers.iter().map(|w| w.url()).collect();

    let info = json!({
        "router_type": "openai",
        "total_workers": stats.total_workers,
        "external_workers": workers.len(),
        "healthy_workers": stats.healthy_workers,
        "total_models": stats.total_models,
        "worker_urls": worker_urls
    });
    (StatusCode::OK, Json(info)).into_response()
}
