//! WorkerSelection step.
//!
//! Planned transition: SelectWorker → LoadPreviousInteraction

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::routers::gemini::{context::RequestContext, state::StepResult};

/// Select a healthy upstream worker for the requested model.
///
/// **Not yet implemented** — currently returns 501. Planned behavior:
///
/// ## Reads
/// - `ctx.input.original_request.model` — the model identifier.
/// - `ctx.components.worker_registry` — the pool of registered workers.
/// - `ctx.input.headers` — forwarded headers (auth extraction).
///
/// ## Writes (planned)
/// - `ctx.processing.worker` — the selected `Arc<dyn Worker>`.
/// - `ctx.processing.upstream_url` — `{worker.url()}/v1/interactions`.
/// - `ctx.state` → `LoadPreviousInteraction`.
pub(crate) async fn worker_selection(_ctx: &mut RequestContext) -> Result<StepResult, Response> {
    // TODO: implement worker selection
    //
    // 1. Extract auth header from ctx.input.headers.
    // 2. Resolve model id from ctx.input.original_request (model or agent field).
    // 3. Call ctx.components.worker_registry to find the least-loaded healthy worker
    //    that supports the model (RuntimeType::External, circuit breaker OK).
    // 4. If no worker found, refresh external worker models and retry.
    // 5. If still no worker:
    //    - If any worker supports the model but all are circuit-broken,
    //      return Err(service_unavailable).
    //    - Otherwise return Err(model_not_found).
    // 6. Set ctx.processing.worker = Some(worker).
    // 7. Set ctx.processing.upstream_url = Some(format!("{}/v1/interactions", worker.url())).
    // 8. Set ctx.state = LoadPreviousInteraction.

    Err((
        StatusCode::NOT_IMPLEMENTED,
        "worker selection not yet implemented",
    )
        .into_response())
}
