//! WorkerSelection step.
//!
//! Transition: SelectWorker → LoadPreviousInteraction

use axum::response::Response;

use crate::routers::gemini::{
    context::RequestContext,
    state::{RequestState, StepResult},
};

/// Select a healthy upstream worker for the requested model.
///
/// ## Reads
/// - `ctx.original_request.model` — the model identifier.
/// - `ctx.shared.worker_registry` — the pool of registered workers.
/// - `ctx.headers` — forwarded headers (auth extraction).
///
/// ## Writes
/// - `ctx.worker` — the selected `Arc<dyn Worker>`.
/// - `ctx.upstream_url` — `{worker.url()}/v1/interactions`.
/// - `ctx.state` → `LoadPreviousInteraction`.
pub(crate) async fn worker_selection(ctx: &mut RequestContext) -> Result<StepResult, Response> {
    // TODO: implement worker selection
    //
    // 1. Extract auth header from ctx.headers.
    // 2. Resolve model id from ctx.original_request (model or agent field).
    // 3. Call worker_registry to find the least-loaded healthy worker
    //    that supports the model (RuntimeType::External, circuit breaker OK).
    // 4. If no worker found, refresh external worker models and retry.
    // 5. If still no worker:
    //    - If any worker supports the model but all are circuit-broken,
    //      return Err(service_unavailable).
    //    - Otherwise return Err(model_not_found).
    // 6. Set ctx.worker = Some(worker).
    // 7. Set ctx.upstream_url = Some(format!("{}/v1/interactions", worker.url())).
    // 8. Set ctx.state = LoadPreviousInteraction.

    ctx.state = RequestState::LoadPreviousInteraction;
    Ok(StepResult::Continue)
}
