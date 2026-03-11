//! WorkerSelection step.
//!
//! Transition: SelectWorker → LoadPreviousInteraction

use axum::response::Response;

use crate::routers::{
    gemini::{
        context::RequestContext,
        state::{RequestState, StepResult},
    },
    worker_selection::{SelectWorkerRequest, WorkerSelector},
};

/// Select a healthy upstream worker for the requested model.
///
/// ## Reads
/// - `ctx.input.original_request.model` / `ctx.input.model_id` — the model identifier.
/// - `ctx.components.worker_registry` — the pool of registered workers.
/// - `ctx.input.headers` — forwarded headers (auth extraction).
///
/// ## Writes
/// - `ctx.processing.worker` — the selected `Arc<dyn Worker>`.
/// - `ctx.processing.upstream_url` — `{worker.url()}/v1/interactions`.
/// - `ctx.state` → `LoadPreviousInteraction`.
pub(crate) async fn worker_selection(ctx: &mut RequestContext) -> Result<StepResult, Response> {
    let model = ctx
        .input
        .model_id
        .as_deref()
        .or(ctx.input.original_request.model.as_deref())
        .or(ctx.input.original_request.agent.as_deref())
        .unwrap_or_default();

    let selector = WorkerSelector::new(&ctx.components.worker_registry, &ctx.components.client);
    let worker = selector
        .select_worker(&SelectWorkerRequest {
            model_id: model,
            headers: ctx.input.headers.as_ref(),
            ..Default::default()
        })
        .await?;

    ctx.processing.upstream_url = Some(format!("{}/v1/interactions", worker.url()));
    ctx.processing.worker = Some(worker);
    ctx.state = RequestState::LoadPreviousInteraction;

    Ok(StepResult::Continue)
}
