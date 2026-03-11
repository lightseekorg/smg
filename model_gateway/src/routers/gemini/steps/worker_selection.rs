//! WorkerSelection step.
//!
//! Transition: SelectWorker → LoadPreviousInteraction

use axum::response::Response;

use crate::{
    core::ProviderType,
    routers::{
        error,
        gemini::{
            context::RequestContext,
            state::{RequestState, StepResult},
        },
        worker_selection::{SelectWorkerRequest, WorkerSelector},
    },
};

/// Select a healthy upstream worker for the requested model.
///
/// ## Reads
/// - `ctx.input.model_id` / `ctx.input.original_request.model` /
///   `ctx.input.original_request.agent` — the model identifier (first non-`None` wins).
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
        .or(ctx.input.original_request.agent.as_deref());

    let model = match model {
        Some(m) if !m.is_empty() => m,
        _ => {
            return Err(error::bad_request(
                "invalid_request",
                "No model identifier provided in request".to_string(),
            ));
        }
    };

    let selector = WorkerSelector::new(&ctx.components.worker_registry, &ctx.components.client);
    let worker = selector
        .select_worker(&SelectWorkerRequest {
            model_id: model,
            headers: ctx.input.headers.as_ref(),
            provider: Some(ProviderType::Gemini),
            ..Default::default()
        })
        .await?;

    ctx.processing.upstream_url = Some(format!("{}/v1/interactions", worker.url()));
    ctx.processing.worker = Some(worker);
    ctx.state = RequestState::LoadPreviousInteraction;

    Ok(StepResult::Continue)
}
