//! Background worker seam.
//!
//! The driver claims jobs from the [`BackgroundResponseRepository`] and hands
//! each one to a [`BackgroundWorker`] for execution. This module defines that
//! seam, the [`HeadlessResponses`] execution abstraction the real worker depends
//! on, the real worker ([`RealBackgroundWorker`]), and a fallback
//! ([`UnavailableBackgroundWorker`]) used when no gRPC router is available.

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::response::Response;
use chrono::{DateTime, Utc};
use openai_protocol::responses::{ResponseStatus, ResponsesRequest, ResponsesResponse};
use serde_json::{from_value, json, to_value, Value};
use smg_data_connector::{
    with_request_context, BackgroundRepositoryError, BackgroundResponseRepository, FinalizeRequest,
    FinalizeStatus, LeasedJob, RequestContext,
};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::{
    config::BackgroundConfig,
    middleware::TenantRequestMeta,
    tenant::{RouteRequestMeta, TenantKey},
};

/// Error code stamped on the response that the [`UnavailableBackgroundWorker`]
/// finalizes. Distinct, greppable, and stable so operators and tests can key
/// off it.
pub const BACKGROUND_EXECUTION_UNAVAILABLE: &str = "background_execution_unavailable";

/// Stable tenant key used for background jobs whose stored `request_context`
/// carries no tenant. Keeps lease/charge identity well-defined without
/// inventing a per-job tenant.
pub const BACKGROUND_TENANT_SENTINEL: &str = "background";

/// Request-context key the gateway uses to carry the resolved tenant identity
/// into storage hooks; reused here to synthesize a `TenantRequestMeta` for a
/// reconstructed background job.
const REQUEST_CONTEXT_TENANT_KEY: &str = "tenant_id";

/// Headless responses execution abstraction.
///
/// The background worker depends on `Arc<dyn HeadlessResponses>` rather than on
/// `GrpcRouter` directly, so it can be unit-tested with a fake that returns
/// canned outcomes. The single implementation in production is `GrpcRouter`
/// (the SMG-local pipeline only exists for the gRPC connection modes).
///
/// Returns a typed `Result<ResponsesResponse, Response>`:
/// - `Ok(resp)` — the pipeline produced a response (status may be `completed`,
///   `incomplete`, `failed`, or `cancelled`); the worker maps the status to a
///   terminal `finalize`.
/// - `Err(Response)` — a pipeline/backend error (treated as transient by the
///   worker, eligible for retry with backoff).
///
/// Implementations MUST NOT persist the response row (the worker's `finalize`
/// is the authoritative terminal write) and MUST NOT assume the produced id is
/// durable (the worker overwrites it with the job's durable id).
#[async_trait]
pub trait HeadlessResponses: Send + Sync {
    async fn execute_responses_headless(
        &self,
        request: ResponsesRequest,
        tenant_meta: TenantRequestMeta,
        request_context: Option<RequestContext>,
        cancel: CancellationToken,
    ) -> Result<ResponsesResponse, Response>;
}

/// Executes a single leased background job end to end.
///
/// The driver owns the concurrency permit and the claim/sweep loops; an
/// implementation of this trait owns everything from "I have a leased job" to
/// "the response row is terminal" (running the model, persisting stream events,
/// honoring cancel/retry, and calling
/// [`BackgroundResponseRepository::finalize`]). The real implementation lands in
/// BGM-PR-07; [`UnavailableBackgroundWorker`] is the placeholder until then.
#[async_trait]
pub trait BackgroundWorker: Send + Sync {
    /// Execute one claimed job. The implementation is responsible for driving
    /// the job to a terminal state (typically via `finalize`); the driver
    /// holds the concurrency permit until this future resolves.
    async fn execute(&self, job: LeasedJob);
}

/// The real background worker.
///
/// Reconstructs a claimed job into a [`ResponsesRequest`], runs the SMG-local
/// responses pipeline headlessly via [`HeadlessResponses`], and writes the
/// terminal outcome through [`BackgroundResponseRepository::finalize`]. While a
/// job runs it keeps the lease alive (heartbeat) and watches for a cooperative
/// cancel (cancel poller). Transient pipeline/backend errors are requeued with
/// exponential backoff until `config.max_retries` is exhausted.
pub struct RealBackgroundWorker {
    repository: Arc<dyn BackgroundResponseRepository>,
    headless: Arc<dyn HeadlessResponses>,
    config: BackgroundConfig,
}

impl RealBackgroundWorker {
    pub fn new(
        repository: Arc<dyn BackgroundResponseRepository>,
        headless: Arc<dyn HeadlessResponses>,
        config: BackgroundConfig,
    ) -> Self {
        Self {
            repository,
            headless,
            config,
        }
    }

    /// Build a `failed` `raw_response` payload (OpenAI-shaped) for terminal
    /// failures where the pipeline produced no response object.
    fn failed_raw_response(job: &LeasedJob, code: &str, message: &str) -> Value {
        json!({
            "id": job.response_id.0,
            "object": "response",
            "status": "failed",
            "model": job.model,
            "output": [],
            "error": { "code": code, "message": message },
        })
    }

    /// Reconstruct the executable [`ResponsesRequest`] from a claimed job.
    ///
    /// - deserializes `request_json` (the accepted client request),
    /// - replaces `input` with the enqueue-time snapshot (`job.input`),
    /// - clears `previous_response_id` and `conversation`: the snapshot already
    ///   folded prior history at enqueue, so leaving these set would make the
    ///   pipeline double-resolve history (`load_conversation_history` /
    ///   `load_previous_messages`),
    /// - forces non-streaming (streaming background is Slice B).
    fn reconstruct_request(job: &LeasedJob) -> Result<ResponsesRequest, serde_json::Error> {
        let mut req: ResponsesRequest = from_value(job.request_json.clone())?;
        req.input = from_value(job.input.clone())?;
        req.previous_response_id = None;
        req.conversation = None;
        req.stream = Some(false);
        Ok(req)
    }

    /// Synthesize per-request tenant metadata for a reconstructed job. Uses the
    /// tenant carried in the stored `request_context` when present, else a
    /// stable background sentinel so lease/charge identity is well-defined.
    fn tenant_meta(request_context: Option<&RequestContext>) -> TenantRequestMeta {
        let tenant = request_context
            .and_then(|ctx| ctx.get(REQUEST_CONTEXT_TENANT_KEY))
            .filter(|t| !t.is_empty())
            .map(TenantKey::new)
            .unwrap_or_else(|| TenantKey::new(BACKGROUND_TENANT_SENTINEL));
        RouteRequestMeta::new(tenant)
    }

    /// Compute the next retry deadline: exponential backoff
    /// (`retry_base_delay * 2^attempt`) capped at `retry_max_delay`, plus jitter
    /// up to the (capped) delay to spread retries across replicas.
    fn next_attempt_at(&self, now: DateTime<Utc>, attempt: u32) -> DateTime<Utc> {
        let base = self.config.retry_base_delay_secs.max(1);
        let cap = self
            .config
            .retry_max_delay_secs
            .max(self.config.retry_base_delay_secs);
        // Saturating exponential: base * 2^attempt, clamped to the cap.
        let scaled = base.saturating_mul(1u64.checked_shl(attempt).unwrap_or(u64::MAX));
        let capped = scaled.min(cap);
        let jitter = if capped > 0 {
            rand::random::<f64>() * capped as f64
        } else {
            0.0
        };
        let delay_secs = capped as f64 + jitter;
        now + chrono::Duration::milliseconds((delay_secs * 1000.0) as i64)
    }

    /// Decide the terminal [`FinalizeStatus`] for a produced response.
    ///
    /// - `Completed` → completed,
    /// - `Incomplete` with a truncation reason (`max_output_tokens` /
    ///   `content_filter`) → incomplete,
    /// - everything else (`failed`, `max_tool_calls`, an `incomplete` without a
    ///   truncation reason, or an unexpected non-terminal status) → failed.
    fn classify_response(resp: &ResponsesResponse) -> FinalizeStatus {
        match resp.status {
            ResponseStatus::Completed => FinalizeStatus::Completed,
            ResponseStatus::Incomplete if resp.incomplete_details.is_some() => {
                FinalizeStatus::Incomplete
            }
            _ => FinalizeStatus::Failed,
        }
    }

    /// Spawn the lease-heartbeat task. Re-extends the lease every
    /// `lease_duration/3` until `stop` fires. Stops early (benignly) if the
    /// lease is lost (`LeaseNotHeld`) — the sweeper has requeued the row or
    /// another worker took it, and this worker's eventual finalize will no-op.
    fn spawn_heartbeat(
        &self,
        job: &LeasedJob,
        stop: CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        let repository = Arc::clone(&self.repository);
        let response_id = job.response_id.clone();
        let worker_id = job.worker_id.clone();
        let lease = self.config.lease_duration();
        // Beat at a third of the lease so two beats fit comfortably inside one
        // lease window even under scheduling jitter.
        let interval = std::cmp::max(lease / 3, Duration::from_secs(1));
        #[expect(
            clippy::disallowed_methods,
            reason = "per-job heartbeat task bounded by `stop`; stops when execution completes"
        )]
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    () = stop.cancelled() => return,
                    () = tokio::time::sleep(interval) => {}
                }
                let now = Utc::now();
                match repository
                    .heartbeat(&response_id, &worker_id, now, lease)
                    .await
                {
                    Ok(()) => {}
                    Err(BackgroundRepositoryError::LeaseNotHeld { .. }) => {
                        debug!(
                            response_id = %response_id.0,
                            "heartbeat: lease lost; stopping heartbeat"
                        );
                        return;
                    }
                    Err(e) => {
                        warn!(
                            response_id = %response_id.0,
                            error = %e,
                            "heartbeat failed; will retry next interval"
                        );
                    }
                }
            }
        })
    }

    /// Spawn the cancel poller. Periodically reads `is_cancel_requested`; when
    /// it observes a cancel it fires `cancel` (the execution token) so the tool
    /// loop converges at its next checkpoint, then exits. Stops when `stop`
    /// fires (execution finished first).
    fn spawn_cancel_poller(
        &self,
        job: &LeasedJob,
        cancel: CancellationToken,
        stop: CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        let repository = Arc::clone(&self.repository);
        let response_id = job.response_id.clone();
        // Reuse the claim poll cadence for cancel observation.
        let interval = std::cmp::max(self.config.poll_interval(), Duration::from_millis(100));
        #[expect(
            clippy::disallowed_methods,
            reason = "per-job cancel poller bounded by `stop`; stops when execution completes or cancel observed"
        )]
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    () = stop.cancelled() => return,
                    () = tokio::time::sleep(interval) => {}
                }
                match repository.is_cancel_requested(&response_id).await {
                    Ok(true) => {
                        debug!(
                            response_id = %response_id.0,
                            "cancel observed; signalling cooperative cancel"
                        );
                        cancel.cancel();
                        return;
                    }
                    Ok(false) => {}
                    Err(e) => {
                        warn!(
                            response_id = %response_id.0,
                            error = %e,
                            "cancel poll failed; will retry next interval"
                        );
                    }
                }
            }
        })
    }

    /// Finalize the job with a terminal status, logging cancel races and
    /// treating a lost lease as benign.
    async fn finalize(&self, job: &LeasedJob, status: FinalizeStatus, raw_response: Value) {
        let now = Utc::now();
        let req = FinalizeRequest::new(
            job.response_id.clone(),
            job.worker_id.clone(),
            status,
            raw_response,
            now,
        );
        match self.repository.finalize(req, now).await {
            Ok(result) => {
                if result.cancel_won {
                    info!(
                        response_id = %job.response_id.0,
                        requested = ?status,
                        "background finalize: cancel won; repository wrote cancelled"
                    );
                } else {
                    debug!(
                        response_id = %job.response_id.0,
                        final_status = ?result.final_status,
                        "background job finalized"
                    );
                }
            }
            Err(BackgroundRepositoryError::LeaseNotHeld { .. }) => {
                // Sweeper requeued or the lease was stolen mid-run: benign, the
                // row will be re-driven (or is already terminal). Don't override.
                info!(
                    response_id = %job.response_id.0,
                    "background finalize skipped: lease no longer held (sweeper/steal)"
                );
            }
            Err(e) => {
                error!(
                    response_id = %job.response_id.0,
                    error = %e,
                    "background finalize failed"
                );
            }
        }
    }

    /// Handle a transient (pipeline/backend) error: requeue with backoff while
    /// retries remain, else finalize `failed`.
    async fn handle_transient_error(&self, job: &LeasedJob) {
        if job.retry_attempt < self.config.max_retries {
            let now = Utc::now();
            let next_attempt_at = self.next_attempt_at(now, job.retry_attempt);
            match self
                .repository
                .requeue_for_retry(&job.response_id, &job.worker_id, now, next_attempt_at)
                .await
            {
                Ok(()) => {
                    info!(
                        response_id = %job.response_id.0,
                        retry_attempt = job.retry_attempt + 1,
                        max_retries = self.config.max_retries,
                        next_attempt_at = %next_attempt_at,
                        "background job hit transient error; requeued for retry"
                    );
                }
                Err(BackgroundRepositoryError::LeaseNotHeld { .. }) => {
                    info!(
                        response_id = %job.response_id.0,
                        "retry requeue skipped: lease no longer held (sweeper/steal/cancel)"
                    );
                }
                Err(e) => {
                    error!(
                        response_id = %job.response_id.0,
                        error = %e,
                        "background retry requeue failed"
                    );
                }
            }
        } else {
            warn!(
                response_id = %job.response_id.0,
                retry_attempt = job.retry_attempt,
                max_retries = self.config.max_retries,
                "background job exhausted retries; finalizing failed"
            );
            let raw = Self::failed_raw_response(
                job,
                "background_execution_failed",
                "Background execution failed after exhausting retries.",
            );
            self.finalize(job, FinalizeStatus::Failed, raw).await;
        }
    }
}

#[async_trait]
impl BackgroundWorker for RealBackgroundWorker {
    async fn execute(&self, job: LeasedJob) {
        // 1. Streaming background is Slice B — not supported here.
        if job.stream_enabled {
            warn!(
                response_id = %job.response_id.0,
                "streaming background not yet supported; finalizing failed"
            );
            let raw = Self::failed_raw_response(
                &job,
                "background_streaming_unsupported",
                "Streaming background responses are not yet supported.",
            );
            self.finalize(&job, FinalizeStatus::Failed, raw).await;
            return;
        }

        // 2. Reconstruct the request from the stored snapshot.
        let mut req = match Self::reconstruct_request(&job) {
            Ok(req) => req,
            Err(e) => {
                error!(
                    response_id = %job.response_id.0,
                    error = %e,
                    "failed to reconstruct background request; finalizing failed"
                );
                let raw = Self::failed_raw_response(
                    &job,
                    "background_request_invalid",
                    "Stored background request could not be reconstructed.",
                );
                self.finalize(&job, FinalizeStatus::Failed, raw).await;
                return;
            }
        };
        // Defensive: keep the model column and the request model in agreement.
        if req.model.is_empty() {
            req.model = job.model.clone();
        }

        // 3. Load the stored request context (storage-hook replay).
        let request_context = match self.repository.load_request_context(&job.response_id).await {
            Ok(ctx) => ctx,
            Err(e) => {
                // Treat as transient: a backend hiccup loading context shouldn't
                // burn the job. Requeue (or fail when exhausted).
                warn!(
                    response_id = %job.response_id.0,
                    error = %e,
                    "failed to load background request context; treating as transient"
                );
                self.handle_transient_error(&job).await;
                return;
            }
        };
        let tenant_meta = Self::tenant_meta(request_context.as_ref());

        // 4 + 5. Spawn the heartbeat and cancel poller, scoped to this run.
        let cancel = CancellationToken::new();
        let stop = CancellationToken::new();
        let heartbeat = self.spawn_heartbeat(&job, stop.clone());
        let cancel_poller = self.spawn_cancel_poller(&job, cancel.clone(), stop.clone());

        // 6. Run inside the storage context so hooked storage sees the replayed
        // request context, exactly like the live request path.
        let ctx_for_scope = request_context.clone().unwrap_or_default();
        let outcome = with_request_context(
            ctx_for_scope,
            self.headless.execute_responses_headless(
                req,
                tenant_meta,
                request_context,
                cancel.clone(),
            ),
        )
        .await;

        // Execution finished: stop the auxiliary tasks and join them so they
        // don't outlive the run (and can't beat after we finalize).
        stop.cancel();
        let _ = heartbeat.await;
        let _ = cancel_poller.await;

        // 7. Map outcome → finalize / retry.
        let cancelled = cancel.is_cancelled();
        match outcome {
            _ if cancelled => {
                // A cancel was observed (token fired) — finalize cancelled. The
                // repository resolves the cancel/complete race authoritatively.
                let raw = match &outcome {
                    Ok(resp) => Self::cancelled_raw_response(&job, resp),
                    Err(_) => Self::cancelled_raw_response_minimal(&job),
                };
                self.finalize(&job, FinalizeStatus::Cancelled, raw).await;
            }
            Ok(mut resp) => {
                // Force the durable id so the stored payload matches the row.
                resp.id = job.response_id.0.clone();
                let status = Self::classify_response(&resp);
                match to_value(&resp) {
                    Ok(raw) => self.finalize(&job, status, raw).await,
                    Err(e) => {
                        error!(
                            response_id = %job.response_id.0,
                            error = %e,
                            "failed to serialize background response; finalizing failed"
                        );
                        let raw = Self::failed_raw_response(
                            &job,
                            "background_response_serialize_failed",
                            "Failed to serialize the produced response.",
                        );
                        self.finalize(&job, FinalizeStatus::Failed, raw).await;
                    }
                }
            }
            Err(response) => {
                // Pipeline / backend error → transient: retry with backoff or
                // fail when exhausted.
                debug!(
                    response_id = %job.response_id.0,
                    status = %response.status(),
                    "background headless execution returned an error response"
                );
                self.handle_transient_error(&job).await;
            }
        }
    }
}

impl RealBackgroundWorker {
    /// Build a `cancelled` `raw_response` from the produced response, forcing
    /// the durable id and cancelled status.
    fn cancelled_raw_response(job: &LeasedJob, resp: &ResponsesResponse) -> Value {
        let mut value =
            to_value(resp).unwrap_or_else(|_| Self::cancelled_raw_response_minimal(job));
        if let Some(obj) = value.as_object_mut() {
            obj.insert("id".to_string(), Value::String(job.response_id.0.clone()));
            obj.insert("status".to_string(), Value::String("cancelled".to_string()));
        }
        value
    }

    /// Minimal `cancelled` payload when no response object is available.
    fn cancelled_raw_response_minimal(job: &LeasedJob) -> Value {
        json!({
            "id": job.response_id.0,
            "object": "response",
            "status": "cancelled",
            "model": job.model,
            "output": [],
        })
    }
}

/// Default [`BackgroundWorker`] that terminalizes every claimed job as `failed`.
///
/// Real execution is not wired yet (BGM-PR-07). Without a worker, a claimed job
/// would sit `in_progress` until its lease expired, get requeued by the sweeper,
/// be re-claimed, and loop forever. This placeholder instead finalizes each
/// claimed job as [`FinalizeStatus::Failed`] with a clear reason, so jobs reach
/// a terminal state cleanly and `GET /v1/responses/{id}` returns a `failed`
/// payload rather than a perpetual `in_progress`.
pub struct UnavailableBackgroundWorker {
    repository: Arc<dyn BackgroundResponseRepository>,
}

impl UnavailableBackgroundWorker {
    pub fn new(repository: Arc<dyn BackgroundResponseRepository>) -> Self {
        Self { repository }
    }

    /// Build the `failed` `raw_response` payload stored for the job. Mirrors the
    /// shape of the queued skeleton produced by the create path (`id`,
    /// `object`, `status`, `model`, `output`) and adds an OpenAI-style `error`
    /// object so the failure reason is visible to clients.
    fn failed_raw_response(job: &LeasedJob) -> Value {
        json!({
            "id": job.response_id.0,
            "object": "response",
            "status": "failed",
            "model": job.model,
            "output": [],
            "error": {
                "code": BACKGROUND_EXECUTION_UNAVAILABLE,
                "message": "background worker not yet implemented",
            },
        })
    }
}

#[async_trait]
impl BackgroundWorker for UnavailableBackgroundWorker {
    async fn execute(&self, job: LeasedJob) {
        let now = Utc::now();
        let finalize = FinalizeRequest::new(
            job.response_id.clone(),
            job.worker_id.clone(),
            FinalizeStatus::Failed,
            Self::failed_raw_response(&job),
            now,
        );

        match self.repository.finalize(finalize, now).await {
            Ok(result) => {
                warn!(
                    response_id = %job.response_id.0,
                    final_status = ?result.final_status,
                    cancel_won = result.cancel_won,
                    "background execution unavailable; finalized job as failed (BGM-PR-07 not yet wired)"
                );
            }
            Err(e) => {
                // A lease that expired before finalize, or a row already made
                // terminal by a concurrent cancel, is expected and harmless:
                // the sweeper requeues genuinely-stuck rows and a terminal row
                // needs no further action. Log and move on.
                warn!(
                    response_id = %job.response_id.0,
                    error = %e,
                    "failed to finalize background job in unavailable worker"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serde_json::json;
    use smg_data_connector::{
        EnqueueRequest, MemoryBackgroundRepository, MemoryResponseStorage, ResponseId,
        ResponseStorage,
    };

    use super::*;

    fn enqueue_req(id: &str) -> EnqueueRequest {
        EnqueueRequest::new(
            ResponseId::from(id),
            "gpt-5.1".to_string(),
            // A valid `ResponsesRequest` shape so the worker can reconstruct it.
            json!({"model": "gpt-5.1", "input": "hi"}),
            json!([]),
            json!({"id": id, "status": "queued"}),
            false,
            0,
        )
    }

    #[tokio::test]
    async fn unavailable_worker_finalizes_claimed_job_as_failed() {
        let rs = Arc::new(MemoryResponseStorage::new());
        let repo: Arc<dyn BackgroundResponseRepository> =
            Arc::new(MemoryBackgroundRepository::new(Arc::clone(&rs)));
        repo.enqueue(enqueue_req("r1"), None, None).await.unwrap();

        let job = repo
            .claim_next("w1", Utc::now(), Duration::from_secs(60))
            .await
            .unwrap()
            .expect("claim");

        let worker = UnavailableBackgroundWorker::new(Arc::clone(&repo));
        worker.execute(job).await;

        // The mirrored response storage must now show a terminal failed payload
        // carrying the unavailable reason.
        let stored = rs
            .get_response(&ResponseId::from("r1"))
            .await
            .unwrap()
            .expect("response present");
        assert_eq!(stored.raw_response["status"], "failed");
        assert_eq!(
            stored.raw_response["error"]["code"],
            BACKGROUND_EXECUTION_UNAVAILABLE
        );

        // The row is terminal, so it is no longer claimable.
        let reclaim = repo
            .claim_next("w2", Utc::now(), Duration::from_secs(60))
            .await
            .unwrap();
        assert!(reclaim.is_none(), "failed job must not be re-claimable");
    }

    // ── RealBackgroundWorker ─────────────────────────────────────────────

    use std::sync::Mutex;

    use axum::{http::StatusCode, response::IntoResponse};

    /// Canned outcome a [`FakeHeadless`] returns.
    enum FakeOutcome {
        /// Return `Ok` with the given status (and optional incomplete reason).
        Ok(ResponseStatus, Option<IncompleteReason>),
        /// Return `Err` with an HTTP error status (a transient pipeline error).
        Err(StatusCode),
        /// Fire the cancel token (simulating the cancel poller) then return the
        /// supplied status — exercises the worker's cancel precedence.
        CancelThen(ResponseStatus),
    }

    use openai_protocol::responses::{IncompleteDetails, IncompleteReason};

    /// A fake [`HeadlessResponses`] returning canned outcomes and recording the
    /// request it received (for reconstruction assertions).
    struct FakeHeadless {
        outcome: FakeOutcome,
        captured: Mutex<Option<ResponsesRequest>>,
    }

    impl FakeHeadless {
        fn new(outcome: FakeOutcome) -> Self {
            Self {
                outcome,
                captured: Mutex::new(None),
            }
        }
    }

    #[async_trait]
    impl HeadlessResponses for FakeHeadless {
        async fn execute_responses_headless(
            &self,
            request: ResponsesRequest,
            _tenant_meta: TenantRequestMeta,
            _request_context: Option<RequestContext>,
            cancel: CancellationToken,
        ) -> Result<ResponsesResponse, Response> {
            *self.captured.lock().unwrap() = Some(request.clone());
            match &self.outcome {
                FakeOutcome::Ok(status, reason) => {
                    let mut resp = ResponsesResponse::builder("resp_fake", &request.model)
                        .status(status.clone())
                        .build();
                    if let Some(r) = reason {
                        resp.incomplete_details = Some(IncompleteDetails { reason: r.clone() });
                    }
                    Ok(resp)
                }
                FakeOutcome::Err(code) => Err(code.into_response()),
                FakeOutcome::CancelThen(status) => {
                    cancel.cancel();
                    let resp = ResponsesResponse::builder("resp_fake", &request.model)
                        .status(status.clone())
                        .build();
                    Ok(resp)
                }
            }
        }
    }

    fn config_with_retries(max_retries: u32) -> BackgroundConfig {
        BackgroundConfig {
            max_retries,
            // Tiny backoff so the retry test reclaims quickly.
            retry_base_delay_secs: 1,
            retry_max_delay_secs: 1,
            lease_duration_secs: 60,
            ..Default::default()
        }
    }

    /// Enqueue + claim a job, returning the leased job (with a valid lease).
    async fn enqueue_and_claim(
        repo: &Arc<dyn BackgroundResponseRepository>,
        req: EnqueueRequest,
        worker_id: &str,
    ) -> LeasedJob {
        repo.enqueue(req, None, None).await.unwrap();
        repo.claim_next(worker_id, Utc::now(), Duration::from_secs(60))
            .await
            .unwrap()
            .expect("claim")
    }

    fn real_worker(
        repo: &Arc<dyn BackgroundResponseRepository>,
        headless: Arc<dyn HeadlessResponses>,
        config: BackgroundConfig,
    ) -> RealBackgroundWorker {
        RealBackgroundWorker::new(Arc::clone(repo), headless, config)
    }

    #[tokio::test]
    async fn real_worker_completed_finalizes_completed() {
        let rs = Arc::new(MemoryResponseStorage::new());
        let repo: Arc<dyn BackgroundResponseRepository> =
            Arc::new(MemoryBackgroundRepository::new(Arc::clone(&rs)));
        let job = enqueue_and_claim(&repo, enqueue_req("r1"), "bg-w").await;

        let headless = Arc::new(FakeHeadless::new(FakeOutcome::Ok(
            ResponseStatus::Completed,
            None,
        )));
        real_worker(&repo, headless, config_with_retries(3))
            .execute(job)
            .await;

        let stored = rs
            .get_response(&ResponseId::from("r1"))
            .await
            .unwrap()
            .expect("present");
        assert_eq!(stored.raw_response["status"], "completed");
        // The durable id must be forced into the stored payload.
        assert_eq!(stored.raw_response["id"], "r1");
    }

    #[tokio::test]
    async fn real_worker_incomplete_truncation_finalizes_incomplete() {
        let rs = Arc::new(MemoryResponseStorage::new());
        let repo: Arc<dyn BackgroundResponseRepository> =
            Arc::new(MemoryBackgroundRepository::new(Arc::clone(&rs)));
        let job = enqueue_and_claim(&repo, enqueue_req("r1"), "bg-w").await;

        let headless = Arc::new(FakeHeadless::new(FakeOutcome::Ok(
            ResponseStatus::Incomplete,
            Some(IncompleteReason::MaxOutputTokens),
        )));
        real_worker(&repo, headless, config_with_retries(3))
            .execute(job)
            .await;

        let stored = rs
            .get_response(&ResponseId::from("r1"))
            .await
            .unwrap()
            .expect("present");
        assert_eq!(stored.raw_response["status"], "incomplete");
    }

    #[tokio::test]
    async fn real_worker_ok_failed_finalizes_failed() {
        let rs = Arc::new(MemoryResponseStorage::new());
        let repo: Arc<dyn BackgroundResponseRepository> =
            Arc::new(MemoryBackgroundRepository::new(Arc::clone(&rs)));
        let job = enqueue_and_claim(&repo, enqueue_req("r1"), "bg-w").await;

        // An Ok response with `failed` status (e.g. max_tool_calls) is terminal
        // failed, NOT a transient retry.
        let headless = Arc::new(FakeHeadless::new(FakeOutcome::Ok(
            ResponseStatus::Failed,
            None,
        )));
        real_worker(&repo, headless, config_with_retries(3))
            .execute(job)
            .await;

        let stored = rs
            .get_response(&ResponseId::from("r1"))
            .await
            .unwrap()
            .expect("present");
        assert_eq!(stored.raw_response["status"], "failed");
        // Terminal: not re-claimable.
        assert!(repo
            .claim_next("probe", Utc::now(), Duration::from_secs(30))
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn real_worker_err_requeues_then_exhausts_to_failed() {
        let rs = Arc::new(MemoryResponseStorage::new());
        let repo: Arc<dyn BackgroundResponseRepository> =
            Arc::new(MemoryBackgroundRepository::new(Arc::clone(&rs)));

        // max_retries = 1: first Err requeues (attempt 0 < 1), second Err on the
        // re-claimed job (attempt 1, not < 1) finalizes failed.
        let config = config_with_retries(1);

        let job = enqueue_and_claim(&repo, enqueue_req("r1"), "bg-w").await;
        assert_eq!(job.retry_attempt, 0);
        let headless = Arc::new(FakeHeadless::new(FakeOutcome::Err(StatusCode::BAD_GATEWAY)));
        real_worker(&repo, Arc::clone(&headless) as _, config.clone())
            .execute(job)
            .await;

        // Not terminal yet — it was requeued. Reclaim after the (1s) backoff.
        let reclaimed = repo
            .claim_next(
                "bg-w",
                Utc::now() + chrono::Duration::seconds(2),
                Duration::from_secs(60),
            )
            .await
            .unwrap()
            .expect("requeued job must be re-claimable after backoff");
        assert_eq!(reclaimed.retry_attempt, 1);

        // Second failure exhausts retries → failed.
        real_worker(&repo, headless, config)
            .execute(reclaimed)
            .await;
        let stored = rs
            .get_response(&ResponseId::from("r1"))
            .await
            .unwrap()
            .expect("present");
        assert_eq!(stored.raw_response["status"], "failed");
    }

    #[tokio::test]
    async fn real_worker_cancel_finalizes_cancelled() {
        let rs = Arc::new(MemoryResponseStorage::new());
        let repo: Arc<dyn BackgroundResponseRepository> =
            Arc::new(MemoryBackgroundRepository::new(Arc::clone(&rs)));
        let job = enqueue_and_claim(&repo, enqueue_req("r1"), "bg-w").await;

        // The fake fires the cancel token mid-run, then returns a (would-be)
        // completed response. The worker must finalize cancelled (token fired).
        let headless = Arc::new(FakeHeadless::new(FakeOutcome::CancelThen(
            ResponseStatus::Completed,
        )));
        real_worker(&repo, headless, config_with_retries(3))
            .execute(job)
            .await;

        let stored = rs
            .get_response(&ResponseId::from("r1"))
            .await
            .unwrap()
            .expect("present");
        assert_eq!(stored.raw_response["status"], "cancelled");
    }

    #[tokio::test]
    async fn real_worker_deserialize_failure_finalizes_failed() {
        let rs = Arc::new(MemoryResponseStorage::new());
        let repo: Arc<dyn BackgroundResponseRepository> =
            Arc::new(MemoryBackgroundRepository::new(Arc::clone(&rs)));

        // request_json is not a valid ResponsesRequest (missing required fields,
        // wrong shape) → reconstruction fails → finalize failed.
        let mut bad = enqueue_req("r1");
        bad.request_json = json!({"not": "a responses request"});
        let job = enqueue_and_claim(&repo, bad, "bg-w").await;

        let headless = Arc::new(FakeHeadless::new(FakeOutcome::Ok(
            ResponseStatus::Completed,
            None,
        )));
        real_worker(&repo, headless, config_with_retries(3))
            .execute(job)
            .await;

        let stored = rs
            .get_response(&ResponseId::from("r1"))
            .await
            .unwrap()
            .expect("present");
        assert_eq!(stored.raw_response["status"], "failed");
        assert_eq!(
            stored.raw_response["error"]["code"],
            "background_request_invalid"
        );
    }

    #[tokio::test]
    async fn real_worker_streaming_finalizes_failed() {
        let rs = Arc::new(MemoryResponseStorage::new());
        let repo: Arc<dyn BackgroundResponseRepository> =
            Arc::new(MemoryBackgroundRepository::new(Arc::clone(&rs)));

        // stream_enabled = true → Slice B not supported → finalize failed.
        let mut req = enqueue_req("r1");
        req.stream_enabled = true;
        let job = enqueue_and_claim(&repo, req, "bg-w").await;

        let headless = Arc::new(FakeHeadless::new(FakeOutcome::Ok(
            ResponseStatus::Completed,
            None,
        )));
        real_worker(&repo, headless, config_with_retries(3))
            .execute(job)
            .await;

        let stored = rs
            .get_response(&ResponseId::from("r1"))
            .await
            .unwrap()
            .expect("present");
        assert_eq!(stored.raw_response["status"], "failed");
        assert_eq!(
            stored.raw_response["error"]["code"],
            "background_streaming_unsupported"
        );
    }

    #[tokio::test]
    async fn real_worker_reconstructs_input_and_clears_chain() {
        let rs = Arc::new(MemoryResponseStorage::new());
        let repo: Arc<dyn BackgroundResponseRepository> =
            Arc::new(MemoryBackgroundRepository::new(Arc::clone(&rs)));

        // request_json carries previous_response_id + conversation; input is the
        // enqueue-time snapshot. The worker must use the snapshot input and clear
        // the chain fields so the pipeline doesn't double-resolve history.
        let mut req = enqueue_req("r1");
        req.request_json = json!({
            "model": "gpt-5.1",
            "input": "ORIGINAL — should be replaced by snapshot",
            "previous_response_id": "resp_prev",
            "conversation": "conv_123",
        });
        req.input = json!([
            {"type": "message", "role": "user", "content": "SNAPSHOT input"}
        ]);
        let job = enqueue_and_claim(&repo, req, "bg-w").await;

        let headless = Arc::new(FakeHeadless::new(FakeOutcome::Ok(
            ResponseStatus::Completed,
            None,
        )));
        real_worker(&repo, Arc::clone(&headless) as _, config_with_retries(3))
            .execute(job)
            .await;

        let captured = headless
            .captured
            .lock()
            .unwrap()
            .clone()
            .expect("headless must have been called");
        assert!(
            captured.previous_response_id.is_none(),
            "previous_response_id must be cleared"
        );
        assert!(
            captured.conversation.is_none(),
            "conversation must be cleared"
        );
        assert_eq!(captured.stream, Some(false), "stream must be forced false");
        // The snapshot input must be installed (a structured array), not the
        // original string from request_json.
        let input_json = to_value(&captured.input).unwrap();
        assert!(input_json.is_array(), "input must be the snapshot array");
    }
}
