//! Background worker seam.
//!
//! The scheduler claims jobs from the [`BackgroundResponseRepository`] and hands
//! each one to a [`BackgroundWorker`] for execution. This module defines that
//! seam plus a default implementation used until the real executor lands in
//! BGM-PR-07.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{json, Value};
use smg_data_connector::{
    BackgroundResponseRepository, FinalizeRequest, FinalizeStatus, LeasedJob,
};
use tracing::warn;

/// Error code stamped on the response that the [`UnavailableBackgroundWorker`]
/// finalizes. Distinct, greppable, and stable so operators and tests can key
/// off it.
pub const BACKGROUND_EXECUTION_UNAVAILABLE: &str = "background_execution_unavailable";

/// Executes a single leased background job end to end.
///
/// The scheduler owns the concurrency permit and the claim/sweep loops; an
/// implementation of this trait owns everything from "I have a leased job" to
/// "the response row is terminal" (running the model, persisting stream events,
/// honoring cancel/retry, and calling
/// [`BackgroundResponseRepository::finalize`]). The real implementation lands in
/// BGM-PR-07; [`UnavailableBackgroundWorker`] is the placeholder until then.
#[async_trait]
pub trait BackgroundWorker: Send + Sync {
    /// Execute one claimed job. The implementation is responsible for driving
    /// the job to a terminal state (typically via `finalize`); the scheduler
    /// holds the concurrency permit until this future resolves.
    async fn execute(&self, job: LeasedJob);
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
            json!({"model": "gpt-5.1"}),
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
}
