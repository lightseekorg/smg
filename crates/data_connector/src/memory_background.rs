//! In-memory `BackgroundResponseRepository`.
//!
//! Non-durable by design: process restart drops all queued / in-progress /
//! terminal state, and clients polling across a restart receive `NotFound`.
//! Suitable for dev, CI, and single-node deployments.
//!
//! `finalize` resolves the cancel/complete race: if `cancel_requested=true`
//! at finalize time, the written status is `Cancelled` regardless of the
//! worker-supplied status and `FinalizeResult::cancel_won=true`.

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde_json::Value;

use crate::{
    background::{
        BackgroundRepositoryError, BackgroundRepositoryResult, BackgroundResponseRepository,
        DeleteResult, EnqueueRequest, FinalizeRequest, FinalizeResult, FinalizeStatus, LeasedJob,
        QueuedResponse, ResumeEventBatch, StoredCancelResult, StoredStreamEvent,
    },
    context::RequestContext,
    core::ResponseId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InternalStatus {
    Queued,
    InProgress,
    Completed,
    Incomplete,
    Failed,
    Cancelled,
}

impl InternalStatus {
    fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Incomplete | Self::Failed | Self::Cancelled
        )
    }

    fn from_finalize(s: FinalizeStatus) -> Self {
        match s {
            FinalizeStatus::Completed => Self::Completed,
            FinalizeStatus::Incomplete => Self::Incomplete,
            FinalizeStatus::Failed => Self::Failed,
            FinalizeStatus::Cancelled => Self::Cancelled,
        }
    }

    fn to_finalize(self) -> Option<FinalizeStatus> {
        match self {
            Self::Completed => Some(FinalizeStatus::Completed),
            Self::Incomplete => Some(FinalizeStatus::Incomplete),
            Self::Failed => Some(FinalizeStatus::Failed),
            Self::Cancelled => Some(FinalizeStatus::Cancelled),
            Self::Queued | Self::InProgress => None,
        }
    }
}

#[derive(Debug, Clone)]
struct Entry {
    status: InternalStatus,

    request_json: Value,
    input: Value,
    raw_response: Value,
    stream_enabled: bool,
    model: String,
    conversation_id: Option<String>,
    previous_response_id: Option<ResponseId>,

    priority: i32,
    retry_attempt: u32,
    worker_id: Option<String>,
    lease_expires_at: Option<DateTime<Utc>>,
    next_attempt_at: DateTime<Utc>,
    created_at: DateTime<Utc>,

    cancel_requested: bool,
    request_context: Option<RequestContext>,

    stream_events: Vec<StoredStreamEvent>,
    next_stream_sequence: i64,

    started_at: Option<DateTime<Utc>>,
    completed_at: Option<DateTime<Utc>>,
}

#[derive(Default)]
struct State {
    entries: HashMap<ResponseId, Entry>,
}

#[derive(Clone, Default)]
pub struct MemoryBackgroundRepository {
    state: Arc<RwLock<State>>,
}

impl MemoryBackgroundRepository {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl BackgroundResponseRepository for MemoryBackgroundRepository {
    async fn enqueue(
        &self,
        req: EnqueueRequest,
        request_context: Option<RequestContext>,
    ) -> BackgroundRepositoryResult<QueuedResponse> {
        let now = Utc::now();
        let mut state = self.state.write();

        if state.entries.contains_key(&req.response_id) {
            return Err(BackgroundRepositoryError::InvalidTransition(format!(
                "response {} already exists",
                req.response_id
            )));
        }

        let response_id = req.response_id.clone();
        let entry = Entry {
            status: InternalStatus::Queued,
            request_json: req.request_json,
            input: req.input,
            raw_response: req.raw_response,
            stream_enabled: req.stream_enabled,
            model: req.model,
            conversation_id: req.conversation_id,
            previous_response_id: req.previous_response_id,
            priority: req.priority,
            retry_attempt: 0,
            worker_id: None,
            lease_expires_at: None,
            next_attempt_at: now,
            created_at: now,
            cancel_requested: false,
            request_context,
            stream_events: Vec::new(),
            next_stream_sequence: 0,
            started_at: None,
            completed_at: None,
        };
        state.entries.insert(response_id.clone(), entry);

        let queue_depth = state
            .entries
            .values()
            .filter(|e| e.status == InternalStatus::Queued)
            .count() as u64;

        Ok(QueuedResponse {
            response_id,
            created_at: now,
            queue_depth_at_insert: queue_depth,
        })
    }

    async fn claim_next(
        &self,
        worker_id: &str,
        now: DateTime<Utc>,
        lease: Duration,
    ) -> BackgroundRepositoryResult<Option<LeasedJob>> {
        let mut state = self.state.write();

        let pick = state
            .entries
            .iter()
            .filter(|(_, e)| e.status == InternalStatus::Queued && e.next_attempt_at <= now)
            .min_by(|(_, a), (_, b)| {
                a.priority
                    .cmp(&b.priority)
                    .then(a.next_attempt_at.cmp(&b.next_attempt_at))
                    .then(a.created_at.cmp(&b.created_at))
            })
            .map(|(id, _)| id.clone());

        let Some(id) = pick else {
            return Ok(None);
        };

        let entry = state.entries.get_mut(&id).ok_or_else(|| {
            BackgroundRepositoryError::Backend(
                "claim_next invariant: picked entry missing under write lock".to_string(),
            )
        })?;

        // Cancel-before-claim: flip straight to Cancelled rather than hand
        // a worker a job it must not execute. The worker sees `None`.
        if entry.cancel_requested {
            entry.status = InternalStatus::Cancelled;
            entry.completed_at = Some(now);
            return Ok(None);
        }

        let expires_at =
            now + chrono::Duration::from_std(lease).unwrap_or(chrono::Duration::zero());
        entry.status = InternalStatus::InProgress;
        entry.worker_id = Some(worker_id.to_string());
        entry.lease_expires_at = Some(expires_at);
        if entry.started_at.is_none() {
            entry.started_at = Some(now);
        }

        Ok(Some(LeasedJob {
            response_id: id,
            retry_attempt: entry.retry_attempt,
            priority: entry.priority,
            worker_id: worker_id.to_string(),
            lease_expires_at: expires_at,
            input: entry.input.clone(),
            request_json: entry.request_json.clone(),
            model: entry.model.clone(),
            conversation_id: entry.conversation_id.clone(),
            previous_response_id: entry.previous_response_id.clone(),
            stream_enabled: entry.stream_enabled,
        }))
    }

    async fn heartbeat(
        &self,
        response_id: &ResponseId,
        worker_id: &str,
        now: DateTime<Utc>,
        lease: Duration,
    ) -> BackgroundRepositoryResult<()> {
        let mut state = self.state.write();
        let entry = state
            .entries
            .get_mut(response_id)
            .ok_or_else(|| BackgroundRepositoryError::NotFound(response_id.clone()))?;

        let held_by_caller = entry.status == InternalStatus::InProgress
            && entry.worker_id.as_deref() == Some(worker_id)
            && entry.lease_expires_at.is_some_and(|exp| exp > now);
        if !held_by_caller {
            return Err(BackgroundRepositoryError::LeaseNotHeld {
                response_id: response_id.clone(),
                worker_id: worker_id.to_string(),
            });
        }

        entry.lease_expires_at =
            Some(now + chrono::Duration::from_std(lease).unwrap_or(chrono::Duration::zero()));
        Ok(())
    }

    async fn request_cancel(
        &self,
        response_id: &ResponseId,
    ) -> BackgroundRepositoryResult<StoredCancelResult> {
        let now = Utc::now();
        let mut state = self.state.write();
        let Some(entry) = state.entries.get_mut(response_id) else {
            return Ok(StoredCancelResult::NotFound);
        };

        if entry.status.is_terminal() {
            return Ok(StoredCancelResult::AlreadyTerminal);
        }

        match entry.status {
            InternalStatus::Queued => {
                entry.status = InternalStatus::Cancelled;
                entry.completed_at = Some(now);
                entry.worker_id = None;
                entry.lease_expires_at = None;
                Ok(StoredCancelResult::QueuedCancelled)
            }
            InternalStatus::InProgress => {
                entry.cancel_requested = true;
                Ok(StoredCancelResult::CancelRequested)
            }
            InternalStatus::Completed
            | InternalStatus::Incomplete
            | InternalStatus::Failed
            | InternalStatus::Cancelled => Ok(StoredCancelResult::AlreadyTerminal),
        }
    }

    async fn append_stream_event(
        &self,
        response_id: &ResponseId,
        event_type: &str,
        data: Value,
    ) -> BackgroundRepositoryResult<StoredStreamEvent> {
        let now = Utc::now();
        let mut state = self.state.write();
        let entry = state
            .entries
            .get_mut(response_id)
            .ok_or_else(|| BackgroundRepositoryError::NotFound(response_id.clone()))?;

        let seq = entry.next_stream_sequence;
        entry.next_stream_sequence = seq
            .checked_add(1)
            .ok_or_else(|| BackgroundRepositoryError::Backend("sequence overflow".to_string()))?;

        let event = StoredStreamEvent {
            response_id: response_id.clone(),
            sequence: seq,
            event_type: event_type.to_string(),
            data,
            created_at: now,
        };
        entry.stream_events.push(event.clone());
        Ok(event)
    }

    async fn load_resume_events(
        &self,
        response_id: &ResponseId,
        starting_after: Option<i64>,
    ) -> BackgroundRepositoryResult<ResumeEventBatch> {
        let state = self.state.read();
        let entry = state
            .entries
            .get(response_id)
            .ok_or_else(|| BackgroundRepositoryError::NotFound(response_id.clone()))?;

        let events: Vec<StoredStreamEvent> = entry
            .stream_events
            .iter()
            .filter(|e| match starting_after {
                Some(after) => e.sequence > after,
                None => true,
            })
            .cloned()
            .collect();
        let terminal = entry.status.is_terminal();
        Ok(ResumeEventBatch { events, terminal })
    }

    async fn finalize(
        &self,
        update: FinalizeRequest,
    ) -> BackgroundRepositoryResult<FinalizeResult> {
        let mut state = self.state.write();
        let entry = state
            .entries
            .get_mut(&update.response_id)
            .ok_or_else(|| BackgroundRepositoryError::NotFound(update.response_id.clone()))?;

        if let Some(prior) = entry.status.to_finalize() {
            return Ok(FinalizeResult {
                response_id: update.response_id,
                final_status: prior,
                cancel_won: false,
            });
        }

        if entry.worker_id.as_deref() != Some(update.worker_id.as_str()) {
            return Err(BackgroundRepositoryError::LeaseNotHeld {
                response_id: update.response_id.clone(),
                worker_id: update.worker_id,
            });
        }

        let (final_status, cancel_won) = if entry.cancel_requested {
            (FinalizeStatus::Cancelled, true)
        } else {
            (update.status, false)
        };

        entry.status = InternalStatus::from_finalize(final_status);
        entry.raw_response = update.raw_response;
        entry.completed_at = Some(update.completed_at);
        entry.worker_id = None;
        entry.lease_expires_at = None;

        Ok(FinalizeResult {
            response_id: update.response_id,
            final_status,
            cancel_won,
        })
    }

    async fn requeue_expired(&self, now: DateTime<Utc>) -> BackgroundRepositoryResult<u64> {
        let mut state = self.state.write();
        let mut count: u64 = 0;
        for entry in state.entries.values_mut() {
            if entry.status == InternalStatus::InProgress
                && entry.lease_expires_at.is_some_and(|exp| exp <= now)
            {
                entry.status = InternalStatus::Queued;
                entry.worker_id = None;
                entry.lease_expires_at = None;
                entry.retry_attempt = entry.retry_attempt.saturating_add(1);
                entry.next_attempt_at = now;
                count += 1;
            }
        }
        Ok(count)
    }

    async fn load_request_context(
        &self,
        response_id: &ResponseId,
    ) -> BackgroundRepositoryResult<Option<RequestContext>> {
        let state = self.state.read();
        let entry = state
            .entries
            .get(response_id)
            .ok_or_else(|| BackgroundRepositoryError::NotFound(response_id.clone()))?;
        Ok(entry.request_context.clone())
    }

    async fn delete_background_response(
        &self,
        response_id: &ResponseId,
    ) -> BackgroundRepositoryResult<DeleteResult> {
        let mut state = self.state.write();
        let Some(entry) = state.entries.get(response_id) else {
            return Ok(DeleteResult::NotFound);
        };

        if entry.status == InternalStatus::InProgress {
            return Ok(DeleteResult::InProgress);
        }

        state.entries.remove(response_id);
        Ok(DeleteResult::Deleted)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

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

    fn finalize_req(id: &str, worker: &str, status: FinalizeStatus) -> FinalizeRequest {
        FinalizeRequest::new(
            ResponseId::from(id),
            worker.to_string(),
            status,
            json!({"id": id, "status": "completed"}),
            Utc::now(),
        )
    }

    #[tokio::test]
    async fn enqueue_creates_queued_response() {
        let repo = MemoryBackgroundRepository::new();
        let ack = repo.enqueue(enqueue_req("r1"), None).await.unwrap();
        assert_eq!(ack.response_id, ResponseId::from("r1"));
        assert_eq!(ack.queue_depth_at_insert, 1);
    }

    #[tokio::test]
    async fn enqueue_rejects_duplicate() {
        let repo = MemoryBackgroundRepository::new();
        repo.enqueue(enqueue_req("r1"), None).await.unwrap();
        let err = repo.enqueue(enqueue_req("r1"), None).await.unwrap_err();
        assert!(matches!(
            err,
            BackgroundRepositoryError::InvalidTransition(_)
        ));
    }

    #[tokio::test]
    async fn claim_next_respects_priority_and_creation_order() {
        let repo = MemoryBackgroundRepository::new();
        let mut r1 = enqueue_req("r1");
        r1.priority = 5;
        let mut r2 = enqueue_req("r2");
        r2.priority = 1;
        let mut r3 = enqueue_req("r3");
        r3.priority = 5;
        repo.enqueue(r1, None).await.unwrap();
        repo.enqueue(r2, None).await.unwrap();
        repo.enqueue(r3, None).await.unwrap();

        let now = Utc::now();
        let lease = Duration::from_secs(60);
        let mut claim_order: Vec<String> = Vec::new();
        for _ in 0..3 {
            let id = repo
                .claim_next("w1", now, lease)
                .await
                .unwrap()
                .map(|j| j.response_id.0)
                .unwrap_or_default();
            claim_order.push(id);
        }
        assert_eq!(claim_order, vec!["r2", "r1", "r3"]);
    }

    #[tokio::test]
    async fn claim_next_empty_returns_none() {
        let repo = MemoryBackgroundRepository::new();
        let now = Utc::now();
        assert!(repo
            .claim_next("w1", now, Duration::from_secs(60))
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn claim_on_cancel_requested_skips_worker_and_marks_cancelled() {
        let repo = MemoryBackgroundRepository::new();
        repo.enqueue(enqueue_req("r1"), None).await.unwrap();
        assert!(matches!(
            repo.request_cancel(&ResponseId::from("r1")).await.unwrap(),
            StoredCancelResult::QueuedCancelled
        ));
        let claimed = repo
            .claim_next("w1", Utc::now(), Duration::from_secs(60))
            .await
            .unwrap();
        assert!(claimed.is_none());
    }

    #[tokio::test]
    async fn heartbeat_requires_active_lease() {
        let repo = MemoryBackgroundRepository::new();
        repo.enqueue(enqueue_req("r1"), None).await.unwrap();

        let err = repo
            .heartbeat(
                &ResponseId::from("r1"),
                "w1",
                Utc::now(),
                Duration::from_secs(60),
            )
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            BackgroundRepositoryError::LeaseNotHeld { .. }
        ));

        let now = Utc::now();
        let _ = repo
            .claim_next("w1", now, Duration::from_secs(60))
            .await
            .unwrap();
        repo.heartbeat(
            &ResponseId::from("r1"),
            "w1",
            now + chrono::Duration::seconds(10),
            Duration::from_secs(60),
        )
        .await
        .unwrap();

        let err = repo
            .heartbeat(
                &ResponseId::from("r1"),
                "w2",
                Utc::now(),
                Duration::from_secs(60),
            )
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            BackgroundRepositoryError::LeaseNotHeld { .. }
        ));
    }

    #[tokio::test]
    async fn request_cancel_returns_variant_for_each_state() {
        let repo = MemoryBackgroundRepository::new();
        assert_eq!(
            repo.request_cancel(&ResponseId::from("missing"))
                .await
                .unwrap(),
            StoredCancelResult::NotFound
        );

        repo.enqueue(enqueue_req("r1"), None).await.unwrap();
        assert_eq!(
            repo.request_cancel(&ResponseId::from("r1")).await.unwrap(),
            StoredCancelResult::QueuedCancelled
        );
        assert_eq!(
            repo.request_cancel(&ResponseId::from("r1")).await.unwrap(),
            StoredCancelResult::AlreadyTerminal
        );

        repo.enqueue(enqueue_req("r2"), None).await.unwrap();
        let _ = repo
            .claim_next("w1", Utc::now(), Duration::from_secs(60))
            .await
            .unwrap();
        assert_eq!(
            repo.request_cancel(&ResponseId::from("r2")).await.unwrap(),
            StoredCancelResult::CancelRequested
        );
    }

    #[tokio::test]
    async fn append_and_load_stream_events_monotonic() {
        let repo = MemoryBackgroundRepository::new();
        repo.enqueue(enqueue_req("r1"), None).await.unwrap();
        let id = ResponseId::from("r1");

        let e0 = repo
            .append_stream_event(&id, "delta", json!({"text": "a"}))
            .await
            .unwrap();
        let e1 = repo
            .append_stream_event(&id, "delta", json!({"text": "b"}))
            .await
            .unwrap();
        let e2 = repo
            .append_stream_event(&id, "delta", json!({"text": "c"}))
            .await
            .unwrap();
        assert_eq!((e0.sequence, e1.sequence, e2.sequence), (0, 1, 2));

        let all = repo.load_resume_events(&id, None).await.unwrap();
        assert_eq!(all.events.len(), 3);
        assert!(!all.terminal);

        let tail = repo.load_resume_events(&id, Some(0)).await.unwrap();
        assert_eq!(tail.events.len(), 2);
        assert_eq!(tail.events[0].sequence, 1);
    }

    #[tokio::test]
    async fn finalize_writes_terminal_and_clears_lease() {
        let repo = MemoryBackgroundRepository::new();
        repo.enqueue(enqueue_req("r1"), None).await.unwrap();
        let _ = repo
            .claim_next("w1", Utc::now(), Duration::from_secs(60))
            .await
            .unwrap()
            .expect("claim");
        let result = repo
            .finalize(finalize_req("r1", "w1", FinalizeStatus::Completed))
            .await
            .unwrap();
        assert_eq!(result.final_status, FinalizeStatus::Completed);
        assert!(!result.cancel_won);

        let result = repo
            .finalize(finalize_req("r1", "w1", FinalizeStatus::Failed))
            .await
            .unwrap();
        assert_eq!(result.final_status, FinalizeStatus::Completed);
    }

    #[tokio::test]
    async fn finalize_cancel_wins_over_worker_status() {
        let repo = MemoryBackgroundRepository::new();
        repo.enqueue(enqueue_req("r1"), None).await.unwrap();
        let _ = repo
            .claim_next("w1", Utc::now(), Duration::from_secs(60))
            .await
            .unwrap()
            .expect("claim");
        assert_eq!(
            repo.request_cancel(&ResponseId::from("r1")).await.unwrap(),
            StoredCancelResult::CancelRequested
        );

        let result = repo
            .finalize(finalize_req("r1", "w1", FinalizeStatus::Completed))
            .await
            .unwrap();
        assert_eq!(result.final_status, FinalizeStatus::Cancelled);
        assert!(result.cancel_won);
    }

    #[tokio::test]
    async fn finalize_rejects_non_lease_holder() {
        let repo = MemoryBackgroundRepository::new();
        repo.enqueue(enqueue_req("r1"), None).await.unwrap();
        let _ = repo
            .claim_next("w1", Utc::now(), Duration::from_secs(60))
            .await
            .unwrap()
            .expect("claim");
        let err = repo
            .finalize(finalize_req("r1", "w2", FinalizeStatus::Completed))
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            BackgroundRepositoryError::LeaseNotHeld { .. }
        ));
    }

    #[tokio::test]
    async fn requeue_expired_moves_in_progress_back_to_queued() {
        let repo = MemoryBackgroundRepository::new();
        repo.enqueue(enqueue_req("r1"), None).await.unwrap();
        let claim_time = Utc::now();
        let _ = repo
            .claim_next("w1", claim_time, Duration::from_secs(60))
            .await
            .unwrap()
            .expect("claim");

        assert_eq!(
            repo.requeue_expired(claim_time + chrono::Duration::seconds(30))
                .await
                .unwrap(),
            0
        );
        assert_eq!(
            repo.requeue_expired(claim_time + chrono::Duration::seconds(120))
                .await
                .unwrap(),
            1
        );

        let job = repo
            .claim_next(
                "w2",
                claim_time + chrono::Duration::seconds(121),
                Duration::from_secs(60),
            )
            .await
            .unwrap()
            .expect("re-claim");
        assert_eq!(job.retry_attempt, 1);
    }

    #[tokio::test]
    async fn load_request_context_round_trips() {
        let repo = MemoryBackgroundRepository::new();
        let mut ctx = RequestContext::default();
        ctx.set("tenant_id", "acme");
        ctx.set("principal", "alice");
        repo.enqueue(enqueue_req("r1"), Some(ctx.clone()))
            .await
            .unwrap();
        let loaded = repo
            .load_request_context(&ResponseId::from("r1"))
            .await
            .unwrap()
            .expect("context must round-trip");
        assert_eq!(loaded.data(), ctx.data());

        repo.enqueue(enqueue_req("r2"), None).await.unwrap();
        assert!(repo
            .load_request_context(&ResponseId::from("r2"))
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn delete_honors_state_machine() {
        let repo = MemoryBackgroundRepository::new();
        assert_eq!(
            repo.delete_background_response(&ResponseId::from("nope"))
                .await
                .unwrap(),
            DeleteResult::NotFound
        );

        repo.enqueue(enqueue_req("r1"), None).await.unwrap();
        assert_eq!(
            repo.delete_background_response(&ResponseId::from("r1"))
                .await
                .unwrap(),
            DeleteResult::Deleted
        );

        repo.enqueue(enqueue_req("r2"), None).await.unwrap();
        let _ = repo
            .claim_next("w1", Utc::now(), Duration::from_secs(60))
            .await
            .unwrap()
            .expect("claim");
        assert_eq!(
            repo.delete_background_response(&ResponseId::from("r2"))
                .await
                .unwrap(),
            DeleteResult::InProgress
        );

        let _ = repo
            .finalize(finalize_req("r2", "w1", FinalizeStatus::Completed))
            .await
            .unwrap();
        assert_eq!(
            repo.delete_background_response(&ResponseId::from("r2"))
                .await
                .unwrap(),
            DeleteResult::Deleted
        );
    }
}
