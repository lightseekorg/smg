//! Background-mode response repository contract.
//!
//! Defines the storage abstraction for SMG's background-mode execution path:
//! durable queue + lease-and-claim + stream-event persistence + terminal-state
//! updates, all owned by the backend implementation's own transaction boundary.
//!
//! This PR ships only the trait and the request/response types. Concrete
//! Postgres / Oracle / in-memory implementations land in later PRs.
//!
//! # Why a new trait instead of extending `ResponseStorage`?
//!
//! [`crate::ResponseStorage`] was designed for whole-response persistence with
//! simple read helpers. Background mode needs:
//!
//! - atomic multi-row updates (`responses` + `background_queue`)
//! - lease / heartbeat / requeue over a queue
//! - append-only stream-event log with monotonic per-response sequences
//! - request-context round-trip from enqueue through finalize
//!
//! Retrofitting those onto `ResponseStorage` would either break its contract or
//! couple unrelated callers to queue semantics. A separate trait keeps the read
//! path unchanged and lets background-mode callers depend only on what they
//! need.
//!
//! The design document is `.claude/designs/background-mode/draft.md`.

use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::{context::RequestContext, core::ResponseId};

// ============================================================================
// Error type
// ============================================================================

/// Errors returned by [`BackgroundResponseRepository`] operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BackgroundRepositoryError {
    /// The target response did not exist.
    #[error("background response not found: {0}")]
    NotFound(ResponseId),

    /// A state-machine transition was rejected (e.g. finalizing an already-terminal
    /// response, heartbeating a lease that no longer belongs to the caller).
    #[error("invalid transition: {0}")]
    InvalidTransition(String),

    /// The worker attempted an operation on a lease it does not hold.
    #[error("lease not held by worker {worker_id} for response {response_id}")]
    LeaseNotHeld {
        response_id: ResponseId,
        worker_id: String,
    },

    /// Stream cursor is older than the retained chunk window.
    #[error("stream cursor expired for response {response_id} at sequence {starting_after}")]
    StreamCursorExpired {
        response_id: ResponseId,
        starting_after: i64,
    },

    /// Queue is at the configured depth ceiling.
    #[error("queue full (depth {current} >= limit {limit})")]
    QueueFull { current: u64, limit: u64 },

    /// A serialization failure encountered while encoding / decoding stored JSON.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// A backend-specific failure (connection pool, SQL error, etc.).
    #[error("backend error: {0}")]
    Backend(String),
}

/// Convenience alias for repository operation results.
pub type BackgroundRepositoryResult<T> = Result<T, BackgroundRepositoryError>;

// ============================================================================
// Enqueue
// ============================================================================

/// Input payload for [`BackgroundResponseRepository::enqueue`].
///
/// Captures both the accepted-client-request (pre-provider-rewrite) and the
/// resolved execution snapshot the worker will use. `raw_response` holds the
/// queued `Response` object that `GET /v1/responses/{id}` will return while
/// the job is pending.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EnqueueRequest {
    /// Response ID. Typically generated via [`ResponseId::new()`], which
    /// produces a ULID string (not a `resp_…`-prefixed identifier).
    pub response_id: ResponseId,

    /// The accepted client request, serialized after validation but before any
    /// provider-specific rewrite. Used by the worker to reproduce the accepted
    /// contract even if the provider adapter rewrites the outbound request.
    pub request_json: Value,

    /// Resolved execution snapshot (e.g. flattened conversation history or
    /// `previous_response_id` chain). The worker reads from this, not from
    /// mutable conversation state.
    pub input: Value,

    /// Initial `raw_response` object stored at enqueue time. Typically the
    /// `queued`-status Response skeleton. `GET /v1/responses/{id}` returns this
    /// verbatim until the worker updates it.
    pub raw_response: Value,

    /// Whether the request was created with `stream=true`; controls whether
    /// `GET /v1/responses/{id}?stream=true` is valid for replay/tail.
    pub stream_enabled: bool,

    /// Queue priority; lower is higher priority.
    pub priority: i32,

    /// Linked conversation, if any. Resolution from the request's
    /// `conversation` field happens at enqueue time.
    pub conversation_id: Option<String>,

    /// Model identifier (e.g. `"gpt-5.1"`).
    pub model: String,

    /// Safety identifier for moderation / rate-limiting, mirrored from the
    /// request.
    pub safety_identifier: Option<String>,

    /// `previous_response_id` resolution metadata captured at enqueue, for
    /// storage and replay. Not the full chain — just the reference.
    pub previous_response_id: Option<ResponseId>,
}

impl EnqueueRequest {
    /// Construct an [`EnqueueRequest`] with the required fields and no
    /// `conversation_id`, `safety_identifier`, or `previous_response_id`.
    ///
    /// `EnqueueRequest` is `#[non_exhaustive]` so external crates cannot use
    /// struct-literal construction. Use this constructor, then assign optional
    /// metadata directly to the public fields (there are no setter methods —
    /// every field is `pub`).
    pub fn new(
        response_id: ResponseId,
        model: String,
        request_json: Value,
        input: Value,
        raw_response: Value,
        stream_enabled: bool,
        priority: i32,
    ) -> Self {
        Self {
            response_id,
            request_json,
            input,
            raw_response,
            stream_enabled,
            priority,
            conversation_id: None,
            model,
            safety_identifier: None,
            previous_response_id: None,
        }
    }
}

/// Acknowledgement returned by [`BackgroundResponseRepository::enqueue`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct QueuedResponse {
    pub response_id: ResponseId,
    /// Server-side enqueue timestamp.
    pub created_at: DateTime<Utc>,
    /// Depth of the queue at the point of insertion (for observability only —
    /// not safe to use for back-pressure decisions).
    pub queue_depth_at_insert: u64,
}

// ============================================================================
// Claim + heartbeat + requeue
// ============================================================================

/// A leased job returned by [`BackgroundResponseRepository::claim_next`].
///
/// The caller holds the lease until `lease_expires_at`; extending requires
/// [`BackgroundResponseRepository::heartbeat`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LeasedJob {
    pub response_id: ResponseId,
    pub retry_attempt: u32,
    pub priority: i32,
    pub worker_id: String,
    pub lease_expires_at: DateTime<Utc>,
    /// Snapshot of the execution input captured at enqueue time. The worker
    /// should not re-read mutable conversation state.
    pub input: Value,
    /// Accepted client request payload for reproducing the client-visible
    /// contract (see [`EnqueueRequest::request_json`]).
    pub request_json: Value,
    pub model: String,
    pub conversation_id: Option<String>,
    pub previous_response_id: Option<ResponseId>,
    pub stream_enabled: bool,
}

// ============================================================================
// Cancel
// ============================================================================

/// Result of a cancel request, reflecting what the repository actually did.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum StoredCancelResult {
    /// Queued → cancelled immediately; the queue row was removed and the
    /// response row transitioned to terminal `cancelled`.
    QueuedCancelled,
    /// In-progress → `cancel_requested = true` recorded. The worker must
    /// observe the flag and converge at its next checkpoint.
    CancelRequested,
    /// The response was already terminal; no state change.
    AlreadyTerminal,
    /// No response existed with that id.
    NotFound,
}

// ============================================================================
// Stream events
// ============================================================================

/// A single persisted SSE event for a background-mode response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StoredStreamEvent {
    pub response_id: ResponseId,
    /// Monotonic per-response sequence, allocated by the repository atomically
    /// with the insert (callers do not own a local sequence counter).
    pub sequence: i64,
    pub event_type: String,
    pub data: Value,
    pub created_at: DateTime<Utc>,
}

/// Batch returned by [`BackgroundResponseRepository::load_resume_events`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ResumeEventBatch {
    /// Events strictly greater than the caller's `starting_after`. Ordered by
    /// `sequence` ascending.
    pub events: Vec<StoredStreamEvent>,
    /// Whether the response has reached a terminal state. If `true` and the
    /// batch is empty, the replay is complete. If `false`, the caller should
    /// continue tailing.
    pub terminal: bool,
}

// ============================================================================
// Finalize
// ============================================================================

/// Terminal status that the worker writes via
/// [`BackgroundResponseRepository::finalize`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FinalizeStatus {
    Completed,
    Incomplete,
    Failed,
    Cancelled,
}

/// Input payload for [`BackgroundResponseRepository::finalize`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FinalizeRequest {
    pub response_id: ResponseId,
    pub worker_id: String,
    pub status: FinalizeStatus,
    /// Final `raw_response` object to store (this is what
    /// `GET /v1/responses/{id}` will return).
    pub raw_response: Value,
    pub completed_at: DateTime<Utc>,
}

impl FinalizeRequest {
    /// Construct a [`FinalizeRequest`]. `FinalizeRequest` is `#[non_exhaustive]`,
    /// so external crates must go through this constructor rather than a
    /// struct literal.
    pub fn new(
        response_id: ResponseId,
        worker_id: String,
        status: FinalizeStatus,
        raw_response: Value,
        completed_at: DateTime<Utc>,
    ) -> Self {
        Self {
            response_id,
            worker_id,
            status,
            raw_response,
            completed_at,
        }
    }
}

/// Result of finalization. The repository resolves cancel races internally; the
/// worker receives the actual persisted status (which may be `Cancelled` even if
/// the worker requested a different terminal status).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct FinalizeResult {
    pub response_id: ResponseId,
    pub final_status: FinalizeStatus,
    /// `true` when the repository observed `cancel_requested=true` and wrote
    /// `Cancelled` instead of the worker-requested status.
    pub cancel_won: bool,
}

// ============================================================================
// Delete
// ============================================================================

/// Result of a delete request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeleteResult {
    /// Response and any associated queue/chunk rows were removed.
    Deleted,
    /// Response was in-progress; delete rejected with `409 response_delete_in_progress`
    /// per the design. Caller must cancel first and delete only after the
    /// response reaches a terminal state.
    InProgress,
    /// No response existed with that id.
    NotFound,
}

// ============================================================================
// Repository trait
// ============================================================================

/// Background-mode response persistence contract.
///
/// Every method that touches both `responses` and `background_queue`
/// (enqueue, claim, cancel, finalize, delete) acquires locks in the order
/// **queue row first, then response row** and commits in a single backend
/// transaction. Implementations are expected to own the transaction boundary
/// internally so the caller never holds cross-method locks.
///
/// See `.claude/designs/background-mode/draft.md` for the state machine and
/// atomicity rules.
#[async_trait]
pub trait BackgroundResponseRepository: Send + Sync {
    /// Insert a new response + queue row atomically, transitioning
    /// `status='queued'`.
    async fn enqueue(
        &self,
        req: EnqueueRequest,
        request_context: Option<RequestContext>,
    ) -> BackgroundRepositoryResult<QueuedResponse>;

    /// Claim the next runnable queue row using `FOR UPDATE SKIP LOCKED` (or the
    /// backend's equivalent). Returns `None` when there is nothing to claim.
    /// The returned [`LeasedJob`] holds the lease until `lease_expires_at`.
    async fn claim_next(
        &self,
        worker_id: &str,
        now: DateTime<Utc>,
        lease: Duration,
    ) -> BackgroundRepositoryResult<Option<LeasedJob>>;

    /// Extend an active lease. Fails with [`BackgroundRepositoryError::LeaseNotHeld`]
    /// if the lease expired or another worker took over.
    ///
    /// `now` is the caller's reference time — used both to check that the
    /// existing lease hasn't expired and to compute the new `lease_expires_at`
    /// as `now + lease`. Passed explicitly (rather than calling `Utc::now()`
    /// internally) so tests can drive the lease clock deterministically and
    /// so `heartbeat` stays consistent with [`Self::claim_next`] and
    /// [`Self::requeue_expired`], which also take `now`.
    async fn heartbeat(
        &self,
        response_id: &ResponseId,
        worker_id: &str,
        now: DateTime<Utc>,
        lease: Duration,
    ) -> BackgroundRepositoryResult<()>;

    /// Request cancellation. The returned [`StoredCancelResult`] reflects
    /// the state the repository observed:
    /// queued → [`StoredCancelResult::QueuedCancelled`];
    /// in-progress → [`StoredCancelResult::CancelRequested`];
    /// terminal → [`StoredCancelResult::AlreadyTerminal`];
    /// missing → [`StoredCancelResult::NotFound`].
    async fn request_cancel(
        &self,
        response_id: &ResponseId,
    ) -> BackgroundRepositoryResult<StoredCancelResult>;

    /// Append one SSE event to the persisted stream log, atomically allocating
    /// the next sequence on the owning response row.
    async fn append_stream_event(
        &self,
        response_id: &ResponseId,
        event_type: &str,
        data: Value,
    ) -> BackgroundRepositoryResult<StoredStreamEvent>;

    /// Load stream events strictly greater than `starting_after` (or all events
    /// when `starting_after` is `None`), plus whether the response has reached
    /// a terminal state.
    async fn load_resume_events(
        &self,
        response_id: &ResponseId,
        starting_after: Option<i64>,
    ) -> BackgroundRepositoryResult<ResumeEventBatch>;

    /// Terminalize a leased job. Fails with
    /// [`BackgroundRepositoryError::LeaseNotHeld`] if the caller's lease has
    /// expired at `now`, even if no sweeper has requeued the row yet — this
    /// prevents a stale worker from committing terminal state after its lease
    /// window. The repository internally resolves the cancel/complete race;
    /// see [`FinalizeResult::cancel_won`].
    async fn finalize(
        &self,
        update: FinalizeRequest,
        now: DateTime<Utc>,
    ) -> BackgroundRepositoryResult<FinalizeResult>;

    /// Re-queue rows whose lease has expired. Idempotent; safe to call from
    /// any replica. Returns the number of rows requeued.
    async fn requeue_expired(&self, now: DateTime<Utc>) -> BackgroundRepositoryResult<u64>;

    /// Load the [`RequestContext`] stored at enqueue time. Used by the worker
    /// to replay storage-hook context during finalize-adjacent writes and
    /// conversation linkage.
    async fn load_request_context(
        &self,
        response_id: &ResponseId,
    ) -> BackgroundRepositoryResult<Option<RequestContext>>;

    /// Delete a background response with the design's rules
    /// (queued/terminal → delete; in-progress → rejected).
    async fn delete_background_response(
        &self,
        response_id: &ResponseId,
    ) -> BackgroundRepositoryResult<DeleteResult>;
}
