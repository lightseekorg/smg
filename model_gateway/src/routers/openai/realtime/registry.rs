//! In-memory session and call registry for Realtime API connections.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use dashmap::DashMap;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

/// Connection state for a realtime session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// WebSocket upgrade accepted but upstream not yet connected.
    Pending,
    /// Bidirectional proxy is active.
    Connected,
    /// Connection has been closed.
    Disconnected,
}

/// A tracked WebSocket session.
#[derive(Debug, Clone)]
pub struct SessionEntry {
    pub session_id: String,
    pub model: String,
    pub worker_url: String,
    pub state: ConnectionState,
    pub created_at: Instant,
    pub cancel_token: CancellationToken,
}

/// A tracked WebRTC call.
#[derive(Debug, Clone)]
pub struct CallEntry {
    pub call_id: String,
    pub model: String,
    pub worker_url: String,
    pub created_at: Instant,
    pub cancel_token: CancellationToken,
}

const DEFAULT_MAX_SESSIONS: usize = 10_000;
const DEFAULT_MAX_CALLS: usize = 10_000;

/// DashMap-backed registry for realtime sessions and WebRTC calls.
///
/// Uses atomic counters for capacity enforcement to avoid TOCTOU races
/// between the length check and the DashMap insert.
#[derive(Debug)]
pub struct RealtimeRegistry {
    sessions: DashMap<String, SessionEntry>,
    calls: DashMap<String, CallEntry>,
    session_count: AtomicUsize,
    call_count: AtomicUsize,
    max_sessions: usize,
    max_calls: usize,
}

impl RealtimeRegistry {
    pub fn new() -> Self {
        Self {
            sessions: DashMap::new(),
            calls: DashMap::new(),
            session_count: AtomicUsize::new(0),
            call_count: AtomicUsize::new(0),
            max_sessions: DEFAULT_MAX_SESSIONS,
            max_calls: DEFAULT_MAX_CALLS,
        }
    }

    pub fn with_capacity(max_sessions: usize, max_calls: usize) -> Self {
        Self {
            sessions: DashMap::new(),
            calls: DashMap::new(),
            session_count: AtomicUsize::new(0),
            call_count: AtomicUsize::new(0),
            max_sessions,
            max_calls,
        }
    }

    // ---- Session methods ----

    pub fn register_session(
        &self,
        session_id: String,
        model: String,
        worker_url: String,
    ) -> Option<SessionEntry> {
        if !self.try_reserve_session() {
            warn!(
                max = self.max_sessions,
                "Session registry at capacity, rejecting registration"
            );
            return None;
        }
        let entry = SessionEntry {
            session_id: session_id.clone(),
            model,
            worker_url,
            state: ConnectionState::Pending,
            created_at: Instant::now(),
            cancel_token: CancellationToken::new(),
        };
        if let Some(old) = self.sessions.insert(session_id, entry.clone()) {
            // Replaced an existing entry — cancel its token so awaiting tasks
            // are notified, and undo the extra reservation.
            old.cancel_token.cancel();
            self.session_count.fetch_sub(1, Ordering::Relaxed);
        }
        Some(entry)
    }

    pub fn set_session_state(&self, session_id: &str, state: ConnectionState) {
        if let Some(mut entry) = self.sessions.get_mut(session_id) {
            entry.state = state;
        }
    }

    pub fn get_session(&self, session_id: &str) -> Option<SessionEntry> {
        self.sessions.get(session_id).map(|e| e.clone())
    }

    pub fn remove_session(&self, session_id: &str) -> Option<SessionEntry> {
        self.sessions.remove(session_id).map(|(_, e)| {
            e.cancel_token.cancel();
            self.session_count.fetch_sub(1, Ordering::Relaxed);
            e
        })
    }

    // ---- Call methods ----

    pub fn register_call(
        &self,
        call_id: String,
        model: String,
        worker_url: String,
    ) -> Option<CallEntry> {
        if !self.try_reserve_call() {
            warn!(
                max = self.max_calls,
                "Call registry at capacity, rejecting registration"
            );
            return None;
        }
        let entry = CallEntry {
            call_id: call_id.clone(),
            model,
            worker_url,
            created_at: Instant::now(),
            cancel_token: CancellationToken::new(),
        };
        if let Some(old) = self.calls.insert(call_id, entry.clone()) {
            // Replaced an existing entry — cancel its token so awaiting tasks
            // are notified, and undo the extra reservation.
            old.cancel_token.cancel();
            self.call_count.fetch_sub(1, Ordering::Relaxed);
        }
        Some(entry)
    }

    pub fn get_call(&self, call_id: &str) -> Option<CallEntry> {
        self.calls.get(call_id).map(|e| e.clone())
    }

    pub fn remove_call(&self, call_id: &str) -> Option<CallEntry> {
        self.calls.remove(call_id).map(|(_, e)| {
            e.cancel_token.cancel();
            self.call_count.fetch_sub(1, Ordering::Relaxed);
            e
        })
    }

    // ---- Atomic reservation helpers ----

    /// Atomically reserve a session slot. Returns `true` if a slot was
    /// successfully claimed, `false` if at capacity.
    fn try_reserve_session(&self) -> bool {
        self.try_reserve(&self.session_count, self.max_sessions)
    }

    /// Atomically reserve a call slot.
    fn try_reserve_call(&self) -> bool {
        self.try_reserve(&self.call_count, self.max_calls)
    }

    /// CAS loop: increment `counter` only if it is below `max`.
    fn try_reserve(&self, counter: &AtomicUsize, max: usize) -> bool {
        loop {
            let current = counter.load(Ordering::Relaxed);
            if current >= max {
                return false;
            }
            if counter
                .compare_exchange_weak(current, current + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
        }
    }

    // ---- Reaper ----

    /// Start a background task that evicts stale entries.
    ///
    /// Returns a `CancellationToken` that stops the reaper when cancelled.
    pub fn start_reaper(
        self: &Arc<Self>,
        max_age: Duration,
        interval: Duration,
    ) -> CancellationToken {
        let shutdown = CancellationToken::new();
        let token = shutdown.clone();
        let registry = Arc::clone(self);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            loop {
                tokio::select! {
                    _ = tick.tick() => {}
                    _ = shutdown.cancelled() => {
                        info!("Realtime registry reaper shutting down");
                        return;
                    }
                }
                let now = Instant::now();

                // Collect stale session keys, then remove and cancel each one
                // so awaiting tasks are properly signaled.
                let stale_session_ids: Vec<String> = registry
                    .sessions
                    .iter()
                    .filter(|e| now.duration_since(e.created_at) > max_age)
                    .map(|e| e.session_id.clone())
                    .collect();

                for id in &stale_session_ids {
                    if let Some((_, entry)) = registry.sessions.remove(id) {
                        entry.cancel_token.cancel();
                    }
                }

                let stale_call_ids: Vec<String> = registry
                    .calls
                    .iter()
                    .filter(|e| now.duration_since(e.created_at) > max_age)
                    .map(|e| e.call_id.clone())
                    .collect();

                for id in &stale_call_ids {
                    if let Some((_, entry)) = registry.calls.remove(id) {
                        entry.cancel_token.cancel();
                    }
                }

                let sessions_reaped = stale_session_ids.len();
                let calls_reaped = stale_call_ids.len();

                // Sync atomic counters.
                if sessions_reaped > 0 {
                    registry
                        .session_count
                        .fetch_sub(sessions_reaped, Ordering::Relaxed);
                }
                if calls_reaped > 0 {
                    registry
                        .call_count
                        .fetch_sub(calls_reaped, Ordering::Relaxed);
                }

                if sessions_reaped > 0 || calls_reaped > 0 {
                    debug!(
                        sessions_reaped,
                        calls_reaped, "Realtime registry reaper cycle"
                    );
                }
            }
        });
        info!("Realtime registry reaper started (max_age={max_age:?}, interval={interval:?})");
        token
    }

    /// Stats for observability.
    pub fn session_count(&self) -> usize {
        self.session_count.load(Ordering::Relaxed)
    }

    pub fn call_count(&self) -> usize {
        self.call_count.load(Ordering::Relaxed)
    }
}

impl Default for RealtimeRegistry {
    fn default() -> Self {
        Self::new()
    }
}
