//! In-memory session and call registry for Realtime API connections.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use dashmap::DashMap;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

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
    pub state: ConnectionState,
    pub created_at: Instant,
    pub cancel_token: CancellationToken,
}

/// DashMap-backed registry for realtime sessions and WebRTC calls.
///
/// No fixed capacity — DashMap grows dynamically. The reaper handles
/// cleanup of stale entries.
#[derive(Debug)]
pub struct RealtimeRegistry {
    sessions: DashMap<String, SessionEntry>,
    calls: DashMap<String, CallEntry>,
}

impl RealtimeRegistry {
    pub fn new() -> Self {
        Self {
            sessions: DashMap::new(),
            calls: DashMap::new(),
        }
    }

    // ---- Session methods ----

    pub fn register_session(
        &self,
        session_id: String,
        model: String,
        worker_url: String,
    ) -> SessionEntry {
        let entry = SessionEntry {
            session_id: session_id.clone(),
            model,
            worker_url,
            state: ConnectionState::Pending,
            created_at: Instant::now(),
            cancel_token: CancellationToken::new(),
        };
        if let Some(old) = self.sessions.insert(session_id, entry.clone()) {
            old.cancel_token.cancel();
        }
        entry
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
            e
        })
    }

    // ---- Call methods ----

    pub fn register_call(&self, call_id: String, model: String, worker_url: String) -> CallEntry {
        let entry = CallEntry {
            call_id: call_id.clone(),
            model,
            worker_url,
            state: ConnectionState::Pending,
            created_at: Instant::now(),
            cancel_token: CancellationToken::new(),
        };
        if let Some(old) = self.calls.insert(call_id, entry.clone()) {
            old.cancel_token.cancel();
        }
        entry
    }

    pub fn get_call(&self, call_id: &str) -> Option<CallEntry> {
        self.calls.get(call_id).map(|e| e.clone())
    }

    pub fn set_call_state(&self, call_id: &str, state: ConnectionState) {
        if let Some(mut entry) = self.calls.get_mut(call_id) {
            entry.state = state;
        }
    }

    pub fn remove_call(&self, call_id: &str) -> Option<CallEntry> {
        self.calls.remove(call_id).map(|(_, e)| {
            e.cancel_token.cancel();
            e
        })
    }

    // ---- Reaper ----

    /// Start a background task that evicts stale entries.
    ///
    /// `pending_max_age` applies to `Pending` sessions (upgrade not completed).
    /// `max_age` applies to `Disconnected` sessions (connection closed but not
    /// yet removed). Active (`Connected`) sessions are never reaped.
    ///
    /// Returns a `CancellationToken` that stops the reaper when cancelled.
    pub fn start_reaper(
        self: &Arc<Self>,
        max_age: Duration,
        pending_max_age: Duration,
        interval: Duration,
    ) -> CancellationToken {
        let shutdown = CancellationToken::new();
        let token = shutdown.clone();
        let registry = Arc::clone(self);
        #[expect(
            clippy::disallowed_methods,
            reason = "reaper task cancelled via returned token"
        )]
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            loop {
                tokio::select! {
                    _ = tick.tick() => {}
                    () = shutdown.cancelled() => {
                        info!("Realtime registry reaper shutting down");
                        return;
                    }
                }
                let now = Instant::now();

                let is_stale = |state: ConnectionState, age: Duration| -> bool {
                    match state {
                        ConnectionState::Connected => false,
                        ConnectionState::Pending => age > pending_max_age,
                        ConnectionState::Disconnected => age > max_age,
                    }
                };

                let stale_session_ids: Vec<String> = registry
                    .sessions
                    .iter()
                    .filter(|e| is_stale(e.state, now.duration_since(e.created_at)))
                    .map(|e| e.session_id.clone())
                    .collect();

                let mut sessions_reaped = 0usize;
                for id in &stale_session_ids {
                    if let Some((_, entry)) = registry.sessions.remove_if(id, |_, e| {
                        is_stale(e.state, now.duration_since(e.created_at))
                    }) {
                        entry.cancel_token.cancel();
                        sessions_reaped += 1;
                    }
                }

                let stale_call_ids: Vec<String> = registry
                    .calls
                    .iter()
                    .filter(|e| is_stale(e.state, now.duration_since(e.created_at)))
                    .map(|e| e.call_id.clone())
                    .collect();

                let mut calls_reaped = 0usize;
                for id in &stale_call_ids {
                    if let Some((_, entry)) = registry.calls.remove_if(id, |_, e| {
                        is_stale(e.state, now.duration_since(e.created_at))
                    }) {
                        entry.cancel_token.cancel();
                        calls_reaped += 1;
                    }
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
        self.sessions.len()
    }

    pub fn call_count(&self) -> usize {
        self.calls.len()
    }
}

impl Default for RealtimeRegistry {
    fn default() -> Self {
        Self::new()
    }
}
