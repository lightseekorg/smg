//! WebSocket transport for the Responses API.
//!
//! Implements `GET /v1/responses` WebSocket handling at the router layer. Each
//! connection owns a single-entry response cache and allows one in-flight
//! `response.create` at a time. Connections are capped at a configurable
//! lifetime (default 60 minutes).

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, OnceLock,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use axum::{
    extract::ws::{close_code, CloseFrame, Message, WebSocket},
    http::HeaderMap,
};
use futures_util::{future::FutureExt as _, SinkExt, StreamExt};
use openai_protocol::responses::{
    ResponseInputOutputItem, ResponseStatus, ResponsesRequest, ResponsesResponse,
};
use serde_json::{json, Value};
use tokio::sync::{mpsc, Mutex, Notify};
use tracing::{debug, info, warn};

const DEFAULT_WS_SESSION_LIFETIME: Duration = Duration::from_secs(60 * 60);
/// Bounded outbound channel capacity. Large enough to absorb bursts from the
/// LLM streaming pipeline while still providing backpressure if a client falls
/// behind.
const OUTBOUND_CHANNEL_CAPACITY: usize = 256;
const ACTIVE_REQUEST_HANDOFF_TIMEOUT: Duration = Duration::from_millis(50);

fn ws_writer_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("SMG_DEBUG_WS_WRITE_TIMING").is_some())
}

fn ws_message_event_type(message: &Message) -> String {
    match message {
        Message::Text(text) => serde_json::from_str::<Value>(text)
            .ok()
            .and_then(|value| {
                value
                    .get("type")
                    .and_then(|value| value.as_str())
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "text".to_string()),
        Message::Binary(_) => "binary".to_string(),
        Message::Ping(_) => "ping".to_string(),
        Message::Pong(_) => "pong".to_string(),
        Message::Close(_) => "close".to_string(),
    }
}

fn unix_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

/// A connection-local cached response and the input items that produced it.
///
/// The WebSocket session keeps the most recent successful response so a
/// follow-up turn referencing it via `previous_response_id` can be served
/// without a durable-storage round trip.
#[derive(Clone, Debug)]
pub struct CachedWsResponse {
    pub response: ResponsesResponse,
    pub input_items: Vec<ResponseInputOutputItem>,
}

impl CachedWsResponse {
    /// Flatten the cached input items + output items into a single conversation
    /// item list for the next turn.
    pub fn to_conversation_items(&self) -> Vec<ResponseInputOutputItem> {
        let mut items = self.input_items.clone();

        for output_item in &self.response.output {
            let Ok(value) = serde_json::to_value(output_item) else {
                warn!("failed to serialize output item for conversation cache");
                continue;
            };
            let Ok(item) = serde_json::from_value::<ResponseInputOutputItem>(value) else {
                warn!(
                    "failed to deserialize output item into ResponseInputOutputItem for conversation cache"
                );
                continue;
            };
            items.push(item);
        }

        items
    }
}

/// Structured client error surfaced over the WebSocket as an `error` event.
#[derive(Clone, Debug)]
pub struct WsClientError {
    pub code: String,
    pub message: String,
    pub status: u16,
    pub error_type: String,
    pub param: Option<String>,
}

impl WsClientError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            status: 400,
            error_type: "invalid_request_error".to_string(),
            param: None,
        }
    }

    pub fn with_status(mut self, status: u16) -> Self {
        self.status = status;
        self
    }

    pub fn with_type(mut self, error_type: impl Into<String>) -> Self {
        self.error_type = error_type.into();
        self
    }

    pub fn with_param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }
}

/// Per-`response.create` options carried alongside the request body.
#[derive(Clone, Debug, Default)]
pub struct WsResponseCreateOptions {
    /// When `Some(false)`, perform a warmup-only response (no generation).
    pub generate: Option<bool>,
}

/// Transport-agnostic backend that executes a `response.create` for a session.
#[async_trait]
pub trait WsResponsesExecutor: Send + Sync {
    /// Run one `response.create`, streaming its events to `outbound_tx`.
    ///
    /// Implementations stream the response events themselves (the session layer
    /// does not forward anything on success) and return the materialized
    /// [`CachedWsResponse`] so the session can answer a follow-up turn's
    /// `previous_response_id` from its connection-local cache.
    ///
    /// - `cached_response` is the session's most recent successful response, so
    ///   a continuation can be served without a durable-storage round trip.
    /// - An `Ok` whose status is `Failed` is NOT cached by the session (a failed
    ///   turn must not become the parent of the next one).
    /// - An `Err` is surfaced to the client as an `error` event; the session
    ///   keeps its prior cache unless the failed turn referenced it.
    async fn execute_response_create(
        &self,
        headers: HeaderMap,
        request: ResponsesRequest,
        options: WsResponseCreateOptions,
        cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::Sender<Message>,
    ) -> Result<CachedWsResponse, WsClientError>;
}

struct WsSessionState {
    active_request: bool,
    cached_response: Option<CachedWsResponse>,
    /// Signalled when `active_request` transitions from true -> false.
    request_done: Arc<Notify>,
}

impl Default for WsSessionState {
    fn default() -> Self {
        Self {
            active_request: false,
            cached_response: None,
            request_done: Arc::new(Notify::new()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct WsRuntimeConfig {
    pub max_session_lifetime: Duration,
}

impl Default for WsRuntimeConfig {
    fn default() -> Self {
        Self {
            max_session_lifetime: DEFAULT_WS_SESSION_LIFETIME,
        }
    }
}

fn format_duration_label(duration: Duration) -> String {
    fn pluralize(value: u128, unit: &str) -> String {
        if unit == "ms" {
            return format!("{value} ms");
        }

        if value == 1 {
            format!("1 {unit}")
        } else {
            format!("{value} {unit}s")
        }
    }

    let millis = duration.as_millis();
    let seconds = duration.as_secs();

    if millis == 0 {
        return "0 ms".to_string();
    }

    if millis < 1_000 {
        return pluralize(millis, "ms");
    }

    if seconds < 60 {
        return pluralize(u128::from(seconds), "second");
    }

    if seconds < 3_600 && seconds.is_multiple_of(60) {
        return pluralize(u128::from(seconds / 60), "minute");
    }

    if seconds.is_multiple_of(3_600) {
        return pluralize(u128::from(seconds / 3_600), "hour");
    }

    format!("{:.1} seconds", duration.as_secs_f64())
}

fn session_lifetime_message(duration: Duration) -> String {
    format!(
        "Responses websocket connection limit reached ({}). Create a new websocket connection to continue.",
        format_duration_label(duration)
    )
}

#[derive(Debug)]
struct ParsedClientEvent {
    event_type: String,
    event_id: Option<String>,
    request: Option<ResponsesRequest>,
    options: WsResponseCreateOptions,
}

pub(crate) async fn serve_responses_ws(
    socket: WebSocket,
    headers: HeaderMap,
    executor: Arc<dyn WsResponsesExecutor>,
) {
    serve_responses_ws_with_config(socket, headers, executor, WsRuntimeConfig::default()).await;
}

pub async fn serve_responses_ws_with_config(
    socket: WebSocket,
    headers: HeaderMap,
    executor: Arc<dyn WsResponsesExecutor>,
    runtime_config: WsRuntimeConfig,
) {
    let (mut sink, mut stream) = socket.split();
    let (outbound_tx, mut outbound_rx) = mpsc::channel::<Message>(OUTBOUND_CHANNEL_CAPACITY);
    let session = Arc::new(Mutex::new(WsSessionState::default()));
    let closing = Arc::new(AtomicBool::new(false));
    // Signalled by the lifetime-cap timeout so the reader loop can stop parking
    // on `stream.next()` (which only wakes on client traffic) and tear the
    // connection down even when an idle/uncooperative client never echoes Close.
    let shutdown = Arc::new(Notify::new());

    #[expect(
        clippy::disallowed_methods,
        reason = "writer task is awaited/aborted before this fn returns; safe to drop on shutdown"
    )]
    let writer = tokio::spawn(async move {
        let session_started_at = Instant::now();
        while let Some(message) = outbound_rx.recv().await {
            let event_type = ws_writer_timing_enabled().then(|| ws_message_event_type(&message));
            let payload_len = match &message {
                Message::Text(text) => text.len(),
                Message::Binary(payload) => payload.len(),
                Message::Ping(payload) | Message::Pong(payload) => payload.len(),
                Message::Close(_) => 0,
            };
            // RFC 6455: the Close frame is the final frame on the connection; no
            // data frames may follow it. Stop the writer once Close is flushed so
            // a late-queued event (e.g. an in-flight stream racing the
            // lifetime-cap Close) is not written after the close handshake.
            let is_close = matches!(message, Message::Close(_));

            if sink.send(message).await.is_err() {
                break;
            }

            if let Some(event_type) = event_type {
                info!(
                    wall_time_ms = unix_timestamp_ms(),
                    session_elapsed_ms = session_started_at.elapsed().as_secs_f64() * 1000.0,
                    event_type,
                    payload_len,
                    "responses websocket writer flushed frame"
                );
            }

            if is_close {
                break;
            }
        }
    });

    let session_timeout = {
        let outbound_tx = outbound_tx.clone();
        let closing = closing.clone();
        let shutdown = shutdown.clone();
        let max_session_lifetime = runtime_config.max_session_lifetime;
        #[expect(
            clippy::disallowed_methods,
            reason = "timeout task is aborted before this fn returns; safe to drop on shutdown"
        )]
        tokio::spawn(async move {
            if max_session_lifetime.is_zero() {
                return;
            }

            tokio::time::sleep(max_session_lifetime).await;
            closing.store(true, Ordering::Release);
            let message = session_lifetime_message(max_session_lifetime);
            send_client_error_json(
                &outbound_tx,
                &WsClientError::new("websocket_connection_limit_reached", &message)
                    .with_type("invalid_request_error"),
                None,
            )
            .await;
            let _ = outbound_tx
                .send(Message::Close(Some(CloseFrame {
                    code: close_code::NORMAL,
                    reason: message.into(),
                })))
                .await;
            // Wake the reader loop so it tears down even if the client never
            // sends another frame (the lifetime cap is otherwise unenforced for
            // an idle connection parked on `stream.next()`).
            shutdown.notify_waiters();
        })
    };

    // Track the current executor task so we can abort it on disconnect.
    let mut executor_handle: Option<tokio::task::JoinHandle<()>> = None;

    // Arm the shutdown waiter before the loop so a lifetime-cap signal that fires
    // between iterations cannot be missed (same `Notified::enable()` idiom as
    // `acquire_request_slot`).
    let shutdown_signal = shutdown.notified();
    tokio::pin!(shutdown_signal);
    shutdown_signal.as_mut().enable();

    loop {
        let message_result = tokio::select! {
            biased;
            () = shutdown_signal.as_mut() => {
                debug!("responses websocket reader stopping: session lifetime reached");
                break;
            }
            message_result = stream.next() => message_result,
        };

        let Some(message_result) = message_result else {
            break;
        };
        let message = match message_result {
            Ok(message) => message,
            Err(err) => {
                debug!("responses websocket receive error: {err}");
                break;
            }
        };

        match message {
            Message::Text(text) => {
                // Closing flag is checked BEFORE accepting a Text frame so a
                // session that has hit its lifetime limit rejects new work.
                if closing.load(Ordering::Acquire) {
                    break;
                }
                // Only an accepted `response.create` yields a handle; error /
                // other events return `None`. Don't overwrite a tracked in-flight
                // handle with `None` (a later error frame must not drop it).
                if let Some(handle) = handle_text_event(
                    text.as_ref(),
                    &headers,
                    executor.clone(),
                    session.clone(),
                    outbound_tx.clone(),
                )
                .await
                {
                    executor_handle = Some(handle);
                }
            }
            Message::Binary(_) => {
                send_error_json(
                    &outbound_tx,
                    "unsupported_message_type",
                    "Binary WebSocket messages are not supported on /v1/responses.",
                    None,
                )
                .await;
            }
            Message::Ping(payload) => {
                let _ = outbound_tx.send(Message::Pong(payload)).await;
            }
            Message::Pong(_) => {}
            Message::Close(_) => break,
        }
    }

    // Client disconnected — abort any in-flight executor task to avoid wasted compute.
    if let Some(handle) = executor_handle.take() {
        handle.abort();
    }
    session_timeout.abort();
    drop(outbound_tx);
    let _ = writer.await;
}

/// Handle an incoming text event. Returns a [`JoinHandle`] when an executor
/// task is spawned so the caller can abort it on disconnect.
async fn handle_text_event(
    payload: &str,
    headers: &HeaderMap,
    executor: Arc<dyn WsResponsesExecutor>,
    session: Arc<Mutex<WsSessionState>>,
    outbound_tx: mpsc::Sender<Message>,
) -> Option<tokio::task::JoinHandle<()>> {
    let raw_event = match parse_client_event(payload) {
        Ok(raw_event) => raw_event,
        Err(err) => {
            send_client_error_json(&outbound_tx, &err, None).await;
            return None;
        }
    };

    match raw_event.event_type.as_str() {
        "response.create" => {
            let Some(request) = raw_event.request else {
                send_error_json(
                    &outbound_tx,
                    "missing_response",
                    "The `response.create` event requires a valid Responses request body.",
                    raw_event.event_id.as_deref(),
                )
                .await;
                return None;
            };

            if !acquire_request_slot(&session).await {
                send_error_json(
                    &outbound_tx,
                    "concurrent_response_create",
                    "Only one in-flight `response.create` is allowed per connection.",
                    raw_event.event_id.as_deref(),
                )
                .await;
                return None;
            }
            let session_guard = session.lock().await;
            let cached_response = session_guard.cached_response.clone();
            drop(session_guard);

            let event_id = raw_event.event_id.clone();
            let options = raw_event.options.clone();
            let referenced_previous_response_id = request.previous_response_id.clone();
            let request_model = request.model.clone();
            let request_store = request.store;
            let request_started_at = Instant::now();
            debug!(
                event_id = event_id.as_deref().unwrap_or(""),
                model = %request_model,
                store = request_store,
                has_previous_response = referenced_previous_response_id.is_some(),
                generate = options.generate.unwrap_or(true),
                "accepted websocket response.create request"
            );
            let headers = headers.clone();
            let session_clone = session.clone();
            let outbound_clone = outbound_tx.clone();
            #[expect(
                clippy::disallowed_methods,
                reason = "handle is tracked by the caller and aborted on client disconnect"
            )]
            let handle = tokio::spawn(async move {
                // Catch a panic in the executor so it routes through the error
                // path below: a panicking turn must release the in-flight slot
                // (and evict a referenced cache entry) instead of wedging the
                // connection so every future `response.create` rejects with
                // `concurrent_response_create`.
                let result = std::panic::AssertUnwindSafe(executor.execute_response_create(
                    headers,
                    request,
                    options,
                    cached_response,
                    outbound_clone.clone(),
                ))
                .catch_unwind()
                .await
                .unwrap_or_else(|_panic| {
                    warn!(
                        event_id = event_id.as_deref().unwrap_or(""),
                        "websocket response.create handler panicked; releasing slot and surfacing error"
                    );
                    Err(WsClientError::new(
                        "internal_error",
                        "Internal error while processing response.create.",
                    )
                    .with_status(500)
                    .with_type("server_error"))
                });

                let mut session_guard = session_clone.lock().await;
                // Update cache *before* clearing active_request to prevent a
                // TOCTOU race where a new request reads stale cache.
                match result {
                    Ok(cached_response) => {
                        debug!(
                            event_id = event_id.as_deref().unwrap_or(""),
                            response_id = %cached_response.response.id,
                            status = ?cached_response.response.status,
                            elapsed_ms = request_started_at.elapsed().as_secs_f64() * 1000.0,
                            "completed websocket response.create request"
                        );
                        session_guard.cached_response = (cached_response.response.status
                            != ResponseStatus::Failed)
                            .then_some(cached_response);
                    }
                    Err(err) => {
                        let should_evict_cached_response = referenced_previous_response_id
                            .as_deref()
                            .is_some_and(|previous_id| {
                                session_guard
                                    .cached_response
                                    .as_ref()
                                    .map(|cached| cached.response.id == previous_id)
                                    .unwrap_or(false)
                            });
                        if should_evict_cached_response {
                            session_guard.cached_response = None;
                        }
                        // Release lock before sending error to avoid holding it
                        // while the bounded channel potentially blocks. Notify is
                        // signalled AFTER the lock is dropped.
                        let notify = session_guard.request_done.clone();
                        session_guard.active_request = false;
                        drop(session_guard);
                        notify.notify_waiters();
                        debug!(
                            event_id = event_id.as_deref().unwrap_or(""),
                            error_code = %err.code,
                            elapsed_ms = request_started_at.elapsed().as_secs_f64() * 1000.0,
                            "websocket response.create request failed"
                        );
                        send_client_error_json(&outbound_clone, &err, event_id.as_deref()).await;
                        return;
                    }
                }
                let notify = session_guard.request_done.clone();
                session_guard.active_request = false;
                drop(session_guard);
                notify.notify_waiters();
            });
            Some(handle)
        }
        other => {
            send_error_json(
                &outbound_tx,
                "unsupported_event",
                format!("Unsupported WebSocket client event type: {other}"),
                raw_event.event_id.as_deref(),
            )
            .await;
            None
        }
    }
}

async fn acquire_request_slot(session: &Arc<Mutex<WsSessionState>>) -> bool {
    // Allow one in-flight `response.create` per connection. A client that just
    // received `response.completed` may send the next turn before the prior
    // request's task has cleared `active_request` (the slot is released only
    // after the response is materialized/cached/persisted). Bridge that handoff
    // with a bounded wait on `request_done`.
    //
    // `Notify::notify_waiters()` stores NO permit, so a completion that fires
    // before this task is registered as a waiter is lost. We therefore arm the
    // waiter via `Notified::enable()` — which registers without awaiting —
    // *before* re-reading `active_request` each iteration. Any `notify_waiters()`
    // from that point is guaranteed to wake us, so the handoff resolves on the
    // real completion (sub-millisecond) instead of stalling for the timeout in
    // the missed-notify window. The bounded `timeout` is retained only as a
    // safety re-check (e.g. a fully lost wakeup); a genuinely concurrent second
    // create still rejects once the deadline passes.
    let deadline = tokio::time::Instant::now() + ACTIVE_REQUEST_HANDOFF_TIMEOUT;

    // The completion `Notify` is connection-stable; clone the handle once.
    let notify = {
        let guard = session.lock().await;
        guard.request_done.clone()
    };
    let notified = notify.notified();
    tokio::pin!(notified);
    loop {
        // Arm the waiter BEFORE checking the slot — closes the missed-notify
        // window: a completion firing after we drop the lock cannot be lost.
        notified.as_mut().enable();

        {
            let mut guard = session.lock().await;
            if !guard.active_request {
                guard.active_request = true;
                return true;
            }
        }

        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return false;
        }
        // Wake on the completion notification (fast path) or after `remaining`
        // (safety net); either way re-check `active_request` under the lock.
        // Re-arm the future for the next iteration.
        let _ = tokio::time::timeout(remaining, notified.as_mut()).await;
        notified.set(notify.notified());
    }
}

fn parse_client_event(payload: &str) -> Result<ParsedClientEvent, WsClientError> {
    let event_value = serde_json::from_str::<Value>(payload).map_err(|err| {
        WsClientError::new(
            "invalid_json",
            format!("Failed to parse WebSocket client event JSON: {err}"),
        )
    })?;

    let mut event_object = match event_value {
        Value::Object(event_object) => event_object,
        _ => {
            return Err(WsClientError::new(
                "invalid_json",
                "WebSocket client event JSON must be an object.",
            ))
        }
    };

    let event_type = take_string_field(&mut event_object, "type").ok_or_else(|| {
        WsClientError::new(
            "invalid_json",
            "WebSocket client event JSON must include a string `type` field.",
        )
    })?;
    let event_id = take_string_field(&mut event_object, "event_id");

    let (request, options) = match event_type.as_str() {
        // The request body may arrive either nested under a `response` key or
        // inlined at the top level (with `type`/`event_id` already removed).
        "response.create" => {
            let mut request_value = match event_object.remove("response") {
                Some(request_value) => request_value,
                // An empty `response.create` (no `response`, no inlined fields)
                // is a warmup-style probe with no request body.
                None if event_object.is_empty() => {
                    return Ok(ParsedClientEvent {
                        event_type,
                        event_id,
                        request: None,
                        options: WsResponseCreateOptions::default(),
                    });
                }
                None => Value::Object(event_object),
            };
            let options = extract_request_options(&mut request_value)?;
            (Some(parse_response_create_request(request_value)?), options)
        }
        _ => (None, WsResponseCreateOptions::default()),
    };

    Ok(ParsedClientEvent {
        event_type,
        event_id,
        request,
        options,
    })
}

/// Deserialize the (options-stripped) `response.create` body into a
/// [`ResponsesRequest`], mapping parse failures to a client error.
fn parse_response_create_request(request_value: Value) -> Result<ResponsesRequest, WsClientError> {
    serde_json::from_value::<ResponsesRequest>(request_value).map_err(|err| {
        WsClientError::new(
            "invalid_request",
            format!("Failed to parse `response.create` payload: {err}"),
        )
    })
}

fn extract_request_options(
    request_value: &mut Value,
) -> Result<WsResponseCreateOptions, WsClientError> {
    let request_object = request_value.as_object_mut().ok_or_else(|| {
        WsClientError::new(
            "invalid_request",
            "The `response.create` payload must be a JSON object.",
        )
    })?;

    Ok(WsResponseCreateOptions {
        generate: take_bool_field(request_object, "generate"),
    })
}

fn take_string_field(object: &mut serde_json::Map<String, Value>, key: &str) -> Option<String> {
    object
        .remove(key)
        .and_then(|value| value.as_str().map(str::to_owned))
}

fn take_bool_field(object: &mut serde_json::Map<String, Value>, key: &str) -> Option<bool> {
    object.remove(key).and_then(|value| value.as_bool())
}

pub(crate) async fn send_error_json(
    outbound_tx: &mpsc::Sender<Message>,
    code: &str,
    message: impl Into<String>,
    event_id: Option<&str>,
) {
    send_client_error_json(outbound_tx, &WsClientError::new(code, message), event_id).await;
}

pub(crate) async fn send_client_error_json(
    outbound_tx: &mpsc::Sender<Message>,
    error: &WsClientError,
    event_id: Option<&str>,
) {
    let mut error_json = json!({
        "type": "error",
        "status": error.status,
        "error": {
            "type": error.error_type,
            "code": error.code,
            "message": error.message,
        },
        "code": error.code,
        "message": error.message,
    });

    if let Some(param) = &error.param {
        error_json["error"]["param"] = Value::String(param.clone());
    }

    if let Some(event_id) = event_id {
        error_json["event_id"] = Value::String(event_id.to_string());
    }

    if outbound_tx
        .send(Message::Text(error_json.to_string().into()))
        .await
        .is_err()
    {
        warn!("responses websocket client disconnected before error delivery");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_duration_label_renders_units() {
        assert_eq!(format_duration_label(Duration::from_millis(0)), "0 ms");
        assert_eq!(format_duration_label(Duration::from_millis(250)), "250 ms");
        assert_eq!(format_duration_label(Duration::from_secs(1)), "1 second");
        assert_eq!(format_duration_label(Duration::from_secs(2)), "2 seconds");
        assert_eq!(format_duration_label(Duration::from_secs(60)), "1 minute");
        assert_eq!(format_duration_label(Duration::from_secs(3_600)), "1 hour");
    }

    /// The first `response.create` always gets the slot.
    #[tokio::test]
    async fn acquire_request_slot_grants_first_caller() {
        let session = Arc::new(Mutex::new(WsSessionState::default()));
        assert!(acquire_request_slot(&session).await);
        assert!(session.lock().await.active_request);
    }

    /// A genuinely concurrent second `response.create` (the prior turn never
    /// releases the slot) must reject once the handoff deadline elapses, rather
    /// than waiting forever.
    #[tokio::test]
    async fn acquire_request_slot_rejects_genuinely_concurrent_second_caller() {
        let session = Arc::new(Mutex::new(WsSessionState::default()));
        assert!(acquire_request_slot(&session).await);

        // Slot is held and never released; the second acquire must time out and
        // reject (the caller surfaces `concurrent_response_create`).
        let start = tokio::time::Instant::now();
        assert!(!acquire_request_slot(&session).await);
        assert!(
            start.elapsed() >= ACTIVE_REQUEST_HANDOFF_TIMEOUT,
            "rejection must only happen after the bounded handoff wait elapses"
        );
        assert!(
            session.lock().await.active_request,
            "a rejected acquire must not clear the in-flight slot"
        );
    }

    /// Missed-notify handoff: `Notify::notify_waiters()` stores no permit, so a
    /// completion firing in the window between the acquirer dropping the lock
    /// and registering `.notified()` is lost. The acquire loop must still bridge
    /// the handoff — re-checking `active_request` under the lock after the
    /// bounded wait — and grant the slot, NOT spuriously reject with
    /// `concurrent_response_create`. Regression for the slot-handoff race
    /// observed under slower history backends (e.g. redis).
    ///
    /// The slot is freed WITHOUT any `notify_waiters()` call, modelling a fully
    /// lost wakeup: the only thing that can rescue the second acquire is the
    /// loop's timeout-driven re-check.
    #[tokio::test]
    async fn acquire_request_slot_recovers_from_missed_notify() {
        let session = Arc::new(Mutex::new(WsSessionState::default()));
        // Prior turn holds the slot.
        assert!(acquire_request_slot(&session).await);

        // Start the second acquire while the slot is still held, so it observes
        // `active_request == true`, clones the notify, and enters the bounded
        // wait on `.notified()`.
        let acquire = {
            let session = session.clone();
            #[expect(
                clippy::disallowed_methods,
                reason = "test task is joined below before the test returns"
            )]
            tokio::spawn(async move { acquire_request_slot(&session).await })
        };

        // Let the spawned acquire reach its wait, then free the slot with NO
        // notify — the loop must recover purely via its bounded re-check.
        tokio::time::sleep(ACTIVE_REQUEST_HANDOFF_TIMEOUT / 5).await;
        session.lock().await.active_request = false;

        let granted = tokio::time::timeout(ACTIVE_REQUEST_HANDOFF_TIMEOUT * 2, acquire)
            .await
            .expect("acquire must resolve within the handoff window despite the missed notify")
            .expect("acquire task must not panic");
        assert!(
            granted,
            "missed notify must degrade to a short re-check, not a rejection"
        );
        assert!(session.lock().await.active_request);
    }
}
