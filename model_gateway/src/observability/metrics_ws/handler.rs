//! WebSocket handler for `/ws/metrics`.
//!
//! Each connection subscribes to topics via [`WatchRegistry`], receives
//! snapshots as watch channels update, and supports re-subscription.

use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::{
    extract::{
        ws::{Message, WebSocket},
        State, WebSocketUpgrade,
    },
    http::StatusCode,
    response::{IntoResponse, Response},
};
use futures::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::sync::watch;
use tracing::{debug, warn};

use super::{
    registry::WatchRegistry,
    types::{ClientMessage, ServerMessage, SnapshotMessage, Topic},
};

/// Max inbound WS message size. Subscribe messages are ~200 bytes;
/// 4 KB is generous headroom while protecting against oversized frames.
const MAX_WS_MESSAGE_SIZE: usize = 4096;

/// Shared state for the metrics WS handler.
#[derive(Clone)]
pub struct MetricsWsState {
    pub registry: Arc<WatchRegistry>,
    pub max_connections: usize,
    pub active_connections: Arc<AtomicUsize>,
}

/// Axum handler for WS upgrade on `/ws/metrics`.
pub async fn ws_metrics_handler(
    ws: WebSocketUpgrade,
    State(state): State<MetricsWsState>,
) -> Response {
    // Atomically claim a connection slot. Roll back if over the limit.
    let prev = state.active_connections.fetch_add(1, Ordering::Relaxed);
    if prev >= state.max_connections {
        state.active_connections.fetch_sub(1, Ordering::Relaxed);
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }

    // Create guard before on_upgrade so it decrements even if the
    // HTTP 101 upgrade fails (client disconnects mid-handshake).
    let guard = ConnGuard(state.active_connections.clone());

    ws.max_message_size(MAX_WS_MESSAGE_SIZE)
        .on_upgrade(move |socket| handle_ws_connection(socket, state, guard))
}

/// Drop guard that decrements the connection counter on drop,
/// ensuring cleanup even if the session panics or upgrade fails.
struct ConnGuard(Arc<AtomicUsize>);

impl Drop for ConnGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

async fn handle_ws_connection(socket: WebSocket, state: MetricsWsState, _guard: ConnGuard) {
    run_ws_session(socket, &state).await;
}

async fn run_ws_session(socket: WebSocket, state: &MetricsWsState) {
    let (mut ws_sink, mut ws_stream) = socket.split();

    // Wait up to 2 seconds for a subscribe message; default to all topics.
    let subscribed_topics = match tokio::time::timeout(
        Duration::from_secs(2),
        wait_for_subscribe(&mut ws_stream),
    )
    .await
    {
        Ok(Some(topics)) => topics,
        _ => Topic::ALL.iter().copied().collect::<HashSet<_>>(),
    };

    debug!(
        "WS metrics client subscribed to {} topics",
        subscribed_topics.len()
    );

    let mut topics: Vec<Topic> = subscribed_topics.into_iter().collect();
    let mut rxs: Vec<watch::Receiver<Option<Value>>> = topics
        .iter()
        .map(|t| state.registry.subscribe(*t))
        .collect();

    // Send initial snapshots (skip None = no data yet)
    if send_snapshots(&topics, &mut rxs, &mut ws_sink)
        .await
        .is_err()
    {
        return;
    }

    // Ping interval for keepalive
    let mut ping_interval = tokio::time::interval(Duration::from_secs(30));
    ping_interval.tick().await; // skip first immediate tick

    loop {
        // Arm the waiter first so publishes that land while we flush
        // pending watch updates still wake this loop.
        let notified = state.registry.notified();
        tokio::pin!(notified);

        // Flush any pending changes before entering select!.
        if send_changed(&topics, &mut rxs, &mut ws_sink).await.is_err() {
            return;
        }

        tokio::select! {
            () = &mut notified => {}
            _ = ping_interval.tick() => {
                if ws_sink.send(Message::Ping(vec![].into())).await.is_err() {
                    return;
                }
            }
            msg = ws_stream.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(ClientMessage::Subscribe {
                            topics: new_topics,
                        }) = serde_json::from_str::<ClientMessage>(&text)
                        {
                            topics = new_topics.into_iter().collect();
                            rxs = topics
                                .iter()
                                .map(|t| state.registry.subscribe(*t))
                                .collect();
                            debug!(
                                "WS client re-subscribed to {} topics",
                                topics.len()
                            );
                            // Send initial snapshots for newly subscribed topics
                            if send_snapshots(&topics, &mut rxs, &mut ws_sink)
                                .await
                                .is_err()
                            {
                                return;
                            }
                        }
                    }
                    Some(Ok(Message::Pong(_))) => {}
                    Some(Ok(Message::Close(_))) | None => return,
                    _ => {}
                }
            }
        }
    }
}

/// Send snapshots for all topics that have data (used on connect and re-subscribe).
async fn send_snapshots(
    topics: &[Topic],
    rxs: &mut [watch::Receiver<Option<Value>>],
    ws_sink: &mut futures::stream::SplitSink<WebSocket, Message>,
) -> Result<(), ()> {
    for (topic, rx) in topics.iter().zip(rxs.iter_mut()) {
        let data = rx.borrow_and_update().clone();
        if let Some(data) = data {
            let msg = ServerMessage::Snapshot(SnapshotMessage::new(*topic, data));
            match serde_json::to_string(&msg) {
                Ok(text) => {
                    if ws_sink.send(Message::Text(text.into())).await.is_err() {
                        return Err(());
                    }
                }
                Err(e) => warn!("failed to serialize snapshot for {topic:?}: {e}"),
            }
        }
    }
    Ok(())
}

/// Send snapshots only for topics that have changed since last check.
async fn send_changed(
    topics: &[Topic],
    rxs: &mut [watch::Receiver<Option<Value>>],
    ws_sink: &mut futures::stream::SplitSink<WebSocket, Message>,
) -> Result<(), ()> {
    for (topic, rx) in topics.iter().zip(rxs.iter_mut()) {
        if rx.has_changed().unwrap_or(false) {
            let data = rx.borrow_and_update().clone();
            if let Some(data) = data {
                let msg = ServerMessage::Snapshot(SnapshotMessage::new(*topic, data));
                match serde_json::to_string(&msg) {
                    Ok(text) => {
                        if ws_sink.send(Message::Text(text.into())).await.is_err() {
                            return Err(());
                        }
                    }
                    Err(e) => warn!("failed to serialize snapshot for {topic:?}: {e}"),
                }
            }
        }
    }
    Ok(())
}

async fn wait_for_subscribe(
    ws_stream: &mut futures::stream::SplitStream<WebSocket>,
) -> Option<HashSet<Topic>> {
    while let Some(Ok(msg)) = ws_stream.next().await {
        if let Message::Text(text) = msg {
            if let Ok(ClientMessage::Subscribe { topics }) =
                serde_json::from_str::<ClientMessage>(&text)
            {
                return Some(topics);
            }
        }
    }
    None
}
