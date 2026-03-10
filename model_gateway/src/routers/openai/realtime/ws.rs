//! WebSocket transport helpers for `/v1/realtime`.
//!
//! The handler logic lives in `OpenAIRouter::route_realtime_ws`.
//! This module provides shared types and URL construction.

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct RealtimeQueryParams {
    pub model: Option<String>,
}

/// Build the upstream WebSocket URL for the realtime endpoint.
///
/// Worker URLs use `http(s)://` but tungstenite requires `ws(s)://`,
/// e.g. `https://api.openai.com` → `wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview`
pub(crate) fn build_upstream_ws_url(worker_url: &str, model: &str) -> String {
    let base = worker_url.trim_end_matches('/');
    let ws_base = if let Some(rest) = base.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = base.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        base.to_string()
    };
    let query = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("model", model)
        .finish();
    format!("{ws_base}/v1/realtime?{query}")
}
