//! WebSocket transport handler for `/v1/realtime`.
//!
//! Server-to-server transport. Client connects with an API key,
//! session is created implicitly when the connection is established.

use std::sync::Arc;

use axum::{
    extract::{
        ws::{WebSocket, WebSocketUpgrade},
        Query, State,
    },
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use tracing::{debug, error, info, warn};

use super::{proxy, rest::select_worker};
use crate::{routers::header_utils::extract_auth_header, server::AppState};

#[derive(Debug, Deserialize)]
pub struct RealtimeQueryParams {
    pub model: Option<String>,
}

/// Handler for `GET /v1/realtime` — WebSocket upgrade.
///
/// 1. Extract model from query params, auth from headers
/// 2. Select worker via WorkerRegistry (model routing)
/// 3. Upgrade to WebSocket
/// 4. Delegate to bidirectional WS proxy
pub async fn ws_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RealtimeQueryParams>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    let model = match params.model {
        Some(m) => m,
        None => {
            warn!("Missing required 'model' query parameter");
            return (
                http::StatusCode::BAD_REQUEST,
                "Missing required 'model' query parameter",
            )
                .into_response();
        }
    };

    let worker = match select_worker(&state, &model) {
        Some(w) => w,
        None => {
            warn!(model, "No available worker for realtime model");
            return http::StatusCode::SERVICE_UNAVAILABLE.into_response();
        }
    };

    let worker_url = worker.url().to_string();

    let upstream_ws_url = build_upstream_ws_url(&worker_url, &model);

    let auth_header_value = extract_auth_header(Some(&headers), worker.api_key());
    let auth_str = match auth_header_value {
        Some(v) => match v.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                warn!("Authorization header contains invalid UTF-8 characters");
                return (
                    http::StatusCode::BAD_REQUEST,
                    "Authorization header contains invalid UTF-8 characters",
                )
                    .into_response();
            }
        },
        None => {
            error!("No authorization available for upstream realtime connection");
            return http::StatusCode::UNAUTHORIZED.into_response();
        }
    };

    let registry = Arc::clone(&state.context.realtime_registry);

    let session_id = uuid::Uuid::now_v7().to_string();
    let entry = registry.register_session(session_id.clone(), model.clone(), worker_url.clone());
    let cancel_token = entry.cancel_token.clone();

    info!(
        session_id,
        model, worker_url, "Upgrading to realtime WebSocket"
    );

    ws.on_upgrade(move |socket: WebSocket| async move {
        if let Err(e) = proxy::run_ws_proxy(
            socket,
            &upstream_ws_url,
            &auth_str,
            registry.clone(),
            session_id.clone(),
            cancel_token,
        )
        .await
        {
            error!(session_id, error = %e, "Realtime WebSocket proxy error");
        }

        // Cleanup: remove session on disconnect
        registry.remove_session(&session_id);
        debug!(session_id, "Realtime session cleaned up");
    })
}

/// Build the upstream WebSocket URL for the realtime endpoint.
///
/// Worker URLs use `http(s)://` but tungstenite requires `ws(s)://`,
/// e.g. `https://api.openai.com` → `wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview`
fn build_upstream_ws_url(worker_url: &str, model: &str) -> String {
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
