//! WebRTC signaling handlers for `/v1/realtime/calls`.
//!
//! SMG acts as a WebRTC relay: it terminates the client's peer connection,
//! establishes its own peer connection to upstream, and bridges data-channel
//! messages plus audio RTP packets between the two.

use std::{net::SocketAddr, sync::Arc};

use axum::{
    body::Bytes,
    extract::{Path, Query, State},
    http::{header::CONTENT_TYPE, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};

use super::{
    rest::{proxy_response, select_worker},
    webrtc_bridge::WebRtcBridge,
};
use crate::{
    core::worker::{Worker, WorkerLoadGuard},
    routers::{error, header_utils::extract_auth_header},
    server::AppState,
};

/// Default STUN server for ICE server-reflexive candidate gathering.
const DEFAULT_STUN_SERVER: &str = "stun.l.google.com:19302";

/// Resolve the default STUN server hostname to an IPv4 `SocketAddr`.
/// Filters for IPv4 since our UDP sockets bind to `0.0.0.0`.
async fn resolve_stun_server() -> Option<SocketAddr> {
    match tokio::net::lookup_host(DEFAULT_STUN_SERVER).await {
        Ok(mut addrs) => addrs.find(|a| a.is_ipv4()),
        Err(e) => {
            tracing::warn!(error = %e, "Failed to resolve STUN server");
            None
        }
    }
}

/// `POST /v1/realtime/calls` — WebRTC SDP signaling.
///
/// Supports two content types:
/// - `multipart/form-data`: Unified interface. Contains `sdp` (SDP offer) and
///   `session` (JSON session config) fields. SMG authenticates with upstream
///   using its own API key.
/// - `application/sdp`: Direct SDP flow. Body is the raw SDP offer.
///   SMG authenticates with upstream using the worker API key.
pub async fn create_call(
    State(state): State<Arc<AppState>>,
    Query(params): Query<super::ws::RealtimeQueryParams>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let content_type = headers
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    if content_type.starts_with("multipart/form-data") {
        create_call_multipart(state, headers, body, &content_type).await
    } else if content_type.starts_with("application/sdp") {
        let model = params.model.as_deref().unwrap_or("");
        create_call_sdp(state, headers, body, model).await
    } else {
        error!(
            content_type,
            "Unsupported Content-Type for /v1/realtime/calls"
        );
        error::bad_request(
            "invalid_content_type",
            "Expected Content-Type: multipart/form-data or application/sdp",
        )
    }
}

/// Unified interface: `multipart/form-data` with `sdp` + `session` fields.
///
/// SMG extracts the model from the session config, selects a worker, creates
/// a dual peer-connection bridge, and returns its own SDP answer to the client.
async fn create_call_multipart(
    state: Arc<AppState>,
    headers: HeaderMap,
    body: Bytes,
    content_type: &str,
) -> Response {
    // -- Parse multipart fields ---------------------------------------------
    let boundary = match multer::parse_boundary(content_type) {
        Ok(b) => b,
        Err(e) => {
            error!(error = %e, "Failed to parse multipart boundary");
            return error::bad_request(
                "invalid_multipart",
                "Missing or invalid multipart boundary",
            );
        }
    };

    let mut multipart = multer::Multipart::new(
        futures::stream::once(async move { Ok::<_, std::io::Error>(body) }),
        boundary,
    );

    let mut sdp_offer: Option<Vec<u8>> = None;
    let mut session_json: Option<serde_json::Value> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        match field.name() {
            Some("sdp") => {
                sdp_offer = field.bytes().await.ok().map(|b| b.to_vec());
            }
            Some("session") => {
                if let Ok(text) = field.text().await {
                    session_json = serde_json::from_str(&text).ok();
                }
            }
            _ => {}
        }
    }

    let sdp_bytes = match sdp_offer {
        Some(s) => s,
        None => return error::bad_request("missing_sdp", "multipart 'sdp' field is required"),
    };

    let sdp_str = match String::from_utf8(sdp_bytes) {
        Ok(s) => s,
        Err(_) => return error::bad_request("invalid_sdp", "SDP is not valid UTF-8"),
    };

    // -- Worker selection ---------------------------------------------------
    let model = session_json
        .as_ref()
        .and_then(|s| s.get("model"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if model.is_empty() {
        return error::bad_request("missing_model", "session.model is required");
    }

    let worker = match select_worker(&state, &model) {
        Some(w) => w,
        None => {
            error!(model, "No available worker for realtime model");
            return StatusCode::SERVICE_UNAVAILABLE.into_response();
        }
    };

    let auth = match extract_auth_header(Some(&headers), worker.api_key()) {
        Some(v) => v,
        None => return StatusCode::UNAUTHORIZED.into_response(),
    };
    let auth_str = auth.to_str().unwrap_or("").to_string();

    let _guard = WorkerLoadGuard::new(worker.clone(), Some(&headers));

    let upstream_url = format!(
        "{}/v1/realtime/calls?model={model}",
        worker.url().trim_end_matches('/')
    );

    // -- Create bridge (dual peer connections) ------------------------------
    let call_id = uuid::Uuid::now_v7().to_string();
    let cancel_token = CancellationToken::new();

    let stun_server = resolve_stun_server().await;
    info!(
        call_id,
        model,
        upstream_url,
        ?stun_server,
        "Creating WebRTC bridge (multipart)"
    );

    let (bridge, client_sdp_answer) = match WebRtcBridge::setup(
        &sdp_str,
        &upstream_url,
        &auth_str,
        session_json,
        call_id.clone(),
        cancel_token.clone(),
        &state.context.client,
        state
            .context
            .webrtc_bind_addr
            .unwrap_or_else(|| std::net::Ipv4Addr::UNSPECIFIED.into()),
        stun_server,
    )
    .await
    {
        Ok(result) => {
            worker.record_outcome(true);
            result
        }
        Err(e) => {
            error!(call_id, error = %e, "Failed to create WebRTC bridge");
            worker.record_outcome(false);
            return StatusCode::BAD_GATEWAY.into_response();
        }
    };

    // -- Register call and spawn bridge task --------------------------------
    let registry: Arc<super::RealtimeRegistry> = Arc::clone(&state.context.realtime_registry);
    let _entry = registry.register_call(call_id.clone(), model.clone(), worker.url().to_string());

    let bridge_registry: Arc<super::RealtimeRegistry> = Arc::clone(&registry);
    let bridge_call_id = call_id.clone();
    #[expect(
        clippy::disallowed_methods,
        reason = "bridge task self-terminates on disconnect/cancel"
    )]
    tokio::spawn(async move {
        Box::pin(bridge.run(bridge_registry.clone())).await;
        bridge_registry.remove_call(&bridge_call_id);
        debug!(call_id = bridge_call_id, "WebRTC bridge task completed");
    });

    debug!(
        call_id,
        model, "WebRTC bridge started, returning SDP answer"
    );

    // -- Return SMG-generated SDP answer ------------------------------------
    #[expect(
        clippy::expect_used,
        reason = "infallible: static header names and valid body"
    )]
    Response::builder()
        .status(StatusCode::CREATED)
        .header("Content-Type", "application/sdp")
        .body(axum::body::Body::from(client_sdp_answer))
        .expect("static response builder")
}

/// Direct SDP flow: `application/sdp` body is the raw SDP offer.
///
/// SMG routes to the correct upstream worker based on the `model` query
/// parameter. A dual peer-connection bridge is created and the client
/// receives SMG's SDP answer.
async fn create_call_sdp(
    state: Arc<AppState>,
    headers: HeaderMap,
    body: Bytes,
    model: &str,
) -> Response {
    if model.is_empty() {
        return error::bad_request(
            "missing_model",
            "query parameter 'model' is required for application/sdp requests",
        );
    }

    let sdp_str = match std::str::from_utf8(&body) {
        Ok(s) => s,
        Err(_) => return error::bad_request("invalid_sdp", "SDP is not valid UTF-8"),
    };

    let worker = match select_worker(&state, model) {
        Some(w) => w,
        None => {
            error!(model, "No available worker for realtime model");
            return StatusCode::SERVICE_UNAVAILABLE.into_response();
        }
    };

    let auth = match extract_auth_header(Some(&headers), worker.api_key()) {
        Some(v) => v,
        None => return StatusCode::UNAUTHORIZED.into_response(),
    };
    let auth_str = auth.to_str().unwrap_or("").to_string();

    let _guard = WorkerLoadGuard::new(worker.clone(), Some(&headers));

    let upstream_url = format!(
        "{}/v1/realtime/calls?model={model}",
        worker.url().trim_end_matches('/')
    );

    // -- Create bridge ------------------------------------------------------
    let call_id = uuid::Uuid::now_v7().to_string();
    let cancel_token = CancellationToken::new();

    let stun_server = resolve_stun_server().await;
    info!(
        call_id,
        model,
        upstream_url,
        ?stun_server,
        "Creating WebRTC bridge (direct SDP)"
    );

    let (bridge, client_sdp_answer) = match WebRtcBridge::setup(
        sdp_str,
        &upstream_url,
        &auth_str,
        None, // no session config in direct SDP path
        call_id.clone(),
        cancel_token.clone(),
        &state.context.client,
        state
            .context
            .webrtc_bind_addr
            .unwrap_or_else(|| std::net::Ipv4Addr::UNSPECIFIED.into()),
        stun_server,
    )
    .await
    {
        Ok(result) => {
            worker.record_outcome(true);
            result
        }
        Err(e) => {
            error!(call_id, model, error = %e, "Failed to create WebRTC bridge (direct SDP)");
            worker.record_outcome(false);
            return StatusCode::BAD_GATEWAY.into_response();
        }
    };

    // -- Register call and spawn bridge task --------------------------------
    let registry: Arc<super::RealtimeRegistry> = Arc::clone(&state.context.realtime_registry);
    let _entry =
        registry.register_call(call_id.clone(), model.to_string(), worker.url().to_string());

    let bridge_registry: Arc<super::RealtimeRegistry> = Arc::clone(&registry);
    let bridge_call_id = call_id.clone();
    #[expect(
        clippy::disallowed_methods,
        reason = "bridge task self-terminates on disconnect/cancel"
    )]
    tokio::spawn(async move {
        Box::pin(bridge.run(bridge_registry.clone())).await;
        bridge_registry.remove_call(&bridge_call_id);
        debug!(
            call_id = bridge_call_id,
            "WebRTC bridge task completed (direct SDP)"
        );
    });

    debug!(call_id, model, "WebRTC bridge started (direct SDP)");

    #[expect(
        clippy::expect_used,
        reason = "infallible: static header names and valid body"
    )]
    Response::builder()
        .status(StatusCode::CREATED)
        .header("Content-Type", "application/sdp")
        .body(axum::body::Body::from(client_sdp_answer))
        .expect("static response builder")
}

/// `POST /v1/realtime/calls/{call_id}/hangup` — Terminate a WebRTC call.
///
/// Cancels the bridge relay task (which tears down both peer connections)
/// and optionally forwards the hangup to upstream.
pub async fn hangup_call(
    State(state): State<Arc<AppState>>,
    Path(call_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let registry: Arc<super::RealtimeRegistry> = Arc::clone(&state.context.realtime_registry);

    // Look up the call to find the worker URL
    let call = registry.get_call(&call_id);

    let (worker_url, model) = match &call {
        Some(entry) => (entry.worker_url.clone(), entry.model.clone()),
        None => {
            // Call not in our registry — try to forward anyway using any available worker
            match select_any_external_worker(&state) {
                Some(w) => (w.url().to_string(), String::new()),
                None => return StatusCode::SERVICE_UNAVAILABLE.into_response(),
            }
        }
    };

    let worker: Arc<dyn Worker> = if model.is_empty() {
        match select_any_external_worker(&state) {
            Some(w) => w,
            None => return StatusCode::SERVICE_UNAVAILABLE.into_response(),
        }
    } else {
        match select_worker(&state, &model) {
            Some(w) => w,
            None => return StatusCode::SERVICE_UNAVAILABLE.into_response(),
        }
    };

    let auth = match extract_auth_header(Some(&headers), worker.api_key()) {
        Some(v) => v,
        None => return StatusCode::UNAUTHORIZED.into_response(),
    };

    let _guard = WorkerLoadGuard::new(worker.clone(), Some(&headers));

    // Cancel the bridge (tears down both peer connections)
    if let Some(entry) = registry.remove_call(&call_id) {
        entry.cancel_token.cancel();
        debug!(call_id, "Cancelled WebRTC bridge via hangup");
    }

    // Also forward hangup to upstream so it cleans up its side
    let upstream_url = format!(
        "{}/v1/realtime/calls/{}/hangup",
        worker_url.trim_end_matches('/'),
        call_id
    );

    info!(call_id, upstream_url, "Forwarding WebRTC hangup");

    let result = state
        .context
        .client
        .post(&upstream_url)
        .header("Authorization", &auth)
        .send()
        .await;

    match result {
        Ok(resp) => {
            let success = resp.status().is_success();
            let response = proxy_response(resp).await;
            worker.record_outcome(success);
            response
        }
        Err(e) => {
            error!(call_id, error = %e, "Failed to forward WebRTC hangup");
            worker.record_outcome(false);
            StatusCode::BAD_GATEWAY.into_response()
        }
    }
}

/// Select any healthy external worker (fallback for `hangup_call`
/// when the call is not in the registry).
fn select_any_external_worker(state: &AppState) -> Option<Arc<dyn Worker>> {
    use crate::core::worker::RuntimeType;
    state
        .context
        .worker_registry
        .get_workers_filtered(None, None, None, Some(RuntimeType::External), true)
        .into_iter()
        .filter(|w| w.circuit_breaker().can_execute())
        .min_by_key(|w| w.load())
}
