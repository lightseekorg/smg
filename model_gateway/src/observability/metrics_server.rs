//! HTTP/WebSocket server for the Prometheus metrics endpoint (port 29000).
//! Serves `GET /metrics` (Prometheus) and `WS /ws/metrics` (real-time state push).

use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::{atomic::AtomicUsize, Arc},
    time::Duration,
};

use axum::{extract::State, response::IntoResponse, routing::get, Router};
use metrics_exporter_prometheus::PrometheusHandle;
use tokio::task::JoinHandle;
use tracing::{error, info};

use super::{
    metrics::UPKEEP_INTERVAL_SECS,
    metrics_ws::{handler, registry::WatchRegistry},
};

/// Default maximum concurrent WebSocket connections on the metrics endpoint.
pub const DEFAULT_MAX_WS_CONNECTIONS: usize = 32;

#[derive(Clone)]
struct MetricsState {
    handle: PrometheusHandle,
}

async fn prometheus_handler(State(state): State<MetricsState>) -> impl IntoResponse {
    (
        [(
            http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        state.handle.render(),
    )
}

async fn bind_metrics_listener(addr: SocketAddr) -> Result<tokio::net::TcpListener, String> {
    tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| format!("failed to bind metrics server on {addr}: {e}"))
}

/// Start the metrics HTTP/WS server. Binds eagerly so callers fail fast on
/// port conflicts or bad addresses.
#[expect(
    clippy::expect_used,
    reason = "startup initialization — metrics server must bind or the process cannot serve metrics"
)]
pub async fn start_metrics_server(
    handle: PrometheusHandle,
    host: String,
    port: u16,
    watch_registry: Arc<WatchRegistry>,
    max_ws_connections: usize,
) -> JoinHandle<()> {
    let ip_addr: IpAddr = host.parse().unwrap_or_else(|e| {
        error!("Failed to parse metrics host '{host}': {e}, falling back to 0.0.0.0");
        IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0))
    });
    let addr = SocketAddr::new(ip_addr, port);

    let listener = bind_metrics_listener(addr)
        .await
        .expect("metrics server bind failed");

    info!("Metrics server listening on {addr} (/metrics + /ws/metrics)");

    // Spawn upkeep task — required by install_recorder() for histogram maintenance.
    let upkeep_handle = handle.clone();
    #[expect(
        clippy::disallowed_methods,
        reason = "upkeep task runs for the lifetime of the process"
    )]
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(UPKEEP_INTERVAL_SECS)).await;
            upkeep_handle.run_upkeep();
        }
    });

    let prom_state = MetricsState { handle };
    let ws_state = handler::MetricsWsState {
        registry: watch_registry,
        max_connections: max_ws_connections,
        active_connections: Arc::new(AtomicUsize::new(0)),
    };

    let app = Router::new()
        .route("/metrics", get(prometheus_handler).with_state(prom_state))
        .route(
            "/ws/metrics",
            get(handler::ws_metrics_handler).with_state(ws_state),
        );

    #[expect(
        clippy::disallowed_methods,
        reason = "metrics server runs for the lifetime of the process"
    )]
    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            error!("Metrics server error: {e}");
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bind_metrics_listener_error_includes_addr() {
        let pre =
            tokio::net::TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0))
                .await
                .unwrap();
        let busy = pre.local_addr().unwrap();
        let err = bind_metrics_listener(busy).await.unwrap_err();
        assert!(err.contains(&busy.to_string()), "got: {err}");
        assert!(err.contains("failed to bind metrics server"), "got: {err}");
    }
}
