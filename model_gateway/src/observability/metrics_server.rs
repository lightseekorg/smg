//! HTTP server for the Prometheus metrics endpoint (port 29000).
//! Later PRs add `/ws/metrics` to this same server.

use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    time::Duration,
};

use axum::{extract::State, response::IntoResponse, routing::get, Router};
use metrics_exporter_prometheus::PrometheusHandle;
use tokio::task::JoinHandle;
use tracing::{error, info};

use super::metrics::UPKEEP_INTERVAL_SECS;

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

/// Start the metrics HTTP server. Binds eagerly so callers fail fast on
/// port conflicts or bad addresses.
#[expect(
    clippy::expect_used,
    reason = "startup initialization — metrics server must bind or the process cannot serve metrics"
)]
pub async fn start_metrics_server(
    handle: PrometheusHandle,
    host: String,
    port: u16,
) -> JoinHandle<()> {
    let ip_addr: IpAddr = host.parse().unwrap_or_else(|e| {
        error!("Failed to parse metrics host '{host}': {e}, falling back to 0.0.0.0");
        IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0))
    });
    let addr = SocketAddr::new(ip_addr, port);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind metrics server");

    info!("Metrics server listening on {addr}");

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

    let state = MetricsState { handle };
    let app = Router::new()
        .route("/metrics", get(prometheus_handler))
        .with_state(state);

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
