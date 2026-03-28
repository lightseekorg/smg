//! HTTP server for the Prometheus metrics endpoint (port 29000).
//! Later PRs add `/ws/metrics` to this same server.

use std::net::SocketAddr;

use axum::{extract::State, response::IntoResponse, routing::get, Router};
use metrics_exporter_prometheus::PrometheusHandle;
use tokio::task::JoinHandle;
use tracing::{error, info};

#[derive(Clone)]
struct MetricsState {
    handle: PrometheusHandle,
}

async fn prometheus_handler(State(state): State<MetricsState>) -> impl IntoResponse {
    state.handle.render()
}

pub fn start_metrics_server(
    handle: PrometheusHandle,
    host: String,
    port: u16,
) -> JoinHandle<()> {
    let state = MetricsState { handle };
    let app = Router::new()
        .route("/metrics", get(prometheus_handler))
        .with_state(state);

    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .unwrap_or_else(|_| SocketAddr::from(([0, 0, 0, 0], port)));

    #[expect(
        clippy::disallowed_methods,
        reason = "metrics server runs for the lifetime of the process"
    )]
    tokio::spawn(async move {
        info!("Metrics server listening on {addr}");
        let listener = match tokio::net::TcpListener::bind(addr).await {
            Ok(l) => l,
            Err(e) => {
                error!("Failed to bind metrics server on {addr}: {e}");
                return;
            }
        };
        if let Err(e) = axum::serve(listener, app).await {
            error!("Metrics server error: {e}");
        }
    })
}
