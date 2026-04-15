//! HTTP server for the Prometheus metrics endpoint (port 29000).
//!
//! The `/metrics` endpoint returns both SMG's own metrics (`smg_*`) and engine
//! metrics (`vllm_*`, `sglang_*`, `nv_trt_*`) in a single Prometheus text
//! response, so Prometheus only needs one scrape target per pod.

use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::{Arc, OnceLock},
    time::Duration,
};

use axum::{extract::State, response::IntoResponse, routing::get, Router};
use metrics_exporter_prometheus::PrometheusHandle;
use tokio::task::JoinHandle;
use tracing::{error, info};

use super::metrics::UPKEEP_INTERVAL_SECS;
use crate::core::{
    worker_manager::{EngineMetricsResult, WorkerManager},
    worker_registry::WorkerRegistry,
};

/// Shared references for engine metrics collection, populated after app init.
pub struct EngineMetricsDeps {
    pub worker_registry: Arc<WorkerRegistry>,
    pub client: reqwest::Client,
}

/// Global handle set once after AppContext is built.
static ENGINE_METRICS_DEPS: OnceLock<EngineMetricsDeps> = OnceLock::new();

/// Register the worker registry and HTTP client for engine metrics collection.
/// Called once after AppContext is initialized.
pub fn register_engine_metrics_deps(worker_registry: Arc<WorkerRegistry>, client: reqwest::Client) {
    let _ = ENGINE_METRICS_DEPS.set(EngineMetricsDeps {
        worker_registry,
        client,
    });
}

#[derive(Clone)]
struct MetricsState {
    handle: PrometheusHandle,
}

async fn prometheus_handler(State(state): State<MetricsState>) -> impl IntoResponse {
    let smg_text = state.handle.render();

    let engine_text = if let Some(deps) = ENGINE_METRICS_DEPS.get() {
        match WorkerManager::get_engine_metrics(&deps.worker_registry, &deps.client).await {
            EngineMetricsResult::Ok(text) => text,
            EngineMetricsResult::Err(_) => String::new(),
        }
    } else {
        String::new()
    };

    (
        [(
            http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        format!("{smg_text}\n{engine_text}"),
    )
}

/// Start the metrics HTTP server. If the port is unavailable, logs an error
/// and returns a no-op handle so the router can still operate.
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

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            error!("failed to bind metrics server on {addr}: {e} — metrics will be unavailable");
            #[expect(
                clippy::disallowed_methods,
                reason = "no-op task for graceful degradation"
            )]
            return tokio::spawn(async {});
        }
    };

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

    let app = Router::new().route("/metrics", get(prometheus_handler).with_state(state));

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
