//! HTTP/WebSocket server for the Prometheus metrics endpoint (port 29000).
//! Serves `GET /metrics` (Prometheus) and `WS /ws/metrics` (real-time state push).
//!
//! The `/metrics` endpoint returns both SMG's own metrics (`smg_*`) and engine
//! metrics (`vllm_*`, `sglang_*`, `nv_trt_*`) in a single Prometheus text
//! response, so Prometheus only needs one scrape target per pod.

use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::{atomic::AtomicUsize, Arc, OnceLock},
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
use crate::worker::{
    manager::{EngineMetricsResult, WorkerManager},
    registry::WorkerRegistry,
};

/// Default maximum concurrent WebSocket connections on the metrics endpoint.
pub const DEFAULT_MAX_WS_CONNECTIONS: usize = 32;

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

/// Start the metrics HTTP/WS server. If the port is unavailable, logs an error
/// and returns a no-op handle so the router can still operate.
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
