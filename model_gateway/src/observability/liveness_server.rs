//! Dedicated liveness server (optional, off by default).
//!
//! The main `/health` endpoint is served from the request runtime and its full
//! middleware stack, so when worker threads are saturated with CPU-bound work
//! the runtime cannot schedule the liveness future within the k8s probe timeout
//! and pods get restarted under load. When `liveness_port` is configured, this
//! module serves a minimal `GET /health` and `GET /liveness` from a dedicated OS
//! thread running its own single-threaded tokio runtime, so the request runtime
//! cannot starve it.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use axum::{http::StatusCode, response::IntoResponse, routing::get, Router};
use tracing::{error, info};

async fn liveness() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

fn liveness_router() -> Router {
    Router::new()
        .route("/health", get(liveness))
        .route("/liveness", get(liveness))
}

async fn serve(addr: SocketAddr) {
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => listener,
        Err(e) => {
            error!("Dedicated liveness server failed to bind on {addr}: {e}");
            return;
        }
    };
    info!("Dedicated liveness server listening on {addr} (/health + /liveness)");
    if let Err(e) = axum::serve(listener, liveness_router()).await {
        error!("Dedicated liveness server error: {e}");
    }
}

/// Spawn the dedicated liveness server on its own OS thread + current-thread
/// runtime so a CPU-saturated request runtime cannot starve the probe.
pub fn spawn_liveness_server(host: &str, port: u16) {
    let ip_addr: IpAddr = host.parse().unwrap_or_else(|e| {
        error!("Failed to parse liveness host '{host}': {e}, falling back to 0.0.0.0");
        IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0))
    });
    let addr = SocketAddr::new(ip_addr, port);

    // A dedicated OS thread + its own runtime is the whole point: it stays
    // schedulable even when the main multi-threaded runtime is saturated.
    let spawn_result = std::thread::Builder::new()
        .name("liveness-server".to_string())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(e) => {
                    error!("Failed to build liveness server runtime: {e}");
                    return;
                }
            };
            runtime.block_on(serve(addr));
        });

    if let Err(e) = spawn_result {
        error!("Failed to spawn liveness server thread: {e}");
    }
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    #[tokio::test]
    async fn liveness_endpoints_return_200_ok() {
        for path in ["/health", "/liveness"] {
            let response = liveness_router()
                .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "path: {path}");
            let body = response.into_body().collect().await.unwrap().to_bytes();
            assert_eq!(&body[..], b"OK", "path: {path}");
        }
    }

    #[test]
    fn spawn_with_bad_host_does_not_panic() {
        // Unparseable host falls back to 0.0.0.0; spawning must not panic even
        // when the dedicated thread cannot ultimately bind.
        spawn_liveness_server("not-an-ip", 0);
    }
}
