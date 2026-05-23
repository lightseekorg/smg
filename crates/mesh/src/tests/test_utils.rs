//! Test utilities for mesh module
use std::{net::SocketAddr, time::Duration};

use tokio::net::TcpListener;

/// Bind to an ephemeral port and return the listener + address.
/// The caller must keep the listener alive and pass it to `mesh_run!`
/// to avoid a TOCTOU port race.
pub async fn bind_node() -> (TcpListener, SocketAddr) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tracing::debug!("Bound node to {}", addr);
    (listener, addr)
}

/// Poll `condition` every 100ms until it returns true or `timeout` expires.
pub async fn wait_for<F>(condition: F, timeout: Duration, msg: &str)
where
    F: Fn() -> bool,
{
    let start = std::time::Instant::now();
    while !condition() {
        assert!(
            start.elapsed() <= timeout,
            "Timeout after {timeout:?} waiting for: {msg}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
