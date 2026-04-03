//! Shared URL normalization and network probe utilities for worker steps.

use std::time::Duration;

use reqwest::Client;

use crate::routers::grpc::client::GrpcClient;

/// Strip protocol prefix (http://, https://, grpc://) from URL.
pub(crate) fn strip_protocol(url: &str) -> String {
    url.trim_start_matches("http://")
        .trim_start_matches("https://")
        .trim_start_matches("grpc://")
        .to_string()
}

/// Ensure URL has an HTTP(S) scheme — handles bare `host:port` and `grpc://` inputs.
pub(crate) fn http_base_url(url: &str) -> String {
    if url.starts_with("http://") || url.starts_with("https://") {
        url.trim_end_matches('/').to_string()
    } else {
        format!("http://{}", strip_protocol(url).trim_end_matches('/'))
    }
}

/// Ensure URL has a gRPC scheme — handles bare `host:port` and `http://` inputs.
pub(crate) fn grpc_base_url(url: &str) -> String {
    if url.starts_with("grpc://") {
        url.trim_end_matches('/').to_string()
    } else {
        format!("grpc://{}", strip_protocol(url).trim_end_matches('/'))
    }
}

/// Try HTTP health check (2xx response required).
pub(crate) async fn try_http_reachable(
    url: &str,
    timeout_secs: u64,
    client: &Client,
) -> Result<(), String> {
    let is_https = url.starts_with("https://");
    let protocol = if is_https { "https" } else { "http" };
    let clean_url = strip_protocol(url);
    let health_url = format!("{protocol}://{clean_url}/health");

    client
        .get(&health_url)
        .timeout(Duration::from_secs(timeout_secs))
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|e| format!("Health check failed: {e}"))?;

    Ok(())
}

/// Perform a single gRPC health check with a specific runtime type.
///
/// Also used by `DetectBackendStep` for runtime identification.
pub(crate) async fn do_grpc_health_check(
    grpc_url: &str,
    timeout_secs: u64,
    runtime_type: &str,
) -> Result<(), String> {
    let connect_future = GrpcClient::connect(grpc_url, runtime_type);
    let client = tokio::time::timeout(Duration::from_secs(timeout_secs), connect_future)
        .await
        .map_err(|_| "gRPC connection timeout".to_string())?
        .map_err(|e| format!("gRPC connection failed: {e}"))?;

    let health_future = client.health_check();
    tokio::time::timeout(Duration::from_secs(timeout_secs), health_future)
        .await
        .map_err(|_| "gRPC health check timeout".to_string())?
        .map_err(|e| format!("gRPC health check failed: {e}"))?;

    Ok(())
}

/// Check if gRPC is reachable by trying all known runtime types in parallel.
///
/// We don't care which runtime it is here — that's `DetectBackendStep`'s job.
/// We just need to know: does this endpoint speak gRPC at all?
pub(crate) async fn try_grpc_reachable(url: &str, timeout_secs: u64) -> Result<(), String> {
    let grpc_url = if url.starts_with("grpc://") {
        url.to_string()
    } else {
        format!("grpc://{}", strip_protocol(url))
    };

    let (sglang, vllm, trtllm, mlx) = tokio::join!(
        do_grpc_health_check(&grpc_url, timeout_secs, "sglang"),
        do_grpc_health_check(&grpc_url, timeout_secs, "vllm"),
        do_grpc_health_check(&grpc_url, timeout_secs, "trtllm"),
        do_grpc_health_check(&grpc_url, timeout_secs, "mlx"),
    );

    match (sglang, vllm, trtllm, mlx) {
        (Ok(()), _, _, _) | (_, Ok(()), _, _) | (_, _, Ok(()), _) | (_, _, _, Ok(())) => Ok(()),
        (Err(e1), Err(e2), Err(e3), Err(e4)) => Err(format!(
            "gRPC not reachable (tried sglang, vllm, trtllm, mlx): sglang={e1}, vllm={e2}, trtllm={e3}, mlx={e4}",
        )),
    }
}
