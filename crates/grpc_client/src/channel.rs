//! Shared `tonic::Channel` builder for SMG gRPC clients.
//!
//! Each engine client (sglang, vllm, trtllm, mlx) connects to its backend
//! the same way: accept either an `http(s)://` or a `grpc(s)://` endpoint,
//! convert the gRPC schemes to tonic-compatible HTTP(S) ones, and build a
//! `Channel` with the same keep-alive / window-size profile. This module
//! centralises that pipeline so adding a new engine — or tuning the
//! transport profile — touches one file instead of four.

use std::time::Duration;

use tonic::transport::Channel;

/// Convert a `grpc://` or `grpcs://` endpoint to a tonic-compatible
/// `http://` or `https://` URI. Other schemes (or schemeless inputs) are
/// returned unchanged so callers can mix `http(s)://` and `grpc(s)://`
/// freely.
pub fn normalize_grpc_endpoint(endpoint: &str) -> String {
    match endpoint.split_once("://") {
        Some(("grpc", rest)) => format!("http://{rest}"),
        Some(("grpcs", rest)) => format!("https://{rest}"),
        _ => endpoint.to_string(),
    }
}

/// Connect a `tonic::Channel` to the given endpoint with the SMG-standard
/// keep-alive and HTTP/2 window profile applied.
///
/// The endpoint may use any of `http://`, `https://`, `grpc://`, or
/// `grpcs://` — gRPC schemes are normalised to their HTTP(S) equivalents
/// before tonic parses them.
pub async fn connect_channel(
    endpoint: &str,
) -> Result<Channel, Box<dyn std::error::Error + Send + Sync>> {
    let http_endpoint = normalize_grpc_endpoint(endpoint);
    let channel = Channel::from_shared(http_endpoint)?
        .http2_keep_alive_interval(Duration::from_secs(30))
        .keep_alive_timeout(Duration::from_secs(10))
        .keep_alive_while_idle(true)
        .tcp_keepalive(Some(Duration::from_secs(60)))
        .tcp_nodelay(true)
        .http2_adaptive_window(true)
        // 16MB stream window, 32MB connection window — sized for the
        // typical inference response (multi-MB tokenized payloads +
        // streaming chunks) without head-of-line blocking.
        .initial_stream_window_size(Some(16 * 1024 * 1024))
        .initial_connection_window_size(Some(32 * 1024 * 1024))
        .connect()
        .await?;
    Ok(channel)
}

#[cfg(test)]
mod tests {
    use super::normalize_grpc_endpoint;

    #[test]
    fn normalize_grpc_to_http() {
        assert_eq!(
            normalize_grpc_endpoint("grpc://worker:8080"),
            "http://worker:8080"
        );
    }

    #[test]
    fn normalize_grpcs_to_https() {
        assert_eq!(
            normalize_grpc_endpoint("grpcs://worker:8443"),
            "https://worker:8443"
        );
    }

    #[test]
    fn normalize_passes_http_through() {
        assert_eq!(
            normalize_grpc_endpoint("http://worker:8080"),
            "http://worker:8080"
        );
    }

    #[test]
    fn normalize_passes_https_through() {
        assert_eq!(
            normalize_grpc_endpoint("https://worker:8443"),
            "https://worker:8443"
        );
    }

    #[test]
    fn normalize_passes_unknown_scheme_through() {
        // Tonic will reject this, but normalize is not a validator —
        // it only rewrites gRPC schemes.
        assert_eq!(
            normalize_grpc_endpoint("tcp://worker:9000"),
            "tcp://worker:9000"
        );
    }

    #[test]
    fn normalize_passes_schemeless_through() {
        assert_eq!(normalize_grpc_endpoint("worker:8080"), "worker:8080");
    }

    #[test]
    fn normalize_handles_path_after_authority() {
        assert_eq!(
            normalize_grpc_endpoint("grpc://worker:8080/some/path"),
            "http://worker:8080/some/path"
        );
    }

    #[test]
    fn normalize_is_case_sensitive_on_scheme() {
        // Schemes are conventionally lowercase; tonic itself is case
        // sensitive on the URI, so we don't rewrite uppercased gRPC.
        assert_eq!(
            normalize_grpc_endpoint("GRPC://worker:8080"),
            "GRPC://worker:8080"
        );
    }
}
