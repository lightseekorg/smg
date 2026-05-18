use std::{path::PathBuf, time::Duration};

use axum::{body::Body, response::Response};
use http::{HeaderMap, HeaderValue, StatusCode};
use smg_mesh::{MTLSConfig, MTLSManager, SpiffeIdentity};
use url::Url;

use super::{
    headers::{
        ALLOWED_MODELS_HEADER, ALLOWED_REGIONS_HEADER, ATTEMPT_HEADER, COMMITTED_MODEL_HEADER,
        FAILOVER_MODE_HEADER, INPUT_MODALITY_HEADER, MAX_RETRY_HEADER, OUTPUT_MODALITY_HEADER,
        REQUEST_MODE_HEADER, REQUEST_MODE_SETTLED, ROUTE_ID_HEADER, SOURCE_SERVICE_HEADER,
        TARGET_REGION_HEADER,
    },
    peers::RegionPeerRequestTarget,
    CrossRegionError, CrossRegionMtlsRuntimeConfig, CrossRegionResult, ExecutionTarget,
    FailoverPolicy, RegionPeerRegistry, RegionRouteDecision, RouteCommit,
};
use crate::{
    config::CrossRegionFailoverMode,
    routers::common::{header_utils, retry::is_retryable_status},
};

const LOCAL_WORKER_SELECTION_HEADERS: &[&str] = &["x-smg-target-worker", "x-smg-routing-key"];

/// Existing SMG endpoint path that cannot be an absolute worker URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExistingSmgPath(String);

impl ExistingSmgPath {
    /// Build a forwarding path for an existing SMG endpoint.
    pub fn new(path: impl Into<String>) -> CrossRegionResult<Self> {
        let path = path.into();
        if path.trim().is_empty() {
            return Err(CrossRegionError::InvalidForwardingTarget {
                reason: "existing SMG path must not be empty".to_string(),
            });
        }
        if path.trim() != path {
            return Err(CrossRegionError::InvalidForwardingTarget {
                reason: "existing SMG path must not contain surrounding whitespace".to_string(),
            });
        }
        if Url::parse(&path).is_ok() || path.starts_with("//") {
            return Err(CrossRegionError::InvalidForwardingTarget {
                reason: "forwarding path must be an existing SMG endpoint path, not a URL"
                    .to_string(),
            });
        }
        if !path.starts_with('/') {
            return Err(CrossRegionError::InvalidForwardingTarget {
                reason: "existing SMG path must start with '/'".to_string(),
            });
        }

        Ok(Self(path))
    }

    /// Return the validated path string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Minimal request envelope reserved for later remote SMG forwarding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardingRequest {
    existing_smg_path: ExistingSmgPath,
    headers: HeaderMap,
    body: Vec<u8>,
}

impl ForwardingRequest {
    /// Build a forwarding request for an existing SMG endpoint path.
    pub fn new(existing_smg_path: impl Into<String>, body: Vec<u8>) -> CrossRegionResult<Self> {
        Ok(Self {
            existing_smg_path: ExistingSmgPath::new(existing_smg_path)?,
            headers: HeaderMap::new(),
            body,
        })
    }

    /// Build a settled forwarding request from an unresolved request and route commit.
    pub fn from_unresolved(
        existing_smg_path: impl Into<String>,
        source_headers: &HeaderMap,
        body: Vec<u8>,
        commit: &RouteCommit,
    ) -> CrossRegionResult<Self> {
        Ok(Self {
            existing_smg_path: ExistingSmgPath::new(existing_smg_path)?,
            headers: settled_forwarding_headers(source_headers, commit)?,
            body,
        })
    }

    /// Return the existing SMG endpoint path for the remote request.
    pub fn existing_smg_path(&self) -> &ExistingSmgPath {
        &self.existing_smg_path
    }

    /// Return headers to send to the remote Region Agent.
    pub fn headers(&self) -> &HeaderMap {
        &self.headers
    }

    /// Return the forwarding request body bytes.
    pub fn body(&self) -> &[u8] {
        &self.body
    }
}

/// Resolved forwarding target that pairs the internal URL with peer identity.
#[derive(Clone, PartialEq, Eq)]
pub struct ForwardingTarget {
    url: Url,
    expected_mtls_identity: String,
}

impl ForwardingTarget {
    /// Build a forwarding target after request path and peer resolution validation.
    fn new(url: Url, expected_mtls_identity: impl Into<String>) -> Self {
        Self {
            url,
            expected_mtls_identity: expected_mtls_identity.into(),
        }
    }

    /// Return the peer identity that outbound mTLS must enforce.
    pub fn expected_mtls_identity(&self) -> &str {
        &self.expected_mtls_identity
    }

    /// Return the endpoint origin for internal diagnostics without exposing a forwarding URL.
    pub(crate) fn endpoint_origin(&self) -> String {
        self.url.origin().ascii_serialization()
    }
}

/// Build the SETTLED header set for a remote Region Agent forward.
fn settled_forwarding_headers(
    source_headers: &HeaderMap,
    commit: &RouteCommit,
) -> CrossRegionResult<HeaderMap> {
    let mut headers = source_headers.clone();
    remove_unresolved_profile_headers(&mut headers);
    remove_local_worker_selection_headers(&mut headers);
    remove_transport_headers(&mut headers);
    insert_static_header(&mut headers, REQUEST_MODE_HEADER, REQUEST_MODE_SETTLED);
    insert_static_header(&mut headers, SOURCE_SERVICE_HEADER, "smg");
    insert_header(&mut headers, TARGET_REGION_HEADER, &commit.target_region)?;
    insert_header(&mut headers, COMMITTED_MODEL_HEADER, &commit.model_id)?;
    insert_header(&mut headers, ROUTE_ID_HEADER, &commit.route_id)?;
    insert_header(&mut headers, ATTEMPT_HEADER, &commit.attempt.to_string())?;
    Ok(headers)
}

/// Remove profile-only headers that the target Region Agent rejects for SETTLED requests.
fn remove_unresolved_profile_headers(headers: &mut HeaderMap) {
    for header in [
        ALLOWED_REGIONS_HEADER,
        ALLOWED_MODELS_HEADER,
        FAILOVER_MODE_HEADER,
        MAX_RETRY_HEADER,
        INPUT_MODALITY_HEADER,
        OUTPUT_MODALITY_HEADER,
    ] {
        headers.remove(header);
    }
}

/// Remove worker-selection controls that are meaningful only inside the source SMG.
fn remove_local_worker_selection_headers(headers: &mut HeaderMap) {
    for header in LOCAL_WORKER_SELECTION_HEADERS {
        headers.remove(*header);
    }
}

/// Remove headers that reqwest must recompute for the outbound hop.
fn remove_transport_headers(headers: &mut HeaderMap) {
    for header in [
        http::header::CONTENT_LENGTH,
        http::header::HOST,
        http::header::CONNECTION,
        http::header::TRANSFER_ENCODING,
    ] {
        headers.remove(header);
    }
}

/// Insert a known-good static header value.
fn insert_static_header(headers: &mut HeaderMap, name: &'static str, value: &'static str) {
    headers.insert(name, HeaderValue::from_static(value));
}

/// Insert route metadata as a validated HTTP header value.
fn insert_header(
    headers: &mut HeaderMap,
    name: &'static str,
    value: &str,
) -> CrossRegionResult<()> {
    let value = HeaderValue::from_str(value).map_err(|error| {
        CrossRegionError::InvalidForwardingTarget {
            reason: format!("route metadata header {name} is invalid: {error}"),
        }
    })?;
    headers.insert(name, value);
    Ok(())
}

/// Convert a reqwest response into an Axum response without buffering the body.
fn response_from_reqwest(response: reqwest::Response) -> Response {
    let status = StatusCode::from_u16(response.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let headers = header_utils::preserve_response_headers(response.headers());
    let body = Body::from_stream(response.bytes_stream());
    let mut forwarded = Response::new(body);
    *forwarded.status_mut() = status;
    *forwarded.headers_mut() = headers;
    forwarded
}

/// Return true when a remote forwarding status can trigger bounded failover.
pub(crate) fn is_retryable_remote_forward_status(status: StatusCode) -> bool {
    is_retryable_status(status)
}

/// Return true when a remote response has begun a streaming body contract.
pub(crate) fn is_streaming_remote_forward_response(response: &Response) -> bool {
    response
        .headers()
        .get(http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value.split(';').next().is_some_and(|media_type| {
                media_type.trim().eq_ignore_ascii_case("text/event-stream")
            })
        })
}

/// Decide whether entry-region failover may try another remote region.
///
/// Attempts are 1-based. `max_retry` is the number of additional retries after
/// the initial committed forwarding attempt, so attempt 1 may fail over when
/// `max_retry >= 1`.
pub(crate) fn should_failover_remote_forward(
    policy: &FailoverPolicy,
    status: StatusCode,
    streaming_response_started: bool,
    attempt: u32,
) -> bool {
    policy.failover_mode == CrossRegionFailoverMode::Automatic
        && !streaming_response_started
        && is_retryable_remote_forward_status(status)
        && attempt <= policy.max_retry
}

/// Return true when a remote response should count as a breaker failure.
pub(crate) fn should_record_remote_forward_failure(
    status: StatusCode,
    streaming_response_started: bool,
) -> bool {
    !streaming_response_started && is_retryable_remote_forward_status(status)
}

/// Build a request-forwarding HTTP client with outbound mTLS identity enforcement.
pub async fn build_request_forwarding_http_client(
    mtls: &CrossRegionMtlsRuntimeConfig,
    target: &ForwardingTarget,
    timeout: Duration,
) -> CrossRegionResult<reqwest::Client> {
    let expected_identity = SpiffeIdentity::parse_region_agent(target.expected_mtls_identity())
        .map_err(|error| CrossRegionError::InvalidForwardingTarget {
            reason: format!(
                "peer expected_mtls_identity is not a Region Agent SPIFFE URI: {error}"
            ),
        })?;
    let mtls_manager = MTLSManager::new(MTLSConfig {
        ca_cert_path: PathBuf::from(&mtls.ca_cert_path),
        server_cert_path: PathBuf::from(&mtls.server_cert_path),
        server_key_path: PathBuf::from(&mtls.server_key_path),
        client_cert_path: PathBuf::from(&mtls.client_cert_path),
        client_key_path: PathBuf::from(&mtls.client_key_path),
        require_client_cert: true,
        ..MTLSConfig::default()
    });
    let tls_config = mtls_manager
        .load_client_config_for_server_identity(&expected_identity)
        .await
        .map_err(|error| CrossRegionError::InvalidForwardingTarget {
            reason: format!("failed to load request-forwarding mTLS client config: {error}"),
        })?;

    reqwest::Client::builder()
        .timeout(timeout)
        .use_preconfigured_tls((*tls_config).clone())
        .build()
        .map_err(|error| CrossRegionError::InvalidForwardingTarget {
            reason: format!("failed to build request-forwarding HTTP client: {error}"),
        })
}

impl std::fmt::Debug for ForwardingTarget {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ForwardingTarget")
            .field("endpoint_origin", &self.endpoint_origin())
            .field("expected_mtls_identity", &self.expected_mtls_identity)
            .finish()
    }
}

/// No-op remote forwarder boundary that resolves only by region id.
#[derive(Debug, Clone)]
pub struct CrossRegionForwarder {
    peer_registry: RegionPeerRegistry,
}

impl CrossRegionForwarder {
    /// Create a forwarder boundary from a peer registry.
    pub fn new(peer_registry: RegionPeerRegistry) -> Self {
        Self { peer_registry }
    }

    /// Resolve the request-plane target for a remote decision.
    pub(crate) fn request_target_for_decision(
        &self,
        decision: &RegionRouteDecision,
    ) -> CrossRegionResult<RegionPeerRequestTarget> {
        match &decision.execution_target {
            ExecutionTarget::RemoteRegion { region_id } => {
                self.peer_registry.request_target(region_id)
            }
            ExecutionTarget::LocalRegion => Err(CrossRegionError::InvalidForwardingTarget {
                reason: "local execution must use the existing local router path".to_string(),
            }),
        }
    }

    /// Resolve the remote Region Agent target for a validated forwarding request.
    pub fn forwarding_target_for_decision(
        &self,
        decision: &RegionRouteDecision,
        request: &ForwardingRequest,
    ) -> CrossRegionResult<ForwardingTarget> {
        if let ExecutionTarget::RemoteRegion { region_id } = &decision.execution_target {
            if decision.target_region != *region_id {
                return Err(CrossRegionError::InvalidForwardingTarget {
                    reason: format!(
                        "route target_region '{}' must match execution target region '{}'",
                        decision.target_region, region_id
                    ),
                });
            }
        }

        let target = self.request_target_for_decision(decision)?;
        let url = target
            .request_url()
            .join(request.existing_smg_path().as_str())
            .map_err(|e| CrossRegionError::InvalidForwardingTarget {
                reason: format!("existing SMG path could not be joined to peer request_url: {e}"),
            })?;

        Ok(ForwardingTarget::new(url, target.expected_mtls_identity()))
    }

    /// Forward a request to a remote Region Agent selected by route decision.
    pub async fn forward(
        &self,
        client: &reqwest::Client,
        decision: &RegionRouteDecision,
        request: ForwardingRequest,
    ) -> CrossRegionResult<Response> {
        let target = self.forwarding_target_for_decision(decision, &request)?;
        Self::forward_to_target(client, target, request).await
    }

    /// Send a settled request to an already-resolved forwarding target.
    pub(crate) async fn forward_to_target(
        client: &reqwest::Client,
        target: ForwardingTarget,
        request: ForwardingRequest,
    ) -> CrossRegionResult<Response> {
        let mut builder = client.post(target.url.clone());
        for (name, value) in request.headers() {
            builder = builder.header(name, value);
        }

        let response = builder.body(request.body).send().await.map_err(|error| {
            CrossRegionError::InvalidForwardingTarget {
                reason: format!(
                    "failed to forward request to remote Region Agent {}: {error}",
                    target.endpoint_origin()
                ),
            }
        })?;
        Ok(response_from_reqwest(response))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::{body::Bytes, extract::State, routing::post, Router};
    use tokio::sync::oneshot;

    use super::*;
    use crate::cross_region::{
        headers::{
            ALLOWED_MODELS_HEADER, ALLOWED_REGIONS_HEADER, ATTEMPT_HEADER, COMMITTED_MODEL_HEADER,
            CONTRACT_VERSION_HEADER, ENTRY_REGION_HEADER, FAILOVER_MODE_HEADER, MAX_RETRY_HEADER,
            OPC_REQUEST_ID_HEADER, REQUEST_MODE_HEADER, REQUEST_MODE_SETTLED,
            REQUEST_MODE_UNRESOLVED, ROUTE_ID_HEADER, SOURCE_SERVICE_HEADER, TARGET_REGION_HEADER,
        },
        RegionPeer, RequestMode, RouteCommit,
    };

    /// Build a remote route decision fixture.
    fn remote_decision(region_id: &str) -> RegionRouteDecision {
        RegionRouteDecision {
            route_id: "route-1".to_string(),
            target_region: region_id.to_string(),
            model_id: "cohere.command-r-plus".to_string(),
            execution_target: ExecutionTarget::RemoteRegion {
                region_id: region_id.to_string(),
            },
        }
    }

    /// Build a committed remote route fixture.
    fn remote_commit(region_id: &str) -> RouteCommit {
        RouteCommit {
            route_id: "route-1".to_string(),
            entry_region: "us-ashburn-1".to_string(),
            target_region: region_id.to_string(),
            model_id: "cohere.command-r-plus".to_string(),
            request_mode: RequestMode::Settled,
            attempt: 1,
            failover_mode: CrossRegionFailoverMode::Automatic,
        }
    }

    /// Build unresolved headers as received from DP-API before route commitment.
    fn unresolved_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        insert(&mut headers, CONTRACT_VERSION_HEADER, "1");
        insert(&mut headers, REQUEST_MODE_HEADER, REQUEST_MODE_UNRESOLVED);
        insert(&mut headers, ENTRY_REGION_HEADER, "us-ashburn-1");
        insert(&mut headers, SOURCE_SERVICE_HEADER, "dp-api");
        insert(&mut headers, OPC_REQUEST_ID_HEADER, "opc-request-1");
        insert(
            &mut headers,
            ALLOWED_REGIONS_HEADER,
            "us-ashburn-1, us-chicago-1",
        );
        insert(&mut headers, ALLOWED_MODELS_HEADER, "cohere.command-r-plus");
        insert(&mut headers, FAILOVER_MODE_HEADER, "AUTO");
        insert(&mut headers, MAX_RETRY_HEADER, "3");
        insert(&mut headers, "x-smg-target-worker", "2");
        insert(&mut headers, "x-smg-routing-key", "local-key");
        insert(&mut headers, http::header::CONTENT_LENGTH.as_str(), "999");
        headers
    }

    /// Add a static test header value.
    fn insert(headers: &mut HeaderMap, name: &'static str, value: &'static str) {
        headers.insert(name, HeaderValue::from_static(value));
    }

    #[test]
    fn forwarding_request_adds_settled_headers_and_removes_unresolved_profile_headers() {
        let request = ForwardingRequest::from_unresolved(
            "/v1/chat/completions",
            &unresolved_headers(),
            br#"{"model":"cohere.command-r-plus"}"#.to_vec(),
            &remote_commit("us-chicago-1"),
        )
        .expect("forwarding request should build");

        assert_eq!(
            request.headers().get(REQUEST_MODE_HEADER),
            Some(&HeaderValue::from_static(REQUEST_MODE_SETTLED))
        );
        assert_eq!(
            request.headers().get(SOURCE_SERVICE_HEADER),
            Some(&HeaderValue::from_static("smg"))
        );
        assert_eq!(
            request.headers().get(TARGET_REGION_HEADER),
            Some(&HeaderValue::from_static("us-chicago-1"))
        );
        assert_eq!(
            request.headers().get(COMMITTED_MODEL_HEADER),
            Some(&HeaderValue::from_static("cohere.command-r-plus"))
        );
        assert_eq!(
            request.headers().get(ROUTE_ID_HEADER),
            Some(&HeaderValue::from_static("route-1"))
        );
        assert_eq!(
            request.headers().get(ATTEMPT_HEADER),
            Some(&HeaderValue::from_static("1"))
        );
        assert!(!request.headers().contains_key(ALLOWED_REGIONS_HEADER));
        assert!(!request.headers().contains_key(ALLOWED_MODELS_HEADER));
        assert!(!request.headers().contains_key(FAILOVER_MODE_HEADER));
        assert!(!request.headers().contains_key(MAX_RETRY_HEADER));
        assert!(!request.headers().contains_key("x-smg-target-worker"));
        assert!(!request.headers().contains_key("x-smg-routing-key"));
        assert!(!request.headers().contains_key(http::header::CONTENT_LENGTH));
        assert_eq!(request.body(), br#"{"model":"cohere.command-r-plus"}"#);
    }

    #[derive(Clone)]
    struct CaptureState {
        sender: Arc<tokio::sync::Mutex<Option<oneshot::Sender<(HeaderMap, Bytes)>>>>,
    }

    /// Capture the headers and body sent to the fake remote Region Agent.
    async fn capture_forwarded_request(
        State(state): State<CaptureState>,
        headers: HeaderMap,
        body: Bytes,
    ) -> (StatusCode, [(&'static str, &'static str); 1], &'static str) {
        if let Some(sender) = state.sender.lock().await.take() {
            let _ = sender.send((headers, body));
        }
        (
            StatusCode::ACCEPTED,
            [("content-type", "text/event-stream")],
            "data: ok\n\n",
        )
    }

    /// Forward one request to a loopback fake Region Agent and return what it observed.
    async fn forward_to_fake_region_agent(
        path: &'static str,
        request_body: Vec<u8>,
    ) -> (Response, HeaderMap, Bytes) {
        let (sender, receiver) = oneshot::channel();
        let app = Router::new()
            .route(path, post(capture_forwarded_request))
            .with_state(CaptureState {
                sender: Arc::new(tokio::sync::Mutex::new(Some(sender))),
            });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener should bind");
        let addr = listener
            .local_addr()
            .expect("test listener should expose local addr");
        #[expect(
            clippy::disallowed_methods,
            reason = "test server lifetime is bounded by the test runtime"
        )]
        tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("test server should serve");
        });

        let target = ForwardingTarget::new(
            Url::parse(&format!("http://{addr}{path}")).expect("target url should parse"),
            "spiffe://oraclecorp.com/oci/oc1/prod/region/us-chicago-1/service/smg-region-agent",
        );
        let request = ForwardingRequest::from_unresolved(
            path,
            &unresolved_headers(),
            request_body,
            &remote_commit("us-chicago-1"),
        )
        .expect("forwarding request should build");

        let response = tokio::time::timeout(
            Duration::from_secs(5),
            CrossRegionForwarder::forward_to_target(&reqwest::Client::new(), target, request),
        )
        .await
        .expect("forwarding should not hang")
        .expect("forwarding should succeed");
        let (headers, body) = tokio::time::timeout(Duration::from_secs(5), receiver)
            .await
            .expect("remote capture should not hang")
            .expect("remote should receive request");
        (response, headers, body)
    }

    #[tokio::test]
    async fn forward_to_target_sends_requests_to_existing_smg_paths() {
        for (path, request_body) in [
            (
                "/v1/chat/completions",
                br#"{"model":"cohere.command-r-plus","stream":true}"#.as_slice(),
            ),
            (
                "/v1/responses",
                br#"{"model":"cohere.command-r-plus","input":"hello","stream":true}"#.as_slice(),
            ),
        ] {
            let (response, headers, body) =
                forward_to_fake_region_agent(path, request_body.to_vec()).await;
            let response_status = response.status();
            let response_content_type = response.headers().get(http::header::CONTENT_TYPE).cloned();
            let response_body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("response body should read");

            assert_eq!(
                headers
                    .get(REQUEST_MODE_HEADER)
                    .and_then(|v| v.to_str().ok()),
                Some(REQUEST_MODE_SETTLED)
            );
            assert_eq!(
                headers
                    .get(TARGET_REGION_HEADER)
                    .and_then(|v| v.to_str().ok()),
                Some("us-chicago-1")
            );
            assert!(!headers.contains_key("x-smg-target-worker"));
            assert!(!headers.contains_key("x-smg-routing-key"));
            assert_eq!(body, Bytes::copy_from_slice(request_body));
            assert_eq!(response_status, StatusCode::ACCEPTED);
            assert_eq!(
                response_content_type
                    .as_ref()
                    .and_then(|value| value.to_str().ok()),
                Some("text/event-stream")
            );
            assert_eq!(response_body, Bytes::from_static(b"data: ok\n\n"));
        }
    }

    #[test]
    fn forwarder_resolves_request_target_without_exposing_peer() {
        let peer = RegionPeer::new(
            "us-chicago-1",
            "https://smg-region-agent.us-chicago-1.internal:8443",
            "https://smg-region-agent.us-chicago-1.internal:9443",
            "oc1",
            "prod",
            None,
        )
        .expect("peer should parse");
        let registry = RegionPeerRegistry::new(vec![peer]).expect("registry should build");
        let forwarder = CrossRegionForwarder::new(registry);

        let _target = forwarder
            .request_target_for_decision(&remote_decision("us-chicago-1"))
            .expect("target should resolve");
    }

    #[test]
    fn forwarder_resolves_request_target_by_region_only() {
        let peer = RegionPeer::new(
            "us-chicago-1",
            "https://smg-region-agent.us-chicago-1.internal:8443",
            "https://smg-region-agent.us-chicago-1.internal:9443",
            "oc1",
            "prod",
            None,
        )
        .expect("peer should parse");
        let registry = RegionPeerRegistry::new(vec![peer]).expect("registry should build");
        let forwarder = CrossRegionForwarder::new(registry);

        let target = forwarder
            .request_target_for_decision(&remote_decision("us-chicago-1"))
            .expect("target should resolve");

        assert_eq!(
            target.request_url().as_str(),
            "https://smg-region-agent.us-chicago-1.internal:8443/"
        );
    }

    #[test]
    fn forwarding_target_uses_peer_identity_and_existing_smg_path() {
        let peer = RegionPeer::new(
            "us-chicago-1",
            "https://smg-region-agent.us-chicago-1.internal:8443",
            "https://smg-region-agent.us-chicago-1.internal:9443",
            "oc1",
            "prod",
            None,
        )
        .expect("peer should parse");
        let registry = RegionPeerRegistry::new(vec![peer]).expect("registry should build");
        let forwarder = CrossRegionForwarder::new(registry);
        let request = ForwardingRequest::new("/v1/chat/completions", b"{}".to_vec())
            .expect("path should parse");

        let target = forwarder
            .forwarding_target_for_decision(&remote_decision("us-chicago-1"), &request)
            .expect("target should resolve");

        assert_eq!(
            target.endpoint_origin(),
            "https://smg-region-agent.us-chicago-1.internal:8443"
        );
        assert_eq!(
            target.expected_mtls_identity(),
            "spiffe://oraclecorp.com/oci/oc1/prod/region/us-chicago-1/service/smg-region-agent"
        );
    }

    #[test]
    fn forwarding_target_reuses_supported_smg_paths() {
        let peer = RegionPeer::new(
            "us-chicago-1",
            "https://smg-region-agent.us-chicago-1.internal:8443",
            "https://smg-region-agent.us-chicago-1.internal:9443",
            "oc1",
            "prod",
            None,
        )
        .expect("peer should parse");
        let registry = RegionPeerRegistry::new(vec![peer]).expect("registry should build");
        let forwarder = CrossRegionForwarder::new(registry);

        for path in ["/v1/chat/completions", "/v1/responses"] {
            let request = ForwardingRequest::new(path, b"{}".to_vec())
                .expect("supported SMG path should parse");
            let target = forwarder
                .forwarding_target_for_decision(&remote_decision("us-chicago-1"), &request)
                .expect("target should resolve");

            assert_eq!(target.url.path(), path);
        }
    }

    #[test]
    fn forwarding_target_rejects_route_target_region_mismatch() {
        let peer = RegionPeer::new(
            "us-chicago-1",
            "https://smg-region-agent.us-chicago-1.internal:8443",
            "https://smg-region-agent.us-chicago-1.internal:9443",
            "oc1",
            "prod",
            None,
        )
        .expect("peer should parse");
        let registry = RegionPeerRegistry::new(vec![peer]).expect("registry should build");
        let forwarder = CrossRegionForwarder::new(registry);
        let mut decision = remote_decision("us-chicago-1");
        decision.target_region = "us-phoenix-1".to_string();
        let request = ForwardingRequest::new("/v1/chat/completions", b"{}".to_vec())
            .expect("path should parse");

        let error = forwarder
            .forwarding_target_for_decision(&decision, &request)
            .expect_err("route metadata mismatch should fail");

        assert!(error.to_string().contains("target_region"));
    }

    #[test]
    fn forwarding_target_debug_hides_joined_path() {
        let peer = RegionPeer::new(
            "us-chicago-1",
            "https://smg-region-agent.us-chicago-1.internal:8443",
            "https://smg-region-agent.us-chicago-1.internal:9443",
            "oc1",
            "prod",
            None,
        )
        .expect("peer should parse");
        let registry = RegionPeerRegistry::new(vec![peer]).expect("registry should build");
        let forwarder = CrossRegionForwarder::new(registry);
        let request = ForwardingRequest::new("/v1/chat/completions", b"{}".to_vec())
            .expect("path should parse");

        let debug = format!(
            "{:?}",
            forwarder
                .forwarding_target_for_decision(&remote_decision("us-chicago-1"), &request)
                .expect("target should resolve")
        );

        assert!(debug.contains("endpoint_origin"));
        assert!(debug.contains("expected_mtls_identity"));
        assert!(!debug.contains("/v1/chat/completions"));
        assert!(!debug.contains("url"));
    }

    #[test]
    fn forwarding_request_rejects_raw_worker_url_as_path() {
        let error = ForwardingRequest::new(
            "https://remote-worker.us-chicago-1.internal:8000/v1/chat/completions",
            b"{}".to_vec(),
        )
        .expect_err("raw worker URL should be rejected");

        assert!(error.to_string().contains("not a URL"));
    }

    #[test]
    fn forwarding_request_rejects_scheme_relative_worker_url_as_path() {
        let error = ForwardingRequest::new(
            "//remote-worker.us-chicago-1.internal:8000/v1/chat/completions",
            b"{}".to_vec(),
        )
        .expect_err("scheme-relative worker URL should be rejected");

        assert!(error.to_string().contains("not a URL"));
    }

    #[test]
    fn remote_forward_retryable_statuses_match_cross_region_contract() {
        for status in [
            StatusCode::REQUEST_TIMEOUT,
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_GATEWAY,
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::GATEWAY_TIMEOUT,
        ] {
            assert!(is_retryable_remote_forward_status(status));
        }

        for status in [
            StatusCode::BAD_REQUEST,
            StatusCode::UNAUTHORIZED,
            StatusCode::FORBIDDEN,
            StatusCode::NOT_FOUND,
        ] {
            assert!(!is_retryable_remote_forward_status(status));
        }
    }

    #[test]
    fn manual_failover_mode_does_not_retry_remote_forwarding() {
        let policy = FailoverPolicy::new(CrossRegionFailoverMode::Manual, 3);

        assert!(!should_failover_remote_forward(
            &policy,
            StatusCode::SERVICE_UNAVAILABLE,
            false,
            1
        ));
        assert!(should_record_remote_forward_failure(
            StatusCode::SERVICE_UNAVAILABLE,
            false
        ));
    }

    #[test]
    fn auto_failover_retries_retryable_non_streaming_response_within_budget() {
        let policy = FailoverPolicy::new(CrossRegionFailoverMode::Automatic, 1);

        assert!(should_failover_remote_forward(
            &policy,
            StatusCode::SERVICE_UNAVAILABLE,
            false,
            1
        ));
        assert!(!should_failover_remote_forward(
            &policy,
            StatusCode::SERVICE_UNAVAILABLE,
            false,
            2
        ));
        assert!(should_record_remote_forward_failure(
            StatusCode::SERVICE_UNAVAILABLE,
            false
        ));
    }

    #[test]
    fn streaming_response_is_not_retried_after_start() {
        let policy = FailoverPolicy::new(CrossRegionFailoverMode::Automatic, 3);

        assert!(!should_failover_remote_forward(
            &policy,
            StatusCode::SERVICE_UNAVAILABLE,
            true,
            1
        ));
        assert!(!should_record_remote_forward_failure(
            StatusCode::SERVICE_UNAVAILABLE,
            true
        ));
    }

    #[test]
    fn retryable_status_opens_breaker_even_when_failover_is_suppressed() {
        for policy in [
            FailoverPolicy::new(CrossRegionFailoverMode::Manual, 3),
            FailoverPolicy::new(CrossRegionFailoverMode::Automatic, 0),
        ] {
            let breaker = crate::cross_region::CrossRegionBreaker::with_failure_threshold(1);

            assert!(!should_failover_remote_forward(
                &policy,
                StatusCode::SERVICE_UNAVAILABLE,
                false,
                1
            ));
            if should_record_remote_forward_failure(StatusCode::SERVICE_UNAVAILABLE, false) {
                breaker.record_failure("us-chicago-1");
            }

            assert_eq!(
                breaker.state_for("us-chicago-1"),
                crate::cross_region::BreakerState::Open
            );
        }
    }

    #[test]
    fn forwarder_rejects_local_execution_target() {
        let forwarder = CrossRegionForwarder::new(RegionPeerRegistry::empty());
        let decision = RegionRouteDecision {
            route_id: "route-1".to_string(),
            target_region: "us-ashburn-1".to_string(),
            model_id: "cohere.command-r-plus".to_string(),
            execution_target: ExecutionTarget::LocalRegion,
        };

        let error = forwarder
            .request_target_for_decision(&decision)
            .expect_err("local target should be rejected");

        assert!(error.to_string().contains("local router path"));
    }
}
