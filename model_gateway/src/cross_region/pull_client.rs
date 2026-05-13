//! Cross-region sync pull client.
//!
//! Each peer gets one task that repeatedly calls the peer's sync endpoint,
//! applies returned envelopes, and advances its cursor. The transport is
//! factored behind a small trait so cursor/reset behavior is unit-testable
//! without standing up HTTP.

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use reqwest::{Client, StatusCode};
use tokio::task::JoinHandle;
use url::Url;

use super::{
    CrossRegionResult, CrossRegionSyncService, Cursor, PullRequest, PullResponse,
    RegionPeerRegistry,
};

const PULL_PATH: &str = "/v1/cross-region/pull";

/// Pull-client retry timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PullClientConfig {
    pub pull_interval: Duration,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
}

impl Default for PullClientConfig {
    fn default() -> Self {
        Self {
            pull_interval: Duration::from_millis(500),
            initial_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(30),
        }
    }
}

/// Result of one transport request.
#[derive(Debug)]
pub enum PullTransportResponse {
    Ok(PullResponse),
    CursorStale,
}

/// Pull transport abstraction. Production uses HTTP; tests inject a scripted
/// transport.
#[async_trait]
pub trait PullTransport: Send + Sync + 'static {
    async fn pull(
        &self,
        sync_url: &Url,
        request: PullRequest,
    ) -> Result<PullTransportResponse, PullClientError>;
}

/// Reqwest-backed transport. The client should be constructed with the
/// sync-plane mTLS identity once that wiring lands.
#[derive(Debug, Clone)]
pub struct HttpPullTransport {
    client: Client,
}

impl HttpPullTransport {
    pub fn new(client: Client) -> Self {
        Self { client }
    }
}

#[async_trait]
impl PullTransport for HttpPullTransport {
    async fn pull(
        &self,
        sync_url: &Url,
        request: PullRequest,
    ) -> Result<PullTransportResponse, PullClientError> {
        let url = sync_url
            .join(PULL_PATH)
            .map_err(|error| PullClientError::Transport {
                reason: format!("invalid sync pull URL: {error}"),
            })?;
        let response = self
            .client
            .post(url)
            .json(&request)
            .send()
            .await
            .map_err(|error| PullClientError::Transport {
                reason: error.to_string(),
            })?;

        match response.status() {
            StatusCode::OK => response
                .json::<PullResponse>()
                .await
                .map(PullTransportResponse::Ok)
                .map_err(|error| PullClientError::Decode {
                    reason: error.to_string(),
                }),
            StatusCode::CONFLICT => Ok(PullTransportResponse::CursorStale),
            status => Err(PullClientError::HttpStatus {
                status: status.as_u16(),
            }),
        }
    }
}

/// One peer's cursor-bearing pull task.
#[derive(Debug)]
pub struct PeerPullTask<T> {
    peer_region_id: String,
    sync_url: Url,
    sync: Arc<CrossRegionSyncService>,
    transport: T,
    cursor: Cursor,
}

impl<T> PeerPullTask<T>
where
    T: PullTransport,
{
    pub fn new(
        peer_region_id: String,
        sync_url: Url,
        sync: Arc<CrossRegionSyncService>,
        transport: T,
    ) -> Self {
        Self {
            peer_region_id,
            sync_url,
            sync,
            transport,
            cursor: 0,
        }
    }

    /// Return the cursor that will be sent on the next request.
    pub fn cursor(&self) -> Cursor {
        self.cursor
    }

    /// Execute one pull cycle. A stale cursor is handled inside this method by
    /// immediately retrying with `cursor = 0` for a full snapshot.
    pub async fn pull_once(&mut self) -> Result<PullStepOutcome, PullClientError> {
        match self
            .transport
            .pull(
                &self.sync_url,
                PullRequest {
                    cursor: self.cursor,
                },
            )
            .await?
        {
            PullTransportResponse::Ok(response) => {
                self.apply_response(response);
                Ok(PullStepOutcome::Applied)
            }
            PullTransportResponse::CursorStale => {
                self.cursor = 0;
                let response = match self
                    .transport
                    .pull(&self.sync_url, PullRequest { cursor: 0 })
                    .await?
                {
                    PullTransportResponse::Ok(response) => response,
                    PullTransportResponse::CursorStale => {
                        return Err(PullClientError::Protocol {
                            reason: "cursor=0 snapshot returned cursor-stale".to_string(),
                        });
                    }
                };
                self.apply_response(response);
                Ok(PullStepOutcome::Resynced)
            }
        }
    }

    async fn run_forever(mut self, config: PullClientConfig) {
        let mut backoff = config.initial_backoff;
        loop {
            match self.pull_once().await {
                Ok(_) => {
                    backoff = config.initial_backoff;
                    tokio::time::sleep(config.pull_interval).await;
                }
                Err(error) => {
                    tracing::warn!(
                        peer_region = %self.peer_region_id,
                        error = %error,
                        "cross-region sync pull failed"
                    );
                    tokio::time::sleep(backoff).await;
                    backoff = doubled_duration(backoff).min(config.max_backoff);
                }
            }
        }
    }

    fn apply_response(&mut self, response: PullResponse) {
        self.sync
            .apply_remote_envelopes(&self.peer_region_id, &response.envelopes);
        self.cursor = response.next_cursor;
    }
}

/// Outcome from a successful pull cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PullStepOutcome {
    Applied,
    Resynced,
}

/// Errors surfaced by the pull client.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PullClientError {
    #[error("cross-region sync transport error: {reason}")]
    Transport { reason: String },
    #[error("cross-region sync peer returned HTTP {status}")]
    HttpStatus { status: u16 },
    #[error("cross-region sync response decode failed: {reason}")]
    Decode { reason: String },
    #[error("cross-region sync protocol violation: {reason}")]
    Protocol { reason: String },
}

/// Owns the spawned peer pull tasks.
#[derive(Debug)]
pub struct PullClientOrchestrator {
    tasks: Vec<JoinHandle<()>>,
}

impl PullClientOrchestrator {
    #[expect(
        clippy::disallowed_methods,
        reason = "tasks are bounded by PullClientOrchestrator which aborts on drop"
    )]
    pub fn start(
        sync: Arc<CrossRegionSyncService>,
        peers: &RegionPeerRegistry,
        client: Client,
        config: PullClientConfig,
    ) -> CrossRegionResult<Self> {
        let tasks = peers
            .sync_targets()?
            .into_iter()
            .map(|target| {
                tracing::debug!(
                    peer_region = %target.region_id(),
                    expected_mtls_identity = %target.expected_mtls_identity(),
                    "starting cross-region sync pull task"
                );
                let task = PeerPullTask::new(
                    target.region_id().to_string(),
                    target.sync_url().clone(),
                    sync.clone(),
                    HttpPullTransport::new(client.clone()),
                );
                tokio::spawn(task.run_forever(config))
            })
            .collect();

        Ok(Self { tasks })
    }

    /// Abort every pull task. Idempotent.
    pub fn abort(&self) {
        for task in &self.tasks {
            task.abort();
        }
    }

    /// Take ownership of the underlying join handles.
    pub fn detach(mut self) -> Vec<JoinHandle<()>> {
        std::mem::take(&mut self.tasks)
    }
}

impl Drop for PullClientOrchestrator {
    fn drop(&mut self) {
        self.abort();
    }
}

fn doubled_duration(duration: Duration) -> Duration {
    duration.checked_mul(2).unwrap_or(Duration::MAX)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use parking_lot::Mutex;

    use super::*;
    use crate::cross_region::{SignalKey, SignalKind, SmgReadinessSignal};

    const LOCAL_REGION: &str = "us-ashburn-1";
    const LOCAL_SERVER: &str = "smg-router-a";
    const PEER_REGION: &str = "us-chicago-1";
    const PEER_SERVER: &str = "smg-router-b";

    #[derive(Debug, Default)]
    struct ScriptedTransport {
        responses: Mutex<VecDeque<Result<PullTransportResponse, PullClientError>>>,
        requests: Mutex<Vec<PullRequest>>,
    }

    impl ScriptedTransport {
        fn new(responses: Vec<Result<PullTransportResponse, PullClientError>>) -> Self {
            Self {
                responses: Mutex::new(VecDeque::from(responses)),
                requests: Mutex::new(Vec::new()),
            }
        }

        fn requests(&self) -> Vec<PullRequest> {
            self.requests.lock().clone()
        }
    }

    #[async_trait]
    impl PullTransport for Arc<ScriptedTransport> {
        async fn pull(
            &self,
            _sync_url: &Url,
            request: PullRequest,
        ) -> Result<PullTransportResponse, PullClientError> {
            self.requests.lock().push(request);
            self.responses
                .lock()
                .pop_front()
                .expect("scripted response should exist")
        }
    }

    fn local_sync() -> Arc<CrossRegionSyncService> {
        Arc::new(
            CrossRegionSyncService::new(LOCAL_REGION.to_string(), LOCAL_SERVER.to_string())
                .expect("sync service should build"),
        )
    }

    fn peer_envelope() -> PullResponse {
        PullResponse {
            envelopes: vec![crate::cross_region::SignalEnvelope {
                key: SignalKey::SmgReadiness {
                    region_id: PEER_REGION.to_string(),
                    server_name: PEER_SERVER.to_string(),
                },
                version: 10,
                actor: PEER_SERVER.to_string(),
                generated_at_ms: 1_000,
                stale_after_ms: 30_000,
                removed: false,
                signal: Some(SignalKind::SmgReadiness(SmgReadinessSignal {
                    region_id: PEER_REGION.to_string(),
                    server_name: PEER_SERVER.to_string(),
                    ready: true,
                })),
            }],
            next_cursor: 7,
        }
    }

    fn task(
        sync: Arc<CrossRegionSyncService>,
        transport: Arc<ScriptedTransport>,
    ) -> PeerPullTask<Arc<ScriptedTransport>> {
        PeerPullTask::new(
            PEER_REGION.to_string(),
            Url::parse("https://smg-region-agent.us-chicago-1.internal:9443")
                .expect("url should parse"),
            sync,
            transport,
        )
    }

    #[tokio::test]
    async fn pull_once_applies_envelopes_and_advances_cursor() {
        let sync = local_sync();
        let transport = Arc::new(ScriptedTransport::new(vec![Ok(PullTransportResponse::Ok(
            peer_envelope(),
        ))]));
        let mut task = task(sync.clone(), transport.clone());

        let outcome = task.pull_once().await.expect("pull should succeed");

        assert_eq!(outcome, PullStepOutcome::Applied);
        assert_eq!(task.cursor(), 7);
        assert_eq!(transport.requests(), vec![PullRequest { cursor: 0 }]);
        let state = sync.state();
        let state = state.read();
        assert!(
            state.readiness_replica(PEER_REGION, PEER_SERVER).is_some(),
            "pulled readiness should materialize into local state"
        );
    }

    #[tokio::test]
    async fn stale_cursor_retries_with_snapshot_and_advances_cursor() {
        let sync = local_sync();
        let transport = Arc::new(ScriptedTransport::new(vec![
            Ok(PullTransportResponse::CursorStale),
            Ok(PullTransportResponse::Ok(peer_envelope())),
        ]));
        let mut task = task(sync, transport.clone());
        task.cursor = 42;

        let outcome = task.pull_once().await.expect("resync should succeed");

        assert_eq!(outcome, PullStepOutcome::Resynced);
        assert_eq!(task.cursor(), 7);
        assert_eq!(
            transport.requests(),
            vec![PullRequest { cursor: 42 }, PullRequest { cursor: 0 }]
        );
    }

    #[tokio::test]
    async fn transport_error_does_not_advance_cursor() {
        let sync = local_sync();
        let transport = Arc::new(ScriptedTransport::new(vec![Err(
            PullClientError::Transport {
                reason: "connection refused".to_string(),
            },
        )]));
        let mut task = task(sync, transport);
        task.cursor = 42;

        let error = task.pull_once().await.expect_err("transport should fail");

        assert_eq!(
            error,
            PullClientError::Transport {
                reason: "connection refused".to_string()
            }
        );
        assert_eq!(task.cursor(), 42);
    }

    #[tokio::test]
    async fn cursor_zero_conflict_is_protocol_error() {
        let sync = local_sync();
        let transport = Arc::new(ScriptedTransport::new(vec![
            Ok(PullTransportResponse::CursorStale),
            Ok(PullTransportResponse::CursorStale),
        ]));
        let mut task = task(sync, transport);
        task.cursor = 42;

        let error = task.pull_once().await.expect_err("protocol error expected");

        assert_eq!(
            error,
            PullClientError::Protocol {
                reason: "cursor=0 snapshot returned cursor-stale".to_string()
            }
        );
        assert_eq!(task.cursor(), 0);
    }
}
