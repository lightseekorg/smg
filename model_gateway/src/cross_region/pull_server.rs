//! Cross-region sync pull endpoint.
//!
//! Peers call this endpoint with their last retained cursor. The handler is
//! stateless: it validates the already-authenticated peer identity, reads the
//! local sync log, and returns either a snapshot (`cursor == 0`) or a delta.

use std::sync::Arc;

use axum::{
    extract::{Extension, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};

use super::{
    AuthenticatedPeerIdentity, CrossRegionError, CrossRegionSyncService, Cursor,
    RegionPeerRegistry, SignalEnvelope, SignalKind,
};

/// Request body for `POST /v1/cross-region/pull`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PullRequest {
    pub cursor: Cursor,
}

/// Successful sync pull response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullResponse {
    pub envelopes: Vec<SignalEnvelope<SignalKind>>,
    pub next_cursor: Cursor,
}

/// Stable error body for sync pull failures.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PullErrorResponse {
    pub error: String,
}

/// Shared state for the pull endpoint.
#[derive(Debug, Clone)]
pub struct PullServerState {
    sync: Arc<CrossRegionSyncService>,
    peers: RegionPeerRegistry,
}

impl PullServerState {
    pub fn new(sync: Arc<CrossRegionSyncService>, peers: RegionPeerRegistry) -> Self {
        Self { sync, peers }
    }
}

/// Build the cross-region sync pull router.
pub fn router(state: PullServerState) -> Router {
    Router::new()
        .route("/v1/cross-region/pull", post(pull_signals))
        .with_state(state)
}

/// Axum handler for the sync pull endpoint.
pub async fn pull_signals(
    State(state): State<PullServerState>,
    Extension(peer_identity): Extension<AuthenticatedPeerIdentity>,
    Json(request): Json<PullRequest>,
) -> Response {
    match handle_pull(&state, &peer_identity, request) {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(error) => error.into_response(),
    }
}

/// Pure pull implementation used by the handler and unit tests.
pub fn handle_pull(
    state: &PullServerState,
    peer_identity: &AuthenticatedPeerIdentity,
    request: PullRequest,
) -> Result<PullResponse, PullServerError> {
    validate_peer(peer_identity, &state.peers)?;

    let (envelopes, next_cursor) = if request.cursor == 0 {
        state.sync.local_log_snapshot()
    } else {
        state
            .sync
            .local_log_delta(request.cursor)
            .map_err(|_| PullServerError::CursorStale)?
    };

    Ok(PullResponse {
        envelopes,
        next_cursor,
    })
}

fn validate_peer(
    peer_identity: &AuthenticatedPeerIdentity,
    peers: &RegionPeerRegistry,
) -> Result<(), PullServerError> {
    let expected_identity = peers
        .expected_mtls_identity(peer_identity.region_id())
        .map_err(PullServerError::from_cross_region)?;

    if peer_identity.mtls_identity() != expected_identity {
        return Err(PullServerError::Unauthorized {
            reason: "authenticated peer mTLS identity does not match configured peer".to_string(),
        });
    }

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PullServerError {
    CursorStale,
    Unauthorized { reason: String },
}

impl PullServerError {
    fn from_cross_region(error: CrossRegionError) -> Self {
        Self::Unauthorized {
            reason: error.to_string(),
        }
    }
}

impl IntoResponse for PullServerError {
    fn into_response(self) -> Response {
        match self {
            Self::CursorStale => (
                StatusCode::CONFLICT,
                Json(PullErrorResponse {
                    error: "cursor-stale".to_string(),
                }),
            )
                .into_response(),
            Self::Unauthorized { reason } => (
                StatusCode::FORBIDDEN,
                Json(PullErrorResponse { error: reason }),
            )
                .into_response(),
        }
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::WorkerStatus;

    use super::*;
    use crate::cross_region::{SignalKey, SmgReadinessSignal, SyncRetention, WorkerHealthSignal};

    const LOCAL_REGION: &str = "us-ashburn-1";
    const LOCAL_SERVER: &str = "smg-router-a";
    const PEER_REGION: &str = "us-chicago-1";
    const PEER_IDENTITY: &str =
        "spiffe://oraclecorp.com/oci/oc1/prod/region/us-chicago-1/service/smg-region-agent";

    fn state() -> PullServerState {
        let sync = Arc::new(
            CrossRegionSyncService::new(LOCAL_REGION.to_string(), LOCAL_SERVER.to_string())
                .expect("sync service should build"),
        );
        PullServerState::new(sync, peer_registry(true))
    }

    fn state_with_retention(retention: SyncRetention) -> PullServerState {
        let sync = Arc::new(
            CrossRegionSyncService::new_with_retention(
                LOCAL_REGION.to_string(),
                LOCAL_SERVER.to_string(),
                retention,
            )
            .expect("sync service should build"),
        );
        PullServerState::new(sync, peer_registry(true))
    }

    fn peer_registry(enabled: bool) -> RegionPeerRegistry {
        let peer = crate::cross_region::RegionPeer::new(
            PEER_REGION,
            "https://smg-region-agent.us-chicago-1.internal:8443",
            "https://smg-region-agent.us-chicago-1.internal:9443",
            "oc1",
            "prod",
            None,
        )
        .expect("peer should build")
        .with_enabled(enabled);
        RegionPeerRegistry::new(vec![peer]).expect("registry should build")
    }

    fn peer_identity() -> AuthenticatedPeerIdentity {
        AuthenticatedPeerIdentity::new(PEER_REGION, PEER_IDENTITY).expect("identity should build")
    }

    fn publish_readiness(state: &PullServerState) {
        state
            .sync
            .publish_signal(
                SignalKey::SmgReadiness {
                    region_id: LOCAL_REGION.to_string(),
                    server_name: LOCAL_SERVER.to_string(),
                },
                SignalKind::SmgReadiness(SmgReadinessSignal {
                    region_id: LOCAL_REGION.to_string(),
                    server_name: LOCAL_SERVER.to_string(),
                    ready: true,
                }),
                30_000,
            )
            .expect("publish should succeed");
    }

    fn publish_health(state: &PullServerState) {
        state
            .sync
            .publish_signal(
                SignalKey::WorkerHealth {
                    region_id: LOCAL_REGION.to_string(),
                    worker_id: "worker-1".to_string(),
                    server_name: LOCAL_SERVER.to_string(),
                },
                SignalKind::WorkerHealth(WorkerHealthSignal {
                    region_id: LOCAL_REGION.to_string(),
                    worker_id: "worker-1".to_string(),
                    server_name: LOCAL_SERVER.to_string(),
                    status: WorkerStatus::Ready,
                }),
                30_000,
            )
            .expect("publish should succeed");
    }

    #[test]
    fn cursor_zero_returns_snapshot() {
        let state = state();
        publish_readiness(&state);
        publish_health(&state);

        let response = handle_pull(&state, &peer_identity(), PullRequest { cursor: 0 })
            .expect("snapshot should succeed");

        assert_eq!(response.envelopes.len(), 2);
        assert_eq!(response.next_cursor, 2);
    }

    #[test]
    fn nonzero_cursor_returns_delta() {
        let state = state();
        publish_readiness(&state);
        let cursor = handle_pull(&state, &peer_identity(), PullRequest { cursor: 0 })
            .expect("snapshot should succeed")
            .next_cursor;
        publish_health(&state);

        let response = handle_pull(&state, &peer_identity(), PullRequest { cursor })
            .expect("delta should succeed");

        assert_eq!(response.envelopes.len(), 1);
        assert_eq!(response.next_cursor, 2);
        assert!(matches!(
            response.envelopes[0].signal,
            Some(SignalKind::WorkerHealth(_))
        ));
    }

    #[test]
    fn stale_cursor_returns_cursor_stale() {
        let state = state_with_retention(SyncRetention {
            tombstone_retention_ms: 1,
            dead_replica_retention_ms: 1,
        });
        publish_readiness(&state);
        publish_health(&state);
        state.sync.gc_log(i64::MAX);
        publish_readiness(&state);

        let snapshot = handle_pull(&state, &peer_identity(), PullRequest { cursor: 0 })
            .expect("cursor zero snapshot should not be stale");
        assert_eq!(snapshot.envelopes.len(), 1);
        assert_eq!(snapshot.next_cursor, 3);

        let error = handle_pull(&state, &peer_identity(), PullRequest { cursor: 1 })
            .expect_err("missing cursor 1 entry should be stale");
        assert_eq!(error, PullServerError::CursorStale);
    }

    #[test]
    fn disabled_peer_is_rejected() {
        let state = PullServerState::new(
            Arc::new(
                CrossRegionSyncService::new(LOCAL_REGION.to_string(), LOCAL_SERVER.to_string())
                    .expect("sync service should build"),
            ),
            peer_registry(false),
        );

        let error = handle_pull(&state, &peer_identity(), PullRequest { cursor: 0 })
            .expect_err("disabled peer should be rejected");

        assert!(matches!(error, PullServerError::Unauthorized { .. }));
    }

    #[test]
    fn mismatched_peer_identity_is_rejected() {
        let state = state();
        let identity = AuthenticatedPeerIdentity::new(
            PEER_REGION,
            "spiffe://oraclecorp.com/oci/oc1/prod/region/us-phoenix-1/service/smg-region-agent",
        )
        .expect("identity should build");

        let error =
            handle_pull(&state, &identity, PullRequest { cursor: 0 }).expect_err("reject identity");

        assert_eq!(
            error,
            PullServerError::Unauthorized {
                reason: "authenticated peer mTLS identity does not match configured peer"
                    .to_string()
            }
        );
    }
}
