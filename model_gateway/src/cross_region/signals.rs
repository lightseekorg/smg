use openai_protocol::worker::{WorkerLoadInfo, WorkerStatus};
use serde::{Deserialize, Serialize};

/// Version for Phase 1 cross-region signal contracts.
pub const SIGNAL_CONTRACT_VERSION: u32 = 1;

/// Key forms used by the cross-region signal sync plane.
///
/// Every signal is **per-replica**: keys carry a trailing `server_name`
/// identifying the publishing replica. Consumers aggregate across replicas
/// at ranking time (sum for load, union/intersection for health/readiness,
/// histogram merge for latency).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SignalKey {
    SmgReadiness {
        region_id: String,
        server_name: String,
    },
    WorkerHealth {
        region_id: String,
        worker_id: String,
        server_name: String,
    },
    WorkerLoad {
        region_id: String,
        worker_id: String,
        server_name: String,
    },
    ClientLatency {
        client_region: String,
        target_region: String,
        server_name: String,
    },
}

impl SignalKey {
    /// Return the stable storage key path for this signal.
    pub fn as_path(&self) -> String {
        match self {
            Self::SmgReadiness {
                region_id,
                server_name,
            } => format!("smg-readiness/{region_id}/{server_name}"),
            Self::WorkerHealth {
                region_id,
                worker_id,
                server_name,
            } => {
                format!("worker-health/{region_id}/{worker_id}/{server_name}")
            }
            Self::WorkerLoad {
                region_id,
                worker_id,
                server_name,
            } => {
                format!("worker-load/{region_id}/{worker_id}/{server_name}")
            }
            Self::ClientLatency {
                client_region,
                target_region,
                server_name,
            } => {
                format!("client-latency/{client_region}/{target_region}/{server_name}")
            }
        }
    }

    /// Region this key belongs to (for the readiness/health/load kinds this
    /// is the owning region; for client-latency it is the client/entry region).
    /// The producer's region-ownership invariant checks this against
    /// `CrossRegionContext::region_id` before publishing.
    pub fn region_segment(&self) -> &str {
        match self {
            Self::SmgReadiness { region_id, .. }
            | Self::WorkerHealth { region_id, .. }
            | Self::WorkerLoad { region_id, .. } => region_id,
            Self::ClientLatency { client_region, .. } => client_region,
        }
    }

    /// Publishing replica's `server_name` (the trailing key segment).
    /// The producer's invariant checks this against `CrossRegionContext::server_name`.
    pub fn server_name_segment(&self) -> &str {
        match self {
            Self::SmgReadiness { server_name, .. }
            | Self::WorkerHealth { server_name, .. }
            | Self::WorkerLoad { server_name, .. }
            | Self::ClientLatency { server_name, .. } => server_name,
        }
    }
}

/// Generic signal wrapper that carries versioning, identity, and freshness
/// metadata. Apply order is `(version, actor)` lexicographic (see design §2c).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignalEnvelope<T> {
    pub key: SignalKey,
    /// Wall-clock-anchored monotonic version.
    /// Assigned by `CrossRegionSyncService::publish_signal` as
    /// `max(now_ms(), prev.version + 1, observed_remote_max_for_key + 1)`.
    pub version: u64,
    /// Writing replica identity (`server_name`). Tiebreaker when two writes
    /// share a `version` (e.g., misconfigured duplicate `server_name` or
    /// restart overlap); see design §2c.
    pub actor: String,
    /// Wall-clock timestamp at publish time.
    pub generated_at_ms: i64,
    /// Freshness bound: consumers exclude this entry from ranking when
    /// `now_ms() - generated_at_ms > stale_after_ms`.
    /// Set to `0` for tombstones (`removed == true`).
    pub stale_after_ms: u32,
    /// Soft-delete flag. When `true`, `signal` carries no payload data and
    /// consumers must exclude the key from aggregation.
    pub removed: bool,
    /// Kind-specific body. `None` when `removed == true`.
    pub signal: Option<T>,
}

/// Local SMG readiness signal for one region.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SmgReadinessSignal {
    pub region_id: String,
    pub server_name: String,
    pub ready: bool,
}

/// Worker health signal for one worker in one region.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerHealthSignal {
    pub region_id: String,
    pub worker_id: String,
    pub server_name: String,
    pub status: WorkerStatus,
}

/// Worker load signal for one worker in one region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerLoadSignal {
    pub region_id: String,
    pub worker_id: String,
    pub server_name: String,
    pub load: WorkerLoadInfo,
}

/// Client-observed latency from one client region to one target region.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientLatencySignal {
    pub client_region: String,
    pub target_region: String,
    pub server_name: String,
    pub p50_latency_ms: u64,
    pub p95_latency_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_key_paths_match_per_replica_contract() {
        assert_eq!(
            SignalKey::SmgReadiness {
                region_id: "us-ashburn-1".to_string(),
                server_name: "smg-router-a".to_string(),
            }
            .as_path(),
            "smg-readiness/us-ashburn-1/smg-router-a"
        );
        assert_eq!(
            SignalKey::WorkerHealth {
                region_id: "us-ashburn-1".to_string(),
                worker_id: "0190ee2b-0001-7000-8000-000000000001".to_string(),
                server_name: "smg-router-a".to_string(),
            }
            .as_path(),
            "worker-health/us-ashburn-1/0190ee2b-0001-7000-8000-000000000001/smg-router-a"
        );
        assert_eq!(
            SignalKey::ClientLatency {
                client_region: "us-phoenix-1".to_string(),
                target_region: "us-chicago-1".to_string(),
                server_name: "smg-router-b".to_string(),
            }
            .as_path(),
            "client-latency/us-phoenix-1/us-chicago-1/smg-router-b"
        );
    }

    #[test]
    fn signal_key_segment_accessors() {
        let key = SignalKey::WorkerLoad {
            region_id: "us-ashburn-1".to_string(),
            worker_id: "w-1".to_string(),
            server_name: "smg-router-a".to_string(),
        };
        assert_eq!(key.region_segment(), "us-ashburn-1");
        assert_eq!(key.server_name_segment(), "smg-router-a");

        let latency = SignalKey::ClientLatency {
            client_region: "us-phoenix-1".to_string(),
            target_region: "us-chicago-1".to_string(),
            server_name: "smg-router-c".to_string(),
        };
        // For client-latency, region_segment is the client/entry region.
        assert_eq!(latency.region_segment(), "us-phoenix-1");
        assert_eq!(latency.server_name_segment(), "smg-router-c");
    }

    #[test]
    fn readiness_signal_serializes_with_envelope_fields() {
        let envelope = SignalEnvelope {
            key: SignalKey::SmgReadiness {
                region_id: "us-ashburn-1".to_string(),
                server_name: "smg-router-a".to_string(),
            },
            version: 1,
            actor: "smg-router-a".to_string(),
            generated_at_ms: 42,
            stale_after_ms: 30_000,
            removed: false,
            signal: Some(SmgReadinessSignal {
                region_id: "us-ashburn-1".to_string(),
                server_name: "smg-router-a".to_string(),
                ready: true,
            }),
        };

        let json = serde_json::to_string(&envelope).expect("serialize signal");

        assert!(json.contains("smg_readiness"));
        assert!(json.contains("us-ashburn-1"));
        assert!(json.contains("smg-router-a"));
        assert!(json.contains("\"actor\":\"smg-router-a\""));
        assert!(json.contains("\"stale_after_ms\":30000"));
        assert!(json.contains("\"removed\":false"));
    }

    #[test]
    fn tombstone_envelope_has_no_signal_payload() {
        let envelope: SignalEnvelope<SmgReadinessSignal> = SignalEnvelope {
            key: SignalKey::SmgReadiness {
                region_id: "us-ashburn-1".to_string(),
                server_name: "smg-router-a".to_string(),
            },
            version: 2,
            actor: "smg-router-a".to_string(),
            generated_at_ms: 100,
            stale_after_ms: 0,
            removed: true,
            signal: None,
        };

        assert!(envelope.signal.is_none());
        assert!(envelope.removed);
    }
}
