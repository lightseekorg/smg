use openai_protocol::worker::{WorkerLoadInfo, WorkerStatus};
use serde::{Deserialize, Serialize};

/// Version for Phase 1 cross-region signal contracts.
pub const SIGNAL_CONTRACT_VERSION: u32 = 1;

/// Mesh/AppStore key forms used by the Phase 1 signal sync plane.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SignalKey {
    SmgReadiness {
        region_id: String,
    },
    WorkerHealth {
        region_id: String,
        worker_id: String,
    },
    WorkerLoad {
        region_id: String,
        worker_id: String,
    },
    ClientLatency {
        client_region: String,
        target_region: String,
    },
}

impl SignalKey {
    /// Return the stable storage key path for this signal.
    pub fn as_path(&self) -> String {
        match self {
            Self::SmgReadiness { region_id } => format!("smg-readiness/{region_id}"),
            Self::WorkerHealth {
                region_id,
                worker_id,
            } => {
                format!("worker-health/{region_id}/{worker_id}")
            }
            Self::WorkerLoad {
                region_id,
                worker_id,
            } => {
                format!("worker-load/{region_id}/{worker_id}")
            }
            Self::ClientLatency {
                client_region,
                target_region,
            } => {
                format!("client-latency/{client_region}/{target_region}")
            }
        }
    }
}

/// Generic signal wrapper that carries version and generation metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignalEnvelope<T> {
    pub key: SignalKey,
    pub version: u64,
    pub generated_at_ms: i64,
    pub signal: T,
}

/// Local SMG readiness signal for one region.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SmgReadinessSignal {
    pub region_id: String,
    pub ready: bool,
}

/// Worker health signal for one worker in one region.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerHealthSignal {
    pub region_id: String,
    pub worker_id: String,
    pub status: WorkerStatus,
}

/// Worker load signal for one worker in one region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerLoadSignal {
    pub region_id: String,
    pub worker_id: String,
    pub load: WorkerLoadInfo,
}

/// Client-observed latency from one client region to one target region.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientLatencySignal {
    pub client_region: String,
    pub target_region: String,
    pub p50_latency_ms: u64,
    pub p95_latency_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_key_paths_match_phase1_contract() {
        assert_eq!(
            SignalKey::SmgReadiness {
                region_id: "us-ashburn-1".to_string()
            }
            .as_path(),
            "smg-readiness/us-ashburn-1"
        );
        assert_eq!(
            SignalKey::ClientLatency {
                client_region: "us-phoenix-1".to_string(),
                target_region: "us-chicago-1".to_string()
            }
            .as_path(),
            "client-latency/us-phoenix-1/us-chicago-1"
        );
    }

    #[test]
    fn readiness_signal_serializes_with_key_envelope() {
        let envelope = SignalEnvelope {
            key: SignalKey::SmgReadiness {
                region_id: "us-ashburn-1".to_string(),
            },
            version: 1,
            generated_at_ms: 42,
            signal: SmgReadinessSignal {
                region_id: "us-ashburn-1".to_string(),
                ready: true,
            },
        };

        let json = serde_json::to_string(&envelope).expect("serialize signal");

        assert!(json.contains("smg_readiness"));
        assert!(json.contains("us-ashburn-1"));
    }
}
