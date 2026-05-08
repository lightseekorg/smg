use std::collections::BTreeSet;

use openai_protocol::{
    model_type::{Endpoint, ModelType},
    worker::WorkerStatus,
};
use serde::{Deserialize, Serialize};

use super::{
    CandidateGatedReason, CrossRegionBreaker, CrossRegionError, CrossRegionResult,
    CrossRegionState, ModalityPolicy, RegionPeerRegistry, RemoteRegionView, RequestMode,
    RoutingProfileContext,
};
use crate::{
    config::CrossRegionFailoverMode,
    worker::{Worker, WorkerRegistry},
};

/// Default maximum age for remote materialized signals used by candidate gating.
pub const DEFAULT_REMOTE_SIGNAL_MAX_AGE_MS: i64 = 30_000;

/// Region-level execution target. Remote variants carry a region, never a worker URL.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExecutionTarget {
    LocalRegion,
    RemoteRegion { region_id: String },
}

/// Region-level route decision emitted by cross-region candidate calculation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionRouteDecision {
    pub route_id: String,
    pub target_region: String,
    pub model_id: String,
    pub execution_target: ExecutionTarget,
}

/// Committed route metadata attached to settled remote requests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteCommit {
    pub route_id: String,
    pub entry_region: String,
    pub target_region: String,
    pub model_id: String,
    pub request_mode: RequestMode,
    pub attempt: u32,
    pub failover_mode: CrossRegionFailoverMode,
}

/// Inspectable route decision context for log and metric emission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteDecisionLogContext {
    pub route_id: String,
    pub entry_region: String,
    pub target_region: String,
    pub model_id: String,
    pub request_mode: RequestMode,
    pub attempt: u32,
    pub failover_mode: CrossRegionFailoverMode,
}

/// Candidate region scoring input produced before a route decision is committed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionCandidate {
    pub region_id: String,
    pub model_id: String,
    pub endpoint_type: Endpoint,
    pub modality: ModalityPolicy,
    pub readiness: bool,
    pub worker_health: WorkerHealthSummary,
    pub healthy: bool,
    pub has_capacity: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_latency_hint_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_load: Option<isize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub freshness_age_ms: Option<i64>,
}

/// Ranking configuration isolated so future weight changes stay local.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateRankingConfig {
    pub missing_client_latency_hint_ms: u64,
    pub missing_worker_load: isize,
}

impl Default for CandidateRankingConfig {
    /// Create the default ranking configuration for absent advisory signals.
    fn default() -> Self {
        Self {
            missing_client_latency_hint_ms: u64::MAX,
            missing_worker_load: isize::MAX,
        }
    }
}

/// Region-level worker health summary carried by a candidate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerHealthSummary {
    pub ready_workers: usize,
    pub total_workers: usize,
}

/// Stable rejection reasons emitted while hard filters gate candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateRejectionReason {
    RegionNotAllowed,
    ModelNotAllowed,
    EndpointUnsupported,
    ModalityUnsupported,
    MissingRemoteSignal,
    StaleRemoteSignal,
    RemoteBreakerOpen,
    PeerUnavailable,
}

impl CandidateRejectionReason {
    /// Map a detailed candidate rejection into the bounded metric label set.
    pub fn metric_reason(self) -> CandidateGatedReason {
        match self {
            Self::RegionNotAllowed
            | Self::ModelNotAllowed
            | Self::EndpointUnsupported
            | Self::ModalityUnsupported => CandidateGatedReason::PolicyMismatch,
            Self::MissingRemoteSignal | Self::StaleRemoteSignal => {
                CandidateGatedReason::StaleSignal
            }
            Self::RemoteBreakerOpen => CandidateGatedReason::BreakerOpen,
            Self::PeerUnavailable => CandidateGatedReason::PeerUnavailable,
        }
    }
}

/// Region/model candidate that was rejected by a hard filter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateRejection {
    pub region_id: String,
    pub model_id: String,
    pub reason: CandidateRejectionReason,
}

impl CandidateRejection {
    /// Return the bounded metric reason associated with this rejection.
    pub fn metric_reason(&self) -> CandidateGatedReason {
        self.reason.metric_reason()
    }
}

/// Candidate construction result with eligible candidates and gated regions.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateCalculationOutput {
    pub candidates: Vec<RegionCandidate>,
    pub rejections: Vec<CandidateRejection>,
}

/// Input bundle for region-level candidate construction and filtering.
#[derive(Debug)]
pub struct CandidateCalculationInput<'a> {
    pub profile: RoutingProfileContext,
    pub local_region: String,
    pub endpoint_type: Endpoint,
    pub local_worker_registry: &'a WorkerRegistry,
    pub remote_state: &'a CrossRegionState,
    pub peer_registry: &'a RegionPeerRegistry,
    pub breaker: &'a CrossRegionBreaker,
    pub client_region: Option<String>,
    pub now_ms: i64,
}

/// Input bundle for deterministic route ranking and commitment.
#[derive(Debug)]
pub struct RouteCalculationInput<'a> {
    pub route_id: String,
    pub entry_region: String,
    pub request_mode: RequestMode,
    pub attempt: u32,
    pub candidate_input: CandidateCalculationInput<'a>,
}

/// Output bundle for a committed cross-region route decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteCalculationOutput {
    pub decision: RegionRouteDecision,
    pub commit: RouteCommit,
    pub log_context: RouteDecisionLogContext,
}

/// Candidate calculator for Phase 1 regional candidate construction.
#[derive(Debug, Clone)]
pub struct CandidateCalculator {
    enabled: bool,
    remote_signal_max_age_ms: i64,
    ranking_config: CandidateRankingConfig,
}

impl Default for CandidateCalculator {
    /// Create the default no-op calculator boundary.
    fn default() -> Self {
        Self {
            enabled: true,
            remote_signal_max_age_ms: DEFAULT_REMOTE_SIGNAL_MAX_AGE_MS,
            ranking_config: CandidateRankingConfig::default(),
        }
    }
}

impl CandidateCalculator {
    /// Create a candidate calculator with the default remote freshness threshold.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the maximum accepted remote signal age.
    pub fn with_remote_signal_max_age_ms(mut self, max_age_ms: i64) -> Self {
        self.remote_signal_max_age_ms = max_age_ms;
        self
    }

    /// Override ranking behavior for absent advisory signal values.
    pub fn with_ranking_config(mut self, ranking_config: CandidateRankingConfig) -> Self {
        self.ranking_config = ranking_config;
        self
    }

    /// Build region-level candidates and hard-filter rejections.
    pub fn build_candidates(
        &self,
        input: CandidateCalculationInput<'_>,
    ) -> CrossRegionResult<CandidateCalculationOutput> {
        validate_input(&input)?;
        if !self.enabled {
            return Ok(CandidateCalculationOutput::default());
        }

        let model_id = input.profile.single_model_id()?.to_string();
        let model_type_hint = derive_model_type_hint(input.local_worker_registry, &model_id);
        let allowed_regions = allowed_region_set(&input.profile);
        let mut output = CandidateCalculationOutput::default();

        for region_id in candidate_region_ids(&input) {
            if !allowed_regions.contains(region_id.as_str()) {
                output.rejections.push(rejection(
                    region_id,
                    model_id.clone(),
                    CandidateRejectionReason::RegionNotAllowed,
                ));
                continue;
            }

            let candidate = if region_id == input.local_region {
                Self::build_local_candidate(&input, &model_id)
            } else {
                self.build_remote_candidate(&input, &region_id, &model_id, model_type_hint)
            };

            match candidate {
                Ok(candidate) => output.candidates.push(candidate),
                Err(reason) => {
                    output
                        .rejections
                        .push(rejection(region_id, model_id.clone(), reason));
                }
            }
        }

        Ok(output)
    }

    /// Rank eligible region candidates and commit the best route.
    pub fn calculate(
        &self,
        input: RouteCalculationInput<'_>,
    ) -> CrossRegionResult<Option<RouteCalculationOutput>> {
        validate_route_input(&input)?;
        if !self.enabled {
            return Ok(None);
        }

        let profile = input.candidate_input.profile.clone();
        let local_region = input.candidate_input.local_region.clone();
        let model_id = profile.single_model_id()?.to_string();
        let failover_mode = profile.failover_policy.failover_mode;
        let candidate_output = self.build_candidates(input.candidate_input)?;
        let maybe_candidate = self.best_candidate(
            &candidate_output.candidates,
            &profile,
            &model_id,
            &local_region,
        );
        let Some(selected_candidate) = maybe_candidate else {
            return Ok(None);
        };

        let target_region = selected_candidate.region_id.clone();
        let decision = RegionRouteDecision {
            route_id: input.route_id.clone(),
            target_region: target_region.clone(),
            model_id,
            execution_target: execution_target_for(&target_region, &local_region),
        };
        let commit = RouteCommit::from_decision(
            decision.clone(),
            input.entry_region,
            input.request_mode,
            input.attempt,
            failover_mode,
        );
        let log_context = RouteDecisionLogContext::from_commit(&commit);

        Ok(Some(RouteCalculationOutput {
            decision,
            commit,
            log_context,
        }))
    }

    /// Return the best eligible candidate according to SMG-11 ranking order.
    fn best_candidate<'a>(
        &self,
        candidates: &'a [RegionCandidate],
        profile: &RoutingProfileContext,
        model_id: &str,
        local_region: &str,
    ) -> Option<&'a RegionCandidate> {
        candidates
            .iter()
            .filter(|candidate| candidate_matches_profile(candidate, profile, model_id))
            .min_by(|left, right| self.compare_candidates(left, right, local_region))
    }

    /// Compare candidates in the configured deterministic ranking order.
    fn compare_candidates(
        &self,
        left: &RegionCandidate,
        right: &RegionCandidate,
        local_region: &str,
    ) -> std::cmp::Ordering {
        right
            .healthy
            .cmp(&left.healthy)
            .then_with(|| right.has_capacity.cmp(&left.has_capacity))
            .then_with(|| self.latency_rank(left).cmp(&self.latency_rank(right)))
            .then_with(|| {
                self.worker_load_rank(left)
                    .cmp(&self.worker_load_rank(right))
            })
            .then_with(|| {
                let left_is_local = left.region_id == local_region;
                let right_is_local = right.region_id == local_region;
                right_is_local.cmp(&left_is_local)
            })
            .then_with(|| left.region_id.cmp(&right.region_id))
    }

    /// Return the comparable client-latency rank for a candidate.
    fn latency_rank(&self, candidate: &RegionCandidate) -> u64 {
        candidate
            .client_latency_hint_ms
            .unwrap_or(self.ranking_config.missing_client_latency_hint_ms)
    }

    /// Return the comparable worker-load rank for a candidate.
    fn worker_load_rank(&self, candidate: &RegionCandidate) -> isize {
        candidate
            .worker_load
            .unwrap_or(self.ranking_config.missing_worker_load)
    }

    /// Build a local-region candidate from the local worker registry.
    fn build_local_candidate(
        input: &CandidateCalculationInput<'_>,
        model_id: &str,
    ) -> Result<RegionCandidate, CandidateRejectionReason> {
        let model_workers = input
            .local_worker_registry
            .get_all()
            .into_iter()
            .filter(|worker| worker.supports_model(model_id))
            .collect::<Vec<_>>();
        if model_workers.is_empty() {
            return Err(CandidateRejectionReason::ModelNotAllowed);
        }

        let endpoint_workers = model_workers
            .into_iter()
            .filter(|worker| worker.supports_endpoint(model_id, input.endpoint_type))
            .collect::<Vec<_>>();
        if endpoint_workers.is_empty() {
            return Err(CandidateRejectionReason::EndpointUnsupported);
        }

        let modality_workers = endpoint_workers
            .into_iter()
            .filter(|worker| worker_supports_modality(worker, model_id, &input.profile.modality))
            .collect::<Vec<_>>();
        if modality_workers.is_empty() {
            return Err(CandidateRejectionReason::ModalityUnsupported);
        }

        let worker_health = worker_health_summary(
            modality_workers.iter().map(|worker| worker.status()),
            modality_workers.len(),
        );
        let worker_load = modality_workers.iter().fold(0isize, |load, worker| {
            load.saturating_add(usize_to_isize(worker.load()))
        });
        let has_capacity = modality_workers.iter().any(|worker| worker.is_available());

        Ok(RegionCandidate {
            region_id: input.local_region.clone(),
            model_id: model_id.to_string(),
            endpoint_type: input.endpoint_type,
            modality: input.profile.modality.clone(),
            readiness: true,
            worker_health,
            healthy: worker_health.ready_workers > 0,
            has_capacity,
            client_latency_hint_ms: None,
            worker_load: Some(worker_load),
            freshness_age_ms: None,
        })
    }

    /// Build a remote-region candidate from materialized sync-plane signals.
    fn build_remote_candidate(
        &self,
        input: &CandidateCalculationInput<'_>,
        region_id: &str,
        model_id: &str,
        model_type_hint: Option<ModelType>,
    ) -> Result<RegionCandidate, CandidateRejectionReason> {
        input
            .peer_registry
            .request_target(region_id)
            .map_err(|_| CandidateRejectionReason::PeerUnavailable)?;

        if !input.breaker.can_attempt(region_id) {
            return Err(CandidateRejectionReason::RemoteBreakerOpen);
        }

        let model_type = model_type_hint.unwrap_or(ModelType::LLM);
        if !model_type.supports_endpoint(input.endpoint_type) {
            return Err(CandidateRejectionReason::EndpointUnsupported);
        }
        if !model_type_supports_modality(model_type, &input.profile.modality) {
            return Err(CandidateRejectionReason::ModalityUnsupported);
        }

        // Phase F: route every read through `RemoteRegionView`. The view
        // applies the per-replica aggregation rules (any-fresh-ready,
        // any-fresh-routable, sum loads by model, min p50 latency) and the
        // freshness window in one place. `freshness_age_ms` here is the
        // worst-case (oldest) contributor across every projection we consult.
        let view = RemoteRegionView::new(
            input.remote_state,
            input.now_ms,
            self.remote_signal_max_age_ms,
        );

        let Some(readiness) = view.readiness(region_id) else {
            if view.has_readiness_replica(region_id) {
                return Err(CandidateRejectionReason::StaleRemoteSignal);
            }
            return Err(CandidateRejectionReason::MissingRemoteSignal);
        };
        let readiness_ready = readiness.ready;
        let mut freshness_age_ms = readiness.freshness_age_ms;

        let worker_ids = view.worker_ids(region_id);
        if worker_ids.is_empty() {
            return Err(CandidateRejectionReason::MissingRemoteSignal);
        }

        let mut selected_statuses = Vec::new();
        let mut selected_load = 0isize;
        let mut saw_model_mismatch = false;
        let mut saw_stale_signal = false;

        for worker_id in worker_ids {
            let Some(projection) = view.worker(region_id, worker_id) else {
                continue;
            };

            let load_resolution = projection.load_for_model(model_id);
            let Some(load_total) = load_resolution.total else {
                if load_resolution.saw_model_mismatch {
                    saw_model_mismatch = true;
                }
                if load_resolution.saw_stale_match {
                    saw_stale_signal = true;
                }
                continue;
            };
            if let Some(age) = load_resolution.freshness_age_ms {
                freshness_age_ms = freshness_age_ms.max(age);
            }

            let status_resolution = projection.aggregated_status();
            let Some(status) = status_resolution.status else {
                if status_resolution.saw_stale_only {
                    saw_stale_signal = true;
                }
                continue;
            };
            if let Some(age) = status_resolution.freshness_age_ms {
                freshness_age_ms = freshness_age_ms.max(age);
            }

            selected_statuses.push(status);
            selected_load = selected_load.saturating_add(load_total);
        }

        if selected_statuses.is_empty() {
            if saw_stale_signal {
                return Err(CandidateRejectionReason::StaleRemoteSignal);
            }
            if saw_model_mismatch {
                return Err(CandidateRejectionReason::ModelNotAllowed);
            }
            return Err(CandidateRejectionReason::MissingRemoteSignal);
        }

        let mut client_latency_hint_ms = None;
        if let Some(client_region) = input.client_region.as_deref() {
            match view.client_latency(client_region, region_id) {
                Some(latency) => {
                    freshness_age_ms = freshness_age_ms.max(latency.freshness_age_ms);
                    client_latency_hint_ms = Some(latency.min_p50_ms);
                }
                None => {
                    if view.has_client_latency_replica(client_region, region_id) {
                        return Err(CandidateRejectionReason::StaleRemoteSignal);
                    }
                }
            }
        }

        let total_workers = selected_statuses.len();
        let worker_health = worker_health_summary(selected_statuses, total_workers);
        let healthy = readiness_ready && worker_health.ready_workers > 0;

        Ok(RegionCandidate {
            region_id: region_id.to_string(),
            model_id: model_id.to_string(),
            endpoint_type: input.endpoint_type,
            modality: input.profile.modality.clone(),
            readiness: readiness_ready,
            worker_health,
            healthy,
            has_capacity: worker_health.ready_workers > 0,
            client_latency_hint_ms,
            worker_load: Some(selected_load),
            freshness_age_ms: Some(freshness_age_ms),
        })
    }
}

/// Validate candidate calculation inputs before reading routing state.
fn validate_input(input: &CandidateCalculationInput<'_>) -> CrossRegionResult<()> {
    if input.local_region.trim().is_empty() {
        return Err(CrossRegionError::InvalidConfig {
            reason: "local_region must not be empty".to_string(),
        });
    }
    if input.now_ms < 0 {
        return Err(CrossRegionError::InvalidConfig {
            reason: "now_ms must not be negative".to_string(),
        });
    }
    input.profile.validate()
}

/// Validate route-commit inputs before ranking candidate output.
fn validate_route_input(input: &RouteCalculationInput<'_>) -> CrossRegionResult<()> {
    if input.entry_region.trim().is_empty() {
        return Err(CrossRegionError::InvalidConfig {
            reason: "entry_region must not be empty".to_string(),
        });
    }
    if input.route_id.trim().is_empty() {
        return Err(CrossRegionError::InvalidConfig {
            reason: "route_id must not be empty".to_string(),
        });
    }
    Ok(())
}

/// Return every region that can produce a candidate or rejection in stable order.
fn candidate_region_ids(input: &CandidateCalculationInput<'_>) -> Vec<String> {
    let mut region_ids = BTreeSet::new();
    region_ids.insert(input.local_region.clone());
    region_ids.extend(input.profile.allowed_regions.iter().cloned());
    region_ids.extend(input.remote_state.regions().into_iter().map(str::to_string));
    region_ids.into_iter().collect()
}

/// Return allowed regions as a stable lookup set.
fn allowed_region_set(profile: &RoutingProfileContext) -> BTreeSet<String> {
    profile.allowed_regions.iter().cloned().collect()
}

/// Build a rejection record for one region/model tuple.
fn rejection(
    region_id: String,
    model_id: String,
    reason: CandidateRejectionReason,
) -> CandidateRejection {
    CandidateRejection {
        region_id,
        model_id,
        reason,
    }
}

/// Derive deterministic local capability hints for remote candidate filtering.
fn derive_model_type_hint(registry: &WorkerRegistry, model_id: &str) -> Option<ModelType> {
    let mut model_types = registry
        .get_all()
        .into_iter()
        .flat_map(|worker| {
            worker
                .models()
                .into_iter()
                .filter(move |model| model.matches(model_id))
                .map(|model| model.model_type)
        })
        .collect::<Vec<_>>();

    model_types.sort_unstable_by_key(|model_type| model_type.bits());
    model_types.dedup();

    let mut model_types = model_types.into_iter();
    let mut capability_intersection = model_types.next()?;
    for model_type in model_types {
        capability_intersection &= model_type;
    }

    Some(capability_intersection)
}

/// Return true when a worker's model capabilities support the requested modalities.
fn worker_supports_modality(
    worker: &std::sync::Arc<dyn Worker>,
    model_id: &str,
    modality: &ModalityPolicy,
) -> bool {
    let model_type = worker
        .models()
        .into_iter()
        .find(|model| model.matches(model_id))
        .map(|model| model.model_type)
        .unwrap_or(ModelType::LLM);
    model_type_supports_modality(model_type, modality)
}

/// Return true when a model capability bitset supports the requested modalities.
fn model_type_supports_modality(model_type: ModelType, modality: &ModalityPolicy) -> bool {
    modality
        .input
        .as_deref()
        .is_none_or(|input| supports_input_modality(model_type, input))
        && modality
            .output
            .as_deref()
            .is_none_or(|output| supports_output_modality(model_type, output))
}

/// Return true when the model can consume the requested input modality.
fn supports_input_modality(model_type: ModelType, modality: &str) -> bool {
    match normalize_modality(modality).as_deref() {
        Some("text") => true,
        Some("image" | "vision") => model_type.supports_vision(),
        Some("audio") => model_type.supports_audio(),
        _ => false,
    }
}

/// Return true when the model can produce the requested output modality.
fn supports_output_modality(model_type: ModelType, modality: &str) -> bool {
    match normalize_modality(modality).as_deref() {
        Some("text") => true,
        Some("image" | "image_gen") => model_type.supports_image_gen(),
        Some("audio") => model_type.supports_audio(),
        _ => false,
    }
}

/// Normalize a modality header value for capability matching.
fn normalize_modality(modality: &str) -> Option<String> {
    let value = modality.trim().to_ascii_lowercase().replace('-', "_");
    (!value.is_empty()).then_some(value)
}

/// Build a worker health summary from candidate worker statuses.
fn worker_health_summary(
    statuses: impl IntoIterator<Item = WorkerStatus>,
    total_workers: usize,
) -> WorkerHealthSummary {
    let ready_workers = statuses
        .into_iter()
        .filter(|status| status.is_routable())
        .count();
    WorkerHealthSummary {
        ready_workers,
        total_workers,
    }
}

/// Convert a usize load counter into isize without overflowing.
fn usize_to_isize(value: usize) -> isize {
    match isize::try_from(value) {
        Ok(value) => value,
        Err(_) => isize::MAX,
    }
}

impl RouteCommit {
    /// Build committed route metadata from a route decision and request metadata.
    pub fn from_decision(
        decision: RegionRouteDecision,
        entry_region: impl Into<String>,
        request_mode: RequestMode,
        attempt: u32,
        failover_mode: CrossRegionFailoverMode,
    ) -> Self {
        Self {
            route_id: decision.route_id,
            entry_region: entry_region.into(),
            target_region: decision.target_region,
            model_id: decision.model_id,
            request_mode,
            attempt,
            failover_mode,
        }
    }
}

impl RouteDecisionLogContext {
    /// Build log and metric context from committed route metadata.
    pub fn from_commit(commit: &RouteCommit) -> Self {
        Self {
            route_id: commit.route_id.clone(),
            entry_region: commit.entry_region.clone(),
            target_region: commit.target_region.clone(),
            model_id: commit.model_id.clone(),
            request_mode: commit.request_mode,
            attempt: commit.attempt,
            failover_mode: commit.failover_mode,
        }
    }
}

/// Return true when a candidate belongs to the active Phase 1 profile.
fn candidate_matches_profile(
    candidate: &RegionCandidate,
    profile: &RoutingProfileContext,
    model_id: &str,
) -> bool {
    candidate.model_id == model_id
        && profile
            .allowed_regions
            .iter()
            .any(|region| region == &candidate.region_id)
}

/// Build the execution target without ever carrying a worker URL.
fn execution_target_for(target_region: &str, local_region: &str) -> ExecutionTarget {
    if target_region == local_region {
        ExecutionTarget::LocalRegion
    } else {
        ExecutionTarget::RemoteRegion {
            region_id: target_region.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::{
        model_card::ModelCard,
        model_type::{Endpoint, ModelType},
        worker::{WorkerLoadInfo, WorkerStatus},
    };

    use super::*;
    use crate::{
        cross_region::{
            BreakerState, CrossRegionBreaker, CrossRegionState, FailoverPolicy, ModalityPolicy,
            RegionPeer, RegionPeerRegistry, SignalVersion, SmgReadinessSignal, WorkerHealthSignal,
            WorkerLoadSignal,
        },
        worker::{BasicWorkerBuilder, Worker, WorkerRegistry},
    };

    #[test]
    fn route_decision_serializes_without_worker_url() {
        let decision = RegionRouteDecision {
            route_id: "route-1".to_string(),
            target_region: "us-chicago-1".to_string(),
            model_id: "cohere.command-r-plus".to_string(),
            execution_target: ExecutionTarget::RemoteRegion {
                region_id: "us-chicago-1".to_string(),
            },
        };

        let json = serde_json::to_string(&decision).expect("serialize route decision");

        assert!(json.contains("us-chicago-1"));
        assert!(!json.contains("worker_url"));
    }

    #[test]
    fn route_calculation_commits_remote_region_without_worker_url() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            10,
        );
        let mut remote_state = CrossRegionState::new();
        add_remote_worker(
            &mut remote_state,
            "us-chicago-1",
            "remote-worker-a",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            1,
            NOW_MS,
        );
        add_client_latency(&mut remote_state, "us-chicago-1", 20, NOW_MS);

        let output = calculate_route(
            &registry,
            &remote_state,
            peer_registry(&["us-chicago-1"]),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1", "us-chicago-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        )
        .expect("route should be committed");
        let commit_json = serde_json::to_string(&output.commit).expect("serialize commit");

        assert_eq!(output.decision.target_region, "us-chicago-1");
        assert_eq!(output.commit.entry_region, "us-ashburn-1");
        assert_eq!(output.commit.model_id, "cohere.command-r-plus");
        assert_eq!(output.log_context.route_id, "route-1");
        assert_eq!(output.log_context.target_region, "us-chicago-1");
        assert!(!commit_json.contains("worker_url"));
        assert!(!commit_json.contains("http://"));
        assert!(!commit_json.contains("https://"));
        assert_eq!(
            output.decision.execution_target,
            ExecutionTarget::RemoteRegion {
                region_id: "us-chicago-1".to_string()
            }
        );
    }

    #[test]
    fn route_ranking_is_deterministic_with_stable_signal_state() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            50,
        );
        let mut remote_state = CrossRegionState::new();
        add_remote_worker(
            &mut remote_state,
            "us-phoenix-1",
            "remote-worker-phx",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            20,
            NOW_MS,
        );
        add_remote_worker(
            &mut remote_state,
            "us-chicago-1",
            "remote-worker-ord",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            20,
            NOW_MS,
        );
        add_client_latency(&mut remote_state, "us-phoenix-1", 30, NOW_MS);
        add_client_latency(&mut remote_state, "us-chicago-1", 20, NOW_MS);

        let profile = profile(
            &["us-ashburn-1", "us-chicago-1", "us-phoenix-1"],
            "cohere.command-r-plus",
            ModalityPolicy::default(),
        );
        let first = route_target_region(
            &registry,
            &remote_state,
            peer_registry(&["us-chicago-1", "us-phoenix-1"]),
            CrossRegionBreaker::new(),
            profile.clone(),
            Endpoint::Chat,
        );
        let second = route_target_region(
            &registry,
            &remote_state,
            peer_registry(&["us-chicago-1", "us-phoenix-1"]),
            CrossRegionBreaker::new(),
            profile,
            Endpoint::Chat,
        );

        assert_eq!(first, "us-chicago-1");
        assert_eq!(first, second);
    }

    #[test]
    fn healthy_candidates_outrank_unhealthy_candidates() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            100,
        );
        let mut remote_state = CrossRegionState::new();
        add_remote_worker(
            &mut remote_state,
            "us-chicago-1",
            "remote-worker-a",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            1,
            NOW_MS,
        );
        remote_state.upsert_readiness(
            SmgReadinessSignal {
                region_id: "us-chicago-1".to_string(),
                ready: false,
            },
            signal_version(NOW_MS),
        );
        add_client_latency(&mut remote_state, "us-chicago-1", 1, NOW_MS);

        let target_region = route_target_region(
            &registry,
            &remote_state,
            peer_registry(&["us-chicago-1"]),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1", "us-chicago-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );

        assert_eq!(target_region, "us-ashburn-1");
    }

    #[test]
    fn available_capacity_order_is_respected() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Pending,
            1,
        );
        let mut remote_state = CrossRegionState::new();
        add_remote_worker(
            &mut remote_state,
            "us-chicago-1",
            "remote-worker-a",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            100,
            NOW_MS,
        );
        remote_state.upsert_readiness(
            SmgReadinessSignal {
                region_id: "us-chicago-1".to_string(),
                ready: false,
            },
            signal_version(NOW_MS),
        );

        let target_region = route_target_region(
            &registry,
            &remote_state,
            peer_registry(&["us-chicago-1"]),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1", "us-chicago-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );

        assert_eq!(target_region, "us-chicago-1");
    }

    #[test]
    fn lower_client_latency_wins_before_worker_load() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            0,
        );
        let mut remote_state = CrossRegionState::new();
        add_remote_worker(
            &mut remote_state,
            "us-chicago-1",
            "remote-worker-ord",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            90,
            NOW_MS,
        );
        add_remote_worker(
            &mut remote_state,
            "us-phoenix-1",
            "remote-worker-phx",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            1,
            NOW_MS,
        );
        add_client_latency(&mut remote_state, "us-chicago-1", 20, NOW_MS);
        add_client_latency(&mut remote_state, "us-phoenix-1", 40, NOW_MS);

        let target_region = route_target_region(
            &registry,
            &remote_state,
            peer_registry(&["us-chicago-1", "us-phoenix-1"]),
            CrossRegionBreaker::new(),
            profile(
                &["us-chicago-1", "us-phoenix-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );

        assert_eq!(target_region, "us-chicago-1");
    }

    #[test]
    fn lower_worker_load_wins_when_higher_order_fields_match() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            0,
        );
        let mut remote_state = CrossRegionState::new();
        add_remote_worker(
            &mut remote_state,
            "us-chicago-1",
            "remote-worker-ord",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            90,
            NOW_MS,
        );
        add_remote_worker(
            &mut remote_state,
            "us-phoenix-1",
            "remote-worker-phx",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            1,
            NOW_MS,
        );
        add_client_latency(&mut remote_state, "us-chicago-1", 20, NOW_MS);
        add_client_latency(&mut remote_state, "us-phoenix-1", 20, NOW_MS);

        let target_region = route_target_region(
            &registry,
            &remote_state,
            peer_registry(&["us-chicago-1", "us-phoenix-1"]),
            CrossRegionBreaker::new(),
            profile(
                &["us-chicago-1", "us-phoenix-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );

        assert_eq!(target_region, "us-phoenix-1");
    }

    #[test]
    fn local_first_tie_break_applies_only_to_equivalent_outcomes() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            10,
        );
        let mut tied_remote_state = CrossRegionState::new();
        add_remote_worker(
            &mut tied_remote_state,
            "us-chicago-1",
            "remote-worker-a",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            10,
            NOW_MS,
        );
        let mut lower_latency_remote_state = tied_remote_state.clone();
        add_client_latency(&mut lower_latency_remote_state, "us-chicago-1", 10, NOW_MS);

        let tied_target_region = route_target_region(
            &registry,
            &tied_remote_state,
            peer_registry(&["us-chicago-1"]),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1", "us-chicago-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );
        let lower_latency_target_region = route_target_region(
            &registry,
            &lower_latency_remote_state,
            peer_registry(&["us-chicago-1"]),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1", "us-chicago-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );

        assert_eq!(tied_target_region, "us-ashburn-1");
        assert_eq!(lower_latency_target_region, "us-chicago-1");
    }

    #[test]
    fn stable_region_id_tie_break_prevents_flapping() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            0,
        );
        let mut remote_state = CrossRegionState::new();
        add_remote_worker(
            &mut remote_state,
            "us-phoenix-1",
            "remote-worker-phx",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            10,
            NOW_MS,
        );
        add_remote_worker(
            &mut remote_state,
            "us-chicago-1",
            "remote-worker-ord",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            10,
            NOW_MS,
        );

        let target_region = route_target_region(
            &registry,
            &remote_state,
            peer_registry(&["us-phoenix-1", "us-chicago-1"]),
            CrossRegionBreaker::new(),
            profile(
                &["us-phoenix-1", "us-chicago-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );

        assert_eq!(target_region, "us-chicago-1");
    }

    #[test]
    fn allowed_region_filter_excludes_regions_outside_profile() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            0,
        );
        let mut remote_state = CrossRegionState::new();
        add_remote_worker(
            &mut remote_state,
            "us-chicago-1",
            "remote-worker-a",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            1,
            NOW_MS,
        );
        let output = build_candidates(
            &registry,
            &remote_state,
            peer_registry(&["us-chicago-1"]),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );

        assert_eq!(candidate_regions(&output), vec!["us-ashburn-1"]);
        assert!(output.rejections.iter().any(|rejection| {
            rejection.region_id == "us-chicago-1"
                && rejection.reason == CandidateRejectionReason::RegionNotAllowed
        }));
    }

    #[test]
    fn single_allowed_model_filter_excludes_other_model_workers() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("meta.llama-3", ModelType::LLM),
            WorkerStatus::Ready,
            0,
        );
        let output = build_candidates(
            &registry,
            &CrossRegionState::new(),
            RegionPeerRegistry::empty(),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );

        assert!(output.candidates.is_empty());
        assert!(output.rejections.iter().any(|rejection| {
            rejection.region_id == "us-ashburn-1"
                && rejection.reason == CandidateRejectionReason::ModelNotAllowed
        }));
    }

    #[test]
    fn remote_candidate_rejects_missing_model_id_signal() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            0,
        );
        let mut remote_state = CrossRegionState::new();
        add_remote_worker(
            &mut remote_state,
            "us-chicago-1",
            "remote-worker-a",
            None,
            WorkerStatus::Ready,
            1,
            NOW_MS,
        );
        let output = build_candidates(
            &registry,
            &remote_state,
            peer_registry(&["us-chicago-1"]),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1", "us-chicago-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );

        assert_eq!(candidate_regions(&output), vec!["us-ashburn-1"]);
        assert!(output.rejections.iter().any(|rejection| {
            rejection.region_id == "us-chicago-1"
                && rejection.reason == CandidateRejectionReason::MissingRemoteSignal
        }));
    }

    #[test]
    fn endpoint_filter_accepts_chat_completions_models() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::CHAT),
            WorkerStatus::Ready,
            0,
        );
        let output = build_candidates(
            &registry,
            &CrossRegionState::new(),
            RegionPeerRegistry::empty(),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );

        assert_eq!(candidate_regions(&output), vec!["us-ashburn-1"]);
        assert_eq!(output.candidates[0].endpoint_type, Endpoint::Chat);
    }

    #[test]
    fn endpoint_filter_rejects_responses_when_model_only_supports_chat() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::CHAT),
            WorkerStatus::Ready,
            0,
        );
        let output = build_candidates(
            &registry,
            &CrossRegionState::new(),
            RegionPeerRegistry::empty(),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Responses,
        );

        assert!(output.candidates.is_empty());
        assert!(output.rejections.iter().any(|rejection| {
            rejection.reason == CandidateRejectionReason::EndpointUnsupported
        }));
    }

    #[test]
    fn endpoint_filter_accepts_responses_models() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::RESPONSES),
            WorkerStatus::Ready,
            0,
        );
        let output = build_candidates(
            &registry,
            &CrossRegionState::new(),
            RegionPeerRegistry::empty(),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Responses,
        );

        assert_eq!(candidate_regions(&output), vec!["us-ashburn-1"]);
        assert_eq!(output.candidates[0].endpoint_type, Endpoint::Responses);
    }

    #[test]
    fn remote_endpoint_filter_is_deterministic_for_conflicting_local_model_cards() {
        let registry = registry_with_workers(vec![
            (
                "http://local-chat-worker:8000",
                model_card("cohere.command-r-plus", ModelType::CHAT),
                WorkerStatus::Ready,
                0,
            ),
            (
                "http://local-responses-worker:8000",
                model_card("cohere.command-r-plus", ModelType::RESPONSES),
                WorkerStatus::Ready,
                0,
            ),
        ]);
        let mut remote_state = CrossRegionState::new();
        add_remote_worker(
            &mut remote_state,
            "us-chicago-1",
            "remote-worker-a",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            1,
            NOW_MS,
        );
        let output = build_candidates(
            &registry,
            &remote_state,
            peer_registry(&["us-chicago-1"]),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1", "us-chicago-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Responses,
        );

        assert_eq!(candidate_regions(&output), vec!["us-ashburn-1"]);
        assert!(output.rejections.iter().any(|rejection| {
            rejection.region_id == "us-chicago-1"
                && rejection.reason == CandidateRejectionReason::EndpointUnsupported
        }));
    }

    #[test]
    fn modality_filter_rejects_vision_input_for_text_only_model() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            0,
        );
        let output = build_candidates(
            &registry,
            &CrossRegionState::new(),
            RegionPeerRegistry::empty(),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1"],
                "cohere.command-r-plus",
                ModalityPolicy {
                    input: Some("image".to_string()),
                    output: Some("text".to_string()),
                },
            ),
            Endpoint::Chat,
        );

        assert!(output.candidates.is_empty());
        assert!(output.rejections.iter().any(|rejection| {
            rejection.reason == CandidateRejectionReason::ModalityUnsupported
        }));
    }

    #[test]
    fn stale_remote_signal_excludes_remote_candidate() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            0,
        );
        let mut remote_state = CrossRegionState::new();
        add_remote_worker(
            &mut remote_state,
            "us-chicago-1",
            "remote-worker-a",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            1,
            NOW_MS - DEFAULT_REMOTE_SIGNAL_MAX_AGE_MS - 1,
        );
        let output = build_candidates(
            &registry,
            &remote_state,
            peer_registry(&["us-chicago-1"]),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1", "us-chicago-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );

        assert_eq!(candidate_regions(&output), vec!["us-ashburn-1"]);
        assert!(output.rejections.iter().any(|rejection| {
            rejection.region_id == "us-chicago-1"
                && rejection.reason == CandidateRejectionReason::StaleRemoteSignal
        }));
    }

    #[test]
    fn missing_peer_excludes_remote_candidate() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            0,
        );
        let mut remote_state = CrossRegionState::new();
        add_remote_worker(
            &mut remote_state,
            "us-chicago-1",
            "remote-worker-a",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            1,
            NOW_MS,
        );
        let output = build_candidates(
            &registry,
            &remote_state,
            RegionPeerRegistry::empty(),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1", "us-chicago-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );

        assert_eq!(candidate_regions(&output), vec!["us-ashburn-1"]);
        assert!(output.rejections.iter().any(|rejection| {
            rejection.region_id == "us-chicago-1"
                && rejection.reason == CandidateRejectionReason::PeerUnavailable
        }));
    }

    #[test]
    fn remote_breaker_excludes_remote_candidate() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            0,
        );
        let mut remote_state = CrossRegionState::new();
        add_remote_worker(
            &mut remote_state,
            "us-chicago-1",
            "remote-worker-a",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            1,
            NOW_MS,
        );
        let output = build_candidates(
            &registry,
            &remote_state,
            peer_registry(&["us-chicago-1"]),
            CrossRegionBreaker::with_default_state(BreakerState::Open),
            profile(
                &["us-ashburn-1", "us-chicago-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );

        assert_eq!(candidate_regions(&output), vec!["us-ashburn-1"]);
        assert!(output.rejections.iter().any(|rejection| {
            rejection.region_id == "us-chicago-1"
                && rejection.reason == CandidateRejectionReason::RemoteBreakerOpen
        }));
    }

    #[test]
    fn candidate_output_contains_region_model_data_without_worker_url_target() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            3,
        );
        let output = build_candidates(
            &registry,
            &CrossRegionState::new(),
            RegionPeerRegistry::empty(),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );
        let candidate = output
            .candidates
            .first()
            .expect("local candidate should be built");
        let json = serde_json::to_string(candidate).expect("serialize candidate");

        assert_eq!(candidate.region_id, "us-ashburn-1");
        assert_eq!(candidate.model_id, "cohere.command-r-plus");
        assert_eq!(candidate.worker_load, Some(3));
        assert!(!json.contains("worker_url"));
        assert!(!json.contains("local-worker"));
    }

    #[test]
    fn local_only_candidate_set_builds_from_local_worker_registry() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            0,
        );
        let output = build_candidates(
            &registry,
            &CrossRegionState::new(),
            RegionPeerRegistry::empty(),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );

        assert_eq!(candidate_regions(&output), vec!["us-ashburn-1"]);
        assert!(output.rejections.is_empty());
    }

    #[test]
    fn local_plus_remote_candidate_set_builds_from_materialized_remote_signals() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            2,
        );
        let mut remote_state = CrossRegionState::new();
        add_remote_worker(
            &mut remote_state,
            "us-chicago-1",
            "remote-worker-a",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            7,
            NOW_MS,
        );
        remote_state.upsert_client_latency(
            crate::cross_region::ClientLatencySignal {
                client_region: "iad".to_string(),
                target_region: "us-chicago-1".to_string(),
                server_name: "remote-iad".to_string(),
                p50_latency_ms: 34,
                p95_latency_ms: 80,
            },
            signal_version(NOW_MS),
        );
        let output = build_candidates(
            &registry,
            &remote_state,
            peer_registry(&["us-chicago-1"]),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1", "us-chicago-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );

        assert_eq!(
            candidate_regions(&output),
            vec!["us-ashburn-1", "us-chicago-1"]
        );
        let remote = output
            .candidates
            .iter()
            .find(|candidate| candidate.region_id == "us-chicago-1")
            .expect("remote candidate should be built");
        assert_eq!(remote.worker_load, Some(7));
        assert_eq!(remote.client_latency_hint_ms, Some(34));
        assert_eq!(remote.freshness_age_ms, Some(0));
    }

    #[test]
    fn missing_remote_signals_degrades_to_local_only_candidates() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            0,
        );
        let output = build_candidates(
            &registry,
            &CrossRegionState::new(),
            peer_registry(&["us-chicago-1"]),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1", "us-chicago-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );

        assert_eq!(candidate_regions(&output), vec!["us-ashburn-1"]);
        assert!(output.rejections.iter().any(|rejection| {
            rejection.region_id == "us-chicago-1"
                && rejection.reason == CandidateRejectionReason::MissingRemoteSignal
        }));
    }

    #[test]
    fn invalid_input_rejects_empty_local_region() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            0,
        );
        let remote_state = CrossRegionState::new();
        let peer_registry = RegionPeerRegistry::empty();
        let breaker = CrossRegionBreaker::new();
        let error = CandidateCalculator::new()
            .build_candidates(CandidateCalculationInput {
                profile: profile(
                    &["us-ashburn-1"],
                    "cohere.command-r-plus",
                    ModalityPolicy::default(),
                ),
                local_region: " ".to_string(),
                endpoint_type: Endpoint::Chat,
                local_worker_registry: &registry,
                remote_state: &remote_state,
                peer_registry: &peer_registry,
                breaker: &breaker,
                client_region: None,
                now_ms: NOW_MS,
            })
            .expect_err("empty local region should be rejected");

        assert!(matches!(error, CrossRegionError::InvalidConfig { .. }));
        assert!(error.to_string().contains("local_region"));
    }

    #[test]
    fn invalid_input_rejects_negative_timestamp() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            0,
        );
        let remote_state = CrossRegionState::new();
        let peer_registry = RegionPeerRegistry::empty();
        let breaker = CrossRegionBreaker::new();
        let error = CandidateCalculator::new()
            .build_candidates(CandidateCalculationInput {
                profile: profile(
                    &["us-ashburn-1"],
                    "cohere.command-r-plus",
                    ModalityPolicy::default(),
                ),
                local_region: "us-ashburn-1".to_string(),
                endpoint_type: Endpoint::Chat,
                local_worker_registry: &registry,
                remote_state: &remote_state,
                peer_registry: &peer_registry,
                breaker: &breaker,
                client_region: None,
                now_ms: -1,
            })
            .expect_err("negative timestamp should be rejected");

        assert!(matches!(error, CrossRegionError::InvalidConfig { .. }));
        assert!(error.to_string().contains("now_ms"));
    }

    #[test]
    fn rejection_metric_reasons_are_bounded() {
        let cases = [
            (
                CandidateRejectionReason::RegionNotAllowed,
                CandidateGatedReason::PolicyMismatch,
            ),
            (
                CandidateRejectionReason::ModelNotAllowed,
                CandidateGatedReason::PolicyMismatch,
            ),
            (
                CandidateRejectionReason::EndpointUnsupported,
                CandidateGatedReason::PolicyMismatch,
            ),
            (
                CandidateRejectionReason::ModalityUnsupported,
                CandidateGatedReason::PolicyMismatch,
            ),
            (
                CandidateRejectionReason::MissingRemoteSignal,
                CandidateGatedReason::StaleSignal,
            ),
            (
                CandidateRejectionReason::StaleRemoteSignal,
                CandidateGatedReason::StaleSignal,
            ),
            (
                CandidateRejectionReason::RemoteBreakerOpen,
                CandidateGatedReason::BreakerOpen,
            ),
            (
                CandidateRejectionReason::PeerUnavailable,
                CandidateGatedReason::PeerUnavailable,
            ),
        ];

        for (reason, metric_reason) in cases {
            let rejection = CandidateRejection {
                region_id: "us-chicago-1".to_string(),
                model_id: "cohere.command-r-plus".to_string(),
                reason,
            };

            assert_eq!(reason.metric_reason(), metric_reason);
            assert_eq!(rejection.metric_reason(), metric_reason);
        }
    }

    #[test]
    fn local_candidate_aggregates_multiple_matching_workers() {
        let registry = registry_with_workers(vec![
            (
                "http://local-ready-worker:8000",
                model_card("cohere.command-r-plus", ModelType::LLM),
                WorkerStatus::Ready,
                2,
            ),
            (
                "http://local-pending-worker:8000",
                model_card("cohere.command-r-plus", ModelType::LLM),
                WorkerStatus::Pending,
                5,
            ),
        ]);
        let output = build_candidates(
            &registry,
            &CrossRegionState::new(),
            RegionPeerRegistry::empty(),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );
        let candidate = output
            .candidates
            .first()
            .expect("local candidate should be built");

        assert_eq!(candidate.worker_health.ready_workers, 1);
        assert_eq!(candidate.worker_health.total_workers, 2);
        assert_eq!(candidate.worker_load, Some(7));
        assert!(candidate.healthy);
        assert!(candidate.has_capacity);
    }

    #[test]
    fn local_candidate_with_only_pending_workers_is_unhealthy_but_retained() {
        let registry = registry_with_worker(
            "http://local-pending-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Pending,
            4,
        );
        let output = build_candidates(
            &registry,
            &CrossRegionState::new(),
            RegionPeerRegistry::empty(),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );
        let candidate = output
            .candidates
            .first()
            .expect("pending local candidate should still be built");

        assert_eq!(candidate.worker_health.ready_workers, 0);
        assert_eq!(candidate.worker_health.total_workers, 1);
        assert_eq!(candidate.worker_load, Some(4));
        assert!(!candidate.healthy);
        assert!(!candidate.has_capacity);
    }

    #[test]
    fn remote_candidate_aggregates_multiple_matching_workers() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            0,
        );
        let mut remote_state = CrossRegionState::new();
        add_remote_worker(
            &mut remote_state,
            "us-chicago-1",
            "remote-ready-worker",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            4,
            NOW_MS,
        );
        add_remote_worker(
            &mut remote_state,
            "us-chicago-1",
            "remote-pending-worker",
            Some("cohere.command-r-plus"),
            WorkerStatus::Pending,
            9,
            NOW_MS,
        );
        let output = build_candidates(
            &registry,
            &remote_state,
            peer_registry(&["us-chicago-1"]),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1", "us-chicago-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );
        let remote = output
            .candidates
            .iter()
            .find(|candidate| candidate.region_id == "us-chicago-1")
            .expect("remote candidate should be built");

        assert_eq!(remote.worker_health.ready_workers, 1);
        assert_eq!(remote.worker_health.total_workers, 2);
        assert_eq!(remote.worker_load, Some(13));
        assert!(remote.healthy);
        assert!(remote.has_capacity);
    }

    #[test]
    fn remote_candidate_aggregates_across_two_replicas_observing_same_worker() {
        // Two SMG replicas in us-chicago-1 each observe one worker (`w1`).
        // Per Phase B aggregation: any-fresh-ready readiness, sum loads,
        // any-fresh-routable status — region should produce one candidate
        // with the summed load and a single ready worker.
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            0,
        );
        let mut remote_state = CrossRegionState::new();
        let v_a = SignalVersion {
            version: 1,
            actor: "smg-a".to_string(),
            updated_at_ms: NOW_MS,
        };
        let v_b = SignalVersion {
            version: 1,
            actor: "smg-b".to_string(),
            updated_at_ms: NOW_MS,
        };
        for (server_name, version) in [("smg-a", &v_a), ("smg-b", &v_b)] {
            remote_state.upsert_readiness(
                SmgReadinessSignal {
                    region_id: "us-chicago-1".to_string(),
                    server_name: server_name.to_string(),
                    ready: true,
                },
                version.clone(),
            );
            remote_state.upsert_worker_health(
                WorkerHealthSignal {
                    region_id: "us-chicago-1".to_string(),
                    worker_id: "w1".to_string(),
                    server_name: server_name.to_string(),
                    status: WorkerStatus::Ready,
                },
                version.clone(),
            );
        }
        remote_state.upsert_worker_load(
            WorkerLoadSignal {
                region_id: "us-chicago-1".to_string(),
                worker_id: "w1".to_string(),
                server_name: "smg-a".to_string(),
                load: WorkerLoadInfo {
                    worker: "w1".to_string(),
                    worker_type: None,
                    load: 4,
                    details: None,
                    region_id: Some("us-chicago-1".to_string()),
                    worker_id: Some("w1".to_string()),
                    model_id: Some("cohere.command-r-plus".to_string()),
                    status: Some(WorkerStatus::Ready),
                    generated_at_ms: Some(NOW_MS),
                    version: Some(1),
                    source: None,
                    remote_workers: None,
                },
            },
            v_a,
        );
        remote_state.upsert_worker_load(
            WorkerLoadSignal {
                region_id: "us-chicago-1".to_string(),
                worker_id: "w1".to_string(),
                server_name: "smg-b".to_string(),
                load: WorkerLoadInfo {
                    worker: "w1".to_string(),
                    worker_type: None,
                    load: 5,
                    details: None,
                    region_id: Some("us-chicago-1".to_string()),
                    worker_id: Some("w1".to_string()),
                    model_id: Some("cohere.command-r-plus".to_string()),
                    status: Some(WorkerStatus::Ready),
                    generated_at_ms: Some(NOW_MS),
                    version: Some(1),
                    source: None,
                    remote_workers: None,
                },
            },
            v_b,
        );

        let output = build_candidates(
            &registry,
            &remote_state,
            peer_registry(&["us-chicago-1"]),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1", "us-chicago-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );
        let remote = output
            .candidates
            .iter()
            .find(|candidate| candidate.region_id == "us-chicago-1")
            .expect("remote candidate should be built from per-replica aggregation");

        assert!(remote.readiness);
        assert!(remote.healthy);
        assert!(remote.has_capacity);
        assert_eq!(
            remote.worker_health.ready_workers, 1,
            "two replicas observing one worker must count as one ready worker, not two",
        );
        assert_eq!(remote.worker_health.total_workers, 1);
        assert_eq!(
            remote.worker_load,
            Some(9),
            "loads from two replicas observing the same worker must sum (4 + 5)",
        );
    }

    #[test]
    fn remote_candidate_rejects_when_only_other_models_have_signals() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            0,
        );
        let mut remote_state = CrossRegionState::new();
        add_remote_worker(
            &mut remote_state,
            "us-chicago-1",
            "remote-worker-a",
            Some("meta.llama-3"),
            WorkerStatus::Ready,
            1,
            NOW_MS,
        );
        let output = build_candidates(
            &registry,
            &remote_state,
            peer_registry(&["us-chicago-1"]),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1", "us-chicago-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );

        assert_eq!(candidate_regions(&output), vec!["us-ashburn-1"]);
        assert!(output.rejections.iter().any(|rejection| {
            rejection.region_id == "us-chicago-1"
                && rejection.reason == CandidateRejectionReason::ModelNotAllowed
        }));
    }

    #[test]
    fn stale_remote_latency_excludes_remote_candidate() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            0,
        );
        let mut remote_state = CrossRegionState::new();
        add_remote_worker(
            &mut remote_state,
            "us-chicago-1",
            "remote-worker-a",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            1,
            NOW_MS,
        );
        remote_state.upsert_client_latency(
            crate::cross_region::ClientLatencySignal {
                client_region: "iad".to_string(),
                target_region: "us-chicago-1".to_string(),
                server_name: "remote-iad".to_string(),
                p50_latency_ms: 34,
                p95_latency_ms: 80,
            },
            signal_version(NOW_MS - DEFAULT_REMOTE_SIGNAL_MAX_AGE_MS - 1),
        );
        let output = build_candidates(
            &registry,
            &remote_state,
            peer_registry(&["us-chicago-1"]),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1", "us-chicago-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );

        assert_eq!(candidate_regions(&output), vec!["us-ashburn-1"]);
        assert!(output.rejections.iter().any(|rejection| {
            rejection.region_id == "us-chicago-1"
                && rejection.reason == CandidateRejectionReason::StaleRemoteSignal
        }));
    }

    #[test]
    fn remote_not_ready_region_is_retained_as_unhealthy_candidate() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            0,
        );
        let mut remote_state = CrossRegionState::new();
        add_remote_worker(
            &mut remote_state,
            "us-chicago-1",
            "remote-worker-a",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            1,
            NOW_MS,
        );
        remote_state.upsert_readiness(
            SmgReadinessSignal {
                region_id: "us-chicago-1".to_string(),
                server_name: "remote-us-chicago-1".to_string(),
                ready: false,
            },
            SignalVersion {
                version: 2,
                actor: "test-actor".to_string(),
                updated_at_ms: NOW_MS,
            },
        );
        let output = build_candidates(
            &registry,
            &remote_state,
            peer_registry(&["us-chicago-1"]),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1", "us-chicago-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );
        let remote = output
            .candidates
            .iter()
            .find(|candidate| candidate.region_id == "us-chicago-1")
            .expect("not-ready remote region should still produce a candidate");

        assert!(!remote.readiness);
        assert_eq!(remote.worker_health.ready_workers, 1);
        assert!(!remote.healthy);
        assert!(remote.has_capacity);
    }

    #[test]
    fn remote_future_signal_timestamp_reports_zero_age() {
        let registry = registry_with_worker(
            "http://local-worker:8000",
            model_card("cohere.command-r-plus", ModelType::LLM),
            WorkerStatus::Ready,
            0,
        );
        let mut remote_state = CrossRegionState::new();
        add_remote_worker(
            &mut remote_state,
            "us-chicago-1",
            "remote-worker-a",
            Some("cohere.command-r-plus"),
            WorkerStatus::Ready,
            1,
            NOW_MS + 1_000,
        );
        let output = build_candidates(
            &registry,
            &remote_state,
            peer_registry(&["us-chicago-1"]),
            CrossRegionBreaker::new(),
            profile(
                &["us-ashburn-1", "us-chicago-1"],
                "cohere.command-r-plus",
                ModalityPolicy::default(),
            ),
            Endpoint::Chat,
        );
        let remote = output
            .candidates
            .iter()
            .find(|candidate| candidate.region_id == "us-chicago-1")
            .expect("future timestamp should not make the signal stale");

        assert_eq!(remote.freshness_age_ms, Some(0));
    }

    /// Fixed test timestamp used for deterministic freshness calculations.
    const NOW_MS: i64 = 1_700_000_000_000;

    /// Build a routing profile fixture with one Phase 1 model.
    fn profile(
        regions: &[&str],
        model_id: &str,
        modality: ModalityPolicy,
    ) -> RoutingProfileContext {
        RoutingProfileContext::new(
            regions.iter().map(|region| (*region).to_string()).collect(),
            vec![model_id.to_string()],
            FailoverPolicy::new(CrossRegionFailoverMode::Manual, 1),
            modality,
        )
        .expect("profile should be valid")
    }

    /// Build a model card with explicit endpoint and modality capabilities.
    fn model_card(model_id: &str, model_type: ModelType) -> ModelCard {
        ModelCard::new(model_id).with_model_type(model_type)
    }

    /// Register one local worker and set its requested load.
    fn registry_with_worker(
        url: &str,
        model: ModelCard,
        status: WorkerStatus,
        load: usize,
    ) -> WorkerRegistry {
        registry_with_workers(vec![(url, model, status, load)])
    }

    /// Register local workers and set their requested loads.
    fn registry_with_workers(
        workers: Vec<(&str, ModelCard, WorkerStatus, usize)>,
    ) -> WorkerRegistry {
        let registry = WorkerRegistry::new();
        for (url, model, status, load) in workers {
            let worker = Arc::new(
                BasicWorkerBuilder::new(url)
                    .model(model)
                    .status(status)
                    .build(),
            );
            for _ in 0..load {
                worker.increment_load();
            }
            registry
                .register(worker)
                .expect("worker should register successfully");
        }
        registry
    }

    /// Build a peer registry with valid Region Agent peers for the supplied regions.
    fn peer_registry(regions: &[&str]) -> RegionPeerRegistry {
        let peers = regions
            .iter()
            .map(|region| {
                RegionPeer::new(
                    *region,
                    format!("https://smg-region-agent.{region}.internal:8443"),
                    format!("https://smg-region-agent.{region}.internal:9443"),
                    "oc1",
                    "prod",
                    None,
                )
                .expect("peer should be valid")
            })
            .collect();
        RegionPeerRegistry::new(peers).expect("peer registry should build")
    }

    /// Build a signal version with a deterministic update timestamp.
    fn signal_version(updated_at_ms: i64) -> SignalVersion {
        SignalVersion {
            version: 1,
            actor: "test-actor".to_string(),
            updated_at_ms,
        }
    }

    /// Add the minimum remote signal set needed for one remote worker candidate.
    fn add_remote_worker(
        state: &mut CrossRegionState,
        region_id: &str,
        worker_id: &str,
        model_id: Option<&str>,
        status: WorkerStatus,
        load: isize,
        updated_at_ms: i64,
    ) {
        let version = signal_version(updated_at_ms);
        let server_name = format!("remote-{region_id}");
        state.upsert_readiness(
            SmgReadinessSignal {
                region_id: region_id.to_string(),
                server_name: server_name.clone(),
                ready: true,
            },
            version.clone(),
        );
        state.upsert_worker_health(
            WorkerHealthSignal {
                region_id: region_id.to_string(),
                worker_id: worker_id.to_string(),
                server_name: server_name.clone(),
                status,
            },
            version.clone(),
        );
        state.upsert_worker_load(
            WorkerLoadSignal {
                region_id: region_id.to_string(),
                worker_id: worker_id.to_string(),
                server_name,
                load: WorkerLoadInfo {
                    worker: worker_id.to_string(),
                    worker_type: None,
                    load,
                    details: None,
                    region_id: Some(region_id.to_string()),
                    worker_id: Some(worker_id.to_string()),
                    model_id: model_id.map(str::to_string),
                    status: Some(status),
                    generated_at_ms: Some(updated_at_ms),
                    version: Some(1),
                    source: None,
                    remote_workers: None,
                },
            },
            version,
        );
    }

    /// Add a client-latency signal for the common test client region.
    fn add_client_latency(
        state: &mut CrossRegionState,
        target_region: &str,
        p50_latency_ms: u64,
        updated_at_ms: i64,
    ) {
        state.upsert_client_latency(
            crate::cross_region::ClientLatencySignal {
                client_region: "iad".to_string(),
                target_region: target_region.to_string(),
                p50_latency_ms,
                p95_latency_ms: p50_latency_ms.saturating_mul(2),
            },
            signal_version(updated_at_ms),
        );
    }

    /// Run candidate construction with the common local-region and client-region fixtures.
    fn build_candidates(
        local_worker_registry: &WorkerRegistry,
        remote_state: &CrossRegionState,
        peer_registry: RegionPeerRegistry,
        breaker: CrossRegionBreaker,
        profile: RoutingProfileContext,
        endpoint_type: Endpoint,
    ) -> CandidateCalculationOutput {
        CandidateCalculator::new()
            .build_candidates(CandidateCalculationInput {
                profile,
                local_region: "us-ashburn-1".to_string(),
                endpoint_type,
                local_worker_registry,
                remote_state,
                peer_registry: &peer_registry,
                breaker: &breaker,
                client_region: Some("iad".to_string()),
                now_ms: NOW_MS,
            })
            .expect("candidate calculation should succeed")
    }

    /// Run route calculation with the common local-region and request fixtures.
    fn calculate_route(
        local_worker_registry: &WorkerRegistry,
        remote_state: &CrossRegionState,
        peer_registry: RegionPeerRegistry,
        breaker: CrossRegionBreaker,
        profile: RoutingProfileContext,
        endpoint_type: Endpoint,
    ) -> Option<RouteCalculationOutput> {
        CandidateCalculator::new()
            .calculate(RouteCalculationInput {
                route_id: "route-1".to_string(),
                entry_region: "us-ashburn-1".to_string(),
                request_mode: RequestMode::Unresolved,
                attempt: 1,
                candidate_input: CandidateCalculationInput {
                    profile,
                    local_region: "us-ashburn-1".to_string(),
                    endpoint_type,
                    local_worker_registry,
                    remote_state,
                    peer_registry: &peer_registry,
                    breaker: &breaker,
                    client_region: Some("iad".to_string()),
                    now_ms: NOW_MS,
                },
            })
            .expect("route calculation should succeed")
    }

    /// Return the committed target region for concise route-ranking assertions.
    fn route_target_region(
        local_worker_registry: &WorkerRegistry,
        remote_state: &CrossRegionState,
        peer_registry: RegionPeerRegistry,
        breaker: CrossRegionBreaker,
        profile: RoutingProfileContext,
        endpoint_type: Endpoint,
    ) -> String {
        calculate_route(
            local_worker_registry,
            remote_state,
            peer_registry,
            breaker,
            profile,
            endpoint_type,
        )
        .expect("route should be committed")
        .decision
        .target_region
    }

    /// Return candidate regions in output order for concise assertions.
    fn candidate_regions(output: &CandidateCalculationOutput) -> Vec<&str> {
        output
            .candidates
            .iter()
            .map(|candidate| candidate.region_id.as_str())
            .collect()
    }
}
