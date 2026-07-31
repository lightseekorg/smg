//! Worker selection stage: Select appropriate worker(s) based on routing mode

use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    http::{HeaderMap, HeaderValue},
    response::Response,
};
use tracing::{error, warn};

use super::PipelineStage;
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    policies::{LoadBalancingPolicy, PolicyRegistry, SelectWorkerInfo, WorkerLeg},
    routers::{
        common::header_utils,
        error,
        grpc::{
            context::{EncodeWorkerAssignment, PdLoadGuards, RequestContext, WorkerSelection},
            multimodal,
        },
        PD_PREFILL_QUEUE_FULL, PD_PREFILL_QUEUE_TIMEOUT,
    },
    worker::{
        acquire_prefill, ConnectionMode, HashRing, PrefillAcquireError, PrefillAdmission,
        PrefillAdmissionRejection, PrefillCandidateError, PrefillSelectionContext, RuntimeType,
        Worker, WorkerLoadGuard, WorkerRegistry, WorkerType, UNKNOWN_MODEL_ID,
    },
};

type PdWorkerPair = (Arc<dyn Worker>, Arc<dyn Worker>, RuntimeType, PdLoadGuards);

type EncodePrefillDecodeWorkerSelection = (
    Vec<EncodeWorkerAssignment>,
    Arc<dyn Worker>,
    Arc<dyn Worker>,
    RuntimeType,
    PdLoadGuards,
);

struct PdCandidate {
    prefill: Arc<dyn Worker>,
    decode: Arc<dyn Worker>,
    runtime_type: RuntimeType,
}

struct EncodePrefillDecodeCandidate {
    encode_assignments: Vec<EncodeWorkerAssignment>,
    prefill: Arc<dyn Worker>,
    decode: Arc<dyn Worker>,
    runtime_type: RuntimeType,
}

struct PdSelectionContext<'a> {
    request_text: Option<&'a str>,
    tokens: Option<&'a [u32]>,
    headers: Option<&'a HeaderMap>,
    hash_ring: Option<Arc<HashRing>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PdSelectionError {
    Unavailable,
    QueueFull,
    QueueTimeout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PdCandidateError {
    Unavailable,
    PrefillAtCapacity,
}

impl PdSelectionError {
    fn into_response(self, model_id: &str) -> Response {
        match self {
            Self::Unavailable => error::model_not_found(model_id),
            Self::QueueFull => error::too_many_requests(
                PD_PREFILL_QUEUE_FULL,
                "The Prefill admission queue is full",
            ),
            Self::QueueTimeout => error::too_many_requests(
                PD_PREFILL_QUEUE_TIMEOUT,
                "Timed out waiting for Prefill admission",
            ),
        }
    }
}

impl From<PdCandidateError> for PdSelectionError {
    fn from(value: PdCandidateError) -> Self {
        match value {
            PdCandidateError::Unavailable => Self::Unavailable,
            PdCandidateError::PrefillAtCapacity => Self::QueueFull,
        }
    }
}

impl From<PrefillAdmissionRejection> for PdSelectionError {
    fn from(value: PrefillAdmissionRejection) -> Self {
        match value {
            PrefillAdmissionRejection::QueueFull => Self::QueueFull,
            PrefillAdmissionRejection::QueueTimeout => Self::QueueTimeout,
            PrefillAdmissionRejection::Unavailable => Self::Unavailable,
        }
    }
}

impl From<PrefillAcquireError<PdCandidateError>> for PdSelectionError {
    fn from(value: PrefillAcquireError<PdCandidateError>) -> Self {
        match value {
            PrefillAcquireError::Candidate(error) => error.into(),
            PrefillAcquireError::Rejected(rejection) => rejection.into(),
        }
    }
}

impl PrefillCandidateError for PdCandidateError {
    fn is_at_capacity(&self) -> bool {
        matches!(self, Self::PrefillAtCapacity)
    }
}

/// Worker selection stage: Select appropriate worker(s) based on routing mode
pub(crate) struct WorkerSelectionStage {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    mode: WorkerSelectionMode,
    prefill_admission: Option<Arc<PrefillAdmission>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WorkerSelectionMode {
    /// Regular mode: select single worker
    Regular,
    /// PD mode: select prefill + decode workers
    PrefillDecode,
    /// EPD mode: select encode + prefill + decode workers
    EncodePrefillDecode,
}

impl WorkerSelectionStage {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        mode: WorkerSelectionMode,
        prefill_admission: Option<Arc<PrefillAdmission>>,
    ) -> Self {
        Self {
            worker_registry,
            policy_registry,
            mode,
            prefill_admission,
        }
    }
}

#[async_trait]
impl PipelineStage for WorkerSelectionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "WorkerSelectionStage::execute",
                "Preparation stage not completed"
            );
            error::internal_error(
                "preparation_stage_not_completed",
                "Preparation stage not completed",
            )
        })?;

        let intermediate = ctx.state.multimodal_intermediate.as_ref();

        let text = prep.routing_text();

        // Get tokens for PrefixHash policy support
        let ids = prep.token_ids();
        let tokens = if ids.is_empty() { None } else { Some(ids) };

        let headers = ctx.input.headers.as_ref();

        let model_id = ctx.input.model_id.as_str();

        let (workers, pd_load_guards) = match self.mode {
            WorkerSelectionMode::Regular => {
                match self.select_single_worker(model_id, text, tokens, headers) {
                    Some(w) => (WorkerSelection::Single { worker: w }, None),
                    None => {
                        error!(
                            function = "WorkerSelectionStage::execute",
                            mode = "Regular",
                            model_id = %model_id,
                            "No available workers for model"
                        );
                        return Err(error::model_not_found(model_id));
                    }
                }
            }
            WorkerSelectionMode::PrefillDecode => {
                match self.select_pd_pair(model_id, text, tokens, headers).await {
                    Ok((prefill, decode, runtime_type, guards)) => (
                        WorkerSelection::Disaggregated {
                            encode_assignments: None,
                            prefill,
                            decode,
                            runtime_type,
                        },
                        Some(guards),
                    ),
                    Err(selection_error) => {
                        error!(
                            function = "WorkerSelectionStage::execute",
                            mode = "PrefillDecode",
                            model_id = %model_id,
                            error = ?selection_error,
                            "Failed to reserve a PD worker pair"
                        );
                        return Err(selection_error.into_response(model_id));
                    }
                }
            }
            WorkerSelectionMode::EncodePrefillDecode => {
                let encode_item_hashes = match encode_item_hashes(intermediate) {
                    Ok(hashes) => hashes,
                    Err(err) => {
                        error!(
                            function = "WorkerSelectionStage::execute",
                            error = %err,
                            "Failed to derive encode item routing hashes"
                        );
                        return Err(error::internal_error(
                            "encode_routing_hash_failed",
                            format!("Failed to derive encode routing hashes: {err}"),
                        ));
                    }
                };
                match self
                    .select_encode_prefill_decode_workers(
                        model_id,
                        text,
                        tokens,
                        headers,
                        &encode_item_hashes,
                    )
                    .await
                {
                    Ok((encode_assignments, prefill, decode, runtime_type, guards)) => (
                        WorkerSelection::Disaggregated {
                            encode_assignments: if encode_assignments.is_empty() {
                                None
                            } else {
                                Some(encode_assignments)
                            },
                            prefill,
                            decode,
                            runtime_type,
                        },
                        Some(guards),
                    ),
                    Err(selection_error) => {
                        error!(
                            function = "WorkerSelectionStage::execute",
                            mode = "EncodePrefillDecode",
                            model_id = %model_id,
                            error = ?selection_error,
                            "Failed to reserve an encode/prefill/decode worker set"
                        );
                        return Err(selection_error.into_response(model_id));
                    }
                }
            }
        };

        // Reject an unsupported (backend, modality) combination now that the
        // runtime is known, before request building fetches/preprocesses media
        // only to fail deep in assembly. The prefill leg builds the request in
        // disaggregated mode, so its runtime is the one that must support the
        // request's modalities.
        if let Some(intermediate) = intermediate {
            if let Err(err) = multimodal::ensure_backend_supports_modalities(
                selection_runtime(&workers),
                intermediate,
            ) {
                return Err(error::bad_request(
                    "multimodal_not_supported",
                    format!("{err}"),
                ));
            }
        }

        ctx.state.workers = Some(workers);
        ctx.state.pd_load_guards = pd_load_guards;
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "WorkerSelection"
    }

    #[cfg(test)]
    fn signature(&self) -> String {
        format!("WorkerSelectionStage({:?})", self.mode)
    }
}

/// Runtime of the leg that builds the generate request: the sole worker in
/// regular mode, the prefill worker in disaggregated (PD/EPD) mode.
fn selection_runtime(workers: &WorkerSelection) -> RuntimeType {
    match workers {
        WorkerSelection::Single { worker } => worker.metadata().spec.runtime_type,
        WorkerSelection::Disaggregated { runtime_type, .. } => *runtime_type,
    }
}

impl WorkerSelectionStage {
    fn select_single_worker(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
    ) -> Option<Arc<dyn Worker>> {
        // Treat "unknown" model as wildcard (match any worker)
        let model_filter = if model_id == UNKNOWN_MODEL_ID {
            None
        } else {
            Some(model_id)
        };

        // Get workers for the specified model, filtered by connection mode
        let workers = self.worker_registry.get_workers_filtered(
            model_filter,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Grpc),
            None,  // any runtime type
            false, // get all workers, we'll filter by is_available() next
        );

        // Use into_iter() to take ownership of Arcs without cloning (avoids atomic inc/dec)
        let available: Vec<Arc<dyn Worker>> =
            workers.into_iter().filter(|w| w.is_available()).collect();

        if available.is_empty() {
            return None;
        }

        // Get the appropriate policy for this model
        let policy = self.policy_registry.get_policy_or_default(model_id);

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self.worker_registry.get_hash_ring(model_id);

        // Select worker via the registry (applies the routing-key sticky override
        // when enabled; otherwise delegates to the configured policy).
        let idx = self.policy_registry.select_worker(
            &policy,
            &available,
            &SelectWorkerInfo {
                request_text: text,
                tokens,
                headers,
                hash_ring,
                leg: WorkerLeg::Single,
            },
        )?;
        let selected = available[idx].clone();

        // Record worker selection metric
        Metrics::record_worker_selection(
            metrics_labels::WORKER_REGULAR,
            metrics_labels::CONNECTION_GRPC,
            model_id,
            policy.name(),
        );

        Some(selected)
    }

    async fn select_pd_pair(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
    ) -> Result<PdWorkerPair, PdSelectionError> {
        let (candidate, prefill_guard) = acquire_prefill(
            self.prefill_admission.as_deref(),
            headers,
            |candidate: &PdCandidate| &candidate.prefill,
            |capacity| self.select_pd_candidate(model_id, text, tokens, headers, capacity),
        )
        .await?;
        let decode_guard = WorkerLoadGuard::new(Arc::clone(&candidate.decode), headers);

        Ok((
            candidate.prefill,
            candidate.decode,
            candidate.runtime_type,
            PdLoadGuards {
                prefill: prefill_guard,
                decode: decode_guard,
            },
        ))
    }

    fn select_pd_candidate(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
        capacity: Option<&PrefillSelectionContext<'_>>,
    ) -> Result<PdCandidate, PdCandidateError> {
        // Treat "unknown" model as wildcard (match any worker)
        let model_filter = if model_id == UNKNOWN_MODEL_ID {
            None
        } else {
            Some(model_id)
        };

        let all_workers = self.worker_registry.get_workers_filtered(
            model_filter,
            None,
            Some(ConnectionMode::Grpc), // Match any gRPC worker
            None,                       // any runtime type
            false,
        );

        let (mut all_prefill, all_decode): (Vec<_>, Vec<_>) =
            all_workers
                .into_iter()
                .fold((Vec::new(), Vec::new()), |mut acc, w| {
                    if w.is_available() {
                        match w.metadata().spec.worker_type {
                            WorkerType::Prefill => acc.0.push(w),
                            WorkerType::Decode => acc.1.push(w),
                            WorkerType::Regular => {}
                            // Encode-prefill-decode selection is handled in select_encode_prefill_decode_workers;
                            // the PD pair fold ignores encode workers.
                            WorkerType::Encode => {}
                        }
                    }
                    acc
                });

        if all_prefill.is_empty() {
            warn!("No available prefill workers");
            return Err(PdCandidateError::Unavailable);
        }

        if all_decode.is_empty() {
            warn!("No available decode workers");
            return Err(PdCandidateError::Unavailable);
        }

        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();
        let targeted = prefill_policy.name() == "consistent_hashing"
            && header_utils::extract_target_worker(headers).is_some();

        all_prefill.retain(|prefill| {
            all_decode.iter().any(|decode| {
                decode.metadata().spec.runtime_type == prefill.metadata().spec.runtime_type
            })
        });
        if all_prefill.is_empty() {
            warn!("No available PD worker pair with a shared runtime");
            return Err(PdCandidateError::Unavailable);
        }
        if let Some(capacity) = capacity {
            if !targeted {
                all_prefill.retain(|worker| capacity.has_capacity(worker));
                if all_prefill.is_empty() {
                    return Err(PdCandidateError::PrefillAtCapacity);
                }
            }
        }

        let target_runtime = all_prefill[0].metadata().spec.runtime_type;

        // Check for mixed runtimes in both prefill and decode pools
        let prefill_mixed = all_prefill
            .iter()
            .any(|w| w.metadata().spec.runtime_type != target_runtime);
        let decode_mixed = all_decode
            .iter()
            .any(|w| w.metadata().spec.runtime_type != target_runtime);

        if prefill_mixed || decode_mixed {
            warn!(
                "Mixed runtime types in PD workers (prefill_mixed={}, decode_mixed={}). Using {:?}.",
                prefill_mixed,
                decode_mixed,
                target_runtime
            );
        }

        // Filter both pools to the target runtime
        let available_prefill: Vec<_> = all_prefill
            .into_iter()
            .filter(|w| w.metadata().spec.runtime_type == target_runtime)
            .collect();
        let available_decode: Vec<_> = all_decode
            .into_iter()
            .filter(|w| w.metadata().spec.runtime_type == target_runtime)
            .collect();

        if available_prefill.is_empty() || available_decode.is_empty() {
            warn!("No available PD pair for runtime {:?}", target_runtime);
            return Err(PdCandidateError::Unavailable);
        }
        if targeted {
            let target = header_utils::extract_target_worker(headers)
                .and_then(|value| value.parse::<usize>().ok())
                .and_then(|index| available_prefill.get(index))
                .ok_or(PdCandidateError::Unavailable)?;
            if capacity.is_some_and(|capacity| !capacity.has_capacity(target)) {
                return Err(PdCandidateError::PrefillAtCapacity);
            }
        }

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self.worker_registry.get_hash_ring(model_id);

        // Prefill and decode are separate pools; tag each leg so the routing-key
        // override keys its sticky map per leg (a key sticks independently).
        let selection = PdSelectionContext {
            request_text: text,
            tokens,
            headers,
            hash_ring,
        };
        let prefill = self.select_disaggregated_worker(
            &available_prefill,
            &prefill_policy,
            &selection,
            WorkerLeg::Prefill,
        )?;
        let decode = self.select_disaggregated_worker(
            &available_decode,
            &decode_policy,
            &selection,
            WorkerLeg::Decode,
        )?;
        let model = model_id;

        // Record worker selection metrics for both prefill and decode
        Metrics::record_worker_selection(
            metrics_labels::WORKER_PREFILL,
            metrics_labels::CONNECTION_GRPC,
            model,
            prefill_policy.name(),
        );
        Metrics::record_worker_selection(
            metrics_labels::WORKER_DECODE,
            metrics_labels::CONNECTION_GRPC,
            model,
            decode_policy.name(),
        );

        Ok(PdCandidate {
            prefill,
            decode,
            runtime_type: target_runtime,
        })
    }

    async fn select_encode_prefill_decode_workers(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
        encode_item_hashes: &[Vec<u8>],
    ) -> Result<EncodePrefillDecodeWorkerSelection, PdSelectionError> {
        let (candidate, prefill_guard) = acquire_prefill(
            self.prefill_admission.as_deref(),
            headers,
            |candidate: &EncodePrefillDecodeCandidate| &candidate.prefill,
            |capacity| {
                self.select_encode_prefill_decode_candidate(
                    model_id,
                    text,
                    tokens,
                    headers,
                    encode_item_hashes,
                    capacity,
                )
            },
        )
        .await?;
        let decode_guard = WorkerLoadGuard::new(Arc::clone(&candidate.decode), headers);

        Ok((
            candidate.encode_assignments,
            candidate.prefill,
            candidate.decode,
            candidate.runtime_type,
            PdLoadGuards {
                prefill: prefill_guard,
                decode: decode_guard,
            },
        ))
    }

    /// Select per-item Encode workers and one Prefill/Decode pair for EPD.
    fn select_encode_prefill_decode_candidate(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
        encode_item_hashes: &[Vec<u8>],
        capacity: Option<&PrefillSelectionContext<'_>>,
    ) -> Result<EncodePrefillDecodeCandidate, PdCandidateError> {
        // Treat "unknown" model as wildcard (match any worker)
        let model_filter = if model_id == UNKNOWN_MODEL_ID {
            None
        } else {
            Some(model_id)
        };

        let all_workers = self.worker_registry.get_workers_filtered(
            model_filter,
            None,
            Some(ConnectionMode::Grpc), // Match any gRPC worker
            None,                       // any runtime type
            false,
        );

        let (all_encode, mut all_prefill, all_decode): (Vec<_>, Vec<_>, Vec<_>) = all_workers
            .into_iter()
            .fold((Vec::new(), Vec::new(), Vec::new()), |mut acc, w| {
                if w.is_available() {
                    match w.metadata().spec.worker_type {
                        WorkerType::Encode => acc.0.push(w),
                        WorkerType::Prefill => acc.1.push(w),
                        WorkerType::Decode => acc.2.push(w),
                        WorkerType::Regular => {}
                    }
                }
                acc
            });

        let needs_encode = !encode_item_hashes.is_empty();
        if needs_encode && all_encode.is_empty() {
            warn!("No available encode workers");
            return Err(PdCandidateError::Unavailable);
        }
        if all_prefill.is_empty() {
            warn!("No available prefill workers");
            return Err(PdCandidateError::Unavailable);
        }
        if all_decode.is_empty() {
            warn!("No available decode workers");
            return Err(PdCandidateError::Unavailable);
        }

        let encode_policy = self.policy_registry.get_encode_policy();
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();
        let targeted = prefill_policy.name() == "consistent_hashing"
            && header_utils::extract_target_worker(headers).is_some();

        // Multimodal EPD currently supports TokenSpeed only.
        all_prefill.retain(|prefill| {
            let runtime = prefill.metadata().spec.runtime_type;
            (!needs_encode || runtime == RuntimeType::TokenSpeed)
                && all_decode
                    .iter()
                    .any(|decode| decode.metadata().spec.runtime_type == runtime)
                && (!needs_encode
                    || all_encode
                        .iter()
                        .any(|encode| encode.metadata().spec.runtime_type == runtime))
        });
        if all_prefill.is_empty() {
            warn!("No available encode/prefill/decode worker set with a shared runtime");
            return Err(PdCandidateError::Unavailable);
        }
        if let Some(capacity) = capacity {
            if !targeted {
                all_prefill.retain(|worker| capacity.has_capacity(worker));
                if all_prefill.is_empty() {
                    return Err(PdCandidateError::PrefillAtCapacity);
                }
            }
        }

        // Disaggregated legs must share a runtime. Pick a runtime that has at
        // least one available worker in every required EPD pool instead of
        // blindly using the first prefill runtime.
        let target_runtime = all_prefill[0].metadata().spec.runtime_type;

        let mixed = all_prefill
            .iter()
            .chain(all_decode.iter())
            .any(|w| w.metadata().spec.runtime_type != target_runtime)
            || (needs_encode
                && all_encode
                    .iter()
                    .any(|w| w.metadata().spec.runtime_type != target_runtime));
        if mixed {
            warn!(
                "Mixed runtime types in encode/prefill/decode workers. Using {:?}.",
                target_runtime
            );
        }

        // Filter all three pools to the target runtime
        let available_encode: Vec<_> = all_encode
            .into_iter()
            .filter(|w| w.metadata().spec.runtime_type == target_runtime)
            .collect();
        let available_prefill: Vec<_> = all_prefill
            .into_iter()
            .filter(|w| w.metadata().spec.runtime_type == target_runtime)
            .collect();
        let available_decode: Vec<_> = all_decode
            .into_iter()
            .filter(|w| w.metadata().spec.runtime_type == target_runtime)
            .collect();

        if (needs_encode && available_encode.is_empty())
            || available_prefill.is_empty()
            || available_decode.is_empty()
        {
            warn!(
                "No available encode/prefill/decode worker set for runtime {:?}",
                target_runtime
            );
            return Err(PdCandidateError::Unavailable);
        }
        if targeted {
            let target = header_utils::extract_target_worker(headers)
                .and_then(|value| value.parse::<usize>().ok())
                .and_then(|index| available_prefill.get(index))
                .ok_or(PdCandidateError::Unavailable)?;
            if capacity.is_some_and(|capacity| !capacity.has_capacity(target)) {
                return Err(PdCandidateError::PrefillAtCapacity);
            }
        }

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self.worker_registry.get_hash_ring(model_id);

        let encode_assignments = assign_encode_workers(
            &available_encode,
            encode_item_hashes,
            model_id,
            encode_policy.as_ref(),
            hash_ring.clone(),
        )
        .ok_or(PdCandidateError::Unavailable)?;

        let selection = PdSelectionContext {
            request_text: text,
            tokens,
            headers,
            hash_ring,
        };
        let prefill = self.select_disaggregated_worker(
            &available_prefill,
            &prefill_policy,
            &selection,
            WorkerLeg::Prefill,
        )?;
        let decode = self.select_disaggregated_worker(
            &available_decode,
            &decode_policy,
            &selection,
            WorkerLeg::Decode,
        )?;
        // Record worker selection metrics for prefill and decode, each tagged
        // with the policy that picked it. Encode item assignment metrics are
        // recorded in assign_encode_workers.
        Metrics::record_worker_selection(
            metrics_labels::WORKER_PREFILL,
            metrics_labels::CONNECTION_GRPC,
            model_id,
            prefill_policy.name(),
        );
        Metrics::record_worker_selection(
            metrics_labels::WORKER_DECODE,
            metrics_labels::CONNECTION_GRPC,
            model_id,
            decode_policy.name(),
        );

        Ok(EncodePrefillDecodeCandidate {
            encode_assignments,
            prefill,
            decode,
            runtime_type: target_runtime,
        })
    }

    fn select_disaggregated_worker(
        &self,
        candidates: &[Arc<dyn Worker>],
        policy: &Arc<dyn LoadBalancingPolicy>,
        request: &PdSelectionContext<'_>,
        leg: WorkerLeg,
    ) -> Result<Arc<dyn Worker>, PdCandidateError> {
        if candidates.is_empty() {
            return Err(PdCandidateError::Unavailable);
        }

        let selected_idx = self
            .policy_registry
            .select_worker(
                policy,
                candidates,
                &SelectWorkerInfo {
                    request_text: request.request_text,
                    tokens: request.tokens,
                    headers: request.headers,
                    hash_ring: request.hash_ring.clone(),
                    leg,
                },
            )
            .filter(|index| *index < candidates.len())
            .ok_or(PdCandidateError::Unavailable)?;

        Ok(Arc::clone(&candidates[selected_idx]))
    }
}

fn encode_item_hashes(
    intermediate: Option<&multimodal::MultimodalIntermediate>,
) -> anyhow::Result<Vec<Vec<u8>>> {
    let Some(intermediate) = intermediate else {
        return Ok(Vec::new());
    };
    multimodal::encode_routing_hashes(intermediate)
}

fn assign_encode_workers(
    encode_workers: &[Arc<dyn Worker>],
    item_hashes: &[Vec<u8>],
    model_id: &str,
    policy: &dyn LoadBalancingPolicy,
    hash_ring: Option<Arc<HashRing>>,
) -> Option<Vec<EncodeWorkerAssignment>> {
    if item_hashes.is_empty() {
        return Some(Vec::new());
    }

    item_hashes
        .iter()
        .enumerate()
        .map(|(item_index, content_hash)| {
            let routing_headers = encode_routing_headers(content_hash);
            let info = SelectWorkerInfo {
                request_text: None,
                tokens: None,
                headers: Some(&routing_headers),
                hash_ring: hash_ring.clone(),
                leg: WorkerLeg::Single,
            };
            let worker_idx = policy.select_worker(encode_workers, &info)?;
            let worker = encode_workers[worker_idx].clone();
            Metrics::record_worker_selection(
                metrics_labels::WORKER_ENCODE,
                metrics_labels::CONNECTION_GRPC,
                model_id,
                policy.name(),
            );
            Some(EncodeWorkerAssignment { item_index, worker })
        })
        .collect()
}

fn encode_routing_headers(content_hash: &[u8]) -> HeaderMap {
    let mut headers = HeaderMap::new();
    let key = hex_encode(content_hash);
    if let Ok(value) = HeaderValue::from_str(&key) {
        headers.insert("x-smg-routing-key", value);
    }
    headers
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};

    use super::*;
    use crate::{
        config::types::PolicyConfig,
        mesh::adapters::tree_sync::RepairEntry,
        policies::{CacheAwareConfig, CacheAwarePolicy, TreeHandle, TreeKind},
        worker::BasicWorkerBuilder,
    };

    fn worker_with_runtime(
        url: &str,
        worker_type: WorkerType,
        runtime_type: RuntimeType,
    ) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(worker_type)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(runtime_type)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    fn worker(url: &str, worker_type: WorkerType) -> Arc<dyn Worker> {
        worker_with_runtime(url, worker_type, RuntimeType::Sglang)
    }

    fn modeled_worker(
        url: &str,
        model_id: &str,
        worker_type: WorkerType,
        runtime_type: RuntimeType,
    ) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new(model_id))
                .worker_type(worker_type)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(runtime_type)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    fn stage(
        worker_registry: Arc<WorkerRegistry>,
        policy: PolicyConfig,
        prefill_admission: Option<Arc<PrefillAdmission>>,
    ) -> WorkerSelectionStage {
        WorkerSelectionStage::new(
            worker_registry,
            Arc::new(PolicyRegistry::new(policy)),
            WorkerSelectionMode::PrefillDecode,
            prefill_admission,
        )
    }

    fn token_tree_has_tenant(policy: &CacheAwarePolicy, tokens: &[u32], worker_url: &str) -> bool {
        policy
            .open_repair_stream(UNKNOWN_MODEL_ID, TreeKind::Token)
            .expect("token tree should be initialized")
            .any(|entry| {
                matches!(
                    entry,
                    RepairEntry::Token {
                        tokens: path,
                        tenants,
                    } if path == tokens
                        && tenants
                            .iter()
                            .any(|(tenant, _)| tenant.as_ref() == worker_url)
                )
            })
    }

    #[tokio::test]
    async fn admission_filters_full_prefill_before_policy_selection() {
        let registry = Arc::new(WorkerRegistry::new());
        let full = worker("grpc://prefill-full:30000", WorkerType::Prefill);
        let available = worker("grpc://prefill-available:30000", WorkerType::Prefill);
        let decode = worker("grpc://decode:30000", WorkerType::Decode);
        for worker in [&full, &available, &decode] {
            registry.register(Arc::clone(worker)).unwrap();
        }

        let admission = Arc::new(PrefillAdmission::new(1, 0, Duration::from_secs(1)));
        let occupied = admission
            .admit(None, {
                let full = Arc::clone(&full);
                move |capacity| capacity.select(Arc::clone(&full), ())
            })
            .await
            .unwrap();
        let stage = stage(
            registry,
            PolicyConfig::RoundRobin,
            Some(Arc::clone(&admission)),
        );

        let (prefill, selected_decode, _, guards) = stage
            .select_pd_pair(UNKNOWN_MODEL_ID, None, None, None)
            .await
            .unwrap();

        assert_eq!(prefill.url(), available.url());
        assert_eq!(selected_decode.url(), decode.url());
        assert_eq!(full.load(), 1);
        assert_eq!(available.load(), 1);
        assert_eq!(decode.load(), 1);
        drop(guards);
        drop(occupied);
        assert_eq!(full.load(), 0);
        assert_eq!(available.load(), 0);
        assert_eq!(decode.load(), 0);
    }

    #[tokio::test]
    async fn admission_selects_an_available_compatible_runtime() {
        let registry = Arc::new(WorkerRegistry::new());
        let full_prefill = worker("grpc://sglang-prefill:30000", WorkerType::Prefill);
        let sglang_decode = worker("grpc://sglang-decode:30000", WorkerType::Decode);
        let available_prefill = worker_with_runtime(
            "grpc://vllm-prefill:30000",
            WorkerType::Prefill,
            RuntimeType::Vllm,
        );
        let vllm_decode = worker_with_runtime(
            "grpc://vllm-decode:30000",
            WorkerType::Decode,
            RuntimeType::Vllm,
        );
        for worker in [
            &full_prefill,
            &sglang_decode,
            &available_prefill,
            &vllm_decode,
        ] {
            registry.register(Arc::clone(worker)).unwrap();
        }

        let admission = Arc::new(PrefillAdmission::new(1, 0, Duration::from_secs(1)));
        let occupied = admission
            .admit(None, {
                let full_prefill = Arc::clone(&full_prefill);
                move |capacity| capacity.select(Arc::clone(&full_prefill), ())
            })
            .await
            .unwrap();
        let stage = stage(
            registry,
            PolicyConfig::RoundRobin,
            Some(Arc::clone(&admission)),
        );

        let (prefill, decode, runtime, guards) = stage
            .select_pd_pair(UNKNOWN_MODEL_ID, None, None, None)
            .await
            .unwrap();

        assert_eq!(prefill.url(), available_prefill.url());
        assert_eq!(decode.url(), vllm_decode.url());
        assert_eq!(runtime, RuntimeType::Vllm);
        assert_eq!(full_prefill.load(), 1);
        assert_eq!(sglang_decode.load(), 0);
        assert_eq!(available_prefill.load(), 1);
        assert_eq!(vllm_decode.load(), 1);

        drop(guards);
        drop(occupied);
        assert_eq!(full_prefill.load(), 0);
        assert_eq!(available_prefill.load(), 0);
        assert_eq!(vllm_decode.load(), 0);
    }

    #[tokio::test]
    async fn unpaired_capacity_does_not_hide_a_full_pairable_prefill() {
        let registry = Arc::new(WorkerRegistry::new());
        let pairable_prefill = worker("grpc://sglang-prefill:30000", WorkerType::Prefill);
        let decode = worker("grpc://sglang-decode:30000", WorkerType::Decode);
        let unpaired_prefill = worker_with_runtime(
            "grpc://vllm-prefill:30000",
            WorkerType::Prefill,
            RuntimeType::Vllm,
        );
        for worker in [&pairable_prefill, &decode, &unpaired_prefill] {
            registry.register(Arc::clone(worker)).unwrap();
        }

        let admission = Arc::new(PrefillAdmission::new(1, 0, Duration::from_secs(1)));
        let occupied = admission
            .admit(None, {
                let pairable_prefill = Arc::clone(&pairable_prefill);
                move |capacity| capacity.select(Arc::clone(&pairable_prefill), ())
            })
            .await
            .unwrap();
        let stage = stage(
            registry,
            PolicyConfig::RoundRobin,
            Some(Arc::clone(&admission)),
        );

        let result = stage
            .select_pd_pair(UNKNOWN_MODEL_ID, None, None, None)
            .await;

        assert!(matches!(result, Err(PdSelectionError::QueueFull)));
        assert_eq!(pairable_prefill.load(), 1);
        assert_eq!(unpaired_prefill.load(), 0);
        assert_eq!(decode.load(), 0);
        drop(occupied);
    }

    #[tokio::test]
    async fn explicit_target_indexes_the_runtime_filtered_prefill_pool() {
        const MODEL_ID: &str = "mixed-runtime-model";

        let registry = Arc::new(WorkerRegistry::new());
        let first = modeled_worker(
            "grpc://sglang-prefill-1:30000",
            MODEL_ID,
            WorkerType::Prefill,
            RuntimeType::Sglang,
        );
        let other_runtime = modeled_worker(
            "grpc://vllm-prefill:30000",
            MODEL_ID,
            WorkerType::Prefill,
            RuntimeType::Vllm,
        );
        let target = modeled_worker(
            "grpc://sglang-prefill-2:30000",
            MODEL_ID,
            WorkerType::Prefill,
            RuntimeType::Sglang,
        );
        let sglang_decode_first = modeled_worker(
            "grpc://sglang-decode-1:30000",
            MODEL_ID,
            WorkerType::Decode,
            RuntimeType::Sglang,
        );
        let sglang_decode_target = modeled_worker(
            "grpc://sglang-decode-2:30000",
            MODEL_ID,
            WorkerType::Decode,
            RuntimeType::Sglang,
        );
        let vllm_decode = modeled_worker(
            "grpc://vllm-decode:30000",
            MODEL_ID,
            WorkerType::Decode,
            RuntimeType::Vllm,
        );
        for worker in [
            &first,
            &other_runtime,
            &target,
            &sglang_decode_first,
            &sglang_decode_target,
            &vllm_decode,
        ] {
            registry.register(Arc::clone(worker)).unwrap();
        }

        let admission = Arc::new(PrefillAdmission::new(1, 0, Duration::from_secs(1)));
        let stage = stage(registry, PolicyConfig::ConsistentHashing, Some(admission));
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-target-worker", "1".parse().unwrap());

        let (prefill, decode, runtime, guards) = stage
            .select_pd_pair(MODEL_ID, None, None, Some(&headers))
            .await
            .unwrap();

        assert_eq!(prefill.url(), target.url());
        assert_eq!(decode.url(), sglang_decode_target.url());
        assert_eq!(runtime, RuntimeType::Sglang);
        assert_eq!(first.load(), 0);
        assert_eq!(other_runtime.load(), 0);
        assert_eq!(target.load(), 1);
        drop(guards);
    }

    #[tokio::test]
    async fn queued_cache_aware_request_commits_only_after_final_worker_selection() {
        let registry = Arc::new(WorkerRegistry::new());
        let first = worker("grpc://prefill-cache-1:30000", WorkerType::Prefill);
        let second = worker("grpc://prefill-cache-2:30000", WorkerType::Prefill);
        let decode = worker("grpc://decode-cache:30000", WorkerType::Decode);
        for worker in [&first, &second, &decode] {
            registry.register(Arc::clone(worker)).unwrap();
        }

        let cache_policy = Arc::new(CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        }));
        cache_policy.init_workers(&[Arc::clone(&first), Arc::clone(&second)]);
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        policy_registry.set_prefill_policy(cache_policy.clone());

        let admission = Arc::new(PrefillAdmission::new(1, 1, Duration::from_secs(5)));
        let occupied_first = admission
            .admit(None, {
                let first = Arc::clone(&first);
                move |capacity| capacity.select(Arc::clone(&first), ())
            })
            .await
            .unwrap();
        let occupied_second = admission
            .admit(None, {
                let second = Arc::clone(&second);
                move |capacity| capacity.select(Arc::clone(&second), ())
            })
            .await
            .unwrap();
        let stage = Arc::new(WorkerSelectionStage::new(
            registry,
            policy_registry,
            WorkerSelectionMode::PrefillDecode,
            Some(Arc::clone(&admission)),
        ));
        let tokens: Vec<u32> = (1..=16).collect();

        #[expect(
            clippy::disallowed_methods,
            reason = "test waiter task is joined before the test ends"
        )]
        let queued = tokio::spawn({
            let stage = Arc::clone(&stage);
            let tokens = tokens.clone();
            async move {
                stage
                    .select_pd_pair(UNKNOWN_MODEL_ID, None, Some(&tokens), None)
                    .await
            }
        });
        for _ in 0..1_000 {
            if admission.queued_requests() == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(admission.queued_requests(), 1);
        assert_eq!(first.processed_requests(), 0);
        assert_eq!(second.processed_requests(), 0);
        assert!(!token_tree_has_tenant(&cache_policy, &tokens, first.url()));
        assert!(!token_tree_has_tenant(&cache_policy, &tokens, second.url()));

        drop(occupied_second);
        let (selected_prefill, _, _, guards) = queued.await.unwrap().unwrap();

        assert_eq!(selected_prefill.url(), second.url());
        assert_eq!(first.processed_requests(), 0);
        assert_eq!(second.processed_requests(), 1);
        assert!(!token_tree_has_tenant(&cache_policy, &tokens, first.url()));
        assert!(token_tree_has_tenant(&cache_policy, &tokens, second.url()));

        drop(guards);
        drop(occupied_first);
    }

    #[tokio::test]
    async fn admission_does_not_reassign_a_full_explicit_target() {
        let registry = Arc::new(WorkerRegistry::new());
        let first = worker("grpc://prefill-1:30000", WorkerType::Prefill);
        let second = worker("grpc://prefill-2:30000", WorkerType::Prefill);
        let decode = worker("grpc://decode:30000", WorkerType::Decode);
        for worker in [&first, &second, &decode] {
            registry.register(Arc::clone(worker)).unwrap();
        }

        let ordered_prefill: Vec<_> = registry
            .get_workers_filtered(
                None,
                Some(WorkerType::Prefill),
                Some(ConnectionMode::Grpc),
                Some(RuntimeType::Sglang),
                false,
            )
            .into_iter()
            .filter(|worker| worker.is_available())
            .collect();
        assert_eq!(ordered_prefill.len(), 2);
        let target = Arc::clone(&ordered_prefill[0]);
        let alternative = Arc::clone(&ordered_prefill[1]);

        let admission = Arc::new(PrefillAdmission::new(1, 0, Duration::from_secs(1)));
        let occupied = admission
            .admit(None, {
                let target = Arc::clone(&target);
                move |capacity| capacity.select(Arc::clone(&target), ())
            })
            .await
            .unwrap();
        let stage = stage(
            registry,
            PolicyConfig::ConsistentHashing,
            Some(Arc::clone(&admission)),
        );
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-target-worker", "0".parse().unwrap());

        let result = stage
            .select_pd_pair(UNKNOWN_MODEL_ID, None, None, Some(&headers))
            .await;

        assert!(matches!(result, Err(PdSelectionError::QueueFull)));
        assert_eq!(target.load(), 1);
        assert_eq!(alternative.load(), 0);
        assert_eq!(decode.load(), 0);
        drop(occupied);
    }
}
