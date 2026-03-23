pub mod classify;
pub mod external;
pub mod local;
pub mod shared;
pub(crate) mod util;

use std::{sync::Arc, time::Duration};

pub use classify::ClassifyWorkerTypeStep;
pub use external::{
    group_models_into_cards, infer_model_type_from_id, CreateExternalWorkersStep,
    DiscoverModelsStep, ModelInfo, ModelsResponse,
};
pub use local::{
    create_worker_removal_workflow, create_worker_removal_workflow_data,
    create_worker_update_workflow, create_worker_update_workflow_data, CreateLocalWorkerStep,
    DetectConnectionModeStep, DiscoverDPInfoStep, DiscoverMetadataStep, DpInfo,
    FindWorkerToUpdateStep, FindWorkersToRemoveStep, RemoveFromPolicyRegistryStep,
    RemoveFromWorkerRegistryStep, UpdatePoliciesForWorkerStep, UpdateRemainingPoliciesStep,
    UpdateWorkerPropertiesStep, WorkerRemovalRequest,
};
use local::{DetectBackendStep, DiscoverDPInfoStep as DPStep, SubmitTokenizerJobStep};
use openai_protocol::worker::WorkerSpec;
pub use shared::{ActivateWorkersStep, RegisterWorkersStep, UpdatePoliciesStep, WorkerList};
use wfaas::{BackoffStrategy, FailureAction, RetryPolicy, StepDefinition, WorkflowDefinition};

use crate::{
    app_context::AppContext, config::RouterConfig, core::steps::workflow_data::WorkerWorkflowData,
};

/// Create the unified worker registration workflow definition.
///
/// DAG structure:
/// ```text
///            classify_worker_type
///                    |
///       +------------+------------------+
///       |  (LOCAL branch)               |  (EXTERNAL branch)
///       v                               v
///  detect_connection_mode         discover_models
///       |                               |
///  detect_backend                       |
///       |                               |
///  discover_metadata                    |
///       |                               |
///  discover_dp_info                     |
///       |                               |
///  create_local_worker           create_external_workers
///       |                               |
///       +---------------+---------------+
///                        |
///                 register_workers  (shared)
///                        |
///           +------------+------------+
///           |            |            |
///      update_policies  submit_tok  activate_workers
///                       (local only)
/// ```
pub fn create_worker_registration_workflow(
    router_config: &RouterConfig,
) -> WorkflowDefinition<WorkerWorkflowData> {
    let detect_timeout = Duration::from_secs(router_config.worker_startup_timeout_secs);

    // Reserve 10% of the startup timeout for workflow overhead (step transitions, etc.).
    // The remaining budget is split into retry attempts: a base number of attempts for
    // the first ATTEMPTS_THRESHOLD seconds, plus one extra attempt per SECS_PER_EXTRA
    // seconds beyond that.
    const EFFECTIVE_TIMEOUT_FACTOR: f64 = 0.9;
    const ATTEMPTS_THRESHOLD_SECS: f64 = 10.0;
    const BASE_ATTEMPTS: u32 = 5;
    const SECS_PER_EXTRA_ATTEMPT: f64 = 5.0;
    const MIN_ATTEMPTS: u32 = 3;

    let timeout_secs = detect_timeout.as_secs() as f64;
    let effective_timeout = timeout_secs * EFFECTIVE_TIMEOUT_FACTOR;
    let max_attempts = if effective_timeout > ATTEMPTS_THRESHOLD_SECS {
        (BASE_ATTEMPTS
            + ((effective_timeout - ATTEMPTS_THRESHOLD_SECS) / SECS_PER_EXTRA_ATTEMPT).ceil()
                as u32)
            .max(MIN_ATTEMPTS)
    } else {
        MIN_ATTEMPTS
    };

    WorkflowDefinition::new("worker_registration", "Worker Registration")
        // Step 0: Classify worker type (Local vs External)
        .add_step(
            StepDefinition::new(
                "classify_worker_type",
                "Classify Worker Type",
                Arc::new(ClassifyWorkerTypeStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_failure_action(FailureAction::FailWorkflow),
        )
        // === LOCAL BRANCH ===
        // Step 1: Detect connection mode (HTTP vs gRPC)
        .add_step(
            StepDefinition::new(
                "detect_connection_mode",
                "Detect Connection Mode",
                Arc::new(DetectConnectionModeStep),
            )
            .with_retry(RetryPolicy {
                max_attempts,
                backoff: BackoffStrategy::Linear {
                    increment: Duration::from_secs(1),
                    max: Duration::from_secs(5),
                },
            })
            .with_timeout(detect_timeout)
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["classify_worker_type"]),
        )
        // Step 1.5: Detect backend runtime (sglang, vllm, trtllm)
        .add_step(
            StepDefinition::new(
                "detect_backend",
                "Detect Backend",
                Arc::new(DetectBackendStep),
            )
            .with_retry(RetryPolicy {
                max_attempts,
                backoff: BackoffStrategy::Linear {
                    increment: Duration::from_secs(1),
                    max: Duration::from_secs(5),
                },
            })
            .with_timeout(Duration::from_secs(10))
            .with_failure_action(FailureAction::ContinueNextStep)
            .depends_on(&["detect_connection_mode"]),
        )
        // Step 2a: Discover metadata
        .add_step(
            StepDefinition::new(
                "discover_metadata",
                "Discover Metadata",
                Arc::new(DiscoverMetadataStep),
            )
            .with_retry(RetryPolicy {
                max_attempts: 3,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
            })
            .with_timeout(Duration::from_secs(10))
            .with_failure_action(FailureAction::ContinueNextStep)
            .depends_on(&["detect_backend"]),
        )
        // Step 2b: Discover DP info (after metadata)
        .add_step(
            StepDefinition::new("discover_dp_info", "Discover DP Info", Arc::new(DPStep))
                .with_retry(RetryPolicy {
                    max_attempts: 3,
                    backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
                })
                .with_timeout(Duration::from_secs(10))
                .with_failure_action(FailureAction::FailWorkflow)
                .depends_on(&["discover_metadata"]),
        )
        // Step 3 (local): Create local worker(s)
        .add_step(
            StepDefinition::new(
                "create_local_worker",
                "Create Local Worker",
                Arc::new(CreateLocalWorkerStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["discover_dp_info"]),
        )
        // === EXTERNAL BRANCH ===
        // Step 1 (external): Discover models from /v1/models
        .add_step(
            StepDefinition::new(
                "discover_models",
                "Discover Models",
                Arc::new(DiscoverModelsStep),
            )
            .with_retry(RetryPolicy {
                max_attempts: 3,
                backoff: BackoffStrategy::Exponential {
                    base: Duration::from_secs(1),
                    max: Duration::from_secs(10),
                },
            })
            .with_timeout(Duration::from_secs(30))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["classify_worker_type"]),
        )
        // Step 2 (external): Create external workers
        .add_step(
            StepDefinition::new(
                "create_external_workers",
                "Create External Workers",
                Arc::new(CreateExternalWorkersStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["discover_models"]),
        )
        // === SHARED (both branches converge) ===
        // Step 4: Register workers
        .add_step(
            StepDefinition::new(
                "register_workers",
                "Register Workers",
                Arc::new(RegisterWorkersStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["create_local_worker", "create_external_workers"]),
        )
        // Step 5a: Submit tokenizer job (local only)
        .add_step(
            StepDefinition::new(
                "submit_tokenizer_job",
                "Submit Tokenizer Job",
                Arc::new(SubmitTokenizerJobStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::ContinueNextStep)
            .depends_on(&["register_workers"]),
        )
        // Step 5b: Update policies
        .add_step(
            StepDefinition::new(
                "update_policies",
                "Update Policies",
                Arc::new(UpdatePoliciesStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::ContinueNextStep)
            .depends_on(&["register_workers"]),
        )
        // Step 5c: Activate workers
        .add_step(
            StepDefinition::new(
                "activate_workers",
                "Activate Workers",
                Arc::new(ActivateWorkersStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["register_workers"]),
        )
}

/// Create initial workflow data for the unified worker registration workflow.
pub fn create_worker_workflow_data(
    config: WorkerSpec,
    app_context: Arc<AppContext>,
) -> WorkerWorkflowData {
    WorkerWorkflowData {
        config,
        worker_kind: None,
        connection_mode: None,
        detected_runtime_type: None,
        discovered_labels: std::collections::HashMap::new(),
        dp_info: None,
        model_cards: Vec::new(),
        workers: None,
        final_labels: std::collections::HashMap::new(),
        app_context: Some(app_context),
        actual_workers: None,
    }
}
