//! Local worker creation step.

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use openai_protocol::{
    model_card::ModelCard,
    worker::{HealthCheckConfig, WorkerSpec},
};
use tracing::debug;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    app_context::AppContext,
    core::{
        circuit_breaker::CircuitBreakerConfig,
        steps::workflow_data::LocalWorkerWorkflowData,
        worker::{RuntimeType, WorkerType},
        BasicWorkerBuilder, ConnectionMode, Worker, UNKNOWN_MODEL_ID,
    },
};

/// Step 3: Create worker object(s) with merged configuration + metadata.
///
/// This step:
/// 1. Merges discovered labels with config labels
/// 2. Determines the model ID from various sources
/// 3. Creates ModelCard with metadata
/// 4. Builds worker(s) - either single worker or multiple DP-aware workers
/// 5. Outputs unified `workers: Vec<Arc<dyn Worker>>` for downstream steps
pub struct CreateLocalWorkerStep;

#[async_trait]
impl StepExecutor<LocalWorkerWorkflowData> for CreateLocalWorkerStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<LocalWorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let connection_mode =
            context.data.connection_mode.as_ref().ok_or_else(|| {
                WorkflowError::ContextValueNotFound("connection_mode".to_string())
            })?;
        let discovered_labels = &context.data.discovered_labels;

        // Check if worker already exists
        if app_context
            .worker_registry
            .get_by_url(&config.url)
            .is_some()
        {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("create_worker"),
                message: format!("Worker {} already exists", config.url),
            });
        }

        // Build labels from config
        let config_labels = config.labels.clone();

        // Merge: discovered labels first, then config labels (config takes precedence)
        let mut final_labels = discovered_labels.clone();
        for (key, value) in &config_labels {
            final_labels.insert(key.clone(), value.clone());
        }

        // Extract KV transfer config (stored in dedicated metadata fields, not labels)
        let kv_connector = final_labels.remove("kv_connector");
        let kv_role = final_labels.remove("kv_role");

        // Determine model_id: config.models > discovered labels > UNKNOWN_MODEL_ID
        let model_id = config
            .models
            .primary()
            .map(|m| m.id.clone())
            .or_else(|| final_labels.get("served_model_name").cloned()) // SGLang
            .or_else(|| final_labels.get("model_id").cloned()) // TensorRT-LLM
            .or_else(|| final_labels.get("model_path").cloned()) // vLLM
            .unwrap_or_else(|| UNKNOWN_MODEL_ID.to_string());

        if model_id != UNKNOWN_MODEL_ID {
            debug!("Using model_id: {}", model_id);
        }

        // Create ModelCard — use config.models if provided, otherwise build from labels
        let model_card = build_model_card(&model_id, config, &final_labels);

        debug!(
            "Creating worker {} with {} discovered + {} config = {} final labels",
            config.url,
            discovered_labels.len(),
            config_labels.len(),
            final_labels.len()
        );

        // Parse worker type
        let worker_type = parse_worker_type(config);

        // Get runtime type (for gRPC workers)
        let runtime_type = determine_runtime_type(connection_mode, &context.data, config);

        // Build circuit breaker config
        let circuit_breaker_config = build_circuit_breaker_config(app_context);

        // Build health config
        let (health_config, health_endpoint) = build_health_config(app_context, config);

        // Normalize URL
        let normalized_url = normalize_url(&config.url, connection_mode);

        if normalized_url != config.url {
            debug!(
                "Normalized worker URL: {} -> {} ({:?})",
                config.url, normalized_url, connection_mode
            );
        }

        // Create workers - always output as Vec for unified downstream handling
        let dp_aware = app_context.router_config.dp_aware;
        let workers = if dp_aware {
            create_dp_aware_workers(
                &context.data,
                &normalized_url,
                model_card,
                worker_type,
                connection_mode,
                runtime_type,
                circuit_breaker_config,
                health_config,
                &health_endpoint,
                config,
                &final_labels,
                kv_connector.as_deref(),
                kv_role.as_deref(),
            )?
        } else {
            create_single_worker(
                &normalized_url,
                model_card,
                worker_type,
                connection_mode,
                runtime_type,
                circuit_breaker_config,
                health_config,
                &health_endpoint,
                config,
                &final_labels,
                kv_connector.as_deref(),
                kv_role.as_deref(),
            )
        };

        // Update workflow data
        context.data.actual_workers = Some(workers);
        context.data.final_labels = final_labels;
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

fn build_model_card(
    model_id: &str,
    config: &WorkerSpec,
    labels: &HashMap<String, String>,
) -> ModelCard {
    // Start from the user-provided proto_card if it matches, preserving all
    // user-supplied fields (display_name, provider, context_length, etc.).
    // Otherwise start from a blank card with just the model_id.
    let mut card = config
        .models
        .find(model_id)
        .cloned()
        .unwrap_or_else(|| ModelCard::new(model_id));
    if let Some(model_type_str) = labels.get("model_type") {
        card = card.with_hf_model_type(model_type_str.clone());
    }
    if let Some(architectures_json) = labels.get("architectures") {
        if let Ok(architectures) = serde_json::from_str::<Vec<String>>(architectures_json) {
            card = card.with_architectures(architectures);
        }
    }

    // Parse classification model id2label mapping
    // The proto field is id2label_json: JSON string like {"0": "negative", "1": "positive"}
    if let Some(id2label_json) = labels.get("id2label_json") {
        if !id2label_json.is_empty() {
            // Parse JSON: keys are string indices, values are label names
            if let Ok(string_map) = serde_json::from_str::<HashMap<String, String>>(id2label_json) {
                // Convert string keys ("0", "1") to u32 keys (0, 1)
                let id2label: HashMap<u32, String> = string_map
                    .into_iter()
                    .filter_map(|(k, v)| k.parse::<u32>().ok().map(|idx| (idx, v)))
                    .collect();

                if !id2label.is_empty() {
                    card = card.with_id2label(id2label);
                    debug!("Parsed id2label with {} classes", card.num_labels);
                }
            }
        }
    }
    // Fallback: if num_labels is set but id2label wasn't parsed, create default labels
    // Match logic in serving_classify.py::_get_id2label_mapping
    else if let Some(num_labels_str) = labels.get("num_labels") {
        if let Ok(num_labels) = num_labels_str.parse::<u32>() {
            if num_labels > 0 {
                // Create default mapping: {0: "LABEL_0", 1: "LABEL_1", ...}
                let id2label: HashMap<u32, String> = (0..num_labels)
                    .map(|i| (i, format!("LABEL_{}", i)))
                    .collect();
                card = card.with_id2label(id2label);
                debug!("Created default id2label with {} classes", num_labels);
            }
        }
    }

    card
}

fn parse_worker_type(config: &WorkerSpec) -> WorkerType {
    config.worker_type
}

fn determine_runtime_type(
    connection_mode: &ConnectionMode,
    data: &LocalWorkerWorkflowData,
    config: &WorkerSpec,
) -> RuntimeType {
    if !matches!(connection_mode, ConnectionMode::Grpc) {
        return RuntimeType::Sglang;
    }

    // Prefer runtime type detected during connection probing
    if let Some(ref detected_runtime) = data.detected_runtime_type {
        match detected_runtime.as_str() {
            "vllm" => RuntimeType::Vllm,
            "trtllm" => RuntimeType::Trtllm,
            _ => RuntimeType::Sglang,
        }
    } else {
        // Fall back to config's runtime_type
        config.runtime_type
    }
}

fn build_circuit_breaker_config(app_context: &AppContext) -> CircuitBreakerConfig {
    let cfg = app_context.router_config.effective_circuit_breaker_config();
    CircuitBreakerConfig {
        failure_threshold: cfg.failure_threshold,
        success_threshold: cfg.success_threshold,
        timeout_duration: Duration::from_secs(cfg.timeout_duration_secs),
        window_duration: Duration::from_secs(cfg.window_duration_secs),
    }
}

fn build_health_config(
    app_context: &AppContext,
    config: &WorkerSpec,
) -> (HealthCheckConfig, String) {
    let base = app_context.router_config.health_check.to_protocol_config();
    let merged = config.health.apply_to(&base);
    (
        merged,
        app_context.router_config.health_check.endpoint.clone(),
    )
}

fn normalize_url(url: &str, connection_mode: &ConnectionMode) -> String {
    if url.starts_with("http://") || url.starts_with("https://") || url.starts_with("grpc://") {
        url.to_string()
    } else {
        match connection_mode {
            ConnectionMode::Http => format!("http://{}", url),
            ConnectionMode::Grpc => format!("grpc://{}", url),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn create_dp_aware_workers(
    data: &LocalWorkerWorkflowData,
    normalized_url: &str,
    model_card: ModelCard,
    worker_type: WorkerType,
    connection_mode: &ConnectionMode,
    runtime_type: RuntimeType,
    circuit_breaker_config: CircuitBreakerConfig,
    health_config: HealthCheckConfig,
    health_endpoint: &str,
    config: &WorkerSpec,
    labels: &HashMap<String, String>,
    kv_connector: Option<&str>,
    kv_role: Option<&str>,
) -> Result<Vec<Arc<dyn Worker>>, WorkflowError> {
    let dp_info = data
        .dp_info
        .as_ref()
        .ok_or_else(|| WorkflowError::ContextValueNotFound("dp_info".to_string()))?;

    debug!(
        "Creating {} DP-aware workers for {} (dp_size: {})",
        dp_info.dp_size, normalized_url, dp_info.dp_size
    );

    let mut workers = Vec::with_capacity(dp_info.dp_size);
    for rank in 0..dp_info.dp_size {
        let mut builder = BasicWorkerBuilder::new(normalized_url.to_string())
            .dp_config(rank, dp_info.dp_size)
            .model(model_card.clone())
            .worker_type(worker_type)
            .connection_mode(*connection_mode)
            .runtime_type(runtime_type)
            .circuit_breaker_config(circuit_breaker_config.clone())
            .health_config(health_config.clone())
            .health_endpoint(health_endpoint)
            .bootstrap_port(config.bootstrap_port)
            .priority(config.priority)
            .cost(config.cost);

        if let Some(ref api_key) = config.api_key {
            builder = builder.api_key(api_key.clone());
        }
        if !labels.is_empty() {
            builder = builder.labels(labels.clone());
        }
        if let Some(connector) = kv_connector {
            builder = builder.kv_connector(connector);
        }
        if let Some(role) = kv_role {
            builder = builder.kv_role(role);
        }

        let worker = Arc::new(builder.build()) as Arc<dyn Worker>;
        if health_config.disable_health_check {
            worker.set_healthy(true);
        } else {
            worker.set_healthy(false);
        }
        workers.push(worker);

        debug!(
            "Created DP-aware worker {}@{}/{} ({:?})",
            normalized_url, rank, dp_info.dp_size, connection_mode
        );
    }

    Ok(workers)
}

#[allow(clippy::too_many_arguments)]
fn create_single_worker(
    normalized_url: &str,
    model_card: ModelCard,
    worker_type: WorkerType,
    connection_mode: &ConnectionMode,
    runtime_type: RuntimeType,
    circuit_breaker_config: CircuitBreakerConfig,
    health_config: HealthCheckConfig,
    health_endpoint: &str,
    config: &WorkerSpec,
    labels: &HashMap<String, String>,
    kv_connector: Option<&str>,
    kv_role: Option<&str>,
) -> Vec<Arc<dyn Worker>> {
    let health_check_disabled = health_config.disable_health_check;

    let mut builder = BasicWorkerBuilder::new(normalized_url.to_string())
        .model(model_card)
        .worker_type(worker_type)
        .connection_mode(*connection_mode)
        .runtime_type(runtime_type)
        .circuit_breaker_config(circuit_breaker_config)
        .health_config(health_config)
        .health_endpoint(health_endpoint)
        .bootstrap_port(config.bootstrap_port)
        .priority(config.priority)
        .cost(config.cost);

    if let Some(ref api_key) = config.api_key {
        builder = builder.api_key(api_key.clone());
    }
    if !labels.is_empty() {
        builder = builder.labels(labels.clone());
    }
    if let Some(connector) = kv_connector {
        builder = builder.kv_connector(connector);
    }
    if let Some(role) = kv_role {
        builder = builder.kv_role(role);
    }

    let worker = Arc::new(builder.build()) as Arc<dyn Worker>;
    if health_check_disabled {
        worker.set_healthy(true);
    } else {
        worker.set_healthy(false);
    }

    debug!(
        "Created worker object for {} ({:?}) with {} labels",
        normalized_url,
        connection_mode,
        labels.len()
    );

    vec![worker]
}
