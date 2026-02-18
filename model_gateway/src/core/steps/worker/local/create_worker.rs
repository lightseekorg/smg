//! Local worker creation step.

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use openai_protocol::{model_card::ModelCard, model_type::ModelType, worker::WorkerSpec};
use tracing::debug;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::core::{
    circuit_breaker::CircuitBreakerConfig, steps::workflow_data::LocalWorkerWorkflowData,
    worker::RuntimeType, BasicWorkerBuilder, ConnectionMode, Worker, UNKNOWN_MODEL_ID,
};

/// Step 3: Create worker object(s) with merged configuration + metadata.
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

        // Merge labels: discovered first, then config (config takes precedence)
        let mut labels = context.data.discovered_labels.clone();
        for (key, value) in &config.labels {
            labels.insert(key.clone(), value.clone());
        }

        // Extract KV transfer config (dedicated metadata fields, not labels)
        let kv_connector = labels.remove("kv_connector");
        let kv_role = labels.remove("kv_role");

        // Determine model_id: config.models > discovered labels > UNKNOWN_MODEL_ID
        let model_id = config
            .models
            .primary()
            .map(|m| m.id.clone())
            .or_else(|| labels.get("served_model_name").cloned())
            .or_else(|| labels.get("model_id").cloned())
            .or_else(|| labels.get("model_path").cloned())
            .unwrap_or_else(|| UNKNOWN_MODEL_ID.to_string());

        let model_card = build_model_card(&model_id, config, &labels);

        let runtime_type = match context.data.detected_runtime_type.as_deref() {
            Some(s) => s.parse::<RuntimeType>().unwrap_or(config.runtime_type),
            None => config.runtime_type,
        };

        // Normalize URL
        let url = normalize_url(&config.url, connection_mode);

        // Build workers
        let cb_cfg = app_context.router_config.effective_circuit_breaker_config();
        let circuit_breaker = CircuitBreakerConfig {
            failure_threshold: cb_cfg.failure_threshold,
            success_threshold: cb_cfg.success_threshold,
            timeout_duration: Duration::from_secs(cb_cfg.timeout_duration_secs),
            window_duration: Duration::from_secs(cb_cfg.window_duration_secs),
        };
        let health_base = app_context.router_config.health_check.to_protocol_config();
        let health_config = config.health.apply_to(&health_base);
        let health_endpoint = &app_context.router_config.health_check.endpoint;

        let dp_ranks: Vec<Option<(usize, usize)>> = if app_context.router_config.dp_aware {
            let dp_info = context
                .data
                .dp_info
                .as_ref()
                .ok_or_else(|| WorkflowError::ContextValueNotFound("dp_info".to_string()))?;
            (0..dp_info.dp_size)
                .map(|r| Some((r, dp_info.dp_size)))
                .collect()
        } else {
            vec![None] // single worker, no DP
        };

        let workers: Vec<Arc<dyn Worker>> = dp_ranks
            .into_iter()
            .map(|dp| {
                let mut builder = BasicWorkerBuilder::new(url.clone())
                    .model(model_card.clone())
                    .worker_type(config.worker_type)
                    .connection_mode(*connection_mode)
                    .runtime_type(runtime_type)
                    .circuit_breaker_config(circuit_breaker.clone())
                    .health_config(health_config.clone())
                    .health_endpoint(health_endpoint)
                    .bootstrap_port(config.bootstrap_port)
                    .priority(config.priority)
                    .cost(config.cost);

                if let Some((rank, size)) = dp {
                    builder = builder.dp_config(rank, size);
                }
                if let Some(ref key) = config.api_key {
                    builder = builder.api_key(key.clone());
                }
                if !labels.is_empty() {
                    builder = builder.labels(labels.clone());
                }
                if let Some(ref c) = kv_connector {
                    builder = builder.kv_connector(c);
                }
                if let Some(ref r) = kv_role {
                    builder = builder.kv_role(r);
                }

                let worker = Arc::new(builder.build()) as Arc<dyn Worker>;
                worker.set_healthy(health_config.disable_health_check);
                worker
            })
            .collect();

        debug!(
            "Created {} worker(s) for {} ({:?}, {} labels)",
            workers.len(),
            url,
            connection_mode,
            labels.len()
        );

        context.data.actual_workers = Some(workers);
        context.data.final_labels = labels;
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
    let user_provided = config.models.find(model_id).is_some();
    let mut card = config
        .models
        .find(model_id)
        .cloned()
        .unwrap_or_else(|| ModelCard::new(model_id));

    if let Some(mt) = labels.get("model_type") {
        card = card.with_hf_model_type(mt.clone());
    }
    if let Some(archs_json) = labels.get("architectures") {
        if let Ok(archs) = serde_json::from_str::<Vec<String>>(archs_json) {
            card = card.with_architectures(archs);
        }
    }

    // Classification model id2label
    if let Some(json) = labels.get("id2label_json").filter(|s| !s.is_empty()) {
        if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(json) {
            let id2label: HashMap<u32, String> = map
                .into_iter()
                .filter_map(|(k, v)| k.parse::<u32>().ok().map(|i| (i, v)))
                .collect();
            if !id2label.is_empty() {
                card = card.with_id2label(id2label);
            }
        }
    } else if let Some(n) = labels
        .get("num_labels")
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| n > 0)
    {
        let id2label: HashMap<u32, String> = (0..n).map(|i| (i, format!("LABEL_{}", i))).collect();
        card = card.with_id2label(id2label);
    }

    // Fill context_length from whichever backend key is available
    if card.context_length.is_none() {
        card.context_length = [
            "context_length",
            "max_context_length",
            "max_model_len",
            "max_total_tokens",
            "max_seq_len",
        ]
        .iter()
        .find_map(|k| labels.get(*k))
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| n > 0);
    }

    // Fill tokenizer_path
    if card.tokenizer_path.is_none() {
        card.tokenizer_path = labels
            .get("tokenizer_path")
            .filter(|s| !s.is_empty())
            .cloned();
    }

    // Infer model_type capabilities from discovered signals
    let has_vision = labels
        .get("supports_vision")
        .or(labels.get("has_image_understanding"))
        .map(|s| s == "true")
        .unwrap_or(false);

    if !user_provided {
        let is_embedding = labels.get("is_embedding").is_some_and(|s| s == "true");
        let is_non_generation = labels.get("is_generation").is_some_and(|s| s == "false");

        if is_embedding || is_non_generation {
            card.model_type = infer_non_generation_type(labels);
        } else if has_vision && !card.model_type.supports_vision() {
            card.model_type |= ModelType::VISION;
        }
    } else if has_vision && !card.model_type.supports_vision() {
        card.model_type |= ModelType::VISION;
    }

    card
}

/// Determine embedding vs rerank from architecture/model_type hints.
fn infer_non_generation_type(labels: &HashMap<String, String>) -> ModelType {
    if let Some(archs_json) = labels.get("architectures") {
        if let Ok(archs) = serde_json::from_str::<Vec<String>>(archs_json) {
            let joined = archs.join(" ").to_lowercase();
            if joined.contains("rerank") || joined.contains("crossencoder") {
                return ModelType::RERANK;
            }
        }
    }
    if let Some(mt) = labels.get("model_type") {
        if mt.to_lowercase().contains("rerank") {
            return ModelType::RERANK;
        }
    }
    ModelType::EMBEDDINGS
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
