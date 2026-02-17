//! External worker creation step.

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use tracing::{debug, info};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::core::{
    circuit_breaker::CircuitBreakerConfig,
    steps::workflow_data::{ExternalWorkerWorkflowData, WorkerList},
    worker::{RuntimeType, WorkerType},
    BasicWorkerBuilder, ConnectionMode, Worker,
};

/// Normalize URL for external APIs (ensure https://).
fn normalize_external_url(url: &str) -> String {
    if url.starts_with("http://") || url.starts_with("https://") {
        url.to_string()
    } else {
        format!("https://{}", url)
    }
}

/// Step 2: Create worker objects for each discovered model.
pub struct CreateExternalWorkersStep;

#[async_trait]
impl StepExecutor<ExternalWorkerWorkflowData> for CreateExternalWorkersStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<ExternalWorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let model_cards = &context.data.model_cards;

        // Build configs from router settings
        let circuit_breaker_config = {
            let cfg = app_context.router_config.effective_circuit_breaker_config();
            CircuitBreakerConfig {
                failure_threshold: cfg.failure_threshold,
                success_threshold: cfg.success_threshold,
                timeout_duration: Duration::from_secs(cfg.timeout_duration_secs),
                window_duration: Duration::from_secs(cfg.window_duration_secs),
            }
        };

        let (health_config, health_endpoint) = {
            let base = app_context.router_config.health_check.to_protocol_config();
            let mut merged = config.health.apply_to(&base);
            // External workers (OpenAI, Anthropic, etc.) should not be health-checked
            // by default — they are third-party APIs that don't expose a /health endpoint.
            // Only apply the default if the user hasn't explicitly overridden it.
            if config.health.disable_health_check.is_none() {
                merged.disable_health_check = true;
            }
            (
                merged,
                app_context.router_config.health_check.endpoint.clone(),
            )
        };

        // Build labels from config
        let labels: HashMap<String, String> = config.labels.clone();

        // Normalize URL (ensure https:// for external APIs)
        let normalized_url = normalize_external_url(&config.url);

        let mut workers = Vec::new();

        // Handle wildcard mode: create a single worker with empty models list
        if model_cards.is_empty() {
            debug!("Creating wildcard worker (no models) for {}", config.url);

            let mut builder = BasicWorkerBuilder::new(normalized_url.clone())
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Http)
                .runtime_type(RuntimeType::External)
                .circuit_breaker_config(circuit_breaker_config.clone())
                .health_config(health_config.clone())
                .health_endpoint(&health_endpoint)
                .priority(config.priority)
                .cost(config.cost);

            if let Some(ref api_key) = config.api_key {
                builder = builder.api_key(api_key.clone());
            }

            if !labels.is_empty() {
                builder = builder.labels(labels.clone());
            }

            let worker = Arc::new(builder.build()) as Arc<dyn Worker>;
            if health_config.disable_health_check {
                worker.set_healthy(true);
            } else {
                worker.set_healthy(false);
            }

            info!(
                "Created wildcard worker at {} (accepts any model, user auth forwarded)",
                normalized_url
            );

            workers.push(worker);
        } else {
            debug!(
                "Creating {} external workers for {}",
                model_cards.len(),
                config.url
            );

            // Create a worker for each model
            for model_card in model_cards.iter() {
                let mut builder = BasicWorkerBuilder::new(normalized_url.clone())
                    .model(model_card.clone())
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Http)
                    .runtime_type(RuntimeType::External)
                    .circuit_breaker_config(circuit_breaker_config.clone())
                    .health_config(health_config.clone())
                    .health_endpoint(&health_endpoint)
                    .priority(config.priority)
                    .cost(config.cost);

                if let Some(ref api_key) = config.api_key {
                    builder = builder.api_key(api_key.clone());
                }

                if !labels.is_empty() {
                    builder = builder.labels(labels.clone());
                }

                let worker = Arc::new(builder.build()) as Arc<dyn Worker>;
                if health_config.disable_health_check {
                    worker.set_healthy(true);
                } else {
                    worker.set_healthy(false);
                }

                debug!(
                    "Created external worker for model {} at {}",
                    model_card.id, normalized_url
                );

                workers.push(worker);
            }

            info!(
                "Created {} external workers from {}",
                workers.len(),
                config.url
            );
        }

        // Store results in workflow data
        context.data.workers = Some(WorkerList::from_workers(&workers));
        context.data.actual_workers = Some(workers);
        context.data.labels = labels;
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
