//! Pipeline orchestrator for Anthropic router
//!
//! This module provides the `MessagesPipeline` that chains all stages together
//! and executes them in sequence to process Messages API requests.

use std::sync::Arc;

use axum::{http::HeaderMap, response::Response};
use openai_protocol::messages::CreateMessageRequest;
use tracing::{debug, error, info};

use super::{
    context::{RequestContext, SharedComponents},
    stages::{
        PipelineStage, RequestBuildingStage, RequestExecutionStage, ResponseProcessingStage,
        WorkerSelectionStage,
    },
};
use crate::routers::error;

// Note: Arc is still used for SharedComponents (shared across stages) but not for
// CreateMessageRequest (flows through sequential stages without shared ownership)

/// Messages API pipeline that processes requests through stages
///
/// The pipeline executes stages in order:
/// 1. Worker Selection - Select appropriate worker
/// 2. Request Building - Build HTTP request
/// 3. Request Execution - Send request to worker
/// 4. Response Processing - Parse response and record metrics
pub(crate) struct MessagesPipeline {
    stages: Vec<Box<dyn PipelineStage>>,
}

impl MessagesPipeline {
    pub fn new(components: Arc<SharedComponents>) -> Self {
        let stages: Vec<Box<dyn PipelineStage>> = vec![
            Box::new(WorkerSelectionStage::new(
                components.worker_registry.clone(),
            )),
            Box::new(RequestBuildingStage::new()),
            Box::new(RequestExecutionStage::new(
                components.http_client.clone(),
                components.request_timeout,
            )),
            Box::new(ResponseProcessingStage::new()),
        ];

        Self { stages }
    }

    /// Execute the pipeline for a Messages API request
    pub async fn execute(
        &self,
        request: CreateMessageRequest,
        headers: Option<HeaderMap>,
        model_id: &str,
    ) -> Response {
        let streaming = request.stream.unwrap_or(false);

        info!(
            model = %model_id,
            streaming = %streaming,
            "Processing Messages API request through pipeline"
        );

        let mut ctx = RequestContext::new(request, headers, model_id);

        for stage in &self.stages {
            let stage_name = stage.name();
            debug!(stage = %stage_name, "Executing pipeline stage");

            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    // Early return (streaming response)
                    debug!(
                        stage = %stage_name,
                        "Stage returned early response (streaming)"
                    );
                    return response;
                }
                Ok(None) => {
                    // Continue to next stage
                    debug!(stage = %stage_name, "Stage completed, continuing");
                }
                Err(response) => {
                    // Error or final response
                    debug!(
                        stage = %stage_name,
                        status = %response.status(),
                        "Stage returned response"
                    );
                    return response;
                }
            }
        }

        error!("Pipeline completed without producing a response");
        error::internal_error(
            "no_response",
            "Internal error: pipeline completed without response",
        )
    }
}

impl std::fmt::Debug for MessagesPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MessagesPipeline")
            .field("stage_count", &self.stages.len())
            .field(
                "stages",
                &self.stages.iter().map(|s| s.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}
