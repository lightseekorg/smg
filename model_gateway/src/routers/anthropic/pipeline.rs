//! Pipeline orchestrator for Anthropic router
//!
//! This module provides the `MessagesPipeline` that chains all stages together
//! and executes them in sequence to process Messages API requests.

use std::sync::Arc;

use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use tracing::{debug, error, info};

use super::{
    context::{RequestContext, SharedComponents},
    stages::{
        DispatchMetadataStage, PipelineStage, RequestBuildingStage, RequestExecutionStage,
        ResponseProcessingStage, ValidationStage, WorkerSelectionStage,
    },
};
use crate::protocols::messages::CreateMessageRequest;

/// Messages API pipeline that processes requests through stages
///
/// The pipeline executes stages in order:
/// 1. Validation - Validate request fields
/// 2. Worker Selection - Select appropriate worker
/// 3. Request Building - Build HTTP request
/// 4. Dispatch Metadata - Generate request ID and timestamps
/// 5. Request Execution - Send request to worker
/// 6. Response Processing - Parse response and record metrics
pub(crate) struct MessagesPipeline {
    stages: Vec<Box<dyn PipelineStage>>,
}

impl MessagesPipeline {
    pub fn new(components: Arc<SharedComponents>) -> Self {
        let stages: Vec<Box<dyn PipelineStage>> = vec![
            Box::new(ValidationStage::new()),
            Box::new(WorkerSelectionStage::new(
                components.worker_registry.clone(),
            )),
            Box::new(RequestBuildingStage::new()),
            Box::new(DispatchMetadataStage::new()),
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
        request: Arc<CreateMessageRequest>,
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
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Internal error: pipeline completed without response",
        )
            .into_response()
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
