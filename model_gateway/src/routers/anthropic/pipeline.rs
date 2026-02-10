//! Pipeline orchestrator for Anthropic router
//!
//! This module provides the `MessagesPipeline` that chains all stages together
//! and executes them in sequence to process Messages API requests.

use std::sync::Arc;

use axum::{http::HeaderMap, response::Response};
use tracing::{debug, error};

use super::{
    context::{RequestContext, SharedComponents},
    stages::{
        PipelineStage, RequestBuildingStage, RequestExecutionStage, ResponseProcessingStage,
        WorkerSelectionStage,
    },
};
use crate::{
    core::WorkerRegistry,
    protocols::messages::{CreateMessageRequest, Message},
    routers::error,
};

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
    pub fn new(components: Arc<SharedComponents>, worker_registry: Arc<WorkerRegistry>) -> Self {
        let stages: Vec<Box<dyn PipelineStage>> = vec![
            Box::new(WorkerSelectionStage::new(worker_registry)),
            Box::new(RequestBuildingStage::new()),
            Box::new(RequestExecutionStage::new(
                components.http_client.clone(),
                components.request_timeout,
            )),
            Box::new(ResponseProcessingStage::new()),
        ];

        Self { stages }
    }

    /// Execute the pipeline for a streaming request, returning the SSE `Response`.
    pub async fn execute(
        &self,
        request: CreateMessageRequest,
        headers: Option<HeaderMap>,
        model_id: &str,
    ) -> Response {
        let mut ctx = RequestContext::new(request, headers, model_id);

        match self.run_stages(&mut ctx, &self.stages).await {
            Some(response) => response,
            None => {
                error!(function = "execute", "No response produced by pipeline");
                error::internal_error("no_response_produced", "No response produced")
            }
        }
    }

    /// Execute the pipeline for a non-streaming request, returning the parsed `Message`.
    pub async fn execute_for_messages(
        &self,
        request: CreateMessageRequest,
        headers: Option<HeaderMap>,
        model_id: &str,
    ) -> Result<Message, Response> {
        if request.stream.unwrap_or(false) {
            return Err(error::bad_request(
                "invalid_request",
                "execute_for_messages does not support streaming requests",
            ));
        }
        let mut ctx = RequestContext::new(request, headers, model_id);

        if let Some(response) = self.run_stages(&mut ctx, &self.stages).await {
            return Err(response);
        }

        match ctx.state.parsed_message {
            Some(message) => Ok(message),
            None => {
                error!(
                    function = "execute_for_messages",
                    "No parsed message produced by pipeline"
                );
                Err(error::internal_error(
                    "no_response_produced",
                    "No response produced",
                ))
            }
        }
    }

    /// Run a set of stages, returning None if all complete with Ok(None),
    /// or Some(Response) if any stage returns early or errors.
    async fn run_stages(
        &self,
        ctx: &mut RequestContext,
        stages: &[Box<dyn PipelineStage>],
    ) -> Option<Response> {
        for stage in stages {
            let stage_name = stage.name();
            debug!(stage = %stage_name, "Executing pipeline stage");

            match stage.execute(ctx).await {
                Ok(Some(response)) => {
                    debug!(
                        stage = %stage_name,
                        "Stage returned early response"
                    );
                    return Some(response);
                }
                Ok(None) => {
                    debug!(stage = %stage_name, "Stage completed, continuing");
                }
                Err(response) => {
                    debug!(
                        stage = %stage_name,
                        status = %response.status(),
                        "Stage returned error response"
                    );
                    return Some(response);
                }
            }
        }
        None
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
