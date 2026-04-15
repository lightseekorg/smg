//! Completion request building stage: build proto GenerateRequest from CompletionRequest
//!
//! Stage 4 for the `/v1/completions` pipeline, parallel to `MessageRequestBuildingStage`
//! from the Messages rollout. Builds backend-specific proto `GenerateRequest` from
//! `PreparationOutput` + `CompletionRequest` sampling parameters.
//!
//! Completions has richer sampling knobs than Messages (frequency_penalty, presence_penalty,
//! repetition_penalty, min_p, n, logprobs, structured output constraints) but no tools
//! and no multimodal.

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;
use uuid::Uuid;

use crate::routers::{
    error,
    grpc::{
        common::stages::{helpers, PipelineStage},
        context::{ClientSelection, RequestContext},
        proto_wrapper::ProtoRequest,
    },
};

pub(crate) struct CompletionRequestBuildingStage {
    inject_pd_metadata: bool,
}

impl CompletionRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool) -> Self {
        Self { inject_pd_metadata }
    }
}

#[async_trait]
impl PipelineStage for CompletionRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "CompletionRequestBuildingStage::execute",
                "Preparation not completed"
            );
            error::internal_error("preparation_not_completed", "Preparation not completed")
        })?;

        let clients = ctx.state.clients.as_ref().ok_or_else(|| {
            error!(
                function = "CompletionRequestBuildingStage::execute",
                "Client acquisition not completed"
            );
            error::internal_error(
                "client_acquisition_not_completed",
                "Client acquisition not completed",
            )
        })?;

        let completion_request = ctx.completion_request_arc();

        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Dual { prefill, .. } => prefill,
        };

        let request_id = format!("cmpl_{}", Uuid::now_v7());

        let mut proto_request = builder_client
            .build_completion_request(
                request_id,
                &completion_request,
                prep.routing_text().unwrap_or_default().to_string(),
                prep.token_ids().to_vec(),
            )
            .map_err(|e| {
                error!(
                    function = "CompletionRequestBuildingStage::execute",
                    error = %e,
                    "Failed to build generate request"
                );
                error::bad_request(
                    "invalid_request_parameters",
                    format!("Invalid request parameters: {e}"),
                )
            })?;

        if self.inject_pd_metadata {
            if let Some(workers) = ctx.state.workers.as_ref() {
                helpers::maybe_inject_pd_metadata(&mut proto_request, workers);
            }
        }

        ctx.state.proto_request = Some(ProtoRequest::Generate(proto_request));
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "CompletionRequestBuilding"
    }
}
