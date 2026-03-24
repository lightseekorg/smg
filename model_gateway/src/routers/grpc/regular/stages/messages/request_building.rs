//! Message request building stage: Build proto GenerateRequest for message requests

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;
use uuid::Uuid;

use crate::routers::{
    error,
    grpc::{
        common::stages::{helpers, PipelineStage},
        context::{ClientSelection, RequestContext},
        multimodal::assemble_multimodal_data,
        proto_wrapper::ProtoRequest,
    },
};

/// Message request building stage
///
/// Builds a backend-specific proto GenerateRequest from the PreparationOutput
/// and CreateMessageRequest sampling parameters.
pub(crate) struct MessageRequestBuildingStage {
    inject_pd_metadata: bool,
}

impl MessageRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool) -> Self {
        Self { inject_pd_metadata }
    }
}

#[async_trait]
impl PipelineStage for MessageRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Take preparation state (last consumer — worker_selection already ran)
        let prep = ctx.state.preparation.take().ok_or_else(|| {
            error!(
                function = "MessageRequestBuildingStage::execute",
                "Preparation not completed"
            );
            error::internal_error("preparation_not_completed", "Preparation not completed")
        })?;

        let clients = ctx.state.clients.as_ref().ok_or_else(|| {
            error!(
                function = "MessageRequestBuildingStage::execute",
                "Client acquisition not completed"
            );
            error::internal_error(
                "client_acquisition_not_completed",
                "Client acquisition not completed",
            )
        })?;

        let messages_request = ctx.messages_request_arc();

        // Get client for building request (use prefill client if PD mode)
        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Dual { prefill, .. } => prefill,
        };

        // Build message request
        let request_id = format!("msg_{}", Uuid::now_v7());

        // Build proto request — take ownership of preparation fields (no clones needed)
        let processed_messages = prep.processed_messages.ok_or_else(|| {
            error!(
                function = "MessageRequestBuildingStage::execute",
                "processed_messages not set in preparation state"
            );
            error::internal_error(
                "processed_messages_missing",
                "processed_messages not set - this is a bug in the pipeline",
            )
        })?;

        // Assemble backend-specific multimodal data now that the backend is known
        let multimodal_data = processed_messages
            .multimodal_intermediate
            .map(|intermediate| assemble_multimodal_data(intermediate, builder_client));

        let mut proto_request = builder_client
            .build_messages_request(
                request_id,
                &messages_request,
                processed_messages.text,
                prep.token_ids,
                multimodal_data,
                prep.tool_constraints,
            )
            .map_err(|e| {
                error!(function = "MessageRequestBuildingStage::execute", error = %e, "Failed to build generate request");
                error::bad_request("invalid_request_parameters", format!("Invalid request parameters: {e}"))
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
        "MessageRequestBuilding"
    }
}
