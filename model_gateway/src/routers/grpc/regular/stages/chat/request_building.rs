//! Chat request building stage: Build proto GenerateRequest for chat requests

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
        proto_wrapper::{ProtoGenerateRequest, ProtoRequest},
        utils,
    },
};

/// Chat request building stage
///
/// Extracts chat-specific request building logic from the old unified RequestBuildingStage.
pub(crate) struct ChatRequestBuildingStage {
    inject_pd_metadata: bool,
}

impl ChatRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool) -> Self {
        Self { inject_pd_metadata }
    }
}

#[async_trait]
impl PipelineStage for ChatRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Take preparation state (last consumer — worker_selection already ran)
        let prep = ctx.state.preparation.take().ok_or_else(|| {
            error!(
                function = "ChatRequestBuildingStage::execute",
                "Preparation not completed"
            );
            error::internal_error("preparation_not_completed", "Preparation not completed")
        })?;

        let clients = ctx.state.clients.as_ref().ok_or_else(|| {
            error!(
                function = "ChatRequestBuildingStage::execute",
                "Client acquisition not completed"
            );
            error::internal_error(
                "client_acquisition_not_completed",
                "Client acquisition not completed",
            )
        })?;

        let chat_request = ctx.chat_request_arc();

        // Get client for building request (use prefill client if PD mode)
        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Dual { prefill, .. } => prefill,
        };

        // Build chat request
        let request_id = format!("chatcmpl-{}", Uuid::now_v7());
        let body_ref = prep.filtered_request.as_ref().unwrap_or(&chat_request);

        // Build proto request — take ownership of preparation fields (no clones needed)
        let processed_messages = prep.processed_messages.ok_or_else(|| {
            error!(
                function = "ChatRequestBuildingStage::execute",
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
            .build_chat_request(
                request_id,
                body_ref,
                processed_messages.text,
                prep.token_ids,
                multimodal_data,
                prep.tool_constraints,
            )
            .map_err(|e| {
                error!(function = "ChatRequestBuildingStage::execute", error = %e, "Failed to build generate request");
                error::bad_request("invalid_request_parameters", format!("Invalid request parameters: {e}"))
            })?;

        // Inject tokenized stop sequences for TRT-LLM requests
        if let ProtoGenerateRequest::Trtllm(ref mut req) = proto_request {
            if let Some(stop) = &body_ref.stop {
                if let Some(tokenizer) = ctx.state.tokenizer.as_ref() {
                    utils::inject_trtllm_stop_words(req, tokenizer.as_ref(), stop);
                }
            }
        }

        if self.inject_pd_metadata {
            if let Some(workers) = ctx.state.workers.as_ref() {
                helpers::maybe_inject_pd_metadata(&mut proto_request, workers);
            }
        }

        ctx.state.proto_request = Some(ProtoRequest::Generate(proto_request));
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ChatRequestBuilding"
    }
}
