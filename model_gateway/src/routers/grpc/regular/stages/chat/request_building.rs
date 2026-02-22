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
        multimodal,
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
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
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
        let request_id = format!("chatcmpl-{}", Uuid::new_v4());
        let body_ref = prep.filtered_request.as_ref().unwrap_or(&chat_request);

        // Build proto request using centralized dispatch
        let processed_messages = prep.processed_messages.as_ref().ok_or_else(|| {
            error!(
                function = "ChatRequestBuildingStage::execute",
                "processed_messages not set in preparation state"
            );
            error::internal_error(
                "processed_messages_missing",
                "processed_messages not set - this is a bug in the pipeline",
            )
        })?;

        // Backend-specific multimodal processing (images were fetched in preparation stage).
        // SGLang: full pixel preprocessing + token expansion.
        // vLLM: raw image bytes only — it handles preprocessing internally.
        // TODO: Token expansion here means WorkerSelectionStage runs on unexpanded tokens.
        // For PrefixHash routing, this reduces KV-cache locality on image-heavy traffic.
        // Consider lifting expansion earlier or passing expanded IDs to the routing policy.
        let (token_ids, multimodal_data) = if let Some(ref images) =
            processed_messages.multimodal_images
        {
            let model_id = ctx.input.model_id.as_deref().unwrap_or(&chat_request.model);
            let tokenizer = ctx.state.tokenizer.as_deref().ok_or_else(|| {
                error!(
                    function = "ChatRequestBuildingStage::execute",
                    "tokenizer not set"
                );
                error::internal_error(
                    "tokenizer_missing",
                    "tokenizer not set for multimodal processing",
                )
            })?;
            let mm_components = ctx.components.multimodal.as_ref().ok_or_else(|| {
                error!(
                    function = "ChatRequestBuildingStage::execute",
                    "multimodal components not initialized"
                );
                error::internal_error(
                    "multimodal_not_configured",
                    "multimodal components not initialized",
                )
            })?;
            let tokenizer_source = ctx
                .components
                .tokenizer_registry
                .get_by_name(model_id)
                .map(|e| e.source)
                .unwrap_or_default();

            let (ids, data) = multimodal::process_for_backend(
                    images,
                    builder_client.is_sglang(),
                    model_id,
                    tokenizer,
                    prep.token_ids.clone(),
                    mm_components,
                    &tokenizer_source,
                )
                .map_err(|e| {
                    error!(function = "ChatRequestBuildingStage::execute", error = %e, "Multimodal processing failed");
                    error::bad_request("multimodal_failed", e.to_string())
                })?;
            (ids, Some(data))
        } else {
            (prep.token_ids.clone(), None)
        };

        let mut proto_request = builder_client
            .build_chat_request(
                request_id,
                body_ref,
                processed_messages.text.clone(),
                token_ids,
                multimodal_data,
                prep.tool_constraints.clone(),
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
