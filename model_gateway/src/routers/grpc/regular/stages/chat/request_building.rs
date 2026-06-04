//! Chat request building stage: Build proto GenerateRequest for chat requests

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;
use uuid::Uuid;

use crate::routers::{
    error,
    grpc::{
        client::GenerateRequestBuildOptions,
        common::stages::{helpers, PipelineStage},
        context::{ClientSelection, PreparationOutput, RequestContext},
        multimodal::assemble_multimodal_data,
        proto_wrapper::ProtoRequest,
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
            ClientSelection::Dual { prefill, .. } | ClientSelection::Triple { prefill, .. } => {
                prefill
            }
        };

        let PreparationOutput::Chat {
            token_ids,
            processed_messages,
            tool_constraints,
        } = prep
        else {
            debug_assert!(false, "pipeline guarantees Chat variant");
            return Err(error::internal_error(
                "wrong_preparation_type",
                "Expected Chat preparation output",
            ));
        };

        // Build chat request
        let request_id = format!("chatcmpl-{}", Uuid::now_v7());

        // Reject multimodal for backends that don't support it, before assembling
        if processed_messages.multimodal_intermediate.is_some() && builder_client.is_mlx() {
            return Err(error::bad_request(
                "multimodal_not_supported",
                "MLX backend does not support multimodal inputs".to_string(),
            ));
        }

        // Assemble backend-specific multimodal data now that the backend is known
        let multimodal_data = processed_messages
            .multimodal_intermediate
            .map(|intermediate| {
                assemble_multimodal_data(intermediate, builder_client, ctx.state.workers.as_ref())
            })
            .transpose()
            .map_err(|e| {
                error!(function = "ChatRequestBuildingStage::execute", error = %e, "Failed to assemble multimodal request");
                error::bad_request("multimodal_not_supported", format!("{e}"))
        })?;

        let require_reasoning = ctx.tokenizer_arc().is_some_and(|tokenizer| {
            utils::should_mark_reasoning_started(
                utils::extract_thinking_from_kwargs(
                    chat_request.chat_template_kwargs.as_ref(),
                    tokenizer.as_ref(),
                ),
                tokenizer.as_ref(),
            )
        });

        let mut proto_request = builder_client
            .build_chat_request(
                request_id,
                &chat_request,
                processed_messages.text,
                token_ids,
                GenerateRequestBuildOptions {
                    multimodal_inputs: multimodal_data,
                    tool_constraints,
                    require_reasoning,
                },
            )
            .map_err(|e| {
                error!(function = "ChatRequestBuildingStage::execute", error = %e, "Failed to build generate request");
                error::bad_request("invalid_request_parameters", format!("Invalid request parameters: {e}"))
            })?;

        helpers::apply_sampling_defaults_to_generate_request(
            &mut proto_request,
            &ctx.input.request_type,
            ctx.state.workers.as_ref(),
        );

        if self.inject_pd_metadata {
            if let Some(workers) = ctx.state.workers.as_ref() {
                helpers::maybe_inject_pd_metadata(&mut proto_request, workers);
            }
        }

        // EPD: the encode stage already dispatched this request's images and
        // recorded the per-item handshakes. Inject them so the prefill receives
        // embeddings over Mooncake, and drop the prefill's pixel_values (it skips
        // the vision tower). Present only in the EPD pipeline (the encode stage is
        // the only writer), so non-EPD requests are untouched. Build-then-strip;
        // a metadata-only assemble that skips serializing the prefill pixels
        // entirely is a follow-up optimization.
        if let Some(handshake) = ctx.state.encode_handshake.take() {
            proto_request.set_encode(handshake);
            proto_request.clear_mm_pixel_values();
        }

        // EPD: inject the prefill->decode KV rendezvous so the prefill ships its
        // KV to the paired decode worker over Mooncake. No-op unless this is a
        // TokenSpeed EPD Triple selection; runs before execute_dual_dispatch
        // clones the request, so both prefill and decode carry the same room.
        if let Some(workers) = ctx.state.workers.as_ref() {
            helpers::maybe_inject_tokenspeed_pd_bootstrap(&mut proto_request, workers);
        }

        ctx.state.proto_request = Some(ProtoRequest::Generate(proto_request));
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ChatRequestBuilding"
    }
}
