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
        context::{
            ClientSelection, ExecutionPlan, ExecutionPlanKind, PreparationOutput, RequestContext,
        },
        multimodal::{assemble_multimodal_data, assemble_multimodal_data_after_encode},
        utils,
    },
};

/// Chat request building stage
///
/// Extracts chat-specific request building logic from the old unified RequestBuildingStage.
pub(crate) struct ChatRequestBuildingStage {
    inject_pd_metadata: bool,
    plan_kind: ExecutionPlanKind,
}

impl ChatRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool, plan_kind: ExecutionPlanKind) -> Self {
        Self {
            inject_pd_metadata,
            plan_kind,
        }
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

        // Get client for building request (use prefill client in disaggregated mode)
        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Disaggregated { prefill, .. } => prefill,
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

        // Assemble backend-specific multimodal data now that the backend is known.
        // In EPD, request building also mints the encode->prefill rooms and
        // carries the encode dispatch plan into the execution plan.
        let mut encode_plan = None;
        let multimodal_data = if let Some(intermediate) = processed_messages.multimodal_intermediate
        {
            let planned_encode =
                helpers::plan_epd_encode(&intermediate, clients, ctx.state.workers.as_ref())
                    .map_err(|e| {
                        error!(function = "ChatRequestBuildingStage::execute", error = %e, "Failed to plan EPD encode");
                        error::bad_request("multimodal_not_supported", format!("{e}"))
                    })?;
            let is_encode_routed = planned_encode.is_some();
            encode_plan = planned_encode;

            let assembled = if is_encode_routed {
                assemble_multimodal_data_after_encode(
                    intermediate,
                    builder_client,
                    ctx.state.workers.as_ref(),
                )
                .await
            } else {
                assemble_multimodal_data(intermediate, builder_client, ctx.state.workers.as_ref())
                    .await
            };
            Some(assembled.map_err(|e| {
                error!(function = "ChatRequestBuildingStage::execute", error = %e, "Failed to assemble multimodal request");
                error::bad_request("multimodal_not_supported", format!("{e}"))
            })?)
        } else {
            None
        };

        let require_reasoning = ctx.tokenizer_arc().is_some_and(|tokenizer| {
            utils::should_mark_reasoning_started(
                utils::resolve_user_thinking(
                    chat_request.chat_template_kwargs.as_ref(),
                    chat_request.reasoning_effort.as_deref(),
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

        // EPD: request building minted the per-item encode bootstrap info and
        // planned the encode-worker dispatch. Inject the bootstrap info into the
        // prefill request; the dispatch plan stays attached to ExecutionPlan.
        let encode_dispatch = if let Some(plan) = encode_plan {
            let (bootstrap_info, dispatch) = plan.into_parts();
            proto_request.set_encode_bootstrap_info(bootstrap_info);
            Some(dispatch)
        } else {
            None
        };

        // EPD: inject the prefill->decode KV rendezvous for backends that carry it
        // in the request. Runs before execute_parallel_pd clones the request, so
        // both prefill and decode carry the same room.
        if let Some(workers) = ctx.state.workers.as_ref() {
            helpers::maybe_inject_pd_rendezvous(&mut proto_request, workers);
        }

        ctx.state.execution_plan = Some(ExecutionPlan::generate(
            self.plan_kind,
            proto_request,
            encode_dispatch,
        ));
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ChatRequestBuilding"
    }
}
