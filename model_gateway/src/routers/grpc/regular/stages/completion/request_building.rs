//! Completion request building stage: build proto GenerateRequest(s) from CompletionRequest
//!
//! Stage 4 for the `/v1/completions` pipeline, parallel to `MessageRequestBuildingStage`
//! from the Messages rollout. Builds backend-specific proto `GenerateRequest`s from
//! `PreparationOutput` + `CompletionRequest` sampling parameters — one per prompt.
//!
//! Completions has richer sampling knobs than Messages (frequency_penalty, presence_penalty,
//! repetition_penalty, min_p, n, logprobs, structured output constraints) but no tools
//! and no multimodal.

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::completion::CompletionRequest;
use tracing::error;
use uuid::Uuid;

use crate::routers::{
    error,
    grpc::{
        client::GrpcClient,
        common::stages::{helpers, PipelineStage},
        context::{
            ClientSelection, CompletionItem, ExecutionPlan, ExecutionPlanKind, PreparationOutput,
            RequestContext, RequestType, WorkerSelection,
        },
        proto_wrapper::ProtoGenerateRequest,
    },
};

pub(crate) struct CompletionRequestBuildingStage {
    inject_pd_metadata: bool,
    plan_kind: ExecutionPlanKind,
}

impl CompletionRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool, plan_kind: ExecutionPlanKind) -> Self {
        Self {
            inject_pd_metadata,
            plan_kind,
        }
    }

    /// Build one backend request for one prompt. PD bootstrap rooms are minted
    /// per call, so injection runs per sub-request rather than
    /// build-once-then-clone.
    #[expect(
        clippy::result_large_err,
        reason = "Response is the standard error type in the pipeline stage pattern"
    )]
    fn build_proto_request(
        &self,
        builder_client: &GrpcClient,
        request_id: String,
        item: &CompletionItem,
        completion_request: &CompletionRequest,
        request_type: &RequestType,
        workers: Option<&WorkerSelection>,
    ) -> Result<ProtoGenerateRequest, Response> {
        let mut proto_request = builder_client
            .build_completion_request(
                request_id,
                completion_request,
                item.text.clone(),
                item.token_ids.clone(),
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

        helpers::apply_sampling_defaults_to_generate_request(
            &mut proto_request,
            request_type,
            workers,
        );

        if self.inject_pd_metadata {
            if let Some(workers) = workers {
                helpers::maybe_inject_pd_metadata(&mut proto_request, workers);
            }
        }

        // EPD: inject the prefill->decode KV rendezvous. Completion EPD is
        // text-only (no encode jobs), so this is the only EPD injection here.
        // No-op unless the backend carries it in the request.
        if let Some(workers) = workers {
            helpers::maybe_inject_pd_rendezvous(&mut proto_request, workers);
        }

        Ok(proto_request)
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

        let PreparationOutput::Completion { items, .. } = prep else {
            error!(
                function = "CompletionRequestBuildingStage::execute",
                "Preparation output is not a completion"
            );
            return Err(error::internal_error(
                "unexpected_preparation_output",
                "Preparation output is not a completion",
            ));
        };

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
            ClientSelection::Disaggregated { prefill, .. } => prefill,
        };

        let shared_request_id = format!("cmpl_{}", Uuid::now_v7());
        let request_type = &ctx.input.request_type;
        let workers = ctx.state.workers.as_ref();

        let plan = match items.as_slice() {
            [] => {
                return Err(error::internal_error(
                    "preparation_not_completed",
                    "No prompts prepared",
                ))
            }
            [item] => ExecutionPlan::generate(
                self.plan_kind,
                self.build_proto_request(
                    builder_client,
                    shared_request_id,
                    item,
                    &completion_request,
                    request_type,
                    workers,
                )?,
            ),
            batch_items => {
                let mut requests = Vec::with_capacity(batch_items.len());
                for (i, item) in batch_items.iter().enumerate() {
                    requests.push(self.build_proto_request(
                        builder_client,
                        format!("{shared_request_id}-p{i}"),
                        item,
                        &completion_request,
                        request_type,
                        workers,
                    )?);
                }
                ExecutionPlan::Batch {
                    kind: self.plan_kind,
                    shared_request_id,
                    requests,
                }
            }
        };

        ctx.state.execution_plan = Some(plan);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "CompletionRequestBuilding"
    }

    #[cfg(test)]
    fn signature(&self) -> String {
        format!(
            "CompletionRequestBuildingStage(inject_pd_metadata={}, {:?})",
            self.inject_pd_metadata, self.plan_kind
        )
    }
}
