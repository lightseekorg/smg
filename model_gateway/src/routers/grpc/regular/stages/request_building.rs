//! Request building stage for chat and generate endpoints

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use super::{chat::ChatRequestBuildingStage, generate::GenerateRequestBuildingStage};
use crate::routers::{
    error as grpc_error,
    grpc::{
        common::stages::PipelineStage,
        context::{RequestContext, RequestType},
    },
};

/// Request building stage for chat and generate pipelines
///
/// These two request types share a single pipeline instance (`new_regular` /
/// `new_pd`) and are dispatched here. All other request types have
/// dedicated pipelines and wire their own request building stages directly.
pub(crate) struct ChatGenerateRequestBuildingStage {
    chat_stage: ChatRequestBuildingStage,
    generate_stage: GenerateRequestBuildingStage,
}

impl ChatGenerateRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool) -> Self {
        Self {
            chat_stage: ChatRequestBuildingStage::new(inject_pd_metadata),
            generate_stage: GenerateRequestBuildingStage::new(inject_pd_metadata),
        }
    }
}

#[async_trait]
impl PipelineStage for ChatGenerateRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        match &ctx.input.request_type {
            RequestType::Chat(_) => self.chat_stage.execute(ctx).await,
            RequestType::Generate(_) => self.generate_stage.execute(ctx).await,
            request_type => {
                error!(
                    function = "ChatGenerateRequestBuildingStage::execute",
                    request_type = %request_type,
                    "{request_type} should not reach this stage"
                );
                Err(grpc_error::internal_error(
                    "wrong_pipeline",
                    format!("{request_type} should use its dedicated pipeline"),
                ))
            }
        }
    }

    fn name(&self) -> &'static str {
        "ChatGenerateRequestBuilding"
    }
}
