//! Preparation stage that delegates to endpoint-specific implementations
//!
//! This stage checks RequestType at runtime and delegates to the appropriate
//! endpoint-specific stage. (ChatPreparationStage, CompletionPreparationStage or GeneratePreparationStage).

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use super::{
    chat::ChatPreparationStage, completion::CompletionPreparationStage,
    generate::GeneratePreparationStage,
};
use crate::routers::{
    error as grpc_error,
    grpc::{
        common::stages::PipelineStage,
        context::{RequestContext, RequestType},
    },
};

/// Preparation stage (delegates to endpoint-specific implementations)
pub(crate) struct PreparationStage {
    chat_stage: ChatPreparationStage,
    generate_stage: GeneratePreparationStage,
    completion_stage: CompletionPreparationStage,
}

impl PreparationStage {
    pub fn new() -> Self {
        Self {
            chat_stage: ChatPreparationStage,
            generate_stage: GeneratePreparationStage,
            completion_stage: CompletionPreparationStage,
        }
    }
}

impl Default for PreparationStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStage for PreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        match &ctx.input.request_type {
            RequestType::Chat(_) => self.chat_stage.execute(ctx).await,
            RequestType::Generate(_) => self.generate_stage.execute(ctx).await,
            RequestType::Completion(_) => self.completion_stage.execute(ctx).await,
            request_type => {
                error!(
                    function = "PreparationStage::execute",
                    request_type = %request_type,
                    "{request_type} request type reached regular preparation stage"
                );
                Err(grpc_error::internal_error(
                    "wrong_pipeline",
                    format!("{request_type} should use its dedicated pipeline"),
                ))
            }
        }
    }

    fn name(&self) -> &'static str {
        "Preparation"
    }
}
