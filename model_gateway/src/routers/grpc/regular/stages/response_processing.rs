//! Response processing stage for the chat + generate pipeline
//!
//! Dispatches to ChatResponseProcessingStage or GenerateResponseProcessingStage
//! based on request type. Only used by new_regular() and new_pd() pipelines.

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use super::{chat::ChatResponseProcessingStage, generate::GenerateResponseProcessingStage};
use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{RequestContext, RequestType},
        regular::{processor, streaming},
    },
};

/// Response processing stage for chat + generate pipelines
pub(crate) struct ChatGenerateResponseProcessingStage {
    chat_stage: ChatResponseProcessingStage,
    generate_stage: GenerateResponseProcessingStage,
}

impl ChatGenerateResponseProcessingStage {
    pub fn new(
        processor: processor::ResponseProcessor,
        streaming_processor: Arc<streaming::StreamingProcessor>,
    ) -> Self {
        Self {
            chat_stage: ChatResponseProcessingStage::new(
                processor.clone(),
                streaming_processor.clone(),
            ),
            generate_stage: GenerateResponseProcessingStage::new(processor, streaming_processor),
        }
    }
}

#[async_trait]
impl PipelineStage for ChatGenerateResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        match &ctx.input.request_type {
            RequestType::Chat(_) => self.chat_stage.execute(ctx).await,
            RequestType::Generate(_) => self.generate_stage.execute(ctx).await,
            request_type => {
                error!(
                    function = "ChatGenerateResponseProcessingStage::execute",
                    request_type = %request_type,
                    "{request_type} should not reach this stage"
                );
                Err(error::internal_error(
                    "wrong_pipeline",
                    format!("{request_type} should use its dedicated pipeline"),
                ))
            }
        }
    }

    fn name(&self) -> &'static str {
        "ChatGenerateResponseProcessing"
    }
}
