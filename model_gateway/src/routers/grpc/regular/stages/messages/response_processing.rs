//! Message response processing stage: non-streaming response processing
//!
//! Collects the backend response, converts it to an Anthropic `Message`,
//! and stores it as FinalResponse::Messages.
//! Streaming support will be added in a follow-up PR.
#![allow(dead_code)] // wired in follow-up PR (pipeline factory)

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{FinalResponse, RequestContext},
        regular::processor,
    },
};

/// Message response processing stage (non-streaming only)
pub(crate) struct MessageResponseProcessingStage {
    processor: processor::ResponseProcessor,
}

impl MessageResponseProcessingStage {
    pub fn new(processor: processor::ResponseProcessor) -> Self {
        Self { processor }
    }
}

#[async_trait]
impl PipelineStage for MessageResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Extract execution result
        let execution_result = ctx.state.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "MessageResponseProcessingStage::execute",
                "No execution result"
            );
            error::internal_error("no_execution_result", "No execution result")
        })?;

        // Get dispatch metadata
        let dispatch = ctx
            .state
            .dispatch
            .as_ref()
            .ok_or_else(|| {
                error!(
                    function = "MessageResponseProcessingStage::execute",
                    "Dispatch metadata not set"
                );
                error::internal_error("dispatch_metadata_not_set", "Dispatch metadata not set")
            })?
            .clone();

        // Get cached tokenizer
        let tokenizer = ctx.tokenizer_arc().ok_or_else(|| {
            error!(
                function = "MessageResponseProcessingStage::execute",
                "Tokenizer not cached in context"
            );
            error::internal_error(
                "tokenizer_not_cached",
                "Tokenizer not cached in context - preparation stage may have been skipped",
            )
        })?;

        let messages_request = ctx.messages_request_arc();

        let stop_decoder = ctx.state.response.stop_decoder.as_mut().ok_or_else(|| {
            error!(
                function = "MessageResponseProcessingStage::execute",
                "Stop decoder not initialized"
            );
            error::internal_error(
                "stop_decoder_not_initialized",
                "Stop decoder not initialized",
            )
        })?;

        // Non-streaming: delegate to ResponseProcessor
        let response = self
            .processor
            .process_non_streaming_messages_response(
                execution_result,
                messages_request,
                dispatch,
                tokenizer,
                stop_decoder,
            )
            .await?;

        // Store the final response
        ctx.state.response.final_response = Some(FinalResponse::Messages(response));

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "MessageResponseProcessing"
    }
}
