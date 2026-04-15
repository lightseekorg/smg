//! Completion response processing stage
//!
//! Stage 7 for the `/v1/completions` pipeline
//!
//! - For streaming: spawns background task and returns SSE response (early exit)
//! - For non-streaming: collects the backend response, converts it to
//!   `CompletionResponse`, and stores it as `FinalResponse::Completion`.

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::{
    routers::{
        error,
        grpc::{
            common::stages::PipelineStage,
            context::{FinalResponse, RequestContext},
            regular::{processor, streaming},
        },
    },
    worker::AttachedBody,
};

/// Completion response processing stage
pub(crate) struct CompletionResponseProcessingStage {
    processor: processor::ResponseProcessor,
    streaming_processor: Arc<streaming::StreamingProcessor>,
}

impl CompletionResponseProcessingStage {
    pub fn new(
        processor: processor::ResponseProcessor,
        streaming_processor: Arc<streaming::StreamingProcessor>,
    ) -> Self {
        Self {
            processor,
            streaming_processor,
        }
    }
}

#[async_trait]
impl PipelineStage for CompletionResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let is_streaming = ctx.is_streaming();

        let execution_result = ctx.state.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "CompletionResponseProcessingStage::execute",
                "No execution result"
            );
            error::internal_error("no_execution_result", "No execution result")
        })?;

        let dispatch = ctx
            .state
            .dispatch
            .as_ref()
            .ok_or_else(|| {
                error!(
                    function = "CompletionResponseProcessingStage::execute",
                    "Dispatch metadata not set"
                );
                error::internal_error("dispatch_metadata_not_set", "Dispatch metadata not set")
            })?
            .clone();

        let tokenizer = ctx.tokenizer_arc().ok_or_else(|| {
            error!(
                function = "CompletionResponseProcessingStage::execute",
                "Tokenizer not cached in context"
            );
            error::internal_error(
                "tokenizer_not_cached",
                "Tokenizer not cached in context - preparation stage may have been skipped",
            )
        })?;

        if is_streaming {
            let response = self
                .streaming_processor
                .clone()
                .process_completion_streaming_response(
                    execution_result,
                    ctx.completion_request_arc(),
                    dispatch,
                    tokenizer,
                );

            let response = match ctx.state.load_guards.take() {
                Some(guards) => AttachedBody::wrap_response(response, guards),
                None => response,
            };

            return Ok(Some(response));
        }

        // Non-streaming path
        let completion_request = ctx.completion_request_arc();

        let stop_decoder = ctx.state.response.stop_decoder.as_mut().ok_or_else(|| {
            error!(
                function = "CompletionResponseProcessingStage::execute",
                "Stop decoder not initialized"
            );
            error::internal_error(
                "stop_decoder_not_initialized",
                "Stop decoder not initialized",
            )
        })?;

        let prompt_text = ctx
            .state
            .preparation
            .as_ref()
            .and_then(|p| p.routing_text())
            .unwrap_or("");

        let response = self
            .processor
            .process_non_streaming_completion_response(
                execution_result,
                completion_request,
                dispatch,
                tokenizer,
                stop_decoder,
                prompt_text,
            )
            .await?;

        ctx.state.response.final_response = Some(FinalResponse::Completion(response));

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "CompletionResponseProcessing"
    }
}
