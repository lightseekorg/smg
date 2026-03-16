//! Completion response processing stage
//!
//! Non-streaming: collects generate results via the shared processor and wraps
//! them as `CompletionResponse`. Streaming: delegates to the completion-aware
//! streaming processor, which emits OpenAI `CompletionStreamResponse` chunks
//! directly from typed proto responses.

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::{
    core::AttachedBody,
    routers::{
        error,
        grpc::{
            common::stages::PipelineStage,
            context::{FinalResponse, RequestContext},
            regular::{processor, streaming},
        },
    },
};

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
        self.process_completion_response(ctx).await
    }

    fn name(&self) -> &'static str {
        "CompletionResponseProcessing"
    }
}

impl CompletionResponseProcessingStage {
    async fn process_completion_response(
        &self,
        ctx: &mut RequestContext,
    ) -> Result<Option<Response>, Response> {
        let is_streaming = ctx.is_streaming();
        let completion_req = ctx.completion_request_arc();

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

        let prompt_text = ctx
            .state
            .preparation
            .as_ref()
            .and_then(|p| p.original_text.clone())
            .unwrap_or_default();

        if is_streaming {
            let response = self
                .streaming_processor
                .clone()
                .process_streaming_completion(
                    execution_result,
                    completion_req.clone(),
                    dispatch,
                    tokenizer,
                    prompt_text,
                );

            let response = match ctx.state.load_guards.take() {
                Some(guards) => AttachedBody::wrap_response(response, guards),
                None => response,
            };
            return Ok(Some(response));
        }

        // Non-streaming
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

        let completion_response = self
            .processor
            .process_non_streaming_completion_response(
                execution_result,
                completion_req,
                dispatch,
                tokenizer,
                stop_decoder,
                &prompt_text,
            )
            .await?;

        ctx.state.response.final_response = Some(FinalResponse::Completion(completion_response));
        Ok(None)
    }
}
