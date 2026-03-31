//! Completion response processing stage (non-streaming)
//!
//! Stage 7 for the `/v1/completions` pipeline, parallel to
//! `MessageResponseProcessingStage` from the Messages rollout.
//! Converts backend `ProtoGenerateComplete` responses into OpenAI
//! `CompletionResponse` format with `CompletionChoice` construction,
//! `echo`/`suffix` handling, and legacy `LogProbs` formatting.
//!
//! Non-streaming only; streaming deferred to follow-up PR.

#![allow(dead_code)] //wired in pipeline factory follow-up PR

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

pub(crate) struct CompletionResponseProcessingStage {
    processor: processor::ResponseProcessor,
}

impl CompletionResponseProcessingStage {
    pub fn new(processor: processor::ResponseProcessor) -> Self {
        Self { processor }
    }
}

#[async_trait]
impl PipelineStage for CompletionResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
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
            .and_then(|p| p.original_text.as_deref())
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
