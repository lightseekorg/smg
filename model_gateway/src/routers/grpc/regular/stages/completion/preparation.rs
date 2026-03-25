//! Completion preparation stage: resolve prompt, tokenize, create stop decoder.
//!
//! This is the `/v1/completions` Stage 1 equivalent. It intentionally builds on top of
//! the native completion pipeline typing introduced in PR #840. It keeps
//! `CompletionRequest` native in the request context instead of laundering it
//! through `GenerateRequest`.

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::common::StringOrArray;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{PreparationOutput, RequestContext},
        utils,
    },
};

pub(crate) struct CompletionPreparationStage;

#[async_trait]
impl PipelineStage for CompletionPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let request = ctx.completion_request_arc();

        let tokenizer =
            utils::resolve_tokenizer(ctx, "CompletionPreparationStage::execute").map_err(|e| *e)?;

        let prompt_text = match &request.prompt {
            StringOrArray::String(text) => text.clone(),
            StringOrArray::Array(_) => {
                return Err(error::bad_request(
                    "batch_prompts_not_supported",
                    "Batched prompt arrays are not supported for gRPC /v1/completions yet",
                ));
            }
        };

        let encoding = tokenizer.encode(&prompt_text, false).map_err(|e| {
            error!(
                function = "CompletionPreparationStage::execute",
                error = %e,
                "Tokenization failed"
            );
            error::bad_request("tokenization_failed", format!("Tokenization failed: {e}"))
        })?;

        let stop_decoder = utils::create_stop_decoder(
            &tokenizer,
            request.stop.as_ref(),
            request.stop_token_ids.as_ref(),
            request.skip_special_tokens,
            request.no_stop_trim,
        );

        ctx.state.preparation = Some(PreparationOutput {
            original_text: Some(prompt_text),
            token_ids: encoding.token_ids().to_vec(),
            processed_messages: None,
            tool_constraints: None,
            filtered_request: None,
            harmony_mode: false,
            selection_text: None,
            harmony_messages: None,
            harmony_stop_ids: None,
        });

        ctx.state.response.stop_decoder = Some(stop_decoder);

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "CompletionPreparation"
    }
}
