//! Chat preparation stage: Filter tools, process messages, tokenize, build constraints

use std::borrow::Cow;

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::chat::ChatCompletionRequest;
use tracing::{debug, error};

use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{PreparationOutput, RequestContext},
        multimodal, utils,
    },
};

/// Chat preparation stage
///
/// Extracts chat-specific preparation logic from the old unified PreparationStage.
/// This is a direct extraction without architectural changes.
pub(crate) struct ChatPreparationStage;

#[async_trait]
impl PipelineStage for ChatPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let request = ctx.chat_request_arc();
        self.prepare_chat(ctx, &request).await?;
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ChatPreparation"
    }
}

impl ChatPreparationStage {
    async fn prepare_chat(
        &self,
        ctx: &mut RequestContext,
        request: &ChatCompletionRequest,
    ) -> Result<(), Response> {
        // Step 0: Resolve tokenizer from registry (cached for reuse in response processing)
        let tokenizer =
            utils::resolve_tokenizer(ctx, "ChatPreparationStage::prepare_chat").map_err(|e| *e)?;

        // Step 1: Filter tools if needed
        let body_ref = utils::filter_chat_request_by_tool_choice(request);

        // Step 2: Process messages and apply chat template
        let processed_messages = match utils::process_chat_messages(&body_ref, &*tokenizer) {
            Ok(msgs) => msgs,
            Err(e) => {
                error!(function = "ChatPreparationStage::execute", error = %e, "Failed to process chat messages");
                return Err(error::bad_request("process_messages_failed", e));
            }
        };

        // Step 3: Tokenize the processed text (no special tokens - chat template already handles them)
        let encoding = match tokenizer.encode(&processed_messages.text, false) {
            Ok(encoding) => encoding,
            Err(e) => {
                error!(function = "ChatPreparationStage::execute", error = %e, "Tokenization failed");
                return Err(error::internal_error(
                    "tokenization_failed",
                    format!("Tokenization failed: {e}"),
                ));
            }
        };

        let mut token_ids = encoding.token_ids().to_vec();
        let mut original_token_ids: Option<Vec<u32>> = None;

        // Step 3.5: Multimodal processing (if images detected in messages)
        let mut multimodal_inputs = None;
        if multimodal::has_multimodal_content(&request.messages) {
            if let Some(mm_components) = ctx.components.multimodal.as_ref() {
                let model_id = ctx.input.model_id.as_deref().unwrap_or(&request.model);

                // Resolve tokenizer source path for loading config files
                let tokenizer_source = ctx
                    .components
                    .tokenizer_registry
                    .get_by_name(model_id)
                    .map(|entry| entry.source)
                    .unwrap_or_default();

                if tokenizer_source.is_empty() {
                    error!(
                        function = "ChatPreparationStage::execute",
                        model = %model_id,
                        "Tokenizer source path not found for multimodal processing"
                    );
                    return Err(error::bad_request(
                        "multimodal_config_missing",
                        format!("Tokenizer source path not found for model: {model_id}"),
                    ));
                }

                match multimodal::process_multimodal(
                    &request.messages,
                    model_id,
                    &*tokenizer,
                    &token_ids,
                    mm_components,
                    &tokenizer_source,
                )
                .await
                {
                    Ok(mm_output) => {
                        debug!(
                            function = "ChatPreparationStage::execute",
                            model = %model_id,
                            original_tokens = token_ids.len(),
                            expanded_tokens = mm_output.expanded_token_ids.len(),
                            "Multimodal processing complete"
                        );
                        original_token_ids = Some(token_ids.clone());
                        token_ids = mm_output.expanded_token_ids;
                        multimodal_inputs = Some(mm_output.multimodal_data);
                    }
                    Err(e) => {
                        error!(
                            function = "ChatPreparationStage::execute",
                            error = %e,
                            "Multimodal processing failed"
                        );
                        return Err(error::bad_request(
                            "multimodal_processing_failed",
                            format!("Multimodal processing failed: {e}"),
                        ));
                    }
                }
            } else {
                error!(
                    function = "ChatPreparationStage::execute",
                    "Multimodal content detected but multimodal components not initialized"
                );
                return Err(error::bad_request(
                    "multimodal_not_supported",
                    "Multimodal content detected but multimodal processing is not available",
                ));
            }
        }

        // Step 4: Build tool constraints if needed
        let tool_call_constraint = if let Some(tools) = body_ref.tools.as_ref() {
            utils::generate_tool_constraints(tools, request.tool_choice.as_ref(), &request.model)
                .map_err(|e| {
                    error!(function = "ChatPreparationStage::execute", error = %e, "Invalid tool configuration");
                    error::bad_request("invalid_tool_configuration", format!("Invalid tool configuration: {e}"))
                })?
        } else {
            None
        };

        // Step 5: Create stop sequence decoder (build once, reuse in non-stream)
        let stop_decoder = utils::create_stop_decoder(
            &tokenizer,
            request.stop.as_ref(),
            request.stop_token_ids.as_ref(),
            request.skip_special_tokens,
            request.no_stop_trim,
        );

        // Update multimodal inputs on processed messages if we have them
        let mut processed_messages = processed_messages;
        processed_messages.multimodal_inputs = multimodal_inputs;

        // Store results in context
        ctx.state.preparation = Some(PreparationOutput {
            original_text: Some(processed_messages.text.clone()),
            token_ids,
            original_token_ids,
            processed_messages: Some(processed_messages),
            tool_constraints: tool_call_constraint,
            filtered_request: if matches!(body_ref, Cow::Owned(_)) {
                Some(body_ref.into_owned())
            } else {
                None
            },
            // Harmony fields (not used for regular preparation)
            harmony_mode: false,
            selection_text: None,
            harmony_messages: None,
            harmony_stop_ids: None,
        });

        // Store stop decoder for reuse in response processing
        ctx.state.response.stop_decoder = Some(stop_decoder);

        Ok(())
    }
}
