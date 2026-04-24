//! Chat preparation stage: Filter tools, process messages, tokenize, build constraints

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{ToolChoice, ToolChoiceValue},
};
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

        // Resolve multimodal context once: placeholder token, model_id, tokenizer_source.
        // The placeholder is passed to process_chat_messages so that string-format chat
        // templates insert it per image instead of stripping image parts.  The remaining
        // fields are reused by process_multimodal to avoid duplicate lookups.
        let is_multimodal = multimodal::has_multimodal_content(&request.messages);
        let (image_placeholder, mm_context) = if is_multimodal {
            if let Some(mm_components) = ctx.components.multimodal.as_ref() {
                let model_id = ctx.input.model_id.as_str();
                let entry = ctx
                    .components
                    .tokenizer_registry
                    .get_by_name(model_id)
                    .or_else(|| ctx.components.tokenizer_registry.get_by_id(model_id));

                let (tokenizer_id, tokenizer_source) = match entry {
                    Some(e) => (e.id.clone(), e.source.clone()),
                    None => {
                        error!(
                            function = "ChatPreparationStage::execute",
                            model = %model_id,
                            "Tokenizer entry not found for multimodal processing"
                        );
                        return Err(error::bad_request(
                            "multimodal_config_missing",
                            format!("Tokenizer not found for model: {model_id}"),
                        ));
                    }
                };

                let placeholder = multimodal::resolve_placeholder_token(
                    model_id,
                    &*tokenizer,
                    mm_components,
                    &tokenizer_id,
                    &tokenizer_source,
                )
                .await
                .map_err(|e| {
                    error!(
                        function = "ChatPreparationStage::execute",
                        model = %model_id,
                        error = %e,
                        "Failed to resolve multimodal placeholder token"
                    );
                    error::internal_error(
                        "multimodal_placeholder_resolution_failed",
                        format!("Failed to resolve multimodal placeholder token: {e}"),
                    )
                })?;

                (
                    placeholder,
                    Some((mm_components, model_id, tokenizer_id, tokenizer_source)),
                )
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
        } else {
            (None, None)
        };

        // Step 2: Process messages and apply chat template
        let processed_messages = match utils::process_chat_messages(
            &body_ref,
            &*tokenizer,
            image_placeholder.as_deref(),
        ) {
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

        // Step 4: Full multimodal processing (fetch + preprocess + expand tokens + hash)
        let mut multimodal_intermediate = None;
        if let Some((mm_components, model_id, tokenizer_id, tokenizer_source)) = mm_context {
            match multimodal::process_multimodal(
                &request.messages,
                model_id,
                &*tokenizer,
                token_ids,
                mm_components,
                &tokenizer_id,
                &tokenizer_source,
            )
            .await
            {
                Ok(output) => {
                    debug!(
                        function = "ChatPreparationStage::execute",
                        expanded_tokens = output.expanded_token_ids.len(),
                        "Multimodal processing complete"
                    );
                    token_ids = output.expanded_token_ids;
                    multimodal_intermediate = Some(output.intermediate);
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
        }

        // Step 4: Build tool constraints if needed
        // The tool parser registry handles both structural tag (for native format
        // parsers like Mistral, KimiK2) and generic JSON schema fallback.
        let tool_call_constraint = if let (Some(tools), Some(tool_choice)) =
            (body_ref.tools.as_ref(), request.tool_choice.as_ref())
        {
            ctx.components
                .tool_parser_factory
                .registry()
                .generate_tool_constraint(
                    ctx.components.configured_tool_parser.as_deref(),
                    tools,
                    tool_choice,
                )
                .map_err(|e| {
                    error!(function = "ChatPreparationStage::execute", error = %e, "Invalid tool configuration");
                    error::bad_request(
                        "invalid_tool_configuration",
                        format!("Invalid tool configuration: {e}"),
                    )
                })?
        } else {
            None
        };

        // Derive skip_special_tokens from constraint type:
        // - json_schema: backend forces JSON, no trigger tokens to preserve
        // - structural_tag or no constraint (auto): parser needs trigger tokens
        let skip_special_tokens = match &tool_call_constraint {
            Some(c) if c.is_json_schema() => request.skip_special_tokens,
            _ if request.tools.is_some()
                && !matches!(
                    request.tool_choice,
                    Some(ToolChoice::Value(ToolChoiceValue::None))
                ) =>
            {
                false
            }
            _ => request.skip_special_tokens,
        };

        // Step 5: Create stop sequence decoder (build once, reuse in non-stream)
        let stop_decoder = utils::create_stop_decoder(
            &tokenizer,
            request.stop.as_ref(),
            request.stop_token_ids.as_ref(),
            skip_special_tokens,
            request.no_stop_trim,
            request.ignore_eos,
        );

        let mut processed_messages = processed_messages;
        processed_messages.multimodal_intermediate = multimodal_intermediate;

        // Store results in context
        ctx.state.preparation = Some(PreparationOutput::Chat {
            token_ids,
            processed_messages,
            tool_constraints: tool_call_constraint.map(|c| c.to_tuple()),
        });

        // Store stop decoder and derived skip_special_tokens for response processing.
        // Stored on ResponseState because PreparationOutput is consumed by
        // request_building before response_processing runs.
        ctx.state.response.stop_decoder = Some(stop_decoder);
        ctx.state.response.skip_special_tokens = Some(skip_special_tokens);

        Ok(())
    }
}
