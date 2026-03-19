//! Message API preparation stage: Convert tools, process messages, tokenize, build constraints

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::{common::StringOrArray, messages::CreateMessageRequest};
use tracing::{debug, error};

use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{PreparationOutput, RequestContext},
        multimodal,
        utils::{self, message_utils},
    },
};

/// Message API preparation stage
///
/// Parallel to `ChatPreparationStage` but works with `CreateMessageRequest`.
/// Converts Anthropic Messages API types into the internal chat template format,
/// tokenizes, and builds tool constraints.
pub(crate) struct MessagePreparationStage;

#[async_trait]
impl PipelineStage for MessagePreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let request = ctx.messages_request_arc();
        self.prepare_messages(ctx, &request).await?;
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "MessagePreparation"
    }
}

impl MessagePreparationStage {
    async fn prepare_messages(
        &self,
        ctx: &mut RequestContext,
        request: &CreateMessageRequest,
    ) -> Result<(), Response> {
        // Step 0: Resolve tokenizer from registry (cached for reuse in response processing)
        let tokenizer = utils::resolve_tokenizer(ctx, "MessagePreparationStage::prepare_messages")
            .map_err(|e| *e)?;

        // Step 1: Convert Messages API tools to chat tools and filter by tool_choice
        let chat_tools = request
            .tools
            .as_deref()
            .map(message_utils::extract_chat_tools);

        let chat_tool_choice = request
            .tool_choice
            .as_ref()
            .map(message_utils::convert_message_tool_choice);

        // Filter tools by tool_choice (reuse chat utility)
        let filtered_tools = match (&chat_tools, &chat_tool_choice) {
            (Some(tools), Some(tc)) => {
                utils::filter_tools_by_tool_choice(tools, Some(tc)).unwrap_or_else(|| tools.clone())
            }
            (Some(tools), None) => tools.clone(),
            _ => Vec::new(),
        };

        let tools_for_template = if filtered_tools.is_empty() {
            None
        } else {
            Some(filtered_tools.as_slice())
        };

        // Step 2: Process messages and apply chat template
        let processed_messages = match message_utils::process_messages(
            request,
            &*tokenizer,
            tools_for_template,
        ) {
            Ok(msgs) => msgs,
            Err(e) => {
                error!(function = "MessagePreparationStage::execute", error = %e, "Failed to process messages");
                return Err(error::bad_request("process_messages_failed", e));
            }
        };

        // Step 3: Tokenize the processed text
        let encoding = match tokenizer.encode(&processed_messages.text, false) {
            Ok(encoding) => encoding,
            Err(e) => {
                error!(function = "MessagePreparationStage::execute", error = %e, "Tokenization failed");
                return Err(error::internal_error(
                    "tokenization_failed",
                    format!("Tokenization failed: {e}"),
                ));
            }
        };

        let mut token_ids = encoding.token_ids().to_vec();

        // Step 3.5: Multimodal processing (fetch + preprocess + expand tokens + hash)
        let mut multimodal_intermediate = None;
        if multimodal::has_multimodal_content_messages(&request.messages) {
            if let Some(mm_components) = ctx.components.multimodal.as_ref() {
                let model_id = ctx.input.model_id.as_deref().unwrap_or(&request.model);
                let tokenizer_source = ctx
                    .components
                    .tokenizer_registry
                    .get_by_name(model_id)
                    .or_else(|| ctx.components.tokenizer_registry.get_by_id(model_id))
                    .map(|e| e.source)
                    .unwrap_or_default();

                if tokenizer_source.is_empty() {
                    error!(
                        function = "MessagePreparationStage::execute",
                        model = %model_id,
                        "Tokenizer source path not found for multimodal processing"
                    );
                    return Err(error::bad_request(
                        "multimodal_config_missing",
                        format!("Tokenizer source path not found for model: {model_id}"),
                    ));
                }

                match multimodal::process_multimodal_messages(
                    &request.messages,
                    model_id,
                    &*tokenizer,
                    token_ids,
                    mm_components,
                    &tokenizer_source,
                )
                .await
                {
                    Ok(output) => {
                        debug!(
                            function = "MessagePreparationStage::execute",
                            expanded_tokens = output.expanded_token_ids.len(),
                            "Multimodal processing complete"
                        );
                        token_ids = output.expanded_token_ids;
                        multimodal_intermediate = Some(output.intermediate);
                    }
                    Err(e) => {
                        error!(
                            function = "MessagePreparationStage::execute",
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
                    function = "MessagePreparationStage::execute",
                    "Multimodal content detected but multimodal components not initialized"
                );
                return Err(error::bad_request(
                    "multimodal_not_supported",
                    "Multimodal content detected but multimodal processing is not available",
                ));
            }
        }

        // Step 4: Build tool constraints if tools present
        let tool_call_constraint = if filtered_tools.is_empty() {
            None
        } else {
            utils::generate_tool_constraints(
                &filtered_tools,
                chat_tool_choice.as_ref(),
                &request.model,
            )
            .map_err(|e| {
                error!(function = "MessagePreparationStage::execute", error = %e, "Invalid tool configuration");
                error::bad_request(
                    "invalid_tool_configuration",
                    format!("Invalid tool configuration: {e}"),
                )
            })?
        };

        // Step 5: Create stop sequence decoder
        let stop_for_decoder = request
            .stop_sequences
            .as_ref()
            .map(|seqs| StringOrArray::Array(seqs.clone()));

        let stop_decoder = utils::create_stop_decoder(
            &tokenizer,
            stop_for_decoder.as_ref(),
            None,  // no stop_token_ids in Messages API
            true,  // always skip special tokens — Messages API never exposes raw tokens
            false, // no_stop_trim default
        );

        let mut processed_messages = processed_messages;
        processed_messages.multimodal_intermediate = multimodal_intermediate;

        // Store results in context
        ctx.state.preparation = Some(PreparationOutput {
            original_text: Some(processed_messages.text.clone()),
            token_ids,
            processed_messages: Some(processed_messages),
            tool_constraints: tool_call_constraint,
            filtered_request: None, // Messages doesn't use Cow<ChatCompletionRequest> pattern
            // Harmony fields (not used for messages)
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
