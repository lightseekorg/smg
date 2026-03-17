//! Harmony Preparation Stage: Harmony encoding for chat and generate requests

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::{
    chat::ChatCompletionRequest,
    responses::{ResponsesRequest, TextFormat},
};
use serde_json::json;
use tracing::error;

use super::super::HarmonyBuilder;
use crate::routers::{
    error,
    grpc::{
        common::{responses::utils::extract_tools_from_response_tools, stages::PipelineStage},
        context::{PreparationOutput, RequestContext, RequestType},
        utils,
    },
};

/// Harmony Preparation stage: Encode requests using Harmony protocol
///
/// Replaces the regular PreparationStage for Harmony models.
/// Converts chat/generate requests to Harmony-encoded token_ids and extraction_text.
pub(crate) struct HarmonyPreparationStage {
    builder: HarmonyBuilder,
}

impl HarmonyPreparationStage {
    /// Create a new Harmony preparation stage
    pub fn new() -> Self {
        Self {
            builder: HarmonyBuilder::new(),
        }
    }
}

impl Default for HarmonyPreparationStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStage for HarmonyPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Clone Arc before match to avoid borrow checker issues
        // Arc clone is cheap (8 bytes) - avoids full request clone (15KB-200KB)
        let is_chat = matches!(&ctx.input.request_type, RequestType::Chat(_));
        let is_responses = matches!(&ctx.input.request_type, RequestType::Responses(_));

        if is_chat {
            let request_arc = ctx.chat_request_arc();
            // Reject ignore_eos for Harmony models: Harmony requires EOS-based stop tokens
            // to produce well-formed output. When ignore_eos is true, some backends skip all
            // stop token checks, causing the Harmony parser to receive malformed token sequences.
            if request_arc.ignore_eos {
                return Err(error::bad_request(
                    "ignore_eos_not_supported",
                    "ignore_eos is not supported for Harmony models",
                ));
            }
            self.prepare_chat(ctx, &request_arc)?;
        } else if is_responses {
            let request_arc = ctx.responses_request_arc();
            self.prepare_responses(ctx, &request_arc)?;
        } else {
            error!(
                function = "HarmonyPreparationStage::execute",
                "Unsupported request type for Harmony pipeline"
            );
            return Err(error::bad_request(
                "harmony_request_type_invalid",
                "Only Chat and Responses requests supported in Harmony pipeline".to_string(),
            ));
        }

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "HarmonyPreparation"
    }
}

impl HarmonyPreparationStage {
    /// Prepare a chat completion request using Harmony encoding
    #[expect(
        clippy::result_large_err,
        reason = "Response is the standard error type in the pipeline stage pattern"
    )]
    fn prepare_chat(
        &self,
        ctx: &mut RequestContext,
        request: &ChatCompletionRequest,
    ) -> Result<Option<Response>, Response> {
        // Step 1: Filter tools if needed
        let body_ref = utils::filter_chat_request_by_tool_choice(request);

        // Step 2: Build tool constraints
        let tool_constraints = if let Some(tools) = body_ref.tools.as_ref() {
            utils::generate_tool_constraints(tools, body_ref.tool_choice.as_ref(), &body_ref.model)
                .map_err(|e| {
                    error!(function = "prepare_chat", error = %e, "Failed to generate tool constraints");
                    error::bad_request("tool_constraint_failed", e)
                })?
        } else {
            None
        };

        // Step 3: Build via Harmony
        let build_output = self.builder.build_from_chat(&body_ref).map_err(|e| {
            error!(
                function = "prepare_chat",
                error = %e,
                "Harmony build failed for chat request"
            );
            error::bad_request("harmony_build_failed", format!("Harmony build failed: {e}"))
        })?;

        // Step 4: Store results
        let bypass_harmony_parser = tool_constraints.is_some();
        ctx.state.preparation = Some(PreparationOutput {
            original_text: None,
            token_ids: build_output.input_ids,
            processed_messages: None,
            tool_constraints,
            filtered_request: if matches!(body_ref, std::borrow::Cow::Owned(_)) {
                Some(body_ref.into_owned())
            } else {
                None
            },
            harmony_mode: true,
            selection_text: Some(build_output.selection_text),
            harmony_messages: Some(build_output.harmony_messages),
            harmony_stop_ids: Some(build_output.stop_token_ids),
            bypass_harmony_parser,
        });

        Ok(None)
    }

    /// Prepare a responses API request using Harmony encoding
    ///
    /// For responses API, we build from conversation history using the same Harmony
    /// encoding that the builder provides. This handles the MCP loop integration.
    #[expect(
        clippy::result_large_err,
        reason = "Response is the standard error type in the pipeline stage pattern"
    )]
    pub fn prepare_responses(
        &self,
        ctx: &mut RequestContext,
        request: &ResponsesRequest,
    ) -> Result<Option<Response>, Response> {
        // Step 1: Extract function tools with schemas from ResponseTools
        let mut function_tools = extract_tools_from_response_tools(request.tools.as_deref());

        // Step 2: Filter tools based on tool_choice (AllowedTools or Function)
        // Note: Tool existence is already validated in ResponsesRequest::validate()
        if let Some(filtered) =
            utils::filter_tools_by_tool_choice(&function_tools, request.tool_choice.as_ref())
        {
            function_tools = filtered;
        }

        // Step 3: Generate tool constraints
        let tool_constraint = if function_tools.is_empty() {
            None
        } else {
            utils::generate_tool_constraints(&function_tools, request.tool_choice.as_ref(), &request.model)
                .map_err(|e| {
                    error!(function = "prepare_responses", error = %e, "Failed to generate tool constraints");
                    error::bad_request("tool_constraint_failed", e)
                })?
        };

        let text_constraint = if let Some(text_config) = &request.text {
            Self::generate_text_format_json_schema(text_config).map_err(|e| *e)?
        } else {
            None
        };

        if tool_constraint.is_some() && text_constraint.is_some() {
            error!(
                function = "prepare_responses",
                "Conflicting constraints: both tool_choice and text format specified"
            );
            return Err(error::bad_request(
                "conflicting_constraints",
                "Cannot use both tool_choice (required/function) and text format (json_object/json_schema) simultaneously".to_string(),
            ));
        }

        let constraint = tool_constraint.or(text_constraint);

        // Step 3: Build via Harmony from responses API request
        let build_output = self.builder.build_from_responses(request).map_err(|e| {
            error!(
                function = "prepare_responses",
                error = %e,
                "Harmony build failed for responses request"
            );
            error::bad_request("harmony_build_failed", format!("Harmony build failed: {e}"))
        })?;

        // Step 4: Store results with constraint
        let bypass_harmony_parser = constraint.is_some();
        ctx.state.preparation = Some(PreparationOutput {
            original_text: None,
            token_ids: build_output.input_ids,
            processed_messages: None,
            tool_constraints: constraint,
            filtered_request: None,
            harmony_mode: true,
            selection_text: Some(build_output.selection_text),
            harmony_messages: Some(build_output.harmony_messages),
            harmony_stop_ids: Some(build_output.stop_token_ids),
            bypass_harmony_parser,
        });

        Ok(None)
    }

    /// Generate json_schema constraint for structured output (text field)
    ///
    /// Converts text.format to a json_schema constraint tuple.
    /// Returns None if text.format is not specified or is "text".
    fn generate_text_format_json_schema(
        text_config: &openai_protocol::responses::TextConfig,
    ) -> Result<Option<(String, String)>, Box<Response>> {
        let Some(format) = &text_config.format else {
            return Ok(None);
        };

        match format {
            TextFormat::Text => Ok(None),
            TextFormat::JsonObject => {
                let schema = serde_json::to_string(&json!({"type": "object"})).map_err(|e| {
                    Box::new(error::internal_error(
                        "json_schema_serialize_failed",
                        e.to_string(),
                    ))
                })?;
                Ok(Some(("json_schema".to_string(), schema)))
            }
            TextFormat::JsonSchema { schema, .. } => {
                let schema_str = serde_json::to_string(schema).map_err(|e| {
                    Box::new(error::internal_error(
                        "json_schema_serialize_failed",
                        e.to_string(),
                    ))
                })?;
                Ok(Some(("json_schema".to_string(), schema_str)))
            }
        }
    }
}
