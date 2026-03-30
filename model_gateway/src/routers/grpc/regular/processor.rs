//! Shared response processing logic for gRPC routers
//!
//! This module contains response processing functions that are shared between
//! the regular router and PD router.

use std::{sync::Arc, time::Instant};

use llm_tokenizer::{
    stop::{SequenceDecoderOutput, StopSequenceDecoder},
    traits::Tokenizer,
};
use openai_protocol::{
    chat::{ChatChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse},
    common::{FunctionCallResponse, Tool as OpenAITool, ToolCall, ToolChoice, ToolChoiceValue},
    generate::{GenerateMetaInfo, GenerateRequest, GenerateResponse},
};
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use serde_json::Value;
use tool_parser::ParserFactory as ToolParserFactory;
use tracing::{error, warn};

use crate::routers::{
    error,
    grpc::{
        common::{response_collection, response_formatting},
        context::{DispatchMetadata, ExecutionResult},
        proto_wrapper::ProtoGenerateComplete,
        utils,
    },
};

/// Unified response processor for both routers
#[derive(Clone)]
pub(crate) struct ResponseProcessor {
    pub tool_parser_factory: ToolParserFactory,
    pub reasoning_parser_factory: ReasoningParserFactory,
    pub configured_tool_parser: Option<String>,
    pub configured_reasoning_parser: Option<String>,
}

impl ResponseProcessor {
    pub fn new(
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        Self {
            tool_parser_factory,
            reasoning_parser_factory,
            configured_tool_parser,
            configured_reasoning_parser,
        }
    }

    /// Process a single choice from GenerateComplete response
    #[expect(clippy::too_many_arguments)]
    pub async fn process_single_choice(
        &self,
        complete: &ProtoGenerateComplete,
        index: usize,
        original_request: &ChatCompletionRequest,
        tokenizer: &Arc<dyn Tokenizer>,
        stop_decoder: &mut StopSequenceDecoder,
        history_tool_calls_count: usize,
        reasoning_parser_available: bool,
        tool_parser_available: bool,
    ) -> Result<ChatChoice, String> {
        stop_decoder.reset();
        // Decode tokens
        let outputs = stop_decoder
            .process_tokens(complete.output_ids())
            .map_err(|e| format!("Failed to process tokens: {e}"))?;

        // Accumulate text with early breaks
        let mut final_text = String::new();
        for output in outputs {
            match output {
                SequenceDecoderOutput::Text(t) => final_text.push_str(&t),
                SequenceDecoderOutput::StoppedWithText(t) => {
                    final_text.push_str(&t);
                    break;
                }
                SequenceDecoderOutput::Stopped => break,
                SequenceDecoderOutput::Held => {}
            }
        }

        // Flush remaining text
        if let SequenceDecoderOutput::Text(t) = stop_decoder.flush() {
            final_text.push_str(&t);
        }

        // Step 1: Handle reasoning content parsing
        let mut reasoning_text: Option<String> = None;
        let mut processed_text = final_text;

        if original_request.separate_reasoning && reasoning_parser_available {
            let pooled_parser = utils::get_reasoning_parser(
                &self.reasoning_parser_factory,
                self.configured_reasoning_parser.as_deref(),
                &original_request.model,
            );

            let mut parser = pooled_parser.lock().await;
            match parser.detect_and_parse_reasoning(&processed_text) {
                Ok(result) => {
                    if !result.reasoning_text.is_empty() {
                        reasoning_text = Some(result.reasoning_text);
                    }
                    processed_text = result.normal_text;
                }
                Err(e) => {
                    warn!("Reasoning parsing error, skipping parsing: {e}");
                }
            }
        }

        // Step 2: Handle tool call parsing
        let mut tool_calls: Option<Vec<ToolCall>> = None;
        let tool_choice_enabled = !matches!(
            &original_request.tool_choice,
            Some(ToolChoice::Value(ToolChoiceValue::None))
        );

        if tool_choice_enabled && original_request.tools.is_some() {
            // Check if JSON schema constraint was used (specific function or required mode)
            let used_json_schema = match &original_request.tool_choice {
                Some(ToolChoice::Function { .. }) => true,
                Some(ToolChoice::Value(ToolChoiceValue::Required)) => true,
                Some(ToolChoice::AllowedTools { mode, .. }) => mode == "required",
                _ => false,
            };

            if used_json_schema {
                (tool_calls, processed_text) = utils::parse_json_schema_response(
                    &processed_text,
                    original_request.tool_choice.as_ref(),
                    &original_request.model,
                    history_tool_calls_count,
                );
            } else if tool_parser_available {
                (tool_calls, processed_text) = self
                    .parse_tool_calls(
                        &processed_text,
                        &original_request.model,
                        original_request.tools.as_deref(),
                        history_tool_calls_count,
                    )
                    .await;
            }
        }

        if tool_calls.is_none() {
            match utils::deterministic_auto_tool_repair(original_request).await {
                Ok(Some(repaired_tool_calls)) => {
                    tool_calls = Some(repaired_tool_calls);
                }
                Ok(None) => {}
                Err(e) => {
                    warn!("Deterministic auto tool repair failed: {e}");
                }
            }
        }

        utils::repair_tool_calls_and_content(original_request, &mut tool_calls, &mut processed_text);

        // Step 3: Use finish reason directly from proto (already OpenAI-compatible string)
        let finish_reason_str = complete.finish_reason();

        // Override finish reason if we have tool calls
        let final_finish_reason_str = if tool_calls.is_some() {
            "tool_calls"
        } else {
            finish_reason_str
        };

        let matched_stop = complete.matched_stop_json();

        // Step 4: Convert output logprobs if present
        let logprobs = complete.output_logprobs().map(|ref proto_logprobs| {
            utils::convert_proto_to_openai_logprobs(proto_logprobs, tokenizer)
        });

        // Step 5: Build ChatCompletionMessage (proper response message type)
        let chat_message = ChatCompletionMessage {
            role: "assistant".to_string(),
            content: if processed_text.is_empty() {
                None
            } else {
                Some(processed_text)
            },
            tool_calls,
            reasoning_content: reasoning_text,
        };

        // Step 6: Build ChatChoice
        Ok(ChatChoice {
            index: index as u32,
            message: chat_message,
            logprobs,
            finish_reason: Some(final_finish_reason_str.to_string()),
            matched_stop,
            hidden_states: None,
        })
    }

    /// Process non-streaming chat response (collects all responses and builds final response)
    pub async fn process_non_streaming_chat_response(
        &self,
        execution_result: ExecutionResult,
        chat_request: Arc<ChatCompletionRequest>,
        dispatch: DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_decoder: &mut StopSequenceDecoder,
        request_logprobs: bool,
    ) -> Result<ChatCompletionResponse, axum::response::Response> {
        // Collect all responses from the execution result
        let all_responses =
            response_collection::collect_responses(execution_result, request_logprobs).await?;

        let history_tool_calls_count = utils::get_history_tool_calls_count(&chat_request);

        // Check parser availability once upfront (not per choice)
        let reasoning_parser_available = chat_request.separate_reasoning
            && utils::check_reasoning_parser_availability(
                &self.reasoning_parser_factory,
                self.configured_reasoning_parser.as_deref(),
                &chat_request.model,
            );

        let tool_choice_enabled = !matches!(
            &chat_request.tool_choice,
            Some(ToolChoice::Value(ToolChoiceValue::None))
        );

        let tool_parser_available = tool_choice_enabled
            && chat_request.tools.is_some()
            && utils::check_tool_parser_availability(
                &self.tool_parser_factory,
                self.configured_tool_parser.as_deref(),
                &chat_request.model,
            );

        // Log once per request (not per choice)
        if chat_request.separate_reasoning && !reasoning_parser_available {
            tracing::debug!(
                "No reasoning parser found for model '{}', skipping reasoning parsing",
                chat_request.model
            );
        }

        if chat_request.tools.is_some() && tool_choice_enabled && !tool_parser_available {
            tracing::debug!(
                "No tool parser found for model '{}', skipping tool call parsing",
                chat_request.model
            );
        }

        // Process all choices
        let mut choices = Vec::new();
        for (index, complete) in all_responses.iter().enumerate() {
            match self
                .process_single_choice(
                    complete,
                    index,
                    &chat_request,
                    &tokenizer,
                    stop_decoder,
                    history_tool_calls_count,
                    reasoning_parser_available,
                    tool_parser_available,
                )
                .await
            {
                Ok(choice) => choices.push(choice),
                Err(e) => {
                    return Err(error::internal_error(
                        "process_choice_failed",
                        format!("Failed to process choice {index}: {e}"),
                    ));
                }
            }
        }

        // Build usage
        let usage = response_formatting::build_usage(&all_responses);

        // Build final ChatCompletionResponse
        Ok(
            ChatCompletionResponse::builder(&dispatch.request_id, &dispatch.model)
                .created(dispatch.created)
                .choices(choices)
                .usage(usage)
                .maybe_system_fingerprint(dispatch.weight_version.clone())
                .build(),
        )
    }

    /// Parse tool calls using model-specific parser
    fn normalize_schema_type(type_name: &str) -> Option<&'static str> {
        match type_name {
            "integer" => Some("integer"),
            "number" => Some("number"),
            "string" => Some("string"),
            "boolean" => Some("boolean"),
            "object" => Some("object"),
            "array" => Some("array"),
            "null" => Some("string"),
            _ => None,
        }
    }

    fn infer_type_from_json_schema(schema: &Value) -> Option<&'static str> {
        let Value::Object(schema_obj) = schema else {
            return None;
        };

        if let Some(type_value) = schema_obj.get("type") {
            match type_value {
                Value::String(type_name) => return Self::normalize_schema_type(type_name),
                Value::Array(type_names) => {
                    for type_name in type_names {
                        if let Some(type_name) = type_name.as_str() {
                            if type_name != "null" {
                                return Self::normalize_schema_type(type_name);
                            }
                        }
                    }
                    return Some("string");
                }
                _ => {}
            }
        }

        for union_key in ["anyOf", "oneOf"] {
            if let Some(Value::Array(schemas)) = schema_obj.get(union_key) {
                let mut inferred_types = Vec::new();
                for sub_schema in schemas {
                    if let Some(inferred_type) = Self::infer_type_from_json_schema(sub_schema) {
                        inferred_types.push(inferred_type);
                    }
                }
                if let Some(first_type) = inferred_types.first().copied() {
                    if inferred_types.iter().all(|ty| *ty == first_type) {
                        return Some(first_type);
                    }
                    if inferred_types.contains(&"string") {
                        return Some("string");
                    }
                    return Some(first_type);
                }
            }
        }

        if let Some(Value::Array(enum_values)) = schema_obj.get("enum") {
            if enum_values.is_empty() {
                return Some("string");
            }

            let inferred = enum_values
                .iter()
                .map(|value| match value {
                    Value::Null => "null",
                    Value::Bool(_) => "boolean",
                    Value::Number(number) if number.is_i64() || number.is_u64() => "integer",
                    Value::Number(_) => "number",
                    Value::String(_) => "string",
                    Value::Array(_) => "array",
                    Value::Object(_) => "object",
                })
                .collect::<std::collections::HashSet<_>>();

            if inferred.len() == 1 {
                return inferred.iter().next().copied();
            }
            return Some("string");
        }

        if let Some(Value::Array(schemas)) = schema_obj.get("allOf") {
            for sub_schema in schemas {
                if let Some(inferred_type) = Self::infer_type_from_json_schema(sub_schema) {
                    if inferred_type != "string" {
                        return Some(inferred_type);
                    }
                }
            }
            return Some("string");
        }

        if schema_obj.contains_key("properties") {
            return Some("object");
        }
        if schema_obj.contains_key("items") {
            return Some("array");
        }

        None
    }

    fn get_argument_type(
        tools: &[OpenAITool],
        function_name: &str,
        argument_name: &str,
    ) -> Option<&'static str> {
        let tool = tools
            .iter()
            .find(|tool| tool.function.name == function_name)?;
        let params = tool.function.parameters.as_object()?;
        let properties = params.get("properties")?.as_object()?;
        let schema = properties.get(argument_name)?;
        Self::infer_type_from_json_schema(schema)
    }

    fn normalize_arguments_to_schema(
        tools: &[OpenAITool],
        function_name: &str,
        arguments_json: &str,
    ) -> String {
        let Ok(Value::Object(arguments)) = serde_json::from_str::<Value>(arguments_json) else {
            return arguments_json.to_string();
        };

        let mut normalized = serde_json::Map::with_capacity(arguments.len());
        for (key, value) in arguments {
            let normalized_value = match Self::get_argument_type(tools, function_name, &key) {
                Some("string") => match value {
                    Value::String(_) => value,
                    Value::Array(array) => Value::String(Value::Array(array).to_string()),
                    Value::Object(object) => Value::String(Value::Object(object).to_string()),
                    Value::Null => Value::String("null".to_string()),
                    Value::Bool(v) => Value::String(v.to_string()),
                    Value::Number(v) => Value::String(v.to_string()),
                },
                Some("number") | Some("integer") => match value {
                    Value::String(number_like) => {
                        if let Ok(int_val) = number_like.parse::<i64>() {
                            Value::Number(int_val.into())
                        } else if let Ok(float_val) = number_like.parse::<f64>() {
                            serde_json::Number::from_f64(float_val)
                                .map(Value::Number)
                                .unwrap_or(Value::String(number_like))
                        } else {
                            Value::String(number_like)
                        }
                    }
                    other => other,
                },
                _ => value,
            };
            normalized.insert(key, normalized_value);
        }

        serde_json::to_string(&normalized).unwrap_or_else(|_| arguments_json.to_string())
    }

    pub async fn parse_tool_calls(
        &self,
        processed_text: &str,
        model: &str,
        request_tools: Option<&[OpenAITool]>,
        history_tool_calls_count: usize,
    ) -> (Option<Vec<ToolCall>>, String) {
        // Get pooled parser for this model
        let pooled_parser = utils::get_tool_parser(
            &self.tool_parser_factory,
            self.configured_tool_parser.as_deref(),
            model,
        );

        // Try parsing directly (parser will handle detection internally)
        let result = {
            let parser = pooled_parser.lock().await;
            parser.parse_complete(processed_text).await
            // Lock is dropped here
        };

        match result {
            Ok((normal_text, parsed_tool_calls)) => {
                if parsed_tool_calls.is_empty() {
                    return (None, normal_text);
                }

                let spec_tool_calls = parsed_tool_calls
                    .into_iter()
                    .enumerate()
                    .map(|(index, tc)| {
                        let function_name = tc.function.name;
                        let normalized_arguments = request_tools.map_or_else(
                            || tc.function.arguments.clone(),
                            |tools| {
                                Self::normalize_arguments_to_schema(
                                    tools,
                                    &function_name,
                                    &tc.function.arguments,
                                )
                            },
                        );
                        // Generate ID for this tool call
                        let id = utils::generate_tool_call_id(
                            model,
                            &function_name,
                            index,
                            history_tool_calls_count,
                        );
                        ToolCall {
                            id,
                            tool_type: "function".to_string(),
                            function: FunctionCallResponse {
                                name: function_name,
                                arguments: Some(normalized_arguments),
                            },
                        }
                    })
                    .collect();
                (Some(spec_tool_calls), normal_text)
            }
            Err(e) => {
                error!("Tool call parsing error: {}", e);
                (None, processed_text.to_string())
            }
        }
    }

    /// Process non-streaming generate response (collects all responses and builds final response array)
    pub async fn process_non_streaming_generate_response(
        &self,
        execution_result: ExecutionResult,
        _generate_request: Arc<GenerateRequest>,
        dispatch: DispatchMetadata,
        stop_decoder: &mut StopSequenceDecoder,
        request_logprobs: bool,
        start_time: Instant,
    ) -> Result<Vec<GenerateResponse>, axum::response::Response> {
        // Collect all responses from the execution result
        let all_responses =
            response_collection::collect_responses(execution_result, request_logprobs).await?;

        // Process each completion
        let mut result_array = Vec::new();
        for complete in all_responses {
            stop_decoder.reset();

            // Process tokens through stop decoder
            let outputs = match stop_decoder.process_tokens(complete.output_ids()) {
                Ok(outputs) => outputs,
                Err(e) => {
                    return Err(error::internal_error(
                        "process_tokens_failed",
                        format!("Failed to process tokens: {e}"),
                    ))
                }
            };

            // Accumulate text with early breaks
            let mut decoded_text = String::new();
            for output in outputs {
                match output {
                    SequenceDecoderOutput::Text(t) => decoded_text.push_str(&t),
                    SequenceDecoderOutput::StoppedWithText(t) => {
                        decoded_text.push_str(&t);
                        break;
                    }
                    SequenceDecoderOutput::Stopped => break,
                    SequenceDecoderOutput::Held => {}
                }
            }

            // Flush remaining text
            if let SequenceDecoderOutput::Text(t) = stop_decoder.flush() {
                decoded_text.push_str(&t);
            }

            let output_ids = complete.output_ids().to_vec();
            let finish_reason_str = complete.finish_reason();

            // Parse finish_reason from string to proper type
            let finish_reason =
                utils::parse_finish_reason(finish_reason_str, complete.completion_tokens());

            let matched_stop = complete.matched_stop_json();

            // Extract logprobs if requested (convert proto types to Generate format)
            let input_token_logprobs = if request_logprobs {
                complete
                    .input_logprobs()
                    .as_ref()
                    .map(utils::convert_generate_input_logprobs)
            } else {
                None
            };

            let output_token_logprobs = if request_logprobs {
                complete
                    .output_logprobs()
                    .as_ref()
                    .map(utils::convert_generate_output_logprobs)
            } else {
                None
            };

            // Build GenerateResponse struct
            let meta_info = GenerateMetaInfo {
                id: dispatch.request_id.clone(),
                finish_reason,
                prompt_tokens: complete.prompt_tokens(),
                weight_version: dispatch
                    .weight_version
                    .clone()
                    .unwrap_or_else(|| "default".to_string()),
                input_token_logprobs,
                output_token_logprobs,
                completion_tokens: complete.completion_tokens(),
                cached_tokens: complete.cached_tokens(),
                e2e_latency: start_time.elapsed().as_secs_f64(),
                matched_stop,
            };

            result_array.push(GenerateResponse {
                text: decoded_text,
                output_ids,
                meta_info,
            });
        }

        Ok(result_array)
    }
}

#[cfg(test)]
mod tests {
    use super::ResponseProcessor;
    use openai_protocol::common::{Function, Tool};
    use serde_json::json;

    #[test]
    fn normalize_arguments_respects_string_schema() {
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "invokeCallback".to_string(),
                description: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "callback": {"type": "string"},
                        "error": {"type": "string"},
                        "value": {"type": "string"}
                    }
                }),
                strict: None,
            },
        }];

        let normalized = ResponseProcessor::normalize_arguments_to_schema(
            &tools,
            "invokeCallback",
            r#"{"callback":"processResult","error":null,"value":"Operation successful"}"#,
        );

        assert_eq!(
            normalized,
            r#"{"callback":"processResult","error":"null","value":"Operation successful"}"#
        );
    }

    #[test]
    fn normalize_arguments_keeps_numeric_schema_for_numbers() {
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "calculate_density".to_string(),
                description: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "country": {"type": "string"},
                        "year": {"type": "string"},
                        "population": {"type": "number"},
                        "land_area": {"type": "number"}
                    }
                }),
                strict: None,
            },
        }];

        let normalized = ResponseProcessor::normalize_arguments_to_schema(
            &tools,
            "calculate_density",
            r#"{"country":"China","year":2000,"population":1267000000,"land_area":9597000}"#,
        );

        assert_eq!(
            normalized,
            r#"{"country":"China","year":"2000","population":1267000000,"land_area":9597000}"#
        );
    }
}
