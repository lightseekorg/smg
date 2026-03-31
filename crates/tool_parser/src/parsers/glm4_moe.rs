use std::collections::HashSet;

use async_trait::async_trait;
use openai_protocol::common::Tool;
use regex::Regex;
use serde_json::Value;

use crate::{
    errors::{ParserError, ParserResult},
    parsers::helpers,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

/// GLM-4 MoE format parser for tool calls
///
/// Handles both GLM-4 MoE and GLM-4.7 MoE formats:
/// - GLM-4: `<tool_call>{name}\n<arg_key>{key}</arg_key>\n<arg_value>{value}</arg_value>\n</tool_call>`
/// - GLM-4.7: `<tool_call>{name}<arg_key>{key}</arg_key><arg_value>{value}</arg_value></tool_call>`
///
/// Features:
/// - XML-style tags for tool calls
/// - Key-value pairs for arguments
/// - Support for multiple sequential tool calls
pub struct Glm4MoeParser {
    /// Regex for extracting complete tool calls
    tool_call_extractor: Regex,
    /// Regex for extracting function details
    func_detail_extractor: Regex,
    /// Regex for extracting argument key-value pairs
    arg_extractor: Regex,

    /// Buffer for accumulating incomplete patterns across chunks
    buffer: String,

    /// Stores complete tool call info (name and arguments) for each tool being parsed
    prev_tool_call_arr: Vec<Value>,

    /// Index of currently streaming tool call (-1 means no active tool)
    current_tool_id: i32,

    /// Tracks raw JSON string content streamed to client for each tool's arguments
    streamed_args_for_tool: Vec<String>,

    /// Token configuration
    bot_token: &'static str,
    eot_token: &'static str,
}

impl Glm4MoeParser {
    fn parse_incomplete_trailing_tool_call(
        &self,
        text: &str,
        tools: Option<&[Tool]>,
    ) -> Option<ToolCall> {
        let start = text.rfind(self.bot_token)?;
        let trailing = &text[start..];

        if trailing.contains(self.eot_token) {
            return None;
        }

        let repaired_block = format!("{trailing}{}", self.eot_token);
        match self.parse_tool_call(&repaired_block, tools) {
            Ok(Some(tool_call)) => Some(tool_call),
            Ok(None) | Err(_) => None,
        }
    }

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
                Value::String(type_name) => {
                    return Self::normalize_schema_type(type_name);
                }
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

            let inferred: HashSet<&str> = enum_values
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
                .collect();

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

    fn get_argument_type(func_name: &str, arg_key: &str, tools: &[Tool]) -> Option<&'static str> {
        let tool = tools.iter().find(|tool| tool.function.name == func_name)?;
        let params = tool.function.parameters.as_object()?;
        let properties = params.get("properties")?.as_object()?;
        let arg_schema = properties.get(arg_key)?;
        Self::infer_type_from_json_schema(arg_schema)
    }

    fn parse_argument_value(value_str: &str, arg_type: Option<&str>) -> Value {
        let value_str = value_str.trim();

        if let Ok(json_val) = serde_json::from_str::<Value>(value_str) {
            return match arg_type {
                Some("string") => match json_val {
                    Value::String(_) => json_val,
                    Value::Array(array) => Value::String(Value::Array(array).to_string()),
                    Value::Object(object) => Value::String(Value::Object(object).to_string()),
                    Value::Null => Value::String(value_str.to_string()),
                    other => Value::String(match other {
                        Value::Bool(v) => v.to_string(),
                        Value::Number(v) => v.to_string(),
                        _ => value_str.to_string(),
                    }),
                },
                Some("number") | Some("integer") => match json_val {
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
                _ => json_val,
            };
        }

        match arg_type {
            Some("string") => Value::String(value_str.to_string()),
            Some("boolean") => match value_str {
                "true" | "True" => Value::Bool(true),
                "false" | "False" => Value::Bool(false),
                _ => Value::String(value_str.to_string()),
            },
            Some("number") | Some("integer") => {
                if let Ok(int_val) = value_str.parse::<i64>() {
                    Value::Number(int_val.into())
                } else if let Ok(float_val) = value_str.parse::<f64>() {
                    serde_json::Number::from_f64(float_val)
                        .map(Value::Number)
                        .unwrap_or(Value::String(value_str.to_string()))
                } else {
                    Value::String(value_str.to_string())
                }
            }
            _ => {
                if value_str == "true" || value_str == "True" {
                    Value::Bool(true)
                } else if value_str == "false" || value_str == "False" {
                    Value::Bool(false)
                } else if value_str == "null" || value_str == "None" {
                    Value::Null
                } else if let Ok(int_val) = value_str.parse::<i64>() {
                    Value::Number(int_val.into())
                } else if let Ok(float_val) = value_str.parse::<f64>() {
                    serde_json::Number::from_f64(float_val)
                        .map(Value::Number)
                        .unwrap_or(Value::String(value_str.to_string()))
                } else {
                    Value::String(value_str.to_string())
                }
            }
        }
    }

    /// Create a new generic GLM MoE parser with a custom func_detail_extractor pattern
    ///
    /// # Arguments
    /// - `func_detail_pattern`: Regex pattern for extracting function name and arguments
    ///   - For GLM-4: `r"(?s)<tool_call>([^\n]*)\n(.*)</tool_call>"`
    ///   - For GLM-4.7: `r"(?s)<tool_call>\s*([^<\s]+)\s*(.*?)</tool_call>"`
    #[expect(
        clippy::expect_used,
        reason = "regex patterns are compile-time string literals"
    )]
    pub(crate) fn new(func_detail_pattern: &str) -> Self {
        // Use (?s) flag for DOTALL mode to handle newlines
        let tool_call_pattern = r"(?s)<tool_call>.*?</tool_call>";
        let tool_call_extractor = Regex::new(tool_call_pattern).expect("Valid regex pattern");

        let func_detail_extractor = Regex::new(func_detail_pattern).expect("Valid regex pattern");

        let arg_pattern = r"(?s)<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>";
        let arg_extractor = Regex::new(arg_pattern).expect("Valid regex pattern");

        Self {
            tool_call_extractor,
            func_detail_extractor,
            arg_extractor,
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            streamed_args_for_tool: Vec::new(),
            bot_token: "<tool_call>",
            eot_token: "</tool_call>",
        }
    }

    /// Create a new GLM-4.5/4.6 MoE parser (with newline-based format)
    pub fn glm45() -> Self {
        Self::new(r"(?s)<tool_call>([^\n]*)\n(.*)</tool_call>")
    }

    /// Create a new GLM-4.7 MoE parser (with whitespace-based format)
    pub fn glm47() -> Self {
        Self::new(r"(?s)<tool_call>\s*([^<\s]+)\s*(.*?)</tool_call>")
    }

    /// Parse arguments from key-value pairs, using tool schema for type inference
    fn parse_arguments(
        &self,
        args_text: &str,
        func_name: &str,
        tools: Option<&[Tool]>,
    ) -> serde_json::Map<String, Value> {
        let mut arguments = serde_json::Map::new();

        for capture in self.arg_extractor.captures_iter(args_text) {
            let key = capture.get(1).map_or("", |m| m.as_str()).trim();
            let value_str = capture.get(2).map_or("", |m| m.as_str()).trim();
            let arg_type =
                tools.and_then(|tool_defs| Self::get_argument_type(func_name, key, tool_defs));
            let value = Self::parse_argument_value(value_str, arg_type);

            arguments.insert(key.to_string(), value);
        }

        arguments
    }

    /// Parse a single tool call block
    fn parse_tool_call(
        &self,
        block: &str,
        tools: Option<&[Tool]>,
    ) -> ParserResult<Option<ToolCall>> {
        if let Some(captures) = self.func_detail_extractor.captures(block) {
            let func_name = captures.get(1).map_or("", |m| m.as_str()).trim();
            let args_text = captures.get(2).map_or("", |m| m.as_str());
            let arguments = self.parse_arguments(args_text, func_name, tools);

            let arguments_str = serde_json::to_string(&arguments)
                .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

            Ok(Some(ToolCall {
                function: FunctionCall {
                    name: func_name.to_string(),
                    arguments: arguments_str,
                },
            }))
        } else {
            Ok(None)
        }
    }

    /// Parse all tool calls from text (shared logic for complete and incremental parsing)
    fn parse_tool_calls_from_text(&self, text: &str, tools: Option<&[Tool]>) -> Vec<ToolCall> {
        let mut parsed_tools = Vec::new();

        for mat in self.tool_call_extractor.find_iter(text) {
            match self.parse_tool_call(mat.as_str(), tools) {
                Ok(Some(tool)) => parsed_tools.push(tool),
                Ok(None) => continue,
                Err(e) => {
                    tracing::debug!("Failed to parse tool call: {}", e);
                    continue;
                }
            }
        }

        parsed_tools
    }
}

impl Default for Glm4MoeParser {
    fn default() -> Self {
        Self::glm45()
    }
}

#[async_trait]
impl ToolParser for Glm4MoeParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        // Check if text contains GLM-4 MoE format
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Find where tool calls begin
        // Safe: has_tool_markers() already confirmed the marker exists
        let idx = text
            .find("<tool_call>")
            .ok_or_else(|| ParserError::ParsingFailed("tool call marker not found".to_string()))?;
        let normal_text = text[..idx].to_string();

        // Parse all tool calls, including incomplete trailing ones
        let mut tools = self.parse_tool_calls_from_text(text, None);
        if let Some(trailing_tool_call) = self.parse_incomplete_trailing_tool_call(text, None) {
            tools.push(trailing_tool_call);
        }

        if tools.is_empty() {
            return Ok((text.to_string(), vec![]));
        }

        Ok((normal_text, tools))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        // Python logic: Wait for complete tool call, then parse it all at once
        self.buffer.push_str(chunk);
        let current_text = &self.buffer.clone();

        // Check if we have bot_token
        let start = current_text.find(self.bot_token);
        if start.is_none() {
            // Preserve partial bot_token at the end of the buffer
            if let Some(partial_len) =
                helpers::ends_with_partial_token(current_text, self.bot_token)
            {
                let split_at = current_text.len() - partial_len;
                let normal_text = if self.current_tool_id >= 0 {
                    String::new()
                } else {
                    current_text[..split_at].to_string()
                };
                self.buffer = current_text[split_at..].to_string();
                return Ok(StreamingParseResult {
                    normal_text,
                    calls: vec![],
                });
            }

            self.buffer.clear();
            let normal_text = if self.current_tool_id >= 0 {
                String::new()
            } else {
                current_text.clone()
            };
            return Ok(StreamingParseResult {
                normal_text,
                calls: vec![],
            });
        }

        // Check if we have eot_token (end of tool call)
        let end = current_text.find(self.eot_token);
        if end.is_some() {
            // Initialize state if this is the first tool call
            if self.current_tool_id == -1 {
                self.current_tool_id = 0;
                self.prev_tool_call_arr = Vec::new();
                self.streamed_args_for_tool = vec![String::new()];
            }

            let tool_indices = helpers::get_tool_indices(tools);

            // Extract normal text before the first tool call
            let idx = current_text.find(self.bot_token);
            let normal_text = if let Some(pos) = idx {
                current_text[..pos].trim().to_string()
            } else {
                String::new()
            };

            let Some(start_pos) = idx else {
                return Ok(StreamingParseResult::default());
            };

            let mut calls = Vec::new();
            let mut parse_cursor = start_pos;

            loop {
                let remaining = &current_text[parse_cursor..];
                let Some(end_rel) = remaining.find(self.eot_token) else {
                    break;
                };

                let block_end = end_rel + self.eot_token.len();
                let block = &remaining[..block_end];

                helpers::ensure_capacity(
                    self.current_tool_id,
                    &mut self.prev_tool_call_arr,
                    &mut self.streamed_args_for_tool,
                );

                let parsed_tools = self.parse_tool_calls_from_text(block, Some(tools));
                if !parsed_tools.is_empty() {
                    let tool_call = &parsed_tools[0];
                    let tool_id = self.current_tool_id as usize;

                    if !tool_indices.contains_key(&tool_call.function.name) {
                        tracing::debug!(
                            "Invalid tool name '{}' - skipping",
                            tool_call.function.name
                        );
                        helpers::reset_current_tool_state(
                            &mut self.buffer,
                            &mut false,
                            &mut self.streamed_args_for_tool,
                            &self.prev_tool_call_arr,
                        );
                        return Ok(StreamingParseResult::default());
                    }

                    calls.push(ToolCallItem {
                        tool_index: tool_id,
                        name: Some(tool_call.function.name.clone()),
                        parameters: tool_call.function.arguments.clone(),
                    });

                    if self.prev_tool_call_arr.len() <= tool_id {
                        self.prev_tool_call_arr
                            .resize_with(tool_id + 1, || Value::Null);
                    }

                    if let Ok(args) = serde_json::from_str::<Value>(&tool_call.function.arguments) {
                        self.prev_tool_call_arr[tool_id] = serde_json::json!({
                            "name": tool_call.function.name,
                            "arguments": args,
                        });
                    }

                    if self.streamed_args_for_tool.len() <= tool_id {
                        self.streamed_args_for_tool
                            .resize_with(tool_id + 1, String::new);
                    }
                    self.streamed_args_for_tool[tool_id].clone_from(&tool_call.function.arguments);

                    self.current_tool_id += 1;
                }

                parse_cursor += block_end;
            }

            self.buffer = current_text[parse_cursor..].to_string();
            return Ok(StreamingParseResult { normal_text, calls });
        }

        // No complete tool call yet - return normal text before start token
        // Safe: start.is_none() case was handled above (early return)
        let Some(start_pos) = start else {
            return Ok(StreamingParseResult::default());
        };
        let normal_text = current_text[..start_pos].to_string();
        self.buffer = current_text[start_pos..].to_string();

        Ok(StreamingParseResult {
            normal_text,
            calls: vec![],
        })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(self.bot_token)
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.prev_tool_call_arr.clear();
        self.current_tool_id = -1;
        self.streamed_args_for_tool.clear();
    }
}
