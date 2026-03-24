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

/// DSML (DeepSeek Markup Language) tag constants used by DeepSeek V3.2
const FUNCTION_CALLS_START: &str = "<\u{ff5c}DSML\u{ff5c}function_calls>";
const FUNCTION_CALLS_END: &str = "</\u{ff5c}DSML\u{ff5c}function_calls>";
const INVOKE_END: &str = "</\u{ff5c}DSML\u{ff5c}invoke>";

/// DeepSeek V3.2 DSML format parser for tool calls
///
/// Handles the DeepSeek V3.2 specific format that uses DSML (DeepSeek Markup Language)
/// with XML-like markup for function calls:
///
/// ```text
/// <｜DSML｜function_calls>
/// <｜DSML｜invoke name="get_weather">
/// <｜DSML｜parameter name="location" string="true">Tokyo</｜DSML｜parameter>
/// <｜DSML｜parameter name="count" string="false">5</｜DSML｜parameter>
/// </｜DSML｜invoke>
/// </｜DSML｜function_calls>
/// ```
///
/// Also supports JSON inside invoke blocks (dual format):
///
/// ```text
/// <｜DSML｜invoke name="get_weather">
/// {"location": "Tokyo", "count": 5}
/// </｜DSML｜invoke>
/// ```
///
/// Features:
/// - DSML XML-like markup with Unicode delimiters
/// - Dual format: XML parameters or inline JSON
/// - `string="true"` for raw string values, `string="false"` for JSON values
/// - Buffer-until-complete-invoke streaming strategy
pub struct DeepSeekV32Parser {
    /// Regex for extracting complete invoke blocks
    invoke_extractor: Regex,
    /// Regex for extracting DSML parameters from an invoke body
    param_extractor: Regex,

    /// Buffer for accumulating incomplete patterns across chunks
    buffer: String,

    /// Stores complete tool call info (name and arguments) for each tool being parsed
    prev_tool_call_arr: Vec<Value>,

    /// Index of currently streaming tool call (-1 means no active tool)
    current_tool_id: i32,

    /// Flag for whether current tool's name has been sent to client
    current_tool_name_sent: bool,

    /// Tracks raw JSON string content streamed to client for each tool's arguments
    streamed_args_for_tool: Vec<String>,
}

impl DeepSeekV32Parser {
    /// Create a new DeepSeek V3.2 DSML parser
    #[expect(
        clippy::expect_used,
        reason = "regex patterns are compile-time string literals"
    )]
    pub fn new() -> Self {
        // Regex for matching complete invoke blocks
        // Captures: (1) function name, (2) invoke body
        let invoke_pattern =
            "(?s)<\u{ff5c}DSML\u{ff5c}invoke name=\"([^\"]+)\">(.*?)</\u{ff5c}DSML\u{ff5c}invoke>";
        let invoke_extractor = Regex::new(invoke_pattern).expect("Valid regex pattern");

        // Regex for matching DSML parameter tags within an invoke body
        // Captures: (1) param name, (2) string flag ("true"|"false"), (3) param value
        let param_pattern = "(?s)<\u{ff5c}DSML\u{ff5c}parameter name=\"([^\"]+)\" string=\"(true|false)\">(.*?)</\u{ff5c}DSML\u{ff5c}parameter>";
        let param_extractor = Regex::new(param_pattern).expect("Valid regex pattern");

        Self {
            invoke_extractor,
            param_extractor,
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
        }
    }

    /// Parse the body of a single invoke block into a ToolCall.
    ///
    /// Supports two formats:
    /// 1. JSON body: the invoke body is valid JSON
    /// 2. XML parameters: the invoke body contains `<｜DSML｜parameter>` tags
    fn parse_invoke(&self, name: &str, body: &str) -> ParserResult<ToolCall> {
        let trimmed = body.trim();

        // Try JSON first (dual format support)
        if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
            let args = if value.is_object() {
                value
            } else {
                serde_json::json!({ "value": value })
            };
            let arguments = serde_json::to_string(&args)
                .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;
            return Ok(ToolCall {
                function: FunctionCall {
                    name: name.to_string(),
                    arguments,
                },
            });
        }

        // Fall back to XML parameter extraction
        let mut args = serde_json::Map::new();
        for cap in self.param_extractor.captures_iter(trimmed) {
            let param_name = cap.get(1).map_or("", |m| m.as_str());
            let is_string = cap.get(2).map_or("true", |m| m.as_str()) == "true";
            let raw_value = cap.get(3).map_or("", |m| m.as_str());

            let value = if is_string {
                // string="true" → always a raw string
                Value::String(raw_value.to_string())
            } else {
                // string="false" → try to parse as JSON, fall back to string
                serde_json::from_str::<Value>(raw_value)
                    .unwrap_or_else(|_| Value::String(raw_value.to_string()))
            };

            args.insert(param_name.to_string(), value);
        }

        if args.is_empty() && !trimmed.is_empty() {
            // Body is non-empty but we couldn't parse it as JSON or XML params
            return Err(ParserError::ParsingFailed(format!(
                "Failed to parse invoke body for '{name}'"
            )));
        }

        let arguments = serde_json::to_string(&Value::Object(args))
            .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

        Ok(ToolCall {
            function: FunctionCall {
                name: name.to_string(),
                arguments,
            },
        })
    }
}

impl Default for DeepSeekV32Parser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for DeepSeekV32Parser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Find where function calls begin
        let idx = text.find(FUNCTION_CALLS_START).ok_or_else(|| {
            ParserError::ParsingFailed("DSML function_calls marker not found".to_string())
        })?;
        let normal_text = text[..idx].to_string();

        // Extract all invoke blocks
        let mut tools = Vec::new();
        for cap in self.invoke_extractor.captures_iter(text) {
            let func_name = cap.get(1).map_or("", |m| m.as_str()).trim();
            let body = cap.get(2).map_or("", |m| m.as_str());

            match self.parse_invoke(func_name, body) {
                Ok(tool) => tools.push(tool),
                Err(e) => {
                    tracing::debug!("Failed to parse DSML invoke block: {}", e);
                    continue;
                }
            }
        }

        // If no tools were successfully parsed despite having markers, return entire text
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
        self.buffer.push_str(chunk);
        let current_text = self.buffer.clone();

        // Check if we have any DSML markers
        let has_dsml = self.has_tool_markers(&current_text)
            || current_text.contains("<\u{ff5c}DSML\u{ff5c}invoke");

        if !has_dsml {
            // No DSML markers detected — return all buffered content as normal text
            let mut normal_text = std::mem::take(&mut self.buffer);
            // Strip out end tokens if present
            for e_token in [FUNCTION_CALLS_END, INVOKE_END] {
                normal_text = normal_text.replace(e_token, "");
            }
            return Ok(StreamingParseResult {
                normal_text,
                calls: vec![],
            });
        }

        let tool_indices = helpers::get_tool_indices(tools);
        let mut calls: Vec<ToolCallItem> = Vec::new();

        // Buffer-until-complete-invoke strategy: look for complete invoke blocks
        while let Some(cap) = self.invoke_extractor.captures(&self.buffer.clone()) {
            let full_match = cap.get(0).map_or("", |m| m.as_str());
            let match_end = cap.get(0).map(|m| m.end()).unwrap_or(0);
            let func_name = cap.get(1).map_or("", |m| m.as_str()).trim();
            let body = cap.get(2).map_or("", |m| m.as_str());

            // Validate tool name
            if !tool_indices.contains_key(func_name) {
                tracing::debug!(
                    "Invalid tool name '{}' in DSML invoke - skipping",
                    func_name
                );
                // Remove the invalid invoke from buffer and continue
                self.buffer = self.buffer.replacen(full_match, "", 1);
                continue;
            }

            // Initialize state if this is the first tool call
            if self.current_tool_id == -1 {
                self.current_tool_id = 0;
                self.prev_tool_call_arr = Vec::new();
                self.streamed_args_for_tool = vec![String::new()];
            }

            // Ensure capacity
            helpers::ensure_capacity(
                self.current_tool_id,
                &mut self.prev_tool_call_arr,
                &mut self.streamed_args_for_tool,
            );

            let tool_id = self.current_tool_id as usize;

            // Parse the invoke body to get arguments
            match self.parse_invoke(func_name, body) {
                Ok(tool_call) => {
                    // Emit name
                    calls.push(ToolCallItem {
                        tool_index: tool_id,
                        name: Some(func_name.to_string()),
                        parameters: String::new(),
                    });

                    // Emit full arguments at once
                    let args_str = &tool_call.function.arguments;
                    if !args_str.is_empty() {
                        calls.push(ToolCallItem {
                            tool_index: tool_id,
                            name: None,
                            parameters: args_str.clone(),
                        });
                        if tool_id < self.streamed_args_for_tool.len() {
                            self.streamed_args_for_tool[tool_id].push_str(args_str);
                        }
                    }

                    // Store the tool call info
                    if tool_id < self.prev_tool_call_arr.len() {
                        self.prev_tool_call_arr[tool_id] = serde_json::json!({
                            "name": func_name,
                            "arguments": serde_json::from_str::<Value>(args_str).unwrap_or(Value::Object(serde_json::Map::new())),
                        });
                    }
                }
                Err(e) => {
                    tracing::debug!("Failed to parse DSML invoke during streaming: {}", e);
                    // Remove the broken invoke from buffer and continue
                    self.buffer = self.buffer.replacen(full_match, "", 1);
                    continue;
                }
            }

            // Remove the completed invoke from buffer
            self.buffer = self.buffer[match_end..].to_string();

            // Advance to next tool
            self.current_tool_id += 1;
            self.current_tool_name_sent = false;
        }

        Ok(StreamingParseResult {
            normal_text: String::new(),
            calls,
        })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(FUNCTION_CALLS_START)
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.prev_tool_call_arr.clear();
        self.current_tool_id = -1;
        self.current_tool_name_sent = false;
        self.streamed_args_for_tool.clear();
    }
}
