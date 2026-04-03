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

/// DeepSeek V3.2 DSML format parser for tool calls
///
/// Handles the DeepSeek V3.2 DSML format:
/// ```text
/// <｜DSML｜function_calls>
/// <｜DSML｜invoke name="func">
/// <｜DSML｜parameter name="key" string="true">value</｜DSML｜parameter>
/// </｜DSML｜invoke>
/// </｜DSML｜function_calls>
/// ```
///
/// Also supports direct JSON inside invoke blocks as a fallback format.
///
/// Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3.2
pub struct DeepSeek32Parser {
    /// Regex for extracting full function_calls block content
    tool_call_complete_regex: Regex,
    /// Regex for extracting complete invoke blocks (name + body)
    invoke_complete_regex: Regex,
    /// Regex for extracting complete parameter tags (name, string attr, value)
    parameter_complete_regex: Regex,
    /// Regex for matching partial parameter tag during streaming (no closing tag)
    partial_parameter_regex: Regex,
    /// Regex for matching invoke blocks (complete or partial, for streaming)
    invoke_regex: Regex,

    /// Buffer for accumulating incomplete patterns across chunks
    buffer: String,
    /// Stores complete tool call info for each tool being parsed
    prev_tool_call_arr: Vec<Value>,
    /// Index of currently streaming tool call (-1 means no active tool)
    current_tool_id: i32,
    /// Flag for whether current tool's name has been sent to client
    current_tool_name_sent: bool,
    /// Tracks raw JSON string content streamed to client for each tool's arguments
    streamed_args_for_tool: Vec<String>,
}

/// DSML fragment suffixes to strip from partial content during streaming.
/// When a DSML closing tag arrives across chunks, the buffer may end with
/// partial fragments like `</｜DSML｜para...` or `</｜DSML｜inv...`.
const DSML_TRAILING_FRAGMENTS: &[&str] = &[
    "</｜DSML｜parameter>",
    "</｜DSML｜parameter",
    "</｜DSML｜paramete",
    "</｜DSML｜paramet",
    "</｜DSML｜parame",
    "</｜DSML｜param",
    "</｜DSML｜para",
    "</｜DSML｜par",
    "</｜DSML｜pa",
    "</｜DSML｜p",
    "</｜DSML｜invoke>",
    "</｜DSML｜invoke",
    "</｜DSML｜invok",
    "</｜DSML｜invo",
    "</｜DSML｜inv",
    "</｜DSML｜in",
    "</｜DSML｜i",
    "</｜DSML｜",
    "</｜DSML",
    "</｜DSM",
    "</｜DS",
    "</｜D",
    "</｜",
    "</",
];

impl DeepSeek32Parser {
    /// Create a new DeepSeek V3.2 parser
    #[expect(
        clippy::expect_used,
        reason = "regex patterns are compile-time string literals"
    )]
    pub fn new() -> Self {
        let tool_call_complete_regex =
            Regex::new(r"(?s)<｜DSML｜function_calls>(.*?)</｜DSML｜function_calls>")
                .expect("Valid regex pattern");

        let invoke_complete_regex =
            Regex::new(r#"(?s)<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>"#)
                .expect("Valid regex pattern");

        let parameter_complete_regex = Regex::new(
            r#"(?s)<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</｜DSML｜parameter>"#,
        )
        .expect("Valid regex pattern");

        let partial_parameter_regex = Regex::new(
            r#"(?s)<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*)$"#,
        )
        .expect("Valid regex pattern");

        let invoke_regex =
            Regex::new(r#"(?s)<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)(</｜DSML｜invoke>|$)"#)
                .expect("Valid regex pattern");

        Self {
            tool_call_complete_regex,
            invoke_complete_regex,
            parameter_complete_regex,
            partial_parameter_regex,
            invoke_regex,
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
        }
    }

    /// Parse DSML parameters from invoke content into a JSON string.
    ///
    /// Supports two formats:
    /// 1. Direct JSON: content starts with `{` — returned as-is
    /// 2. XML parameters: `<｜DSML｜parameter name="k" string="true|false">v</｜DSML｜parameter>`
    ///
    /// When `allow_partial` is true (streaming), also matches open parameter tags
    /// and strips trailing DSML fragments.
    fn parse_parameters_from_dsml(&self, invoke_content: &str, allow_partial: bool) -> String {
        let trimmed = invoke_content.trim();

        // Direct JSON path
        if trimmed.starts_with('{') {
            if allow_partial {
                let mut result = trimmed.to_string();
                for fragment in DSML_TRAILING_FRAGMENTS {
                    if let Some(stripped) = result.strip_suffix(fragment) {
                        result = stripped.to_string();
                        break;
                    }
                }
                return result;
            } else if trimmed.ends_with('}') {
                return trimmed.to_string();
            }
        }

        // XML parameter path
        let mut params = serde_json::Map::new();

        for cap in self.parameter_complete_regex.captures_iter(invoke_content) {
            let name = cap.get(1).map_or("", |m| m.as_str());
            let is_string = cap.get(2).map_or("true", |m| m.as_str());
            let value = cap.get(3).map_or("", |m| m.as_str());

            let json_value = if is_string == "true" {
                Value::String(value.to_string())
            } else {
                serde_json::from_str(value.trim())
                    .unwrap_or_else(|_| Value::String(value.to_string()))
            };

            params.insert(name.to_string(), json_value);
        }

        // Partial parameter matching for streaming
        if allow_partial {
            if let Some(cap) = self.partial_parameter_regex.captures(invoke_content) {
                let name = cap.get(1).map_or("", |m| m.as_str());
                let is_string = cap.get(2).map_or("true", |m| m.as_str());
                let raw_value = cap.get(3).map_or("", |m| m.as_str()).trim();

                // Strip trailing DSML fragments from partial value
                let mut value = raw_value.to_string();
                for fragment in DSML_TRAILING_FRAGMENTS {
                    if let Some(stripped) = value.strip_suffix(fragment) {
                        value = stripped.to_string();
                        break;
                    }
                }
                let value = value.trim();

                // Only add if we have actual content and this param isn't already complete
                if !value.is_empty() && !params.contains_key(name) {
                    let json_value = if is_string == "true" {
                        Value::String(value.to_string())
                    } else {
                        serde_json::from_str(value)
                            .unwrap_or_else(|_| Value::String(value.to_string()))
                    };
                    params.insert(name.to_string(), json_value);
                }
            }
        }

        serde_json::to_string(&Value::Object(params)).unwrap_or_else(|_| "{}".to_string())
    }

    /// Parse a single complete invoke block into a ToolCall
    fn parse_invoke(&self, name: &str, content: &str) -> ToolCall {
        let arguments = self.parse_parameters_from_dsml(content, false);

        ToolCall {
            function: FunctionCall {
                name: name.trim().to_string(),
                arguments,
            },
        }
    }
}

impl Default for DeepSeek32Parser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for DeepSeek32Parser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        let idx = text
            .find("<｜DSML｜function_calls>")
            .ok_or_else(|| ParserError::ParsingFailed("DSML marker not found".to_string()))?;
        let normal_text = text[..idx].trim_end().to_string();

        let mut tools = Vec::new();

        for fc_cap in self.tool_call_complete_regex.captures_iter(text) {
            let fc_content = fc_cap.get(1).map_or("", |m| m.as_str());

            for inv_cap in self.invoke_complete_regex.captures_iter(fc_content) {
                let func_name = inv_cap.get(1).map_or("", |m| m.as_str());
                let invoke_content = inv_cap.get(2).map_or("", |m| m.as_str());

                tools.push(self.parse_invoke(func_name, invoke_content));
            }
        }

        if tools.is_empty() {
            return Ok((text.to_string(), vec![]));
        }

        Ok((normal_text, tools))
    }

    async fn parse_incremental(
        &mut self,
        _chunk: &str,
        _tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        // Placeholder — implemented in Task 2
        Ok(StreamingParseResult::default())
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<｜DSML｜function_calls>")
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
