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

/// DeepSeek V3.1 format parser for tool calls
///
/// Handles the DeepSeek V3.1 format:
/// `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{name}<｜tool▁sep｜>{json_args}<｜tool▁call▁end｜><｜tool▁calls▁end｜>`
///
/// Differences from V3:
/// - No `function` type prefix before `<｜tool▁sep｜>`
/// - No markdown code block wrapping around JSON arguments
/// - Function name directly after `<｜tool▁call▁begin｜>`
/// - Raw JSON arguments directly after `<｜tool▁sep｜>`
///
/// Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3.1
pub struct DeepSeek31Parser {
    /// Regex for extracting complete tool call blocks
    tool_call_extractor: Regex,
    /// Regex for extracting function name and arguments from a complete block
    func_detail_extractor: Regex,
    /// Regex for matching partial tool calls during streaming (used in Task 3)
    #[expect(dead_code, reason = "used by parse_incremental, implemented in Task 3")]
    partial_tool_call_regex: Regex,
    /// Regex for removing completed tool calls from buffer (used in Task 3)
    #[expect(dead_code, reason = "used by parse_incremental, implemented in Task 3")]
    tool_call_end_pattern: Regex,

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

impl DeepSeek31Parser {
    /// Create a new DeepSeek V3.1 parser
    #[expect(
        clippy::expect_used,
        reason = "regex patterns are compile-time string literals"
    )]
    pub fn new() -> Self {
        let tool_call_extractor =
            Regex::new(r"(?s)<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>")
                .expect("Valid regex pattern");

        let func_detail_extractor =
            Regex::new(r"(?s)<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)<｜tool▁call▁end｜>")
                .expect("Valid regex pattern");

        let partial_tool_call_regex =
            Regex::new(r"(?s)<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)")
                .expect("Valid regex pattern");

        let tool_call_end_pattern =
            Regex::new(r"(?s)<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>")
                .expect("Valid regex pattern");

        Self {
            tool_call_extractor,
            func_detail_extractor,
            partial_tool_call_regex,
            tool_call_end_pattern,
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
        }
    }

    /// Parse a single complete tool call block
    fn parse_tool_call(&self, block: &str) -> ParserResult<ToolCall> {
        let captures = self.func_detail_extractor.captures(block).ok_or_else(|| {
            ParserError::ParsingFailed("Failed to match tool call pattern".to_string())
        })?;

        let func_name = captures.get(1).map_or("", |m| m.as_str()).trim();
        if func_name.is_empty() {
            return Err(ParserError::ParsingFailed(
                "Empty function name".to_string(),
            ));
        }

        let json_args = captures.get(2).map_or("{}", |m| m.as_str()).trim();

        let value = serde_json::from_str::<Value>(json_args)
            .map_err(|e| ParserError::ParsingFailed(format!("Invalid JSON: {e}")))?;

        let args = if value.is_object() {
            value
        } else {
            serde_json::json!({ "value": value })
        };

        let arguments =
            serde_json::to_string(&args).map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

        Ok(ToolCall {
            function: FunctionCall {
                name: func_name.to_string(),
                arguments,
            },
        })
    }
}

impl Default for DeepSeek31Parser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for DeepSeek31Parser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        let idx = text
            .find("<｜tool▁calls▁begin｜>")
            .ok_or_else(|| ParserError::ParsingFailed("tool call marker not found".to_string()))?;
        let normal_text = text[..idx].to_string();

        let mut tools = Vec::new();
        for mat in self.tool_call_extractor.find_iter(text) {
            match self.parse_tool_call(mat.as_str()) {
                Ok(tool) => tools.push(tool),
                Err(e) => {
                    tracing::debug!("Failed to parse tool call: {}", e);
                    continue;
                }
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
        // Placeholder — implemented in Task 3
        Ok(StreamingParseResult::default())
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<｜tool▁calls▁begin｜>")
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
