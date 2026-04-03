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

/// DSML token fragments for stripping partial closing tags during streaming.
/// Applied in reverse order using character-level right-trimming, following
/// SGLang's exact fragment definitions.
const DSML_PARAM_END_FRAGMENTS: &[&str] = &["</", "｜DSML｜", "parameter"];
const DSML_INVOKE_END_FRAGMENTS: &[&str] = &["</", "｜DSML｜", "inv", "oke"];

/// Strip trailing DSML fragment characters from a string.
/// Iterates fragments in reverse, stripping any trailing characters
/// that appear in each fragment (mimics Python's `str.rstrip`).
fn strip_dsml_trailing(s: &str, fragments: &[&str]) -> String {
    let mut result = s.to_string();
    for fragment in fragments.iter().rev() {
        result = result
            .trim_end_matches(|c: char| fragment.contains(c))
            .to_string();
    }
    result
}

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
                return strip_dsml_trailing(trimmed, DSML_INVOKE_END_FRAGMENTS);
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
        // Following SGLang: strip DSML fragments from remaining content BEFORE
        // running the partial regex, so the regex captures a clean value.
        if allow_partial {
            // Find where the last complete parameter match ended
            let last_match_end = self
                .parameter_complete_regex
                .find_iter(invoke_content)
                .last()
                .map(|m| m.end())
                .unwrap_or(0);

            let remaining = &invoke_content[last_match_end..];
            let cleaned = strip_dsml_trailing(remaining, DSML_PARAM_END_FRAGMENTS);

            if let Some(cap) = self.partial_parameter_regex.captures(&cleaned) {
                let name = cap.get(1).map_or("", |m| m.as_str());
                let is_string = cap.get(2).map_or("true", |m| m.as_str());
                let value = cap.get(3).map_or("", |m| m.as_str()).trim();

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
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);
        let current_text = self.buffer.clone();

        // Check for DSML markers or partial DSML prefixes
        let has_dsml =
            self.has_tool_markers(&current_text) || current_text.contains("<｜DSML｜invoke");
        let has_partial_prefix = current_text.ends_with('<')
            || current_text.ends_with("<｜")
            || current_text.ends_with("</")
            || current_text.ends_with("</｜");

        if !has_dsml && !has_partial_prefix {
            let mut normal_text = std::mem::take(&mut self.buffer);
            for end_token in [
                "</｜DSML｜function_calls>",
                "</｜DSML｜invoke>",
                "</｜DSML｜parameter>",
                "<｜end▁of▁sentence｜>",
            ] {
                normal_text = normal_text.replace(end_token, "");
            }
            return Ok(StreamingParseResult {
                normal_text,
                calls: vec![],
            });
        }

        // If we have partial prefix but no actual DSML content, buffer and wait
        if !has_dsml && has_partial_prefix {
            return Ok(StreamingParseResult::default());
        }

        let tool_indices = helpers::get_tool_indices(tools);
        let mut all_calls: Vec<ToolCallItem> = Vec::new();

        // Process invoke blocks in a loop (handles multiple complete invokes in buffer)
        loop {
            let buf_snapshot = self.buffer.clone();
            let invoke_match = self.invoke_regex.captures(&buf_snapshot);

            let captures = match invoke_match {
                Some(c) => c,
                None => break,
            };

            let func_name = captures
                .get(1)
                .map_or(String::new(), |m| m.as_str().trim().to_string());
            let invoke_content = captures
                .get(2)
                .map_or(String::new(), |m| m.as_str().to_string());
            let is_complete = captures
                .get(3)
                .is_some_and(|m| m.as_str().contains("</｜DSML｜invoke>"));
            let match_end = captures.get(0).map(|m| m.end());
            drop(captures);

            // Skip if tool name is not in provided tools list
            if !func_name.is_empty() && !tool_indices.contains_key(func_name.as_str()) {
                tracing::debug!("Invalid tool name '{}' - skipping", func_name);
                if is_complete {
                    // Complete invalid invoke — advance buffer past it and try next
                    if let Some(end) = match_end {
                        self.buffer = self.buffer[end..].to_string();
                    }
                    continue;
                } else {
                    // Incomplete invalid invoke — reset state and wait for more data
                    helpers::reset_current_tool_state(
                        &mut self.buffer,
                        &mut self.current_tool_name_sent,
                        &mut self.streamed_args_for_tool,
                        &self.prev_tool_call_arr,
                    );
                    return Ok(StreamingParseResult::default());
                }
            }

            // Initialize state on first tool
            if self.current_tool_id == -1 {
                self.current_tool_id = 0;
                self.prev_tool_call_arr = Vec::new();
                self.streamed_args_for_tool = vec![String::new()];
            }

            helpers::ensure_capacity(
                self.current_tool_id,
                &mut self.prev_tool_call_arr,
                &mut self.streamed_args_for_tool,
            );

            // Emit tool name if not sent
            if !self.current_tool_name_sent && !func_name.is_empty() {
                all_calls.push(ToolCallItem {
                    tool_index: self.current_tool_id as usize,
                    name: Some(func_name.to_string()),
                    parameters: String::new(),
                });
                self.current_tool_name_sent = true;

                let tool_id = self.current_tool_id as usize;
                if self.prev_tool_call_arr.len() <= tool_id {
                    self.prev_tool_call_arr
                        .resize_with(tool_id + 1, || Value::Null);
                }
                self.prev_tool_call_arr[tool_id] = serde_json::json!({
                    "name": func_name,
                    "arguments": {},
                });
            }

            // Parse current arguments (partial or complete)
            let current_args = self.parse_parameters_from_dsml(&invoke_content, !is_complete);
            let tool_id = self.current_tool_id as usize;

            // Compute diff against what we've already sent
            let sent_len = self
                .streamed_args_for_tool
                .get(tool_id)
                .map(|s| s.len())
                .unwrap_or(0);

            let prev_args = if tool_id < self.prev_tool_call_arr.len() {
                self.prev_tool_call_arr[tool_id]
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            } else {
                None
            };

            let argument_diff = if is_complete {
                if sent_len < current_args.len() {
                    Some(current_args[sent_len..].to_string())
                } else {
                    Some(String::new())
                }
            } else if let Some(prev) = &prev_args {
                if current_args == *prev {
                    None
                } else {
                    let prefix = helpers::find_common_prefix(prev, &current_args);
                    if prefix.len() > sent_len {
                        Some(prefix[sent_len..].to_string())
                    } else {
                        None
                    }
                }
            } else {
                None
            };

            if let Some(diff) = argument_diff {
                if !diff.is_empty() {
                    if tool_id < self.streamed_args_for_tool.len() {
                        self.streamed_args_for_tool[tool_id].push_str(&diff);
                    }
                    all_calls.push(ToolCallItem {
                        tool_index: tool_id,
                        name: None,
                        parameters: diff,
                    });
                }
            }

            // Update prev state
            if tool_id < self.prev_tool_call_arr.len() {
                self.prev_tool_call_arr[tool_id] = serde_json::json!({
                    "name": func_name,
                    "arguments": current_args,
                });
            }

            // If invoke is complete, advance to next tool
            if is_complete {
                if let Some(end) = match_end {
                    self.buffer = self.buffer[end..].to_string();
                } else {
                    self.buffer.clear();
                }
                self.current_tool_id += 1;
                self.current_tool_name_sent = false;
                continue;
            } else {
                break;
            }
        }

        Ok(StreamingParseResult {
            normal_text: String::new(),
            calls: all_calls,
        })
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
