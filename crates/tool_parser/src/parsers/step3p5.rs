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

/// Step-3.5 XML format parser for tool calls.
///
/// Ported from vLLM's `step3p5_tool_parser.py` (`Step3p5ToolParser` /
/// `StreamingXMLToolCallParser`). Step-3.5 wraps each call in `<tool_call>` …
/// `</tool_call>`, declares the function with `<function=name>` … `</function>`,
/// and emits each argument as `<parameter=key>value</parameter>`:
///
/// ```text
/// <tool_call>
/// <function=name>
/// <parameter=key>value</parameter>
/// </function>
/// </tool_call>
/// ```
///
/// The vLLM parser coerces each parameter value by its declared JSON-schema type
/// (`_get_param_type` / `_convert_param_value`): `string`-family types stay verbatim,
/// `int`/`num`/`float` parse to numbers, `bool` to booleans, and a literal `null`
/// becomes JSON null. When no tool schema is available the value is inferred
/// JSON-first, with a string fallback.
pub struct Step3p5Parser {
    /// Regex for extracting complete `<tool_call>…</tool_call>` blocks.
    extractor: Regex,
    /// Regex for extracting the `<function=name>` declaration.
    xml_function_pattern: Regex,
    /// Regex for extracting `<parameter=key>value</parameter>` pairs.
    xml_param_pattern: Regex,

    /// Buffer for accumulating incomplete patterns across chunks.
    buffer: String,

    /// Stores complete tool call info (name and arguments) for each tool being parsed.
    prev_tool_call_arr: Vec<Value>,

    /// Index of currently streaming tool call (-1 means no active tool).
    current_tool_id: i32,

    /// Flag for whether current tool's name has been sent to the client.
    current_tool_name_sent: bool,

    /// Tracks raw JSON string content streamed to client for each tool's arguments.
    streamed_args_for_tool: Vec<String>,

    /// Token configuration (exactly matching the vLLM source).
    tool_call_start_token: &'static str,
    tool_call_end_token: &'static str,

    /// XML format streaming state.
    in_tool_call: bool,
    current_function_name: String,
    current_parameters: serde_json::Map<String, Value>,
}

/// Coerce a raw parameter value by its declared JSON-schema type, mirroring vLLM's
/// `_convert_param_value`. A literal `null` always maps to JSON null. Returns `None`
/// when no declared type is known so the caller can fall back to inference.
fn coerce_value(raw: &str, declared_type: Option<&str>) -> Option<Value> {
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("null") {
        return Some(Value::Null);
    }

    let declared = declared_type?;
    let ty = declared.trim().to_lowercase();

    if ty == "string"
        || ty == "str"
        || ty == "text"
        || ty == "varchar"
        || ty == "char"
        || ty == "enum"
    {
        return Some(Value::String(raw.to_string()));
    }
    if ty.starts_with("int")
        || ty.starts_with("uint")
        || ty.starts_with("long")
        || ty.starts_with("short")
        || ty.starts_with("unsigned")
    {
        if let Ok(n) = trimmed.parse::<i64>() {
            return Some(Value::Number(n.into()));
        }
        // vLLM degrades to string on parse failure.
        return Some(Value::String(raw.to_string()));
    }
    if ty.starts_with("num") || ty.starts_with("float") {
        if let Ok(f) = trimmed.parse::<f64>() {
            if f.fract() == 0.0 && f.is_finite() && f.abs() < (i64::MAX as f64) {
                return Some(Value::Number((f as i64).into()));
            }
            if let Some(n) = serde_json::Number::from_f64(f) {
                return Some(Value::Number(n));
            }
        }
        return Some(Value::String(raw.to_string()));
    }
    if ty == "boolean" || ty == "bool" || ty == "binary" {
        return Some(Value::Bool(trimmed.eq_ignore_ascii_case("true")));
    }
    if ty == "object"
        || ty == "array"
        || ty == "arr"
        || ty == "sequence"
        || ty.starts_with("dict")
        || ty.starts_with("list")
    {
        if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
            return Some(v);
        }
        return Some(Value::String(raw.to_string()));
    }

    // Unknown declared type: treat as string (vLLM's `repair_param_type` fallback).
    Some(Value::String(raw.to_string()))
}

/// Infer a parameter value when no schema type is known. Tries JSON first
/// (numbers, booleans, null, objects, arrays), then Python-style literals, and
/// finally falls back to the trimmed string.
fn infer_value(raw: &str) -> Value {
    let trimmed = raw.trim();

    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return v;
    }

    match trimmed {
        "True" => return Value::Bool(true),
        "False" => return Value::Bool(false),
        "None" | "null" => return Value::Null,
        _ => {}
    }

    Value::String(trimmed.to_string())
}

impl Step3p5Parser {
    /// Create a new Step-3.5 XML parser.
    #[expect(
        clippy::expect_used,
        reason = "regex patterns are compile-time string literals"
    )]
    pub fn new() -> Self {
        let extractor = Regex::new(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>")
            .expect("Valid tool_call regex pattern");
        let xml_function_pattern =
            Regex::new(r"<function=([^>]+)>").expect("Valid XML function pattern");
        let xml_param_pattern = Regex::new(r"(?s)<parameter=([^>]+)>(.*?)</parameter>")
            .expect("Valid XML parameter pattern");

        Self {
            extractor,
            xml_function_pattern,
            xml_param_pattern,
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
            tool_call_start_token: "<tool_call>",
            tool_call_end_token: "</tool_call>",
            in_tool_call: false,
            current_function_name: String::new(),
            current_parameters: serde_json::Map::new(),
        }
    }

    /// Parse a single `<function=…>…</function>` block into a [`ToolCall`],
    /// coercing each parameter by its declared schema type when available.
    fn parse_xml_format(&self, content: &str, tools: &[Tool]) -> ParserResult<Option<ToolCall>> {
        let function_captures = self
            .xml_function_pattern
            .captures(content)
            .ok_or_else(|| ParserError::ParsingFailed("No function name found".to_string()))?;

        let function_name = function_captures
            .get(1)
            .ok_or_else(|| ParserError::ParsingFailed("Function name capture failed".to_string()))?
            .as_str()
            .trim()
            .to_string();

        if function_name.is_empty() {
            return Ok(None);
        }

        let param_types = helpers::param_types_for_function(tools, &function_name);

        let mut parameters = serde_json::Map::new();
        for cap in self.xml_param_pattern.captures_iter(content) {
            if let (Some(key_match), Some(value_match)) = (cap.get(1), cap.get(2)) {
                let key = key_match.as_str().trim().to_string();
                let value = value_match.as_str();
                let declared = param_types.get(&key).map(String::as_str);
                let json_value =
                    coerce_value(value, declared).unwrap_or_else(|| infer_value(value));
                parameters.insert(key, json_value);
            }
        }

        let arguments = serde_json::to_string(&parameters)
            .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

        Ok(Some(ToolCall {
            function: FunctionCall {
                name: function_name,
                arguments,
            },
        }))
    }

    /// Extract all complete tool calls from `text` (shared by complete parsing
    /// with and without schema information).
    fn extract_all(&self, text: &str, tools: &[Tool]) -> (String, Vec<ToolCall>) {
        let Some(idx) = text.find(self.tool_call_start_token) else {
            return (text.to_string(), vec![]);
        };
        let normal_text = text[..idx].to_string();

        let mut calls = Vec::new();
        for captures in self.extractor.captures_iter(text) {
            if let Some(content_str) = captures.get(1) {
                let content = content_str.as_str().trim();
                match self.parse_xml_format(content, tools) {
                    Ok(Some(tool)) => calls.push(tool),
                    Ok(None) => continue,
                    Err(e) => {
                        tracing::warn!("Failed to parse Step-3.5 tool call: {:?}", e);
                        continue;
                    }
                }
            }
        }

        if calls.is_empty() {
            return (text.to_string(), vec![]);
        }

        (normal_text, calls)
    }

    /// Parse and stream complete parameters from the buffer, coercing by the
    /// supplied schema types. Returns the tool call argument deltas to emit.
    fn parse_and_stream_parameters(
        &mut self,
        param_types: &std::collections::HashMap<String, String>,
    ) -> Vec<ToolCallItem> {
        let mut calls: Vec<ToolCallItem> = vec![];

        let mut new_params = serde_json::Map::new();
        for cap in self.xml_param_pattern.captures_iter(&self.buffer) {
            if let (Some(key_match), Some(value_match)) = (cap.get(1), cap.get(2)) {
                let key = key_match.as_str().trim().to_string();
                let value = value_match.as_str();
                let declared = param_types.get(&key).map(String::as_str);
                let json_value =
                    coerce_value(value, declared).unwrap_or_else(|| infer_value(value));
                new_params.insert(key, json_value);
            }
        }

        if new_params != self.current_parameters {
            let current_args = &mut self.streamed_args_for_tool[self.current_tool_id as usize];

            if self.current_parameters.is_empty() {
                let mut items = Vec::new();
                for (key, value) in &new_params {
                    let key_json =
                        serde_json::to_string(key).unwrap_or_else(|_| format!("\"{key}\""));
                    let value_json = serde_json::to_string(value).unwrap_or_default();
                    items.push(format!("{key_json}: {value_json}"));
                }
                let json_fragment = format!("{{{}", items.join(", "));

                calls.push(ToolCallItem {
                    tool_index: self.current_tool_id as usize,
                    name: None,
                    parameters: json_fragment.clone(),
                });
                *current_args = json_fragment;
            } else {
                let new_keys: Vec<_> = new_params
                    .keys()
                    .filter(|k| !self.current_parameters.contains_key(*k))
                    .collect();

                if !new_keys.is_empty() {
                    let mut continuation_parts = Vec::new();
                    for key in new_keys {
                        if let Some(value) = new_params.get(key) {
                            let key_json =
                                serde_json::to_string(key).unwrap_or_else(|_| format!("\"{key}\""));
                            let value_json = serde_json::to_string(value).unwrap_or_default();
                            continuation_parts.push(format!("{key_json}: {value_json}"));
                        }
                    }

                    let json_fragment = format!(", {}", continuation_parts.join(", "));

                    calls.push(ToolCallItem {
                        tool_index: self.current_tool_id as usize,
                        name: None,
                        parameters: json_fragment.clone(),
                    });
                    current_args.push_str(&json_fragment);
                }
            }

            self.current_parameters.clone_from(&new_params);
            if let Some(tool_obj) =
                self.prev_tool_call_arr[self.current_tool_id as usize].as_object_mut()
            {
                tool_obj.insert("arguments".to_string(), Value::Object(new_params));
            }
        }

        calls
    }

    /// Reset streaming state for the next tool call.
    fn reset_streaming_state(&mut self) {
        self.in_tool_call = false;
        self.current_tool_name_sent = false;
        self.current_function_name.clear();
        self.current_parameters.clear();
    }
}

impl Default for Step3p5Parser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for Step3p5Parser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        Ok(self.extract_all(text, &[]))
    }

    async fn parse_complete_with_tools(
        &self,
        text: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        Ok(self.extract_all(text, tools))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);

        let mut normal_text = String::new();
        let mut calls: Vec<ToolCallItem> = vec![];

        let tool_indices = helpers::get_tool_indices(tools);

        loop {
            // Not in a tool call and no start token: flush as normal text.
            if !self.in_tool_call && !self.buffer.contains(self.tool_call_start_token) {
                if helpers::ends_with_partial_token(&self.buffer, self.tool_call_start_token)
                    .is_none()
                {
                    normal_text.push_str(&self.buffer);
                    self.buffer.clear();
                }
                break;
            }

            // Look for tool call start.
            if !self.in_tool_call {
                if let Some(s) = self.buffer.find(self.tool_call_start_token) {
                    normal_text.push_str(&self.buffer[..s]);
                    self.buffer = self.buffer[s + self.tool_call_start_token.len()..].to_string();
                    self.in_tool_call = true;
                    self.current_tool_name_sent = false;
                    self.current_function_name.clear();
                    self.current_parameters.clear();
                    continue;
                }
                break;
            }

            // Parse function name if not sent yet.
            if !self.current_tool_name_sent {
                if let Some(captures) = self.xml_function_pattern.captures(&self.buffer) {
                    if let Some(name_match) = captures.get(1) {
                        let function_name = name_match.as_str().trim().to_string();

                        if tool_indices.contains_key(&function_name) {
                            self.current_function_name.clone_from(&function_name);
                            self.current_tool_name_sent = true;

                            if self.current_tool_id == -1 {
                                self.current_tool_id = 0;
                            }

                            helpers::ensure_capacity(
                                self.current_tool_id,
                                &mut self.prev_tool_call_arr,
                                &mut self.streamed_args_for_tool,
                            );

                            self.prev_tool_call_arr[self.current_tool_id as usize] = serde_json::json!({
                                "name": function_name,
                                "arguments": {}
                            });

                            calls.push(ToolCallItem {
                                tool_index: self.current_tool_id as usize,
                                name: Some(function_name),
                                parameters: String::new(),
                            });

                            // Safe: group 0 is the entire match.
                            self.buffer =
                                self.buffer[captures.get(0).map_or(0, |m| m.end())..].to_string();
                            continue;
                        }

                        tracing::warn!("Invalid function name: {}", function_name);
                        self.reset_streaming_state();
                        normal_text.push_str(&self.buffer);
                        self.buffer.clear();
                        break;
                    }
                }
                // Function name not complete yet.
                break;
            }

            // Parse parameters (only complete ones).
            if self.current_tool_name_sent {
                let param_types =
                    helpers::param_types_for_function(tools, &self.current_function_name);
                let param_calls = self.parse_and_stream_parameters(&param_types);
                calls.extend(param_calls);

                if let Some(end_pos) = self.buffer.find(self.tool_call_end_token) {
                    let current_args = &self.streamed_args_for_tool[self.current_tool_id as usize];
                    if !current_args.is_empty() {
                        let open_braces = current_args.matches('{').count();
                        let close_braces = current_args.matches('}').count();
                        if open_braces > close_braces {
                            calls.push(ToolCallItem {
                                tool_index: self.current_tool_id as usize,
                                name: None,
                                parameters: "}".to_string(),
                            });
                            self.streamed_args_for_tool[self.current_tool_id as usize].push('}');
                        }
                    }

                    self.buffer =
                        self.buffer[end_pos + self.tool_call_end_token.len()..].to_string();
                    self.reset_streaming_state();
                    self.current_tool_id += 1;
                    continue;
                }
                // Tool call not complete yet.
                break;
            }

            break;
        }

        Ok(StreamingParseResult { normal_text, calls })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(self.tool_call_start_token)
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn reset(&mut self) {
        helpers::reset_parser_state(
            &mut self.buffer,
            &mut self.prev_tool_call_arr,
            &mut self.current_tool_id,
            &mut self.current_tool_name_sent,
            &mut self.streamed_args_for_tool,
        );
        self.reset_streaming_state();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coerce_value_string_keeps_numeric_text() {
        assert_eq!(
            coerce_value("42", Some("string")),
            Some(Value::String("42".to_string()))
        );
    }

    #[test]
    fn test_coerce_value_integer() {
        assert_eq!(coerce_value("42", Some("integer")), Some(Value::from(42)));
        assert_eq!(coerce_value(" 7 ", Some("int")), Some(Value::from(7)));
    }

    #[test]
    fn test_coerce_value_number() {
        assert_eq!(coerce_value("1.5", Some("number")), Some(Value::from(1.5)));
        // Whole floats degrade to integers, matching vLLM.
        assert_eq!(coerce_value("3.0", Some("number")), Some(Value::from(3)));
    }

    #[test]
    fn test_coerce_value_boolean() {
        assert_eq!(
            coerce_value("true", Some("boolean")),
            Some(Value::Bool(true))
        );
        assert_eq!(
            coerce_value("False", Some("bool")),
            Some(Value::Bool(false))
        );
    }

    #[test]
    fn test_coerce_value_null() {
        assert_eq!(coerce_value("null", Some("string")), Some(Value::Null));
        assert_eq!(coerce_value("NULL", None), Some(Value::Null));
    }

    #[test]
    fn test_coerce_value_object() {
        assert_eq!(
            coerce_value(r#"{"a": 1}"#, Some("object")),
            Some(serde_json::json!({"a": 1}))
        );
    }

    #[test]
    fn test_infer_value_fallbacks() {
        assert_eq!(infer_value("42"), Value::from(42));
        assert_eq!(infer_value("true"), Value::Bool(true));
        assert_eq!(infer_value("None"), Value::Null);
        assert_eq!(infer_value("hello"), Value::String("hello".to_string()));
    }
}
