use std::{collections::HashMap, fmt::Write as FmtWrite};

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

/// MiniMax M2 format parser for tool calls
///
/// Handles the MiniMax M2 specific format:
/// `<minimax:tool_call><invoke name="func"><parameter name="key">value</parameter></invoke></minimax:tool_call>`
///
/// Features:
/// - Namespaced XML tags (`minimax:tool_call`)
/// - Function wrapped in `<invoke name="...">` tags
/// - Parameters as `<parameter name="key">value</parameter>`
/// - Incremental JSON streaming for parameters
///
/// Reference: https://huggingface.co/MiniMaxAI/MiniMax-M2?chat_template=default
pub struct MinimaxM2Parser {
    // Regex patterns
    tool_call_extractor: Regex,
    invoke_extractor: Regex,
    param_extractor: Regex,

    // Streaming state
    buffer: String,
    prev_tool_call_arr: Vec<Value>,
    current_tool_id: i32,
    streamed_args_for_tool: Vec<String>,
    current_function_name: String,
    current_parameters: HashMap<String, Value>,
    in_tool_call: bool,
    function_name_sent: bool,
    waiting_for_tool_call_end: bool,

    // Token configuration
    tool_call_start_token: &'static str,
    tool_call_end_token: &'static str,
    invoke_end_token: &'static str,
}

impl MinimaxM2Parser {
    /// Parse a value from string with consistent logic
    #[inline]
    fn parse_value(text: &str) -> Value {
        // Try parsing as common literals first
        match text {
            "true" | "True" => return Value::Bool(true),
            "false" | "False" => return Value::Bool(false),
            "null" | "None" => return Value::Null,
            _ => {}
        }

        // Try parsing as number
        if let Ok(num) = text.parse::<i64>() {
            return Value::Number(num.into());
        }

        if let Ok(num) = text.parse::<f64>() {
            if let Some(n) = serde_json::Number::from_f64(num) {
                return Value::Number(n);
            }
        }

        // Default to string
        Value::String(text.to_string())
    }

    /// Create a new MiniMax M2 parser
    #[expect(
        clippy::expect_used,
        reason = "regex patterns are compile-time string literals"
    )]
    pub fn new() -> Self {
        // Use (?s) flag for DOTALL mode to handle newlines
        let tool_call_pattern = r"(?s)<minimax:tool_call>.*?</minimax:tool_call>";
        let tool_call_extractor = Regex::new(tool_call_pattern).expect("Valid regex pattern");

        let invoke_pattern = r#"(?s)<invoke\s+name="([^"]+)">(.*?)</invoke>"#;
        let invoke_extractor = Regex::new(invoke_pattern).expect("Valid regex pattern");

        let param_pattern = r#"(?s)<parameter\s+name="([^"]+)">(.*?)</parameter>"#;
        let param_extractor = Regex::new(param_pattern).expect("Valid regex pattern");

        Self {
            tool_call_extractor,
            invoke_extractor,
            param_extractor,
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            streamed_args_for_tool: Vec::new(),
            current_function_name: String::new(),
            current_parameters: HashMap::new(),
            in_tool_call: false,
            function_name_sent: false,
            waiting_for_tool_call_end: false,
            tool_call_start_token: "<minimax:tool_call>",
            tool_call_end_token: "</minimax:tool_call>",
            invoke_end_token: "</invoke>",
        }
    }

    /// Extract the set of JSON-schema types for a given parameter of a given
    /// function from the tool list. Returns empty slice if not found — caller
    /// should fall back to the schema-unaware `parse_value`.
    fn schema_types_for_param(
        tools: &[Tool],
        func_name: &str,
        param_name: &str,
    ) -> Vec<String> {
        let Some(tool) = tools.iter().find(|t| t.function.name == func_name) else {
            return Vec::new();
        };

        let params: &Value = &tool.function.parameters;
        let Some(properties) = params.get("properties").and_then(Value::as_object) else {
            return Vec::new();
        };
        let Some(schema) = properties.get(param_name) else {
            return Vec::new();
        };

        Self::extract_types_from_schema(schema)
    }

    /// Collect every type name that a JSON-schema definition can represent.
    /// Handles `type`, `type: [..]`, `enum`, `anyOf`, `oneOf`, `allOf`.
    fn extract_types_from_schema(schema: &Value) -> Vec<String> {
        let mut types = std::collections::BTreeSet::new();
        let Some(obj) = schema.as_object() else {
            return vec!["string".to_string()];
        };

        if let Some(t) = obj.get("type") {
            match t {
                Value::String(s) => {
                    types.insert(s.clone());
                }
                Value::Array(arr) => {
                    for v in arr {
                        if let Some(s) = v.as_str() {
                            types.insert(s.to_string());
                        }
                    }
                }
                _ => {}
            }
        }

        if let Some(Value::Array(enum_vals)) = obj.get("enum") {
            for v in enum_vals {
                let t = match v {
                    Value::Null => "null",
                    Value::Bool(_) => "boolean",
                    Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
                    Value::Number(_) => "number",
                    Value::String(_) => "string",
                    Value::Array(_) => "array",
                    Value::Object(_) => "object",
                };
                types.insert(t.to_string());
            }
        }

        for key in ["anyOf", "oneOf", "allOf"] {
            if let Some(Value::Array(arr)) = obj.get(key) {
                for sub in arr {
                    for t in Self::extract_types_from_schema(sub) {
                        types.insert(t);
                    }
                }
            }
        }

        if types.is_empty() {
            vec!["string".to_string()]
        } else {
            types.into_iter().collect()
        }
    }

    /// Convert a parameter's raw string value to the JSON type declared in the
    /// tool schema. Mirrors sglang's `_convert_param_value_with_types` so
    /// schema-aware parsing behaves identically across HTTP and gRPC paths.
    fn convert_with_schema(value: &str, param_types: &[String]) -> Value {
        let normalized: Vec<String> =
            param_types.iter().map(|t| t.to_lowercase()).collect();

        // Only coerce "null"/"none"/"nil" to JSON null when the schema
        // actually accepts null.  Previously this was an unconditional
        // early return, which broke cases like fc-dash
        // `bfcl/simple_javascript_43` where the tool description tells the
        // model `error` can be the STRING `'null'`; the model faithfully
        // emits `null` and the parser must not coerce it.
        let looks_null = value.eq_ignore_ascii_case("null")
            || value.eq_ignore_ascii_case("none")
            || value.eq_ignore_ascii_case("nil");
        if looks_null && normalized.iter().any(|t| t == "null") {
            return Value::Null;
        }

        // Priority: integer > number > boolean > object > array > string
        let priority: &[&str] = &[
            "integer", "int", "number", "float", "boolean", "bool", "object",
            "array", "string", "str", "text",
        ];

        for pt in priority {
            if !normalized.iter().any(|t| t == pt) {
                continue;
            }
            match *pt {
                "string" | "str" | "text" => return Value::String(value.to_string()),
                "integer" | "int" => {
                    if let Ok(n) = value.parse::<i64>() {
                        return Value::Number(n.into());
                    }
                }
                "number" | "float" => {
                    // Match Python `json.dumps` int-vs-float distinction:
                    // if the raw text parses as an integer, keep it as an
                    // integer so the serialized argument is `85` not `85.0`.
                    // sglang's Python parser does `int(x) or float(x)`, we
                    // mirror it.  JSON Schema `"type": "number"` accepts both.
                    if !value.contains(|c: char| matches!(c, '.' | 'e' | 'E')) {
                        if let Ok(n) = value.parse::<i64>() {
                            return Value::Number(n.into());
                        }
                    }
                    if let Ok(n) = value.parse::<f64>() {
                        if let Some(nn) = serde_json::Number::from_f64(n) {
                            return Value::Number(nn);
                        }
                    }
                }
                "boolean" | "bool" => {
                    let lo = value.to_lowercase();
                    match lo.trim() {
                        "true" | "1" | "yes" | "on" => return Value::Bool(true),
                        "false" | "0" | "no" | "off" => return Value::Bool(false),
                        _ => {}
                    }
                }
                "object" | "array" => {
                    if let Ok(v) = serde_json::from_str::<Value>(value) {
                        return v;
                    }
                }
                _ => {}
            }
        }

        // Final fallback: try json, else plain string
        if let Ok(v) = serde_json::from_str::<Value>(value) {
            return v;
        }
        Value::String(value.to_string())
    }

    /// Parse parameters from parameter tags. When `tools` is non-empty the
    /// tool schema is consulted to preserve declared types (e.g. keep
    /// numeric-looking strings as strings when the schema says string).
    fn parse_parameters(
        &self,
        params_text: &str,
        func_name: &str,
        tools: &[Tool],
    ) -> serde_json::Map<String, Value> {
        let mut parameters = serde_json::Map::new();

        for capture in self.param_extractor.captures_iter(params_text) {
            let key = capture.get(1).map_or("", |m| m.as_str()).trim();
            let value_str = capture.get(2).map_or("", |m| m.as_str());
            let decoded = Self::decode_xml_entities(value_str);

            let value = if tools.is_empty() {
                Self::parse_value(&decoded)
            } else {
                let types = Self::schema_types_for_param(tools, func_name, key);
                if types.is_empty() {
                    Self::parse_value(&decoded)
                } else {
                    Self::convert_with_schema(&decoded, &types)
                }
            };

            parameters.insert(key.to_string(), value);
        }

        parameters
    }

    /// Decode common XML entities
    fn decode_xml_entities(text: &str) -> String {
        text.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
    }

    /// Parse all `<invoke>` elements inside a single `<minimax:tool_call>` block.
    ///
    /// MiniMax wraps parallel tool calls in a **single** `<minimax:tool_call>`
    /// block with multiple `<invoke>` children, so we must iterate all matches.
    fn parse_tool_call_block(
        &self,
        block: &str,
        tools: &[Tool],
    ) -> ParserResult<Vec<ToolCall>> {
        let mut results = Vec::new();

        for captures in self.invoke_extractor.captures_iter(block) {
            let func_name = captures.get(1).map_or("", |m| m.as_str()).trim();
            let params_text = captures.get(2).map_or("", |m| m.as_str());
            let parameters = self.parse_parameters(params_text, func_name, tools);

            // Use Python-`json.dumps`-compatible spacing so
            // `tool_calls[].function.arguments` matches what sglang's HTTP
            // path produces (`{"a": 1, "b": 2}` not `{"a":1,"b":2}`).
            let arguments_str = helpers::python_json_to_string(&parameters)
                .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

            results.push(ToolCall {
                function: FunctionCall {
                    name: func_name.to_string(),
                    arguments: arguments_str,
                },
            });
        }

        Ok(results)
    }

    /// Parse all tool calls from text and return first valid position
    fn parse_tool_calls_from_text(
        &self,
        text: &str,
        tools: &[Tool],
    ) -> (Vec<ToolCall>, Option<usize>) {
        let mut results = Vec::new();
        let mut first_valid_pos = None;

        for mat in self.tool_call_extractor.find_iter(text) {
            match self.parse_tool_call_block(mat.as_str(), tools) {
                Ok(block_tools) => {
                    if !block_tools.is_empty() && first_valid_pos.is_none() {
                        first_valid_pos = Some(mat.start());
                    }
                    results.extend(block_tools);
                }
                Err(e) => {
                    tracing::debug!("Failed to parse tool call block: {}", e);
                    continue;
                }
            }
        }

        (results, first_valid_pos)
    }

    /// Parse and stream parameters incrementally
    fn parse_and_stream_parameters(&mut self, text: &str, tools: &[Tool]) -> Vec<ToolCallItem> {
        let mut calls = Vec::new();

        let func_name = self.current_function_name.clone();
        // Find all complete parameter patterns in the buffer
        let param_matches: Vec<_> = self
            .param_extractor
            .captures_iter(text)
            .map(|cap| {
                let name = cap.get(1).map_or("", |m| m.as_str()).trim().to_string();
                let value_str = cap.get(2).map_or("", |m| m.as_str());
                let decoded = Self::decode_xml_entities(value_str);

                // Schema-aware conversion when tools are available
                let value = if !tools.is_empty() {
                    let types = Self::schema_types_for_param(tools, &func_name, &name);
                    if types.is_empty() {
                        // Fallback: JSON parse for arrays/objects, else parse_value
                        if decoded.starts_with('{') || decoded.starts_with('[') {
                            serde_json::from_str::<Value>(&decoded)
                                .unwrap_or_else(|_| Self::parse_value(&decoded))
                        } else {
                            Self::parse_value(&decoded)
                        }
                    } else {
                        Self::convert_with_schema(&decoded, &types)
                    }
                } else if decoded.starts_with('{') || decoded.starts_with('[') {
                    serde_json::from_str::<Value>(&decoded)
                        .unwrap_or_else(|_| Self::parse_value(&decoded))
                } else {
                    Self::parse_value(&decoded)
                };

                (name, value)
            })
            .collect();

        // Build new parameters map
        let mut new_params = HashMap::new();
        for (name, value) in param_matches {
            new_params.insert(name, value);
        }

        // If we have new parameters that weren't in current_parameters, stream them
        if !new_params.is_empty() && new_params != self.current_parameters {
            let tool_id = self.current_tool_id as usize;

            // Ensure we have enough capacity
            while self.streamed_args_for_tool.len() <= tool_id {
                self.streamed_args_for_tool.push(String::new());
            }

            // Build incremental JSON with single allocation
            if self.current_parameters.is_empty() {
                // First parameters - start JSON object but don't close it
                let mut json_fragment = String::with_capacity(256);
                json_fragment.push('{');

                let mut first = true;
                for (key, value) in &new_params {
                    if !first {
                        json_fragment.push_str(", ");
                    }
                    // serde_json::to_string for String/Value is infallible; write! to String is infallible
                    let key_json = serde_json::to_string(key).unwrap_or_default();
                    let value_json = serde_json::to_string(value).unwrap_or_default();
                    let _ = write!(&mut json_fragment, "{key_json}: {value_json}");
                    first = false;
                }

                calls.push(ToolCallItem {
                    tool_index: tool_id,
                    name: None,
                    parameters: json_fragment.clone(),
                });

                self.streamed_args_for_tool[tool_id] = json_fragment;
            } else {
                // Additional parameters - add them incrementally
                let new_keys: Vec<_> = new_params
                    .keys()
                    .filter(|k| !self.current_parameters.contains_key(*k))
                    .collect();

                if !new_keys.is_empty() {
                    let mut json_fragment = String::with_capacity(128);

                    for key in new_keys {
                        let value = &new_params[key];
                        // serde_json::to_string for String/Value is infallible; write! to String is infallible
                        let key_json = serde_json::to_string(key).unwrap_or_default();
                        let value_json = serde_json::to_string(value).unwrap_or_default();
                        let _ = write!(&mut json_fragment, ", {key_json}: {value_json}");
                    }

                    calls.push(ToolCallItem {
                        tool_index: tool_id,
                        name: None,
                        parameters: json_fragment.clone(),
                    });

                    self.streamed_args_for_tool[tool_id].push_str(&json_fragment);
                }
            }

            // Update current parameters
            self.current_parameters = new_params;

            // Update prev_tool_call_arr
            while self.prev_tool_call_arr.len() <= tool_id {
                self.prev_tool_call_arr.push(Value::Null);
            }
            self.prev_tool_call_arr[tool_id] = serde_json::json!({
                "name": self.current_function_name,
                "arguments": self.current_parameters,
            });
        }

        calls
    }
}

impl Default for MinimaxM2Parser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for MinimaxM2Parser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        // Without a schema, fall back to string/number/bool coercion.
        self.parse_complete_with_tools(text, &[]).await
    }

    async fn parse_complete_with_tools(
        &self,
        text: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        let (results, first_valid_tool_pos) = self.parse_tool_calls_from_text(text, tools);

        if results.is_empty() {
            return Ok((text.to_string(), vec![]));
        }

        let normal_text = if let Some(pos) = first_valid_tool_pos {
            text[..pos].to_string()
        } else {
            text.to_string()
        };

        Ok((normal_text, results))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);
        let mut normal_text = String::new();
        let mut calls = Vec::new();

        // Build tool indices for validation
        let tool_indices = helpers::get_tool_indices(tools);

        loop {
            // After </invoke>, we're waiting for either another <invoke> or
            // the closing </minimax:tool_call>.
            if self.waiting_for_tool_call_end {
                let next_invoke = self.buffer.find("<invoke");
                let end_tag = self.buffer.find(self.tool_call_end_token);

                match (next_invoke, end_tag) {
                    (Some(inv), Some(end)) if inv < end => {
                        // Another <invoke> before end tag — continue parsing
                        self.waiting_for_tool_call_end = false;
                        self.in_tool_call = true;
                        continue;
                    }
                    (_, Some(end_pos)) => {
                        self.buffer =
                            self.buffer[end_pos + self.tool_call_end_token.len()..].to_string();
                        self.in_tool_call = false;
                        self.waiting_for_tool_call_end = false;
                        continue;
                    }
                    (Some(_), None) => {
                        // Another invoke but no end tag yet — keep parsing
                        self.waiting_for_tool_call_end = false;
                        self.in_tool_call = true;
                        continue;
                    }
                    (None, None) => {
                        break;
                    }
                }
            }

            // If we're not in a tool call and don't see a start token, return normal text
            if !self.in_tool_call && !self.buffer.contains(self.tool_call_start_token) {
                // Check if buffer might contain a partial start token at the end
                if let Some(partial_len) =
                    helpers::ends_with_partial_token(&self.buffer, self.tool_call_start_token)
                {
                    // Return everything except the potential partial token
                    let end = self.buffer.len() - partial_len;
                    normal_text = self.buffer[..end].to_string();
                    self.buffer = self.buffer[end..].to_string();
                } else {
                    // No partial token, return all as normal text
                    normal_text.clone_from(&self.buffer);
                    self.buffer.clear();
                }
                break;
            }

            // Look for tool call start
            if !self.in_tool_call {
                if let Some(start) = self.buffer.find(self.tool_call_start_token) {
                    normal_text = self.buffer[..start].to_string();
                    self.buffer =
                        self.buffer[start + self.tool_call_start_token.len()..].to_string();

                    self.in_tool_call = true;
                    self.function_name_sent = false;
                    self.current_function_name.clear();
                    self.current_parameters.clear();

                    continue;
                } else {
                    // No start token found
                    break;
                }
            }

            // We're in a tool call, try to parse function name if not sent yet
            if !self.function_name_sent {
                // Use regex to extract function name from <invoke name="..."> pattern
                // Check if we have enough text to match the invoke pattern
                if let Some(captures) = self.invoke_extractor.captures(&self.buffer) {
                    let function_name = captures
                        .get(1)
                        .map_or("", |m| m.as_str())
                        .trim()
                        .to_string();

                    // Validate function name
                    if tool_indices.contains_key(&function_name) {
                        self.current_function_name.clone_from(&function_name);
                        self.function_name_sent = true;

                        // Initialize tool call tracking
                        if self.current_tool_id == -1 {
                            self.current_tool_id = 0;
                        }

                        // Ensure tracking arrays are large enough
                        helpers::ensure_capacity(
                            self.current_tool_id,
                            &mut self.prev_tool_call_arr,
                            &mut self.streamed_args_for_tool,
                        );

                        // Send tool name with empty parameters
                        calls.push(ToolCallItem {
                            tool_index: self.current_tool_id as usize,
                            name: Some(function_name),
                            parameters: String::new(),
                        });

                        // Find the position after the opening invoke tag (after the >)
                        // We only want to remove up to the opening tag, not the full match
                        if let Some(pos) = self.buffer.find('>') {
                            self.buffer = self.buffer[pos + 1..].to_string();
                        }
                        continue;
                    } else {
                        // Invalid function name, reset state
                        tracing::debug!("Invalid function name: {}", function_name);
                        self.in_tool_call = false;
                        normal_text.push_str(&self.buffer);
                        self.buffer.clear();
                        break;
                    }
                }
                // No complete invoke pattern found yet, wait for more text
                break;
            }

            // Parse parameters incrementally
            if self.function_name_sent {
                // Process parameters and get any calls to emit
                // Note: We need to be careful here - parse_and_stream_parameters needs
                // to work with the buffer but we can't pass &self.buffer directly
                // due to borrow checker. Instead, we'll refactor slightly.
                // For now, keep the clone but mark it as a TODO for future optimization
                let buffer_copy = self.buffer.clone(); // TODO: Optimize this
                let parameter_calls = self.parse_and_stream_parameters(&buffer_copy, tools);
                calls.extend(parameter_calls);

                // Check if tool call is complete (</invoke> found)
                if let Some(invoke_end) = self.buffer.find(self.invoke_end_token) {
                    // Add closing brace to complete the JSON object
                    let tool_id = self.current_tool_id as usize;
                    if tool_id < self.streamed_args_for_tool.len() {
                        let current_streamed = &self.streamed_args_for_tool[tool_id];
                        if !current_streamed.is_empty() && !current_streamed.ends_with('}') {
                            // Count opening and closing braces to check if JSON is complete
                            let open_braces = current_streamed.matches('{').count();
                            let close_braces = current_streamed.matches('}').count();
                            if open_braces > close_braces {
                                calls.push(ToolCallItem {
                                    tool_index: tool_id,
                                    name: None,
                                    parameters: "}".to_string(),
                                });
                                self.streamed_args_for_tool[tool_id].push('}');
                            }
                        }
                    }

                    // Move buffer past the </invoke>
                    self.buffer =
                        self.buffer[invoke_end + self.invoke_end_token.len()..].to_string();

                    // Advance tool_id and reset per-invoke state for the next
                    // tool call within the same <minimax:tool_call> block.
                    self.current_tool_id += 1;
                    self.function_name_sent = false;
                    self.current_function_name.clear();
                    self.current_parameters.clear();

                    // Check if there's another <invoke> before </minimax:tool_call>.
                    // MiniMax puts multiple <invoke> blocks inside one wrapper.
                    let next_invoke = self.buffer.find("<invoke");
                    let end_tag = self.buffer.find(self.tool_call_end_token);

                    match (next_invoke, end_tag) {
                        (Some(inv), Some(end)) if inv < end => {
                            // Another invoke coming — stay in_tool_call, loop
                            continue;
                        }
                        (_, Some(end_pos)) => {
                            // End tag found, no more invokes
                            self.buffer =
                                self.buffer[end_pos + self.tool_call_end_token.len()..].to_string();
                            self.in_tool_call = false;
                            continue;
                        }
                        (Some(_), None) => {
                            // Another invoke exists but end tag not received yet — keep parsing
                            continue;
                        }
                        (None, None) => {
                            // Neither found yet — wait for more data
                            self.waiting_for_tool_call_end = true;
                            break;
                        }
                    }
                }
                // Tool call not complete yet, wait for more text
                break;
            }
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
        self.buffer.clear();
        self.prev_tool_call_arr.clear();
        self.current_tool_id = -1;
        self.streamed_args_for_tool.clear();
        self.current_function_name.clear();
        self.current_parameters.clear();
        self.in_tool_call = false;
        self.function_name_sent = false;
        self.waiting_for_tool_call_end = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn types(slice: &[&str]) -> Vec<String> {
        slice.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn convert_with_schema_string_keeps_null_as_string() {
        // Regression: fc-dash bfcl/simple_javascript_43.
        // Schema declares `error: string` and the tool description tells the
        // model the value may be the string `"null"`.  The parser must NOT
        // coerce it to JSON null.
        let v = MinimaxM2Parser::convert_with_schema("null", &types(&["string"]));
        assert_eq!(v, Value::String("null".to_string()));

        let v = MinimaxM2Parser::convert_with_schema("None", &types(&["string"]));
        assert_eq!(v, Value::String("None".to_string()));

        let v = MinimaxM2Parser::convert_with_schema("nil", &types(&["string"]));
        assert_eq!(v, Value::String("nil".to_string()));
    }

    #[test]
    fn convert_with_schema_nullable_type_becomes_null() {
        // Schema `["string", "null"]` – literal "null" should coerce to Null.
        let v = MinimaxM2Parser::convert_with_schema("null", &types(&["string", "null"]));
        assert_eq!(v, Value::Null);

        let v = MinimaxM2Parser::convert_with_schema("None", &types(&["integer", "null"]));
        assert_eq!(v, Value::Null);
    }

    #[test]
    fn convert_with_schema_integer_still_parses_numbers() {
        let v = MinimaxM2Parser::convert_with_schema("42", &types(&["integer"]));
        assert_eq!(v, Value::Number(42i64.into()));
    }

    #[test]
    fn convert_with_schema_boolean_still_works() {
        let v = MinimaxM2Parser::convert_with_schema("true", &types(&["boolean"]));
        assert_eq!(v, Value::Bool(true));
        let v = MinimaxM2Parser::convert_with_schema("false", &types(&["boolean"]));
        assert_eq!(v, Value::Bool(false));
    }

    #[test]
    fn convert_with_schema_number_keeps_int_as_int() {
        // Regression for fc-dash `bfcl/parallel_138`: schema says `number`,
        // model emits `85` (no decimal) — must serialize as `85`, not `85.0`,
        // to match sglang's Python path (`int(x) or float(x)`).
        let v = MinimaxM2Parser::convert_with_schema("85", &types(&["number"]));
        assert_eq!(v, Value::Number(85i64.into()));

        let v = MinimaxM2Parser::convert_with_schema("1.8", &types(&["number"]));
        // 1.8 should still be a float
        assert!(v.is_f64(), "1.8 should parse as float, got {v:?}");
        assert_eq!(
            v.as_f64().unwrap().to_string(),
            "1.8"
        );

        let v = MinimaxM2Parser::convert_with_schema("1267000000", &types(&["number"]));
        assert_eq!(v, Value::Number(1267000000i64.into()));

        // Scientific notation should still be float
        let v = MinimaxM2Parser::convert_with_schema("1e5", &types(&["number"]));
        assert!(v.is_f64());
    }

    #[tokio::test]
    async fn parse_complete_arguments_use_spaces_like_python() {
        // Regression for gRPC/HTTP parity: fc-dash `bfcl/parallel_138` etc
        // compare `tool_calls[].function.arguments` as a string and expect
        // Python's `json.dumps` default spacing (`{"a": 1, "b": 2}`).
        let parser = MinimaxM2Parser::new();
        let text = r#"<minimax:tool_call>
<invoke name="calculate_BMI">
<parameter name="weight_kg">85</parameter>
<parameter name="height_m">1.8</parameter>
</invoke>
</minimax:tool_call>"#;

        let (_, calls) = parser.parse_complete(text).await.unwrap();
        assert_eq!(calls.len(), 1);
        let args = &calls[0].function.arguments;
        // The ordering in the argument JSON follows input order; we assert
        // only the whitespace invariant that must match Python.
        assert!(
            args.contains(", "),
            "expected `, ` between items, got: {args}"
        );
        assert!(
            args.contains(": "),
            "expected `: ` between keys and values, got: {args}"
        );
        // And verify no `":":` (no-space) appears
        assert!(!args.contains("\":\""), "unexpected compact JSON: {args}");
        assert!(!args.contains(",\""), "unexpected compact JSON: {args}");
    }
}
