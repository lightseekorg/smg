//! Kimi-K3 (XTML) tool-call parser
//!
//! Ports the Kimi-K3 reference tool-call parser: turns the generated XTML
//! `response` and `tools` channels back into plain `content` and `ToolCall`s.
//!
//! # Format
//!
//! K3 assistant tool calls live in a nested `tools` channel, emitted after a
//! sibling `response` channel that is unwrapped into content:
//! ```text
//! <|open|>response<|sep|> CONTENT <|close|>response<|sep|>
//! <|open|>tools<|sep|>
//!   <|open|>call tool="NAME" index="N"<|sep|>
//!     <|open|>argument key="K" type="T"<|sep|>VALUE<|close|>argument<|sep|>
//!   <|close|>call<|sep|>
//! <|close|>tools<|sep|>
//! ```
//! `<|open|>`, `<|close|>`, `<|sep|>` are literal strings in the detokenized
//! text (not special tokens from this parser's point of view). Markers
//! tolerate optional whitespace within them (e.g. `<|open|> tools <|sep|>`)
//! as defense in depth; this is a no-op on clean input. Bodies use
//! non-greedy matching so each block stops at its own first closing marker.
//!
//! # Argument decoding
//! - `type="string"` -> the value is raw text, used as-is (no unescaping).
//! - any other type -> the value is JSON-decoded; on JSON error, falls back
//!   to the raw string rather than erroring mid-stream.
//!
//! # Attribute escaping
//! Attribute values (`tool=`, `index=`, `key=`, `type=`) are escaped on the
//! encode side (`&` -> `&amp;`, `"` -> `&quot;`); decoding reverses that in
//! order (`&quot;` -> `"` then `&amp;` -> `&`).
//!
//! # Tool-call id
//! SMG's [`ToolCall`] has no `id` field (one is assigned later by
//! `model_gateway`), so this parser only produces `name` + `arguments` (+
//! content). Streaming [`ToolCallItem::tool_index`] is a per-message
//! zero-based ordinal assigned in emission order, independent of the XTML
//! `index` attribute. Deriving it from the ordinal — rather than trusting the
//! model-supplied `index` — keeps streamed indices (and the ids later built
//! from them) unique and monotonic even when a model emits duplicate, sparse,
//! or out-of-order `index` values, and matches the non-streaming path, which
//! also indexes tool calls by their emission order.
//!
//! Known limitation (inherited from the reference): because string argument
//! and response bodies are emitted raw, a value that literally contains
//! `<|close|>argument<|sep|>` or `<|close|>response<|sep|>` is
//! indistinguishable from a real closing marker.

use std::collections::HashMap;

use async_trait::async_trait;
use openai_protocol::common::Tool;
use regex::Regex;
use serde_json::Value;

use crate::{
    errors::ParserResult,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

const TOOLS_OPEN: &str = "<|open|>tools<|sep|>";
const RESPONSE_OPEN: &str = "<|open|>response<|sep|>";
const RESPONSE_CLOSE: &str = "<|close|>response<|sep|>";

/// A decoded `<|open|>call ...<|sep|>...<|close|>call<|sep|>` block.
///
/// The XTML `index` attribute is deliberately not captured: streamed tool
/// indices are assigned by emission order (see the module-level `Tool-call id`
/// docs), so a model-supplied `index` never influences the parser output.
struct DecodedCall {
    name: String,
    /// Compact JSON object string.
    arguments: String,
}

/// Kimi-K3 XTML tool-call parser.
///
/// Handles both non-streaming (`parse_complete`) and streaming
/// (`parse_incremental`) extraction of the `tools` channel, and unwraps the
/// sibling `response` channel into `content`.
pub struct KimiK3Parser {
    /// Matches `<|open|>tools<|sep|>` (tolerant of inner whitespace).
    tools_open_re: Regex,
    /// Matches `<|close|>tools<|sep|>`.
    tools_close_re: Regex,
    /// Matches `<|open|>response<|sep|>`.
    response_open_re: Regex,
    /// Matches `<|close|>response<|sep|>`.
    response_close_re: Regex,
    /// Matches a stray `<|close|>message<|sep|>` (stripped from content).
    message_close_re: Regex,
    /// Matches one `call` block, capturing `attrs` and `body`.
    call_re: Regex,
    /// Matches one `argument` block, capturing `attrs` and `val`.
    arg_re: Regex,
    /// Matches one `key="value"` attribute pair.
    attr_re: Regex,
    /// Matches a complete, unwrapped `response` channel, capturing `c`.
    response_re: Regex,

    /// Accumulates every chunk seen so far (plays the role of the streaming
    /// `current_text`, which the engine rebuilds from scratch each call).
    buffer: String,
    /// Byte offset into `buffer` up to which response content has already
    /// been emitted.
    sent_content_idx: usize,
    /// Number of tool calls already emitted.
    sent_tool_call_count: usize,
}

impl KimiK3Parser {
    /// Create a new Kimi-K3 parser.
    #[expect(
        clippy::expect_used,
        reason = "regex patterns are compile-time string literals"
    )]
    pub fn new() -> Self {
        Self {
            tools_open_re: Regex::new(r"<\|open\|>\s*tools\s*<\|sep\|>")
                .expect("valid regex"),
            tools_close_re: Regex::new(r"<\|close\|>\s*tools\s*<\|sep\|>")
                .expect("valid regex"),
            response_open_re: Regex::new(r"<\|open\|>\s*response\s*<\|sep\|>")
                .expect("valid regex"),
            response_close_re: Regex::new(r"<\|close\|>\s*response\s*<\|sep\|>")
                .expect("valid regex"),
            message_close_re: Regex::new(r"<\|close\|>\s*message\s*<\|sep\|>")
                .expect("valid regex"),
            call_re: Regex::new(
                r"(?s)<\|open\|>\s*call\s+(?P<attrs>.*?)<\|sep\|>(?P<body>.*?)<\|close\|>\s*call\s*<\|sep\|>",
            )
            .expect("valid regex"),
            arg_re: Regex::new(
                r"(?s)<\|open\|>\s*argument\s+(?P<attrs>.*?)<\|sep\|>(?P<val>.*?)<\|close\|>\s*argument\s*<\|sep\|>",
            )
            .expect("valid regex"),
            attr_re: Regex::new(r#"(?P<k>\w+)="(?P<v>[^"]*)""#).expect("valid regex"),
            response_re: Regex::new(
                r"(?s)<\|open\|>\s*response\s*<\|sep\|>(?P<c>.*?)<\|close\|>\s*response\s*<\|sep\|>",
            )
            .expect("valid regex"),
            buffer: String::new(),
            sent_content_idx: 0,
            sent_tool_call_count: 0,
        }
    }

    /// Parse the `key="value"` attribute pairs in `s`, unescaping each value
    /// (`&quot;` -> `"` then `&amp;` -> `&`, the reverse of the encode order).
    fn attrs(&self, s: &str) -> HashMap<String, String> {
        self.attr_re
            .captures_iter(s)
            .map(|m| {
                let key = m.name("k").map_or("", |g| g.as_str()).to_string();
                let value = m
                    .name("v")
                    .map_or("", |g| g.as_str())
                    .replace("&quot;", "\"")
                    .replace("&amp;", "&");
                (key, value)
            })
            .collect()
    }

    /// Decode one `call` block (its `attrs` segment and `body`) into a
    /// [`DecodedCall`]. Each argument is re-typed per its `type=` tag:
    /// strings pass through raw, everything else is JSON-decoded (falling
    /// back to the raw string on malformed JSON, so a partial stream never
    /// errors). Returns `None` when no tool name is present.
    fn decode_call(&self, attrs: &str, body: &str) -> Option<DecodedCall> {
        let call_attrs = self.attrs(attrs);
        let tool_name = call_attrs.get("tool").cloned().unwrap_or_default();
        if tool_name.is_empty() {
            return None;
        }

        let mut arguments = serde_json::Map::new();
        for arg_match in self.arg_re.captures_iter(body) {
            let arg_attrs_text = arg_match.name("attrs").map_or("", |g| g.as_str());
            let arg_attrs = self.attrs(arg_attrs_text);
            let key = arg_attrs.get("key").cloned().unwrap_or_default();
            let arg_type = arg_attrs.get("type").map_or("string", String::as_str);
            let raw_value = arg_match.name("val").map_or("", |g| g.as_str());

            let value = if arg_type == "string" {
                Value::String(raw_value.to_string())
            } else {
                serde_json::from_str::<Value>(raw_value)
                    .unwrap_or_else(|_| Value::String(raw_value.to_string()))
            };
            arguments.insert(key, value);
        }

        let arguments_str = serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".to_string());

        Some(DecodedCall {
            name: tool_name,
            arguments: arguments_str,
        })
    }

    /// Strip XTML response/message markers from generated response text.
    ///
    /// In chat serving, `<|open|>response<|sep|>` is often part of the
    /// prompt generation prefix, so the model output may only contain the
    /// body plus `<|close|>response<|sep|>`. Handles both that
    /// consumed-prefix shape and a complete
    /// `<|open|>response<|sep|>...<|close|>response<|sep|>` wrapper.
    fn strip_response_content(&self, text: &str) -> Option<String> {
        let stripped = if let Some(m_open) = self.response_open_re.find(text) {
            if let Some(m_close) = self.response_close_re.find_at(text, m_open.end()) {
                text[m_open.end()..m_close.start()].to_string()
            } else {
                text[m_open.end()..].to_string()
            }
        } else {
            self.response_close_re.replace_all(text, "").into_owned()
        };
        let stripped = self
            .message_close_re
            .replace_all(&stripped, "")
            .into_owned();
        if stripped.is_empty() {
            None
        } else {
            Some(stripped)
        }
    }

    /// Compute non-streaming content: prefer the unwrapped `response`
    /// channel; else fall back to stripping markers from `before` (the text
    /// preceding the `tools` channel, or the whole output when there is no
    /// `tools` channel at all).
    fn content(&self, model_output: &str, before: &str) -> Option<String> {
        if let Some(m) = self.response_re.captures(model_output) {
            let c = m.name("c").map_or("", |g| g.as_str());
            return if c.is_empty() {
                None
            } else {
                Some(c.to_string())
            };
        }
        self.strip_response_content(before)
    }

    /// Compute the streaming-safe slice of response content, updating
    /// `sent_content_idx`. This is what keeps split markers from leaking:
    /// content is only released up to the start of the next recognized
    /// marker (or up to the point where a partial marker might still be
    /// growing at the tail of `current_text`).
    fn extract_response_content(&mut self, current_text: &str) -> Option<String> {
        let m_open = self.response_open_re.find(current_text);
        let body_start = m_open.map_or(0, |m| m.end());

        let tools_start = self
            .tools_open_re
            .find_at(current_text, body_start)
            .map(|m| m.start());
        let response_end = self
            .response_close_re
            .find_at(current_text, body_start)
            .map(|m| m.start());

        let sendable_idx = match (tools_start, response_end) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => {
                let overlap = partial_tag_overlap(current_text, RESPONSE_OPEN)
                    .max(partial_tag_overlap(current_text, RESPONSE_CLOSE))
                    .max(partial_tag_overlap(current_text, TOOLS_OPEN));
                current_text.len() - overlap
            }
        };

        if sendable_idx <= body_start {
            return None;
        }
        if self.sent_content_idx < body_start {
            self.sent_content_idx = body_start;
        }
        if sendable_idx <= self.sent_content_idx {
            return None;
        }

        let content = current_text[self.sent_content_idx..sendable_idx].to_string();
        self.sent_content_idx = sendable_idx;
        if content.is_empty() {
            None
        } else {
            Some(content)
        }
    }

    /// Decode every complete `call` block in `section` (a slice starting
    /// just past the `tools`-open marker), skipping blocks with no tool
    /// name.
    fn decode_calls_in_section(&self, section: &str) -> Vec<DecodedCall> {
        let mut calls = Vec::new();
        for m in self.call_re.captures_iter(section) {
            let attrs_text = m.name("attrs").map_or("", |g| g.as_str());
            let body = m.name("body").map_or("", |g| g.as_str());
            if let Some(decoded) = self.decode_call(attrs_text, body) {
                calls.push(decoded);
            }
        }
        calls
    }
}

/// Length of the longest prefix of `tag` that `text` ends with (up to
/// `tag.len() - 1`, since a full match would have been caught by the marker
/// regexes already). Used to hold back a possibly-still-growing partial
/// marker at the tail of the streamed text instead of leaking it as content.
fn partial_tag_overlap(text: &str, tag: &str) -> usize {
    let max_len = text.len().min(tag.len().saturating_sub(1));
    for n in (1..=max_len).rev() {
        if text.ends_with(&tag[..n]) {
            return n;
        }
    }
    0
}

impl Default for KimiK3Parser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for KimiK3Parser {
    async fn parse_complete(&self, output: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        let Some(m_open) = self.tools_open_re.find(output) else {
            let content = self.content(output, output);
            return Ok((content.unwrap_or_default(), vec![]));
        };

        let before = &output[..m_open.start()];
        let start = m_open.end();
        let section_end = self
            .tools_close_re
            .find_at(output, start)
            .map_or(output.len(), |m| m.start());
        let section = &output[start..section_end];

        let calls = self
            .decode_calls_in_section(section)
            .into_iter()
            .map(|decoded| ToolCall {
                function: FunctionCall {
                    name: decoded.name,
                    arguments: decoded.arguments,
                },
            })
            .collect();

        let content = self.content(output, before);
        Ok((content.unwrap_or_default(), calls))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        _tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);
        let current_text = self.buffer.clone();

        let content = self.extract_response_content(&current_text);

        let Some(m_tools) = self.tools_open_re.find(&current_text) else {
            return Ok(StreamingParseResult {
                normal_text: content.unwrap_or_default(),
                calls: vec![],
            });
        };

        let section = &current_text[m_tools.end()..];
        let decoded_calls = self.decode_calls_in_section(section);

        if decoded_calls.len() <= self.sent_tool_call_count {
            return Ok(StreamingParseResult {
                normal_text: content.unwrap_or_default(),
                calls: vec![],
            });
        }

        let calls = decoded_calls
            .iter()
            .skip(self.sent_tool_call_count)
            .enumerate()
            .map(|(i, decoded)| ToolCallItem {
                // Per-message zero-based ordinal in emission order; the XTML
                // `index` attribute is deliberately ignored (see module docs).
                tool_index: self.sent_tool_call_count + i,
                name: Some(decoded.name.clone()),
                parameters: decoded.arguments.clone(),
            })
            .collect();
        self.sent_tool_call_count = decoded_calls.len();

        Ok(StreamingParseResult {
            normal_text: content.unwrap_or_default(),
            calls,
        })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        self.tools_open_re.is_match(text)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.sent_content_idx = 0;
        self.sent_tool_call_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;

    const OPEN: &str = "<|open|>";
    const CLOSE: &str = "<|close|>";
    const SEP: &str = "<|sep|>";

    fn _arg(key: &str, typ: &str, value: &str) -> String {
        format!(r#"{OPEN}argument key="{key}" type="{typ}"{SEP}{value}{CLOSE}argument{SEP}"#)
    }

    fn _call(tool: &str, index: i32, args: &[String]) -> String {
        let body: String = args.concat();
        format!(r#"{OPEN}call tool="{tool}" index="{index}"{SEP}{body}{CLOSE}call{SEP}"#)
    }

    fn _response(content: &str) -> String {
        format!("{OPEN}response{SEP}{content}{CLOSE}response{SEP}")
    }

    fn _tools(calls: &[String]) -> String {
        let body: String = calls.concat();
        format!("{OPEN}tools{SEP}{body}{CLOSE}tools{SEP}")
    }

    #[tokio::test]
    async fn test_parse_complete_response_and_typed_arguments() {
        let parser = KimiK3Parser::new();
        let input = _response("answer")
            + &_tools(&[_call(
                "calc",
                1,
                &[
                    _arg("x", "number", "1"),
                    _arg("flag", "boolean", "true"),
                    _arg("text", "string", "raw"),
                ],
            )]);

        let (content, calls) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(content, "answer");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "calc");
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(
            args,
            serde_json::json!({"x": 1, "flag": true, "text": "raw"})
        );
    }

    #[tokio::test]
    async fn test_parse_complete_unescapes_attributes() {
        let parser = KimiK3Parser::new();
        let input = _tools(&[_call(
            "a&amp;b&quot;c",
            1,
            &[_arg("k&amp;q", "string", "v")],
        )]);

        let (_content, calls) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "a&b\"c");
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args, serde_json::json!({"k&q": "v"}));
    }

    #[tokio::test]
    async fn test_parse_complete_allows_less_than_in_attributes() {
        let parser = KimiK3Parser::new();
        let input = _tools(&[_call("calc<beta", 1, &[_arg("foo<bar", "string", "raw")])]);

        let (_content, calls) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "calc<beta");
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args, serde_json::json!({"foo<bar": "raw"}));
    }

    #[tokio::test]
    async fn test_parse_complete_no_tools_channel() {
        let parser = KimiK3Parser::new();
        let input = _response("hi");

        let (content, calls) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(content, "hi");
        assert_eq!(calls.len(), 0);
    }

    #[tokio::test]
    async fn test_parse_complete_whitespace_degraded_markers() {
        let parser = KimiK3Parser::new();
        let input = format!("{OPEN} response {SEP}answer{CLOSE} response {SEP}");

        let (content, calls) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(content, "answer");
        assert_eq!(calls.len(), 0);
    }

    #[tokio::test]
    async fn test_parse_complete_malformed_json_argument_falls_back_to_raw() {
        let parser = KimiK3Parser::new();
        let input = _tools(&[_call("calc", 1, &[_arg("x", "number", "not-json")])]);

        let (_content, calls) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(calls.len(), 1);
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args, serde_json::json!({"x": "not-json"}));
    }

    #[tokio::test]
    async fn test_parse_complete_multiple_tool_calls() {
        let parser = KimiK3Parser::new();
        let input = _tools(&[
            _call("first", 1, &[_arg("a", "number", "1")]),
            _call("second", 2, &[_arg("b", "number", "2")]),
        ]);

        let (_content, calls) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "first");
        assert_eq!(calls[1].function.name, "second");
    }

    #[tokio::test]
    async fn test_decode_call_without_index_attribute() {
        let parser = KimiK3Parser::new();
        let input = format!(
            r#"{OPEN}tools{SEP}{OPEN}call tool="calc"{SEP}{CLOSE}call{SEP}{CLOSE}tools{SEP}"#
        );

        let (_content, calls) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "calc");
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn test_has_tool_markers() {
        let parser = KimiK3Parser::new();
        assert!(parser.has_tool_markers(&_tools(&[])));
        assert!(!parser.has_tool_markers(&_response("hi")));
        assert!(!parser.has_tool_markers("plain text"));
    }

    #[tokio::test]
    async fn test_streaming_split_markers_do_not_leak() {
        let mut parser = KimiK3Parser::new();
        let hi_chunk = format!("{SEP}Hi");
        let call_open_chunk = format!(r#"{OPEN}call tool="calc" index="1"{SEP}"#);
        let arg_chunk = _arg("x", "number", "1");
        let close_call_chunk = format!("{CLOSE}call");

        let chunks: [&str; 10] = [
            OPEN,
            "response",
            hi_chunk.as_str(),
            OPEN,
            "tools",
            SEP,
            call_open_chunk.as_str(),
            arg_chunk.as_str(),
            close_call_chunk.as_str(),
            SEP,
        ];

        let mut content = String::new();
        let mut calls = Vec::new();
        for chunk in chunks {
            let result = parser.parse_incremental(chunk, &[]).await.unwrap();
            content.push_str(&result.normal_text);
            calls.extend(result.calls);
        }

        assert_eq!(content, "Hi");
        assert!(!content.contains(OPEN));
        assert!(!content.contains(SEP));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name.as_deref(), Some("calc"));
        let args: Value = serde_json::from_str(&calls[0].parameters).unwrap();
        assert_eq!(args, serde_json::json!({"x": 1}));
    }

    #[tokio::test]
    async fn test_streaming_consumed_response_prefix_no_call_keeps_content() {
        let mut parser = KimiK3Parser::new();
        let close_response_chunk = format!("response{SEP}");
        let chunks: [&str; 4] = ["O", "K", CLOSE, close_response_chunk.as_str()];

        let mut content = String::new();
        for chunk in chunks {
            let result = parser.parse_incremental(chunk, &[]).await.unwrap();
            assert!(!result.normal_text.contains(CLOSE));
            content.push_str(&result.normal_text);
        }
        assert_eq!(content, "OK");
    }

    #[tokio::test]
    async fn test_streaming_multiple_tool_calls_increment_tool_index() {
        let mut parser = KimiK3Parser::new();
        let input = _tools(&[
            _call("first", 1, &[_arg("a", "number", "1")]),
            _call("second", 2, &[_arg("b", "number", "2")]),
        ]);

        let result = parser.parse_incremental(&input, &[]).await.unwrap();
        assert_eq!(result.calls.len(), 2);
        assert_eq!(result.calls[0].tool_index, 0);
        assert_eq!(result.calls[1].tool_index, 1);
    }

    #[tokio::test]
    async fn test_streaming_duplicate_xtml_index_uses_ordinal() {
        // A model that emits colliding, sparse, or out-of-order `index`
        // attributes must still yield unique, monotonic streamed indices: the
        // parser assigns them by emission order and ignores the attribute.
        let mut parser = KimiK3Parser::new();
        let input = _tools(&[
            _call("first", 7, &[_arg("a", "number", "1")]),
            _call("second", 7, &[_arg("b", "number", "2")]),
            _call("third", 3, &[_arg("c", "number", "3")]),
        ]);

        let result = parser.parse_incremental(&input, &[]).await.unwrap();
        assert_eq!(result.calls.len(), 3);
        assert_eq!(result.calls[0].tool_index, 0);
        assert_eq!(result.calls[1].tool_index, 1);
        assert_eq!(result.calls[2].tool_index, 2);
    }

    #[tokio::test]
    async fn test_reset_clears_streaming_state() {
        let mut parser = KimiK3Parser::new();

        let primed = format!("{OPEN}response{SEP}Hello{CLOSE}response{SEP}")
            + &_tools(&[_call("calc", 1, &[_arg("x", "number", "1")])]);
        let primed_result = parser.parse_incremental(&primed, &[]).await.unwrap();
        assert_eq!(primed_result.calls.len(), 1);

        parser.reset();

        let fresh_content = format!("{OPEN}response{SEP}Fresh{CLOSE}response{SEP}");
        let content_result = parser.parse_incremental(&fresh_content, &[]).await.unwrap();
        assert_eq!(content_result.normal_text, "Fresh");

        let fresh_call = _tools(&[_call("calc", 1, &[_arg("x", "number", "2")])]);
        let call_result = parser.parse_incremental(&fresh_call, &[]).await.unwrap();
        assert_eq!(call_result.calls.len(), 1);
        assert_eq!(call_result.calls[0].name.as_deref(), Some("calc"));
    }

    #[test]
    fn test_factory_resolves_moonshot_kimi_k3() {
        let factory = crate::factory::ParserFactory::new();
        assert_eq!(
            factory
                .registry()
                .resolve_model_to_parser("moonshotai/Kimi-K3"),
            Some("kimi_k3".to_string())
        );
    }

    #[test]
    fn test_factory_resolves_underscore_kimi_k3() {
        let factory = crate::factory::ParserFactory::new();
        for id in ["kimi_k3", "Kimi_K3", "moonshotai/Kimi_K3"] {
            assert_eq!(
                factory.registry().resolve_model_to_parser(id),
                Some("kimi_k3".to_string()),
                "expected `{id}` to resolve to the kimi_k3 parser"
            );
        }
    }
}
