// Kimi K3 reasoning parser for the XTML tag format.
//
// K3 uses structural special tokens <|open|>, <|close|>, <|sep|> to delimit
// channels. This parser extracts the `think` channel into `reasoning_text` and
// delivers the remainder (with `response`/`message` wrapper tokens removed) as
// `normal_text`. The `tools` channel is preserved verbatim for the tool parser.
//
// Ported from the Kimi-K3 reference reasoning parser (XTML `think` channel).

use regex::Regex;

use crate::traits::{ParseError, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE};

/// Literal marker strings (used for streaming overlap detection).
const THINK_OPEN: &str = "<|open|>think<|sep|>";
const THINK_CLOSE: &str = "<|close|>think<|sep|>";
const RESPONSE_OPEN: &str = "<|open|>response<|sep|>";
const RESPONSE_CLOSE: &str = "<|close|>response<|sep|>";
const MESSAGE_OPEN: &str = "<|open|>message<|sep|>";
const MESSAGE_CLOSE: &str = "<|close|>message<|sep|>";

/// Reasoning parser for the Kimi K3 (XTML) chat format.
///
/// Extracts the `think` channel into `reasoning_text` and strips the
/// `response`/`message` wrapper tokens from `normal_text` by substitution,
/// preserving any `tools` channel for downstream parsing.
#[derive(Debug, Clone)]
pub struct KimiK3Parser {
    think_open_re: Regex,
    think_close_re: Regex,
    response_open_re: Regex,
    response_close_re: Regex,
    message_open_re: Regex,
    message_close_re: Regex,

    // Shared state
    in_reasoning: bool,

    // Streaming accumulation
    buffer: String,
    reason_phase_ended: bool,
    /// Byte offset in `buffer` where post-think content begins.
    content_tail_start: usize,
    /// Safe-to-emit reasoning already returned to the caller.
    emitted_reasoning: String,
    /// Safe-to-emit content already returned to the caller.
    emitted_content: String,
}

impl KimiK3Parser {
    /// Create a new `KimiK3Parser` with pre-compiled tolerant XTML regexes.
    #[expect(
        clippy::expect_used,
        reason = "regex patterns are hardcoded valid literals; failure indicates a programming error"
    )]
    pub fn new() -> Self {
        let open = r"<\|open\|>";
        let close = r"<\|close\|>";
        let sep = r"<\|sep\|>";

        Self {
            think_open_re: Regex::new(&format!(r"{open}\s*think\s*{sep}"))
                .expect("valid think-open regex"),
            think_close_re: Regex::new(&format!(r"{close}\s*think\s*{sep}"))
                .expect("valid think-close regex"),
            response_open_re: Regex::new(&format!(r"{open}\s*response\s*{sep}"))
                .expect("valid response-open regex"),
            response_close_re: Regex::new(&format!(r"{close}\s*response\s*{sep}"))
                .expect("valid response-close regex"),
            message_open_re: Regex::new(&format!(r"{open}\s*message\s*{sep}"))
                .expect("valid message-open regex"),
            message_close_re: Regex::new(&format!(r"{close}\s*message\s*{sep}"))
                .expect("valid message-close regex"),

            in_reasoning: false,
            buffer: String::new(),
            reason_phase_ended: false,
            content_tail_start: 0,
            emitted_reasoning: String::new(),
            emitted_content: String::new(),
        }
    }

    /// Strip the `response` and `message` wrapper markers
    /// (`<|open|>response<|sep|>`, `<|close|>response<|sep|>`,
    /// `<|open|>message<|sep|>`, `<|close|>message<|sep|>`) from `text` via
    /// substitution.
    ///
    /// Unlike a structured extraction that pulls only the response body,
    /// this approach removes only the wrapper markers, preserving any
    /// `<|open|>tools<|sep|>…` channel that follows for the tool parser.
    fn strip_content_wrapper(&self, text: &str) -> String {
        let text = self.response_open_re.replace_all(text, "");
        let text = self.response_close_re.replace_all(&text, "");
        let text = self.message_open_re.replace_all(&text, "");
        self.message_close_re.replace_all(&text, "").into_owned()
    }

    /// Return the prefix of accumulated reasoning text that is safe to stream.
    ///
    /// Skips past the think-open marker if present, then holds back any
    /// partial marker suffix that could complete as a think-open or think-close.
    fn reasoning_text_ready_to_emit<'a>(&self, text: &'a str) -> &'a str {
        let after_open = if let Some(m) = self.think_open_re.find(text) {
            &text[m.end()..]
        } else {
            text
        };

        let overlap = Self::compute_overlap(after_open, &[THINK_OPEN, THINK_CLOSE]);
        if overlap > 0 {
            &after_open[..after_open.len() - overlap]
        } else {
            after_open
        }
    }

    /// Return the prefix of post-reasoning content that is safe to stream.
    ///
    /// Strips the response-open prefix, removes complete response-close and
    /// message-close markers, then holds back any partial marker suffix.
    ///
    /// The held-back partial-marker suffix is intentionally never force-flushed:
    /// the [`ReasoningParser`] trait has no finalize/end-of-stream hook, and the
    /// only bytes ever withheld are a proper prefix of a control marker (e.g.
    /// `<|clo`). A complete generation always ends on a whole marker, so nothing
    /// is lost in practice; a stream truncated mid-marker-prefix is already
    /// incomplete output, and dropping that dangling fragment is preferable to
    /// leaking a partial control token as content.
    fn content_ready_to_emit(&self, text: &str) -> String {
        // Strip the response-open marker as a prefix (first occurrence only).
        // In the K3 grammar the response channel opens exactly once and its
        // opening marker is a prefix of the content tail, so a slice from its
        // end is correct and a second occurrence cannot legitimately appear.
        // The response-close / message-open / message-close markers, by
        // contrast, are terminators that may sit anywhere in the tail, so they
        // are removed globally below. This asymmetry is deliberate.
        let text = if let Some(m) = self.response_open_re.find(text) {
            &text[m.end()..]
        } else {
            text
        };

        // Remove all complete response-close, message-open, and message-close markers
        let text = self.response_close_re.replace_all(text, "");
        let text = self.message_open_re.replace_all(&text, "");
        let text = self.message_close_re.replace_all(&text, "");

        // Hold back any partial marker suffix (see the fn-level note above:
        // withheld only while it could still complete into a control marker).
        let text_str: &str = &text;
        let overlap = Self::compute_overlap(
            text_str,
            &[RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_OPEN, MESSAGE_CLOSE],
        );
        if overlap > 0 {
            text_str[..text_str.len() - overlap].to_owned()
        } else {
            text_str.to_owned()
        }
    }

    /// Compute the maximum length of a suffix of `text` that is also a
    /// non-empty prefix of any marker in `markers`.
    fn compute_overlap(text: &str, markers: &[&str]) -> usize {
        let mut overlap = 0usize;
        for marker in markers {
            let max_check = marker.len().saturating_sub(1).min(text.len());
            for n in (1..=max_check).rev() {
                if text.ends_with(&marker[..n]) {
                    overlap = overlap.max(n);
                    break;
                }
            }
        }
        overlap
    }
}

impl Default for KimiK3Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for KimiK3Parser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        // Guard against oversized single-shot input, mirroring the streaming
        // entry point (`BaseReasoningParser` and `InklingParser` apply the same
        // guard to their non-streaming path).
        if text.len() > DEFAULT_MAX_BUFFER_SIZE {
            return Err(ParseError::BufferOverflow(text.len()));
        }

        let m_open = self.think_open_re.find(text);
        let reasoning_start = m_open.map(|m| m.end()).unwrap_or(0);

        // No think channel markers at all.
        if m_open.is_none() && self.think_close_re.find(text).is_none() {
            if self.in_reasoning {
                // Generation prefix consumed the open marker; think not yet closed.
                //
                // This diverges from the reference non-streaming extraction
                // path, which is stateless and always treats marker-free text as
                // plain content. SMG instead tracks `in_reasoning` explicitly
                // (set via `mark_reasoning_started()` when the caller's
                // generation prompt already pre-fills K3's thinking-mode
                // prefix `<|open|>think<|sep|>`). So if the model's output is
                // truncated before a close marker appears, the entire output
                // IS reasoning — classify it all as reasoning here rather
                // than losing it as content.
                return Ok(ParserResult::reasoning(text.to_owned()));
            }
            // Plain content with no think channel.
            return Ok(ParserResult::normal(self.strip_content_wrapper(text)));
        }

        // Search for the close marker starting after the open (or from the start).
        let m_close = self.think_close_re.find_at(text, reasoning_start);
        if let Some(m_close) = m_close {
            let reasoning = text[reasoning_start..m_close.start()].to_owned();
            let normal = self.strip_content_wrapper(&text[m_close.end()..]);
            self.in_reasoning = false;
            Ok(ParserResult::new(normal, reasoning))
        } else {
            // Think opened but not yet closed — still in reasoning.
            self.in_reasoning = true;
            Ok(ParserResult::reasoning(text[reasoning_start..].to_owned()))
        }
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        // Guard against unbounded buffer growth, mirroring `BaseReasoningParser`.
        let buffered_size = self.buffer.len() + text.len();
        if buffered_size > DEFAULT_MAX_BUFFER_SIZE {
            return Err(ParseError::BufferOverflow(buffered_size));
        }

        self.buffer.push_str(text);

        // Without a prefilled think channel, the stream begins in normal content.
        // Hold only a possible split think-open marker; otherwise route the
        // accumulated text downstream so structural tool tokens reach the tool parser.
        if !self.in_reasoning
            && !self.reason_phase_ended
            && self.think_open_re.find(&self.buffer).is_none()
        {
            if THINK_OPEN.starts_with(self.buffer.as_str()) {
                return Ok(ParserResult::default());
            }
            self.reason_phase_ended = true;
            self.content_tail_start = 0;
        }

        if self.reason_phase_ended {
            // Content phase: emit new content from the post-think tail.
            let content_tail = &self.buffer[self.content_tail_start..];
            let current_safe = self.content_ready_to_emit(content_tail);
            // Invariant: `emitted_content` is always a prefix of `current_safe`;
            // on the impossible mismatch, emit nothing rather than double-emit.
            let new_content = if current_safe.starts_with(self.emitted_content.as_str()) {
                current_safe[self.emitted_content.len()..].to_owned()
            } else {
                String::new()
            };
            self.emitted_content = current_safe;
            return Ok(ParserResult {
                normal_text: new_content,
                reasoning_text: String::new(),
            });
        }

        // Check whether the think-close marker now completes in the buffer.
        if let Some(m_close) = self.think_close_re.find(&self.buffer) {
            // Transition: reasoning phase ends.
            self.reason_phase_ended = true;
            self.content_tail_start = m_close.end();
            self.in_reasoning = false;

            // Final reasoning delta.
            let r_start = self
                .think_open_re
                .find(&self.buffer)
                .map(|m| m.end())
                .unwrap_or(0);
            let full_reasoning = &self.buffer[r_start..m_close.start()];
            // Invariant: `emitted_reasoning` is always a prefix of `full_reasoning`;
            // on the impossible mismatch, emit nothing rather than double-emit.
            let reasoning_delta = if full_reasoning.starts_with(self.emitted_reasoning.as_str()) {
                full_reasoning[self.emitted_reasoning.len()..].to_owned()
            } else {
                String::new()
            };

            // Initial content delta (safe prefix of the content tail).
            let content_tail = &self.buffer[m_close.end()..];
            let content_safe = self.content_ready_to_emit(content_tail);
            self.emitted_content.clone_from(&content_safe);

            Ok(ParserResult {
                normal_text: content_safe,
                reasoning_text: reasoning_delta,
            })
        } else {
            // Still in reasoning — emit the safe prefix of accumulated reasoning.
            // The tool parser is gated on `is_in_reasoning()`, so the flag must
            // stay set until the think-close transition above clears it.
            self.in_reasoning = true;
            let current_safe = self.reasoning_text_ready_to_emit(&self.buffer).to_owned();
            // Invariant: `emitted_reasoning` is always a prefix of `current_safe`;
            // on the impossible mismatch, emit nothing rather than double-emit.
            let reasoning_delta = if current_safe.starts_with(self.emitted_reasoning.as_str()) {
                current_safe[self.emitted_reasoning.len()..].to_owned()
            } else {
                String::new()
            };
            self.emitted_reasoning = current_safe;
            Ok(ParserResult {
                normal_text: String::new(),
                reasoning_text: reasoning_delta,
            })
        }
    }

    fn reset(&mut self) {
        self.in_reasoning = false;
        self.buffer.clear();
        self.reason_phase_ended = false;
        self.content_tail_start = 0;
        self.emitted_reasoning.clear();
        self.emitted_content.clear();
    }

    fn model_type(&self) -> &str {
        "kimi_k3"
    }

    fn requires_special_tokens(&self) -> bool {
        true
    }

    fn is_in_reasoning(&self) -> bool {
        self.in_reasoning
    }

    fn mark_reasoning_started(&mut self) {
        self.in_reasoning = true;
    }

    fn mark_think_start_stripped(&mut self) {
        // For K3 streaming, the absence of a think-open marker in the buffer
        // is already handled by treating reasoning_start as 0. No additional
        // state is needed.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPEN: &str = "<|open|>";
    const CLOSE: &str = "<|close|>";
    const SEP: &str = "<|sep|>";
    fn think_open() -> String {
        format!("{OPEN}think{SEP}")
    }
    fn think_close() -> String {
        format!("{CLOSE}think{SEP}")
    }
    fn response_open() -> String {
        format!("{OPEN}response{SEP}")
    }
    fn response_close() -> String {
        format!("{CLOSE}response{SEP}")
    }
    fn message_open() -> String {
        format!("{OPEN}message{SEP}")
    }
    fn message_close() -> String {
        format!("{CLOSE}message{SEP}")
    }

    // -------------------------------------------------------------------------
    // Non-streaming tests
    // -------------------------------------------------------------------------

    #[test]
    fn model_type_and_special_tokens() {
        let p = KimiK3Parser::new();
        assert_eq!(p.model_type(), "kimi_k3");
        assert!(p.requires_special_tokens());
    }

    #[test]
    fn extract_with_xtml_tags() {
        let mut p = KimiK3Parser::new();
        let input = format!(
            "{}step{}{}answer",
            think_open(),
            think_close(),
            response_open()
        );
        let r = p.detect_and_parse_reasoning(&input).unwrap();
        assert_eq!(r.reasoning_text, "step");
        assert_eq!(r.normal_text, "answer");
    }

    #[test]
    fn extract_with_generation_prefix_consumed() {
        // open think marker absent (was the prompt prefix)
        let mut p = KimiK3Parser::new();
        p.mark_reasoning_started();
        let input = format!("step{}{}answer", think_close(), response_open());
        let r = p.detect_and_parse_reasoning(&input).unwrap();
        assert_eq!(r.reasoning_text, "step");
        assert_eq!(r.normal_text, "answer");
    }

    #[test]
    fn strips_response_wrapper_with_close_and_keeps_tools_channel() {
        let mut p = KimiK3Parser::new();
        let input = format!(
            "{}step{}{}answer{}{}tools{}CALLS{}tools{}",
            think_open(),
            think_close(),
            response_open(),
            response_close(),
            OPEN,
            SEP,
            CLOSE,
            SEP
        );
        let r = p.detect_and_parse_reasoning(&input).unwrap();
        assert_eq!(r.reasoning_text, "step");
        // response wrapper removed; tools channel preserved for the tool parser
        assert_eq!(
            r.normal_text,
            format!("answer{OPEN}tools{SEP}CALLS{CLOSE}tools{SEP}")
        );
    }

    #[test]
    fn strips_message_wrapper_like_response_wrapper() {
        // The `message` wrapper is stripped symmetrically (open + close), just
        // like the `response` wrapper, leaving only the answer body.
        let mut p = KimiK3Parser::new();
        let input = format!(
            "{}r{}{}answer{}",
            think_open(),
            think_close(),
            message_open(),
            message_close()
        );
        let r = p.detect_and_parse_reasoning(&input).unwrap();
        assert_eq!(r.reasoning_text, "r");
        assert_eq!(r.normal_text, "answer");
    }

    #[test]
    fn no_think_channel_is_all_content() {
        let mut p = KimiK3Parser::new();
        let r = p.detect_and_parse_reasoning("just an answer").unwrap();
        assert_eq!(r.reasoning_text, "");
        assert_eq!(r.normal_text, "just an answer");
    }

    #[test]
    fn marker_free_text_while_in_reasoning_is_all_reasoning() {
        // Once the caller has told us we're already in the think channel
        // (e.g. the generation prompt pre-filled `<|open|>think<|sep|>`),
        // truncated output with no markers at all must come back as
        // reasoning in full, not be misclassified as content.
        let mut p = KimiK3Parser::new();
        p.mark_reasoning_started();
        let text = "still thinking, no markers";
        let r = p.detect_and_parse_reasoning(text).unwrap();
        assert_eq!(r.reasoning_text, text);
        assert_eq!(r.normal_text, "");
    }

    #[test]
    fn detect_and_parse_reasoning_buffer_overflow_is_guarded() {
        let mut p = KimiK3Parser::new();
        let oversized = "a".repeat(DEFAULT_MAX_BUFFER_SIZE + 1);
        let result = p.detect_and_parse_reasoning(&oversized);
        assert!(matches!(
            result,
            Err(ParseError::BufferOverflow(size)) if size == DEFAULT_MAX_BUFFER_SIZE + 1
        ));
    }

    // -------------------------------------------------------------------------
    // Streaming tests
    // -------------------------------------------------------------------------

    #[test]
    fn streaming_thinking_disabled_routes_tools_to_normal_content() {
        let mut p = KimiK3Parser::new();
        let chunks = [
            "<|close|>response",
            "<|sep|>",
            "<|open|>tools",
            "<|sep|>",
            r#"<|open|>call tool="kvv_walle_case" index="1""#,
            "<|sep|>",
            "<|close|>call",
            "<|sep|>",
            "<|close|>tools",
            "<|sep|>",
        ];

        let mut reasoning = String::new();
        let mut normal = String::new();
        for chunk in chunks {
            let result = p.parse_reasoning_streaming_incremental(chunk).unwrap();
            reasoning.push_str(&result.reasoning_text);
            normal.push_str(&result.normal_text);
        }

        assert_eq!(reasoning, "");
        assert_eq!(
            normal,
            concat!(
                "<|open|>tools<|sep|>",
                r#"<|open|>call tool="kvv_walle_case" index="1"<|sep|>"#,
                "<|close|>call<|sep|><|close|>tools<|sep|>"
            )
        );
        assert!(!p.is_in_reasoning());
    }

    #[test]
    fn streaming_split_open_marker_held_back() {
        let mut p = KimiK3Parser::new();
        assert_eq!(
            p.parse_reasoning_streaming_incremental("<|open|>")
                .unwrap()
                .reasoning_text,
            ""
        );
        assert_eq!(
            p.parse_reasoning_streaming_incremental("think")
                .unwrap()
                .reasoning_text,
            ""
        );
        let r = p
            .parse_reasoning_streaming_incremental("<|sep|>step")
            .unwrap();
        assert_eq!(r.reasoning_text, "step");
    }

    #[test]
    fn streaming_split_close_hands_content_downstream() {
        let mut p = KimiK3Parser::new();
        p.parse_reasoning_streaming_incremental("<|open|>think<|sep|>step")
            .unwrap();
        let partial = p
            .parse_reasoning_streaming_incremental("<|close|>")
            .unwrap();
        assert_eq!(partial.reasoning_text, ""); // partial close held back
        assert_eq!(partial.normal_text, "");
        let closed = p
            .parse_reasoning_streaming_incremental("think<|sep|><|open|>response<|sep|>answer")
            .unwrap();
        assert_eq!(closed.reasoning_text, "");
        assert_eq!(closed.normal_text, "answer"); // response-open stripped
    }

    #[test]
    fn streaming_split_message_wrapper_markers_do_not_leak_into_content() {
        // Analogous to `streaming_split_close_hands_content_downstream`, but
        // the marker split during the post-reasoning content phase is the
        // `message` wrapper rather than the think-close marker.
        let mut p = KimiK3Parser::new();
        // Reach the content phase in one shot.
        p.parse_reasoning_streaming_incremental("<|open|>think<|sep|>step<|close|>think<|sep|>")
            .unwrap();

        // Split message-open marker across chunks.
        let partial_open = p
            .parse_reasoning_streaming_incremental("<|open|>mess")
            .unwrap();
        assert_eq!(partial_open.normal_text, ""); // partial open marker held back
        let after_open = p
            .parse_reasoning_streaming_incremental("age<|sep|>answer")
            .unwrap();
        assert_eq!(after_open.normal_text, "answer"); // message-open stripped, no leak

        // Split message-close marker across chunks.
        let partial_close = p
            .parse_reasoning_streaming_incremental(" more<|clo")
            .unwrap();
        assert_eq!(partial_close.normal_text, " more"); // partial close marker held back
        let after_close = p
            .parse_reasoning_streaming_incremental("se|>message<|sep|> tail")
            .unwrap();
        assert_eq!(after_close.normal_text, " tail"); // message-close stripped, no leak
    }

    #[test]
    fn streaming_maintains_in_reasoning_flag() {
        let mut p = KimiK3Parser::new();
        // While accumulating reasoning content, the parser reports in-reasoning.
        p.parse_reasoning_streaming_incremental("<|open|>think<|sep|>step")
            .unwrap();
        assert!(p.is_in_reasoning());
        // A partial close marker is held back but we are still in reasoning.
        p.parse_reasoning_streaming_incremental("<|close|>")
            .unwrap();
        assert!(p.is_in_reasoning());
        // Completing the think-close transition clears the flag.
        p.parse_reasoning_streaming_incremental("think<|sep|><|open|>response<|sep|>answer")
            .unwrap();
        assert!(!p.is_in_reasoning());
    }

    #[test]
    fn streaming_buffer_overflow_is_guarded() {
        let mut p = KimiK3Parser::new();
        let oversized = "a".repeat(DEFAULT_MAX_BUFFER_SIZE + 1);
        let result = p.parse_reasoning_streaming_incremental(&oversized);
        assert!(matches!(
            result,
            Err(ParseError::BufferOverflow(size)) if size == DEFAULT_MAX_BUFFER_SIZE + 1
        ));
    }

    #[test]
    fn reset_clears_state() {
        let mut p = KimiK3Parser::new();
        // Put the parser into reasoning mode via streaming.
        p.parse_reasoning_streaming_incremental("<|open|>think<|sep|>step")
            .unwrap();
        assert!(p.is_in_reasoning() || !p.buffer.is_empty());
        p.reset();
        assert!(!p.is_in_reasoning());
        assert!(p.buffer.is_empty());
        // After reset, a fresh non-streaming parse should work correctly.
        let r = p.detect_and_parse_reasoning("just content").unwrap();
        assert_eq!(r.normal_text, "just content");
        assert_eq!(r.reasoning_text, "");
    }
}
