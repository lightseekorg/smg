// Mistral / Magistral specific reasoning parser.
// Mistral/Magistral reasoning models emit reasoning between `[THINK]` and
// `[/THINK]` markers. A valid reasoning trace always starts with `[THINK]`;
// if the `[THINK]` token is never generated, all output is normal content.
// The template does NOT prefill the start token, so always_in_reasoning=false.

use crate::{
    parsers::BaseReasoningParser,
    traits::{ParseError, ParserConfig, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE},
};

/// Mistral / Magistral reasoning parser.
///
/// Uses `[THINK]` / `[/THINK]` markers (vLLM `SpecialTokens.begin_think` /
/// `SpecialTokens.end_think`). Output starts as normal content; reasoning is
/// only entered when an explicit `[THINK]` token appears, so this parser uses
/// `always_in_reasoning=false`.
pub struct MistralParser {
    base: BaseReasoningParser,
}

impl MistralParser {
    /// Create a new Mistral / Magistral parser.
    pub fn new() -> Self {
        let config = ParserConfig {
            think_start_token: "[THINK]".to_string(),
            think_end_token: "[/THINK]".to_string(),
            stream_reasoning: true,
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
            always_in_reasoning: false,
        };

        Self {
            base: BaseReasoningParser::new(config).with_model_type("mistral".to_string()),
        }
    }
}

impl Default for MistralParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for MistralParser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        self.base.detect_and_parse_reasoning(text)
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        self.base.parse_reasoning_streaming_incremental(text)
    }

    fn reset(&mut self) {
        self.base.reset();
    }

    fn model_type(&self) -> &str {
        self.base.model_type()
    }

    fn is_in_reasoning(&self) -> bool {
        self.base.is_in_reasoning()
    }

    fn mark_reasoning_started(&mut self) {
        self.base.mark_reasoning_started();
    }

    fn mark_think_start_stripped(&mut self) {
        self.base.mark_think_start_stripped();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type() {
        let parser = MistralParser::new();
        assert_eq!(parser.model_type(), "mistral");
    }

    #[test]
    fn test_fresh_parser_not_in_reasoning() {
        // always_in_reasoning=false: a fresh parser starts outside reasoning.
        let parser = MistralParser::new();
        assert!(!parser.is_in_reasoning());
    }

    #[test]
    fn test_complete_extraction() {
        let mut parser = MistralParser::new();
        let result = parser
            .detect_and_parse_reasoning("[THINK]reasoning here[/THINK]normal text")
            .unwrap();
        assert_eq!(result.reasoning_text, "reasoning here");
        assert_eq!(result.normal_text, "normal text");
    }

    #[test]
    fn test_complete_extraction_preserves_whitespace() {
        let mut parser = MistralParser::new();
        let result = parser
            .detect_and_parse_reasoning("[THINK]foo[/THINK] hello world")
            .unwrap();
        assert_eq!(result.reasoning_text, "foo");
        assert_eq!(result.normal_text, " hello world");
    }

    #[test]
    fn test_no_reasoning_passthrough() {
        // No [THINK] token at all => everything is normal content.
        let mut parser = MistralParser::new();
        let result = parser
            .detect_and_parse_reasoning("just a plain answer with no reasoning")
            .unwrap();
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "just a plain answer with no reasoning");
    }

    #[test]
    fn test_truncated_reasoning_no_end_token() {
        let mut parser = MistralParser::new();
        let result = parser
            .detect_and_parse_reasoning("[THINK]reasoning was cut off")
            .unwrap();
        assert_eq!(result.reasoning_text, "reasoning was cut off");
        assert_eq!(result.normal_text, "");
    }

    #[test]
    fn test_streaming_incremental() {
        let mut parser = MistralParser::new();

        // Start token + reasoning content in one chunk.
        let result1 = parser
            .parse_reasoning_streaming_incremental("[THINK]thinking about")
            .unwrap();
        assert_eq!(result1.reasoning_text, "thinking about");
        assert_eq!(result1.normal_text, "");

        // More reasoning content.
        let result2 = parser
            .parse_reasoning_streaming_incremental(" the problem")
            .unwrap();
        assert_eq!(result2.reasoning_text, " the problem");
        assert_eq!(result2.normal_text, "");

        // End token followed by normal text.
        let result3 = parser
            .parse_reasoning_streaming_incremental("[/THINK]the answer")
            .unwrap();
        assert_eq!(result3.reasoning_text, "");
        assert_eq!(result3.normal_text, "the answer");
    }

    #[test]
    fn test_streaming_split_inside_start_marker() {
        let mut parser = MistralParser::new();

        // "[THINK]" arrives split across chunk boundaries: "[TH" then "INK]".
        let r1 = parser.parse_reasoning_streaming_incremental("[TH").unwrap();
        // Partial start token must be buffered, emitting nothing yet.
        assert_eq!(r1.reasoning_text, "");
        assert_eq!(r1.normal_text, "");

        let r2 = parser
            .parse_reasoning_streaming_incremental("INK]reasoning")
            .unwrap();
        assert_eq!(r2.reasoning_text, "reasoning");
        assert_eq!(r2.normal_text, "");
    }

    #[test]
    fn test_streaming_split_inside_end_marker() {
        let mut parser = MistralParser::new();

        // Enter reasoning.
        let r1 = parser
            .parse_reasoning_streaming_incremental("[THINK]deep thought")
            .unwrap();
        assert_eq!(r1.reasoning_text, "deep thought");
        assert_eq!(r1.normal_text, "");

        // End marker "[/THINK]" split across boundaries: "[/TH" then "INK]done".
        let r2 = parser
            .parse_reasoning_streaming_incremental("[/TH")
            .unwrap();
        // Partial end token must be buffered, emitting nothing yet.
        assert_eq!(r2.reasoning_text, "");
        assert_eq!(r2.normal_text, "");

        let r3 = parser
            .parse_reasoning_streaming_incremental("INK]done")
            .unwrap();
        assert_eq!(r3.reasoning_text, "");
        assert_eq!(r3.normal_text, "done");
    }

    #[test]
    fn test_streaming_no_reasoning_passthrough() {
        let mut parser = MistralParser::new();
        let result = parser
            .parse_reasoning_streaming_incremental("plain content")
            .unwrap();
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "plain content");
    }

    #[test]
    fn test_reset_behavior() {
        let mut parser = MistralParser::new();

        // Drive the parser into reasoning state.
        parser
            .parse_reasoning_streaming_incremental("[THINK]partial reasoning")
            .unwrap();
        assert!(parser.is_in_reasoning());

        // Reset restores the always_in_reasoning=false starting state.
        parser.reset();
        assert!(!parser.is_in_reasoning());

        // After reset, parsing starts cleanly again.
        let result = parser
            .detect_and_parse_reasoning("[THINK]again[/THINK]answer")
            .unwrap();
        assert_eq!(result.reasoning_text, "again");
        assert_eq!(result.normal_text, "answer");
    }
}
