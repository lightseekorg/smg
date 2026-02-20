// NanoV3 specific reasoning parser.
// Uses the same format as DeepSeek-R1 (<think>...</think>) with initial_in_reasoning=true.

use crate::{
    parsers::BaseReasoningParser,
    traits::{ParseError, ParserConfig, ParserResult, ReasoningParser},
};

/// NanoV3 reasoning parser.
///
/// Uses the same reasoning format as DeepSeek-R1: `<think>...</think>`.
/// Starts with `initial_in_reasoning=true`, assuming all output is reasoning
/// until a `</think>` token is encountered.
pub struct NanoV3Parser {
    base: BaseReasoningParser,
}

impl NanoV3Parser {
    /// Create a new NanoV3 parser.
    pub fn new() -> Self {
        let config = ParserConfig {
            think_start_token: "<think>".to_string(),
            think_end_token: "</think>".to_string(),
            stream_reasoning: true,
            max_buffer_size: 65536,
            initial_in_reasoning: true,
        };

        Self {
            base: BaseReasoningParser::new(config).with_model_type("nano_v3".to_string()),
        }
    }
}

impl Default for NanoV3Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for NanoV3Parser {
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
        self.base.reset()
    }

    fn model_type(&self) -> &str {
        self.base.model_type()
    }

    fn is_in_reasoning(&self) -> bool {
        self.base.is_in_reasoning()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nano_v3_initial_state() {
        let mut parser = NanoV3Parser::new();

        // Should treat text as reasoning even without start token
        let result = parser
            .detect_and_parse_reasoning("This is reasoning content")
            .unwrap();
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "This is reasoning content");
    }

    #[test]
    fn test_nano_v3_with_end_token() {
        let mut parser = NanoV3Parser::new();

        let result = parser
            .detect_and_parse_reasoning("reasoning content</think>answer")
            .unwrap();
        assert_eq!(result.normal_text, "answer");
        assert_eq!(result.reasoning_text, "reasoning content");
    }

    #[test]
    fn test_nano_v3_with_both_tokens() {
        let mut parser = NanoV3Parser::new();

        let result = parser
            .detect_and_parse_reasoning("<think>reasoning content</think>answer")
            .unwrap();
        assert_eq!(result.normal_text, "answer");
        assert_eq!(result.reasoning_text, "reasoning content");
    }

    #[test]
    fn test_nano_v3_streaming() {
        let mut parser = NanoV3Parser::new();

        let result1 = parser
            .parse_reasoning_streaming_incremental("reasoning text ")
            .unwrap();
        assert_eq!(result1.normal_text, "");
        assert_eq!(result1.reasoning_text, "reasoning text ");

        let result2 = parser
            .parse_reasoning_streaming_incremental("more reasoning</think>answer")
            .unwrap();
        assert_eq!(result2.normal_text, "answer");
        assert_eq!(result2.reasoning_text, "more reasoning");
    }

    #[test]
    fn test_model_type() {
        let parser = NanoV3Parser::new();
        assert_eq!(parser.model_type(), "nano_v3");
    }
}
