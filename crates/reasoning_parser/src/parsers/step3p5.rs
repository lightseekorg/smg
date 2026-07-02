// Step-3.5 specific reasoning parser.

use crate::{
    parsers::BaseReasoningParser,
    traits::{ParseError, ParserConfig, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE},
};

/// Step-3.5 reasoning parser.
///
/// Ported from vLLM's `Step3p5ReasoningParser`, which extends
/// `BaseThinkingReasoningParser` with `<think>` / `</think>` tokens. Step-3.5
/// emits reasoning at the start of generation (the template injects the thinking
/// prefix), so output begins inside reasoning even when the literal `<think>`
/// token is absent — modeled here with `always_in_reasoning: true`, matching the
/// existing `step3` parser.
pub struct Step3p5Parser {
    base: BaseReasoningParser,
}

impl Step3p5Parser {
    /// Create a new Step-3.5 parser.
    pub fn new() -> Self {
        let config = ParserConfig {
            think_start_token: "<think>".to_string(),
            think_end_token: "</think>".to_string(),
            stream_reasoning: true,
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
            always_in_reasoning: true,
        };

        Self {
            base: BaseReasoningParser::new(config).with_model_type("step3p5".to_string()),
        }
    }
}

impl Default for Step3p5Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for Step3p5Parser {
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
        let parser = Step3p5Parser::new();
        assert_eq!(parser.model_type(), "step3p5");
    }

    #[test]
    fn test_initial_state_is_in_reasoning() {
        let parser = Step3p5Parser::new();
        // always_in_reasoning=true seeds in_reasoning on a fresh parser.
        assert!(parser.is_in_reasoning());
    }

    #[test]
    fn test_reasoning_without_start_token() {
        let mut parser = Step3p5Parser::new();

        // Output begins inside reasoning even without a leading <think>.
        let result = parser
            .detect_and_parse_reasoning("reasoning content</think>answer")
            .unwrap();
        assert_eq!(result.reasoning_text, "reasoning content");
        assert_eq!(result.normal_text, "answer");
    }

    #[test]
    fn test_reasoning_with_both_tokens() {
        let mut parser = Step3p5Parser::new();

        let result = parser
            .detect_and_parse_reasoning("<think>thinking here</think>normal text")
            .unwrap();
        assert_eq!(result.reasoning_text, "thinking here");
        assert_eq!(result.normal_text, "normal text");
    }

    #[test]
    fn test_no_think_passthrough() {
        let mut parser = Step3p5Parser::new();

        // No end token: everything is reasoning (always_in_reasoning=true).
        let result = parser
            .detect_and_parse_reasoning("just reasoning, no end token")
            .unwrap();
        assert_eq!(result.reasoning_text, "just reasoning, no end token");
        assert_eq!(result.normal_text, "");
    }

    #[test]
    fn test_streaming_incremental() {
        let mut parser = Step3p5Parser::new();

        // First chunk - treated as reasoning.
        let result1 = parser
            .parse_reasoning_streaming_incremental("reasoning text ")
            .unwrap();
        assert_eq!(result1.normal_text, "");
        assert_eq!(result1.reasoning_text, "reasoning text ");

        // Second chunk - continues reasoning until end token, then content.
        let result2 = parser
            .parse_reasoning_streaming_incremental("more</think>answer")
            .unwrap();
        assert_eq!(result2.normal_text, "answer");
        assert_eq!(result2.reasoning_text, "more");
    }

    #[test]
    fn test_reset() {
        let mut parser = Step3p5Parser::new();

        parser
            .parse_reasoning_streaming_incremental("partial reasoning")
            .unwrap();
        parser.reset();

        // After reset, in_reasoning is restored to the always_in_reasoning flag.
        assert!(parser.is_in_reasoning());

        let result = parser
            .detect_and_parse_reasoning("fresh</think>content")
            .unwrap();
        assert_eq!(result.reasoning_text, "fresh");
        assert_eq!(result.normal_text, "content");
    }
}
