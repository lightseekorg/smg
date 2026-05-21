// Passthrough reasoning parser.
//
// Forwards all input as `normal_text` byte-faithfully and never produces
// `reasoning_text`, regardless of whether the input contains `<think>` /
// `</think>` (or any other) markers. Used both as the explicit
// `--reasoning-parser passthrough` option and as the fallback for models that
// don't match any registered pattern.

use crate::traits::{ParseError, ParserResult, ReasoningParser};

/// Parser that performs no reasoning extraction.
///
/// Every byte received is forwarded to `normal_text` unchanged;
/// `reasoning_text` is always empty. Has no internal state.
#[derive(Debug, Clone, Default)]
pub struct PassthroughParser;

impl PassthroughParser {
    pub fn new() -> Self {
        Self
    }
}

impl ReasoningParser for PassthroughParser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        Ok(ParserResult::normal(text.to_string()))
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        Ok(ParserResult::normal(text.to_string()))
    }

    fn reset(&mut self) {}

    fn model_type(&self) -> &str {
        "passthrough"
    }

    fn is_in_reasoning(&self) -> bool {
        false
    }

    fn mark_reasoning_started(&mut self) {}

    fn mark_think_start_stripped(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_whitespace_non_streaming() {
        let mut parser = PassthroughParser::new();
        let result = parser
            .detect_and_parse_reasoning("  leading and trailing  \n")
            .unwrap();
        assert_eq!(result.normal_text, "  leading and trailing  \n");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn keeps_think_tags_in_normal_text() {
        let mut parser = PassthroughParser::new();
        let result = parser
            .detect_and_parse_reasoning("<think>cot</think>answer")
            .unwrap();
        assert_eq!(result.normal_text, "<think>cot</think>answer");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn streaming_preserves_whitespace_across_chunks() {
        let mut parser = PassthroughParser::new();

        let r1 = parser
            .parse_reasoning_streaming_incremental("  hello")
            .unwrap();
        assert_eq!(r1.normal_text, "  hello");
        assert_eq!(r1.reasoning_text, "");

        let r2 = parser
            .parse_reasoning_streaming_incremental(" world  \n")
            .unwrap();
        assert_eq!(r2.normal_text, " world  \n");
        assert_eq!(r2.reasoning_text, "");
    }

    #[test]
    fn empty_input_yields_empty_result() {
        let mut parser = PassthroughParser::new();
        let result = parser.detect_and_parse_reasoning("").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn model_type_is_passthrough() {
        let parser = PassthroughParser::new();
        assert_eq!(parser.model_type(), "passthrough");
    }
}
