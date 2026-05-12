// No-op reasoning parser.
//
// Returns all input text as `normal_text` and never produces reasoning text,
// regardless of whether the input contains `<think>`/`</think>` (or any other)
// markers. Use this when the model emits a single content stream and the
// caller does not want any portion of it separated into `reasoning_content`.

use crate::traits::{ParseError, ParserResult, ReasoningParser};

/// Parser that performs no reasoning extraction.
///
/// Every byte received is forwarded to `normal_text`; `reasoning_text` is always
/// empty. State is trivial: no buffering, no tokens, no flags.
#[derive(Debug, Clone, Default)]
pub struct NoneParser;

impl NoneParser {
    pub fn new() -> Self {
        Self
    }
}

impl ReasoningParser for NoneParser {
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
        "none"
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
    fn plain_text_goes_to_normal() {
        let mut parser = NoneParser::new();
        let result = parser
            .detect_and_parse_reasoning("just some content")
            .unwrap();
        assert_eq!(result.normal_text, "just some content");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn think_tags_are_kept_in_normal_text() {
        let mut parser = NoneParser::new();
        let result = parser
            .detect_and_parse_reasoning("<think>cot</think>answer")
            .unwrap();
        assert_eq!(result.normal_text, "<think>cot</think>answer");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn streaming_passes_chunks_through_unchanged() {
        let mut parser = NoneParser::new();

        let r1 = parser
            .parse_reasoning_streaming_incremental("<think>")
            .unwrap();
        assert_eq!(r1.normal_text, "<think>");
        assert_eq!(r1.reasoning_text, "");

        let r2 = parser
            .parse_reasoning_streaming_incremental("hidden cot")
            .unwrap();
        assert_eq!(r2.normal_text, "hidden cot");
        assert_eq!(r2.reasoning_text, "");

        let r3 = parser
            .parse_reasoning_streaming_incremental("</think>visible")
            .unwrap();
        assert_eq!(r3.normal_text, "</think>visible");
        assert_eq!(r3.reasoning_text, "");
    }

    #[test]
    fn empty_input_is_normal_and_empty() {
        let mut parser = NoneParser::new();
        let result = parser.detect_and_parse_reasoning("").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn mark_helpers_do_not_change_behavior() {
        let mut parser = NoneParser::new();
        parser.mark_reasoning_started();
        parser.mark_think_start_stripped();
        assert!(!parser.is_in_reasoning());

        let result = parser
            .detect_and_parse_reasoning("<think>x</think>y")
            .unwrap();
        assert_eq!(result.normal_text, "<think>x</think>y");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn model_type_is_none() {
        let parser = NoneParser::new();
        assert_eq!(parser.model_type(), "none");
    }
}
