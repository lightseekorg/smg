// GLM reasoning parser.
// Uses the same format as Qwen3 but has its own implementation for debugging.

use crate::{
    parsers::BaseReasoningParser,
    traits::{ParseError, ParserConfig, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE},
};

/// GLM reasoning parser.
///
/// GLM series (4.5, 4.7, 5, 5.1, etc.) all use the same format as Qwen3
/// (<think>...</think>) but has its own implementation for better debugging
/// and potential future customization.
pub struct GlmParser {
    base: BaseReasoningParser,
}

impl GlmParser {
    /// Create a new GLM parser.
    pub fn new() -> Self {
        let config = ParserConfig {
            think_start_token: "<think>".to_string(),
            think_end_token: "</think>".to_string(),
            stream_reasoning: true,
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
            always_in_reasoning: false,
        };

        Self {
            base: BaseReasoningParser::new(config).with_model_type("glm".to_string()),
        }
    }
}

impl Default for GlmParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for GlmParser {
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
    fn test_glm_initial_state() {
        let mut parser = GlmParser::new();

        // Should NOT treat text as reasoning without start token
        let result = parser
            .detect_and_parse_reasoning("This is normal content")
            .unwrap();
        assert_eq!(result.normal_text, "This is normal content");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn test_glm_with_tokens() {
        let mut parser = GlmParser::new();

        // Should extract reasoning with proper tokens
        let result = parser
            .detect_and_parse_reasoning("<think>reasoning content</think>answer")
            .unwrap();
        assert_eq!(result.normal_text, "answer");
        assert_eq!(result.reasoning_text, "reasoning content");
    }

    #[test]
    fn test_glm_streaming() {
        let mut parser = GlmParser::new();

        // First chunk - normal text
        let result1 = parser
            .parse_reasoning_streaming_incremental("normal text ")
            .unwrap();
        assert_eq!(result1.normal_text, "normal text ");
        assert_eq!(result1.reasoning_text, "");

        // Second chunk - enters reasoning
        let result2 = parser
            .parse_reasoning_streaming_incremental("<think>reasoning")
            .unwrap();
        assert_eq!(result2.normal_text, "");
        assert_eq!(result2.reasoning_text, "reasoning");

        // Third chunk - exits reasoning
        let result3 = parser
            .parse_reasoning_streaming_incremental("</think>answer")
            .unwrap();
        assert_eq!(result3.normal_text, "answer");
        assert_eq!(result3.reasoning_text, "");
    }

    #[test]
    fn test_model_type() {
        let parser = GlmParser::new();
        assert_eq!(parser.model_type(), "glm");
    }
}
