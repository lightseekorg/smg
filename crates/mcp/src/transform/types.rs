//! Response transformation types.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::transformer::extract_image_generation_fallback_text;
use crate::core::config::ResponseFormatConfig;

/// Format for transforming MCP responses to API-specific formats.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Pass through MCP result unchanged as mcp_call output
    #[default]
    Passthrough,
    /// Transform to OpenAI web_search_call format
    WebSearchCall,
    /// Transform to OpenAI code_interpreter_call format
    CodeInterpreterCall,
    /// Transform to OpenAI image_generation_call format
    ImageGenerationCall,
    /// Transform to OpenAI file_search_call format
    FileSearchCall,
}

impl ResponseFormat {
    /// Compact tool output before feeding it back into model context.
    pub fn compact_tool_output_for_model_context(&self, is_error: bool, output: &Value) -> String {
        if matches!(self, ResponseFormat::ImageGenerationCall) && is_error {
            if let Some(text) = extract_image_generation_fallback_text(output) {
                return text;
            }
        }
        if is_error {
            match output {
                Value::String(text) => text.clone(),
                _ => serde_json::to_string(output).unwrap_or_else(|e| {
                    format!("{{\"error\": \"Failed to serialize tool output: {e}\"}}")
                }),
            }
        } else {
            match output {
                Value::String(text) => text.clone(),
                _ => output.to_string(),
            }
        }
    }
}

impl From<ResponseFormatConfig> for ResponseFormat {
    fn from(config: ResponseFormatConfig) -> Self {
        match config {
            ResponseFormatConfig::Passthrough => ResponseFormat::Passthrough,
            ResponseFormatConfig::WebSearchCall => ResponseFormat::WebSearchCall,
            ResponseFormatConfig::CodeInterpreterCall => ResponseFormat::CodeInterpreterCall,
            ResponseFormatConfig::ImageGenerationCall => ResponseFormat::ImageGenerationCall,
            ResponseFormatConfig::FileSearchCall => ResponseFormat::FileSearchCall,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_format_serde() {
        let formats = vec![
            (ResponseFormat::Passthrough, "\"passthrough\""),
            (ResponseFormat::WebSearchCall, "\"web_search_call\""),
            (
                ResponseFormat::CodeInterpreterCall,
                "\"code_interpreter_call\"",
            ),
            (
                ResponseFormat::ImageGenerationCall,
                "\"image_generation_call\"",
            ),
            (ResponseFormat::FileSearchCall, "\"file_search_call\""),
        ];

        for (format, expected) in formats {
            let serialized = serde_json::to_string(&format).unwrap();
            assert_eq!(serialized, expected);

            let deserialized: ResponseFormat = serde_json::from_str(&serialized).unwrap();
            assert_eq!(deserialized, format);
        }
    }

    #[test]
    fn test_response_format_default() {
        assert_eq!(ResponseFormat::default(), ResponseFormat::Passthrough);
    }
}
