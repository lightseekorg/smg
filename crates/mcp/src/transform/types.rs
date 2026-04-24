//! Response transformation types.

use serde::{Deserialize, Serialize};

use crate::core::config::{BuiltinToolType, ResponseFormatConfig};

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
    /// Transform to OpenAI file_search_call format
    FileSearchCall,
    /// Transform to OpenAI image_generation_call format
    ImageGenerationCall,
}

impl ResponseFormat {
    /// Inverse of [`BuiltinToolType::response_format`]: returns the hosted-tool
    /// kind this response format corresponds to, or `None` for the non-hosted
    /// `Passthrough` format.
    ///
    /// Router dispatch paths use this to look up caller-declared overrides
    /// for the current tool's hosted-tool kind without threading
    /// [`BuiltinToolType`] separately through session bindings.
    pub fn to_builtin_tool_type(&self) -> Option<BuiltinToolType> {
        match self {
            ResponseFormat::Passthrough => None,
            ResponseFormat::WebSearchCall => Some(BuiltinToolType::WebSearchPreview),
            ResponseFormat::CodeInterpreterCall => Some(BuiltinToolType::CodeInterpreter),
            ResponseFormat::FileSearchCall => Some(BuiltinToolType::FileSearch),
            ResponseFormat::ImageGenerationCall => Some(BuiltinToolType::ImageGeneration),
        }
    }
}

impl From<ResponseFormatConfig> for ResponseFormat {
    fn from(config: ResponseFormatConfig) -> Self {
        match config {
            ResponseFormatConfig::Passthrough => ResponseFormat::Passthrough,
            ResponseFormatConfig::WebSearchCall => ResponseFormat::WebSearchCall,
            ResponseFormatConfig::CodeInterpreterCall => ResponseFormat::CodeInterpreterCall,
            ResponseFormatConfig::FileSearchCall => ResponseFormat::FileSearchCall,
            ResponseFormatConfig::ImageGenerationCall => ResponseFormat::ImageGenerationCall,
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
            (ResponseFormat::FileSearchCall, "\"file_search_call\""),
            (
                ResponseFormat::ImageGenerationCall,
                "\"image_generation_call\"",
            ),
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

    #[test]
    fn test_to_builtin_tool_type_round_trip() {
        // Non-passthrough formats should round-trip cleanly through
        // BuiltinToolType::response_format.
        let kinds = [
            BuiltinToolType::WebSearchPreview,
            BuiltinToolType::CodeInterpreter,
            BuiltinToolType::FileSearch,
            BuiltinToolType::ImageGeneration,
        ];
        for kind in kinds {
            let fmt: ResponseFormat = kind.response_format().into();
            assert_eq!(fmt.to_builtin_tool_type(), Some(kind));
        }

        // Passthrough is not a hosted-tool kind.
        assert_eq!(ResponseFormat::Passthrough.to_builtin_tool_type(), None);
    }
}
