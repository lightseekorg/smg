//! Response transformation types.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::debug;

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
    pub fn compact_tool_output_for_model_context(&self, output: &Value) -> String {
        let input = output.to_string();
        let result = if matches!(self, ResponseFormat::ImageGenerationCall) {
            // image tool outputs are wrapped as content blocks
            // [{"type":"text","text":"{\"result\":\"<base64>\",\"status\":\"completed\"}"}]
            // Parse each `text` JSON payload and remove `result` to avoid large base64 in model context.
            let items = output
                .as_array()
                .expect("ImageGenerationCall output must be a wrapped content array");
            let mut sanitized = items.clone();
            for item in &mut sanitized {
                let Some(item_obj) = item.as_object_mut() else {
                    continue;
                };
                if item_obj.remove("result").is_some() {
                    continue;
                }
                let Some(text) = item_obj.get("text").and_then(Value::as_str) else {
                    continue;
                };
                if let Ok(Value::Object(mut obj)) = serde_json::from_str::<Value>(text) {
                    obj.remove("result");
                    item_obj.insert(
                        "text".to_string(),
                        Value::String(Value::Object(obj).to_string()),
                    );
                }
            }
            Value::Array(sanitized).to_string()
        } else {
            output.to_string()
        };

        debug!(
            response_format = ?self,
            input = %input,
            compressed_result = %result,
            "compact_tool_output_for_model_context"
        );

        result
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
