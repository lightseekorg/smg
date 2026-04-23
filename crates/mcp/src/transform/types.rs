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
    /// Transform to OpenAI file_search_call format
    FileSearchCall,
    /// Transform to OpenAI image_generation_call format
    ImageGenerationCall,
}

impl ResponseFormat {
    /// Compact the raw MCP tool output before feeding it back into model
    /// context on subsequent turns.
    ///
    /// For most formats this is a no-op — the full raw output is fed back
    /// verbatim (it's a short JSON blob, not a multi-megabyte base64).
    ///
    /// For `ImageGenerationCall`, the MCP tool response is of the shape
    /// `[{"type":"text","text":"{\"result\":\"<base64>\",\"revised_prompt\":\"...\"}"}]`.
    /// Echoing the full base64 into subsequent model context would waste
    /// many thousands of tokens per turn; the model only needs to know that
    /// an image was generated (plus any `revised_prompt` it rewrote).
    /// This strips `result` / `data` (the base64 payloads) from the text
    /// blocks, keeping status and revised prompt.
    pub fn compact_tool_output_for_model_context(&self, output: &Value) -> String {
        let compacted = match self {
            ResponseFormat::Passthrough
            | ResponseFormat::WebSearchCall
            | ResponseFormat::CodeInterpreterCall
            | ResponseFormat::FileSearchCall => output.to_string(),
            ResponseFormat::ImageGenerationCall => compact_image_generation(output),
        };

        debug!(
            response_format = ?self,
            input_len = output.to_string().len(),
            compacted_len = compacted.len(),
            "compact_tool_output_for_model_context"
        );

        compacted
    }
}

/// Strip base64 image payloads from MCP image-generation tool output so the
/// model only sees metadata (status, revised_prompt) when the transcript
/// replays on subsequent turns.
///
/// Input shapes we tolerate:
/// - `[{"type":"text","text":"{\"result\":\"<base64>\",\"revised_prompt\":\"...\"}"}]`
///   — typical success: stringified JSON embedded in a text content block.
/// - `[{"type":"text","result":"<base64>", ...}]` — flat variant where
///   `result`/`data` sit directly on the content item.
///
/// Any shape we do not recognize falls through unchanged via
/// `output.to_string()` so we never silently drop information.
fn compact_image_generation(output: &Value) -> String {
    let Some(items) = output.as_array() else {
        return output.to_string();
    };

    let mut sanitized = items.clone();
    for item in &mut sanitized {
        let Some(item_obj) = item.as_object_mut() else {
            continue;
        };

        // Flat variant: strip base64 keys directly off the item.
        let stripped_flat = item_obj.remove("result").is_some() | item_obj.remove("data").is_some();
        if stripped_flat {
            continue;
        }

        // Embedded variant: parse the `text` field as JSON, drop base64 keys,
        // re-stringify.
        let Some(text) = item_obj.get("text").and_then(Value::as_str) else {
            continue;
        };
        let Ok(Value::Object(mut obj)) = serde_json::from_str::<Value>(text) else {
            continue;
        };
        obj.remove("result");
        obj.remove("data");
        item_obj.insert(
            "text".to_string(),
            Value::String(Value::Object(obj).to_string()),
        );
    }
    Value::Array(sanitized).to_string()
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
    use serde_json::json;

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
    fn compact_image_generation_strips_base64_from_embedded_text() {
        let output = json!([
            {
                "type": "text",
                "text": "{\"result\":\"AAAAAAAA\",\"revised_prompt\":\"a cat\",\"status\":\"completed\"}"
            }
        ]);
        let compacted =
            ResponseFormat::ImageGenerationCall.compact_tool_output_for_model_context(&output);
        assert!(!compacted.contains("AAAAAAAA"));
        assert!(compacted.contains("revised_prompt"));
        assert!(compacted.contains("a cat"));
    }

    #[test]
    fn compact_image_generation_strips_flat_result_and_data() {
        let output = json!([
            {"type": "image", "result": "BBBBBBBB", "data": "CCCCCCCC", "size": "1024x1024"}
        ]);
        let compacted =
            ResponseFormat::ImageGenerationCall.compact_tool_output_for_model_context(&output);
        assert!(!compacted.contains("BBBBBBBB"));
        assert!(!compacted.contains("CCCCCCCC"));
        assert!(compacted.contains("1024x1024"));
    }

    #[test]
    fn compact_noop_formats_preserve_output_verbatim() {
        let output = json!({"result": "keep-me", "status": "completed"});
        for format in [
            ResponseFormat::Passthrough,
            ResponseFormat::WebSearchCall,
            ResponseFormat::CodeInterpreterCall,
            ResponseFormat::FileSearchCall,
        ] {
            let compacted = format.compact_tool_output_for_model_context(&output);
            assert_eq!(compacted, output.to_string());
        }
    }
}
