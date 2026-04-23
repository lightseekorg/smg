//! Request-time built-in tool argument overrides.
//!
//! Built-in tools (like `image_generation`) let callers pin configuration on
//! the request itself — for example, forcing `size: "1024x1024"` or
//! `quality: "low"`. When the model later emits a tool call, its arguments may
//! or may not echo that configuration. To respect the caller's intent, we
//! merge request-level tool config on top of the model-emitted arguments
//! right before dispatching the call to MCP. Request-level values win for
//! overlapping keys.
//!
//! This mechanism is opt-in per `ResponseFormat`. Today it only applies to
//! `ResponseFormat::ImageGenerationCall`; other formats pass through
//! unchanged. Adding a new builtin tool that needs request-pinned arguments
//! means extending `request_tool_overrides`.

use openai_protocol::responses::{ResponseTool, ResponsesRequest};
use serde_json::{to_value, Value};
use smg_mcp::ResponseFormat;

/// Returns request-level builtin-tool argument overrides for the given
/// response format, or `None` if this format does not support pinning.
///
/// For `ResponseFormat::ImageGenerationCall`, this locates the first
/// `ResponseTool::ImageGeneration` entry in the request's `tools` list,
/// serializes the `ImageGenerationTool` struct to a JSON object, drops fields
/// that serialized to `null` (so unset caller fields do not clobber valid
/// model-emitted values), and returns the remaining key-value pairs.
///
/// Because `ImageGenerationTool` uses `#[serde_with::skip_serializing_none]`,
/// unset `Option` fields already serialize as absent, not `null`. The
/// explicit null filter is a defensive second line of defense in case the
/// attribute is ever removed or a wrapped type serializes `null` explicitly.
///
/// The match on `ResponseFormat` is exhaustive; adding a new format requires
/// touching this function to decide whether it supports overrides.
pub(crate) fn request_tool_overrides(
    response_format: &ResponseFormat,
    original_body: &ResponsesRequest,
) -> Option<Value> {
    match response_format {
        ResponseFormat::ImageGenerationCall => {}
        ResponseFormat::Passthrough
        | ResponseFormat::WebSearchCall
        | ResponseFormat::CodeInterpreterCall
        | ResponseFormat::FileSearchCall => return None,
    }

    let tools = original_body.tools.as_ref()?;

    tools.iter().find_map(|tool| {
        let mut serialized = match tool {
            ResponseTool::ImageGeneration(image_tool) => match to_value(image_tool).ok()? {
                Value::Object(obj) => obj,
                _ => return None,
            },
            _ => return None,
        };
        serialized.retain(|_, v| !v.is_null());
        if serialized.is_empty() {
            None
        } else {
            Some(Value::Object(serialized))
        }
    })
}

/// Merge request-level tool-argument overrides into model-emitted tool call
/// arguments, with request-level values winning on key conflicts.
///
/// `arguments` is mutated in place. If the format has no overrides, or the
/// request does not carry a matching tool config, or `arguments` is not a
/// JSON object, this is a no-op.
pub(crate) fn apply_request_tool_overrides(
    response_format: &ResponseFormat,
    original_body: &ResponsesRequest,
    arguments: &mut Value,
) {
    let Some(overrides) = request_tool_overrides(response_format, original_body) else {
        return;
    };
    let Some(args_obj) = arguments.as_object_mut() else {
        return;
    };
    let Some(override_obj) = overrides.as_object() else {
        return;
    };
    for (k, v) in override_obj {
        args_obj.insert(k.clone(), v.clone());
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::responses::{ImageGenerationTool, ResponseTool, ResponsesRequest};
    use serde_json::json;
    use smg_mcp::ResponseFormat;

    use super::{apply_request_tool_overrides, request_tool_overrides};

    fn request_with_image_tool(tool: ImageGenerationTool) -> ResponsesRequest {
        ResponsesRequest {
            tools: Some(vec![ResponseTool::ImageGeneration(tool)]),
            ..Default::default()
        }
    }

    #[test]
    fn overrides_are_none_for_non_image_formats() {
        let request = request_with_image_tool(ImageGenerationTool {
            size: Some("512x512".to_string()),
            ..Default::default()
        });
        for format in [
            ResponseFormat::Passthrough,
            ResponseFormat::WebSearchCall,
            ResponseFormat::CodeInterpreterCall,
            ResponseFormat::FileSearchCall,
        ] {
            assert!(request_tool_overrides(&format, &request).is_none());
        }
    }

    #[test]
    fn overrides_are_none_when_no_image_tool_present() {
        let request = ResponsesRequest::default();
        assert!(request_tool_overrides(&ResponseFormat::ImageGenerationCall, &request).is_none());
    }

    #[test]
    fn overrides_drop_null_fields() {
        // ImageGenerationTool uses skip_serializing_none, so fields that are
        // `None` serialize as absent. Empty tool struct produces no overrides.
        let request = request_with_image_tool(ImageGenerationTool::default());
        assert!(request_tool_overrides(&ResponseFormat::ImageGenerationCall, &request).is_none());
    }

    #[test]
    fn request_values_override_model_arguments() {
        let request = request_with_image_tool(ImageGenerationTool {
            size: Some("512x512".to_string()),
            quality: Some("low".to_string()),
            ..Default::default()
        });
        let mut arguments = json!({
            "size": "1024x1024",
            "quality": "high",
            "prompt": "keep-me"
        });

        apply_request_tool_overrides(
            &ResponseFormat::ImageGenerationCall,
            &request,
            &mut arguments,
        );

        assert_eq!(
            arguments.get("size").and_then(|v| v.as_str()),
            Some("512x512")
        );
        assert_eq!(
            arguments.get("quality").and_then(|v| v.as_str()),
            Some("low")
        );
        assert_eq!(
            arguments.get("prompt").and_then(|v| v.as_str()),
            Some("keep-me")
        );
    }

    #[test]
    fn apply_is_noop_when_arguments_not_object() {
        let request = request_with_image_tool(ImageGenerationTool {
            size: Some("512x512".to_string()),
            ..Default::default()
        });
        let mut arguments = json!("not-an-object");
        apply_request_tool_overrides(
            &ResponseFormat::ImageGenerationCall,
            &request,
            &mut arguments,
        );
        assert_eq!(arguments, json!("not-an-object"));
    }

    #[test]
    fn apply_is_noop_for_non_image_format() {
        let request = request_with_image_tool(ImageGenerationTool {
            size: Some("512x512".to_string()),
            ..Default::default()
        });
        let mut arguments = json!({"size": "1024x1024"});
        apply_request_tool_overrides(&ResponseFormat::WebSearchCall, &request, &mut arguments);
        assert_eq!(
            arguments.get("size").and_then(|v| v.as_str()),
            Some("1024x1024")
        );
    }

    #[test]
    fn first_image_tool_wins_when_multiple_present() {
        let request = ResponsesRequest {
            tools: Some(vec![
                ResponseTool::ImageGeneration(ImageGenerationTool {
                    size: Some("512x512".to_string()),
                    ..Default::default()
                }),
                ResponseTool::ImageGeneration(ImageGenerationTool {
                    size: Some("1024x1024".to_string()),
                    ..Default::default()
                }),
            ]),
            ..Default::default()
        };
        let mut arguments = json!({});
        apply_request_tool_overrides(
            &ResponseFormat::ImageGenerationCall,
            &request,
            &mut arguments,
        );
        assert_eq!(
            arguments.get("size").and_then(|v| v.as_str()),
            Some("512x512")
        );
    }
}
