use openai_protocol::responses::{ResponseTool, ResponsesRequest};
use serde_json::{to_value, Value};
use smg_mcp::ResponseFormat;

/// Returns request-level builtin-tool argument overrides for the given response format.
///
/// Input source:
/// - `original_body.tools` from the incoming Responses request.
///
/// Extensibility note:
/// - This function intentionally uses exhaustive matching on `ResponseFormat`.
///   When a new `ResponseFormat` variant is added, Rust will require this
///   function to handle it explicitly (or intentionally return `None`).
///
/// Current behavior:
/// - `ResponseFormat::ImageGenerationCall`: reads the first
///   `ResponseTool::ImageGeneration` entry, serializes it to a JSON object,
///   removes null fields, and returns the remaining key-value pairs.
/// - All other response formats: returns `None`.
///
/// Return value:
/// - `Some(Value::Object(...))` when non-null overrides are present.
/// - `None` when no applicable overrides exist.
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
        // Serialize image tool config into a JSON object for merge.
        let mut serialized = match tool {
            ResponseTool::ImageGeneration(image_tool) => match to_value(image_tool).ok()? {
                Value::Object(obj) => obj,
                _ => return None,
            },
            _ => return None,
        };
        // Drop nulls so absent fields do not overwrite generated call arguments.
        serialized.retain(|_, v| !v.is_null());
        if serialized.is_empty() {
            None
        } else {
            Some(Value::Object(serialized))
        }
    })
}

pub(crate) fn apply_request_tool_overrides(
    response_format: &ResponseFormat,
    original_body: &ResponsesRequest,
    arguments: &mut Value,
) {
    // Merge order is intentional: request-level tool config wins over model-emitted
    // tool-call arguments for overlapping keys. This lets API callers pin
    // tool behavior (for example, image size/quality) even if the model chooses
    // different values in the generated function call.
    if let (Some(overrides), Some(args_obj)) = (
        request_tool_overrides(response_format, original_body),
        arguments.as_object_mut(),
    ) {
        let Some(override_obj) = overrides.as_object() else {
            return;
        };
        for (k, v) in override_obj {
            args_obj.insert(k.clone(), v.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::responses::{ImageGenerationTool, ResponseTool, ResponsesRequest};
    use serde_json::json;
    use smg_mcp::ResponseFormat;

    use super::apply_request_tool_overrides;

    #[test]
    fn apply_request_tool_overrides_request_values_override_model_arguments() {
        let request = ResponsesRequest {
            tools: Some(vec![ResponseTool::ImageGeneration(ImageGenerationTool {
                size: Some("512x512".to_string()),
                quality: Some("low".to_string()),
                ..Default::default()
            })]),
            ..Default::default()
        };
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
}
