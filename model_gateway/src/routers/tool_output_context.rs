use serde_json::{json, Value};
use smg_mcp::ResponseFormat;

/// Build tool output text for model context.
///
/// This is format-driven and intended to support per-tool compaction policies.
/// Currently only `ResponseFormat::ImageGenerationCall` is compacted into a
/// minimal fixed summary (no payload/status) to avoid feeding large binary
/// image data back into the next model turn. Other formats are currently no-op
/// and return `output.to_string()` unchanged.
pub fn compact_tool_output_for_model_context(
    response_format: &ResponseFormat,
    output: &Value,
) -> String {
    match response_format {
        ResponseFormat::ImageGenerationCall => {
            let (status, error) = extract_image_status_and_error(output);
            let mut summary = json!({
                "tool": "generate_image",
                "status": status,
                "note": "binary image payload omitted from model context"
            });
            if let Some(err) = error {
                summary["error"] = json!(err);
            }
            summary.to_string()
        }
        // No-op for other tools for now: preserve raw string outputs as-is.
        _ => match output {
            Value::String(text) => text.clone(),
            _ => output.to_string(),
        },
    }
}

fn extract_image_status_and_error(output: &Value) -> (&'static str, Option<String>) {
    if output
        .as_object()
        .and_then(|o| o.get("result"))
        .and_then(|v| v.as_object())
        .and_then(|r| r.get("isError"))
        .and_then(|v| v.as_bool())
        == Some(true)
    {
        let error_text = output
            .as_object()
            .and_then(|obj| obj.get("result"))
            .and_then(|v| v.as_object())
            .and_then(|result_obj| {
                result_obj
                    .get("error")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
                    .or_else(|| {
                        result_obj
                            .get("content")
                            .and_then(|v| v.as_array())
                            .and_then(|content| {
                                content.iter().find_map(|item| {
                                    item.as_object()
                                        .and_then(|o| o.get("text"))
                                        .and_then(|v| v.as_str())
                                        .filter(|t| !t.trim().is_empty())
                                        .map(str::to_string)
                                })
                            })
                    })
            });
        return ("failed", error_text);
    }

    ("completed", None)
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use smg_mcp::ResponseFormat;

    use super::compact_tool_output_for_model_context;

    #[test]
    fn test_compact_image_output_with_jsonrpc_wrapped_text_error() {
        let input = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": "Error executing tool generate_image: 1 validation error for generate_imageArguments\noutput_format\n  Input should be 'png', 'jpeg' or 'webp' [type=literal_error, input_value='spng', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.12/v/literal_error"
                    }
                ],
                "isError": true
            }
        });

        let compact =
            compact_tool_output_for_model_context(&ResponseFormat::ImageGenerationCall, &input);
        let parsed: serde_json::Value =
            serde_json::from_str(&compact).expect("compact output should be valid JSON");

        assert_eq!(
            parsed.get("status").and_then(|v| v.as_str()),
            Some("failed")
        );
        assert!(parsed
            .get("error")
            .and_then(|v| v.as_str())
            .is_some_and(|s| s.contains("Input should be 'png', 'jpeg' or 'webp'")));
    }

    #[test]
    fn test_compact_image_output_with_jsonrpc_wrapped_text_non_error() {
        let input = json!({
            "jsonrpc": "2.0",
            "id": 4,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": "{\"result\":\"ZmFrZV9iYXNlNjQ=\"}"
                    }
                ],
                "isError": false
            }
        });

        let compact =
            compact_tool_output_for_model_context(&ResponseFormat::ImageGenerationCall, &input);
        let parsed: serde_json::Value =
            serde_json::from_str(&compact).expect("compact output should be valid JSON");

        assert_eq!(
            parsed.get("status").and_then(|v| v.as_str()),
            Some("completed")
        );
        assert!(parsed.get("error").is_none());
    }
}
