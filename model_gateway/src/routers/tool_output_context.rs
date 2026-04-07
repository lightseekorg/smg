use smg_mcp::ResponseFormat;
use serde_json::{json, Value};

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
    if let Some(error) = find_error_text(output) {
        return ("failed", Some(error));
    }

    if let Some(status) = output
        .as_object()
        .and_then(|o| o.get("status"))
        .and_then(|v| v.as_str())
    {
        if status.eq_ignore_ascii_case("failed") || status.eq_ignore_ascii_case("error") {
            return ("failed", None);
        }
        return ("completed", None);
    }

    ("completed", None)
}

fn find_error_text(output: &Value) -> Option<String> {
    match output {
        Value::Object(obj) => {
            if let Some(err) = obj.get("error").and_then(|v| v.as_str()) {
                return Some(err.to_string());
            }
            if let Some(result) = obj.get("result").and_then(|v| v.as_str()) {
                if result.contains("Error executing tool") {
                    return Some(result.to_string());
                }
            }
            None
        }
        Value::Array(items) => items.iter().find_map(|item| {
            let text = item
                .as_object()
                .and_then(|o| o.get("text"))
                .and_then(|v| v.as_str());
            if text.is_some_and(|t| t.contains("Error executing tool")) {
                text.map(str::to_string)
            } else {
                None
            }
        }),
        Value::String(text) => text
            .contains("Error executing tool")
            .then_some(text.to_string()),
        _ => None,
    }
}
