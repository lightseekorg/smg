use serde_json::{json, Value};

const MAX_TOOL_OUTPUT_CONTEXT_CHARS: usize = 4096;

pub fn compact_tool_output_for_model_context(
    is_image_generation: bool,
    output: &Value,
    is_error: bool,
) -> String {
    if is_image_generation {
        let status = output
            .as_object()
            .and_then(|o| o.get("status"))
            .and_then(|v| v.as_str())
            .unwrap_or(if is_error { "failed" } else { "completed" });
        let error = output
            .as_object()
            .and_then(|o| o.get("error"))
            .and_then(|v| v.as_str());
        let has_result = output
            .as_object()
            .and_then(|o| o.get("result"))
            .is_some();
        return json!({
            "tool": "generate_image",
            "status": status,
            "has_result": has_result,
            "error": error,
            "note": "binary image payload omitted from model context"
        })
        .to_string();
    }

    let raw = output.to_string();
    if raw.chars().count() > MAX_TOOL_OUTPUT_CONTEXT_CHARS {
        let preview: String = raw.chars().take(MAX_TOOL_OUTPUT_CONTEXT_CHARS).collect();
        json!({
            "truncated": true,
            "original_chars": raw.chars().count(),
            "preview": preview
        })
        .to_string()
    } else {
        raw
    }
}
