use serde_json::{json, Value};
use smg_mcp::{extract_image_generation_fallback_text, ResponseFormat};

/// Build tool output text for model context.
///
/// This is format-driven and intended to support per-tool compaction policies.
/// Currently only `ResponseFormat::ImageGenerationCall` is compacted into a
/// minimal fixed summary (tool/status/note) to avoid feeding large binary
/// image data back into the next model turn. Other formats are currently no-op:
/// string outputs are preserved as-is, non-strings use `output.to_string()`.
pub fn compact_tool_output_for_model_context(
    response_format: &ResponseFormat,
    is_error: bool,
    output: &Value,
) -> String {
    match response_format {
        ResponseFormat::ImageGenerationCall => {
            let note = if is_error {
                extract_image_generation_fallback_text(output).unwrap_or_default()
            } else {
                "Successfully generated the image".to_string()
            };
            let summary = json!({
                "tool": "generate_image",
                "status": if is_error { "failed" } else { "completed" },
                "note": note
            });
            summary.to_string()
        }
        // No-op for other tools for now: preserve raw string outputs as-is.
        _ => match output {
            Value::String(text) => text.clone(),
            _ => output.to_string(),
        },
    }
}
