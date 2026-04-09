use serde_json::{json, Value};
use smg_mcp::{extract_image_generation_fallback_text, is_image_generation_error, ResponseFormat};

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
            let is_error = is_image_generation_error(output);
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
