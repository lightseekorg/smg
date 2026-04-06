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
        ResponseFormat::ImageGenerationCall => json!({
            "tool": "generate_image",
            "note": "binary image payload omitted from model context"
        })
        .to_string(),
        // No-op for other tools for now: keep full output text in model context.
        _ => output.to_string(),
    }
}
