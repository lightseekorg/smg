//! Shared helpers for the gRPC Harmony Responses surface.
//!
//! This module keeps Harmony-specific image-generation tool normalization.
//!
//! - `strip_image_generation_from_request_tools` — harmony-specific
//!   builder workaround for the
//!   [`openai_protocol::responses::ResponseTool::ImageGeneration`] tag
//!   when an MCP server has taken ownership of `image_generation`.

use openai_protocol::responses::{ResponseTool, ResponsesRequest};
use smg_mcp::{McpToolSession, ResponseFormat};
use tracing::debug;

/// Strip `ResponseTool::ImageGeneration` once an MCP session exposes an
/// MCP-routed replacement.
///
/// The harmony builder synthesizes a function-tool description named
/// Once an MCP server takes ownership of `image_generation`, the MCP-exposed
/// function tool is the single advertisement the model should see.
pub(super) fn strip_image_generation_from_request_tools(
    request: &mut ResponsesRequest,
    session: &McpToolSession<'_>,
) {
    let mcp_has_image_generation = session
        .mcp_tools()
        .iter()
        .any(|entry| matches!(entry.response_format, ResponseFormat::ImageGenerationCall));
    if !mcp_has_image_generation {
        return;
    }
    if let Some(tools) = request.tools.as_mut() {
        let before = tools.len();
        tools.retain(|t| !matches!(t, ResponseTool::ImageGeneration(_)));
        let after = tools.len();
        if before != after {
            debug!(
                removed = before - after,
                "Stripped ResponseTool::ImageGeneration from request.tools because the \
                 MCP session exposes an image_generation-routed dispatcher",
            );
        }
    }
}
