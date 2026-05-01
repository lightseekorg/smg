//! Harmony-side MCP tool advertisement helper.

use openai_protocol::responses::ResponseTool;
use smg_mcp::McpToolSession;

pub(crate) fn convert_mcp_tools_to_response_tools(
    session: &McpToolSession<'_>,
) -> Vec<ResponseTool> {
    session.build_response_tools()
}
