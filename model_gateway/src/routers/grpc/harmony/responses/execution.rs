//! Harmony-side MCP tool helper kept after the agent-loop refactor.
//!
//! Tool execution itself now lives in
//! `routers/common/agent_loop/tooling.rs::execute_planned_tool`; the
//! only thing that remains harmony-specific is converting the
//! request-scoped `McpToolSession` into the `ResponseTool` array the
//! harmony pipeline sees as advertised function tools.

use openai_protocol::responses::ResponseTool;
use smg_mcp::McpToolSession;

pub(crate) fn convert_mcp_tools_to_response_tools(
    session: &McpToolSession<'_>,
) -> Vec<ResponseTool> {
    session.build_response_tools()
}
