//! MCP (Model Context Protocol) module for the OpenAI router.

mod tool_prep;

pub(crate) use tool_prep::prepare_mcp_tools_as_functions;
