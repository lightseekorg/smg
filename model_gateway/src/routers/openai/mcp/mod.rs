//! MCP (Model Context Protocol) module for the OpenAI router.
//!
//! Contains tool loop orchestration and streaming tool call handling,
//! extracted from `responses/` for separation of concerns.

mod tool_handler;
mod tool_prep;

// Re-export types used by responses/streaming.rs
pub(crate) use tool_handler::{FunctionCallInProgress, StreamAction, StreamingToolHandler};
pub(crate) use tool_prep::prepare_mcp_tools_as_functions;
