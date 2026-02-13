//! Tool use and MCP integration for Messages API
//!
//! This module handles:
//! - Converting MCP tools to Messages API format
//! - Executing tool calls via MCP orchestrator
//! - Assembling tool results

use openai_protocol::messages::{ContentBlock, ToolUseBlock};

pub fn extract_tool_calls(content: &[ContentBlock]) -> Vec<ToolUseBlock> {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolUse { id, name, input } => Some(ToolUseBlock {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
                cache_control: None,
            }),
            _ => None,
        })
        .collect()
}
