//! Shared Responses agent-loop runtime setup.
//!
//! This helper intentionally stops before constructing a surface adapter or
//! running the loop. OpenAI/gRPC regular/gRPC harmony still own their upstream
//! call details; this module only packages the common prepared input, MCP
//! server bindings, loop state, and tool budget.

use openai_protocol::responses::ResponsesRequest;
use smg_mcp::McpServerBinding;

use super::{
    agent_loop::{AgentLoopState, PreparedLoopInput},
    responses_history::PreparedRequestHistory,
};

pub(crate) struct ResponsesLoopSetup {
    pub(crate) current_request: ResponsesRequest,
    pub(crate) prepared: PreparedLoopInput,
    pub(crate) state: AgentLoopState,
    pub(crate) max_tool_calls: Option<usize>,
    pub(crate) mcp_servers: Vec<McpServerBinding>,
}

impl ResponsesLoopSetup {
    pub(crate) fn from_history(
        history: PreparedRequestHistory,
        mcp_servers: Vec<McpServerBinding>,
    ) -> Self {
        Self::new(
            history.request,
            history.prepared,
            history.existing_mcp_list_tools_labels,
            mcp_servers,
        )
    }

    pub(crate) fn new(
        current_request: ResponsesRequest,
        prepared: PreparedLoopInput,
        existing_mcp_list_tools_labels: impl IntoIterator<Item = String>,
        mcp_servers: Vec<McpServerBinding>,
    ) -> Self {
        let max_tool_calls = current_request.max_tool_calls.map(|n| n as usize);
        let state = AgentLoopState::new(
            prepared.upstream_input.clone(),
            existing_mcp_list_tools_labels.into_iter().collect(),
        );

        Self {
            current_request,
            prepared,
            state,
            max_tool_calls,
            mcp_servers,
        }
    }
}
