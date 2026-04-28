//! Shared agent loop driving every Responses API surface.
//!
//! Every Responses request — OpenAI passthrough, gRPC chat-adapter
//! ("regular"), and gRPC harmony — enters the same state machine here
//! instead of branching outside the loop on "has MCP tools" and friends.
//! The driver decides the next action from loop state; surface adapters
//! plug in only the parts that are genuinely surface-specific:
//!
//! - building the upstream request from the canonical transcript
//! - executing one model turn (non-streaming or streaming)
//! - rendering the final response item shape
//!
//! See `docs/contributing/unified-agent-loop.md` (on
//! `design/unified-agent-loop`) for the full design and naming contract;
//! the `state`, `prepared`, `events`, `error`, `tooling`, and `driver`
//! submodules below mirror its module layout.

pub(crate) mod build_response;
pub(crate) mod driver;
pub(crate) mod error;
pub(crate) mod events;
pub(crate) mod prepared;
pub(crate) mod presentation;
pub(crate) mod state;
pub(crate) mod tooling;

pub(crate) use build_response::{build_response_from_state, ResponseBuildHooks, UsageShape};
pub(crate) use driver::{run_agent_loop, AgentLoopAdapter, AgentLoopContext, RenderMode};
pub(crate) use error::AgentLoopError;
pub(crate) use events::{LoopEvent, NoopSink, StreamSink};
pub(crate) use prepared::PreparedLoopInput;
pub(crate) use presentation::{
    normalize_output_item_id, OutputFamily, ToolPresentation, ToolTransferDescriptor,
    ToolVisibility,
};
pub(crate) use state::{
    AgentLoopState, ExecutedCall, LoopModelTurn, LoopToolCall, PendingToolExecution,
};
