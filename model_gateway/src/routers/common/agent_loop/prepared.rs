//! Canonical loop input — the boundary between surface-specific history
//! loading and the shared driver.
//!
//! Every surface (openai, grpc-regular, grpc-harmony) builds a
//! `PreparedLoopInput` from its incoming `ResponsesRequest` plus any
//! response-chain or conversation-storage items, then hands it to the
//! driver. The driver does not care which source the items came from.

use openai_protocol::responses::{ResponseInput, ResponseInputOutputItem};

/// Loop-facing view of an incoming `ResponsesRequest` after history has
/// been hydrated and normalized. Mirrors the shape spelled out in the
/// design doc's "Prepared loop input" section.
#[derive(Debug, Clone)]
pub(crate) struct PreparedLoopInput {
    /// Items the upstream model should see — the LLM-consumable
    /// transcript form. `mcp_call` history items have already been
    /// expanded into `function_call` / `function_call_output` pairs
    /// here; client-only metadata items live in `control_items` instead.
    pub upstream_input: ResponseInput,

    /// Client-visible-only items that drive loop control decisions but
    /// must not reach the upstream model. Today this includes:
    /// - `mcp_list_tools` (already-emitted server inventory)
    /// - `mcp_approval_request` / `mcp_approval_response`
    pub control_items: Vec<ResponseInputOutputItem>,
}

impl PreparedLoopInput {
    /// Construct from already-prepared parts. Use this from a surface
    /// adapter's history loader.
    pub(crate) fn new(
        upstream_input: ResponseInput,
        control_items: Vec<ResponseInputOutputItem>,
    ) -> Self {
        Self {
            upstream_input,
            control_items,
        }
    }
}
