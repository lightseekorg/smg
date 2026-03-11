//! OpenAI-compatible responses handling module
//!
//! This module provides comprehensive support for OpenAI Responses API with:
//! - Streaming and non-streaming response handling
//! - MCP (Model Context Protocol) tool interception and execution
//! - SSE (Server-Sent Events) parsing and forwarding
//! - Response accumulation for persistence
//! - Tool call detection and output index remapping

mod accumulator;
mod common;
mod non_streaming;
mod streaming;
mod utils;

// Re-exported for openai::mcp::tool_handler (cross-module dependency)
pub(crate) use accumulator::StreamingResponseAccumulator;
pub(crate) use common::{extract_output_index, get_event_type};
pub use non_streaming::handle_non_streaming_response;
pub use streaming::handle_streaming_response;
