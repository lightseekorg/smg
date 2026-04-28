//! OpenAI-compatible responses handling module
//!
//! This module provides comprehensive support for OpenAI Responses API with:
//! - Streaming and non-streaming response handling
//! - MCP (Model Context Protocol) tool preparation and common-loop handoff
//! - SSE (Server-Sent Events) parsing and forwarding
//! - Response accumulation for persistence
//! - Upstream tool-call parsing and output index remapping
//! - Input history loading from conversations and response chains
//! - Shared helpers for response retrieval-related logic

mod accumulator;
mod agent_loop_adapter;
mod common;
pub(crate) mod history;
mod non_streaming;
pub(crate) mod route;
mod streaming;
mod upstream_stream_parser;
mod utils;

pub(crate) use accumulator::StreamingResponseAccumulator;
pub(crate) use common::{extract_output_index, get_event_type};
pub use non_streaming::handle_non_streaming_response;
pub use streaming::handle_streaming_response;
