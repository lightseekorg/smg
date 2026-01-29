//! Anthropic Messages API support for OpenAI router
//!
//! This module implements the Anthropic Messages API (`/v1/messages`) within the
//! OpenAI router, reusing existing infrastructure for policy selection, worker
//! routing, and observability.
//!
//! ## Architecture
//!
//! - `handler.rs`: Core request handling (non-streaming)
//! - `streaming.rs`: SSE streaming support
//! - `tools.rs`: Tool use and MCP integration
//!
//! ## Status
//!
//! - [x] Foundation (PR #1)
//! - [ ] Non-streaming (PR #2)
//! - [ ] Streaming (PR #3)
//! - [ ] Tools (PR #4)

pub mod handler;
pub mod streaming;
pub mod tools;

pub use handler::MessagesHandler;
