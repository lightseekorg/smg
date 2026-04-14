//! Cross-router shared utilities.
//!
//! This module collects helpers that every router (HTTP, gRPC,
//! OpenAI, Anthropic, Gemini, etc.) needs but no individual router
//! owns. Putting them here keeps `routers/mod.rs` focused on the
//! `RouterTrait` definition and the per-protocol submodules.
//!
//! Submodules:
//! - [`header_utils`] — request header parsing helpers
//!   (`extract_routing_key`, `extract_target_worker`, etc.)
//! - [`mcp_utils`] — Model Context Protocol tool-call orchestration
//! - [`persistence_utils`] — response/conversation persistence
//!   helpers shared across the chat / responses / messages routes
//! - [`worker_selection`] — per-request worker-selection helpers used
//!   by every routing path (regular, PD, fallback, external provider)
//! - [`retry`] — generic async retry executor + backoff calculator,
//!   used by every router for transport-level retries. Has zero
//!   coupling to the `Worker` trait — it lived in `worker/` for
//!   historical reasons before this extraction.

pub mod header_utils;
pub mod mcp_utils;
pub mod persistence_utils;
pub mod retry;
pub mod worker_selection;
