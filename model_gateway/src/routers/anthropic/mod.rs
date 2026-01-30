//! Anthropic API implementations
//!
//! This module provides Anthropic-specific API routing including:
//! - Messages API (/v1/messages) with SSE streaming, tool use, and extended thinking
//! - Models API (/v1/models) for listing available models

pub mod messages;
pub mod models;
mod router;

pub use router::AnthropicRouter;
