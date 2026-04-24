//! Response transformation for MCP to API-specific formats.
//!
//! This module provides transformation from MCP `CallToolResult` responses
//! to OpenAI Responses API formats (web_search_call, code_interpreter_call, etc.).
//!
//! # Example
//!
//! ```ignore
//! use smg_mcp::transform::{ResponseFormat, ResponseTransformer};
//!
//! let mcp_result = serde_json::json!({"results": [{"url": "https://example.com"}]});
//! let output = ResponseTransformer::transform(
//!     &mcp_result,
//!     &ResponseFormat::WebSearchCall,
//!     "call-123",
//!     "brave",
//!     "web_search",
//!     "{}",
//! );
//! ```

mod overrides;
mod transformer;
mod types;

pub use overrides::{apply_hosted_tool_overrides, extract_hosted_tool_overrides};
pub use transformer::{
    compact_image_generation_output, extract_embedded_openai_responses, mcp_response_item_id,
    ResponseTransformer,
};
pub use types::ResponseFormat;
