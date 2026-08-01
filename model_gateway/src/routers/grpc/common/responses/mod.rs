//! Shared response functionality used by both regular and harmony implementations

pub(crate) mod context;
pub(crate) mod handlers;
pub(crate) mod streaming;
pub(crate) mod utils;

// Re-export commonly used items
pub(crate) use context::ResponsesContext;
use openai_protocol::{
    common::{Usage, UsageInfo},
    responses::ResponsesUsage,
};
use serde_json::{json, Value};
pub(crate) use streaming::build_sse_response;
pub(crate) use utils::{ensure_mcp_connection, persist_response_if_needed};

pub(crate) use crate::routers::common::mcp_utils::collect_user_function_names;

/// Convert Chat usage into the Responses API's modern usage wire shape.
pub(crate) fn responses_usage_from_chat_usage(usage: &Usage) -> ResponsesUsage {
    let usage_info = UsageInfo {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
        reasoning_tokens: usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens),
        prompt_tokens_details: usage.prompt_tokens_details.clone(),
    };

    ResponsesUsage::Modern(usage_info.to_response_usage())
}

/// Build the usage object embedded in regular Responses `response.completed` events.
pub(crate) fn response_completed_usage(usage: &Usage) -> Value {
    let usage = responses_usage_from_chat_usage(usage).to_response_usage();
    json!({
        "input_tokens": usage.input_tokens,
        "input_tokens_details": {
            "cached_tokens": usage
                .input_tokens_details
                .as_ref()
                .map_or(0, |details| details.cached_tokens),
        },
        "output_tokens": usage.output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": usage
                .output_tokens_details
                .as_ref()
                .map_or(0, |details| details.reasoning_tokens),
        },
        "total_tokens": usage.total_tokens,
    })
}
