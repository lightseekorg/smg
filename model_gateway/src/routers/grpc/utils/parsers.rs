//! Reasoning and tool parser helpers.

use llm_tokenizer::{
    chat_template::{ThinkingKeyName, ThinkingToggle},
    traits::Tokenizer,
};
use reasoning_parser::{
    ParserFactory as ReasoningParserFactory, PooledParser as ReasoningPooledParser, ReasoningParser,
};
use serde_json::Value;
use tool_parser::{
    ParserFactory as ToolParserFactory, PooledParser as ToolPooledParser, ToolParser,
};
use tracing::warn;

/// Determine if thinking is effectively ON based on the template's thinking
/// toggle and the user's request.
///
/// `user_thinking`: `Some(true)` = user enabled thinking, `Some(false)` = user
/// disabled it, `None` = not specified (use template default).
pub(crate) fn should_mark_reasoning_started(
    user_thinking: Option<bool>,
    tokenizer: &dyn Tokenizer,
) -> bool {
    match tokenizer.thinking_toggle() {
        ThinkingToggle::None => false,
        ThinkingToggle::DefaultOn => user_thinking != Some(false),
        ThinkingToggle::DefaultOff => user_thinking == Some(true),
    }
}

/// Extract the user's thinking preference from chat_template_kwargs.
///
/// Only checks the key that the template actually uses (e.g. `enable_thinking`
/// for Qwen3, `thinking` for Kimi-K2.5). This prevents mismatches where the
/// user passes the wrong key name and the template ignores it.
pub(crate) fn extract_thinking_from_kwargs(
    kwargs: Option<&std::collections::HashMap<String, Value>>,
    tokenizer: &dyn Tokenizer,
) -> Option<bool> {
    let kwargs = kwargs?;
    match tokenizer.thinking_key_name() {
        Some(ThinkingKeyName::EnableThinking) => kwargs.get("enable_thinking"),
        Some(ThinkingKeyName::Thinking) => kwargs.get("thinking"),
        None => None,
    }
    .and_then(|v| v.as_bool())
}

/// Check if a reasoning parser is available for the given model
pub(crate) fn check_reasoning_parser_availability(
    reasoning_parser_factory: &ReasoningParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> bool {
    if let Some(parser_name) = configured_parser {
        reasoning_parser_factory.registry().has_parser(parser_name)
    } else {
        reasoning_parser_factory
            .registry()
            .has_parser_for_model(model)
    }
}

/// Check if a tool parser is available for the given model
pub(crate) fn check_tool_parser_availability(
    tool_parser_factory: &ToolParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> bool {
    if let Some(parser_name) = configured_parser {
        tool_parser_factory.registry().has_parser(parser_name)
    } else {
        tool_parser_factory.registry().has_parser_for_model(model)
    }
}

/// Get the appropriate reasoning parser for a model
///
/// If a parser name is explicitly configured, use that parser.
/// Otherwise, auto-detect based on the model name.
/// Get a pooled reasoning parser (for non-streaming where state doesn't matter)
pub(crate) fn get_reasoning_parser(
    reasoning_parser_factory: &ReasoningParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> ReasoningPooledParser {
    if let Some(parser_name) = configured_parser {
        // Use configured parser if specified
        reasoning_parser_factory
            .registry()
            .get_pooled_parser(parser_name)
            .unwrap_or_else(|| {
                warn!(
                    "Configured reasoning parser '{}' not found, falling back to model-based selection",
                    parser_name
                );
                reasoning_parser_factory.get_pooled(model)
            })
    } else {
        // Auto-detect based on model
        reasoning_parser_factory.get_pooled(model)
    }
}

/// Create a fresh reasoning parser instance (for streaming where state isolation is needed)
pub(crate) fn create_reasoning_parser(
    reasoning_parser_factory: &ReasoningParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> Option<Box<dyn ReasoningParser>> {
    if let Some(parser_name) = configured_parser {
        // Use configured parser if specified
        reasoning_parser_factory
            .registry()
            .create_parser(parser_name)
            .or_else(|| {
                warn!(
                    "Configured reasoning parser '{}' not found, falling back to model-based selection",
                    parser_name
                );
                reasoning_parser_factory.registry().create_for_model(model)
            })
    } else {
        // Auto-detect based on model
        reasoning_parser_factory.registry().create_for_model(model)
    }
}

/// Get the appropriate tool parser for a model
///
/// If a parser name is explicitly configured, use that parser.
/// Otherwise, auto-detect based on the model name.
/// Get a pooled tool parser (for non-streaming where state doesn't matter)
pub(crate) fn get_tool_parser(
    tool_parser_factory: &ToolParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> ToolPooledParser {
    if let Some(parser_name) = configured_parser {
        // Use configured parser if specified
        tool_parser_factory
            .registry()
            .get_pooled_parser(parser_name)
            .unwrap_or_else(|| {
                warn!(
                    "Configured tool parser '{}' not found, falling back to model-based selection",
                    parser_name
                );
                tool_parser_factory.get_pooled(model)
            })
    } else {
        // Auto-detect based on model
        tool_parser_factory.get_pooled(model)
    }
}

/// Create a fresh tool parser instance (for streaming where state isolation is needed)
pub(crate) fn create_tool_parser(
    tool_parser_factory: &ToolParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> Option<Box<dyn ToolParser>> {
    if let Some(parser_name) = configured_parser {
        // Use configured parser if specified
        tool_parser_factory
            .registry()
            .create_parser(parser_name)
            .or_else(|| {
                warn!(
                    "Configured tool parser '{}' not found, falling back to model-based selection",
                    parser_name
                );
                tool_parser_factory.registry().create_for_model(model)
            })
    } else {
        // Auto-detect based on model
        tool_parser_factory.registry().create_for_model(model)
    }
}
