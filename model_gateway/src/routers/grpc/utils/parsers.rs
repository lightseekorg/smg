//! Reasoning and tool parser helpers.

use llm_tokenizer::{
    chat_template::{ThinkingKeyName, ThinkingToggle},
    traits::Tokenizer,
};
use openai_protocol::common::{ResponseFormat, ToolChoice, ToolChoiceValue};
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

/// Returns `true` when constrained decoding is active, meaning the model output
/// is structured JSON rather than free-form text. In that case the reasoning
/// parser must be skipped — otherwise it captures the constrained JSON as
/// `reasoning_content` and leaves `content` empty.
///
/// Constrained decoding is triggered by:
/// - `tool_choice` = a specific function, `required`, or `allowed_tools` with
///   `mode == "required"`
/// - `response_format` = `json_object` or `json_schema`
pub(crate) fn has_constrained_output(
    tool_choice: Option<&ToolChoice>,
    response_format: Option<&ResponseFormat>,
) -> bool {
    let constrained_tool_choice = matches!(
        tool_choice,
        Some(ToolChoice::Function { .. }) | Some(ToolChoice::Value(ToolChoiceValue::Required))
    ) || matches!(
        tool_choice,
        Some(ToolChoice::AllowedTools { mode, .. }) if mode == "required"
    );

    let constrained_response_format = matches!(
        response_format,
        Some(ResponseFormat::JsonObject)
            | Some(ResponseFormat::JsonSchema { .. })
            | Some(ResponseFormat::Regex { .. })
    );

    constrained_tool_choice || constrained_response_format
}

#[cfg(test)]
mod tests {
    use super::*;
    use openai_protocol::common::{
        FunctionChoice, JsonSchemaFormat, ToolReference,
    };

    // ── has_constrained_output: tool_choice variants ────────────────────

    #[test]
    fn no_tool_choice_no_response_format_is_unconstrained() {
        assert!(!has_constrained_output(None, None));
    }

    #[test]
    fn tool_choice_auto_is_unconstrained() {
        let tc = ToolChoice::Value(ToolChoiceValue::Auto);
        assert!(!has_constrained_output(Some(&tc), None));
    }

    #[test]
    fn tool_choice_none_is_unconstrained() {
        let tc = ToolChoice::Value(ToolChoiceValue::None);
        assert!(!has_constrained_output(Some(&tc), None));
    }

    #[test]
    fn tool_choice_required_is_constrained() {
        let tc = ToolChoice::Value(ToolChoiceValue::Required);
        assert!(has_constrained_output(Some(&tc), None));
    }

    #[test]
    fn tool_choice_specific_function_is_constrained() {
        let tc = ToolChoice::Function {
            tool_type: "function".to_string(),
            function: FunctionChoice {
                name: "get_weather".to_string(),
            },
        };
        assert!(has_constrained_output(Some(&tc), None));
    }

    #[test]
    fn allowed_tools_required_is_constrained() {
        let tc = ToolChoice::AllowedTools {
            tool_type: "allowed_tools".to_string(),
            mode: "required".to_string(),
            tools: vec![ToolReference::Function {
                name: "search".to_string(),
            }],
        };
        assert!(has_constrained_output(Some(&tc), None));
    }

    #[test]
    fn allowed_tools_auto_is_unconstrained() {
        let tc = ToolChoice::AllowedTools {
            tool_type: "allowed_tools".to_string(),
            mode: "auto".to_string(),
            tools: vec![ToolReference::Function {
                name: "search".to_string(),
            }],
        };
        assert!(!has_constrained_output(Some(&tc), None));
    }

    // ── has_constrained_output: response_format variants ────────────────

    #[test]
    fn response_format_text_is_unconstrained() {
        assert!(!has_constrained_output(None, Some(&ResponseFormat::Text)));
    }

    #[test]
    fn response_format_json_object_is_constrained() {
        assert!(has_constrained_output(
            None,
            Some(&ResponseFormat::JsonObject)
        ));
    }

    #[test]
    fn response_format_json_schema_is_constrained() {
        let rf = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaFormat {
                name: "feedback".to_string(),
                schema: serde_json::json!({"type": "object"}),
                strict: Some(true),
            },
        };
        assert!(has_constrained_output(None, Some(&rf)));
    }

    // ── has_constrained_output: combinations ────────────────────────────

    #[test]
    fn tool_choice_auto_with_json_object_is_constrained() {
        let tc = ToolChoice::Value(ToolChoiceValue::Auto);
        assert!(has_constrained_output(
            Some(&tc),
            Some(&ResponseFormat::JsonObject)
        ));
    }

    #[test]
    fn tool_choice_auto_with_text_format_is_unconstrained() {
        let tc = ToolChoice::Value(ToolChoiceValue::Auto);
        assert!(!has_constrained_output(
            Some(&tc),
            Some(&ResponseFormat::Text)
        ));
    }

    #[test]
    fn tool_choice_required_with_json_schema_both_constrain() {
        let tc = ToolChoice::Value(ToolChoiceValue::Required);
        let rf = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaFormat {
                name: "output".to_string(),
                schema: serde_json::json!({"type": "object"}),
                strict: None,
            },
        };
        assert!(has_constrained_output(Some(&tc), Some(&rf)));
    }

    #[test]
    fn tool_choice_none_with_json_object_is_constrained_via_format() {
        let tc = ToolChoice::Value(ToolChoiceValue::None);
        assert!(has_constrained_output(
            Some(&tc),
            Some(&ResponseFormat::JsonObject)
        ));
    }
}
