//! DeepSeek V3.1 Parser Integration Tests
mod common;

use tool_parser::{DeepSeek31Parser, ToolParser};

#[tokio::test]
async fn test_deepseek31_complete_single_tool() {
    let parser = DeepSeek31Parser::new();

    let input = concat!(
        "Let me check that for you.",
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>",
        r#"{"location": "Tokyo", "units": "celsius"}"#,
        "<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
    );

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(normal_text, "Let me check that for you.");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Tokyo");
    assert_eq!(args["units"], "celsius");
}

#[tokio::test]
async fn test_deepseek31_complete_multiple_tools() {
    let parser = DeepSeek31Parser::new();

    let input = concat!(
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>search<｜tool▁sep｜>",
        r#"{"query": "rust programming"}"#,
        "<｜tool▁call▁end｜>",
        "<｜tool▁call▁begin｜>translate<｜tool▁sep｜>",
        r#"{"text": "Hello World", "to": "ja"}"#,
        "<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
    );

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.name, "search");
    assert_eq!(tools[1].function.name, "translate");
}

#[tokio::test]
async fn test_deepseek31_complete_nested_json() {
    let parser = DeepSeek31Parser::new();

    let input = concat!(
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>process<｜tool▁sep｜>",
        r#"{"data": {"nested": {"deep": [1, 2, 3]}}}"#,
        "<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
    );

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "process");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert!(args["data"]["nested"]["deep"].is_array());
}

#[tokio::test]
async fn test_deepseek31_complete_malformed_json() {
    let parser = DeepSeek31Parser::new();

    let input = concat!(
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>search<｜tool▁sep｜>",
        "{invalid json}",
        "<｜tool▁call▁end｜>",
        "<｜tool▁call▁begin｜>translate<｜tool▁sep｜>",
        r#"{"text": "hello", "to": "ja"}"#,
        "<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
    );

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "translate");
}

#[test]
fn test_deepseek31_format_detection() {
    let parser = DeepSeek31Parser::new();

    assert!(parser.has_tool_markers("<｜tool▁calls▁begin｜>"));
    assert!(parser.has_tool_markers("text with <｜tool▁calls▁begin｜> marker"));

    assert!(!parser.has_tool_markers("[TOOL_CALLS]"));
    assert!(!parser.has_tool_markers("<tool_call>"));
    assert!(!parser.has_tool_markers("plain text"));
}

#[tokio::test]
async fn test_deepseek31_no_tool_calls() {
    let parser = DeepSeek31Parser::new();

    let input = "Just a normal response with no tools.";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(normal_text, input);
    assert!(tools.is_empty());
}
