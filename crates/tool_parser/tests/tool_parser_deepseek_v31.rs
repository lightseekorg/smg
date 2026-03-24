//! DeepSeek V3.1 Parser Integration Tests
//!
//! V3.1 format differs from V3: no `function` type prefix, no code fence around JSON args.
//! Format: `<｜tool▁call▁begin｜>{name}<｜tool▁sep｜>{raw_json}<｜tool▁call▁end｜>`
mod common;

use common::create_test_tools;
use tool_parser::{DeepSeekV31Parser, ToolParser};

#[tokio::test]
async fn test_deepseek_v31_complete_parsing() {
    let parser = DeepSeekV31Parser::new();

    let input = "Let me help you with that.\n\
        <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>\
        {\"location\": \"Tokyo\", \"units\": \"celsius\"}\
        <｜tool▁call▁end｜><｜tool▁calls▁end｜>";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "Let me help you with that.\n");
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Tokyo");
    assert_eq!(args["units"], "celsius");
}

#[tokio::test]
async fn test_deepseek_v31_multiple_tools() {
    let parser = DeepSeekV31Parser::new();

    let input = "<｜tool▁calls▁begin｜>\
        <｜tool▁call▁begin｜>search<｜tool▁sep｜>{\"query\": \"rust programming\"}<｜tool▁call▁end｜>\
        <｜tool▁call▁begin｜>translate<｜tool▁sep｜>{\"text\": \"Hello World\", \"to\": \"ja\"}<｜tool▁call▁end｜>\
        <｜tool▁calls▁end｜>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.name, "search");
    assert_eq!(tools[1].function.name, "translate");

    let args0: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args0["query"], "rust programming");

    let args1: serde_json::Value = serde_json::from_str(&tools[1].function.arguments).unwrap();
    assert_eq!(args1["text"], "Hello World");
    assert_eq!(args1["to"], "ja");
}

#[tokio::test]
async fn test_deepseek_v31_streaming() {
    let tools = create_test_tools();

    let mut parser = DeepSeekV31Parser::new();

    // Simulate streaming chunks — V3.1 format (no "function" prefix, no code fence)
    let chunks = vec![
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>",
        "get_weather<｜tool▁sep｜>",
        r#"{"location": "#,
        r#""Beijing", "#,
        r#""units": "metric"}"#,
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    ];

    let mut found_name = false;
    let mut collected_args = String::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                found_name = true;
            }
            if !call.parameters.is_empty() {
                collected_args.push_str(&call.parameters);
            }
        }
    }

    assert!(found_name, "Should have found tool name during streaming");
    assert!(
        !collected_args.is_empty(),
        "Should have collected argument chunks"
    );
}

#[tokio::test]
async fn test_deepseek_v31_no_type_field() {
    // V3.1 should NOT require "function" type prefix
    let parser = DeepSeekV31Parser::new();

    // V3 format (with "function" type) should fail or not match in V3.1 parser
    let v3_input = "<｜tool▁calls▁begin｜>\
        <｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n\
        ```json\n{\"location\": \"Tokyo\"}\n```\
        <｜tool▁call▁end｜><｜tool▁calls▁end｜>";

    let (_text, tools) = parser.parse_complete(v3_input).await.unwrap();
    // The V3 format should either fail to parse JSON or produce an incorrect name
    // because V3.1 treats everything before <｜tool▁sep｜> as the function name
    // and everything after as raw JSON (which includes the code fence markers).
    // Either way, it should not produce a valid "get_weather" tool call.
    let valid_tools: Vec<_> = tools
        .iter()
        .filter(|t| t.function.name == "get_weather")
        .collect();
    assert!(
        valid_tools.is_empty(),
        "V3 format should not parse correctly with V3.1 parser as a clean get_weather call"
    );

    // V3.1 format (no type, no code fence) should work
    let v31_input = "<｜tool▁calls▁begin｜>\
        <｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"location\": \"Tokyo\"}\
        <｜tool▁call▁end｜><｜tool▁calls▁end｜>";

    let (_text, tools) = parser.parse_complete(v31_input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");
}

#[test]
fn test_deepseek_v31_format_detection() {
    let parser = DeepSeekV31Parser::new();

    // Should detect DeepSeek format (same markers as V3)
    assert!(parser.has_tool_markers("<｜tool▁calls▁begin｜>"));
    assert!(parser.has_tool_markers(
        "text with <｜tool▁calls▁begin｜> marker"
    ));

    // Should not detect other formats
    assert!(!parser.has_tool_markers("[TOOL_CALLS]"));
    assert!(!parser.has_tool_markers("<tool_call>"));
    assert!(!parser.has_tool_markers("plain text"));
}

#[tokio::test]
async fn test_deepseek_v31_nested_json() {
    let parser = DeepSeekV31Parser::new();

    let input = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>process<｜tool▁sep｜>\
        {\"data\": {\"nested\": {\"deep\": [1, 2, 3]}}}\
        <｜tool▁call▁end｜><｜tool▁calls▁end｜>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "process");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert!(args["data"]["nested"]["deep"].is_array());
}

#[tokio::test]
async fn test_deepseek_v31_malformed_json_handling() {
    let parser = DeepSeekV31Parser::new();

    // Malformed JSON should be skipped, valid one parsed
    let input = "<｜tool▁calls▁begin｜>\
        <｜tool▁call▁begin｜>search<｜tool▁sep｜>{invalid json}<｜tool▁call▁end｜>\
        <｜tool▁call▁begin｜>search<｜tool▁sep｜>{\"query\": \"valid\"}<｜tool▁call▁end｜>\
        <｜tool▁calls▁end｜>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "search");
}

#[tokio::test]
async fn test_deepseek_v31_no_tools_in_text() {
    let parser = DeepSeekV31Parser::new();

    let input = "Just a normal response with no tool calls.";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(normal_text, input);
    assert!(tools.is_empty());
}

#[tokio::test]
async fn test_deepseek_v31_text_before_tools() {
    let parser = DeepSeekV31Parser::new();

    let input = "I'll search for that information.\n\
        <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>\
        {\"query\": \"rust async\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(normal_text, "I'll search for that information.\n");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "search");
}
