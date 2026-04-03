//! DeepSeek V3.2 Parser Integration Tests
mod common;

use tool_parser::{DeepSeek32Parser, ToolParser};

#[tokio::test]
async fn test_deepseek32_complete_single_tool() {
    let parser = DeepSeek32Parser::new();

    let input = concat!(
        "Let me check that.\n\n",
        "<｜DSML｜function_calls>\n",
        "<｜DSML｜invoke name=\"get_weather\">\n",
        "<｜DSML｜parameter name=\"location\" string=\"true\">Tokyo</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"units\" string=\"true\">celsius</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜function_calls>",
    );

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(normal_text, "Let me check that.");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Tokyo");
    assert_eq!(args["units"], "celsius");
}

#[tokio::test]
async fn test_deepseek32_complete_multiple_tools() {
    let parser = DeepSeek32Parser::new();

    let input = concat!(
        "<｜DSML｜function_calls>\n",
        "<｜DSML｜invoke name=\"search\">\n",
        "<｜DSML｜parameter name=\"query\" string=\"true\">rust programming</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "<｜DSML｜invoke name=\"translate\">\n",
        "<｜DSML｜parameter name=\"text\" string=\"true\">Hello World</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"to\" string=\"true\">ja</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜function_calls>",
    );

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.name, "search");
    assert_eq!(tools[1].function.name, "translate");
}

#[tokio::test]
async fn test_deepseek32_complete_direct_json() {
    let parser = DeepSeek32Parser::new();

    let input = concat!(
        "<｜DSML｜function_calls>\n",
        "<｜DSML｜invoke name=\"get_weather\">\n",
        "{\"location\": \"Beijing\", \"date\": \"2024-01-16\"}\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜function_calls>",
    );

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Beijing");
    assert_eq!(args["date"], "2024-01-16");
}

#[tokio::test]
async fn test_deepseek32_complete_mixed_types() {
    let parser = DeepSeek32Parser::new();

    let input = concat!(
        "<｜DSML｜function_calls>\n",
        "<｜DSML｜invoke name=\"process\">\n",
        "<｜DSML｜parameter name=\"text\" string=\"true\">hello</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"count\" string=\"false\">42</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"enabled\" string=\"false\">true</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜function_calls>",
    );

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "hello");
    assert_eq!(args["count"], 42);
    assert_eq!(args["enabled"], true);
}

#[tokio::test]
async fn test_deepseek32_complete_nested_json_param() {
    let parser = DeepSeek32Parser::new();

    let input = concat!(
        "<｜DSML｜function_calls>\n",
        "<｜DSML｜invoke name=\"process\">\n",
        "<｜DSML｜parameter name=\"data\" string=\"false\">{\"nested\": [1, 2, 3]}</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜function_calls>",
    );

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert!(args["data"]["nested"].is_array());
}

#[tokio::test]
async fn test_deepseek32_complete_malformed_skips() {
    let parser = DeepSeek32Parser::new();

    let input = concat!(
        "<｜DSML｜function_calls>\n",
        "<｜DSML｜invoke name=\"search\">\n",
        "not valid at all\n",
        "</｜DSML｜invoke>\n",
        "<｜DSML｜invoke name=\"translate\">\n",
        "<｜DSML｜parameter name=\"text\" string=\"true\">hello</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"to\" string=\"true\">ja</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜function_calls>",
    );

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert!(tools.len() >= 1);
    assert!(tools.iter().any(|t| t.function.name == "translate"));
}

#[test]
fn test_deepseek32_format_detection() {
    let parser = DeepSeek32Parser::new();

    assert!(parser.has_tool_markers("<｜DSML｜function_calls>"));
    assert!(parser.has_tool_markers("text with <｜DSML｜function_calls> marker"));

    assert!(!parser.has_tool_markers("<｜tool▁calls▁begin｜>"));
    assert!(!parser.has_tool_markers("[TOOL_CALLS]"));
    assert!(!parser.has_tool_markers("plain text"));
}

#[tokio::test]
async fn test_deepseek32_no_tool_calls() {
    let parser = DeepSeek32Parser::new();

    let input = "Just a normal response.";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(normal_text, input);
    assert!(tools.is_empty());
}
