//! GLM-4.7 MoE Parser Integration Tests
mod common;

use common::create_test_tools;
use openai_protocol::common::{Function, Tool};
use serde_json::json;
use tool_parser::{Glm4MoeParser, ToolParser};

#[tokio::test]
async fn test_glm47_complete_parsing() {
    let parser = Glm4MoeParser::glm47();

    let input = r"Let me search for that.
<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value><arg_key>date</arg_key><arg_value>2024-12-25</arg_value></tool_call>
The weather will be...";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "Let me search for that.\n");
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Beijing");
    assert_eq!(args["date"], "2024-12-25");
}

#[tokio::test]
async fn test_glm47_multiple_tools() {
    let parser = Glm4MoeParser::glm47();

    let input = r"<tool_call>search<arg_key>query</arg_key><arg_value>rust tutorials</arg_value></tool_call><tool_call>translate<arg_key>text</arg_key><arg_value>Hello World</arg_value><arg_key>target_lang</arg_key><arg_value>zh</arg_value></tool_call>";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(normal_text, "");
    assert_eq!(tools[0].function.name, "search");
    assert_eq!(tools[1].function.name, "translate");
}

#[tokio::test]
async fn test_glm47_complete_parsing_salvages_incomplete_trailing_tool() {
    let parser = Glm4MoeParser::glm47();

    let input = concat!(
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>New York, NY</arg_value></tool_call>",
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>San Francisco, CA</arg_value></tool_call>",
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Chicago, IL</arg_value>"
    );

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(normal_text, "");
    assert_eq!(tools.len(), 3);
    assert_eq!(tools[2].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[2].function.arguments).unwrap();
    assert_eq!(args["city"], "Chicago, IL");
}

#[tokio::test]
async fn test_glm47_type_conversion() {
    let parser = Glm4MoeParser::glm47();

    let input = r"<tool_call>process<arg_key>count</arg_key><arg_value>42</arg_value><arg_key>rate</arg_key><arg_value>1.5</arg_value><arg_key>enabled</arg_key><arg_value>true</arg_value><arg_key>data</arg_key><arg_value>null</arg_value><arg_key>text</arg_key><arg_value>string value</arg_value></tool_call>";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["count"], 42);
    assert_eq!(args["rate"], 1.5);
    assert_eq!(args["enabled"], true);
    assert_eq!(args["data"], serde_json::Value::Null);
    assert_eq!(args["text"], "string value");
}

#[tokio::test]
async fn test_glm47_streaming() {
    let mut parser = Glm4MoeParser::glm47();

    let tools = create_test_tools();

    // Simulate streaming chunks
    let chunks = vec![
        "<tool_call>",
        "get_weather",
        "<arg_key>city</arg_key>",
        "<arg_value>Shanghai</arg_value>",
        "<arg_key>units</arg_key>",
        "<arg_value>celsius</arg_value>",
        "</tool_call>",
    ];

    let mut found_name = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                found_name = true;
            }
        }
    }

    assert!(found_name, "Should have found tool name during streaming");
}

#[tokio::test]
async fn test_glm47_streaming_multiple_complete_tools_in_one_chunk() {
    let mut parser = Glm4MoeParser::glm47();
    let tools = create_test_tools();

    let chunk = concat!(
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Shanghai</arg_value></tool_call>",
        "<tool_call>translate<arg_key>text</arg_key><arg_value>Hello</arg_value><arg_key>target_lang</arg_key><arg_value>zh</arg_value></tool_call>"
    );

    let result = parser.parse_incremental(chunk, &tools).await.unwrap();

    assert_eq!(result.calls.len(), 2);
    assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
    assert_eq!(result.calls[1].name.as_deref(), Some("translate"));
}

#[tokio::test]
async fn test_glm47_streaming_preserves_partial_next_tool_prefix_between_chunks() {
    let mut parser = Glm4MoeParser::glm47();
    let tools = create_test_tools();

    let first_chunk = concat!(
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>New York, NY</arg_value></tool_call>",
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>San Francisco, CA</arg_value></tool_call>",
        "<tool_ca"
    );

    let second_chunk =
        "ll>get_weather<arg_key>city</arg_key><arg_value>Chicago, IL</arg_value></tool_call>";

    let first_result = parser.parse_incremental(first_chunk, &tools).await.unwrap();
    assert_eq!(first_result.calls.len(), 2);
    assert_eq!(first_result.calls[0].name.as_deref(), Some("get_weather"));
    assert_eq!(first_result.calls[1].name.as_deref(), Some("get_weather"));

    let second_result = parser
        .parse_incremental(second_chunk, &tools)
        .await
        .unwrap();
    assert_eq!(second_result.calls.len(), 1);
    assert_eq!(second_result.calls[0].name.as_deref(), Some("get_weather"));
    assert_eq!(
        second_result.calls[0].parameters,
        r#"{"city":"Chicago, IL"}"#
    );
}

#[tokio::test]
async fn test_glm47_streaming_uses_schema_for_string_arguments() {
    let mut parser = Glm4MoeParser::glm47();
    let tools = vec![Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "invokeCallback".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "callback": {"type": "string"},
                    "error": {"type": "string"},
                    "value": {"type": "string"}
                }
            }),
            strict: None,
        },
    }];

    let result = parser
        .parse_incremental(
            "<tool_call>invokeCallback<arg_key>callback</arg_key><arg_value>processResult</arg_value><arg_key>error</arg_key><arg_value>null</arg_value><arg_key>value</arg_key><arg_value>Operation successful</arg_value></tool_call>",
            &tools,
        )
        .await
        .unwrap();

    assert_eq!(result.calls.len(), 1);
    assert_eq!(result.calls[0].name.as_deref(), Some("invokeCallback"));
    assert_eq!(
        result.calls[0].parameters,
        r#"{"callback":"processResult","error":"null","value":"Operation successful"}"#
    );
}

#[test]
fn test_glm47_format_detection() {
    let parser = Glm4MoeParser::glm47();

    // Should detect GLM-4 format
    assert!(parser.has_tool_markers("<tool_call>"));
    assert!(parser.has_tool_markers("text with <tool_call> marker"));

    // Should not detect other formats
    assert!(!parser.has_tool_markers("[TOOL_CALLS]"));
    assert!(!parser.has_tool_markers("<｜tool▁calls▁begin｜>"));
    assert!(!parser.has_tool_markers("plain text"));
}

#[tokio::test]
async fn test_python_literals() {
    let parser = Glm4MoeParser::glm47();

    let input = r"<tool_call>test_func<arg_key>bool_true</arg_key><arg_value>True</arg_value><arg_key>bool_false</arg_key><arg_value>False</arg_value><arg_key>none_val</arg_key><arg_value>None</arg_value></tool_call>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "test_func");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["bool_true"], true);
    assert_eq!(args["bool_false"], false);
    assert_eq!(args["none_val"], serde_json::Value::Null);
}

#[tokio::test]
async fn test_glm47_nested_json_in_arg_values() {
    let parser = Glm4MoeParser::glm47();

    let input = r#"<tool_call>process<arg_key>data</arg_key><arg_value>{"nested": {"key": "value"}}</arg_value><arg_key>list</arg_key><arg_value>[1, 2, 3]</arg_value></tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert!(args["data"].is_object());
    assert!(args["list"].is_array());
}
