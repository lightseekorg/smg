//! GLM-4.7 MoE Parser Integration Tests
mod common;

use common::create_test_tools;
use openai_protocol::common::{Function, Tool};
use serde_json::json;
use tool_parser::{Glm4MoeParser, ToolParser};

// =============================================================================
// Non-streaming (parse_complete) tests
// =============================================================================

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

#[test]
fn test_glm47_format_detection() {
    let parser = Glm4MoeParser::glm47();

    assert!(parser.has_tool_markers("<tool_call>"));
    assert!(parser.has_tool_markers("text with <tool_call> marker"));
    assert!(!parser.has_tool_markers("[TOOL_CALLS]"));
    assert!(!parser.has_tool_markers("plain text"));
}

// =============================================================================
// Streaming (parse_incremental) tests — Two-phase emission
// =============================================================================

#[tokio::test]
async fn test_streaming_two_phase_single_tool() {
    let mut parser = Glm4MoeParser::glm47();
    let tools = create_test_tools();

    // Phase A: send <tool_call> + name + start of args
    let r1 = parser
        .parse_incremental("<tool_call>get_weather<arg_key>city</arg_key>", &tools)
        .await
        .unwrap();
    assert_eq!(r1.calls.len(), 1, "Phase A should emit the tool name");
    assert_eq!(r1.calls[0].name.as_deref(), Some("get_weather"));
    assert_eq!(r1.calls[0].parameters, "");

    // Phase B: send value + close
    let r2 = parser
        .parse_incremental(
            "<arg_value>Shanghai</arg_value></tool_call>",
            &tools,
        )
        .await
        .unwrap();
    assert_eq!(r2.calls.len(), 1, "Phase B should emit arguments");
    assert!(r2.calls[0].name.is_none(), "Phase B should not repeat name");
    assert!(
        r2.calls[0].parameters.contains("Shanghai"),
        "Arguments should contain the city"
    );
}

#[tokio::test]
async fn test_streaming_name_emitted_before_close() {
    let mut parser = Glm4MoeParser::glm47();
    let tools = create_test_tools();

    // Chunk 1: just the start tag + name, no </tool_call> yet
    let r1 = parser
        .parse_incremental("<tool_call>get_weather<arg_key>", &tools)
        .await
        .unwrap();
    assert_eq!(r1.calls.len(), 1);
    assert_eq!(r1.calls[0].name.as_deref(), Some("get_weather"));
    assert_eq!(r1.calls[0].parameters, "");

    // Chunk 2: finish the args + close
    let r2 = parser
        .parse_incremental(
            "city</arg_key><arg_value>NYC</arg_value></tool_call>",
            &tools,
        )
        .await
        .unwrap();
    assert_eq!(r2.calls.len(), 1);
    assert!(r2.calls[0].name.is_none());
    assert!(r2.calls[0].parameters.contains("NYC"));
}

#[tokio::test]
async fn test_streaming_normal_text_before_tool() {
    let mut parser = Glm4MoeParser::glm47();
    let tools = create_test_tools();

    let r = parser
        .parse_incremental(
            "Let me check the weather.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>",
            &tools,
        )
        .await
        .unwrap();

    assert!(
        r.normal_text.contains("Let me check the weather"),
        "Normal text before tool call should be returned"
    );
    // Should have name emission + args emission
    assert!(r.calls.len() >= 1);
    assert_eq!(r.calls[0].name.as_deref(), Some("get_weather"));
}

#[tokio::test]
async fn test_streaming_multiple_complete_tools_in_one_chunk() {
    let mut parser = Glm4MoeParser::glm47();
    let tools = create_test_tools();

    let chunk = concat!(
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Shanghai</arg_value></tool_call>",
        "<tool_call>translate<arg_key>text</arg_key><arg_value>Hello</arg_value><arg_key>target_lang</arg_key><arg_value>zh</arg_value></tool_call>"
    );

    let result = parser.parse_incremental(chunk, &tools).await.unwrap();

    // Each tool should produce a name call + args call = 4 total
    let names: Vec<_> = result
        .calls
        .iter()
        .filter_map(|c| c.name.as_deref())
        .collect();
    assert_eq!(names, vec!["get_weather", "translate"]);
    assert_eq!(result.calls.len(), 4); // 2 names + 2 args
}

#[tokio::test]
async fn test_streaming_partial_bot_token_across_chunks() {
    let mut parser = Glm4MoeParser::glm47();
    let tools = create_test_tools();

    // First chunk: two complete tools + partial start of third
    let first_chunk = concat!(
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>New York, NY</arg_value></tool_call>",
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>San Francisco, CA</arg_value></tool_call>",
        "<tool_ca"
    );

    let r1 = parser.parse_incremental(first_chunk, &tools).await.unwrap();
    let names1: Vec<_> = r1.calls.iter().filter_map(|c| c.name.as_deref()).collect();
    assert_eq!(names1, vec!["get_weather", "get_weather"]);

    // Second chunk: complete the third tool
    let second_chunk =
        "ll>get_weather<arg_key>city</arg_key><arg_value>Chicago, IL</arg_value></tool_call>";

    let r2 = parser.parse_incremental(second_chunk, &tools).await.unwrap();
    let names2: Vec<_> = r2.calls.iter().filter_map(|c| c.name.as_deref()).collect();
    assert_eq!(names2, vec!["get_weather"]);

    // Verify the third tool's args
    let args_calls: Vec<_> = r2.calls.iter().filter(|c| c.name.is_none()).collect();
    assert!(!args_calls.is_empty());
    assert!(args_calls[0].parameters.contains("Chicago, IL"));
}

#[tokio::test]
async fn test_streaming_end_of_stream_flush() {
    let mut parser = Glm4MoeParser::glm47();
    let tools = create_test_tools();

    // Send a tool call without closing tag (simulates end-of-stream)
    let r = parser
        .parse_incremental(
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Berlin</arg_value>",
            &tools,
        )
        .await
        .unwrap();

    // Name should be emitted
    assert_eq!(r.calls.len(), 1);
    assert_eq!(r.calls[0].name.as_deref(), Some("get_weather"));

    // Stream ends — get_unstreamed_tool_args should flush the buffered args
    let unstreamed = parser.get_unstreamed_tool_args();
    assert!(
        unstreamed.is_some(),
        "Should have unstreamed args from buffer"
    );
    let items = unstreamed.unwrap();
    assert!(!items.is_empty());
    // Name was already sent, so this should be args-only
    assert!(items[0].name.is_none());
    assert!(items[0].parameters.contains("Berlin"));
}

#[tokio::test]
async fn test_streaming_end_of_stream_flush_name_not_sent() {
    let mut parser = Glm4MoeParser::glm47();
    let tools = create_test_tools();

    // Two complete tools + one incomplete without </tool_call>
    let r = parser
        .parse_incremental(
            concat!(
                "<tool_call>get_weather<arg_key>city</arg_key><arg_value>A</arg_value></tool_call>",
                "<tool_call>get_weather<arg_key>city</arg_key><arg_value>B</arg_value></tool_call>",
                "<tool_call>get_weather<arg_key>city</arg_key><arg_value>C</arg_value>"
            ),
            &tools,
        )
        .await
        .unwrap();

    // First two are complete (name + args each), third has name emitted
    let names: Vec<_> = r.calls.iter().filter_map(|c| c.name.as_deref()).collect();
    assert!(names.len() >= 2, "At least two tool names should be emitted");

    // Flush remaining
    let unstreamed = parser.get_unstreamed_tool_args();
    assert!(unstreamed.is_some());
    let items = unstreamed.unwrap();
    assert!(items.iter().any(|i| i.parameters.contains("C")));
}

#[tokio::test]
async fn test_streaming_no_arg_function() {
    let mut parser = Glm4MoeParser::glm47();
    let tools = vec![Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "ping".to_string(),
            description: None,
            parameters: json!({"type": "object", "properties": {}}),
            strict: None,
        },
    }];

    let r = parser
        .parse_incremental("<tool_call>ping</tool_call>", &tools)
        .await
        .unwrap();

    let names: Vec<_> = r.calls.iter().filter_map(|c| c.name.as_deref()).collect();
    assert_eq!(names, vec!["ping"]);

    let args_calls: Vec<_> = r.calls.iter().filter(|c| c.name.is_none()).collect();
    assert_eq!(args_calls.len(), 1);
    assert_eq!(args_calls[0].parameters, "{}");
}

#[tokio::test]
async fn test_streaming_uses_schema_for_string_arguments() {
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

    // Name + args
    assert!(result.calls.len() >= 2);
    assert_eq!(result.calls[0].name.as_deref(), Some("invokeCallback"));

    // Combine all arg fragments
    let all_args: String = result
        .calls
        .iter()
        .filter(|c| c.name.is_none())
        .map(|c| c.parameters.as_str())
        .collect();
    let args: serde_json::Value = serde_json::from_str(&all_args).unwrap();
    assert_eq!(args["callback"], "processResult");
    assert_eq!(args["error"], "null");
    assert_eq!(args["value"], "Operation successful");
}

#[tokio::test]
async fn test_streaming_char_by_char() {
    let mut parser = Glm4MoeParser::glm47();
    let tools = create_test_tools();

    let full = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>NYC</arg_value></tool_call>";

    let mut found_name = false;
    let mut found_args = false;

    for ch in full.chars() {
        let r = parser
            .parse_incremental(&ch.to_string(), &tools)
            .await
            .unwrap();

        for call in &r.calls {
            if call.name.as_deref() == Some("get_weather") {
                found_name = true;
            }
            if call.name.is_none() && call.parameters.contains("NYC") {
                found_args = true;
            }
        }
    }

    assert!(found_name, "Tool name should have been emitted");
    assert!(found_args, "Tool args should have been emitted");
}

#[tokio::test]
async fn test_streaming_reset_clears_state() {
    let mut parser = Glm4MoeParser::glm47();
    let tools = create_test_tools();

    // Process one tool call
    let _ = parser
        .parse_incremental(
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>X</arg_value></tool_call>",
            &tools,
        )
        .await
        .unwrap();

    // Reset
    parser.reset();

    // Process another — should work from clean state
    let r = parser
        .parse_incremental(
            "<tool_call>translate<arg_key>text</arg_key><arg_value>Hi</arg_value></tool_call>",
            &tools,
        )
        .await
        .unwrap();

    let names: Vec<_> = r.calls.iter().filter_map(|c| c.name.as_deref()).collect();
    assert_eq!(names, vec!["translate"]);
    assert_eq!(r.calls[0].tool_index, 0, "Tool index should restart at 0");
}

#[tokio::test]
async fn test_streaming_five_tools_last_split_across_chunks() {
    let mut parser = Glm4MoeParser::glm47();
    let tools = create_test_tools();

    // Chunk 1: 4 complete tools + beginning of 5th
    let chunk1 = concat!(
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>A</arg_value></tool_call>",
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>B</arg_value></tool_call>",
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>C</arg_value></tool_call>",
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>D</arg_value></tool_call>",
        "<tool_call>get_weather<arg_key>city</arg_key>"
    );

    let r1 = parser.parse_incremental(chunk1, &tools).await.unwrap();
    let names1: Vec<_> = r1.calls.iter().filter_map(|c| c.name.as_deref()).collect();
    // 4 complete tools (name + args each) + 5th name emitted
    assert_eq!(names1.len(), 5, "All 5 names should be emitted: got {:?}", names1);

    let args1: Vec<_> = r1.calls.iter().filter(|c| c.name.is_none()).collect();
    assert_eq!(args1.len(), 4, "4 complete arg sets should be emitted");

    // Chunk 2: finish the 5th tool
    let r2 = parser
        .parse_incremental("<arg_value>E</arg_value></tool_call>", &tools)
        .await
        .unwrap();
    let args2: Vec<_> = r2.calls.iter().filter(|c| c.name.is_none()).collect();
    assert_eq!(args2.len(), 1, "5th tool's args should be emitted");
    assert!(
        args2[0].parameters.contains("E"),
        "5th tool should have city E, got: {}",
        args2[0].parameters
    );
}

#[tokio::test]
async fn test_glm47_streaming_five_parallel_tools() {
    let mut parser = Glm4MoeParser::glm47();
    let tools = create_test_tools();

    let input = concat!(
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>A</arg_value></tool_call>",
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>B</arg_value></tool_call>",
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>C</arg_value></tool_call>",
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>D</arg_value></tool_call>",
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>E</arg_value></tool_call>"
    );

    let result = parser.parse_incremental(input, &tools).await.unwrap();
    let names: Vec<_> = result
        .calls
        .iter()
        .filter_map(|c| c.name.as_deref())
        .collect();
    assert_eq!(names.len(), 5, "All 5 tool names should be emitted");

    let arg_items: Vec<_> = result.calls.iter().filter(|c| c.name.is_none()).collect();
    assert_eq!(arg_items.len(), 5, "All 5 argument sets should be emitted");
}
