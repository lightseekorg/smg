//! DeepSeek V3.2 DSML Parser Integration Tests
mod common;

use common::create_test_tools;
use tool_parser::{DeepSeekV32Parser, ToolParser};

#[tokio::test]
async fn test_dsml_complete_xml_parameters() {
    let parser = DeepSeekV32Parser::new();

    let input = "Let me check that for you.\n\
        <\u{ff5c}DSML\u{ff5c}function_calls>\n\
        <\u{ff5c}DSML\u{ff5c}invoke name=\"get_weather\">\n\
        <\u{ff5c}DSML\u{ff5c}parameter name=\"location\" string=\"true\">Tokyo</\u{ff5c}DSML\u{ff5c}parameter>\n\
        <\u{ff5c}DSML\u{ff5c}parameter name=\"units\" string=\"true\">celsius</\u{ff5c}DSML\u{ff5c}parameter>\n\
        </\u{ff5c}DSML\u{ff5c}invoke>\n\
        </\u{ff5c}DSML\u{ff5c}function_calls>";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "Let me check that for you.\n");
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Tokyo");
    assert_eq!(args["units"], "celsius");
}

#[tokio::test]
async fn test_dsml_complete_json_body() {
    let parser = DeepSeekV32Parser::new();

    let input = "<\u{ff5c}DSML\u{ff5c}function_calls>\n\
        <\u{ff5c}DSML\u{ff5c}invoke name=\"get_weather\">\n\
        {\"location\": \"Tokyo\", \"units\": \"celsius\"}\n\
        </\u{ff5c}DSML\u{ff5c}invoke>\n\
        </\u{ff5c}DSML\u{ff5c}function_calls>";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "");
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Tokyo");
    assert_eq!(args["units"], "celsius");
}

#[tokio::test]
async fn test_dsml_string_false_type_coercion() {
    let parser = DeepSeekV32Parser::new();

    let input = "<\u{ff5c}DSML\u{ff5c}function_calls>\n\
        <\u{ff5c}DSML\u{ff5c}invoke name=\"process\">\n\
        <\u{ff5c}DSML\u{ff5c}parameter name=\"count\" string=\"false\">5</\u{ff5c}DSML\u{ff5c}parameter>\n\
        <\u{ff5c}DSML\u{ff5c}parameter name=\"rate\" string=\"false\">7.5</\u{ff5c}DSML\u{ff5c}parameter>\n\
        <\u{ff5c}DSML\u{ff5c}parameter name=\"enabled\" string=\"false\">true</\u{ff5c}DSML\u{ff5c}parameter>\n\
        <\u{ff5c}DSML\u{ff5c}parameter name=\"text\" string=\"true\">hello world</\u{ff5c}DSML\u{ff5c}parameter>\n\
        </\u{ff5c}DSML\u{ff5c}invoke>\n\
        </\u{ff5c}DSML\u{ff5c}function_calls>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "process");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    // string="false" values should be parsed as their JSON types
    assert_eq!(args["count"], 5);
    assert_eq!(args["rate"], 7.5);
    assert_eq!(args["enabled"], true);
    // string="true" value should be a raw string
    assert_eq!(args["text"], "hello world");
}

#[tokio::test]
async fn test_dsml_string_false_fallback_to_string() {
    let parser = DeepSeekV32Parser::new();

    // When string="false" but the value is not valid JSON, it should fall back to string
    let input = "<\u{ff5c}DSML\u{ff5c}function_calls>\n\
        <\u{ff5c}DSML\u{ff5c}invoke name=\"search\">\n\
        <\u{ff5c}DSML\u{ff5c}parameter name=\"query\" string=\"false\">not valid json here</\u{ff5c}DSML\u{ff5c}parameter>\n\
        </\u{ff5c}DSML\u{ff5c}invoke>\n\
        </\u{ff5c}DSML\u{ff5c}function_calls>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "search");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    // Should fall back to string when JSON parse fails
    assert_eq!(args["query"], "not valid json here");
}

#[tokio::test]
async fn test_dsml_multiple_invocations() {
    let parser = DeepSeekV32Parser::new();

    let input = "<\u{ff5c}DSML\u{ff5c}function_calls>\n\
        <\u{ff5c}DSML\u{ff5c}invoke name=\"search\">\n\
        <\u{ff5c}DSML\u{ff5c}parameter name=\"query\" string=\"true\">rust programming</\u{ff5c}DSML\u{ff5c}parameter>\n\
        </\u{ff5c}DSML\u{ff5c}invoke>\n\
        <\u{ff5c}DSML\u{ff5c}invoke name=\"translate\">\n\
        {\"text\": \"Hello World\", \"to\": \"ja\"}\n\
        </\u{ff5c}DSML\u{ff5c}invoke>\n\
        </\u{ff5c}DSML\u{ff5c}function_calls>";

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
async fn test_dsml_streaming_chunked() {
    let tools = create_test_tools();
    let mut parser = DeepSeekV32Parser::new();

    // Simulate streaming chunks that build up a complete DSML invoke
    let chunks = vec![
        "<\u{ff5c}DSML\u{ff5c}function_calls>\n",
        "<\u{ff5c}DSML\u{ff5c}invoke name=\"get_weather\">\n",
        "{\"location\": ",
        "\"Beijing\", ",
        "\"units\": \"metric\"}",
        "\n</\u{ff5c}DSML\u{ff5c}invoke>",
        "\n</\u{ff5c}DSML\u{ff5c}function_calls>",
    ];

    let mut found_name = false;
    let mut found_args = false;
    let mut collected_args_chunks: Vec<String> = Vec::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        for call in result.calls {
            if let Some(name) = &call.name {
                assert_eq!(name, "get_weather");
                found_name = true;
            }
            if call.name.is_none() && !call.parameters.is_empty() {
                found_args = true;
                collected_args_chunks.push(call.parameters.clone());
            }
        }
    }

    assert!(found_name, "Should have found tool name during streaming");
    assert!(
        found_args,
        "Should have found tool arguments during streaming"
    );

    let full_args: String = collected_args_chunks.join("");
    let _: serde_json::Value =
        serde_json::from_str(&full_args).expect("Collected argument chunks should form valid JSON");
}

#[tokio::test]
async fn test_dsml_streaming_multiple_invocations() {
    let tools = create_test_tools();
    let mut parser = DeepSeekV32Parser::new();

    let chunks = vec![
        "<\u{ff5c}DSML\u{ff5c}function_calls>\n",
        "<\u{ff5c}DSML\u{ff5c}invoke name=\"search\">\n",
        "{\"query\": \"rust\"}\n",
        "</\u{ff5c}DSML\u{ff5c}invoke>\n",
        "<\u{ff5c}DSML\u{ff5c}invoke name=\"translate\">\n",
        "{\"text\": \"hello\", \"to\": \"ja\"}\n",
        "</\u{ff5c}DSML\u{ff5c}invoke>\n",
        "</\u{ff5c}DSML\u{ff5c}function_calls>",
    ];

    let mut names_found: Vec<String> = Vec::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        for call in result.calls {
            if let Some(name) = call.name {
                names_found.push(name);
            }
        }
    }

    assert_eq!(names_found.len(), 2);
    assert_eq!(names_found[0], "search");
    assert_eq!(names_found[1], "translate");
}

#[test]
fn test_dsml_format_detection() {
    let parser = DeepSeekV32Parser::new();

    // Should detect DSML format
    assert!(parser.has_tool_markers("<\u{ff5c}DSML\u{ff5c}function_calls>"));
    assert!(parser.has_tool_markers("text with <\u{ff5c}DSML\u{ff5c}function_calls> marker"));

    // Should not detect other formats
    assert!(!parser.has_tool_markers("<\u{ff5c}tool\u{2581}calls\u{2581}begin\u{ff5c}>"));
    assert!(!parser.has_tool_markers("[TOOL_CALLS]"));
    assert!(!parser.has_tool_markers("<tool_call>"));
    assert!(!parser.has_tool_markers("plain text"));
}

#[tokio::test]
async fn test_dsml_normal_text_before_block() {
    let parser = DeepSeekV32Parser::new();

    let input = "Sure, I'll look that up for you!\n\
        <\u{ff5c}DSML\u{ff5c}function_calls>\n\
        <\u{ff5c}DSML\u{ff5c}invoke name=\"search\">\n\
        {\"query\": \"rust async\"}\n\
        </\u{ff5c}DSML\u{ff5c}invoke>\n\
        </\u{ff5c}DSML\u{ff5c}function_calls>";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(normal_text, "Sure, I'll look that up for you!\n");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "search");
}

#[tokio::test]
async fn test_dsml_no_tool_markers() {
    let parser = DeepSeekV32Parser::new();

    let input = "This is just normal text with no tool calls.";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(normal_text, input);
    assert!(tools.is_empty());
}

#[tokio::test]
async fn test_dsml_nested_json_in_parameter() {
    let parser = DeepSeekV32Parser::new();

    let input = "<\u{ff5c}DSML\u{ff5c}function_calls>\n\
        <\u{ff5c}DSML\u{ff5c}invoke name=\"process\">\n\
        <\u{ff5c}DSML\u{ff5c}parameter name=\"data\" string=\"false\">{\"nested\": {\"deep\": [1, 2, 3]}}</\u{ff5c}DSML\u{ff5c}parameter>\n\
        </\u{ff5c}DSML\u{ff5c}invoke>\n\
        </\u{ff5c}DSML\u{ff5c}function_calls>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "process");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert!(args["data"]["nested"]["deep"].is_array());
    assert_eq!(args["data"]["nested"]["deep"][0], 1);
}

#[tokio::test]
async fn test_dsml_empty_args() {
    let parser = DeepSeekV32Parser::new();

    // Invoke with empty body -- should produce empty object
    let input = "<\u{ff5c}DSML\u{ff5c}function_calls>\n\
        <\u{ff5c}DSML\u{ff5c}invoke name=\"ping\">\n\
        </\u{ff5c}DSML\u{ff5c}invoke>\n\
        </\u{ff5c}DSML\u{ff5c}function_calls>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "ping");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert!(args.is_object());
}

#[tokio::test]
async fn test_dsml_streaming_xml_parameters() {
    let tools = create_test_tools();
    let mut parser = DeepSeekV32Parser::new();

    // Stream an invoke with XML parameters
    let chunks = vec![
        "<\u{ff5c}DSML\u{ff5c}function_calls>\n",
        "<\u{ff5c}DSML\u{ff5c}invoke name=\"process\">\n",
        "<\u{ff5c}DSML\u{ff5c}parameter name=\"count\" ",
        "string=\"false\">42",
        "</\u{ff5c}DSML\u{ff5c}parameter>\n",
        "<\u{ff5c}DSML\u{ff5c}parameter name=\"text\" string=\"true\">hello</\u{ff5c}DSML\u{ff5c}parameter>\n",
        "</\u{ff5c}DSML\u{ff5c}invoke>",
        "\n</\u{ff5c}DSML\u{ff5c}function_calls>",
    ];

    let mut found_name = false;
    let mut found_args = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        for call in result.calls {
            if let Some(name) = &call.name {
                assert_eq!(name, "process");
                found_name = true;
            }
            if call.name.is_none() && !call.parameters.is_empty() {
                found_args = true;
                // Verify the arguments are valid JSON
                let args: serde_json::Value = serde_json::from_str(&call.parameters).unwrap();
                assert_eq!(args["count"], 42);
                assert_eq!(args["text"], "hello");
            }
        }
    }

    assert!(found_name, "Should have found tool name during streaming");
    assert!(
        found_args,
        "Should have found tool arguments during streaming"
    );
}

#[tokio::test]
async fn test_dsml_reset() {
    let tools = create_test_tools();
    let mut parser = DeepSeekV32Parser::new();

    // Feed partial data
    let _ = parser
        .parse_incremental(
            "<\u{ff5c}DSML\u{ff5c}function_calls>\n<\u{ff5c}DSML\u{ff5c}invoke name=\"search\">\n",
            &tools,
        )
        .await
        .unwrap();

    // Reset
    parser.reset();

    // After reset, reuse the SAME parser instance to verify state was cleared
    let chunks = vec![
        "<\u{ff5c}DSML\u{ff5c}function_calls>\n",
        "<\u{ff5c}DSML\u{ff5c}invoke name=\"get_weather\">\n",
        "{\"location\": \"Paris\"}\n",
        "</\u{ff5c}DSML\u{ff5c}invoke>\n",
        "</\u{ff5c}DSML\u{ff5c}function_calls>",
    ];

    let mut found_name = false;
    let mut found_args = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = &call.name {
                assert_eq!(name, "get_weather");
                found_name = true;
            }
            if call.name.is_none() && !call.parameters.is_empty() {
                let args: serde_json::Value = serde_json::from_str(&call.parameters).unwrap();
                assert_eq!(args["location"], "Paris");
                found_args = true;
            }
        }
    }

    assert!(found_name, "After reset, parser should find tool name");
    assert!(found_args, "After reset, parser should find tool arguments");
}
