//! DeepSeek V3 Parser Integration Tests
mod common;

use common::create_test_tools;
use tool_parser::{DeepSeekParser, ToolParser};

#[tokio::test]
async fn test_deepseek_complete_parsing() {
    let parser = DeepSeekParser::new();

    let input = r#"Let me help you with that.
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"location": "Tokyo", "units": "celsius"}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
The weather in Tokyo is..."#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "Let me help you with that.\n");
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Tokyo");
    assert_eq!(args["units"], "celsius");
}

#[tokio::test]
async fn test_deepseek_multiple_tools() {
    let parser = DeepSeekParser::new();

    let input = r#"<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>search
```json
{"query": "rust programming"}
```<｜tool▁call▁end｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>translate
```json
{"text": "Hello World", "to": "ja"}
```<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.name, "search");
    assert_eq!(tools[1].function.name, "translate");
}

#[tokio::test]
async fn test_deepseek_streaming() {
    let tools = create_test_tools();

    let mut parser = DeepSeekParser::new();

    // Simulate streaming chunks
    let chunks = vec![
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>",
        "function<｜tool▁sep｜>get_weather\n",
        "```json\n",
        r#"{"location": "#,
        r#""Beijing", "#,
        r#""units": "metric"}"#,
        "\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
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
async fn test_deepseek_nested_json() {
    let parser = DeepSeekParser::new();

    let input = r#"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>process
```json
{
    "data": {
        "nested": {
            "deep": [1, 2, 3]
        }
    }
}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "process");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert!(args["data"]["nested"]["deep"].is_array());
}

#[test]
fn test_deepseek_format_detection() {
    let parser = DeepSeekParser::new();

    // Should detect DeepSeek format
    assert!(parser.has_tool_markers("<｜tool▁calls▁begin｜>"));
    assert!(parser.has_tool_markers("text with <｜tool▁calls▁begin｜> marker"));

    // Should not detect other formats
    assert!(!parser.has_tool_markers("[TOOL_CALLS]"));
    assert!(!parser.has_tool_markers("<tool_call>"));
    assert!(!parser.has_tool_markers("plain text"));
}

#[tokio::test]
async fn test_deepseek_malformed_json_handling() {
    let parser = DeepSeekParser::new();

    // Malformed JSON should be skipped
    let input = r#"<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>broken
```json
{invalid json}
```<｜tool▁call▁end｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>valid
```json
{"key": "value"}
```<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    // Only the valid tool call should be parsed
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "valid");
}

#[tokio::test]
async fn test_deepseek_streaming_final_chunk_strips_fence_and_markers() {
    // Regression: when the closing ```` ``` ```` fence and end tokens arrive in the
    // same chunk as the final JSON bytes, the greedy partial regex captures them into
    // the arguments group. They must be stripped before the diff/completion checks,
    // otherwise the fence/markers get streamed as argument content and
    // is_complete_json never returns true (the tool call never completes).
    let tools = create_test_tools();
    let mut parser = DeepSeekParser::new();

    // First chunk: emit through the function name so the args stream begins.
    parser
        .parse_incremental(
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n",
            &tools,
        )
        .await
        .unwrap();

    // Final chunk: complete JSON immediately followed by the closing fence and end markers.
    let final_chunk =
        "{\"location\": \"Tokyo\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>";
    let result = parser.parse_incremental(final_chunk, &tools).await.unwrap();

    // Accumulate every argument delta streamed on the final chunk.
    let streamed_args: String = result
        .calls
        .iter()
        .filter(|c| c.name.is_none())
        .map(|c| c.parameters.as_str())
        .collect();

    // No fence or end-marker bytes may leak into streamed argument content.
    assert!(
        !streamed_args.contains("```"),
        "streamed args leaked the closing fence: {streamed_args:?}"
    );
    assert!(
        !streamed_args.contains('｜'),
        "streamed args leaked an end marker: {streamed_args:?}"
    );

    // The streamed arguments must be exactly the clean JSON object and parse cleanly.
    assert_eq!(streamed_args, r#"{"location": "Tokyo"}"#);
    let parsed: serde_json::Value = serde_json::from_str(&streamed_args).unwrap();
    assert_eq!(parsed["location"], "Tokyo");

    // The tool call must complete on this final chunk: the parser advances to the next
    // tool and consumes the finished call from its buffer. A following second tool call
    // is therefore parsed as a new tool (index 1) with its own clean name and args -
    // which only happens if the first call completed instead of staying open.
    let second = parser
        .parse_incremental(
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"location\": \"Paris\"}\n```<｜tool▁call▁end｜>",
            &tools,
        )
        .await
        .unwrap();
    assert!(
        second
            .calls
            .iter()
            .any(|c| c.name.as_deref() == Some("get_weather") && c.tool_index == 1),
        "second tool call not started as a new tool: {:?}",
        second.calls
    );
}

#[tokio::test]
async fn test_deepseek_streaming_normal_text_strips_end_of_sentence() {
    // Regression: streamed normal text (no tool call) must strip the end-of-sentence
    // marker, matching deepseek31 — otherwise it leaks into client-visible content.
    let tools = create_test_tools();
    let mut parser = DeepSeekParser::new();

    let result = parser
        .parse_incremental("All done.<｜end▁of▁sentence｜>", &tools)
        .await
        .unwrap();

    assert!(
        !result.normal_text.contains('｜'),
        "end-of-sentence marker leaked into normal text: {:?}",
        result.normal_text
    );
    assert_eq!(result.normal_text, "All done.");
    assert!(result.calls.is_empty());
}

#[tokio::test]
async fn test_multiple_tool_calls() {
    let parser = DeepSeekParser::new();

    let input = r#"<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"location": "Tokyo"}
```<｜tool▁call▁end｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"location": "Paris"}
```<｜tool▁call▁end｜>
<｜tool▁calls▁end｜><｜end▁of▁sentence｜>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.name, "get_weather");
    assert_eq!(tools[1].function.name, "get_weather");
}
