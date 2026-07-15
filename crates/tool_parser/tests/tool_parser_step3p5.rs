//! Step-3.5 Tool Parser Integration Tests
//!
//! Tests for the Step-3.5 XML tool parser, which handles the framing:
//! <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
mod common;

use common::create_test_tools;
use tool_parser::{ParserFactory, Step3p5Parser, ToolParser};

#[tokio::test]
async fn test_step3p5_factory_routing() {
    let factory = ParserFactory::new();
    let registry = factory.registry();

    assert!(factory.has_parser("step3p5"));

    // Step-3.5 IDs route to step3p5 (longest trailing-* prefix wins).
    for model in [
        "step3.5",
        "step3.5-large",
        "step-3.5",
        "Step-3.5",
        "stepfun-ai/step3.5",
        "stepfun-ai/Step-3.5-chat",
    ] {
        assert_eq!(
            registry.resolve_model_to_parser(model).as_deref(),
            Some("step3p5"),
            "model {model} should resolve to step3p5"
        );
    }

    // Plain Step3 IDs still resolve to step3.
    for model in ["step3", "step3-model", "Step-3", "Step-3-large"] {
        assert_eq!(
            registry.resolve_model_to_parser(model).as_deref(),
            Some("step3"),
            "model {model} should resolve to step3"
        );
    }
}

#[tokio::test]
async fn test_step3p5_single_tool() {
    let parser = Step3p5Parser::new();
    let input = r"<tool_call>
<function=get_weather>
<parameter=city>Beijing</parameter>
<parameter=units>celsius</parameter>
</function>
</tool_call>";

    let (_normal, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Beijing");
    assert_eq!(args["units"], "celsius");
}

#[tokio::test]
async fn test_step3p5_normal_text_before_call() {
    let parser = Step3p5Parser::new();
    let input = r"Let me check that for you.
<tool_call>
<function=search>
<parameter=query>rust async</parameter>
</function>
</tool_call>";

    let (normal, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "search");
    assert_eq!(normal, "Let me check that for you.\n");
}

#[tokio::test]
async fn test_step3p5_parallel_tools() {
    let parser = Step3p5Parser::new();
    let input = r"<tool_call>
<function=get_weather>
<parameter=city>Tokyo</parameter>
</function>
</tool_call>
<tool_call>
<function=search>
<parameter=query>weather forecast</parameter>
</function>
</tool_call>";

    let (_normal, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.name, "get_weather");
    assert_eq!(tools[1].function.name, "search");
}

#[tokio::test]
async fn test_step3p5_empty_parameters() {
    let parser = Step3p5Parser::new();
    let input = r"<tool_call>
<function=get_time>
</function>
</tool_call>";

    let (_normal, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_time");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args, serde_json::json!({}));
}

#[tokio::test]
async fn test_step3p5_empty_parameter_value() {
    let parser = Step3p5Parser::new();
    let input = r"<tool_call>
<function=process>
<parameter=text></parameter>
<parameter=count>5</parameter>
</function>
</tool_call>";

    let (_normal, tools) = parser
        .parse_complete_with_tools(input, &create_test_tools())
        .await
        .unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "");
    // count is declared "number" in the test schema.
    assert_eq!(args["count"], 5);
}

#[tokio::test]
async fn test_step3p5_no_markers_passthrough() {
    let parser = Step3p5Parser::new();
    let input = "This is a plain response with no tool calls at all. \
Even if it mentions get_weather or search, they are not calls.";

    let (normal, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal, input);
}

#[tokio::test]
async fn test_step3p5_format_detection() {
    let parser = Step3p5Parser::new();
    assert!(parser.has_tool_markers("<tool_call>"));
    assert!(parser.has_tool_markers("hello <tool_call>"));
    assert!(!parser.has_tool_markers("plain text"));
    assert!(!parser.has_tool_markers("<function=test>"));
}

#[tokio::test]
async fn test_step3p5_type_coercion_with_schema() {
    let parser = Step3p5Parser::new();
    let tools = create_test_tools();

    // `process` declares: count(number), rate(number), enabled(boolean),
    // data(object), text(string).
    let input = r#"<tool_call>
<function=process>
<parameter=count>42</parameter>
<parameter=rate>1.5</parameter>
<parameter=enabled>true</parameter>
<parameter=data>{"k": [1, 2]}</parameter>
<parameter=text>123</parameter>
</function>
</tool_call>"#;

    let (_normal, calls) = parser
        .parse_complete_with_tools(input, &tools)
        .await
        .unwrap();
    assert_eq!(calls.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["count"], 42);
    assert_eq!(args["rate"], 1.5);
    assert_eq!(args["enabled"], true);
    assert_eq!(args["data"], serde_json::json!({"k": [1, 2]}));
    // Declared as string, so a numeric-looking value stays a string.
    assert_eq!(args["text"], "123");
}

#[tokio::test]
async fn test_step3p5_inference_without_schema() {
    let parser = Step3p5Parser::new();

    // No tools supplied: values are inferred JSON-first.
    let input = r"<tool_call>
<function=process>
<parameter=count>42</parameter>
<parameter=text>hello</parameter>
</function>
</tool_call>";

    let (_normal, calls) = parser.parse_complete(input).await.unwrap();
    assert_eq!(calls.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    // Inferred: numeric text parses as a number.
    assert_eq!(args["count"], 42);
    assert_eq!(args["text"], "hello");
}

#[tokio::test]
async fn test_step3p5_streaming_basic() {
    let mut parser = Step3p5Parser::new();
    let tools = create_test_tools();

    let chunks = vec![
        "<tool_call>",
        "<function=get_weather>",
        "<parameter=city>Shanghai</parameter>",
        "<parameter=units>celsius</parameter>",
        "</function>",
        "</tool_call>",
    ];

    let mut found_name = false;
    let mut found_params = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                found_name = true;
            }
            if !call.parameters.is_empty() {
                found_params = true;
            }
        }
    }

    assert!(found_name, "Should have found tool name during streaming");
    assert!(found_params, "Should have streamed parameters");
}

#[tokio::test]
async fn test_step3p5_streaming_across_chunk_boundaries() {
    let mut parser = Step3p5Parser::new();
    let tools = create_test_tools();

    // Tags split mid-token across chunk boundaries.
    let chunks = vec![
        "<tool_c",
        "all><function=",
        "get_weather><param",
        "eter=city>Bei",
        "jing</parameter></func",
        "tion></tool_call>",
    ];

    let mut found_name = false;
    let mut streamed_args = String::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                found_name = true;
            }
            streamed_args.push_str(&call.parameters);
        }
    }

    assert!(found_name, "Should parse function name from partial chunks");
    assert!(
        streamed_args.contains("city"),
        "Should stream the city parameter, got: {streamed_args}"
    );
}

#[tokio::test]
async fn test_step3p5_streaming_realistic_char_chunks() {
    let mut parser = Step3p5Parser::new();
    let tools = create_test_tools();

    let input = r"<tool_call>
<function=get_weather>
<parameter=city>Tokyo</parameter>
<parameter=units>celsius</parameter>
</function>
</tool_call>";

    let mut got_tool_name = false;
    for chunk in common::streaming_helpers::create_realistic_chunks(input) {
        let result = parser.parse_incremental(&chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                got_tool_name = true;
            }
        }
    }

    assert!(
        got_tool_name,
        "Should parse tool name from char-level chunks"
    );
}

#[tokio::test]
async fn test_step3p5_streaming_no_markers_returns_text() {
    let mut parser = Step3p5Parser::new();
    let tools = create_test_tools();

    let result = parser
        .parse_incremental("just some normal text", &tools)
        .await
        .unwrap();
    assert_eq!(result.normal_text, "just some normal text");
    assert!(result.calls.is_empty());
}

#[tokio::test]
async fn test_step3p5_streaming_parallel() {
    let mut parser = Step3p5Parser::new();
    let tools = create_test_tools();

    let chunks = vec![
        "<tool_call><function=get_weather><parameter=city>Tokyo</parameter></function></tool_call>",
        "<tool_call><function=search><parameter=query>forecast</parameter></function></tool_call>",
    ];

    let mut tool_names = Vec::new();
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                tool_names.push(name);
            }
        }
    }

    assert_eq!(tool_names, vec!["get_weather", "search"]);
}

#[tokio::test]
async fn test_step3p5_reset_between_requests() {
    let mut parser = Step3p5Parser::new();
    let tools = create_test_tools();

    // First request.
    for chunk in [
        "<tool_call><function=get_weather>",
        "<parameter=city>London</parameter>",
        "</function></tool_call>",
    ] {
        parser.parse_incremental(chunk, &tools).await.unwrap();
    }

    parser.reset();

    // Second request after reset.
    let mut second_tool_name = None;
    for chunk in [
        "<tool_call><function=search>",
        "<parameter=query>rust</parameter>",
        "</function></tool_call>",
    ] {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                second_tool_name = Some(name);
            }
        }
    }

    assert_eq!(second_tool_name, Some("search".to_string()));
}

#[tokio::test]
async fn test_step3p5_invalid_function_name_skipped() {
    let mut parser = Step3p5Parser::new();
    let tools = create_test_tools();

    let chunks = vec![
        "<tool_call>",
        "<function=not_a_real_tool>",
        "<parameter=x>1</parameter>",
        "</function></tool_call>",
    ];

    let mut found_invalid = false;
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if call.name.as_deref() == Some("not_a_real_tool") {
                found_invalid = true;
            }
        }
    }

    assert!(!found_invalid, "Invalid function should not be emitted");
}

// ============================================================================
// Net-new robustness coverage (parallels qwen_xml / minimax_m2 suites).
// ============================================================================

#[tokio::test]
async fn test_step3p5_text_before_and_after_call() {
    let parser = Step3p5Parser::new();
    let input = r"I'll analyze the weather for you now.
<tool_call>
<function=get_weather>
<parameter=city>Boston</parameter>
<parameter=units>celsius</parameter>
</function>
</tool_call>
Based on the analysis, here's what I found.";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");

    // `extract_all` returns only the text preceding the first `<tool_call>`;
    // trailing prose after the block is not part of normal_text.
    assert!(normal_text.contains("I'll analyze the weather for you now."));
    assert!(!normal_text.contains("Based on the analysis"));

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Boston");
    assert_eq!(args["units"], "celsius");
}

#[tokio::test]
async fn test_step3p5_incomplete_tool_call_returns_text() {
    let parser = Step3p5Parser::new();

    // Open `<tool_call>` with no `</tool_call>`: the extractor regex requires a
    // closing tag, so no call is produced and the input is returned verbatim.
    let input = r"<tool_call>
<function=get_weather>
<parameter=city>Chicago</parameter>";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(
        tools.len(),
        0,
        "must not emit a bogus call for a truncated block"
    );
    assert_eq!(normal_text, input);
}

#[tokio::test]
async fn test_step3p5_malformed_function_tag_unclosed() {
    let parser = Step3p5Parser::new();

    // `<function=get_weather` is never closed with its own `>`. REAL behavior:
    // `xml_function_pattern` is `<function=([^>]+)>`, and `[^>]+` is greedy and
    // matches across newlines, so it swallows everything up to the FIRST `>` it
    // can find -- the one closing `<parameter=city>`. The captured "name" is thus
    // the malformed blob `get_weather\n<parameter=city`, NOT a clean `get_weather`.
    // The parser still emits one call (the name is non-empty), documenting that
    // it does not silently drop a malformed-but-non-empty function tag.
    let input = r"<tool_call>
<function=get_weather
<parameter=city>Miami</parameter>
</function>
</tool_call>";

    let (_normal, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_ne!(
        tools[0].function.name, "get_weather",
        "the unclosed tag over-captures, so the name is not the clean value"
    );
    assert!(
        tools[0].function.name.contains("get_weather"),
        "over-captured name still contains the intended prefix, got: {}",
        tools[0].function.name
    );
}

#[tokio::test]
async fn test_step3p5_missing_function_name() {
    let parser = Step3p5Parser::new();

    // `<function=>` has an empty name: the `[^>]+` group requires at least one
    // char, so it does not match; the call is dropped.
    let input = r"<tool_call>
<function=>
<parameter=city>Miami</parameter>
</function>
</tool_call>";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input);
}

#[tokio::test]
async fn test_step3p5_special_characters_in_values() {
    let parser = Step3p5Parser::new();

    // Note: `<`/`>`-like text inside a value works as long as it does not form a
    // literal `</parameter>`; the non-greedy regex stops at the first close tag.
    let input = r#"<tool_call>
<function=process>
<parameter=text>Special chars: @#$%^&*()</parameter>
<parameter=emoji>🦀 Rust 🚀 你好</parameter>
<parameter=quotes>"double" and 'single' quotes</parameter>
<parameter=angles>a < b and c > d</parameter>
</function>
</tool_call>"#;

    let (_normal, tools) = parser
        .parse_complete_with_tools(input, &create_test_tools())
        .await
        .unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    // `text` is declared `string`, so it is kept verbatim.
    assert_eq!(args["text"], "Special chars: @#$%^&*()");
    assert_eq!(args["emoji"], "🦀 Rust 🚀 你好");
    assert_eq!(args["quotes"], "\"double\" and 'single' quotes");
    assert_eq!(args["angles"], "a < b and c > d");
}

#[tokio::test]
async fn test_step3p5_whitespace_handling() {
    let parser = Step3p5Parser::new();
    let tools = create_test_tools();

    // Indented tags and padded values. `text` is a declared `string` so it is
    // preserved verbatim (no trim); inferred values are trimmed before parsing.
    let input = r"<tool_call>
    <function=process>
        <parameter=text>  spaces around  </parameter>
        <parameter=count>  7  </parameter>
    </function>
</tool_call>";

    let (_normal, tools) = parser
        .parse_complete_with_tools(input, &tools)
        .await
        .unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    // string schema keeps surrounding whitespace verbatim.
    assert_eq!(args["text"], "  spaces around  ");
    // number schema trims before parsing.
    assert_eq!(args["count"], 7);
}

#[tokio::test]
async fn test_step3p5_many_parameters() {
    let parser = Step3p5Parser::new();

    let mut params_xml = String::new();
    for i in 1..=10 {
        params_xml.push_str(&format!("<parameter=param{i}>value{i}</parameter>\n"));
    }
    let input =
        format!("<tool_call>\n<function=complex_func>\n{params_xml}</function>\n</tool_call>");

    let (_normal, tools) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "complex_func");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    for i in 1..=10 {
        assert_eq!(args[format!("param{i}")], format!("value{i}"));
    }
}

#[tokio::test]
async fn test_step3p5_unknown_function_complete_is_forwarded() {
    let parser = Step3p5Parser::new();
    let tools = create_test_tools();

    // REAL behavior: `parse_xml_format` (used by `extract_all` for complete
    // parsing) only rejects an *empty* function name; it does NOT validate the
    // name against the tool list. So an unknown function is FORWARDED as a call
    // in complete parsing. (Contrast streaming `parse_incremental`, which checks
    // `tool_indices` and drops unknown names — see
    // `test_step3p5_invalid_function_name_skipped`.)
    let input = r"<tool_call>
<function=not_a_real_tool>
<parameter=x>1</parameter>
</function>
</tool_call>";

    let (_normal, calls) = parser
        .parse_complete_with_tools(input, &tools)
        .await
        .unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "not_a_real_tool");
}

#[tokio::test]
async fn test_step3p5_empty_tools_slice_infers_json_first() {
    let parser = Step3p5Parser::new();

    // No schema: values are inferred JSON-first with a string fallback
    // (`infer_value`): numbers/bools/null/objects/arrays parse, everything else
    // stays a string.
    let input = r#"<tool_call>
<function=process>
<parameter=int_val>42</parameter>
<parameter=float_val>2.5</parameter>
<parameter=bool_val>true</parameter>
<parameter=null_val>null</parameter>
<parameter=obj_val>{"a": 1}</parameter>
<parameter=arr_val>[1, 2, 3]</parameter>
<parameter=str_val>hello world</parameter>
</function>
</tool_call>"#;

    let (_normal, calls) = parser.parse_complete_with_tools(input, &[]).await.unwrap();
    assert_eq!(calls.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["int_val"], 42);
    assert_eq!(args["float_val"], 2.5);
    assert_eq!(args["bool_val"], true);
    assert_eq!(args["null_val"], serde_json::Value::Null);
    assert_eq!(args["obj_val"], serde_json::json!({"a": 1}));
    assert_eq!(args["arr_val"], serde_json::json!([1, 2, 3]));
    assert_eq!(args["str_val"], "hello world");
}

#[tokio::test]
async fn test_step3p5_invalid_value_with_object_schema_falls_back_to_string() {
    let parser = Step3p5Parser::new();
    let tools = create_test_tools();

    // `data` is declared `object` in the `process` schema. An unparseable value
    // must degrade gracefully to the raw string (vLLM's `repair_param_type`
    // fallback in `coerce_value`), not error out the whole call.
    let input = r"<tool_call>
<function=process>
<parameter=data>{not valid json at all</parameter>
<parameter=text>ok</parameter>
</function>
</tool_call>";

    let (_normal, calls) = parser
        .parse_complete_with_tools(input, &tools)
        .await
        .unwrap();
    assert_eq!(calls.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["data"], "{not valid json at all");
    assert_eq!(args["text"], "ok");
}

#[tokio::test]
async fn test_step3p5_multiline_parameter_values() {
    let parser = Step3p5Parser::new();
    let tools = create_test_tools();

    let input = r"<tool_call>
<function=process>
<parameter=text>Line 1
Line 2
Line 3</parameter>
</function>
</tool_call>";

    let (_normal, tools) = parser
        .parse_complete_with_tools(input, &tools)
        .await
        .unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    // string schema preserves the embedded newlines verbatim.
    assert_eq!(args["text"], "Line 1\nLine 2\nLine 3");
}

#[tokio::test]
async fn test_step3p5_multiple_functions_in_one_block() {
    let parser = Step3p5Parser::new();

    // The grammar wraps one function per `<tool_call>` block. If a model emits
    // two `<function=>` declarations inside a single block, `parse_xml_format`
    // captures only the FIRST function name (`xml_function_pattern.captures`),
    // but `captures_iter` over the parameter pattern collects EVERY parameter in
    // the block. Documenting this real behavior.
    let input = r"<tool_call>
<function=get_weather>
<parameter=city>Tokyo</parameter>
</function>
<function=search>
<parameter=query>forecast</parameter>
</function>
</tool_call>";

    let (_normal, calls) = parser.parse_complete(input).await.unwrap();
    assert_eq!(calls.len(), 1, "one tool_call block yields one call");
    assert_eq!(calls[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    // Both blocks' params are merged into the single emitted call.
    assert_eq!(args["city"], "Tokyo");
    assert_eq!(args["query"], "forecast");
}

#[tokio::test]
async fn test_step3p5_back_to_back_blocks_streaming() {
    let mut parser = Step3p5Parser::new();
    let tools = create_test_tools();

    // Two complete blocks concatenated with no separator, fed in one chunk.
    let chunk = "<tool_call><function=get_weather><parameter=city>Tokyo</parameter></function></tool_call><tool_call><function=search><parameter=query>rust</parameter></function></tool_call>";

    let mut names = Vec::new();
    let result = parser.parse_incremental(chunk, &tools).await.unwrap();
    for call in result.calls {
        if let Some(name) = call.name {
            names.push(name);
        }
    }
    assert_eq!(names, vec!["get_weather", "search"]);
}

#[tokio::test]
async fn test_step3p5_streaming_split_inside_function_tag() {
    let mut parser = Step3p5Parser::new();
    let tools = create_test_tools();

    // The split lands in the middle of `<function=get_weather>`.
    let chunks = vec![
        "<tool_call><function=get_w",
        "eather><parameter=city>Berlin</parameter></function></tool_call>",
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
    assert!(
        found_name,
        "name must resolve once the function tag completes"
    );
}

#[tokio::test]
async fn test_step3p5_streaming_split_inside_parameter_tag() {
    let mut parser = Step3p5Parser::new();
    let tools = create_test_tools();

    // The split lands in the middle of `<parameter=city>`.
    let chunks = vec![
        "<tool_call><function=get_weather><param",
        "eter=city>Berlin</parameter></function></tool_call>",
    ];

    let mut streamed_args = String::new();
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            streamed_args.push_str(&call.parameters);
        }
    }
    assert!(
        streamed_args.contains("city"),
        "city parameter must stream once the tag completes, got: {streamed_args}"
    );
}

#[tokio::test]
async fn test_step3p5_streaming_char_by_char_with_surrounding_text() {
    let mut parser = Step3p5Parser::new();
    let tools = create_test_tools();

    let complete = "Let me help. <tool_call><function=get_weather><parameter=city>Seattle</parameter></function></tool_call> Done.";

    let mut content = String::new();
    let mut tool_name_found = false;
    // Feed one byte at a time (ASCII-only input, so byte == char here).
    for i in 0..complete.len() {
        let delta = &complete[i..i + 1];
        let result = parser.parse_incremental(delta, &tools).await.unwrap();
        content.push_str(&result.normal_text);
        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                tool_name_found = true;
            }
        }
    }

    assert!(
        tool_name_found,
        "name must be found under char-by-char streaming"
    );
    assert!(content.contains("Let me help."));
}

#[tokio::test]
async fn test_step3p5_streaming_tiny_multibyte_bursts() {
    let mut parser = Step3p5Parser::new();
    let tools = create_test_tools();

    // Multi-byte value streamed via realistic 2-3 char chunks (the helper splits
    // on char boundaries, so multi-byte chars are never torn).
    let input = r"<tool_call>
<function=get_weather>
<parameter=city>北京市</parameter>
</function>
</tool_call>";

    let mut found_name = false;
    let mut streamed_args = String::new();
    for chunk in common::streaming_helpers::create_realistic_chunks(input) {
        let result = parser.parse_incremental(&chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                found_name = true;
            }
            streamed_args.push_str(&call.parameters);
        }
    }
    assert!(found_name);
    assert!(
        streamed_args.contains("北京市"),
        "multi-byte value must stream intact, got: {streamed_args}"
    );
}

#[tokio::test]
async fn test_step3p5_streaming_coercion_branches_with_schema() {
    let mut parser = Step3p5Parser::new();
    let tools = create_test_tools();

    // Drive every coercion branch end-to-end through streaming using the
    // `process` schema (count/rate: number, enabled: boolean, text: string) plus
    // a literal `null`. The float-with-whole-value branch (rate=2.0 -> 2) and the
    // string-kept-verbatim branch (text="123") are both exercised here.
    let chunks = vec![
        "<tool_call><function=process>",
        "<parameter=count>42</parameter>",
        "<parameter=rate>2.0</parameter>",
        "<parameter=enabled>true</parameter>",
        "<parameter=text>123</parameter>",
        "<parameter=data>null</parameter>",
        "</function></tool_call>",
    ];

    let mut args_json = String::new();
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if call.name.is_none() {
                args_json.push_str(&call.parameters);
            }
        }
    }

    let args: serde_json::Value = serde_json::from_str(&args_json).unwrap();
    assert_eq!(args["count"], 42, "int branch");
    assert_eq!(args["rate"], 2, "float->int when whole");
    assert_eq!(args["enabled"], true, "bool branch");
    assert_eq!(args["text"], "123", "string kept verbatim");
    assert_eq!(args["data"], serde_json::Value::Null, "null literal branch");
}

#[tokio::test]
async fn test_step3p5_streaming_float_kept_as_float() {
    let mut parser = Step3p5Parser::new();
    let tools = create_test_tools();

    // Non-whole float must stay a float through streaming.
    let chunks = vec![
        "<tool_call><function=process>",
        "<parameter=rate>1.5</parameter>",
        "</function></tool_call>",
    ];

    let mut args_json = String::new();
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if call.name.is_none() {
                args_json.push_str(&call.parameters);
            }
        }
    }

    let args: serde_json::Value = serde_json::from_str(&args_json).unwrap();
    assert_eq!(args["rate"], 1.5);
}
