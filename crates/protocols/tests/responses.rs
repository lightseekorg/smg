//! Protocol-surface roundtrip tests for the OpenAI Responses API types.

use openai_protocol::{
    common::{
        ContextManagementType, Detail, PromptCacheRetention, ToolChoice as ChatToolChoice,
        ToolChoiceValue as ChatToolChoiceValue, ToolReference,
    },
    responses::*,
};
use serde_json::json;

#[test]
fn summary_text_content_round_trips_spec_shape() {
    // Spec: `summary: array of SummaryTextContent { text, type: "summary_text" }`.
    let item: ResponseOutputItem = serde_json::from_value(json!({
        "type": "reasoning",
        "id": "r_1",
        "summary": [{"text": "step 1", "type": "summary_text"}],
        "content": [],
        "status": "completed",
    }))
    .expect("spec-shape reasoning should deserialize");

    match &item {
        ResponseOutputItem::Reasoning { summary, .. } => {
            assert_eq!(summary.len(), 1);
            match &summary[0] {
                SummaryTextContent::SummaryText { text } => {
                    assert_eq!(text, "step 1");
                }
            }
        }
        _ => panic!("expected Reasoning variant"),
    }

    let v = serde_json::to_value(&item).expect("serialize");
    assert_eq!(
        v["summary"],
        json!([{"text": "step 1", "type": "summary_text"}])
    );
}

#[test]
fn legacy_vec_string_summary_fails_to_deserialize() {
    let legacy = r#"{"type":"reasoning","id":"r_x","summary":["text"]}"#;
    let result: Result<ResponseInputOutputItem, _> = serde_json::from_str(legacy);
    assert!(
        result.is_err(),
        "legacy Vec<String> summary must no longer deserialize"
    );
}

#[test]
fn test_responses_request_omitted_top_p_deserializes_to_none() {
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": "hello"
    }))
    .expect("request should deserialize");

    assert_eq!(request.top_p, None);

    let serialized = serde_json::to_value(&request).expect("request should serialize");
    assert!(serialized.get("top_p").is_none());
}

#[test]
fn test_responses_request_null_top_p_deserializes_to_none() {
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": "hello",
        "top_p": null
    }))
    .expect("request should deserialize");

    assert_eq!(request.top_p, None);

    let serialized = serde_json::to_value(&request).expect("request should serialize");
    assert!(serialized.get("top_p").is_none());
}

#[test]
fn test_responses_request_explicit_top_p_preserved() {
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": "hello",
        "top_p": 0.9
    }))
    .expect("request should deserialize");

    assert_eq!(request.top_p, Some(0.9));
}

#[test]
fn test_require_approval_string_round_trip() {
    let tool: McpTool = serde_json::from_value(json!({
        "server_label": "deepwiki",
        "require_approval": "never"
    }))
    .expect("mcp tool should deserialize");

    assert_eq!(
        tool.require_approval,
        Some(RequireApproval::Mode(RequireApprovalMode::Never))
    );

    let serialized = serde_json::to_value(&tool).expect("mcp tool should serialize");
    assert_eq!(serialized["require_approval"], json!("never"));
}

#[test]
fn test_require_approval_object_round_trip() {
    let tool: McpTool = serde_json::from_value(json!({
        "server_label": "deepwiki",
        "require_approval": {
            "always": null,
            "never": {
                "tool_names": ["ask_question", "read_wiki_structure"],
                "read_only": null
            }
        }
    }))
    .expect("mcp tool should deserialize");

    assert_eq!(
        tool.require_approval,
        Some(RequireApproval::Rules(RequireApprovalRules {
            always: None,
            never: Some(RequireApprovalFilter {
                tool_names: Some(vec![
                    "ask_question".to_string(),
                    "read_wiki_structure".to_string()
                ]),
                read_only: None,
            }),
        }))
    );

    let serialized = serde_json::to_value(&tool).expect("mcp tool should serialize");
    assert_eq!(
        serialized["require_approval"],
        json!({
            "never": {
                "tool_names": ["ask_question", "read_wiki_structure"]
            }
        })
    );
}

#[test]
fn test_include_field_web_search_call_variants_round_trip() {
    let fields: Vec<IncludeField> = serde_json::from_value(json!([
        "web_search_call.action.sources",
        "web_search_call.results",
    ]))
    .expect("include fields should deserialize");

    assert_eq!(
        fields,
        vec![
            IncludeField::WebSearchCallActionSources,
            IncludeField::WebSearchCallResults,
        ]
    );

    let serialized = serde_json::to_value(&fields).expect("include fields should serialize");
    assert_eq!(
        serialized,
        json!(["web_search_call.action.sources", "web_search_call.results"])
    );
}

#[test]
fn test_file_search_tool_round_trip() {
    // Spec fixture (openai-responses-api-spec.md §tools): `file_search` with
    // vector_store_ids, compound filter, and ranking_options incl. hybrid_search,
    // ranker enum, and score_threshold.
    let payload = json!({
        "type": "file_search",
        "vector_store_ids": ["vs_1234567890"],
        "max_num_results": 20,
        "filters": {
            "type": "and",
            "filters": [
                {"type": "eq", "key": "region", "value": "us-east-1"},
                {"type": "in", "key": "tag", "value": ["alpha", "beta"]}
            ]
        },
        "ranking_options": {
            "ranker": "default-2024-11-15",
            "score_threshold": 0.5,
            "hybrid_search": {"embedding_weight": 0.7, "text_weight": 0.3}
        }
    });

    let tool: ResponseTool =
        serde_json::from_value(payload.clone()).expect("file_search tool should deserialize");
    assert!(matches!(tool, ResponseTool::FileSearch(_)));

    let serialized = serde_json::to_value(&tool).expect("file_search tool should serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_file_search_tool_round_trip_hybrid_weights_omitted() {
    // Spec (openai-responses-api-spec.md §tools): `embedding_weight` and
    // `text_weight` are optional. When omitted they must deserialize to
    // `None` and be absent again on re-serialization (absent→None→absent).
    let payload = json!({
        "type": "file_search",
        "vector_store_ids": ["vs_1234567890"],
        "ranking_options": {
            "hybrid_search": {}
        }
    });

    let tool: ResponseTool = serde_json::from_value(payload.clone())
        .expect("file_search tool with empty hybrid_search should deserialize");
    match &tool {
        ResponseTool::FileSearch(fs) => {
            let ranking = fs
                .ranking_options
                .as_ref()
                .expect("ranking_options should be present");
            let hybrid = ranking
                .hybrid_search
                .as_ref()
                .expect("hybrid_search should be present");
            assert!(hybrid.embedding_weight.is_none());
            assert!(hybrid.text_weight.is_none());
        }
        other => panic!("expected FileSearch variant, got {other:?}"),
    }

    let serialized = serde_json::to_value(&tool).expect("file_search tool should serialize");
    assert_eq!(serialized, payload);
}

// ------------------------------------------------------------------
// T3: non-preview web_search tool + WebSearchCall.results
// ------------------------------------------------------------------

/// Spec fixture (openai-responses-api-spec.md §tools line 439):
/// `{ type: "web_search" | "web_search_2025_08_26", filters? { allowed_domains? },
/// search_context_size?: "low"|"medium"|"high", user_location? }`. Covers the
/// canonical tag, the versioned alias, and full-field nested shape.
#[test]
fn test_web_search_tool_round_trip() {
    let payload = json!({
        "type": "web_search",
        "filters": {"allowed_domains": ["example.com", "rust-lang.org"]},
        "search_context_size": "high",
        "user_location": {
            "type": "approximate",
            "city": "San Francisco",
            "country": "US",
            "region": "California",
            "timezone": "America/Los_Angeles"
        }
    });
    let tool: ResponseTool =
        serde_json::from_value(payload.clone()).expect("web_search tool should deserialize");
    assert!(matches!(tool, ResponseTool::WebSearch(_)));
    assert_eq!(
        serde_json::to_value(&tool).expect("web_search tool should serialize"),
        payload
    );

    // Versioned alias deserializes into the same variant (canonical serialization re-tested above).
    let alias: ResponseTool = serde_json::from_value(json!({"type": "web_search_2025_08_26"}))
        .expect("web_search_2025_08_26 alias should deserialize");
    assert!(matches!(alias, ResponseTool::WebSearch(_)));
}

/// Acceptance: `web_search_call` output item carries an optional typed
/// `results` field populated when callers request `web_search_call.results`
/// via the top-level `include[]` array. When absent, the default wire shape
/// `{id, action, status, type}` must stay spec-byte-identical.
#[test]
fn test_web_search_call_results_round_trip() {
    let with_results = json!({
        "type": "web_search_call",
        "id": "ws_abc",
        "status": "completed",
        "action": {"type": "search", "query": "rust", "queries": ["rust"]},
        "results": [
            {"url": "https://tokio.rs", "title": "Tokio", "snippet": "rt", "score": 0.5},
            {"url": "https://async.rs"}
        ]
    });
    let item: ResponseOutputItem = serde_json::from_value(with_results.clone())
        .expect("web_search_call with results should deserialize");
    let ResponseOutputItem::WebSearchCall { results, .. } = &item else {
        panic!("expected WebSearchCall");
    };
    let results = results.as_ref().expect("results present");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].score, Some(0.5));
    assert!(results[1].title.is_none());
    assert_eq!(
        serde_json::to_value(&item).expect("web_search_call should serialize"),
        with_results
    );

    // Absent results: deserializes to None and re-serializes without the key.
    let no_results = json!({
        "type": "web_search_call",
        "id": "ws_no",
        "status": "completed",
        "action": {"type": "search"}
    });
    let bare: ResponseOutputItem = serde_json::from_value(no_results.clone())
        .expect("web_search_call without results should deserialize");
    let ResponseOutputItem::WebSearchCall { results, .. } = &bare else {
        panic!("expected WebSearchCall");
    };
    assert!(results.is_none());
    let serialized = serde_json::to_value(&bare).expect("web_search_call should serialize");
    assert_eq!(serialized, no_results);
    assert!(serialized.get("results").is_none());
}

// ------------------------------------------------------------------
// P2: new top-level ResponsesRequest fields
// ------------------------------------------------------------------

/// Acceptance: the six new top-level fields deserialize and re-serialize
/// without loss (`prompt`, `prompt_cache_key`, `prompt_cache_retention`,
/// `safety_identifier`, `stream_options`, `context_management`).
#[test]
fn test_responses_request_new_top_level_fields_round_trip() {
    let payload = json!({
        "model": "gpt-5.4",
        "input": "hello",
        "prompt": {
            "id": "pmpt_abc",
            "variables": {
                "name": "ada",
                "picture": {
                    "type": "input_image",
                    "image_url": "https://example.com/pic.png",
                    "detail": "high"
                }
            },
            "version": "1"
        },
        "prompt_cache_key": "pck-123",
        "prompt_cache_retention": "24h",
        "safety_identifier": "sid-123",
        "stream_options": { "include_obfuscation": false },
        "context_management": [{
            "type": "compaction",
            "compact_threshold": 4096
        }]
    });

    let request: ResponsesRequest =
        serde_json::from_value(payload.clone()).expect("request should deserialize");

    assert!(request.prompt.is_some());
    assert_eq!(
        request.prompt.as_ref().map(|p| p.id.as_str()),
        Some("pmpt_abc")
    );
    assert_eq!(
        request.prompt.as_ref().and_then(|p| p.version.as_deref()),
        Some("1")
    );
    assert_eq!(request.prompt_cache_key.as_deref(), Some("pck-123"));
    assert_eq!(
        request.prompt_cache_retention,
        Some(PromptCacheRetention::Duration24h)
    );
    assert_eq!(request.safety_identifier.as_deref(), Some("sid-123"));
    assert_eq!(
        request
            .stream_options
            .as_ref()
            .and_then(|s| s.include_obfuscation),
        Some(false)
    );
    let ctx = request
        .context_management
        .as_ref()
        .expect("context_management must round-trip");
    assert_eq!(ctx.len(), 1);
    assert_eq!(ctx[0].r#type, ContextManagementType::Compaction);
    assert_eq!(ctx[0].compact_threshold, Some(4096));

    // Re-serialize and confirm the wire form matches the inputs.
    let reserialized = serde_json::to_value(&request).expect("should serialize");
    assert_eq!(reserialized["prompt"]["id"], "pmpt_abc");
    assert_eq!(reserialized["prompt_cache_key"], "pck-123");
    assert_eq!(reserialized["prompt_cache_retention"], "24h");
    assert_eq!(reserialized["safety_identifier"], "sid-123");
    assert_eq!(reserialized["stream_options"]["include_obfuscation"], false);
    assert_eq!(reserialized["context_management"][0]["type"], "compaction");
    assert_eq!(
        reserialized["context_management"][0]["compact_threshold"],
        4096
    );
}

/// `prompt_cache_retention` accepts the other spec value and serializes
/// back with the hyphenated rename.
#[test]
fn test_prompt_cache_retention_in_memory_round_trip() {
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": "hello",
        "prompt_cache_retention": "in-memory"
    }))
    .expect("should deserialize");

    assert_eq!(
        request.prompt_cache_retention,
        Some(PromptCacheRetention::InMemory)
    );

    let reserialized = serde_json::to_value(&request).expect("should serialize");
    assert_eq!(reserialized["prompt_cache_retention"], "in-memory");
}

/// Absent fields must stay absent on the wire (no `"prompt": null` etc.),
/// matching every other `Option<_>` field on `ResponsesRequest`.
#[test]
fn test_responses_request_new_fields_omitted_when_absent() {
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": "hello"
    }))
    .expect("should deserialize");

    assert!(request.prompt.is_none());
    assert!(request.prompt_cache_key.is_none());
    assert!(request.prompt_cache_retention.is_none());
    assert!(request.safety_identifier.is_none());
    assert!(request.stream_options.is_none());
    assert!(request.context_management.is_none());

    let serialized = serde_json::to_value(&request).expect("should serialize");
    for key in [
        "prompt",
        "prompt_cache_key",
        "prompt_cache_retention",
        "safety_identifier",
        "stream_options",
        "context_management",
    ] {
        assert!(
            serialized.get(key).is_none(),
            "field {key} should be skipped when absent"
        );
    }
}

// ---- P1: rich content parts + typed annotations ----

#[test]
fn content_part_input_image_roundtrip() {
    // `original` is the spec-new detail level; exercising it here
    // also covers the rest of the variant shape.
    let raw = json!({
        "type": "input_image",
        "detail": "original",
        "image_url": "https://example.com/cat.png"
    });
    let part: ResponseContentPart =
        serde_json::from_value(raw.clone()).expect("input_image should deserialize");
    match &part {
        ResponseContentPart::InputImage {
            detail,
            file_id,
            image_url,
        } => {
            assert!(matches!(detail, Some(Detail::Original)));
            assert!(file_id.is_none());
            assert_eq!(image_url.as_deref(), Some("https://example.com/cat.png"));
        }
        other => panic!("expected InputImage, got {other:?}"),
    }
    let roundtripped = serde_json::to_value(&part).unwrap();
    assert_eq!(roundtripped, raw);
}

#[test]
fn content_part_input_file_roundtrip() {
    let raw = json!({
        "type": "input_file",
        "detail": "low",
        "file_data": "BASE64DATA",
        "filename": "report.pdf"
    });
    let part: ResponseContentPart =
        serde_json::from_value(raw.clone()).expect("input_file should deserialize");
    match &part {
        ResponseContentPart::InputFile {
            detail,
            file_data,
            file_id,
            file_url,
            filename,
        } => {
            assert!(matches!(detail, Some(FileDetail::Low)));
            assert_eq!(file_data.as_deref(), Some("BASE64DATA"));
            assert!(file_id.is_none());
            assert!(file_url.is_none());
            assert_eq!(filename.as_deref(), Some("report.pdf"));
        }
        other => panic!("expected InputFile, got {other:?}"),
    }
    let roundtripped = serde_json::to_value(&part).unwrap();
    assert_eq!(roundtripped, raw);
}

#[test]
fn content_part_refusal_roundtrip() {
    let raw = json!({ "type": "refusal", "refusal": "I cannot help with that." });
    let part: ResponseContentPart =
        serde_json::from_value(raw.clone()).expect("refusal should deserialize");
    match &part {
        ResponseContentPart::Refusal { refusal } => {
            assert_eq!(refusal, "I cannot help with that.");
        }
        other => panic!("expected Refusal, got {other:?}"),
    }
    let roundtripped = serde_json::to_value(&part).unwrap();
    assert_eq!(roundtripped, raw);
}

#[test]
fn content_part_output_text_with_typed_annotations() {
    let raw = json!({
        "type": "output_text",
        "text": "See [1] and [2].",
        "annotations": [
            {
                "type": "file_citation",
                "file_id": "file-abc",
                "filename": "source.pdf",
                "index": 5
            },
            {
                "type": "url_citation",
                "url": "https://example.com",
                "title": "Example",
                "start_index": 0,
                "end_index": 18
            },
            {
                "type": "container_file_citation",
                "container_id": "cntr-1",
                "file_id": "file-xyz",
                "filename": "out.txt",
                "start_index": 0,
                "end_index": 4
            },
            {
                "type": "file_path",
                "file_id": "file-gen",
                "index": 7
            }
        ]
    });
    let part: ResponseContentPart =
        serde_json::from_value(raw.clone()).expect("output_text should deserialize");
    match &part {
        ResponseContentPart::OutputText {
            text,
            annotations,
            logprobs,
        } => {
            assert_eq!(text, "See [1] and [2].");
            assert_eq!(annotations.len(), 4);
            assert!(logprobs.is_none());
            assert!(matches!(&annotations[0], Annotation::FileCitation { .. }));
            assert!(matches!(&annotations[1], Annotation::UrlCitation { .. }));
            assert!(matches!(
                &annotations[2],
                Annotation::ContainerFileCitation { .. }
            ));
            assert!(matches!(&annotations[3], Annotation::FilePath { .. }));
        }
        other => panic!("expected OutputText, got {other:?}"),
    }
    let roundtripped = serde_json::to_value(&part).unwrap();
    assert_eq!(roundtripped, raw);
}

#[test]
fn content_part_unknown_type_fails_fast() {
    // Previously `#[serde(other)] Unknown` silently swallowed unknown
    // types; P1 removes that arm so spec-invalid payloads fail cleanly.
    let raw = json!({ "type": "totally_made_up", "text": "x" });
    let err = serde_json::from_value::<ResponseContentPart>(raw)
        .expect_err("unknown content-part type must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("totally_made_up") || msg.contains("unknown variant"),
        "error should mention the unknown variant, got: {msg}"
    );
}

// ------------------------------------------------------------------
// P5: ResponseInputOutputItem fail-fast on unknown `type` discriminator
//
// `ResponseInputOutputItem` is `#[serde(tag = "type")]` with a single
// `#[serde(untagged)] SimpleInputMessage` fallback. Before P5, that
// fallback accepted any `{role, content, type: "<anything>"}` payload
// because `type` was `Option<String>`. The spec (EasyInputMessage.type)
// constrains `type` to absent or `"message"`, so P5 replaces the loose
// `Option<String>` with `Option<SimpleInputMessageTypeTag>` — unknown
// discriminators must now return a deserialize error (→ 400).
// ------------------------------------------------------------------

#[test]
fn input_item_unknown_type_fails_fast() {
    // Spec-invalid discriminator must NOT silently fall into
    // SimpleInputMessage. Previously `type: "totally_made_up"` deserialized
    // cleanly because `r#type` was `Option<String>`.
    let raw = json!({
        "type": "totally_made_up",
        "role": "user",
        "content": "hello"
    });
    let err = serde_json::from_value::<ResponseInputOutputItem>(raw)
        .expect_err("unknown input-item type must fail");
    let msg = err.to_string();
    assert!(
        !msg.is_empty(),
        "deserialize error must carry a message, got empty: {msg}"
    );
}

#[test]
fn simple_input_message_without_type_still_deserializes() {
    // Per spec, `EasyInputMessage.type` is optional. Omitting it must
    // continue to route to the untagged SimpleInputMessage variant.
    let raw = json!({
        "role": "user",
        "content": "hello"
    });
    let item: ResponseInputOutputItem =
        serde_json::from_value(raw).expect("missing type must still deserialize");
    match item {
        ResponseInputOutputItem::SimpleInputMessage {
            role,
            content,
            r#type,
            ..
        } => {
            assert_eq!(role, "user");
            assert!(matches!(content, StringOrContentParts::String(ref s) if s == "hello"));
            assert!(r#type.is_none(), "absent type must deserialize as None");
        }
        other => panic!("expected SimpleInputMessage, got {other:?}"),
    }
}

#[test]
fn simple_input_message_with_type_message_deserializes() {
    // Per spec, `type: "message"` is the only non-null value permitted on
    // EasyInputMessage. Round-trip it intact.
    let raw = json!({
        "type": "message",
        "role": "user",
        "content": "hi"
    });
    let item: ResponseInputOutputItem =
        serde_json::from_value(raw.clone()).expect("`type: message` must deserialize");
    match &item {
        ResponseInputOutputItem::SimpleInputMessage { r#type, .. } => {
            assert_eq!(*r#type, Some(SimpleInputMessageTypeTag::Message));
        }
        other => panic!("expected SimpleInputMessage, got {other:?}"),
    }
    // Serializes back to the same wire bytes (canonical shape).
    let roundtripped = serde_json::to_value(&item).expect("serialize");
    assert_eq!(roundtripped, raw);
}

// ------------------------------------------------------------------
// P7: ResponsesToolChoice round-trip tests (all 8 spec variants)
// ------------------------------------------------------------------

/// Options: `"none"` | `"auto"` | `"required"` deserialize as bare strings
/// and re-serialize identically.
#[test]
fn responses_tool_choice_options_round_trip() {
    for (payload, expected) in [
        (json!("none"), ToolChoiceOptions::None),
        (json!("auto"), ToolChoiceOptions::Auto),
        (json!("required"), ToolChoiceOptions::Required),
    ] {
        let choice: ResponsesToolChoice =
            serde_json::from_value(payload.clone()).expect("options string should deserialize");
        match &choice {
            ResponsesToolChoice::Options(opt) => assert_eq!(opt, &expected),
            other => panic!("expected Options, got {other:?}"),
        }
        let reserialized = serde_json::to_value(&choice).expect("serialize");
        assert_eq!(reserialized, payload);
    }
}

/// Types: `{"type": "<built-in>"}` for every hosted tool the spec enumerates.
#[test]
fn responses_tool_choice_types_round_trip() {
    for (payload, expected) in [
        (
            json!({"type": "file_search"}),
            BuiltInToolChoiceType::FileSearch,
        ),
        (
            json!({"type": "web_search"}),
            BuiltInToolChoiceType::WebSearch,
        ),
        (
            json!({"type": "web_search_preview"}),
            BuiltInToolChoiceType::WebSearchPreview,
        ),
        (
            json!({"type": "web_search_preview_2025_03_11"}),
            BuiltInToolChoiceType::WebSearchPreview20250311,
        ),
        (
            json!({"type": "image_generation"}),
            BuiltInToolChoiceType::ImageGeneration,
        ),
        (
            json!({"type": "computer_use_preview"}),
            BuiltInToolChoiceType::ComputerUsePreview,
        ),
        (
            json!({"type": "code_interpreter"}),
            BuiltInToolChoiceType::CodeInterpreter,
        ),
    ] {
        let choice: ResponsesToolChoice =
            serde_json::from_value(payload.clone()).expect("types object should deserialize");
        match &choice {
            ResponsesToolChoice::Types { tool_type } => assert_eq!(tool_type, &expected),
            other => panic!("expected Types, got {other:?}"),
        }
        let reserialized = serde_json::to_value(&choice).expect("serialize");
        assert_eq!(reserialized, payload);
    }
}

/// Function: `{"type": "function", "name": "..."}` — spec-canonical flat
/// shape round-trips unchanged.
#[test]
fn responses_tool_choice_function_round_trip() {
    let payload = json!({"type": "function", "name": "get_weather"});
    let choice: ResponsesToolChoice =
        serde_json::from_value(payload.clone()).expect("function flat shape should deserialize");
    match &choice {
        ResponsesToolChoice::Function(payload) => assert_eq!(payload.name, "get_weather"),
        other => panic!("expected Function, got {other:?}"),
    }
    assert_eq!(choice.function_name(), Some("get_weather"));
    let reserialized = serde_json::to_value(&choice).expect("serialize");
    assert_eq!(reserialized, payload);
}

/// Backward compat: the Chat-style nested `{"function": {"name": ...}}`
/// wire shape was accepted on the Responses endpoint before the P7
/// split and existing smg clients + e2e tests rely on it. We accept it
/// on deserialize (Postel: liberal on input) but always emit the
/// canonical flat shape on serialize (conservative on output).
#[test]
fn responses_tool_choice_function_accepts_legacy_nested() {
    let legacy_nested = json!({
        "type": "function",
        "function": {"name": "search_web"}
    });
    let choice: ResponsesToolChoice = serde_json::from_value(legacy_nested)
        .expect("legacy nested function shape must deserialize for backward compat");

    // Internal state is normalized to the flat `name` field regardless of
    // which wire shape we read.
    match &choice {
        ResponsesToolChoice::Function(payload) => {
            assert_eq!(payload.name, "search_web");
            assert_eq!(payload.tool_type, FunctionToolChoiceTag::Function);
        }
        other => panic!("expected Function, got {other:?}"),
    }
    assert_eq!(choice.function_name(), Some("search_web"));

    // Serialize MUST emit the canonical flat shape, not the nested legacy
    // shape that came in on the wire.
    let reserialized = serde_json::to_value(&choice).expect("serialize");
    assert_eq!(
        reserialized,
        json!({"type": "function", "name": "search_web"}),
        "Responses `Function` must serialize as canonical flat even when \
         deserialized from the legacy nested shape"
    );
    assert!(
        reserialized.get("function").is_none(),
        "canonical flat serialize must not emit a nested `function` object"
    );
}

/// Payloads without either `name` or `function.name` must fail —
/// we accept both wire shapes, but at least one of them must carry the
/// function name. (The exact error string comes from serde's untagged
/// enum routing which swallows inner errors, so we assert only that
/// deserialization fails.)
#[test]
fn responses_tool_choice_function_rejects_missing_name() {
    let payload = json!({"type": "function"});
    assert!(
        serde_json::from_value::<ResponsesToolChoice>(payload).is_err(),
        "function without `name` or `function.name` must be rejected",
    );
}

/// AllowedTools: `{"type": "allowed_tools", "mode", "tools"}` with Function
/// references; validates ToolReference interop.
#[test]
fn responses_tool_choice_allowed_tools_round_trip() {
    let payload = json!({
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [{"type": "function", "name": "get_weather"}]
    });
    let choice: ResponsesToolChoice =
        serde_json::from_value(payload.clone()).expect("allowed_tools should deserialize");
    match &choice {
        ResponsesToolChoice::AllowedTools { mode, tools, .. } => {
            assert_eq!(mode, "auto");
            assert_eq!(tools.len(), 1);
            match &tools[0] {
                ToolReference::Function { name } => assert_eq!(name, "get_weather"),
                other => panic!("expected Function reference, got {other:?}"),
            }
        }
        other => panic!("expected AllowedTools, got {other:?}"),
    }
    let reserialized = serde_json::to_value(&choice).expect("serialize");
    assert_eq!(reserialized, payload);
}

/// Mcp: `{"type": "mcp", "server_label", "name"?}` with and without the
/// optional `name`.
#[test]
fn responses_tool_choice_mcp_round_trip() {
    // With `name`
    let payload_with_name = json!({
        "type": "mcp",
        "server_label": "my_server",
        "name": "search"
    });
    let choice: ResponsesToolChoice = serde_json::from_value(payload_with_name.clone())
        .expect("mcp with name should deserialize");
    match &choice {
        ResponsesToolChoice::Mcp {
            server_label, name, ..
        } => {
            assert_eq!(server_label, "my_server");
            assert_eq!(name.as_deref(), Some("search"));
        }
        other => panic!("expected Mcp, got {other:?}"),
    }
    let reserialized = serde_json::to_value(&choice).expect("serialize");
    assert_eq!(reserialized, payload_with_name);

    // Without `name`
    let payload_no_name = json!({
        "type": "mcp",
        "server_label": "my_server"
    });
    let choice: ResponsesToolChoice = serde_json::from_value(payload_no_name.clone())
        .expect("mcp without name should deserialize");
    match &choice {
        ResponsesToolChoice::Mcp {
            server_label, name, ..
        } => {
            assert_eq!(server_label, "my_server");
            assert!(name.is_none());
        }
        other => panic!("expected Mcp, got {other:?}"),
    }
    let reserialized = serde_json::to_value(&choice).expect("serialize");
    assert_eq!(reserialized, payload_no_name);
}

/// Custom: `{"type": "custom", "name": "..."}` — user-registered tool by name.
#[test]
fn responses_tool_choice_custom_round_trip() {
    let payload = json!({"type": "custom", "name": "my_custom_tool"});
    let choice: ResponsesToolChoice =
        serde_json::from_value(payload.clone()).expect("custom should deserialize");
    match &choice {
        ResponsesToolChoice::Custom { name, .. } => assert_eq!(name, "my_custom_tool"),
        other => panic!("expected Custom, got {other:?}"),
    }
    let reserialized = serde_json::to_value(&choice).expect("serialize");
    assert_eq!(reserialized, payload);
}

/// ApplyPatch: `{"type": "apply_patch"}`.
#[test]
fn responses_tool_choice_apply_patch_round_trip() {
    let payload = json!({"type": "apply_patch"});
    let choice: ResponsesToolChoice =
        serde_json::from_value(payload.clone()).expect("apply_patch should deserialize");
    assert!(matches!(choice, ResponsesToolChoice::ApplyPatch { .. }));
    let reserialized = serde_json::to_value(&choice).expect("serialize");
    assert_eq!(reserialized, payload);
}

/// Shell: `{"type": "shell"}`.
#[test]
fn responses_tool_choice_shell_round_trip() {
    let payload = json!({"type": "shell"});
    let choice: ResponsesToolChoice =
        serde_json::from_value(payload.clone()).expect("shell should deserialize");
    assert!(matches!(choice, ResponsesToolChoice::Shell { .. }));
    let reserialized = serde_json::to_value(&choice).expect("serialize");
    assert_eq!(reserialized, payload);
}

/// Projection onto Chat Completions' ToolChoice enum.
/// - Options map to the same Value variants.
/// - Function flattens to Chat's nested `{function: {name}}` wire form.
/// - AllowedTools preserves mode and tool list.
/// - Hosted / custom / apply_patch / shell / mcp have no Chat equivalent
///   and fall back to `auto`.
#[test]
fn responses_tool_choice_to_chat_projection() {
    assert!(matches!(
        ResponsesToolChoice::Options(ToolChoiceOptions::None).to_chat_tool_choice(),
        ChatToolChoice::Value(ChatToolChoiceValue::None)
    ));
    assert!(matches!(
        ResponsesToolChoice::Options(ToolChoiceOptions::Auto).to_chat_tool_choice(),
        ChatToolChoice::Value(ChatToolChoiceValue::Auto)
    ));
    assert!(matches!(
        ResponsesToolChoice::Options(ToolChoiceOptions::Required).to_chat_tool_choice(),
        ChatToolChoice::Value(ChatToolChoiceValue::Required)
    ));

    let fn_choice = ResponsesToolChoice::Function(ResponsesFunctionToolChoice {
        tool_type: FunctionToolChoiceTag::Function,
        name: "get_weather".to_string(),
    });
    match fn_choice.to_chat_tool_choice() {
        ChatToolChoice::Function {
            tool_type,
            function,
        } => {
            assert_eq!(tool_type, "function");
            assert_eq!(function.name, "get_weather");
        }
        other => panic!("expected Function, got {other:?}"),
    }

    let allowed = ResponsesToolChoice::AllowedTools {
        tool_type: AllowedToolsToolChoiceTag::AllowedTools,
        mode: "auto".to_string(),
        tools: vec![ToolReference::Function {
            name: "get_weather".to_string(),
        }],
    };
    match allowed.to_chat_tool_choice() {
        ChatToolChoice::AllowedTools {
            tool_type,
            mode,
            tools,
        } => {
            assert_eq!(tool_type, "allowed_tools");
            assert_eq!(mode, "auto");
            assert_eq!(tools.len(), 1);
        }
        other => panic!("expected AllowedTools, got {other:?}"),
    }

    // Responses-only variants collapse onto Chat's `auto` — there is no
    // spec-valid Chat projection for hosted / mcp / custom / apply_patch / shell.
    for variant in [
        ResponsesToolChoice::Types {
            tool_type: BuiltInToolChoiceType::FileSearch,
        },
        ResponsesToolChoice::Mcp {
            tool_type: McpToolChoiceTag::Mcp,
            server_label: "s".into(),
            name: None,
        },
        ResponsesToolChoice::Custom {
            tool_type: CustomToolChoiceTag::Custom,
            name: "c".into(),
        },
        ResponsesToolChoice::ApplyPatch {
            tool_type: ApplyPatchToolChoiceTag::ApplyPatch,
        },
        ResponsesToolChoice::Shell {
            tool_type: ShellToolChoiceTag::Shell,
        },
    ] {
        assert!(matches!(
            variant.to_chat_tool_choice(),
            ChatToolChoice::Value(ChatToolChoiceValue::Auto)
        ));
    }
}

/// The Default impl is `Options(Auto)`.
#[test]
fn responses_tool_choice_default_is_auto() {
    assert!(matches!(
        ResponsesToolChoice::default(),
        ResponsesToolChoice::Options(ToolChoiceOptions::Auto)
    ));
}

// ------------------------------------------------------------------
// T4: image_generation tool + image_generation_call item round-trips
// ------------------------------------------------------------------

/// `ResponseTool::ImageGeneration` — bare `{type}` and full spec payload
/// round-trip with every optional field preserved byte-for-byte.
#[test]
fn image_generation_tool_round_trips_spec_shape() {
    // Minimal spec shape: just the discriminator, no config.
    let minimal = json!({"type": "image_generation"});
    let tool: ResponseTool =
        serde_json::from_value(minimal.clone()).expect("minimal image_generation deserialize");
    match &tool {
        ResponseTool::ImageGeneration(cfg) => {
            assert!(cfg.action.is_none());
            assert!(cfg.background.is_none());
            assert!(cfg.input_fidelity.is_none());
            assert!(cfg.input_image_mask.is_none());
            assert!(cfg.model.is_none());
            assert!(cfg.moderation.is_none());
            assert!(cfg.output_compression.is_none());
            assert!(cfg.output_format.is_none());
            assert!(cfg.partial_images.is_none());
            assert!(cfg.quality.is_none());
            assert!(cfg.size.is_none());
        }
        other => panic!("expected ImageGeneration, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&tool).expect("serialize"), minimal);

    // Full spec shape: every optional field populated.
    let full = json!({
        "type": "image_generation",
        "action": "edit",
        "background": "transparent",
        "input_fidelity": "high",
        "input_image_mask": {
            "file_id": "file_abc",
            "image_url": "https://example.com/mask.png"
        },
        "model": "gpt-image-1",
        "moderation": "low",
        "output_compression": 80,
        "output_format": "png",
        "partial_images": 2,
        "quality": "high",
        "size": "1024x1024"
    });
    let tool: ResponseTool =
        serde_json::from_value(full.clone()).expect("full image_generation deserialize");
    match &tool {
        ResponseTool::ImageGeneration(cfg) => {
            assert_eq!(cfg.action.as_deref(), Some("edit"));
            assert_eq!(cfg.background.as_deref(), Some("transparent"));
            assert_eq!(cfg.input_fidelity.as_deref(), Some("high"));
            let mask = cfg.input_image_mask.as_ref().expect("mask present");
            assert_eq!(mask.file_id.as_deref(), Some("file_abc"));
            assert_eq!(
                mask.image_url.as_deref(),
                Some("https://example.com/mask.png")
            );
            assert_eq!(cfg.model.as_deref(), Some("gpt-image-1"));
            assert_eq!(cfg.moderation.as_deref(), Some("low"));
            assert_eq!(cfg.output_compression, Some(80));
            assert_eq!(cfg.output_format.as_deref(), Some("png"));
            assert_eq!(cfg.partial_images, Some(2));
            assert_eq!(cfg.quality.as_deref(), Some("high"));
            assert_eq!(cfg.size.as_deref(), Some("1024x1024"));
        }
        other => panic!("expected ImageGeneration, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&tool).expect("serialize"), full);
}

/// `ResponseOutputItem::ImageGenerationCall` — `{id, result, status, type}`
/// round-trips with every spec-listed status variant. Absent
/// `revised_prompt` deserializes to `None` and is omitted on serialize, so
/// the default wire shape stays byte-identical to the spec example.
#[test]
fn image_generation_call_output_item_round_trips_spec_shape() {
    for (status_json, expected) in [
        ("in_progress", ImageGenerationCallStatus::InProgress),
        ("completed", ImageGenerationCallStatus::Completed),
        ("generating", ImageGenerationCallStatus::Generating),
        ("failed", ImageGenerationCallStatus::Failed),
    ] {
        let payload = json!({
            "type": "image_generation_call",
            "id": "ig_1",
            "result": "aGVsbG8=",
            "status": status_json,
        });
        let item: ResponseOutputItem = serde_json::from_value(payload.clone())
            .expect("image_generation_call output item deserialize");
        match &item {
            ResponseOutputItem::ImageGenerationCall {
                id,
                result,
                revised_prompt,
                status,
            } => {
                assert_eq!(id, "ig_1");
                assert_eq!(result, "aGVsbG8=");
                assert!(revised_prompt.is_none());
                assert_eq!(*status, expected);
            }
            other => panic!("expected ImageGenerationCall, got {other:?}"),
        }
        assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
    }
}

/// `ResponseOutputItem::ImageGenerationCall` preserves `revised_prompt`
/// through a full (de)serialization cycle. OpenAI populates this when
/// the mainline model rewrites the user prompt before dispatching the
/// image-generation call; dropping it here would silently lose prompt
/// provenance during storage/replay.
#[test]
fn image_generation_call_output_item_round_trips_with_revised_prompt() {
    let payload = json!({
        "type": "image_generation_call",
        "id": "ig_1",
        "result": "aGVsbG8=",
        "revised_prompt": "A red fox sitting on a rock at sunset.",
        "status": "completed",
    });
    let item: ResponseOutputItem = serde_json::from_value(payload.clone())
        .expect("image_generation_call output item deserialize");
    match &item {
        ResponseOutputItem::ImageGenerationCall {
            id,
            result,
            revised_prompt,
            status,
        } => {
            assert_eq!(id, "ig_1");
            assert_eq!(result, "aGVsbG8=");
            assert_eq!(
                revised_prompt.as_deref(),
                Some("A red fox sitting on a rock at sunset.")
            );
            assert_eq!(*status, ImageGenerationCallStatus::Completed);
        }
        other => panic!("expected ImageGenerationCall, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

/// `ResponseInputOutputItem::ImageGenerationCall` mirrors the output-item
/// shape so clients can replay a prior-turn image generation through the
/// `input` array for stateless round-trips.
#[test]
fn image_generation_call_input_item_round_trips_spec_shape() {
    let payload = json!({
        "type": "image_generation_call",
        "id": "ig_2",
        "result": "d29ybGQ=",
        "status": "completed",
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("image_generation_call input item deserialize");
    match &item {
        ResponseInputOutputItem::ImageGenerationCall {
            id,
            result,
            revised_prompt,
            status,
        } => {
            assert_eq!(id, "ig_2");
            assert_eq!(result.as_deref(), Some("d29ybGQ="));
            assert!(revised_prompt.is_none());
            assert_eq!(*status, Some(ImageGenerationCallStatus::Completed));
        }
        other => panic!("expected ImageGenerationCall, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

/// `ResponseInputOutputItem::ImageGenerationCall` accepts the documented
/// multi-turn id-only reference form: clients resubmitting
/// `{ "type": "image_generation_call", "id": ... }` to continue an
/// image-edit conversation must deserialize and round-trip without
/// forcing the full `result` / `status` payload. (See OpenAI
/// Responses API image-generation tool guide.)
#[test]
fn image_generation_call_input_item_accepts_id_only_reference() {
    let payload = json!({
        "type": "image_generation_call",
        "id": "ig_3",
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("id-only image_generation_call input item deserialize");
    match &item {
        ResponseInputOutputItem::ImageGenerationCall {
            id,
            result,
            revised_prompt,
            status,
        } => {
            assert_eq!(id, "ig_3");
            assert!(result.is_none());
            assert!(revised_prompt.is_none());
            assert!(status.is_none());
        }
        other => panic!("expected ImageGenerationCall, got {other:?}"),
    }
    // Serialized form must stay minimal — no `"result": null` or
    // `"status": null` leaks through `skip_serializing_if`.
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

/// `ResponseInputOutputItem::ImageGenerationCall` preserves
/// `revised_prompt` on replay — the stateless client path must carry the
/// rewritten prompt forward to match what the output variant emitted on
/// the originating turn.
#[test]
fn image_generation_call_input_item_round_trips_with_revised_prompt() {
    let payload = json!({
        "type": "image_generation_call",
        "id": "ig_2",
        "result": "d29ybGQ=",
        "revised_prompt": "A cozy cabin in a snowy pine forest at dusk.",
        "status": "completed",
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("image_generation_call input item deserialize");
    match &item {
        ResponseInputOutputItem::ImageGenerationCall {
            id,
            result,
            revised_prompt,
            status,
        } => {
            assert_eq!(id, "ig_2");
            assert_eq!(result.as_deref(), Some("d29ybGQ="));
            assert_eq!(
                revised_prompt.as_deref(),
                Some("A cozy cabin in a snowy pine forest at dusk.")
            );
            assert_eq!(*status, Some(ImageGenerationCallStatus::Completed));
        }
        other => panic!("expected ImageGenerationCall, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn compaction_output_item_round_trips_spec_shape() {
    let payload = json!({
        "type": "compaction",
        "id": "cmp_abc123",
        "encrypted_content": "opaque-base64-blob",
    });
    let item: ResponseOutputItem =
        serde_json::from_value(payload.clone()).expect("compaction output item deserialize");
    match &item {
        ResponseOutputItem::Compaction {
            id,
            encrypted_content,
        } => {
            assert_eq!(id, "cmp_abc123");
            assert_eq!(encrypted_content, "opaque-base64-blob");
        }
        other => panic!("expected Compaction, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn compaction_input_item_round_trips_spec_shape() {
    let payload = json!({
        "type": "compaction",
        "id": "cmp_abc123",
        "encrypted_content": "opaque-base64-blob",
    });
    let item: ResponseInputOutputItem =
        serde_json::from_value(payload.clone()).expect("compaction input item deserialize");
    match &item {
        ResponseInputOutputItem::Compaction {
            encrypted_content,
            id,
        } => {
            assert_eq!(encrypted_content, "opaque-base64-blob");
            assert_eq!(id.as_deref(), Some("cmp_abc123"));
        }
        other => panic!("expected Compaction, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn compaction_input_item_accepts_missing_id() {
    let payload = json!({
        "type": "compaction",
        "encrypted_content": "opaque-base64-blob",
    });
    let item: ResponseInputOutputItem =
        serde_json::from_value(payload.clone()).expect("id-less compaction input item deserialize");
    match &item {
        ResponseInputOutputItem::Compaction {
            encrypted_content,
            id,
        } => {
            assert_eq!(encrypted_content, "opaque-base64-blob");
            assert!(id.is_none());
        }
        other => panic!("expected Compaction, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn computer_tool_round_trips_spec_shape() {
    let payload = json!({ "type": "computer" });
    let tool: ResponseTool = serde_json::from_value(payload.clone()).expect("deserialize");
    assert!(matches!(tool, ResponseTool::Computer));
    assert_eq!(serde_json::to_value(&tool).expect("serialize"), payload);
}

#[test]
fn computer_use_preview_tool_round_trips_spec_shape() {
    let payload = json!({
        "type": "computer_use_preview",
        "display_height": 1080,
        "display_width": 1920,
        "environment": "browser",
    });
    let tool: ResponseTool = serde_json::from_value(payload.clone()).expect("deserialize");
    match &tool {
        ResponseTool::ComputerUsePreview(cup) => {
            assert_eq!(cup.display_height, 1080);
            assert_eq!(cup.display_width, 1920);
            assert_eq!(cup.environment, ComputerEnvironment::Browser);
        }
        other => panic!("expected ComputerUsePreview, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&tool).expect("serialize"), payload);
}

#[test]
fn computer_action_variants_round_trip() {
    let fixtures = [
        json!({"type": "click", "button": "left", "x": 10, "y": 20}),
        json!({"type": "double_click", "x": 100, "y": 200}),
        json!({
            "type": "drag",
            "path": [{"x": 0, "y": 0}, {"x": 10, "y": 10}],
        }),
        json!({"type": "keypress", "keys": ["ctrl", "c"]}),
        json!({"type": "move", "x": 50, "y": 60}),
        json!({"type": "screenshot"}),
        json!({
            "type": "scroll",
            "scroll_x": 0,
            "scroll_y": -120,
            "x": 100,
            "y": 200,
        }),
        json!({"type": "type", "text": "hello"}),
        json!({"type": "wait"}),
    ];
    for payload in fixtures {
        let action: ComputerAction =
            serde_json::from_value(payload.clone()).expect("action deserialize");
        assert_eq!(
            serde_json::to_value(&action).expect("action serialize"),
            payload,
        );
    }
}

#[test]
fn test_custom_tool_text_format_round_trip() {
    // Spec (openai-responses-api-spec.md §tools, L471-474):
    // `Custom { name, type: "custom", defer_loading?, description?, format? }`
    // with `format: CustomToolInputFormat = Text { type: "text" }`.
    let payload = json!({
        "type": "custom",
        "name": "run_sql",
        "description": "Run a SELECT against the analytics warehouse",
        "defer_loading": false,
        "format": {"type": "text"}
    });

    let tool: ResponseTool = serde_json::from_value(payload.clone())
        .expect("custom tool with text format should deserialize");
    match &tool {
        ResponseTool::Custom(c) => {
            assert_eq!(c.name, "run_sql");
            assert_eq!(c.defer_loading, Some(false));
            assert!(matches!(c.format, Some(CustomToolInputFormat::Text)));
        }
        other => panic!("expected ResponseTool::Custom, got {other:?}"),
    }

    let serialized = serde_json::to_value(&tool).expect("custom tool should serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_custom_tool_grammar_format_round_trip() {
    // Spec (openai-responses-api-spec.md §tools, L472-474):
    // `Grammar { definition, syntax: "lark" | "regex", type: "grammar" }`.
    for syntax in ["lark", "regex"] {
        let payload = json!({
            "type": "custom",
            "name": "parse_expr",
            "format": {
                "type": "grammar",
                "definition": "start: NUMBER",
                "syntax": syntax
            }
        });

        let tool: ResponseTool = serde_json::from_value(payload.clone())
            .expect("custom tool with grammar format should deserialize");
        match &tool {
            ResponseTool::Custom(c) => match &c.format {
                Some(CustomToolInputFormat::Grammar(g)) => {
                    assert_eq!(g.definition, "start: NUMBER");
                    let expected = if syntax == "lark" {
                        CustomToolGrammarSyntax::Lark
                    } else {
                        CustomToolGrammarSyntax::Regex
                    };
                    assert_eq!(g.syntax, expected);
                }
                other => panic!("expected Grammar format, got {other:?}"),
            },
            other => panic!("expected ResponseTool::Custom, got {other:?}"),
        }

        let serialized = serde_json::to_value(&tool).expect("custom tool should serialize");
        assert_eq!(serialized, payload);
    }
}

#[test]
fn test_namespace_tool_with_function_round_trip() {
    // Spec (openai-responses-api-spec.md §tools L475):
    // `Namespace { description, name, tools: array of Function | Custom,
    // type: "namespace" }`. Inner elements reuse the top-level Function
    // shape but are restricted by the spec to Function or Custom.
    let payload = json!({
        "type": "namespace",
        "name": "sql",
        "description": "Tools for interacting with the warehouse",
        "tools": [
            {
                "type": "function",
                "name": "select",
                "description": "Run a read-only SELECT",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                },
                "strict": true
            }
        ]
    });

    let tool: ResponseTool = serde_json::from_value(payload.clone())
        .expect("namespace tool with function element should deserialize");
    match &tool {
        ResponseTool::Namespace {
            description,
            name,
            tools,
        } => {
            assert_eq!(name, "sql");
            assert_eq!(description, "Tools for interacting with the warehouse");
            assert_eq!(tools.len(), 1);
            match &tools[0] {
                NamespaceTool::Function(ft) => {
                    assert_eq!(ft.function.name, "select");
                    assert_eq!(ft.function.strict, Some(true));
                }
                other @ NamespaceTool::Custom(_) => {
                    panic!("expected NamespaceTool::Function, got {other:?}")
                }
            }
        }
        other => panic!("expected ResponseTool::Namespace, got {other:?}"),
    }

    let serialized = serde_json::to_value(&tool).expect("namespace tool should serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_namespace_tool_with_custom_text_format_round_trip() {
    // Spec (openai-responses-api-spec.md §tools L471-475): namespace elements
    // may be Custom tools. `Custom { name, type: "custom", defer_loading?,
    // description?, format? }` with `format: Text { type: "text" }`.
    let payload = json!({
        "type": "namespace",
        "name": "shell",
        "description": "User-owned shell helpers",
        "tools": [
            {
                "type": "custom",
                "name": "raw_cmd",
                "description": "Emit a free-form shell command",
                "format": {"type": "text"}
            }
        ]
    });

    let tool: ResponseTool = serde_json::from_value(payload.clone())
        .expect("namespace tool with custom/text element should deserialize");
    match &tool {
        ResponseTool::Namespace { tools, .. } => {
            assert_eq!(tools.len(), 1);
            match &tools[0] {
                NamespaceTool::Custom(c) => {
                    assert_eq!(c.name, "raw_cmd");
                    assert!(matches!(c.format, Some(CustomToolInputFormat::Text)));
                }
                other @ NamespaceTool::Function(_) => {
                    panic!("expected NamespaceTool::Custom, got {other:?}")
                }
            }
        }
        other => panic!("expected ResponseTool::Namespace, got {other:?}"),
    }

    let serialized = serde_json::to_value(&tool).expect("namespace tool should serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_namespace_tool_with_custom_grammar_format_round_trip() {
    // Spec (openai-responses-api-spec.md §tools L472-475): namespace elements
    // may be Custom tools whose `format: Grammar { definition, syntax: "lark"
    // | "regex", type: "grammar" }` constrains free-form input at decode time.
    for syntax in ["lark", "regex"] {
        let payload = json!({
            "type": "namespace",
            "name": "parse",
            "description": "Grammar-constrained parsers",
            "tools": [
                {
                    "type": "custom",
                    "name": "expr_parser",
                    "format": {
                        "type": "grammar",
                        "definition": "start: NUMBER",
                        "syntax": syntax
                    }
                }
            ]
        });

        let tool: ResponseTool = serde_json::from_value(payload.clone())
            .expect("namespace tool with custom/grammar element should deserialize");
        match &tool {
            ResponseTool::Namespace { tools, .. } => match &tools[0] {
                NamespaceTool::Custom(c) => match &c.format {
                    Some(CustomToolInputFormat::Grammar(g)) => {
                        assert_eq!(g.definition, "start: NUMBER");
                        let expected = if syntax == "lark" {
                            CustomToolGrammarSyntax::Lark
                        } else {
                            CustomToolGrammarSyntax::Regex
                        };
                        assert_eq!(g.syntax, expected);
                    }
                    other => panic!("expected Grammar format, got {other:?}"),
                },
                other @ NamespaceTool::Function(_) => {
                    panic!("expected NamespaceTool::Custom, got {other:?}")
                }
            },
            other => panic!("expected ResponseTool::Namespace, got {other:?}"),
        }

        let serialized = serde_json::to_value(&tool).expect("namespace tool should serialize");
        assert_eq!(serialized, payload);
    }
}

#[test]
fn test_namespace_tool_mixed_elements_round_trip() {
    // Spec (openai-responses-api-spec.md §tools L475): a single namespace may
    // mix Function and Custom elements in any order.
    let payload = json!({
        "type": "namespace",
        "name": "fs",
        "description": "Filesystem helpers",
        "tools": [
            {
                "type": "function",
                "name": "read_file",
                "description": "Read a file by path",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                },
                "strict": true
            },
            {
                "type": "custom",
                "name": "write_file",
                "description": "Write a file by free-form payload"
            }
        ]
    });

    let tool: ResponseTool = serde_json::from_value(payload.clone())
        .expect("namespace tool with mixed elements should deserialize");
    match &tool {
        ResponseTool::Namespace { tools, .. } => {
            assert_eq!(tools.len(), 2);
            assert!(
                matches!(&tools[0], NamespaceTool::Function(ft) if ft.function.name == "read_file")
            );
            assert!(matches!(&tools[1], NamespaceTool::Custom(c) if c.name == "write_file"));
        }
        other => panic!("expected ResponseTool::Namespace, got {other:?}"),
    }

    let serialized = serde_json::to_value(&tool).expect("namespace tool should serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn computer_call_input_item_round_trips_spec_shape() {
    let payload = json!({
        "type": "computer_call",
        "id": "cc_1",
        "call_id": "call_1",
        "action": {"type": "screenshot"},
        "status": "completed",
        "pending_safety_checks": [],
    });
    let item: ResponseInputOutputItem =
        serde_json::from_value(payload.clone()).expect("deserialize");
    assert!(matches!(item, ResponseInputOutputItem::ComputerCall { .. }));
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn computer_call_output_input_item_round_trips_spec_shape() {
    let payload = json!({
        "type": "computer_call_output",
        "call_id": "call_1",
        "output": {"type": "computer_screenshot", "file_id": "file_1"},
    });
    let item: ResponseInputOutputItem =
        serde_json::from_value(payload.clone()).expect("deserialize");
    assert!(matches!(
        item,
        ResponseInputOutputItem::ComputerCallOutput { .. }
    ));
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn computer_call_output_item_round_trips_spec_shape() {
    let payload = json!({
        "type": "computer_call",
        "id": "cc_1",
        "call_id": "call_1",
        "action": {"type": "click", "button": "left", "x": 10, "y": 20},
        "status": "in_progress",
        "pending_safety_checks": [],
    });
    let item: ResponseOutputItem = serde_json::from_value(payload.clone()).expect("deserialize");
    assert!(matches!(item, ResponseOutputItem::ComputerCall { .. }));
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn computer_call_output_output_item_round_trips_spec_shape() {
    let payload = json!({
        "type": "computer_call_output",
        "call_id": "call_1",
        "output": {"type": "computer_screenshot", "image_url": "https://example/s.png"},
    });
    let item: ResponseOutputItem = serde_json::from_value(payload.clone()).expect("deserialize");
    assert!(matches!(
        item,
        ResponseOutputItem::ComputerCallOutput { .. }
    ));
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn computer_safety_check_accepts_optional_code_and_message() {
    // Per SDK v2.8.1 (types/responses/response_computer_tool_call.py::PendingSafetyCheck),
    // both `code` and `message` are `Optional[str] = None`. Payloads that omit
    // either field must deserialize, and the serialization round-trip must
    // preserve the missing fields (not invent empty strings).
    let payload = json!({
        "type": "computer_call",
        "id": "cc_1",
        "call_id": "call_1",
        "action": {"type": "screenshot"},
        "status": "in_progress",
        "pending_safety_checks": [
            {"id": "psc_only_id"},
            {"id": "psc_with_code", "code": "malicious_instructions"},
            {"id": "psc_with_message", "message": "details"},
            {"id": "psc_full", "code": "c", "message": "m"},
        ],
    });
    let item: ResponseOutputItem = serde_json::from_value(payload.clone()).expect("deserialize");
    match &item {
        ResponseOutputItem::ComputerCall {
            pending_safety_checks,
            ..
        } => {
            assert_eq!(pending_safety_checks.len(), 4);
            assert!(pending_safety_checks[0].code.is_none());
            assert!(pending_safety_checks[0].message.is_none());
            assert_eq!(
                pending_safety_checks[1].code.as_deref(),
                Some("malicious_instructions")
            );
            assert!(pending_safety_checks[1].message.is_none());
            assert!(pending_safety_checks[2].code.is_none());
            assert_eq!(pending_safety_checks[2].message.as_deref(), Some("details"));
            assert_eq!(pending_safety_checks[3].code.as_deref(), Some("c"));
            assert_eq!(pending_safety_checks[3].message.as_deref(), Some("m"));
        }
        other => panic!("expected ComputerCall, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn test_custom_tool_call_input_item_round_trip() {
    // Spec (openai-responses-api-spec.md L272-273):
    // `CustomToolCall object { call_id, input, name, type, id, namespace }`
    // `type: "custom_tool_call"`.
    let payload = json!({
        "type": "custom_tool_call",
        "call_id": "call_abc123",
        "input": "SELECT count(*) FROM events",
        "name": "run_sql",
        "id": "ctc_01",
        "namespace": "analytics"
    });

    let item: ResponseInputOutputItem =
        serde_json::from_value(payload.clone()).expect("custom_tool_call should deserialize");
    match &item {
        ResponseInputOutputItem::CustomToolCall {
            call_id,
            input,
            name,
            id,
            namespace,
        } => {
            assert_eq!(call_id, "call_abc123");
            assert_eq!(input, "SELECT count(*) FROM events");
            assert_eq!(name, "run_sql");
            assert_eq!(id.as_deref(), Some("ctc_01"));
            assert_eq!(namespace.as_deref(), Some("analytics"));
        }
        other => panic!("expected CustomToolCall, got {other:?}"),
    }

    let serialized = serde_json::to_value(&item).expect("serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_custom_tool_call_output_string_round_trip() {
    // Spec (openai-responses-api-spec.md L268-270):
    // `CustomToolCallOutput object { call_id, output, type, id }` where
    // `output: string or array of ResponseInputText | ResponseInputImage |
    // ResponseInputFile`. No `status` field per spec.
    let payload = json!({
        "type": "custom_tool_call_output",
        "call_id": "call_abc123",
        "output": "42",
        "id": "ctco_01"
    });

    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("custom_tool_call_output should deserialize");
    match &item {
        ResponseInputOutputItem::CustomToolCallOutput {
            call_id,
            output,
            id,
        } => {
            assert_eq!(call_id, "call_abc123");
            assert_eq!(id.as_deref(), Some("ctco_01"));
            match output {
                CustomToolCallOutputContent::Text(s) => assert_eq!(s, "42"),
                CustomToolCallOutputContent::Parts(_) => panic!("expected Text output"),
            }
        }
        other => panic!("expected CustomToolCallOutput, got {other:?}"),
    }

    let serialized = serde_json::to_value(&item).expect("serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_custom_tool_call_output_array_round_trip() {
    // Spec (openai-responses-api-spec.md L269): array of
    // ResponseInputText | ResponseInputImage | ResponseInputFile.
    let payload = json!({
        "type": "custom_tool_call_output",
        "call_id": "call_abc123",
        "output": [
            {"type": "input_text", "text": "rows returned:"},
            {"type": "input_file", "filename": "result.csv", "file_id": "file_xyz"}
        ]
    });

    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("array-form custom_tool_call_output should deserialize");
    match &item {
        ResponseInputOutputItem::CustomToolCallOutput { output, .. } => match output {
            CustomToolCallOutputContent::Parts(parts) => {
                assert_eq!(parts.len(), 2);
                assert!(matches!(
                    parts[0],
                    CustomToolInputContentPart::InputText { .. }
                ));
                assert!(matches!(
                    parts[1],
                    CustomToolInputContentPart::InputFile { .. }
                ));
            }
            CustomToolCallOutputContent::Text(_) => panic!("expected Parts output"),
        },
        other => panic!("expected CustomToolCallOutput, got {other:?}"),
    }

    let serialized = serde_json::to_value(&item).expect("serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_custom_tool_call_output_parts_reject_output_text() {
    // Spec (openai-responses-api-spec.md L269): the `output` array only
    // accepts input-typed parts. Assistant-facing shapes such as
    // `output_text` and `refusal` must be rejected at the type boundary
    // rather than silently coerced.
    let payload_output_text = json!({
        "type": "custom_tool_call_output",
        "call_id": "call_abc123",
        "output": [
            {"type": "output_text", "text": "not allowed here", "annotations": []}
        ]
    });
    let err = serde_json::from_value::<ResponseInputOutputItem>(payload_output_text)
        .expect_err("output_text must not be accepted inside custom_tool_call_output parts");
    let msg = err.to_string();
    assert!(
        msg.contains("output_text") || msg.contains("did not match any variant"),
        "expected untagged-variant rejection carrying `output_text`, got: {msg}"
    );

    let payload_refusal = json!({
        "type": "custom_tool_call_output",
        "call_id": "call_abc123",
        "output": [
            {"type": "refusal", "refusal": "nope"}
        ]
    });
    assert!(
        serde_json::from_value::<ResponseInputOutputItem>(payload_refusal).is_err(),
        "refusal must not be accepted inside custom_tool_call_output parts"
    );

    // Sanity: an input-typed part still deserializes cleanly.
    let payload_ok = json!({
        "type": "custom_tool_call_output",
        "call_id": "call_abc123",
        "output": [
            {"type": "input_text", "text": "ok"}
        ]
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload_ok)
        .expect("input_text part must still deserialize after the tightening");
    match &item {
        ResponseInputOutputItem::CustomToolCallOutput { output, .. } => match output {
            CustomToolCallOutputContent::Parts(parts) => {
                assert_eq!(parts.len(), 1);
                assert!(matches!(
                    parts[0],
                    CustomToolInputContentPart::InputText { .. }
                ));
            }
            CustomToolCallOutputContent::Text(_) => panic!("expected Parts output"),
        },
        other => panic!("expected CustomToolCallOutput, got {other:?}"),
    }
}

#[test]
fn test_custom_tool_call_output_validation_rejects_empty_text_and_parts() {
    use validator::Validate;

    // cross-parameter validation (L2145-2160) requires a Message/SimpleInputMessage
    // alongside any tool item, so every payload below pairs the custom_tool_call_output
    // with a plain user message to isolate the CustomToolCallOutput branch we want
    // to cover.
    let user_msg = json!({"role": "user", "content": "hi"});

    // Empty string output — validate_input_item's
    // `CustomToolCallOutputContent::Text(s) if s.is_empty()` branch.
    let empty_text: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": [
            user_msg,
            {
                "type": "custom_tool_call_output",
                "call_id": "call_abc123",
                "output": ""
            }
        ]
    }))
    .expect("request should deserialize");
    assert!(
        empty_text.validate().is_err(),
        "empty CustomToolCallOutput text must be rejected"
    );

    // Empty parts array output — validate_input_item's
    // `CustomToolCallOutputContent::Parts(parts) if parts.is_empty()` branch.
    let empty_parts: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": [
            user_msg,
            {
                "type": "custom_tool_call_output",
                "call_id": "call_abc123",
                "output": []
            }
        ]
    }))
    .expect("request should deserialize");
    assert!(
        empty_parts.validate().is_err(),
        "empty CustomToolCallOutput parts array must be rejected"
    );

    // Sanity: a non-empty string output passes the same validator.
    let ok_text: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": [
            user_msg,
            {
                "type": "custom_tool_call_output",
                "call_id": "call_abc123",
                "output": "42"
            }
        ]
    }))
    .expect("request should deserialize");
    assert!(
        ok_text.validate().is_ok(),
        "non-empty CustomToolCallOutput must validate",
    );
}
