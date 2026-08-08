//! Verify that `HuggingFaceTokenizer` selects the right chat-template renderer
//! based on `config.json::architectures`.
#[cfg(test)]
mod tests {
    use std::{collections::HashMap, fs};

    use llm_tokenizer::{
        chat_template::{ChatTemplateParams, ThinkingKeyName, ThinkingToggle},
        huggingface::HuggingFaceTokenizer,
        TokenizerTrait,
    };
    use serde_json::json;
    use tempfile::TempDir;
    /// A minimal tokenizer.json that loads cleanly. The only requirement is that
    /// it parses; the encoder logic does not call back into the tokenizer here.
    const MIN_TOKENIZER_JSON: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "BPE",
            "vocab": { "hello": 0, "<s>": 1, "</s>": 2 },
            "merges": []
        }
    }"#;
    fn write_dir(architectures: Option<&[&str]>) -> (TempDir, String) {
        let temp = TempDir::new().unwrap();
        let tok_path = temp.path().join("tokenizer.json");
        fs::write(&tok_path, MIN_TOKENIZER_JSON).unwrap();
        if let Some(archs) = architectures {
            let cfg_path = temp.path().join("config.json");
            let body = json!({ "architectures": archs }).to_string();
            fs::write(&cfg_path, body).unwrap();
        }
        let p = tok_path.to_str().unwrap().to_string();
        (temp, p)
    }

    fn write_named_v4_dir(model_name: &str) -> (TempDir, String) {
        let temp = TempDir::new().unwrap();
        let model_dir = temp.path().join(model_name);
        fs::create_dir(&model_dir).unwrap();
        let tok_path = model_dir.join("tokenizer.json");
        fs::write(&tok_path, MIN_TOKENIZER_JSON).unwrap();
        fs::write(
            model_dir.join("config.json"),
            json!({ "architectures": ["DeepseekV4ForCausalLM"] }).to_string(),
        )
        .unwrap();
        let path = tok_path.to_str().unwrap().to_string();
        (temp, path)
    }

    #[test]
    fn config_with_deepseek_v32_arch_uses_v32_renderer() {
        let (_tmp, tok) = write_dir(Some(&["DeepseekV32ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let kwargs: HashMap<String, serde_json::Value> = HashMap::new();
        let params = ChatTemplateParams {
            template_kwargs: Some(&kwargs),
            ..Default::default()
        };
        let out = tokenizer.apply_chat_template(&messages, params).unwrap();
        // V3.2 emits BOS + <｜User｜>Hello<｜Assistant｜></think> in chat mode.
        assert!(out.contains("<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>"));
        assert!(out.contains("<\u{FF5C}User\u{FF5C}>Hello<\u{FF5C}Assistant\u{FF5C}>"));
        assert!(out.ends_with("</think>"));
    }
    #[test]
    fn config_with_deepseek_v4_arch_uses_v4_renderer() {
        let (_tmp, tok) = write_dir(Some(&["DeepseekV4ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let kwargs: HashMap<String, serde_json::Value> = HashMap::new();
        let params = ChatTemplateParams {
            template_kwargs: Some(&kwargs),
            ..Default::default()
        };
        let out = tokenizer.apply_chat_template(&messages, params).unwrap();
        // V4 emits BOS + <｜User｜>Hello<｜Assistant｜></think> in chat mode.
        assert!(out.contains("<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>"));
        assert!(out.contains("<\u{FF5C}User\u{FF5C}>Hello<\u{FF5C}Assistant\u{FF5C}>"));
        assert!(out.ends_with("</think>"));
    }

    #[test]
    fn config_with_unrelated_arch_falls_back_to_jinja() {
        // A non-DeepSeek architecture should keep using the Jinja renderer; with
        // no chat_template set, applying the template should error rather than
        // silently picking a DeepSeek encoder.
        let (_tmp, tok) = write_dir(Some(&["LlamaForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let result = tokenizer.apply_chat_template(&messages, ChatTemplateParams::default());
        assert!(
            result.is_err(),
            "expected error from missing Jinja template"
        );
    }
    #[test]
    fn no_config_json_falls_back_to_jinja() {
        // No sibling config.json — must still default to Jinja and not blow up.
        let (_tmp, tok) = write_dir(None);
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let result = tokenizer.apply_chat_template(&messages, ChatTemplateParams::default());
        // Without a chat template registered, the Jinja renderer surfaces an error.
        // The important thing is that we did NOT auto-select a DeepSeek encoder.
        assert!(result.is_err());
    }
    #[test]
    fn malformed_config_json_falls_back_to_jinja() {
        let temp = TempDir::new().unwrap();
        let tok_path = temp.path().join("tokenizer.json");
        fs::write(&tok_path, MIN_TOKENIZER_JSON).unwrap();
        fs::write(temp.path().join("config.json"), "{ this is not json").unwrap();
        let tokenizer = HuggingFaceTokenizer::from_file(tok_path.to_str().unwrap()).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let result = tokenizer.apply_chat_template(&messages, ChatTemplateParams::default());
        assert!(result.is_err());
    }
    #[test]
    fn deepseek_v4_injects_tools_into_system_message() {
        // Client passes tools at the request level (params.tools), not embedded
        // in messages. The shim must attach them to a system message so the
        // encoder renders the tools block.
        let (_tmp, tok) = write_dir(Some(&["DeepseekV4ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }
            }
        })];
        let params = ChatTemplateParams {
            tools: Some(&tools),
            ..Default::default()
        };
        let out = tokenizer.apply_chat_template(&messages, params).unwrap();
        assert!(out.contains("## Tools"), "tools block missing: {out}");
        assert!(
            out.contains("get_weather"),
            "tool name missing from prompt: {out}"
        );
        assert!(
            out.contains("<\u{FF5C}DSML\u{FF5C}tool_calls>"),
            "V4 DSML invocation grammar missing: {out}"
        );
    }

    #[test]
    fn deepseek_v32_injects_tools_into_system_message() {
        let (_tmp, tok) = write_dir(Some(&["DeepseekV32ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hi" })];
        let tools = vec![json!({
            "type": "function",
            "function": { "name": "ping", "description": "ping", "parameters": {} }
        })];
        let params = ChatTemplateParams {
            tools: Some(&tools),
            ..Default::default()
        };
        let out = tokenizer.apply_chat_template(&messages, params).unwrap();
        assert!(out.contains("## Tools"), "tools block missing: {out}");
        assert!(out.contains("ping"), "tool name missing: {out}");
        // V3.2 uses function_calls (vs V4's tool_calls).
        assert!(
            out.contains("<\u{FF5C}DSML\u{FF5C}function_calls>"),
            "V3.2 DSML invocation grammar missing: {out}"
        );
    }

    #[test]
    fn deepseek_v4_attaches_tools_to_existing_system_message() {
        // When a system message is already present, tools should attach to it
        // rather than inserting a second system block.
        let (_tmp, tok) = write_dir(Some(&["DeepseekV4ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![
            json!({ "role": "system", "content": "Be concise." }),
            json!({ "role": "user", "content": "Hi" }),
        ];
        let tools = vec![json!({
            "type": "function",
            "function": { "name": "ping", "description": "ping", "parameters": {} }
        })];
        let params = ChatTemplateParams {
            tools: Some(&tools),
            ..Default::default()
        };
        let out = tokenizer.apply_chat_template(&messages, params).unwrap();
        assert!(
            out.contains("Be concise."),
            "existing system content lost: {out}"
        );
        assert!(out.contains("ping"), "tool not attached: {out}");
    }

    #[test]
    fn deepseek_renderers_report_thinking_introspection() {
        // V3.2 / V4 inject `<think>` in the prefill when thinking is on, and
        // gate thinking on the `thinking` kwarg. The trait methods must
        // surface this so the gateway can call `mark_reasoning_started` on
        // the conditional reasoning parser (deepseek_v31 etc).
        for arch in &["DeepseekV32ForCausalLM", "DeepseekV4ForCausalLM"] {
            let (_tmp, tok) = write_dir(Some(&[*arch]));
            let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
            assert_eq!(
                tokenizer.thinking_toggle(),
                ThinkingToggle::DefaultOff,
                "{arch}: expected DefaultOff toggle"
            );
            assert_eq!(
                tokenizer.thinking_key_name(),
                Some(ThinkingKeyName::Thinking),
                "{arch}: expected Thinking key name"
            );
            assert!(
                tokenizer.think_in_prefill(),
                "{arch}: expected think_in_prefill=true"
            );
        }
    }

    #[test]
    fn deepseek_renderers_honor_thinking_kwarg_only() {
        // `thinking: true` → prompt ends with <think> (thinking mode).
        // `enable_thinking: true` alone → ignored (chat mode), matching
        // `thinking_key_name() == Some(Thinking)` and sglang's DeepSeek path.
        for arch in &["DeepseekV32ForCausalLM", "DeepseekV4ForCausalLM"] {
            let (_tmp, tok) = write_dir(Some(&[*arch]));
            let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
            let messages = vec![json!({ "role": "user", "content": "Hi" })];

            let mut thinking_kwargs: HashMap<String, serde_json::Value> = HashMap::new();
            thinking_kwargs.insert("thinking".to_string(), serde_json::Value::Bool(true));
            let out_thinking = tokenizer
                .apply_chat_template(
                    &messages,
                    ChatTemplateParams {
                        template_kwargs: Some(&thinking_kwargs),
                        ..Default::default()
                    },
                )
                .unwrap();
            assert!(
                out_thinking.ends_with("<think>"),
                "{arch}: thinking=true should enter thinking mode: {out_thinking}"
            );

            let mut enable_thinking_kwargs: HashMap<String, serde_json::Value> = HashMap::new();
            enable_thinking_kwargs
                .insert("enable_thinking".to_string(), serde_json::Value::Bool(true));
            let out_enable = tokenizer
                .apply_chat_template(
                    &messages,
                    ChatTemplateParams {
                        template_kwargs: Some(&enable_thinking_kwargs),
                        ..Default::default()
                    },
                )
                .unwrap();
            assert!(
                out_enable.ends_with("</think>"),
                "{arch}: enable_thinking alone must NOT enter thinking mode: {out_enable}"
            );
        }
    }

    #[test]
    fn deepseek_v4_renderer_passes_reasoning_effort() {
        let (_tmp, tok) = write_named_v4_dir("DeepSeek-V4-Flash-0731");
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let mut kwargs: HashMap<String, serde_json::Value> = HashMap::new();
        kwargs.insert(
            "reasoning_effort".to_string(),
            serde_json::Value::String("max".to_string()),
        );
        kwargs.insert("thinking".to_string(), serde_json::Value::Bool(true));
        let params = ChatTemplateParams {
            template_kwargs: Some(&kwargs),
            ..Default::default()
        };
        let out = tokenizer.apply_chat_template(&messages, params).unwrap();
        assert!(
            out.contains("Reasoning Effort: Beyond maximum"),
            "expected 0731 max reasoning-effort prefix in V4 output"
        );
        assert!(
            out.ends_with("<think>"),
            "thinking mode should leave a <think> token open"
        );
    }

    #[test]
    fn deepseek_v4_maps_openai_efforts_to_0731_buckets() {
        let (_tmp, tok) = write_named_v4_dir("DeepSeek-V4-Flash-0731");
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];

        for (effort, expected_prefix) in [
            ("minimal", None),
            ("low", None),
            ("medium", Some("Reasoning Effort: Absolute maximum")),
            ("high", Some("Reasoning Effort: Absolute maximum")),
            ("xhigh", Some("Reasoning Effort: Beyond maximum")),
            ("max", Some("Reasoning Effort: Beyond maximum")),
        ] {
            let out = tokenizer
                .apply_chat_template(
                    &messages,
                    ChatTemplateParams {
                        thinking: Some(true),
                        reasoning_effort: Some(effort),
                        ..Default::default()
                    },
                )
                .unwrap();

            match expected_prefix {
                Some(prefix) => assert!(out.contains(prefix), "effort {effort}: {out}"),
                None => assert!(!out.contains("Reasoning Effort:"), "effort {effort}: {out}"),
            }
            assert!(out.ends_with("<think>"), "effort {effort}: {out}");
        }
    }

    #[test]
    fn deepseek_v4_top_level_and_template_efforts_render_identically() {
        let (_tmp, tok) = write_named_v4_dir("DeepSeek-V4-Flash-0731");
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];

        for (effort, expected_prefix) in [
            ("low", None),
            ("high", Some("Reasoning Effort: Absolute maximum")),
            ("max", Some("Reasoning Effort: Beyond maximum")),
        ] {
            let top_level = tokenizer
                .apply_chat_template(
                    &messages,
                    ChatTemplateParams {
                        thinking: Some(true),
                        reasoning_effort: Some(effort),
                        ..Default::default()
                    },
                )
                .unwrap();
            let template_kwargs = HashMap::from([("reasoning_effort".to_string(), json!(effort))]);
            let native = tokenizer
                .apply_chat_template(
                    &messages,
                    ChatTemplateParams {
                        template_kwargs: Some(&template_kwargs),
                        ..Default::default()
                    },
                )
                .unwrap();

            assert_eq!(native, top_level, "effort {effort}");
            assert!(native.ends_with("<think>"), "effort {effort}: {native}");
            match expected_prefix {
                Some(prefix) => assert!(native.contains(prefix), "effort {effort}: {native}"),
                None => assert!(
                    !native.contains("Reasoning Effort:"),
                    "effort {effort}: {native}"
                ),
            }
        }
    }

    #[test]
    fn deepseek_v4_distinguishes_merged_public_effort_from_native_template_effort() {
        let (_tmp, tok) = write_named_v4_dir("DeepSeek-V4-Flash");
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let kwargs = HashMap::from([("reasoning_effort".to_string(), json!("medium"))]);

        let output = tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    thinking: Some(true),
                    reasoning_effort: Some("medium"),
                    template_kwargs: Some(&kwargs),
                    ..Default::default()
                },
            )
            .unwrap();

        assert!(output.contains("Reasoning Effort: Absolute maximum"));
        assert!(!output.contains("Reasoning Effort: Beyond maximum"));
    }

    #[test]
    fn deepseek_v4_native_template_effort_overrides_openai_effort() {
        let (_tmp, tok) = write_named_v4_dir("DeepSeek-V4-Flash-0731");
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let native_effort = json!("max");

        let out = tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    thinking: Some(true),
                    reasoning_effort: Some("medium"),
                    template_reasoning_effort: Some(&native_effort),
                    ..Default::default()
                },
            )
            .unwrap();

        assert!(out.contains("Reasoning Effort: Beyond maximum"));
        assert!(!out.contains("Reasoning Effort: Absolute maximum"));
    }

    #[test]
    fn deepseek_v4_rejects_invalid_native_template_effort() {
        let messages = vec![json!({ "role": "user", "content": "Hello" })];

        for model_name in [
            "DeepSeek-V4-Flash",
            "DeepSeek-V4-Flash-0731",
            "DeepSeek-V4-Flash-DSpark",
            "DeepSeek-V4-Pro",
        ] {
            let (_tmp, tok) = write_named_v4_dir(model_name);
            let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
            for native_effort in [json!("medium"), json!(3)] {
                let error = tokenizer
                    .apply_chat_template(
                        &messages,
                        ChatTemplateParams {
                            thinking: Some(true),
                            template_reasoning_effort: Some(&native_effort),
                            ..Default::default()
                        },
                    )
                    .unwrap_err();
                assert!(
                    error.to_string().contains(
                        "chat_template_kwargs.reasoning_effort must be one of low, high, max"
                    ),
                    "{model_name}: unexpected error: {error}"
                );
            }
        }

        let (_tmp, tok) = write_dir(Some(&["DeepseekV4ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        for native_effort in [json!("medium"), json!(3)] {
            let error = tokenizer
                .apply_chat_template(
                    &messages,
                    ChatTemplateParams {
                        thinking: Some(true),
                        template_reasoning_effort: Some(&native_effort),
                        ..Default::default()
                    },
                )
                .unwrap_err();
            assert!(
                error.to_string().contains(
                    "chat_template_kwargs.reasoning_effort must be one of low, high, max"
                ),
                "anonymous V4: unexpected error: {error}"
            );
        }
    }

    #[test]
    fn deepseek_v4_explicit_thinking_overrides_effort_defaults() {
        let (_tmp, tok) = write_named_v4_dir("DeepSeek-V4-Flash-0731");
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];

        let mut effort_only = HashMap::new();
        effort_only.insert("reasoning_effort".to_string(), json!("max"));
        let out = tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    template_kwargs: Some(&effort_only),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(out.ends_with("<think>"));
        assert!(out.contains("Reasoning Effort: Beyond maximum"));

        let thinking_off = HashMap::from([("thinking".to_string(), json!(false))]);
        let out = tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    template_kwargs: Some(&thinking_off),
                    thinking: Some(true),
                    reasoning_effort: Some("max"),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(out.ends_with("</think>"));
        assert!(!out.contains("Reasoning Effort:"));

        let thinking_on = HashMap::from([("thinking".to_string(), json!(true))]);
        let out = tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    template_kwargs: Some(&thinking_on),
                    thinking: Some(false),
                    reasoning_effort: Some("none"),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(out.ends_with("<think>"));
        assert!(!out.contains("Reasoning Effort:"));

        let out = tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    reasoning_effort: Some("turbo"),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(out.ends_with("</think>"));
        assert!(!out.contains("Reasoning Effort:"));
    }

    #[test]
    fn deepseek_v4_effort_encoding_is_independent_of_local_model_path() {
        for model_name in [
            "DeepSeek-V4-Flash-0731",
            "DeepSeek-V4-Flash",
            "DeepSeek-V4-Flash-DSpark",
            "DeepSeek-V4-Pro",
        ] {
            let (_tmp, tok) = write_named_v4_dir(model_name);
            let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
            let messages = vec![json!({ "role": "user", "content": "Hello" })];
            let native_effort = json!("max");
            let output = tokenizer
                .apply_chat_template(
                    &messages,
                    ChatTemplateParams {
                        thinking: Some(true),
                        template_reasoning_effort: Some(&native_effort),
                        ..Default::default()
                    },
                )
                .unwrap();

            assert!(
                output.contains("Reasoning Effort: Beyond maximum"),
                "{model_name}: {output}"
            );
            assert!(
                !output.contains("Reasoning Effort: Absolute maximum"),
                "{model_name}: {output}"
            );
        }
    }

    #[test]
    fn deepseek_v4_anonymous_renderer_uses_0731_effort_encoding() {
        let (_tmp, tok) = write_dir(Some(&["DeepseekV4ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let native_effort = json!("max");
        let output = tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    thinking: Some(true),
                    template_reasoning_effort: Some(&native_effort),
                    ..Default::default()
                },
            )
            .unwrap();

        assert!(output.contains("Reasoning Effort: Beyond maximum"));
        assert!(!output.contains("Reasoning Effort: Absolute maximum"));
    }
}
