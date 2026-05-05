//! End-to-end render test: load a TiktokenTokenizer with the real Kimi-K2.5
//! chat_template.jinja and a config.json declaring KimiK25ForConditionalGeneration,
//! send a chat with tools, assert the rendered prompt contains the TS namespace
//! block (not the JSON fallback).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

#[cfg(test)]
mod tests {
    use std::fs;

    use llm_tokenizer::{
        chat_template::ChatTemplateParams, traits::Tokenizer as TokenizerTrait, TiktokenTokenizer,
    };
    use serde_json::json;
    use tempfile::TempDir;

    const MIN_TIKTOKEN_MODEL: &str = "aGVsbG8= 0\n";

    fn build_dir() -> TempDir {
        let dir = TempDir::new().unwrap();
        let template = fs::read_to_string(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/kimi_k25/chat_template.jinja"),
        )
        .expect("vendored chat_template.jinja must exist");
        fs::write(dir.path().join("tiktoken.model"), MIN_TIKTOKEN_MODEL).unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"architectures": ["KimiK25ForConditionalGeneration"]}"#,
        )
        .unwrap();
        fs::write(dir.path().join("chat_template.jinja"), template).unwrap();
        // tokenizer_config without an inline template — loader will discover chat_template.jinja.
        fs::write(dir.path().join("tokenizer_config.json"), "{}").unwrap();
        dir
    }

    #[test]
    fn renders_tools_as_typescript_namespace() {
        let dir = build_dir();
        let tok = TiktokenTokenizer::from_dir(dir.path()).expect("tokenizer should load");

        let messages = vec![json!({"role": "user", "content": "what's 2+2?"})];
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "calc",
                "description": "Compute an expression.",
                "parameters": {
                    "type": "object",
                    "properties": {"expr": {"type": "string"}},
                    "required": ["expr"]
                }
            }
        })];
        let rendered = tok
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    add_generation_prompt: true,
                    tools: Some(&tools),
                    ..Default::default()
                },
            )
            .expect("render should succeed");

        assert!(
            rendered.contains("namespace functions"),
            "expected TS namespace block in rendered prompt, got:\n{rendered}"
        );
        assert!(
            rendered.contains("type calc = (_:"),
            "expected TS function declaration, got:\n{rendered}"
        );
        assert!(
            !rendered.contains(r#"[{"function":"#),
            "rendered prompt fell into JSON fallback path:\n{rendered}"
        );
    }

    #[test]
    fn renders_without_tools_block_when_tools_empty() {
        let dir = build_dir();
        let tok = TiktokenTokenizer::from_dir(dir.path()).expect("tokenizer should load");
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let rendered = tok
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    add_generation_prompt: true,
                    ..Default::default()
                },
            )
            .expect("render should succeed");

        assert!(
            !rendered.contains("namespace functions"),
            "unexpected tools block:\n{rendered}"
        );
        assert!(
            !rendered.contains("# Tools"),
            "unexpected tools block:\n{rendered}"
        );
    }
}
