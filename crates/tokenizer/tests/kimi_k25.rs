//! Essential coverage for the Kimi-K2.5 tool renderer:
//!   - encoder shapes a nested-object schema with `?` for optional fields
//!   - encoder emits a TS union for `enum` schemas
//!   - end-to-end: a tokenizer loaded with `KimiK25ForConditionalGeneration`
//!     dispatches tools through the TS-namespace encoder (not the JSON fallback)

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::fs;

use llm_tokenizer::{
    chat_template::ChatTemplateParams, encoders::kimi_k25_tools::encode_tools_to_typescript,
    traits::Tokenizer as TokenizerTrait, TiktokenTokenizer,
};
use serde_json::{json, Value};
use tempfile::TempDir;

const MIN_TIKTOKEN_MODEL: &str = "aGVsbG8= 0\n";

#[test]
fn encoder_renders_nested_object_with_required() {
    let tools: Vec<Value> = serde_json::from_str(
        r#"[{
            "type": "function",
            "function": {
                "name": "create_user",
                "description": "Create a new user record.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer"}
                            },
                            "required": ["name"]
                        }
                    },
                    "required": ["user"]
                }
            }
        }]"#,
    )
    .unwrap();

    let expected = "# Tools\n\n## functions\nnamespace functions {\n// Create a new user record.\ntype create_user = (_: {\n  user: {\n    name: string,\n    age?: number\n  }\n}) => any;\n}\n";
    assert_eq!(
        encode_tools_to_typescript(&tools).as_deref(),
        Some(expected)
    );
}

#[test]
fn encoder_renders_enum_as_union() {
    let tools: Vec<Value> = serde_json::from_str(
        r#"[{
            "type": "function",
            "function": {
                "name": "set_status",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {"enum": ["active", "paused", "done"]}
                    },
                    "required": ["status"]
                }
            }
        }]"#,
    )
    .unwrap();

    let expected = "# Tools\n\n## functions\nnamespace functions {\ntype set_status = (_: {\n  status: \"active\" | \"paused\" | \"done\"\n}) => any;\n}\n";
    assert_eq!(
        encode_tools_to_typescript(&tools).as_deref(),
        Some(expected)
    );
}

#[test]
fn chat_template_renders_typescript_namespace() {
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
    fs::write(dir.path().join("tokenizer_config.json"), "{}").unwrap();

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
        rendered.contains("namespace functions") && rendered.contains("type calc = (_:"),
        "expected TS namespace block, got:\n{rendered}"
    );
    assert!(
        !rendered.contains(r#"[{"function":"#),
        "rendered prompt fell into JSON fallback path:\n{rendered}"
    );
}
