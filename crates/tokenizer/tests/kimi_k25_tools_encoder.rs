//! Golden-fixture tests for the Kimi-K2.5 TS-namespace encoder. Each pair of
//! files in `fixtures/kimi_k25/{schemas,expected}/` is asserted byte-equal
//! against the encoder output. Failure prints a unified diff for triage.

use std::{fs, path::PathBuf};

use llm_tokenizer::encoders::kimi_k25_tools::encode_tools_to_typescript;
use serde_json::Value;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("kimi_k25")
}

#[expect(clippy::panic, reason = "test helper — panics are intentional")]
#[expect(clippy::expect_used, reason = "test helper — panics are intentional")]
fn run_fixture(name: &str) {
    let schema_path = fixtures_dir().join("schemas").join(format!("{name}.json"));
    let expected_path = fixtures_dir().join("expected").join(format!("{name}.txt"));
    let schema_str = fs::read_to_string(&schema_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", schema_path.display()));
    let expected = fs::read_to_string(&expected_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", expected_path.display()));
    let tools: Vec<Value> = serde_json::from_str(&schema_str).expect("parse schema json");
    let actual = encode_tools_to_typescript(&tools).unwrap_or_default();
    assert!(
        actual == expected,
        "fixture mismatch for {name}\n--- expected ---\n{expected}\n--- actual ---\n{actual}\n"
    );
}

#[test]
fn fixture_01_simple_string() {
    run_fixture("01_simple_string");
}
