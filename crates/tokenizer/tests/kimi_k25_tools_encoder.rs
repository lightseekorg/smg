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

#[test]
fn fixture_02_optional_number() {
    run_fixture("02_optional_number");
}

#[test]
fn fixture_03_nested_object() {
    run_fixture("03_nested_object");
}

#[test]
fn fixture_04_array_of_strings() {
    run_fixture("04_array_of_strings");
}

#[test]
fn fixture_05_two_tools() {
    run_fixture("05_two_tools");
}

#[test]
fn fixture_06_enum() {
    run_fixture("06_enum");
}

#[test]
fn fixture_07_anyof() {
    run_fixture("07_anyof");
}

#[test]
fn fixture_08_type_list() {
    run_fixture("08_type_list");
}

#[test]
fn fixture_09_ref_local() {
    run_fixture("09_ref_local");
}

#[test]
fn fixture_11_multiline_description() {
    run_fixture("11_multiline_description");
}

#[test]
fn fixture_12_unicode() {
    run_fixture("12_unicode");
}

#[test]
fn fixture_13_empty_params() {
    run_fixture("13_empty_params");
}

#[test]
fn fixture_14_unknown_type() {
    run_fixture("14_unknown_type");
}
