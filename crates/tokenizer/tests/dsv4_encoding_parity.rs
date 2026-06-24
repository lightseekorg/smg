//! Byte-for-byte parity between SMG's DeepSeek-V4 prompt encoder and the
//! reference `encoding_dsv4.py` that ships with the model (and that vLLM serves
//! from). Fixtures live in `tests/fixtures/dsv4/` — see the README there.
//!
//! A mismatch means the SMG serving path renders a different prompt than vLLM,
//! shifting the model off its training distribution (this caught a JSON
//! separator divergence in embedded tool schemas).
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::{fs, path::PathBuf};

use llm_tokenizer::encoders::deepseek_v4::{encode_messages, EncodeParams, ThinkingMode};
use serde_json::Value;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv4")
}

fn first_diff(got: &str, gold: &str) -> String {
    let (g, r): (Vec<char>, Vec<char>) = (got.chars().collect(), gold.chars().collect());
    let n = g.len().min(r.len());
    for i in 0..n {
        if g[i] != r[i] {
            let s = i.saturating_sub(50);
            return format!(
                "first diff at char {i}\n   GOT ...{}\n  GOLD ...{}",
                g[s..(i + 40).min(g.len())].iter().collect::<String>(),
                r[s..(i + 40).min(r.len())].iter().collect::<String>(),
            );
        }
    }
    format!(
        "identical prefix; length differs GOT={} GOLD={}",
        g.len(),
        r.len()
    )
}

fn assert_parity(name: &str, messages: &[Value], mode: ThinkingMode) {
    let dir = fixtures_dir();
    let gold = fs::read_to_string(dir.join(format!("test_output_{name}.txt"))).unwrap();
    let got = encode_messages(messages, mode, &EncodeParams::default())
        .unwrap_or_else(|e| panic!("case {name} encode failed: {e}"));
    assert!(
        got == gold,
        "case {name} mismatch\n{}",
        first_diff(&got, &gold)
    );
}

fn load(name: &str) -> Value {
    let raw = fs::read_to_string(fixtures_dir().join(format!("test_input_{name}.json"))).unwrap();
    serde_json::from_str(&raw).unwrap()
}

/// Thinking mode, multi-turn tool calling with tool results merged into user.
#[test]
fn case_1_thinking_tools_multiturn() {
    let td = load("1");
    let mut messages = td["messages"].as_array().unwrap().clone();
    // The reference harness attaches the tool list to the system message.
    if let Some(obj) = messages[0].as_object_mut() {
        obj.insert("tools".into(), td["tools"].clone());
    }
    assert_parity("1", &messages, ThinkingMode::Thinking);
}

/// Thinking mode, no tools — earlier-turn reasoning is dropped.
#[test]
fn case_2_thinking_no_tools() {
    let messages = load("2");
    assert_parity("2", messages.as_array().unwrap(), ThinkingMode::Thinking);
}

/// Interleaved thinking + search: developer message with tools, latest_reminder.
#[test]
fn case_3_interleaved_search() {
    let messages = load("3");
    assert_parity("3", messages.as_array().unwrap(), ThinkingMode::Thinking);
}

/// Chat mode quick-instruction task with latest_reminder.
#[test]
fn case_4_quick_instruction() {
    let messages = load("4");
    assert_parity("4", messages.as_array().unwrap(), ThinkingMode::Chat);
}
