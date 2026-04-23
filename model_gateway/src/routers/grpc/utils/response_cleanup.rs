//! Post-parsing response cleanup utilities.
//!
//! When tool/reasoning parsers are active, backends send `skip_special_tokens=false`
//! so parsers can see structural tokens. After parsing completes, residual special
//! tokens must be stripped before the response reaches the user.

use llm_tokenizer::traits::Tokenizer;

/// Common ChatML-family tokens that may leak when `skip_special_tokens=false`.
///
/// Used as fallback when `tokenizer.get_special_tokens().additional_special_tokens`
/// is empty — this happens for tiktoken-based tokenizers whose `tokenizer_config.json`
/// lacks an `additional_special_tokens` array (e.g. Kimi K2, non-Cl100kBase models).
const CHATML_FALLBACK_TOKENS: &[&str] = &[
    "<|im_end|>",
    "<|im_start|>",
    "<|im_user|>",
    "<|im_assistant|>",
    "<|im_system|>",
    "<|im_middle|>",
    "</think>",
];

/// Strip special tokens that leaked into text after tool/reasoning parsing.
///
/// Prefers `tokenizer.get_special_tokens().additional_special_tokens` (dynamic,
/// model-specific). Falls back to [`CHATML_FALLBACK_TOKENS`] when that list is
/// empty, so Kimi / tiktoken models still get cleaned up.
///
/// Returns `true` if any tokens were removed.
pub(crate) fn strip_leaked_special_tokens(text: &mut String, tokenizer: &dyn Tokenizer) -> bool {
    let special = tokenizer.get_special_tokens();

    let mut changed = false;
    if special.additional_special_tokens.is_empty() {
        for token in CHATML_FALLBACK_TOKENS {
            if text.contains(token) {
                *text = text.replace(token, "");
                changed = true;
            }
        }
    } else {
        for token in &special.additional_special_tokens {
            if text.contains(token.as_str()) {
                *text = text.replace(token.as_str(), "");
                changed = true;
            }
        }
    }
    changed
}

/// Streaming variant: strip leaked special tokens from a delta string,
/// returning the cleaned string.
pub(crate) fn strip_leaked_special_tokens_from_delta(
    delta: String,
    tokenizer: &dyn Tokenizer,
) -> String {
    let special = tokenizer.get_special_tokens();

    let mut result = delta;
    if special.additional_special_tokens.is_empty() {
        for token in CHATML_FALLBACK_TOKENS {
            if result.contains(token) {
                result = result.replace(token, "");
            }
        }
    } else {
        for token in &special.additional_special_tokens {
            if result.contains(token.as_str()) {
                result = result.replace(token.as_str(), "");
            }
        }
    }
    result
}

/// Clean up a JSON response by removing markdown fences and trailing garbage.
///
/// Models sometimes wrap JSON output in `` ```json ... ``` `` fences or append
/// text after the JSON object. This function:
/// 1. Trims leading whitespace so detection works on indented output
/// 2. Strips markdown code fences if present (any `` ``` `` opener, case-insensitive)
/// 3. Truncates at the first complete top-level JSON object/array boundary
pub(crate) fn clean_json_response(text: &mut String) {
    let trimmed = text.trim_start();
    if trimmed.len() != text.len() {
        *text = trimmed.to_string();
    }
    strip_markdown_json_fence(text);
    truncate_to_json_boundary(text);
}

/// Strip `` ```[lang]\n...\n``` `` wrapping if present.
///
/// Accepts any triple-backtick opener regardless of language tag or casing
/// (`` ``` ``, `` ```json ``, `` ```JSON ``, `` ```Json ``, etc.).
fn strip_markdown_json_fence(text: &mut String) {
    if !text.starts_with("```") {
        return;
    }
    if let Some(start) = text.find('\n') {
        let inner = &text[start + 1..];
        if let Some(end) = inner.rfind("```") {
            *text = inner[..end].trim().to_string();
        }
    }
}

/// Truncate trailing content after the first balanced top-level JSON value.
///
/// Handles both `{...}` objects and `[...]` arrays. Uses a simple brace/bracket
/// depth counter with string-escape awareness.
fn truncate_to_json_boundary(text: &mut String) {
    let (open, close) = match text.chars().next() {
        Some('{') => ('{', '}'),
        Some('[') => ('[', ']'),
        _ => return,
    };

    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    let mut json_end = None;

    for (i, ch) in text.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                json_end = Some(i + ch.len_utf8());
                break;
            }
        }
    }
    if let Some(end) = json_end {
        if end < text.len() {
            text.truncate(end);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── JSON cleanup tests ──────────────────────────────────────────────

    #[test]
    fn test_strip_markdown_fence_json() {
        let mut text = "```json\n{\"key\": \"value\"}\n```".to_string();
        clean_json_response(&mut text);
        assert_eq!(text, r#"{"key": "value"}"#);
    }

    #[test]
    fn test_strip_markdown_fence_json_uppercase() {
        let mut text = "```JSON\n{\"key\": 1}\n```".to_string();
        clean_json_response(&mut text);
        assert_eq!(text, r#"{"key": 1}"#);
    }

    #[test]
    fn test_strip_plain_fence() {
        let mut text = "```\n{\"key\": 1}\n```".to_string();
        clean_json_response(&mut text);
        assert_eq!(text, r#"{"key": 1}"#);
    }

    #[test]
    fn test_strip_fence_with_leading_whitespace() {
        let mut text = "\n  ```json\n{\"key\": 1}\n```".to_string();
        clean_json_response(&mut text);
        assert_eq!(text, r#"{"key": 1}"#);
    }

    #[test]
    fn test_strip_markdown_fence_with_trailing_whitespace() {
        let mut text = "```json\n  {\"a\": 1}  \n```".to_string();
        clean_json_response(&mut text);
        assert_eq!(text, r#"{"a": 1}"#);
    }

    #[test]
    fn test_truncate_trailing_text_after_object() {
        let mut text = r#"{"a": 1} some trailing garbage"#.to_string();
        clean_json_response(&mut text);
        assert_eq!(text, r#"{"a": 1}"#);
    }

    #[test]
    fn test_truncate_trailing_text_after_array() {
        let mut text = "[1, 2, 3] extra stuff".to_string();
        clean_json_response(&mut text);
        assert_eq!(text, "[1, 2, 3]");
    }

    #[test]
    fn test_truncate_with_leading_whitespace() {
        let mut text = "  {\"a\": 1} trailing".to_string();
        clean_json_response(&mut text);
        assert_eq!(text, r#"{"a": 1}"#);
    }

    #[test]
    fn test_nested_objects() {
        let mut text = r#"{"a": {"b": {"c": 1}}} trailing"#.to_string();
        clean_json_response(&mut text);
        assert_eq!(text, r#"{"a": {"b": {"c": 1}}}"#);
    }

    #[test]
    fn test_braces_inside_strings() {
        let mut text = r#"{"a": "not {real} braces"} trailing"#.to_string();
        clean_json_response(&mut text);
        assert_eq!(text, r#"{"a": "not {real} braces"}"#);
    }

    #[test]
    fn test_escaped_quotes_in_strings() {
        let mut text = r#"{"a": "he said \"hi\""} extra"#.to_string();
        clean_json_response(&mut text);
        assert_eq!(text, r#"{"a": "he said \"hi\""}"#);
    }

    #[test]
    fn test_no_trailing_text_is_noop() {
        let mut text = r#"{"a": 1}"#.to_string();
        let original = text.clone();
        clean_json_response(&mut text);
        assert_eq!(text, original);
    }

    #[test]
    fn test_plain_text_is_noop() {
        let mut text = "just some text".to_string();
        let original = text.clone();
        clean_json_response(&mut text);
        assert_eq!(text, original);
    }

    #[test]
    fn test_fence_then_truncate_combined() {
        let mut text = "```json\n{\"a\": 1} garbage\n```".to_string();
        clean_json_response(&mut text);
        assert_eq!(text, r#"{"a": 1}"#);
    }

    #[test]
    fn test_nested_array_in_object() {
        let mut text = r#"{"items": [1, [2, 3], 4]} done"#.to_string();
        clean_json_response(&mut text);
        assert_eq!(text, r#"{"items": [1, [2, 3], 4]}"#);
    }

    #[test]
    fn test_empty_object() {
        let mut text = "{} trailing".to_string();
        clean_json_response(&mut text);
        assert_eq!(text, "{}");
    }

    #[test]
    fn test_empty_array() {
        let mut text = "[] trailing".to_string();
        clean_json_response(&mut text);
        assert_eq!(text, "[]");
    }

    #[test]
    fn test_backslash_at_string_end() {
        let mut text = r#"{"path": "C:\\Users\\"} after"#.to_string();
        clean_json_response(&mut text);
        assert_eq!(text, r#"{"path": "C:\\Users\\"}"#);
    }

    // ── Fallback token stripping tests ──────────────────────────────────

    /// MockTokenizer has empty `additional_special_tokens`, so it exercises the
    /// fallback to [`CHATML_FALLBACK_TOKENS`].
    fn mock_tokenizer() -> llm_tokenizer::MockTokenizer {
        llm_tokenizer::MockTokenizer::new()
    }

    #[test]
    fn test_fallback_strips_chatml_tokens() {
        let tok = mock_tokenizer();
        let mut text =
            "Hello world<|im_end|><|im_assistant|>".to_string();
        let changed = strip_leaked_special_tokens(&mut text, &tok);
        assert!(changed);
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn test_fallback_strips_think_tag() {
        let tok = mock_tokenizer();
        let mut text = "some output</think>".to_string();
        let changed = strip_leaked_special_tokens(&mut text, &tok);
        assert!(changed);
        assert_eq!(text, "some output");
    }

    #[test]
    fn test_fallback_noop_on_clean_text() {
        let tok = mock_tokenizer();
        let mut text = "clean text".to_string();
        let changed = strip_leaked_special_tokens(&mut text, &tok);
        assert!(!changed);
        assert_eq!(text, "clean text");
    }

    #[test]
    fn test_delta_fallback_strips_chatml() {
        let tok = mock_tokenizer();
        let delta = "Hello<|im_end|>".to_string();
        let result = strip_leaked_special_tokens_from_delta(delta, &tok);
        assert_eq!(result, "Hello");
    }

    #[test]
    fn test_strip_preserves_whitespace() {
        let tok = mock_tokenizer();
        let mut text = "  indented<|im_end|>  ".to_string();
        let changed = strip_leaked_special_tokens(&mut text, &tok);
        assert!(changed);
        assert_eq!(text, "  indented  ");
    }
}
