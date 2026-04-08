//! Post-parsing response cleanup utilities.
//!
//! When tool/reasoning parsers are active, backends send `skip_special_tokens=false`
//! so parsers can see structural tokens. After parsing completes, residual special
//! tokens must be stripped before the response reaches the user. These utilities use
//! the tokenizer's own special-token list so they work for any model family.

use llm_tokenizer::traits::Tokenizer;

/// Strip special tokens that leaked into text after tool/reasoning parsing.
///
/// Uses `tokenizer.get_special_tokens().additional_special_tokens` — the
/// model-specific added tokens (e.g. `<|im_end|>` for ChatML, `<|eot_id|>` for
/// Llama, `</think>` for thinking models). This is dynamic per-model, so no
/// hardcoded token list is needed.
pub(crate) fn strip_leaked_special_tokens(text: &mut String, tokenizer: &dyn Tokenizer) {
    let special = tokenizer.get_special_tokens();
    if special.additional_special_tokens.is_empty() {
        return;
    }

    let mut changed = false;
    for token in &special.additional_special_tokens {
        if text.contains(token.as_str()) {
            *text = text.replace(token.as_str(), "");
            changed = true;
        }
    }
    if changed {
        let trimmed = text.trim();
        if trimmed.len() != text.len() {
            *text = trimmed.to_string();
        }
    }
}

/// Streaming variant: strip leaked special tokens from a delta string,
/// returning the cleaned string.
pub(crate) fn strip_leaked_special_tokens_from_delta(
    delta: String,
    tokenizer: &dyn Tokenizer,
) -> String {
    let special = tokenizer.get_special_tokens();
    if special.additional_special_tokens.is_empty() {
        return delta;
    }

    let mut result = delta;
    for token in &special.additional_special_tokens {
        if result.contains(token.as_str()) {
            result = result.replace(token.as_str(), "");
        }
    }
    result
}

/// Clean up a JSON response by removing markdown fences and trailing garbage.
///
/// Models sometimes wrap JSON output in `` ```json ... ``` `` fences or append
/// text after the JSON object. This function:
/// 1. Strips markdown code fences if present
/// 2. Truncates at the first complete top-level JSON object/array boundary
pub(crate) fn clean_json_response(text: &mut String) {
    strip_markdown_json_fence(text);
    truncate_to_json_boundary(text);
}

/// Strip `` ```json\n...\n``` `` wrapping if present.
fn strip_markdown_json_fence(text: &mut String) {
    if !(text.starts_with("```json") || text.starts_with("```JSON")) {
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
    let first = match text.chars().next() {
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
        if ch == first.0 {
            depth += 1;
        } else if ch == first.1 {
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
}
