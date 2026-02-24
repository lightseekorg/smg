use std::collections::HashMap;

use serde_json::Value;

use crate::core::ConversationMetadata;

/// Parse raw JSON string into `ConversationMetadata` (`JsonMap<String, Value>`).
///
/// Shared across Postgres, Redis, and Oracle conversation storage backends.
/// Returns `Ok(None)` for `None`, empty strings, and the literal `"null"`.
pub(super) fn parse_conversation_metadata(
    raw: Option<String>,
) -> Result<Option<ConversationMetadata>, String> {
    match raw {
        Some(s) if !s.is_empty() => {
            let s = s.trim();
            if s.is_empty() || s.eq_ignore_ascii_case("null") {
                return Ok(None);
            }
            serde_json::from_str::<ConversationMetadata>(s)
                .map(Some)
                .map_err(|e| e.to_string())
        }
        _ => Ok(None),
    }
}

pub(super) fn parse_tool_calls(raw: Option<String>) -> Result<Vec<Value>, String> {
    match raw {
        Some(s) if !s.is_empty() => serde_json::from_str(&s).map_err(|e| e.to_string()),
        _ => Ok(Vec::new()),
    }
}

pub(super) fn parse_metadata(raw: Option<String>) -> Result<HashMap<String, Value>, String> {
    match raw {
        Some(s) if !s.is_empty() => serde_json::from_str(&s).map_err(|e| e.to_string()),
        _ => Ok(HashMap::new()),
    }
}

pub(super) fn parse_raw_response(raw: Option<String>) -> Result<Value, String> {
    match raw {
        Some(s) if !s.is_empty() => serde_json::from_str(&s).map_err(|e| e.to_string()),
        _ => Ok(Value::Null),
    }
}

pub(super) fn parse_json_value(raw: Option<String>) -> Result<Value, String> {
    match raw {
        Some(s) if !s.is_empty() => serde_json::from_str(&s).map_err(|e| e.to_string()),
        _ => Ok(Value::Array(vec![])),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn parse_tool_calls_handles_empty_input() {
        assert!(parse_tool_calls(None).unwrap().is_empty());
        assert!(parse_tool_calls(Some(String::new())).unwrap().is_empty());
    }

    #[test]
    fn parse_tool_calls_round_trips() {
        let payload = json!([{ "type": "test", "value": 1 }]).to_string();
        let parsed = parse_tool_calls(Some(payload)).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["type"], "test");
        assert_eq!(parsed[0]["value"], 1);
    }

    #[test]
    fn parse_metadata_defaults_to_empty_map() {
        assert!(parse_metadata(None).unwrap().is_empty());
    }

    #[test]
    fn parse_metadata_round_trips() {
        let payload = json!({"key": "value", "nested": {"bool": true}}).to_string();
        let parsed = parse_metadata(Some(payload)).unwrap();
        assert_eq!(parsed.get("key").unwrap(), "value");
        assert_eq!(parsed["nested"]["bool"], true);
    }

    #[test]
    fn parse_raw_response_handles_null() {
        assert_eq!(parse_raw_response(None).unwrap(), Value::Null);
    }

    #[test]
    fn parse_raw_response_round_trips() {
        let payload = json!({"id": "abc"}).to_string();
        let parsed = parse_raw_response(Some(payload)).unwrap();
        assert_eq!(parsed["id"], "abc");
    }

    #[test]
    fn parse_conversation_metadata_none_returns_ok_none() {
        assert!(parse_conversation_metadata(None).unwrap().is_none());
    }

    #[test]
    fn parse_conversation_metadata_empty_string_returns_ok_none() {
        assert!(parse_conversation_metadata(Some(String::new()))
            .unwrap()
            .is_none());
    }

    #[test]
    fn parse_conversation_metadata_null_string_returns_ok_none() {
        assert!(parse_conversation_metadata(Some("null".to_string()))
            .unwrap()
            .is_none());
        // Also test case-insensitive
        assert!(parse_conversation_metadata(Some("NULL".to_string()))
            .unwrap()
            .is_none());
        assert!(parse_conversation_metadata(Some("Null".to_string()))
            .unwrap()
            .is_none());
    }

    #[test]
    fn parse_conversation_metadata_valid_json_object() {
        let payload = json!({"key": "value", "count": 42}).to_string();
        let parsed = parse_conversation_metadata(Some(payload))
            .unwrap()
            .expect("should be Some");
        assert_eq!(parsed.get("key").expect("key should exist"), "value");
        assert_eq!(
            parsed
                .get("count")
                .expect("count should exist")
                .as_i64()
                .expect("should be i64"),
            42
        );
    }

    #[test]
    fn parse_conversation_metadata_invalid_json_returns_err() {
        let result = parse_conversation_metadata(Some("not json".to_string()));
        assert!(result.is_err());
    }
}
