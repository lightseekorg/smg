//! Header parsing utilities for the Responses API.

use axum::http::HeaderMap;
use serde::Deserialize;
use tracing::debug;

static HEADER_CONVERSATION_MEMORY_CONFIG: http::header::HeaderName =
    http::header::HeaderName::from_static("x-conversation-memory-config");

/// Memory configuration parsed from the `x-conversation-memory-config` request header.
///
/// Returns defaults when the header is absent or unparsable.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ConversationMemoryConfig {
    #[serde(default)]
    pub long_term_memory: LongTermMemoryConfig,
    #[serde(default)]
    pub short_term_memory: ShortTermMemoryConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct LongTermMemoryConfig {
    #[serde(default)]
    pub enabled: bool,
    pub subject_id: Option<String>,
    pub embedding_model_id: Option<String>,
    pub extraction_model_id: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ShortTermMemoryConfig {
    #[serde(default)]
    pub enabled: bool,
    pub condenser_model_id: Option<String>,
}

/// Extract memory configuration from the `x-conversation-memory-config` JSON header.
///
/// Returns defaults when the header is absent or unparsable.
pub(crate) fn extract_conversation_memory_config(
    headers: Option<&HeaderMap>,
) -> ConversationMemoryConfig {
    let Some(value) = headers.and_then(|h| h.get(&HEADER_CONVERSATION_MEMORY_CONFIG)) else {
        return ConversationMemoryConfig::default();
    };

    let Ok(raw) = value.to_str() else {
        debug!("Invalid UTF-8 in x-conversation-memory-config header; using defaults");
        return ConversationMemoryConfig::default();
    };

    if raw.is_empty() {
        return ConversationMemoryConfig::default();
    }

    match serde_json::from_str::<ConversationMemoryConfig>(raw) {
        Ok(mut cfg) => {
            cfg.long_term_memory.subject_id =
                normalize_optional_string(cfg.long_term_memory.subject_id);
            cfg.long_term_memory.embedding_model_id =
                normalize_optional_string(cfg.long_term_memory.embedding_model_id);
            cfg.long_term_memory.extraction_model_id =
                normalize_optional_string(cfg.long_term_memory.extraction_model_id);
            cfg.short_term_memory.condenser_model_id =
                normalize_optional_string(cfg.short_term_memory.condenser_model_id);
            cfg
        }
        Err(e) => {
            debug!(error = %e, "Failed to parse x-conversation-memory-config header; using defaults");
            ConversationMemoryConfig::default()
        }
    }
}

fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderMap;

    use super::extract_conversation_memory_config;

    #[test]
    fn extract_conversation_memory_config_with_valid_json_populates_all_fields() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-conversation-memory-config",
            r#"{"long_term_memory":{"enabled":true,"subject_id":"sub-1","embedding_model_id":"emb-model","extraction_model_id":"ext-model"},"short_term_memory":{"enabled":true,"condenser_model_id":"cond-model"}}"#
                .parse()
                .unwrap(),
        );

        let cfg = extract_conversation_memory_config(Some(&headers));

        assert!(cfg.long_term_memory.enabled);
        assert_eq!(cfg.long_term_memory.subject_id.as_deref(), Some("sub-1"));
        assert_eq!(
            cfg.long_term_memory.embedding_model_id.as_deref(),
            Some("emb-model")
        );
        assert_eq!(
            cfg.long_term_memory.extraction_model_id.as_deref(),
            Some("ext-model")
        );
        assert!(cfg.short_term_memory.enabled);
        assert_eq!(
            cfg.short_term_memory.condenser_model_id.as_deref(),
            Some("cond-model")
        );
    }

    #[test]
    fn extract_conversation_memory_config_with_invalid_json_returns_defaults() {
        let mut headers = HeaderMap::new();
        headers.insert("x-conversation-memory-config", "not-json".parse().unwrap());

        let cfg = extract_conversation_memory_config(Some(&headers));

        assert!(!cfg.long_term_memory.enabled);
        assert!(!cfg.short_term_memory.enabled);
    }

    #[test]
    fn extract_conversation_memory_config_with_absent_header_returns_defaults() {
        let headers = HeaderMap::new();

        let cfg = extract_conversation_memory_config(Some(&headers));

        assert!(!cfg.long_term_memory.enabled);
        assert!(cfg.long_term_memory.subject_id.is_none());
        assert!(!cfg.short_term_memory.enabled);
        assert!(cfg.short_term_memory.condenser_model_id.is_none());
    }

    #[test]
    fn extract_conversation_memory_config_with_blank_json_fields_normalizes_to_none() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-conversation-memory-config",
            r#"{"long_term_memory":{"enabled":true,"subject_id":"   ","embedding_model_id":"","extraction_model_id":"   "},"short_term_memory":{"enabled":true,"condenser_model_id":"   "}}"#
                .parse()
                .unwrap(),
        );

        let cfg = extract_conversation_memory_config(Some(&headers));

        assert!(cfg.long_term_memory.enabled);
        assert!(cfg.short_term_memory.enabled);
        assert!(cfg.long_term_memory.subject_id.is_none());
        assert!(cfg.long_term_memory.embedding_model_id.is_none());
        assert!(cfg.long_term_memory.extraction_model_id.is_none());
        assert!(cfg.short_term_memory.condenser_model_id.is_none());
    }
}
