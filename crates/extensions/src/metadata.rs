use chrono::{DateTime, Utc};
use smg_data_connector::RequestContext as StorageRequestContext;

#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConversationTurnInfo {
    pub user_turns: u32,
    pub total_items: u32,
    pub raw_stored_item_count: Option<u32>,
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct RequestMetadata {
    pub request_id: String,
    pub safety_identifier: Option<String>,
    pub tenant_id: Option<String>,
    pub originated_at: DateTime<Utc>,
    pub storage_request_context: Option<StorageRequestContext>,
}

impl RequestMetadata {
    pub fn build_from(
        request_id: impl Into<String>,
        safety_identifier: Option<String>,
        tenant_id: Option<String>,
        storage_request_context: Option<StorageRequestContext>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            safety_identifier,
            tenant_id,
            originated_at: Utc::now(),
            storage_request_context,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_info_default_is_zero() {
        let info = ConversationTurnInfo::default();
        assert_eq!(info.user_turns, 0);
        assert_eq!(info.total_items, 0);
        assert_eq!(info.raw_stored_item_count, None);
    }

    #[test]
    fn turn_info_destructure_with_rest_pattern() {
        let info = ConversationTurnInfo {
            user_turns: 4,
            total_items: 9,
            raw_stored_item_count: Some(10),
        };
        let ConversationTurnInfo { user_turns, .. } = info;
        assert_eq!(user_turns, 4);
    }

    #[test]
    fn request_metadata_build_from_populates_originated_at() {
        let md = RequestMetadata::build_from("req_123", Some("user_abc".into()), None, None);
        assert_eq!(md.request_id, "req_123");
        assert_eq!(md.safety_identifier.as_deref(), Some("user_abc"));
        let elapsed = Utc::now().signed_duration_since(md.originated_at);
        assert!(elapsed.num_seconds() < 2);
    }
}
