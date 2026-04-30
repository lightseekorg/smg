use async_trait::async_trait;

use crate::interceptor::ResponsesInterceptor;

/// Interceptor that does nothing on either phase.
///
/// Useful in tests, in default registries, and as a baseline for measuring
/// per-interceptor overhead.
#[derive(Default, Debug, Clone)]
pub struct NoOpInterceptor;

impl NoOpInterceptor {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ResponsesInterceptor for NoOpInterceptor {
    fn name(&self) -> &'static str {
        "noop"
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use chrono::Utc;
    use smg_data_connector::ConversationItemStorage;

    use super::*;
    use crate::context::{AfterPersistCtx, BeforeModelCtx};
    use crate::metadata::{ConversationTurnInfo, RequestMetadata};

    fn make_metadata() -> RequestMetadata {
        RequestMetadata {
            request_id: "req_test".into(),
            safety_identifier: None,
            tenant_id: None,
            originated_at: Utc::now(),
            storage_request_context: None,
        }
    }

    #[tokio::test]
    async fn noop_name_is_stable() {
        let i = NoOpInterceptor::new();
        assert_eq!(i.name(), "noop");
    }

    #[tokio::test]
    async fn noop_default_methods_do_nothing() {
        let i = NoOpInterceptor::new();
        let metadata = make_metadata();
        let history: Arc<dyn ConversationItemStorage> = Arc::new(
            smg_data_connector::MemoryConversationItemStorage::new(),
        );
        let mut request = openai_protocol::responses::ResponsesRequest::default();
        let headers = axum::http::HeaderMap::new();

        let mut before = BeforeModelCtx {
            headers: &headers,
            request: &mut request,
            conversation_id: None,
            history: history.clone(),
            turn_info: ConversationTurnInfo::default(),
            request_metadata: &metadata,
        };
        i.before_model(&mut before).await;

        let after = AfterPersistCtx {
            headers: &headers,
            request: &request,
            response_json: None,
            response_id: None,
            conversation_id: None,
            turn_info: ConversationTurnInfo::default(),
            persisted_item_ids: &[],
            request_metadata: &metadata,
        };
        i.after_persist(&after).await;
    }
}
