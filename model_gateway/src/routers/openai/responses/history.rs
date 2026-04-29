//! OpenAI Responses wrapper around the shared history preparer.

use axum::response::Response;
use openai_protocol::responses::ResponsesRequest;

use crate::routers::{
    common::{
        header_utils::ConversationMemoryConfig,
        responses_history::{
            prepare_request_history as prepare_shared_request_history, PreparedRequestHistory,
        },
    },
    openai::context::ResponsesComponents,
};

pub(crate) async fn prepare_request_history(
    components: &ResponsesComponents,
    request: &ResponsesRequest,
) -> Result<PreparedRequestHistory, Response> {
    prepare_shared_request_history(
        &components.response_storage,
        &components.conversation_storage,
        &components.conversation_item_storage,
        request,
    )
    .await
}

/// Memory hook entrypoint for Responses API.
///
/// This is intentionally a no-op in this PR: it confirms header parsing is
/// connected to request flow and logs activation state for follow-up retrieval work.
pub(crate) fn inject_memory_context(
    config: &ConversationMemoryConfig,
    _request_body: &mut ResponsesRequest,
) {
    if config.long_term_memory.enabled {
        tracing::debug!(
            has_subject_id = config.long_term_memory.subject_id.is_some(),
            has_embedding_model = config.long_term_memory.embedding_model_id.is_some(),
            has_extraction_model = config.long_term_memory.extraction_model_id.is_some(),
            "LTM recall requested - retrieval not yet implemented"
        );
    }

    if config.short_term_memory.enabled {
        tracing::debug!(
            has_condenser_model = config.short_term_memory.condenser_model_id.is_some(),
            "STM recall requested - retrieval not yet implemented"
        );
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::responses::{ResponseInput, ResponsesRequest};

    use super::inject_memory_context;
    use crate::routers::common::header_utils::{
        ConversationMemoryConfig, LongTermMemoryConfig, ShortTermMemoryConfig,
    };

    #[test]
    fn inject_memory_context_is_no_op_for_now() {
        let config = ConversationMemoryConfig {
            long_term_memory: LongTermMemoryConfig {
                enabled: true,
                policy: None,
                subject_id: Some("subj-1".to_string()),
                embedding_model_id: Some("embed-1".to_string()),
                extraction_model_id: Some("extract-1".to_string()),
            },
            short_term_memory: ShortTermMemoryConfig {
                enabled: true,
                condenser_model_id: Some("condense-1".to_string()),
            },
        };
        let mut request = ResponsesRequest {
            input: ResponseInput::Text("hello".to_string()),
            ..Default::default()
        };

        inject_memory_context(&config, &mut request);

        match request.input {
            ResponseInput::Text(text) => assert_eq!(text, "hello"),
            ResponseInput::Items(_) => {
                panic!("request input should remain unchanged for no-op hook")
            }
        }
    }
}
