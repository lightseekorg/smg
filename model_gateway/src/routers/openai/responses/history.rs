//! OpenAI Responses wrapper around the shared history loader.

use std::collections::HashSet;

use axum::response::Response;
use openai_protocol::responses::{
    self, ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponsesRequest,
};

use crate::routers::{
    common::{
        agent_loop::PreparedLoopInput,
        header_utils::ConversationMemoryConfig,
        responses_history::load_request_history,
        transcript_lower::{
            extract_control_items, extract_mcp_list_tools_server_labels, lower_transcript,
        },
    },
    openai::context::ResponsesComponents,
};

pub(crate) struct PreparedResponsesHistory {
    pub request: ResponsesRequest,
    pub prepared: PreparedLoopInput,
    pub existing_mcp_list_tools_labels: HashSet<String>,
    pub previous_response_id: Option<String>,
}

pub(crate) async fn prepare_request_history(
    components: &ResponsesComponents,
    request: &ResponsesRequest,
) -> Result<PreparedResponsesHistory, Response> {
    let loaded = load_request_history(
        &components.response_storage,
        &components.conversation_storage,
        &components.conversation_item_storage,
        request,
    )
    .await?;

    let mut existing_mcp_list_tools_labels = loaded.existing_mcp_list_tools_labels;
    let mut control_items = loaded.control_items;
    let mut modified_request = request.clone();
    let mut combined = loaded.items;

    match &modified_request.input {
        ResponseInput::Items(items) => {
            existing_mcp_list_tools_labels.extend(extract_mcp_list_tools_server_labels(items));
            control_items.extend(extract_control_items(items));
            for item in items {
                combined.push(responses::normalize_input_item(item));
            }
        }
        ResponseInput::Text(text) => {
            combined.push(ResponseInputOutputItem::Message {
                id: format!("msg_u_{}", uuid::Uuid::now_v7()),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText { text: text.clone() }],
                status: Some("completed".to_string()),
                phase: None,
            });
        }
    }

    let previous_response_id = modified_request.previous_response_id.take();
    modified_request.input = ResponseInput::Items(lower_transcript(combined));
    modified_request.conversation = None;

    Ok(PreparedResponsesHistory {
        prepared: PreparedLoopInput::new(modified_request.input.clone(), control_items),
        request: modified_request,
        existing_mcp_list_tools_labels,
        previous_response_id,
    })
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
