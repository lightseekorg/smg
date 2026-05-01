//! Surface-side helpers for regular Responses.
//!
//! - `ResponsesCallContext` — request-scoped knobs the handler bundles
//!   for both modes.

use axum::http;
use openai_protocol::responses::{
    ResponseContentPart, ResponseInputOutputItem, ResponseReasoningContent,
};

use crate::{middleware::TenantRequestMeta, routers::common::agent_loop::AgentLoopState};

/// Per-request parameters for chat pipeline execution. Bundles values
/// that are always threaded together through the regular responses
/// call chain.
pub(super) struct ResponsesCallContext {
    pub headers: Option<http::HeaderMap>,
    pub model_id: String,
    pub response_id: Option<String>,
    pub tenant_request_meta: TenantRequestMeta,
}

pub(super) fn append_assistant_prefix_to_transcript(
    state: &mut AgentLoopState,
    request_id: &str,
    reasoning_text: Option<&str>,
    message_text: Option<&str>,
) {
    if let Some(text) = reasoning_text.filter(|text| !text.is_empty()) {
        state
            .transcript
            .push(ResponseInputOutputItem::new_reasoning(
                format!("reasoning_{request_id}"),
                vec![],
                vec![ResponseReasoningContent::ReasoningText {
                    text: text.to_string(),
                }],
                Some("completed".to_string()),
            ));
    }

    if let Some(text) = message_text.filter(|text| !text.is_empty()) {
        state.transcript.push(ResponseInputOutputItem::Message {
            id: format!("msg_{request_id}"),
            role: "assistant".to_string(),
            content: vec![ResponseContentPart::OutputText {
                text: text.to_string(),
                annotations: vec![],
                logprobs: None,
            }],
            status: Some("completed".to_string()),
            phase: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::responses::{ResponseInput, ResponseReasoningContent};

    use super::*;

    #[test]
    fn assistant_prefix_replay_preserves_reasoning_and_text_order() {
        let mut state = AgentLoopState::new(ResponseInput::Items(vec![]), Default::default());

        append_assistant_prefix_to_transcript(
            &mut state,
            "resp_1",
            Some("analysis"),
            Some("partial"),
        );

        assert!(matches!(
            state.transcript.as_slice(),
            [
                ResponseInputOutputItem::Reasoning { id, content, .. },
                ResponseInputOutputItem::Message { id: msg_id, content: msg_content, .. },
            ] if id == "reasoning_resp_1"
                && matches!(
                    content.as_slice(),
                    [ResponseReasoningContent::ReasoningText { text }] if text == "analysis"
                )
                && msg_id == "msg_resp_1"
                && matches!(
                    msg_content.as_slice(),
                    [ResponseContentPart::OutputText { text, .. }] if text == "partial"
                )
        ));
    }
}
