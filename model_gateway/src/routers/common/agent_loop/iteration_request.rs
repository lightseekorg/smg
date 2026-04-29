//! Shared per-iteration request assembly for Responses agent-loop adapters.

use openai_protocol::responses::{
    self, ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseTool,
    ResponsesRequest, ResponsesToolChoice, ToolChoiceOptions,
};
use uuid::Uuid;

use super::state::AgentLoopState;

#[derive(Clone, Copy)]
pub(crate) struct IterationInputOptions {
    normalize_items: bool,
}

impl IterationInputOptions {
    pub(crate) const fn normalized_message() -> Self {
        Self {
            normalize_items: true,
        }
    }

    pub(crate) const fn preserved_message() -> Self {
        Self {
            normalize_items: false,
        }
    }
}

pub(crate) enum IterationTools {
    Original,
    Override(Option<Vec<ResponseTool>>),
}

pub(crate) struct IterationRequestOptions {
    pub(crate) input: IterationInputOptions,
    pub(crate) stream: Option<bool>,
    pub(crate) tools: IterationTools,
}

impl IterationRequestOptions {
    pub(crate) fn with_original_tools(input: IterationInputOptions, stream: Option<bool>) -> Self {
        Self {
            input,
            stream,
            tools: IterationTools::Original,
        }
    }

    pub(crate) fn with_tool_override(
        input: IterationInputOptions,
        stream: Option<bool>,
        tools: Option<Vec<ResponseTool>>,
    ) -> Self {
        Self {
            input,
            stream,
            tools: IterationTools::Override(tools),
        }
    }
}

/// Build the model-visible input for one loop iteration from the prepared
/// upstream input plus the driver-owned transcript.
pub(crate) fn build_iteration_input_items(
    state: &AgentLoopState,
    options: IterationInputOptions,
) -> Vec<ResponseInputOutputItem> {
    let upstream_items = match &state.upstream_input {
        ResponseInput::Items(items) if options.normalize_items => items
            .iter()
            .map(responses::normalize_input_item)
            .collect::<Vec<_>>(),
        ResponseInput::Items(items) => items.clone(),
        ResponseInput::Text(text) => vec![text_input_item(text)],
    };

    let mut combined = Vec::with_capacity(upstream_items.len() + state.transcript.len());
    combined.extend(upstream_items);
    combined.extend(state.transcript.iter().cloned());
    combined
}

/// Build a typed Responses sub-request for gRPC-backed adapters.
pub(crate) fn build_responses_iteration_request(
    original: &ResponsesRequest,
    state: &AgentLoopState,
    options: IterationRequestOptions,
) -> ResponsesRequest {
    let mut request = original.clone();
    request.input = ResponseInput::Items(build_iteration_input_items(state, options.input));
    request.store = Some(false);
    request.previous_response_id = None;
    request.conversation = None;
    if let Some(stream) = options.stream {
        request.stream = Some(stream);
    }

    match options.tools {
        IterationTools::Original => {
            if state.tool_budget_exhausted {
                request.tools = None;
            }
        }
        IterationTools::Override(tools) => {
            request.tools = if state.tool_budget_exhausted {
                None
            } else {
                tools
            };
        }
    }

    if state.tool_budget_exhausted {
        request.tool_choice = None;
    } else if state.iteration > 1 {
        request.tool_choice = Some(ResponsesToolChoice::Options(ToolChoiceOptions::Auto));
    }

    request
}

fn text_input_item(text: &str) -> ResponseInputOutputItem {
    ResponseInputOutputItem::Message {
        id: format!("msg_u_{}", Uuid::now_v7()),
        role: "user".to_string(),
        content: vec![ResponseContentPart::InputText {
            text: text.to_string(),
        }],
        status: Some("completed".to_string()),
        phase: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_with_text(text: &str) -> AgentLoopState {
        AgentLoopState::new(ResponseInput::Text(text.to_string()), Default::default())
    }

    #[test]
    fn builds_message_text_input_for_regular_style() {
        let items = build_iteration_input_items(
            &state_with_text("hello"),
            IterationInputOptions::normalized_message(),
        );

        assert!(matches!(
            items.as_slice(),
            [ResponseInputOutputItem::Message { role, content, .. }]
                if role == "user"
                    && matches!(
                        content.as_slice(),
                        [ResponseContentPart::InputText { text }] if text == "hello"
                    )
        ));
    }
}
