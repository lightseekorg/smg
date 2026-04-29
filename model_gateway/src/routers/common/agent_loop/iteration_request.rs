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

pub(crate) enum IterationRequestFlavor {
    /// gRPC regular converts the typed Responses sub-request into a Chat
    /// Completion request, so normalize input items before conversion and keep
    /// the caller's original Responses tools on the sub-request.
    RegularChat { stream: Option<bool> },
    /// Responses-shaped upstreams (gRPC harmony + OpenAI passthrough) consume
    /// typed Responses requests directly and receive their model-visible tool
    /// set from the caller.
    Responses {
        stream: Option<bool>,
        tools: Option<Vec<ResponseTool>>,
    },
}

impl IterationRequestFlavor {
    fn input_options(&self) -> IterationInputOptions {
        match self {
            IterationRequestFlavor::RegularChat { .. } => {
                IterationInputOptions::normalized_message()
            }
            IterationRequestFlavor::Responses { .. } => IterationInputOptions::preserved_message(),
        }
    }

    fn stream(&self) -> Option<bool> {
        match self {
            IterationRequestFlavor::RegularChat { stream }
            | IterationRequestFlavor::Responses { stream, .. } => *stream,
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
    flavor: IterationRequestFlavor,
) -> ResponsesRequest {
    let mut request = original.clone();
    request.input =
        ResponseInput::Items(build_iteration_input_items(state, flavor.input_options()));
    request.store = Some(false);
    request.previous_response_id = None;
    request.conversation = None;
    if let Some(stream) = flavor.stream() {
        request.stream = Some(stream);
    }

    match flavor {
        IterationRequestFlavor::RegularChat { .. } => {
            if state.tool_budget_exhausted {
                request.tools = None;
            }
        }
        IterationRequestFlavor::Responses { tools, .. } => {
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

    #[test]
    fn regular_chat_flavor_sets_stream_and_keeps_original_tools() {
        let original = ResponsesRequest {
            input: ResponseInput::Text("hello".to_string()),
            tools: Some(vec![ResponseTool::Computer]),
            ..Default::default()
        };

        let request = build_responses_iteration_request(
            &original,
            &state_with_text("hello"),
            IterationRequestFlavor::RegularChat { stream: Some(true) },
        );

        assert_eq!(request.stream, Some(true));
        assert!(matches!(
            request.tools.as_deref(),
            Some([ResponseTool::Computer])
        ));
    }

    #[test]
    fn responses_flavor_uses_adapter_tool_override() {
        let original = ResponsesRequest {
            input: ResponseInput::Text("hello".to_string()),
            tools: None,
            ..Default::default()
        };

        let request = build_responses_iteration_request(
            &original,
            &state_with_text("hello"),
            IterationRequestFlavor::Responses {
                stream: None,
                tools: Some(vec![ResponseTool::Computer]),
            },
        );

        assert!(matches!(
            request.tools.as_deref(),
            Some([ResponseTool::Computer])
        ));
    }
}
