//! Streaming tool call handling for MCP interception.

use std::collections::HashMap;

use openai_protocol::event_types::{
    is_function_call_type, FunctionCallEvent, ItemType, OutputItemEvent, ResponseEvent,
};
use serde_json::Value;
use tracing::warn;

use crate::routers::openai::responses::{
    extract_output_index, get_event_type, StreamingResponseAccumulator,
};

/// Item-type discriminators for output items that the streaming tool loop
/// closes with its own `output_item.done` umbrella event. Upstream emits its
/// own duplicate umbrella before the structured `<type>.completed` sub-event,
/// which would violate the spec requirement that `output_item.done` is the
/// LAST event for a given item (see `.claude/_audit/openai-responses-api-spec.md`
/// §streaming). Whenever an `output_item.done` arrives for one of these types
/// while a tool call is pending, we suppress the upstream copy and let the
/// tool loop emit the umbrella at the correct position.
const TOOL_CALL_ITEM_TYPES: &[&str] = &[
    ItemType::FUNCTION_CALL,
    ItemType::FUNCTION_TOOL_CALL,
    ItemType::IMAGE_GENERATION_CALL,
    ItemType::WEB_SEARCH_CALL,
    ItemType::CODE_INTERPRETER_CALL,
    ItemType::FILE_SEARCH_CALL,
];

/// Check whether an item-type string designates a tool-call item whose
/// umbrella `output_item.done` is re-emitted by the tool loop (and therefore
/// must be suppressed when forwarded by upstream).
fn is_tool_call_item_type(item_type: &str) -> bool {
    TOOL_CALL_ITEM_TYPES.contains(&item_type)
}

/// Action to take based on streaming event processing
#[derive(Debug)]
pub(crate) enum StreamAction {
    /// Pass this event to the client.
    Forward,
    /// Accumulate the event for tool execution, do not send downstream.
    Buffer,
    /// The upstream signalled tool-call completion — run the MCP tool now.
    ///
    /// `forward_triggering_event` distinguishes two upstream shapes:
    ///
    /// * When the upstream emits `function_call_arguments.done` we want to
    ///   forward the event (it becomes `mcp_call_arguments.done`, a sub-event
    ///   that belongs BEFORE `response.<type>.completed`).
    /// * When the upstream skips the delta/done arguments events and signals
    ///   tool completion with `output_item.done` for the `function_call`
    ///   item, forwarding it would emit a duplicate umbrella
    ///   `response.output_item.done` event that fires BEFORE the sub-event
    ///   `response.<type>.completed` — violating the spec sub-event ordering
    ///   (spec §streaming: the umbrella `output_item.done` is always the
    ///   LAST event for a given item). The tool_loop's
    ///   `send_tool_call_completion_events` emits the correct umbrella
    ///   `output_item.done` in its proper position.
    ExecuteTools { forward_triggering_event: bool },
}

/// Maps upstream output indices to sequential downstream indices
#[derive(Debug, Default)]
pub(crate) struct OutputIndexMapper {
    next_index: usize,
    // Map upstream output_index -> remapped output_index
    assigned: HashMap<usize, usize>,
}

impl OutputIndexMapper {
    pub fn with_start(next_index: usize) -> Self {
        Self {
            next_index,
            assigned: HashMap::new(),
        }
    }

    pub fn ensure_mapping(&mut self, upstream_index: usize) -> usize {
        *self.assigned.entry(upstream_index).or_insert_with(|| {
            let assigned = self.next_index;
            self.next_index += 1;
            assigned
        })
    }

    pub fn lookup(&self, upstream_index: usize) -> Option<usize> {
        self.assigned.get(&upstream_index).copied()
    }

    pub fn allocate_synthetic(&mut self) -> usize {
        let assigned = self.next_index;
        self.next_index += 1;
        assigned
    }

    pub fn next_index(&self) -> usize {
        self.next_index
    }
}

/// Represents a function call being accumulated across delta events
#[derive(Debug, Clone)]
pub(crate) struct FunctionCallInProgress {
    pub call_id: String,
    pub name: String,
    pub arguments_buffer: String,
    pub item_id: Option<String>,
    pub output_index: usize,
    pub last_obfuscation: Option<String>,
    pub assigned_output_index: Option<usize>,
}

impl FunctionCallInProgress {
    pub fn new(call_id: String, output_index: usize) -> Self {
        Self {
            call_id,
            name: String::new(),
            arguments_buffer: String::new(),
            item_id: None,
            output_index,
            last_obfuscation: None,
            assigned_output_index: None,
        }
    }

    pub fn is_complete(&self) -> bool {
        !self.name.is_empty()
    }

    pub fn effective_output_index(&self) -> usize {
        self.assigned_output_index.unwrap_or(self.output_index)
    }
}

/// Handles streaming responses with MCP tool call interception
pub(crate) struct StreamingToolHandler {
    /// Accumulator for response persistence
    pub accumulator: StreamingResponseAccumulator,
    /// Function calls being built from deltas
    pub pending_calls: Vec<FunctionCallInProgress>,
    /// Manage output_index remapping so they increment per item
    output_index_mapper: OutputIndexMapper,
    /// Original response id captured from the first response.created event
    pub original_response_id: Option<String>,
}

impl StreamingToolHandler {
    pub fn with_starting_index(start: usize) -> Self {
        Self {
            accumulator: StreamingResponseAccumulator::new(),
            pending_calls: Vec::new(),
            output_index_mapper: OutputIndexMapper::with_start(start),
            original_response_id: None,
        }
    }

    pub fn ensure_output_index(&mut self, upstream_index: usize) -> usize {
        self.output_index_mapper.ensure_mapping(upstream_index)
    }

    pub fn mapped_output_index(&self, upstream_index: usize) -> Option<usize> {
        self.output_index_mapper.lookup(upstream_index)
    }

    pub fn allocate_synthetic_output_index(&mut self) -> usize {
        self.output_index_mapper.allocate_synthetic()
    }

    pub fn next_output_index(&self) -> usize {
        self.output_index_mapper.next_index()
    }

    pub fn original_response_id(&self) -> Option<&str> {
        self.original_response_id
            .as_deref()
            .or_else(|| self.accumulator.original_response_id())
    }

    pub fn snapshot_final_response(&self) -> Option<Value> {
        self.accumulator.snapshot_final_response()
    }

    /// Process an SSE event and determine what action to take
    pub fn process_event(&mut self, event_name: Option<&str>, data: &str) -> StreamAction {
        // Always feed to accumulator for storage
        self.accumulator.ingest_block(&format!(
            "{}data: {}",
            event_name
                .map(|n| format!("event: {n}\n"))
                .unwrap_or_default(),
            data
        ));

        let parsed: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return StreamAction::Forward,
        };

        match get_event_type(event_name, &parsed) {
            ResponseEvent::CREATED => {
                if self.original_response_id.is_none() {
                    self.original_response_id = parsed
                        .get("response")
                        .and_then(|v| v.get("id"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                }
                StreamAction::Forward
            }
            ResponseEvent::COMPLETED => StreamAction::Forward,
            OutputItemEvent::ADDED => self.handle_output_item_added(&parsed),
            FunctionCallEvent::ARGUMENTS_DELTA => self.handle_arguments_delta(&parsed),
            FunctionCallEvent::ARGUMENTS_DONE => self.handle_arguments_done(&parsed),
            OutputItemEvent::DELTA => self.process_output_delta(&parsed),
            OutputItemEvent::DONE => {
                if let Some(output_index) = extract_output_index(&parsed) {
                    self.ensure_output_index(output_index);
                }
                // Only suppress the umbrella `output_item.done` when it is
                // closing a tool-call item that we are intercepting — an
                // unrelated `output_item.done` for a message, reasoning
                // item, or mcp_list_tools block must reach the client
                // untouched even when a pending tool call is queued.
                //
                // The gate covers every tool-call item type that emits its
                // own structured `<type>.completed` sub-event (function_call
                // and the hosted builtins: image_generation_call,
                // web_search_call, code_interpreter_call, file_search_call).
                // When OpenAI cloud passthrough streams a native hosted tool
                // directly (not wrapped as function_call), the upstream
                // umbrella `output_item.done` fires BEFORE
                // `<type>.completed`, and forwarding it breaks the spec
                // ordering invariant. See PR #1365 CI failure log for
                // image_generation_call evidence.
                let is_tool_call_done = parsed
                    .get("item")
                    .and_then(|item| item.get("type"))
                    .and_then(|value| value.as_str())
                    .is_some_and(is_tool_call_item_type);
                if is_tool_call_done && self.has_complete_calls() {
                    // Suppress the upstream umbrella event — the tool loop
                    // will emit its own `output_item.done` at the correct
                    // position, AFTER `response.<type>.completed`.
                    StreamAction::ExecuteTools {
                        forward_triggering_event: false,
                    }
                } else {
                    StreamAction::Forward
                }
            }
            _ => StreamAction::Forward,
        }
    }

    fn handle_output_item_added(&mut self, parsed: &Value) -> StreamAction {
        let cached_output_index = extract_output_index(parsed);
        if let Some(output_index) = cached_output_index {
            self.ensure_output_index(output_index);
        }

        let Some(item) = parsed.get("item") else {
            return StreamAction::Forward;
        };
        let Some(item_type) = item.get("type").and_then(|v| v.as_str()) else {
            return StreamAction::Forward;
        };

        if !is_function_call_type(item_type) {
            return StreamAction::Forward;
        }

        let Some(output_index) = cached_output_index else {
            warn!(
                "Missing output_index in function_call added event, \
                 forwarding without processing for tool execution"
            );
            return StreamAction::Forward;
        };

        let assigned_index = self.ensure_output_index(output_index);
        let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
        let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");

        let call = self.get_or_create_call(output_index, item);
        call.call_id = call_id.to_string();
        call.name = name.to_string();
        call.item_id = item
            .get("id")
            .and_then(|v| v.as_str())
            .map(|id| id.to_string());
        call.assigned_output_index = Some(assigned_index);

        StreamAction::Forward
    }

    fn handle_arguments_delta(&mut self, parsed: &Value) -> StreamAction {
        let Some(output_index) = extract_output_index(parsed) else {
            return StreamAction::Forward;
        };

        let assigned_index = self.ensure_output_index(output_index);

        if let Some(delta) = parsed.get("delta").and_then(|v| v.as_str()) {
            if let Some(call) = self.find_call_mut(output_index) {
                call.arguments_buffer.push_str(delta);
                if let Some(obfuscation) = parsed.get("obfuscation").and_then(|v| v.as_str()) {
                    call.last_obfuscation = Some(obfuscation.to_string());
                }
                if call.assigned_output_index.is_none() {
                    call.assigned_output_index = Some(assigned_index);
                }
            }
        }
        StreamAction::Forward
    }

    fn handle_arguments_done(&mut self, parsed: &Value) -> StreamAction {
        if let Some(output_index) = extract_output_index(parsed) {
            let assigned_index = self.ensure_output_index(output_index);
            if let Some(call) = self.find_call_mut(output_index) {
                if call.assigned_output_index.is_none() {
                    call.assigned_output_index = Some(assigned_index);
                }
            }
        }

        if self.has_complete_calls() {
            // Forward the triggering event — `function_call_arguments.done`
            // becomes `mcp_call_arguments.done` which is a sub-event
            // belonging BEFORE `response.<type>.completed` and therefore
            // safe to forward inline here.
            StreamAction::ExecuteTools {
                forward_triggering_event: true,
            }
        } else {
            StreamAction::Forward
        }
    }

    fn find_call_mut(&mut self, output_index: usize) -> Option<&mut FunctionCallInProgress> {
        self.pending_calls
            .iter_mut()
            .find(|c| c.output_index == output_index)
    }

    /// Process output delta events to detect and accumulate function calls
    fn process_output_delta(&mut self, event: &Value) -> StreamAction {
        let output_index = extract_output_index(event).unwrap_or(0);
        let assigned_index = self.ensure_output_index(output_index);

        let delta = match event.get("delta") {
            Some(d) => d,
            None => return StreamAction::Forward,
        };

        let item_type = delta.get("type").and_then(|v| v.as_str());

        if item_type.is_some_and(is_function_call_type) {
            // Get or create function call for this output index
            let call = self.get_or_create_call(output_index, delta);
            call.assigned_output_index = Some(assigned_index);

            if let Some(call_id) = delta.get("call_id").and_then(|v| v.as_str()) {
                call.call_id = call_id.to_string();
            }
            if let Some(item_id) = delta.get("id").and_then(|v| v.as_str()) {
                call.item_id = Some(item_id.to_string());
            }
            if let Some(name) = delta.get("name").and_then(|v| v.as_str()) {
                call.name.push_str(name);
            }
            if let Some(args) = delta.get("arguments").and_then(|v| v.as_str()) {
                call.arguments_buffer.push_str(args);
            }

            if let Some(obfuscation) = delta.get("obfuscation").and_then(|v| v.as_str()) {
                call.last_obfuscation = Some(obfuscation.to_string());
            }

            return StreamAction::Buffer;
        }

        StreamAction::Forward
    }

    fn get_or_create_call(
        &mut self,
        output_index: usize,
        delta: &Value,
    ) -> &mut FunctionCallInProgress {
        if let Some(pos) = self
            .pending_calls
            .iter()
            .position(|c| c.output_index == output_index)
        {
            return &mut self.pending_calls[pos];
        }

        let call_id = delta
            .get("call_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let mut call = FunctionCallInProgress::new(call_id, output_index);
        if let Some(obfuscation) = delta.get("obfuscation").and_then(|v| v.as_str()) {
            call.last_obfuscation = Some(obfuscation.to_string());
        }

        self.pending_calls.push(call);
        #[expect(
            clippy::expect_used,
            reason = "just pushed an element; last_mut is infallible"
        )]
        self.pending_calls
            .last_mut()
            .expect("Just pushed to pending_calls, must have at least one element")
    }

    fn has_complete_calls(&self) -> bool {
        !self.pending_calls.is_empty() && self.pending_calls.iter().all(|c| c.is_complete())
    }

    pub fn take_pending_calls(&mut self) -> Vec<FunctionCallInProgress> {
        std::mem::take(&mut self.pending_calls)
    }
}

#[cfg(test)]
mod tests {
    //! Regression tests for R6.7 Gap B — streaming event ordering for
    //! image-generation-style built-in tool calls.
    //!
    //! The bug: when the upstream signalled tool completion with a direct
    //! `response.output_item.done` event (no preceding
    //! `response.function_call_arguments.done`), the streaming layer used
    //! to forward that upstream event to the client. The tool loop then
    //! emitted its own umbrella `output_item.done` AFTER emitting the
    //! `<tool>.completed` sub-event — producing the wire sequence
    //!
    //!     response.output_item.added
    //!     response.<tool>.in_progress
    //!     response.output_item.done       ← duplicate (upstream-forwarded)
    //!     response.<tool>.generating
    //!     response.<tool>.completed
    //!     response.output_item.done       ← tool-loop synthesized
    //!
    //! which violates the spec invariant that `response.output_item.done`
    //! is the LAST event emitted for a given item (see
    //! `.claude/_audit/openai-responses-api-spec.md` §streaming, events
    //! L1054-L1072).

    use super::*;

    /// Feed an `output_item.added` for a function_call item so the handler
    /// has a complete pending call registered.
    fn bootstrap_function_call_added(handler: &mut StreamingToolHandler) {
        let data = r#"{
          "type": "response.output_item.added",
          "output_index": 0,
          "item": {
            "type": "function_call",
            "id": "fc_test",
            "call_id": "call_test",
            "name": "image_generation"
          }
        }"#;
        let _ = handler.process_event(Some("response.output_item.added"), data);
    }

    #[test]
    fn output_item_done_triggers_execute_tools_without_forwarding() {
        // Gap B: when the upstream signals tool-call completion via
        // `output_item.done` for a function_call item (no preceding
        // `function_call_arguments.done`), the handler must ask the
        // caller NOT to forward the triggering event — the tool loop
        // will emit its own umbrella `output_item.done` at the correct
        // position after `response.<tool>.completed`.
        let mut handler = StreamingToolHandler::with_starting_index(0);
        bootstrap_function_call_added(&mut handler);

        let done_event = r#"{
          "type": "response.output_item.done",
          "output_index": 0,
          "item": {
            "type": "function_call",
            "id": "fc_test",
            "call_id": "call_test",
            "name": "image_generation",
            "arguments": "{}",
            "status": "completed"
          }
        }"#;

        let action = handler.process_event(Some("response.output_item.done"), done_event);

        match action {
            StreamAction::ExecuteTools {
                forward_triggering_event,
            } => assert!(
                !forward_triggering_event,
                "output_item.done must be suppressed to avoid a duplicate umbrella event"
            ),
            other => panic!("expected ExecuteTools, got {other:?}"),
        }
    }

    /// Assert that an `output_item.done` for `item_type` is suppressed (the
    /// gate returns `ExecuteTools { forward_triggering_event: false }`) when
    /// a tool call is already pending. This is the R6.7b regression: R6.7's
    /// original gate only matched `function_call`/`function_tool_call`, so
    /// OpenAI cloud passthrough of native hosted tools (which emits the
    /// item directly with its hosted-tool type) slipped through and the
    /// duplicate umbrella event reached the wire. See PR #1365 CI log for
    /// the `image_generation_call` reproduction.
    fn assert_output_item_done_suppressed_for_hosted_tool(item_type: &str) {
        let mut handler = StreamingToolHandler::with_starting_index(0);
        bootstrap_function_call_added(&mut handler);

        let done_event = format!(
            r#"{{
              "type": "response.output_item.done",
              "output_index": 0,
              "item": {{
                "type": "{item_type}",
                "id": "hosted_test",
                "status": "completed"
              }}
            }}"#
        );

        let action = handler.process_event(Some("response.output_item.done"), &done_event);

        match action {
            StreamAction::ExecuteTools {
                forward_triggering_event,
            } => assert!(
                !forward_triggering_event,
                "output_item.done for {item_type} must be suppressed to avoid a duplicate umbrella event"
            ),
            other => panic!("expected ExecuteTools for {item_type}, got {other:?}"),
        }
    }

    #[test]
    fn output_item_done_suppressed_for_image_generation_call() {
        // R6.7b: OpenAI cloud passthrough emits the `image_generation_call`
        // item directly — the umbrella `output_item.done` that upstream
        // sends before `response.image_generation_call.completed` must be
        // suppressed so the tool loop's umbrella lands at the right spot
        // (see PR #1365 CI failure: `completed` at index 9, `output_item.done`
        // at index 3).
        assert_output_item_done_suppressed_for_hosted_tool(ItemType::IMAGE_GENERATION_CALL);
    }

    #[test]
    fn output_item_done_suppressed_for_web_search_call() {
        // R6.7b: hosted `web_search_call` items emitted directly by
        // upstream must also trigger suppression — the tool loop emits
        // `response.web_search_call.completed` followed by its own
        // `output_item.done` at the correct position.
        assert_output_item_done_suppressed_for_hosted_tool(ItemType::WEB_SEARCH_CALL);
    }

    #[test]
    fn output_item_done_suppressed_for_code_interpreter_call() {
        // R6.7b: hosted `code_interpreter_call` items share the ordering
        // contract with the other tool-call item types.
        assert_output_item_done_suppressed_for_hosted_tool(ItemType::CODE_INTERPRETER_CALL);
    }

    #[test]
    fn output_item_done_suppressed_for_file_search_call() {
        // R6.7b: hosted `file_search_call` items share the ordering
        // contract with the other tool-call item types.
        assert_output_item_done_suppressed_for_hosted_tool(ItemType::FILE_SEARCH_CALL);
    }

    #[test]
    fn arguments_done_triggers_execute_tools_with_forwarding() {
        // Forwarding is safe for `function_call_arguments.done` because it
        // becomes `mcp_call_arguments.done` — a sub-event that belongs
        // BEFORE `response.<tool>.completed` per spec sub-event ordering.
        let mut handler = StreamingToolHandler::with_starting_index(0);
        bootstrap_function_call_added(&mut handler);

        let args_done = r#"{
          "type": "response.function_call_arguments.done",
          "output_index": 0,
          "item_id": "fc_test",
          "arguments": "{}"
        }"#;

        let action =
            handler.process_event(Some("response.function_call_arguments.done"), args_done);

        match action {
            StreamAction::ExecuteTools {
                forward_triggering_event,
            } => assert!(
                forward_triggering_event,
                "function_call_arguments.done forwards as mcp_call_arguments.done"
            ),
            other => panic!("expected ExecuteTools, got {other:?}"),
        }
    }

    #[test]
    fn output_item_done_for_unrelated_item_forwards_even_with_pending_call() {
        // Regression for PR #1369 review (chatgpt-codex-connector P1):
        // suppression must be gated on the done event's item type. When a
        // function_call is queued AND an unrelated `output_item.done` for a
        // message / reasoning / mcp_list_tools block arrives, the umbrella
        // event for that sibling item must pass through untouched so the
        // client sees its completion. Only the function_call's own
        // `output_item.done` should be suppressed (the tool loop will emit
        // the correct umbrella at its proper position).
        let mut handler = StreamingToolHandler::with_starting_index(0);
        bootstrap_function_call_added(&mut handler);

        let done_event = r#"{
          "type": "response.output_item.done",
          "output_index": 1,
          "item": {
            "type": "message",
            "id": "msg_sibling",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello"}]
          }
        }"#;

        let action = handler.process_event(Some("response.output_item.done"), done_event);

        assert!(
            matches!(action, StreamAction::Forward),
            "sibling output_item.done must forward even while a function_call is pending; got {action:?}"
        );
    }

    #[test]
    fn output_item_done_without_pending_calls_forwards() {
        // Non-function-call umbrella `output_item.done` events (messages,
        // mcp_list_tools items, etc.) must pass through unchanged. This
        // prevents regressing the path where the assistant emits a plain
        // text message after tool execution completes.
        let mut handler = StreamingToolHandler::with_starting_index(0);

        let done_event = r#"{
          "type": "response.output_item.done",
          "output_index": 0,
          "item": {
            "type": "message",
            "id": "msg_test",
            "status": "completed",
            "role": "assistant",
            "content": []
          }
        }"#;

        let action = handler.process_event(Some("response.output_item.done"), done_event);

        assert!(
            matches!(action, StreamAction::Forward),
            "expected Forward for an umbrella output_item.done on a non-tool item, got {action:?}"
        );
    }
}
