//! Presentation-side renderer for tool calls surfaced through the
//! agent loop.
//!
//! The agent loop's tool call lifecycle has two facets that have to
//! stay in lock-step on the wire:
//!
//! 1. **Identity** — the call's `id` / `call_id` / `name` /
//!    `arguments` / `server_label`, which drive how the gateway
//!    actually dispatches the tool.
//! 2. **Presentation** — the *family* of streaming events and output
//!    item shape the client should see. A passthrough MCP tool maps
//!    to the `mcp_call.*` event family and an `mcp_call` output item;
//!    a hosted builtin maps to its own family
//!    (`web_search_call.*`, `code_interpreter_call.*`, etc.); a
//!    caller-declared function tool maps to `function_call.*`.
//!
//! This module owns the second facet. [`OutputFamily`] is the
//! agent-loop-local enum describing every wire family the loop can
//! surface — including caller-declared function tools, which are *not*
//! an MCP concept and therefore have no `ResponseFormat` variant.
//! [`ToolPresentation`] wraps an `OutputFamily` and centralizes the
//! rules for translating the call onto the wire (event family, item
//! shape, argument streaming policy).

use std::collections::HashMap;

use serde_json::{json, Value};
use smg_mcp::ResponseFormat;

use super::state::{ExecutedCall, LoopToolCall};

/// Wire family for one tool call. Spans every shape the agent loop
/// can put on the wire — `Function` is exclusively caller-declared,
/// the rest correspond 1:1 with [`smg_mcp::ResponseFormat`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputFamily {
    /// Caller-declared function tool. Surfaces as `function_call`
    /// output item; the gateway never executes it.
    Function,
    /// Gateway-routed MCP tool. Surfaces as `mcp_call`.
    McpCall,
    WebSearchCall,
    CodeInterpreterCall,
    FileSearchCall,
    ImageGenerationCall,
}

impl OutputFamily {
    /// Map an MCP-side [`ResponseFormat`] to the corresponding output
    /// family. The `Function` variant is *only* reachable for
    /// caller-declared tools and is constructed directly — it never
    /// flows through this conversion.
    pub(crate) const fn from_mcp_format(format: &ResponseFormat) -> Self {
        match format {
            ResponseFormat::Passthrough => OutputFamily::McpCall,
            ResponseFormat::WebSearchCall => OutputFamily::WebSearchCall,
            ResponseFormat::CodeInterpreterCall => OutputFamily::CodeInterpreterCall,
            ResponseFormat::FileSearchCall => OutputFamily::FileSearchCall,
            ResponseFormat::ImageGenerationCall => OutputFamily::ImageGenerationCall,
        }
    }

    /// Whether this family streams its arguments via
    /// `*_arguments.{delta,done}`. `Function` and `McpCall` do; the
    /// hosted builtins surface progress through structured
    /// `*.in_progress` / `*.searching` / `*.completed` events instead.
    pub(crate) const fn streams_arguments(self) -> bool {
        matches!(self, OutputFamily::Function | OutputFamily::McpCall)
    }

    /// Whether the family emits an intermediate searching /
    /// interpreting / generating event between `*.in_progress` and
    /// `*.completed`. Only the hosted builtins do.
    pub(crate) const fn has_searching_event(self) -> bool {
        matches!(
            self,
            OutputFamily::WebSearchCall
                | OutputFamily::CodeInterpreterCall
                | OutputFamily::FileSearchCall
                | OutputFamily::ImageGenerationCall
        )
    }

    /// Whether `*.failed` is part of this family's lifecycle. Only
    /// `McpCall` has a dedicated failure event today; hosted builtins
    /// surface failure inside `*.completed`, and caller-declared
    /// function calls have no execute phase at all.
    pub(crate) const fn has_failed_event(self) -> bool {
        matches!(self, OutputFamily::McpCall)
    }

    pub(crate) const fn id_prefix(self) -> &'static str {
        match self {
            OutputFamily::Function => "fc_",
            OutputFamily::McpCall => "mcp_",
            OutputFamily::WebSearchCall => "ws_",
            OutputFamily::CodeInterpreterCall => "ci_",
            OutputFamily::FileSearchCall => "fs_",
            OutputFamily::ImageGenerationCall => "ig_",
        }
    }
}

pub(crate) fn normalize_output_item_id(family: OutputFamily, source_id: &str) -> String {
    let prefix = family.id_prefix();
    if source_id.starts_with(prefix) {
        return source_id.to_string();
    }

    source_id
        .strip_prefix("fc_")
        .or_else(|| source_id.strip_prefix("call_"))
        .map(|stripped| format!("{prefix}{stripped}"))
        .unwrap_or_else(|| format!("{prefix}{source_id}"))
}

/// Whether a model-emitted tool call should be visible during the
/// current streaming turn. Approval-gated calls are hidden because the
/// driver will surface a single `mcp_approval_request` boundary instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolVisibility {
    Visible,
    SuppressedForApproval,
}

/// Common transfer descriptor consumed by streaming sinks. It bundles
/// the wire family and stream-time visibility decision so sinks do not
/// hold raw MCP response-format / approval-policy snapshots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ToolTransferDescriptor {
    pub family: OutputFamily,
    pub visibility: ToolVisibility,
}

impl ToolTransferDescriptor {
    pub(crate) const fn caller_function() -> Self {
        Self {
            family: OutputFamily::Function,
            visibility: ToolVisibility::Visible,
        }
    }

    pub(crate) const fn from_family_and_approval(
        family: OutputFamily,
        requires_approval: bool,
    ) -> Self {
        Self {
            family,
            visibility: if requires_approval {
                ToolVisibility::SuppressedForApproval
            } else {
                ToolVisibility::Visible
            },
        }
    }

    pub(crate) fn map_from_mcp_snapshots(
        presentation_snapshot: HashMap<String, ResponseFormat>,
        approval_snapshot: HashMap<String, bool>,
    ) -> HashMap<String, ToolTransferDescriptor> {
        presentation_snapshot
            .into_iter()
            .map(|(name, fmt)| {
                let requires_approval = approval_snapshot.get(&name).copied().unwrap_or(false);
                (
                    name,
                    ToolTransferDescriptor::from_family_and_approval(
                        OutputFamily::from_mcp_format(&fmt),
                        requires_approval,
                    ),
                )
            })
            .collect()
    }
}

/// Self-describing presentation rules for one tool call. Wraps an
/// [`OutputFamily`] and centralizes every rule for translating the
/// call onto the wire — event family, id prefix, argument streaming
/// policy, intermediate status events, initial / final output item
/// shape, and server-label visibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToolPresentation {
    pub family: OutputFamily,
}

impl ToolPresentation {
    pub(crate) const fn from_family(family: OutputFamily) -> Self {
        Self { family }
    }

    /// Build a presentation for a gateway tool whose MCP-side format
    /// is already known.
    pub(crate) const fn from_mcp_format(format: &ResponseFormat) -> Self {
        Self::from_family(OutputFamily::from_mcp_format(format))
    }

    pub(crate) const fn streams_arguments(&self) -> bool {
        self.family.streams_arguments()
    }

    pub(crate) const fn has_searching_event(&self) -> bool {
        self.family.has_searching_event()
    }

    pub(crate) const fn has_failed_event(&self) -> bool {
        self.family.has_failed_event()
    }

    /// Build the `output_item.added` JSON shape for the in-progress
    /// half of this call's lifecycle. Family-specific so each builtin
    /// gets its required wire fields (e.g. `web_search_call.action`,
    /// `image_generation_call.result`); the sink only has to allocate
    /// an item id and call this method.
    pub(crate) fn render_initial_item(
        &self,
        call: &LoopToolCall,
        server_label: &str,
        item_id: &str,
    ) -> Value {
        match self.family {
            OutputFamily::McpCall => json!({
                "id": item_id,
                "type": "mcp_call",
                "server_label": server_label,
                "name": call.name,
                "arguments": "",
                "approval_request_id": call.approval_request_id,
                "error": null,
                "output": null,
                "status": "in_progress",
            }),
            OutputFamily::WebSearchCall => json!({
                "id": item_id,
                "type": "web_search_call",
                // `action.type = "search"` is the family's documented
                // initial action shape; `query` is empty until the
                // executor parses arguments. Matches the OpenAI wire
                // contract for the in-progress added event.
                "action": { "type": "search", "query": "" },
                "status": "in_progress",
            }),
            OutputFamily::CodeInterpreterCall => json!({
                "id": item_id,
                "type": "code_interpreter_call",
                // `container_id` is required on the wire. We do not
                // know it before execution, so emit an empty string —
                // the matching `output_item.done` will replace it from
                // `transformed_item`.
                "container_id": "",
                "code": null,
                "outputs": [],
                "status": "in_progress",
            }),
            OutputFamily::FileSearchCall => json!({
                "id": item_id,
                "type": "file_search_call",
                // `queries` defaults to empty array; the executor
                // populates it inside the rendered output item.
                "queries": [],
                "results": null,
                "status": "in_progress",
            }),
            OutputFamily::ImageGenerationCall => json!({
                "id": item_id,
                "type": "image_generation_call",
                "result": null,
                "status": "in_progress",
            }),
            OutputFamily::Function => json!({
                "id": item_id,
                "type": "function_call",
                "name": call.name,
                "call_id": call.call_id,
                "arguments": "",
                "status": "in_progress",
            }),
        }
    }

    /// Build the `output_item.done` JSON shape for the completion half
    /// of this call's lifecycle. Prefers the executor's
    /// `transformed_item` and re-stamps `id` to the streaming-allocated
    /// item id so the lifecycle correlates client-side. On failure
    /// paths the renderer also forces `status: "failed"`, populates
    /// `error`, and clears `output`.
    pub(crate) fn render_final_item(&self, executed: &ExecutedCall, item_id: &str) -> Value {
        let mut value = match &executed.transformed_item {
            Some(item) => serde_json::to_value(item)
                .unwrap_or_else(|_| self.synthesize_fallback(executed, item_id)),
            None => self.synthesize_fallback(executed, item_id),
        };

        if let Some(obj) = value.as_object_mut() {
            // Re-stamp the streaming-allocated id so the wire-side
            // lifecycle correlates `output_item.added` ↔
            // `output_item.done` regardless of what id the executor
            // chose for `transformed_item`.
            obj.insert("id".to_string(), Value::String(item_id.to_string()));

            // Backfill `approval_request_id` for mcp_call output items
            // produced from an approval-continuation path. The
            // transformer (which renders `transformed_item`) cannot see
            // the original `mcpr_*` id; the loop carries it on
            // [`ExecutedCall`] and stamps it here so the final wire
            // payload echoes the request-side approval id.
            if matches!(self.family, OutputFamily::McpCall) {
                if let Some(ref approval_id) = executed.approval_request_id {
                    obj.insert(
                        "approval_request_id".to_string(),
                        Value::String(approval_id.clone()),
                    );
                }
            }

            if executed.is_error {
                obj.insert("status".to_string(), json!("failed"));
                // `error` exists only on `mcp_call` per spec; hosted
                // families convey failure through `status` alone.
                if matches!(self.family, OutputFamily::McpCall) {
                    let error_text = if executed.output_string.is_empty() {
                        "tool execution failed".to_string()
                    } else {
                        executed.output_string.clone()
                    };
                    obj.insert("error".to_string(), json!(error_text));
                }
                if obj.contains_key("output") {
                    obj.insert("output".to_string(), Value::Null);
                }
            }
        }
        value
    }

    fn synthesize_fallback(&self, executed: &ExecutedCall, item_id: &str) -> Value {
        // Fallback used when the executor did not populate
        // `transformed_item`; produces the per-family wire shape.
        let (output, error, status) = if executed.is_error {
            (Value::Null, json!(executed.output_string), "failed")
        } else {
            (json!(executed.output_string), Value::Null, "completed")
        };
        match self.family {
            OutputFamily::McpCall => json!({
                "id": item_id,
                "type": "mcp_call",
                "server_label": "",
                "name": executed.name,
                "arguments": executed.arguments,
                "approval_request_id": executed.approval_request_id,
                "error": error,
                "output": output,
                "status": status,
            }),
            // Hosted families have no `error` slot per spec; failure
            // surfaces through `status` alone.
            OutputFamily::WebSearchCall => json!({
                "id": item_id,
                "type": "web_search_call",
                "action": { "type": "search", "query": "" },
                "status": status,
            }),
            OutputFamily::CodeInterpreterCall => json!({
                "id": item_id,
                "type": "code_interpreter_call",
                "container_id": "",
                "code": null,
                "outputs": [],
                "status": status,
            }),
            OutputFamily::FileSearchCall => json!({
                "id": item_id,
                "type": "file_search_call",
                "queries": [],
                "results": null,
                "status": status,
            }),
            OutputFamily::ImageGenerationCall => json!({
                "id": item_id,
                "type": "image_generation_call",
                "result": null,
                "status": status,
            }),
            // Function (caller fc) has no execute phase, so this fallback
            // is only used when the loop's emission events flow through
            // sink without a paired transformer output. Keep the shape
            // wire-aligned with `function_call`.
            OutputFamily::Function => json!({
                "id": item_id,
                "type": "function_call",
                "name": executed.name,
                "call_id": executed.call_id,
                "arguments": executed.arguments,
                "status": status,
            }),
        }
    }
}

impl Default for ToolPresentation {
    fn default() -> Self {
        Self::from_mcp_format(&ResponseFormat::Passthrough)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loop_call(name: &str, arguments: &str) -> LoopToolCall {
        LoopToolCall {
            call_id: "call_x".to_string(),
            item_id: "item_x".to_string(),
            name: name.to_string(),
            arguments: arguments.to_string(),
            approval_request_id: None,
        }
    }

    /// Renderer signatures take `server_label` separately now (it lives
    /// on `PlannedToolExecution` / `PendingToolExecution`, not on
    /// `LoopToolCall`). Tests pass it explicitly via this helper.
    const TEST_SERVER_LABEL: &str = "deepwiki";

    fn executed_call(is_error: bool, output: &str) -> ExecutedCall {
        ExecutedCall {
            call_id: "call_x".to_string(),
            item_id: "item_x".to_string(),
            name: "search".to_string(),
            arguments: "{}".to_string(),
            output_string: output.to_string(),
            transformed_item: None,
            is_error,
            approval_request_id: None,
        }
    }

    #[test]
    fn passthrough_lifecycle_flags() {
        let p = ToolPresentation::from_mcp_format(&ResponseFormat::Passthrough);
        assert!(p.streams_arguments());
        assert!(!p.has_searching_event());
        assert!(p.has_failed_event());
    }

    #[test]
    fn function_lifecycle_flags() {
        let p = ToolPresentation::from_family(OutputFamily::Function);
        assert!(p.streams_arguments());
        assert!(!p.has_searching_event());
        assert!(!p.has_failed_event());
    }

    #[test]
    fn web_search_lifecycle_flags() {
        let p = ToolPresentation::from_mcp_format(&ResponseFormat::WebSearchCall);
        assert!(!p.streams_arguments());
        assert!(p.has_searching_event());
        assert!(!p.has_failed_event());
    }

    #[test]
    fn passthrough_initial_item_carries_required_fields() {
        let p = ToolPresentation::from_mcp_format(&ResponseFormat::Passthrough);
        let v = p.render_initial_item(
            &loop_call("search", "{\"q\":1}"),
            TEST_SERVER_LABEL,
            "mcp_abc",
        );
        let obj = v.as_object().unwrap();
        assert_eq!(obj["type"], "mcp_call");
        assert_eq!(obj["status"], "in_progress");
        assert_eq!(obj["server_label"], "deepwiki");
        assert!(obj.contains_key("approval_request_id"));
        assert!(obj.contains_key("error"));
        assert!(obj.contains_key("output"));
        assert!(
            !obj.contains_key("call_id"),
            "call_id must be omitted on initial item per OpenAI parity"
        );
    }

    #[test]
    fn web_search_initial_item_includes_action() {
        let p = ToolPresentation::from_mcp_format(&ResponseFormat::WebSearchCall);
        let v = p.render_initial_item(&loop_call("search", ""), TEST_SERVER_LABEL, "ws_abc");
        let obj = v.as_object().unwrap();
        assert_eq!(obj["type"], "web_search_call");
        assert_eq!(obj["action"]["type"], "search");
    }

    #[test]
    fn code_interpreter_initial_item_includes_container_id() {
        let p = ToolPresentation::from_mcp_format(&ResponseFormat::CodeInterpreterCall);
        let v = p.render_initial_item(&loop_call("python", ""), TEST_SERVER_LABEL, "ci_abc");
        let obj = v.as_object().unwrap();
        assert_eq!(obj["type"], "code_interpreter_call");
        assert!(obj.contains_key("container_id"));
    }

    #[test]
    fn file_search_initial_item_includes_queries() {
        let p = ToolPresentation::from_mcp_format(&ResponseFormat::FileSearchCall);
        let v = p.render_initial_item(&loop_call("file_search", ""), TEST_SERVER_LABEL, "fs_abc");
        let obj = v.as_object().unwrap();
        assert_eq!(obj["type"], "file_search_call");
        assert!(obj["queries"].is_array());
    }

    #[test]
    fn render_final_item_for_error_overrides_status_and_clears_output() {
        let p = ToolPresentation::from_mcp_format(&ResponseFormat::Passthrough);
        let exec = executed_call(true, "Repository not found");
        let v = p.render_final_item(&exec, "mcp_xyz");
        let obj = v.as_object().unwrap();
        assert_eq!(obj["status"], "failed");
        assert_eq!(obj["error"], "Repository not found");
        assert!(obj["output"].is_null());
    }

    #[test]
    fn render_final_item_for_success_keeps_output() {
        let p = ToolPresentation::from_mcp_format(&ResponseFormat::Passthrough);
        let exec = executed_call(false, "result text");
        let v = p.render_final_item(&exec, "mcp_xyz");
        let obj = v.as_object().unwrap();
        assert_eq!(obj["status"], "completed");
        assert_eq!(obj["output"], "result text");
        assert!(obj["error"].is_null());
    }

    #[test]
    fn render_final_item_backfills_approval_request_id() {
        let p = ToolPresentation::from_mcp_format(&ResponseFormat::Passthrough);
        let mut exec = executed_call(false, "result");
        exec.approval_request_id = Some("mcpr_call_42".to_string());
        let v = p.render_final_item(&exec, "mcp_xyz");
        assert_eq!(v["approval_request_id"], "mcpr_call_42");
    }
}
