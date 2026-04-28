//! Lower a Responses transcript to a backend-friendly form.
//!
//! The Responses API surfaces a number of "high-level" item types that
//! every backend pipeline reaches for differently: hosted-MCP calls
//! (`mcp_call`), MCP listings (`mcp_list_tools`), MCP approval
//! request/response pairs, and image-generation calls. Every backend we
//! support — chat-completions and the harmony pipeline alike — speaks
//! only in the small core vocabulary (`message`, `reasoning`,
//! `function_call`, `function_call_output`). Lowering converts the
//! high-level vocabulary to the core one once, in a single shared
//! place, so adapters never have to know that hosted MCP, image
//! generation, or any future hosted-tool concept exists.
//!
//! ## Adding a new high-level type
//!
//! Implement [`Lowering`] for the input variant and register the impl
//! in [`lower_item`]. The dispatch is exhaustive so the compiler will
//! force a match arm for every new variant added to
//! `ResponseInputOutputItem`.
//!
//! ## What "lower" means
//!
//! 1. **Project**: hosted-tool calls expand into the equivalent
//!    `function_call` (+ optional `function_call_output`) pair so
//!    backends see the same shape they would for caller-declared
//!    function tools.
//! 2. **Drop**: hosted-tool listings, approval pairs, and other
//!    metadata items that have no direct counterpart in the prompt are
//!    discarded; backends never had a way to render them and emitting
//!    them again on output is the responsibility of the agent loop, not
//!    the prompt builder.
//! 3. **Pass through**: everything else (`message`, `reasoning`,
//!    `function_call`, `function_call_output`, `simple_input_message`,
//!    etc.) passes through unchanged.

use std::collections::HashSet;

use openai_protocol::responses::{ResponseInput, ResponseInputOutputItem};
use serde_json::{json, Value};

/// Lower an input transcript to the core item set every backend
/// understands. Idempotent — running the function twice yields the
/// same result.
pub fn lower_transcript(items: Vec<ResponseInputOutputItem>) -> Vec<ResponseInputOutputItem> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        out.extend(lower_item(item));
    }
    out
}

/// Lower a single item. Most variants pass through; the high-level
/// ones project to a small list of core items or drop entirely.
fn lower_item(item: ResponseInputOutputItem) -> Vec<ResponseInputOutputItem> {
    match item {
        // Hosted-MCP call: project to a function_call (+ optional
        // function_call_output) pair so harmony / chat backends see
        // the standard tool exchange shape. A failed prior call must
        // surface its `error` text on the output side — otherwise the
        // model sees a tool_call with no result and silently retries.
        ResponseInputOutputItem::McpCall {
            id,
            arguments,
            name,
            output,
            error,
            status,
            // server_label, approval_request_id are MCP metadata
            // the backend prompt cannot represent — discard.
            ..
        } => {
            let mut out = Vec::with_capacity(2);
            // `call_id` is the cross-reference key the harmony / chat
            // builders use to pair call ↔ output. Reuse `id` for both
            // identifiers — the original mcp_call only had one.
            out.push(ResponseInputOutputItem::FunctionToolCall {
                id: id.clone(),
                call_id: id.clone(),
                name,
                arguments,
                output: None,
                status: status.clone().or_else(|| Some("completed".to_string())),
            });
            // Successful call: take the rendered output verbatim.
            // Failed call: project `error` into the output slot,
            // wrapped so the backend prompt can distinguish it from a
            // normal result. Filter out empty `output` strings before
            // falling through — some surfaces persist failed calls
            // with `output: Some("")` *and* a populated `error`, and
            // taking the empty string verbatim would erase the
            // failure context the prompt needs.
            let output_text = output
                .filter(|s| !s.is_empty())
                .or_else(|| error.map(|e| format!("Tool call failed: {e}")));
            if let Some(text) = output_text {
                out.push(ResponseInputOutputItem::FunctionCallOutput {
                    id: None,
                    call_id: id,
                    output: text,
                    status: status.or_else(|| Some("completed".to_string())),
                });
            }
            out
        }

        // Hosted-tool metadata items — these only exist for the client
        // to round-trip and should not reach the backend prompt. The
        // agent loop re-emits `mcp_list_tools` from session state on
        // output; drop on input.
        ResponseInputOutputItem::McpListTools { .. }
        | ResponseInputOutputItem::McpApprovalRequest { .. }
        | ResponseInputOutputItem::McpApprovalResponse { .. } => Vec::new(),

        // Image-generation hosted calls historically reached harmony as
        // "Unsupported input item type". Drop on input — the result
        // (if any) is best surfaced on output, which is rendered by the
        // image-generation tool path itself.
        ResponseInputOutputItem::ImageGenerationCall { .. } => Vec::new(),

        // Hosted shell-family tools (`shell_call`, `apply_patch_call`,
        // `local_shell_call`) — like hosted MCP, they round-trip through
        // function_call exchanges on the model side. Project the call
        // to a `function_call` with a synthetic name so the backend
        // prompt can replay the exchange via the same shape it knows.
        // The backend does not need the original action / operation
        // payload structure, so serialize it into the `arguments` blob.
        ResponseInputOutputItem::ShellCall {
            id,
            call_id,
            action,
            environment,
            status,
            ..
        } => {
            let arguments = serialize_hosted_args(json!({
                "action": action,
                "environment": environment,
            }));
            vec![ResponseInputOutputItem::FunctionToolCall {
                id: id.unwrap_or_else(|| call_id.clone()),
                call_id,
                name: "shell".to_string(),
                arguments,
                output: None,
                status: status
                    .map(|s| serde_json::to_string(&s).unwrap_or_else(|_| "completed".to_string()))
                    .map(|s| s.trim_matches('"').to_string())
                    .or_else(|| Some("completed".to_string())),
            }]
        }
        ResponseInputOutputItem::ShellCallOutput {
            call_id,
            output,
            id,
            status,
            ..
        } => {
            // `output` is a `Vec<ShellOutputChunk>`; serialize verbatim so
            // the backend sees the full stdout/stderr the original tool
            // produced. Status mirrors what `ShellCall` lowered to.
            let serialized_output = serde_json::to_string(&output).unwrap_or_default();
            vec![ResponseInputOutputItem::FunctionCallOutput {
                id,
                call_id,
                output: serialized_output,
                status: status
                    .map(|s| serde_json::to_string(&s).unwrap_or_else(|_| "completed".to_string()))
                    .map(|s| s.trim_matches('"').to_string())
                    .or_else(|| Some("completed".to_string())),
            }]
        }
        ResponseInputOutputItem::ApplyPatchCall {
            id,
            call_id,
            operation,
            status,
        } => {
            let arguments = serialize_hosted_args(json!({ "operation": operation }));
            vec![ResponseInputOutputItem::FunctionToolCall {
                id: id.unwrap_or_else(|| call_id.clone()),
                call_id,
                name: "apply_patch".to_string(),
                arguments,
                output: None,
                status: Some(
                    serde_json::to_string(&status)
                        .unwrap_or_else(|_| "\"completed\"".to_string())
                        .trim_matches('"')
                        .to_string(),
                ),
            }]
        }
        ResponseInputOutputItem::ApplyPatchCallOutput {
            call_id,
            status,
            id,
            output,
        } => vec![ResponseInputOutputItem::FunctionCallOutput {
            id,
            call_id,
            output: output.unwrap_or_default(),
            status: Some(
                serde_json::to_string(&status)
                    .unwrap_or_else(|_| "\"completed\"".to_string())
                    .trim_matches('"')
                    .to_string(),
            ),
        }],
        ResponseInputOutputItem::LocalShellCall {
            id,
            call_id: _,
            action,
            status,
        } => {
            // `LocalShellCallOutput` has only `id` (no `call_id`),
            // so the only key both sides share is `id`. Mirror it
            // into the synthesized `function_call`'s `call_id` so
            // the matching `LocalShellCallOutput` lowering below
            // can pair via the same key. The native `call_id` on
            // the input is dropped — backends that consume the
            // lowered transcript only look up function_call_output
            // by its `call_id`.
            let arguments = serialize_hosted_args(json!({ "action": action }));
            vec![ResponseInputOutputItem::FunctionToolCall {
                id: id.clone(),
                call_id: id,
                name: "local_shell".to_string(),
                arguments,
                output: None,
                status: Some(
                    serde_json::to_string(&status)
                        .unwrap_or_else(|_| "\"completed\"".to_string())
                        .trim_matches('"')
                        .to_string(),
                ),
            }]
        }
        ResponseInputOutputItem::LocalShellCallOutput { id, output, status } => {
            vec![ResponseInputOutputItem::FunctionCallOutput {
                id: Some(id.clone()),
                // Mirror `id` into `call_id` — see the symmetric
                // comment on `LocalShellCall` above.
                call_id: id,
                output,
                status: status.map(|s| {
                    serde_json::to_string(&s)
                        .unwrap_or_else(|_| "\"completed\"".to_string())
                        .trim_matches('"')
                        .to_string()
                }),
            }]
        }

        // Everything else passes through. Listed exhaustively so adding
        // a new variant to ResponseInputOutputItem is a forced compile
        // error here, not a silent drop.
        item @ (ResponseInputOutputItem::Message { .. }
        | ResponseInputOutputItem::SimpleInputMessage { .. }
        | ResponseInputOutputItem::Reasoning { .. }
        | ResponseInputOutputItem::FunctionToolCall { .. }
        | ResponseInputOutputItem::FunctionCallOutput { .. }
        | ResponseInputOutputItem::ComputerCall { .. }
        | ResponseInputOutputItem::ComputerCallOutput { .. }
        | ResponseInputOutputItem::CustomToolCall { .. }
        | ResponseInputOutputItem::CustomToolCallOutput { .. }
        | ResponseInputOutputItem::Compaction { .. }
        | ResponseInputOutputItem::ItemReference { .. }) => vec![item],
    }
}

/// Serialize a hosted-tool argument bundle to a JSON string. Used by
/// the shell / apply_patch / local_shell projections to flatten
/// structured action / operation fields into the `function_call`
/// `arguments` blob (which is documented as a JSON-string).
fn serialize_hosted_args(value: Value) -> String {
    serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string())
}

/// Convenience: lower the items inside a `ResponseInput`. `Text`
/// variants pass through unchanged.
pub fn lower_response_input(input: ResponseInput) -> ResponseInput {
    match input {
        ResponseInput::Items(items) => ResponseInput::Items(lower_transcript(items)),
        text @ ResponseInput::Text(_) => text,
    }
}

/// Collect `server_label` values for every `mcp_list_tools` item in a
/// transcript. Callers feed this into the agent loop's
/// `existing_mcp_list_tools_labels` set so a server whose listing the
/// caller already replayed (via `previous_response_id` history,
/// `conversation` items, or hand-stitched input items) is not
/// re-emitted on the way out.
///
/// Run **before** [`lower_transcript`] — the lowering pass drops
/// `mcp_list_tools` items, so calling this after lowering would always
/// return an empty set.
pub fn extract_mcp_list_tools_server_labels(items: &[ResponseInputOutputItem]) -> HashSet<String> {
    items
        .iter()
        .filter_map(|item| match item {
            ResponseInputOutputItem::McpListTools { server_label, .. } => {
                Some(server_label.clone())
            }
            _ => None,
        })
        .collect()
}

/// Pull out the loop's *control items* — items that drive loop-entry
/// decisions but must not reach the upstream model:
///
/// - `mcp_list_tools` — server inventory snapshot from a prior turn
/// - `mcp_approval_request` — model's last-turn request for user
///   approval
/// - `mcp_approval_response` — caller's approve / deny answer for
///   the matching `mcp_approval_request`
///
/// Returns the extracted items in source order. Run **before**
/// [`lower_transcript`] — the lowering pass drops these item types
/// once they reach the prompt-build path, so a post-lowering scan
/// would always come back empty.
pub fn extract_control_items(items: &[ResponseInputOutputItem]) -> Vec<ResponseInputOutputItem> {
    items
        .iter()
        .filter(|item| {
            matches!(
                item,
                ResponseInputOutputItem::McpListTools { .. }
                    | ResponseInputOutputItem::McpApprovalRequest { .. }
                    | ResponseInputOutputItem::McpApprovalResponse { .. }
            )
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use openai_protocol::responses::{ResponseContentPart, StringOrContentParts};

    use super::*;

    fn user_msg(text: &str) -> ResponseInputOutputItem {
        ResponseInputOutputItem::SimpleInputMessage {
            content: StringOrContentParts::String(text.to_string()),
            role: "user".to_string(),
            r#type: None,
            phase: None,
        }
    }

    #[test]
    fn mcp_call_with_output_lowers_to_function_call_pair() {
        let items = vec![
            user_msg("hi"),
            ResponseInputOutputItem::McpCall {
                id: "mcp_abc".to_string(),
                arguments: "{\"q\":1}".to_string(),
                name: "search".to_string(),
                server_label: "deepwiki".to_string(),
                approval_request_id: None,
                error: None,
                output: Some("[{\"r\":1}]".to_string()),
                status: Some("completed".to_string()),
            },
        ];

        let lowered = lower_transcript(items);
        assert_eq!(lowered.len(), 3);
        match &lowered[1] {
            ResponseInputOutputItem::FunctionToolCall {
                id,
                call_id,
                name,
                arguments,
                output,
                ..
            } => {
                assert_eq!(id, "mcp_abc");
                assert_eq!(call_id, "mcp_abc");
                assert_eq!(name, "search");
                assert_eq!(arguments, "{\"q\":1}");
                assert!(output.is_none());
            }
            other => panic!("expected FunctionToolCall, got {other:?}"),
        }
        match &lowered[2] {
            ResponseInputOutputItem::FunctionCallOutput {
                call_id, output, ..
            } => {
                assert_eq!(call_id, "mcp_abc");
                assert_eq!(output, "[{\"r\":1}]");
            }
            other => panic!("expected FunctionCallOutput, got {other:?}"),
        }
    }

    #[test]
    fn mcp_call_with_error_surfaces_error_on_output_side() {
        // A failed prior call has `error` populated and `output` empty.
        // Lowering must still produce a `function_call_output` so the
        // backend prompt sees the failure — otherwise the model's next
        // turn sees an unmatched tool_call and silently retries.
        let items = vec![ResponseInputOutputItem::McpCall {
            id: "mcp_err".to_string(),
            arguments: "{}".to_string(),
            name: "search".to_string(),
            server_label: "deepwiki".to_string(),
            approval_request_id: None,
            error: Some("Repository not found".to_string()),
            output: None,
            status: Some("failed".to_string()),
        }];

        let lowered = lower_transcript(items);
        assert_eq!(lowered.len(), 2);
        match &lowered[1] {
            ResponseInputOutputItem::FunctionCallOutput {
                call_id, output, ..
            } => {
                assert_eq!(call_id, "mcp_err");
                assert!(
                    output.contains("Repository not found"),
                    "expected error text on output side, got {output}"
                );
            }
            other => panic!("expected FunctionCallOutput, got {other:?}"),
        }
    }

    #[test]
    fn mcp_call_without_output_emits_call_only() {
        let items = vec![ResponseInputOutputItem::McpCall {
            id: "mcp_pending".to_string(),
            arguments: "{}".to_string(),
            name: "search".to_string(),
            server_label: "deepwiki".to_string(),
            approval_request_id: None,
            error: None,
            output: None,
            status: Some("in_progress".to_string()),
        }];

        let lowered = lower_transcript(items);
        assert_eq!(lowered.len(), 1);
        assert!(matches!(
            lowered[0],
            ResponseInputOutputItem::FunctionToolCall { .. }
        ));
    }

    #[test]
    fn mcp_metadata_items_drop_silently() {
        let items = vec![
            user_msg("hello"),
            ResponseInputOutputItem::McpListTools {
                id: "mcpl_1".to_string(),
                server_label: "deepwiki".to_string(),
                tools: vec![],
                error: None,
            },
            ResponseInputOutputItem::McpApprovalRequest {
                id: "mcpr_1".to_string(),
                server_label: "deepwiki".to_string(),
                name: "search".to_string(),
                arguments: "{}".to_string(),
            },
            ResponseInputOutputItem::McpApprovalResponse {
                id: None,
                approval_request_id: "mcpr_1".to_string(),
                approve: true,
                reason: None,
            },
        ];
        let lowered = lower_transcript(items);
        assert_eq!(lowered.len(), 1);
        assert!(matches!(
            lowered[0],
            ResponseInputOutputItem::SimpleInputMessage { .. }
        ));
    }

    #[test]
    fn image_generation_call_drops_silently() {
        let items = vec![
            user_msg("hello"),
            ResponseInputOutputItem::ImageGenerationCall {
                id: "ig_1".to_string(),
                action: None,
                background: None,
                output_format: None,
                quality: None,
                result: None,
                revised_prompt: None,
                size: None,
                status: None,
            },
        ];
        let lowered = lower_transcript(items);
        assert_eq!(lowered.len(), 1);
    }

    #[test]
    fn pass_through_items_are_preserved() {
        let items = vec![
            user_msg("hello"),
            ResponseInputOutputItem::Message {
                id: "msg_1".to_string(),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: "hi".to_string(),
                    annotations: vec![],
                    logprobs: None,
                }],
                status: Some("completed".to_string()),
                phase: None,
            },
            ResponseInputOutputItem::FunctionToolCall {
                id: "fc_1".to_string(),
                call_id: "fc_1".to_string(),
                name: "f".to_string(),
                arguments: "{}".to_string(),
                output: None,
                status: Some("completed".to_string()),
            },
        ];
        let lowered = lower_transcript(items);
        assert_eq!(lowered.len(), 3);
    }

    #[test]
    fn extract_labels_picks_up_hand_stitched_mcp_list_tools() {
        let items = vec![
            user_msg("ping"),
            ResponseInputOutputItem::McpListTools {
                id: "mcpl_1".to_string(),
                server_label: "deepwiki".to_string(),
                tools: vec![],
                error: None,
            },
            ResponseInputOutputItem::McpListTools {
                id: "mcpl_2".to_string(),
                server_label: "openai-developer-docs".to_string(),
                tools: vec![],
                error: None,
            },
        ];
        let labels = extract_mcp_list_tools_server_labels(&items);
        assert_eq!(labels.len(), 2);
        assert!(labels.contains("deepwiki"));
        assert!(labels.contains("openai-developer-docs"));
    }

    #[test]
    fn lowering_is_idempotent() {
        // `ResponseInputOutputItem` does not implement `PartialEq`,
        // so compare the lowered transcripts via their stable JSON
        // representation. The shape — not the types — is what matters
        // for the idempotency contract.
        let items = vec![ResponseInputOutputItem::McpCall {
            id: "mcp_x".to_string(),
            arguments: "{}".to_string(),
            name: "search".to_string(),
            server_label: "deepwiki".to_string(),
            approval_request_id: None,
            error: None,
            output: Some("ok".to_string()),
            status: Some("completed".to_string()),
        }];
        let once = lower_transcript(items);
        let twice = lower_transcript(once.clone());
        let once_json = serde_json::to_value(&once).unwrap();
        let twice_json = serde_json::to_value(&twice).unwrap();
        assert_eq!(once_json, twice_json);
    }
}
