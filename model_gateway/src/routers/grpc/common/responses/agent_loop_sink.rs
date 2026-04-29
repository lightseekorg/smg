//! gRPC `StreamSink` implementation backing both regular and harmony
//! Responses streaming.
//!
//! The sink owns an [`ResponseStreamEventEmitter`] and the SSE channel.
//! `LoopEvent`s the agent-loop driver emits map onto the emitter's
//! high-level "response.created / response.completed / response.incomplete"
//! calls; chunk-level
//! events (message deltas, content_part.added/done, output_text.delta,
//! function_call_arguments.delta for *caller-declared* function tools)
//! stay inside the surface adapter — those need parser-context the
//! driver does not have.
//!
//! ## Boundary contract
//!
//! Gateway-owned tool call lifecycle (`mcp_call.*`,
//! `web_search_call.*`, `code_interpreter_call.*`,
//! `file_search_call.*`, `image_generation_call.*` plus the matching
//! `output_item.added` / `output_item.done`) is owned exclusively by
//! this sink. Surface parsers (harmony streaming processor, regular
//! chat-stream `process_chunk`) MUST NOT emit those events inline —
//! they only collect tool-call announcements into accumulator state
//! and surface them through `LoopModelTurn`'s
//! `pending_gateway_tool_calls` / `ToolCallsFound` so the driver can
//! fire `LoopEvent::ToolCallStarted` / `ToolCompleted` with the
//! correct ordering relative to actual tool execution. The harmony
//! processor still emits inline events for *user-managed*
//! `function_call` tools (which the driver never executes, so they
//! never see a `ToolCallStarted`); that is the only inline tool
//! emission that survives the boundary.
//!
//! ## Approval gating
//!
//! Tools whose policy is `require_approval: always` must NOT surface
//! a streamed `mcp_call.*` lifecycle on the turn the model emits them
//! — only the `mcp_approval_request` item that the driver appends via
//! `RenderMode::ApprovalInterrupt` is allowed to reach the wire.
//! The common presentation layer passes the sink a per-name transfer
//! descriptor (`OutputFamily` + visibility). For approval-gated calls
//! the descriptor says `SuppressedForApproval`, so the sink drops
//! `ToolCallEmissionStarted` / `Fragment` / `Done` without allocating
//! an `output_index` (the `mcp_approval_request` then takes index 0).
//! On the continuation turn the driver primes execution and emits a
//! single [`LoopEvent::ApprovedToolReplay`], which the sink expands into
//! a compressed open-half lifecycle (`output_item.added` →
//! `mcp_call.in_progress` → `mcp_call_arguments.delta/done`); the
//! matching close-half fires later via `ToolCompleted`.
//!
//! Once the connection drops the sink stops calling `send_event`; the
//! driver still pumps events into it, but they become no-ops. This keeps
//! the loop's control flow independent of transport-level disconnects.

use std::{collections::HashMap, io};

use bytes::Bytes;
use openai_protocol::responses::ResponseOutputItem;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::routers::common::{
    agent_loop::{
        ExecutedCall, LoopEvent, LoopToolCall, OutputFamily, PendingToolExecution, StreamSink,
        ToolPresentation, ToolTransferDescriptor, ToolVisibility,
    },
    responses_streaming::{OutputItemType, ResponseStreamEventEmitter},
};

/// Bookkeeping for a single in-flight tool call's wire-side lifecycle.
/// Populated on `ToolCallEmissionStarted`, consumed on subsequent
/// `ToolCallArgumentsFragment` / `ToolCallEmissionDone` (for caller
/// `function_call`) or `ToolCompleted` (for gateway tools after
/// execution). Keyed by `call_id` — the cross-event correlation key.
struct ToolCallTracking {
    output_index: Option<usize>,
    item_id: Option<String>,
    presentation: ToolPresentation,
    /// Tool name as the model emitted it. Cached so
    /// `ToolCallEmissionDone` can synthesize the closing
    /// `function_call` output item without the adapter re-passing
    /// the name on every event.
    name: String,
    /// `mcp_call.server_label` for gateway calls. Approval-continuation
    /// replays carry the original server_label here so the rendered
    /// `output_item.added(mcp_call)` matches the prompt the gateway
    /// emitted on the prior turn. Empty for caller-declared function
    /// tools and hosted builtins where no MCP server label exists.
    server_label: String,
    /// `mcpr_*` approval-request id, only set for approval-continuation
    /// replays. Threaded through to `output_item.added(mcp_call)` so
    /// clients can correlate the streamed lifecycle with the prior-turn
    /// approval prompt.
    approval_request_id: Option<String>,
    /// Argument fragments captured from the model stream. Gateway tools flush
    /// them per item immediately before `*_arguments.done`.
    argument_fragments: Vec<String>,
    full_args: Option<String>,
    open_emitted: bool,
}

/// Shared streaming sink for both gRPC Responses surfaces.
pub(crate) struct GrpcResponseStreamSink {
    pub(crate) emitter: ResponseStreamEventEmitter,
    pub(crate) tx: mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    /// Latches once a `send_event` fails so subsequent emits become
    /// no-ops without needing the driver to thread a disconnect signal.
    disconnected: bool,
    /// Final-turn usage bundle the adapter stages from inside its
    /// `render_final` (which the driver calls *before*
    /// `Sink::emit(LoopEvent::ResponseFinished)`). Carries the numbers
    /// the terminal response payload reports back to the client.
    final_usage: Option<Value>,
    /// Per-call_id wire-side state. Used to pair
    /// `ToolCallEmissionStarted` with later `ArgumentsFragment` /
    /// `EmissionDone` (and, for gateway tools, the subsequent
    /// `ToolCompleted` after execution) into the same
    /// `output_index` / `item_id`.
    tool_call_tracking: HashMap<String, ToolCallTracking>,
    /// `name → transfer descriptor` map built by the common
    /// presentation layer at sink construction. Tools not in the map
    /// default to caller-declared [`OutputFamily::Function`] with
    /// visible streaming.
    tool_transfers: HashMap<String, ToolTransferDescriptor>,
    /// Per-call_index → call_id mapping used by the chat-stream
    /// translation path ([`process_chat_chunk`]). Chat completion
    /// streams only carry `id` + `name` on the *first* delta for a
    /// given `tool_calls[].index`; later deltas just carry argument
    /// fragments. The sink looks the call_id up here so subsequent
    /// `ToolCallArgumentsFragment` events fire on the correct id.
    chat_call_ids_by_index: HashMap<u32, String>,
    /// Per-call_id accumulated argument buffer used by the chat-
    /// stream translation path. Each `ToolCallArgumentsFragment`
    /// appends here; on chunked finish or stream end the sink uses
    /// the accumulated full buffer to fire `ToolCallEmissionDone`.
    /// The harmony processor does **not** populate this — it issues
    /// `ToolCallEmissionDone` itself when the harmony stream's
    /// `Complete` variant arrives, with full args from its parser.
    chat_args_accumulator: HashMap<String, String>,
    /// Exposed tool name → MCP server label snapshot.
    tool_server_labels: HashMap<String, String>,
}

impl GrpcResponseStreamSink {
    pub(crate) fn new(
        emitter: ResponseStreamEventEmitter,
        tx: mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        tool_transfers: HashMap<String, ToolTransferDescriptor>,
        tool_server_labels: HashMap<String, String>,
    ) -> Self {
        Self {
            emitter,
            tx,
            disconnected: false,
            final_usage: None,
            tool_call_tracking: HashMap::new(),
            tool_transfers,
            chat_call_ids_by_index: HashMap::new(),
            chat_args_accumulator: HashMap::new(),
            tool_server_labels,
        }
    }

    /// Look up a tool's transfer descriptor by exposed-name. Unknown
    /// names (caller-declared function tools that the gateway never
    /// registered under an MCP exposure) default to visible
    /// [`OutputFamily::Function`].
    fn transfer_descriptor(&self, name: &str) -> ToolTransferDescriptor {
        self.tool_transfers
            .get(name)
            .copied()
            .unwrap_or_else(ToolTransferDescriptor::caller_function)
    }

    /// Adapter-side hook for the surface-specific tail. The adapter
    /// computes its own usage shape (Harmony has reasoning-token
    /// details, Chat completion has only the classic triple) and stages
    /// it before the driver fires `ResponseFinished`.
    pub(crate) fn set_final_usage(&mut self, usage: Option<Value>) {
        self.final_usage = usage;
    }

    /// Translate one chat-completion stream chunk into wire events.
    ///
    /// Handles the two orthogonal pieces a chat stream carries:
    /// - **text + finish**: forwarded through the underlying
    ///   `ResponseStreamEventEmitter::process_chunk` (response.* +
    ///   output_text deltas + content_part lifecycle + message close).
    /// - **tool_calls deltas**: per-call_index, recognized here and
    ///   re-emitted as the loop's family-agnostic
    ///   `LoopEvent::ToolCallEmissionStarted` (first chunk per index)
    ///   and `LoopEvent::ToolCallArgumentsFragment` (each subsequent
    ///   chunk that brings argument tokens). Sink-internal classifier
    ///   then dispatches them onto `function_call` / `mcp_call` /
    ///   hosted-builtin wire shapes.
    ///
    /// `LoopEvent::ToolCallEmissionDone` is fired once the chunk
    /// carries `finish_reason` (any value, since either `tool_calls`
    /// or `stop` ends an in-flight tool's args), with the full
    /// accumulated arguments string. After firing, the per-call_index
    /// and per-call_id buffers are cleared so the next iteration's
    /// chunks start fresh.
    pub(crate) fn process_chat_chunk(
        &mut self,
        chunk: &openai_protocol::chat::ChatCompletionStreamResponse,
    ) {
        if self.disconnected {
            return;
        }
        // `process_chunk` errs when the receiver drops; latch the flag
        // so later emit_* calls short-circuit.
        if self.emitter.process_chunk(chunk, &self.tx).is_err() {
            self.disconnected = true;
            return;
        }

        let Some(choice) = chunk.choices.first() else {
            return;
        };

        if let Some(deltas) = &choice.delta.tool_calls {
            for delta in deltas {
                let idx = delta.index;
                let id_in_delta = delta.id.as_deref();
                if let Some(id) = id_in_delta {
                    // First chunk for this call_index — it carries
                    // both `id` and `name`. Register the mapping and
                    // open the wire lifecycle.
                    let name = delta
                        .function
                        .as_ref()
                        .and_then(|f| f.name.as_deref())
                        .unwrap_or("");
                    self.chat_call_ids_by_index.insert(idx, id.to_string());
                    self.chat_args_accumulator
                        .insert(id.to_string(), String::new());
                    self.emit(LoopEvent::ToolCallEmissionStarted {
                        call_id: id,
                        item_id: id,
                        name,
                    });
                    if let Some(args) = delta
                        .function
                        .as_ref()
                        .and_then(|f| f.arguments.as_deref())
                        .filter(|s| !s.is_empty())
                    {
                        if let Some(buf) = self.chat_args_accumulator.get_mut(id) {
                            buf.push_str(args);
                        }
                        self.emit(LoopEvent::ToolCallArgumentsFragment {
                            call_id: id,
                            fragment: args,
                        });
                    }
                } else if let Some(call_id) = self.chat_call_ids_by_index.get(&idx).cloned() {
                    // Subsequent chunk for an in-flight call —
                    // append args.
                    if let Some(args) = delta
                        .function
                        .as_ref()
                        .and_then(|f| f.arguments.as_deref())
                        .filter(|s| !s.is_empty())
                    {
                        if let Some(buf) = self.chat_args_accumulator.get_mut(&call_id) {
                            buf.push_str(args);
                        }
                        self.emit(LoopEvent::ToolCallArgumentsFragment {
                            call_id: &call_id,
                            fragment: args,
                        });
                    }
                }
            }
        }

        // On any finish, close out every in-flight tool call's
        // emission lifecycle with the accumulated full args. Sink
        // emits `*_arguments.done` for streaming families and (for
        // `function_call` only) the closing `output_item.done`.
        if choice.finish_reason.is_some() {
            let mut call_ids: Vec<(u32, String)> = self
                .chat_call_ids_by_index
                .iter()
                .map(|(idx, call_id)| (*idx, call_id.clone()))
                .collect();
            call_ids.sort_by_key(|(idx, _)| *idx);
            for (_, call_id) in call_ids {
                let full_args = self
                    .chat_args_accumulator
                    .get(&call_id)
                    .cloned()
                    .unwrap_or_default();
                self.emit(LoopEvent::ToolCallEmissionDone {
                    call_id: &call_id,
                    full_args: &full_args,
                });
            }
            self.chat_call_ids_by_index.clear();
            self.chat_args_accumulator.clear();
        }
    }

    fn send(&mut self, value: &Value) {
        if self.disconnected {
            return;
        }
        if self.emitter.send_event(value, &self.tx).is_err() {
            self.disconnected = true;
        }
    }

    fn emit_response_started(&mut self) {
        let event = self.emitter.emit_created();
        self.send(&event);
        let event = self.emitter.emit_in_progress();
        self.send(&event);
    }

    fn emit_response_completed(&mut self, incomplete_reason: Option<&str>) {
        let mut usage = self.final_usage.take();
        if let Some(reason) = incomplete_reason {
            let event = self.emitter.emit_incomplete(reason, usage.as_ref());
            self.send(&event);
            return;
        }
        // Reattach in case nobody read it (idempotent close).
        if usage.is_none() {
            usage = self.final_usage.take();
        }
        let event = self.emitter.emit_completed(usage.as_ref());
        self.send(&event);
    }

    fn emit_mcp_list_tools(&mut self, item: &ResponseOutputItem) {
        let ResponseOutputItem::McpListTools {
            id,
            server_label,
            tools,
            ..
        } = item
        else {
            return;
        };
        if self.disconnected {
            return;
        }

        // Emit the same four-event sequence as
        // `ResponseStreamEventEmitter::emit_mcp_list_tools_sequence`,
        // but feed pre-rendered `McpToolInfo`s straight from the
        // `ResponseOutputItem` rather than re-deriving them from a
        // session's `ToolEntry`s. The driver only knows about output
        // items, so this lets the sink stay surface-agnostic.
        let tool_items: Vec<Value> = tools
            .iter()
            .map(|t| serde_json::to_value(t).unwrap_or(Value::Null))
            .collect();

        let item_in_progress = json!({
            "id": id,
            "type": "mcp_list_tools",
            "server_label": server_label,
            "tools": [],
        });
        // The emitter's per-output-index allocator is what assigns the
        // numeric index that `output_item.added` and `.done` must agree
        // on; reuse it so the index space stays consistent with whatever
        // the adapter has already emitted.
        let (output_index, _allocated_id) = self
            .emitter
            .allocate_output_index(OutputItemType::McpListTools);
        let event = self
            .emitter
            .emit_output_item_added(output_index, &item_in_progress);
        self.send(&event);
        let event = self
            .emitter
            .emit_mcp_list_tools_in_progress(output_index, id);
        self.send(&event);
        let event = self
            .emitter
            .emit_mcp_list_tools_completed(output_index, id, &tool_items);
        self.send(&event);

        let item_done = json!({
            "id": id,
            "type": "mcp_list_tools",
            "server_label": server_label,
            "tools": tool_items,
        });
        let event = self.emitter.emit_output_item_done(output_index, &item_done);
        self.send(&event);
        self.emitter.complete_output_item(output_index);
    }

    /// Adapter saw a tool call's first chunk in the model token
    /// stream — open the wire lifecycle for it. The sink reads the
    /// common transfer descriptor by name (default
    /// [`OutputFamily::Function`] + visible) and either:
    ///
    /// - Approval-gated: drop *every* wire event silently and skip
    ///   `output_index` allocation. The driver's
    ///   `RenderMode::ApprovalInterrupt` will surface a
    ///   `mcp_approval_request` item at index 0; clients must not see
    ///   a phantom `mcp_call` slot for the gated call.
    /// - Caller function_call: allocate `output_index` / `item_id` and emit
    ///   the open lifecycle immediately because the gateway will not execute it.
    /// - Gateway-owned tools: record the model-emitted call but defer
    ///   `output_index` allocation until `ToolCallExecutionStarted`, so skipped
    ///   calls do not create hidden index gaps.
    ///
    /// Tracking is created only for non-suppressed calls; subsequent
    /// `ToolCallArgumentsFragment` / `ToolCallEmissionDone` /
    /// `ToolCompleted` events for a suppressed call are no-ops.
    fn emit_tool_call_emission_started(&mut self, call_id: &str, item_id_hint: &str, name: &str) {
        if self.disconnected {
            return;
        }
        let descriptor = self.transfer_descriptor(name);
        if matches!(descriptor.visibility, ToolVisibility::SuppressedForApproval) {
            // Drop the entire stream-time lifecycle. No tracking, no
            // index allocation. The driver surfaces this call as
            // `mcp_approval_request` instead.
            return;
        }
        let presentation = ToolPresentation::from_family(descriptor.family);
        let family = presentation.family;

        let server_label = if matches!(family, OutputFamily::McpCall) {
            self.tool_server_labels
                .get(name)
                .cloned()
                .unwrap_or_default()
        } else {
            String::new()
        };
        let open_emitted = matches!(family, OutputFamily::Function);
        let (output_index, item_id) = if open_emitted {
            let output_item_type =
                ResponseStreamEventEmitter::output_item_type_for_family(Some(family));
            let (output_index, allocated_item_id) =
                self.emitter.allocate_output_index(output_item_type);
            let call = LoopToolCall {
                call_id: call_id.to_string(),
                item_id: item_id_hint.to_string(),
                name: name.to_string(),
                arguments: String::new(),
                approval_request_id: None,
            };
            let item = presentation.render_initial_item(&call, &server_label, &allocated_item_id);
            let event = self.emitter.emit_output_item_added(output_index, &item);
            self.send(&event);
            (Some(output_index), Some(allocated_item_id))
        } else {
            (None, None)
        };

        self.tool_call_tracking.insert(
            call_id.to_string(),
            ToolCallTracking {
                output_index,
                item_id,
                presentation,
                name: name.to_string(),
                server_label,
                approval_request_id: None,
                argument_fragments: Vec::new(),
                full_args: None,
                open_emitted,
            },
        );
    }

    fn flush_tool_call_open_and_arguments(&mut self, call_id: &str, fallback_args: &str) {
        if self.disconnected {
            return;
        }

        let snapshot = {
            let Some(tracking) = self.tool_call_tracking.get_mut(call_id) else {
                return;
            };
            if tracking.open_emitted {
                return;
            }
            let full_args = tracking
                .full_args
                .clone()
                .unwrap_or_else(|| fallback_args.to_string());
            (
                tracking.presentation.clone(),
                tracking.name.clone(),
                tracking.server_label.clone(),
                tracking.approval_request_id.clone(),
                tracking.argument_fragments.clone(),
                full_args,
            )
        };

        let (presentation, name, server_label, approval_request_id, fragments, full_args) =
            snapshot;
        let family = presentation.family;
        let output_item_type =
            ResponseStreamEventEmitter::output_item_type_for_family(Some(family));
        let (output_index, item_id) = self.emitter.allocate_output_index(output_item_type);
        let call = LoopToolCall {
            call_id: call_id.to_string(),
            item_id: item_id.clone(),
            name,
            arguments: String::new(),
            approval_request_id,
        };
        let initial_item = presentation.render_initial_item(&call, &server_label, &item_id);

        if let Some(tracking) = self.tool_call_tracking.get_mut(call_id) {
            tracking.output_index = Some(output_index);
            tracking.item_id = Some(item_id.clone());
            tracking.open_emitted = true;
        }

        let event = self
            .emitter
            .emit_output_item_added(output_index, &initial_item);
        self.send(&event);

        if let Some(event) = self
            .emitter
            .emit_tool_call_in_progress(output_index, &item_id, family)
        {
            self.send(&event);
        }

        if presentation.has_searching_event() {
            if let Some(event) =
                self.emitter
                    .emit_tool_call_searching(output_index, &item_id, family)
            {
                self.send(&event);
            }
        }

        if presentation.streams_arguments() {
            if fragments.is_empty() && !full_args.is_empty() {
                if let Some(event) = self.emitter.emit_tool_call_arguments_delta(
                    output_index,
                    &item_id,
                    &full_args,
                    family,
                ) {
                    self.send(&event);
                }
            } else {
                for fragment in fragments {
                    if let Some(event) = self.emitter.emit_tool_call_arguments_delta(
                        output_index,
                        &item_id,
                        &fragment,
                        family,
                    ) {
                        self.send(&event);
                    }
                }
            }
            if let Some(event) = self.emitter.emit_tool_call_arguments_done(
                output_index,
                &item_id,
                &full_args,
                family,
            ) {
                self.send(&event);
            }
        }
    }

    /// Emit a streaming argument fragment for an in-flight tool call.
    /// Argument-streaming families (`mcp_call`, `function_call`)
    /// dispatch to their own delta event family; hosted builtins skip
    /// silently because their progress rides on structured events.
    /// Calls that were suppressed at `EmissionStarted` (approval-gated)
    /// have no tracking entry, so this is a no-op for them.
    fn emit_tool_call_arguments_fragment(&mut self, call_id: &str, fragment: &str) {
        if self.disconnected {
            return;
        }
        let Some(tracking) = self.tool_call_tracking.get_mut(call_id) else {
            return;
        };
        let family = tracking.presentation.family;
        if !matches!(family, OutputFamily::Function) {
            tracking.argument_fragments.push(fragment.to_string());
            return;
        }
        let (Some(output_index), Some(item_id)) = (tracking.output_index, tracking.item_id.clone())
        else {
            return;
        };
        if let Some(event) =
            self.emitter
                .emit_tool_call_arguments_delta(output_index, &item_id, fragment, family)
        {
            self.send(&event);
        }
    }

    /// Adapter signaled the tool call's arguments are done. For
    /// argument-streaming families: emit `*_arguments.done`. For
    /// caller `function_call` (no execute phase): also close the
    /// wire lifecycle with `output_item.done(status=completed)` and
    /// drop tracking. For gateway tools, leave tracking alive — the
    /// later `ToolCompleted` event finishes the lifecycle.
    fn emit_tool_call_emission_done(&mut self, call_id: &str, full_args: &str) {
        if self.disconnected {
            return;
        }
        // Snapshot what we need from tracking before any `send_event`
        // call mutably borrows `self`. We don't drop tracking here
        // unless this is a `Function` (caller fc, no execute phase).
        let (family, output_index, item_id, name) = {
            let Some(tracking) = self.tool_call_tracking.get_mut(call_id) else {
                return;
            };
            if !matches!(tracking.presentation.family, OutputFamily::Function) {
                tracking.full_args = Some(full_args.to_string());
                return;
            }
            let (Some(output_index), Some(item_id)) =
                (tracking.output_index, tracking.item_id.clone())
            else {
                return;
            };
            (
                tracking.presentation.family,
                output_index,
                item_id,
                tracking.name.clone(),
            )
        };

        if let Some(event) =
            self.emitter
                .emit_tool_call_arguments_done(output_index, &item_id, full_args, family)
        {
            self.send(&event);
        }

        if matches!(family, OutputFamily::Function) {
            // Caller fc has no execute phase. Build the final
            // `function_call` output item and close the lifecycle now.
            let item_done = json!({
                "id": item_id,
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": full_args,
                "status": "completed",
            });
            let event = self.emitter.emit_output_item_done(output_index, &item_done);
            self.send(&event);
            self.emitter.complete_output_item(output_index);
            self.tool_call_tracking.remove(call_id);
        }
    }

    /// Emit the open-half lifecycle for an approval-continuation call
    /// in one shot. The model is *not* invoked on a continuation turn
    /// — the driver primes `ExecuteTools` directly — so there is no
    /// live model-stream to pump `EmissionStarted/Fragment/Done`
    /// through. Sink expands [`LoopEvent::ApprovedToolReplay`] into:
    ///
    /// - `output_item.added(<family>)`
    /// - family `*.in_progress` (when one exists)
    /// - family `*_arguments.delta` + `*_arguments.done`
    ///   (for streaming families)
    ///
    /// Tracking is then registered so the matching `ToolCompleted`
    /// fires the close-half (`*_completed` / `*.failed` +
    /// `output_item.done`) normally.
    #[expect(
        clippy::too_many_arguments,
        reason = "callsite mirrors LoopEvent::ApprovedToolReplay payload 1:1; bundling into a struct here forces it to live on the LoopEvent borrow surface as well, which fights its design as a flat-borrow event."
    )]
    fn emit_approved_tool_replay(
        &mut self,
        call_id: &str,
        item_id_hint: &str,
        name: &str,
        full_args: &str,
        family: OutputFamily,
        server_label: &str,
        approval_request_id: Option<&str>,
    ) {
        if self.disconnected {
            return;
        }
        let presentation = ToolPresentation::from_family(family);
        let output_item_type =
            ResponseStreamEventEmitter::output_item_type_for_family(Some(family));
        let (output_index, allocated_item_id) =
            self.emitter.allocate_output_index(output_item_type);

        let call = LoopToolCall {
            call_id: call_id.to_string(),
            item_id: item_id_hint.to_string(),
            name: name.to_string(),
            arguments: full_args.to_string(),
            approval_request_id: approval_request_id.map(str::to_string),
        };
        let item = presentation.render_initial_item(&call, server_label, &allocated_item_id);
        let event = self.emitter.emit_output_item_added(output_index, &item);
        self.send(&event);

        if let Some(event) =
            self.emitter
                .emit_tool_call_in_progress(output_index, &allocated_item_id, family)
        {
            self.send(&event);
        }

        if presentation.streams_arguments() {
            if let Some(event) = self.emitter.emit_tool_call_arguments_delta(
                output_index,
                &allocated_item_id,
                full_args,
                family,
            ) {
                self.send(&event);
            }
            if let Some(event) = self.emitter.emit_tool_call_arguments_done(
                output_index,
                &allocated_item_id,
                full_args,
                family,
            ) {
                self.send(&event);
            }
        }

        self.tool_call_tracking.insert(
            call_id.to_string(),
            ToolCallTracking {
                output_index: Some(output_index),
                item_id: Some(allocated_item_id),
                presentation,
                name: name.to_string(),
                server_label: server_label.to_string(),
                approval_request_id: approval_request_id.map(str::to_string),
                argument_fragments: Vec::new(),
                full_args: Some(full_args.to_string()),
                open_emitted: true,
            },
        );
    }

    /// Emit the completion half of a gateway tool call lifecycle.
    /// Family-specific decisions (whether to emit `*.failed` vs
    /// `*.completed` on error, what shape the rendered output_item
    /// takes) all flow through [`ToolPresentation`] — the sink itself
    /// just dispatches.
    fn emit_tool_completed(&mut self, executed: &ExecutedCall) {
        if self.disconnected {
            return;
        }
        self.flush_tool_call_open_and_arguments(&executed.call_id, &executed.arguments);
        let Some(tracking) = self.tool_call_tracking.remove(&executed.call_id) else {
            // No matching `ToolCallStarted`. The driver always pairs
            // them so this only fires if the started event was
            // suppressed (e.g., approval interrupt before execution).
            // Drop silently rather than fabricate a half-lifecycle.
            return;
        };

        let family = tracking.presentation.family;
        let (Some(output_index), Some(item_id)) = (tracking.output_index, tracking.item_id) else {
            return;
        };
        // Families with a dedicated failure event (`mcp_call.failed`
        // today) emit it on `is_error`; the rest surface failure
        // inside the `*.completed` payload.
        if executed.is_error && tracking.presentation.has_failed_event() {
            let event =
                self.emitter
                    .emit_mcp_call_failed(output_index, &item_id, &executed.output_string);
            self.send(&event);
        } else if let Some(event) =
            self.emitter
                .emit_tool_call_completed(output_index, &item_id, family)
        {
            self.send(&event);
        }

        // Family-specific final item shape comes from the renderer.
        // It re-stamps `id` to the streaming-allocated item id and
        // forces `status: failed` / `error: <msg>` / `output: null`
        // on the error path so the wire-side `output_item.done`
        // payload never contradicts the preceding `*.failed` event.
        let mut item = tracking.presentation.render_final_item(executed, &item_id);

        // Backfill stream-time-known metadata onto the rendered item
        // when the executor's `transformed_item` left them blank.
        if matches!(family, OutputFamily::McpCall) {
            if let Some(obj) = item.as_object_mut() {
                if !tracking.server_label.is_empty()
                    && obj
                        .get("server_label")
                        .and_then(|v| v.as_str())
                        .is_none_or(|s| s.is_empty())
                {
                    obj.insert(
                        "server_label".to_string(),
                        Value::String(tracking.server_label.clone()),
                    );
                }
                if let Some(ref approval_id) = tracking.approval_request_id {
                    obj.insert(
                        "approval_request_id".to_string(),
                        Value::String(approval_id.clone()),
                    );
                }
            }
        }

        let event = self.emitter.emit_output_item_done(output_index, &item);
        self.send(&event);
        self.emitter.complete_output_item(output_index);
    }

    /// Emit the `mcp_approval_request` output item lifecycle for the
    /// single gated tool call the driver chose to surface this turn.
    /// Per the OpenAI Responses contract a turn surfaces one approval
    /// boundary at a time; remaining gated calls re-issue on
    /// continuation. The approval request has no `*.in_progress` /
    /// `*.completed` family events — open and close carry the same
    /// payload.
    fn emit_approval_requested(&mut self, pending: &PendingToolExecution) {
        if self.disconnected {
            return;
        }
        let (output_index, _allocated_item_id) = self
            .emitter
            .allocate_output_index(OutputItemType::McpApprovalRequest);
        // Use the canonical `mcpr_<call_id>` id that the driver's
        // NS-side `build_mcp_approval_request_items` uses
        // (`driver.rs:692`). The allocator-issued id would put
        // streaming on a different wire id from the persisted /
        // history form, so a continuation that copies the streamed
        // id and hits the chain on a later turn could orphan-reject.
        let item_id = format!("mcpr_{}", pending.call.call_id);

        let item = json!({
            "id": item_id,
            "type": "mcp_approval_request",
            "server_label": pending.server_label,
            "name": pending.call.name,
            "arguments": pending.call.arguments,
        });
        let event = self.emitter.emit_output_item_added(output_index, &item);
        self.send(&event);
        let event = self.emitter.emit_output_item_done(output_index, &item);
        self.send(&event);
        self.emitter.complete_output_item(output_index);
    }
}

impl StreamSink for GrpcResponseStreamSink {
    fn emit(&mut self, event: LoopEvent<'_>) {
        match event {
            LoopEvent::ResponseStarted => self.emit_response_started(),
            LoopEvent::ResponseFinished => self.emit_response_completed(None),
            LoopEvent::ResponseIncomplete { reason } => {
                self.emit_response_completed(Some(reason));
            }
            LoopEvent::McpListToolsItem { item } => self.emit_mcp_list_tools(item),
            LoopEvent::ToolCallEmissionStarted {
                call_id,
                item_id,
                name,
            } => self.emit_tool_call_emission_started(call_id, item_id, name),
            LoopEvent::ToolCallArgumentsFragment { call_id, fragment } => {
                self.emit_tool_call_arguments_fragment(call_id, fragment);
            }
            LoopEvent::ToolCallEmissionDone { call_id, full_args } => {
                self.emit_tool_call_emission_done(call_id, full_args);
            }
            LoopEvent::ToolCallExecutionStarted { call_id, full_args } => {
                self.flush_tool_call_open_and_arguments(call_id, full_args);
            }
            LoopEvent::ToolCompleted { executed } => self.emit_tool_completed(executed),
            LoopEvent::ApprovedToolReplay {
                call_id,
                item_id,
                name,
                full_args,
                family,
                server_label,
                approval_request_id,
            } => self.emit_approved_tool_replay(
                call_id,
                item_id,
                name,
                full_args,
                family,
                server_label,
                approval_request_id,
            ),
            LoopEvent::ApprovalRequested { pending } => self.emit_approval_requested(pending),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Wire-level sink behaviour for contracts that are cheaper to
    //! verify here than through integration / e2e suites.
    use std::collections::HashMap;

    use serde_json::Value;
    use tokio::sync::mpsc;

    use super::*;
    use crate::routers::common::{
        agent_loop::{
            state::{LoopToolCall, PendingToolExecution},
            OutputFamily, ToolTransferDescriptor,
        },
        responses_streaming::ResponseStreamEventEmitter,
    };

    /// Build a sink + receiver pair primed with common transfer
    /// descriptors derived from the supplied family / approval maps.
    /// Pulls a `mpsc::unbounded_channel` for SSE events; tests
    /// `drain_events` it after every emit batch and match against the
    /// parsed JSON payloads.
    fn make_sink(
        families: HashMap<String, OutputFamily>,
        requires_approval: HashMap<String, bool>,
    ) -> (
        GrpcResponseStreamSink,
        mpsc::UnboundedReceiver<Result<Bytes, io::Error>>,
    ) {
        make_sink_with_labels(families, requires_approval, HashMap::new())
    }

    fn make_sink_with_labels(
        families: HashMap<String, OutputFamily>,
        requires_approval: HashMap<String, bool>,
        labels: HashMap<String, String>,
    ) -> (
        GrpcResponseStreamSink,
        mpsc::UnboundedReceiver<Result<Bytes, io::Error>>,
    ) {
        let (tx, rx) = mpsc::unbounded_channel();
        let emitter =
            ResponseStreamEventEmitter::new("resp_test".to_string(), "model_x".to_string(), 0);
        let transfers = families
            .into_iter()
            .map(|(name, family)| {
                let requires = requires_approval.get(&name).copied().unwrap_or(false);
                (
                    name,
                    ToolTransferDescriptor::from_family_and_approval(family, requires),
                )
            })
            .collect();
        let sink = GrpcResponseStreamSink::new(emitter, tx, transfers, labels);
        (sink, rx)
    }

    /// Pull every event already in the receiver, parse the SSE payload
    /// out of each chunk's `data: ...` line, and return one Value per
    /// chunk. SSE error events (no `data:`) are skipped.
    fn drain_events(rx: &mut mpsc::UnboundedReceiver<Result<Bytes, io::Error>>) -> Vec<Value> {
        let mut out = Vec::new();
        while let Ok(Some(Ok(bytes))) = tokio::runtime::Handle::try_current()
            .map(|_| ())
            .map(|()| rx.try_recv().map(Some))
            .unwrap_or_else(|_| rx.try_recv().map(Some))
        {
            let s = String::from_utf8_lossy(&bytes).into_owned();
            for line in s.lines() {
                if let Some(payload) = line.strip_prefix("data: ") {
                    if let Ok(v) = serde_json::from_str::<Value>(payload) {
                        out.push(v);
                    }
                }
            }
        }
        out
    }

    #[test]
    fn incomplete_finish_emits_response_incomplete() {
        let (mut sink, mut rx) = make_sink(HashMap::new(), HashMap::new());

        sink.emit(LoopEvent::ResponseIncomplete {
            reason: "max_output_tokens",
        });

        let events = drain_events(&mut rx);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["type"], "response.incomplete");
        assert_eq!(events[0]["response"]["status"], "incomplete");
        assert_eq!(
            events[0]["response"]["incomplete_details"]["reason"],
            "max_output_tokens"
        );
    }

    #[test]
    fn streamed_mcp_list_tools_items_omit_status() {
        let (mut sink, mut rx) = make_sink(HashMap::new(), HashMap::new());
        let item = ResponseOutputItem::McpListTools {
            id: "mcpl_test".to_string(),
            server_label: "deepwiki".to_string(),
            tools: vec![],
            error: None,
        };

        sink.emit(LoopEvent::McpListToolsItem { item: &item });

        let events = drain_events(&mut rx);
        let added = events
            .iter()
            .find(|event| event["type"] == "response.output_item.added")
            .expect("mcp_list_tools added event");
        assert!(added["item"].get("status").is_none());
        let done = events
            .iter()
            .find(|event| event["type"] == "response.output_item.done")
            .expect("mcp_list_tools done event");
        assert!(done["item"].get("status").is_none());
    }

    /// Caller-declared function tools stream through
    /// `response.function_call_arguments.*`, not `response.mcp_call_arguments.*`.
    #[test]
    fn caller_fc_streams_function_call_arguments() {
        let (mut sink, mut rx) = make_sink(HashMap::new(), HashMap::new());
        sink.emit(LoopEvent::ToolCallEmissionStarted {
            call_id: "call_user_1",
            item_id: "fc_1",
            name: "get_weather",
        });
        sink.emit(LoopEvent::ToolCallArgumentsFragment {
            call_id: "call_user_1",
            fragment: "{\"city\":",
        });
        sink.emit(LoopEvent::ToolCallEmissionDone {
            call_id: "call_user_1",
            full_args: "{\"city\":\"sf\"}",
        });

        let events = drain_events(&mut rx);
        let types: Vec<&str> = events
            .iter()
            .filter_map(|e| e.get("type").and_then(|t| t.as_str()))
            .collect();
        assert!(
            types.contains(&"response.function_call_arguments.delta"),
            "expected function_call_arguments.delta, got {types:?}"
        );
        assert!(
            types.contains(&"response.function_call_arguments.done"),
            "expected function_call_arguments.done, got {types:?}"
        );
        assert!(
            !types.iter().any(|t| t.starts_with("response.mcp_call")),
            "must not emit mcp_call.* events for caller fc, got {types:?}"
        );
    }

    #[test]
    fn mcp_call_flushes_on_execution_started_with_server_label() {
        let mut families = HashMap::new();
        families.insert("ask_question".to_string(), OutputFamily::McpCall);
        let mut labels = HashMap::new();
        labels.insert("ask_question".to_string(), "deepwiki".to_string());
        let (mut sink, mut rx) = make_sink_with_labels(families, HashMap::new(), labels);

        sink.emit(LoopEvent::ToolCallEmissionStarted {
            call_id: "call_mcp_1",
            item_id: "fc_mcp_1",
            name: "ask_question",
        });
        sink.emit(LoopEvent::ToolCallArgumentsFragment {
            call_id: "call_mcp_1",
            fragment: "{\"q\":",
        });
        sink.emit(LoopEvent::ToolCallArgumentsFragment {
            call_id: "call_mcp_1",
            fragment: "\"hi\"}",
        });
        sink.emit(LoopEvent::ToolCallEmissionDone {
            call_id: "call_mcp_1",
            full_args: "{\"q\":\"hi\"}",
        });

        let before_execution = drain_events(&mut rx);
        assert!(
            before_execution.is_empty(),
            "MCP call must stay buffered until execution starts: {before_execution:?}"
        );

        sink.emit(LoopEvent::ToolCallExecutionStarted {
            call_id: "call_mcp_1",
            full_args: "{\"q\":\"hi\"}",
        });

        let events = drain_events(&mut rx);
        let types: Vec<&str> = events
            .iter()
            .filter_map(|e| e.get("type").and_then(|t| t.as_str()))
            .collect();
        assert_eq!(
            types,
            vec![
                "response.output_item.added",
                "response.mcp_call.in_progress",
                "response.mcp_call_arguments.delta",
                "response.mcp_call_arguments.delta",
                "response.mcp_call_arguments.done",
            ]
        );
        assert_eq!(events[0]["item"]["server_label"], "deepwiki");
        assert_eq!(events[2]["delta"], "{\"q\":");
        assert_eq!(events[3]["delta"], "\"hi\"}");
    }

    #[test]
    fn skipped_mcp_call_does_not_consume_output_index() {
        let mut families = HashMap::new();
        families.insert("ask_question".to_string(), OutputFamily::McpCall);
        let mut labels = HashMap::new();
        labels.insert("ask_question".to_string(), "deepwiki".to_string());
        let (mut sink, mut rx) = make_sink_with_labels(families, HashMap::new(), labels);

        sink.emit(LoopEvent::ToolCallEmissionStarted {
            call_id: "call_skipped",
            item_id: "fc_skipped",
            name: "ask_question",
        });
        sink.emit(LoopEvent::ToolCallArgumentsFragment {
            call_id: "call_skipped",
            fragment: "{\"q\":\"skip\"}",
        });
        sink.emit(LoopEvent::ToolCallEmissionDone {
            call_id: "call_skipped",
            full_args: "{\"q\":\"skip\"}",
        });

        assert!(
            drain_events(&mut rx).is_empty(),
            "buffered gateway calls must not emit or allocate until execution starts"
        );

        sink.emit(LoopEvent::ToolCallEmissionStarted {
            call_id: "call_executed",
            item_id: "fc_executed",
            name: "ask_question",
        });
        sink.emit(LoopEvent::ToolCallArgumentsFragment {
            call_id: "call_executed",
            fragment: "{\"q\":\"run\"}",
        });
        sink.emit(LoopEvent::ToolCallEmissionDone {
            call_id: "call_executed",
            full_args: "{\"q\":\"run\"}",
        });
        sink.emit(LoopEvent::ToolCallExecutionStarted {
            call_id: "call_executed",
            full_args: "{\"q\":\"run\"}",
        });

        let events = drain_events(&mut rx);
        assert_eq!(events[0]["type"], "response.output_item.added");
        assert_eq!(events[0]["output_index"], 0);
    }

    /// Approval-gated MCP tools must not consume a streaming `output_index`.
    #[test]
    fn approval_gated_calls_do_not_allocate_output_index() {
        let mut families = HashMap::new();
        families.insert("ask_question".to_string(), OutputFamily::McpCall);
        let mut gate = HashMap::new();
        gate.insert("ask_question".to_string(), true);
        let (mut sink, mut rx) = make_sink(families, gate);

        // Drive the streamed lifecycle for a gated tool — sink must
        // suppress every event.
        sink.emit(LoopEvent::ToolCallEmissionStarted {
            call_id: "call_a",
            item_id: "fc_a",
            name: "ask_question",
        });
        sink.emit(LoopEvent::ToolCallArgumentsFragment {
            call_id: "call_a",
            fragment: "{\"q\":\"hi\"}",
        });
        sink.emit(LoopEvent::ToolCallEmissionDone {
            call_id: "call_a",
            full_args: "{\"q\":\"hi\"}",
        });
        let suppressed_events = drain_events(&mut rx);
        assert!(
            suppressed_events.is_empty(),
            "gated call must not emit any wire events, got {suppressed_events:?}"
        );

        // Driver surfaces the gated call as a single approval request.
        let pending = PendingToolExecution {
            call: LoopToolCall {
                call_id: "call_a".to_string(),
                item_id: "fc_a".to_string(),
                name: "ask_question".to_string(),
                arguments: "{\"q\":\"hi\"}".to_string(),
                approval_request_id: None,
            },
            server_label: "deepwiki".to_string(),
        };
        sink.emit(LoopEvent::ApprovalRequested { pending: &pending });

        let approval_events = drain_events(&mut rx);
        let added = approval_events
            .iter()
            .find(|e| e.get("type").and_then(|t| t.as_str()) == Some("response.output_item.added"))
            .expect("approval emission must include output_item.added");
        assert_eq!(
            added["output_index"].as_u64(),
            Some(0),
            "approval request must take output_index=0 (got {})",
            added["output_index"]
        );
        assert_eq!(
            added["item"]["type"], "mcp_approval_request",
            "first visible item must be the approval request, not a phantom mcp_call"
        );
    }

    /// `ApprovedToolReplay` preserves the original `mcpr_*` id on the
    /// streamed `mcp_call.approval_request_id`.
    #[test]
    fn approved_replay_stamps_approval_request_id() {
        let mut families = HashMap::new();
        families.insert("ask_question".to_string(), OutputFamily::McpCall);
        let (mut sink, mut rx) = make_sink(families, HashMap::new());

        sink.emit(LoopEvent::ApprovedToolReplay {
            call_id: "call_a",
            item_id: "fc_a",
            name: "ask_question",
            full_args: "{\"q\":\"hi\"}",
            family: OutputFamily::McpCall,
            server_label: "deepwiki",
            approval_request_id: Some("mcpr_call_a"),
        });

        let events = drain_events(&mut rx);
        let added = events
            .iter()
            .find(|e| e.get("type").and_then(|t| t.as_str()) == Some("response.output_item.added"))
            .expect("replay must include output_item.added");
        assert_eq!(added["item"]["type"], "mcp_call");
        assert_eq!(added["item"]["server_label"], "deepwiki");
        assert_eq!(added["item"]["approval_request_id"], "mcpr_call_a");
    }

    /// Subsequent `ToolCompleted` for the replayed call must close
    /// the lifecycle (`mcp_call.completed` + `output_item.done`)
    /// with `approval_request_id` still echoed onto the final item.
    #[test]
    fn approved_replay_completion_keeps_approval_request_id() {
        let mut families = HashMap::new();
        families.insert("ask_question".to_string(), OutputFamily::McpCall);
        let (mut sink, mut rx) = make_sink(families, HashMap::new());

        sink.emit(LoopEvent::ApprovedToolReplay {
            call_id: "call_a",
            item_id: "fc_a",
            name: "ask_question",
            full_args: "{\"q\":\"hi\"}",
            family: OutputFamily::McpCall,
            server_label: "deepwiki",
            approval_request_id: Some("mcpr_call_a"),
        });
        let _ = drain_events(&mut rx);

        let executed = ExecutedCall {
            call_id: "call_a".to_string(),
            item_id: "fc_a".to_string(),
            name: "ask_question".to_string(),
            arguments: "{\"q\":\"hi\"}".to_string(),
            output_string: "ok".to_string(),
            transformed_item: None,
            is_error: false,
            approval_request_id: Some("mcpr_call_a".to_string()),
        };
        sink.emit(LoopEvent::ToolCompleted {
            executed: &executed,
        });

        let events = drain_events(&mut rx);
        let types: Vec<&str> = events
            .iter()
            .filter_map(|e| e.get("type").and_then(|t| t.as_str()))
            .collect();
        assert!(
            types.contains(&"response.mcp_call.completed"),
            "completed event missing: {types:?}"
        );
        let done = events
            .iter()
            .find(|e| e.get("type").and_then(|t| t.as_str()) == Some("response.output_item.done"))
            .expect("output_item.done missing");
        assert_eq!(done["item"]["approval_request_id"], "mcpr_call_a");
    }

    /// `process_chat_chunk`-driven caller fc completes with
    /// `output_item.done(status=completed)` and emits
    /// `function_call_arguments.done`, never the mcp_call counterpart.
    /// Sanity check that the chat-stream path goes through the same
    /// family-aware dispatch as the harmony path tested above.
    #[test]
    fn chat_stream_caller_fc_emits_function_call_args_done() {
        use openai_protocol::{
            chat::{ChatCompletionStreamResponse, ChatMessageDelta, ChatStreamChoice},
            common::{FunctionCallDelta, ToolCallDelta},
        };

        let (mut sink, mut rx) = make_sink(HashMap::new(), HashMap::new());

        let first = ChatCompletionStreamResponse {
            id: String::from("c1"),
            object: String::from("chat.completion.chunk"),
            created: 0,
            model: String::from("m"),
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: ChatMessageDelta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![ToolCallDelta {
                        index: 0,
                        id: Some("call_x".to_string()),
                        tool_type: Some("function".to_string()),
                        function: Some(FunctionCallDelta {
                            name: Some("get_weather".to_string()),
                            arguments: Some("{\"city\":".to_string()),
                        }),
                    }]),
                },
                finish_reason: None,
                logprobs: None,
                matched_stop: None,
            }],
            usage: None,
            system_fingerprint: None,
        };
        let last = ChatCompletionStreamResponse {
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: ChatMessageDelta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![ToolCallDelta {
                        index: 0,
                        id: None,
                        tool_type: None,
                        function: Some(FunctionCallDelta {
                            name: None,
                            arguments: Some("\"sf\"}".to_string()),
                        }),
                    }]),
                },
                finish_reason: Some("tool_calls".to_string()),
                logprobs: None,
                matched_stop: None,
            }],
            ..first.clone()
        };
        sink.process_chat_chunk(&first);
        sink.process_chat_chunk(&last);

        let events = drain_events(&mut rx);
        let types: Vec<&str> = events
            .iter()
            .filter_map(|e| e.get("type").and_then(|t| t.as_str()))
            .collect();
        assert!(
            types.contains(&"response.function_call_arguments.delta"),
            "missing function_call_arguments.delta in {types:?}"
        );
        assert!(
            types.contains(&"response.function_call_arguments.done"),
            "missing function_call_arguments.done in {types:?}"
        );
        let done = events
            .iter()
            .find(|e| e.get("type").and_then(|t| t.as_str()) == Some("response.output_item.done"))
            .expect("function_call must close with output_item.done");
        assert_eq!(done["item"]["type"], "function_call");
        assert_eq!(done["item"]["status"], "completed");
        assert!(
            !types.iter().any(|t| t.starts_with("response.mcp_call")),
            "caller fc must not emit mcp_call.* events: {types:?}"
        );
    }
}
