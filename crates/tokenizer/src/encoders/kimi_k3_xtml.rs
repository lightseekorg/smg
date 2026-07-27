//! Kimi-K3 XTML chat-template renderer.
//!
//! Ported from the upstream Python reference `encoding_k3.py::build_chat_segments`
//! (entry point). Unlike the Kimi-K2.5 encoder, Kimi-K3 ships **no Jinja chat
//! template**: its prompt is rendered entirely in Python into XTML using the
//! `<|open|>` / `<|close|>` / `<|sep|>` / `<|end_of_msg|>` control tokens. This
//! module reproduces that rendering so SMG's gRPC path can build the prompt
//! itself, dispatched via `Renderer::KimiK3Xtml`.
//!
//! # Encoding fidelity limitation
//!
//! The Python reference distinguishes *structural markers* (encoded as tiktoken
//! special-token IDs) from *user/tool text* (encoded as ordinary BPE) by
//! emitting a list of `EncodeSegment { text, allow_special }`. This port instead
//! concatenates every segment's `text` into a flat `String`. A later single
//! `encode` therefore treats all `<|...|>` occurrences uniformly — a known
//! limitation for user/tool text that literally contains control-token strings
//! (e.g. a user typing `<|open|>`). Matching the flat-`String` contract of the
//! sibling renderers (`kimi_k25_tools`, `deepseek_v32`) is intentional here;
//! segment-aware encoding is out of scope.
//!
//! HTML-escaping the control tokens in user/tool text is **not** the fix: the
//! reference emits that text as ordinary BPE without escaping, so the model was
//! trained on the raw bytes; rewriting `<|open|>` to `&lt;|open|&gt;` (or
//! similar) would both feed the model text it never saw and corrupt legitimate
//! content. The correct fix is to carry the reference's per-segment
//! `allow_special` flag through to the tokenizer so structural markers keep
//! their special-token ids while surrounding text is encoded as ordinary BPE —
//! a cross-cutting change across this renderer family, tracked as future work.
//!
//! # Deferred (TODO)
//!
//! The following `build_chat_segments` branches are intentionally not ported yet
//! (not exercised by the current gRPC path / golden fixtures):
//! - multimodal `image_prompts` substitution (image parts render as the bare
//!   `<|kimi_image_placeholder|>` marker; a separate multimodal task will wire
//!   real image prompts through).

use anyhow::{anyhow, Result};
use serde_json::{Map, Value};

use crate::chat_template::ChatTemplateParams;

const OPEN_TOKEN: &str = "<|open|>";
const CLOSE_TOKEN: &str = "<|close|>";
const SEP_TOKEN: &str = "<|sep|>";
const END_OF_MSG_TOKEN: &str = "<|end_of_msg|>";
const IMAGE_PLACEHOLDER: &str = "<|kimi_image_placeholder|>";

/// Render a Kimi-K3 chat prompt to an XTML `String`.
///
/// `params.tools` (when non-empty) produces the leading `tool-declare` system
/// message. `params.add_generation_prompt` appends the assistant generation
/// prompt tail. Thinking mode is resolved from `template_kwargs["thinking"]`
/// then `params.thinking`, defaulting to `true` to match the Python
/// `build_chat_segments(thinking=True)` default; it selects `think` vs
/// `response` for both the generation-prompt tail and (per-turn) any prior
/// assistant reasoning.
pub fn apply_kimi_k3_xtml(messages: &[Value], params: &ChatTemplateParams) -> Result<String> {
    // Re-sort tool results by tool_call_id, then normalize each message
    // (deep-sort tool schemas, coerce tool-call arguments) — both side-effect
    // free, mirroring the Python entry point.
    let reordered = normalize_xtml_tool_result_messages(messages);
    let normalized: Vec<Value> = reordered.iter().map(normalize_message).collect();

    // Top-level tools declaration is deep-sorted before compact JSON encoding.
    let tools_sorted: Option<Value> = params
        .tools
        .filter(|t| !t.is_empty())
        .map(|t| deep_sort(&Value::Array(t.to_vec())));

    let thinking = params
        .template_kwargs
        .and_then(|k| k.get("thinking"))
        .and_then(Value::as_bool)
        .or(params.thinking)
        .unwrap_or(true);

    let mut out = String::new();

    if let Some(tools) = &tools_sorted {
        push_tool_declare(&mut out, tools, false)?;
    }

    // Effort directive (`thinking-effort` internal system message). Only emitted
    // while thinking is on, mirroring the Python reference (both the validation
    // and the emit are gated on `thinking`).
    //
    // Precedence:
    //   1. An explicit `thinking_effort` (via `chat_template_kwargs`) wins and is
    //      validated strictly — a provided-but-unsupported value is a hard error,
    //      matching the reference `assert thinking_effort in _VALID_THINKING_EFFORTS`.
    //   2. Otherwise the OpenAI-standard top-level `reasoning_effort` level is
    //      used, but only when it names a supported K3 effort (`low`/`high`/`max`).
    //      `minimal`/`none` already switch thinking off upstream and `medium` has
    //      no K3 equivalent, so such values emit no directive rather than erroring
    //      on an otherwise-valid OpenAI field.
    //
    // Absent both, no directive is emitted and the model applies its intrinsic
    // `max` default — matching the reference, which injects nothing when
    // `thinking_effort` is unset. `preserve_thinking` (which some callers send
    // alongside the effort) is not read by the reference renderer and is ignored.
    if thinking {
        if let Some(effort_val) = params
            .template_kwargs
            .and_then(|k| k.get("thinking_effort"))
            .filter(|v| !v.is_null())
        {
            match effort_val.as_str() {
                Some(effort @ ("low" | "high" | "max")) => {
                    push_thinking_effort(&mut out, effort);
                }
                _ => {
                    return Err(anyhow!(
                        "Unsupported thinking_effort={effort_val}; \
                         supported values are [\"high\", \"low\", \"max\"]."
                    ));
                }
            }
        } else if let Some(effort) = params
            .template_kwargs
            .and_then(|k| k.get("reasoning_effort"))
            .and_then(Value::as_str)
            .filter(|e| matches!(*e, "low" | "high" | "max"))
        {
            push_thinking_effort(&mut out, effort);
        }
    }

    // Tracks the most recent assistant `tool_calls` so tool messages can resolve
    // a missing name by position, mirroring the Python module-local state.
    let mut current_tool_calls: Option<&Vec<Value>> = None;
    let mut tool_index: usize = 0;

    for message in &normalized {
        let obj = match message.as_object() {
            Some(o) => o,
            None => continue,
        };
        let role = obj.get("role").and_then(Value::as_str).unwrap_or("");
        match role {
            "user" => {
                let mut attrs = vec![("role", "user".to_string())];
                if let Some(name) = nonempty_name(obj) {
                    attrs.push(("name", name));
                }
                push_open_tag(&mut out, "message", &attrs);
                push_content(&mut out, obj.get("content"));
                push_close_tag(&mut out, "message");
                push_end_of_msg(&mut out);
            }
            "system" => {
                if let Some(tools) = obj
                    .get("tools")
                    .filter(|t| t.as_array().is_some_and(|a| !a.is_empty()))
                {
                    // Dynamic tool declaration; already deep-sorted by
                    // `normalize_message`.
                    push_tool_declare(&mut out, tools, true)?;
                } else {
                    let mut attrs = vec![("role", "system".to_string())];
                    if let Some(name) = nonempty_name(obj) {
                        attrs.push(("name", name));
                    }
                    push_open_tag(&mut out, "message", &attrs);
                    push_content(&mut out, obj.get("content"));
                    push_close_tag(&mut out, "message");
                    push_end_of_msg(&mut out);
                }
            }
            "tool" => {
                tool_index += 1;
                let mut tool_name = obj
                    .get("tool")
                    .and_then(Value::as_str)
                    .or_else(|| obj.get("name").and_then(Value::as_str))
                    .map(str::to_string);
                if tool_name.is_none() {
                    if let Some(tcs) = current_tool_calls {
                        if tool_index <= tcs.len() {
                            let tc = &tcs[tool_index - 1];
                            let fnobj = function_or_self(tc);
                            tool_name = fnobj
                                .get("name")
                                .and_then(Value::as_str)
                                .map(str::to_string);
                        }
                    }
                }
                let tool_name = tool_name.ok_or_else(|| {
                    anyhow!(
                        "Kimi K3 tool messages need a resolvable tool name: carry `tool`/`name`, \
                         or match a preceding assistant tool_call by order."
                    )
                })?;
                push_open_tag(
                    &mut out,
                    "message",
                    &[
                        ("role", "tool".to_string()),
                        ("tool", tool_name),
                        ("index", tool_index.to_string()),
                    ],
                );
                push_content(&mut out, obj.get("content"));
                push_close_tag(&mut out, "message");
                push_end_of_msg(&mut out);
            }
            "assistant" => {
                current_tool_calls = obj.get("tool_calls").and_then(Value::as_array);
                tool_index = 0;
                let mut attrs = vec![("role", "assistant".to_string())];
                if let Some(name) = nonempty_name(obj) {
                    attrs.push(("name", name));
                }
                push_open_tag(&mut out, "message", &attrs);
                render_assistant_segments(&mut out, obj, thinking)?;
                push_close_tag(&mut out, "message");
                push_end_of_msg(&mut out);
            }
            // Unknown roles produce no output, matching the Python loop's lack
            // of a matching branch.
            _ => {}
        }
    }

    // Post-loop `tool_choice` / `response_format` internal system messages,
    // emitted after the conversation and before the generation-prompt tail to
    // mirror `build_chat_segments`.
    match params
        .template_kwargs
        .and_then(|kwargs| kwargs.get("tool_choice"))
        .and_then(Value::as_str)
    {
        Some("required") => push_internal_system_message(
            &mut out,
            "tool-choice",
            "The system is invoked with `tool_choice=required`.\n\
             You MUST call tools in the next message.",
        ),
        Some("none") => push_internal_system_message(
            &mut out,
            "tool-choice",
            "The system is invoked with `tool_choice=none`.\n\
             You MUST NOT call any tools in the next message.",
        ),
        _ => {}
    }

    if let Some(response_format) = params
        .template_kwargs
        .and_then(|kwargs| kwargs.get("response_format"))
    {
        push_response_format(&mut out, response_format, params.template_kwargs)?;
    }

    if params.add_generation_prompt {
        push_open_tag(&mut out, "message", &[("role", "assistant".to_string())]);
        push_open_tag(&mut out, if thinking { "think" } else { "response" }, &[]);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Rendering helpers (segment text emitted straight into the output String)
// ---------------------------------------------------------------------------

fn escape_attr_value(value: &str) -> String {
    value.replace('&', "&amp;").replace('"', "&quot;")
}

fn push_attr(out: &mut String, key: &str, value: &str) {
    out.push(' ');
    out.push_str(key);
    out.push_str("=\"");
    out.push_str(&escape_attr_value(value));
    out.push('"');
}

fn push_open_tag(out: &mut String, tag: &str, attrs: &[(&str, String)]) {
    out.push_str(OPEN_TOKEN);
    out.push_str(tag);
    for (key, value) in attrs {
        push_attr(out, key, value);
    }
    out.push_str(SEP_TOKEN);
}

fn push_close_tag(out: &mut String, tag: &str) {
    out.push_str(CLOSE_TOKEN);
    out.push_str(tag);
    out.push_str(SEP_TOKEN);
}

fn push_end_of_msg(out: &mut String) {
    out.push_str(END_OF_MSG_TOKEN);
}

/// Emit an internal `role="system"` message with the given `type` and a
/// (stripped) body, mirroring the Python `_internal_system_message` helper.
fn push_internal_system_message(out: &mut String, message_type: &str, body: &str) {
    push_open_tag(
        out,
        "message",
        &[
            ("role", "system".to_string()),
            ("type", message_type.to_string()),
        ],
    );
    out.push_str(body.trim());
    push_close_tag(out, "message");
    push_end_of_msg(out);
}

/// Render `content` (string, or an OpenAI content-part array) into `out`.
///
/// Image parts emit the bare `<|kimi_image_placeholder|>` marker; real
/// `image_prompts` substitution is deferred (see module docs).
fn push_content(out: &mut String, content: Option<&Value>) {
    match content {
        Some(Value::String(s)) => out.push_str(s),
        Some(Value::Array(parts)) => {
            for part in parts {
                let ty = part.get("type").and_then(Value::as_str);
                if matches!(ty, Some("image") | Some("image_url")) {
                    out.push_str(IMAGE_PLACEHOLDER);
                } else if let Some(text) = part.get("text").and_then(Value::as_str) {
                    out.push_str(text);
                }
            }
        }
        _ => {}
    }
}

fn push_tool_declare(out: &mut String, tools: &Value, dynamic: bool) -> Result<()> {
    let compact = json_compact(tools)?;
    let body = if dynamic {
        format!(
            "## New Tools Available\n\
             The system dynamically extends the toolset via lazy-loading.\n\
             You have access to all existing and extended tools.\n\
             Here are the specs for the extended tools.\n\n\
             ```json\n{compact}\n```"
        )
    } else {
        format!(
            "# Tools\n\
             Here are the available tools, described in JSONSchema.\n\n\
             ```json\n{compact}\n```"
        )
    };
    push_open_tag(
        out,
        "message",
        &[
            ("role", "system".to_string()),
            ("type", "tool-declare".to_string()),
        ],
    );
    out.push_str(&body);
    push_close_tag(out, "message");
    push_end_of_msg(out);
    Ok(())
}

/// Emit the `thinking-effort` internal system message.
///
/// Mirrors the Python reference `_internal_system_message("thinking-effort", …)`:
/// a `role="system" type="thinking-effort"` message whose (stripped) body states
/// the requested effort. `effort` is pre-validated to `low`/`high`/`max`; the
/// body text intentionally reproduces the reference verbatim (including its
/// mention of `medium`, which the validation set does not accept).
fn push_thinking_effort(out: &mut String, effort: &str) {
    let body = format!(
        concat!(
            "`thinking_effort` guides on how much to think in your ",
            "thinking channel (not including the response channel), ",
            "supported values include `low`, `medium`, `high`, and `max`.\n",
            "Now the system is invoked with `thinking_effort={effort}`.",
        ),
        effort = effort,
    );
    push_internal_system_message(out, "thinking-effort", &body);
}

/// Emit the `response-format` internal system message for `json_object` /
/// `json_schema`, mirroring the Python reference. The schema is resolved from an
/// explicit `response_schema` template kwarg, falling back to
/// `response_format.json_schema.schema`; it is deep-sorted then compacted so the
/// output is stable and byte-identical to the reference renderer.
fn push_response_format(
    out: &mut String,
    response_format: &Value,
    template_kwargs: Option<&std::collections::HashMap<String, Value>>,
) -> Result<()> {
    let response_type = response_format
        .get("type")
        .and_then(Value::as_str)
        .or_else(|| response_format.as_str());
    match response_type {
        Some("json_object") => push_internal_system_message(
            out,
            "response-format",
            "The system is invoked with `response_format=json_object`.\n\
             Your response must be raw JSON data without markdown code \
             blocks (```json) or any additional formatting.",
        ),
        Some("json_schema") => {
            let response_schema = template_kwargs
                .and_then(|kwargs| kwargs.get("response_schema"))
                .or_else(|| {
                    response_format
                        .get("json_schema")
                        .and_then(|schema| schema.get("schema"))
                });
            let schema = response_schema
                .map(deep_sort)
                .map(|schema| json_compact(&schema))
                .transpose()?
                .unwrap_or_else(|| "null".to_string());
            let body = format!(
                "The system is invoked with `response_format=json_schema`.\n\
                 Your response must be raw JSON data without markdown code \
                 blocks (```json) or any additional formatting.\n\
                 The JSON data must match the following schema:\n\
                 ```json\n{schema}\n```"
            );
            push_internal_system_message(out, "response-format", &body);
        }
        _ => {}
    }
    Ok(())
}

fn render_assistant_segments(
    out: &mut String,
    msg: &Map<String, Value>,
    thinking: bool,
) -> Result<()> {
    // The `<think>` channel is structural: in thinking mode every assistant
    // message carries the open/close tags even when there is no reasoning content
    // to fill in. In non-thinking mode the channel is dropped entirely. Mirrors
    // the Python `_render_assistant_segments(..., thinking)`.
    if thinking {
        // `reasoning_content or reasoning` — Python truthiness picks the first
        // non-falsy value, falling back to `reasoning` otherwise.
        let reasoning = msg
            .get("reasoning_content")
            .filter(|v| is_truthy(v))
            .or_else(|| msg.get("reasoning"));
        push_open_tag(out, "think", &[]);
        if let Some(rc) = reasoning {
            let rc_str = plain_string(rc);
            if !rc_str.trim().is_empty() {
                out.push_str(&rc_str);
            }
        }
        push_close_tag(out, "think");
    }

    push_open_tag(out, "response", &[]);
    push_content(out, msg.get("content"));
    push_close_tag(out, "response");

    if let Some(tool_calls) = msg
        .get("tool_calls")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
    {
        push_open_tag(out, "tools", &[]);
        for (idx, tool_call) in tool_calls.iter().enumerate() {
            let index = idx + 1;
            let fnobj = function_or_self(tool_call);
            let name = fnobj
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("Kimi K3 tool call is missing a function name"))?;
            push_open_tag(
                out,
                "call",
                &[("tool", name.to_string()), ("index", index.to_string())],
            );
            let json_block = fnobj.get("_xtml_json_block").and_then(Value::as_str);
            if let Some(block) = json_block {
                push_open_tag(out, "json", &[("type", "object".to_string())]);
                out.push_str(block);
                push_close_tag(out, "json");
            } else if let Some(args) = fnobj.get("arguments").and_then(Value::as_object) {
                for (key, value) in args {
                    push_open_tag(
                        out,
                        "argument",
                        &[("key", key.clone()), ("type", xtml_type(value))],
                    );
                    out.push_str(&xtml_value(value));
                    push_close_tag(out, "argument");
                }
            }
            push_close_tag(out, "call");
        }
        push_close_tag(out, "tools");
    }

    Ok(())
}

/// `tool_call.get("function", tool_call)`: use the `function` object when
/// present, otherwise treat the tool-call object itself as the function shape
/// (arguments are attached at the top level in that case).
fn function_or_self(tool_call: &Value) -> &Value {
    match tool_call.get("function") {
        Some(f) if f.is_object() => f,
        _ => tool_call,
    }
}

fn nonempty_name(obj: &Map<String, Value>) -> Option<String> {
    obj.get("name")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

// ---------------------------------------------------------------------------
// XTML value typing
// ---------------------------------------------------------------------------

fn xtml_type(value: &Value) -> String {
    match value {
        Value::Bool(_) => "boolean",
        Value::Null => "null",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Object(_) => "object",
        Value::Array(_) => "array",
    }
    .to_string()
}

/// `_xtml_value`: strings pass through verbatim; everything else is JSON-encoded.
///
/// The Python reference uses `json.dumps(value, ensure_ascii=False)` with
/// Python's default `(", ", ": ")` separators, whereas `serde_json` emits
/// compact `(",", ":")` separators — so object/array argument values differ in
/// separator spacing. Scalar (number/bool/null) values are identical. Argument
/// values in practice (and in the golden fixtures) are strings, so this only
/// affects rarely-seen structured argument values.
fn xtml_value(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// `str(value)` for text emission: strings verbatim, everything else JSON.
fn plain_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::Number(n) => n.as_f64().is_none_or(|f| f != 0.0),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `str(scalar)` used for tool-call-id keys.
fn scalar_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Null => "None".to_string(),
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

/// Compact JSON with `serde_json`'s `(",", ":")` separators and no non-ASCII
/// escaping — matching `json.dumps(..., ensure_ascii=False, separators=(",", ":"))`.
/// Callers deep-sort the value beforehand so object keys serialize in sorted
/// order despite the crate's `preserve_order` feature.
fn json_compact(value: &Value) -> Result<String> {
    serde_json::to_string(value).map_err(Into::into)
}

fn empty_object() -> Value {
    Value::Object(Map::new())
}

/// Recursively sort object keys (ascending); array order is preserved.
///
/// Required because the crate enables `serde_json`'s `preserve_order` feature,
/// so serialization would otherwise keep the caller's insertion order.
fn deep_sort(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by_key(|entry| entry.0);
            let mut sorted = Map::with_capacity(entries.len());
            for (key, val) in entries {
                sorted.insert(key.clone(), deep_sort(val));
            }
            Value::Object(sorted)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(deep_sort).collect()),
        other => other.clone(),
    }
}

// ---------------------------------------------------------------------------
// Message normalization (side-effect free — inputs are never mutated)
// ---------------------------------------------------------------------------

/// Coerce a tool call's `arguments` into `(object, optional_raw_json_block)`.
///
/// Mirrors Python's `normalize_tool_arguments`, but is infallible: malformed
/// argument types that the Python reference would `raise` on instead degrade to
/// empty arguments (or, for unparsable strings, a raw JSON block), so a single
/// bad tool call cannot abort the whole render.
fn normalize_tool_arguments(arguments: Option<&Value>) -> (Value, Option<String>) {
    match arguments {
        None | Some(Value::Null) => (empty_object(), None),
        Some(Value::Object(m)) => (Value::Object(m.clone()), None),
        Some(Value::String(s)) => {
            if s.trim().is_empty() {
                return (empty_object(), None);
            }
            match serde_json::from_str::<Value>(s) {
                Ok(Value::Object(m)) => (Value::Object(m), None),
                // Non-object JSON (Python raises); keep the raw text as a block.
                Ok(_) => (empty_object(), Some(s.clone())),
                // Unparsable (Python's `except: return {}, arguments`).
                Err(_) => (empty_object(), Some(s.clone())),
            }
        }
        // Non-str/non-dict (Python raises TypeError); drop to empty arguments.
        Some(_) => (empty_object(), None),
    }
}

/// Deep-sort a message's `tools` and normalize any `tool_calls[].arguments`.
fn normalize_message(message: &Value) -> Value {
    let obj = match message.as_object() {
        Some(o) => o,
        None => return message.clone(),
    };
    let mut normalized = obj.clone();

    if let Some(tools) = normalized.get("tools") {
        if !tools.is_null() {
            let sorted = deep_sort(tools);
            normalized.insert("tools".to_string(), sorted);
        }
    }

    let tool_calls = match normalized.get("tool_calls").and_then(Value::as_array) {
        Some(tc) if !tc.is_empty() => tc.clone(),
        _ => return Value::Object(normalized),
    };

    let mut normalized_calls: Vec<Value> = Vec::with_capacity(tool_calls.len());
    for tool_call in &tool_calls {
        let tc_obj = match tool_call.as_object() {
            Some(o) => o,
            None => {
                normalized_calls.push(tool_call.clone());
                continue;
            }
        };
        let mut tc = tc_obj.clone();
        match tc.get("function").and_then(|f| f.as_object()).cloned() {
            Some(mut fnmap) => {
                let (args, json_block) = normalize_tool_arguments(fnmap.get("arguments"));
                fnmap.insert("arguments".to_string(), args);
                apply_json_block(&mut fnmap, json_block);
                tc.insert("function".to_string(), Value::Object(fnmap));
            }
            None => {
                let (args, json_block) = normalize_tool_arguments(tc.get("arguments"));
                tc.insert("arguments".to_string(), args);
                apply_json_block(&mut tc, json_block);
            }
        }
        normalized_calls.push(Value::Object(tc));
    }
    normalized.insert("tool_calls".to_string(), Value::Array(normalized_calls));
    Value::Object(normalized)
}

fn apply_json_block(map: &mut Map<String, Value>, json_block: Option<String>) {
    match json_block {
        None => {
            map.remove("_xtml_json_block");
        }
        Some(block) => {
            map.insert("_xtml_json_block".to_string(), Value::String(block));
        }
    }
}

/// Map assistant `tool_calls[].id` to `(1-based position, function name)`.
///
/// Every entry advances the position (even an id-less one); duplicate ids keep
/// their first occurrence.
fn tool_call_id_index(tool_calls: &Value) -> Vec<(String, usize, Option<String>)> {
    let mut index: Vec<(String, usize, Option<String>)> = Vec::new();
    let arr = match tool_calls.as_array() {
        Some(a) => a,
        None => return index,
    };
    for (pos0, tool_call) in arr.iter().enumerate() {
        let position = pos0 + 1;
        let tc_obj = match tool_call.as_object() {
            Some(o) => o,
            None => continue,
        };
        let call_id = match tc_obj.get("id") {
            Some(v) if !v.is_null() => v,
            _ => continue,
        };
        let key = scalar_str(call_id);
        if index.iter().any(|(k, _, _)| k == &key) {
            continue;
        }
        let name = match tc_obj.get("function").and_then(|f| f.as_object()) {
            Some(f) => f.get("name").and_then(Value::as_str).map(str::to_string),
            None => tc_obj
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_string),
        };
        index.push((key, position, name));
    }
    index
}

fn lookup_index<'a>(
    index: &'a [(String, usize, Option<String>)],
    key: &str,
) -> Option<&'a (String, usize, Option<String>)> {
    index.iter().find(|(k, _, _)| k == key)
}

/// Re-sort each run of consecutive `tool` messages into the most recent
/// assistant `tool_calls` order, matching by `tool_call_id == tool_calls[].id`.
///
/// A fully-matched run is sorted by the matched 1-based position (stable on
/// original offset) and each matched message's `tool`/`name` is rewritten to the
/// authoritative call name. A run that cannot be fully matched is left
/// untouched. Side-effect free: matched messages are shallow-copied; every other
/// message is cloned through unchanged. Re-running is idempotent.
fn normalize_xtml_tool_result_messages(messages: &[Value]) -> Vec<Value> {
    let mut output: Vec<Value> = Vec::with_capacity(messages.len());
    let mut current_index: Vec<(String, usize, Option<String>)> = Vec::new();
    let n = messages.len();
    let mut i = 0;

    while i < n {
        let message = &messages[i];
        let role = role_of(message);

        if role == Some("assistant") {
            current_index = match message.get("tool_calls") {
                Some(v) if v.as_array().is_some_and(|a| !a.is_empty()) => tool_call_id_index(v),
                _ => Vec::new(),
            };
            output.push(message.clone());
            i += 1;
            continue;
        }

        if role != Some("tool") {
            output.push(message.clone());
            i += 1;
            continue;
        }

        // Gather a run of consecutive tool messages.
        let mut run: Vec<(Option<usize>, usize, &Value, Option<String>)> = Vec::new();
        let mut unresolved = false;
        let mut offset = 0;
        while i < n && role_of(&messages[i]) == Some("tool") {
            let tool_message = &messages[i];
            let call_id = tool_message
                .get("tool_call_id")
                .or_else(|| tool_message.get("id"))
                .filter(|v| !v.is_null());
            let matched = call_id
                .map(scalar_str)
                .and_then(|k| lookup_index(&current_index, &k).cloned());
            match matched {
                None => {
                    unresolved = true;
                    run.push((None, offset, tool_message, None));
                }
                Some((_, position, name)) => {
                    run.push((Some(position), offset, tool_message, name));
                }
            }
            offset += 1;
            i += 1;
        }

        if unresolved {
            for item in &run {
                output.push(item.2.clone());
            }
        } else {
            run.sort_by_key(|item| (item.0, item.1));
            for (_, _, tool_message, name) in run {
                match name {
                    None => output.push(tool_message.clone()),
                    Some(nm) => {
                        let mut resolved = tool_message.as_object().cloned().unwrap_or_default();
                        resolved.insert("tool".to_string(), Value::String(nm.clone()));
                        if resolved.contains_key("name") {
                            resolved.insert("name".to_string(), Value::String(nm));
                        }
                        output.push(Value::Object(resolved));
                    }
                }
            }
        }
    }

    output
}

fn role_of(message: &Value) -> Option<&str> {
    message
        .as_object()
        .and_then(|o| o.get("role"))
        .and_then(Value::as_str)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::*;

    fn params(thinking: Option<bool>, tools: Option<&[Value]>) -> ChatTemplateParams<'_> {
        ChatTemplateParams {
            add_generation_prompt: true,
            tools,
            thinking,
            ..Default::default()
        }
    }

    fn params_kw(
        thinking: Option<bool>,
        template_kwargs: &HashMap<String, Value>,
        add_generation_prompt: bool,
    ) -> ChatTemplateParams<'_> {
        ChatTemplateParams {
            add_generation_prompt,
            thinking,
            template_kwargs: Some(template_kwargs),
            ..Default::default()
        }
    }

    #[test]
    fn thinking_defaults_on_when_unspecified() {
        let messages = vec![json!({"role": "user", "content": "Hi"})];
        let rendered = apply_kimi_k3_xtml(&messages, &params(None, None)).unwrap();
        assert!(
            rendered.ends_with("<|open|>think<|sep|>"),
            "got: {rendered}"
        );
    }

    #[test]
    fn deep_sort_orders_nested_keys() {
        let value = json!({"b": 1, "a": {"d": 2, "c": 3}});
        let sorted = json_compact(&deep_sort(&value)).unwrap();
        assert_eq!(sorted, r#"{"a":{"c":3,"d":2},"b":1}"#);
    }

    #[test]
    fn attr_values_are_escaped() {
        let mut out = String::new();
        push_attr(&mut out, "name", "a&b\"c");
        assert_eq!(out, " name=\"a&amp;b&quot;c\"");
    }

    #[test]
    fn string_arguments_render_verbatim() {
        let messages = vec![json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {"name": "f", "arguments": "{\"k\": \"v\"}"}
            }]
        })];
        let rendered = apply_kimi_k3_xtml(&messages, &params(Some(true), None)).unwrap();
        assert!(
            rendered.contains(
                "<|open|>argument key=\"k\" type=\"string\"<|sep|>v<|close|>argument<|sep|>"
            ),
            "got: {rendered}"
        );
    }

    // --- Effort directive: reasoning_effort -> thinking_effort bridge ---------

    #[test]
    fn reasoning_effort_emits_thinking_effort_directive() {
        let messages = vec![json!({"role": "user", "content": "Hi"})];
        let kwargs = HashMap::from([("reasoning_effort".to_string(), json!("high"))]);
        let rendered = apply_kimi_k3_xtml(&messages, &params_kw(None, &kwargs, true)).unwrap();
        assert!(
            rendered.contains("<|open|>message role=\"system\" type=\"thinking-effort\"<|sep|>"),
            "missing thinking-effort message: {rendered}"
        );
        assert!(
            rendered.contains(
                "Now the system is invoked with `thinking_effort=high`.<|close|>message<|sep|>"
            ),
            "wrong effort level: {rendered}"
        );
    }

    #[test]
    fn explicit_thinking_effort_wins_over_reasoning_effort() {
        let messages = vec![json!({"role": "user", "content": "Hi"})];
        let kwargs = HashMap::from([
            ("reasoning_effort".to_string(), json!("high")),
            ("thinking_effort".to_string(), json!("low")),
        ]);
        let rendered = apply_kimi_k3_xtml(&messages, &params_kw(None, &kwargs, true)).unwrap();
        assert!(
            rendered.contains("Now the system is invoked with `thinking_effort=low`."),
            "explicit thinking_effort should win: {rendered}"
        );
        assert!(
            !rendered.contains("thinking_effort=high"),
            "reasoning_effort should not leak when overridden: {rendered}"
        );
    }

    #[test]
    fn unsupported_reasoning_effort_emits_no_directive() {
        // `medium` is a valid OpenAI reasoning_effort but has no K3 equivalent:
        // it must be ignored (no directive), never error like an explicit value.
        let messages = vec![json!({"role": "user", "content": "Hi"})];
        let kwargs = HashMap::from([("reasoning_effort".to_string(), json!("medium"))]);
        let rendered = apply_kimi_k3_xtml(&messages, &params_kw(None, &kwargs, true)).unwrap();
        assert!(
            !rendered.contains("type=\"thinking-effort\""),
            "medium must not emit a directive: {rendered}"
        );
    }

    #[test]
    fn unsupported_explicit_thinking_effort_errors() {
        let messages = vec![json!({"role": "user", "content": "Hi"})];
        let kwargs = HashMap::from([("thinking_effort".to_string(), json!("medium"))]);
        assert!(apply_kimi_k3_xtml(&messages, &params_kw(None, &kwargs, true)).is_err());
    }

    #[test]
    fn effort_directive_suppressed_when_thinking_off() {
        let messages = vec![json!({"role": "user", "content": "Hi"})];
        let kwargs = HashMap::from([("reasoning_effort".to_string(), json!("low"))]);
        let rendered =
            apply_kimi_k3_xtml(&messages, &params_kw(Some(false), &kwargs, true)).unwrap();
        assert!(
            !rendered.contains("type=\"thinking-effort\""),
            "no effort directive when thinking is off: {rendered}"
        );
    }

    #[test]
    fn no_effort_directive_when_unspecified() {
        // Byte-parity with the reference: absent both keys, nothing is injected
        // (the model applies its intrinsic `max` default).
        let messages = vec![json!({"role": "user", "content": "Hi"})];
        let kwargs = HashMap::new();
        let rendered = apply_kimi_k3_xtml(&messages, &params_kw(None, &kwargs, true)).unwrap();
        assert!(
            !rendered.contains("type=\"thinking-effort\""),
            "no directive should be injected by default: {rendered}"
        );
    }

    // --- Structural <think> channel ------------------------------------------

    #[test]
    fn think_channel_is_structural_when_thinking() {
        // Assistant history message with no reasoning still carries empty tags.
        let messages = vec![
            json!({"role": "user", "content": "Hi"}),
            json!({"role": "assistant", "content": "ok"}),
        ];
        let kwargs = HashMap::new();
        let rendered =
            apply_kimi_k3_xtml(&messages, &params_kw(Some(true), &kwargs, false)).unwrap();
        assert!(
            rendered.contains(
                "<|open|>think<|sep|><|close|>think<|sep|>\
                 <|open|>response<|sep|>ok<|close|>response<|sep|>"
            ),
            "empty think channel must still be emitted: {rendered}"
        );
    }

    #[test]
    fn think_channel_dropped_when_not_thinking() {
        let messages = vec![
            json!({"role": "user", "content": "Hi"}),
            json!({"role": "assistant", "content": "ok"}),
        ];
        let kwargs = HashMap::new();
        let rendered =
            apply_kimi_k3_xtml(&messages, &params_kw(Some(false), &kwargs, false)).unwrap();
        assert!(
            !rendered.contains("<|open|>think<|sep|>"),
            "think channel must be dropped in non-thinking mode: {rendered}"
        );
        assert!(
            rendered.contains("<|open|>response<|sep|>ok<|close|>response<|sep|>"),
            "response channel must still render: {rendered}"
        );
    }

    // --- tool_choice / response_format internal messages ---------------------

    #[test]
    fn tool_choice_required_emits_internal_message() {
        let messages = vec![json!({"role": "user", "content": "Hi"})];
        let kwargs = HashMap::from([("tool_choice".to_string(), json!("required"))]);
        let rendered = apply_kimi_k3_xtml(&messages, &params_kw(None, &kwargs, true)).unwrap();
        assert!(
            rendered.contains(
                "<|open|>message role=\"system\" type=\"tool-choice\"<|sep|>\
                 The system is invoked with `tool_choice=required`.\n\
                 You MUST call tools in the next message.<|close|>message<|sep|>"
            ),
            "got: {rendered}"
        );
    }

    #[test]
    fn tool_choice_none_emits_internal_message() {
        let messages = vec![json!({"role": "user", "content": "Hi"})];
        let kwargs = HashMap::from([("tool_choice".to_string(), json!("none"))]);
        let rendered = apply_kimi_k3_xtml(&messages, &params_kw(None, &kwargs, true)).unwrap();
        assert!(
            rendered.contains(
                "The system is invoked with `tool_choice=none`.\n\
                 You MUST NOT call any tools in the next message."
            ),
            "got: {rendered}"
        );
    }

    #[test]
    fn tool_choice_auto_emits_nothing() {
        let messages = vec![json!({"role": "user", "content": "Hi"})];
        let kwargs = HashMap::from([("tool_choice".to_string(), json!("auto"))]);
        let rendered = apply_kimi_k3_xtml(&messages, &params_kw(None, &kwargs, true)).unwrap();
        assert!(
            !rendered.contains("type=\"tool-choice\""),
            "got: {rendered}"
        );
    }

    #[test]
    fn response_format_json_object_emits_internal_message() {
        let messages = vec![json!({"role": "user", "content": "Hi"})];
        let kwargs = HashMap::from([(
            "response_format".to_string(),
            json!({"type": "json_object"}),
        )]);
        let rendered = apply_kimi_k3_xtml(&messages, &params_kw(None, &kwargs, true)).unwrap();
        assert!(
            rendered.contains(
                "<|open|>message role=\"system\" type=\"response-format\"<|sep|>\
                 The system is invoked with `response_format=json_object`."
            ),
            "got: {rendered}"
        );
    }

    #[test]
    fn response_format_json_schema_emits_sorted_schema() {
        let messages = vec![json!({"role": "user", "content": "Hi"})];
        let kwargs = HashMap::from([(
            "response_format".to_string(),
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "x",
                    "schema": {"type": "object", "properties": {"a": {"type": "string"}}}
                }
            }),
        )]);
        let rendered = apply_kimi_k3_xtml(&messages, &params_kw(None, &kwargs, true)).unwrap();
        assert!(
            rendered.contains("The system is invoked with `response_format=json_schema`."),
            "got: {rendered}"
        );
        assert!(
            rendered.contains(
                "```json\n{\"properties\":{\"a\":{\"type\":\"string\"}},\"type\":\"object\"}\n```"
            ),
            "schema must be deep-sorted and compacted: {rendered}"
        );
    }
}
