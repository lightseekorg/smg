//! ``smg-as-tokenspeed-dependency`` — protocol-layer entry points exposed to
//! Python so an inference engine like TokenSpeed can drop its own
//! tokenization / function-call / reasoning-parser / OAI-server code and
//! call into smg's Rust implementations directly.
//!
//! Direction agreed on 2026-04-27/28 with @syuoni: tokenspeed remains the
//! Python entry (``ts serve``), boots ``AsyncLLM`` as before, and imports
//! this module via PyO3. The Rust side then runs the OAI HTTP layer and
//! drives ``AsyncLLM`` in-process — no gRPC, no IPC, single Python process.
//!
//! The current shape:
//!
//! * ``parse_tool_call_complete``, ``parse_reasoning_complete`` —
//!   thin pyfunctions over the existing ``tool_parser`` and
//!   ``reasoning_parser`` workspace crates so callers can verify the
//!   integration end-to-end before the HTTP server lands.
//! * ``serve_oai`` — runs an axum HTTP server in-process, drives the
//!   supplied ``AsyncLLM``-shaped engine via ``pyo3-async-runtimes``,
//!   and serves OAI-compatible ``/v1/chat/completions``. First cut is
//!   non-streaming and skips the chat-template render (passes
//!   ``messages[-1].content`` straight through as text) so the bridge
//!   itself can be exercised; chat templates, streaming and tool /
//!   reasoning parsing land in follow-ups.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use futures::stream::StreamExt;
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use serde_json::{json, Value};
use tokio::runtime::Runtime;
use tracing::{debug, error};

/// Re-export of pyo3's owned-Python-object handle. ``Py<PyAny>`` is what
/// pyo3 0.28 uses where older versions exposed ``PyObject`` from the
/// prelude.
type PyObject = Py<PyAny>;

/// Lazily-initialized tokio runtime shared across blocking PyO3 entries
/// (the parser pyfunctions). The HTTP server (``serve_oai``) builds its
/// own multi-threaded runtime and registers it with pyo3-async-runtimes.
fn shared_runtime() -> PyResult<&'static Runtime> {
    static RT: OnceLock<Runtime> = OnceLock::new();
    if let Some(rt) = RT.get() {
        return Ok(rt);
    }
    let rt = Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to start tokio runtime: {e}")))?;
    RT.set(rt)
        .map_err(|_| PyRuntimeError::new_err("tokio runtime initialized twice"))?;
    Ok(RT.get().expect("just set"))
}

// =====================================================================
// Tool-call parsing
// =====================================================================

/// Parse a complete (non-streaming) model output for tool calls.
///
/// ``parser_name`` selects which detector to run: one of the values returned
/// by :py:func:`get_available_tool_call_parsers` (e.g. ``"llama"``, ``"qwen"``,
/// ``"kimi_k2"``, ``"deepseek_v3"``, ``"json"``, ...).
///
/// Returns a ``dict`` with two keys:
///
/// * ``"normal_text"``: the prose part of the output, with any tool-call
///   payload stripped.
/// * ``"tool_calls"``: a ``list[dict]`` of ``{"name": str, "arguments": str}``
///   entries — ``arguments`` is the raw JSON string the model emitted (callers
///   usually ``json.loads`` it).
///
/// Raises :class:`ValueError` if the parser name is unknown,
/// :class:`RuntimeError` if the parser fails (malformed payload, partial
/// JSON that the detector chose to reject, etc.).
#[pyfunction]
#[pyo3(signature = (output, parser_name))]
fn parse_tool_call_complete(py: Python<'_>, output: &str, parser_name: &str) -> PyResult<PyObject> {
    let factory = tool_parser::ParserFactory::new();
    let mut parser = factory
        .registry()
        .create_parser(parser_name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown tool parser: {parser_name:?}")))?;

    let rt = shared_runtime()?;
    let (remaining, calls) = rt
        .block_on(async { parser.parse_complete(output).await })
        .map_err(|e| PyRuntimeError::new_err(format!("tool parser failed: {e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("normal_text", remaining)?;

    let list = PyList::empty(py);
    for call in calls {
        let item = PyDict::new(py);
        item.set_item("name", call.function.name)?;
        item.set_item("arguments", call.function.arguments)?;
        list.append(item)?;
    }
    dict.set_item("tool_calls", list)?;
    Ok(dict.into())
}

// =====================================================================
// Reasoning parsing
// =====================================================================

/// Detect and split a model output's reasoning block (e.g. ``<think>...</think>``,
/// Qwen3 thinking tags, DeepSeek-R1 reasoning markers) from the user-visible
/// content.
///
/// ``parser_name`` selects the model-family detector — one of the values
/// returned by :py:func:`get_available_reasoning_parsers` (e.g. ``"qwen3"``,
/// ``"deepseek_r1"``, ``"glm45"``, ...).
///
/// Returns a ``dict`` with:
///
/// * ``"normal_text"``: the prose / answer portion, reasoning stripped.
/// * ``"reasoning_text"``: the raw text inside the reasoning block (or
///   empty string if the model didn't emit one).
#[pyfunction]
#[pyo3(signature = (output, parser_name))]
fn parse_reasoning_complete(py: Python<'_>, output: &str, parser_name: &str) -> PyResult<PyObject> {
    let factory = reasoning_parser::ParserFactory::new();
    let mut parser = factory
        .registry()
        .create_parser(parser_name)
        .ok_or_else(|| {
            PyValueError::new_err(format!("unknown reasoning parser: {parser_name:?}"))
        })?;

    let result = parser
        .detect_and_parse_reasoning(output)
        .map_err(|e| PyRuntimeError::new_err(format!("reasoning parser failed: {e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("normal_text", result.normal_text)?;
    dict.set_item("reasoning_text", result.reasoning_text)?;
    Ok(dict.into())
}

// =====================================================================
// OAI HTTP server
// =====================================================================

/// Shared state held by every axum handler:
///
/// * the Python ``AsyncLLM``-shaped engine (cheap to clone — ``Py<PyAny>``
///   is reference-counted; we acquire the GIL when actually calling
///   into it).
/// * the ``TaskLocals`` captured at server start. Without this,
///   ``into_future`` from inside a tokio worker thread fails with
///   ``RuntimeError: no running event loop`` because the asyncio loop
///   pyo3-async-runtimes set up on the supervisor thread is invisible
///   to the rest of the runtime. Each handler clones these locals and
///   uses :func:`pyo3_async_runtimes::tokio::into_future_with_locals`
///   so the Python coroutine gets scheduled on the right loop.
/// * an optional Jinja chat-template processor — when present, the
///   handler renders the request's ``messages`` array into a single
///   prompt string before sending to the engine; when ``None`` we fall
///   back to taking ``messages[-1].content`` verbatim.
/// * optional names of tool / reasoning parsers from the workspace
///   crates' factories. Each streaming request instantiates fresh
///   parser instances (the state machines are stateful) so we only
///   store the names here, not the parser objects themselves.
#[derive(Clone)]
struct AppState {
    engine: Arc<Py<PyAny>>,
    locals: Arc<pyo3_async_runtimes::TaskLocals>,
    chat_template: Option<Arc<llm_tokenizer::chat_template::ChatTemplateProcessor>>,
    tool_parser_name: Option<Arc<String>>,
    reasoning_parser_name: Option<Arc<String>>,
}

/// Drive ``engine.generate_request(GenerateReqInput(text=..., sampling_params=...))``
/// to completion and return the final response dict.
///
/// The Python side is a TokenSpeed-style ``AsyncLLM``: ``generate_request``
/// returns an async generator yielding output dicts of shape
/// ``{"text": str, "output_ids": [int], "meta_info": {"finish_reason": str|dict, ...}}``.
/// For non-streaming we drive the generator with a tiny inline coroutine
/// (``await aiter.__anext__()`` until ``StopAsyncIteration``) and keep
/// the **last** yield — that's the final, complete response.
///
/// Returns the JSON-serializable dict as ``serde_json::Value`` for axum to
/// emit.
async fn drive_generate(
    engine: Arc<Py<PyAny>>,
    locals: Arc<pyo3_async_runtimes::TaskLocals>,
    prompt: String,
    sampling: serde_json::Map<String, Value>,
    request_id: String,
) -> PyResult<Value> {
    // We cross the language boundary by JSON-encoding the
    // sampling-params payload and asking Python's ``json.loads`` to
    // decode it. That is intentionally one syscall-style hop instead
    // of a recursive ``IntoPyObject`` walk: the values are tiny
    // (a handful of scalars per request), the engine itself has to
    // JSON-encode the response anyway, and avoiding pyo3's
    // type-conversion traits keeps this module portable across
    // pyo3 minor versions.
    let sampling_json = serde_json::to_string(&Value::Object(sampling.clone()))
        .map_err(|e| PyRuntimeError::new_err(format!("encode sampling_params: {e}")))?;

    // pyo3-async-runtimes' ``into_future`` reads its TaskLocals (the
    // asyncio loop reference) from a tokio task-local. Axum handlers
    // run on tokio worker threads with no such task-local set, so we
    // wrap the engine call in ``tokio::scope(locals, ...)`` — that
    // populates the task-local for the duration of this future, then
    // ``into_future`` (called from inside ``Python::attach``) finds
    // it and schedules the Python coroutine on the right loop.
    let locals_for_scope = (*locals).clone();
    let last_obj = pyo3_async_runtimes::tokio::scope(locals_for_scope, async move {
        let coro_fut = Python::attach(|py| -> PyResult<_> {
            let io_struct = py.import("tokenspeed.runtime.engine.io_struct")?;
            let generate_req_input_cls = io_struct.getattr("GenerateReqInput")?;
            let json_module = py.import("json")?;
            let sampling_dict = json_module.call_method1("loads", (sampling_json,))?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("text", prompt)?;
            kwargs.set_item("sampling_params", sampling_dict)?;
            kwargs.set_item("stream", false)?;
            let req_obj = generate_req_input_cls.call((), Some(&kwargs))?;
            // ``rid`` is declared ``field(default=None, init=False)`` on
            // GenerateReqInput, so it can't go through the constructor —
            // assign it post-init like the gRPC servicer does (which is
            // also what tokenspeed's own ``normalize_batch_and_arguments``
            // path expects to find when n>1 fan-out runs).
            req_obj.setattr("rid", request_id)?;

            // Inline coroutine helper that drains the engine's async
            // generator and returns its **final** yielded dict (the one
            // with ``meta_info.finish_reason`` set). Compiled fresh per
            // call — cheap and keeps the helper from leaking into
            // ``sys.modules``.
            let helpers = PyModule::from_code(
                py,
                std::ffi::CString::new(CONSUME_HELPER_SRC)
                    .expect("static cstring")
                    .as_c_str(),
                std::ffi::CString::new("smg_rs_serve_helpers.py")
                    .expect("static cstring")
                    .as_c_str(),
                std::ffi::CString::new("smg_rs_serve_helpers")
                    .expect("static cstring")
                    .as_c_str(),
            )?;
            let consume = helpers.getattr("_consume_to_last")?;
            let coro = consume.call1((engine.bind(py), req_obj))?;

            pyo3_async_runtimes::tokio::into_future(coro)
        })?;
        coro_fut.await
    })
    .await?;

    // 2. Convert the final Python dict to ``serde_json::Value`` via the
    //    same JSON hop in reverse: ``json.dumps`` on the Python side,
    //    ``serde_json::from_str`` on ours. The dict shape is well-defined
    //    by the TokenSpeed contract (``{"text": str, "output_ids": [int],
    //    "meta_info": {...}}``) so dumping is total.
    let json_str = Python::attach(|py| -> PyResult<String> {
        let json_module = py.import("json")?;
        let s: String = json_module
            .call_method1("dumps", (last_obj.bind(py),))?
            .extract()?;
        Ok(s)
    })?;

    serde_json::from_str(&json_str)
        .map_err(|e| PyRuntimeError::new_err(format!("decode engine response: {e}")))
}

/// Inline Python helper. Consumes the async generator and returns the
/// final yielded dict — the one with ``meta_info.finish_reason`` set.
/// Bundled as a string so the wheel doesn't need a sibling ``.py`` file.
const CONSUME_HELPER_SRC: &str = r#"
async def _consume_to_last(engine, obj):
    """Drain engine.generate_request(obj); return the final output dict."""
    last = None
    async for out in engine.generate_request(obj):
        last = out
    if last is None:
        raise RuntimeError("engine.generate_request yielded no output")
    return last
"#;

/// Render an OAI-shape ``messages`` array into a prompt string the
/// engine can tokenize.
///
/// * If a chat-template processor is wired into the server, run it
///   over the full ``messages`` array — that is the OAI contract and
///   the only path that produces a correct prompt for assistant /
///   tool / system roles.
/// * Otherwise fall back to ``messages[-1].content`` verbatim. That
///   single-turn shortcut is what the first cut shipped with; it stays
///   in place so callers that already pre-render their own template
///   (and just want a transport for raw text) don't break.
fn render_prompt_from_oai(
    body: &Value,
    chat_template: Option<&llm_tokenizer::chat_template::ChatTemplateProcessor>,
) -> Result<String, String> {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "request body missing 'messages' array".to_string())?;

    if let Some(processor) = chat_template {
        let params = llm_tokenizer::chat_template::ChatTemplateParams {
            add_generation_prompt: true,
            ..Default::default()
        };
        return processor
            .apply_chat_template(messages, params)
            .map_err(|e| format!("chat_template render failed: {e}"));
    }

    let last = messages
        .last()
        .ok_or_else(|| "'messages' is empty".to_string())?;
    let content = last
        .get("content")
        .ok_or_else(|| "last message has no 'content'".to_string())?;
    match content {
        Value::String(s) => Ok(s.clone()),
        // Vision / multipart content arrays are a follow-up.
        Value::Array(_) => Err(
            "multipart/array message content is not supported without a chat_template; \
             pass a plain string or wire chat_template into serve_oai"
                .to_string(),
        ),
        other => Err(format!("unsupported content shape: {other}")),
    }
}

/// Pull the sampling params out of an OAI-shape body into a flat dict that
/// ``GenerateReqInput.sampling_params`` understands. Only the well-known
/// subset is forwarded — engine-specific keys like ``regex``,
/// ``json_schema``, ``ebnf``, ``stop_token_ids`` etc. land in a
/// follow-up alongside tool_choice plumbing.
fn extract_sampling_from_oai(body: &Value) -> serde_json::Map<String, Value> {
    let mut out = serde_json::Map::new();
    let copy = |out: &mut serde_json::Map<String, Value>, key_oai: &str, key_engine: &str| {
        if let Some(v) = body.get(key_oai) {
            if !v.is_null() {
                out.insert(key_engine.to_string(), v.clone());
            }
        }
    };

    copy(&mut out, "temperature", "temperature");
    copy(&mut out, "top_p", "top_p");
    copy(&mut out, "top_k", "top_k");
    copy(&mut out, "frequency_penalty", "frequency_penalty");
    copy(&mut out, "presence_penalty", "presence_penalty");
    copy(&mut out, "max_tokens", "max_new_tokens");
    copy(&mut out, "max_completion_tokens", "max_new_tokens");
    copy(&mut out, "stop", "stop");

    out
}

/// Bridge ``engine.generate_request(stream=True)`` — an async generator
/// of cumulative ``{"text": ..., "meta_info": {...}}`` dicts — to an
/// axum SSE stream of OAI ``chat.completion.chunk`` events.
///
/// Mechanics:
///
/// 1. Spawn a tokio task scoped under the supplied ``TaskLocals`` (so
///    ``into_future`` finds the asyncio loop).
/// 2. The task builds the ``GenerateReqInput`` with ``stream=True``
///    once and grabs the resulting Python async iterator.
/// 3. Each iteration: call ``__anext__()``, bridge to a Rust future
///    via ``into_future``, await. On ``StopAsyncIteration`` we stop;
///    on any other ``PyErr`` we surface it as an SSE ``error`` event.
/// 4. Each yielded dict is JSON-roundtripped to ``serde_json::Value``,
///    we compute the **delta** against the cumulative ``text`` we've
///    already streamed, and emit a ``chat.completion.chunk`` event.
/// 5. After the final chunk we emit a chunk with the actual
///    ``finish_reason`` and then ``data: [DONE]`` per OAI convention.
///
/// Chunks are funneled through a bounded ``mpsc::channel`` so axum
/// can flush them downstream as fast as the client reads.
fn stream_response(
    engine: Arc<Py<PyAny>>,
    locals: Arc<pyo3_async_runtimes::TaskLocals>,
    prompt: String,
    sampling: serde_json::Map<String, Value>,
    request_id: String,
    model_label: String,
    tool_parser_name: Option<Arc<String>>,
    reasoning_parser_name: Option<Arc<String>>,
    tools: Vec<openai_protocol::common::Tool>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = futures::channel::mpsc::unbounded::<Result<Event, Infallible>>();

    let request_id_for_task = request_id.clone();
    let model_label_for_task = model_label.clone();
    let locals_for_scope = (*locals).clone();

    tokio::spawn(async move {
        // Initial role chunk so OAI clients see {"role": "assistant"} once
        // up front — same convention as openai-python's stream parser.
        let _ = tx.unbounded_send(Ok(Event::default().data(
            json!({
                "id": &request_id_for_task,
                "object": "chat.completion.chunk",
                "model": &model_label_for_task,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": null,
                }],
            })
            .to_string(),
        )));

        // Clone the sender for the inner scope so each block owns its
        // own handle (closures inside ``scope``'s async-move can't
        // borrow ``tx`` from the outer task without escaping the move).
        let tx_inner = tx.clone();

        // Build per-stream parser instances. These are stateful state
        // machines (they buffer partial JSON / partial reasoning tags
        // across chunks), so each request needs its own. Construction
        // is fallible only if the caller passed an unknown name; we
        // surface that as an SSE error event and fall back to raw text.
        let mut tool_parser: Option<Box<dyn tool_parser::ToolParser>> =
            tool_parser_name.as_ref().and_then(|name| {
                let factory = tool_parser::ParserFactory::new();
                let p = factory.registry().create_parser(name);
                if p.is_none() {
                    let _ = tx.unbounded_send(Ok(Event::default().event("error").data(
                        json!({"error": format!("unknown tool_parser: {}", name)}).to_string(),
                    )));
                }
                p
            });
        let mut reasoning_parser_instance: Option<
            Box<dyn reasoning_parser::ReasoningParser>,
        > = reasoning_parser_name.as_ref().and_then(|name| {
            let factory = reasoning_parser::ParserFactory::new();
            let p = factory.registry().create_parser(name);
            if p.is_none() {
                let _ = tx.unbounded_send(Ok(Event::default().event("error").data(
                    json!({"error": format!("unknown reasoning_parser: {}", name)}).to_string(),
                )));
            }
            p
        });

        // Tool indexes we've already emitted ``id`` for. The first chunk
        // for a tool index carries the OAI ``id`` and ``function.name``;
        // subsequent chunks for the same index only ship incremental
        // ``function.arguments`` JSON.
        let mut seen_tool_indexes = std::collections::HashSet::<usize>::new();

        let result: PyResult<()> = pyo3_async_runtimes::tokio::scope(locals_for_scope, async move {
            // Build the request and grab the async iterator object once.
            let aiter: Py<PyAny> = Python::attach(|py| -> PyResult<_> {
                let io_struct = py.import("tokenspeed.runtime.engine.io_struct")?;
                let generate_req_input_cls = io_struct.getattr("GenerateReqInput")?;
                let json_module = py.import("json")?;
                let sampling_json = serde_json::to_string(&Value::Object(sampling.clone()))
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!("encode sampling_params: {e}"))
                    })?;
                let sampling_dict = json_module.call_method1("loads", (sampling_json,))?;

                let kwargs = PyDict::new(py);
                kwargs.set_item("text", prompt)?;
                kwargs.set_item("sampling_params", sampling_dict)?;
                kwargs.set_item("stream", true)?;
                let req_obj = generate_req_input_cls.call((), Some(&kwargs))?;
                req_obj.setattr("rid", &request_id_for_task)?;

                let aiter = engine.bind(py).call_method1("generate_request", (req_obj,))?;
                Ok(aiter.unbind())
            })?;

            let mut prev_text = String::new();
            let mut last_finish_reason: Option<String> = None;
            let mut last_completion_tokens: u64 = 0;
            let mut last_prompt_tokens: u64 = 0;

            loop {
                // Pull the next chunk via __anext__; bridge to a Rust
                // future so axum's tokio reactor stays unblocked.
                let next_fut = Python::attach(|py| -> PyResult<_> {
                    let bound = aiter.bind(py);
                    let anext = bound.call_method0("__anext__")?;
                    pyo3_async_runtimes::tokio::into_future(anext)
                })?;

                let chunk_obj = match next_fut.await {
                    Ok(o) => o,
                    Err(e) => {
                        let stop = Python::attach(|py| {
                            e.is_instance_of::<PyStopAsyncIteration>(py)
                        });
                        if stop {
                            break;
                        }
                        return Err(e);
                    }
                };

                // JSON-roundtrip the dict so the rest of the loop can
                // work on serde_json::Value without touching pyo3.
                let chunk_json: Value = Python::attach(|py| -> PyResult<Value> {
                    let json_module = py.import("json")?;
                    let s: String = json_module
                        .call_method1("dumps", (chunk_obj.bind(py),))?
                        .extract()?;
                    Ok(serde_json::from_str(&s).unwrap_or(Value::Null))
                })?;

                let cur_text = chunk_json
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                // ``out["text"]`` is the cumulative output so far per
                // tokenspeed's contract; the delta we forward to the OAI
                // client is the suffix beyond what we've streamed before.
                let delta = if cur_text.starts_with(&prev_text) {
                    cur_text[prev_text.len()..].to_string()
                } else {
                    // Engine restarted / produced a non-prefix sequence
                    // (rare). Fall back to the full text — the client
                    // will see a duplicate but semantically nothing is
                    // lost.
                    cur_text.clone()
                };
                prev_text = cur_text;

                if let Some(meta) = chunk_json.get("meta_info") {
                    if let Some(fr) = meta.get("finish_reason") {
                        last_finish_reason = match fr {
                            Value::String(s) => Some(s.clone()),
                            Value::Object(m) => {
                                m.get("type").and_then(Value::as_str).map(str::to_owned)
                            }
                            _ => None,
                        };
                    }
                    if let Some(c) = meta.get("completion_tokens").and_then(Value::as_u64) {
                        last_completion_tokens = c;
                    }
                    if let Some(p) = meta.get("prompt_tokens").and_then(Value::as_u64) {
                        last_prompt_tokens = p;
                    }
                }

                if delta.is_empty() {
                    continue;
                }

                // 1. Reasoning parser strips ``<think>...</think>`` (or
                //    Qwen3 thinking tags / DSeek-R1 markers) from the
                //    delta and gives back ``(visible_text, reasoning_text)``.
                //    Apply it FIRST — reasoning blocks usually wrap the
                //    rest of the content, and feeding them into the tool
                //    parser would just confuse it.
                let (after_reasoning, reasoning_delta) = if let Some(rp) =
                    reasoning_parser_instance.as_mut()
                {
                    match rp.parse_reasoning_streaming_incremental(&delta) {
                        Ok(r) => (r.normal_text, r.reasoning_text),
                        Err(e) => {
                            error!(error = %e, "reasoning_parser streaming failed");
                            (delta.clone(), String::new())
                        }
                    }
                } else {
                    (delta.clone(), String::new())
                };

                // 2. Tool parser detects partial JSON / format-specific
                //    tool-call activity in the remaining text and gives
                //    back ``(prose_text, [ToolCallItem ...])``.
                let (visible_text, tool_calls): (String, Vec<tool_parser::types::ToolCallItem>) =
                    if let Some(tp) = tool_parser.as_mut() {
                        match tp.parse_incremental(&after_reasoning, &tools).await {
                            Ok(r) => (r.normal_text, r.calls),
                            Err(e) => {
                                error!(error = %e, "tool_parser streaming failed");
                                (after_reasoning, Vec::new())
                            }
                        }
                    } else {
                        (after_reasoning, Vec::new())
                    };

                // 3. Compose the OAI delta event. Multiple delta fields
                //    can coexist on a single chunk per OAI spec.
                let mut delta_obj = serde_json::Map::new();
                if !reasoning_delta.is_empty() {
                    // OAI extension popularized by DeepSeek-R1 / Qwen3
                    // and accepted by most current chat clients.
                    delta_obj.insert("reasoning_content".into(), Value::String(reasoning_delta));
                }
                if !visible_text.is_empty() {
                    delta_obj.insert("content".into(), Value::String(visible_text));
                }
                if !tool_calls.is_empty() {
                    let tcs: Vec<Value> = tool_calls
                        .into_iter()
                        .map(|item| {
                            let mut function_obj = serde_json::Map::new();
                            if let Some(name) = item.name.as_ref() {
                                function_obj.insert("name".into(), Value::String(name.clone()));
                            }
                            if !item.parameters.is_empty() {
                                function_obj.insert(
                                    "arguments".into(),
                                    Value::String(item.parameters.clone()),
                                );
                            }

                            let mut tc = serde_json::Map::new();
                            tc.insert("index".into(), Value::from(item.tool_index));
                            // The first chunk for a tool index gets the
                            // ``id``+``type`` envelope; later chunks for
                            // the same index just ship argument deltas.
                            if seen_tool_indexes.insert(item.tool_index) {
                                tc.insert(
                                    "id".into(),
                                    Value::String(format!(
                                        "call_{}_{}",
                                        &request_id_for_task, item.tool_index
                                    )),
                                );
                                tc.insert("type".into(), Value::String("function".into()));
                            }
                            tc.insert("function".into(), Value::Object(function_obj));
                            Value::Object(tc)
                        })
                        .collect();
                    delta_obj.insert("tool_calls".into(), Value::Array(tcs));
                }

                // Skip empty deltas (parser swallowed the chunk into its
                // buffer waiting for more) — pushing an empty chunk
                // confuses openai-python's parser.
                if delta_obj.is_empty() {
                    continue;
                }

                let _ = tx_inner.unbounded_send(Ok(Event::default().data(
                    json!({
                        "id": &request_id_for_task,
                        "object": "chat.completion.chunk",
                        "model": &model_label_for_task,
                        "choices": [{
                            "index": 0,
                            "delta": Value::Object(delta_obj),
                            "finish_reason": null,
                        }],
                    })
                    .to_string(),
                )));
            }

            // Final terminating chunk carries the finish_reason and a
            // (non-OAI but commonly accepted) usage block.
            let _ = tx_inner.unbounded_send(Ok(Event::default().data(
                json!({
                    "id": &request_id_for_task,
                    "object": "chat.completion.chunk",
                    "model": &model_label_for_task,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": last_finish_reason.unwrap_or_else(|| "stop".to_string()),
                    }],
                    "usage": {
                        "prompt_tokens": last_prompt_tokens,
                        "completion_tokens": last_completion_tokens,
                        "total_tokens": last_prompt_tokens + last_completion_tokens,
                    },
                })
                .to_string(),
            )));
            // OAI sentinel so client SSE parsers know to stop.
            let _ = tx_inner.unbounded_send(Ok(Event::default().data("[DONE]")));
            Ok(())
        })
        .await;

        if let Err(e) = result {
            error!(error = %e, "AsyncLLM streaming call failed");
            let _ = tx.unbounded_send(Ok(Event::default()
                .event("error")
                .data(json!({"error": e.to_string()}).to_string())));
        }
    });

    Sse::new(rx).keep_alive(KeepAlive::default())
}

/// ``POST /v1/chat/completions`` handler.
///
/// Branches on ``body.stream``:
///
/// * ``stream: true`` (or omitted-and-defaulting to false but with
///   ``Accept: text/event-stream``) — open an SSE response and pump
///   each chunk yielded by ``engine.generate_request`` as an OAI
///   ``chat.completion.chunk`` event, terminated by ``data: [DONE]``.
/// * ``stream: false`` — drain the generator, return a single
///   ``chat.completion`` JSON body.
///
/// Chat-template render and tool / reasoning post-processing are still
/// follow-ups; the streaming path emits raw deltas of ``out["text"]``.
async fn chat_completions_handler(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    let prompt = match render_prompt_from_oai(&body, state.chat_template.as_deref()) {
        Ok(p) => p,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": e}))).into_response(),
    };
    let sampling = extract_sampling_from_oai(&body);
    let model_label = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("default")
        .to_string();
    let stream_mode = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());

    debug!(
        prompt_len = prompt.len(),
        sampling = ?sampling,
        rid = %request_id,
        stream = stream_mode,
        "dispatching to AsyncLLM",
    );

    let engine = state.engine.clone();
    let locals = state.locals.clone();

    // Pull the OAI ``tools`` array out of the request body — the
    // streaming json/llama/qwen tool parsers want it to validate
    // emitted tool names against. Drop silently on parse error
    // (clients that don't pass tools just get raw text deltas).
    let tools: Vec<openai_protocol::common::Tool> = body
        .get("tools")
        .cloned()
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default();

    if stream_mode {
        return stream_response(
            engine,
            locals,
            prompt,
            sampling,
            request_id,
            model_label,
            state.tool_parser_name.clone(),
            state.reasoning_parser_name.clone(),
            tools,
        )
        .into_response();
    }

    let result = drive_generate(engine, locals, prompt, sampling, request_id.clone()).await;

    match result {
        Ok(out) => {
            let text = out
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let finish_reason = out
                .get("meta_info")
                .and_then(|m| m.get("finish_reason"))
                .map(|fr| match fr {
                    Value::String(s) => s.clone(),
                    Value::Object(m) => m
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("stop")
                        .to_string(),
                    _ => "stop".to_string(),
                })
                .unwrap_or_else(|| "stop".to_string());
            let prompt_tokens = out
                .get("meta_info")
                .and_then(|m| m.get("prompt_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let completion_tokens = out
                .get("meta_info")
                .and_then(|m| m.get("completion_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0);

            let body = json!({
                "id": request_id,
                "object": "chat.completion",
                "created": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                "model": model_label,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            });
            (StatusCode::OK, Json(body)).into_response()
        }
        Err(e) => {
            error!(error = %e, "AsyncLLM call failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
                .into_response()
        }
    }
}

/// ``POST /v1/completions`` handler — legacy OAI completions endpoint.
///
/// Differs from chat-completions in that it takes a raw ``prompt``
/// string (no chat template, no messages array) and returns ``text``
/// instead of ``message``. We intentionally do NOT branch on
/// ``stream: true`` for this first cut — most modern callers go through
/// chat-completions, and adding a second SSE path would duplicate the
/// streaming code without insight. Returns a single JSON ``completion``
/// object.
async fn completions_handler(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    let prompt = match body.get("prompt") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => {
            // OAI accepts an array of prompts. We only honor the first
            // one for now (batch=1); the engine doesn't currently
            // multiplex /v1/completions batches across choices, and
            // openai-python's typical usage is a single prompt anyway.
            match arr.first() {
                Some(Value::String(s)) => s.clone(),
                _ => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({"error": "'prompt' array must contain strings"})),
                    )
                        .into_response();
                }
            }
        }
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "request body missing 'prompt' string"})),
            )
                .into_response();
        }
    };
    let sampling = extract_sampling_from_oai(&body);
    let model_label = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("default")
        .to_string();
    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4().simple());

    let engine = state.engine.clone();
    let locals = state.locals.clone();
    let result = drive_generate(engine, locals, prompt, sampling, request_id.clone()).await;

    match result {
        Ok(out) => {
            let text = out
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let finish_reason = out
                .get("meta_info")
                .and_then(|m| m.get("finish_reason"))
                .map(|fr| match fr {
                    Value::String(s) => s.clone(),
                    Value::Object(m) => m
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("stop")
                        .to_string(),
                    _ => "stop".to_string(),
                })
                .unwrap_or_else(|| "stop".to_string());
            let prompt_tokens = out
                .get("meta_info")
                .and_then(|m| m.get("prompt_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let completion_tokens = out
                .get("meta_info")
                .and_then(|m| m.get("completion_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0);

            let body = json!({
                "id": request_id,
                "object": "text_completion",
                "created": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                "model": model_label,
                "choices": [{
                    "index": 0,
                    "text": text,
                    "logprobs": null,
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            });
            (StatusCode::OK, Json(body)).into_response()
        }
        Err(e) => {
            error!(error = %e, "AsyncLLM call failed (completions)");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
                .into_response()
        }
    }
}

/// Run smg's OAI-compatible HTTP server in-process, driving the supplied
/// engine via PyO3 callbacks.
///
/// Parameters
/// ----------
/// engine
///     A TokenSpeed-style ``AsyncLLM`` instance. Only
///     ``async def generate_request(obj)`` is exercised; the contract
///     is "yield output dicts of shape
///     ``{'text': str, 'output_ids': [int], 'meta_info': {...}}`` until
///     ``StopAsyncIteration``."
/// host, port
///     Listen socket. Defaults match openai-python's expectations.
/// chat_template
///     Optional Jinja chat-template string (the value of the
///     ``chat_template`` field in a HuggingFace ``tokenizer_config.json``).
///     When supplied, requests' ``messages`` arrays are rendered through
///     this template before being sent to the engine — that's the
///     OAI-correct path. When ``None``, ``messages[-1].content`` is
///     forwarded verbatim (single-turn shortcut for callers that already
///     pre-render their own template).
///
/// Intended call site (``tokenspeed serve``)::
///
///     async_llm = AsyncLLM(server_args)
///     chat_template = open(args.chat_template).read() if args.chat_template else None
///     smg_rs.serve_oai(
///         engine=async_llm,
///         host=args.host,
///         port=args.port,
///         chat_template=chat_template,
///     )
///
/// The function blocks the calling Python thread for the lifetime of the
/// server (until the process is signaled). It releases the GIL while
/// awaiting HTTP requests and re-acquires it for each engine call.
///
/// **Current limitations** (all tracked as separate follow-ups):
///
/// * No tool / reasoning streaming post-processing — the parsers are
///   exposed separately as :py:func:`parse_tool_call_complete` and
///   :py:func:`parse_reasoning_complete` for non-streaming use.
/// * Only ``/v1/chat/completions`` is wired; ``/v1/completions``,
///   ``/v1/responses``, ``/v1/embeddings`` etc. land later.
#[pyfunction]
#[pyo3(signature = (
    engine,
    host = "127.0.0.1",
    port = 8000,
    chat_template = None,
    tool_parser = None,
    reasoning_parser = None,
))]
fn serve_oai(
    py: Python<'_>,
    engine: PyObject,
    host: &str,
    port: u16,
    chat_template: Option<String>,
    tool_parser: Option<String>,
    reasoning_parser: Option<String>,
) -> PyResult<()> {
    let host_owned = host.to_string();
    let addr: SocketAddr = format!("{host_owned}:{port}")
        .parse()
        .map_err(|e| PyValueError::new_err(format!("invalid host:port {host_owned}:{port}: {e}")))?;

    // Compile the chat template up front so a malformed template fails
    // at startup rather than on the first request.
    let chat_template_arc: Option<Arc<llm_tokenizer::chat_template::ChatTemplateProcessor>> =
        match chat_template {
            Some(template_src) => {
                let processor =
                    llm_tokenizer::chat_template::ChatTemplateProcessor::new(template_src)
                        .map_err(|e| {
                            PyValueError::new_err(format!(
                                "chat_template failed to parse: {e}"
                            ))
                        })?;
                Some(Arc::new(processor))
            }
            None => None,
        };

    // Validate parser names eagerly: if the operator passes ``"qwen"`` and
    // we don't have one, we'd rather fail at server startup than per
    // request.
    if let Some(name) = tool_parser.as_ref() {
        let factory = ::tool_parser::ParserFactory::new();
        if !factory.has_parser(name) {
            return Err(PyValueError::new_err(format!(
                "unknown tool_parser: {name:?} (available: {:?})",
                factory.list_parsers()
            )));
        }
    }
    if let Some(name) = reasoning_parser.as_ref() {
        let factory = ::reasoning_parser::ParserFactory::new();
        if !factory.list_parsers().contains(name) {
            return Err(PyValueError::new_err(format!(
                "unknown reasoning_parser: {name:?} (available: {:?})",
                factory.list_parsers()
            )));
        }
    }

    let engine_arc = Arc::new(engine);

    debug!(
        %addr,
        has_chat_template = chat_template_arc.is_some(),
        tool_parser = ?tool_parser,
        reasoning_parser = ?reasoning_parser,
        "smg_rs.serve_oai starting",
    );

    // ``pyo3_async_runtimes::tokio::run`` sets up an asyncio event loop
    // on a supervisor thread, builds a tokio runtime, registers both
    // with the bridge, and runs the supplied Rust future on the tokio
    // runtime. Inside the future we capture the supervisor loop's
    // ``TaskLocals`` so each axum handler can pass them to
    // ``into_future_with_locals`` — handlers run on tokio worker
    // threads where the loop reference is otherwise invisible.
    pyo3_async_runtimes::tokio::run(py, async move {
        // ``tokio::run`` set TaskLocals as a tokio task-local for this
        // future. Pull them out so we can clone them per axum handler
        // (each handler runs as its own tokio task and would otherwise
        // not see them).
        let locals = Python::attach(|py| -> PyResult<_> {
            pyo3_async_runtimes::tokio::get_current_locals(py)
        })?;

        let state = AppState {
            engine: engine_arc,
            locals: Arc::new(locals),
            chat_template: chat_template_arc,
            tool_parser_name: tool_parser.map(Arc::new),
            reasoning_parser_name: reasoning_parser.map(Arc::new),
        };
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions_handler))
            .route("/v1/completions", post(completions_handler))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("bind {addr} failed: {e}")))?;
        axum::serve(listener, app)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("axum serve failed: {e}")))
    })?;

    Ok(())
}

// =====================================================================
// Module wiring
// =====================================================================

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_tool_call_complete, m)?)?;
    m.add_function(wrap_pyfunction!(parse_reasoning_complete, m)?)?;
    m.add_function(wrap_pyfunction!(serve_oai, m)?)?;
    Ok(())
}
