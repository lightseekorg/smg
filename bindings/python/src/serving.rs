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

use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
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

/// Shared state held by every axum handler — just a reference to the
/// Python ``AsyncLLM``-shaped engine. Cloning is cheap (``Py<PyAny>``
/// is reference-counted under the hood; we acquire the GIL when we
/// actually call into it).
#[derive(Clone)]
struct AppState {
    engine: Arc<Py<PyAny>>,
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
    prompt: String,
    sampling: serde_json::Map<String, Value>,
    request_id: String,
) -> PyResult<Value> {
    // 1. Build the GenerateReqInput on the Python side under the GIL.
    //    Then hand the resulting coroutine off to pyo3-async-runtimes
    //    so axum's tokio reactor doesn't block on the engine.
    //
    //    We cross the language boundary by JSON-encoding the
    //    sampling-params payload and asking Python's ``json.loads`` to
    //    decode it. That is intentionally one syscall-style hop instead
    //    of a recursive ``IntoPyObject`` walk: the values are tiny
    //    (a handful of scalars per request), the engine itself has to
    //    JSON-encode the response anyway, and avoiding pyo3's
    //    type-conversion traits keeps this module portable across
    //    pyo3 minor versions.
    let sampling_json = serde_json::to_string(&Value::Object(sampling.clone()))
        .map_err(|e| PyRuntimeError::new_err(format!("encode sampling_params: {e}")))?;

    let coro_fut = Python::with_gil(|py| -> PyResult<_> {
        let io_struct = py.import("tokenspeed.runtime.engine.io_struct")?;
        let generate_req_input_cls = io_struct.getattr("GenerateReqInput")?;
        let json_module = py.import("json")?;
        let sampling_dict = json_module.call_method1("loads", (sampling_json,))?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("text", prompt)?;
        kwargs.set_item("sampling_params", sampling_dict)?;
        kwargs.set_item("rid", request_id)?;
        kwargs.set_item("stream", false)?;
        let req_obj = generate_req_input_cls.call((), Some(&kwargs))?;

        // Inline coroutine helper that drains the engine's async generator
        // and returns its **final** yielded dict (the one with
        // ``meta_info.finish_reason`` set). Compiled fresh per call —
        // cheap and keeps the helper from leaking into ``sys.modules``.
        let helpers = pyo3::types::PyModule::from_code(
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

    let last_obj = coro_fut.await?;

    // 2. Convert the final Python dict to ``serde_json::Value`` via the
    //    same JSON hop in reverse: ``json.dumps`` on the Python side,
    //    ``serde_json::from_str`` on ours. The dict shape is well-defined
    //    by the TokenSpeed contract (``{"text": str, "output_ids": [int],
    //    "meta_info": {...}}``) so dumping is total.
    let json_str = Python::with_gil(|py| -> PyResult<String> {
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

/// Pull the user's last message text out of an OAI-shape body. First-cut
/// shortcut: skip chat-template rendering and just take whatever the user
/// said most recently. We do this on Rust side because chat-template
/// support requires loading the model's HF tokenizer; lands as a
/// follow-up.
fn extract_prompt_from_oai(body: &Value) -> Result<String, String> {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "request body missing 'messages' array".to_string())?;
    let last = messages
        .last()
        .ok_or_else(|| "'messages' is empty".to_string())?;
    let content = last
        .get("content")
        .ok_or_else(|| "last message has no 'content'".to_string())?;
    match content {
        Value::String(s) => Ok(s.clone()),
        // Vision / multipart content arrays are documented as a follow-up.
        Value::Array(_) => Err(
            "multipart/array message content is not supported in the first-cut serve_oai; \
             pass a plain string"
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

/// ``POST /v1/chat/completions`` handler.
///
/// First-cut behavior: extract the last user message, drive ``AsyncLLM``,
/// pack the engine's text response into an OAI ``ChatCompletion`` shape.
/// No streaming, no chat template, no tool/reasoning post-processing.
async fn chat_completions_handler(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    let prompt = match extract_prompt_from_oai(&body) {
        Ok(p) => p,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": e}))).into_response(),
    };
    let sampling = extract_sampling_from_oai(&body);
    let model_label = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("default")
        .to_string();
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());

    debug!(
        prompt_len = prompt.len(),
        sampling = ?sampling,
        rid = %request_id,
        "dispatching to AsyncLLM",
    );

    let engine = state.engine.clone();
    let result = drive_generate(engine, prompt, sampling, request_id.clone()).await;

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

/// Run smg's OAI-compatible HTTP server in-process, driving the supplied
/// engine via PyO3 callbacks.
///
/// Intended call site (``tokenspeed serve``)::
///
///     async_llm = AsyncLLM(server_args)
///     smg_rs.serve_oai(
///         engine=async_llm,
///         host=args.host,
///         port=args.port,
///     )
///
/// The function blocks the calling Python thread for the lifetime of the
/// server (until the process is signaled). It releases the GIL while
/// awaiting HTTP requests and re-acquires it for each engine call.
///
/// **First-cut limitations** (all tracked as separate follow-ups):
///
/// * Non-streaming only — streaming SSE goes through pyo3-async-runtimes
///   wrapping ``__anext__`` per chunk.
/// * Skips chat-template rendering — passes the last ``messages[-1].content``
///   directly to the engine. The full path will use the ``llm-tokenizer``
///   crate's minijinja chat template.
/// * No tool / reasoning post-processing — the parsers are exposed
///   separately as :py:func:`parse_tool_call_complete` and
///   :py:func:`parse_reasoning_complete` for now.
/// * Only ``/v1/chat/completions`` is wired; ``/v1/completions``,
///   ``/v1/responses``, ``/v1/embeddings`` etc. land later.
#[pyfunction]
#[pyo3(signature = (engine, host = "127.0.0.1", port = 8000))]
fn serve_oai(py: Python<'_>, engine: PyObject, host: &str, port: u16) -> PyResult<()> {
    let host_owned = host.to_string();
    let addr: SocketAddr = format!("{host_owned}:{port}")
        .parse()
        .map_err(|e| PyValueError::new_err(format!("invalid host:port {host_owned}:{port}: {e}")))?;

    let state = AppState {
        engine: Arc::new(engine),
    };
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions_handler))
        .with_state(state);

    // Build a tokio runtime that pyo3-async-runtimes will treat as the
    // bridge between Python coroutines (engine.generate_request) and
    // axum's request loop. The runtime is leaked for the lifetime of
    // the process — this function is the entry point and never returns
    // until the server exits.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("smg-oai-server")
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime build failed: {e}")))?;

    // Hand the runtime to pyo3-async-runtimes so ``into_future`` knows
    // where to schedule Python coroutines. Must happen before the first
    // ``into_future`` call.
    let rt_handle = rt.handle().clone();
    pyo3_async_runtimes::tokio::init_with_runtime(&rt)
        .map_err(|e| PyRuntimeError::new_err(format!("pyo3-async-runtimes init failed: {e}")))?;

    debug!(%addr, "smg_rs.serve_oai starting on {addr}");

    // Release the GIL while the server is running so Python coroutines
    // dispatched by axum handlers can re-acquire it themselves.
    py.detach(|| {
        rt_handle.block_on(async move {
            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("bind {addr} failed: {e}"))
                })?;
            axum::serve(listener, app)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("axum serve failed: {e}")))
        })
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
