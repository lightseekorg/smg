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
//! This module is the *skeleton*: the parser entry points are real (they
//! re-use the ``tool_parser`` and ``reasoning_parser`` workspace crates as
//! libraries) so callers can verify the integration works end-to-end; the
//! ``serve_oai`` HTTP entry is a stub that raises a clear error message
//! pointing at the follow-up work.
//!
//! See ``crates/protocols``, ``crates/tool_parser``, ``crates/reasoning_parser``,
//! ``crates/tokenizer`` for the full library surface that will land here over
//! the next few iterations.

use std::sync::OnceLock;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::{Py, PyAny};
use tokio::runtime::Runtime;

/// Re-export of pyo3's owned-Python-object handle. ``Py<PyAny>`` is what
/// pyo3 0.28 uses where older versions exposed ``PyObject`` from the
/// prelude.
type PyObject = Py<PyAny>;

/// Lazily-initialized tokio runtime shared across blocking PyO3 entries.
///
/// One Runtime per process is enough for the read-only parser entries; the
/// HTTP server (``serve_oai``) will spawn its own multi-threaded runtime
/// when implemented.
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
// OAI HTTP server (stub)
// =====================================================================

/// Run smg's OAI-compatible HTTP server in-process, driving the supplied
/// engine via PyO3 callbacks.
///
/// Intended call site (``tokenspeed serve``)::
///
///     async_llm = AsyncLLM(server_args)
///     await smg_rs.serve_oai(
///         engine=async_llm,
///         host=args.host,
///         port=args.port,
///         chat_template=args.chat_template,
///     )
///
/// **Currently a stub.** The full implementation will spin up an axum
/// router that:
///
/// 1. Renders chat templates server-side (``llm-tokenizer`` crate).
/// 2. Tokenizes input via the model's HF tokenizer.
/// 3. Builds ``SamplingParams`` from the request.
/// 4. Awaits ``engine.async_generate(input_ids, sampling_params)`` via
///    ``pyo3-async-runtimes``, pulling tokens out as a Rust ``Stream``.
/// 5. Streams parser-detected tool / reasoning chunks back as SSE.
///
/// Tracking: ``crates/pylib`` direction note in CHANGELOG, and the
/// "smg-as-pylib" thread on Slack with @syuoni.
#[pyfunction]
#[pyo3(signature = (engine, host, port, chat_template = None))]
#[allow(unused_variables)]
fn serve_oai(
    engine: PyObject,
    host: &str,
    port: u16,
    chat_template: Option<&str>,
) -> PyResult<()> {
    Err(PyRuntimeError::new_err(
        "serve_oai is not implemented yet — landing in a follow-up commit on \
         feat/pylib-protocol. Integration shape: smg axum HTTP server in-process, \
         driving AsyncLLM via pyo3-async-runtimes. See bindings/python/src/serving.rs \
         doc comment for the planned signature and the call sites that will use it.",
    ))
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
