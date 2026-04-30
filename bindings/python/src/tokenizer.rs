//! PyO3 bindings for the `llm-tokenizer` crate.
//!
//! Mirrors the semantics of `bindings/golang/src/tokenizer.rs`, exposed as a
//! Python class with PyO3 idioms (exceptions instead of error codes,
//! lifetime via `Drop` instead of explicit `_free` calls).

use std::sync::Arc;

use llm_tokenizer::{chat_template::ChatTemplateParams, traits::Tokenizer as TokenizerTrait};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use serde_json::Value;

/// Python-facing tokenizer handle.
///
/// Wraps `Arc<dyn llm_tokenizer::traits::Tokenizer>`. Constructed via
/// `Tokenizer.from_file(path)`. Supports `encode`, `decode`, `apply_chat_template`.
// The Python-facing name strips the `Py` prefix (`Tokenizer`, not
// `PyTokenizer`). This is the convention for new bindings modules; older
// classes in `lib.rs` (e.g. `PyRole`, `PyOracleConfig`) predate it.
#[pyclass(name = "Tokenizer", module = "smg_rs")]
pub struct PyTokenizer {
    inner: Arc<dyn TokenizerTrait>,
}

#[pymethods]
impl PyTokenizer {
    /// Create a tokenizer from a local file path or HuggingFace model ID.
    ///
    /// If `path` is a local path, the tokenizer is loaded from disk.
    /// Otherwise it is fetched from the HuggingFace Hub (set `HF_TOKEN` env var
    /// for gated models).
    ///
    /// `from_file` raises `ValueError` on invalid inputs (bad path, malformed
    /// arguments). `encode`, `decode`, and `apply_chat_template` raise
    /// `RuntimeError` on tokenizer operation failures.
    #[staticmethod]
    #[pyo3(signature = (path))]
    fn from_file(py: Python<'_>, path: &str) -> PyResult<Self> {
        let inner = py
            .detach(|| llm_tokenizer::create_tokenizer(path))
            .map_err(|e| {
                PyValueError::new_err(format!("failed to load tokenizer from {path}: {e}"))
            })?;
        Ok(PyTokenizer { inner })
    }

    /// Encode `text` to a list of token IDs.
    ///
    /// `add_special_tokens` controls whether the tokenizer's BOS/EOS tokens are
    /// included; defaults to true to match the HuggingFace `__call__` default.
    #[pyo3(signature = (text, add_special_tokens = true))]
    fn encode(&self, py: Python<'_>, text: &str, add_special_tokens: bool) -> PyResult<Vec<u32>> {
        let encoding = py
            .detach(|| self.inner.encode(text, add_special_tokens))
            .map_err(|e| PyRuntimeError::new_err(format!("encode failed: {e}")))?;
        Ok(encoding.token_ids().to_vec())
    }

    /// Decode a list of token IDs to text.
    ///
    /// `skip_special_tokens` controls whether BOS/EOS tokens are stripped from
    /// the output; defaults to true.
    #[pyo3(signature = (token_ids, skip_special_tokens = true))]
    fn decode(
        &self,
        py: Python<'_>,
        token_ids: Vec<u32>,
        skip_special_tokens: bool,
    ) -> PyResult<String> {
        if token_ids.is_empty() {
            return Ok(String::new());
        }
        py.detach(|| self.inner.decode(&token_ids, skip_special_tokens))
            .map_err(|e| PyRuntimeError::new_err(format!("decode failed: {e}")))
    }

    /// Apply the tokenizer's chat template to a list of messages.
    ///
    /// `messages` is a list of dicts (e.g. `[{"role": "user", "content": "..."}]`).
    /// `tools`, when provided, is a list of OpenAI-style tool dicts that the
    /// chat template can render into the prompt. When `None` (the default),
    /// `tools` is left **undefined** in the template context so guards like
    /// `{% if tools is defined %}` work correctly.
    ///
    /// `add_generation_prompt` defaults to true (matches HuggingFace and
    /// SGLang/vLLM convention for serving).
    ///
    /// The tokenizer's special tokens (`bos_token`, `eos_token`, etc.) are
    /// always injected so templates that reference them render correctly.
    #[pyo3(signature = (messages, tools = None, add_generation_prompt = true))]
    fn apply_chat_template(
        &self,
        py: Python<'_>,
        messages: &Bound<'_, PyAny>,
        tools: Option<&Bound<'_, PyAny>>,
        add_generation_prompt: bool,
    ) -> PyResult<String> {
        // py_to_json must run with the GIL held.
        let messages_value = py_to_json(messages)?;
        let messages_vec: Vec<Value> = match messages_value {
            Value::Array(arr) => arr,
            _ => {
                return Err(PyValueError::new_err(
                    "messages must be a list of message dicts",
                ));
            }
        };

        let tools_vec: Option<Vec<Value>> = match tools {
            None => None,
            Some(t) => match py_to_json(t)? {
                Value::Array(arr) => Some(arr),
                _ => {
                    return Err(PyValueError::new_err("tools must be a list of tool dicts"));
                }
            },
        };

        let inner = Arc::clone(&self.inner);
        py.detach(move || {
            let params = ChatTemplateParams {
                add_generation_prompt,
                tools: tools_vec.as_deref(),
                documents: None,
                template_kwargs: None,
                special_tokens: Some(inner.get_special_tokens()),
            };
            inner.apply_chat_template(&messages_vec, params)
        })
        .map_err(|e| PyRuntimeError::new_err(format!("apply_chat_template failed: {e}")))
    }

    fn __repr__(&self) -> String {
        format!("Tokenizer(vocab_size={})", self.inner.vocab_size())
    }
}

/// Convert a Python object (dict, list, etc.) to `serde_json::Value` by going
/// through Python's JSON serializer. This is the simplest correct way to
/// bridge arbitrary nested Python data to serde without writing a custom
/// `FromPyObject` for every shape.
fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    let py = obj.py();
    let json = py.import("json")?;
    let s: String = json.call_method1("dumps", (obj,))?.extract()?;
    serde_json::from_str(&s)
        .map_err(|e| PyValueError::new_err(format!("failed to convert Python object to JSON: {e}")))
}
