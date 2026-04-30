//! PyO3 bindings for the `llm-tokenizer` crate.
//!
//! Mirrors the semantics of `bindings/golang/src/tokenizer.rs`, exposed as a
//! Python class with PyO3 idioms (exceptions instead of error codes,
//! lifetime via `Drop` instead of explicit `_free` calls).

use std::sync::Arc;

use llm_tokenizer::traits::Tokenizer as TokenizerTrait;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
    /// All errors are currently raised as `ValueError`; future versions may
    /// distinguish input errors (`ValueError`) from runtime/network errors
    /// (`RuntimeError`).
    #[staticmethod]
    #[pyo3(signature = (path))]
    fn from_file(path: &str) -> PyResult<Self> {
        let inner = llm_tokenizer::create_tokenizer(path).map_err(|e| {
            PyValueError::new_err(format!("failed to load tokenizer from {path}: {e}"))
        })?;
        Ok(PyTokenizer { inner })
    }

    fn __repr__(&self) -> String {
        format!("Tokenizer(vocab_size={})", self.inner.vocab_size())
    }
}
