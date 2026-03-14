//! LoRA adapter lifecycle management.
//!
//! Each [`BasicWorker`] holds a [`WorkerLoraState`] initialised at
//! worker-registration time.  The state handles:
//! - URI classification (via [`openai_protocol::lora`])
//! - First-request loading into the engine (with per-path deduplication)
//! - Per-worker caching of loaded adapter names
//! - Request JSON rewriting for the target runtime (SGLang vs vLLM)
//!
//! ## v1 scope
//! Only local filesystem paths and pre-loaded adapter names are supported.
//! Remote URI schemes (`s3://`, `hermes://`, …) return an explicit 400 error.

pub mod state;

pub use state::{LoraStateError, WorkerLoraState};
