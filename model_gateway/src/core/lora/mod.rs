//! LoRA adapter lifecycle: load local adapters into SGLang/vLLM on first use,
//! rewrite inference requests to reference the engine-side adapter name.

pub mod engine_client;
pub mod middleware;
pub mod uri;

pub use engine_client::{EngineClientError, LoraEngineClient};
pub use middleware::{LoraError, LoraMiddleware};
pub use uri::{classify, AdapterUri, UnsupportedSchemeError};
