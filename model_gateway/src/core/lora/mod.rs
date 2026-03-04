//! LoRA adapter lifecycle management.
//!
//! Each [`BasicWorker`] optionally holds a [`WorkerLoraState`] initialised at
//! worker-registration time from `RouterConfig.lora`.
//!
//! ## v1 scope
//! Only local filesystem paths and pre-loaded adapter names are supported.
//! Remote URI schemes (`s3://`, `hermes://`, …) return an explicit 400 error;
//! download support will be added in a future version.

pub mod engine_client;
pub mod state;
pub mod uri;

pub use engine_client::{EngineClientError, LoraEngineClient};
pub use state::{LoraStateError, WorkerLoraState};
pub use uri::{classify, AdapterUri, UnsupportedSchemeError};
