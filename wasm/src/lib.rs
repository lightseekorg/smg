//! WebAssembly (WASM) module support for Shepherd Model Gateway
//!
//! This crate provides WASM component execution capabilities using the WebAssembly Component Model.
//! It supports middleware execution at various attach points (OnRequest, OnResponse) with async support.

pub mod config;
pub mod errors;
pub mod module;
pub mod module_manager;
pub mod runtime;
pub mod spec;
pub mod types;

// Re-export commonly used types
pub use config::WasmRuntimeConfig;
pub use errors::{Result, WasmError, WasmManagerError, WasmModuleError, WasmRuntimeError};
pub use module::{
    MiddlewareAttachPoint, WasmMetrics, WasmModule, WasmModuleAddRequest, WasmModuleAddResponse,
    WasmModuleAddResult, WasmModuleAttachPoint, WasmModuleDescriptor, WasmModuleListResponse,
    WasmModuleMeta, WasmModuleType,
};
pub use module_manager::WasmModuleManager;
pub use runtime::WasmRuntime;
pub use spec::{apply_modify_action_to_headers, build_wasm_headers_from_axum_headers, smg, Smg};
pub use types::{WasiState, WasmComponentInput, WasmComponentOutput};
