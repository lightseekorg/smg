//! WASM Component Type System
//!
//! Provides generic input/output types for WASM component execution
//! based on attach points.

use wasmtime::{component::ResourceTable, StoreLimits};
use wasmtime_wasi::{WasiCtx, WasiCtxView, WasiView};

use crate::spec::smg::gateway::middleware_types;

/// Generic input type for WASM component execution
///
/// This enum represents all possible input types that can be passed
/// to a WASM component, determined by the attach_point.
#[derive(Debug, Clone)]
pub enum WasmComponentInput {
    /// Middleware OnRequest input
    MiddlewareRequest(middleware_types::Request),
    /// Middleware OnResponse input
    MiddlewareResponse(middleware_types::Response),
}

/// Generic output type from WASM component execution
///
/// This enum represents all possible output types that can be returned
/// from a WASM component, determined by the attach_point.
#[derive(Debug, Clone)]
pub enum WasmComponentOutput {
    /// Middleware Action output
    MiddlewareAction(middleware_types::Action),
}

pub struct WasiState {
    pub ctx: WasiCtx,
    pub table: ResourceTable,
    pub limits: StoreLimits,
}

impl WasiView for WasiState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.ctx,
            table: &mut self.table,
        }
    }
}
