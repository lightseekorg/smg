//! WebAssembly Interface Bindings for Storage Hooks.
//!
//! Invokes `wasmtime::component::bindgen!` at compile time to generate
//! host-side bindings from `interface/storage/storage-hooks.wit`.

wasmtime::component::bindgen!({
    path: "src/interface/storage",
    world: "storage-hook",
    imports: { default: async | trappable },
    exports: { default: async },
});
