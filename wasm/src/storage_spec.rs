//! WebAssembly Interface Bindings for Storage Hooks
//!
//! Generated from `interface/storage/storage-hooks.wit` using wasmtime's
//! component model bindgen.

wasmtime::component::bindgen!({
    path: "src/interface/storage",
    world: "storage-hook",
    imports: { default: async | trappable },
    exports: { default: async },
});
