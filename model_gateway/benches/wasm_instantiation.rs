//! Benchmark: WASM Component Instantiation Overhead
//!
//! Measures the per-request cost of creating a Store, WasiCtx, ResourceTable,
//! StoreLimits and calling `Linker::instantiate_async()` — the exact hot path
//! in `WasmThreadPool::execute_component_in_worker`.
//!
//! Three scenarios:
//!   1. `cold_compile_and_instantiate` — Component::new() + Store + instantiate (first request)
//!   2. `cached_component_instantiate` — cached Component + fresh Store + instantiate (steady state)
//!   3. `store_creation_only` — just Store + WasiCtx + ResourceTable + StoreLimits (isolation)
//!
//! Run: cargo bench --bench wasm_instantiation

use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use wasmtime::{
    component::{Component, Linker, ResourceTable},
    Config, Engine, InstanceAllocationStrategy, PoolingAllocationConfig, Store, StoreLimitsBuilder,
};
use wasmtime_wasi::{WasiCtx, WasiCtxView, WasiView};

// WasiState replica (mirrors wasm/src/types.rs)

struct BenchWasiState {
    ctx: WasiCtx,
    table: ResourceTable,
    limits: wasmtime::StoreLimits,
}

impl WasiView for BenchWasiState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.ctx,
            table: &mut self.table,
        }
    }
}

// Minimal valid WASM component (WAT)

// A minimal component that exports two no-op functions matching the shape
// of our middleware interface (takes a record, returns a variant).
// We use a simple core module wrapped in a component since the real
// middleware's guest logic is irrelevant — we're measuring instantiation.
const MINIMAL_COMPONENT_WAT: &str = r#"
(component
  (core module $m
    (memory (export "memory") 1)
    (func (export "run") (result i32)
      i32.const 0
    )
  )
  (core instance $i (instantiate $m))
)
"#;

// Engine + config helpers (mirrors WasmThreadPool::worker_loop)

fn make_pooling_engine() -> Engine {
    let mut pool_config = PoolingAllocationConfig::default();
    let max_memory_bytes: usize = 1024 * 65536; // 1024 pages = 64MB (default config)
    pool_config.total_core_instances(20);
    pool_config.max_memory_size(max_memory_bytes);
    pool_config.max_component_instance_size(max_memory_bytes);
    pool_config.max_tables_per_component(5);

    let mut config = Config::new();
    config.allocation_strategy(InstanceAllocationStrategy::Pooling(pool_config));
    config.async_stack_size(1024 * 1024); // 1MB (default)
    config.async_support(true);
    config.wasm_component_model(true);
    config.epoch_interruption(true);
    Engine::new(&config).expect("Failed to create pooling engine")
}

fn make_standard_engine() -> Engine {
    let mut config = Config::new();
    config.async_support(true);
    config.wasm_component_model(true);
    config.epoch_interruption(true);
    Engine::new(&config).expect("Failed to create standard engine")
}

fn make_store(engine: &Engine) -> Store<BenchWasiState> {
    let memory_limit_bytes: usize = 1024 * 65536; // 64MB
    let limits = StoreLimitsBuilder::new()
        .memory_size(memory_limit_bytes)
        .trap_on_grow_failure(true)
        .build();

    let mut store = Store::new(
        engine,
        BenchWasiState {
            ctx: WasiCtx::builder().build(),
            table: ResourceTable::new(),
            limits,
        },
    );

    store.limiter(|state| &mut state.limits);

    let deadline_epochs = (1000u64 / 100).max(1); // 1s timeout, 100ms epoch interval
    store.set_epoch_deadline(deadline_epochs);
    store.epoch_deadline_callback(|_store| {
        Err(wasmtime::Error::msg("execution time limit exceeded"))
    });

    store
}

fn make_linker(engine: &Engine) -> Linker<BenchWasiState> {
    let mut linker = Linker::<BenchWasiState>::new(engine);
    wasmtime_wasi::p2::add_to_linker_async(&mut linker).expect("Failed to add WASI to linker");
    linker
}

// Benchmarks

/// Scenario 1: Cold path — compile component from WAT + create Store + instantiate.
/// This is what happens on the very first request with a new module.
fn bench_cold_compile_and_instantiate(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let engine = make_pooling_engine();
    let linker = make_linker(&engine);

    // Pre-encode WAT to binary once (wasm text → binary is NOT part of production path)
    let wasm_bytes = wasmtime::wat::parse_str(MINIMAL_COMPONENT_WAT).expect("Failed to parse WAT");

    c.bench_function("wasm_cold_compile_and_instantiate", |b| {
        b.iter(|| {
            rt.block_on(async {
                // 1. Compile component from bytes (production does this on cache miss)
                let component =
                    Component::new(&engine, &wasm_bytes).expect("Failed to compile component");

                // 2. Create fresh Store (production does this EVERY request)
                let mut store = make_store(&engine);

                // 3. Instantiate (production does this EVERY request)
                let _instance = linker
                    .instantiate_async(&mut store, &component)
                    .await
                    .expect("Failed to instantiate component");
            });
        });
    });
}

/// Scenario 2: Steady-state — component already compiled/cached, just Store + instantiate.
/// This is what happens on every request after the first (cache hit).
fn bench_cached_component_instantiate(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let engine = make_pooling_engine();
    let linker = make_linker(&engine);

    // Pre-compile component (this is the cached state)
    let wasm_bytes = wasmtime::wat::parse_str(MINIMAL_COMPONENT_WAT).expect("Failed to parse WAT");
    let component = Component::new(&engine, &wasm_bytes).expect("Failed to compile component");

    c.bench_function("wasm_cached_component_instantiate", |b| {
        b.iter(|| {
            rt.block_on(async {
                // 1. Clone component handle (cheap — just an Arc bump)
                let comp = component.clone();

                // 2. Create fresh Store (production does this EVERY request)
                let mut store = make_store(&engine);

                // 3. Instantiate (production does this EVERY request)
                let _instance = linker
                    .instantiate_async(&mut store, &comp)
                    .await
                    .expect("Failed to instantiate component");
            });
        });
    });
}

/// Scenario 3: Isolate Store creation cost (no compilation, no instantiation).
/// Shows how much of the per-request time is just building WasiCtx + ResourceTable.
fn bench_store_creation_only(c: &mut Criterion) {
    let engine = make_pooling_engine();

    c.bench_function("wasm_store_creation_only", |b| {
        b.iter(|| {
            let _store = make_store(&engine);
        });
    });
}

/// Scenario 4: Pooling vs standard engine instantiation comparison.
/// Shows the benefit of PoolingAllocationStrategy.
fn bench_standard_vs_pooling(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Standard engine (no pooling)
    let std_engine = make_standard_engine();
    let std_linker = make_linker(&std_engine);
    let wasm_bytes = wasmtime::wat::parse_str(MINIMAL_COMPONENT_WAT).expect("Failed to parse WAT");
    let std_component =
        Component::new(&std_engine, &wasm_bytes).expect("Failed to compile component");

    // Pooling engine
    let pool_engine = make_pooling_engine();
    let pool_linker = make_linker(&pool_engine);
    let pool_component =
        Component::new(&pool_engine, &wasm_bytes).expect("Failed to compile component");

    let mut group = c.benchmark_group("wasm_instantiate_engine_comparison");

    group.bench_function("standard_engine", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut store = make_store(&std_engine);
                let _instance = std_linker
                    .instantiate_async(&mut store, &std_component)
                    .await
                    .expect("Failed to instantiate");
            });
        });
    });

    group.bench_function("pooling_engine", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut store = make_store(&pool_engine);
                let _instance = pool_linker
                    .instantiate_async(&mut store, &pool_component)
                    .await
                    .expect("Failed to instantiate");
            });
        });
    });

    group.finish();
}

/// Scenario 5: LRU cache lookup overhead with Vec<u8> keys.
/// Measures the cost of hashing large byte slices for cache lookups.
fn bench_lru_cache_lookup_overhead(c: &mut Criterion) {
    use std::num::NonZeroUsize;

    use lru::LruCache;

    let engine = make_pooling_engine();
    let wasm_bytes = wasmtime::wat::parse_str(MINIMAL_COMPONENT_WAT).expect("Failed to parse WAT");
    let component = Component::new(&engine, &wasm_bytes).expect("Failed to compile component");

    let cache_capacity = NonZeroUsize::new(10).unwrap();
    let mut cache: LruCache<Vec<u8>, Component> = LruCache::new(cache_capacity);
    cache.push(wasm_bytes.clone(), component);

    // Simulate different key sizes to show scaling
    let mut group = c.benchmark_group("wasm_lru_cache_lookup");

    // Small key (our minimal WAT)
    let small_key = wasm_bytes.clone();
    group.bench_function("small_key_lookup", |b| {
        b.iter(|| {
            let _ = cache.get(&small_key);
        });
    });

    // Simulate a realistic 500KB WASM module key
    let large_key: Vec<u8> = vec![0u8; 500_000];
    group.bench_function("500kb_key_lookup_miss", |b| {
        b.iter(|| {
            let _ = cache.get(&large_key);
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(100);
    targets =
        bench_cold_compile_and_instantiate,
        bench_cached_component_instantiate,
        bench_store_creation_only,
        bench_standard_vs_pooling,
        bench_lru_cache_lookup_overhead
}
criterion_main!(benches);
