//! Benchmarks for mesh state serialization.
//!
//! Measures serialization/deserialization performance and output sizes for
//! the mesh CRDT store layer. These benchmarks establish a baseline before
//! switching from JSON to bincode.
//!
//! Run with: cargo bench --bench mesh_serialization -p smg-mesh
//!
//! For quick summary: cargo bench --bench mesh_serialization -p smg-mesh -- benchmark_summary --exact
#![expect(
    clippy::unwrap_used,
    clippy::print_stderr,
    reason = "benchmark code: panicking on setup failure is expected, eprintln used for benchmark output"
)]

use std::{collections::BTreeMap, hint::black_box, sync::Mutex};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use smg_mesh::{TreeInsertOp, TreeKey, TreeOperation, TreeState};

// ═══════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════

/// Test configurations: (operations, tokens_per_op, label)
///
/// Production reality:
/// - MAX_TREE_OPERATIONS = 2048 (compacts to 1024)
/// - Real prompts: 1000-8000 tokens per operation
/// - Multiple models: 1-20 models per gateway
/// - Total wire payload = sum of all models' PolicyState entries
const TEST_CONFIGS: [(usize, usize, &str); 6] = [
    (64, 100, "64ops_100tok"),
    (256, 500, "256ops_500tok"),
    (512, 1000, "512ops_1000tok"),
    (1024, 1000, "1024ops_1000tok"),
    (1024, 4000, "1024ops_4000tok"),
    (2048, 4000, "2048ops_4000tok"),
];

/// Multi-model configs: (num_models, ops_per_model, tokens_per_op, label)
/// Simulates total PolicyStore wire payload across multiple models
const MULTI_MODEL_CONFIGS: [(usize, usize, usize, &str); 3] = [
    (3, 1024, 2000, "3models_1024ops_2000tok"),
    (5, 1024, 4000, "5models_1024ops_4000tok"),
    (10, 1024, 4000, "10models_1024ops_4000tok"),
];

/// Mesh sync interval (seconds) — used for throughput calculations
const SYNC_INTERVAL_SECS: u64 = 1;

/// Max message size limit (bytes) — same as flow_control::MAX_MESSAGE_SIZE
const MAX_MESSAGE_SIZE: usize = 10 * 1024 * 1024;

// ═══════════════════════════════════════════════════════════════════
// Results collection for summary table
// ═══════════════════════════════════════════════════════════════════

lazy_static::lazy_static! {
    static ref RESULTS: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());
}

fn add_result(key: &str, value: String) {
    let mut results = RESULTS.lock().unwrap();
    let index = results.len();
    results.insert(format!("{index:03}_{key}"), value);
}

// ═══════════════════════════════════════════════════════════════════
// Fixture generators
// ═══════════════════════════════════════════════════════════════════

/// Generate a realistic TreeState with token-based operations.
/// Uses a deterministic seed for reproducible benchmark results across runs.
fn make_tree_state(model_id: &str, num_ops: usize, tokens_per_op: usize) -> TreeState {
    let mut rng = StdRng::seed_from_u64(42);
    let mut state = TreeState::new(model_id.to_string());
    for i in 0..num_ops {
        let tokens: Vec<u32> = (0..tokens_per_op)
            .map(|_| rng.random_range(0..50000u32))
            .collect();
        let op = TreeOperation::Insert(TreeInsertOp {
            key: TreeKey::Tokens(tokens),
            tenant: format!("http://worker-{i}:8000"),
        });
        state.add_operation(op);
    }
    state
}

/// PolicyState mirrors the mesh store type to avoid depending on internal
/// module structure. Keep in sync with `crates/mesh/src/stores.rs::PolicyState`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PolicyState {
    model_id: String,
    policy_type: String,
    config: Vec<u8>,
    version: u64,
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Benchmark: TreeState serialization
// ═══════════════════════════════════════════════════════════════════

fn bench_tree_state_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_state_serialize");

    for (ops, tokens, label) in TEST_CONFIGS {
        let state = make_tree_state("test-model", ops, tokens);
        let json_size = serde_json::to_vec(&state).unwrap().len();
        let bincode_size = bincode::serialize(&state).unwrap().len();

        group.bench_with_input(BenchmarkId::new("json", label), &state, |b, state| {
            b.iter(|| {
                let bytes = serde_json::to_vec(black_box(state)).unwrap();
                black_box(bytes);
            });
        });

        group.bench_with_input(BenchmarkId::new("bincode", label), &state, |b, state| {
            b.iter(|| {
                let bytes = bincode::serialize(black_box(state)).unwrap();
                black_box(bytes);
            });
        });

        // Sizes reported in summary table
        let _ = (json_size, bincode_size);
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Benchmark: TreeState deserialization
// ═══════════════════════════════════════════════════════════════════

fn bench_tree_state_deserialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_state_deserialize");

    for (ops, tokens, label) in TEST_CONFIGS {
        let state = make_tree_state("test-model", ops, tokens);
        let json_bytes = serde_json::to_vec(&state).unwrap();
        let bincode_bytes = bincode::serialize(&state).unwrap();

        group.bench_with_input(BenchmarkId::new("json", label), &json_bytes, |b, bytes| {
            b.iter(|| {
                let state: TreeState = serde_json::from_slice(black_box(bytes)).unwrap();
                black_box(state);
            });
        });

        group.bench_with_input(
            BenchmarkId::new("bincode", label),
            &bincode_bytes,
            |b, bytes| {
                b.iter(|| {
                    let state: TreeState = bincode::deserialize(black_box(bytes)).unwrap();
                    black_box(state);
                });
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Benchmark: Full wire path (TreeState → PolicyState → wire bytes)
// ═══════════════════════════════════════════════════════════════════

fn bench_policy_state_full_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_state_full_path");

    for (ops, tokens, label) in [(256, 500, "256ops_500tok"), (1024, 1000, "1024ops_1000tok")] {
        let tree_state = make_tree_state("test-model", ops, tokens);

        // JSON → JSON (old behavior)
        group.bench_with_input(
            BenchmarkId::new("json_json", label),
            &tree_state,
            |b, ts| {
                b.iter(|| {
                    let inner = serde_json::to_vec(black_box(ts)).unwrap();
                    let ps = PolicyState {
                        model_id: "test-model".to_string(),
                        policy_type: "tree_state".to_string(),
                        config: inner,
                        version: 1,
                    };
                    let outer = serde_json::to_vec(&ps).unwrap();
                    black_box(outer);
                });
            },
        );

        // bincode → bincode (new behavior)
        group.bench_with_input(
            BenchmarkId::new("bincode_bincode", label),
            &tree_state,
            |b, ts| {
                b.iter(|| {
                    let inner = bincode::serialize(black_box(ts)).unwrap();
                    let ps = PolicyState {
                        model_id: "test-model".to_string(),
                        policy_type: "tree_state".to_string(),
                        config: inner,
                        version: 1,
                    };
                    let outer = bincode::serialize(&ps).unwrap();
                    black_box(outer);
                });
            },
        );

        // Wire path sizes reported in summary table
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Benchmark: Multi-model aggregate wire payload
// ═══════════════════════════════════════════════════════════════════

fn bench_multi_model(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_model_aggregate");
    group.sample_size(10);

    for (num_models, ops, tokens, label) in MULTI_MODEL_CONFIGS {
        // Build all model TreeStates
        let states: Vec<TreeState> = (0..num_models)
            .map(|i| make_tree_state(&format!("model-{i}"), ops, tokens))
            .collect();

        // Simulate full PolicyStore serialization (all models serialized per sync cycle)
        group.bench_with_input(
            BenchmarkId::new("json_total", label),
            &states,
            |b, states| {
                b.iter(|| {
                    let mut total = 0usize;
                    for ts in black_box(states) {
                        let inner = serde_json::to_vec(ts).unwrap();
                        let ps = PolicyState {
                            model_id: ts.model_id.clone(),
                            policy_type: "tree_state".to_string(),
                            config: inner,
                            version: 1,
                        };
                        let wire = serde_json::to_vec(&ps).unwrap();
                        total += wire.len();
                    }
                    black_box(total);
                });
            },
        );

        // Compute total wire size
        let mut total_wire_size = 0usize;
        for ts in &states {
            let inner = serde_json::to_vec(ts).unwrap();
            let ps = PolicyState {
                model_id: ts.model_id.clone(),
                policy_type: "tree_state".to_string(),
                config: inner,
                version: 1,
            };
            total_wire_size += serde_json::to_vec(&ps).unwrap().len();
        }

        // Multi-model sizes reported in summary table
        let _ = total_wire_size;
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Summary benchmark — prints formatted results table
// ═══════════════════════════════════════════════════════════════════

fn bench_summary(c: &mut Criterion) {
    use std::time::Instant;

    let mut group = c.benchmark_group("benchmark_summary");
    group.sample_size(10);

    // ── Comprehensive comparison table with timing ──
    for (ops, tokens, label) in TEST_CONFIGS {
        let state = make_tree_state("test-model", ops, tokens);

        // Size comparison
        let json_bytes = serde_json::to_vec(&state).unwrap();
        let bincode_bytes = bincode::serialize(&state).unwrap();

        // Full wire path sizes
        let json_ps = PolicyState {
            model_id: "test-model".to_string(),
            policy_type: "tree_state".to_string(),
            config: json_bytes.clone(),
            version: 1,
        };
        let json_wire_size = serde_json::to_vec(&json_ps).unwrap().len();

        let bincode_ps = PolicyState {
            model_id: "test-model".to_string(),
            policy_type: "tree_state".to_string(),
            config: bincode_bytes.clone(),
            version: 1,
        };
        let bincode_wire_size = bincode::serialize(&bincode_ps).unwrap().len();

        // Manual timing — scale iterations inversely with payload size
        let iters = (100_000 / (ops * tokens).max(1)).clamp(5, 100);

        let start = Instant::now();
        for _ in 0..iters {
            black_box(serde_json::to_vec(black_box(&state)).unwrap());
        }
        let json_ser_us = start.elapsed().as_micros() as f64 / iters as f64;

        let start = Instant::now();
        for _ in 0..iters {
            black_box(bincode::serialize(black_box(&state)).unwrap());
        }
        let bin_ser_us = start.elapsed().as_micros() as f64 / iters as f64;

        let start = Instant::now();
        for _ in 0..iters {
            let _: TreeState = black_box(serde_json::from_slice(black_box(&json_bytes))).unwrap();
        }
        let json_de_us = start.elapsed().as_micros() as f64 / iters as f64;

        let start = Instant::now();
        for _ in 0..iters {
            let _: TreeState = black_box(bincode::deserialize(black_box(&bincode_bytes))).unwrap();
        }
        let bin_de_us = start.elapsed().as_micros() as f64 / iters as f64;

        add_result(
            "comparison",
            format!(
                "{:>18} | {:>10} {:>10} {:>5.1}x | {:>10} {:>10} {:>5.1}x | {:>10} {:>10} {:>5.1}x",
                label,
                format_size(json_bytes.len()),
                format_size(bincode_bytes.len()),
                json_bytes.len() as f64 / bincode_bytes.len() as f64,
                format_time(json_ser_us),
                format_time(bin_ser_us),
                json_ser_us / bin_ser_us,
                format_time(json_de_us),
                format_time(bin_de_us),
                json_de_us / bin_de_us,
            ),
        );

        add_result(
            "wire",
            format!(
                "{:>18} | JSON wire {:>10} | bincode wire {:>10} | {:>5.1}x smaller",
                label,
                format_size(json_wire_size),
                format_size(bincode_wire_size),
                json_wire_size as f64 / bincode_wire_size as f64,
            ),
        );

        // No-op to register in Criterion
        group.bench_function(BenchmarkId::new("summary", label), |b| {
            b.iter(|| black_box(json_bytes.len()));
        });
    }

    // Multi-model aggregate
    for (num_models, ops, tokens, label) in MULTI_MODEL_CONFIGS {
        let states: Vec<TreeState> = (0..num_models)
            .map(|i| make_tree_state(&format!("model-{i}"), ops, tokens))
            .collect();

        let mut json_total = 0usize;
        let mut bincode_total = 0usize;
        for ts in &states {
            let inner_json = serde_json::to_vec(ts).unwrap();
            let ps = PolicyState {
                model_id: ts.model_id.clone(),
                policy_type: "tree_state".to_string(),
                config: inner_json,
                version: 1,
            };
            json_total += serde_json::to_vec(&ps).unwrap().len();

            let inner_bin = bincode::serialize(ts).unwrap();
            let bin_ps = PolicyState {
                model_id: ts.model_id.clone(),
                policy_type: "tree_state".to_string(),
                config: inner_bin,
                version: 1,
            };
            bincode_total += bincode::serialize(&bin_ps).unwrap().len();
        }

        add_result(
            "multi",
            format!(
                "{:>30} | {:>3} models | JSON {:>10} | bincode {:>10} | {:>5.1}x",
                label,
                num_models,
                format_size(json_total),
                format_size(bincode_total),
                json_total as f64 / bincode_total as f64,
            ),
        );
    }

    group.finish();
    print_summary();
}

fn format_time(us: f64) -> String {
    if us >= 1000.0 {
        format!("{:.1}ms", us / 1000.0)
    } else {
        format!("{us:.0}µs")
    }
}

fn print_summary() {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();

    let results = RESULTS.lock().unwrap();

    let mut comparison_results = Vec::new();
    let mut wire_results = Vec::new();
    let mut multi_results = Vec::new();

    for (key, value) in results.iter() {
        let category = key.split('_').skip(1).collect::<Vec<_>>().join("_");
        match category.as_str() {
            "comparison" => comparison_results.push(value.clone()),
            "wire" | "wire_path" => wire_results.push(value.clone()),
            "multi" | "multi_model" => multi_results.push(value.clone()),
            _ => {}
        }
    }

    eprintln!();
    eprintln!("{}", "═".repeat(120));
    eprintln!("MESH SERIALIZATION BENCHMARK — JSON vs bincode");
    eprintln!("{}", "═".repeat(120));
    eprintln!();
    eprintln!("Configuration:");
    eprintln!("  Sync interval:    {SYNC_INTERVAL_SECS} second(s)");
    let max_mb = MAX_MESSAGE_SIZE / (1024 * 1024);
    eprintln!("  Max message size: {max_mb}MB");
    eprintln!("  RNG seed:         42 (deterministic)");
    eprintln!();

    if !comparison_results.is_empty() {
        eprintln!("{}", "─".repeat(120));
        eprintln!(
            "{:>18} | {:>10} {:>10} {:>5} | {:>10} {:>10} {:>5} | {:>10} {:>10} {:>5}",
            "Config",
            "JSON size",
            "bin size",
            "ratio",
            "JSON ser",
            "bin ser",
            "ratio",
            "JSON de",
            "bin de",
            "ratio",
        );
        eprintln!("{}", "─".repeat(120));
        for v in &comparison_results {
            eprintln!("{v}");
        }
        eprintln!();
    }

    if !wire_results.is_empty() {
        eprintln!("{}", "─".repeat(120));
        eprintln!("FULL WIRE PATH (TreeState → PolicyState.config → wire bytes)");
        eprintln!("{}", "─".repeat(120));
        for v in &wire_results {
            eprintln!("{v}");
        }
        eprintln!();
    }

    if !multi_results.is_empty() {
        eprintln!("{}", "─".repeat(120));
        eprintln!("MULTI-MODEL AGGREGATE (total per sync cycle)");
        eprintln!("{}", "─".repeat(120));
        for v in &multi_results {
            eprintln!("{v}");
        }
        eprintln!();
    }

    eprintln!("{}", "═".repeat(120));
}

criterion_group!(
    benches,
    bench_tree_state_serialization,
    bench_tree_state_deserialization,
    bench_policy_state_full_path,
    bench_multi_model,
    bench_summary,
);
criterion_main!(benches);
