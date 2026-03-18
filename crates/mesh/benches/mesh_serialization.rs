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

/// Generate a TreeState with text-based operations (smaller payloads).
fn make_text_tree_state(model_id: &str, num_ops: usize) -> TreeState {
    let mut state = TreeState::new(model_id.to_string());
    for i in 0..num_ops {
        let op = TreeOperation::Insert(TreeInsertOp {
            key: TreeKey::Text(format!(
                "You are a helpful assistant. User query #{i}: Tell me about topic {i}"
            )),
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

        group.bench_with_input(BenchmarkId::new("json", label), &state, |b, state| {
            b.iter(|| {
                let bytes = serde_json::to_vec(black_box(state)).unwrap();
                black_box(bytes);
            });
        });

        add_result(
            "serialize",
            format!(
                "{:>18} | {:>10} | {:>10} | {:>8}",
                label,
                format_size(json_size),
                if json_size > MAX_MESSAGE_SIZE {
                    "EXCEEDS"
                } else {
                    "OK"
                },
                format!("{ops}×{tokens}"),
            ),
        );
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

        group.bench_with_input(BenchmarkId::new("json", label), &json_bytes, |b, bytes| {
            b.iter(|| {
                let state: TreeState = serde_json::from_slice(black_box(bytes)).unwrap();
                black_box(state);
            });
        });
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

        let inner = serde_json::to_vec(&tree_state).unwrap();
        let inner_size = inner.len();
        let ps = PolicyState {
            model_id: "test-model".to_string(),
            policy_type: "tree_state".to_string(),
            config: inner,
            version: 1,
        };
        let wire_size = serde_json::to_vec(&ps).unwrap().len();

        add_result(
            "wire_path",
            format!(
                "{:>18} | {:>10} → {:>10} | {:>5.1}x blowup | {}",
                label,
                format_size(inner_size),
                format_size(wire_size),
                wire_size as f64 / inner_size as f64,
                if wire_size > MAX_MESSAGE_SIZE {
                    "⚠ EXCEEDS 10MB LIMIT"
                } else {
                    "OK"
                },
            ),
        );
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

        add_result(
            "multi_model",
            format!(
                "{:>30} | {:>3} models | {:>10} total wire | {}",
                label,
                num_models,
                format_size(total_wire_size),
                if total_wire_size > MAX_MESSAGE_SIZE {
                    "⚠ EXCEEDS 10MB LIMIT"
                } else {
                    "OK"
                },
            ),
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Summary benchmark — prints formatted results table
// ═══════════════════════════════════════════════════════════════════

fn bench_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("benchmark_summary");
    group.sample_size(10);

    // Compute all sizes for the summary
    for (ops, tokens, label) in TEST_CONFIGS {
        let state = make_tree_state("test-model", ops, tokens);
        let json_inner = serde_json::to_vec(&state).unwrap();
        let ps = PolicyState {
            model_id: "test-model".to_string(),
            policy_type: "tree_state".to_string(),
            config: json_inner.clone(),
            version: 1,
        };
        let json_wire = serde_json::to_vec(&ps).unwrap();

        add_result(
            "size",
            format!(
                "{:>18} | TreeState: {:>10} | PolicyState (wire): {:>10} | {:>5.1}x | {}",
                label,
                format_size(json_inner.len()),
                format_size(json_wire.len()),
                json_wire.len() as f64 / json_inner.len() as f64,
                if json_wire.len() > MAX_MESSAGE_SIZE {
                    "⚠ EXCEEDS LIMIT"
                } else {
                    "OK"
                },
            ),
        );

        // No-op benchmark to register in Criterion output
        group.bench_function(BenchmarkId::new("size_kb", label), |b| {
            b.iter(|| black_box(json_inner.len()));
        });
    }

    // Text-based comparison
    let text_state = make_text_tree_state("test-model", 256);
    let json_text = serde_json::to_vec(&text_state).unwrap();
    add_result(
        "size",
        format!(
            "{:>18} | TreeState: {:>10} | (text keys, no token blowup)",
            "256ops_text",
            format_size(json_text.len()),
        ),
    );

    group.finish();

    // Print summary
    print_summary();
}

fn print_summary() {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();

    let results = RESULTS.lock().unwrap();

    let mut serialize_results = Vec::new();
    let mut wire_results = Vec::new();
    let mut size_results = Vec::new();
    let mut multi_model_results = Vec::new();

    for (key, value) in results.iter() {
        let category = key.split('_').skip(1).collect::<Vec<_>>().join("_");
        match category.as_str() {
            "serialize" => serialize_results.push(value.clone()),
            "wire_path" | "wire" => wire_results.push(value.clone()),
            "size" => size_results.push(value.clone()),
            "multi_model" | "multi" => multi_model_results.push(value.clone()),
            _ => {}
        }
    }

    eprintln!();
    eprintln!("{}", "═".repeat(90));
    eprintln!("MESH SERIALIZATION BENCHMARK");
    eprintln!("{}", "═".repeat(90));
    eprintln!();
    eprintln!("Configuration:");
    eprintln!("  Sync interval:    {SYNC_INTERVAL_SECS} second(s)");
    let max_mb = MAX_MESSAGE_SIZE / (1024 * 1024);
    eprintln!("  Max message size: {max_mb}MB");
    eprintln!("  Serialization:    serde_json (current)");
    eprintln!("  RNG seed:         42 (deterministic)");
    eprintln!();

    if !serialize_results.is_empty() {
        eprintln!("{}", "─".repeat(90));
        eprintln!("TREESTATE SERIALIZATION (JSON)");
        eprintln!("{}", "─".repeat(90));
        eprintln!(
            "{:>18} | {:>10} | {:>10} | {:>8}",
            "Config", "JSON Size", "Limit", "Ops"
        );
        eprintln!("{}", "─".repeat(90));
        for v in &serialize_results {
            eprintln!("{v}");
        }
        eprintln!();
    }

    if !wire_results.is_empty() {
        eprintln!("{}", "─".repeat(90));
        eprintln!("FULL WIRE PATH: TreeState → PolicyState.config → JSON wire");
        eprintln!("{}", "─".repeat(90));
        for v in &wire_results {
            eprintln!("{v}");
        }
        eprintln!();
    }

    if !size_results.is_empty() {
        eprintln!("{}", "─".repeat(90));
        eprintln!("SIZE SUMMARY (single model)");
        eprintln!("{}", "─".repeat(90));
        for v in &size_results {
            eprintln!("{v}");
        }
        eprintln!();
    }

    if !multi_model_results.is_empty() {
        eprintln!("{}", "─".repeat(90));
        eprintln!("MULTI-MODEL AGGREGATE (total PolicyStore wire per sync cycle)");
        eprintln!("{}", "─".repeat(90));
        for v in &multi_model_results {
            eprintln!("{v}");
        }
        eprintln!();
    }

    eprintln!("{}", "═".repeat(90));
    eprintln!("Key insight: PolicyState JSON is ~3x larger than TreeState JSON because");
    eprintln!("Vec<u8> serializes as a JSON array of decimal integers [240, 159, ...]");
    eprintln!("With 10 models × 4000-token prompts, total wire payload reaches ~500MB+.");
    eprintln!("{}", "═".repeat(90));
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
