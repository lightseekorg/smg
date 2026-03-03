//! Benchmarks for the radix tree implementations used in cache-aware routing.
//!
//! This benchmark tests all three implementations:
//! - StringTree: Character-based tree for HTTP router (text input)
//! - TokenTree: Token-based tree for gRPC router (pre-tokenized input)
//! - PositionalIndexer: Event-driven indexer for gRPC router (KV cache events)
//!
//! Run with: cargo bench --bench radix_tree_benchmark
//!
//! For quick validation: cargo bench --bench radix_tree_benchmark -- benchmark_summary --exact
#![expect(
    clippy::unwrap_used,
    clippy::print_stderr,
    reason = "benchmark code: panicking on setup failure is expected, eprintln used for benchmark output"
)]

use std::{
    collections::BTreeMap,
    hint::black_box,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Instant,
};

use criterion::{criterion_group, criterion_main, Criterion};
use kv_index::{
    compute_content_hash, compute_request_content_hashes, ContentHash, PositionalIndexer,
    SequenceHash, StoredBlock, StringTree, TokenTree,
};
use rand::{
    distr::{Alphanumeric, SampleString},
    rng as thread_rng, Rng,
};

// Global results storage for summary
lazy_static::lazy_static! {
    static ref BENCHMARK_RESULTS: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());
}

fn add_result(category: &str, result: String) {
    let mut results = BENCHMARK_RESULTS.lock().unwrap();
    let index = results.len();
    let key = format!("{index:03}_{category}");
    results.insert(key, result);
}

/// Common conversation prefixes that create shared tree paths
const CONVERSATION_PREFIXES: [&str; 6] = [
    "<|system|>\nYou are a helpful assistant.\n<|user|>\n",
    "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n",
    "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n",
    "Human: ",
    "User: ",
    "### Instruction:\n",
];

/// Token ID type
type TokenId = u32;

/// Generate random ASCII strings of given length
fn random_ascii_string(len: usize) -> String {
    Alphanumeric.sample_string(&mut thread_rng(), len)
}

/// Generate fixed-size strings for benchmarks
fn generate_fixed_char_strings(count: usize, char_len: usize) -> Vec<String> {
    let mut rng = thread_rng();
    (0..count)
        .map(|_| {
            let prefix_idx = rng.random_range(0..CONVERSATION_PREFIXES.len());
            let prefix = CONVERSATION_PREFIXES[prefix_idx];
            let remaining = char_len.saturating_sub(prefix.len());
            format!("{}{}", prefix, random_ascii_string(remaining))
        })
        .collect()
}

/// Generate fixed-size token sequences for size-specific benchmarks
fn generate_fixed_token_sequences(count: usize, token_len: usize) -> Vec<Vec<TokenId>> {
    let mut rng = thread_rng();
    (0..count)
        .map(|_| (0..token_len).map(|_| rng.random_range(0..50000)).collect())
        .collect()
}

/// Generate worker endpoint URLs for scaling tests
fn generate_worker_endpoints(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| {
            if i % 4 == 0 {
                format!("grpc://worker-{i}.sglang.svc.cluster.local:50051")
            } else {
                format!("http://worker-{i}.sglang.svc.cluster.local:8000")
            }
        })
        .collect()
}

// ============================================================================
// Benchmark Macros
// ============================================================================

/// Macro for INSERT benchmarks with string-based trees (StringTree)
macro_rules! bench_string_insert {
    ($group:expr, $num_workers:expr, $char_size:expr, $workers:expr, $strings:expr,
     $prefix:literal, $category:literal, $tree_new:expr, $insert_method:ident) => {{
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!(concat!($prefix, "_insert_{}w_{}c"), $num_workers, $char_size);
        let workers_clone = $workers.clone();
        let strings_clone = $strings.clone();

        $group.bench_function(&bench_name, |b| {
            let workers = workers_clone.clone();
            let strings = strings_clone.clone();
            let printed = printed.clone();

            b.iter_custom(|iters| {
                let tree = $tree_new;
                let start = Instant::now();
                for i in 0..iters {
                    let tenant = &workers[i as usize % workers.len()];
                    let text = &strings[i as usize % strings.len()];
                    tree.$insert_method(black_box(text), tenant);
                }
                let duration = start.elapsed();

                if !printed.swap(true, Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let latency_us = duration.as_nanos() as f64 / iters as f64 / 1000.0;
                    let throughput_mb = (ops_per_sec * $char_size as f64) / 1_000_000.0;
                    add_result(
                        $category,
                        format!(
                            "{:>3}w | {:>5} chars | INSERT | {:>8.0} ops/s | {:>6.1} µs | {:>7.1} MB/s",
                            $num_workers, $char_size, ops_per_sec, latency_us, throughput_mb
                        ),
                    );
                }

                duration
            });
        });
    }};
}

/// Macro for MATCH benchmarks with string-based trees
macro_rules! bench_string_match {
    ($group:expr, $num_workers:expr, $char_size:expr, $tree:expr, $strings:expr,
     $prefix:literal, $category:literal, $match_method:ident) => {{
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!(concat!($prefix, "_match_{}w_{}c"), $num_workers, $char_size);
        let tree_clone = $tree.clone();
        let strings_clone = $strings.clone();

        $group.bench_function(&bench_name, |b| {
            let tree = tree_clone.clone();
            let strings = strings_clone.clone();
            let mut idx = 0;
            let printed = printed.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let result = tree.$match_method(black_box(&strings[idx % strings.len()]));
                    black_box(result);
                    idx += 1;
                }
                let duration = start.elapsed();

                if !printed.swap(true, Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let latency_us = duration.as_nanos() as f64 / iters as f64 / 1000.0;
                    let throughput_mb = (ops_per_sec * $char_size as f64) / 1_000_000.0;
                    add_result(
                        $category,
                        format!(
                            "{:>3}w | {:>5} chars | MATCH  | {:>8.0} ops/s | {:>6.1} µs | {:>7.1} MB/s",
                            $num_workers, $char_size, ops_per_sec, latency_us, throughput_mb
                        ),
                    );
                }

                duration
            });
        });
    }};
}

/// Macro for CONCURRENT benchmarks with string-based trees
macro_rules! bench_string_concurrent {
    ($group:expr, $num_workers:expr, $workers:expr, $num_threads:expr, $ops_per_thread:expr,
     $prefix:literal, $category:literal, $tree_new:expr, $insert_method:ident, $match_method:ident) => {{
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!(concat!($prefix, "_concurrent_{}w"), $num_workers);
        let workers_clone = $workers.clone();

        $group.bench_function(&bench_name, |b| {
            let printed = printed.clone();
            let workers = workers_clone.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let tree = Arc::new($tree_new);
                    let workers_ref = &workers;
                    let handles: Vec<_> = (0..$num_threads)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let worker = workers_ref[t % workers_ref.len()].clone();
                            thread::spawn(move || {
                                for i in 0..$ops_per_thread {
                                    let text = format!(
                                        "{}thread{}_request{}_padding{}",
                                        CONVERSATION_PREFIXES[i % CONVERSATION_PREFIXES.len()],
                                        t,
                                        i,
                                        "x".repeat(1000)
                                    );
                                    if i % 3 == 0 {
                                        tree.$match_method(&text);
                                    } else {
                                        tree.$insert_method(&text, &worker);
                                    }
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                }
                let duration = start.elapsed();

                if !printed.swap(true, Ordering::Relaxed) {
                    let total_ops = iters * $num_threads as u64 * $ops_per_thread as u64;
                    let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                    add_result(
                        $category,
                        format!(
                            "{:>3}w | CONCURRENT | {:>7.0} ops/s | {} threads x {} ops",
                            $num_workers, ops_per_sec, $num_threads, $ops_per_thread
                        ),
                    );
                }

                duration
            });
        });
    }};
}

/// Macro for INSERT benchmarks with TokenTree
macro_rules! bench_token_insert {
    ($group:expr, $num_workers:expr, $token_size:expr, $workers:expr, $sequences:expr) => {{
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!("token_insert_{}w_{}tok", $num_workers, $token_size);
        let workers_clone = $workers.clone();
        let seqs_clone = $sequences.clone();

        $group.bench_function(&bench_name, |b| {
            let workers = workers_clone.clone();
            let seqs = seqs_clone.clone();
            let printed = printed.clone();

            b.iter_custom(|iters| {
                let tree = TokenTree::new();
                let start = Instant::now();
                for i in 0..iters {
                    let tenant = &workers[i as usize % workers.len()];
                    let tokens = &seqs[i as usize % seqs.len()];
                    tree.insert_tokens(black_box(tokens), tenant);
                }
                let duration = start.elapsed();

                if !printed.swap(true, Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let latency_us = duration.as_nanos() as f64 / iters as f64 / 1000.0;
                    let throughput_mtok = (ops_per_sec * $token_size as f64) / 1_000_000.0;
                    add_result(
                        "token",
                        format!(
                            "{:>3}w | {:>5} tokens | INSERT | {:>8.0} ops/s | {:>6.1} µs | {:>6.1} Mtok/s",
                            $num_workers, $token_size, ops_per_sec, latency_us, throughput_mtok
                        ),
                    );
                }

                duration
            });
        });
    }};
}

/// Macro for MATCH benchmarks with TokenTree
macro_rules! bench_token_match {
    ($group:expr, $num_workers:expr, $token_size:expr, $tree:expr, $sequences:expr) => {{
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!("token_match_{}w_{}tok", $num_workers, $token_size);
        let tree_clone = $tree.clone();
        let seqs_clone = $sequences.clone();

        $group.bench_function(&bench_name, |b| {
            let tree = tree_clone.clone();
            let seqs = seqs_clone.clone();
            let mut idx = 0;
            let printed = printed.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let result = tree.prefix_match_legacy(black_box(&seqs[idx % seqs.len()]));
                    black_box(result);
                    idx += 1;
                }
                let duration = start.elapsed();

                if !printed.swap(true, Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let latency_us = duration.as_nanos() as f64 / iters as f64 / 1000.0;
                    let throughput_mtok = (ops_per_sec * $token_size as f64) / 1_000_000.0;
                    add_result(
                        "token",
                        format!(
                            "{:>3}w | {:>5} tokens | MATCH  | {:>8.0} ops/s | {:>6.1} µs | {:>6.1} Mtok/s",
                            $num_workers, $token_size, ops_per_sec, latency_us, throughput_mtok
                        ),
                    );
                }

                duration
            });
        });
    }};
}

/// Macro for CONCURRENT benchmarks with TokenTree
macro_rules! bench_token_concurrent {
    ($group:expr, $num_workers:expr, $workers:expr, $num_threads:expr, $ops_per_thread:expr) => {{
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!("token_concurrent_{}w", $num_workers);
        let workers_clone = $workers.clone();

        $group.bench_function(&bench_name, |b| {
            let printed = printed.clone();
            let workers = workers_clone.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let tree = Arc::new(TokenTree::new());
                    let workers_ref = &workers;
                    let handles: Vec<_> = (0..$num_threads)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let worker = workers_ref[t % workers_ref.len()].clone();
                            thread::spawn(move || {
                                for i in 0..$ops_per_thread {
                                    // Generate deterministic token sequence
                                    let tokens: Vec<TokenId> = (0..1024)
                                        .map(|j| (t * 10000 + i * 100 + j) as u32)
                                        .collect();
                                    if i % 3 == 0 {
                                        tree.prefix_match_legacy(&tokens);
                                    } else {
                                        tree.insert_tokens(&tokens, &worker);
                                    }
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                }
                let duration = start.elapsed();

                if !printed.swap(true, Ordering::Relaxed) {
                    let total_ops = iters * $num_threads as u64 * $ops_per_thread as u64;
                    let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                    add_result(
                        "token",
                        format!(
                            "{:>3}w | CONCURRENT | {:>7.0} ops/s | {} threads x {} ops",
                            $num_workers, ops_per_sec, $num_threads, $ops_per_thread
                        ),
                    );
                }

                duration
            });
        });
    }};
}

// ============================================================================
// PositionalIndexer Helpers
// ============================================================================

/// Generate token chunks of `block_size` tokens each.
fn generate_token_chunks(num_blocks: usize, block_size: usize) -> Vec<Vec<TokenId>> {
    let mut rng = thread_rng();
    (0..num_blocks)
        .map(|_| {
            (0..block_size)
                .map(|_| rng.random_range(0..50000))
                .collect()
        })
        .collect()
}

fn chunks_to_stored_blocks(chunks: &[Vec<TokenId>]) -> Vec<StoredBlock> {
    chunks
        .iter()
        .enumerate()
        .map(|(i, tokens)| StoredBlock {
            seq_hash: SequenceHash(i as u64 + 1),
            content_hash: compute_content_hash(tokens),
        })
        .collect()
}

fn flatten_tokens(chunks: &[Vec<TokenId>]) -> Vec<TokenId> {
    chunks.iter().flat_map(|c| c.iter().copied()).collect()
}

/// Build a populated indexer. Workers share `shared_prefix_blocks` initial blocks
/// (simulating common system prompt) then diverge.
fn build_populated_indexer(
    workers: &[String],
    blocks_per_worker: usize,
    block_size: usize,
    shared_prefix_blocks: usize,
    jump_size: usize,
) -> (Arc<PositionalIndexer>, Vec<Vec<Vec<TokenId>>>) {
    let indexer = Arc::new(PositionalIndexer::new(jump_size));

    let shared_chunks = generate_token_chunks(shared_prefix_blocks, block_size);
    let shared_blocks = chunks_to_stored_blocks(&shared_chunks);

    let mut all_worker_chunks = Vec::with_capacity(workers.len());

    for worker in workers {
        indexer.apply_stored(worker, &shared_blocks, None).unwrap();

        let unique_count = blocks_per_worker.saturating_sub(shared_prefix_blocks);
        let unique_chunks = generate_token_chunks(unique_count, block_size);
        let unique_blocks: Vec<StoredBlock> = unique_chunks
            .iter()
            .enumerate()
            .map(|(i, tokens)| StoredBlock {
                seq_hash: SequenceHash((shared_prefix_blocks + i) as u64 + 1),
                content_hash: compute_content_hash(tokens),
            })
            .collect();

        if !unique_blocks.is_empty() {
            let parent = SequenceHash(shared_prefix_blocks as u64);
            indexer
                .apply_stored(worker, &unique_blocks, Some(parent))
                .unwrap();
        }

        let mut worker_chunks = shared_chunks.clone();
        worker_chunks.extend(unique_chunks);
        all_worker_chunks.push(worker_chunks);
    }

    (indexer, all_worker_chunks)
}

// ============================================================================
// PositionalIndexer Benchmark Macros
// ============================================================================

/// Macro for STORE benchmarks with PositionalIndexer
macro_rules! bench_indexer_store {
    ($group:expr, $num_workers:expr, $blocks_per_worker:expr, $block_size:expr, $workers:expr) => {{
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!(
            "indexer_store_{}w_{}blk_{}bs",
            $num_workers, $blocks_per_worker, $block_size
        );
        let workers_clone = $workers.clone();

        $group.bench_function(&bench_name, |b| {
            let workers = workers_clone.clone();
            let printed = printed.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let indexer = PositionalIndexer::new(64);
                    for worker in &workers {
                        let chunks = generate_token_chunks($blocks_per_worker, $block_size);
                        let blocks = chunks_to_stored_blocks(&chunks);
                        let _ = indexer.apply_stored(black_box(worker), black_box(&blocks), None);
                    }
                }
                let duration = start.elapsed();

                if !printed.swap(true, Ordering::Relaxed) {
                    let total_blocks =
                        iters as f64 * $num_workers as f64 * $blocks_per_worker as f64;
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let latency_us = duration.as_nanos() as f64 / iters as f64 / 1000.0;
                    let blocks_per_sec = total_blocks / duration.as_secs_f64();
                    add_result(
                        "indexer",
                        format!(
                            "{:>3}w | {:>4}blk x {:>2}tok | STORE  | {:>8.0} ops/s | {:>7.1} µs | {:>8.0} blk/s",
                            $num_workers, $blocks_per_worker, $block_size, ops_per_sec, latency_us, blocks_per_sec
                        ),
                    );
                }

                duration
            });
        });
    }};
}

/// Macro for MATCH benchmarks with PositionalIndexer (find_matches — the hot path)
macro_rules! bench_indexer_match {
    ($group:expr, $num_workers:expr, $query_blocks:expr, $block_size:expr,
     $indexer:expr, $query_hashes:expr) => {{
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!(
            "indexer_match_{}w_{}qblk_{}bs",
            $num_workers, $query_blocks, $block_size
        );
        let indexer_clone = $indexer.clone();
        let hashes_clone = $query_hashes.clone();

        $group.bench_function(&bench_name, |b| {
            let indexer = indexer_clone.clone();
            let hashes = hashes_clone.clone();
            let mut idx = 0;
            let printed = printed.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let result = indexer.find_matches(black_box(&hashes[idx % hashes.len()]));
                    black_box(result);
                    idx += 1;
                }
                let duration = start.elapsed();

                if !printed.swap(true, Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let latency_us = duration.as_nanos() as f64 / iters as f64 / 1000.0;
                    let throughput_mblk =
                        (ops_per_sec * $query_blocks as f64) / 1_000_000.0;
                    add_result(
                        "indexer",
                        format!(
                            "{:>3}w | {:>4}blk x {:>2}tok | MATCH  | {:>8.0} ops/s | {:>7.1} µs | {:>5.2} Mblk/s",
                            $num_workers, $query_blocks, $block_size, ops_per_sec, latency_us, throughput_mblk
                        ),
                    );
                }

                duration
            });
        });
    }};
}

/// Macro for CONCURRENT benchmarks with PositionalIndexer
macro_rules! bench_indexer_concurrent {
    ($group:expr, $num_workers:expr, $block_size:expr, $num_threads:expr, $ops_per_thread:expr) => {{
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!("indexer_concurrent_{}w", $num_workers);

        $group.bench_function(&bench_name, |b| {
            let printed = printed.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let workers = generate_worker_endpoints($num_workers);
                    let (indexer, worker_chunks) =
                        build_populated_indexer(&workers, 64, $block_size, 8, 64);

                    let handles: Vec<_> = (0..$num_threads)
                        .map(|t| {
                            let indexer = Arc::clone(&indexer);
                            let worker = workers[t % workers.len()].clone();
                            let chunks = worker_chunks[t % workers.len()].clone();
                            let block_size = $block_size;

                            thread::spawn(move || {
                                let mut rng = thread_rng();
                                for i in 0..$ops_per_thread {
                                    if i % 3 == 0 {
                                        // Read: find_matches
                                        let query_tokens = flatten_tokens(&chunks);
                                        let content_hashes = compute_request_content_hashes(
                                            &query_tokens,
                                            block_size,
                                        );
                                        black_box(indexer.find_matches(&content_hashes));
                                    } else {
                                        // Write: apply_stored
                                        let new_chunks = generate_token_chunks(4, block_size);
                                        let blocks = chunks_to_stored_blocks(&new_chunks);
                                        let parent = SequenceHash(rng.random_range(1u64..65));
                                        let _ =
                                            indexer.apply_stored(&worker, &blocks, Some(parent));
                                    }
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                }
                let duration = start.elapsed();

                if !printed.swap(true, Ordering::Relaxed) {
                    let total_ops = iters * $num_threads as u64 * $ops_per_thread as u64;
                    let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                    add_result(
                        "indexer",
                        format!(
                            "{:>3}w | CONCURRENT | {:>7.0} ops/s | {} threads x {} ops",
                            $num_workers, ops_per_sec, $num_threads, $ops_per_thread
                        ),
                    );
                }

                duration
            });
        });
    }};
}

// ============================================================================
// Main Benchmark
// ============================================================================

/// Main benchmark for StringTree, TokenTree, and PositionalIndexer
fn bench_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("benchmark_summary");

    // Reduce warmup and measurement time for faster runs
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(2));
    group.sample_size(50);

    // Configuration constants
    const TREE_SIZE: usize = 2_000;
    const INSERT_POOL_SIZE: usize = 2_000;
    const NUM_THREADS: usize = 32;
    const OPS_PER_THREAD: usize = 100;

    // Worker counts and sizes to test (reduced for faster runs)
    const WORKER_COUNTS: [usize; 3] = [10, 100, 500];
    const TOKEN_SIZES: [usize; 3] = [1024, 4096, 16384];
    const CHAR_SIZES: [usize; 3] = [4096, 16384, 65536];

    // ========================================================================
    // StringTree Benchmark
    // ========================================================================
    for &num_workers in &WORKER_COUNTS {
        let workers = generate_worker_endpoints(num_workers);

        for &char_size in &CHAR_SIZES {
            let fixed_strings = generate_fixed_char_strings(INSERT_POOL_SIZE, char_size);

            // Pre-populate tree for MATCH
            let string_tree = Arc::new(StringTree::new());
            for (i, s) in fixed_strings.iter().take(TREE_SIZE).enumerate() {
                let tenant = &workers[i % workers.len()];
                string_tree.insert_text(s, tenant);
            }

            bench_string_insert!(
                group,
                num_workers,
                char_size,
                workers,
                fixed_strings,
                "string",
                "string",
                StringTree::new(),
                insert_text
            );
            bench_string_match!(
                group,
                num_workers,
                char_size,
                string_tree,
                fixed_strings,
                "string",
                "string",
                prefix_match_legacy
            );
        }

        bench_string_concurrent!(
            group,
            num_workers,
            workers,
            NUM_THREADS,
            OPS_PER_THREAD,
            "string",
            "string",
            StringTree::new(),
            insert_text,
            prefix_match_legacy
        );
    }

    // ========================================================================
    // TokenTree Benchmark
    // ========================================================================
    for &num_workers in &WORKER_COUNTS {
        let workers = generate_worker_endpoints(num_workers);

        for &token_size in &TOKEN_SIZES {
            let fixed_sequences = generate_fixed_token_sequences(INSERT_POOL_SIZE, token_size);

            // Pre-populate tree for MATCH
            let token_tree = Arc::new(TokenTree::new());
            for (i, seq) in fixed_sequences.iter().take(TREE_SIZE).enumerate() {
                let tenant = &workers[i % workers.len()];
                token_tree.insert_tokens(seq, tenant);
            }

            bench_token_insert!(group, num_workers, token_size, workers, fixed_sequences);
            bench_token_match!(group, num_workers, token_size, token_tree, fixed_sequences);
        }

        bench_token_concurrent!(group, num_workers, workers, NUM_THREADS, OPS_PER_THREAD);
    }

    // ========================================================================
    // PositionalIndexer Benchmark
    // ========================================================================
    const BLOCK_SIZES: [usize; 2] = [16, 64];
    const BLOCKS_PER_WORKER: [usize; 3] = [64, 256, 1024];
    const QUERY_BLOCK_COUNTS: [usize; 3] = [32, 128, 512];
    const SHARED_PREFIX_BLOCKS: usize = 8;
    const JUMP_SIZE: usize = 64;

    for &num_workers in &WORKER_COUNTS {
        let workers = generate_worker_endpoints(num_workers);

        for &block_size in &BLOCK_SIZES {
            // STORE benchmarks
            for &blocks_per_worker in &BLOCKS_PER_WORKER {
                bench_indexer_store!(group, num_workers, blocks_per_worker, block_size, workers);
            }

            // MATCH benchmarks: build a populated indexer, then query it
            let max_blocks = *BLOCKS_PER_WORKER.last().unwrap();
            let (indexer, worker_chunks) = build_populated_indexer(
                &workers,
                max_blocks,
                block_size,
                SHARED_PREFIX_BLOCKS,
                JUMP_SIZE,
            );

            for &query_blocks in &QUERY_BLOCK_COUNTS {
                // Generate query hashes: mix of cached (from worker 0) and novel tokens
                let num_queries = 100;
                let query_hashes: Vec<Vec<ContentHash>> = (0..num_queries)
                    .map(|i| {
                        let chunks = &worker_chunks[i % worker_chunks.len()];
                        let take = query_blocks.min(chunks.len());
                        let tokens = flatten_tokens(&chunks[..take]);
                        compute_request_content_hashes(&tokens, block_size)
                    })
                    .collect();

                bench_indexer_match!(
                    group,
                    num_workers,
                    query_blocks,
                    block_size,
                    indexer,
                    query_hashes
                );
            }
        }

        bench_indexer_concurrent!(group, num_workers, 16, NUM_THREADS, OPS_PER_THREAD);
    }

    group.finish();
}

/// Print final summary table
fn print_summary() {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();

    let results = BENCHMARK_RESULTS.lock().unwrap();

    // Collect results by category
    let mut string_results = Vec::new();
    let mut token_results = Vec::new();
    let mut indexer_results = Vec::new();

    for (key, value) in results.iter() {
        let category = key.split('_').skip(1).collect::<Vec<_>>().join("_");
        match category.as_str() {
            "string" => string_results.push(value.clone()),
            "token" => token_results.push(value.clone()),
            "indexer" => indexer_results.push(value.clone()),
            _ => {}
        }
    }

    eprintln!("\n{}", "=".repeat(90));
    eprintln!("STRINGTREE (kv_index::StringTree)");
    eprintln!("{}", "=".repeat(90));
    eprintln!(
        "{:>4} | {:>12} | {:>6} | {:>10} | {:>8} | {:>12}",
        "Work", "Size", "Op", "Throughput", "Latency", "Bandwidth"
    );
    eprintln!("{}", "-".repeat(90));
    for v in &string_results {
        eprintln!("{v}");
    }

    eprintln!("\n{}", "=".repeat(90));
    eprintln!("TOKENTREE (kv_index::TokenTree)");
    eprintln!("{}", "=".repeat(90));
    eprintln!(
        "{:>4} | {:>12} | {:>6} | {:>10} | {:>8} | {:>12}",
        "Work", "Size", "Op", "Throughput", "Latency", "Bandwidth"
    );
    eprintln!("{}", "-".repeat(90));
    for v in &token_results {
        eprintln!("{v}");
    }

    eprintln!("\n{}", "=".repeat(95));
    eprintln!("POSITIONALINDEXER (kv_index::PositionalIndexer) — event-driven KV cache routing");
    eprintln!("{}", "=".repeat(95));
    eprintln!(
        "{:>4} | {:>15} | {:>6} | {:>10} | {:>9} | {:>12}",
        "Work", "Size", "Op", "Throughput", "Latency", "Bandwidth"
    );
    eprintln!("{}", "-".repeat(95));
    for v in &indexer_results {
        eprintln!("{v}");
    }

    eprintln!("\n{}", "=".repeat(95));
}

fn run_benchmarks(c: &mut Criterion) {
    bench_summary(c);
    print_summary();
}

criterion_group!(benches, run_benchmarks);
criterion_main!(benches);
