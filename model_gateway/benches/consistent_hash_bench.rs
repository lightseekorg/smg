use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use smg_mesh::consistent_hash::ConsistentHashRing;

fn build_ring(num_nodes: usize) -> ConsistentHashRing {
    let mut ring = ConsistentHashRing::new();
    for i in 0..num_nodes {
        ring.add_node(&format!("node-{i}"));
    }
    ring
}

fn bench_is_owner(c: &mut Criterion) {
    let mut group = c.benchmark_group("consistent_hash");
    let num_nodes = 50;
    let ring = build_ring(num_nodes);

    // Generate some keys to check
    let keys: Vec<String> = (0..500).map(|i| format!("test-key-{i}")).collect();

    // Test a node that exists
    let node_true = "node-25";
    // Test a node that does NOT exist to measure the worst-case fallback
    let node_false = "node-nonexistent";

    group.bench_function("is_owner_baseline", |b| {
        b.iter(|| {
            for key in &keys {
                // We mix hits and misses to get an average runtime check
                black_box(ring.is_owner(black_box(key), black_box(node_true)));
                black_box(ring.is_owner(black_box(key), black_box(node_false)));
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_is_owner);
criterion_main!(benches);
