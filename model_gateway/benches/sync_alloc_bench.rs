use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
/// Simulates the old hot path with the unconditional allocation
#[inline(never)]
fn sync_insert_operation_old(
    _model_id: &str,
    tokens: &[u32],
    _tenant: &str,
    mesh_sync_active: bool,
) {
    // Unconditionally allocate the vec here, similar to the caller:
    // self.sync_insert_operation(model_id, TreeKey::Tokens(tokens.to_vec()), tenant)
    let key = tokens.to_vec();

    // Simulating checking self.mesh_sync.read().clone() inside sync_insert_operation
    if mesh_sync_active {
        // use key
        black_box(&key);
    }
}

/// Simulates the new hot path avoiding allocation unless active
#[inline(never)]
fn sync_insert_operation_new(
    _model_id: &str,
    tokens: &[u32],
    _tenant: &str,
    mesh_sync_active: bool,
) {
    // Simulating self.sync_insert_tokens(...)
    if mesh_sync_active {
        let key = tokens.to_vec();
        black_box(&key);
    }
}

fn bench_sync_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing_sync_allocation");
    let tokens = vec![1_u32; 4000]; // simulate a 4k token prompt

    // We assume the typical path is single-node / no active mesh sync
    let mesh_sync_active = false;

    group.bench_with_input(
        BenchmarkId::new("Unconditional Allocation (Old)", tokens.len()),
        &tokens,
        |b, tokens| {
            b.iter(|| {
                sync_insert_operation_old(
                    black_box("mock-model"),
                    black_box(tokens),
                    black_box("tenant-a"),
                    black_box(mesh_sync_active),
                );
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Deferred Allocation (New)", tokens.len()),
        &tokens,
        |b, tokens| {
            b.iter(|| {
                sync_insert_operation_new(
                    black_box("mock-model"),
                    black_box(tokens),
                    black_box("tenant-a"),
                    black_box(mesh_sync_active),
                );
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_sync_allocation);
criterion_main!(benches);
