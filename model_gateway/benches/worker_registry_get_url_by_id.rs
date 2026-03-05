use std::{collections::HashMap, hint::black_box, sync::Arc};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg::core::{BasicWorkerBuilder, CircuitBreakerConfig, WorkerId, WorkerRegistry};

/// Build a registry pre-populated with `count` workers.
///
/// Returns `(registry, first_id, last_id, all_urls)` where:
///   - `first_id`  = the WorkerId of the first worker registered
///   - `last_id`   = the WorkerId of the last worker registered
///   - `all_urls`  = every URL (for the O(1) baseline benchmark)
fn setup_registry(count: usize) -> (Arc<WorkerRegistry>, WorkerId, WorkerId, Vec<String>) {
    let registry = Arc::new(WorkerRegistry::new());
    let mut first_id: Option<WorkerId> = None;
    let mut last_id: Option<WorkerId> = None;
    let mut urls = Vec::with_capacity(count);

    for i in 0..count {
        let url = format!("http://worker-{i}:8000");
        urls.push(url.clone());

        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "bench-model".to_string());

        let worker = BasicWorkerBuilder::new(url)
            .labels(labels)
            .circuit_breaker_config(CircuitBreakerConfig::default())
            .build();

        let id = registry.register(Arc::from(worker));

        if i == 0 {
            first_id = Some(id.clone());
        }
        if i == count - 1 {
            last_id = Some(id);
        }
    }

    (
        registry,
        first_id.expect("at least one worker"),
        last_id.expect("at least one worker"),
        urls,
    )
}

/// A WorkerId not present in any registry.
/// Simulates a missing-ID lookup: exercises the `id_to_url` false-miss path (O(1)).
fn unknown_id() -> WorkerId {
    WorkerId::from_string("00000000-0000-0000-0000-000000000000".to_string())
}

/// Benchmark `get_url_by_id` at several fleet sizes.
fn bench_get_url_by_id(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_url_by_id");

    // Test at realistic fleet sizes: 10, 100, 500, 2 000 workers.
    for &size in &[10usize, 100, 500, 2_000] {
        let (registry, first_id, last_id, _urls) = setup_registry(size);
        let missing = unknown_id();

        // hot: look up the last-registered worker (best/average DashMap case)
        group.bench_with_input(
            BenchmarkId::new("hot (last registered)", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(registry.get_url_by_id(black_box(&last_id)));
                });
            },
        );

        // cold: look up the first-registered worker (worst-case for scan order)
        group.bench_with_input(
            BenchmarkId::new("cold (first registered)", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(registry.get_url_by_id(black_box(&first_id)));
                });
            },
        );

        // missing: WorkerId absent from registry → exercises O(1) id_to_url miss path.
        group.bench_with_input(
            BenchmarkId::new("missing (O(1) miss)", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(registry.get_url_by_id(black_box(&missing)));
                });
            },
        );
    }

    group.finish();
}

fn bench_get_by_url_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_by_url (O(1) baseline)");

    for &size in &[10usize, 100, 500, 2_000] {
        let (registry, _first_id, _last_id, urls) = setup_registry(size);

        // Look up the first URL (arbitrary all lookups are O(1))
        let first_url = urls[0].clone();
        let last_url = urls[urls.len() - 1].clone();

        group.bench_with_input(BenchmarkId::new("first url", size), &size, |b, _| {
            b.iter(|| {
                black_box(registry.get_by_url(black_box(&first_url)));
            });
        });

        group.bench_with_input(BenchmarkId::new("last url", size), &size, |b, _| {
            b.iter(|| {
                black_box(registry.get_by_url(black_box(&last_url)));
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_get_url_by_id, bench_get_by_url_baseline);
criterion_main!(benches);
