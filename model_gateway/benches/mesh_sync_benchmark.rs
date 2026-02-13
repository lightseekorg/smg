use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use smg_mesh::{MeshSyncManager, StateStores};
use smg::core::{BasicWorkerBuilder, Worker, WorkerType};
use smg::policies::{CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, SelectWorkerInfo};

fn benchmark_mesh_sync_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("mesh_sync_overhead");
    
    // Setup runtime
    let rt = tokio::runtime::Runtime::new().unwrap();

    // 1. Setup workers
    let workers: Vec<Arc<dyn Worker>> = vec![
        Arc::new(
            BasicWorkerBuilder::new("http://worker1:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        ),
        Arc::new(
            BasicWorkerBuilder::new("http://worker2:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        ),
    ];

    // 2. Setup Policy WITHOUT Mesh (Baseline)
    let config_no_mesh = CacheAwareConfig {
        eviction_interval_secs: 0,
        ..Default::default()
    };
    let policy_no_mesh = CacheAwarePolicy::with_config(config_no_mesh);
    policy_no_mesh.init_workers(&workers);

    // 3. Setup Policy WITH Mesh
    let config_mesh = CacheAwareConfig {
        eviction_interval_secs: 0,
        ..Default::default()
    };
    let mut policy_mesh = CacheAwarePolicy::with_config(config_mesh);
    policy_mesh.init_workers(&workers);

    // Create mesh sync manager
    let stores = Arc::new(StateStores::with_self_name("bench_node".to_string()));
    let mesh_sync = Arc::new(MeshSyncManager::new(stores.clone(), "bench_node".to_string()));
    policy_mesh.set_mesh_sync(Some(mesh_sync));

    // Define workload
    let request_text = "benchmark request text that triggers insertion";
    
    group.throughput(Throughput::Elements(1));

    // Benchmark No Mesh
    group.bench_function("select_worker_no_mesh", |b| {
        b.iter(|| {
            let info = SelectWorkerInfo {
                request_text: Some(request_text),
                ..Default::default()
            };
            policy_no_mesh.select_worker(&workers, &info)
        })
    });

    // Benchmark With Mesh (Sync)
    group.bench_function("select_worker_with_mesh_sync", |b| {
        
        b.iter(|| {
            let info = SelectWorkerInfo {
                request_text: Some(request_text),
                ..Default::default()
            };
            // This will trigger sync_tree_operation every time
            policy_mesh.select_worker(&workers, &info)
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_mesh_sync_overhead);
criterion_main!(benches);
