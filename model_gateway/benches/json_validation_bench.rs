use criterion::{black_box, criterion_group, criterion_main, Criterion};
use serde::de::IgnoredAny;
use serde_json::Value;

fn bench_json_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_validation_streaming_simulation");

    // A moderately sized JSON payload simulating a tool call
    let base_json = r#"{"name": "test_tool", "arguments": {"query": "what is the weather in san francisco today?", "date": "2023-10-25", "extra": [1,2,3,4,5,6,7,8,9,10]}}"#;
    let mut chunks = Vec::new();
    let mut current = String::new();

    // Create increasingly large partial JSON strings to simulate streaming chunk-by-chunk arrivals.
    // In actual code, `is_complete_json` is called on the accumulating buffer for every new chunk.
    for ch in base_json.chars() {
        current.push(ch);
        chunks.push(current.clone());
    }

    // Baseline: allocating full Value AST
    group.bench_function("baseline_value_ast", |b| {
        b.iter(|| {
            let mut complete_count = 0;
            for chunk in &chunks {
                if serde_json::from_str::<Value>(black_box(chunk)).is_ok() {
                    complete_count += 1;
                }
            }
            black_box(complete_count);
        })
    });

    // Optimized: using IgnoredAny for zero-allocation validation
    group.bench_function("optimized_ignored_any", |b| {
        b.iter(|| {
            let mut complete_count = 0;
            for chunk in &chunks {
                if serde_json::from_str::<IgnoredAny>(black_box(chunk)).is_ok() {
                    complete_count += 1;
                }
            }
            black_box(complete_count);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_json_validation);
criterion_main!(benches);
