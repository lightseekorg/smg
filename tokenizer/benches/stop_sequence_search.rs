use std::hint::black_box;

use aho_corasick::AhoCorasick;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};

/// Naive implementation using nested loops with find()
fn find_stop_sequence_naive(text: &str, stop_sequences: &[String]) -> Option<(usize, usize)> {
    for stop_seq in stop_sequences {
        if let Some(pos) = text.find(stop_seq) {
            return Some((pos, pos + stop_seq.len()));
        }
    }
    None
}

/// Optimized implementation using Aho-Corasick automaton
fn find_stop_sequence_aho(text: &str, ac: &AhoCorasick) -> Option<(usize, usize)> {
    ac.find(text).map(|mat| (mat.start(), mat.end()))
}

fn bench_stop_sequence_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("stop_sequence_search");

    // Simulate realistic streaming text from LLM
    let text = "User: Hello! Assistant: How can I help you today? ".repeat(1000);
    group.throughput(Throughput::Bytes(text.len() as u64));

    for token_count in [5, 20, 50] {
        let stop_sequences: Vec<String> = (0..token_count)
            .map(|i| format!("<|stop_sequence_{}|>", i))
            .collect();

        let ac = AhoCorasick::new(&stop_sequences).unwrap();

        group.bench_function(format!("naive_{}_sequences", token_count), |b| {
            b.iter(|| find_stop_sequence_naive(black_box(&text), black_box(&stop_sequences)))
        });

        group.bench_function(format!("aho_corasick_{}_sequences", token_count), |b| {
            b.iter(|| find_stop_sequence_aho(black_box(&text), black_box(&ac)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_stop_sequence_search);
criterion_main!(benches);
