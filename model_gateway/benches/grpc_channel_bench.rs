//! gRPC Channel Benchmark — Baseline (Single Channel)
//!
//! Measures throughput and latency of concurrent streaming gRPC requests
//! through a single tonic::Channel to quantify the bottleneck.
//! This benchmark:
//! 1. Spins up an in-process mock SglangScheduler gRPC server
//! 2. Connects a single SglangSchedulerClient (one channel = one TCP connection)
//! 3. Fires N concurrent streaming Generate RPCs
//! 4. Reports throughput (req/s) and latency percentiles (p50/p95/p99)

use std::{
    net::SocketAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use futures::StreamExt;
// Access the proto types and server trait from grpc_client
use smg_grpc_client::sglang_proto;
use tokio::sync::Barrier;

// Mock gRPC Server

/// A mock SglangScheduler that returns streaming chunks with configurable delay.
/// This simulates realistic token-by-token generation without actual inference.
#[derive(Clone)]
struct MockScheduler {
    /// Number of streaming chunks to send per Generate request
    chunks_per_request: usize,
    /// Simulated inter-token delay
    chunk_delay: Duration,
}

#[tonic::async_trait]
impl sglang_proto::sglang_scheduler_server::SglangScheduler for MockScheduler {
    type GenerateStream = tokio_stream::wrappers::ReceiverStream<
        Result<sglang_proto::GenerateResponse, tonic::Status>,
    >;

    async fn generate(
        &self,
        request: tonic::Request<sglang_proto::GenerateRequest>,
    ) -> Result<tonic::Response<Self::GenerateStream>, tonic::Status> {
        let req = request.into_inner();
        let request_id = req.request_id.clone();
        let chunks = self.chunks_per_request;
        let delay = self.chunk_delay;

        let (tx, rx) = tokio::sync::mpsc::channel(32);

        tokio::spawn(async move {
            // Send streaming chunks (simulating token generation)
            for i in 0..chunks {
                if delay > Duration::ZERO {
                    tokio::time::sleep(delay).await;
                }

                let response = sglang_proto::GenerateResponse {
                    request_id: request_id.clone(),
                    response: Some(if i < chunks - 1 {
                        // Intermediate chunk
                        sglang_proto::generate_response::Response::Chunk(
                            sglang_proto::GenerateStreamChunk {
                                token_ids: vec![100 + i as u32],
                                prompt_tokens: 10,
                                completion_tokens: (i + 1) as u32,
                                cached_tokens: 0,
                                ..Default::default()
                            },
                        )
                    } else {
                        // Final complete message
                        sglang_proto::generate_response::Response::Complete(
                            sglang_proto::GenerateComplete {
                                output_ids: (0..chunks as u32).map(|j| 100 + j).collect(),
                                finish_reason: "stop".to_string(),
                                prompt_tokens: 10,
                                completion_tokens: chunks as u32,
                                cached_tokens: 0,
                                ..Default::default()
                            },
                        )
                    }),
                };

                if tx.send(Ok(response)).await.is_err() {
                    break; // Client disconnected
                }
            }
        });

        Ok(tonic::Response::new(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        ))
    }

    async fn embed(
        &self,
        _request: tonic::Request<sglang_proto::EmbedRequest>,
    ) -> Result<tonic::Response<sglang_proto::EmbedResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("not used in benchmark"))
    }

    async fn health_check(
        &self,
        _request: tonic::Request<sglang_proto::HealthCheckRequest>,
    ) -> Result<tonic::Response<sglang_proto::HealthCheckResponse>, tonic::Status> {
        Ok(tonic::Response::new(sglang_proto::HealthCheckResponse {
            healthy: true,
            message: "ok".to_string(),
        }))
    }

    async fn abort(
        &self,
        _request: tonic::Request<sglang_proto::AbortRequest>,
    ) -> Result<tonic::Response<sglang_proto::AbortResponse>, tonic::Status> {
        Ok(tonic::Response::new(sglang_proto::AbortResponse {
            success: true,
            message: "aborted".to_string(),
        }))
    }

    async fn get_model_info(
        &self,
        _request: tonic::Request<sglang_proto::GetModelInfoRequest>,
    ) -> Result<tonic::Response<sglang_proto::GetModelInfoResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("not used in benchmark"))
    }

    async fn get_server_info(
        &self,
        _request: tonic::Request<sglang_proto::GetServerInfoRequest>,
    ) -> Result<tonic::Response<sglang_proto::GetServerInfoResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("not used in benchmark"))
    }

    async fn get_loads(
        &self,
        _request: tonic::Request<sglang_proto::GetLoadsRequest>,
    ) -> Result<tonic::Response<sglang_proto::GetLoadsResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("not used in benchmark"))
    }
}

// Benchmark Runner

/// Results from a single concurrency-level benchmark run
struct BenchResult {
    concurrency: usize,
    total_requests: usize,
    throughput_rps: f64,
    latency_p50_ms: f64,
    latency_p95_ms: f64,
    latency_p99_ms: f64,
    latency_max_ms: f64,
    errors: u64,
}

/// Start the mock gRPC server and return its address
async fn start_mock_server(chunks_per_request: usize, chunk_delay: Duration) -> SocketAddr {
    let mock = MockScheduler {
        chunks_per_request,
        chunk_delay,
    };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("Failed to bind mock server");
    let addr = listener.local_addr().unwrap();

    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);

    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(sglang_proto::sglang_scheduler_server::SglangSchedulerServer::new(mock))
            .serve_with_incoming(incoming)
            .await
            .expect("Mock server failed");
    });

    // Give server a moment to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    addr
}

/// Run benchmark at a specific concurrency level
async fn run_benchmark(
    client: &smg_grpc_client::SglangSchedulerClient,
    concurrency: usize,
    requests_per_task: usize,
) -> BenchResult {
    let total_requests = concurrency * requests_per_task;
    let error_count = Arc::new(AtomicU64::new(0));
    let barrier = Arc::new(Barrier::new(concurrency));

    // Collect all latencies
    let latencies: Arc<tokio::sync::Mutex<Vec<f64>>> =
        Arc::new(tokio::sync::Mutex::new(Vec::with_capacity(total_requests)));

    let overall_start = Instant::now();

    let mut handles = Vec::with_capacity(concurrency);

    for task_id in 0..concurrency {
        let client = client.clone();
        let barrier = barrier.clone();
        let error_count = error_count.clone();
        let latencies = latencies.clone();

        handles.push(tokio::spawn(async move {
            // Wait for all tasks to be ready (synchronized start)
            barrier.wait().await;

            let mut task_latencies = Vec::with_capacity(requests_per_task);

            for req_idx in 0..requests_per_task {
                let request_id = format!("bench-{}-{}", task_id, req_idx);
                let req = sglang_proto::GenerateRequest {
                    request_id: request_id.clone(),
                    tokenized: Some(sglang_proto::TokenizedInput {
                        original_text: "Hello, benchmark world!".to_string(),
                        input_ids: vec![9906, 11, 23513, 1917, 0],
                    }),
                    sampling_params: Some(sglang_proto::SamplingParams {
                        temperature: 0.7,
                        top_p: 0.9,
                        top_k: 50,
                        max_new_tokens: Some(128),
                        skip_special_tokens: true,
                        spaces_between_special_tokens: true,
                        repetition_penalty: 1.0,
                        n: 1,
                        ..Default::default()
                    }),
                    stream: true,
                    ..Default::default()
                };

                let req_start = Instant::now();

                match client.generate(req).await {
                    Ok(mut stream) => {
                        // Consume all streaming chunks
                        let mut chunk_count = 0u32;
                        while let Some(result) = stream.next().await {
                            match result {
                                Ok(_) => chunk_count += 1,
                                Err(_e) => {
                                    error_count.fetch_add(1, Ordering::Relaxed);
                                    break;
                                }
                            }
                        }
                        if chunk_count > 0 {
                            stream.mark_completed();
                        }
                        let elapsed = req_start.elapsed();
                        task_latencies.push(elapsed.as_secs_f64() * 1000.0); // ms
                    }
                    Err(_e) => {
                        error_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }

            // Batch-write latencies
            let mut guard = latencies.lock().await;
            guard.extend(task_latencies);
        }));
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.expect("Task panicked");
    }

    let overall_elapsed = overall_start.elapsed();
    let errors = error_count.load(Ordering::Relaxed);
    let successful_requests = total_requests as u64 - errors;
    let throughput_rps = successful_requests as f64 / overall_elapsed.as_secs_f64();

    // Calculate percentiles
    let mut latency_values = latencies.lock().await;
    latency_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let (p50, p95, p99, max_lat) = if latency_values.is_empty() {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        let len = latency_values.len();
        let p50_idx = (len as f64 * 0.50) as usize;
        let p95_idx = (len as f64 * 0.95) as usize;
        let p99_idx = ((len as f64 * 0.99) as usize).min(len - 1);
        (
            latency_values[p50_idx],
            latency_values[p95_idx.min(len - 1)],
            latency_values[p99_idx],
            latency_values[len - 1],
        )
    };

    BenchResult {
        concurrency,
        total_requests,
        throughput_rps,
        latency_p50_ms: p50,
        latency_p95_ms: p95,
        latency_p99_ms: p99,
        latency_max_ms: max_lat,
        errors,
    }
}

fn print_header() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                     gRPC Channel Benchmark — Single Channel (Baseline)                        ║");
    println!("╠═══════════════╦═══════════╦═══════════════╦══════════╦══════════╦══════════╦════════╦══════════╣");
    println!("║  Concurrency  ║  Total    ║  Throughput   ║  p50     ║  p95     ║  p99     ║  Errs  ║  Max     ║");
    println!("║               ║  Requests ║  (req/s)      ║  (ms)    ║  (ms)    ║  (ms)    ║        ║  (ms)    ║");
    println!("╠═══════════════╬═══════════╬═══════════════╬══════════╬══════════╬══════════╬════════╬══════════╣");
}

fn print_result(r: &BenchResult) {
    println!(
        "║  {:>11}  ║  {:>7}  ║  {:>11.2}  ║  {:>6.2}  ║  {:>6.2}  ║  {:>6.2}  ║  {:>4}  ║  {:>6.2}  ║",
        r.concurrency, r.total_requests, r.throughput_rps,
        r.latency_p50_ms, r.latency_p95_ms, r.latency_p99_ms,
        r.errors, r.latency_max_ms,
    );
}

fn print_footer() {
    println!("╚═══════════════╩═══════════╩═══════════════╩══════════╩══════════╩══════════╩════════╩══════════╝");
    println!();
}

// Main Entry Point

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to build tokio runtime");

    rt.block_on(async {
        // Configuration
        let chunks_per_request = 10; // 10 streaming chunks per generate request
        let chunk_delay = Duration::from_micros(100); // 100μs inter-chunk delay (fast, isolates channel)
        let requests_per_task = 20; // Each concurrent task sends 20 requests

        println!("\n=== Benchmark Configuration ===");
        println!("  Chunks per request:  {}", chunks_per_request);
        println!("  Chunk delay:         {:?}", chunk_delay);
        println!("  Requests per task:   {}", requests_per_task);
        println!("  Channel type:        Single tonic::Channel (1 TCP connection)");

        // Start mock server
        let addr = start_mock_server(chunks_per_request, chunk_delay).await;
        println!("  Mock server:         http://{}", addr);

        // Connect single-channel client (current production behavior)
        let client = smg_grpc_client::SglangSchedulerClient::connect(&format!("http://{}", addr))
            .await
            .expect("Failed to connect to mock server");

        // Warm up
        println!("\n  Warming up...");
        let _ = run_benchmark(&client, 2, 5).await;
        println!("  Warm-up complete.\n");

        // Run at various concurrency levels
        let concurrency_levels = [1, 10, 50, 100, 200, 500];

        print_header();

        for &concurrency in &concurrency_levels {
            let result = run_benchmark(&client, concurrency, requests_per_task).await;
            print_result(&result);

            // Brief pause between levels to let things settle
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        print_footer();
    });
}
