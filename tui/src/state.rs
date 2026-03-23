use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, RwLock},
};

use chrono::{DateTime, Utc};
use openai_protocol::messages::ListModelsResponse;

use crate::client::{
    ClusterStatusResponse, LoadsResponse, MeshHealthResponse, RateLimitStats, SmgClient,
    WorkersResponse,
};

/// Cached gateway state from the most recent poll cycle.
#[derive(Debug, Default)]
pub struct GatewayState {
    pub connected: bool,
    pub healthy: bool,
    pub last_updated: Option<DateTime<Utc>>,
    pub last_error: Option<String>,

    pub workers: Option<WorkersResponse>,
    pub loads: Option<LoadsResponse>,
    pub cluster: Option<ClusterStatusResponse>,
    pub mesh_health: Option<MeshHealthResponse>,
    pub rate_limits: Option<RateLimitStats>,
    pub models: Option<ListModelsResponse>,

    /// Rolling aggregate throughput history (capacity: 20).
    pub throughput_history: VecDeque<f64>,
    /// Rolling aggregate cache hit rate history (capacity: 20).
    pub cache_hit_history: VecDeque<f64>,
    /// Per-worker throughput history.
    pub per_worker_throughput: HashMap<String, VecDeque<f64>>,
    /// Per-worker cache hit rate history.
    pub per_worker_cache_hit: HashMap<String, VecDeque<f64>>,

    /// Previous total request count from Prometheus (for computing req/s).
    pub prev_request_count: Option<u64>,
    /// Rolling requests-per-second history (for external workers without gen_throughput).
    pub requests_per_sec_history: VecDeque<f64>,

    /// Previous total token counts from Prometheus (for computing tok/s).
    pub prev_input_tokens: Option<u64>,
    pub prev_output_tokens: Option<u64>,
    /// Rolling tokens-per-second history.
    pub input_tps_history: VecDeque<f64>,
    pub output_tps_history: VecDeque<f64>,

    /// Rolling avg latency history (seconds, from Prometheus duration sum/count).
    pub avg_latency_history: VecDeque<f64>,
    /// Previous duration sum/count for computing per-interval avg latency.
    pub prev_duration_sum: Option<f64>,
    pub prev_duration_count: Option<u64>,
    /// Active HTTP connections (gauge).
    pub active_connections: u64,
    /// In-flight requests (gauge).
    pub inflight_requests: u64,
    /// Per-worker previous request counts (for computing per-worker req/s).
    pub prev_worker_requests: HashMap<String, u64>,
    /// Per-worker req/s.
    pub worker_rps: HashMap<String, f64>,

    /// GPU information from nvidia-smi (None if not available).
    pub gpus: Option<Vec<GpuInfo>>,

    /// Circuit breaker status parsed from Prometheus metrics.
    pub circuit_breakers: CircuitBreakerSummary,
}

/// Summary of circuit breaker states across all workers.
#[derive(Debug, Clone, Default)]
pub struct CircuitBreakerSummary {
    pub closed: u32,
    pub open: u32,
    pub total_failures: u64,
}

/// GPU information parsed from nvidia-smi.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub utilization_pct: u32,
    pub temperature_c: u32,
}

/// Thread-safe shared handle to the gateway state.
pub type SharedState = Arc<RwLock<GatewayState>>;

/// Spawn a background poller that periodically fetches data from all SMG
/// endpoints and updates [`SharedState`].
pub fn spawn_poller(
    client: SmgClient,
    state: SharedState,
    interval_secs: u64,
) -> tokio::task::JoinHandle<()> {
    // Safety: fire-and-forget background poller that runs for the app's lifetime
    #[expect(clippy::disallowed_methods)]
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
        loop {
            ticker.tick().await;
            poll_once(&client, &state, interval_secs).await;
        }
    })
}

/// Query nvidia-smi for GPU information. Returns None if nvidia-smi is not available.
async fn query_gpus() -> Option<Vec<GpuInfo>> {
    let output = tokio::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let gpus: Vec<GpuInfo> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| {
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() < 6 {
                return None;
            }
            Some(GpuInfo {
                index: parts[0].parse().ok()?,
                name: parts[1].to_string(),
                memory_used_mb: parts[2].parse().ok()?,
                memory_total_mb: parts[3].parse().ok()?,
                utilization_pct: parts[4].parse().ok()?,
                temperature_c: parts[5].parse().ok()?,
            })
        })
        .collect();

    if gpus.is_empty() {
        None
    } else {
        Some(gpus)
    }
}

/// Parse circuit breaker state from Prometheus metrics.
fn parse_circuit_breakers(metrics_text: &str) -> CircuitBreakerSummary {
    let mut closed = 0u32;
    let mut open = 0u32;
    let mut total_failures = 0u64;

    for line in metrics_text.lines() {
        if line.starts_with("smg_worker_cb_state{") {
            if let Some(val) = line
                .rsplit_once(' ')
                .and_then(|(_, v)| v.parse::<i32>().ok())
            {
                match val {
                    0 => closed += 1,
                    1 => open += 1,
                    _ => {} // -1 = stale, skip
                }
            }
        } else if line.starts_with("smg_worker_cb_consecutive_failures{") {
            if let Some(val) = line
                .rsplit_once(' ')
                .and_then(|(_, v)| v.parse::<u64>().ok())
            {
                total_failures += val;
            }
        }
    }

    CircuitBreakerSummary {
        closed,
        open,
        total_failures,
    }
}

/// Parse total request count from Prometheus metrics text.
fn parse_request_count(metrics_text: &str) -> u64 {
    metrics_text
        .lines()
        .filter(|line| line.starts_with("smg_router_requests_total{"))
        .filter_map(|line| {
            line.rsplit_once(' ')
                .and_then(|(_, v)| v.parse::<u64>().ok())
        })
        .sum()
}

/// Parse request duration sum and count from Prometheus (for avg latency).
fn parse_duration_stats(metrics_text: &str) -> (f64, u64) {
    let mut sum = 0.0f64;
    let mut count = 0u64;
    for line in metrics_text.lines() {
        if line.starts_with("smg_router_request_duration_seconds_sum{") {
            if let Some(v) = line
                .rsplit_once(' ')
                .and_then(|(_, v)| v.parse::<f64>().ok())
            {
                sum += v;
            }
        } else if line.starts_with("smg_router_request_duration_seconds_count{") {
            if let Some(v) = line
                .rsplit_once(' ')
                .and_then(|(_, v)| v.parse::<u64>().ok())
            {
                count += v;
            }
        }
    }
    (sum, count)
}

/// Parse active connections and in-flight requests from Prometheus.
fn parse_gauges(metrics_text: &str) -> (u64, u64) {
    let mut connections = 0u64;
    let mut inflight = 0u64;
    for line in metrics_text.lines() {
        if line.starts_with("smg_http_connections_active ") {
            if let Some(v) = line
                .rsplit_once(' ')
                .and_then(|(_, v)| v.parse::<u64>().ok())
            {
                connections = v;
            }
        } else if line.starts_with("smg_http_inflight_request_age_count{") {
            if let Some(v) = line
                .rsplit_once(' ')
                .and_then(|(_, v)| v.parse::<u64>().ok())
            {
                inflight += v;
            }
        }
    }
    (connections, inflight)
}

/// Parse per-worker request counts from Prometheus (success outcomes).
fn parse_worker_request_counts(metrics_text: &str) -> HashMap<String, u64> {
    let mut counts = HashMap::new();
    for line in metrics_text.lines() {
        // smg_worker_cb_outcomes_total{worker="grpc://127.0.0.1:46361",outcome="success"} 4024
        if line.starts_with("smg_worker_cb_outcomes_total{") && line.contains("outcome=\"success\"")
        {
            if let (Some(worker_start), Some(val)) = (
                line.find("worker=\"").map(|i| i + 8),
                line.rsplit_once(' ')
                    .and_then(|(_, v)| v.parse::<u64>().ok()),
            ) {
                if let Some(worker_end) = line[worker_start..].find('"') {
                    let worker = line[worker_start..worker_start + worker_end].to_string();
                    counts.insert(worker, val);
                }
            }
        }
    }
    counts
}

/// Parse total token counts from Prometheus metrics (input and output separately).
fn parse_token_counts(metrics_text: &str) -> (u64, u64) {
    let mut input = 0u64;
    let mut output = 0u64;
    for line in metrics_text.lines() {
        if line.starts_with("smg_router_tokens_total{") {
            if let Some(val) = line
                .rsplit_once(' ')
                .and_then(|(_, v)| v.parse::<u64>().ok())
            {
                if line.contains("token_type=\"input\"") {
                    input += val;
                } else if line.contains("token_type=\"output\"") {
                    output += val;
                }
            }
        }
    }
    (input, output)
}

async fn poll_once(client: &SmgClient, state: &SharedState, interval_secs: u64) {
    // Fire all requests concurrently.
    let (alive, health, workers, loads, cluster, mesh, rates, models, metrics, gpus) = tokio::join!(
        client.check_alive(),
        client.check_health(),
        client.list_workers(),
        client.get_loads(),
        client.get_cluster_status(),
        client.get_mesh_health(),
        client.get_rate_limit_stats(),
        client.list_models(),
        client.fetch_metrics(),
        query_gpus(),
    );

    // Safety: RwLock is not poisoned in practice — no panics while holding the lock
    #[expect(clippy::unwrap_used)]
    let mut s = state.write().unwrap();

    // Connection (liveness) vs readiness (health) are distinct:
    // connected = server is reachable; healthy = readiness check passes
    s.connected = alive.is_ok();
    match health {
        Ok(()) => {
            s.healthy = true;
            s.last_error = None;
        }
        Err(e) => {
            s.healthy = false;
            if !s.connected {
                s.last_error = Some(e.to_string());
            }
        }
    }

    // Each endpoint: update on success, keep stale data on failure.
    if let Ok(w) = workers {
        s.workers = Some(w);
    }
    if let Ok(l) = loads {
        s.loads = Some(l);
    }
    if let Ok(c) = cluster {
        s.cluster = Some(c);
    }
    if let Ok(m) = mesh {
        s.mesh_health = Some(m);
    }
    if let Ok(r) = rates {
        s.rate_limits = Some(r);
    }
    if let Ok(m) = models {
        s.models = Some(m);
    }
    s.gpus = gpus;

    s.last_updated = Some(Utc::now());

    const SPARKLINE_CAP: usize = 20;

    // Extract per-worker metrics before mutably borrowing sparkline buffers.
    let worker_metrics: Vec<(String, f64, f64)> = if let Some(ref loads_resp) = s.loads {
        loads_resp
            .workers
            .iter()
            .filter_map(|wl| {
                wl.details.as_ref().map(|details| {
                    let throughput: f64 = details.loads.iter().map(|s| s.gen_throughput).sum();
                    let cache_hit: f64 = if details.loads.is_empty() {
                        0.0
                    } else {
                        details.loads.iter().map(|s| s.cache_hit_rate).sum::<f64>()
                            / details.loads.len() as f64
                    };
                    (wl.worker.clone(), throughput, cache_hit)
                })
            })
            .collect()
    } else {
        Vec::new()
    };

    let has_worker_metrics = !worker_metrics.is_empty();
    if has_worker_metrics {
        let mut total_cache_hits = 0.0_f64;
        let worker_count = worker_metrics.len() as u32;

        for (worker_name, throughput, cache_hit) in worker_metrics {
            total_cache_hits += cache_hit;

            let th = s
                .per_worker_throughput
                .entry(worker_name.clone())
                .or_insert_with(|| VecDeque::with_capacity(SPARKLINE_CAP));
            if th.len() >= SPARKLINE_CAP {
                th.pop_front();
            }
            th.push_back(throughput);

            let ch = s
                .per_worker_cache_hit
                .entry(worker_name)
                .or_insert_with(|| VecDeque::with_capacity(SPARKLINE_CAP));
            if ch.len() >= SPARKLINE_CAP {
                ch.pop_front();
            }
            ch.push_back(cache_hit);
        }

        let avg_cache = if worker_count > 0 {
            total_cache_hits / worker_count as f64
        } else {
            0.0
        };
        if s.cache_hit_history.len() >= SPARKLINE_CAP {
            s.cache_hit_history.pop_front();
        }
        s.cache_hit_history.push_back(avg_cache);
    }

    // Compute requests/sec from Prometheus counter (works for external workers).
    if let Ok(metrics_text) = metrics {
        s.circuit_breakers = parse_circuit_breakers(&metrics_text);
        let current_count = parse_request_count(&metrics_text);
        if let Some(prev) = s.prev_request_count {
            let delta = current_count.saturating_sub(prev);
            let rps = delta as f64 / interval_secs as f64;
            if s.requests_per_sec_history.len() >= SPARKLINE_CAP {
                s.requests_per_sec_history.pop_front();
            }
            s.requests_per_sec_history.push_back(rps);

            // Use req/s as primary throughput sparkline (always works)
            if s.throughput_history.len() >= SPARKLINE_CAP {
                s.throughput_history.pop_front();
            }
            s.throughput_history.push_back(rps);
        }
        s.prev_request_count = Some(current_count);

        // Compute tok/s from Prometheus token counters
        let (cur_input, cur_output) = parse_token_counts(&metrics_text);
        if let (Some(prev_in), Some(prev_out)) = (s.prev_input_tokens, s.prev_output_tokens) {
            let in_delta = cur_input.saturating_sub(prev_in);
            let out_delta = cur_output.saturating_sub(prev_out);
            let in_tps = in_delta as f64 / interval_secs as f64;
            let out_tps = out_delta as f64 / interval_secs as f64;
            if s.input_tps_history.len() >= SPARKLINE_CAP {
                s.input_tps_history.pop_front();
            }
            if s.output_tps_history.len() >= SPARKLINE_CAP {
                s.output_tps_history.pop_front();
            }
            s.input_tps_history.push_back(in_tps);
            s.output_tps_history.push_back(out_tps);
        }
        s.prev_input_tokens = Some(cur_input);
        s.prev_output_tokens = Some(cur_output);

        // Compute avg latency from duration sum/count delta
        let (cur_sum, cur_count) = parse_duration_stats(&metrics_text);
        if let (Some(prev_sum), Some(prev_count)) = (s.prev_duration_sum, s.prev_duration_count) {
            let delta_sum = if cur_sum >= prev_sum {
                cur_sum - prev_sum
            } else {
                0.0
            };
            let delta_count = cur_count.saturating_sub(prev_count);
            let avg_latency = if delta_count > 0 {
                delta_sum / delta_count as f64
            } else {
                0.0
            };
            if s.avg_latency_history.len() >= SPARKLINE_CAP {
                s.avg_latency_history.pop_front();
            }
            s.avg_latency_history.push_back(avg_latency);
        }
        s.prev_duration_sum = Some(cur_sum);
        s.prev_duration_count = Some(cur_count);

        // Parse gauges
        let (connections, inflight) = parse_gauges(&metrics_text);
        s.active_connections = connections;
        s.inflight_requests = inflight;

        // Compute per-worker req/s
        let cur_worker_reqs = parse_worker_request_counts(&metrics_text);
        for (worker, cur_count) in &cur_worker_reqs {
            if let Some(prev) = s.prev_worker_requests.get(worker) {
                let delta = cur_count.saturating_sub(*prev);
                let rps = delta as f64 / interval_secs as f64;
                s.worker_rps.insert(worker.clone(), rps);
            }
        }
        s.prev_worker_requests = cur_worker_reqs;
    }
}
