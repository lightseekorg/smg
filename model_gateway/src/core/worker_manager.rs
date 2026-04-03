//! Worker Management Module
//!
//! Provides worker lifecycle operations and fan-out request utilities.

use std::{collections::HashMap, sync::Arc, time::Duration};

use axum::response::{IntoResponse, Response};
use futures::{
    future,
    stream::{self, StreamExt},
};
use http::StatusCode;
use openai_protocol::worker::{
    FlushCacheResult, WorkerGroupKey, WorkerLoadInfo, WorkerLoadResponse, WorkerLoadsResult,
};
use tokio::{
    sync::{watch, Mutex},
    task::JoinHandle,
};
use tracing::{debug, info};

use crate::{
    core::{
        metrics_aggregator::{self, MetricPack},
        ConnectionMode, Worker, WorkerLoadManager, WorkerRegistry, WorkerType,
    },
    policies::PolicyRegistry,
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CONCURRENT: usize = 32;

/// Result of a fan-out request to a single worker
struct WorkerResponse {
    url: String,
    result: Result<reqwest::Response, reqwest::Error>,
}

/// Fan out requests to workers in parallel
async fn fan_out(
    workers: &[Arc<dyn Worker>],
    client: &reqwest::Client,
    endpoint: &str,
    method: reqwest::Method,
) -> Vec<WorkerResponse> {
    let futures: Vec<_> = workers
        .iter()
        .map(|worker| {
            let client = client.clone();
            let url = worker.url().to_string();
            let full_url = format!("{url}/{endpoint}");
            let api_key = worker.api_key().cloned();
            let method = method.clone();

            async move {
                let mut req = client.request(method, &full_url).timeout(REQUEST_TIMEOUT);
                if let Some(key) = api_key {
                    req = req.bearer_auth(key);
                }
                WorkerResponse {
                    url,
                    result: req.send().await,
                }
            }
        })
        .collect();

    stream::iter(futures)
        .buffer_unordered(MAX_CONCURRENT)
        .collect()
        .await
}

pub enum EngineMetricsResult {
    Ok(String),
    Err(String),
}

impl IntoResponse for EngineMetricsResult {
    fn into_response(self) -> Response {
        match self {
            Self::Ok(text) => (StatusCode::OK, text).into_response(),
            Self::Err(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response(),
        }
    }
}

pub struct WorkerManager;

impl WorkerManager {
    pub fn get_worker_urls(registry: &Arc<WorkerRegistry>) -> Vec<String> {
        registry
            .get_all()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    pub async fn flush_cache_all(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> FlushCacheResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        let http_workers: Vec<_> = workers
            .into_iter()
            .filter(|w| matches!(w.connection_mode(), ConnectionMode::Http))
            .collect();

        if http_workers.is_empty() {
            return FlushCacheResult {
                successful: vec![],
                failed: vec![],
                total_workers,
                http_workers: 0,
                message: "No HTTP workers available for cache flush".to_string(),
            };
        }

        info!(
            "Flushing cache on {} HTTP workers (out of {} total)",
            http_workers.len(),
            total_workers
        );

        let responses = fan_out(&http_workers, client, "flush_cache", reqwest::Method::POST).await;

        let mut successful = Vec::new();
        let mut failed = Vec::new();

        for resp in responses {
            match resp.result {
                Ok(r) if r.status().is_success() => successful.push(resp.url),
                Ok(r) => failed.push((resp.url, format!("HTTP {}", r.status()))),
                Err(e) => failed.push((resp.url, e.to_string())),
            }
        }

        let message = if failed.is_empty() {
            format!(
                "Successfully flushed cache on all {} HTTP workers",
                successful.len()
            )
        } else {
            format!(
                "Cache flush: {} succeeded, {} failed",
                successful.len(),
                failed.len()
            )
        };

        info!("{}", message);

        FlushCacheResult {
            successful,
            failed,
            total_workers,
            http_workers: http_workers.len(),
            message,
        }
    }

    pub async fn get_all_worker_loads(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> WorkerLoadsResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        let futures: Vec<_> = workers
            .iter()
            .map(|worker| {
                let worker_type = match worker.worker_type() {
                    WorkerType::Regular => None,
                    WorkerType::Prefill => Some("prefill".to_string()),
                    WorkerType::Decode => Some("decode".to_string()),
                };
                let connection_mode = worker.connection_mode();
                let client = client.clone();
                let worker = Arc::clone(worker);

                async move {
                    let details = match connection_mode {
                        ConnectionMode::Http => Self::fetch_http_load(&client, &worker).await,
                        ConnectionMode::Grpc => Self::fetch_grpc_load(&worker).await,
                    };
                    let load = details
                        .as_ref()
                        .map(|d| d.total_used_tokens() as isize)
                        .unwrap_or(-1);
                    WorkerLoadInfo {
                        worker: worker.url().to_string(),
                        worker_type,
                        load,
                        details,
                    }
                }
            })
            .collect();

        let loads = future::join_all(futures).await;
        let successful = loads.iter().filter(|l| l.load >= 0).count();
        let failed = loads.iter().filter(|l| l.load < 0).count();

        WorkerLoadsResult {
            loads,
            total_workers,
            successful,
            failed,
        }
    }

    /// Fetch load via HTTP using the /v1/loads endpoint.
    /// Returns the full `WorkerLoadResponse` or `None` on failure.
    async fn fetch_http_load(
        client: &reqwest::Client,
        worker: &Arc<dyn Worker>,
    ) -> Option<WorkerLoadResponse> {
        let url = worker.url();
        let load_url = format!("{url}/v1/loads?include=core");
        let mut req = client.get(&load_url).timeout(REQUEST_TIMEOUT);
        if let Some(key) = worker.api_key() {
            req = req.bearer_auth(key);
        }

        let resp = match req.send().await {
            Ok(r) if r.status().is_success() => r,
            _ => return None,
        };

        let response: WorkerLoadResponse = resp.json().await.ok()?;

        if response.loads.is_empty() {
            return None;
        }

        Some(response)
    }

    /// Fetch load via gRPC using the GetLoads RPC.
    /// Only supported for SGLang backends.
    async fn fetch_grpc_load(worker: &Arc<dyn Worker>) -> Option<WorkerLoadResponse> {
        let grpc_client = match worker.get_grpc_client().await {
            Ok(Some(client)) => client,
            Ok(None) => {
                debug!("No gRPC client for worker {}", worker.url());
                return None;
            }
            Err(e) => {
                debug!("Failed to get gRPC client for {}: {e}", worker.url());
                return None;
            }
        };

        match grpc_client.get_loads().await {
            Ok(load) if !load.loads.is_empty() => Some(load),
            Ok(_) => None,
            Err(e) => {
                debug!("gRPC GetLoads failed for {}: {e}", worker.url());
                None
            }
        }
    }

    pub async fn get_engine_metrics(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> EngineMetricsResult {
        let workers = worker_registry.get_all();

        if workers.is_empty() {
            return EngineMetricsResult::Err("No available workers".to_string());
        }

        let responses = fan_out(&workers, client, "metrics", reqwest::Method::GET).await;

        let mut metric_packs = Vec::new();
        for resp in responses {
            if let Ok(r) = resp.result {
                if r.status().is_success() {
                    if let Ok(text) = r.text().await {
                        metric_packs.push(MetricPack {
                            labels: vec![("worker_addr".into(), resp.url)],
                            metrics_text: text,
                        });
                    }
                }
            }
        }

        if metric_packs.is_empty() {
            return EngineMetricsResult::Err("All backend requests failed".to_string());
        }

        match metrics_aggregator::aggregate_metrics(metric_packs) {
            Ok(text) => EngineMetricsResult::Ok(text),
            Err(e) => EngineMetricsResult::Err(format!("Failed to aggregate metrics: {e}")),
        }
    }
}

/// Load monitoring service that periodically fetches worker loads.
///
/// Maintains separate polling loops per worker group (model_id × worker_type × connection_mode).
/// Groups are started/stopped automatically when workers are registered/removed.
pub struct LoadMonitor {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    pub worker_load_manager: Arc<WorkerLoadManager>,
    client: reqwest::Client,
    default_interval: Duration,
    tx: watch::Sender<HashMap<String, WorkerLoadResponse>>,
    rx: watch::Receiver<HashMap<String, WorkerLoadResponse>>,
    /// Per-group polling handles
    group_handles: Arc<Mutex<HashMap<WorkerGroupKey, JoinHandle<()>>>>,
}

impl LoadMonitor {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        default_interval_secs: u64,
    ) -> Self {
        let (tx, rx) = watch::channel(HashMap::new());

        Self {
            worker_registry,
            policy_registry,
            worker_load_manager: Arc::new(WorkerLoadManager::new()),
            client,
            default_interval: Duration::from_secs(default_interval_secs),
            tx,
            rx,
            group_handles: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Start polling for a worker group. If the group already has a running loop, this is a no-op.
    ///
    /// `interval` is the per-worker override from `WorkerSpec.load_monitor_interval_secs`.
    /// Falls back to `self.default_interval` when `None`.
    pub async fn on_group_added(&self, key: WorkerGroupKey, interval: Option<u64>) {
        let mut handles = self.group_handles.lock().await;
        if handles.contains_key(&key) {
            debug!("Load monitor group already running: {key}");
            return;
        }

        // Floor at 1s to prevent tight-loop DoS from a zero interval.
        let interval = interval
            .map(|s| Duration::from_secs(s.max(1)))
            .unwrap_or(self.default_interval)
            .max(Duration::from_secs(1));

        info!("Starting load monitor for group {key} with interval {interval:?}");

        let worker_registry = Arc::clone(&self.worker_registry);
        let policy_registry = Arc::clone(&self.policy_registry);
        let worker_load_manager = Arc::clone(&self.worker_load_manager);
        let client = self.client.clone();
        let tx = self.tx.clone();
        let group_key = key.clone();

        #[expect(
            clippy::disallowed_methods,
            reason = "Load monitor loop: runs for the lifetime of the group, handle is stored and abort() is called on removal"
        )]
        let handle = tokio::spawn(async move {
            Self::group_monitor_loop(
                group_key,
                worker_registry,
                policy_registry,
                worker_load_manager,
                client,
                interval,
                tx,
            )
            .await;
        });

        handles.insert(key, handle);
    }

    /// Stop polling for a worker group and clean up its entries from the shared load map.
    ///
    /// `worker_urls` must be provided by the caller because this is called *after*
    /// workers have been removed from the registry (so we can't look them up).
    pub async fn on_group_removed(&self, key: &WorkerGroupKey, worker_urls: &[String]) {
        let handle = {
            let mut handles = self.group_handles.lock().await;
            handles.remove(key)
        };
        if let Some(handle) = handle {
            info!("Stopping load monitor for group {key}");
            handle.abort();
            let _ = handle.await;
        }

        // Remove stale load entries regardless of whether a handle was found,
        // since entries could exist from a previous monitoring cycle.
        if !worker_urls.is_empty() {
            self.tx.send_modify(|map| {
                for url in worker_urls {
                    map.remove(url);
                }
            });
        }
    }

    /// Stop all group polling loops.
    pub async fn stop(&self) {
        let handles: HashMap<WorkerGroupKey, JoinHandle<()>> = {
            let mut guard = self.group_handles.lock().await;
            std::mem::take(&mut *guard)
        };

        if handles.is_empty() {
            return;
        }

        info!("Stopping all {} load monitor groups", handles.len());
        for (key, handle) in handles {
            debug!("Stopping load monitor group: {key}");
            handle.abort();
            let _ = handle.await;
        }
    }

    pub fn subscribe(&self) -> watch::Receiver<HashMap<String, WorkerLoadResponse>> {
        self.rx.clone()
    }

    /// Polling loop for a single worker group.
    async fn group_monitor_loop(
        group_key: WorkerGroupKey,
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        worker_load_manager: Arc<WorkerLoadManager>,
        client: reqwest::Client,
        interval: Duration,
        tx: watch::Sender<HashMap<String, WorkerLoadResponse>>,
    ) {
        let mut interval_timer = tokio::time::interval(interval);

        loop {
            interval_timer.tick().await;

            let power_of_two_policies = policy_registry.get_all_power_of_two_policies();
            if power_of_two_policies.is_empty() && policy_registry.get_dp_rank_policy().is_none() {
                debug!("No PowerOfTwo policies found, skipping load fetch for group {group_key}");
                continue;
            }

            // Get workers for this specific group
            let workers = worker_registry.get_workers_filtered(
                Some(&group_key.model_id),
                Some(group_key.worker_type),
                Some(group_key.connection_mode),
                None,
                false,
            );

            if workers.is_empty() {
                debug!("No workers in group {group_key}, skipping");
                continue;
            }

            // Fetch loads for all workers in this group
            let futures: Vec<_> = workers
                .iter()
                .map(|worker| {
                    let client = client.clone();
                    let worker = Arc::clone(worker);
                    let connection_mode = group_key.connection_mode;

                    async move {
                        let response = match connection_mode {
                            ConnectionMode::Http => {
                                WorkerManager::fetch_http_load(&client, &worker).await
                            }
                            ConnectionMode::Grpc => WorkerManager::fetch_grpc_load(&worker).await,
                        };
                        (worker.url().to_string(), response)
                    }
                })
                .collect();

            let results = future::join_all(futures).await;

            // Collect successful loads
            let mut group_loads: HashMap<String, WorkerLoadResponse> = HashMap::new();
            let mut group_dp_loads: HashMap<String, HashMap<isize, isize>> = HashMap::new();
            for (url, response) in results {
                if let Some(load) = response {
                    group_loads.insert(url.clone(), load.clone());
                    let dp_rank_loads = load.dp_rank_loads();
                    group_dp_loads.insert(url, dp_rank_loads);
                }
            }

            if group_loads.is_empty() {
                debug!("No loads fetched for group {group_key}");
                continue;
            }

            debug!(
                "Fetched loads from {}/{} workers in group {group_key}",
                group_loads.len(),
                workers.len()
            );

            // Update policies with this group's loads
            for policy in &power_of_two_policies {
                policy.update_loads(&group_loads);
            }
            worker_load_manager.update_dp_loads(&group_dp_loads);

            // Atomically merge into the shared watch channel.
            // Remove all group URLs first to clear stale entries from workers
            // that failed this tick, then insert successful loads.
            let all_group_urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();
            tx.send_modify(|map| {
                for url in &all_group_urls {
                    map.remove(url);
                }
                map.extend(group_loads);
            });
        }
    }

    /// Check if any group polling loop is running.
    pub async fn is_running(&self) -> bool {
        let handles = self.group_handles.lock().await;
        !handles.is_empty()
    }
}

impl Drop for LoadMonitor {
    fn drop(&mut self) {
        if let Ok(mut handles) = self.group_handles.try_lock() {
            for (_, handle) in handles.drain() {
                handle.abort();
            }
        }
    }
}
