use std::{sync::Arc, time::Duration};

use reqwest::Client;
use serde_json::Value;

use crate::{
    store::MetricsStore,
    types::{MetricSource, WorkerSnapshot},
};

pub struct DirectScraper {
    store: Arc<MetricsStore>,
    client: Client,
    interval: Duration,
}

impl DirectScraper {
    pub fn new(store: Arc<MetricsStore>, interval: Duration) -> Self {
        Self {
            store,
            client: Client::builder()
                .timeout(Duration::from_secs(3))
                .build()
                .unwrap_or_default(),
            interval,
        }
    }

    pub async fn run<F, Fut>(&self, get_urls: F)
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Vec<String>>,
    {
        let mut ticker = tokio::time::interval(self.interval);
        loop {
            ticker.tick().await;
            let urls = get_urls().await;

            for url in urls {
                let store = Arc::clone(&self.store);
                let client = self.client.clone();
                let w_url = url.clone();

                tokio::spawn(async move {
                    // Try to fetch `/get_load` first for token usage
                    let load_url = format!("{}/get_load", w_url);
                    let mut snapshot =
                        WorkerSnapshot::new(w_url.clone(), MetricSource::DirectScrape);

                    if let Ok(resp) = client.get(&load_url).send().await {
                        if let Ok(Value::Array(arr)) = resp.json::<Value>().await {
                            let total_tokens: i64 = arr
                                .iter()
                                .filter_map(|e| e.get("num_tokens").and_then(|v| v.as_i64()))
                                .sum();
                            snapshot.kv_cache_tokens = Some(total_tokens as isize);
                        }
                    }

                    // Fetch `/metrics` for custom prometheus metrics
                    let metrics_url = format!("{}/metrics", w_url);
                    if let Ok(resp) = client.get(&metrics_url).send().await {
                        if let Ok(text) = resp.text().await {
                            for line in text.lines() {
                                let line = line.trim();
                                if line.is_empty() || line.starts_with('#') {
                                    continue;
                                }

                                // Format is typically: metric_name{label="val"} value
                                if let Some((key_part, val_part)) = line.rsplit_once(' ') {
                                    if let Ok(val) = val_part.parse::<f64>() {
                                        // Extract just the metric name, stripping labels
                                        let key = match key_part.find('{') {
                                            Some(idx) => &key_part[..idx],
                                            None => key_part,
                                        };

                                        // Route standard routing metrics directly to native fields
                                        match key {
                                            "sglang:in_flight_requests"
                                            | "vllm:num_requests_running" => {
                                                snapshot.in_flight_requests = val as isize;
                                            }
                                            "sglang:avg_tokens_per_req" => {
                                                snapshot.avg_tokens_per_req = val as isize;
                                            }
                                            _ => {
                                                snapshot
                                                    .custom_metrics
                                                    .insert(key.to_string(), val);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    store.update(snapshot);
                });
            }
        }
    }
}
