use std::{sync::Arc, time::Duration};

use reqwest::Client;
use serde::Deserialize;
use tracing::{debug, warn};

use crate::{
    store::MetricsStore,
    types::{MetricSource, WorkerSnapshot},
};

pub struct PrometheusScraper {
    store: Arc<MetricsStore>,
    client: Client,
    interval: Duration,
    prom_url: String,
    /// PromQL selector for which metrics to pull.
    /// Defaults to `{__name__=~"^custom_.*"}`.
    metric_selector: String,
}

#[derive(Deserialize)]
struct PromResponse {
    status: String,
    data: Option<PromData>,
}

#[derive(Deserialize)]
struct PromData {
    result: Vec<PromResult>,
}

#[derive(Deserialize)]
struct PromResult {
    metric: std::collections::HashMap<String, String>,
    value: (f64, String),
}

impl PrometheusScraper {
    pub fn new(store: Arc<MetricsStore>, prom_url: String, interval: Duration) -> Self {
        Self::with_selector(
            store,
            prom_url,
            interval,
            r#"{__name__=~"^custom_.*"}"#.to_string(),
        )
    }

    /// Create with a custom PromQL selector string.
    ///
    /// For example:
    /// - `{__name__=~"^sglang:.*"}` to pull all SGLang metrics
    /// - `{job="sgw_workers"}` to pull all metrics for a specific job
    pub fn with_selector(
        store: Arc<MetricsStore>,
        prom_url: String,
        interval: Duration,
        metric_selector: String,
    ) -> Self {
        Self {
            store,
            client: Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap_or_default(),
            interval,
            prom_url,
            metric_selector,
        }
    }

    pub async fn run(&self) {
        if self.interval.is_zero() {
            tracing::warn!("PrometheusScraper: scrape interval is zero; disabling scraper");
            return;
        }

        let mut ticker = tokio::time::interval(self.interval);
        let mut consecutive_failures: u32 = 0;

        loop {
            if consecutive_failures > 0 {
                // Exponential backoff: base * 2^failures, capped at 60 seconds
                let multiplier = 2_u32.pow(consecutive_failures.min(6));
                let backoff = std::cmp::min(self.interval * multiplier, Duration::from_secs(60));
                tokio::time::sleep(backoff).await;
            } else {
                ticker.tick().await;
            }

            let query_url = format!(
                "{}/api/v1/query?query={}",
                self.prom_url,
                urlencoding::encode(&self.metric_selector),
            );

            match self.client.get(&query_url).send().await {
                Ok(resp) => {
                    if !resp.status().is_success() {
                        consecutive_failures += 1;
                        warn!(
                            consecutive_failures,
                            status = %resp.status(),
                            url = %query_url,
                            "PrometheusScraper: HTTP error from Prometheus"
                        );
                        continue;
                    }

                    match resp.json::<PromResponse>().await {
                        Ok(prom_resp) if prom_resp.status == "success" => {
                            consecutive_failures = 0;
                            let count = self.ingest(prom_resp.data);
                            debug!(metrics_ingested = count, "PrometheusScraper: scrape OK");
                        }
                        Ok(prom_resp) => {
                            consecutive_failures += 1;
                            warn!(
                                consecutive_failures,
                                status = %prom_resp.status,
                                "PrometheusScraper: Prometheus returned non-success status"
                            );
                        }
                        Err(e) => {
                            consecutive_failures += 1;
                            warn!(
                                consecutive_failures,
                                error = %e,
                                "PrometheusScraper: failed to parse Prometheus response"
                            );
                        }
                    }
                }
                Err(e) => {
                    consecutive_failures += 1;
                    warn!(
                        consecutive_failures,
                        error = %e,
                        url = %self.prom_url,
                        "PrometheusScraper: HTTP request failed"
                    );
                }
            }
        }
    }

    /// Ingest metric results, returning the number of metrics pushed.
    fn ingest(&self, data: Option<PromData>) -> usize {
        let Some(data) = data else { return 0 };

        // Aggregate all metrics for the same worker into a single snapshot
        // so we call store.update once per worker, not once per metric.
        let mut by_worker: std::collections::HashMap<String, WorkerSnapshot> =
            std::collections::HashMap::new();
        let mut count = 0;

        for res in data.result {
            let (Some(worker_url), Some(metric_name)) =
                (res.metric.get("worker_url"), res.metric.get("__name__"))
            else {
                continue;
            };

            match res.value.1.parse::<f64>() {
                Ok(val) => {
                    if val.is_finite() {
                        let snapshot = by_worker.entry(worker_url.clone()).or_insert_with(|| {
                            WorkerSnapshot::new(worker_url.clone(), MetricSource::Prometheus)
                        });
                        snapshot.custom_metrics.insert(metric_name.clone(), val);
                        count += 1;
                    }
                }
                Err(e) => {
                    // C4: log parse failures so operators can diagnose bad metric values
                    debug!(
                        metric = %metric_name,
                        raw_value = %res.value.1,
                        worker = %worker_url,
                        error = %e,
                        "PrometheusScraper: failed to parse metric value as f64"
                    );
                }
            }
        }

        for (_url, snapshot) in by_worker {
            self.store.update(snapshot);
        }

        count
    }
}
