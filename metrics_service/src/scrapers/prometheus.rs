use std::{sync::Arc, time::Duration};

use reqwest::Client;
use serde::Deserialize;

use crate::{
    store::MetricsStore,
    types::{MetricSource, WorkerSnapshot},
};

pub struct PrometheusScraper {
    store: Arc<MetricsStore>,
    client: Client,
    interval: Duration,
    prom_url: String,
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
        Self {
            store,
            client: Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap_or_default(),
            interval,
            prom_url,
        }
    }

    pub async fn run(&self) {
        let mut ticker = tokio::time::interval(self.interval);

        loop {
            ticker.tick().await;

            // Example custom query metric collection (all custom metrics prefixed with custom_)
            let query_url = format!(
                "{}/api/v1/query?query={{__name__=~\"^custom_.*\"}}",
                self.prom_url
            );

            if let Ok(resp) = self.client.get(&query_url).send().await {
                if let Ok(prom_resp) = resp.json::<PromResponse>().await {
                    if prom_resp.status == "success" {
                        if let Some(data) = prom_resp.data {
                            for res in data.result {
                                if let (Some(worker_url), Some(metric_name)) =
                                    (res.metric.get("worker_url"), res.metric.get("__name__"))
                                {
                                    if let Ok(val) = res.value.1.parse::<f64>() {
                                        let mut snapshot = WorkerSnapshot::new(
                                            worker_url.clone(),
                                            MetricSource::Prometheus,
                                        );
                                        snapshot.custom_metrics.insert(metric_name.clone(), val);
                                        self.store.update(snapshot);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
