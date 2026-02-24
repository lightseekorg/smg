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

                    // A real implementation might also hit `/metrics` here for custom_metrics
                    // and merge into snapshot before pushing. Let's push token count.
                    store.update(snapshot);
                });
            }
        }
    }
}
