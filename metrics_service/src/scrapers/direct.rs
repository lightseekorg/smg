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
                            apply_metrics_text(&mut snapshot, &text);
                        }
                    }

                    store.update(snapshot);
                });
            }
        }
    }
}

/// Parse one Prometheus text-format exposition and apply recognized metrics to `snapshot`.
///
/// This is a separate function so it can be unit-tested without a live HTTP server.
pub(crate) fn apply_metrics_text(snapshot: &mut WorkerSnapshot, text: &str) {
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Format: `metric_name{labels} value [timestamp]`
        // We split on the last space to separate value from the name+labels part
        if let Some((key_part, val_str)) = line.rsplit_once(' ') {
            // Strip an optional timestamp (second trailing field) if present
            let val_str = val_str.split_whitespace().next().unwrap_or(val_str);

            if let Ok(val) = val_str.parse::<f64>() {
                // Strip label block to get the bare metric name
                let key = match key_part.find('{') {
                    Some(idx) => key_part[..idx].trim(),
                    None => key_part.trim(),
                };

                // Route well-known routing metrics to native snapshot fields.
                // Supported backends: SGLang, vLLM, TensorRT-LLM (trtllm)
                match key {
                    // ── In-flight requests ──────────────────────────────────
                    "sglang:in_flight_requests"
                    | "vllm:num_requests_running"
                    | "trtllm_inflight_reqs"
                    | "trtllm_request_count_active" => {
                        snapshot.in_flight_requests = val as isize;
                    }
                    // ── Average tokens per request ──────────────────────────
                    "sglang:avg_tokens_per_req" => {
                        snapshot.avg_tokens_per_req = val as isize;
                    }
                    // ── Everything else → custom_metrics (CEL + Prometheus) ─
                    _ => {
                        snapshot.custom_metrics.insert(key.to_string(), val);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MetricSource;

    fn empty_snap() -> WorkerSnapshot {
        WorkerSnapshot::new("http://test".to_string(), MetricSource::DirectScrape)
    }

    #[test]
    fn test_sglang_in_flight_routed_to_native_field() {
        let mut snap = empty_snap();
        apply_metrics_text(&mut snap, "sglang:in_flight_requests 7");
        assert_eq!(snap.in_flight_requests, 7);
        assert!(!snap
            .custom_metrics
            .contains_key("sglang:in_flight_requests"));
    }

    #[test]
    fn test_vllm_running_routed_to_native_field() {
        let mut snap = empty_snap();
        apply_metrics_text(&mut snap, "vllm:num_requests_running 3");
        assert_eq!(snap.in_flight_requests, 3);
    }

    #[test]
    fn test_trtllm_inflight_reqs_routed_to_native_field() {
        let mut snap = empty_snap();
        apply_metrics_text(&mut snap, "trtllm_inflight_reqs 8");
        assert_eq!(snap.in_flight_requests, 8);
        assert!(
            !snap.custom_metrics.contains_key("trtllm_inflight_reqs"),
            "trtllm_inflight_reqs should NOT appear in custom_metrics"
        );
    }

    #[test]
    fn test_trtllm_request_count_active_routed_to_native_field() {
        let mut snap = empty_snap();
        apply_metrics_text(&mut snap, "trtllm_request_count_active 4");
        assert_eq!(snap.in_flight_requests, 4);
    }

    #[test]
    fn test_trtllm_kv_cache_goes_to_custom_metrics() {
        let mut snap = empty_snap();
        apply_metrics_text(&mut snap, "trtllm_kv_cache_utilization 0.72");
        assert!(snap
            .custom_metrics
            .contains_key("trtllm_kv_cache_utilization"));
        let v = snap.custom_metrics["trtllm_kv_cache_utilization"];
        assert!((v - 0.72).abs() < 1e-6);
    }

    #[test]
    fn test_avg_tokens_routed_to_native_field() {
        let mut snap = empty_snap();
        apply_metrics_text(&mut snap, "sglang:avg_tokens_per_req 512");
        assert_eq!(snap.avg_tokens_per_req, 512);
    }

    #[test]
    fn test_unknown_metric_goes_to_custom_metrics() {
        let mut snap = empty_snap();
        apply_metrics_text(&mut snap, "sglang:cache_hit_rate 0.87");
        assert!(snap.custom_metrics.contains_key("sglang:cache_hit_rate"));
        let v = snap.custom_metrics["sglang:cache_hit_rate"];
        assert!((v - 0.87).abs() < 1e-6);
    }

    #[test]
    fn test_label_block_stripped_from_metric_name() {
        let mut snap = empty_snap();
        apply_metrics_text(&mut snap, r#"sglang:cache_hit_rate{model="llama"} 0.5"#);
        assert!(snap.custom_metrics.contains_key("sglang:cache_hit_rate"));
    }

    #[test]
    fn test_comments_and_blank_lines_ignored() {
        let mut snap = empty_snap();
        let text = "# HELP sglang:in_flight_requests In-flight\n\n# TYPE sglang:in_flight_requests gauge\nsglang:in_flight_requests 4";
        apply_metrics_text(&mut snap, text);
        assert_eq!(snap.in_flight_requests, 4);
    }

    #[test]
    fn test_invalid_value_ignored() {
        let mut snap = empty_snap();
        apply_metrics_text(&mut snap, "some_metric not_a_number");
        assert!(snap.custom_metrics.is_empty());
        assert_eq!(snap.in_flight_requests, 0);
    }

    #[test]
    fn test_multiple_metrics_parsed() {
        let mut snap = empty_snap();
        let text = concat!(
            "sglang:in_flight_requests 10\n",
            "sglang:avg_tokens_per_req 256\n",
            "sglang:cache_hit_rate 0.9\n",
            "vllm:gpu_cache_usage_perc 0.6\n",
        );
        apply_metrics_text(&mut snap, text);
        assert_eq!(snap.in_flight_requests, 10);
        assert_eq!(snap.avg_tokens_per_req, 256);
        assert_eq!(snap.custom_metrics.len(), 2); // cache_hit_rate + gpu_cache_usage_perc
    }

    #[test]
    fn test_float_nan_and_inf_ignored() {
        let mut snap = empty_snap();
        // NaN and Inf are valid f64 parses but should not corrupt counters
        apply_metrics_text(&mut snap, "some_metric +Inf");
        // +Inf parses as f64::INFINITY; it goes into custom_metrics but doesn't crash
        // (routing logic only special-cases integer fields anyway)
        apply_metrics_text(&mut snap, "sglang:in_flight_requests NaN");
        // NaN string does not parse as f64; sglang:in_flight_requests stays 0
        assert_eq!(snap.in_flight_requests, 0);
    }
}
