use std::{sync::Arc, time::Duration};

use reqwest::Client;
use tokio::sync::Semaphore;

use crate::{
    store::MetricsStore,
    types::{MetricSource, WorkerSnapshot},
};

/// Maximum number of concurrent per-worker scrape tasks.
const MAX_CONCURRENT_SCRAPES: usize = 16;

pub struct DirectScraper {
    store: Arc<MetricsStore>,
    client: Client,
    interval: Duration,
    semaphore: Arc<Semaphore>,
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
            semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_SCRAPES)),
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
                let semaphore = Arc::clone(&self.semaphore);

                #[expect(
                    clippy::disallowed_methods,
                    reason = "background scrape task; gateway shutdown without waiting for it is acceptable"
                )]
                tokio::spawn(async move {
                    // Acquire a permit before any network I/O; bounds concurrent
                    // scrape tasks to MAX_CONCURRENT_SCRAPES regardless of fleet size.
                    let Ok(_permit) = semaphore.acquire_owned().await else {
                        return; // Semaphore closed — shutting down
                    };

                    let mut snapshot =
                        WorkerSnapshot::new(w_url.clone(), MetricSource::DirectScrape);
                    let mut should_publish = false;

                    // SGLang + vLLM: Prometheus text format (Content-Type: text/plain)
                    // TRT-LLM:       JSON array format     (Content-Type: application/json)
                    // We detect the format from the Content-Type header and dispatch accordingly.
                    // Capped at 1 MiB to prevent OOM from a malicious/misconfigured worker.
                    // Body is read chunk-by-chunk so the cap aborts the read early, before
                    // the full response is buffered.
                    const SIZE_CAP: usize = 1 << 20; // 1 MiB
                    let metrics_url = format!("{w_url}/metrics");
                    match client.get(&metrics_url).send().await {
                        Ok(resp) => {
                            // Sniff content type before consuming the body
                            let is_json = resp
                                .headers()
                                .get(reqwest::header::CONTENT_TYPE)
                                .and_then(|v| v.to_str().ok())
                                .map(|ct| ct.contains("application/json"))
                                .unwrap_or(false);

                            // Stream body chunk-by-chunk; bail out if we exceed 1 MiB
                            let mut body: Vec<u8> = Vec::new();
                            let mut stream = resp.bytes_stream();
                            let mut capped = false;
                            loop {
                                use futures::StreamExt as _;
                                match stream.next().await {
                                    None => break,
                                    Some(Err(e)) => {
                                        tracing::debug!(
                                            worker = %w_url,
                                            error = %e,
                                            "failed to read /metrics response body"
                                        );
                                        return;
                                    }
                                    Some(Ok(chunk)) => {
                                        let remaining_bytes = SIZE_CAP.saturating_sub(body.len());
                                        let take = std::cmp::min(chunk.len(), remaining_bytes);
                                        body.extend_from_slice(&chunk[..take]);

                                        if chunk.len() > remaining_bytes {
                                            tracing::warn!(
                                                worker = %w_url,
                                                bytes = body.len(),
                                                "/metrics response exceeds 1 MiB size cap; skipping"
                                            );
                                            capped = true;
                                            break;
                                        }
                                    }
                                }
                            }

                            if !capped {
                                if is_json {
                                    // TRT-LLM: JSON array of iteration stats
                                    apply_trtllm_json(&mut snapshot, &body, &w_url);
                                    should_publish = true;
                                } else {
                                    // SGLang / vLLM: Prometheus text exposition
                                    match std::str::from_utf8(&body) {
                                        Ok(text) => {
                                            apply_metrics_text(&mut snapshot, text);
                                            should_publish = true;
                                        }
                                        Err(e) => {
                                            tracing::debug!(
                                                worker = %w_url,
                                                error = %e,
                                                "/metrics response is not valid UTF-8"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            tracing::debug!(
                                worker = %w_url,
                                error = %e,
                                "failed to fetch /metrics"
                            );
                        }
                    }

                    if should_publish {
                        store.update(snapshot);
                    }
                });
            }
        }
    }
}

/// Parse TRT-LLM's JSON iteration-stats format and apply to `snapshot`.
///
/// TRT-LLM `/metrics` returns a JSON array of iteration records:
/// ```json
/// [{ "numActiveRequests": 1, "gpuMemUsage": 76665782272,
///    "kvCacheStats": { "usedNumBlocks": 3, "maxNumBlocks": 101256,
///                      "cacheHitRate": 0.00128, "tokensPerBlock": 32 } }]
/// ```
/// We use the **last** entry (most recent iteration) and:
/// - `numActiveRequests` → `in_flight_requests`
/// - `usedNumBlocks × tokensPerBlock` → `kv_cache_tokens`
/// - `cacheHitRate` → `custom_metrics["trtllm_kv_cache_hit_rate"]`
/// - `gpuMemUsage`  → `custom_metrics["trtllm_gpu_mem_usage_bytes"]`
pub(crate) fn apply_trtllm_json(snapshot: &mut WorkerSnapshot, bytes: &[u8], worker_url: &str) {
    let Ok(arr) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        tracing::debug!(
            worker = %worker_url,
            "/metrics TRT-LLM JSON parse failed"
        );
        return;
    };

    // Use the last entry — TRT-LLM returns one object per iteration in the queue.
    let entry = match &arr {
        serde_json::Value::Array(a) => match a.last() {
            Some(e) => e,
            None => {
                tracing::debug!(worker = %worker_url, "/metrics TRT-LLM JSON array is empty");
                return;
            }
        },
        // Some TRT-LLM builds return a single object rather than an array.
        obj @ serde_json::Value::Object(_) => obj,
        _ => {
            tracing::debug!(
                worker = %worker_url,
                "/metrics TRT-LLM JSON is neither array nor object"
            );
            return;
        }
    };

    // in_flight_requests ← numActiveRequests
    if let Some(n) = entry.get("numActiveRequests").and_then(|v| v.as_i64()) {
        snapshot.in_flight_requests = n as isize;
    }

    // GPU memory → custom metric
    if let Some(mem) = entry.get("gpuMemUsage").and_then(|v| v.as_f64()) {
        snapshot
            .custom_metrics
            .insert("trtllm_gpu_mem_usage_bytes".to_string(), mem);
    }

    // KV cache stats
    if let Some(kv) = entry.get("kvCacheStats").and_then(|v| v.as_object()) {
        // kv_cache_tokens ← usedNumBlocks × tokensPerBlock
        let used = kv
            .get("usedNumBlocks")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let tokens_per_block = kv
            .get("tokensPerBlock")
            .and_then(|v| v.as_i64())
            .unwrap_or(1);
        if used > 0 {
            snapshot.kv_cache_tokens = Some((used * tokens_per_block) as isize);
        }

        // KV cache utilization fraction → custom metric
        if let Some(hit) = kv.get("cacheHitRate").and_then(|v| v.as_f64()) {
            snapshot
                .custom_metrics
                .insert("trtllm_kv_cache_hit_rate".to_string(), hit);
        }

        // Free/used/max blocks → custom metrics (useful for CEL policies)
        for key in &["freeNumBlocks", "usedNumBlocks", "maxNumBlocks"] {
            if let Some(v) = kv.get(*key).and_then(|v| v.as_f64()) {
                snapshot.custom_metrics.insert(format!("trtllm_{key}"), v);
            }
        }
    }

    // iterLatencyMS → custom metric
    if let Some(lat) = entry.get("iterLatencyMS").and_then(|v| v.as_f64()) {
        snapshot
            .custom_metrics
            .insert("trtllm_iter_latency_ms".to_string(), lat);
    }
}

/// Parse one Prometheus text-format exposition and apply recognized metrics to `snapshot`.
///
/// Used for SGLang and vLLM, which expose standard Prometheus text format at `/metrics`.
/// This is a separate function so it can be unit-tested without a live HTTP server.
pub(crate) fn apply_metrics_text(snapshot: &mut WorkerSnapshot, text: &str) {
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Format: `metric_name{labels} value [timestamp_ms]`
        // Prometheus allows an optional Unix-ms timestamp as a third token.
        // We must split on the FIRST space that follows the metric/label part,
        // not the last space — otherwise the timestamp is mistaken for the value.
        // Strategy: collect up to 3 whitespace-delimited tokens.
        let mut tokens = line.splitn(3, [' ', '\t']);
        let key_part = match tokens.next() {
            Some(k) => k,
            None => continue,
        };
        let val_str = match tokens.next() {
            Some(v) => v.trim(),
            None => continue,
        };
        // Third token (if present) is the timestamp — ignored.

        if let Ok(val) = val_str.parse::<f64>() {
            // Skip non-finite values (+Inf, -Inf, NaN) — they are valid f64
            // parses (Prometheus uses +Inf in histograms) but would corrupt
            // counters (as isize) and break CEL scoring.
            if !val.is_finite() {
                continue;
            }
            // Strip label block to get the bare metric name
            let key = match key_part.find('{') {
                Some(idx) => key_part[..idx].trim(),
                None => key_part.trim(),
            };

            // Route well-known routing metrics to native snapshot fields.
            // SGLang:  sglang: prefix — names confirmed at docs.sglang.io/references/production_metrics
            // vLLM:    vllm: prefix   — names confirmed at docs.vllm.ai/en/stable/design/metrics/
            // TRT-LLM: trtllm_ prefix — stable v1 exposes 5 histogram/counter metrics only.
            //          trtllm_inflight_reqs and trtllm_request_count_active are forward-compatible
            //          aliases for when TRT-LLM ships inflight batcher gauge support.
            match key {
                //  In-flight requests
                "sglang:num_running_reqs"
                | "sglang:in_flight_requests"
                | "vllm:num_requests_running"
                | "trtllm_inflight_reqs" // forward-compatible: not yet in stable TRT-LLM
                | "trtllm_request_count_active" => {
                    // forward-compatible: not yet in stable TRT-LLM
                    snapshot.in_flight_requests = val as isize;
                }
                //  KV cache occupancy
                //  SGLang: num_used_tokens → raw token count (primary)
                //          token_usage     → fraction 0-1 (secondary; used when num_used_tokens absent)
                //  vLLM:   kv_cache_usage_perc → fraction 0-1, scaled ×1000
                //  TRT-LLM via Prometheus: kv_cache_utilization → fraction 0-1, scaled ×1000
                //  TRT-LLM via JSON: apply_trtllm_json() sets kv_cache_tokens directly
                "sglang:num_used_tokens" => {
                    snapshot.kv_cache_tokens = Some(val as isize);
                }
                "sglang:token_usage"
                | "vllm:kv_cache_usage_perc"
                | "trtllm_kv_cache_utilization" => {
                    // Fraction (0–1) → scale ×1000 so load balancer comparisons are meaningful.
                    // For sglang:token_usage, only use it if num_used_tokens hasn't been set yet.
                    if key == "sglang:token_usage" && snapshot.kv_cache_tokens.is_some() {
                        snapshot.custom_metrics.insert(key.to_string(), val);
                    } else {
                        snapshot.kv_cache_tokens = Some((val * 1000.0) as isize);
                    }
                }
                // Everything else → custom_metrics (CEL + Prometheus)
                _ => {
                    snapshot.custom_metrics.insert(key.to_string(), val);
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

    // Prometheus text parser tests (SGLang / vLLM)

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
    fn test_sglang_num_running_reqs_routed_to_native_field() {
        let mut snap = empty_snap();
        apply_metrics_text(&mut snap, "sglang:num_running_reqs 9");
        assert_eq!(snap.in_flight_requests, 9);
    }

    #[test]
    fn test_sglang_num_used_tokens_routed_to_kv_cache() {
        let mut snap = empty_snap();
        apply_metrics_text(&mut snap, "sglang:num_used_tokens 123859");
        assert_eq!(snap.kv_cache_tokens, Some(123859));
        assert!(!snap.custom_metrics.contains_key("sglang:num_used_tokens"));
    }

    #[test]
    fn test_vllm_running_routed_to_native_field() {
        let mut snap = empty_snap();
        apply_metrics_text(&mut snap, "vllm:num_requests_running 3");
        assert_eq!(snap.in_flight_requests, 3);
    }

    #[test]
    fn test_vllm_kv_cache_usage_perc_scaled() {
        let mut snap = empty_snap();
        apply_metrics_text(&mut snap, "vllm:kv_cache_usage_perc 0.75");
        // 0.75 × 1000 = 750
        assert_eq!(snap.kv_cache_tokens, Some(750));
        assert!(!snap.custom_metrics.contains_key("vllm:kv_cache_usage_perc"));
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
    fn test_sglang_token_usage_routed_to_kv_cache_when_no_num_used_tokens() {
        // token_usage is the KV cache fraction (0-1); used when num_used_tokens is absent
        let mut snap = empty_snap();
        apply_metrics_text(&mut snap, "sglang:token_usage 0.6");
        assert_eq!(snap.kv_cache_tokens, Some(600)); // 0.6 × 1000
        assert!(!snap.custom_metrics.contains_key("sglang:token_usage"));
    }

    #[test]
    fn test_sglang_token_usage_deferred_to_custom_when_num_used_tokens_present() {
        // If num_used_tokens already set kv_cache_tokens, token_usage goes to custom_metrics
        let mut snap = empty_snap();
        apply_metrics_text(&mut snap, "sglang:num_used_tokens 50000");
        apply_metrics_text(&mut snap, "sglang:token_usage 0.6");
        assert_eq!(snap.kv_cache_tokens, Some(50000)); // num_used_tokens wins
        assert!(snap.custom_metrics.contains_key("sglang:token_usage")); // token_usage still stored
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
            "sglang:num_used_tokens 40000\n",
            "sglang:cache_hit_rate 0.9\n",
            "vllm:gpu_cache_usage_perc 0.6\n",
        );
        apply_metrics_text(&mut snap, text);
        assert_eq!(snap.in_flight_requests, 10);
        assert_eq!(snap.kv_cache_tokens, Some(40000));
        assert_eq!(snap.custom_metrics.len(), 2); // cache_hit_rate + gpu_cache_usage_perc
    }

    #[test]
    fn test_float_nan_and_inf_ignored() {
        let mut snap = empty_snap();
        // +Inf is a valid f64 parse but is non-finite — must be skipped
        apply_metrics_text(&mut snap, "some_metric +Inf");
        assert!(
            snap.custom_metrics.is_empty(),
            "+Inf must not enter custom_metrics"
        );
        // NaN string does not parse as f64 at all
        apply_metrics_text(&mut snap, "sglang:in_flight_requests NaN");
        assert_eq!(snap.in_flight_requests, 0);
    }

    // TRT-LLM JSON parser tests

    #[test]
    fn test_trtllm_json_array_basic() {
        let mut snap = empty_snap();
        let json = r#"[{"numActiveRequests": 5, "gpuMemUsage": 1234567,
                         "kvCacheStats": {"usedNumBlocks": 3, "maxNumBlocks": 100,
                                          "tokensPerBlock": 32, "cacheHitRate": 0.5,
                                          "freeNumBlocks": 97}}]"#;
        apply_trtllm_json(&mut snap, json.as_bytes(), "http://test");
        assert_eq!(snap.in_flight_requests, 5);
        assert_eq!(snap.kv_cache_tokens, Some(3 * 32)); // usedNumBlocks × tokensPerBlock
        assert!(snap.custom_metrics.contains_key("trtllm_kv_cache_hit_rate"));
        assert!(snap
            .custom_metrics
            .contains_key("trtllm_gpu_mem_usage_bytes"));
    }

    #[test]
    fn test_trtllm_json_uses_last_entry() {
        let mut snap = empty_snap();
        // Two iterations — we want the last one (numActiveRequests=9)
        let json = r#"[{"numActiveRequests": 1, "kvCacheStats": {}},
                        {"numActiveRequests": 9, "kvCacheStats": {}}]"#;
        apply_trtllm_json(&mut snap, json.as_bytes(), "http://test");
        assert_eq!(snap.in_flight_requests, 9);
    }

    #[test]
    fn test_trtllm_json_single_object() {
        // Some TRT-LLM builds return an object, not an array
        let mut snap = empty_snap();
        let json = r#"{"numActiveRequests": 3, "kvCacheStats": {"usedNumBlocks": 10, "tokensPerBlock": 16}}"#;
        apply_trtllm_json(&mut snap, json.as_bytes(), "http://test");
        assert_eq!(snap.in_flight_requests, 3);
        assert_eq!(snap.kv_cache_tokens, Some(10 * 16));
    }

    #[test]
    fn test_trtllm_json_empty_array_ignored() {
        let mut snap = empty_snap();
        apply_trtllm_json(&mut snap, b"[]", "http://test");
        assert_eq!(snap.in_flight_requests, 0);
        assert_eq!(snap.kv_cache_tokens, None);
    }

    #[test]
    fn test_trtllm_json_invalid_ignored() {
        let mut snap = empty_snap();
        apply_trtllm_json(&mut snap, b"not json at all", "http://test");
        assert_eq!(snap.in_flight_requests, 0);
    }
}
