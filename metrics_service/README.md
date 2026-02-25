# metrics-service

## Overview

`metrics-service` is an event-driven crate that collects real-time operational metrics from LLM inference workers (SGLang, vLLM, and TensorRT-LLM) and makes them available to the `model_gateway` for load-balancing decisions and Prometheus observability.

The crate implements a three-tier collection pipeline:

```
Workers                 metrics-service            model_gateway
──────────────────────────────────────────────────────────────
/metrics (SGLang etc.) ──► DirectScraper  ─┐
Prometheus server      ──► PromScraper    ─┼──► MetricsStore ──► EventBus
Response headers/gRPC  ──► Piggyback      ─┘         │
                                                      ├──► CelPolicyEngine   (routing)
                                                      └──► ObsExporter       (Prometheus /metrics)
```

Key design decisions:
- **Source priority**: Piggyback (100) > DirectScrape (50) > Prometheus (25) — a higher-quality source never gets overwritten by a lower-quality one while fresh.
- **Monotonic seq-numbers**: Stale updates from the same source are silently dropped.
- **`custom_metrics` merging**: Updates from different sources are merged rather than replaced, so e.g. a Prometheus update for `gpu_util` doesn't erase a DirectScrape value for `cache_hit_rate`.
- **Snapshot-then-subscribe**: `MetricsStore::subscribe()` returns both existing snapshots and a live receiver in one atomic step, eliminating subscriber cold-start gaps.

## Source Map

```
metrics_service/
  src/
    lib.rs          – crate re-exports
    types.rs        – WorkerSnapshot, MetricSource
    store.rs        – MetricsStore (DashMap cache, priority/seq merging, subscribe API)
    bus.rs          – EventBus (tokio broadcast), recv_or_skip lag helper
    scrapers/
      mod.rs
      direct.rs     – DirectScraper (polls /get_load + /metrics per worker)
      prometheus.rs – PrometheusScraper (configurable PromQL selector)
```

## Backend Support

The crate is backend-agnostic: every worker backend exposes standard Prometheus text format on `/metrics`. The `DirectScraper` routes the following well-known metric names to native `WorkerSnapshot` fields:

| Metric name (Prometheus) | Backend | Mapped to |
|---|---|---|
| `sglang:in_flight_requests` | SGLang | `in_flight_requests` |
| `sglang:avg_tokens_per_req` | SGLang | `avg_tokens_per_req` |
| `vllm:num_requests_running` | vLLM | `in_flight_requests` |
| `trtllm_inflight_reqs` | TensorRT-LLM | `in_flight_requests` |
| `trtllm_request_count_active` | TensorRT-LLM | `in_flight_requests` |
| `trtllm_kv_cache_utilization` | TensorRT-LLM | mapped via `custom_metrics` |

All other metrics (SGLang, vLLM, TRT-LLM, user-defined) are stored in `WorkerSnapshot.custom_metrics` and:
1. Forwarded to `CelPolicyEngine` CEL expressions (accessible by name)
2. Exported as `smg_worker_<name>` Prometheus gauges by the observability exporter

### Piggyback collection

HTTP workers can also proactively push metrics via response headers (no extra round-trip):

| Header | Mapped to |
|---|---|
| `X-SGLang-KV-Cache-Tokens` | `kv_cache_tokens` |
| `X-SGLang-In-Flight-Requests` | `in_flight_requests` |
| `X-SGLang-Avg-Tokens-Per-Req` | `avg_tokens_per_req` |
| `X-SGLang-*` (arbitrary) | `custom_metrics` |

## Core Types

### `WorkerSnapshot` (`types.rs`)

```rust
pub struct WorkerSnapshot {
    pub url: String,
    pub seq_no: u64,
    pub source: MetricSource,
    pub timestamp: SystemTime,

    // Routing-critical fields
    pub kv_cache_tokens: Option<isize>,
    pub in_flight_requests: isize,
    pub avg_tokens_per_req: isize,

    // All other metrics (CEL-accessible and Prometheus-exported)
    pub custom_metrics: HashMap<String, f64>,
}
```

### `MetricSource` (priority ordering)

```rust
pub enum MetricSource {
    Prometheus = 25,
    DirectScrape = 50,
    Piggyback = 100,   // highest priority — real-time, zero-latency
}
```

## `MetricsStore` (`store.rs`)

Thread-safe metric cache backed by `DashMap`. Key methods:

| Method | Description |
|---|---|
| `update(snapshot)` | Apply a new snapshot; enforces source priority and monotonic seq-no |
| `get(url)` | Look up the latest snapshot for a specific worker URL |
| `get_all()` | Return snapshots for all known workers |
| `subscribe()` | Snapshot-then-subscribe: returns `(receiver, Vec<current_snapshots>)` atomically |

## `EventBus` (`bus.rs`)

Thin wrapper around `tokio::broadcast`. Exposes:

- `publish(snapshot)` — emit to all subscribers
- `subscribe()` — raw `broadcast::Receiver`
- `subscribe_with_lag_tracking()` — returns receiver + `Arc<AtomicU64>` lag counter
- `recv_or_skip(rx, counter)` — lag-safe receive; skips missed messages, logs and counts drops

## Scrapers

### `DirectScraper` (`scrapers/direct.rs`)

Polls each live worker over HTTP on a configurable interval:

1. `GET {url}/get_load` → JSON array of `{num_tokens}` objects → `kv_cache_tokens`
2. `GET {url}/metrics` → Prometheus text format → parses and routes metrics

Worker URLs are resolved via a `Weak<WorkerRegistry>` callback injected at startup — the scraper shares the live registry and stops gracefully when the gateway shuts down.

Configuration (from `RouterConfig.metrics_scraper`):
```toml
[metrics_scraper]
scrape_interval_secs = 5
```

### `PrometheusScraper` (`scrapers/prometheus.rs`)

Polls a Prometheus HTTP API endpoint with a configurable PromQL selector:

```rust
PrometheusScraper::with_selector(store, url, interval, r#"{job="sgw_workers"}"#.to_string())
```

Default selector: `{__name__=~"^custom_.*"}`.

Configuration:
```toml
[metrics_scraper]
prometheus_url = "http://prometheus:9090"           # required to enable
prometheus_scrape_interval_secs = 15
```

Consecutive failure tracking logs structured warnings on every error type (HTTP error, non-success Prometheus status, parse failure, connection error).

## Configuration Reference

All fields are optional (`#[serde(default)]`). Existing configs parse without changes.

```toml
[metrics_scraper]
scrape_interval_secs = 5               # DirectScraper poll interval (default: 5)
prometheus_url = ""                    # Prometheus base URL; empty = disabled (default: disabled)
prometheus_scrape_interval_secs = 15   # PromScraper interval (default: 15)
staleness_threshold_secs = 60          # Age after which a snapshot is considered stale (default: 60)
```

## Metrics-Driven Routing Policy

The `MetricsDriven` routing policy uses `CelPolicyEngine` to score workers using CEL expressions against live `WorkerSnapshot` data:

```yaml
# config.yaml
policy:
  type: metrics_driven
  strategy: "min_kv_cache_tokens"   # or "min_in_flight", or any CEL expression
  fresh_threshold_secs: 10
  stale_threshold_secs: 60
```

Named strategies: `min_kv_cache_tokens` (default) · `min_in_flight`

Custom CEL expressions have access to:
- `kv_cache_tokens` (int)
- `in_flight_requests` (int)
- `avg_tokens_per_req` (int)
- `custom_metrics` (map of string → double) — e.g. `custom_metrics["sglang:cache_hit_rate"]`

Tiered fallback: **Fresh** (< `fresh_threshold_secs`) → **Stale** (< `stale_threshold_secs`) → **Round-Robin** → 503.

## Running Tests

```bash
# metrics-service unit tests (26 total across store, bus, direct scraper)
cargo test -p metrics-service

# model_gateway policy tests (CEL engine, factory, registry)
cargo test -p smg -- policies::

# Type-check everything
cargo check
```

## Known Limitations

- `kv_cache_tokens` is populated by `DirectScraper`'s `/get_load` call and by `X-SGLang-KV-Cache-Tokens` response headers. TRT-LLM does not expose the same JSON format; use `custom_metrics["trtllm_kv_cache_utilization"]` in a CEL expression for equivalent routing.
- gRPC streaming piggyback emits `in_flight = 1` at stream start only; the actual count is corrected on the next `DirectScraper` poll cycle.
- The `PrometheusScraper` only ships one snapshot per metric per scrape; if multiple workers share the same `worker_url` label value the last one wins.
