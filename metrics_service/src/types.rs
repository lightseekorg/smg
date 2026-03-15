use std::{collections::HashMap, time::SystemTime};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MetricSource {
    Prometheus = 25,
    DirectScrape = 50,
    Piggyback = 100,
}

#[derive(Debug, Clone)]
pub struct WorkerSnapshot {
    pub url: String,
    pub seq_no: u64,
    pub source: MetricSource,
    pub timestamp: SystemTime,

    // Core metrics used by load balancer
    pub kv_cache_tokens: Option<isize>,
    pub in_flight_requests: isize,
    pub avg_tokens_per_req: isize,

    // Extensibility for CEL policies and observability
    pub custom_metrics: HashMap<String, f64>,
}

impl WorkerSnapshot {
    pub fn new(url: String, source: MetricSource) -> Self {
        Self {
            url,
            seq_no: 0,
            source,
            timestamp: SystemTime::now(),
            kv_cache_tokens: None,
            in_flight_requests: 0,
            avg_tokens_per_req: 0,
            custom_metrics: Default::default(),
        }
    }
}
