// Ported from the Apache-2.0 reference `vllm-engine-core-client`
// (vllm-project/vllm): protocol/stats.rs.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::codec::OpaqueValue;

/// Base cache hit statistics. Mirrors Python `BaseCacheStats`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BaseCacheStats {
    pub reset: bool,
    pub requests: u64,
    pub queries: u64,
    pub hits: u64,
}

/// Prefix cache hit statistics. Mirrors Python `PrefixCacheStats`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PrefixCacheStats {
    #[serde(flatten)]
    pub base: BaseCacheStats,
    pub preempted_requests: u64,
    pub preempted_queries: u64,
    pub preempted_hits: u64,
}

/// Single KV cache block eviction sample. Mirrors Python `KvCacheEvictionEvent`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct KvCacheEvictionEvent {
    pub lifetime_seconds: f64,
    pub idle_seconds: f64,
    pub reuse_gaps_seconds: Vec<f64>,
}

/// Per-step speculative-decoding stats. Mirrors Python `SpecDecodingStats`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SpecDecodingStats {
    pub num_spec_tokens: u64,
    pub num_drafts: u64,
    pub num_draft_tokens: u64,
    pub num_accepted_tokens: u64,
    pub num_accepted_tokens_per_pos: Vec<u64>,
}

/// Breakdown of a scheduled prefill computation.
///
/// Python models this as a plain `@dataclass`, so msgspec serializes it as a
/// map (named fields), not the array-like form used by `EngineCoreOutput`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillStats {
    #[serde(default)]
    pub num_prompt_tokens: u32,
    #[serde(default)]
    pub num_computed_tokens: u32,
    #[serde(default)]
    pub num_cached_tokens: u32,
    #[serde(default)]
    pub num_local_cached_tokens: u32,
    #[serde(default)]
    pub num_external_cached_tokens: u32,
    #[serde(default)]
    pub num_cache_creation_tokens: u32,
}

/// Debug stats for the metrics calculation. Mirrors Python `DebugPerfStats`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DebugPerfStats {
    pub calc_duration: f64,
    pub num_prefill_requests: u64,
    pub num_decode_requests: u64,
    pub context_breakdown: Option<BTreeMap<String, u64>>,
    pub num_flops_per_gpu_breakdown: Option<BTreeMap<String, u64>>,
    pub num_read_bytes_per_gpu_breakdown: Option<BTreeMap<String, u64>>,
    pub num_write_bytes_per_gpu_breakdown: Option<BTreeMap<String, u64>>,
}

/// Estimated MFU/performance stats. Mirrors Python `PerfStats`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PerfStats {
    pub num_flops_per_gpu: u64,
    pub num_read_bytes_per_gpu: u64,
    pub num_write_bytes_per_gpu: u64,
    pub debug_stats: Option<DebugPerfStats>,
}

/// CUDA graph runtime stats. Mirrors Python `CudagraphStats`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CudagraphStats {
    pub num_unpadded_tokens: u64,
    pub num_padded_tokens: u64,
    pub num_paddings: u64,
    pub runtime_mode: String,
}

/// Scheduler stats piggybacked on every output batch. Mirrors Python
/// `SchedulerStats`.
///
/// `num_running_reqs` / `num_waiting_reqs` are SMG's per-rank DP load signal —
/// fresher than any polled `GetLoads`, and the reason the connector consumes
/// these instead of the reference's client-side load balancer.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SchedulerStats {
    /// Number of requests in model execution batches.
    pub num_running_reqs: u64,
    /// Length of the "waiting" request queue.
    pub num_waiting_reqs: u64,
    /// Length of the "skipped waiting" queue.
    #[serde(default)]
    pub num_skipped_waiting_reqs: u64,
    /// Internal DP load-balancing step counter.
    pub step_counter: u64,
    /// Internal DP load-balancing wave number.
    pub current_wave: u64,
    /// KV-cache usage. `1.0` means 100% usage.
    pub kv_cache_usage: f64,
    /// Local prefix cache statistics.
    pub prefix_cache_stats: PrefixCacheStats,
    /// External connector prefix cache statistics, when configured.
    pub connector_prefix_cache_stats: Option<PrefixCacheStats>,
    /// Sampled KV cache eviction events for residency metrics.
    pub kv_cache_eviction_events: Vec<KvCacheEvictionEvent>,
    /// Speculative decoding scheduler stats, when enabled.
    pub spec_decoding_stats: Option<SpecDecodingStats>,
    /// Connector-specific KV transfer stats, kept opaque for now.
    pub kv_connector_stats: Option<BTreeMap<String, OpaqueValue>>,
    /// CUDA graph runtime stats when graph metrics are enabled.
    pub cudagraph_stats: Option<CudagraphStats>,
    /// Estimated MFU/performance stats, when enabled.
    pub perf_stats: Option<PerfStats>,
}
