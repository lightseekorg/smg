//! Host-DRAM LRU cache of preprocessed, serialized per-image encoder inputs for
//! the gateway multimodal encoder path (EPD P1).
//!
//! A repeated image (same raw bytes, same model/preprocessor config, same wire
//! dtype) is preprocessed and serialized once; subsequent requests reuse the
//! cached payload, skipping the gateway-CPU heavy steps (resize + normalize +
//! patchify + f32->bf16 serialize) for that image. Decode still runs in the fetch
//! stage (post-decode lookup); the cache key is the raw-bytes blake3 hash that the
//! media connector already computes.
//!
//! The cached value is a pre-serialized per-image encoder payload (a down-cast
//! 16-bit wire tensor). A backend assembler benefits from it only if it accepts
//! that pre-serialized form; the backends that instead serialize the full-precision
//! tensor themselves declare `ensure_unserialized_encoder` as a precondition and
//! never consume the cache. Today the TokenSpeed assembler is the one that consumes
//! it, so operators enable the cache on a gateway whose multimodal backend reads the
//! pre-serialized form. Disabled by default (`SMG_MM_PIXEL_CACHE_MB` unset / 0 =>
//! zero behavior change).

use std::{
    collections::HashMap,
    sync::{Arc, OnceLock},
};

use lru::LruCache;
use parking_lot::Mutex;

use super::proto_wrapper::TensorBytes;

/// Identifies a preprocessed image output. Hashing the raw image bytes alone is
/// not sufficient: the same bytes preprocess differently under a different model
/// or preprocessor config, and serialize to different bytes under a different wire
/// dtype. The fingerprint is process-constant today but is in the key so a future
/// multi-model gateway (or a config hot-reload) cannot return stale pixels.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct PixelCacheKey {
    /// blake3 hex digest of the raw encoded image bytes (`ImageFrame.hash`).
    pub image_hash: String,
    /// Stable hash of (model identity + preprocessor config) for this deployment.
    pub config_fingerprint: u64,
    /// Wire dtype the encoder input was serialized to (e.g. "bfloat16").
    pub dtype: String,
}

/// The cached, fully-serialized per-image encoder payload (the down-cast wire form).
/// Holds everything `assemble_tokenspeed` needs for one item except the
/// request-positional fields (placeholders, content_hash) which are cheap to
/// recompute and depend on where the image sits in the prompt.
#[derive(Debug)]
pub(crate) struct CachedEncodeItem {
    /// Serialized encoder input (bf16 little-endian wire bytes). The heavy buffer.
    pub encoder_input: Vec<u8>,
    pub encoder_input_shape: Vec<u32>,
    pub encoder_input_dtype: String,
    /// Per-image model-specific tensors, already serialized (e.g. image_grid_thw).
    pub model_specific_tensors: HashMap<String, TensorBytes>,
    /// Number of encoder feature tokens for this image (drives placeholder expansion).
    pub feature_token_count: usize,
}

impl CachedEncodeItem {
    /// Approximate heap footprint of this entry's payload, for the byte budget.
    fn heap_bytes(&self) -> usize {
        let model_specific: usize = self
            .model_specific_tensors
            .iter()
            .map(|(key, tensor)| {
                key.len() + tensor.data.len() + tensor.shape.len() * 4 + tensor.dtype.len()
            })
            .sum();
        self.encoder_input.len()
            + self.encoder_input_shape.len() * 4
            + self.encoder_input_dtype.len()
            + model_specific
    }
}

/// Per-key bookkeeping overhead charged against the budget (key string + Arc + node).
const KEY_OVERHEAD: usize = 128;

struct Inner {
    map: LruCache<PixelCacheKey, Arc<CachedEncodeItem>>,
    cur_bytes: usize,
    max_bytes: usize,
}

/// A thread-safe, byte-budgeted LRU of serialized per-image encoder payloads.
/// Values are `Arc`-shared: lookups and the carry through the request pipeline are
/// refcount bumps, never deep copies; the heavy bytes are copied only once, when
/// the proto item is materialized on the leg that actually ships pixels.
pub(crate) struct PixelCache {
    inner: Mutex<Inner>,
}

impl PixelCache {
    pub(crate) fn new(max_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                // Unbounded by count; we evict by the byte budget instead.
                map: LruCache::unbounded(),
                cur_bytes: 0,
                max_bytes,
            }),
        }
    }

    /// Look up an entry, bumping its recency. Returns a cheap `Arc` clone on a hit.
    pub(crate) fn get(&self, key: &PixelCacheKey) -> Option<Arc<CachedEncodeItem>> {
        let mut inner = self.inner.lock();
        inner.map.get(key).cloned()
    }

    /// Insert an entry, evicting least-recently-used entries until back under
    /// budget. An item larger than the whole budget is skipped (inline bypass:
    /// the caller already holds the value, so the request still succeeds).
    pub(crate) fn insert(&self, key: PixelCacheKey, value: Arc<CachedEncodeItem>) {
        let entry_bytes = value.heap_bytes() + KEY_OVERHEAD;
        let mut inner = self.inner.lock();
        if entry_bytes > inner.max_bytes {
            return;
        }
        let Inner {
            map,
            cur_bytes,
            max_bytes,
        } = &mut *inner;
        if let Some(previous) = map.put(key, value) {
            *cur_bytes = cur_bytes.saturating_sub(previous.heap_bytes() + KEY_OVERHEAD);
        }
        *cur_bytes += entry_bytes;
        while *cur_bytes > *max_bytes {
            match map.pop_lru() {
                Some((_, evicted)) => {
                    *cur_bytes = cur_bytes.saturating_sub(evicted.heap_bytes() + KEY_OVERHEAD);
                }
                None => break,
            }
        }
    }

    #[cfg(test)]
    fn current_bytes(&self) -> usize {
        self.inner.lock().cur_bytes
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.inner.lock().map.len()
    }
}

/// Build the process-wide pixel cache from `SMG_MM_PIXEL_CACHE_MB`. Returns `None`
/// when unset or 0 (the default), in which case the multimodal path is byte-for-byte
/// unchanged. Parsed once and reused for the process lifetime.
pub(crate) fn pixel_cache_from_env() -> Option<Arc<PixelCache>> {
    static CACHE: OnceLock<Option<Arc<PixelCache>>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let mb = std::env::var("SMG_MM_PIXEL_CACHE_MB")
                .ok()
                .and_then(|raw| raw.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if mb == 0 {
                return None;
            }
            let max_bytes = mb.saturating_mul(1024 * 1024);
            tracing::info!(
                target: "smg::request",
                cache_mb = mb,
                "multimodal pixel_values cache enabled (host DRAM, pre-serialized encoder payloads)"
            );
            Some(Arc::new(PixelCache::new(max_bytes)))
        })
        .clone()
}

/// Stable per-deployment fingerprint folding model identity (`tokenizer_id`) and
/// the full model config into the cache key. Constant for a running gateway; it is
/// in the key so a different model/config can never collide on the same raw-bytes
/// hash. `config` is the model config.json (a `serde_json::Value`) whose
/// `to_string()` is deterministic for a given value.
pub(crate) fn config_fingerprint(tokenizer_id: &str, config: &serde_json::Value) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(tokenizer_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(config.to_string().as_bytes());
    let digest = hasher.finalize();
    // blake3 output is 32 bytes, so the first 8 always exist.
    let mut head = [0u8; 8];
    head.copy_from_slice(&digest.as_bytes()[..8]);
    u64::from_le_bytes(head)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(token_count: usize, payload: usize) -> Arc<CachedEncodeItem> {
        Arc::new(CachedEncodeItem {
            encoder_input: vec![0u8; payload],
            encoder_input_shape: vec![1, payload as u32],
            encoder_input_dtype: "bfloat16".to_string(),
            model_specific_tensors: HashMap::new(),
            feature_token_count: token_count,
        })
    }

    fn key(hash: &str) -> PixelCacheKey {
        PixelCacheKey {
            image_hash: hash.to_string(),
            config_fingerprint: 7,
            dtype: "bfloat16".to_string(),
        }
    }

    #[test]
    fn hit_returns_shared_arc() {
        let cache = PixelCache::new(1024 * 1024);
        let value = item(10, 64);
        cache.insert(key("a"), value.clone());
        let got = cache.get(&key("a")).expect("hit");
        assert_eq!(got.feature_token_count, 10);
        // Same allocation, not a copy.
        assert!(Arc::ptr_eq(&got, &value));
        assert!(cache.get(&key("missing")).is_none());
    }

    #[test]
    fn key_distinguishes_fingerprint_and_dtype() {
        let cache = PixelCache::new(1024 * 1024);
        let mut k_f16 = key("a");
        k_f16.dtype = "float16".to_string();
        let mut k_other_model = key("a");
        k_other_model.config_fingerprint = 99;
        cache.insert(key("a"), item(1, 8));
        cache.insert(k_f16.clone(), item(2, 8));
        cache.insert(k_other_model.clone(), item(3, 8));
        assert_eq!(cache.get(&key("a")).unwrap().feature_token_count, 1);
        assert_eq!(cache.get(&k_f16).unwrap().feature_token_count, 2);
        assert_eq!(cache.get(&k_other_model).unwrap().feature_token_count, 3);
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn evicts_lru_when_over_budget() {
        // Entry footprint = payload + shape (2 * u32) + dtype + per-key overhead.
        let entry = 4096 + 2 * 4 + "bfloat16".len() + KEY_OVERHEAD;
        // Budget fits exactly two entries, never three.
        let budget = entry * 2 + entry / 2;
        let cache = PixelCache::new(budget);
        cache.insert(key("a"), item(1, 4096));
        cache.insert(key("b"), item(2, 4096));
        assert!(cache.get(&key("a")).is_some(), "both fit before overflow");
        assert!(cache.get(&key("b")).is_some());
        // Touch "a" so "b" becomes the LRU victim, then overflow with "c".
        assert!(cache.get(&key("a")).is_some());
        cache.insert(key("c"), item(3, 4096));
        assert!(cache.get(&key("a")).is_some(), "recently used survives");
        assert!(cache.get(&key("b")).is_none(), "LRU evicted");
        assert!(cache.get(&key("c")).is_some(), "newest survives");
        assert!(cache.current_bytes() <= budget);
    }

    #[test]
    fn oversized_item_is_skipped_not_stored() {
        let cache = PixelCache::new(1024);
        cache.insert(key("big"), item(1, 4096));
        assert!(cache.get(&key("big")).is_none());
        assert_eq!(cache.current_bytes(), 0);
    }

    #[test]
    fn reinsert_same_key_does_not_double_count() {
        let cache = PixelCache::new(1024 * 1024);
        cache.insert(key("a"), item(1, 4096));
        let after_first = cache.current_bytes();
        cache.insert(key("a"), item(2, 4096));
        assert_eq!(cache.current_bytes(), after_first);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&key("a")).unwrap().feature_token_count, 2);
    }

    #[test]
    fn fingerprint_is_stable_and_sensitive() {
        let cfg_a = serde_json::json!({"model_type": "qwen3_vl", "x": 1});
        let cfg_b = serde_json::json!({"model_type": "qwen3_vl", "x": 2});
        assert_eq!(
            config_fingerprint("tok", &cfg_a),
            config_fingerprint("tok", &cfg_a)
        );
        assert_ne!(
            config_fingerprint("tok", &cfg_a),
            config_fingerprint("tok", &cfg_b)
        );
        assert_ne!(
            config_fingerprint("tok-1", &cfg_a),
            config_fingerprint("tok-2", &cfg_a)
        );
    }
}
