use std::{collections::BTreeSet, path::Path};

use serde_json::Value;

use crate::traits::TokenIdType;

fn collect_eos_ids(cfg: &Value, ids: &mut BTreeSet<TokenIdType>) {
    match cfg.get("eos_token_id") {
        Some(Value::Number(n)) => {
            if let Some(id) = n.as_u64() {
                ids.insert(id as TokenIdType);
            }
        }
        Some(Value::Array(arr)) => {
            for v in arr {
                if let Some(id) = v.as_u64() {
                    ids.insert(id as TokenIdType);
                }
            }
        }
        _ => {}
    }
}

/// Load and merge EOS token IDs from `config.json` and `generation_config.json`.
///
/// Models may define different EOS tokens in each file (e.g. Kimi-K2.5 uses
/// `[EOS]` (163585) in config.json and `<|im_end|>` (163586) in
/// generation_config.json). We merge both into a deduplicated, sorted list so
/// the StopDecoder can strip any of them before decoding.
///
/// This matches how vllm and sglang resolve EOS:
/// - vllm: `hf_config.eos_token_id` (from config.json via AutoConfig) +
///   `generation_config.eos_token_id` (merged in `update_from_generation_config`)
/// - sglang: `model_info["eos_token_ids"]` (from model config, includes both sources)
pub fn load_eos_token_ids(dir: &Path) -> Vec<TokenIdType> {
    let mut ids = BTreeSet::new();

    for filename in ["config.json", "generation_config.json"] {
        if let Ok(content) = std::fs::read_to_string(dir.join(filename)) {
            if let Ok(cfg) = serde_json::from_str::<Value>(&content) {
                collect_eos_ids(&cfg, &mut ids);
            }
        }
    }

    ids.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_single_int() {
        let cfg: Value = serde_json::from_str(r#"{"eos_token_id": 42}"#).unwrap();
        let mut ids = BTreeSet::new();
        collect_eos_ids(&cfg, &mut ids);
        assert_eq!(ids.into_iter().collect::<Vec<_>>(), vec![42]);
    }

    #[test]
    fn test_collect_array() {
        let cfg: Value = serde_json::from_str(r#"{"eos_token_id": [10, 20, 30]}"#).unwrap();
        let mut ids = BTreeSet::new();
        collect_eos_ids(&cfg, &mut ids);
        assert_eq!(ids.into_iter().collect::<Vec<_>>(), vec![10, 20, 30]);
    }

    #[test]
    fn test_collect_missing_field() {
        let cfg: Value = serde_json::from_str(r#"{"model_type": "llama"}"#).unwrap();
        let mut ids = BTreeSet::new();
        collect_eos_ids(&cfg, &mut ids);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_merge_deduplicates() {
        let mut ids = BTreeSet::new();
        let cfg1: Value = serde_json::from_str(r#"{"eos_token_id": 100}"#).unwrap();
        let cfg2: Value = serde_json::from_str(r#"{"eos_token_id": [100, 200]}"#).unwrap();
        collect_eos_ids(&cfg1, &mut ids);
        collect_eos_ids(&cfg2, &mut ids);
        assert_eq!(ids.into_iter().collect::<Vec<_>>(), vec![100, 200]);
    }

    #[test]
    fn test_load_from_nonexistent_dir() {
        let ids = load_eos_token_ids(Path::new("/nonexistent/path"));
        assert!(ids.is_empty());
    }
}
