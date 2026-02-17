use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use anyhow::{Error, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use rustc_hash::FxHashMap;
use tiktoken_rs::{cl100k_base, p50k_base, p50k_edit, r50k_base, CoreBPE};

use crate::{
    chat_template::{
        load_chat_template_from_file, ChatTemplateContentFormat, ChatTemplateParams,
        ChatTemplateState,
    },
    factory::discover_chat_template_in_dir,
    traits::{Decoder, Encoder, Encoding, SpecialTokens, TokenIdType, Tokenizer as TokenizerTrait},
};

/// Regex pattern for cl100k_base tokenization.
///
/// This pattern is correct for OpenAI models and most open-source tiktoken models (e.g.
/// DeepSeek, Kimi K2). Some models use a different regex — for example, Kimi K2's native
/// regex includes `\p{Han}` for Chinese character splitting — but encode/decode roundtrips
/// still work correctly because BPE vocab handles tokenization; the regex only affects exact
/// token boundary placement. A future enhancement could parse the regex from HuggingFace's
/// `generation_config.json` or similar metadata.
const CL100K_BASE_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

type Rank = u32;

// ---------------------------------------------------------------------------
// Tiktoken-specific config parsing (from tokenizer_config.json)
// ---------------------------------------------------------------------------

/// Parsed `tokenizer_config.json` for tiktoken-based models.
#[derive(Default)]
struct TiktokenConfig {
    special_tokens: SpecialTokens,
    /// Token string -> ID mapping from `added_tokens_decoder`
    added_tokens: HashMap<String, TokenIdType>,
    chat_template: Option<String>,
}

/// Parse `tokenizer_config.json` for tiktoken-based models.
fn load_tiktoken_config(config_path: &Path) -> Result<TiktokenConfig> {
    let content = std::fs::read_to_string(config_path)?;
    let config: serde_json::Value = serde_json::from_str(&content)?;

    let added_tokens = parse_added_tokens_decoder(&config);
    let special_tokens = parse_special_tokens(&config);

    let chat_template = config
        .get("chat_template")
        .and_then(|v| v.as_str())
        .map(String::from);

    Ok(TiktokenConfig {
        special_tokens,
        added_tokens,
        chat_template,
    })
}

/// Parse `added_tokens_decoder` from config JSON.
///
/// Format: `{ "163584": { "content": "[BOS]", "special": true }, ... }`
fn parse_added_tokens_decoder(config: &serde_json::Value) -> HashMap<String, TokenIdType> {
    let mut tokens = HashMap::new();
    if let Some(added) = config
        .get("added_tokens_decoder")
        .and_then(|v| v.as_object())
    {
        for (id_str, token_info) in added {
            if let (Ok(id), Some(content)) = (
                id_str.parse::<TokenIdType>(),
                token_info.get("content").and_then(|v| v.as_str()),
            ) {
                tokens.insert(content.to_string(), id);
            }
        }
    }
    tokens
}

/// Extract named special tokens (bos, eos, unk, etc.) from config JSON.
///
/// Handles both string-valued tokens (`"bos_token": "<s>"`) and object-valued tokens
/// (`"bos_token": {"content": "<s>", "lstrip": false, ...}`) found in some HuggingFace models.
fn parse_special_tokens(config: &serde_json::Value) -> SpecialTokens {
    let get_str = |key: &str| {
        config.get(key).and_then(|v| {
            v.as_str()
                .map(String::from)
                .or_else(|| v.get("content").and_then(|c| c.as_str()).map(String::from))
        })
    };

    let additional: Vec<String> = config
        .get("additional_special_tokens")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    v.as_str()
                        .map(String::from)
                        .or_else(|| v.get("content").and_then(|c| c.as_str()).map(String::from))
                })
                .collect()
        })
        .unwrap_or_default();

    SpecialTokens {
        bos_token: get_str("bos_token"),
        eos_token: get_str("eos_token"),
        unk_token: get_str("unk_token"),
        sep_token: get_str("sep_token"),
        pad_token: get_str("pad_token"),
        cls_token: get_str("cls_token"),
        mask_token: get_str("mask_token"),
        additional_special_tokens: additional,
    }
}

/// Tiktoken tokenizer wrapper — supports both built-in OpenAI encodings and hub-loaded models.
pub struct TiktokenTokenizer {
    tokenizer: CoreBPE,
    #[allow(dead_code)]
    model: Option<TiktokenModel>,
    special_tokens: SpecialTokens,
    vocab: HashMap<String, TokenIdType>,
    reverse_vocab: HashMap<TokenIdType, String>,
    vocab_size: usize,
    chat_template: ChatTemplateState,
}

/// Supported Tiktoken models
#[derive(Debug, Clone, Copy)]
pub enum TiktokenModel {
    /// GPT-4, GPT-3.5-turbo, text-embedding-ada-002
    Cl100kBase,
    /// Codex models, text-davinci-002, text-davinci-003
    P50kBase,
    /// Use for edit models like text-davinci-edit-001, code-davinci-edit-001
    P50kEdit,
    /// GPT-3 models like davinci
    R50kBase,
}

impl TiktokenTokenizer {
    /// Create a new Tiktoken tokenizer for the specified built-in model
    pub fn new(model: TiktokenModel) -> Result<Self> {
        let tokenizer =
            match model {
                TiktokenModel::Cl100kBase => cl100k_base()
                    .map_err(|e| Error::msg(format!("Failed to load cl100k_base: {}", e)))?,
                TiktokenModel::P50kBase => p50k_base()
                    .map_err(|e| Error::msg(format!("Failed to load p50k_base: {}", e)))?,
                TiktokenModel::P50kEdit => p50k_edit()
                    .map_err(|e| Error::msg(format!("Failed to load p50k_edit: {}", e)))?,
                TiktokenModel::R50kBase => r50k_base()
                    .map_err(|e| Error::msg(format!("Failed to load r50k_base: {}", e)))?,
            };

        let special_tokens = Self::get_special_tokens_for_model(model);

        let vocab_size = match model {
            TiktokenModel::Cl100kBase => 100256,
            TiktokenModel::P50kBase | TiktokenModel::P50kEdit => 50281,
            TiktokenModel::R50kBase => 50257,
        };

        Ok(TiktokenTokenizer {
            tokenizer,
            model: Some(model),
            special_tokens,
            vocab: HashMap::new(),
            reverse_vocab: HashMap::new(),
            vocab_size,
            chat_template: ChatTemplateState::new(None),
        })
    }

    /// Create from a directory containing tiktoken.model + tokenizer_config.json
    pub fn from_dir(dir: &Path) -> Result<Self> {
        Self::from_dir_with_chat_template(dir, None)
    }

    /// Create from a directory with an optional chat template file path.
    /// Discovers the tiktoken model file automatically via `find_tiktoken_file`.
    pub fn from_dir_with_chat_template(
        dir: &Path,
        chat_template_path: Option<&str>,
    ) -> Result<Self> {
        let tiktoken_path = find_tiktoken_file(dir)?;
        Self::load_from_path(&tiktoken_path, chat_template_path)
    }

    /// Create from an exact tiktoken file path (`.tiktoken` or `tiktoken.model`).
    /// Looks for `tokenizer_config.json` in the same directory.
    pub fn from_file(tiktoken_path: &Path) -> Result<Self> {
        Self::from_file_with_chat_template(tiktoken_path, None)
    }

    /// Create from an exact tiktoken file path with an optional chat template.
    pub fn from_file_with_chat_template(
        tiktoken_path: &Path,
        chat_template_path: Option<&str>,
    ) -> Result<Self> {
        Self::load_from_path(tiktoken_path, chat_template_path)
    }

    /// Core loading logic shared by `from_dir` and `from_file` constructors.
    fn load_from_path(tiktoken_path: &Path, chat_template_path: Option<&str>) -> Result<Self> {
        // 1. Load BPE encoder from the exact file
        let tiktoken_path_str = tiktoken_path
            .to_str()
            .ok_or_else(|| Error::msg("Tiktoken file path is not valid UTF-8"))?;
        let encoder = load_tiktoken_bpe(tiktoken_path_str)?;

        // 2. Parse tokenizer_config.json from the same directory
        let dir = tiktoken_path
            .parent()
            .ok_or_else(|| Error::msg("Cannot determine parent directory of tiktoken file"))?;
        let config_path = dir.join("tokenizer_config.json");
        let config = if config_path.exists() {
            load_tiktoken_config(&config_path)?
        } else {
            TiktokenConfig::default()
        };

        // 3. Build special tokens encoder for CoreBPE (needs FxHashMap)
        let special_tokens_encoder: FxHashMap<String, Rank> = config
            .added_tokens
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();

        // 4. Calculate true vocab size from max token ID (handles sparse/reserved IDs),
        //    build string-based vocab maps (borrows encoder), then pass encoder by value to CoreBPE
        let vocab_size = encoder
            .values()
            .copied()
            .chain(special_tokens_encoder.values().copied())
            .max()
            .map(|id| id as usize + 1)
            .unwrap_or(0);
        let (vocab, reverse_vocab) = build_vocab_maps(&encoder, &config.added_tokens);
        let tokenizer = CoreBPE::new(encoder, special_tokens_encoder, CL100K_BASE_PATTERN)?;

        // 5. Load chat template — propagate errors for explicit paths,
        //    silently fall back for auto-discovery
        let chat_template = if let Some(p) = chat_template_path {
            load_chat_template_from_file(p)?
        } else {
            config.chat_template.or_else(|| {
                discover_chat_template_in_dir(dir)
                    .and_then(|p| load_chat_template_from_file(&p).ok().flatten())
            })
        };

        Ok(TiktokenTokenizer {
            tokenizer,
            model: None,
            special_tokens: config.special_tokens,
            vocab,
            reverse_vocab,
            vocab_size,
            chat_template: ChatTemplateState::new(chat_template),
        })
    }

    /// Create a tokenizer from a model string (e.g., "gpt-4", "gpt-3.5-turbo")
    pub fn from_model_name(model_name: &str) -> Result<Self> {
        let model = Self::model_from_name(model_name)?;
        Self::new(model)
    }

    /// Determine the appropriate model from a model name
    fn model_from_name(model_name: &str) -> Result<TiktokenModel> {
        if model_name.contains("gpt-4")
            || model_name.contains("gpt-3.5")
            || model_name.contains("turbo")
        {
            Ok(TiktokenModel::Cl100kBase)
        } else if model_name.contains("davinci-002")
            || model_name.contains("davinci-003")
            || model_name.contains("codex")
        {
            Ok(TiktokenModel::P50kBase)
        } else if model_name.contains("edit") {
            Ok(TiktokenModel::P50kEdit)
        } else if model_name.contains("davinci")
            || model_name.contains("curie")
            || model_name.contains("babbage")
            || model_name.contains("ada")
        {
            Ok(TiktokenModel::R50kBase)
        } else {
            Err(anyhow::anyhow!(
                "Unrecognized OpenAI model name: '{}'. Expected GPT-3, GPT-3.5, GPT-4, or related model names",
                model_name
            ))
        }
    }

    /// Get special tokens for a specific model
    fn get_special_tokens_for_model(model: TiktokenModel) -> SpecialTokens {
        match model {
            TiktokenModel::Cl100kBase => SpecialTokens {
                bos_token: Some("<|endoftext|>".to_string()),
                eos_token: Some("<|endoftext|>".to_string()),
                unk_token: None,
                sep_token: None,
                pad_token: Some("<|endoftext|>".to_string()),
                cls_token: None,
                mask_token: None,
                additional_special_tokens: vec![
                    "<|fim_prefix|>".to_string(),
                    "<|fim_middle|>".to_string(),
                    "<|fim_suffix|>".to_string(),
                    "<|endofprompt|>".to_string(),
                ],
            },
            _ => SpecialTokens {
                bos_token: Some("<|endoftext|>".to_string()),
                eos_token: Some("<|endoftext|>".to_string()),
                unk_token: None,
                sep_token: None,
                pad_token: Some("<|endoftext|>".to_string()),
                cls_token: None,
                mask_token: None,
                additional_special_tokens: vec![],
            },
        }
    }
}

/// Parse a .tiktoken / tiktoken.model file into a BPE encoder.
///
/// Format: each line is `<base64-encoded-token-bytes> <rank>`
fn load_tiktoken_bpe(path: &str) -> Result<FxHashMap<Vec<u8>, Rank>> {
    let content = std::fs::read_to_string(path)?;
    let mut encoder =
        FxHashMap::with_capacity_and_hasher(content.lines().count(), Default::default());
    for line in content.lines() {
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let token_b64 = parts
            .next()
            .ok_or_else(|| Error::msg("missing token in tiktoken file"))?;
        let rank_str = parts
            .next()
            .ok_or_else(|| Error::msg("missing rank in tiktoken file"))?;
        let token_bytes = STANDARD.decode(token_b64)?;
        let rank: Rank = rank_str.parse()?;
        encoder.insert(token_bytes, rank);
    }
    Ok(encoder)
}

/// Build string-level vocab from byte-level encoder + added tokens.
fn build_vocab_maps(
    encoder: &FxHashMap<Vec<u8>, Rank>,
    added_tokens: &HashMap<String, TokenIdType>,
) -> (HashMap<String, TokenIdType>, HashMap<TokenIdType, String>) {
    let capacity = encoder.len() + added_tokens.len();
    let mut vocab = HashMap::with_capacity(capacity);
    let mut reverse_vocab = HashMap::with_capacity(capacity);

    // BPE tokens (only valid UTF-8 sequences get string entries)
    for (token_bytes, &rank) in encoder {
        if let Ok(token_str) = std::str::from_utf8(token_bytes) {
            vocab.insert(token_str.to_string(), rank);
            reverse_vocab.insert(rank, token_str.to_string());
        }
    }

    // Special/added tokens (always valid UTF-8)
    for (token_str, &id) in added_tokens {
        vocab.insert(token_str.clone(), id);
        reverse_vocab.insert(id, token_str.clone());
    }

    (vocab, reverse_vocab)
}

/// Find a tiktoken model file in the given directory.
///
/// Looks for `tiktoken.model` first, then any `*.tiktoken` file.
fn find_tiktoken_file(dir: &Path) -> Result<PathBuf> {
    let tiktoken_model = dir.join("tiktoken.model");
    if tiktoken_model.exists() {
        return Ok(tiktoken_model);
    }

    // Look for *.tiktoken files
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".tiktoken") {
                    return Ok(entry.path());
                }
            }
        }
    }

    Err(Error::msg(format!(
        "No tiktoken model file found in '{}'",
        dir.display()
    )))
}

/// Check whether a directory contains a tiktoken model file.
pub fn has_tiktoken_file(dir: &Path) -> bool {
    if dir.join("tiktoken.model").exists() {
        return true;
    }
    std::fs::read_dir(dir)
        .ok()
        .map(|entries| {
            entries.flatten().any(|e| {
                e.file_name()
                    .to_str()
                    .is_some_and(|n| n.ends_with(".tiktoken"))
            })
        })
        .unwrap_or(false)
}

/// Check whether a single file is a tiktoken model file (by name).
pub fn is_tiktoken_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|name| name == "tiktoken.model" || name.ends_with(".tiktoken"))
}

impl Encoder for TiktokenTokenizer {
    fn encode(&self, input: &str, _add_special_tokens: bool) -> Result<Encoding> {
        let tokens = self.tokenizer.encode_ordinary(input);
        Ok(Encoding::Tiktoken(tokens))
    }

    fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
        inputs
            .iter()
            .map(|input| self.encode(input, add_special_tokens))
            .collect()
    }
}

impl Decoder for TiktokenTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], _skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(token_ids.to_vec())
            .map_err(|e| Error::msg(format!("Decoding failed: {}", e)))
    }
}

impl TokenizerTrait for TiktokenTokenizer {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn token_to_id(&self, token: &str) -> Option<TokenIdType> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: TokenIdType) -> Option<String> {
        self.reverse_vocab.get(&id).cloned()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn apply_chat_template(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> Result<String> {
        self.chat_template.apply(messages, params)
    }

    fn chat_template_content_format(&self) -> ChatTemplateContentFormat {
        self.chat_template.content_format()
    }

    fn set_chat_template(&mut self, template: String) {
        self.chat_template.set(template);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{Decoder, Encoder, Tokenizer};

    #[test]
    fn test_tiktoken_creation() {
        let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap();
        assert_eq!(tokenizer.vocab_size(), 100256);
    }

    #[test]
    fn test_model_from_name() {
        assert!(matches!(
            TiktokenTokenizer::model_from_name("gpt-4").unwrap(),
            TiktokenModel::Cl100kBase
        ));
        assert!(matches!(
            TiktokenTokenizer::model_from_name("gpt-3.5-turbo").unwrap(),
            TiktokenModel::Cl100kBase
        ));
        assert!(matches!(
            TiktokenTokenizer::model_from_name("text-davinci-003").unwrap(),
            TiktokenModel::P50kBase
        ));
        assert!(matches!(
            TiktokenTokenizer::model_from_name("text-davinci-edit-001").unwrap(),
            TiktokenModel::P50kEdit
        ));
        assert!(matches!(
            TiktokenTokenizer::model_from_name("davinci").unwrap(),
            TiktokenModel::R50kBase
        ));
    }

    #[test]
    fn test_encode_decode() {
        let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap();

        let text = "Hello, world!";
        let encoding = tokenizer.encode(text, false).unwrap();

        let decoded = tokenizer.decode(encoding.token_ids(), false).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_batch_encode() {
        let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap();

        let texts = vec!["Hello", "World", "Test"];
        let encodings = tokenizer.encode_batch(&texts, false).unwrap();

        assert_eq!(encodings.len(), 3);
        for (i, encoding) in encodings.iter().enumerate() {
            let decoded = tokenizer.decode(encoding.token_ids(), false).unwrap();
            assert_eq!(decoded, texts[i]);
        }
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap();
        let special_tokens = tokenizer.get_special_tokens();

        assert!(special_tokens.eos_token.is_some());
        assert_eq!(special_tokens.eos_token.as_ref().unwrap(), "<|endoftext|>");
    }

    #[test]
    fn test_unrecognized_model_name_returns_error() {
        let result = TiktokenTokenizer::from_model_name("distilgpt-2");
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Unrecognized OpenAI model name"));
        }

        let result = TiktokenTokenizer::from_model_name("bert-base-uncased");
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Unrecognized OpenAI model name"));
        }

        let result = TiktokenTokenizer::from_model_name("llama-7b");
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Unrecognized OpenAI model name"));
        }
    }

    #[test]
    fn test_recognized_model_names() {
        assert!(TiktokenTokenizer::from_model_name("gpt-4").is_ok());
        assert!(TiktokenTokenizer::from_model_name("gpt-3.5-turbo").is_ok());
        assert!(TiktokenTokenizer::from_model_name("text-davinci-003").is_ok());
        assert!(TiktokenTokenizer::from_model_name("code-davinci-002").is_ok());
        assert!(TiktokenTokenizer::from_model_name("text-curie-001").is_ok());
        assert!(TiktokenTokenizer::from_model_name("text-babbage-001").is_ok());
        assert!(TiktokenTokenizer::from_model_name("text-ada-001").is_ok());
    }

    #[test]
    fn test_builtin_tokenizer_has_empty_vocab_maps() {
        let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap();
        // Built-in path: vocab maps are empty, token_to_id returns None
        assert_eq!(tokenizer.token_to_id("hello"), None);
        assert_eq!(tokenizer.id_to_token(0), None);
    }

    #[test]
    fn test_load_tiktoken_bpe() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.tiktoken");
        let mut f = std::fs::File::create(&file_path).unwrap();
        // "IQ==" is base64 for byte 0x21 ('!'), rank 0
        // "Ig==" is base64 for byte 0x22 ('"'), rank 1
        writeln!(f, "IQ== 0").unwrap();
        writeln!(f, "Ig== 1").unwrap();

        let encoder = load_tiktoken_bpe(file_path.to_str().unwrap()).unwrap();
        assert_eq!(encoder.len(), 2);
        assert_eq!(encoder.get(&vec![0x21u8]), Some(&0));
        assert_eq!(encoder.get(&vec![0x22u8]), Some(&1));
    }

    #[test]
    fn test_build_vocab_maps() {
        let mut encoder = FxHashMap::default();
        encoder.insert(b"hello".to_vec(), 42u32);
        encoder.insert(vec![0xFF, 0xFE], 99u32); // invalid UTF-8

        let mut added = HashMap::new();
        added.insert("<|special|>".to_string(), 1000u32);

        let (vocab, reverse_vocab) = build_vocab_maps(&encoder, &added);

        // Valid UTF-8 token present
        assert_eq!(vocab.get("hello"), Some(&42));
        assert_eq!(reverse_vocab.get(&42), Some(&"hello".to_string()));

        // Invalid UTF-8 token excluded from vocab
        assert!(!vocab.contains_key("\u{FFFD}")); // not lossy-inserted

        // Added token present
        assert_eq!(vocab.get("<|special|>"), Some(&1000));
        assert_eq!(reverse_vocab.get(&1000), Some(&"<|special|>".to_string()));
    }

    #[test]
    fn test_has_tiktoken_file() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!has_tiktoken_file(dir.path()));

        std::fs::write(dir.path().join("tiktoken.model"), "test").unwrap();
        assert!(has_tiktoken_file(dir.path()));
    }

    #[test]
    fn test_find_tiktoken_file_model() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tiktoken.model"), "test").unwrap();
        let found = find_tiktoken_file(dir.path()).unwrap();
        assert_eq!(found.file_name().unwrap(), "tiktoken.model");
    }

    #[test]
    fn test_find_tiktoken_file_extension() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("vocab.tiktoken"), "test").unwrap();
        let found = find_tiktoken_file(dir.path()).unwrap();
        assert!(found
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .ends_with(".tiktoken"));
    }

    #[test]
    fn test_is_tiktoken_file() {
        assert!(is_tiktoken_file(Path::new("tiktoken.model")));
        assert!(is_tiktoken_file(Path::new("vocab.tiktoken")));
        assert!(!is_tiktoken_file(Path::new("tokenizer.json")));
        assert!(!is_tiktoken_file(Path::new("model.bin")));
    }

    #[test]
    fn test_parse_added_tokens_decoder() {
        let config: serde_json::Value = serde_json::json!({
            "added_tokens_decoder": {
                "163584": { "content": "[BOS]", "special": true },
                "163585": { "content": "[EOS]", "special": true },
                "163586": { "content": "<|im_end|>", "special": true }
            }
        });
        let tokens = parse_added_tokens_decoder(&config);
        assert_eq!(tokens.get("[BOS]"), Some(&163584));
        assert_eq!(tokens.get("[EOS]"), Some(&163585));
        assert_eq!(tokens.get("<|im_end|>"), Some(&163586));
    }

    #[test]
    fn test_parse_special_tokens() {
        let config: serde_json::Value = serde_json::json!({
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "additional_special_tokens": ["<|im_end|>", "<|im_user|>"]
        });
        let special = parse_special_tokens(&config);
        assert_eq!(special.bos_token.as_deref(), Some("[BOS]"));
        assert_eq!(special.eos_token.as_deref(), Some("[EOS]"));
        assert_eq!(special.unk_token.as_deref(), Some("[UNK]"));
        assert_eq!(special.pad_token.as_deref(), Some("[PAD]"));
        assert_eq!(special.additional_special_tokens.len(), 2);
    }

    #[test]
    fn test_parse_special_tokens_object_valued() {
        let config: serde_json::Value = serde_json::json!({
            "bos_token": {"content": "<s>", "lstrip": false, "rstrip": false, "single_word": false, "special": true},
            "eos_token": "</s>",
            "unk_token": {"content": "<unk>", "special": true}
        });
        let special = parse_special_tokens(&config);
        assert_eq!(special.bos_token.as_deref(), Some("<s>"));
        assert_eq!(special.eos_token.as_deref(), Some("</s>"));
        assert_eq!(special.unk_token.as_deref(), Some("<unk>"));
    }

    #[test]
    fn test_tiktoken_config_default() {
        let config = TiktokenConfig::default();
        assert!(config.special_tokens.bos_token.is_none());
        assert!(config.added_tokens.is_empty());
        assert!(config.chat_template.is_none());
    }
}
