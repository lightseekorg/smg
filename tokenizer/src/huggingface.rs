use std::collections::HashMap;

use anyhow::{Error, Result};
use tokenizers::{processors::template::TemplateProcessing, tokenizer::Tokenizer as HfTokenizer};
use tracing::debug;

use crate::{
    chat_template::{
        load_chat_template_from_file, ChatTemplateContentFormat, ChatTemplateParams,
        ChatTemplateState,
    },
    traits::{Decoder, Encoder, Encoding, SpecialTokens, TokenIdType, Tokenizer as TokenizerTrait},
};

/// HuggingFace tokenizer wrapper
pub struct HuggingFaceTokenizer {
    tokenizer: HfTokenizer,
    special_tokens: SpecialTokens,
    vocab: HashMap<String, TokenIdType>,
    reverse_vocab: HashMap<TokenIdType, String>,
    chat_template: ChatTemplateState,
}

impl HuggingFaceTokenizer {
    /// Create a tokenizer from a HuggingFace tokenizer JSON file
    pub fn from_file(file_path: &str) -> Result<Self> {
        // Try to auto-discover chat template if not explicitly provided
        let path = std::path::Path::new(file_path);
        let chat_template_path = path
            .parent()
            .and_then(crate::factory::discover_chat_template_in_dir);
        Self::from_file_with_chat_template(file_path, chat_template_path.as_deref())
    }

    /// Create a tokenizer from a HuggingFace tokenizer JSON file with an optional chat template
    pub fn from_file_with_chat_template(
        file_path: &str,
        chat_template_path: Option<&str>,
    ) -> Result<Self> {
        let mut tokenizer = HfTokenizer::from_file(file_path)
            .map_err(|e| Error::msg(format!("Failed to load tokenizer: {}", e)))?;

        // Extract special tokens
        let special_tokens = Self::extract_special_tokens(&tokenizer);

        // Build vocab mappings (include special tokens to get added_tokens like <|im_start|>)
        let vocab = tokenizer.get_vocab(true); // true = include special tokens and added_tokens
        let reverse_vocab: HashMap<TokenIdType, String> = vocab
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();

        // Always load tokenizer_config.json for add_bos_token/add_eos_token,
        // then override only the chat template string when an explicit path is provided.
        let (mut chat_template_str, add_bos_token, add_eos_token) =
            Self::load_chat_template_and_config(file_path);

        if let Some(template_path) = chat_template_path {
            chat_template_str = load_chat_template_from_file(template_path)?;
        }

        // Configure post_processor based on tokenizer_config.json (matches Python transformers)
        // Only modify when at least one setting is explicitly true
        let needs_eos = add_eos_token == Some(true);
        let needs_bos = match add_bos_token {
            Some(true) => true,
            Some(false) => false,
            // Not set: preserve existing behavior from tokenizer.json
            None => needs_eos && Self::tokenizer_adds_special_tokens(&tokenizer),
        };

        if needs_bos || needs_eos {
            if let Some(post_processor) =
                Self::build_post_processor(needs_bos, needs_eos, &special_tokens, &vocab)
            {
                debug!(needs_bos, needs_eos, "Configured post_processor");
                tokenizer.with_post_processor(Some(post_processor));
            }
        }

        Ok(HuggingFaceTokenizer {
            tokenizer,
            special_tokens,
            vocab,
            reverse_vocab,
            chat_template: ChatTemplateState::new(chat_template_str),
        })
    }

    /// Check if the tokenizer's post_processor adds special tokens (e.g., BOS)
    fn tokenizer_adds_special_tokens(tokenizer: &HfTokenizer) -> bool {
        tokenizer
            .encode("", true)
            .map(|enc| !enc.get_ids().is_empty())
            .unwrap_or(false)
    }

    /// Build a TemplateProcessing post_processor (matches Python transformers' update_post_processor)
    /// Template format: "{bos}:0 $A:0 {eos}:0" with optional BOS/EOS based on config
    fn build_post_processor(
        add_bos_token: bool,
        add_eos_token: bool,
        special_tokens: &SpecialTokens,
        vocab: &HashMap<String, TokenIdType>,
    ) -> Option<TemplateProcessing> {
        // Build template string exactly like Python:
        // single = f"{(bos + ':0 ') if add_bos_token else ''}$A:0{(' ' + eos + ':0') if add_eos_token else ''}"
        let mut template = String::with_capacity(32);
        let mut tokens = Vec::with_capacity(2);

        if add_bos_token {
            let bos = special_tokens.bos_token.as_ref()?;
            let bos_id = vocab.get(bos).copied()?;
            template.push_str(bos);
            template.push_str(":0 ");
            tokens.push((bos.clone(), bos_id));
        }

        template.push_str("$A:0");

        if add_eos_token {
            let eos = special_tokens.eos_token.as_ref()?;
            let eos_id = vocab.get(eos).copied()?;
            template.push(' ');
            template.push_str(eos);
            template.push_str(":0");
            tokens.push((eos.clone(), eos_id));
        }

        TemplateProcessing::builder()
            .try_single(template.as_str())
            .ok()?
            .special_tokens(tokens)
            .build()
            .ok()
    }

    /// Create from an existing HuggingFace tokenizer
    pub fn from_tokenizer(tokenizer: HfTokenizer) -> Self {
        let special_tokens = Self::extract_special_tokens(&tokenizer);
        let vocab = tokenizer.get_vocab(true); // true = include special tokens and added_tokens
        let reverse_vocab: HashMap<TokenIdType, String> = vocab
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();

        HuggingFaceTokenizer {
            tokenizer,
            special_tokens,
            vocab,
            reverse_vocab,
            chat_template: ChatTemplateState::new(None),
        }
    }

    /// Extract special tokens from the tokenizer
    fn extract_special_tokens(tokenizer: &HfTokenizer) -> SpecialTokens {
        // Get vocab with special tokens included (added_tokens like <|im_start|>)
        let vocab = tokenizer.get_vocab(true);

        let find_token = |patterns: &[&str]| -> Option<String> {
            for pattern in patterns {
                if vocab.contains_key(*pattern) {
                    return Some(pattern.to_string());
                }
            }
            None
        };

        // Extract additional special tokens using the tokenizers library API
        let additional_special_tokens: Vec<String> = tokenizer
            .get_added_tokens_decoder()
            .iter()
            .filter(|(_id, token)| token.special) // Only tokens marked as special: true
            .map(|(_id, token)| token.content.clone())
            .collect();

        SpecialTokens {
            bos_token: find_token(&["<s>", "<|startoftext|>", "<BOS>", "[CLS]"]),
            eos_token: find_token(&["</s>", "<|endoftext|>", "<EOS>", "[SEP]"]),
            unk_token: find_token(&["<unk>", "<UNK>", "[UNK]"]),
            sep_token: find_token(&["[SEP]", "<sep>", "<SEP>"]),
            pad_token: find_token(&["<pad>", "<PAD>", "[PAD]"]),
            cls_token: find_token(&["[CLS]", "<cls>", "<CLS>"]),
            mask_token: find_token(&["[MASK]", "<mask>", "<MASK>"]),
            additional_special_tokens,
        }
    }

    /// Load chat template and special token settings from tokenizer_config.json
    /// Returns Option<bool> to distinguish between explicit false vs not set
    fn load_chat_template_and_config(
        tokenizer_path: &str,
    ) -> (Option<String>, Option<bool>, Option<bool>) {
        (|| {
            let path = std::path::Path::new(tokenizer_path);
            let config_path = path.parent()?.join("tokenizer_config.json");

            if !config_path.exists() {
                return None;
            }

            let config_str = config_path.to_str()?;
            let content = std::fs::read_to_string(&config_path).ok()?;
            let config: serde_json::Value = serde_json::from_str(&content).ok()?;

            let chat_template = super::chat_template::load_chat_template_from_config(config_str)
                .ok()
                .flatten();

            let add_bos_token = config.get("add_bos_token").and_then(|v| v.as_bool());
            let add_eos_token = config.get("add_eos_token").and_then(|v| v.as_bool());

            Some((chat_template, add_bos_token, add_eos_token))
        })()
        .unwrap_or((None, None, None))
    }
}

impl Encoder for HuggingFaceTokenizer {
    fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
        self.tokenizer
            .encode(input, add_special_tokens)
            .map_err(|e| Error::msg(format!("Encoding failed: {}", e)))
            .map(|encoding| Encoding::Hf(Box::new(encoding)))
    }

    fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
        self.tokenizer
            .encode_batch(inputs.to_vec(), add_special_tokens)
            .map_err(|e| Error::msg(format!("Batch encoding failed: {}", e)))
            .map(|encodings| {
                encodings
                    .into_iter()
                    .map(|e| Encoding::Hf(Box::new(e)))
                    .collect()
            })
    }
}

impl Decoder for HuggingFaceTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| Error::msg(format!("Decoding failed: {}", e)))
    }
}

impl TokenizerTrait for HuggingFaceTokenizer {
    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
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
