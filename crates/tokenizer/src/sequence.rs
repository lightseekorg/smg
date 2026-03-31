use std::sync::Arc;

use anyhow::Result;

use crate::traits::{TokenIdType, Tokenizer as TokenizerTrait};

/// Maintains state for an ongoing sequence of tokens and their decoded text.
///
/// Mirrors the design of the native `DecodeStream` in the HuggingFace `tokenizers`
/// crate but works through the `dyn Tokenizer` trait so it supports all backends
/// (HuggingFace, Tiktoken, Mock).
///
/// Key design decisions (matching native `DecodeStream`):
/// - **Token draining**: consumed tokens are drained from the buffer after each
///   successful step, keeping memory bounded regardless of generation length.
/// - **Prefix caching**: the decoded prefix string is cached between calls,
///   avoiding a redundant `decode()` on the next step.
pub struct Sequence {
    /// The tokenizer used for encoding/decoding
    tokenizer: Arc<dyn TokenizerTrait>,

    /// Sliding window of token ids needed for correct incremental decoding.
    /// Consumed tokens are drained after each successful step.
    token_ids: Vec<TokenIdType>,

    /// Total number of tokens ever appended (does NOT reset on drain).
    /// Used by callers that need the logical sequence length.
    total_tokens: usize,

    /// Index within `token_ids` that marks the start of the "prefix" window.
    /// Everything before this has already been decoded and can be drained.
    prefix_index: usize,

    /// Cached decoded prefix string from the previous successful step.
    /// On the next call we skip one `decode()` by reusing this.
    cached_prefix: String,

    /// Whether to skip special tokens when decoding
    skip_special_tokens: bool,
}

impl std::fmt::Debug for Sequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequence")
            .field("tokenizer", &"Arc<dyn Tokenizer>")
            .field(
                "token_ids",
                &format_args!("{}", {
                    let token_ids = &self.token_ids;
                    if token_ids.len() <= 20 {
                        format!("{token_ids:?}")
                    } else {
                        let first_ten = &token_ids[..10];
                        let last_ten = &token_ids[token_ids.len() - 10..];
                        format!("{first_ten:?} ... {last_ten:?}")
                    }
                }),
            )
            .field("prefix_index", &self.prefix_index)
            .field("buffer_len", &self.token_ids.len())
            .field("total_tokens", &self.total_tokens)
            .finish()
    }
}

impl Sequence {
    /// Create a new empty sequence
    pub fn new(tokenizer: Arc<dyn TokenizerTrait>) -> Self {
        Self::new_with_options(tokenizer, false)
    }

    /// Create a new empty sequence with skip_special_tokens option
    pub fn new_with_options(tokenizer: Arc<dyn TokenizerTrait>, skip_special_tokens: bool) -> Self {
        Self {
            tokenizer,
            token_ids: Vec::new(),
            total_tokens: 0,
            prefix_index: 0,
            cached_prefix: String::new(),
            skip_special_tokens,
        }
    }

    /// Create a sequence with initial tokens
    pub fn with_tokens(tokenizer: Arc<dyn TokenizerTrait>, token_ids: Vec<TokenIdType>) -> Self {
        Self::with_tokens_and_options(tokenizer, token_ids, false)
    }

    /// Create a sequence with initial tokens and skip_special_tokens option
    pub fn with_tokens_and_options(
        tokenizer: Arc<dyn TokenizerTrait>,
        token_ids: Vec<TokenIdType>,
        skip_special_tokens: bool,
    ) -> Self {
        let len = token_ids.len();
        Self {
            tokenizer,
            token_ids,
            total_tokens: len,
            prefix_index: 0,
            cached_prefix: String::new(),
            skip_special_tokens,
        }
    }

    /// Check if the sequence is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_tokens == 0
    }

    /// Get the total number of tokens appended (logical length, not buffer size)
    #[inline]
    pub fn len(&self) -> usize {
        self.total_tokens
    }

    /// Clear the sequence
    pub fn clear(&mut self) {
        self.token_ids.clear();
        self.total_tokens = 0;
        self.prefix_index = 0;
        self.cached_prefix.clear();
    }

    /// Append text to the sequence by encoding it
    ///
    /// Set `add_special_tokens` to `true` for embeddings, or `false` for chat completion
    /// where the chat template already handles special tokens.
    pub fn append_text(&mut self, input: &str, add_special_tokens: bool) -> Result<()> {
        let encoding = self.tokenizer.encode(input, add_special_tokens)?;
        let ids = encoding.token_ids();
        self.token_ids.extend(ids);
        self.total_tokens += ids.len();
        Ok(())
    }

    /// Append a single token to the sequence and return newly decoded text.
    ///
    /// Uses the same algorithm as the native `DecodeStream` in the HuggingFace
    /// `tokenizers` crate: cache the decoded prefix string between calls and
    /// drain consumed tokens to keep memory bounded.
    #[inline]
    pub fn append_token(&mut self, token_id: TokenIdType) -> Result<String> {
        self.token_ids.push(token_id);
        self.total_tokens += 1;

        // If the cached prefix is empty but we have buffered tokens, compute it.
        // This handles the first call and recovery after incomplete UTF-8.
        if self.cached_prefix.is_empty() && self.token_ids.len() > 1 {
            let new_prefix = self.tokenizer.decode(
                &self.token_ids[..self.token_ids.len() - 1],
                self.skip_special_tokens,
            )?;
            if !new_prefix.ends_with("�") {
                self.prefix_index = self.token_ids.len() - 1;
                self.cached_prefix = new_prefix;
            }
        }

        // Decode the full buffer (prefix context + new token)
        let string = self
            .tokenizer
            .decode(&self.token_ids, self.skip_special_tokens)?;

        if string.len() > self.cached_prefix.len() && !string.ends_with("�") {
            // Find the char-safe split point
            let mut split_at = self.cached_prefix.len();
            while !string.is_char_boundary(split_at) && split_at > 0 {
                split_at -= 1;
            }

            let incremental_text = string[split_at..].to_string().replace("�", "");

            // Drain consumed tokens and recompute the cached prefix for next call.
            // This mirrors native DecodeStream: keeps buffer small, avoids unbounded growth.
            let new_prefix_len = self.token_ids.len() - self.prefix_index;
            self.token_ids.drain(..self.prefix_index);
            self.prefix_index = new_prefix_len;
            self.cached_prefix = self
                .tokenizer
                .decode(&self.token_ids, self.skip_special_tokens)?;

            Ok(incremental_text)
        } else {
            // Incomplete UTF-8 or no new text — wait for more tokens
            Ok(String::new())
        }
    }

    /// Get a reference to the tokenizer
    #[inline]
    pub fn tokenizer(&self) -> &Arc<dyn TokenizerTrait> {
        &self.tokenizer
    }

    /// Get the current token ids in the buffer (sliding window, not full history)
    #[inline]
    pub fn token_ids(&self) -> &[TokenIdType] {
        &self.token_ids
    }

    /// Decode the current buffer to text
    pub fn text(&self) -> Result<String> {
        self.tokenizer
            .decode(&self.token_ids, self.skip_special_tokens)
    }

    /// Get whether special tokens are skipped during decoding
    #[inline]
    pub fn skip_special_tokens(&self) -> bool {
        self.skip_special_tokens
    }
}

#[cfg(test)]
mod tests {
    use crate::{mock::MockTokenizer, *};

    #[test]
    fn test_sequence_new() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let seq = Sequence::new(tokenizer);
        assert!(seq.is_empty());
        assert_eq!(seq.len(), 0);
    }

    #[test]
    fn test_sequence_append_text() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let mut seq = Sequence::new(tokenizer);

        seq.append_text("Hello", false).unwrap();
        assert!(!seq.is_empty());

        let text = seq.text().unwrap();
        assert_eq!(text, "Hello");
    }

    #[test]
    fn test_sequence_append_token() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let mut seq = Sequence::new(tokenizer.clone());

        // Start with an empty sequence and append token 1 ("Hello")
        let text1 = seq.append_token(1).unwrap();
        assert_eq!(text1, "Hello");

        // Now append token 2 ("world")
        // The mock tokenizer will decode [1, 2] as "Hello world" (with a space)
        let text2 = seq.append_token(2).unwrap();
        // The incremental text should be " world" (with the space that the mock tokenizer adds)
        assert_eq!(text2, " world");
    }

    #[test]
    fn test_sequence_clear() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let mut seq = Sequence::new(tokenizer);

        seq.append_text("Hello world", false).unwrap();
        assert!(!seq.is_empty());

        seq.clear();
        assert!(seq.is_empty());
        assert_eq!(seq.len(), 0);
    }

    #[test]
    fn test_sequence_debug() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let mut seq = Sequence::new(tokenizer);

        seq.append_text("Test", false).unwrap();
        let debug_str = format!("{seq:?}");
        assert!(debug_str.contains("Sequence"));
        assert!(debug_str.contains("total_tokens"));
    }

    #[test]
    fn test_sequence_token_drain() {
        // Verify that the token buffer stays bounded after many appends
        let tokenizer = Arc::new(MockTokenizer::new());
        let mut seq = Sequence::new(tokenizer);

        // Append many tokens
        for i in 0..100 {
            let token_id = (i % 5) + 1; // cycle through mock tokens
            let _ = seq.append_token(token_id);
        }

        // Logical length should reflect all tokens
        assert_eq!(seq.len(), 100);

        // Buffer should be much smaller than 100 due to draining
        assert!(
            seq.token_ids().len() < 100,
            "Token buffer should be drained, but has {} entries",
            seq.token_ids().len()
        );
    }
}
