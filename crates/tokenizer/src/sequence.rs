use std::sync::Arc;

use anyhow::Result;

use crate::traits::{TokenIdType, Tokenizer as TokenizerTrait};

const INCREMENTAL_DETOKENIZATION_OVERLAP: usize = 32;

#[inline]
fn incremental_text(
    prefix_text: &str,
    new_text: &str,
    allow_incomplete_suffix: bool,
) -> Option<String> {
    if !allow_incomplete_suffix && new_text.ends_with('\u{FFFD}') {
        return None;
    }

    let mut split_at = prefix_text.len();
    while split_at > 0
        && (split_at > new_text.len()
            || !new_text.is_char_boundary(split_at)
            || !prefix_text.is_char_boundary(split_at))
    {
        split_at -= 1;
    }

    if new_text.len() > split_at && new_text.starts_with(&prefix_text[..split_at]) {
        let incremental = new_text[split_at..].to_string();
        if !incremental.is_empty() {
            return Some(incremental);
        }
    }

    let stable_prefix = prefix_text.trim_end_matches('\u{FFFD}');
    let mut matched_len = 0;
    let mut right_chars = new_text.char_indices();

    for left_ch in stable_prefix.chars() {
        match right_chars.next() {
            Some((idx, right_ch)) if left_ch == right_ch => {
                matched_len = idx + right_ch.len_utf8();
            }
            _ => break,
        }
    }

    let incremental = new_text[matched_len..].to_string();
    if incremental.is_empty() {
        None
    } else {
        Some(incremental)
    }
}

/// Maintains state for an ongoing sequence of tokens and their decoded text
/// This provides a cleaner abstraction for managing token sequences
pub struct Sequence {
    /// The tokenizer used for encoding/decoding
    tokenizer: Arc<dyn TokenizerTrait>,

    /// The current sequence of token ids
    token_ids: Vec<TokenIdType>,

    /// The position in the current sequence the last decoded token completed
    prefix_offset: usize,

    /// Current position in the sequence
    read_offset: usize,

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
                    let token_ids = self.token_ids();
                    if token_ids.len() <= 20 {
                        format!("{token_ids:?}")
                    } else {
                        let first_ten = &token_ids[..10];
                        let last_ten = &token_ids[token_ids.len() - 10..];
                        format!("{first_ten:?} ... {last_ten:?}")
                    }
                }),
            )
            .field("prefix_offset", &self.prefix_offset)
            .field("read_offset", &self.read_offset)
            .field("token count", &self.token_ids.len())
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
            prefix_offset: 0,
            read_offset: 0,
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
            prefix_offset: len.saturating_sub(INCREMENTAL_DETOKENIZATION_OVERLAP),
            read_offset: len,
            skip_special_tokens,
        }
    }

    /// Check if the sequence is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }

    /// Get the length of the sequence
    #[inline]
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    /// Clear the sequence
    pub fn clear(&mut self) {
        self.token_ids.clear();
        self.prefix_offset = 0;
        self.read_offset = 0;
    }

    /// Append text to the sequence by encoding it
    ///
    /// Set `add_special_tokens` to `true` for embeddings, or `false` for chat completion
    /// where the chat template already handles special tokens.
    pub fn append_text(&mut self, input: &str, add_special_tokens: bool) -> Result<()> {
        let encoding = self.tokenizer.encode(input, add_special_tokens)?;
        self.token_ids.extend(encoding.token_ids());
        Ok(())
    }

    /// Append a single token to the sequence and return newly decoded text.
    /// Based on HuggingFace TGI incremental decoding, but with a bounded
    /// sliding window so each call is O(1) in tokenizer work regardless of
    /// total sequence length.
    #[inline]
    pub fn append_token(&mut self, token_id: TokenIdType) -> Result<String> {
        self.token_ids.push(token_id);

        let len = self.token_ids.len();
        let prefix_text = self.tokenizer.decode(
            &self.token_ids[self.prefix_offset..self.read_offset],
            self.skip_special_tokens,
        )?;
        let new_text = self.tokenizer.decode(
            &self.token_ids[self.prefix_offset..],
            self.skip_special_tokens,
        )?;

        if let Some(text) = incremental_text(&prefix_text, &new_text, false) {
            self.prefix_offset = self.read_offset;
            self.read_offset = len;
            return Ok(text);
        }

        let decode_window_len = len.saturating_sub(self.prefix_offset);
        if !new_text.ends_with('\u{FFFD}') {
            self.prefix_offset = len.saturating_sub(INCREMENTAL_DETOKENIZATION_OVERLAP);
            self.read_offset = len;
        } else if decode_window_len > INCREMENTAL_DETOKENIZATION_OVERLAP + 2 {
            self.prefix_offset = len.saturating_sub(INCREMENTAL_DETOKENIZATION_OVERLAP);
            self.read_offset = self.prefix_offset;
        }

        Ok(String::new())
    }

    pub(crate) fn flush_pending(&mut self) -> Result<Option<String>> {
        if self.read_offset < self.token_ids.len() {
            let prefix_start = self
                .read_offset
                .saturating_sub(INCREMENTAL_DETOKENIZATION_OVERLAP);
            let prefix_text = self.tokenizer.decode(
                &self.token_ids[prefix_start..self.read_offset],
                self.skip_special_tokens,
            )?;
            let remaining = self
                .tokenizer
                .decode(&self.token_ids[prefix_start..], self.skip_special_tokens)?;

            self.prefix_offset = self.read_offset;
            self.read_offset = self.token_ids.len();

            if let Some(remaining) = incremental_text(&prefix_text, &remaining, true) {
                return Ok(Some(remaining));
            }
        }

        Ok(None)
    }

    /// Get a reference to the tokenizer
    #[inline]
    pub fn tokenizer(&self) -> &Arc<dyn TokenizerTrait> {
        &self.tokenizer
    }

    /// Get the current token ids
    #[inline]
    pub fn token_ids(&self) -> &[TokenIdType] {
        &self.token_ids
    }

    /// Decode the entire sequence to text
    pub fn text(&self) -> Result<String> {
        self.tokenizer
            .decode(&self.token_ids, self.skip_special_tokens)
    }

    /// Get the prefix offset
    #[inline]
    pub fn prefix_offset(&self) -> usize {
        self.prefix_offset
    }

    /// Get the read offset
    #[inline]
    pub fn read_offset(&self) -> usize {
        self.read_offset
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

        assert_eq!(seq.text().unwrap(), "Hello world");
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
        assert_eq!(seq.prefix_offset(), 0);
        assert_eq!(seq.read_offset(), 0);
    }

    #[test]
    fn test_sequence_debug() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let mut seq = Sequence::new(tokenizer);

        seq.append_text("Test", false).unwrap();
        let debug_str = format!("{seq:?}");
        assert!(debug_str.contains("Sequence"));
        assert!(debug_str.contains("token count"));
    }
}
