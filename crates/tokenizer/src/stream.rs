// src/tokenizer/stream.rs

use std::sync::Arc;

use anyhow::Result;

use crate::traits::{self, TokenIdType};

const INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET: usize = 5;

/// DecodeStream will keep the state necessary to produce individual chunks of
/// strings given an input stream of token_ids
pub struct DecodeStream {
    /// The tokenizer used to decode token_ids
    tokenizer: Arc<dyn traits::Tokenizer>,

    skip_special_tokens: bool,

    /// A temporary buffer of the necessary token_ids needed
    /// to produce valid string chunks
    all_token_ids: Vec<TokenIdType>,

    prefix_offset: usize,
    read_offset: usize,
}

impl DecodeStream {
    pub fn new(
        tokenizer: Arc<dyn traits::Tokenizer>,
        prompt_token_ids: &[TokenIdType],
        skip_special_tokens: bool,
    ) -> Self {
        let num_input_tokens = prompt_token_ids.len();
        let prompt_token_ids = prompt_token_ids.to_vec();
        Self {
            tokenizer,
            skip_special_tokens,
            all_token_ids: prompt_token_ids,
            prefix_offset: num_input_tokens
                .saturating_sub(INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET),
            read_offset: num_input_tokens,
        }
    }

    /// Step appends a token_id to the internal state and tries to produce a text chunk.
    /// Returning `None` means the given id is not enough to produce a chunk.
    ///
    /// Uses content-based comparison (not byte-length) to correctly handle
    /// byte-fallback tokenizers where `\u{FFFD}` resolves to a character of
    /// the same byte length.
    #[inline]
    pub fn step(&mut self, id: TokenIdType) -> Result<Option<String>> {
        self.all_token_ids.push(id);

        let prefix_text = self.tokenizer.decode(
            &self.all_token_ids[self.prefix_offset..self.read_offset],
            self.skip_special_tokens,
        )?;

        let new_text = self.tokenizer.decode(
            &self.all_token_ids[self.prefix_offset..],
            self.skip_special_tokens,
        )?;

        // Trailing \u{FFFD} means an incomplete UTF-8 byte sequence — wait
        if new_text.ends_with('\u{FFFD}') {
            return Ok(None);
        }

        // Find where new_text diverges from prefix_text by comparing bytes.
        // This correctly handles the case where \u{FFFD} (3 bytes) resolves
        // to a real character of the same byte length (e.g., CJK), which the
        // previous byte-length comparison missed entirely.
        let common_len = prefix_text
            .as_bytes()
            .iter()
            .zip(new_text.as_bytes())
            .take_while(|(a, b)| a == b)
            .count();

        // Back up to the nearest UTF-8 char boundary
        let mut split_at = common_len;
        while split_at > 0 && !new_text.is_char_boundary(split_at) {
            split_at -= 1;
        }

        let incremental = &new_text[split_at..];
        if !incremental.is_empty() {
            self.prefix_offset = self.read_offset;
            self.read_offset = self.all_token_ids.len();
            return Ok(Some(incremental.to_string()));
        }

        Ok(None)
    }

    /// Process multiple tokens at once
    pub fn step_batch(&mut self, token_ids: &[u32]) -> Result<Vec<String>> {
        // Pre-allocate with capacity - most tokens produce output
        let mut chunks = Vec::with_capacity(token_ids.len());

        for &token_id in token_ids {
            if let Some(text) = self.step(token_id)? {
                chunks.push(text);
            }
        }

        Ok(chunks)
    }

    /// Force flush any remaining text
    pub fn flush(&mut self) -> Result<Option<String>> {
        if self.read_offset < self.all_token_ids.len() {
            let remaining = self.tokenizer.decode(
                &self.all_token_ids[self.read_offset..],
                self.skip_special_tokens,
            )?;

            self.read_offset = self.all_token_ids.len();

            if !remaining.is_empty() {
                return Ok(Some(remaining));
            }
        }

        Ok(None)
    }

    /// Get all tokens processed so far
    pub fn tokens(&self) -> &[u32] {
        &self.all_token_ids
    }
}
