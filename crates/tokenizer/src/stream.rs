// src/tokenizer/stream.rs

use std::sync::Arc;

use anyhow::Result;

use crate::traits::{self, TokenIdType};

const INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET: usize = 32;

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
    #[inline]
    pub fn step(&mut self, id: TokenIdType) -> Result<Option<String>> {
        self.all_token_ids.push(id);

        let len = self.all_token_ids.len();
        let prefix_text = self.tokenizer.decode(
            &self.all_token_ids[self.prefix_offset..self.read_offset],
            self.skip_special_tokens,
        )?;

        let new_text = self.tokenizer.decode(
            &self.all_token_ids[self.prefix_offset..],
            self.skip_special_tokens,
        )?;

        if let Some(new_text) = incremental_text(&prefix_text, &new_text, false) {
            self.prefix_offset = self.read_offset;
            self.read_offset = len;
            return Ok(Some(new_text));
        }

        let decode_window_len = len.saturating_sub(self.prefix_offset);
        if !new_text.ends_with('\u{FFFD}') {
            self.prefix_offset = len.saturating_sub(INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET);
            self.read_offset = len;
        } else if decode_window_len > INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET + 2 {
            self.prefix_offset = len.saturating_sub(INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET);
            self.read_offset = self.prefix_offset;
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
            let prefix_start = self
                .read_offset
                .saturating_sub(INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET);
            let prefix_text = self.tokenizer.decode(
                &self.all_token_ids[prefix_start..self.read_offset],
                self.skip_special_tokens,
            )?;
            let remaining = self.tokenizer.decode(
                &self.all_token_ids[prefix_start..],
                self.skip_special_tokens,
            )?;

            self.prefix_offset = self.read_offset;
            self.read_offset = self.all_token_ids.len();

            if let Some(remaining) = incremental_text(&prefix_text, &remaining, true) {
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
