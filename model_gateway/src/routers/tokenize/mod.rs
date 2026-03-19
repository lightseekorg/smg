//! Tokenize module for tokenization and detokenization operations
//!
//! This module provides HTTP handlers for:
//! - Tokenizing text into token IDs
//! - Detokenizing token IDs back to text
//! - Managing tokenizers (add, list, get, remove)

pub mod grpc_service;
mod handlers;

pub use handlers::{
    add_tokenizer, detokenize, get_tokenizer_info, get_tokenizer_status, list_tokenizers,
    remove_tokenizer, render_chat, render_chat_core, render_completion, render_completion_core,
    tokenize,
};
