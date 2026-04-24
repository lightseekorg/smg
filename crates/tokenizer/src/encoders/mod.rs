//! Custom chat-template encoders for models that bypass Jinja templates.
//!
//! Some model families (notably DeepSeek V3.2 and V4) use a custom Python
//! encoding step that the bundled Jinja chat template in
//! `tokenizer_config.json` does not fully reproduce — DSML tool-call format,
//! conditional thinking-block drops, V4 quick-instruction tokens, and the
//! reasoning-effort prefix all live outside the template.
//!
//! Both vllm and sglang upstream bypass the Jinja template for these models
//! and call the Python encoder directly. The encoders in this module are
//! straight Rust ports of the canonical HuggingFace references; see each
//! file's header comment for the upstream source.

pub mod deepseek_v32;
pub mod deepseek_v4;
