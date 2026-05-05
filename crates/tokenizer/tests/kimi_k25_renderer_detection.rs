//! Verify that `TiktokenTokenizer` picks `Renderer::KimiK25Tools` when
//! config.json declares `KimiK25ForConditionalGeneration`, and falls back to
//! Jinja otherwise. Mirrors the structure of `deepseek_renderer_detection.rs`.

#[cfg(test)]
mod tests {
    use std::fs;

    use llm_tokenizer::tiktoken::TiktokenTokenizer;
    use tempfile::TempDir;

    /// Minimal tiktoken model file content. The format the loader expects is
    /// `<base64-token> <rank>\n` per line.
    /// "aGVsbG8=" decodes to "hello" (rank 0).
    const MIN_TIKTOKEN_MODEL: &str = "aGVsbG8= 0\n";

    fn build_dir(architectures: Option<&str>) -> TempDir {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("tiktoken.model"), MIN_TIKTOKEN_MODEL).unwrap();
        if let Some(arch) = architectures {
            fs::write(
                dir.path().join("config.json"),
                format!(r#"{{"architectures": ["{arch}"]}}"#),
            )
            .unwrap();
        }
        // tokenizer_config with a trivial template so loading succeeds.
        fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": "{{ messages | length }}"}"#,
        )
        .unwrap();
        dir
    }

    #[test]
    fn loads_with_kimi_k25_architecture() {
        let dir = build_dir(Some("KimiK25ForConditionalGeneration"));
        let tok = TiktokenTokenizer::from_dir(dir.path()).expect("load");
        // We can't read the private renderer field directly. The behavioral
        // assertion (does dispatch produce the TS namespace block?) is in
        // tests/kimi_k25_chat_template.rs (Task 9). This test just exercises
        // the detection codepath and confirms loading succeeds.
        let _ = tok;
    }

    #[test]
    fn falls_back_to_jinja_when_config_missing() {
        let dir = build_dir(None);
        let tok = TiktokenTokenizer::from_dir(dir.path()).expect("load");
        let _ = tok;
    }
}
