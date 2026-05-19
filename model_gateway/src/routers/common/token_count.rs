use openai_protocol::common::GenerationRequest;

/// Estimate request token count for routing and rate limiting.
///
/// Falls back to a conservative character-based estimate when no tokenizer is
/// registered for the model or tokenization fails.
pub fn count_tokens<T: GenerationRequest>(
    tokenizer_registry: &llm_tokenizer::registry::TokenizerRegistry,
    body: &T,
    model_id: &str,
) -> u32 {
    let text = body.extract_text_for_routing();
    if text.is_empty() {
        return 1;
    }

    let fallback_estimate = || ((text.chars().count() as u32) / 4).max(1);

    let Some(tokenizer) = tokenizer_registry.get(model_id) else {
        return fallback_estimate();
    };

    tokenizer
        .encode(&text, false)
        .map(|encoding| encoding.token_ids().len() as u32)
        .unwrap_or_else(|_| fallback_estimate())
        .max(1)
}
