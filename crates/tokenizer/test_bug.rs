use std::sync::Arc;
use llm_tokenizer::traits::{Tokenizer, Decoder, Encoder, Encoding, SpecialTokens};
use llm_tokenizer::sequence::Sequence;
use anyhow::Result;

struct BadTokenizer;
impl Encoder for BadTokenizer {
    fn encode(&self, _: &str, _: bool) -> Result<Encoding> { Ok(Encoding::Plain(vec![])) }
    fn encode_batch(&self, _: &[&str], _: bool) -> Result<Vec<Encoding>> { Ok(vec![]) }
}
impl Decoder for BadTokenizer {
    fn decode(&self, ids: &[u32], _: bool) -> Result<String> {
        // Always return replacement character
        let mut s = String::new();
        for _ in ids {
            s.push('\u{FFFD}');
        }
        Ok(s)
    }
}
impl Tokenizer for BadTokenizer {
    fn vocab_size(&self) -> usize { 10 }
    fn get_special_tokens(&self) -> &SpecialTokens { &SpecialTokens::default() }
    fn token_to_id(&self, _: &str) -> Option<u32> { None }
    fn id_to_token(&self, _: u32) -> Option<String> { None }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

fn main() {
    let tokenizer = Arc::new(BadTokenizer);
    let mut seq = Sequence::new(tokenizer);
    for i in 0..100 {
        seq.append_token(1).unwrap();
    }
    println!("prefix_offset: {}, read_offset: {}, len: {}", seq.prefix_offset(), seq.read_offset(), seq.len());
}
