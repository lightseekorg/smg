use std::sync::Arc;

use crate::{
    mock,
    traits::{Decoder, Encoder},
    Tokenizer,
};

#[test]
fn test_mock_tokenizer_encode() {
    let tokenizer = mock::MockTokenizer::new();
    let encoding = tokenizer.encode("Hello world", false).unwrap();
    let token_ids = encoding.token_ids();
    assert_eq!(token_ids, &[1, 2]); // "Hello" -> 1, "world" -> 2
}

#[test]
fn test_mock_tokenizer_decode() {
    let tokenizer = mock::MockTokenizer::new();
    let text = tokenizer.decode(&[1, 2], false).unwrap();
    assert_eq!(text, "Hello world");
}

#[test]
fn test_mock_tokenizer_decode_skip_special() {
    let tokenizer = mock::MockTokenizer::new();

    // With special tokens
    let text = tokenizer.decode(&[1000, 1, 2, 999], false).unwrap();
    assert_eq!(text, "<bos> Hello world <eos>");

    // Without special tokens
    let text = tokenizer.decode(&[1000, 1, 2, 999], true).unwrap();
    assert_eq!(text, "Hello world");
}

#[test]
fn test_tokenizer_wrapper() {
    let mock_tokenizer = Arc::new(mock::MockTokenizer::new());
    let tokenizer = Tokenizer::from_arc(mock_tokenizer);

    let encoding = tokenizer.encode("Hello world", false).unwrap();
    assert_eq!(encoding.token_ids(), &[1, 2]);

    let text = tokenizer.decode(&[1, 2], false).unwrap();
    assert_eq!(text, "Hello world");

    assert_eq!(tokenizer.vocab_size(), 14);

    assert_eq!(tokenizer.token_to_id("Hello"), Some(1));
    assert_eq!(tokenizer.token_to_id("unknown"), None);

    assert_eq!(tokenizer.id_to_token(1), Some("Hello".to_string()));
    assert_eq!(tokenizer.id_to_token(9999), None);
}

#[test]
fn test_decode_stream_basic() {
    let mock_tokenizer = Arc::new(mock::MockTokenizer::new());
    let tokenizer = Tokenizer::from_arc(mock_tokenizer);

    // Create a decode stream with initial tokens [1, 2] = "Hello world"
    let initial_tokens = vec![1, 2];
    let mut stream = tokenizer.decode_stream(&initial_tokens, false);

    // Step with token 3 ("test").
    // DecodeStream computes:
    //   prefix_text = decode([1, 2]) = "Hello world"
    //   new_text    = decode([1, 2, 3]) = "Hello world test"
    // The incremental output is the suffix beyond prefix_text: " test"
    let result = stream.step(3).unwrap();
    assert_eq!(result, Some(" test".to_string()));

    // The stream should now track all three tokens
    assert_eq!(stream.tokens(), &[1, 2, 3]);
}

#[test]
fn test_decode_stream_multiple_steps() {
    let mock_tokenizer = Arc::new(mock::MockTokenizer::new());
    let tokenizer = Tokenizer::from_arc(mock_tokenizer);

    // Start with a single token [1] = "Hello"
    let initial_tokens = vec![1];
    let mut stream = tokenizer.decode_stream(&initial_tokens, false);

    // Step with token 2 ("world"):
    //   prefix_text = decode([1]) = "Hello"
    //   new_text    = decode([1, 2]) = "Hello world"
    //   incremental = " world"
    let result = stream.step(2).unwrap();
    assert_eq!(result, Some(" world".to_string()));

    // Step with token 3 ("test"):
    //   prefix_text = decode([2]) = "world"
    //   new_text    = decode([2, 3]) = "world test"
    //   incremental = " test"
    let result = stream.step(3).unwrap();
    assert_eq!(result, Some(" test".to_string()));

    assert_eq!(stream.tokens(), &[1, 2, 3]);
}

#[test]
fn test_decode_stream_flush() {
    let mock_tokenizer = Arc::new(mock::MockTokenizer::new());
    let tokenizer = Tokenizer::from_arc(mock_tokenizer);

    let initial_tokens = vec![1]; // "Hello"
    let mut stream = tokenizer.decode_stream(&initial_tokens, false);

    // Both steps produce text, advancing read_offset each time
    let step1 = stream.step(2).unwrap();
    assert_eq!(step1, Some(" world".to_string()));

    let step2 = stream.step(3).unwrap();
    assert_eq!(step2, Some(" test".to_string()));

    // After successful steps that consumed all tokens, read_offset == all_token_ids.len(),
    // so flush has nothing remaining and returns None.
    let flushed = stream.flush().unwrap();
    assert_eq!(flushed, None);
}

#[test]
fn test_special_tokens() {
    let mock_tokenizer = Arc::new(mock::MockTokenizer::new());
    let tokenizer = Tokenizer::from_arc(mock_tokenizer);

    let special_tokens = tokenizer.get_special_tokens();
    assert_eq!(special_tokens.bos_token, Some("<bos>".to_string()));
    assert_eq!(special_tokens.eos_token, Some("<eos>".to_string()));
    assert_eq!(special_tokens.unk_token, Some("<unk>".to_string()));
    assert!(special_tokens.sep_token.is_none());
    assert!(special_tokens.pad_token.is_none());
}

#[test]
fn test_batch_encode() {
    let tokenizer = mock::MockTokenizer::new();
    let inputs = vec!["Hello", "world", "test"];
    let encodings = tokenizer.encode_batch(&inputs, false).unwrap();

    assert_eq!(encodings.len(), 3);
    assert_eq!(encodings[0].token_ids(), &[1]); // "Hello" -> 1
    assert_eq!(encodings[1].token_ids(), &[2]); // "world" -> 2
    assert_eq!(encodings[2].token_ids(), &[3]); // "test" -> 3
}

#[test]
fn test_thread_safety() {
    use std::thread;

    let mock_tokenizer = Arc::new(mock::MockTokenizer::new());
    let tokenizer = Tokenizer::from_arc(mock_tokenizer);

    // Spawn multiple threads that use the same tokenizer
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let tokenizer_clone = tokenizer.clone();
            thread::spawn(move || {
                let text = "Hello test".to_string();
                let encoding = tokenizer_clone.encode(&text, false).unwrap();
                let decoded = tokenizer_clone.decode(encoding.token_ids(), false).unwrap();
                assert!(decoded.contains("Hello") || decoded.contains("test"));
                i
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
}

/// Regression test: DecodeStream must not panic when prefix_text.len()
/// falls mid-codepoint in new_text.  This happens with real tokenizers
/// that use byte-fallback — partial byte tokens merge into multi-byte
/// characters when more context arrives, changing the byte length of
/// the prefix portion.
#[test]
fn test_decode_stream_multibyte_char_boundary() {
    use anyhow::Result;

    use crate::{
        stream::DecodeStream,
        traits::{Decoder, Encoder, Encoding, SpecialTokens, Tokenizer as TokenizerTrait},
    };

    /// Mock tokenizer simulating byte-fallback context sensitivity.
    ///
    /// decode([1, 2])   → "abc"  (3 bytes — incomplete byte rendered as ASCII)
    /// decode([1, 2, 3]) → "ab🎉" (6 bytes — merged into 4-byte emoji)
    ///
    /// prefix_text.len() = 3 lands inside the emoji (bytes 2..6 in new_text).
    struct MultiByteTokenizer {
        special_tokens: SpecialTokens,
    }

    impl Encoder for MultiByteTokenizer {
        fn encode(&self, _input: &str, _add_special_tokens: bool) -> Result<Encoding> {
            Ok(Encoding::Plain(vec![]))
        }

        fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
            inputs
                .iter()
                .map(|s| self.encode(s, add_special_tokens))
                .collect()
        }
    }

    impl Decoder for MultiByteTokenizer {
        fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
            Ok(match token_ids {
                [1, 2] => "abc".into(),
                [1, 2, 3] => "ab\u{1F389}".into(), // "ab🎉"
                _ => String::new(),
            })
        }
    }

    impl TokenizerTrait for MultiByteTokenizer {
        fn vocab_size(&self) -> usize {
            10
        }
        fn get_special_tokens(&self) -> &SpecialTokens {
            &self.special_tokens
        }
        fn token_to_id(&self, _token: &str) -> Option<u32> {
            None
        }
        fn id_to_token(&self, _id: u32) -> Option<String> {
            None
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    let tokenizer: Arc<dyn TokenizerTrait> = Arc::new(MultiByteTokenizer {
        special_tokens: SpecialTokens::default(),
    });
    let prompt_tokens = vec![1, 2];
    let mut stream = DecodeStream::new(tokenizer, &prompt_tokens, false);

    // Without the char-boundary fix this panics:
    //   "byte index 3 is not a char boundary; it is inside '🎉' (bytes 2..6)"
    let result = stream.step(3).unwrap();
    assert_eq!(result, Some("\u{1F389}".to_string()));
}
