mod llama4;
mod llava;
mod phi3_v;
mod qwen3_vl;
mod qwen_vl;
mod traits;

use llama4::Llama4Spec;
use llava::{LlavaNextSpec, LlavaSpec};
use once_cell::sync::Lazy;
use phi3_v::Phi3VisionSpec;
use qwen3_vl::Qwen3VLVisionSpec;
use qwen_vl::QwenVLVisionSpec;
// Re-export for use by spec modules within the crate.
pub(crate) use traits::image_sizes_hw;
// Re-export public API from traits.
pub use traits::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult};

pub struct ModelRegistry {
    specs: Vec<LazySpec>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            specs: vec![
                LazySpec::new("llama4", || Box::new(Llama4Spec)),
                // LlavaNext must be registered before Llava so "llava_next" model_type matches first.
                LazySpec::new("llava_next", || Box::new(LlavaNextSpec)),
                LazySpec::new("llava", || Box::new(LlavaSpec)),
                // Qwen3-VL must be registered before QwenVL so "qwen3" matches first.
                LazySpec::new("qwen3_vl", || Box::new(Qwen3VLVisionSpec)),
                LazySpec::new("qwen_vl", || Box::new(QwenVLVisionSpec)),
                LazySpec::new("phi3_v", || Box::new(Phi3VisionSpec)),
            ],
        }
    }

    pub fn lookup<'a>(&'a self, metadata: &ModelMetadata) -> Option<&'a dyn ModelProcessorSpec> {
        for spec in &self.specs {
            let spec_ref = spec.get();
            if spec_ref.matches(metadata) {
                return Some(spec_ref);
            }
        }
        None
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

struct LazySpec {
    inner: Lazy<Box<dyn ModelProcessorSpec>>,
}

impl LazySpec {
    fn new(_id: &'static str, factory: fn() -> Box<dyn ModelProcessorSpec>) -> Self {
        Self {
            inner: Lazy::new(factory),
        }
    }

    fn get(&self) -> &dyn ModelProcessorSpec {
        self.inner.as_ref()
    }
}

#[cfg(test)]
pub(super) mod test_helpers {
    use std::collections::HashMap;

    use llm_tokenizer::{Decoder, Encoder, Encoding, SpecialTokens, TokenizerTrait};
    use once_cell::sync::Lazy;

    use crate::{
        types::ImageSize,
        vision::image_processor::{ModelSpecificValue, PreprocessedImages},
    };

    pub struct TestTokenizer {
        vocab: HashMap<String, u32>,
    }

    impl TestTokenizer {
        pub fn new(pairs: &[(&str, u32)]) -> Self {
            let vocab = pairs
                .iter()
                .map(|(token, id)| ((*token).to_string(), *id))
                .collect();
            Self { vocab }
        }
    }

    impl Encoder for TestTokenizer {
        fn encode(&self, _input: &str, _add_special_tokens: bool) -> anyhow::Result<Encoding> {
            Ok(Encoding::Plain(Vec::new()))
        }

        fn encode_batch(
            &self,
            inputs: &[&str],
            add_special_tokens: bool,
        ) -> anyhow::Result<Vec<Encoding>> {
            inputs
                .iter()
                .map(|_| self.encode("", add_special_tokens))
                .collect()
        }
    }

    impl Decoder for TestTokenizer {
        fn decode(&self, _token_ids: &[u32], _skip_special_tokens: bool) -> anyhow::Result<String> {
            Ok(String::new())
        }
    }

    impl TokenizerTrait for TestTokenizer {
        fn vocab_size(&self) -> usize {
            self.vocab.len()
        }

        fn get_special_tokens(&self) -> &SpecialTokens {
            static TOKENS: Lazy<SpecialTokens> = Lazy::new(|| SpecialTokens {
                bos_token: None,
                eos_token: None,
                unk_token: None,
                sep_token: None,
                pad_token: None,
                cls_token: None,
                mask_token: None,
                additional_special_tokens: vec![],
            });
            &TOKENS
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.vocab.get(token).copied()
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            self.vocab
                .iter()
                .find(|(_, &v)| v == id)
                .map(|(k, _)| k.clone())
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// Build a minimal `PreprocessedImages` for testing prompt_replacements.
    pub fn test_preprocessed(image_sizes: &[ImageSize]) -> PreprocessedImages {
        test_preprocessed_with_tokens(image_sizes, &vec![0; image_sizes.len()])
    }

    pub fn test_preprocessed_with_tokens(
        image_sizes: &[ImageSize],
        num_img_tokens: &[usize],
    ) -> PreprocessedImages {
        let sizes: Vec<(u32, u32)> = image_sizes.iter().map(|s| (s.height, s.width)).collect();
        PreprocessedImages {
            pixel_values: ndarray::ArrayD::zeros(vec![1, 3, 336, 336]),
            num_img_tokens: num_img_tokens.to_vec(),
            image_sizes: sizes,
            model_specific: HashMap::new(),
        }
    }

    /// Build `PreprocessedImages` with explicit aspect_ratios (for Llama4 tests).
    pub fn test_preprocessed_with_aspects(
        image_sizes: &[ImageSize],
        aspect_ratios: &[(i64, i64)],
    ) -> PreprocessedImages {
        let sizes: Vec<(u32, u32)> = image_sizes.iter().map(|s| (s.height, s.width)).collect();
        let flat: Vec<i64> = aspect_ratios
            .iter()
            .flat_map(|&(h, w)| vec![h, w])
            .collect();
        let batch = aspect_ratios.len();
        let mut model_specific = HashMap::new();
        model_specific.insert(
            "aspect_ratios".to_string(),
            ModelSpecificValue::IntTensor {
                data: flat,
                shape: vec![batch, 2],
            },
        );
        PreprocessedImages {
            pixel_values: ndarray::ArrayD::zeros(vec![1, 3, 336, 336]),
            num_img_tokens: vec![0; sizes.len()],
            image_sizes: sizes,
            model_specific,
        }
    }
}
