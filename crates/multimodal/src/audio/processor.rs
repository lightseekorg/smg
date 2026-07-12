use std::{collections::HashMap, sync::Arc};

use serde_json::Value;

use super::Qwen3AudioProcessor;
use crate::{
    encoder_inputs::PreprocessedEncoderInputs, error::TransformError, types::AudioClip,
    vision::PreProcessorConfig,
};

/// Audio preprocessing contract selected by [`AudioProcessorRegistry`].
pub trait AudioPreProcessor: Send + Sync {
    fn preprocess(
        &self,
        clips: &[Arc<AudioClip>],
    ) -> Result<PreprocessedEncoderInputs, TransformError>;
}

pub type AudioProcessorFactory = fn(&Value, &PreProcessorConfig) -> Box<dyn AudioPreProcessor>;

/// Registry of model-specific audio processor factories.
///
/// Audio processors are created with the current model config because their
/// feature shapes and quantization parameters can be checkpoint-specific.
/// Model-family detection remains the responsibility of `ModelRegistry`; this
/// registry is keyed by the resolved model spec name so that matching logic is
/// not duplicated across capability and processor registries.
pub struct AudioProcessorRegistry {
    factories: HashMap<String, AudioProcessorFactory>,
}

impl AudioProcessorRegistry {
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    pub fn register(&mut self, model_spec: impl Into<String>, factory: AudioProcessorFactory) {
        self.factories.insert(model_spec.into(), factory);
    }

    pub fn create(
        &self,
        model_spec: &str,
        model_config: &Value,
        preprocessor_config: &PreProcessorConfig,
    ) -> Option<Box<dyn AudioPreProcessor>> {
        self.factories
            .get(model_spec)
            .copied()
            .map(|factory| factory(model_config, preprocessor_config))
    }

    pub fn with_defaults() -> Self {
        fn qwen3_audio(
            config: &Value,
            preprocessor_config: &PreProcessorConfig,
        ) -> Box<dyn AudioPreProcessor> {
            Box::new(Qwen3AudioProcessor::from_configs(
                config,
                preprocessor_config,
            ))
        }

        let mut registry = Self::new();
        registry.register("qwen3_asr", qwen3_audio);
        registry.register("qwen3_omni", qwen3_audio);
        registry
    }
}

impl Default for AudioProcessorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;
    use crate::{audio::DecodedAudio, types::AudioSource};

    fn clip() -> Arc<AudioClip> {
        Arc::new(AudioClip::new(
            Bytes::from_static(b"audio"),
            DecodedAudio {
                samples: vec![0.0; 800],
                sample_rate: 16_000,
            },
            AudioSource::InlineBytes,
            "audio-hash".to_string(),
        ))
    }

    #[test]
    fn qwen_registry_applies_preprocessor_config() {
        let registry = AudioProcessorRegistry::with_defaults();
        let preprocessor_config = PreProcessorConfig::from_json(
            r#"{"feature_size": 16, "sampling_rate": 16000, "n_fft": 400, "hop_length": 160}"#,
        )
        .unwrap();
        let processor = registry
            .create("qwen3_asr", &serde_json::json!({}), &preprocessor_config)
            .expect("Qwen audio processor");

        let result = processor.preprocess(&[clip()]).unwrap();
        assert_eq!(result.encoder_input.shape(), &[1, 16, 5]);
    }
}
