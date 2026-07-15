use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    audio::{AudioPreProcessor, InklingAudioProcessor},
    encoder_inputs::PreprocessedEncoderInputs,
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{EncoderFieldLayouts, FieldLayout, Modality, PromptReplacement, TokenId},
    vision::PreProcessorConfig,
};

pub(super) struct InklingSpec;

impl InklingSpec {
    const CONTENT_IMAGE: &'static str = "<|content_image|>";
    const CONTENT_AUDIO_INPUT: &'static str = "<|content_audio_input|>";
    const AUDIO: &'static str = "<|audio|>";

    fn image_transport_fill_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.token_id(Self::CONTENT_IMAGE)
    }

    fn audio_placeholder_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.token_id(Self::AUDIO)
    }

    fn tower_enabled(metadata: &ModelMetadata, config_key: &str) -> bool {
        metadata
            .config
            .get(config_key)
            .and_then(|config| config.get("decoder_dmodel"))
            .is_some_and(|value| !value.is_null())
    }
}

impl ModelProcessorSpec for InklingSpec {
    fn name(&self) -> &'static str {
        "inkling"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        id.contains("inkling")
            || metadata
                .config_model_type()
                .is_some_and(|mt| mt == "inkling_mm_model")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(Self::CONTENT_IMAGE.to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::image_transport_fill_id(metadata)
    }

    fn placeholder_token_for(
        &self,
        _metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => Ok(Self::CONTENT_IMAGE.to_string()),
            Modality::Audio => Ok(Self::CONTENT_AUDIO_INPUT.to_string()),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn placeholder_token_id_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<TokenId> {
        match modality {
            Modality::Image => Self::image_transport_fill_id(metadata),
            Modality::Audio => Self::audio_placeholder_id(metadata),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn modality_limits(
        &self,
        metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        let mut limits = HashMap::new();
        if Self::tower_enabled(metadata, "vision_config") {
            limits.insert(Modality::Image, 10);
        }
        if Self::tower_enabled(metadata, "audio_config") {
            limits.insert(Modality::Audio, 10);
        }
        Ok(limits)
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn audio_processor(
        &self,
        model_config: &Value,
        _preprocessor_config: &PreProcessorConfig,
    ) -> Option<Box<dyn AudioPreProcessor>> {
        Some(Box::new(InklingAudioProcessor::from_model_config(
            model_config,
        )))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        self.prompt_replacements_for(metadata, preprocessed, Modality::Image)
    }

    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        match modality {
            Modality::Image => {
                let content_image_id = metadata.token_id(Self::CONTENT_IMAGE)?;
                Ok(preprocessed
                    .feature_token_counts
                    .iter()
                    .map(|&num_tokens| {
                        let mut tokens = Vec::with_capacity(num_tokens + 1);
                        tokens.push(content_image_id);
                        // TML transports images as a typed span and does not
                        // define a positive image target token. TokenSpeed only
                        // needs real ids until it rewrites the explicit feature
                        // offsets to content-derived MM pad values, so reuse the
                        // public content token as an internal transport fill.
                        tokens.extend(std::iter::repeat_n(content_image_id, num_tokens));
                        PromptReplacement::sequence(Modality::Image, Self::CONTENT_IMAGE, tokens)
                            .with_feature_span(1, num_tokens)
                    })
                    .collect())
            }
            Modality::Audio => {
                let content_audio_id = metadata.token_id(Self::CONTENT_AUDIO_INPUT)?;
                let audio_id = Self::audio_placeholder_id(metadata)?;
                Ok(preprocessed
                    .feature_token_counts
                    .iter()
                    .map(|&num_tokens| {
                        let mut tokens = Vec::with_capacity(num_tokens + 1);
                        tokens.push(content_audio_id);
                        tokens.extend(std::iter::repeat_n(audio_id, num_tokens));
                        PromptReplacement::sequence(
                            Modality::Audio,
                            Self::CONTENT_AUDIO_INPUT,
                            tokens,
                        )
                        .with_feature_span(1, num_tokens)
                    })
                    .collect())
            }
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn encoder_field_layouts_for(&self, modality: Modality) -> EncoderFieldLayouts {
        match modality {
            Modality::Image | Modality::Audio => EncoderFieldLayouts::new(
                FieldLayout::flat("tokens_per_item"),
                HashMap::from([("tokens_per_item".to_string(), FieldLayout::Batched)]),
            ),
            Modality::Video | Modality::ImageEmbeds => EncoderFieldLayouts::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::InklingSpec;
    use crate::{
        registry::{test_helpers::*, ModelMetadata, ModelProcessorSpec, ModelRegistry},
        types::ImageSize,
    };

    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new(&[
            ("<|content_image|>", 200005),
            ("<|content_audio_input|>", 200020),
            ("<|audio|>", 200023),
        ])
    }

    fn config() -> serde_json::Value {
        json!({
            "model_type": "inkling_mm_model",
            "architectures": ["InklingForConditionalGeneration"],
            "vision_config": {"decoder_dmodel": 6144},
            "audio_config": {"decoder_dmodel": 6144}
        })
    }

    #[test]
    fn inkling_matches_model_type() {
        let tokenizer = tokenizer();
        let config = config();
        let metadata = ModelMetadata {
            model_id: "local-checkpoint",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("inkling spec");
        assert_eq!(spec.name(), "inkling");
    }

    #[test]
    fn inkling_matches_family_name_without_model_type() {
        let tokenizer = tokenizer();
        let config = json!({});
        let metadata = ModelMetadata {
            model_id: "org/inkling-chat",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("inkling spec");
        assert_eq!(spec.name(), "inkling");
    }

    #[test]
    fn inkling_spec_builds_audio_processor() {
        use std::sync::Arc;

        use bytes::Bytes;

        use crate::{
            audio::DecodedAudio,
            types::{AudioClip, AudioSource},
            vision::PreProcessorConfig,
        };

        let config = json!({"audio_config": {"n_mel_bins": 8}});
        let processor = InklingSpec
            .audio_processor(&config, &PreProcessorConfig::default())
            .expect("inkling spec must provide an audio processor");

        let clip = Arc::new(AudioClip::new(
            Bytes::from_static(b"audio"),
            DecodedAudio {
                samples: vec![0.0; 800],
                sample_rate: 16_000,
            },
            AudioSource::InlineBytes,
            "audio-hash".to_string(),
        ));
        let result = processor.preprocess(&[clip]).unwrap();
        assert_eq!(result.encoder_input.shape(), &[1, 8]);
    }

    #[test]
    fn modality_limits_follow_configured_towers() {
        let tokenizer = tokenizer();
        for (config, expected) in [
            (
                json!({
                    "vision_config": {"decoder_dmodel": 6144},
                    "audio_config": {"decoder_dmodel": 6144}
                }),
                HashMap::from([
                    (crate::types::Modality::Image, 10),
                    (crate::types::Modality::Audio, 10),
                ]),
            ),
            (
                json!({
                    "vision_config": {"decoder_dmodel": null},
                    "audio_config": {"decoder_dmodel": 6144}
                }),
                HashMap::from([(crate::types::Modality::Audio, 10)]),
            ),
            (
                json!({
                    "vision_config": {"decoder_dmodel": 6144},
                    "audio_config": {}
                }),
                HashMap::from([(crate::types::Modality::Image, 10)]),
            ),
            (json!({}), HashMap::new()),
        ] {
            let metadata = ModelMetadata {
                model_id: "inkling-test",
                tokenizer: &tokenizer,
                config: &config,
            };
            assert_eq!(InklingSpec.modality_limits(&metadata).unwrap(), expected);
        }
    }

    #[test]
    fn image_replacement_preserves_content_token_and_adds_patch_span() {
        let tokenizer = tokenizer();
        let config = config();
        let metadata = ModelMetadata {
            model_id: "local-checkpoint",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("inkling spec");

        let replacements = spec
            .prompt_replacements_for(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(80, 40)], &[3]),
                crate::types::Modality::Image,
            )
            .unwrap();
        assert_eq!(replacements[0].tokens, vec![200005, 200005, 200005, 200005]);
        assert_eq!(
            replacements[0].feature_ranges,
            Some(vec![crate::types::PlaceholderRange {
                offset: 1,
                length: 3
            }])
        );
        assert_eq!(
            spec.placeholder_token_id_for(&metadata, crate::types::Modality::Image)
                .unwrap(),
            200005
        );
    }

    #[test]
    fn audio_replacement_preserves_content_token_and_adds_placeholder_span() {
        let tokenizer = tokenizer();
        let config = config();
        let metadata = ModelMetadata {
            model_id: "local-checkpoint",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("inkling spec");

        let replacements = spec
            .prompt_replacements_for(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(80, 2)], &[2]),
                crate::types::Modality::Audio,
            )
            .unwrap();
        assert_eq!(replacements[0].tokens, vec![200020, 200023, 200023]);
        assert_eq!(
            replacements[0].feature_ranges,
            Some(vec![crate::types::PlaceholderRange {
                offset: 1,
                length: 2
            }])
        );
        assert_eq!(
            spec.placeholder_token_id_for(&metadata, crate::types::Modality::Audio)
                .unwrap(),
            200023
        );
    }
}
