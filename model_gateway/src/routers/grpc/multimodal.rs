//! Multimodal processing integration for gRPC chat pipeline.
//!
//! This module bridges the `llm-multimodal` crate with the gRPC router pipeline,
//! handling the full processing chain: extract content parts → fetch images →
//! preprocess pixels → expand placeholder tokens → build proto MultimodalInputs.

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    sync::Arc,
};

use anyhow::{Context, Result};
use dashmap::DashMap;
use llm_multimodal::{
    AsyncMultiModalTracker, ChatContentPart, ConversationSegment, ImageDetail, ImageFrame,
    ImageProcessorRegistry, ImageSize, MediaConnector, MediaConnectorConfig, Modality,
    ModelMetadata, ModelRegistry, ModelSpecificValue, PlaceholderRange, PreProcessorConfig,
    PreprocessedImages, PromptReplacement, TrackedMedia, TrackerConfig, TrackerOutput,
};
use llm_tokenizer::TokenizerTrait;
use openai_protocol::{
    chat::{ChatMessage, MessageContent},
    common::ContentPart,
};
use prost_types::value::Kind;
use smg_grpc_client::sglang_proto;
use tracing::{debug, warn};

fn base64_encode(data: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(data)
}

/// Cached model configuration files loaded from the tokenizer directory.
#[derive(Debug, Clone)]
pub(crate) struct MultimodalModelConfig {
    /// Model config.json (HuggingFace format)
    pub config: serde_json::Value,
    /// Preprocessor config (preprocessor_config.json)
    pub preprocessor_config: PreProcessorConfig,
}

/// Shared multimodal components injected at router creation time.
pub(crate) struct MultimodalComponents {
    pub media_connector: Arc<MediaConnector>,
    pub image_processor_registry: Arc<ImageProcessorRegistry>,
    pub model_registry: Arc<ModelRegistry>,
    /// Lazily-loaded model configs, keyed by model_id.
    pub model_configs: DashMap<String, Arc<MultimodalModelConfig>>,
}

impl MultimodalComponents {
    /// Create multimodal components with default registries.
    pub fn new() -> Result<Self> {
        let client = reqwest::Client::new();
        let media_connector = MediaConnector::new(client, MediaConnectorConfig::default())
            .context("Failed to create MediaConnector")?;

        Ok(Self {
            media_connector: Arc::new(media_connector),
            image_processor_registry: Arc::new(ImageProcessorRegistry::with_defaults()),
            model_registry: Arc::new(ModelRegistry::default()),
            model_configs: DashMap::new(),
        })
    }

    /// Load or retrieve cached model config for a given model and tokenizer source path.
    pub fn get_or_load_config(
        &self,
        model_id: &str,
        tokenizer_source: &str,
    ) -> Result<Arc<MultimodalModelConfig>> {
        if let Some(cached) = self.model_configs.get(model_id) {
            return Ok(cached.clone());
        }

        let base_dir = Path::new(tokenizer_source)
            .parent()
            .unwrap_or(Path::new(tokenizer_source));

        // Load config.json
        let config_path = base_dir.join("config.json");
        let config: serde_json::Value = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config.json at {}", config_path.display()))
            .and_then(|s| {
                serde_json::from_str(&s).with_context(|| {
                    format!("Failed to parse config.json at {}", config_path.display())
                })
            })?;

        // Load preprocessor_config.json
        let pp_config_path = base_dir.join("preprocessor_config.json");
        let preprocessor_config = std::fs::read_to_string(&pp_config_path)
            .with_context(|| {
                format!(
                    "Failed to read preprocessor_config.json at {}",
                    pp_config_path.display()
                )
            })
            .and_then(|s| {
                PreProcessorConfig::from_json(&s).with_context(|| {
                    format!(
                        "Failed to parse preprocessor_config.json at {}",
                        pp_config_path.display()
                    )
                })
            })?;

        let model_config = Arc::new(MultimodalModelConfig {
            config,
            preprocessor_config,
        });

        self.model_configs
            .insert(model_id.to_string(), model_config.clone());
        Ok(model_config)
    }
}

/// Output of the multimodal processing pipeline.
pub(crate) struct MultimodalOutput {
    /// Token IDs with placeholder tokens expanded to the correct count per image.
    pub expanded_token_ids: Vec<u32>,
    /// Proto-ready multimodal inputs for the SGLang GenerateRequest.
    pub proto_mm_inputs: sglang_proto::MultimodalInputs,
}

/// Check if any messages in the request contain multimodal content (images).
pub(crate) fn has_multimodal_content(messages: &[ChatMessage]) -> bool {
    messages.iter().any(|msg| {
        let content = match msg {
            ChatMessage::User { content, .. } => Some(content),
            ChatMessage::System { content, .. } => Some(content),
            ChatMessage::Developer { content, .. } => Some(content),
            _ => None,
        };
        content.is_some_and(|c| match c {
            MessageContent::Parts(parts) => parts
                .iter()
                .any(|p| matches!(p, ContentPart::ImageUrl { .. })),
            MessageContent::Text(_) => false,
        })
    })
}

/// Extract multimodal content parts from OpenAI chat messages,
/// converting protocol `ContentPart` to multimodal crate `ChatContentPart`.
fn extract_content_parts(messages: &[ChatMessage]) -> Vec<ChatContentPart> {
    let mut parts = Vec::new();

    for msg in messages {
        let content = match msg {
            ChatMessage::User { content, .. } => Some(content),
            ChatMessage::System { content, .. } => Some(content),
            ChatMessage::Developer { content, .. } => Some(content),
            _ => None,
        };

        if let Some(MessageContent::Parts(message_parts)) = content {
            for part in message_parts {
                match part {
                    ContentPart::ImageUrl { image_url } => {
                        let detail = image_url.detail.as_deref().and_then(parse_detail);
                        parts.push(ChatContentPart::ImageUrl {
                            url: image_url.url.clone(),
                            detail,
                            uuid: None,
                        });
                    }
                    ContentPart::Text { text } => {
                        parts.push(ChatContentPart::Text { text: text.clone() });
                    }
                    _ => {} // Skip VideoUrl etc. for now
                }
            }
        }
    }

    parts
}

/// Parse OpenAI detail string to multimodal ImageDetail enum.
fn parse_detail(detail: &str) -> Option<ImageDetail> {
    match detail.to_ascii_lowercase().as_str() {
        "auto" => Some(ImageDetail::Auto),
        "low" => Some(ImageDetail::Low),
        "high" => Some(ImageDetail::High),
        _ => None,
    }
}

/// Reconstruct the text content from tracker conversation segments.
///
/// Placeholder segments produce the placeholder token string (e.g., "<image>").
#[allow(dead_code)]
fn reconstruct_text_from_segments(segments: &[ConversationSegment]) -> String {
    let mut text = String::new();
    for segment in segments {
        match segment {
            ConversationSegment::Text(s) => text.push_str(s),
            ConversationSegment::Placeholder { token } => text.push_str(token),
        }
    }
    text
}

/// Full multimodal processing pipeline.
///
/// This function:
/// 1. Extracts image content parts from messages
/// 2. Uses AsyncMultiModalTracker to fetch images + insert placeholders
/// 3. Preprocesses images via ImagePreProcessor (pixel values)
/// 4. Computes prompt replacements (token expansion rules)
/// 5. Expands placeholder tokens in token_ids
/// 6. Builds proto MultimodalInputs
pub(crate) async fn process_multimodal(
    messages: &[ChatMessage],
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    token_ids: Vec<u32>,
    components: &MultimodalComponents,
    tokenizer_source: &str,
) -> Result<MultimodalOutput> {
    // Load model configs (config.json + preprocessor_config.json)
    let model_config = components.get_or_load_config(model_id, tokenizer_source)?;

    let metadata = ModelMetadata {
        model_id,
        tokenizer,
        config: &model_config.config,
    };

    // Look up model spec in registry
    let spec = components
        .model_registry
        .lookup(&metadata)
        .ok_or_else(|| anyhow::anyhow!("Multimodal not supported for model: {}", model_id))?;

    debug!(
        model_id,
        spec_name = spec.name(),
        "Found multimodal model spec"
    );

    // Build tracker config from spec
    let placeholder_token = spec
        .placeholder_token(&metadata)
        .map_err(|e| anyhow::anyhow!("Failed to get placeholder token: {}", e))?;
    let modality_limits = spec
        .modality_limits(&metadata)
        .map_err(|e| anyhow::anyhow!("Failed to get modality limits: {}", e))?;

    let tracker_config = TrackerConfig {
        placeholder_tokens: {
            let mut m = HashMap::new();
            m.insert(Modality::Image, placeholder_token.clone());
            m
        },
        modality_limits,
    };

    // Extract content parts and run tracker
    let content_parts = extract_content_parts(messages);
    let mut tracker =
        AsyncMultiModalTracker::new(components.media_connector.clone(), tracker_config);

    for part in content_parts {
        tracker
            .push_part(part)
            .map_err(|e| anyhow::anyhow!("Failed to push content part: {}", e))?;
    }

    let tracker_output: TrackerOutput = tracker
        .finalize()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to finalize multimodal tracker: {}", e))?;

    // Collect fetched images from tracker output
    let images: Vec<Arc<ImageFrame>> = tracker_output
        .data
        .get(&Modality::Image)
        .map(|media_vec| {
            media_vec
                .iter()
                .filter_map(|m| match m {
                    TrackedMedia::Image(frame) => Some(frame.clone()),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default();

    if images.is_empty() {
        return Err(anyhow::anyhow!(
            "No images were successfully fetched for multimodal request"
        ));
    }

    debug!(
        image_count = images.len(),
        "Fetched images for multimodal processing"
    );

    // Find image processor for this model
    let image_processor = components
        .image_processor_registry
        .find(model_id)
        .ok_or_else(|| anyhow::anyhow!("No image processor found for model: {}", model_id))?;

    // Preprocess images (compute pixel values)
    // Clone needed: ImagePreProcessor::preprocess takes &[DynamicImage] but images are behind Arc<ImageFrame>.
    // Cost is negligible vs. the preprocessing work itself.
    let dynamic_images: Vec<image::DynamicImage> = images.iter().map(|f| f.image.clone()).collect();

    let preprocessed: PreprocessedImages = image_processor
        .preprocess(&dynamic_images, &model_config.preprocessor_config)
        .map_err(|e| anyhow::anyhow!("Image preprocessing failed: {}", e))?;

    debug!(
        num_images = preprocessed.num_img_tokens.len(),
        total_tokens = preprocessed.num_img_tokens.iter().sum::<usize>(),
        "Image preprocessing complete"
    );

    // Compute image sizes for prompt replacement calculation
    let image_sizes: Vec<ImageSize> = preprocessed
        .image_sizes
        .iter()
        .map(|&(w, h)| ImageSize {
            width: w,
            height: h,
        })
        .collect();

    // Get prompt replacements (token expansion rules) from spec
    let prompt_replacements = spec
        .prompt_replacements(&metadata, &image_sizes)
        .map_err(|e| anyhow::anyhow!("Failed to compute prompt replacements: {}", e))?;

    // Expand placeholder tokens in the token ID sequence
    let placeholder_token_id = tokenizer.token_to_id(&placeholder_token);
    let (expanded_token_ids, mm_placeholders) =
        expand_tokens(&token_ids, placeholder_token_id, &prompt_replacements);

    debug!(
        original_len = token_ids.len(),
        expanded_len = expanded_token_ids.len(),
        placeholder_count = mm_placeholders.len(),
        "Token expansion complete"
    );

    // Build proto MultimodalInputs
    let proto_mm_inputs = build_proto_multimodal_inputs(&preprocessed, &mm_placeholders, &images);

    Ok(MultimodalOutput {
        expanded_token_ids,
        proto_mm_inputs,
    })
}

/// Expand placeholder tokens in the token ID sequence.
///
/// For each placeholder token found, replace it with the expanded token sequence
/// from the corresponding `PromptReplacement`. Also track placeholder ranges.
fn expand_tokens(
    token_ids: &[u32],
    placeholder_token_id: Option<u32>,
    replacements: &[PromptReplacement],
) -> (Vec<u32>, Vec<PlaceholderRange>) {
    let Some(placeholder_id) = placeholder_token_id else {
        // If we can't resolve the placeholder token, return unchanged
        warn!("Could not resolve placeholder token ID; skipping token expansion");
        return (token_ids.to_vec(), vec![]);
    };

    let mut expanded = Vec::with_capacity(token_ids.len());
    let mut placeholders = Vec::new();
    let mut replacement_idx = 0;

    for &token in token_ids {
        if token == placeholder_id && replacement_idx < replacements.len() {
            let repl = &replacements[replacement_idx];
            let offset = expanded.len();
            // PromptReplacement uses TokenId = i32, convert to u32
            for &t in &repl.tokens {
                expanded.push(t as u32);
            }
            placeholders.push(PlaceholderRange {
                offset,
                length: repl.tokens.len(),
            });
            replacement_idx += 1;
        } else {
            expanded.push(token);
        }
    }

    if replacement_idx < replacements.len() {
        warn!(
            expected = replacements.len(),
            found = replacement_idx,
            "Fewer placeholder tokens found in sequence than expected"
        );
    }

    (expanded, placeholders)
}

/// Build proto `MultimodalInputs` from preprocessed images and placeholder ranges.
fn build_proto_multimodal_inputs(
    preprocessed: &PreprocessedImages,
    placeholders: &[PlaceholderRange],
    images: &[Arc<ImageFrame>],
) -> sglang_proto::MultimodalInputs {
    // Serialize pixel values as raw f32 bytes
    let pixel_bytes = pixel_values_to_bytes(preprocessed);

    // Build processed_features as a protobuf Struct
    let mut fields = BTreeMap::new();

    // pixel_values as base64-encoded bytes stored in a string value
    // (protobuf Struct doesn't have a bytes type, so we encode as base64)
    let pixel_b64 = base64_encode(&pixel_bytes);
    fields.insert(
        "pixel_values".to_string(),
        prost_types::Value {
            kind: Some(Kind::StringValue(pixel_b64)),
        },
    );

    // pixel_values shape
    let shape_list: Vec<prost_types::Value> = preprocessed
        .pixel_values
        .shape()
        .iter()
        .map(|&dim| prost_types::Value {
            kind: Some(Kind::NumberValue(dim as f64)),
        })
        .collect();
    fields.insert(
        "pixel_values_shape".to_string(),
        prost_types::Value {
            kind: Some(Kind::ListValue(prost_types::ListValue {
                values: shape_list,
            })),
        },
    );

    // pixel_values dtype
    fields.insert(
        "pixel_values_dtype".to_string(),
        prost_types::Value {
            kind: Some(Kind::StringValue("float32".to_string())),
        },
    );

    // Image sizes
    let sizes_list: Vec<prost_types::Value> = preprocessed
        .image_sizes
        .iter()
        .map(|&(w, h)| prost_types::Value {
            kind: Some(Kind::ListValue(prost_types::ListValue {
                values: vec![
                    prost_types::Value {
                        kind: Some(Kind::NumberValue(w as f64)),
                    },
                    prost_types::Value {
                        kind: Some(Kind::NumberValue(h as f64)),
                    },
                ],
            })),
        })
        .collect();
    fields.insert(
        "image_sizes".to_string(),
        prost_types::Value {
            kind: Some(Kind::ListValue(prost_types::ListValue {
                values: sizes_list,
            })),
        },
    );

    // Placeholder ranges for token alignment
    let placeholder_list: Vec<prost_types::Value> = placeholders
        .iter()
        .map(|p| {
            let mut ph_fields = BTreeMap::new();
            ph_fields.insert(
                "offset".to_string(),
                prost_types::Value {
                    kind: Some(Kind::NumberValue(p.offset as f64)),
                },
            );
            ph_fields.insert(
                "length".to_string(),
                prost_types::Value {
                    kind: Some(Kind::NumberValue(p.length as f64)),
                },
            );
            prost_types::Value {
                kind: Some(Kind::StructValue(prost_types::Struct { fields: ph_fields })),
            }
        })
        .collect();
    fields.insert(
        "mm_placeholders".to_string(),
        prost_types::Value {
            kind: Some(Kind::ListValue(prost_types::ListValue {
                values: placeholder_list,
            })),
        },
    );

    // num_img_tokens per image
    let token_counts: Vec<prost_types::Value> = preprocessed
        .num_img_tokens
        .iter()
        .map(|&count| prost_types::Value {
            kind: Some(Kind::NumberValue(count as f64)),
        })
        .collect();
    fields.insert(
        "num_img_tokens".to_string(),
        prost_types::Value {
            kind: Some(Kind::ListValue(prost_types::ListValue {
                values: token_counts,
            })),
        },
    );

    // Add model-specific values
    for (key, value) in &preprocessed.model_specific {
        if let Some(proto_value) = model_specific_to_proto(value) {
            fields.insert(key.clone(), proto_value);
        }
    }

    // Collect raw image bytes for the image_data field
    let image_data: Vec<Vec<u8>> = images
        .iter()
        .map(|frame| frame.raw_bytes.to_vec())
        .collect();

    sglang_proto::MultimodalInputs {
        image_urls: vec![],
        video_urls: vec![],
        audio_urls: vec![],
        processed_features: Some(prost_types::Struct { fields }),
        image_data,
        video_data: vec![],
        audio_data: vec![],
        modalities: vec!["image".to_string()],
    }
}

/// Serialize pixel values (ArrayD<f32>) to raw little-endian f32 bytes.
fn pixel_values_to_bytes(preprocessed: &PreprocessedImages) -> Vec<u8> {
    let slice = preprocessed
        .pixel_values
        .as_slice()
        .unwrap_or_else(|| preprocessed.pixel_values.as_slice_memory_order().unwrap());
    let mut bytes = Vec::with_capacity(slice.len() * 4);
    for &val in slice {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Convert a model-specific value to a protobuf Value.
fn model_specific_to_proto(value: &ModelSpecificValue) -> Option<prost_types::Value> {
    match value {
        ModelSpecificValue::Tensor { data, shape } => {
            // Encode tensor as base64 bytes + shape
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            let b64 = base64_encode(&bytes);

            let mut fields = BTreeMap::new();
            fields.insert(
                "data".to_string(),
                prost_types::Value {
                    kind: Some(Kind::StringValue(b64)),
                },
            );
            fields.insert(
                "shape".to_string(),
                prost_types::Value {
                    kind: Some(Kind::ListValue(prost_types::ListValue {
                        values: shape
                            .iter()
                            .map(|&d| prost_types::Value {
                                kind: Some(Kind::NumberValue(d as f64)),
                            })
                            .collect(),
                    })),
                },
            );
            fields.insert(
                "dtype".to_string(),
                prost_types::Value {
                    kind: Some(Kind::StringValue("float32".to_string())),
                },
            );
            Some(prost_types::Value {
                kind: Some(Kind::StructValue(prost_types::Struct { fields })),
            })
        }
        ModelSpecificValue::IntTensor { data, shape } => {
            let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
            let b64 = base64_encode(&bytes);

            let mut fields = BTreeMap::new();
            fields.insert(
                "data".to_string(),
                prost_types::Value {
                    kind: Some(Kind::StringValue(b64)),
                },
            );
            fields.insert(
                "shape".to_string(),
                prost_types::Value {
                    kind: Some(Kind::ListValue(prost_types::ListValue {
                        values: shape
                            .iter()
                            .map(|&d| prost_types::Value {
                                kind: Some(Kind::NumberValue(d as f64)),
                            })
                            .collect(),
                    })),
                },
            );
            fields.insert(
                "dtype".to_string(),
                prost_types::Value {
                    kind: Some(Kind::StringValue("int64".to_string())),
                },
            );
            Some(prost_types::Value {
                kind: Some(Kind::StructValue(prost_types::Struct { fields })),
            })
        }
        ModelSpecificValue::UintTensor { data, shape } => {
            let bytes: Vec<u8> = data.iter().flat_map(|u| u.to_le_bytes()).collect();
            let b64 = base64_encode(&bytes);

            let mut fields = BTreeMap::new();
            fields.insert(
                "data".to_string(),
                prost_types::Value {
                    kind: Some(Kind::StringValue(b64)),
                },
            );
            fields.insert(
                "shape".to_string(),
                prost_types::Value {
                    kind: Some(Kind::ListValue(prost_types::ListValue {
                        values: shape
                            .iter()
                            .map(|&d| prost_types::Value {
                                kind: Some(Kind::NumberValue(d as f64)),
                            })
                            .collect(),
                    })),
                },
            );
            fields.insert(
                "dtype".to_string(),
                prost_types::Value {
                    kind: Some(Kind::StringValue("uint32".to_string())),
                },
            );
            Some(prost_types::Value {
                kind: Some(Kind::StructValue(prost_types::Struct { fields })),
            })
        }
        ModelSpecificValue::Int(v) => Some(prost_types::Value {
            kind: Some(Kind::NumberValue(*v as f64)),
        }),
        ModelSpecificValue::Float(v) => Some(prost_types::Value {
            kind: Some(Kind::NumberValue(*v)),
        }),
        ModelSpecificValue::IntVec(v) => Some(prost_types::Value {
            kind: Some(Kind::ListValue(prost_types::ListValue {
                values: v
                    .iter()
                    .map(|&i| prost_types::Value {
                        kind: Some(Kind::NumberValue(i as f64)),
                    })
                    .collect(),
            })),
        }),
        ModelSpecificValue::UintVec(v) => Some(prost_types::Value {
            kind: Some(Kind::ListValue(prost_types::ListValue {
                values: v
                    .iter()
                    .map(|&u| prost_types::Value {
                        kind: Some(Kind::NumberValue(u as f64)),
                    })
                    .collect(),
            })),
        }),
        ModelSpecificValue::FloatVec(v) => Some(prost_types::Value {
            kind: Some(Kind::ListValue(prost_types::ListValue {
                values: v
                    .iter()
                    .map(|&f| prost_types::Value {
                        kind: Some(Kind::NumberValue(f as f64)),
                    })
                    .collect(),
            })),
        }),
        ModelSpecificValue::TupleVec(v) => Some(prost_types::Value {
            kind: Some(Kind::ListValue(prost_types::ListValue {
                values: v
                    .iter()
                    .map(|&(a, b)| prost_types::Value {
                        kind: Some(Kind::ListValue(prost_types::ListValue {
                            values: vec![
                                prost_types::Value {
                                    kind: Some(Kind::NumberValue(a as f64)),
                                },
                                prost_types::Value {
                                    kind: Some(Kind::NumberValue(b as f64)),
                                },
                            ],
                        })),
                    })
                    .collect(),
            })),
        }),
        ModelSpecificValue::Bool(v) => Some(prost_types::Value {
            kind: Some(Kind::BoolValue(*v)),
        }),
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::ImageUrl;

    use super::*;

    #[test]
    fn test_has_multimodal_content_with_images() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "What is this?".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/cat.jpg".to_string(),
                        detail: None,
                    },
                },
            ]),
            name: None,
        }];

        assert!(has_multimodal_content(&messages));
    }

    #[test]
    fn test_has_multimodal_content_text_only() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Text("Hello".to_string()),
            name: None,
        }];

        assert!(!has_multimodal_content(&messages));
    }

    #[test]
    fn test_has_multimodal_content_parts_text_only() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![ContentPart::Text {
                text: "Just text".to_string(),
            }]),
            name: None,
        }];

        assert!(!has_multimodal_content(&messages));
    }

    #[test]
    fn test_extract_content_parts() {
        let messages = vec![
            ChatMessage::System {
                content: MessageContent::Text("You are helpful".to_string()),
                name: None,
            },
            ChatMessage::User {
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "Describe this:".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image.jpg".to_string(),
                            detail: Some("high".to_string()),
                        },
                    },
                ]),
                name: None,
            },
        ];

        let parts = extract_content_parts(&messages);
        assert_eq!(parts.len(), 2);

        match &parts[0] {
            ChatContentPart::Text { text } => assert_eq!(text, "Describe this:"),
            _ => panic!("Expected Text part"),
        }

        match &parts[1] {
            ChatContentPart::ImageUrl { url, detail, .. } => {
                assert_eq!(url, "https://example.com/image.jpg");
                assert_eq!(*detail, Some(ImageDetail::High));
            }
            _ => panic!("Expected ImageUrl part"),
        }
    }

    #[test]
    fn test_expand_tokens_basic() {
        let token_ids = vec![1, 2, 100, 3, 4]; // 100 is the placeholder
        let replacements = vec![PromptReplacement {
            modality: Modality::Image,
            placeholder_token: "<image>".to_string(),
            tokens: vec![50, 50, 50, 50], // Expand to 4 tokens
        }];

        let (expanded, placeholders) = expand_tokens(&token_ids, Some(100), &replacements);

        assert_eq!(expanded, vec![1, 2, 50, 50, 50, 50, 3, 4]);
        assert_eq!(placeholders.len(), 1);
        assert_eq!(placeholders[0].offset, 2);
        assert_eq!(placeholders[0].length, 4);
    }

    #[test]
    fn test_expand_tokens_no_placeholder() {
        let token_ids = vec![1, 2, 3];
        let (expanded, placeholders) = expand_tokens(&token_ids, None, &[]);

        assert_eq!(expanded, vec![1, 2, 3]);
        assert!(placeholders.is_empty());
    }

    #[test]
    fn test_expand_tokens_multiple_images() {
        let token_ids = vec![1, 100, 2, 100, 3]; // Two placeholder tokens
        let replacements = vec![
            PromptReplacement {
                modality: Modality::Image,
                placeholder_token: "<image>".to_string(),
                tokens: vec![50, 50], // 2 tokens for first image
            },
            PromptReplacement {
                modality: Modality::Image,
                placeholder_token: "<image>".to_string(),
                tokens: vec![60, 60, 60], // 3 tokens for second image
            },
        ];

        let (expanded, placeholders) = expand_tokens(&token_ids, Some(100), &replacements);

        assert_eq!(expanded, vec![1, 50, 50, 2, 60, 60, 60, 3]);
        assert_eq!(placeholders.len(), 2);
        assert_eq!(placeholders[0].offset, 1);
        assert_eq!(placeholders[0].length, 2);
        assert_eq!(placeholders[1].offset, 4);
        assert_eq!(placeholders[1].length, 3);
    }

    #[test]
    fn test_parse_detail() {
        assert_eq!(parse_detail("auto"), Some(ImageDetail::Auto));
        assert_eq!(parse_detail("Auto"), Some(ImageDetail::Auto));
        assert_eq!(parse_detail("LOW"), Some(ImageDetail::Low));
        assert_eq!(parse_detail("high"), Some(ImageDetail::High));
        assert_eq!(parse_detail("unknown"), None);
    }

    #[test]
    fn test_reconstruct_text_from_segments() {
        let segments = vec![
            ConversationSegment::Text("Hello ".to_string()),
            ConversationSegment::Placeholder {
                token: "<image>".to_string(),
            },
            ConversationSegment::Text(" world".to_string()),
        ];

        let text = reconstruct_text_from_segments(&segments);
        assert_eq!(text, "Hello <image> world");
    }
}
