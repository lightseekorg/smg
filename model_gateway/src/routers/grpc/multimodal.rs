//! Multimodal processing integration for gRPC chat pipeline.
//!
//! This module bridges the `llm-multimodal` crate with the gRPC router pipeline,
//! handling the full processing chain: extract content parts → fetch images →
//! preprocess pixels → expand placeholder tokens → build proto MultimodalInputs.

use std::{collections::HashMap, path::Path, sync::Arc};

use anyhow::{Context, Result};
use dashmap::DashMap;
use llm_multimodal::{
    AsyncMultiModalTracker, ChatContentPart, ImageDetail, ImageFrame, ImageProcessorRegistry,
    ImageSize, MediaConnector, MediaConnectorConfig, Modality, ModelMetadata, ModelProcessorSpec,
    ModelRegistry, ModelSpecificValue, PlaceholderRange, PreProcessorConfig, PreprocessedImages,
    PromptReplacement, TrackedMedia, TrackerConfig, TrackerOutput,
};
use llm_tokenizer::TokenizerTrait;
use openai_protocol::{
    chat::{ChatMessage, MessageContent},
    common::ContentPart,
};
use tracing::{debug, warn};

use crate::routers::grpc::{MultimodalData, TensorBytes};

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
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("Failed to create reqwest client")?;
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

        let base_dir = Path::new(tokenizer_source);

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

/// Load model config and look up the multimodal model spec for the given model.
fn resolve_model_spec<'a>(
    components: &'a MultimodalComponents,
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    tokenizer_source: &str,
) -> Result<(Arc<MultimodalModelConfig>, &'a dyn ModelProcessorSpec)> {
    let model_config = components.get_or_load_config(model_id, tokenizer_source)?;
    let metadata = ModelMetadata {
        model_id,
        tokenizer,
        config: &model_config.config,
    };
    let spec = components
        .model_registry
        .lookup(&metadata)
        .ok_or_else(|| anyhow::anyhow!("Multimodal not supported for model: {model_id}"))?;
    Ok((model_config, spec))
}

/// Output of the multimodal processing pipeline.
pub(crate) struct MultimodalOutput {
    /// Token IDs with placeholder tokens expanded to the correct count per image.
    pub expanded_token_ids: Vec<u32>,
    /// Backend-agnostic multimodal data.
    pub multimodal_data: MultimodalData,
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
                    ContentPart::VideoUrl { .. } => {} // Skip VideoUrl for now
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

/// Phase 1: Fetch images from chat messages (backend-agnostic).
///
/// Uses the multimodal tracker to extract image URLs, fetch them, and enforce
/// modality limits. No pixel preprocessing or token expansion — those are
/// deferred to Phase 2 where the backend type is known.
pub(crate) async fn fetch_images(
    messages: &[ChatMessage],
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    components: &MultimodalComponents,
    tokenizer_source: &str,
) -> Result<Vec<Arc<ImageFrame>>> {
    let (model_config, spec) =
        resolve_model_spec(components, model_id, tokenizer, tokenizer_source)?;

    debug!(
        model_id,
        spec_name = spec.name(),
        "Found multimodal model spec"
    );

    // Build tracker config from spec (for modality limits)
    let metadata = ModelMetadata {
        model_id,
        tokenizer,
        config: &model_config.config,
    };
    let placeholder_token = spec
        .placeholder_token(&metadata)
        .map_err(|e| anyhow::anyhow!("Failed to get placeholder token: {e}"))?;
    let modality_limits = spec
        .modality_limits(&metadata)
        .map_err(|e| anyhow::anyhow!("Failed to get modality limits: {e}"))?;

    let tracker_config = TrackerConfig {
        placeholder_tokens: {
            let mut m = HashMap::new();
            m.insert(Modality::Image, placeholder_token);
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
            .map_err(|e| anyhow::anyhow!("Failed to push content part: {e}"))?;
    }

    let tracker_output: TrackerOutput = tracker
        .finalize()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to finalize multimodal tracker: {e}"))?;

    // Collect fetched images
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

    Ok(images)
}

/// Phase 2 (SGLang): Preprocess images and expand placeholder tokens.
///
/// Takes raw fetched images and produces the full `MultimodalOutput` with
/// preprocessed pixel values, model-specific tensors, and expanded token IDs.
pub(crate) fn preprocess_for_sglang(
    images: &[Arc<ImageFrame>],
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    token_ids: Vec<u32>,
    components: &MultimodalComponents,
    tokenizer_source: &str,
) -> Result<MultimodalOutput> {
    let (model_config, spec) =
        resolve_model_spec(components, model_id, tokenizer, tokenizer_source)?;

    // Find image processor for this model
    let image_processor = components
        .image_processor_registry
        .find(model_id)
        .ok_or_else(|| anyhow::anyhow!("No image processor found for model: {model_id}"))?;

    // Preprocess images (compute pixel values)
    // Clone needed: ImagePreProcessor::preprocess takes &[DynamicImage] but images are behind Arc<ImageFrame>.
    // Cost is negligible vs. the preprocessing work itself.
    let dynamic_images: Vec<image::DynamicImage> = images.iter().map(|f| f.image.clone()).collect();

    let preprocessed: PreprocessedImages = image_processor
        .preprocess(&dynamic_images, &model_config.preprocessor_config)
        .map_err(|e| anyhow::anyhow!("Image preprocessing failed: {e}"))?;

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

    // Reconstruct metadata for spec calls that need it
    let metadata = ModelMetadata {
        model_id,
        tokenizer,
        config: &model_config.config,
    };

    // Get prompt replacements (token expansion rules) from spec
    let prompt_replacements = spec
        .prompt_replacements(&metadata, &image_sizes)
        .map_err(|e| anyhow::anyhow!("Failed to compute prompt replacements: {e}"))?;

    // Two token IDs may differ for the same placeholder:
    // - search_token_id: what the tokenizer actually emits (e.g. 200090 for "<|image|>")
    // - im_token_id: what the model config declares (e.g. config.image_token_index = 200092)
    //
    // expand_tokens searches input_ids for search_token_id and replaces each with
    // the replacement sequence from prompt_replacements (which uses im_token_id).
    // The proto im_token_id tells sglang's pad_input_tokens what to look for
    // in the *expanded* output.
    let placeholder_token = spec
        .placeholder_token(&metadata)
        .map_err(|e| anyhow::anyhow!("Failed to get placeholder token: {e}"))?;
    let search_token_id = tokenizer.token_to_id(&placeholder_token);
    let im_token_id: Option<u32> = spec
        .placeholder_token_id(&metadata)
        .ok()
        .map(|id| id as u32)
        .or(search_token_id);

    let (expanded_token_ids, mm_placeholders) =
        expand_tokens(&token_ids, search_token_id, &prompt_replacements);

    debug!(
        original_len = token_ids.len(),
        expanded_len = expanded_token_ids.len(),
        placeholder_count = mm_placeholders.len(),
        ?search_token_id,
        ?im_token_id,
        "Token expansion complete"
    );

    let multimodal_data =
        build_multimodal_data(&preprocessed, im_token_id, &mm_placeholders, images);

    Ok(MultimodalOutput {
        expanded_token_ids,
        multimodal_data,
    })
}

/// Phase 2 (vLLM): Build minimal multimodal data with raw image bytes only.
///
/// vLLM handles image preprocessing and token expansion internally,
/// so we only send the raw JPEG/PNG bytes.
fn build_vllm_multimodal_data(images: &[Arc<ImageFrame>]) -> MultimodalData {
    MultimodalData {
        image_data: images
            .iter()
            .map(|frame| frame.raw_bytes.to_vec())
            .collect(),
        pixel_values: Vec::new(),
        pixel_values_shape: Vec::new(),
        model_specific_tensors: HashMap::new(),
        im_token_id: None,
        mm_placeholders: Vec::new(),
    }
}

/// Phase 2: Backend-specific multimodal processing.
///
/// For SGLang: preprocesses images, expands placeholder tokens, builds full MultimodalData.
/// For vLLM/others: extracts raw image bytes only, returns original token IDs unchanged.
pub(crate) fn process_for_backend(
    images: &[Arc<ImageFrame>],
    is_sglang: bool,
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    token_ids: Vec<u32>,
    components: &MultimodalComponents,
    tokenizer_source: &str,
) -> Result<(Vec<u32>, MultimodalData)> {
    if is_sglang {
        let output = preprocess_for_sglang(
            images,
            model_id,
            tokenizer,
            token_ids,
            components,
            tokenizer_source,
        )?;
        Ok((output.expanded_token_ids, output.multimodal_data))
    } else {
        Ok((token_ids, build_vllm_multimodal_data(images)))
    }
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
            expanded.extend(repl.tokens.iter().map(|&t| t as u32));
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

/// Build backend-agnostic `MultimodalData` from preprocessed images.
fn build_multimodal_data(
    preprocessed: &PreprocessedImages,
    im_token_id: Option<u32>,
    placeholders: &[PlaceholderRange],
    images: &[Arc<ImageFrame>],
) -> MultimodalData {
    // Serialize pixel values as raw little-endian f32 bytes
    let pixel_bytes: Vec<u8> = if let Some(pixel_slice) = preprocessed
        .pixel_values
        .as_slice()
        .or_else(|| preprocessed.pixel_values.as_slice_memory_order())
    {
        pixel_slice.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else {
        // Fallback for non-contiguous arrays
        preprocessed
            .pixel_values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    };
    let pixel_shape: Vec<u32> = preprocessed
        .pixel_values
        .shape()
        .iter()
        .map(|&d| d as u32)
        .collect();

    // Build model-specific tensors
    let mut model_specific_tensors = HashMap::new();
    for (key, value) in &preprocessed.model_specific {
        if let Some(tensor) = model_specific_to_tensor_bytes(value) {
            model_specific_tensors.insert(key.clone(), tensor);
        }
    }

    // Collect raw image bytes for the image_data field
    let image_data: Vec<Vec<u8>> = images
        .iter()
        .map(|frame| frame.raw_bytes.to_vec())
        .collect();

    // Convert placeholder ranges
    let mm_placeholders = placeholders
        .iter()
        .map(|p| (p.offset as u32, p.length as u32))
        .collect();

    MultimodalData {
        image_data,
        pixel_values: pixel_bytes,
        pixel_values_shape: pixel_shape,
        model_specific_tensors,
        im_token_id,
        mm_placeholders,
    }
}

/// Convert a model-specific value to backend-agnostic TensorBytes.
fn model_specific_to_tensor_bytes(value: &ModelSpecificValue) -> Option<TensorBytes> {
    match value {
        ModelSpecificValue::Tensor { data, shape } => Some(TensorBytes {
            data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            shape: shape.iter().map(|&d| d as u32).collect(),
            dtype: "float32".to_string(),
        }),
        ModelSpecificValue::IntTensor { data, shape } => Some(TensorBytes {
            data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            shape: shape.iter().map(|&d| d as u32).collect(),
            dtype: "int64".to_string(),
        }),
        ModelSpecificValue::UintTensor { data, shape } => Some(TensorBytes {
            data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            shape: shape.iter().map(|&d| d as u32).collect(),
            dtype: "uint32".to_string(),
        }),
        ModelSpecificValue::UintVec(v) => Some(TensorBytes {
            data: v.iter().flat_map(|val| val.to_le_bytes()).collect(),
            shape: vec![v.len() as u32],
            dtype: "uint32".to_string(),
        }),
        ModelSpecificValue::IntVec(v) => Some(TensorBytes {
            data: v.iter().flat_map(|val| val.to_le_bytes()).collect(),
            shape: vec![v.len() as u32],
            dtype: "int64".to_string(),
        }),
        ModelSpecificValue::FloatVec(v) => Some(TensorBytes {
            data: v.iter().flat_map(|val| val.to_le_bytes()).collect(),
            shape: vec![v.len() as u32],
            dtype: "float32".to_string(),
        }),
        // Scalar/tuple/bool types not used by any current processor; skip.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use llm_multimodal::ConversationSegment;
    use openai_protocol::common::ImageUrl;

    use super::*;

    /// Reconstruct the text content from tracker conversation segments.
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
