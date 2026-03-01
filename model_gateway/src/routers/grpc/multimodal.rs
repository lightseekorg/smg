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
    MediaConnector, MediaConnectorConfig, Modality, ModelMetadata, ModelProcessorSpec,
    ModelRegistry, ModelSpecificValue, PlaceholderRange, PreProcessorConfig, PreprocessedImages,
    PromptReplacement, TrackedMedia, TrackerOutput,
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

/// Process multimodal content: fetch images, preprocess pixels, expand tokens, collect hashes.
///
/// Single entry point called from preparation.rs. Handles the full pipeline:
/// extract content parts → fetch images → preprocess → expand tokens → collect hashes.
pub(crate) async fn process_multimodal(
    messages: &[ChatMessage],
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    token_ids: Vec<u32>,
    components: &MultimodalComponents,
    tokenizer_source: &str,
) -> Result<MultimodalOutput> {
    // Step 1: Fetch images
    let content_parts = extract_content_parts(messages);
    let mut tracker = AsyncMultiModalTracker::new(components.media_connector.clone());

    for part in content_parts {
        tracker
            .push_part(part)
            .map_err(|e| anyhow::anyhow!("Failed to push content part: {e}"))?;
    }

    let tracker_output: TrackerOutput = tracker
        .finalize()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to finalize multimodal tracker: {e}"))?;

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

    // Step 2: Resolve model spec and preprocess images
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

    let image_processor = components
        .image_processor_registry
        .find(model_id)
        .ok_or_else(|| anyhow::anyhow!("No image processor found for model: {model_id}"))?;

    // ImagePreProcessor::preprocess takes &[DynamicImage]; images are behind Arc<ImageFrame>.
    // Clone cost is negligible vs. the preprocessing work itself.
    let dynamic_images: Vec<image::DynamicImage> = images.iter().map(|f| f.image.clone()).collect();

    let preprocessed: PreprocessedImages = image_processor
        .preprocess(&dynamic_images, &model_config.preprocessor_config)
        .map_err(|e| anyhow::anyhow!("Image preprocessing failed: {e}"))?;

    debug!(
        num_images = preprocessed.num_img_tokens.len(),
        total_tokens = preprocessed.num_img_tokens.iter().sum::<usize>(),
        "Image preprocessing complete"
    );

    // Step 3: Compute prompt replacements and expand tokens.
    let prompt_replacements = spec
        .prompt_replacements(&metadata, &preprocessed)
        .map_err(|e| anyhow::anyhow!("Failed to compute prompt replacements: {e}"))?;

    // Two token IDs may differ for the same placeholder:
    // - search_token_id: what the tokenizer actually emits (e.g. 200090 for "<|image|>")
    // - im_token_id: what the model config declares (e.g. config.image_token_index = 200092)
    let placeholder_token = spec
        .placeholder_token(&metadata)
        .map_err(|e| anyhow::anyhow!("Failed to get placeholder token: {e}"))?;
    let search_token_id = tokenizer.token_to_id(&placeholder_token);
    let im_token_id: Option<u32> = spec
        .placeholder_token_id(&metadata)
        .ok()
        .map(|id| id as u32)
        .or(search_token_id);

    let expanded = expand_tokens(
        &token_ids,
        search_token_id,
        im_token_id,
        &prompt_replacements,
    );

    debug!(
        original_len = token_ids.len(),
        expanded_len = expanded.token_ids.len(),
        placeholder_count = expanded.placeholders.len(),
        ?search_token_id,
        ?im_token_id,
        "Token expansion complete"
    );

    // Step 4: Build multimodal data (includes hash collection)
    let multimodal_data = build_multimodal_data(
        &preprocessed,
        im_token_id,
        &expanded.placeholders,
        expanded.patch_offsets,
        &images,
        spec,
    );

    Ok(MultimodalOutput {
        expanded_token_ids: expanded.token_ids,
        multimodal_data,
    })
}

/// Output of token expansion, containing both full structural and patch-only ranges.
struct ExpandedTokens {
    /// The expanded token ID sequence.
    token_ids: Vec<u32>,
    /// Full structural placeholder ranges (offset, length) covering the entire
    /// replacement including structural tokens. Used by vLLM (which filters via is_embed).
    placeholders: Vec<PlaceholderRange>,
    /// Patch-only placeholder ranges: contiguous runs of `im_token_id` within each
    /// expansion. Used by sglang (which expects offsets aligned 1:1 with vision
    /// encoder output). `None` when `im_token_id` is not set.
    patch_offsets: Option<Vec<(u32, u32)>>,
}

/// Expand placeholder tokens in the token ID sequence.
///
/// For each placeholder token found, replace it with the expanded token sequence
/// from the corresponding `PromptReplacement`. Also track both the full structural
/// placeholder ranges and patch-only offsets (contiguous runs of `im_token_id`)
/// in a single pass — no extra iteration needed.
fn expand_tokens(
    token_ids: &[u32],
    placeholder_token_id: Option<u32>,
    im_token_id: Option<u32>,
    replacements: &[PromptReplacement],
) -> ExpandedTokens {
    let Some(placeholder_id) = placeholder_token_id else {
        // If we can't resolve the placeholder token, return unchanged
        warn!("Could not resolve placeholder token ID; skipping token expansion");
        return ExpandedTokens {
            token_ids: token_ids.to_vec(),
            placeholders: vec![],
            patch_offsets: None,
        };
    };

    let mut expanded = Vec::with_capacity(token_ids.len());
    let mut placeholders = Vec::new();
    let mut patch_offsets: Option<Vec<(u32, u32)>> = im_token_id.map(|_| Vec::new());
    let mut replacement_idx = 0;

    for &token in token_ids {
        if token == placeholder_id && replacement_idx < replacements.len() {
            let repl = &replacements[replacement_idx];
            let offset = expanded.len();

            // Track patch-only runs while extending
            if let (Some(im_id), Some(ref mut offsets)) = (im_token_id, &mut patch_offsets) {
                let mut run_start: Option<u32> = None;
                for (i, &t) in repl.tokens.iter().enumerate() {
                    let pos = (offset + i) as u32;
                    if t as u32 == im_id {
                        if run_start.is_none() {
                            run_start = Some(pos);
                        }
                    } else if let Some(s) = run_start {
                        offsets.push((s, pos - s));
                        run_start = None;
                    }
                }
                if let Some(s) = run_start {
                    offsets.push((s, (offset + repl.tokens.len()) as u32 - s));
                }
            }

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

    ExpandedTokens {
        token_ids: expanded,
        placeholders,
        patch_offsets,
    }
}

/// Build backend-agnostic `MultimodalData` from preprocessed images.
fn build_multimodal_data(
    preprocessed: &PreprocessedImages,
    im_token_id: Option<u32>,
    placeholders: &[PlaceholderRange],
    sglang_patch_offsets: Option<Vec<(u32, u32)>>,
    images: &[Arc<ImageFrame>],
    spec: &dyn ModelProcessorSpec,
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

    // Collect raw image bytes and hashes in a single pass
    let (image_data, mm_hashes): (Vec<Vec<u8>>, Vec<String>) = images
        .iter()
        .map(|frame| (frame.raw_bytes.to_vec(), frame.hash.clone()))
        .unzip();

    // Convert placeholder ranges
    let mm_placeholders = placeholders
        .iter()
        .map(|p| (p.offset as u32, p.length as u32))
        .collect();

    // Field layout classification comes from the model spec in the multimodal crate.
    let layouts = spec.field_layouts();
    let batched_keys = PreprocessedImages::batched_keys(&layouts);
    let flat_keys = PreprocessedImages::flat_keys(&layouts);

    MultimodalData {
        image_data,
        pixel_values: pixel_bytes,
        pixel_values_shape: pixel_shape,
        model_specific_tensors,
        im_token_id,
        mm_placeholders,
        sglang_patch_offsets,
        mm_hashes,
        batched_keys,
        flat_keys,
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

        let result = expand_tokens(&token_ids, Some(100), None, &replacements);

        assert_eq!(result.token_ids, vec![1, 2, 50, 50, 50, 50, 3, 4]);
        assert_eq!(result.placeholders.len(), 1);
        assert_eq!(result.placeholders[0].offset, 2);
        assert_eq!(result.placeholders[0].length, 4);
        assert!(result.patch_offsets.is_none());
    }

    #[test]
    fn test_expand_tokens_no_placeholder() {
        let token_ids = vec![1, 2, 3];
        let result = expand_tokens(&token_ids, None, None, &[]);

        assert_eq!(result.token_ids, vec![1, 2, 3]);
        assert!(result.placeholders.is_empty());
        assert!(result.patch_offsets.is_none());
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

        let result = expand_tokens(&token_ids, Some(100), None, &replacements);

        assert_eq!(result.token_ids, vec![1, 50, 50, 2, 60, 60, 60, 3]);
        assert_eq!(result.placeholders.len(), 2);
        assert_eq!(result.placeholders[0].offset, 1);
        assert_eq!(result.placeholders[0].length, 2);
        assert_eq!(result.placeholders[1].offset, 4);
        assert_eq!(result.placeholders[1].length, 3);
    }

    #[test]
    fn test_expand_tokens_patch_offsets_with_structural() {
        // Simulates Llama-4: placeholder expands to structural + patch tokens
        // 88=image_start, 92=patch(im_token_id), 93=separator, 89=image_end
        let token_ids = vec![1, 100, 2]; // 100 is the placeholder
        let replacements = vec![PromptReplacement {
            modality: Modality::Image,
            placeholder_token: "<image>".to_string(),
            tokens: vec![88, 92, 92, 92, 93, 92, 92, 92, 89], // start + patches + sep + patches + end
        }];

        let result = expand_tokens(&token_ids, Some(100), Some(92), &replacements);

        // Full structural range
        assert_eq!(result.placeholders.len(), 1);
        assert_eq!(result.placeholders[0].offset, 1);
        assert_eq!(result.placeholders[0].length, 9);

        // Patch-only offsets: two runs of token 92
        let patch = result.patch_offsets.unwrap();
        assert_eq!(patch.len(), 2);
        assert_eq!(patch[0], (2, 3)); // offset=2, length=3
        assert_eq!(patch[1], (6, 3)); // offset=6, length=3
    }

    #[test]
    fn test_parse_detail() {
        assert_eq!(parse_detail("auto"), Some(ImageDetail::Auto));
        assert_eq!(parse_detail("Auto"), Some(ImageDetail::Auto));
        assert_eq!(parse_detail("LOW"), Some(ImageDetail::Low));
        assert_eq!(parse_detail("high"), Some(ImageDetail::High));
        assert_eq!(parse_detail("unknown"), None);
    }
}
