//! Multimodal processing integration for gRPC pipeline (chat + messages).
//!
//! This module bridges the `llm-multimodal` crate with the gRPC router pipeline,
//! handling the full processing chain: extract content parts → fetch images →
//! preprocess pixels → expand placeholder tokens → build proto MultimodalInputs.
//!
//! Both the chat completion pipeline and the Messages API pipeline share the same
//! processing core (`process_multimodal_parts`). Only the detection and extraction
//! functions differ because they work with different input types (`ChatMessage` vs
//! `InputMessage`).

use std::{collections::HashMap, io::Write, mem::size_of, path::Path, sync::Arc, time::Instant};

use anyhow::{Context, Result};
use dashmap::DashMap;
use llm_multimodal::{
    AsyncMultiModalTracker, FieldLayout, ImageDetail, ImageFrame, ImageProcessorRegistry,
    MediaConnector, MediaConnectorConfig, MediaContentPart, Modality, ModelMetadata, ModelRegistry,
    ModelSpecificValue, PlaceholderRange, PreProcessorConfig, PreprocessedImages,
    PromptReplacement, TrackedMedia, TrackerOutput, VideoClip,
};
use llm_tokenizer::TokenizerTrait;
use openai_protocol::{
    chat::{ChatMessage, MessageContent},
    common::ContentPart,
    messages::{ImageSource, InputContent, InputContentBlock, InputMessage, Role},
};
use tracing::{debug, info, warn};

use crate::routers::grpc::{
    client::GrpcClient,
    context::WorkerSelection,
    proto_wrapper::{
        tokenspeed_shm_transport_enabled_for_bytes, write_tokenspeed_shm_with,
        SglangMultimodalData, TensorBytes, TokenSpeedModality, TokenSpeedMultimodalData,
        TokenSpeedTensorBytes, TrtllmMultimodalData, VllmMultimodalData,
    },
    MultimodalData,
};

/// Cached model configuration files loaded from the tokenizer directory.
#[derive(Debug, Clone)]
pub(crate) struct MultimodalModelConfig {
    /// Model config.json (HuggingFace format)
    pub config: serde_json::Value,
    /// Preprocessor config (preprocessor_config.json)
    pub preprocessor_config: PreProcessorConfig,
    /// Video-specific preprocessor config, when provided by the model repo.
    pub video_preprocessor_config: Option<PreProcessorConfig>,
}

/// Shared cache of multimodal model configuration files keyed by tokenizer UUID.
///
/// Sources of data:
/// 1. Preloaded from `GetTokenizer` bundles during tokenizer registration.
/// 2. Lazy-loaded from local disk / HF on first multimodal request.
pub struct MultimodalConfigRegistry {
    configs: DashMap<String, Arc<MultimodalModelConfig>>,
}

fn log_mm_timing_enabled() -> bool {
    std::env::var("SMG_LOG_MM_TIMING")
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

impl MultimodalConfigRegistry {
    pub(crate) fn new() -> Self {
        Self {
            configs: DashMap::new(),
        }
    }

    pub(crate) fn get(&self, tokenizer_id: &str) -> Option<Arc<MultimodalModelConfig>> {
        self.configs.get(tokenizer_id).map(|r| r.clone())
    }

    pub(crate) fn insert(&self, tokenizer_id: String, config: Arc<MultimodalModelConfig>) {
        self.configs.insert(tokenizer_id, config);
    }

    /// Drop the cached config for a tokenizer. Called when a tokenizer is
    /// removed so stale entries don't accumulate across re-registrations
    /// (tokenizer IDs are regenerated on each registration via `Uuid::now_v7`).
    pub(crate) fn remove(&self, tokenizer_id: &str) -> Option<Arc<MultimodalModelConfig>> {
        self.configs.remove(tokenizer_id).map(|(_, v)| v)
    }

    /// Return a cached config if present; otherwise load from `tokenizer_source`
    /// (local dir or HF cache/download via `llm_multimodal::hub`), cache under
    /// `tokenizer_id`, and return it.
    pub(crate) async fn get_or_load(
        &self,
        tokenizer_id: &str,
        tokenizer_source: &str,
    ) -> Result<Arc<MultimodalModelConfig>> {
        if let Some(cached) = self.get(tokenizer_id) {
            debug!(%tokenizer_id, "multimodal config cache hit");
            return Ok(cached);
        }

        debug!(
            %tokenizer_id,
            %tokenizer_source,
            "multimodal config cache miss, loading"
        );

        let base_dir = llm_multimodal::hub::resolve_model_config_dir(tokenizer_source)
            .await
            .with_context(|| {
                format!("Failed to resolve model config directory for '{tokenizer_source}'")
            })?;

        let config_path = base_dir.join("config.json");
        let config: serde_json::Value = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config.json at {}", config_path.display()))
            .and_then(|s| {
                serde_json::from_str(&s).with_context(|| {
                    format!("Failed to parse config.json at {}", config_path.display())
                })
            })?;

        // preprocessor_config.json is optional — each image processor supplies
        // its own model-specific defaults, so missing/unparsable files fall
        // back to `PreProcessorConfig::default()`. This matches the bundle
        // preload path in `try_load_multimodal_config`.
        let pp_config_path = base_dir.join("preprocessor_config.json");
        let preprocessor_config =
            load_preprocessor_config_file(&pp_config_path, "preprocessor_config.json")
                .unwrap_or_else(|| {
                    debug!(
                        path = %pp_config_path.display(),
                        "No preprocessor_config.json found; using PreProcessorConfig defaults"
                    );
                    PreProcessorConfig::default()
                });
        let video_preprocessor_config = load_video_preprocessor_config(&base_dir);

        let model_config = Arc::new(MultimodalModelConfig {
            config,
            preprocessor_config,
            video_preprocessor_config,
        });

        self.configs
            .insert(tokenizer_id.to_string(), model_config.clone());

        debug!(%tokenizer_id, "multimodal config loaded and cached");
        Ok(model_config)
    }
}

impl Default for MultimodalConfigRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn load_preprocessor_config_file(
    path: &Path,
    label: &str,
) -> Option<PreProcessorConfig> {
    if !path.exists() {
        return None;
    }

    match std::fs::read_to_string(path) {
        Ok(config_str) => match PreProcessorConfig::from_json(&config_str) {
            Ok(config) => Some(config),
            Err(e) => {
                warn!(
                    path = %path.display(),
                    error = %e,
                    "Failed to parse {label}"
                );
                None
            }
        },
        Err(e) => {
            warn!(
                path = %path.display(),
                error = %e,
                "Failed to read {label}"
            );
            None
        }
    }
}

pub(crate) fn load_video_preprocessor_config(base_dir: &Path) -> Option<PreProcessorConfig> {
    let video_path = base_dir.join("video_preprocessor_config.json");
    if let Some(config) =
        load_preprocessor_config_file(&video_path, "video_preprocessor_config.json")
    {
        return Some(config);
    }

    let processor_path = base_dir.join("processor_config.json");
    if !processor_path.exists() {
        return None;
    }

    match std::fs::read_to_string(&processor_path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.get("video_processor").cloned())
        .and_then(|v| PreProcessorConfig::from_value(v).ok())
    {
        Some(config) => Some(config),
        None => {
            warn!(
                path = %processor_path.display(),
                "Failed to load video_processor from processor_config.json"
            );
            None
        }
    }
}

/// Shared multimodal components injected at router creation time.
pub(crate) struct MultimodalComponents {
    pub media_connector: Arc<MediaConnector>,
    pub image_processor_registry: Arc<ImageProcessorRegistry>,
    pub model_registry: Arc<ModelRegistry>,
    /// Shared reference to the app-level multimodal config cache.
    pub config_registry: Arc<MultimodalConfigRegistry>,
}

impl MultimodalComponents {
    /// Create multimodal components with default registries and a reference
    /// to the shared `MultimodalConfigRegistry` owned by `AppContext`.
    pub fn new(config_registry: Arc<MultimodalConfigRegistry>) -> Result<Self> {
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
            config_registry,
        })
    }
}

/// Output of the multimodal processing pipeline.
pub(crate) struct MultimodalOutput {
    /// Token IDs with placeholder tokens expanded to the correct count per image.
    pub expanded_token_ids: Vec<u32>,
    /// Lightweight intermediate holding preprocessing results.
    /// Assembled into backend-specific `MultimodalData` in request_building.
    pub intermediate: MultimodalIntermediate,
}

/// Lightweight intermediate from the preparation stage.
///
/// Holds all preprocessing results without serializing tensors to bytes.
/// The assembly stage converts this into a backend-specific [`MultimodalData`]
/// variant once the target backend is known (after worker selection).
#[derive(Debug)]
pub(crate) enum MultimodalIntermediate {
    Precomputed(PrecomputedMultimodalIntermediate),
}

#[derive(Debug)]
pub(crate) struct PrecomputedMultimodalIntermediate {
    /// Active modality for this preprocessed payload.
    pub modality: Modality,
    /// Preprocessed pixel values and model-specific tensors (not yet serialized).
    pub preprocessed: PreprocessedImages,
    /// Raw image frames (bytes + blake3 hashes).
    pub images: Vec<Arc<ImageFrame>>,
    /// Raw video clips (bytes + blake3 hashes + sampled frames).
    pub videos: Vec<Arc<VideoClip>>,
    /// Full structural placeholder ranges (offset, length).
    pub placeholders: Vec<PlaceholderRange>,
    /// Patch-only placeholder offsets for sglang.
    pub patch_offsets: Option<Vec<(u32, u32)>>,
    /// Placeholder token ID from model config for the active modality.
    pub placeholder_token_id: Option<u32>,
    /// Per-tensor field layout classification from the model spec.
    pub field_layouts: HashMap<String, FieldLayout>,
    /// Tensor keys that should remain on CPU (vLLM `keep_on_cpu` hint).
    pub keep_on_cpu_keys: Vec<String>,
}

/// Resolve the placeholder token string for a multimodal model.
///
/// Loads the model config (via the shared registry, keyed by `tokenizer_id`)
/// and looks up the model spec to get the placeholder token (e.g.
/// `"<|image|>"` for Phi-3-vision). Returns `None` if the model is not
/// recognized as multimodal.
pub(crate) async fn resolve_placeholder_token(
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    components: &MultimodalComponents,
    tokenizer_id: &str,
    tokenizer_source: &str,
    modality: Modality,
) -> Result<Option<String>> {
    let model_config = components
        .config_registry
        .get_or_load(tokenizer_id, tokenizer_source)
        .await?;
    let metadata = ModelMetadata {
        model_id,
        tokenizer,
        config: &model_config.config,
    };
    let spec = match components.model_registry.lookup(&metadata) {
        Some(s) => s,
        None => return Ok(None),
    };
    Ok(Some(
        spec.placeholder_token_for(&metadata, modality)
            .map_err(|e| anyhow::anyhow!("Failed to get placeholder token: {e}"))?,
    ))
}

/// Return the multimodal modalities present in OpenAI chat messages.
pub(crate) fn chat_modalities(messages: &[ChatMessage]) -> Vec<Modality> {
    let mut modalities = Vec::new();
    let mut push_unique = |modality| {
        if !modalities.contains(&modality) {
            modalities.push(modality);
        }
    };

    for msg in messages {
        let content = match msg {
            ChatMessage::User { content, .. } => Some(content),
            ChatMessage::System { content, .. } => Some(content),
            ChatMessage::Developer { content, .. } => Some(content),
            ChatMessage::Tool { content, .. } => Some(content),
            _ => None,
        };

        if let Some(MessageContent::Parts(parts)) = content {
            for part in parts {
                match part {
                    ContentPart::ImageUrl { .. } => push_unique(Modality::Image),
                    ContentPart::VideoUrl { .. } => push_unique(Modality::Video),
                    ContentPart::Text { .. } => {}
                }
            }
        }
    }

    modalities
}

/// Check if any messages in the request contain multimodal content.
#[cfg(test)]
pub(crate) fn has_multimodal_content(messages: &[ChatMessage]) -> bool {
    !chat_modalities(messages).is_empty()
}

/// Extract multimodal content parts from OpenAI chat messages,
/// converting protocol `ContentPart` to multimodal crate `MediaContentPart`.
fn extract_content_parts(messages: &[ChatMessage]) -> Vec<MediaContentPart> {
    let mut parts = Vec::new();

    for msg in messages {
        let content = match msg {
            ChatMessage::User { content, .. } => Some(content),
            ChatMessage::System { content, .. } => Some(content),
            ChatMessage::Developer { content, .. } => Some(content),
            ChatMessage::Tool { content, .. } => Some(content),
            _ => None,
        };

        if let Some(MessageContent::Parts(message_parts)) = content {
            for part in message_parts {
                match part {
                    ContentPart::ImageUrl { image_url } => {
                        let detail = image_url.detail.as_deref().and_then(parse_detail);
                        parts.push(MediaContentPart::ImageUrl {
                            url: image_url.url.clone(),
                            detail,
                            uuid: None,
                        });
                    }
                    ContentPart::Text { text } => {
                        parts.push(MediaContentPart::Text { text: text.clone() });
                    }
                    ContentPart::VideoUrl { video_url } => {
                        parts.push(MediaContentPart::VideoUrl {
                            url: video_url.url.clone(),
                            uuid: None,
                        });
                    }
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

// ---------------------------------------------------------------------------
// Messages API multimodal detection and extraction
// ---------------------------------------------------------------------------

/// Check if any messages in a Messages API request contain multimodal content.
pub(crate) fn has_multimodal_content_messages(messages: &[InputMessage]) -> bool {
    messages.iter().any(|msg| {
        if msg.role != Role::User {
            return false;
        }
        match &msg.content {
            InputContent::Blocks(blocks) => blocks
                .iter()
                .any(|block| matches!(block, InputContentBlock::Image(_))),
            InputContent::String(_) => false,
        }
    })
}

/// Extract multimodal content parts from Messages API input messages,
/// converting `InputContentBlock::Image` to multimodal crate `MediaContentPart`.
fn extract_content_parts_messages(messages: &[InputMessage]) -> Vec<MediaContentPart> {
    let mut parts = Vec::new();

    for msg in messages {
        if msg.role != Role::User {
            continue;
        }
        let blocks = match &msg.content {
            InputContent::Blocks(blocks) => blocks,
            InputContent::String(_) => continue,
        };

        for block in blocks {
            match block {
                InputContentBlock::Image(image_block) => match &image_block.source {
                    ImageSource::Base64 { media_type, data } => {
                        // Convert base64 to data URL for the media connector
                        let data_url = format!("data:{media_type};base64,{data}");
                        parts.push(MediaContentPart::ImageUrl {
                            url: data_url,
                            detail: None,
                            uuid: None,
                        });
                    }
                    ImageSource::Url { url } => {
                        parts.push(MediaContentPart::ImageUrl {
                            url: url.clone(),
                            detail: None,
                            uuid: None,
                        });
                    }
                },
                InputContentBlock::Text(text_block) => {
                    parts.push(MediaContentPart::Text {
                        text: text_block.text.clone(),
                    });
                }
                _ => {}
            }
        }
    }

    parts
}

/// Process multimodal content from Messages API input messages.
pub(crate) async fn process_multimodal_messages(
    messages: &[InputMessage],
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    token_ids: Vec<u32>,
    components: &MultimodalComponents,
    tokenizer_id: &str,
    tokenizer_source: &str,
) -> Result<MultimodalOutput> {
    let content_parts = extract_content_parts_messages(messages);
    process_multimodal_parts(
        content_parts,
        model_id,
        tokenizer,
        token_ids,
        components,
        tokenizer_id,
        tokenizer_source,
    )
    .await
}

/// Process multimodal content: fetch images, preprocess pixels, expand tokens, collect hashes.
///
/// Single entry point called from preparation.rs. Handles the full pipeline:
pub(crate) async fn process_multimodal(
    messages: &[ChatMessage],
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    token_ids: Vec<u32>,
    components: &MultimodalComponents,
    tokenizer_id: &str,
    tokenizer_source: &str,
) -> Result<MultimodalOutput> {
    let content_parts = extract_content_parts(messages);
    process_multimodal_parts(
        content_parts,
        model_id,
        tokenizer,
        token_ids,
        components,
        tokenizer_id,
        tokenizer_source,
    )
    .await
}

/// Shared multimodal processing core.
///
/// Takes pre-extracted `MediaContentPart`s (from either chat or messages pipeline)
/// and runs the full processing chain: fetch → preprocess → expand → build intermediate.
async fn process_multimodal_parts(
    content_parts: Vec<MediaContentPart>,
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    token_ids: Vec<u32>,
    components: &MultimodalComponents,
    tokenizer_id: &str,
    tokenizer_source: &str,
) -> Result<MultimodalOutput> {
    let log_timing = log_mm_timing_enabled();
    let total_started = Instant::now();
    let media_started = Instant::now();
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

    let videos: Vec<Arc<VideoClip>> = tracker_output
        .data
        .get(&Modality::Video)
        .map(|media_vec| {
            media_vec
                .iter()
                .filter_map(|m| match m {
                    TrackedMedia::Video(clip) => Some(clip.clone()),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default();

    let media_elapsed_ms = media_started.elapsed().as_secs_f64() * 1000.0;
    let modality = match (images.is_empty(), videos.is_empty()) {
        (false, true) => Modality::Image,
        (true, false) => Modality::Video,
        (false, false) => {
            return Err(anyhow::anyhow!(
                "Mixed image and video multimodal requests are not supported yet"
            ));
        }
        (true, true) => {
            return Err(anyhow::anyhow!(
                "No media was successfully fetched for multimodal request"
            ));
        }
    };

    if modality == Modality::Video && videos.len() != 1 {
        return Err(anyhow::anyhow!(
            "Exactly one video is supported per request for the initial video path"
        ));
    }

    match modality {
        Modality::Image => {
            debug!(
                image_count = images.len(),
                image_sizes = ?images.iter().map(|f| (f.image.width(), f.image.height())).collect::<Vec<_>>(),
                "Fetched images for multimodal processing"
            );
        }
        Modality::Video => {
            debug!(
                video_count = videos.len(),
                frame_count = videos.first().map_or(0, |v| v.frames.len()),
                "Fetched video for multimodal processing"
            );
        }
        _ => {}
    }

    // Step 2: Resolve model spec and preprocess media.
    let config_started = Instant::now();
    let model_config = components
        .config_registry
        .get_or_load(tokenizer_id, tokenizer_source)
        .await?;
    let model_type = model_config
        .config
        .get("model_type")
        .and_then(|v| v.as_str());
    let metadata = ModelMetadata {
        model_id,
        tokenizer,
        config: &model_config.config,
    };
    let spec = components
        .model_registry
        .lookup(&metadata)
        .ok_or_else(|| anyhow::anyhow!("Multimodal not supported for model: {model_id}"))?;
    let config_elapsed_ms = config_started.elapsed().as_secs_f64() * 1000.0;

    // Run CPU-intensive image preprocessing on a blocking thread pool so it
    // doesn't block the tokio async runtime under concurrent load.
    // TODO: consider making the thread pool size configurable.
    let pp_config = match modality {
        Modality::Video => model_config
            .video_preprocessor_config
            .clone()
            .unwrap_or_else(|| model_config.preprocessor_config.clone()),
        _ => model_config.preprocessor_config.clone(),
    };
    let registry = components.image_processor_registry.clone();
    let model_id_owned = model_id.to_string();
    let model_type_owned = model_type.map(String::from);
    let images_for_preprocess = images.clone(); // cheap Arc refcount bumps
    let videos_for_preprocess = videos.clone(); // cheap Arc refcount bumps

    let preprocess_started = Instant::now();
    let preprocessed: PreprocessedImages = tokio::task::spawn_blocking(move || {
        let processor = registry
            .find(&model_id_owned, model_type_owned.as_deref())
            .ok_or_else(|| {
                anyhow::anyhow!("No image processor found for model: {model_id_owned}")
            })?;

        match modality {
            Modality::Image => {
                // Extract DynamicImages inside the blocking closure so the expensive
                // clone happens off the tokio async runtime.
                let raw_images: Vec<image::DynamicImage> = images_for_preprocess
                    .iter()
                    .map(|f| f.image.clone())
                    .collect();
                processor
                    .preprocess(&raw_images, &pp_config)
                    .map_err(|e| anyhow::anyhow!("Image preprocessing failed: {e}"))
            }
            Modality::Video => {
                let video = videos_for_preprocess
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("No video available for preprocessing"))?;

                if !video.frames().is_empty() {
                    return processor
                        .preprocess_video(video.frames(), &pp_config)
                        .map_err(|e| anyhow::anyhow!("Video preprocessing failed: {e}"));
                }

                if let Some(rgb_video) = video.rgb_video() {
                    let frame_refs = rgb_video.frame_refs();
                    if frame_refs.len() == rgb_video.frames.len() {
                        if let Ok(preprocessed) =
                            processor.preprocess_video_rgb(&frame_refs, &pp_config)
                        {
                            return Ok(preprocessed);
                        }
                    }
                }

                let frames = video
                    .materialized_frames()
                    .map_err(|e| anyhow::anyhow!("Video frame materialization failed: {e}"))?;
                processor
                    .preprocess_video(&frames, &pp_config)
                    .map_err(|e| anyhow::anyhow!("Video preprocessing failed: {e}"))
            }
            _ => Err(anyhow::anyhow!(
                "Unsupported modality for preprocessing: {modality}"
            )),
        }
    })
    .await
    .map_err(|e| anyhow::anyhow!("Preprocessing task panicked: {e}"))??;
    let preprocess_elapsed_ms = preprocess_started.elapsed().as_secs_f64() * 1000.0;

    debug!(
        ?modality,
        item_count = preprocessed.num_img_tokens.len(),
        total_tokens = preprocessed.num_img_tokens.iter().sum::<usize>(),
        "Multimodal preprocessing complete"
    );

    // Step 3: Compute prompt replacements and expand tokens.
    let expansion_started = Instant::now();
    let prompt_replacements = spec
        .prompt_replacements_for(&metadata, &preprocessed, modality)
        .map_err(|e| anyhow::anyhow!("Failed to compute prompt replacements: {e}"))?;

    // Two token IDs may differ for the same placeholder:
    // - search_token_id: what the tokenizer actually emits (e.g. 200090 for "<|image|>")
    // - placeholder_token_id: what the model config declares (e.g. image_token_id/video_token_id)
    let placeholder_token = spec
        .placeholder_token_for(&metadata, modality)
        .map_err(|e| anyhow::anyhow!("Failed to get placeholder token: {e}"))?;
    let search_token_id = tokenizer.token_to_id(&placeholder_token);
    let placeholder_token_id: Option<u32> = match spec.placeholder_token_id_for(&metadata, modality)
    {
        Ok(id) => Some(id as u32),
        Err(e) => {
            warn!(
                error = %e,
                ?search_token_id,
                "Failed to resolve placeholder_token_id from config, falling back to tokenizer lookup"
            );
            search_token_id
        }
    };

    let expanded = expand_tokens(
        &token_ids,
        search_token_id,
        placeholder_token_id,
        &prompt_replacements,
    );

    debug!(
        original_len = token_ids.len(),
        expanded_len = expanded.token_ids.len(),
        placeholder_count = expanded.placeholders.len(),
        ?search_token_id,
        ?placeholder_token_id,
        "Token expansion complete"
    );
    let expansion_elapsed_ms = expansion_started.elapsed().as_secs_f64() * 1000.0;
    let image_count = images.len();
    let video_count = videos.len();
    let video_frame_count = videos.first().map_or(0, |video| {
        if !video.frames().is_empty() {
            video.frames().len()
        } else {
            video
                .rgb_video()
                .map_or(0, |rgb_video| rgb_video.frames.len())
        }
    });
    let original_tokens = token_ids.len();
    let expanded_tokens = expanded.token_ids.len();

    // Step 4: Build lightweight intermediate (defers tensor serialization to assembly)
    let intermediate = MultimodalIntermediate::Precomputed(PrecomputedMultimodalIntermediate {
        modality,
        preprocessed,
        images,
        videos,
        placeholders: expanded.placeholders,
        patch_offsets: expanded.patch_offsets,
        placeholder_token_id,
        field_layouts: spec.field_layouts(),
        keep_on_cpu_keys: spec.keep_on_cpu_keys(),
    });

    if log_timing {
        info!(
            modality = ?modality,
            image_count,
            video_count,
            video_frame_count,
            media_fetch_decode_ms = media_elapsed_ms,
            config_lookup_ms = config_elapsed_ms,
            preprocess_ms = preprocess_elapsed_ms,
            token_expand_ms = expansion_elapsed_ms,
            total_ms = total_started.elapsed().as_secs_f64() * 1000.0,
            original_tokens,
            expanded_tokens,
            "smg_mm_timing process_multimodal_parts"
        );
    }

    Ok(MultimodalOutput {
        expanded_token_ids: expanded.token_ids,
        intermediate,
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

// ---------------------------------------------------------------------------
// Assembly: convert MultimodalIntermediate → backend-specific MultimodalData
// ---------------------------------------------------------------------------

/// Assemble backend-specific multimodal data from the intermediate.
///
/// Called in request_building after worker selection, when the backend is known.
#[expect(
    clippy::unreachable,
    reason = "MLX multimodal rejected by caller before reaching here"
)]
pub(crate) fn assemble_multimodal_data(
    intermediate: MultimodalIntermediate,
    client: &GrpcClient,
    workers: Option<&WorkerSelection>,
) -> Result<MultimodalData> {
    match intermediate {
        MultimodalIntermediate::Precomputed(precomputed) => match client {
            GrpcClient::Sglang(_) => {
                ensure_image_only(&precomputed, "SGLang")?;
                Ok(MultimodalData::Sglang(assemble_sglang(precomputed)))
            }
            GrpcClient::Vllm(_) => {
                ensure_image_only(&precomputed, "vLLM")?;
                Ok(MultimodalData::Vllm(assemble_vllm(precomputed)))
            }
            GrpcClient::Trtllm(_) => {
                ensure_image_only(&precomputed, "TRT-LLM")?;
                Ok(MultimodalData::Trtllm(assemble_trtllm(precomputed)))
            }
            GrpcClient::TokenSpeed(_) => Ok(MultimodalData::TokenSpeed(assemble_tokenspeed(
                precomputed,
                workers,
            ))),
            GrpcClient::Mlx(_) => unreachable!(
                "caller rejects multimodal for MLX in build_chat_request/build_messages_request"
            ),
        },
    }
}

fn ensure_image_only(
    intermediate: &PrecomputedMultimodalIntermediate,
    backend: &str,
) -> Result<()> {
    if intermediate.modality != Modality::Image {
        return Err(anyhow::anyhow!(
            "{backend} multimodal path currently supports image inputs only; got {}",
            intermediate.modality
        ));
    }
    Ok(())
}

fn assemble_sglang(intermediate: PrecomputedMultimodalIntermediate) -> SglangMultimodalData {
    let (pixel_values, pixel_values_shape) = serialize_pixel_values(&intermediate.preprocessed);
    let model_specific_tensors = serialize_model_specific(intermediate.preprocessed.model_specific);
    let image_data = intermediate
        .images
        .iter()
        .map(|f| f.raw_bytes.to_vec())
        .collect();
    // Use patch-only offsets when available and non-empty; fall back to full structural ranges.
    let mm_placeholders = intermediate
        .patch_offsets
        .filter(|offsets| !offsets.is_empty())
        .unwrap_or_else(|| {
            intermediate
                .placeholders
                .iter()
                .map(|p| (p.offset as u32, p.length as u32))
                .collect()
        });

    SglangMultimodalData {
        image_data,
        pixel_values,
        pixel_values_shape,
        model_specific_tensors,
        im_token_id: intermediate.placeholder_token_id,
        mm_placeholders,
    }
}

fn assemble_vllm(intermediate: PrecomputedMultimodalIntermediate) -> VllmMultimodalData {
    let (pixel_values, pixel_values_shape) = serialize_pixel_values(&intermediate.preprocessed);
    let model_specific_tensors = serialize_model_specific(intermediate.preprocessed.model_specific);
    let mm_hashes = intermediate.images.iter().map(|f| f.hash.clone()).collect();
    let mm_placeholders = intermediate
        .placeholders
        .iter()
        .map(|p| (p.offset as u32, p.length as u32))
        .collect();
    let batched_keys = PreprocessedImages::batched_keys(&intermediate.field_layouts);
    let flat_keys = PreprocessedImages::flat_keys(&intermediate.field_layouts);

    VllmMultimodalData {
        pixel_values,
        pixel_values_shape,
        model_specific_tensors,
        im_token_id: intermediate.placeholder_token_id,
        mm_placeholders,
        mm_hashes,
        batched_keys,
        flat_keys,
        keep_on_cpu_keys: intermediate.keep_on_cpu_keys,
    }
}

fn assemble_trtllm(intermediate: PrecomputedMultimodalIntermediate) -> TrtllmMultimodalData {
    let image_data = intermediate
        .images
        .iter()
        .map(|f| f.raw_bytes.to_vec())
        .collect();
    TrtllmMultimodalData { image_data }
}

fn assemble_tokenspeed(
    intermediate: PrecomputedMultimodalIntermediate,
    workers: Option<&WorkerSelection>,
) -> TokenSpeedMultimodalData {
    let log_timing = log_mm_timing_enabled();
    let total_started = Instant::now();
    // Use patch-only offsets when available and non-empty; fall back to full structural ranges.
    let encoder_input_dtype = tokenspeed_encoder_input_dtype(intermediate.modality, workers);
    let pixel_started = Instant::now();
    let encoder_input = serialize_pixel_values_as_tokenspeed_tensor(
        &intermediate.preprocessed,
        &encoder_input_dtype,
    );
    let pixel_serialize_ms = pixel_started.elapsed().as_secs_f64() * 1000.0;
    let model_specific_started = Instant::now();
    let model_specific_tensors = serialize_model_specific(intermediate.preprocessed.model_specific);
    let model_specific_serialize_ms = model_specific_started.elapsed().as_secs_f64() * 1000.0;
    let mm_placeholders = intermediate
        .patch_offsets
        .filter(|offsets| !offsets.is_empty())
        .unwrap_or_else(|| {
            intermediate
                .placeholders
                .iter()
                .map(|p| (p.offset as u32, p.length as u32))
                .collect()
        });

    let modality = match intermediate.modality {
        Modality::Image => TokenSpeedModality::Image,
        Modality::Video => TokenSpeedModality::Video,
        Modality::Audio => TokenSpeedModality::Audio,
        Modality::ImageEmbeds => TokenSpeedModality::Image,
    };
    let content_hash = match intermediate.modality {
        Modality::Image => hash_hex_strings(intermediate.images.iter().map(|f| f.hash.as_str())),
        Modality::Video => hash_hex_strings(intermediate.videos.iter().map(|v| v.hash.as_str())),
        _ => Vec::new(),
    };

    if log_timing {
        info!(
            modality = ?modality,
            encoder_input_dtype = %encoder_input.dtype,
            encoder_input_bytes = encoder_input.nbytes(),
            encoder_input_shape = ?encoder_input.shape,
            model_specific_tensor_count = model_specific_tensors.len(),
            pixel_serialize_ms,
            model_specific_serialize_ms,
            total_ms = total_started.elapsed().as_secs_f64() * 1000.0,
            "smg_mm_timing assemble_tokenspeed"
        );
    }

    TokenSpeedMultimodalData {
        modality,
        encoder_input,
        model_specific_tensors,
        placeholder_token_id: intermediate.placeholder_token_id,
        mm_placeholders,
        content_hash,
    }
}

fn hash_hex_strings<'a>(hashes: impl Iterator<Item = &'a str>) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new();
    for hash in hashes {
        hasher.update(hash.as_bytes());
    }
    hasher.finalize().as_bytes().to_vec()
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

/// Serialize pixel values ndarray to raw little-endian f32 bytes + shape.
fn serialize_pixel_values(preprocessed: &PreprocessedImages) -> (Vec<u8>, Vec<u32>) {
    let pixel_bytes: Vec<u8> = if let Some(pixel_slice) = preprocessed
        .pixel_values
        .as_slice()
        .or_else(|| preprocessed.pixel_values.as_slice_memory_order())
    {
        // Zero-copy reinterpret: &[f32] → &[u8] on little-endian (x86).
        // This replaces the per-element flat_map(to_le_bytes) which was the
        // #1 CPU hotspot (13% of SMG CPU in profiling).
        #[cfg(target_endian = "little")]
        {
            let byte_slice: &[u8] = bytemuck::cast_slice(pixel_slice);
            byte_slice.to_vec()
        }
        #[cfg(not(target_endian = "little"))]
        {
            pixel_slice.iter().flat_map(|v| v.to_le_bytes()).collect()
        }
    } else {
        // Non-C-contiguous array: .iter() walks in logical (row-major) order,
        // which matches the shape — unlike as_slice_memory_order() which would
        // silently serialize in wrong dimension order for Fortran-contiguous arrays.
        preprocessed
            .pixel_values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    };
    (pixel_bytes, pixel_values_shape(preprocessed))
}

/// Serialize pixel values to the requested wire dtype.
fn serialize_pixel_values_as_tokenspeed_tensor(
    preprocessed: &PreprocessedImages,
    dtype: &str,
) -> TokenSpeedTensorBytes {
    let dtype = match canonical_float_dtype(dtype).as_deref() {
        Some("float32") => "float32".to_string(),
        Some("bfloat16") => "bfloat16".to_string(),
        Some("float16") => "float16".to_string(),
        _ => {
            warn!(
                dtype,
                "Unsupported TokenSpeed encoder input dtype; falling back to float32"
            );
            "float32".to_string()
        }
    };
    let shape = pixel_values_shape(preprocessed);
    let nbytes = preprocessed.pixel_values.len()
        * match dtype.as_str() {
            "float32" => size_of::<f32>(),
            "bfloat16" | "float16" => size_of::<u16>(),
            _ => unreachable!("canonical dtype is constrained above"),
        };

    if tokenspeed_shm_transport_enabled_for_bytes(nbytes) {
        let started = Instant::now();
        match write_tokenspeed_shm_with(nbytes, |file| {
            write_pixel_values_as_dtype(file, preprocessed, &dtype)
        }) {
            Ok(handle) => {
                if log_mm_timing_enabled() {
                    info!(
                        nbytes,
                        elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
                        "smg_mm_timing tokenspeed_shm_write_direct"
                    );
                }
                return TokenSpeedTensorBytes::shm(handle, shape, dtype);
            }
            Err(error) => {
                warn!(
                    ?error,
                    nbytes,
                    dtype = %dtype,
                    "Failed to write TokenSpeed encoder input directly to SHM; falling back to bytes path"
                );
            }
        }
    }

    let (data, shape, dtype) = serialize_pixel_values_as_dtype(preprocessed, &dtype);
    TokenSpeedTensorBytes::bytes(data, shape, dtype)
}

fn write_pixel_values_as_dtype(
    writer: &mut impl Write,
    preprocessed: &PreprocessedImages,
    dtype: &str,
) -> std::io::Result<()> {
    match dtype {
        "float32" => write_pixel_values_as_f32(writer, preprocessed),
        "bfloat16" => write_pixel_values_as_u16(writer, preprocessed, f32_to_bf16_bits),
        "float16" => write_pixel_values_as_u16(writer, preprocessed, f32_to_f16_bits),
        _ => unreachable!("canonical dtype is constrained before direct SHM write"),
    }
}

fn write_pixel_values_as_f32(
    writer: &mut impl Write,
    preprocessed: &PreprocessedImages,
) -> std::io::Result<()> {
    if let Some(pixel_slice) = preprocessed
        .pixel_values
        .as_slice()
        .or_else(|| preprocessed.pixel_values.as_slice_memory_order())
    {
        #[cfg(target_endian = "little")]
        {
            return writer.write_all(bytemuck::cast_slice(pixel_slice));
        }
        #[cfg(not(target_endian = "little"))]
        {
            for value in pixel_slice {
                writer.write_all(&value.to_le_bytes())?;
            }
            return Ok(());
        }
    }

    for value in preprocessed.pixel_values.iter() {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_pixel_values_as_u16<F>(
    writer: &mut impl Write,
    preprocessed: &PreprocessedImages,
    convert: F,
) -> std::io::Result<()>
where
    F: Fn(f32) -> u16 + Copy,
{
    const CHUNK_VALUES: usize = 256 * 1024;

    let mut converted = Vec::with_capacity(CHUNK_VALUES);
    let mut flush = |converted: &mut Vec<u16>| -> std::io::Result<()> {
        if converted.is_empty() {
            return Ok(());
        }
        #[cfg(target_endian = "little")]
        {
            writer.write_all(bytemuck::cast_slice(converted.as_slice()))?;
        }
        #[cfg(not(target_endian = "little"))]
        {
            for value in converted.iter() {
                writer.write_all(&value.to_le_bytes())?;
            }
        }
        converted.clear();
        Ok(())
    };

    if let Some(pixel_slice) = preprocessed.pixel_values.as_slice() {
        for &value in pixel_slice {
            converted.push(convert(value));
            if converted.len() == CHUNK_VALUES {
                flush(&mut converted)?;
            }
        }
    } else {
        for &value in preprocessed.pixel_values.iter() {
            converted.push(convert(value));
            if converted.len() == CHUNK_VALUES {
                flush(&mut converted)?;
            }
        }
    }
    flush(&mut converted)
}

fn serialize_pixel_values_as_dtype(
    preprocessed: &PreprocessedImages,
    dtype: &str,
) -> (Vec<u8>, Vec<u32>, String) {
    match canonical_float_dtype(dtype).as_deref() {
        Some("float32") => {
            let (data, shape) = serialize_pixel_values(preprocessed);
            (data, shape, "float32".to_string())
        }
        Some("bfloat16") => (
            serialize_pixel_values_as_u16_bytes(preprocessed, f32_to_bf16_bits),
            pixel_values_shape(preprocessed),
            "bfloat16".to_string(),
        ),
        Some("float16") => (
            serialize_pixel_values_as_u16_bytes(preprocessed, f32_to_f16_bits),
            pixel_values_shape(preprocessed),
            "float16".to_string(),
        ),
        _ => {
            warn!(
                dtype,
                "Unsupported TokenSpeed encoder input dtype; falling back to float32"
            );
            let (data, shape) = serialize_pixel_values(preprocessed);
            (data, shape, "float32".to_string())
        }
    }
}

fn serialize_pixel_values_as_u16_bytes<F>(preprocessed: &PreprocessedImages, convert: F) -> Vec<u8>
where
    F: Fn(f32) -> u16 + Copy,
{
    let element_count = preprocessed.pixel_values.len();
    let mut converted = Vec::with_capacity(element_count);

    if let Some(pixel_slice) = preprocessed.pixel_values.as_slice() {
        converted.extend(pixel_slice.iter().map(|&value| convert(value)));
    } else {
        converted.extend(
            preprocessed
                .pixel_values
                .iter()
                .map(|&value| convert(value)),
        );
    }

    #[cfg(target_endian = "little")]
    {
        bytemuck::cast_slice(&converted).to_vec()
    }
    #[cfg(not(target_endian = "little"))]
    {
        let mut bytes = Vec::with_capacity(element_count * std::mem::size_of::<u16>());
        for value in converted {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

fn tokenspeed_encoder_input_dtype(modality: Modality, workers: Option<&WorkerSelection>) -> String {
    if let Some(dtype) = tokenspeed_encoder_input_dtype_from_env(modality) {
        return dtype;
    }
    if let Some(dtype) = tokenspeed_encoder_input_dtype_from_worker(workers) {
        return dtype;
    }
    "float32".to_string()
}

fn tokenspeed_encoder_input_dtype_from_env(modality: Modality) -> Option<String> {
    let modality_env = match modality {
        Modality::Image | Modality::ImageEmbeds => "SMG_TOKENSPEED_IMAGE_ENCODER_INPUT_DTYPE",
        Modality::Video => "SMG_TOKENSPEED_VIDEO_ENCODER_INPUT_DTYPE",
        Modality::Audio => "SMG_TOKENSPEED_AUDIO_ENCODER_INPUT_DTYPE",
    };
    std::env::var(modality_env)
        .or_else(|_| std::env::var("SMG_TOKENSPEED_ENCODER_INPUT_DTYPE"))
        .ok()
        .filter(|dtype| !dtype.is_empty())
}

fn tokenspeed_encoder_input_dtype_from_worker(workers: Option<&WorkerSelection>) -> Option<String> {
    let worker = match workers? {
        WorkerSelection::Single { worker } => worker,
        WorkerSelection::Dual { prefill, .. } => prefill,
    };
    worker
        .metadata()
        .spec
        .labels
        .get("multimodal_encoder_dtype")
        .filter(|dtype| !dtype.is_empty())
        .cloned()
}

fn canonical_float_dtype(dtype: &str) -> Option<String> {
    match dtype.trim().to_ascii_lowercase().as_str() {
        "float32" | "fp32" | "f32" => Some("float32".to_string()),
        "bfloat16" | "bf16" => Some("bfloat16".to_string()),
        "float16" | "fp16" | "f16" | "half" => Some("float16".to_string()),
        _ => None,
    }
}

fn pixel_values_shape(preprocessed: &PreprocessedImages) -> Vec<u32> {
    preprocessed
        .pixel_values
        .shape()
        .iter()
        .map(|&d| d as u32)
        .collect()
}

#[inline]
fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fff + lsb;
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

#[inline]
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;

    if exp == 0xff {
        return if mant == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7e00
        };
    }

    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x800000;
        let shift = (14 - half_exp) as u32;
        let mut half_mant = (mantissa >> shift) as u16;
        let round_bit = (mantissa >> (shift - 1)) & 1;
        let sticky = mantissa & ((1u32 << (shift - 1)) - 1);
        if round_bit != 0 && (sticky != 0 || (half_mant & 1) != 0) {
            half_mant += 1;
        }
        return sign | half_mant;
    }

    let mut half = sign | ((half_exp as u16) << 10) | ((mant >> 13) as u16);
    let round = mant & 0x1fff;
    if round > 0x1000 || (round == 0x1000 && (half & 1) != 0) {
        half += 1;
    }
    half
}

/// Serialize model-specific values to TensorBytes, consuming the map to avoid key clones.
fn serialize_model_specific(
    model_specific: HashMap<String, ModelSpecificValue>,
) -> HashMap<String, TensorBytes> {
    model_specific
        .into_iter()
        .filter_map(|(key, value)| match model_specific_to_tensor_bytes(&value) {
            Some(tensor) => Some((key, tensor)),
            None => {
                warn!(tensor_key = %key, "Dropping unsupported model_specific value during multimodal serialization");
                None
            }
        })
        .collect()
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
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use openai_protocol::common::{ImageUrl, VideoUrl};
    use tempfile::TempDir;

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
    fn test_has_multimodal_content_with_video() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![ContentPart::VideoUrl {
                video_url: VideoUrl {
                    url: "https://example.com/clip.mp4".to_string(),
                },
            }]),
            name: None,
        }];

        assert!(has_multimodal_content(&messages));
        assert_eq!(chat_modalities(&messages), vec![Modality::Video]);
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
            MediaContentPart::Text { text } => assert_eq!(text, "Describe this:"),
            _ => panic!("Expected Text part"),
        }

        match &parts[1] {
            MediaContentPart::ImageUrl { url, detail, .. } => {
                assert_eq!(url, "https://example.com/image.jpg");
                assert_eq!(*detail, Some(ImageDetail::High));
            }
            _ => panic!("Expected ImageUrl part"),
        }
    }

    #[test]
    fn test_extract_video_content_parts() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![ContentPart::VideoUrl {
                video_url: VideoUrl {
                    url: "https://example.com/video.mp4".to_string(),
                },
            }]),
            name: None,
        }];

        let parts = extract_content_parts(&messages);
        assert_eq!(parts.len(), 1);
        match &parts[0] {
            MediaContentPart::VideoUrl { url, .. } => {
                assert_eq!(url, "https://example.com/video.mp4");
            }
            _ => panic!("Expected VideoUrl part"),
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

    // ------------------------------------------------------------------
    // MultimodalConfigRegistry tests
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn registry_get_or_load_reads_from_local_dir_and_caches() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("config.json"),
            r#"{"model_type":"phi3_v","image_token_index":32044}"#,
        )
        .unwrap();
        fs::write(
            tmp.path().join("preprocessor_config.json"),
            r#"{"image_processor_type":"Phi3VImageProcessor"}"#,
        )
        .unwrap();
        let source = tmp.path().to_string_lossy().into_owned();

        let reg = MultimodalConfigRegistry::new();
        let first = reg.get_or_load("tok-uuid-2", &source).await.unwrap();
        assert_eq!(first.config["model_type"].as_str(), Some("phi3_v"));

        let second = reg.get_or_load("tok-uuid-2", &source).await.unwrap();
        assert!(
            Arc::ptr_eq(&first, &second),
            "second call must hit cache and return same Arc"
        );
    }

    #[tokio::test]
    async fn registry_get_or_load_falls_back_when_preprocessor_config_missing() {
        // Mirrors the bundle-preload behavior in try_load_multimodal_config:
        // a local dir without preprocessor_config.json must still load and
        // cache an entry using PreProcessorConfig::default().
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        let source = tmp.path().to_string_lossy().into_owned();

        let reg = MultimodalConfigRegistry::new();
        let loaded = reg
            .get_or_load("tok-uuid-nopp", &source)
            .await
            .expect("must fall back to default preprocessor_config");
        assert_eq!(loaded.config["model_type"].as_str(), Some("llama"));
        assert!(reg.get("tok-uuid-nopp").is_some());
    }

    #[tokio::test]
    async fn registry_remove_drops_cached_entry() {
        let reg = MultimodalConfigRegistry::new();
        let cfg = Arc::new(MultimodalModelConfig {
            config: serde_json::json!({"model_type":"phi3_v"}),
            preprocessor_config: PreProcessorConfig::from_json(
                r#"{"image_processor_type":"Phi3VImageProcessor"}"#,
            )
            .unwrap(),
            video_preprocessor_config: None,
        });
        reg.insert("tok-uuid-rm".to_string(), cfg.clone());
        assert!(reg.get("tok-uuid-rm").is_some());

        let removed = reg.remove("tok-uuid-rm").expect("remove returns the entry");
        assert!(Arc::ptr_eq(&removed, &cfg));
        assert!(reg.get("tok-uuid-rm").is_none());
        assert!(reg.remove("tok-uuid-rm").is_none());
    }

    #[tokio::test]
    async fn registry_get_or_load_hits_preloaded_entry_without_touching_source() {
        // Regression test for the IGW bug: preload populates the registry
        // under the tokenizer UUID; `get_or_load` must return it without
        // consulting `tokenizer_source` (which in IGW points to an
        // unreachable worker-only path).
        let reg = MultimodalConfigRegistry::new();
        let cfg = Arc::new(MultimodalModelConfig {
            config: serde_json::json!({"model_type":"phi3_v"}),
            preprocessor_config: PreProcessorConfig::from_json(
                r#"{"image_processor_type":"Phi3VImageProcessor"}"#,
            )
            .unwrap(),
            video_preprocessor_config: None,
        });
        reg.insert("tok-uuid-3".to_string(), cfg.clone());

        let bad_source = "/nonexistent/worker-only/path-that-would-fail";
        let got = reg
            .get_or_load("tok-uuid-3", bad_source)
            .await
            .expect("preloaded entry must be returned without touching source");
        assert!(Arc::ptr_eq(&got, &cfg));
    }
}
