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

use std::{
    collections::HashMap,
    mem::size_of,
    path::Path,
    sync::{Arc, OnceLock},
    time::Instant,
};

use anyhow::{Context, Result};
use dashmap::DashMap;
#[cfg(test)]
use llm_multimodal::EncoderInput;
use llm_multimodal::{
    vision::transforms::preprocess_parallelism, AsyncMultiModalTracker,
    DeferredNormalizedEncoderInput, FieldLayout, ImageDetail, ImageFrame, MediaConnector,
    MediaConnectorConfig, MediaContentPart, Modality, ModalityInput, ModalityPreProcessor,
    ModelMetadata, ModelRegistry, ModelSpecificValue, MultimodalRuntime, OutputPreference,
    PlaceholderRange, PreProcessorConfig, PreprocessRequest, PreprocessedEncoderInputs,
    PromptReplacement, TrackedMedia, TrackerOutput, VideoClip, VideoInput, VisionProcessorRegistry,
};
use llm_tokenizer::TokenizerTrait;
use ndarray::{ArrayD, ArrayViewD, Axis, Slice};
use openai_protocol::{
    chat::{ChatMessage, MessageContent},
    common::ContentPart,
    messages::{ImageSource, InputContent, InputContentBlock, InputMessage, Role},
};
use rayon::prelude::*;
use tracing::{debug, info, warn};

use crate::routers::grpc::{
    client::GrpcClient,
    context::WorkerSelection,
    proto_wrapper::{
        tokenspeed_mm_shm_min_bytes, tokenspeed_mm_tensor_transport_mode,
        tokenspeed_shm_dev_writable, write_tokenspeed_shm_mapped, SglangMultimodalData,
        TensorBytes, TokenSpeedModality, TokenSpeedMultimodalData, TokenSpeedMultimodalItem,
        TokenSpeedTensor, TrtllmMultimodalData, VllmMultimodalData,
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
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SMG_LOG_MM_TIMING")
            .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
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

        // preprocessor_config.json is optional — each vision processor supplies
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

    let processor_config = match std::fs::read_to_string(&processor_path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
    {
        Some(config) => config,
        None => {
            warn!(
                path = %processor_path.display(),
                "Failed to load processor_config.json for video_processor"
            );
            return None;
        }
    };

    let video_processor = processor_config.get("video_processor")?;
    match PreProcessorConfig::from_value(video_processor.clone()) {
        Ok(config) => Some(config),
        Err(error) => {
            warn!(
                path = %processor_path.display(),
                error = %error,
                "Failed to parse video_processor from processor_config.json"
            );
            None
        }
    }
}

/// Shared multimodal components injected at router creation time.
pub(crate) struct MultimodalComponents {
    pub runtime: Arc<MultimodalRuntime>,
    pub media_connector: Arc<MediaConnector>,
    pub vision_processor_registry: Arc<VisionProcessorRegistry>,
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
        let runtime =
            Arc::new(MultimodalRuntime::new().context("Failed to create multimodal runtime")?);
        let media_connector = MediaConnector::new_with_runtime(
            client,
            MediaConnectorConfig::default(),
            runtime.clone(),
        )
        .context("Failed to create MediaConnector")?;

        Ok(Self {
            runtime,
            media_connector: Arc::new(media_connector),
            vision_processor_registry: Arc::new(VisionProcessorRegistry::with_defaults()),
            model_registry: Arc::new(ModelRegistry::default()),
            config_registry,
        })
    }
}

/// Output of the multimodal processing pipeline.
pub(crate) struct MultimodalOutput {
    /// Token IDs with placeholder tokens expanded to the correct count per media item.
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
    /// Preprocessed encoder input and model-specific tensors (not yet serialized).
    pub preprocessed: Arc<PreprocessedEncoderInputs>,
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
    let total_started = log_timing.then(Instant::now);
    let media_started = log_timing.then(Instant::now);
    let mut tracker = AsyncMultiModalTracker::new(components.media_connector.clone());

    for part in content_parts {
        tracker
            .push_part(part)
            .map_err(|e| anyhow::anyhow!("Failed to push content part: {e}"))?;
    }

    let media_future = async move {
        let tracker_output: TrackerOutput = tracker
            .finalize()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to finalize multimodal tracker: {e}"))?;
        Ok::<(TrackerOutput, f64), anyhow::Error>((
            tracker_output,
            media_started
                .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or_default(),
        ))
    };

    let config_future = async {
        let config_started = log_timing.then(Instant::now);
        let model_config = components
            .config_registry
            .get_or_load(tokenizer_id, tokenizer_source)
            .await?;
        Ok::<(Arc<MultimodalModelConfig>, f64), anyhow::Error>((
            model_config,
            config_started
                .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or_default(),
        ))
    };

    let (tracker_result, config_result) = tokio::join!(media_future, config_future);
    let (tracker_output, media_elapsed_ms) = tracker_result?;

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
                item_sizes = ?images.iter().map(|f| (f.image.width(), f.image.height())).collect::<Vec<_>>(),
                "Fetched images for multimodal processing"
            );
        }
        Modality::Video => {
            debug!(
                video_count = videos.len(),
                frame_count = videos.first().map_or(0, |v| v.frame_count()),
                "Fetched video for multimodal processing"
            );
        }
        _ => {}
    }

    // Step 2: Resolve model spec and preprocess media.
    let (model_config, config_elapsed_ms) = config_result?;
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

    // Run CPU-intensive vision preprocessing on a blocking thread pool so it
    // doesn't block the tokio async runtime under concurrent load.
    // TODO: consider making the thread pool size configurable.
    let pp_config = match modality {
        Modality::Video => model_config
            .video_preprocessor_config
            .clone()
            .unwrap_or_else(|| model_config.preprocessor_config.clone()),
        _ => model_config.preprocessor_config.clone(),
    };
    let registry = components.vision_processor_registry.clone();
    let model_id_owned = model_id.to_string();
    let model_type_owned = model_type.map(String::from);

    let preprocess_started = log_timing.then(Instant::now);
    let model_type_for_preprocess = model_type_owned.clone();
    let images_for_preprocess = images.clone(); // cheap Arc refcount bumps
    let videos_for_preprocess = videos.clone(); // cheap Arc refcount bumps
    let runtime = components.runtime.clone();
    let preprocess_task = tokio::task::spawn_blocking(move || {
        runtime.run_cpu(|| {
            preprocess_media(
                &registry,
                &model_id_owned,
                model_type_for_preprocess.as_deref(),
                modality,
                &images_for_preprocess,
                &videos_for_preprocess,
                &pp_config,
            )
        })
    });

    let placeholder_ids_result: Result<(Option<u32>, Option<u32>)> = (|| {
        // These IDs depend only on the model config/tokenizer, so resolve them
        // while CPU vision preprocessing is running on the blocking pool.
        let placeholder_token = spec
            .placeholder_token_for(&metadata, modality)
            .map_err(|e| anyhow::anyhow!("Failed to get placeholder token: {e}"))?;
        let search_token_id = tokenizer.token_to_id(&placeholder_token);
        let placeholder_token_id: Option<u32> = match spec
            .placeholder_token_id_for(&metadata, modality)
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
        Ok((search_token_id, placeholder_token_id))
    })();

    let preprocessed = Arc::new(
        preprocess_task
            .await
            .map_err(|e| anyhow::anyhow!("Preprocessing task panicked: {e}"))
            .and_then(|inner| inner)?,
    );
    let preprocess_elapsed_ms = preprocess_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or_default();

    debug!(
        ?modality,
        item_count = preprocessed.feature_token_counts.len(),
        total_tokens = preprocessed.feature_token_counts.iter().sum::<usize>(),
        "Multimodal preprocessing complete"
    );

    // Step 3: Compute prompt replacements and expand tokens.
    let expansion_started = log_timing.then(Instant::now);
    let prompt_replacements = spec
        .prompt_replacements_for(&metadata, &preprocessed, modality)
        .map_err(|e| anyhow::anyhow!("Failed to compute prompt replacements: {e}"))?;

    // Two token IDs may differ for the same placeholder:
    // - search_token_id: what the tokenizer actually emits (e.g. 200090 for "<|image|>")
    // - placeholder_token_id: what the model config declares (e.g. image_token_id/video_token_id)
    let (search_token_id, placeholder_token_id) = placeholder_ids_result?;

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
    let expansion_elapsed_ms = expansion_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or_default();
    let timing_counts = log_timing.then(|| {
        let video_frame_count = videos.first().map_or(0, |video| video.frame_count());
        (
            images.len(),
            videos.len(),
            video_frame_count,
            token_ids.len(),
            expanded.token_ids.len(),
        )
    });

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

    if let Some((image_count, video_count, video_frame_count, original_tokens, expanded_tokens)) =
        timing_counts
    {
        info!(
            modality = ?modality,
            image_count,
            video_count,
            video_frame_count,
            media_fetch_decode_ms = media_elapsed_ms,
            config_lookup_ms = config_elapsed_ms,
            preprocess_ms = preprocess_elapsed_ms,
            token_expand_ms = expansion_elapsed_ms,
            total_ms = total_started
                .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or_default(),
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

fn preprocess_media(
    registry: &VisionProcessorRegistry,
    model_id: &str,
    model_type: Option<&str>,
    modality: Modality,
    images: &[Arc<ImageFrame>],
    videos: &[Arc<VideoClip>],
    config: &PreProcessorConfig,
) -> Result<PreprocessedEncoderInputs> {
    let processor = registry
        .find(model_id, model_type)
        .ok_or_else(|| anyhow::anyhow!("No vision processor found for model: {model_id}"))?;

    match modality {
        Modality::Image => {
            let raw_images: Vec<&image::DynamicImage> =
                images.iter().map(|frame| &frame.image).collect();
            processor
                .preprocess_input(
                    PreprocessRequest {
                        input: ModalityInput::Images(&raw_images),
                        output: if images.len() == 1 {
                            OutputPreference::CompactAllowed
                        } else {
                            OutputPreference::Materialized
                        },
                    },
                    config,
                )
                .map_err(|error| anyhow::anyhow!("Image preprocessing failed: {error}"))
        }
        Modality::Video => preprocess_video(processor, videos, config),
        _ => Err(anyhow::anyhow!(
            "Unsupported modality for preprocessing: {modality}"
        )),
    }
}

fn preprocess_video(
    processor: &dyn llm_multimodal::VisionPreProcessor,
    videos: &[Arc<VideoClip>],
    config: &PreProcessorConfig,
) -> Result<PreprocessedEncoderInputs> {
    let video = videos
        .first()
        .ok_or_else(|| anyhow::anyhow!("No video available for preprocessing"))?;

    if let Some(stream) = video
        .take_rgb_stream()
        .map_err(|error| anyhow::anyhow!("Video frame stream unavailable: {error}"))?
    {
        return processor
            .preprocess_input(
                PreprocessRequest {
                    input: ModalityInput::Video(VideoInput::RgbStream(stream)),
                    output: OutputPreference::CompactAllowed,
                },
                config,
            )
            .map_err(|error| anyhow::anyhow!("Video stream preprocessing failed: {error}"));
    }

    if !video.frames().is_empty() {
        return processor
            .preprocess_input(
                PreprocessRequest {
                    input: ModalityInput::Video(VideoInput::Frames(video.frames())),
                    output: OutputPreference::Materialized,
                },
                config,
            )
            .map_err(|error| anyhow::anyhow!("Video preprocessing failed: {error}"));
    }

    if let Some(rgb_video) = video.rgb_video() {
        match rgb_video.frame_refs() {
            Ok(frame_refs) => match processor.preprocess_input(
                PreprocessRequest {
                    input: ModalityInput::Video(VideoInput::Rgb(&frame_refs)),
                    output: OutputPreference::CompactAllowed,
                },
                config,
            ) {
                Ok(preprocessed) => return Ok(preprocessed),
                Err(error) => {
                    warn!(
                        error = %error,
                        "RGB video preprocessing fast path failed; falling back to materialized frames"
                    );
                }
            },
            Err(error) => {
                warn!(
                    error = %error,
                    "RGB video frame refs are invalid; falling back to materialized frames"
                );
            }
        }
    }

    let frames = video
        .materialized_frames()
        .map_err(|error| anyhow::anyhow!("Video frame materialization failed: {error}"))?;
    processor
        .preprocess_input(
            PreprocessRequest {
                input: ModalityInput::Video(VideoInput::Frames(&frames)),
                output: OutputPreference::Materialized,
            },
            config,
        )
        .map_err(|error| anyhow::anyhow!("Video preprocessing failed: {error}"))
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

    let replacement_extra_capacity = replacements
        .iter()
        .try_fold(0usize, |acc, repl| {
            acc.checked_add(repl.tokens.len().saturating_sub(1))
        })
        .unwrap_or(0);
    let expanded_capacity = token_ids
        .len()
        .checked_add(replacement_extra_capacity)
        .unwrap_or(token_ids.len());
    let mut expanded = Vec::with_capacity(expanded_capacity);
    let mut placeholders = Vec::with_capacity(replacements.len());
    let mut patch_offsets: Option<Vec<(u32, u32)>> =
        im_token_id.map(|_| Vec::with_capacity(replacements.len()));
    let mut replacement_idx = 0;

    for &token in token_ids {
        if token == placeholder_id && replacement_idx < replacements.len() {
            let repl = &replacements[replacement_idx];
            let offset = expanded.len();
            let repl_len = repl.tokens.len();
            let mut repeated_patch_token: Option<u32> = None;

            // Track patch-only runs while extending
            if let (Some(im_id), Some(ref mut offsets)) = (im_token_id, &mut patch_offsets) {
                if matches!(repl.tokens.first(), Some(&token) if token as u32 == im_id)
                    && repl.tokens.iter().all(|&token| token as u32 == im_id)
                {
                    offsets.push((offset as u32, repl_len as u32));
                    repeated_patch_token = Some(im_id);
                } else {
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
                        offsets.push((s, (offset + repl_len) as u32 - s));
                    }
                }
            }

            // PromptReplacement uses TokenId = i32, convert to u32
            if let Some(token) = repeated_patch_token {
                expanded.resize(expanded.len() + repl_len, token);
            } else {
                expanded.extend(repl.tokens.iter().map(|&t| t as u32));
            }
            placeholders.push(PlaceholderRange {
                offset,
                length: repl_len,
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
    runtime: &MultimodalRuntime,
) -> Result<MultimodalData> {
    runtime.run_cpu(|| match intermediate {
        MultimodalIntermediate::Precomputed(precomputed) => match client {
            GrpcClient::Sglang(_) => {
                ensure_image_only(&precomputed, "SGLang")?;
                Ok(MultimodalData::Sglang(assemble_sglang(
                    materialize_encoder_input(precomputed)?,
                )?))
            }
            GrpcClient::Vllm(_) => {
                ensure_image_only(&precomputed, "vLLM")?;
                Ok(MultimodalData::Vllm(assemble_vllm(
                    materialize_encoder_input(precomputed)?,
                )?))
            }
            GrpcClient::Trtllm(_) => {
                ensure_image_only(&precomputed, "TRT-LLM")?;
                Ok(MultimodalData::Trtllm(assemble_trtllm(precomputed)))
            }
            GrpcClient::TokenSpeed(_) => Ok(MultimodalData::TokenSpeed(assemble_tokenspeed(
                precomputed,
                workers,
            )?)),
            GrpcClient::Mlx(_) => unreachable!(
                "caller rejects multimodal for MLX in build_chat_request/build_messages_request"
            ),
        },
    })
}

fn materialize_encoder_input(
    mut intermediate: PrecomputedMultimodalIntermediate,
) -> Result<PrecomputedMultimodalIntermediate> {
    Arc::make_mut(&mut intermediate.preprocessed)
        .materialize_encoder_input()
        .map_err(|error| anyhow::anyhow!("failed to materialize encoder input: {error}"))?;
    Ok(intermediate)
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

fn assemble_sglang(
    intermediate: PrecomputedMultimodalIntermediate,
) -> Result<SglangMultimodalData> {
    let (pixel_values, pixel_values_shape) = serialize_encoder_input(&intermediate.preprocessed)?;
    let model_specific_tensors =
        serialize_model_specific(&intermediate.preprocessed.model_specific);
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

    Ok(SglangMultimodalData {
        image_data,
        pixel_values,
        pixel_values_shape,
        model_specific_tensors,
        im_token_id: intermediate.placeholder_token_id,
        mm_placeholders,
    })
}

fn assemble_vllm(intermediate: PrecomputedMultimodalIntermediate) -> Result<VllmMultimodalData> {
    let (pixel_values, pixel_values_shape) = serialize_encoder_input(&intermediate.preprocessed)?;
    let model_specific_tensors =
        serialize_model_specific(&intermediate.preprocessed.model_specific);
    let mm_hashes = intermediate.images.iter().map(|f| f.hash.clone()).collect();
    let mm_placeholders = intermediate
        .placeholders
        .iter()
        .map(|p| (p.offset as u32, p.length as u32))
        .collect();
    let batched_keys = PreprocessedEncoderInputs::batched_keys(&intermediate.field_layouts);
    let flat_keys = PreprocessedEncoderInputs::flat_keys(&intermediate.field_layouts);

    Ok(VllmMultimodalData {
        pixel_values,
        pixel_values_shape,
        model_specific_tensors,
        im_token_id: intermediate.placeholder_token_id,
        mm_placeholders,
        mm_hashes,
        batched_keys,
        flat_keys,
        keep_on_cpu_keys: intermediate.keep_on_cpu_keys,
    })
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
    mut intermediate: PrecomputedMultimodalIntermediate,
    workers: Option<&WorkerSelection>,
) -> Result<TokenSpeedMultimodalData> {
    let log_timing = log_mm_timing_enabled();
    let total_started = log_timing.then(Instant::now);
    // Resolve the multimodal tensor transport once per request: `shm` always on,
    // `auto` only when the worker is verified to share /dev/shm (matching
    // namespace token), otherwise inline. See `worker_shares_dev_shm`.
    let shm_enabled = resolve_tokenspeed_shm_enabled(intermediate.modality, workers);
    // Use patch-only offsets when available and non-empty; fall back to full structural ranges.
    let encoder_input_dtype = tokenspeed_encoder_input_dtype(intermediate.modality, workers);
    let encoder_input_dtype = canonical_tokenspeed_encoder_dtype(&encoder_input_dtype);
    if encoder_input_dtype != "bfloat16" && intermediate.preprocessed.encoder_input.is_deferred() {
        Arc::make_mut(&mut intermediate.preprocessed)
            .materialize_encoder_input()
            .map_err(|error| anyhow::anyhow!("failed to materialize encoder input: {error}"))?;
    }
    let patch_offsets = intermediate
        .patch_offsets
        .as_deref()
        .filter(|offsets| !offsets.is_empty())
        .unwrap_or(&[]);

    let modality = match intermediate.modality {
        Modality::Image => TokenSpeedModality::Image,
        Modality::Video => TokenSpeedModality::Video,
        Modality::Audio => TokenSpeedModality::Audio,
        Modality::ImageEmbeds => TokenSpeedModality::Image,
    };

    let item_count = precomputed_multimodal_item_count(&intermediate)?;
    let flat_spans = flat_item_spans(
        &intermediate.preprocessed.model_specific,
        &intermediate.field_layouts,
        item_count,
    )?;
    validate_tokenspeed_item_spans(
        intermediate.preprocessed.as_ref(),
        &intermediate.field_layouts,
        &flat_spans,
        item_count,
    )?;
    let mm_placeholders_by_item = placeholders_for_items(&intermediate.placeholders, patch_offsets);
    anyhow::ensure!(
        mm_placeholders_by_item.len() == item_count,
        "precomputed multimodal assembly placeholder item count mismatch: modality={}, placeholder_item_count={}, item_count={item_count}",
        intermediate.modality,
        mm_placeholders_by_item.len()
    );
    let mut mm_placeholders_by_item = mm_placeholders_by_item.into_iter();
    if let Some(deferred) = intermediate
        .preprocessed
        .encoder_input
        .deferred_normalized()
    {
        anyhow::ensure!(
            item_count == 1,
            "deferred TokenSpeed encoder input currently requires one multimodal item"
        );
        // TODO: Add native typed payload support for vLLM/SGLang before using
        // deferred BF16 outside TokenSpeed; converting BF16 back to FP32 would
        // not preserve their current FP32 contract.
        let model_specific_started = log_timing.then(Instant::now);
        let model_specific_tensors = serialize_model_specific_for_item(
            &intermediate.preprocessed.model_specific,
            &intermediate.field_layouts,
            &flat_spans,
            0,
        )?;
        let model_specific_serialize_ms =
            model_specific_started.map(|started| started.elapsed().as_secs_f64() * 1000.0);
        let mm_placeholders = mm_placeholders_by_item
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing placeholders for multimodal item 0"))?;
        let content_hash = content_hash_for_item(intermediate.modality, &intermediate, 0);
        let encoder_input_started = log_timing.then(Instant::now);
        let encoder_input = serialize_deferred_bf16_tokenspeed_tensor(
            deferred,
            shm_enabled,
            tokenspeed_mm_shm_min_bytes(),
            log_timing,
        )?;
        let encoder_input_serialize_ms =
            encoder_input_started.map(|started| started.elapsed().as_secs_f64() * 1000.0);
        if log_timing {
            info!(
                modality = ?modality,
                item_index = 0,
                encoder_input_dtype = %encoder_input.dtype,
                encoder_input_bytes = encoder_input.nbytes(),
                encoder_input_shape = ?encoder_input.shape,
                model_specific_tensor_count = model_specific_tensors.len(),
                encoder_input_serialize_ms = encoder_input_serialize_ms.unwrap_or_default(),
                model_specific_serialize_ms = model_specific_serialize_ms.unwrap_or_default(),
                "smg_mm_timing assemble_tokenspeed_item"
            );
        }
        if let Some(total_started) = total_started {
            info!(
                modality = ?modality,
                item_count = 1,
                total_ms = total_started.elapsed().as_secs_f64() * 1000.0,
                "smg_mm_timing assemble_tokenspeed"
            );
        }
        return Ok(TokenSpeedMultimodalData {
            items: vec![TokenSpeedMultimodalItem {
                modality,
                encoder_input,
                model_specific_tensors,
                placeholder_token_id: intermediate.placeholder_token_id,
                mm_placeholders,
                content_hash,
            }],
            shm_enabled,
        });
    }

    let mut pending_items: Vec<PendingTokenSpeedItem<'_>> = Vec::with_capacity(item_count);
    for item_index in 0..item_count {
        let item_encoder_input = encoder_input_for_item(
            &intermediate.preprocessed,
            &intermediate.field_layouts,
            &flat_spans,
            item_index,
        )?;
        let model_specific_started = log_timing.then(Instant::now);
        let model_specific_tensors = serialize_model_specific_for_item(
            &intermediate.preprocessed.model_specific,
            &intermediate.field_layouts,
            &flat_spans,
            item_index,
        )?;
        let model_specific_serialize_ms =
            model_specific_started.map(|started| started.elapsed().as_secs_f64() * 1000.0);
        let mm_placeholders = mm_placeholders_by_item.next().ok_or_else(|| {
            anyhow::anyhow!("missing placeholders for multimodal item {item_index}")
        })?;
        let content_hash = content_hash_for_item(intermediate.modality, &intermediate, item_index);

        pending_items.push(PendingTokenSpeedItem {
            encoder_input: item_encoder_input,
            model_specific_tensors,
            mm_placeholders,
            content_hash,
            model_specific_serialize_ms,
        });
    }

    let encoder_input_started = log_timing.then(Instant::now);
    let encoder_inputs = if item_count == 1 {
        let min_shm_bytes = tokenspeed_mm_shm_min_bytes();
        pending_items
            .iter()
            .map(|item| {
                serialize_array_as_tokenspeed_tensor(
                    &item.encoder_input,
                    &encoder_input_dtype,
                    shm_enabled,
                    min_shm_bytes,
                    log_timing,
                )
            })
            .collect()
    } else {
        serialize_arrays_as_tokenspeed_tensors(
            pending_items.iter().map(|item| &item.encoder_input),
            &encoder_input_dtype,
            shm_enabled,
        )
    };
    let encoder_input_serialize_ms =
        encoder_input_started.map(|started| started.elapsed().as_secs_f64() * 1000.0);

    let mut items: Vec<TokenSpeedMultimodalItem> = Vec::with_capacity(item_count);
    for (item_index, (pending, encoder_input)) in
        pending_items.into_iter().zip(encoder_inputs).enumerate()
    {
        if log_timing {
            info!(
                modality = ?modality,
                item_index,
                encoder_input_dtype = %encoder_input.dtype,
                encoder_input_bytes = encoder_input.nbytes(),
                encoder_input_shape = ?encoder_input.shape,
                model_specific_tensor_count = pending.model_specific_tensors.len(),
                encoder_input_serialize_ms = encoder_input_serialize_ms.unwrap_or_default(),
                model_specific_serialize_ms = pending
                    .model_specific_serialize_ms
                    .unwrap_or_default(),
                "smg_mm_timing assemble_tokenspeed_item"
            );
        }

        items.push(TokenSpeedMultimodalItem {
            modality,
            encoder_input,
            model_specific_tensors: pending.model_specific_tensors,
            placeholder_token_id: intermediate.placeholder_token_id,
            mm_placeholders: pending.mm_placeholders,
            content_hash: pending.content_hash,
        });
    }

    if let Some(total_started) = total_started {
        info!(
            modality = ?modality,
            item_count = items.len(),
            total_ms = total_started.elapsed().as_secs_f64() * 1000.0,
            "smg_mm_timing assemble_tokenspeed"
        );
    }

    Ok(TokenSpeedMultimodalData { items, shm_enabled })
}

struct PendingTokenSpeedItem<'a> {
    encoder_input: ArrayViewD<'a, f32>,
    model_specific_tensors: HashMap<String, TensorBytes>,
    mm_placeholders: Vec<(u32, u32)>,
    content_hash: Vec<u8>,
    model_specific_serialize_ms: Option<f64>,
}

type FlatItemSpans = HashMap<String, Vec<(usize, usize)>>;

fn precomputed_multimodal_item_count(
    intermediate: &PrecomputedMultimodalIntermediate,
) -> Result<usize> {
    let media_count = match intermediate.modality {
        Modality::Image | Modality::ImageEmbeds => intermediate.images.len(),
        Modality::Video => intermediate.videos.len(),
        Modality::Audio => 0,
    };
    let token_count = intermediate.preprocessed.feature_token_counts.len();
    let placeholder_count = intermediate.placeholders.len();
    let item_count = token_count.max(media_count).max(placeholder_count);
    anyhow::ensure!(
        item_count > 0,
        "precomputed multimodal assembly requires at least one item"
    );
    if media_count > 0 {
        anyhow::ensure!(
            media_count == item_count,
            "precomputed multimodal assembly media count mismatch: modality={}, media_count={media_count}, item_count={item_count}",
            intermediate.modality
        );
    }
    anyhow::ensure!(
        token_count == item_count,
        "precomputed multimodal assembly token count mismatch: modality={}, token_count={token_count}, item_count={item_count}",
        intermediate.modality
    );
    anyhow::ensure!(
        placeholder_count == item_count,
        "precomputed multimodal assembly placeholder count mismatch: modality={}, placeholder_count={placeholder_count}, item_count={item_count}",
        intermediate.modality
    );
    Ok(item_count)
}

fn flat_item_spans(
    model_specific: &HashMap<String, ModelSpecificValue>,
    field_layouts: &HashMap<String, FieldLayout>,
    item_count: usize,
) -> Result<FlatItemSpans> {
    let mut spans_by_sizes_key = HashMap::new();
    for layout in field_layouts.values() {
        let FieldLayout::Flat { sizes_key } = layout else {
            continue;
        };
        if spans_by_sizes_key.contains_key(sizes_key) {
            continue;
        }

        let sizes_value = model_specific
            .get(sizes_key)
            .ok_or_else(|| anyhow::anyhow!("missing flat sizes tensor {sizes_key}"))?;
        spans_by_sizes_key.insert(
            sizes_key.clone(),
            item_spans_from_model_specific_sizes(sizes_key, sizes_value, item_count)?,
        );
    }
    Ok(spans_by_sizes_key)
}

fn item_spans_from_model_specific_sizes(
    sizes_key: &str,
    value: &ModelSpecificValue,
    item_count: usize,
) -> Result<Vec<(usize, usize)>> {
    let sizes_len = match value {
        ModelSpecificValue::IntTensor { data, .. } => data.len(),
        ModelSpecificValue::UintTensor { data, .. } => data.len(),
        ModelSpecificValue::IntVec(values) => values.len(),
        ModelSpecificValue::UintVec(values) => values.len(),
        _ => anyhow::bail!("unsupported flat sizes value type"),
    };
    anyhow::ensure!(
        sizes_len == item_count,
        "flat sizes tensor {sizes_key} length mismatch: sizes_len={sizes_len}, item_count={item_count}",
    );

    let mut spans = Vec::with_capacity(item_count);
    let mut start = 0usize;

    match value {
        ModelSpecificValue::IntTensor { data, .. } => {
            for &len in data {
                push_item_span_from_i64(&mut spans, &mut start, len)?;
            }
        }
        ModelSpecificValue::UintTensor { data, .. } => {
            for &len in data {
                push_item_span(&mut spans, &mut start, len as usize)?;
            }
        }
        ModelSpecificValue::IntVec(values) => {
            for &len in values {
                push_item_span_from_i64(&mut spans, &mut start, len)?;
            }
        }
        ModelSpecificValue::UintVec(values) => {
            for &len in values {
                push_item_span(&mut spans, &mut start, len as usize)?;
            }
        }
        _ => anyhow::bail!("unsupported flat sizes value type"),
    }
    Ok(spans)
}

fn push_item_span_from_i64(
    spans: &mut Vec<(usize, usize)>,
    start: &mut usize,
    len: i64,
) -> Result<()> {
    let len = usize::try_from(len).context("negative flat size")?;
    push_item_span(spans, start, len)
}

fn push_item_span(spans: &mut Vec<(usize, usize)>, start: &mut usize, len: usize) -> Result<()> {
    spans.push((*start, len));
    *start = (*start)
        .checked_add(len)
        .ok_or_else(|| anyhow::anyhow!("flat size offset overflow"))?;
    Ok(())
}

fn validate_tokenspeed_item_spans(
    preprocessed: &PreprocessedEncoderInputs,
    field_layouts: &HashMap<String, FieldLayout>,
    flat_spans: &FlatItemSpans,
    item_count: usize,
) -> Result<()> {
    let encoder_shape = preprocessed.encoder_input_shape();
    let encoder_first_dim = *encoder_shape
        .first()
        .ok_or_else(|| anyhow::anyhow!("encoder_input tensor must have a first dimension"))?;
    let encoder_layout = field_layouts
        .get("pixel_values")
        .unwrap_or(&FieldLayout::Batched);
    validate_tokenspeed_layout_first_dim(
        "pixel_values",
        encoder_layout,
        encoder_first_dim,
        flat_spans,
        item_count,
    )?;

    for (key, value) in &preprocessed.model_specific {
        let Some(layout) = field_layouts.get(key) else {
            continue;
        };
        let first_dim = model_specific_first_dim(key, value)?;
        validate_tokenspeed_layout_first_dim(key, layout, first_dim, flat_spans, item_count)?;
    }

    Ok(())
}

fn validate_tokenspeed_layout_first_dim(
    tensor_key: &str,
    layout: &FieldLayout,
    first_dim: usize,
    flat_spans: &FlatItemSpans,
    item_count: usize,
) -> Result<()> {
    match layout {
        FieldLayout::Batched => {
            anyhow::ensure!(
                first_dim == item_count,
                "batched tensor {tensor_key} first dimension mismatch: first_dim={first_dim}, item_count={item_count}"
            );
        }
        FieldLayout::Flat { sizes_key } => {
            let spans = flat_spans.get(sizes_key).ok_or_else(|| {
                anyhow::anyhow!("missing flat spans for sizes tensor {sizes_key}")
            })?;
            let span_total = spans.iter().try_fold(0usize, |acc, (_, len)| {
                acc.checked_add(*len)
                    .ok_or_else(|| anyhow::anyhow!("flat span total overflow for {tensor_key}"))
            })?;
            anyhow::ensure!(
                span_total == first_dim,
                "flat tensor {tensor_key} first dimension mismatch: span_total={span_total}, first_dim={first_dim}, sizes_key={sizes_key}"
            );
        }
    }
    Ok(())
}

fn model_specific_first_dim(key: &str, value: &ModelSpecificValue) -> Result<usize> {
    match value {
        ModelSpecificValue::Tensor { shape, .. }
        | ModelSpecificValue::IntTensor { shape, .. }
        | ModelSpecificValue::UintTensor { shape, .. } => shape.first().copied().ok_or_else(|| {
            anyhow::anyhow!("model_specific tensor {key} must have a first dimension")
        }),
        ModelSpecificValue::IntVec(values) => Ok(values.len()),
        ModelSpecificValue::UintVec(values) => Ok(values.len()),
        ModelSpecificValue::FloatVec(values) => Ok(values.len()),
        ModelSpecificValue::TupleVec(values) => Ok(values.len()),
        ModelSpecificValue::Int(_) | ModelSpecificValue::Float(_) | ModelSpecificValue::Bool(_) => {
            anyhow::bail!("model_specific value {key} has no first dimension")
        }
    }
}

fn flat_item_span(
    flat_spans: &FlatItemSpans,
    sizes_key: &str,
    item_index: usize,
) -> Result<(usize, usize)> {
    flat_spans
        .get(sizes_key)
        .and_then(|spans| spans.get(item_index))
        .copied()
        .ok_or_else(|| {
            anyhow::anyhow!("missing flat span for sizes tensor {sizes_key} item {item_index}")
        })
}

fn encoder_input_for_item<'a>(
    preprocessed: &'a PreprocessedEncoderInputs,
    field_layouts: &HashMap<String, FieldLayout>,
    flat_spans: &FlatItemSpans,
    item_index: usize,
) -> Result<ArrayViewD<'a, f32>> {
    // The field layout key remains "pixel_values" because it is the established
    // model vision input name. Internally this tensor is the modality encoder
    // input we pass to TokenSpeed.
    let layout = field_layouts
        .get("pixel_values")
        .unwrap_or(&FieldLayout::Batched);
    let encoder_input = preprocessed
        .encoder_input
        .dense()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    match layout {
        FieldLayout::Batched => slice_array_axis0(encoder_input, item_index, 1),
        FieldLayout::Flat { sizes_key } => {
            let (start, len) = flat_item_span(flat_spans, sizes_key, item_index)?;
            slice_array_axis0(encoder_input, start, len)
        }
    }
}

fn serialize_model_specific_for_item(
    model_specific: &HashMap<String, ModelSpecificValue>,
    field_layouts: &HashMap<String, FieldLayout>,
    flat_spans: &FlatItemSpans,
    item_index: usize,
) -> Result<HashMap<String, TensorBytes>> {
    let mut serialized = HashMap::with_capacity(model_specific.len());
    for (key, value) in model_specific {
        let tensor = match field_layouts.get(key) {
            Some(FieldLayout::Batched) => {
                let item_value = value
                    .slice_first_dim(item_index, 1)
                    .with_context(|| format!("failed to slice model_specific tensor {key}"))?;
                model_specific_to_tensor_bytes(&item_value)
            }
            Some(FieldLayout::Flat { sizes_key }) => {
                let (start, len) = flat_item_span(flat_spans, sizes_key, item_index)?;
                let item_value = value
                    .slice_first_dim(start, len)
                    .with_context(|| format!("failed to slice flat model_specific tensor {key}"))?;
                model_specific_to_tensor_bytes(&item_value)
            }
            None => model_specific_to_tensor_bytes(value),
        };
        if let Some(tensor) = tensor {
            serialized.insert(key.clone(), tensor);
        } else {
            warn!(tensor_key = %key, "Dropping unsupported model_specific value during multimodal serialization");
        }
    }
    Ok(serialized)
}

fn placeholders_for_items(
    placeholders: &[PlaceholderRange],
    patch_offsets: &[(u32, u32)],
) -> Vec<Vec<(u32, u32)>> {
    if placeholders.len() == 1 {
        return vec![placeholders_for_item(&placeholders[0], patch_offsets)];
    }

    if patch_offsets.is_empty() {
        return placeholders
            .iter()
            .map(|placeholder| vec![full_placeholder_range(placeholder)])
            .collect();
    }

    if patch_offsets.len() == placeholders.len() {
        let mut by_item = Vec::with_capacity(placeholders.len());
        let mut one_patch_run_per_item = true;
        for (placeholder, &(offset, length)) in placeholders.iter().zip(patch_offsets) {
            let start = placeholder.offset as u32;
            let end = start + placeholder.length as u32;
            if offset < start || offset.saturating_add(length) > end {
                one_patch_run_per_item = false;
                break;
            }
            by_item.push(vec![(offset, length)]);
        }
        if one_patch_run_per_item {
            return by_item;
        }
    }

    if !placeholder_ranges_sorted(placeholders) || !patch_offsets_sorted(patch_offsets) {
        return placeholders
            .iter()
            .map(|placeholder| placeholders_for_item(placeholder, patch_offsets))
            .collect();
    }

    let mut by_item = Vec::with_capacity(placeholders.len());
    let mut patch_idx = 0usize;
    for placeholder in placeholders {
        let start = placeholder.offset as u32;
        let end = start + placeholder.length as u32;
        while patch_idx < patch_offsets.len() && patch_offsets[patch_idx].0 < start {
            patch_idx += 1;
        }

        let mut item_patch_offsets = Vec::new();
        let mut scan_idx = patch_idx;
        while scan_idx < patch_offsets.len() {
            let (offset, length) = patch_offsets[scan_idx];
            if offset >= end {
                break;
            }
            if offset >= start && offset.saturating_add(length) <= end {
                item_patch_offsets.push((offset, length));
            }
            scan_idx += 1;
        }
        patch_idx = scan_idx;

        if item_patch_offsets.is_empty() {
            by_item.push(vec![(start, end - start)]);
        } else {
            by_item.push(item_patch_offsets);
        }
    }
    by_item
}

fn placeholders_for_item(
    placeholder: &PlaceholderRange,
    patch_offsets: &[(u32, u32)],
) -> Vec<(u32, u32)> {
    let start = placeholder.offset as u32;
    let end = start + placeholder.length as u32;
    if patch_offsets.is_empty() {
        return vec![(start, end - start)];
    }
    if patch_offsets.len() == 1 {
        let (offset, length) = patch_offsets[0];
        return if offset >= start && offset.saturating_add(length) <= end {
            vec![(offset, length)]
        } else {
            vec![(start, end - start)]
        };
    }

    let item_patch_offsets = patch_offsets
        .iter()
        .copied()
        .filter(|(offset, length)| *offset >= start && offset.saturating_add(*length) <= end)
        .collect::<Vec<_>>();
    if item_patch_offsets.is_empty() {
        vec![(start, end - start)]
    } else {
        item_patch_offsets
    }
}

fn full_placeholder_range(placeholder: &PlaceholderRange) -> (u32, u32) {
    let start = placeholder.offset as u32;
    (start, placeholder.length as u32)
}

fn placeholder_ranges_sorted(placeholders: &[PlaceholderRange]) -> bool {
    placeholders
        .windows(2)
        .all(|window| window[0].offset <= window[1].offset)
}

fn patch_offsets_sorted(patch_offsets: &[(u32, u32)]) -> bool {
    patch_offsets
        .windows(2)
        .all(|window| window[0].0 <= window[1].0)
}

fn content_hash_for_item(
    modality: Modality,
    intermediate: &PrecomputedMultimodalIntermediate,
    item_index: usize,
) -> Vec<u8> {
    match modality {
        Modality::Image | Modality::ImageEmbeds => intermediate
            .images
            .get(item_index)
            .map(|image| hash_hex_strings(std::iter::once(image.hash.as_str())))
            .unwrap_or_default(),
        Modality::Video => intermediate
            .videos
            .get(item_index)
            .map(|video| hash_hex_strings(std::iter::once(video.hash.as_str())))
            .unwrap_or_default(),
        Modality::Audio => Vec::new(),
    }
}

fn slice_array_axis0(array: &ArrayD<f32>, start: usize, len: usize) -> Result<ArrayViewD<'_, f32>> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| anyhow::anyhow!("array slice range overflow"))?;
    let rows = array.shape().first().copied().unwrap_or(0);
    anyhow::ensure!(
        end <= rows,
        "array first-dimension slice {start}..{end} exceeds {rows}"
    );
    Ok(array.slice_axis(Axis(0), Slice::from(start..end)))
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

fn serialize_deferred_bf16_tokenspeed_tensor(
    encoder_input: &DeferredNormalizedEncoderInput,
    shm_enabled: bool,
    min_shm_bytes: usize,
    log_timing: bool,
) -> Result<TokenSpeedTensor> {
    let nbytes = encoder_input
        .len()
        .checked_mul(size_of::<u16>())
        .ok_or_else(|| anyhow::anyhow!("deferred BF16 encoder input size overflow"))?;
    let shape = encoder_input
        .shape()
        .iter()
        .map(|&dimension| {
            u32::try_from(dimension)
                .map_err(|_| anyhow::anyhow!("encoder input dimension exceeds u32"))
        })
        .collect::<Result<Vec<_>>>()?;

    if shm_enabled && nbytes >= min_shm_bytes {
        let timing_started = log_timing.then(Instant::now);
        let handle = write_tokenspeed_shm_mapped(nbytes, |output| {
            encoder_input
                .fill_bf16_le_bytes(output)
                .map_err(|error| std::io::Error::other(error.to_string()))
        })?;
        if log_timing {
            info!(
                nbytes,
                elapsed_ms = timing_started
                    .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or_default(),
                "smg_mm_timing tokenspeed_shm_write_deferred_bf16"
            );
        }
        return Ok(TokenSpeedTensor::shm(handle, shape, "bfloat16".to_string()));
    }

    let mut data = vec![0; nbytes];
    encoder_input
        .fill_bf16_le_bytes(&mut data)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(TokenSpeedTensor::inline(
        data,
        shape,
        "bfloat16".to_string(),
    ))
}

/// Serialize the primary encoder input ndarray to raw little-endian f32 bytes + shape.
fn serialize_encoder_input(
    preprocessed: &PreprocessedEncoderInputs,
) -> Result<(Vec<u8>, Vec<u32>)> {
    let encoder_input = preprocessed
        .encoder_input
        .dense()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(serialize_array(encoder_input))
}

fn serialize_array(encoder_input: &ArrayD<f32>) -> (Vec<u8>, Vec<u32>) {
    let encoder_bytes: Vec<u8> = if let Some(encoder_slice) = encoder_input
        // Fast path only for C-contiguous arrays, whose memory order equals
        // logical (row-major) order. A non-C-contiguous array (e.g. a
        // Fortran-contiguous view) falls through to logical `.iter()` below;
        // `as_slice_memory_order()` is deliberately NOT used as a fallback
        // because it would serialize such arrays in the wrong dimension order.
        .as_slice()
    {
        // Zero-copy reinterpret: &[f32] → &[u8] on little-endian (x86).
        // This replaces the per-element flat_map(to_le_bytes) which was the
        // #1 CPU hotspot (13% of SMG CPU in profiling).
        #[cfg(target_endian = "little")]
        {
            let byte_slice: &[u8] = bytemuck::cast_slice(encoder_slice);
            byte_slice.to_vec()
        }
        #[cfg(not(target_endian = "little"))]
        {
            f32_values_to_le_bytes(encoder_slice.iter().copied(), encoder_slice.len())
        }
    } else {
        // Non-C-contiguous array: `.iter()` walks in logical (row-major) order,
        // which matches the shape.
        f32_values_to_le_bytes(encoder_input.iter().copied(), encoder_input.len())
    };
    (encoder_bytes, array_shape(encoder_input))
}

/// Serialize encoder input to the requested wire dtype.
fn serialize_arrays_as_tokenspeed_tensors<'view, 'item>(
    encoder_inputs: impl ExactSizeIterator<Item = &'item ArrayViewD<'view, f32>>,
    dtype: &str,
    shm_enabled: bool,
) -> Vec<TokenSpeedTensor>
where
    'view: 'item,
{
    let min_shm_bytes = tokenspeed_mm_shm_min_bytes();
    let log_timing = log_mm_timing_enabled();
    let item_count = encoder_inputs.len();
    if shm_enabled && item_count >= 2 {
        let encoder_inputs = encoder_inputs.collect::<Vec<_>>();
        if let Some(tensors) = serialize_arrays_as_packed_tokenspeed_shm(
            &encoder_inputs,
            dtype,
            min_shm_bytes,
            log_timing,
        ) {
            return tensors;
        }
        return encoder_inputs
            .iter()
            .map(|&encoder_input| {
                serialize_array_as_tokenspeed_tensor(
                    encoder_input,
                    dtype,
                    shm_enabled,
                    min_shm_bytes,
                    log_timing,
                )
            })
            .collect();
    }

    encoder_inputs
        .map(|encoder_input| {
            serialize_array_as_tokenspeed_tensor(
                encoder_input,
                dtype,
                shm_enabled,
                min_shm_bytes,
                log_timing,
            )
        })
        .collect()
}

fn serialize_arrays_as_packed_tokenspeed_shm(
    encoder_inputs: &[&ArrayViewD<'_, f32>],
    dtype: &str,
    min_bytes: usize,
    log_timing: bool,
) -> Option<Vec<TokenSpeedTensor>> {
    if encoder_inputs.len() < 2 {
        return None;
    }

    let dtype = canonical_tokenspeed_encoder_dtype(dtype);
    let mut offsets = Vec::with_capacity(encoder_inputs.len());
    let mut nbytes_by_item = Vec::with_capacity(encoder_inputs.len());
    let mut shapes = Vec::with_capacity(encoder_inputs.len());
    let mut total_nbytes = 0usize;
    for &encoder_input in encoder_inputs {
        let nbytes = tokenspeed_encoder_input_nbytes(encoder_input, &dtype)?;
        offsets.push(total_nbytes);
        nbytes_by_item.push(nbytes);
        shapes.push(array_view_shape(encoder_input));
        total_nbytes = total_nbytes.checked_add(nbytes)?;
    }
    if total_nbytes < min_bytes {
        return None;
    }

    let timing_started = log_timing.then(Instant::now);
    match write_tokenspeed_shm_mapped(total_nbytes, |output| {
        for ((&encoder_input, &offset), &nbytes) in
            encoder_inputs.iter().zip(&offsets).zip(&nbytes_by_item)
        {
            fill_array_as_dtype(&mut output[offset..offset + nbytes], encoder_input, &dtype)?;
        }
        Ok(())
    }) {
        Ok(base_handle) => {
            if log_timing {
                info!(
                    item_count = encoder_inputs.len(),
                    nbytes = total_nbytes,
                    elapsed_ms = timing_started
                        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or_default(),
                    "smg_mm_timing tokenspeed_shm_write_packed"
                );
            }

            Some(
                shapes
                    .into_iter()
                    .zip(offsets)
                    .zip(nbytes_by_item)
                    .map(|((shape, offset), nbytes)| {
                        let mut handle = base_handle.clone();
                        handle.offset = offset as u64;
                        handle.nbytes = nbytes as u64;
                        TokenSpeedTensor::shm(handle, shape, dtype.clone())
                    })
                    .collect(),
            )
        }
        Err(error) => {
            use crate::observability::metrics::Metrics;
            warn!(
                ?error,
                item_count = encoder_inputs.len(),
                nbytes = total_nbytes,
                dtype = %dtype,
                "Failed to write packed TokenSpeed encoder inputs to SHM; falling back to per-item transport"
            );
            Metrics::record_mm_shm_write_failure("tokenspeed");
            None
        }
    }
}

fn serialize_array_as_tokenspeed_tensor(
    encoder_input: &ArrayViewD<'_, f32>,
    dtype: &str,
    shm_enabled: bool,
    min_shm_bytes: usize,
    log_timing: bool,
) -> TokenSpeedTensor {
    let dtype = canonical_tokenspeed_encoder_dtype(dtype);
    let shape = array_view_shape(encoder_input);
    let Some(nbytes) = tokenspeed_encoder_input_nbytes(encoder_input, &dtype) else {
        warn!(
            dtype = %dtype,
            shape = ?shape,
            "TokenSpeed encoder input byte length overflow; falling back to inline serialization"
        );
        let (data, shape, dtype) = serialize_array_view_as_dtype(encoder_input, &dtype);
        return TokenSpeedTensor::inline(data, shape, dtype);
    };

    if shm_enabled && nbytes >= min_shm_bytes {
        let timing_started = log_timing.then(Instant::now);
        match write_tokenspeed_shm_mapped(nbytes, |output| {
            fill_array_as_dtype(output, encoder_input, &dtype)
        }) {
            Ok(handle) => {
                if log_timing {
                    info!(
                        nbytes,
                        elapsed_ms = timing_started
                            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                            .unwrap_or_default(),
                        "smg_mm_timing tokenspeed_shm_write_direct"
                    );
                }
                return TokenSpeedTensor::shm(handle, shape, dtype);
            }
            Err(error) => {
                use crate::observability::metrics::Metrics;
                warn!(
                    ?error,
                    nbytes,
                    dtype = %dtype,
                    "Failed to write TokenSpeed encoder input directly to SHM; falling back to bytes path"
                );
                Metrics::record_mm_shm_write_failure("tokenspeed");
            }
        }
    }

    let (data, shape, dtype) = serialize_array_view_as_dtype(encoder_input, &dtype);
    TokenSpeedTensor::inline(data, shape, dtype)
}

fn canonical_tokenspeed_encoder_dtype(dtype: &str) -> String {
    match canonical_float_dtype(dtype).as_deref() {
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
    }
}

fn tokenspeed_encoder_input_nbytes(
    encoder_input: &ArrayViewD<'_, f32>,
    dtype: &str,
) -> Option<usize> {
    encoder_input
        .len()
        .checked_mul(tokenspeed_encoder_input_element_size(dtype))
}

fn tokenspeed_encoder_input_element_size(dtype: &str) -> usize {
    if dtype == "bfloat16" || dtype == "float16" {
        size_of::<u16>()
    } else {
        size_of::<f32>()
    }
}

fn fill_array_as_dtype(
    output: &mut [u8],
    encoder_input: &ArrayViewD<'_, f32>,
    dtype: &str,
) -> std::io::Result<()> {
    let expected = tokenspeed_encoder_input_nbytes(encoder_input, dtype).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "TokenSpeed encoder input byte length overflow",
        )
    })?;
    if output.len() != expected {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "TokenSpeed encoder output has an unexpected byte length",
        ));
    }

    match dtype {
        "float32" => {
            fill_array_as_f32_bytes(output, encoder_input);
            Ok(())
        }
        "bfloat16" => {
            fill_array_as_u16_bytes(output, encoder_input, f32_to_bf16_bits);
            Ok(())
        }
        "float16" => {
            fill_array_as_u16_bytes(output, encoder_input, f32_to_f16_bits);
            Ok(())
        }
        other => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("unsupported TokenSpeed encoder input dtype: {other}"),
        )),
    }
}

fn fill_array_as_f32_bytes(output: &mut [u8], encoder_input: &ArrayViewD<'_, f32>) {
    if let Some(encoder_slice) = encoder_input.as_slice() {
        #[cfg(target_endian = "little")]
        output.copy_from_slice(bytemuck::cast_slice(encoder_slice));
        #[cfg(not(target_endian = "little"))]
        fill_f32_values_as_f32_bytes(output, encoder_slice.iter().copied());
    } else {
        fill_f32_values_as_f32_bytes(output, encoder_input.iter().copied());
    }
}

fn fill_f32_values_as_f32_bytes<I>(output: &mut [u8], values: I)
where
    I: IntoIterator<Item = f32>,
{
    for (output, value) in output.chunks_exact_mut(size_of::<f32>()).zip(values) {
        output.copy_from_slice(&value.to_le_bytes());
    }
}

fn fill_array_as_u16_bytes<F>(output: &mut [u8], encoder_input: &ArrayViewD<'_, f32>, convert: F)
where
    F: Fn(f32) -> u16 + Copy + Send + Sync,
{
    if let Some(encoder_slice) = encoder_input.as_slice() {
        fill_f32_slice_as_u16_bytes(output, encoder_slice, convert);
    } else {
        fill_f32_values_as_u16_bytes(output, encoder_input.iter().copied(), convert);
    }
}

fn serialize_array_view_as_dtype(
    encoder_input: &ArrayViewD<'_, f32>,
    dtype: &str,
) -> (Vec<u8>, Vec<u32>, String) {
    match canonical_float_dtype(dtype).as_deref() {
        Some("float32") => {
            let data = serialize_array_view_f32_bytes(encoder_input);
            (data, array_view_shape(encoder_input), "float32".to_string())
        }
        Some("bfloat16") => (
            serialize_array_view_as_u16_bytes(encoder_input, f32_to_bf16_bits),
            array_view_shape(encoder_input),
            "bfloat16".to_string(),
        ),
        Some("float16") => (
            serialize_array_view_as_u16_bytes(encoder_input, f32_to_f16_bits),
            array_view_shape(encoder_input),
            "float16".to_string(),
        ),
        _ => {
            warn!(
                dtype,
                "Unsupported TokenSpeed encoder input dtype; falling back to float32"
            );
            let data = serialize_array_view_f32_bytes(encoder_input);
            (data, array_view_shape(encoder_input), "float32".to_string())
        }
    }
}

fn serialize_array_view_f32_bytes(encoder_input: &ArrayViewD<'_, f32>) -> Vec<u8> {
    if let Some(encoder_slice) = encoder_input
        // Fast path only for C-contiguous views, whose memory order equals
        // logical (row-major) order. Non-C-contiguous views fall through to
        // logical `.iter()` below, preserving the wire order.
        .as_slice()
    {
        #[cfg(target_endian = "little")]
        {
            return bytemuck::cast_slice(encoder_slice).to_vec();
        }
        #[cfg(not(target_endian = "little"))]
        {
            return f32_values_to_le_bytes(encoder_slice.iter().copied(), encoder_slice.len());
        }
    }

    f32_values_to_le_bytes(encoder_input.iter().copied(), encoder_input.len())
}

fn f32_values_to_le_bytes<I>(values: I, len: usize) -> Vec<u8>
where
    I: IntoIterator<Item = f32>,
{
    let mut bytes = Vec::with_capacity(len * size_of::<f32>());
    extend_f32_le_bytes(&mut bytes, values);
    bytes
}

fn extend_f32_le_bytes<I>(bytes: &mut Vec<u8>, values: I)
where
    I: IntoIterator<Item = f32>,
{
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn serialize_array_view_as_u16_bytes<F>(encoder_input: &ArrayViewD<'_, f32>, convert: F) -> Vec<u8>
where
    F: Fn(f32) -> u16 + Copy + Send + Sync,
{
    let element_count = encoder_input.len();
    let mut bytes = vec![0u8; element_count * size_of::<u16>()];

    if let Some(encoder_slice) = encoder_input.as_slice() {
        fill_f32_slice_as_u16_bytes(&mut bytes, encoder_slice, convert);
    } else {
        fill_f32_values_as_u16_bytes(&mut bytes, encoder_input.iter().copied(), convert);
    }
    bytes
}

fn fill_f32_slice_as_u16_bytes<F>(bytes: &mut [u8], values: &[f32], convert: F)
where
    F: Fn(f32) -> u16 + Copy + Send + Sync,
{
    debug_assert_eq!(bytes.len(), values.len() * size_of::<u16>());
    let workers = preprocess_parallelism(bytes.len(), values.len());
    if workers <= 1 {
        fill_f32_values_as_u16_bytes(bytes, values.iter().copied(), convert);
        return;
    }

    let chunk_values = values.len().div_ceil(workers);
    bytes
        .par_chunks_mut(chunk_values * size_of::<u16>())
        .zip(values.par_chunks(chunk_values))
        .for_each(|(output, values)| {
            fill_f32_values_as_u16_bytes(output, values.iter().copied(), convert);
        });
}

fn fill_f32_values_as_u16_bytes<I, F>(bytes: &mut [u8], values: I, convert: F)
where
    I: IntoIterator<Item = f32>,
    F: Fn(f32) -> u16 + Copy,
{
    for (output, value) in bytes.chunks_exact_mut(size_of::<u16>()).zip(values) {
        output.copy_from_slice(&convert(value).to_le_bytes());
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
    static IMAGE_DTYPE: OnceLock<Option<String>> = OnceLock::new();
    static VIDEO_DTYPE: OnceLock<Option<String>> = OnceLock::new();
    static AUDIO_DTYPE: OnceLock<Option<String>> = OnceLock::new();
    static DEFAULT_DTYPE: OnceLock<Option<String>> = OnceLock::new();

    let modality_dtype = match modality {
        Modality::Image | Modality::ImageEmbeds => {
            cached_env_dtype(&IMAGE_DTYPE, "SMG_TOKENSPEED_IMAGE_ENCODER_INPUT_DTYPE")
        }
        Modality::Video => {
            cached_env_dtype(&VIDEO_DTYPE, "SMG_TOKENSPEED_VIDEO_ENCODER_INPUT_DTYPE")
        }
        Modality::Audio => {
            cached_env_dtype(&AUDIO_DTYPE, "SMG_TOKENSPEED_AUDIO_ENCODER_INPUT_DTYPE")
        }
    };
    modality_dtype
        .or_else(|| cached_env_dtype(&DEFAULT_DTYPE, "SMG_TOKENSPEED_ENCODER_INPUT_DTYPE"))
}

fn cached_env_dtype(cell: &'static OnceLock<Option<String>>, name: &str) -> Option<String> {
    cell.get_or_init(|| std::env::var(name).ok().filter(|dtype| !dtype.is_empty()))
        .clone()
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

/// Resolve whether large multimodal tensors should use the SHM transport for
/// this request. `shm` = always (legacy explicit opt-in); `auto` = only when the
/// worker is known to share SMG's `/dev/shm`; anything else (including unset or
/// `inline`) keeps the inline gRPC path.
fn resolve_tokenspeed_shm_enabled(modality: Modality, workers: Option<&WorkerSelection>) -> bool {
    let configured_mode = tokenspeed_mm_tensor_transport_mode();
    let mode = effective_tokenspeed_transport_mode(modality, &configured_mode);
    log_tokenspeed_transport_config_once(&configured_mode, &mode, modality);
    match mode.as_str() {
        // SHM only ever happens when SMG can actually write /dev/shm.
        "shm" => tokenspeed_shm_dev_writable(),
        "auto" => worker_shares_dev_shm(workers) && tokenspeed_shm_dev_writable(),
        "" | "inline" => false,
        other => {
            log_unknown_tokenspeed_transport_once(other);
            false
        }
    }
}

fn effective_tokenspeed_transport_mode(modality: Modality, configured_mode: &str) -> String {
    if !configured_mode.is_empty() {
        return configured_mode.to_string();
    }

    match modality {
        Modality::Video => "auto".to_string(),
        Modality::Image | Modality::ImageEmbeds | Modality::Audio => "inline".to_string(),
    }
}

fn log_tokenspeed_transport_config_once(
    configured_mode: &str,
    effective_mode: &str,
    modality: Modality,
) {
    static LOGGED: OnceLock<()> = OnceLock::new();
    LOGGED.get_or_init(|| {
        info!(
            configured_mode,
            effective_mode,
            ?modality,
            shm_min_bytes = tokenspeed_mm_shm_min_bytes(),
            dev_writable = tokenspeed_shm_dev_writable(),
            "TokenSpeed multimodal tensor transport configured"
        );
    });
}

fn log_unknown_tokenspeed_transport_once(value: &str) {
    static WARNED: OnceLock<()> = OnceLock::new();
    WARNED.get_or_init(|| {
        warn!(
            value,
            "Unknown SMG_TOKENSPEED_MM_TENSOR_TRANSPORT value; expected inline|shm|auto, using inline"
        );
    });
}

/// Whether the worker is *verified* to share SMG's `/dev/shm`, making the SHM
/// transport safe under `auto`.
///
/// Rather than inferring locality from the worker URL (TCP loopback proves only
/// network locality, not a shared `/dev/shm`), the worker advertises its
/// `/dev/shm` filesystem identity (`<boot_id>:<st_dev of /dev/shm>`) via
/// `GetServerInfo`, which discovery stores in the worker's `shm_namespace_id`
/// label. Two processes share `/dev/shm` iff these tokens match: `boot_id` pins
/// the host, and `st_dev` is the tmpfs superblock device, identical whenever the
/// same tmpfs backs both `/dev/shm` mounts — including separate containers that
/// share it via `--ipc`/bind-mount (where mount-namespace inodes differ but the
/// underlying superblock is the same). We compare the worker's token to ours:
/// equal ⇒ shared. A missing/empty token or any mismatch is treated as
/// non-sharing, so `auto` safely falls back to inline.
fn worker_shares_dev_shm(workers: Option<&WorkerSelection>) -> bool {
    let Some(local) = local_shm_namespace_id() else {
        return false;
    };
    let worker = match workers {
        Some(WorkerSelection::Single { worker }) => worker,
        Some(WorkerSelection::Dual { prefill, .. }) => prefill,
        None => return false,
    };
    worker
        .metadata()
        .spec
        .labels
        .get("shm_namespace_id")
        .is_some_and(|id| !id.is_empty() && id == local)
}

/// This process's `/dev/shm` filesystem identity: `<boot_id>:<st_dev of /dev/shm>`.
/// `boot_id` pins the host (it is not namespaced) and `st_dev` is the tmpfs
/// superblock device backing `/dev/shm`; together they identify the tmpfs so two
/// processes sharing it (even across containers via `--ipc`/bind-mount) produce
/// the same token. Computed once; `None` if it can't be determined (then `auto`
/// stays inline).
fn local_shm_namespace_id() -> Option<&'static str> {
    static ID: OnceLock<Option<String>> = OnceLock::new();
    ID.get_or_init(compute_shm_namespace_id).as_deref()
}

#[cfg(unix)]
fn compute_shm_namespace_id() -> Option<String> {
    use std::os::unix::fs::MetadataExt;
    let boot_id = std::fs::read_to_string("/proc/sys/kernel/random/boot_id").ok()?;
    let shm_dev = std::fs::metadata("/dev/shm").ok()?.dev();
    Some(format!("{}:{shm_dev}", boot_id.trim()))
}

#[cfg(not(unix))]
fn compute_shm_namespace_id() -> Option<String> {
    None
}

fn canonical_float_dtype(dtype: &str) -> Option<String> {
    match dtype.trim().to_ascii_lowercase().as_str() {
        "float32" | "fp32" | "f32" => Some("float32".to_string()),
        "bfloat16" | "bf16" => Some("bfloat16".to_string()),
        "float16" | "fp16" | "f16" | "half" => Some("float16".to_string()),
        _ => None,
    }
}

fn array_shape(encoder_input: &ArrayD<f32>) -> Vec<u32> {
    encoder_input.shape().iter().map(|&d| d as u32).collect()
}

fn array_view_shape(encoder_input: &ArrayViewD<'_, f32>) -> Vec<u32> {
    encoder_input.shape().iter().map(|&d| d as u32).collect()
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

/// Serialize model-specific values to TensorBytes.
fn serialize_model_specific(
    model_specific: &HashMap<String, ModelSpecificValue>,
) -> HashMap<String, TensorBytes> {
    model_specific
        .iter()
        .filter_map(|(key, value)| match model_specific_to_tensor_bytes(value) {
            Some(tensor) => Some((key.clone(), tensor)),
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
            data: f32_slice_to_le_bytes(data),
            shape: shape.iter().map(|&d| d as u32).collect(),
            dtype: "float32".to_string(),
        }),
        ModelSpecificValue::IntTensor { data, shape } => Some(TensorBytes {
            data: i64_slice_to_le_bytes(data),
            shape: shape.iter().map(|&d| d as u32).collect(),
            dtype: "int64".to_string(),
        }),
        ModelSpecificValue::UintTensor { data, shape } => Some(TensorBytes {
            data: u32_slice_to_le_bytes(data),
            shape: shape.iter().map(|&d| d as u32).collect(),
            dtype: "uint32".to_string(),
        }),
        ModelSpecificValue::UintVec(v) => Some(TensorBytes {
            data: u32_slice_to_le_bytes(v),
            shape: vec![v.len() as u32],
            dtype: "uint32".to_string(),
        }),
        ModelSpecificValue::IntVec(v) => Some(TensorBytes {
            data: i64_slice_to_le_bytes(v),
            shape: vec![v.len() as u32],
            dtype: "int64".to_string(),
        }),
        ModelSpecificValue::FloatVec(v) => Some(TensorBytes {
            data: f32_slice_to_le_bytes(v),
            shape: vec![v.len() as u32],
            dtype: "float32".to_string(),
        }),
        _ => None,
    }
}

fn f32_slice_to_le_bytes(values: &[f32]) -> Vec<u8> {
    #[cfg(target_endian = "little")]
    {
        bytemuck::cast_slice(values).to_vec()
    }
    #[cfg(not(target_endian = "little"))]
    {
        f32_values_to_le_bytes(values.iter().copied(), values.len())
    }
}

fn i64_slice_to_le_bytes(values: &[i64]) -> Vec<u8> {
    #[cfg(target_endian = "little")]
    {
        bytemuck::cast_slice(values).to_vec()
    }
    #[cfg(not(target_endian = "little"))]
    {
        let mut bytes = Vec::with_capacity(values.len() * size_of::<i64>());
        for &value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

fn u32_slice_to_le_bytes(values: &[u32]) -> Vec<u8> {
    #[cfg(target_endian = "little")]
    {
        bytemuck::cast_slice(values).to_vec()
    }
    #[cfg(not(target_endian = "little"))]
    {
        let mut bytes = Vec::with_capacity(values.len() * size_of::<u32>());
        for &value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        io::{Read, Seek, SeekFrom, Write},
        mem::size_of,
    };

    use ndarray::IxDyn;
    use openai_protocol::common::{ImageUrl, VideoUrl};
    use tempfile::TempDir;

    use super::*;
    use crate::routers::grpc::proto_wrapper::{
        cleanup_tokenspeed_shm_handles, write_tokenspeed_shm_with, TokenSpeedTensorStorage,
    };

    #[test]
    #[cfg(target_os = "linux")]
    fn local_shm_namespace_id_resolves_on_linux() {
        // /proc/.../boot_id and /dev/shm both exist on the Linux CI/runtime
        // image, so the token must resolve to `<boot_id>:<st_dev>`. If it ever
        // returned None, `auto` would silently never enable SHM.
        let id = local_shm_namespace_id().expect("shm namespace id should resolve on Linux");
        assert!(
            id.contains(':'),
            "token must be <boot_id>:<st_dev>, got {id:?}"
        );
        let dev = id.rsplit(':').next().unwrap();
        assert!(
            dev.parse::<u64>().is_ok(),
            "st_dev component must be numeric, got {id:?}"
        );
    }

    #[test]
    fn tokenspeed_transport_default_uses_video_auto_only() {
        assert_eq!(
            effective_tokenspeed_transport_mode(Modality::Image, ""),
            "inline"
        );
        assert_eq!(
            effective_tokenspeed_transport_mode(Modality::ImageEmbeds, ""),
            "inline"
        );
        assert_eq!(
            effective_tokenspeed_transport_mode(Modality::Audio, ""),
            "inline"
        );
        assert_eq!(
            effective_tokenspeed_transport_mode(Modality::Video, ""),
            "auto"
        );
    }

    #[test]
    fn tokenspeed_transport_explicit_mode_overrides_modality_default() {
        assert_eq!(
            effective_tokenspeed_transport_mode(Modality::Image, "auto"),
            "auto"
        );
        assert_eq!(
            effective_tokenspeed_transport_mode(Modality::Video, "inline"),
            "inline"
        );
        assert_eq!(
            effective_tokenspeed_transport_mode(Modality::Video, "shm"),
            "shm"
        );
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
    fn test_expand_tokens_patch_offsets_all_repeated_patch_tokens() {
        let token_ids = vec![1, 100, 2];
        let replacements = vec![PromptReplacement {
            modality: Modality::Image,
            placeholder_token: "<image>".to_string(),
            tokens: vec![92, 92, 92, 92],
        }];

        let result = expand_tokens(&token_ids, Some(100), Some(92), &replacements);

        assert_eq!(result.token_ids, vec![1, 92, 92, 92, 92, 2]);
        assert_eq!(result.placeholders[0].offset, 1);
        assert_eq!(result.placeholders[0].length, 4);
        assert_eq!(result.patch_offsets.unwrap(), vec![(1, 4)]);
    }

    #[test]
    fn test_placeholders_for_items_uses_patch_offsets_in_one_pass() {
        let placeholders = vec![
            PlaceholderRange {
                offset: 1,
                length: 9,
            },
            PlaceholderRange {
                offset: 11,
                length: 4,
            },
        ];
        let patch_offsets = vec![(2, 3), (6, 3), (12, 2)];

        let by_item = placeholders_for_items(&placeholders, &patch_offsets);

        assert_eq!(by_item, vec![vec![(2, 3), (6, 3)], vec![(12, 2)]]);
    }

    #[test]
    fn test_placeholders_for_items_single_item_fast_path_matches_patch_offsets() {
        let placeholders = vec![PlaceholderRange {
            offset: 1,
            length: 9,
        }];
        let patch_offsets = vec![(2, 3), (6, 3)];

        let by_item = placeholders_for_items(&placeholders, &patch_offsets);

        assert_eq!(by_item, vec![vec![(2, 3), (6, 3)]]);
    }

    #[test]
    fn test_placeholders_for_items_single_patch_run_fast_path() {
        let placeholders = vec![PlaceholderRange {
            offset: 1,
            length: 9,
        }];
        let patch_offsets = vec![(2, 3)];

        let by_item = placeholders_for_items(&placeholders, &patch_offsets);

        assert_eq!(by_item, vec![vec![(2, 3)]]);
    }

    #[test]
    fn test_placeholders_for_items_one_patch_run_per_item_fast_path() {
        let placeholders = vec![
            PlaceholderRange {
                offset: 1,
                length: 4,
            },
            PlaceholderRange {
                offset: 8,
                length: 6,
            },
        ];
        let patch_offsets = vec![(1, 4), (9, 3)];

        let by_item = placeholders_for_items(&placeholders, &patch_offsets);

        assert_eq!(by_item, vec![vec![(1, 4)], vec![(9, 3)]]);
    }

    #[test]
    fn test_placeholders_for_items_falls_back_to_full_ranges() {
        let placeholders = vec![
            PlaceholderRange {
                offset: 1,
                length: 2,
            },
            PlaceholderRange {
                offset: 4,
                length: 3,
            },
        ];

        let by_item = placeholders_for_items(&placeholders, &[]);

        assert_eq!(by_item, vec![vec![(1, 2)], vec![(4, 3)]]);
    }

    #[test]
    fn test_flat_item_spans_precomputes_prefix_offsets() {
        let model_specific = HashMap::from([(
            "patches_per_image".to_string(),
            ModelSpecificValue::UintTensor {
                data: vec![2, 3, 1],
                shape: vec![3],
            },
        )]);
        let field_layouts = HashMap::from([(
            "pixel_values".to_string(),
            FieldLayout::flat("patches_per_image"),
        )]);

        let spans = flat_item_spans(&model_specific, &field_layouts, 3).unwrap();

        assert_eq!(spans["patches_per_image"], vec![(0, 2), (2, 3), (5, 1)]);
        assert_eq!(
            flat_item_span(&spans, "patches_per_image", 1).unwrap(),
            (2, 3)
        );
    }

    #[test]
    fn test_validate_tokenspeed_item_spans_rejects_flat_under_consumption() {
        let preprocessed = PreprocessedEncoderInputs {
            encoder_input: EncoderInput::Dense(
                ArrayD::from_shape_vec(IxDyn(&[4, 2]), vec![0.0; 8]).unwrap(),
            ),
            feature_token_counts: vec![1, 1],
            item_sizes: vec![(1, 1), (1, 1)],
            model_specific: HashMap::from([(
                "patches_per_image".to_string(),
                ModelSpecificValue::UintTensor {
                    data: vec![2, 1],
                    shape: vec![2],
                },
            )]),
        };
        let field_layouts = HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("patches_per_image".to_string(), FieldLayout::Batched),
        ]);
        let flat_spans = flat_item_spans(&preprocessed.model_specific, &field_layouts, 2).unwrap();

        let error = validate_tokenspeed_item_spans(&preprocessed, &field_layouts, &flat_spans, 2)
            .unwrap_err();

        assert!(error
            .to_string()
            .contains("flat tensor pixel_values first dimension mismatch"));
    }

    #[test]
    fn test_validate_tokenspeed_item_spans_rejects_batched_extra_rows() {
        let preprocessed = PreprocessedEncoderInputs {
            encoder_input: EncoderInput::Dense(
                ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![0.0; 6]).unwrap(),
            ),
            feature_token_counts: vec![1, 1],
            item_sizes: vec![(1, 1), (1, 1)],
            model_specific: HashMap::new(),
        };
        let field_layouts = HashMap::new();
        let flat_spans = HashMap::new();

        let error = validate_tokenspeed_item_spans(&preprocessed, &field_layouts, &flat_spans, 2)
            .unwrap_err();

        assert!(error
            .to_string()
            .contains("batched tensor pixel_values first dimension mismatch"));
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
    fn assemble_tokenspeed_splits_image_items() {
        let mut model_specific = HashMap::new();
        model_specific.insert(
            "patches_per_image".to_string(),
            ModelSpecificValue::UintTensor {
                data: vec![2, 2],
                shape: vec![2],
            },
        );
        model_specific.insert(
            "image_grid_thw".to_string(),
            ModelSpecificValue::UintTensor {
                data: vec![1, 2, 3, 4, 5, 6],
                shape: vec![2, 3],
            },
        );

        let preprocessed = PreprocessedEncoderInputs {
            encoder_input: EncoderInput::Dense(
                ArrayD::from_shape_vec(
                    IxDyn(&[4, 2]),
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                )
                .unwrap(),
            ),
            feature_token_counts: vec![2, 2],
            item_sizes: vec![(1, 1), (1, 1)],
            model_specific,
        };

        let images = vec![
            Arc::new(ImageFrame::new(
                image::DynamicImage::new_rgb8(1, 1),
                bytes::Bytes::from_static(b"a"),
                ImageDetail::Auto,
                llm_multimodal::ImageSource::InlineBytes,
                "hash-a".to_string(),
            )),
            Arc::new(ImageFrame::new(
                image::DynamicImage::new_rgb8(1, 1),
                bytes::Bytes::from_static(b"b"),
                ImageDetail::Auto,
                llm_multimodal::ImageSource::InlineBytes,
                "hash-b".to_string(),
            )),
        ];

        let intermediate = PrecomputedMultimodalIntermediate {
            modality: Modality::Image,
            preprocessed: Arc::new(preprocessed),
            images,
            videos: vec![],
            placeholders: vec![
                PlaceholderRange {
                    offset: 10,
                    length: 2,
                },
                PlaceholderRange {
                    offset: 20,
                    length: 2,
                },
            ],
            patch_offsets: Some(vec![(10, 2), (20, 2)]),
            placeholder_token_id: Some(151655),
            field_layouts: HashMap::from([
                (
                    "pixel_values".to_string(),
                    FieldLayout::flat("patches_per_image"),
                ),
                ("patches_per_image".to_string(), FieldLayout::Batched),
                ("image_grid_thw".to_string(), FieldLayout::Batched),
            ]),
            keep_on_cpu_keys: vec![],
        };

        let assembled = assemble_tokenspeed(intermediate, None).unwrap();
        assert_eq!(assembled.items.len(), 2);

        let first = &assembled.items[0];
        assert_eq!(first.modality, TokenSpeedModality::Image);
        assert_eq!(first.encoder_input.shape, vec![2, 2]);
        assert_eq!(first.encoder_input.nbytes(), 4 * size_of::<f32>());
        assert_eq!(first.mm_placeholders, vec![(10, 2)]);
        assert_eq!(
            first.content_hash,
            hash_hex_strings(std::iter::once("hash-a"))
        );
        assert_eq!(
            first.model_specific_tensors["image_grid_thw"].shape,
            vec![1, 3]
        );
        assert_eq!(
            first.model_specific_tensors["patches_per_image"].shape,
            vec![1]
        );

        let second = &assembled.items[1];
        assert_eq!(second.encoder_input.shape, vec![2, 2]);
        assert_eq!(second.mm_placeholders, vec![(20, 2)]);
        assert_eq!(
            second.content_hash,
            hash_hex_strings(std::iter::once("hash-b"))
        );
        assert_eq!(
            second.model_specific_tensors["image_grid_thw"].shape,
            vec![1, 3]
        );
    }

    #[test]
    fn assemble_tokenspeed_splits_video_items() {
        let mut model_specific = HashMap::new();
        model_specific.insert(
            "patches_per_video".to_string(),
            ModelSpecificValue::UintTensor {
                data: vec![2, 2],
                shape: vec![2],
            },
        );
        model_specific.insert(
            "video_grid_thw".to_string(),
            ModelSpecificValue::UintTensor {
                data: vec![1, 2, 3, 4, 5, 6],
                shape: vec![2, 3],
            },
        );

        let preprocessed = PreprocessedEncoderInputs {
            encoder_input: EncoderInput::Dense(
                ArrayD::from_shape_vec(
                    IxDyn(&[4, 2]),
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                )
                .unwrap(),
            ),
            feature_token_counts: vec![2, 2],
            item_sizes: vec![(1, 1), (1, 1)],
            model_specific,
        };

        let videos = vec![
            Arc::new(VideoClip::new(
                vec![image::DynamicImage::new_rgb8(1, 1)],
                bytes::Bytes::from_static(b"a"),
                llm_multimodal::VideoSource::InlineBytes,
                "video-hash-a".to_string(),
            )),
            Arc::new(VideoClip::new(
                vec![image::DynamicImage::new_rgb8(1, 1)],
                bytes::Bytes::from_static(b"b"),
                llm_multimodal::VideoSource::InlineBytes,
                "video-hash-b".to_string(),
            )),
        ];

        let intermediate = PrecomputedMultimodalIntermediate {
            modality: Modality::Video,
            preprocessed: Arc::new(preprocessed),
            images: vec![],
            videos,
            placeholders: vec![
                PlaceholderRange {
                    offset: 30,
                    length: 2,
                },
                PlaceholderRange {
                    offset: 40,
                    length: 2,
                },
            ],
            patch_offsets: Some(vec![(30, 2), (40, 2)]),
            placeholder_token_id: Some(151656),
            field_layouts: HashMap::from([
                (
                    "pixel_values".to_string(),
                    FieldLayout::flat("patches_per_video"),
                ),
                ("patches_per_video".to_string(), FieldLayout::Batched),
                ("video_grid_thw".to_string(), FieldLayout::Batched),
            ]),
            keep_on_cpu_keys: vec![],
        };

        let assembled = assemble_tokenspeed(intermediate, None).unwrap();
        assert_eq!(assembled.items.len(), 2);

        let first = &assembled.items[0];
        assert_eq!(first.modality, TokenSpeedModality::Video);
        assert_eq!(first.encoder_input.shape, vec![2, 2]);
        assert_eq!(first.encoder_input.nbytes(), 4 * size_of::<f32>());
        assert_eq!(first.mm_placeholders, vec![(30, 2)]);
        assert_eq!(
            first.content_hash,
            hash_hex_strings(std::iter::once("video-hash-a"))
        );
        assert_eq!(
            first.model_specific_tensors["video_grid_thw"].shape,
            vec![1, 3]
        );
        assert_eq!(
            first.model_specific_tensors["patches_per_video"].shape,
            vec![1]
        );

        let second = &assembled.items[1];
        assert_eq!(second.encoder_input.shape, vec![2, 2]);
        assert_eq!(second.mm_placeholders, vec![(40, 2)]);
        assert_eq!(
            second.content_hash,
            hash_hex_strings(std::iter::once("video-hash-b"))
        );
        assert_eq!(
            second.model_specific_tensors["video_grid_thw"].shape,
            vec![1, 3]
        );
    }

    #[test]
    #[cfg(unix)]
    fn serialize_tokenspeed_encoder_inputs_packs_large_items_into_one_shm_segment() {
        if !tokenspeed_shm_dev_writable() {
            return;
        }

        let first_values = vec![1.0f32; 16 * 1024];
        let second_values = vec![2.0f32; 16 * 1024];
        let first = ArrayD::from_shape_vec(IxDyn(&[16 * 1024]), first_values).unwrap();
        let second = ArrayD::from_shape_vec(IxDyn(&[16 * 1024]), second_values).unwrap();
        let first_view = first.view();
        let second_view = second.view();

        let tensors = serialize_arrays_as_packed_tokenspeed_shm(
            &[&first_view, &second_view],
            "float32",
            1,
            false,
        )
        .expect("large encoder inputs should be packed");

        let handles = tensors
            .iter()
            .map(|tensor| match &tensor.storage {
                TokenSpeedTensorStorage::Shm(handle) => handle.clone(),
                TokenSpeedTensorStorage::Inline(_) => panic!("expected packed SHM tensor"),
            })
            .collect::<Vec<_>>();

        assert_eq!(handles.len(), 2);
        assert_eq!(handles[0].name, handles[1].name);
        assert_eq!(handles[0].offset, 0);
        assert_eq!(handles[0].nbytes, (16 * 1024 * size_of::<f32>()) as u64);
        assert_eq!(handles[1].offset, handles[0].nbytes);
        assert_eq!(handles[1].nbytes, handles[0].nbytes);

        let mut file = fs::File::open(format!("/dev/shm/{}", handles[0].name)).unwrap();
        let mut bytes = [0u8; size_of::<f32>()];
        file.read_exact(&mut bytes).unwrap();
        assert_eq!(f32::from_le_bytes(bytes), 1.0);
        file.seek(SeekFrom::Start(handles[1].offset)).unwrap();
        file.read_exact(&mut bytes).unwrap();
        assert_eq!(f32::from_le_bytes(bytes), 2.0);

        cleanup_tokenspeed_shm_handles(&handles);
    }

    #[test]
    #[cfg(unix)]
    fn serialize_tokenspeed_encoder_inputs_packs_when_combined_size_reaches_threshold() {
        if !tokenspeed_shm_dev_writable() {
            return;
        }

        let first = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0f32; 4]).unwrap();
        let second = ArrayD::from_shape_vec(IxDyn(&[4]), vec![2.0f32; 4]).unwrap();
        let first_view = first.view();
        let second_view = second.view();
        let item_nbytes = (4 * size_of::<f32>()) as u64;

        let tensors = serialize_arrays_as_packed_tokenspeed_shm(
            &[&first_view, &second_view],
            "float32",
            32,
            false,
        )
        .expect("combined encoder input size should trigger packed SHM");

        let handles = tensors
            .iter()
            .map(|tensor| match &tensor.storage {
                TokenSpeedTensorStorage::Shm(handle) => handle.clone(),
                TokenSpeedTensorStorage::Inline(_) => panic!("expected packed SHM tensor"),
            })
            .collect::<Vec<_>>();

        assert_eq!(handles.len(), 2);
        assert_eq!(handles[0].name, handles[1].name);
        assert_eq!(handles[0].nbytes, item_nbytes);
        assert_eq!(handles[1].offset, item_nbytes);
        assert_eq!(handles[1].nbytes, item_nbytes);

        cleanup_tokenspeed_shm_handles(&handles);
    }

    #[test]
    #[cfg(unix)]
    fn write_tokenspeed_shm_with_rejects_unexpected_byte_count() {
        if !tokenspeed_shm_dev_writable() {
            return;
        }

        let err = write_tokenspeed_shm_with(4, |file| file.write_all(&[1, 2]))
            .expect_err("short SHM writes must be rejected");

        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn serialize_tokenspeed_encoder_input_view_matches_owned_slice_bytes() {
        let array = ArrayD::from_shape_vec(
            IxDyn(&[3, 4]),
            (0..12).map(|value| value as f32 + 0.25).collect(),
        )
        .unwrap();
        let view = array.slice_axis(Axis(1), Slice::from(1..3));
        let owned = view.to_owned();

        for dtype in ["float32", "bfloat16", "float16"] {
            let (view_data, view_shape, view_dtype) = serialize_array_view_as_dtype(&view, dtype);
            let (owned_data, owned_shape, owned_dtype) =
                serialize_array_view_as_dtype(&owned.view(), dtype);

            assert_eq!(view_shape, owned_shape);
            assert_eq!(view_dtype, owned_dtype);
            assert_eq!(view_data, owned_data);

            let mut view_written = vec![0; view_data.len()];
            fill_array_as_dtype(&mut view_written, &view, dtype).unwrap();
            let mut owned_written = vec![0; owned_data.len()];
            fill_array_as_dtype(&mut owned_written, &owned.view(), dtype).unwrap();
            assert_eq!(view_written, owned_written);
        }
    }

    #[test]
    fn serialize_tokenspeed_u16_inline_bytes_are_little_endian() {
        let array = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0_f32, -2.0_f32]).unwrap();

        let (bf16, _, bf16_dtype) = serialize_array_view_as_dtype(&array.view(), "bfloat16");
        let (f16, _, f16_dtype) = serialize_array_view_as_dtype(&array.view(), "float16");

        assert_eq!(bf16_dtype, "bfloat16");
        assert_eq!(bf16, vec![0x80, 0x3f, 0x00, 0xc0]);
        assert_eq!(f16_dtype, "float16");
        assert_eq!(f16, vec![0x00, 0x3c, 0x00, 0xc0]);
    }

    #[test]
    fn serialize_tokenspeed_u16_parallel_matches_scalar_conversion() {
        let values: Vec<f32> = (0..300_000)
            .map(|index| (index as f32 - 150_000.0) / 257.0)
            .collect();
        let array = ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.clone()).unwrap();

        for (dtype, convert) in [
            ("bfloat16", f32_to_bf16_bits as fn(f32) -> u16),
            ("float16", f32_to_f16_bits as fn(f32) -> u16),
        ] {
            let (actual, _, _) = serialize_array_view_as_dtype(&array.view(), dtype);
            let expected: Vec<u8> = values
                .iter()
                .flat_map(|&value| convert(value).to_le_bytes())
                .collect();
            assert_eq!(actual, expected);

            let mut written = vec![0; actual.len()];
            fill_array_as_dtype(&mut written, &array.view(), dtype).unwrap();
            assert_eq!(written, expected);
        }
    }

    #[test]
    fn model_specific_tensor_bytes_are_little_endian() {
        let float_tensor =
            model_specific_to_tensor_bytes(&ModelSpecificValue::FloatVec(vec![1.0_f32, -2.0_f32]))
                .unwrap();
        let int_tensor =
            model_specific_to_tensor_bytes(&ModelSpecificValue::IntVec(vec![1_i64, -2_i64]))
                .unwrap();
        let uint_tensor =
            model_specific_to_tensor_bytes(&ModelSpecificValue::UintVec(vec![1_u32, 0x11223344]))
                .unwrap();

        assert_eq!(
            float_tensor.data,
            vec![0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0xc0]
        );
        assert_eq!(
            int_tensor.data,
            vec![
                0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xff,
                0xff, 0xff,
            ]
        );
        assert_eq!(
            uint_tensor.data,
            vec![0x01, 0x00, 0x00, 0x00, 0x44, 0x33, 0x22, 0x11]
        );
    }

    #[test]
    fn deferred_bf16_tokenspeed_tensor_matches_reference_conversion() {
        let lut: [[f32; 256]; 3] = std::array::from_fn(|channel| {
            std::array::from_fn(|value| value as f32 * (channel + 1) as f32 / 255.0)
        });
        let raw = vec![0, 127, 255, 1, 128, 254];
        let deferred =
            DeferredNormalizedEncoderInput::new(raw.clone(), vec![2, 3], lut, 1).unwrap();

        let tensor =
            serialize_deferred_bf16_tokenspeed_tensor(&deferred, false, usize::MAX, false).unwrap();
        let TokenSpeedTensorStorage::Inline(data) = tensor.storage else {
            panic!("expected inline deferred tensor");
        };
        let expected = raw
            .iter()
            .enumerate()
            .flat_map(|(index, &value)| {
                f32_to_bf16_bits(lut[index % 3][value as usize]).to_le_bytes()
            })
            .collect::<Vec<_>>();

        assert_eq!(data, expected);
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.dtype, "bfloat16");
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

    #[test]
    fn load_video_preprocessor_config_ignores_missing_video_processor_key() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("processor_config.json"),
            r#"{"image_processor":{"image_processor_type":"Qwen3VLImageProcessor"}}"#,
        )
        .unwrap();

        assert!(load_video_preprocessor_config(tmp.path()).is_none());
    }

    #[test]
    fn load_video_preprocessor_config_reads_video_processor_key() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("processor_config.json"),
            r#"{"video_processor":{"image_processor_type":"Qwen3VLVideoProcessor","do_resize":true}}"#,
        )
        .unwrap();

        let config =
            load_video_preprocessor_config(tmp.path()).expect("video_processor should parse");
        assert_eq!(
            config.image_processor_type.as_deref(),
            Some("Qwen3VLVideoProcessor")
        );
        assert_eq!(config.do_resize, Some(true));
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
