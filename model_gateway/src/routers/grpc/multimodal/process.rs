//! Multimodal processing core: fetch media → preprocess pixels → expand
//! placeholder tokens → build the lightweight [`MultimodalIntermediate`].
//!
//! The chat and Messages API pipelines share `process_multimodal_parts`; only
//! the content extraction differs (see [`super::detect`]).

use std::{sync::Arc, time::Instant};

use anyhow::Result;
use llm_multimodal::{
    AsyncMultiModalTracker, ImageFrame, Modality, ModelMetadata, PlaceholderRange,
    PreprocessedEncoderInputs, PromptReplacement, TrackedMedia, TrackerOutput, VideoClip,
};
use llm_tokenizer::TokenizerTrait;
use openai_protocol::{chat::ChatMessage, messages::InputMessage};
use tracing::{debug, info, warn};

use super::{
    config::MultimodalComponents,
    detect::{extract_content_parts, extract_content_parts_messages},
    log_mm_timing_enabled, MultimodalIntermediate, MultimodalOutput,
    PrecomputedMultimodalIntermediate,
};

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
    content_parts: Vec<llm_multimodal::MediaContentPart>,
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
                item_sizes = ?images.iter().map(|f| (f.image.width(), f.image.height())).collect::<Vec<_>>(),
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
    let preprocess_started = Instant::now();

    let registry = components.vision_processor_registry.clone();
    let model_id_owned = model_id.to_string();
    let model_type_owned = model_type.map(String::from);
    let images_for_preprocess = images.clone(); // cheap Arc refcount bumps
    let videos_for_preprocess = videos.clone(); // cheap Arc refcount bumps
    let preprocessed: PreprocessedEncoderInputs = tokio::task::spawn_blocking(move || {
        let processor = registry
            .find(&model_id_owned, model_type_owned.as_deref())
            .ok_or_else(|| {
                anyhow::anyhow!("No vision processor found for model: {model_id_owned}")
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
                    match rgb_video.frame_refs() {
                        Ok(frame_refs) => {
                            match processor.preprocess_video_rgb(&frame_refs, &pp_config) {
                                Ok(preprocessed) => return Ok(preprocessed),
                                Err(error) => {
                                    warn!(
                                        error = %error,
                                        "RGB video preprocessing fast path failed; falling back to materialized frames"
                                    );
                                }
                            }
                        }
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
        item_count = preprocessed.feature_token_counts.len(),
        total_tokens = preprocessed.feature_token_counts.iter().sum::<usize>(),
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
        if video.frames().is_empty() {
            video
                .rgb_video()
                .map_or(0, |rgb_video| rgb_video.frames.len())
        } else {
            video.frames().len()
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
    let mut extra_placeholders = 0usize;

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
            // A placeholder token seen after all replacements are consumed is
            // left in place (unchanged behavior) but counted so we can warn.
            if token == placeholder_id {
                extra_placeholders += 1;
            }
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
    if extra_placeholders > 0 {
        warn!(
            extra_placeholders,
            replacements = replacements.len(),
            "More placeholder tokens than replacements; extra placeholders left unexpanded"
        );
    }

    ExpandedTokens {
        token_ids: expanded,
        placeholders,
        patch_offsets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_expand_tokens_more_placeholders_than_replacements() {
        // Two placeholder tokens but only one replacement: the first is
        // expanded, the second is left in place unchanged (and warned about).
        let token_ids = vec![1, 100, 2, 100, 3];
        let replacements = vec![PromptReplacement {
            modality: Modality::Image,
            placeholder_token: "<image>".to_string(),
            tokens: vec![50, 50],
        }];

        let result = expand_tokens(&token_ids, Some(100), None, &replacements);

        // Output is unchanged behavior: excess placeholder (100) stays as-is.
        assert_eq!(result.token_ids, vec![1, 50, 50, 2, 100, 3]);
        assert_eq!(result.placeholders.len(), 1);
        assert_eq!(result.placeholders[0].offset, 1);
        assert_eq!(result.placeholders[0].length, 2);
    }
}
