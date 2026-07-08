//! Assembly: convert a [`MultimodalIntermediate`] into backend-specific
//! `MultimodalData` once the target backend is known (after worker selection).
//!
//! For TokenSpeed this also splits the batched preprocessing output into
//! per-item encoder inputs and per-item content hashes used for encode routing.

use std::{collections::HashMap, time::Instant};

use anyhow::{Context, Result};
use llm_multimodal::{
    FieldLayout, Modality, ModelSpecificValue, PlaceholderRange, PreprocessedEncoderInputs,
};
use ndarray::ArrayViewD;
use tracing::{info, warn};

use super::{
    log_mm_timing_enabled,
    serialize::{
        model_specific_to_tensor_bytes, serialize_array_as_tokenspeed_tensor,
        serialize_encoder_input, serialize_model_specific, slice_array_axis0,
    },
    transport::{mm_encoder_input_dtype, resolve_mm_shm_enabled, resolve_mm_shm_min_bytes},
    MultimodalIntermediate, PrecomputedMultimodalIntermediate,
};
use crate::routers::grpc::{
    client::GrpcClient,
    context::WorkerSelection,
    proto_wrapper::{
        cleanup_tokenspeed_items_encoder_shm, SglangMultimodalData, TensorBytes,
        TokenSpeedModality, TokenSpeedMultimodalData, TokenSpeedMultimodalItem, TokenSpeedTensor,
        TrtllmMultimodalData, VllmMultimodalData,
    },
    MultimodalData,
};

/// Assemble backend-specific multimodal data from the intermediate.
///
/// Called in request_building after worker selection, when the backend is known.
pub(crate) async fn assemble_multimodal_data(
    intermediate: MultimodalIntermediate,
    client: &GrpcClient,
    workers: Option<&WorkerSelection>,
) -> Result<MultimodalData> {
    assemble_multimodal_data_impl(intermediate, client, workers, false).await
}

/// Assemble multimodal data for a prefill request whose item embeddings will
/// arrive out-of-band from encode workers.
pub(crate) async fn assemble_multimodal_data_after_encode(
    intermediate: MultimodalIntermediate,
    client: &GrpcClient,
    workers: Option<&WorkerSelection>,
) -> Result<MultimodalData> {
    assemble_multimodal_data_impl(intermediate, client, workers, true).await
}

#[expect(
    clippy::unreachable,
    reason = "MLX multimodal rejected by caller before reaching here"
)]
async fn assemble_multimodal_data_impl(
    intermediate: MultimodalIntermediate,
    client: &GrpcClient,
    workers: Option<&WorkerSelection>,
    omit_prefill_pixels: bool,
) -> Result<MultimodalData> {
    match intermediate {
        MultimodalIntermediate::Precomputed(precomputed) => match client {
            GrpcClient::Sglang(_) => {
                ensure_image_only(&precomputed, "SGLang")?;
                Ok(MultimodalData::Sglang(assemble_sglang(precomputed)))
            }
            GrpcClient::Vllm(_) => {
                ensure_image_only(&precomputed, "vLLM")?;
                Ok(MultimodalData::Vllm(assemble_vllm(precomputed, workers)))
            }
            GrpcClient::Trtllm(_) => {
                ensure_image_only(&precomputed, "TRT-LLM")?;
                Ok(MultimodalData::Trtllm(assemble_trtllm(precomputed)))
            }
            GrpcClient::TokenSpeed(_) => {
                let options =
                    tokenspeed_assembly_options(precomputed.modality, workers, omit_prefill_pixels);
                let pending = tokio::task::spawn_blocking(move || {
                    assemble_tokenspeed_with_options(&precomputed, options)
                        .map(PendingTokenSpeedAssembly::new)
                })
                .await
                .context("TokenSpeed multimodal assembly task failed")??;
                Ok(MultimodalData::TokenSpeed(pending.into_inner()?))
            }
            GrpcClient::Mlx(_) => unreachable!(
                "caller rejects multimodal for MLX in build_chat_request/build_messages_request"
            ),
        },
    }
}

/// Owns SHM-backed assembly output until the awaiting task accepts it.
///
/// Dropping a `spawn_blocking` join handle does not cancel its task. If the
/// request future is cancelled, Tokio drops the completed task output instead;
/// this guard unlinks any SHM files before that output is discarded.
struct PendingTokenSpeedAssembly {
    data: Option<TokenSpeedMultimodalData>,
}

impl PendingTokenSpeedAssembly {
    fn new(data: TokenSpeedMultimodalData) -> Self {
        Self { data: Some(data) }
    }

    fn into_inner(mut self) -> Result<TokenSpeedMultimodalData> {
        self.data
            .take()
            .context("pending TokenSpeed assembly is missing data")
    }
}

impl Drop for PendingTokenSpeedAssembly {
    fn drop(&mut self) {
        if let Some(data) = &self.data {
            cleanup_tokenspeed_items_encoder_shm(&data.items, None);
        }
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
    let (pixel_values, pixel_values_shape) = serialize_encoder_input(&intermediate.preprocessed);
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

fn assemble_vllm(
    intermediate: PrecomputedMultimodalIntermediate,
    workers: Option<&WorkerSelection>,
) -> VllmMultimodalData {
    let (pixel_values, pixel_values_shape) = serialize_encoder_input(&intermediate.preprocessed);
    let model_specific_tensors = serialize_model_specific(intermediate.preprocessed.model_specific);
    let mm_hashes = intermediate.images.iter().map(|f| f.hash.clone()).collect();
    let mm_placeholders = intermediate
        .placeholders
        .iter()
        .map(|p| (p.offset as u32, p.length as u32))
        .collect();
    let batched_keys = PreprocessedEncoderInputs::batched_keys(&intermediate.field_layouts);
    let flat_keys = PreprocessedEncoderInputs::flat_keys(&intermediate.field_layouts);

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
        shm_enabled: resolve_mm_shm_enabled(workers, false),
        shm_min_bytes: resolve_mm_shm_min_bytes(workers),
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

pub(crate) fn assemble_tokenspeed(
    intermediate: &PrecomputedMultimodalIntermediate,
    workers: Option<&WorkerSelection>,
    skip_pixel_values: bool,
) -> Result<TokenSpeedMultimodalData> {
    let options = tokenspeed_assembly_options(intermediate.modality, workers, skip_pixel_values);
    assemble_tokenspeed_with_options(intermediate, options)
}

struct TokenSpeedAssemblyOptions {
    shm_enabled: bool,
    shm_min_bytes: usize,
    encoder_input_dtype: String,
    skip_pixel_values: bool,
}

fn tokenspeed_assembly_options(
    modality: Modality,
    workers: Option<&WorkerSelection>,
    skip_pixel_values: bool,
) -> TokenSpeedAssemblyOptions {
    TokenSpeedAssemblyOptions {
        shm_enabled: resolve_mm_shm_enabled(workers, skip_pixel_values),
        shm_min_bytes: resolve_mm_shm_min_bytes(workers),
        encoder_input_dtype: mm_encoder_input_dtype(modality, workers),
        skip_pixel_values,
    }
}

fn assemble_tokenspeed_with_options(
    intermediate: &PrecomputedMultimodalIntermediate,
    options: TokenSpeedAssemblyOptions,
) -> Result<TokenSpeedMultimodalData> {
    let log_timing = log_mm_timing_enabled();
    let total_started = Instant::now();
    let TokenSpeedAssemblyOptions {
        shm_enabled,
        shm_min_bytes,
        encoder_input_dtype,
        skip_pixel_values,
    } = options;
    // Use patch-only offsets when available and non-empty; fall back to full structural ranges.
    let patch_offsets = intermediate
        .patch_offsets
        .clone()
        .filter(|offsets| !offsets.is_empty())
        .unwrap_or_default();

    let modality = match intermediate.modality {
        Modality::Image => TokenSpeedModality::Image,
        Modality::Video => TokenSpeedModality::Video,
        Modality::Audio => TokenSpeedModality::Audio,
        Modality::ImageEmbeds => TokenSpeedModality::Image,
    };

    let item_count = precomputed_multimodal_item_count(intermediate)?;
    // Build items imperatively so that if any step fails partway we can unlink
    // the /dev/shm segments already created for prior items' encoder inputs
    // (and this item's, once created). `?`/`collect` would drop those
    // `TokenSpeedTensor::Shm` handles without ever reaching the send-path
    // cleanup, leaking files until the next sweep.
    let mut items: Vec<TokenSpeedMultimodalItem> = Vec::with_capacity(item_count);
    for item_index in 0..item_count {
        let encoder_input_started = Instant::now();
        // EPD prefill: the embedding arrives over Mooncake and this item's
        // encoder_input is stripped downstream (clear_mm_pixel_values), so skip
        // the per-item slice + serialize entirely when skip_pixel_values is set.
        let encoder_input = if skip_pixel_values {
            TokenSpeedTensor::inline(Vec::new(), Vec::new(), encoder_input_dtype.clone())
        } else {
            let item_encoder_input = match encoder_input_for_item(
                &intermediate.preprocessed,
                &intermediate.field_layouts,
                item_index,
            ) {
                Ok(value) => value,
                Err(error) => {
                    cleanup_tokenspeed_items_encoder_shm(&items, None);
                    return Err(error);
                }
            };
            serialize_array_as_tokenspeed_tensor(
                &item_encoder_input,
                &encoder_input_dtype,
                shm_enabled,
                shm_min_bytes,
            )
        };
        let encoder_input_serialize_ms = encoder_input_started.elapsed().as_secs_f64() * 1000.0;
        let model_specific_started = Instant::now();
        let model_specific_tensors = match serialize_model_specific_for_item(
            &intermediate.preprocessed.model_specific,
            &intermediate.field_layouts,
            item_index,
        ) {
            Ok(value) => value,
            Err(error) => {
                // `encoder_input` (possibly SHM) was created for this item but the
                // item isn't built; clean it plus all prior items.
                cleanup_tokenspeed_items_encoder_shm(&items, Some(&encoder_input));
                return Err(error);
            }
        };
        let model_specific_serialize_ms = model_specific_started.elapsed().as_secs_f64() * 1000.0;
        let mm_placeholders =
            placeholders_for_item(item_index, &intermediate.placeholders, &patch_offsets);
        let content_hash = content_hash_for_item(intermediate.modality, intermediate, item_index);

        if log_timing {
            info!(
                modality = ?modality,
                item_index,
                encoder_input_dtype = %encoder_input.dtype,
                encoder_input_bytes = encoder_input.nbytes(),
                encoder_input_shape = ?encoder_input.shape,
                model_specific_tensor_count = model_specific_tensors.len(),
                encoder_input_serialize_ms,
                model_specific_serialize_ms,
                "smg_mm_timing assemble_tokenspeed_item"
            );
        }

        items.push(TokenSpeedMultimodalItem {
            modality,
            encoder_input,
            model_specific_tensors,
            placeholder_token_id: intermediate.placeholder_token_id,
            mm_placeholders,
            content_hash,
        });
    }

    if log_timing {
        info!(
            modality = ?modality,
            item_count = items.len(),
            total_ms = total_started.elapsed().as_secs_f64() * 1000.0,
            "smg_mm_timing assemble_tokenspeed"
        );
    }

    Ok(TokenSpeedMultimodalData {
        items,
        shm_enabled,
        shm_min_bytes,
    })
}

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

pub(crate) fn precomputed_encode_routing_hashes(
    intermediate: &PrecomputedMultimodalIntermediate,
) -> Result<Vec<Vec<u8>>> {
    let item_count = precomputed_multimodal_item_count(intermediate)?;
    Ok((0..item_count)
        .map(|item_index| content_hash_for_item(intermediate.modality, intermediate, item_index))
        .collect())
}

fn encoder_input_for_item<'a>(
    preprocessed: &'a PreprocessedEncoderInputs,
    field_layouts: &HashMap<String, FieldLayout>,
    item_index: usize,
) -> Result<ArrayViewD<'a, f32>> {
    // The field layout key remains "pixel_values" because it mirrors the
    // HuggingFace/vLLM vision kwargs contract. Internally this tensor is the
    // modality encoder input we pass to TokenSpeed.
    let layout = field_layouts
        .get("pixel_values")
        .unwrap_or(&FieldLayout::Batched);
    match layout {
        FieldLayout::Batched => slice_array_axis0(&preprocessed.encoder_input, item_index, 1),
        FieldLayout::Flat { sizes_key } => {
            let sizes = tensor_sizes_from_model_specific(&preprocessed.model_specific, sizes_key)?;
            let (start, len) = item_span(&sizes, item_index)?;
            slice_array_axis0(&preprocessed.encoder_input, start, len)
        }
    }
}

fn serialize_model_specific_for_item(
    model_specific: &HashMap<String, ModelSpecificValue>,
    field_layouts: &HashMap<String, FieldLayout>,
    item_index: usize,
) -> Result<HashMap<String, TensorBytes>> {
    let mut serialized = HashMap::with_capacity(model_specific.len());
    for (key, value) in model_specific {
        let item_value = match field_layouts.get(key) {
            Some(FieldLayout::Batched) => value
                .slice_first_dim(item_index, 1)
                .with_context(|| format!("failed to slice model_specific tensor {key}"))?,
            Some(FieldLayout::Flat { sizes_key }) => {
                let sizes = tensor_sizes_from_model_specific(model_specific, sizes_key)?;
                let (start, len) = item_span(&sizes, item_index)?;
                value
                    .slice_first_dim(start, len)
                    .with_context(|| format!("failed to slice flat model_specific tensor {key}"))?
            }
            None => value.clone(),
        };
        if let Some(tensor) = model_specific_to_tensor_bytes(&item_value) {
            serialized.insert(key.clone(), tensor);
        } else {
            warn!(tensor_key = %key, "Dropping unsupported model_specific value during multimodal serialization");
        }
    }
    Ok(serialized)
}

fn placeholders_for_item(
    item_index: usize,
    placeholders: &[PlaceholderRange],
    patch_offsets: &[(u32, u32)],
) -> Vec<(u32, u32)> {
    let Some(placeholder) = placeholders.get(item_index) else {
        return Vec::new();
    };
    let start = placeholder.offset as u32;
    let end = start + placeholder.length as u32;
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

fn tensor_sizes_from_model_specific(
    model_specific: &HashMap<String, ModelSpecificValue>,
    key: &str,
) -> Result<Vec<usize>> {
    let value = model_specific
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("missing flat sizes tensor {key}"))?;
    value
        .as_flat_sizes()
        .with_context(|| format!("invalid flat sizes tensor {key}"))
}

fn item_span(sizes: &[usize], item_index: usize) -> Result<(usize, usize)> {
    let len = *sizes
        .get(item_index)
        .ok_or_else(|| anyhow::anyhow!("missing flat size for item {item_index}"))?;
    let start = sizes[..item_index]
        .iter()
        .try_fold(0usize, |acc, &size| acc.checked_add(size))
        .ok_or_else(|| anyhow::anyhow!("flat size offset overflow"))?;
    Ok((start, len))
}

fn hash_hex_strings<'a>(hashes: impl Iterator<Item = &'a str>) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new();
    for hash in hashes {
        hasher.update(hash.as_bytes());
    }
    hasher.finalize().as_bytes().to_vec()
}

#[cfg(test)]
mod tests {
    use std::{mem::size_of, path::Path, sync::Arc};

    use llm_multimodal::{ImageDetail, ImageFrame, VideoClip};
    use ndarray::{ArrayD, IxDyn};

    use super::*;
    use crate::routers::grpc::proto_wrapper::write_tokenspeed_shm_with;

    #[cfg(target_os = "linux")]
    fn pending_tokenspeed_shm_assembly() -> (PendingTokenSpeedAssembly, std::path::PathBuf) {
        let handle = write_tokenspeed_shm_with(4, |output| {
            output.copy_from_slice(&[1, 2, 3, 4]);
            Ok(())
        })
        .unwrap();
        let path = Path::new("/dev/shm").join(&handle.name);
        let data = TokenSpeedMultimodalData {
            items: vec![TokenSpeedMultimodalItem {
                modality: TokenSpeedModality::Image,
                encoder_input: TokenSpeedTensor::shm(handle, vec![2], "bfloat16".to_string()),
                model_specific_tensors: HashMap::new(),
                placeholder_token_id: None,
                mm_placeholders: vec![],
                content_hash: vec![],
            }],
            shm_enabled: true,
            shm_min_bytes: 0,
        };
        (PendingTokenSpeedAssembly::new(data), path)
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn pending_tokenspeed_assembly_cleans_shm_when_dropped() {
        let (pending, path) = pending_tokenspeed_shm_assembly();
        assert!(path.exists());

        drop(pending);

        assert!(!path.exists());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn pending_tokenspeed_assembly_transfers_shm_ownership() {
        let (pending, path) = pending_tokenspeed_shm_assembly();
        let data = pending.into_inner().unwrap();
        assert!(path.exists());

        cleanup_tokenspeed_items_encoder_shm(&data.items, None);

        assert!(!path.exists());
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
            encoder_input: ArrayD::from_shape_vec(
                IxDyn(&[4, 2]),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            )
            .unwrap(),
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
            preprocessed,
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

        let assembled = assemble_tokenspeed(&intermediate, None, false).unwrap();
        assert_eq!(assembled.items.len(), 2);

        let first = &assembled.items[0];
        assert_eq!(first.modality, TokenSpeedModality::Image);
        assert_eq!(first.encoder_input.shape, vec![2, 2]);
        // bf16 is the default TokenSpeed encoder_input wire dtype (2 bytes/elem).
        assert_eq!(first.encoder_input.nbytes(), 4 * size_of::<u16>());
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
            encoder_input: ArrayD::from_shape_vec(
                IxDyn(&[4, 2]),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            )
            .unwrap(),
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
            preprocessed,
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

        let assembled = assemble_tokenspeed(&intermediate, None, false).unwrap();
        assert_eq!(assembled.items.len(), 2);

        let first = &assembled.items[0];
        assert_eq!(first.modality, TokenSpeedModality::Video);
        assert_eq!(first.encoder_input.shape, vec![2, 2]);
        // bf16 is the default TokenSpeed encoder_input wire dtype (2 bytes/elem).
        assert_eq!(first.encoder_input.nbytes(), 4 * size_of::<u16>());
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
}
