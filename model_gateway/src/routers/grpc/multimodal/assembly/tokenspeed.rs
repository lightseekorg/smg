mod transport;

use std::{collections::HashMap, mem::size_of, sync::Arc, time::Instant};

use anyhow::{Context, Result};
use llm_multimodal::{
    vision::transforms::preprocess_parallelism, DeferredNormalizedEncoderInput, FieldLayout,
    Modality, ModelSpecificValue, PlaceholderRange, PreprocessedEncoderInputs,
};
use ndarray::{ArrayD, ArrayViewD, Axis, Slice};
use rayon::prelude::*;
use tracing::{info, warn};
#[cfg(test)]
pub(in crate::routers::grpc::multimodal) use transport::{
    effective_tokenspeed_transport_mode, local_shm_namespace_id,
};
use transport::{resolve_tokenspeed_shm_enabled, tokenspeed_encoder_input_dtype};

use super::{
    super::{log_mm_timing_enabled, PrecomputedMultimodalIntermediate},
    serialization::model_specific_to_tensor_bytes,
};
use crate::routers::grpc::{
    context::WorkerSelection,
    proto_wrapper::{
        tokenspeed_mm_shm_min_bytes, write_tokenspeed_shm_mapped, TensorBytes, TokenSpeedModality,
        TokenSpeedMultimodalData, TokenSpeedMultimodalItem, TokenSpeedTensor,
    },
};

pub(in crate::routers::grpc::multimodal) fn assemble_tokenspeed(
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

pub(in crate::routers::grpc::multimodal) fn flat_item_spans(
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

pub(in crate::routers::grpc::multimodal) fn validate_tokenspeed_item_spans(
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

pub(in crate::routers::grpc::multimodal) fn flat_item_span(
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

pub(in crate::routers::grpc::multimodal) fn placeholders_for_items(
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

pub(in crate::routers::grpc::multimodal) fn hash_hex_strings<'a>(
    hashes: impl Iterator<Item = &'a str>,
) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new();
    for hash in hashes {
        hasher.update(hash.as_bytes());
    }
    hasher.finalize().as_bytes().to_vec()
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

pub(in crate::routers::grpc::multimodal) fn serialize_deferred_bf16_tokenspeed_tensor(
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

pub(in crate::routers::grpc::multimodal) fn serialize_arrays_as_packed_tokenspeed_shm(
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

pub(in crate::routers::grpc::multimodal) fn fill_array_as_dtype(
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

pub(in crate::routers::grpc::multimodal) fn serialize_array_view_as_dtype(
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

fn canonical_float_dtype(dtype: &str) -> Option<String> {
    match dtype.trim().to_ascii_lowercase().as_str() {
        "float32" | "fp32" | "f32" => Some("float32".to_string()),
        "bfloat16" | "bf16" => Some("bfloat16".to_string()),
        "float16" | "fp16" | "f16" | "half" => Some("float16".to_string()),
        _ => None,
    }
}

fn array_view_shape(encoder_input: &ArrayViewD<'_, f32>) -> Vec<u32> {
    encoder_input.shape().iter().map(|&d| d as u32).collect()
}

#[inline]
pub(in crate::routers::grpc::multimodal) fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fff + lsb;
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

#[inline]
pub(in crate::routers::grpc::multimodal) fn f32_to_f16_bits(value: f32) -> u16 {
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
