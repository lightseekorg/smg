// ---------------------------------------------------------------------------
// Assembly: convert MultimodalIntermediate → backend-specific MultimodalData
// ---------------------------------------------------------------------------

mod serialization;
mod tokenspeed;

use std::sync::Arc;

use anyhow::Result;
use llm_multimodal::{ImageFrame, MultimodalRuntime, PreprocessedEncoderInputs};
#[cfg(test)]
pub(super) use serialization::model_specific_to_tensor_bytes;
use serialization::{serialize_encoder_input, serialize_model_specific};
pub(super) use tokenspeed::assemble_tokenspeed;
#[cfg(test)]
pub(super) use tokenspeed::{
    effective_tokenspeed_transport_mode, f32_to_bf16_bits, f32_to_f16_bits, fill_array_as_dtype,
    flat_item_span, flat_item_spans, hash_hex_strings, local_shm_namespace_id,
    placeholders_for_items, serialize_array_view_as_dtype,
    serialize_arrays_as_packed_tokenspeed_shm, serialize_deferred_bf16_tokenspeed_tensor,
    validate_tokenspeed_item_spans,
};

use super::{MultimodalIntermediate, PrecomputedMultimodalIntermediate, PreparedMedia};
use crate::routers::grpc::{
    client::GrpcClient,
    context::WorkerSelection,
    proto_wrapper::{SglangMultimodalData, TrtllmMultimodalData, VllmMultimodalData},
    MultimodalData,
};

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
            GrpcClient::Sglang(_) => Ok(MultimodalData::Sglang(assemble_sglang(
                materialize_encoder_input(precomputed)?,
            )?)),
            GrpcClient::Vllm(_) => Ok(MultimodalData::Vllm(assemble_vllm(
                materialize_encoder_input(precomputed)?,
            )?)),
            GrpcClient::Trtllm(_) => Ok(MultimodalData::Trtllm(assemble_trtllm(precomputed)?)),
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

fn assemble_sglang(
    intermediate: PrecomputedMultimodalIntermediate,
) -> Result<SglangMultimodalData> {
    let (pixel_values, pixel_values_shape) = serialize_encoder_input(&intermediate.preprocessed)?;
    let model_specific_tensors =
        serialize_model_specific(&intermediate.preprocessed.model_specific);
    let image_data = prepared_images(&intermediate.media, "SGLang")?
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
    let mm_hashes = prepared_images(&intermediate.media, "vLLM")?
        .iter()
        .map(|frame| frame.hash.clone())
        .collect();
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

fn assemble_trtllm(
    intermediate: PrecomputedMultimodalIntermediate,
) -> Result<TrtllmMultimodalData> {
    let image_data = prepared_images(&intermediate.media, "TRT-LLM")?
        .iter()
        .map(|f| f.raw_bytes.to_vec())
        .collect();
    Ok(TrtllmMultimodalData { image_data })
}

fn prepared_images<'a>(media: &'a PreparedMedia, backend: &str) -> Result<&'a [Arc<ImageFrame>]> {
    match media {
        PreparedMedia::Images(images) => Ok(images),
        PreparedMedia::Videos(_) => Err(anyhow::anyhow!(
            "{backend} multimodal path currently supports image inputs only; got {}",
            media.modality()
        )),
    }
}
