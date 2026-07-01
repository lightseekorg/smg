/// Serialize the primary encoder input ndarray to raw little-endian f32 bytes + shape.
use std::{collections::HashMap, mem::size_of};

use anyhow::Result;
use llm_multimodal::{ModelSpecificValue, PreprocessedEncoderInputs};
use ndarray::ArrayD;
use tracing::warn;

use crate::routers::grpc::TensorBytes;

pub(super) fn serialize_encoder_input(
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

fn array_shape(encoder_input: &ArrayD<f32>) -> Vec<u32> {
    encoder_input.shape().iter().map(|&d| d as u32).collect()
}

fn f32_values_to_le_bytes<I>(values: I, len: usize) -> Vec<u8>
where
    I: Iterator<Item = f32>,
{
    let mut bytes = Vec::with_capacity(len * size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Serialize model-specific values to TensorBytes.
pub(super) fn serialize_model_specific(
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
pub(in crate::routers::grpc::multimodal) fn model_specific_to_tensor_bytes(
    value: &ModelSpecificValue,
) -> Option<TensorBytes> {
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
