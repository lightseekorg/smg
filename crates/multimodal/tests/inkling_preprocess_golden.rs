//! Inkling audio preprocessing regression checks.

#![allow(clippy::expect_used, clippy::panic)]

use llm_multimodal::{
    audio::{DecodedAudio, InklingAudioProcessor},
    vision::processor::{ModelSpecificValue, PreprocessedEncoderInputs},
};
use serde::Deserialize;

#[derive(Deserialize)]
struct GoldenDocument {
    audio_cases: Vec<AudioCase>,
}

#[derive(Deserialize)]
struct AudioCase {
    name: String,
    sample_rate: usize,
    samples: usize,
    seed: u32,
    #[serde(default = "default_audio_amplitude")]
    amplitude: f32,
    shape: Vec<usize>,
    tokens_per_item: Vec<i64>,
    feature_token_counts: Vec<usize>,
    fnv1a_i32: String,
}

fn default_audio_amplitude() -> f32 {
    1.0
}

fn make_audio_samples(count: usize, seed: u32, amplitude: f32) -> Vec<f32> {
    if seed == 0 {
        return vec![0.0; count];
    }
    (0..count)
        .map(|i| {
            let raw = ((i as u32 * 73 + seed * 977) % 65_536) as i32 - 32_768;
            raw as f32 / 32_768.0 * amplitude
        })
        .collect()
}

fn tokens_per_item(result: &PreprocessedEncoderInputs) -> Vec<i64> {
    match result.model_specific.get("tokens_per_item") {
        Some(ModelSpecificValue::IntTensor { data, shape }) => {
            assert_eq!(shape, &[data.len()]);
            data.clone()
        }
        value => panic!("expected tokens_per_item IntTensor, got {value:?}"),
    }
}

fn fnv1a_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn fnv1a_i32(values: &[f32]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        let bin = *value as i32;
        assert!(
            (*value - bin as f32).abs() < f32::EPSILON,
            "audio dMel bin is not an integer: {value}"
        );
        hash = fnv1a_update(hash, &bin.to_le_bytes());
    }
    format!("{hash:016x}")
}

fn load_golden() -> GoldenDocument {
    serde_json::from_str(include_str!(
        "fixtures/golden/inkling_preprocess_fingerprints.json"
    ))
    .expect("invalid checked-in Inkling golden fixture")
}

#[test]
fn inkling_audio_preprocess_matches_checked_in_golden() {
    let golden = load_golden();
    let processor = InklingAudioProcessor::new();
    for case in &golden.audio_cases {
        let decoded = DecodedAudio {
            samples: make_audio_samples(case.samples, case.seed, case.amplitude),
            sample_rate: case.sample_rate,
        };
        let result = processor
            .preprocess_decoded_clips(vec![decoded])
            .expect("Inkling audio preprocessing failed");

        assert_eq!(
            result.encoder_input.shape(),
            case.shape,
            "audio shape changed for {}",
            case.name
        );
        assert_eq!(
            result.feature_token_counts, case.feature_token_counts,
            "audio token counts changed for {}",
            case.name
        );
        assert_eq!(
            tokens_per_item(&result),
            case.tokens_per_item,
            "audio tokens_per_item changed for {}",
            case.name
        );

        let values = result
            .encoder_input
            .as_slice_memory_order()
            .expect("Inkling audio encoder input must be contiguous");
        assert_eq!(
            fnv1a_i32(values),
            case.fnv1a_i32,
            "Inkling audio int32 fingerprint changed for {}",
            case.name
        );
    }
}
