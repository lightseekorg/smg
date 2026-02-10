use std::sync::Arc;

use bytes::Bytes;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use llm_multimodal::types::{
    ImageDetail, ImageFrame, ImageSource, MultiModalInputs, MultiModalTensor, MultiModalValue,
};
use serde_json::to_vec;

fn current_image_ingestion_path(data: &Bytes) -> Arc<ImageFrame> {
    let raw = data.clone();

    let dummy_img = image::DynamicImage::ImageRgb8(image::RgbImage::new(1, 1));

    Arc::new(ImageFrame::new(
        dummy_img,
        raw,
        ImageDetail::Auto,
        ImageSource::InlineBytes,
    ))
}

fn create_multimodal_payload(data: &Bytes) -> MultiModalInputs {
    let mut inputs = MultiModalInputs::new(vec![1, 2, 3]);

    let tensor = MultiModalTensor {
        shape: vec![1, 3, 224, 224],
        dtype: "float16".to_string(),
        data: data.clone(),
    };

    inputs
        .mm_kwargs
        .insert("images".to_string(), vec![MultiModalValue::Tensor(tensor)]);
    inputs
}

use prost::Message;

#[derive(Clone, PartialEq, Message)]
pub struct ProtoMultiModalTensor {
    #[prost(uint64, repeated, tag = "1")]
    pub shape: Vec<u64>,
    #[prost(string, tag = "2")]
    pub dtype: String,
    #[prost(bytes, tag = "3")]
    pub data: Bytes,
}

fn bench_multimodal_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("multimodal_payload_cloning");

    for size_mb in [1, 5, 10].iter() {
        let size_bytes = size_mb * 1024 * 1024;
        let dummy_data = vec![0u8; size_bytes];
        let bytes_data = Bytes::from(dummy_data);

        group.throughput(Throughput::Bytes(size_bytes as u64));

        // Ingestion Latency
        group.bench_with_input(
            BenchmarkId::new("ingestion_clone", format!("{}MB", size_mb)),
            &bytes_data,
            |b, data| b.iter(|| current_image_ingestion_path(black_box(data))),
        );

        // Payload Creation Latency
        group.bench_with_input(
            BenchmarkId::new("payload_creation_clone", format!("{}MB", size_mb)),
            &bytes_data,
            |b, data| b.iter(|| create_multimodal_payload(black_box(data))),
        );

        // JSON Serialization Latency (Baseline)
        let payload = create_multimodal_payload(&bytes_data);
        group.bench_with_input(
            BenchmarkId::new("json_serialization", format!("{}MB", size_mb)),
            &payload,
            |b, p| {
                b.iter(|| {
                    let encoded = to_vec(black_box(p)).unwrap();
                    black_box(encoded);
                })
            },
        );

        // Prost (gRPC) Serialization Latency
        let proto_tensor = ProtoMultiModalTensor {
            shape: vec![1, 3, 224, 224],
            dtype: "float16".to_string(),
            data: bytes_data.clone(),
        };
        group.bench_with_input(
            BenchmarkId::new("prost_serialization", format!("{}MB", size_mb)),
            &proto_tensor,
            |b, p| {
                b.iter(|| {
                    let encoded = p.encode_to_vec();
                    black_box(encoded);
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_multimodal_overhead);
criterion_main!(benches);
