//! Benchmark: SMG image preprocessing vs HF processor baseline.
//!
//! Measures the time for model-specific image preprocessing (resize, normalize,
//! patchify) at various image sizes. Compare results with the companion Python
//! script `scripts/bench_image_preprocess.py` which benchmarks HF transformers.
//!
//! Run:  cargo bench -p llm-multimodal --bench image_preprocess

#![allow(clippy::unwrap_used, clippy::expect_used)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use image::{imageops::FilterType, DynamicImage, RgbImage};
use llm_multimodal::vision::{
    image_processor::ImagePreProcessor,
    preprocessor_config::PreProcessorConfig,
    processors::{Llama4VisionProcessor, Qwen2VLProcessor, Qwen3VLProcessor},
    transforms,
};

/// Create a synthetic RGB image with some variation (not all zeros).
fn make_test_image(width: u32, height: u32) -> DynamicImage {
    let img = RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    });
    DynamicImage::ImageRgb8(img)
}

fn load_preprocessor_config(model_path: &str) -> Option<PreProcessorConfig> {
    let config_path = format!("{model_path}/preprocessor_config.json");
    let json = std::fs::read_to_string(&config_path).ok()?;
    PreProcessorConfig::from_json(&json).ok()
}

// ── Full pipeline benchmarks ─────────────────────────────────────

fn bench_qwen3_vl(c: &mut Criterion) {
    let processor = Qwen3VLProcessor::new();
    let config =
        load_preprocessor_config("/raid/models/Qwen/Qwen3-VL-8B-Instruct").unwrap_or_else(|| {
            PreProcessorConfig::from_json(
                r#"{"do_resize": true, "size": {"shortest_edge": 3136, "longest_edge": 12845056}}"#,
            )
            .unwrap()
        });

    let sizes: &[(u32, u32)] = &[
        (224, 224),
        (640, 480),
        (1024, 768),
        (1920, 1080),
        (3840, 2160),
    ];

    let mut group = c.benchmark_group("qwen3_vl_preprocess");
    for &(w, h) in sizes {
        let image = make_test_image(w, h);
        let images = [image];
        group.bench_with_input(
            BenchmarkId::new("single", format!("{w}x{h}")),
            &images,
            |b, imgs| {
                b.iter(|| processor.preprocess(imgs, &config).unwrap());
            },
        );
    }
    group.finish();

    // Batch benchmarks
    let mut group = c.benchmark_group("qwen3_vl_batch");
    for batch_size in [3, 5, 10] {
        let images: Vec<DynamicImage> = (0..batch_size)
            .map(|i| make_test_image(640 + i * 10, 480 + i * 10))
            .collect();
        group.bench_with_input(
            BenchmarkId::new("640x480", format!("batch{batch_size}")),
            &images,
            |b, imgs| {
                b.iter(|| processor.preprocess(imgs, &config).unwrap());
            },
        );
    }
    group.finish();

    // Extreme: very small and very large
    let mut group = c.benchmark_group("qwen3_vl_extreme");
    let extremes: &[(u32, u32, &str)] = &[
        (32, 32, "tiny_32x32"),
        (50, 50, "small_50x50"),
        (100, 2000, "tall_100x2000"),
        (2000, 100, "wide_2000x100"),
        (4096, 4096, "huge_4096x4096"),
    ];
    for &(w, h, label) in extremes {
        let image = make_test_image(w, h);
        let images = [image];
        group.bench_with_input(BenchmarkId::new("single", label), &images, |b, imgs| {
            b.iter(|| processor.preprocess(imgs, &config).unwrap());
        });
    }
    group.finish();
}

fn bench_qwen2_vl(c: &mut Criterion) {
    let processor = Qwen2VLProcessor::new();
    let config =
        load_preprocessor_config("/raid/models/Qwen/Qwen2-VL-2B-Instruct").unwrap_or_else(|| {
            PreProcessorConfig::from_json(
                r#"{"do_resize": true, "size": {"shortest_edge": 3136, "longest_edge": 12845056}}"#,
            )
            .unwrap()
        });

    let sizes: &[(u32, u32)] = &[(224, 224), (640, 480), (1024, 768), (1920, 1080)];

    let mut group = c.benchmark_group("qwen2_vl_preprocess");
    for &(w, h) in sizes {
        let image = make_test_image(w, h);
        let images = [image];
        group.bench_with_input(
            BenchmarkId::new("single", format!("{w}x{h}")),
            &images,
            |b, imgs| {
                b.iter(|| processor.preprocess(imgs, &config).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_llama4(c: &mut Criterion) {
    let processor = Llama4VisionProcessor::new();
    let config =
        load_preprocessor_config("/raid/models/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
            .unwrap_or_else(|| {
                PreProcessorConfig::from_json(
                    r#"{"do_resize": true, "size": {"height": 336, "width": 336}}"#,
                )
                .unwrap()
            });

    let sizes: &[(u32, u32)] = &[
        (224, 224),
        (336, 336),
        (640, 480),
        (1024, 768),
        (1920, 1080),
    ];

    let mut group = c.benchmark_group("llama4_preprocess");
    for &(w, h) in sizes {
        let image = make_test_image(w, h);
        let images = [image];
        group.bench_with_input(
            BenchmarkId::new("single", format!("{w}x{h}")),
            &images,
            |b, imgs| {
                b.iter(|| processor.preprocess(imgs, &config).unwrap());
            },
        );
    }
    group.finish();
}

// ── Per-step profiling benchmarks ────────────────────────────────

fn bench_individual_steps(c: &mut Criterion) {
    let sizes: &[(u32, u32)] = &[(640, 480), (1024, 768), (1920, 1080)];

    // Step 1: Resize only
    let mut group = c.benchmark_group("step_resize");
    for &(w, h) in sizes {
        let image = make_test_image(w, h);
        // Qwen3-VL target: smart_resize result
        let processor = Qwen3VLProcessor::new();
        let (th, tw) = processor.smart_resize(h as usize, w as usize).unwrap();
        group.bench_with_input(
            BenchmarkId::new("fir_bilinear", format!("{w}x{h}")),
            &image,
            |b, img| {
                b.iter(|| transforms::resize(img, tw as u32, th as u32, FilterType::Triangle));
            },
        );
    }
    group.finish();

    // Step 2: to_tensor only
    let mut group = c.benchmark_group("step_to_tensor");
    for &(w, h) in sizes {
        // Use a pre-resized image to isolate to_tensor cost
        let image = make_test_image(w, h);
        group.bench_with_input(
            BenchmarkId::new("rgb8", format!("{w}x{h}")),
            &image,
            |b, img| {
                b.iter(|| transforms::to_tensor(img));
            },
        );
    }
    group.finish();

    // Step 3: normalize only
    let mut group = c.benchmark_group("step_normalize");
    let mean = [0.5, 0.5, 0.5];
    let std = [0.5, 0.5, 0.5];
    for &(w, h) in sizes {
        let image = make_test_image(w, h);
        let tensor = transforms::to_tensor(&image);
        group.bench_with_input(
            BenchmarkId::new("f32", format!("{w}x{h}")),
            &tensor,
            |b, t| {
                b.iter_batched(
                    || t.clone(),
                    |mut fresh| transforms::normalize(&mut fresh, &mean, &std),
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_llama4_steps(c: &mut Criterion) {
    let processor = Llama4VisionProcessor::new();
    let config =
        load_preprocessor_config("/raid/models/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
            .unwrap_or_else(|| {
                PreProcessorConfig::from_json(
                    r#"{"do_resize": true, "size": {"height": 336, "width": 336}}"#,
                )
                .unwrap()
            });

    // 1024x768 is the worst case (1.8x slower than HF)
    let sizes: &[(u32, u32)] = &[(640, 480), (1024, 768), (1920, 1080)];

    let mut group = c.benchmark_group("llama4_steps");
    for &(w, h) in sizes {
        let image = make_test_image(w, h);

        // Full preprocess
        group.bench_with_input(
            BenchmarkId::new("full_preprocess", format!("{w}x{h}")),
            &image,
            |b, img| {
                let imgs = [img.clone()];
                b.iter(|| processor.preprocess(&imgs, &config).unwrap());
            },
        );

        // Just resize
        group.bench_with_input(
            BenchmarkId::new("resize_only", format!("{w}x{h}")),
            &image,
            |b, img| {
                b.iter(|| transforms::resize(img, 336, 336, FilterType::Triangle));
            },
        );

        // to_tensor_and_normalize on tile-sized image
        let tile_img = make_test_image(336, 336);
        group.bench_with_input(
            BenchmarkId::new("tensor_normalize_336", format!("{w}x{h}")),
            &tile_img,
            |b, img| {
                b.iter(|| {
                    transforms::to_tensor_and_normalize(img, &[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5])
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_qwen3_vl,
    bench_qwen2_vl,
    bench_llama4,
    bench_llama4_steps,
    bench_individual_steps,
    bench_fused_to_tensor_normalize,
    bench_to_rgb8,
    bench_resize_detailed,
);
criterion_main!(benches);

fn bench_fused_to_tensor_normalize(c: &mut Criterion) {
    let sizes: &[(u32, u32)] = &[(640, 480), (1024, 768), (1920, 1080)];
    let mean = [0.5, 0.5, 0.5];
    let std = [0.5, 0.5, 0.5];

    let mut group = c.benchmark_group("step_to_tensor_normalize_fused");
    for &(w, h) in sizes {
        let image = make_test_image(w, h);
        group.bench_with_input(
            BenchmarkId::new("fused", format!("{w}x{h}")),
            &image,
            |b, img| {
                b.iter(|| transforms::to_tensor_and_normalize(img, &mean, &std));
            },
        );
    }
    group.finish();
}

fn bench_to_rgb8(c: &mut Criterion) {
    let sizes: &[(u32, u32)] = &[(640, 480), (1024, 768), (1920, 1080)];

    let mut group = c.benchmark_group("step_to_rgb8");
    for &(w, h) in sizes {
        let image = make_test_image(w, h);
        group.bench_with_input(
            BenchmarkId::new("rgb8", format!("{w}x{h}")),
            &image,
            |b, img| {
                b.iter(|| img.to_rgb8());
            },
        );
    }
    group.finish();

    // Also test when image is already RGB8 (should be free)
    let mut group = c.benchmark_group("step_to_rgb8_noop");
    for &(w, h) in sizes {
        let image = make_test_image(w, h);
        let rgb = DynamicImage::ImageRgb8(image.to_rgb8());
        group.bench_with_input(
            BenchmarkId::new("already_rgb8", format!("{w}x{h}")),
            &rgb,
            |b, img| {
                b.iter(|| img.to_rgb8());
            },
        );
    }
    group.finish();
}

fn bench_resize_detailed(c: &mut Criterion) {
    let processor = Qwen3VLProcessor::new();

    // Benchmark: make_test_image + resize + convert back to DynamicImage
    let mut group = c.benchmark_group("step_resize_full_pipeline");
    let sizes: &[(u32, u32)] = &[(640, 480), (1024, 768), (1920, 1080)];
    for &(w, h) in sizes {
        let image = make_test_image(w, h);
        let (th, tw) = processor.smart_resize(h as usize, w as usize).unwrap();

        // fir resize (our path)
        group.bench_with_input(
            BenchmarkId::new("fir", format!("{w}x{h}->{tw}x{th}")),
            &image,
            |b, img| {
                b.iter(|| transforms::resize(img, tw as u32, th as u32, FilterType::Triangle));
            },
        );

        // image crate resize (old path, for comparison)
        group.bench_with_input(
            BenchmarkId::new("image_crate", format!("{w}x{h}->{tw}x{th}")),
            &image,
            |b, img| {
                b.iter(|| img.resize_exact(tw as u32, th as u32, FilterType::Triangle));
            },
        );
    }
    group.finish();
}
