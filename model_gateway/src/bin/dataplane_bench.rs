//! GPU-free gateway data-plane microbench: N threads x (jpeg decode -> preprocess
//! -> bf16 serialize), the exact per-image path a multimodal request takes through
//! the gateway. Isolates allocator/serialization changes from cluster noise.
//!
//! Free modes model how the tokio gateway releases buffers:
//!   local   - same-thread alloc+free (glibc's best case; v1/v2 behavior)
//!   xthread - results dropped by dedicated freer threads (work-stealing pattern:
//!             buffers allocated on one thread are freed on another)
//!   hold    - freer keeps a FIFO of `hold` results before dropping (models the
//!             second-scale lifetimes of buffers pinned by in-flight requests)
//!
//! Usage: dataplane_bench <preprocessor_config.json> [threads] [secs] [w] [h] [batch] [mode] [hold]
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use llm_multimodal::{ImageProcessorRegistry, PreProcessorConfig, PreprocessedImages};

fn synth_jpeg(w: u32, h: u32) -> Vec<u8> {
    let mut buf = image::RgbImage::new(w, h);
    let mut state: u32 = 0x1234_5678;
    for p in buf.pixels_mut() {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        p.0 = [(state >> 8) as u8, (state >> 16) as u8, (state >> 24) as u8];
    }
    let img = image::DynamicImage::ImageRgb8(buf);
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Jpeg)
        .expect("jpeg encode");
    out.into_inner()
}

type ReqResult = (PreprocessedImages, Vec<u8>, Vec<u32>);

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cfg_path = args
        .get(1)
        .expect("usage: dataplane_bench <preprocessor_config.json> [threads] [secs] [w] [h] [batch] [mode] [hold]");
    let threads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let secs: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);
    let w: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1920);
    let h: u32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1080);
    let batch: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(8);
    let mode: String = args.get(7).cloned().unwrap_or_else(|| "local".into());
    let hold: usize = args.get(8).and_then(|s| s.parse().ok()).unwrap_or(128);

    let cfg: PreProcessorConfig =
        serde_json::from_str(&std::fs::read_to_string(cfg_path).expect("read config"))
            .expect("parse config");
    let registry = ImageProcessorRegistry::with_defaults();
    let jpeg = synth_jpeg(w, h);
    eprintln!(
        "jpeg_bytes={} threads={} secs={} res={}x{} batch={} mode={} hold={}",
        jpeg.len(),
        threads,
        secs,
        w,
        h,
        batch,
        mode,
        hold
    );

    let stop = AtomicBool::new(false);
    let count = AtomicU64::new(0);
    // Bounded channel for xthread/hold: backpressure ~2 in-flight results per worker.
    let (tx, rx) = mpsc::sync_channel::<ReqResult>(threads * 2);
    let t0 = Instant::now();
    std::thread::scope(|s| {
        if mode != "local" {
            let hold_n = if mode == "hold" { hold } else { 0 };
            s.spawn(move || {
                let mut fifo: VecDeque<ReqResult> = VecDeque::with_capacity(hold_n + 1);
                while let Ok(r) = rx.recv() {
                    fifo.push_back(r);
                    while fifo.len() > hold_n {
                        drop(fifo.pop_front());
                    }
                }
            });
        } else {
            drop(rx);
        }
        for _ in 0..threads {
            let tx = tx.clone();
            s.spawn(|| {
                let tx = tx; // move the clone in
                let model_id =
                    std::env::var("DP_MODEL_ID").unwrap_or_else(|_| "qwen3-vl-397b".into());
                let proc_ = registry
                    .find(&model_id, None)
                    .expect("no processor matched DP_MODEL_ID");
                while !stop.load(Ordering::Relaxed) {
                    // One iteration = one request: `batch` decoded images preprocessed
                    // as a single batch, matching the gateway's per-request shape
                    // (the batched pixel array is what exceeds glibc's adaptive
                    // mmap-threshold ceiling).
                    let imgs: Vec<image::DynamicImage> = (0..batch)
                        .map(|_| image::load_from_memory(&jpeg).expect("jpeg decode"))
                        .collect();
                    let pre = proc_.preprocess(&imgs, &cfg).expect("preprocess");
                    let (bytes, shape) = smg::bench_serialize_pixel_values(&pre, "bfloat16");
                    if mode == "local" {
                        std::hint::black_box((&pre, &bytes, &shape));
                    } else if tx.send((pre, bytes, shape)).is_err() {
                        break;
                    }
                    count.fetch_add(batch as u64, Ordering::Relaxed);
                }
            });
        }
        drop(tx); // freer exits once all workers hang up
        s.spawn(|| {
            std::thread::sleep(std::time::Duration::from_secs(secs));
            stop.store(true, Ordering::Relaxed);
        });
    });
    let el = t0.elapsed().as_secs_f64();
    let n = count.load(Ordering::Relaxed);
    println!(
        "imgs={} secs={:.1} img_per_s={:.1} thread_ms_per_img={:.1}",
        n,
        el,
        n as f64 / el,
        el * 1000.0 * threads as f64 / n as f64
    );
}
