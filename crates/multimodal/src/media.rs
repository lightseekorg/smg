use std::{
    collections::HashSet,
    io::Write,
    path::PathBuf,
    process::{Output, Stdio},
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use bytes::Bytes;
#[cfg(feature = "opencv-video")]
use opencv::{core::Mat, imgproc, prelude::*, videoio};
use reqwest::Client;
use tokio::{fs, io::AsyncReadExt, process::Command, task, time};
use tracing::info;
use url::Url;

const DEFAULT_VIDEO_PROCESS_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_VIDEO_MAX_DECODED_BYTES: usize = 1024 * 1024 * 1024;
static VIDEO_DECODE_BACKEND: OnceLock<Option<String>> = OnceLock::new();
static LOG_VIDEO_DECODE_TIMING: OnceLock<bool> = OnceLock::new();
static VIDEO_PROCESS_TIMEOUT: OnceLock<Duration> = OnceLock::new();
static VIDEO_MAX_DECODED_BYTES: OnceLock<usize> = OnceLock::new();

use super::{
    error::MediaConnectorError,
    types::{
        DecodedRgbFrame, DecodedRgbVideo, ImageDetail, ImageFrame, ImageSource, VideoClip,
        VideoSource,
    },
};

#[derive(Clone)]
pub struct MediaConnectorConfig {
    pub allowed_domains: Option<Vec<String>>,
    pub allowed_local_media_path: Option<PathBuf>,
    pub fetch_timeout: Duration,
}

impl Default for MediaConnectorConfig {
    fn default() -> Self {
        Self {
            allowed_domains: None,
            allowed_local_media_path: None,
            fetch_timeout: Duration::from_secs(10),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ImageFetchConfig {
    pub detail: ImageDetail,
}

impl Default for ImageFetchConfig {
    fn default() -> Self {
        Self {
            detail: ImageDetail::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VideoFetchConfig {
    pub min_frames: usize,
    pub max_frames: usize,
    pub sample_fps: f32,
}

impl Default for VideoFetchConfig {
    fn default() -> Self {
        Self {
            min_frames: 4,
            max_frames: 768,
            sample_fps: 2.0,
        }
    }
}

#[derive(Debug, Clone)]
pub enum MediaSource {
    Url(String),
    DataUrl(String),
    InlineBytes(Vec<u8>),
    File(PathBuf),
}

#[derive(Clone)]
pub struct MediaConnector {
    client: Client,
    allowed_domains: Option<HashSet<String>>,
    allowed_local_media_path: Option<PathBuf>,
    fetch_timeout: Duration,
}

impl MediaConnector {
    pub fn new(client: Client, config: MediaConnectorConfig) -> Result<Self, MediaConnectorError> {
        let allowed_domains = config.allowed_domains.map(|domains| {
            domains
                .into_iter()
                .map(|d| d.to_ascii_lowercase())
                .collect::<HashSet<_>>()
        });

        let allowed_local_media_path = if let Some(path) = config.allowed_local_media_path {
            Some(std::fs::canonicalize(path)?)
        } else {
            None
        };

        Ok(Self {
            client,
            allowed_domains,
            allowed_local_media_path,
            fetch_timeout: config.fetch_timeout,
        })
    }

    pub async fn fetch_image(
        &self,
        source: MediaSource,
        cfg: ImageFetchConfig,
    ) -> Result<Arc<ImageFrame>, MediaConnectorError> {
        match source {
            MediaSource::Url(url) => self.fetch_http_image(url, cfg).await,
            MediaSource::DataUrl(data_url) => self.fetch_data_url(data_url, cfg).await,
            MediaSource::InlineBytes(bytes) => {
                self.decode_image(bytes.into(), cfg.detail, ImageSource::InlineBytes)
                    .await
            }
            MediaSource::File(path) => self.fetch_file(path, cfg).await,
        }
    }

    pub async fn fetch_video(
        &self,
        source: MediaSource,
        cfg: VideoFetchConfig,
    ) -> Result<Arc<VideoClip>, MediaConnectorError> {
        // TODO: add a configurable max-video-bytes guard before fully buffering
        // URL/data/file/inline payloads. VideoClip retains the original bytes,
        // so oversized inputs should be rejected before decode.
        match source {
            MediaSource::Url(url) => self.fetch_http_video(url, cfg).await,
            MediaSource::DataUrl(data_url) => self.fetch_video_data_url(data_url, cfg).await,
            MediaSource::InlineBytes(bytes) => {
                self.decode_video(bytes.into(), cfg, VideoSource::InlineBytes)
                    .await
            }
            MediaSource::File(path) => self.fetch_video_file(path, cfg).await,
        }
    }

    async fn fetch_http_image(
        &self,
        url: String,
        cfg: ImageFetchConfig,
    ) -> Result<Arc<ImageFrame>, MediaConnectorError> {
        let parsed = Url::parse(&url).map_err(|_| MediaConnectorError::InvalidUrl(url.clone()))?;
        self.ensure_domain_allowed(&parsed)?;

        let mut req = self.client.get(parsed.as_str());
        if self.fetch_timeout > Duration::ZERO {
            req = req.timeout(self.fetch_timeout);
        }

        let resp = req.send().await.map_err(|err| {
            if err.is_timeout() {
                MediaConnectorError::Timeout(self.fetch_timeout)
            } else {
                MediaConnectorError::Http(err)
            }
        })?;

        let resp = resp.error_for_status()?;
        let bytes = resp.bytes().await?;
        self.decode_image(
            bytes,
            cfg.detail,
            ImageSource::Url {
                url: parsed.to_string(),
            },
        )
        .await
    }

    async fn fetch_data_url(
        &self,
        data_url: String,
        cfg: ImageFetchConfig,
    ) -> Result<Arc<ImageFrame>, MediaConnectorError> {
        let (metadata, data) = data_url
            .split_once(',')
            .ok_or_else(|| MediaConnectorError::DataUrl("missing comma in data url".into()))?;

        if !metadata.ends_with(";base64") {
            return Err(MediaConnectorError::DataUrl(
                "only base64 encoded data URLs are supported".into(),
            ));
        }

        let data = data.trim();
        let decoded = decode_base64_data_url_payload(data).await?;
        self.decode_image(decoded, cfg.detail, ImageSource::DataUrl)
            .await
    }

    async fn fetch_video_data_url(
        &self,
        data_url: String,
        cfg: VideoFetchConfig,
    ) -> Result<Arc<VideoClip>, MediaConnectorError> {
        let (metadata, data) = data_url
            .split_once(',')
            .ok_or_else(|| MediaConnectorError::DataUrl("missing comma in data url".into()))?;

        if !metadata.ends_with(";base64") {
            return Err(MediaConnectorError::DataUrl(
                "only base64 encoded data URLs are supported".into(),
            ));
        }

        let data = data.trim();
        let decoded = decode_base64_data_url_payload(data).await?;
        self.decode_video(decoded, cfg, VideoSource::DataUrl).await
    }

    async fn fetch_file(
        &self,
        path: PathBuf,
        cfg: ImageFetchConfig,
    ) -> Result<Arc<ImageFrame>, MediaConnectorError> {
        let allowed_root = self
            .allowed_local_media_path
            .as_ref()
            .ok_or_else(|| MediaConnectorError::DisallowedLocalPath(path.display().to_string()))?;

        let canonical = fs::canonicalize(&path).await?;
        if !canonical.starts_with(allowed_root) {
            return Err(MediaConnectorError::DisallowedLocalPath(
                path.display().to_string(),
            ));
        }

        let bytes = fs::read(&canonical).await?;
        self.decode_image(
            bytes.into(),
            cfg.detail,
            ImageSource::File { path: canonical },
        )
        .await
    }

    async fn fetch_http_video(
        &self,
        url: String,
        cfg: VideoFetchConfig,
    ) -> Result<Arc<VideoClip>, MediaConnectorError> {
        let parsed = Url::parse(&url).map_err(|_| MediaConnectorError::InvalidUrl(url.clone()))?;
        self.ensure_domain_allowed(&parsed)?;

        let mut req = self.client.get(parsed.as_str());
        if self.fetch_timeout > Duration::ZERO {
            req = req.timeout(self.fetch_timeout);
        }

        let resp = req.send().await.map_err(|err| {
            if err.is_timeout() {
                MediaConnectorError::Timeout(self.fetch_timeout)
            } else {
                MediaConnectorError::Http(err)
            }
        })?;

        let resp = resp.error_for_status()?;
        let bytes = resp.bytes().await?;
        self.decode_video(
            bytes,
            cfg,
            VideoSource::Url {
                url: parsed.to_string(),
            },
        )
        .await
    }

    async fn fetch_video_file(
        &self,
        path: PathBuf,
        cfg: VideoFetchConfig,
    ) -> Result<Arc<VideoClip>, MediaConnectorError> {
        let allowed_root = self
            .allowed_local_media_path
            .as_ref()
            .ok_or_else(|| MediaConnectorError::DisallowedLocalPath(path.display().to_string()))?;

        let canonical = fs::canonicalize(&path).await?;
        if !canonical.starts_with(allowed_root) {
            return Err(MediaConnectorError::DisallowedLocalPath(
                path.display().to_string(),
            ));
        }

        validate_video_fetch_config(cfg)?;
        let source = VideoSource::File {
            path: canonical.clone(),
        };

        let bytes = Bytes::from(fs::read(&canonical).await?);
        let bytes_for_hash = bytes.clone();
        let hash = async move {
            task::spawn_blocking(move || crate::hasher::hash_video(&bytes_for_hash))
                .await
                .map_err(MediaConnectorError::Blocking)
        };
        let decode = decode_video_frames_from_path(&canonical, bytes.len(), Some(&bytes), cfg);
        let (hash, decoded) = tokio::try_join!(hash, decode)?;

        Ok(Arc::new(video_clip_from_decoded(
            decoded, bytes, source, hash,
        )))
    }

    fn ensure_domain_allowed(&self, url: &Url) -> Result<(), MediaConnectorError> {
        if let Some(allowed) = &self.allowed_domains {
            let host = url
                .host_str()
                .map(|h| h.to_ascii_lowercase())
                .ok_or_else(|| MediaConnectorError::InvalidUrl(url.to_string()))?;
            if !allowed.contains(&host) {
                return Err(MediaConnectorError::DisallowedDomain(host));
            }
        }
        Ok(())
    }

    async fn decode_image(
        &self,
        bytes: Bytes,
        detail: ImageDetail,
        source: ImageSource,
    ) -> Result<Arc<ImageFrame>, MediaConnectorError> {
        let hash = crate::hasher::hash_image(&bytes);
        let image = decode_image(bytes.clone()).await?;

        Ok(Arc::new(ImageFrame::new(
            image, bytes, detail, source, hash,
        )))
    }

    async fn decode_video(
        &self,
        bytes: Bytes,
        cfg: VideoFetchConfig,
        source: VideoSource,
    ) -> Result<Arc<VideoClip>, MediaConnectorError> {
        validate_video_fetch_config(cfg)?;
        let bytes_for_hash = bytes.clone();
        let bytes_for_decode = bytes.clone();
        let hash = async move {
            task::spawn_blocking(move || crate::hasher::hash_video(&bytes_for_hash))
                .await
                .map_err(MediaConnectorError::Blocking)
        };
        let decode = decode_video_frames(bytes_for_decode, cfg);
        let (hash, decoded) = tokio::try_join!(hash, decode)?;

        Ok(Arc::new(video_clip_from_decoded(
            decoded, bytes, source, hash,
        )))
    }
}

async fn decode_base64_data_url_payload(data: &str) -> Result<Bytes, MediaConnectorError> {
    let data = data.to_owned();
    let decoded = task::spawn_blocking(move || BASE64_STANDARD.decode(data))
        .await
        .map_err(MediaConnectorError::Blocking)??;
    Ok(Bytes::from(decoded))
}

async fn decode_image(bytes: Bytes) -> Result<image::DynamicImage, MediaConnectorError> {
    // Decode JPEGs through libjpeg-turbo with PIL-compatible defaults
    // (accurate IDCT + fancy upsampling). The pure-Rust decoder can diverge by
    // a few pixel levels, which the vision encoder amplifies into an embedding
    // shift. Non-JPEG inputs and any turbojpeg failure fall back to the `image`
    // crate.
    task::spawn_blocking(
        move || -> Result<image::DynamicImage, MediaConnectorError> {
            if let Some(img) = crate::jpeg_turbo::decode_jpeg_rgb(&bytes) {
                return Ok(img);
            }
            let cursor = std::io::Cursor::new(bytes);
            let reader = image::ImageReader::new(cursor).with_guessed_format()?;
            Ok(reader.decode()?)
        },
    )
    .await
    .map_err(MediaConnectorError::Blocking)?
}

#[derive(Clone)]
enum DecodedVideoFrames {
    Images(Vec<image::DynamicImage>),
    Rgb(DecodedRgbVideo),
}

async fn decode_video_frames(
    bytes: Bytes,
    cfg: VideoFetchConfig,
) -> Result<DecodedVideoFrames, MediaConnectorError> {
    let input_bytes = bytes.len();
    let input_file = {
        let bytes = bytes.clone();
        task::spawn_blocking(move || write_temp_video_file(&bytes))
            .await
            .map_err(MediaConnectorError::Blocking)??
    };
    let input_path = input_file.path().to_path_buf();
    decode_video_frames_from_path(&input_path, input_bytes, Some(&bytes), cfg).await
}

async fn decode_video_frames_from_path(
    input_path: &std::path::Path,
    input_bytes: usize,
    input_data: Option<&Bytes>,
    cfg: VideoFetchConfig,
) -> Result<DecodedVideoFrames, MediaConnectorError> {
    match video_decode_backend_override() {
        Some("ffmpeg") => decode_video_with_ffmpeg(input_path, input_bytes, input_data, cfg).await,
        Some("opencv") => {
            #[cfg(feature = "opencv-video")]
            {
                let input_path = input_path.to_path_buf();
                task::spawn_blocking(move || {
                    decode_video_with_opencv_logged(&input_path, input_bytes, cfg)
                })
                .await
                .map_err(MediaConnectorError::Blocking)?
            }
            #[cfg(not(feature = "opencv-video"))]
            {
                Err(MediaConnectorError::VideoDecode(
                    "SMG_VIDEO_DECODE_BACKEND=opencv requires the opencv-video feature".to_string(),
                ))
            }
        }
        Some(backend) => Err(MediaConnectorError::VideoDecode(format!(
            "unsupported SMG_VIDEO_DECODE_BACKEND={backend}; expected auto, opencv, or ffmpeg"
        ))),
        None => {
            #[cfg(feature = "opencv-video")]
            {
                // OpenCV samples by frame index while the FFmpeg fallback uses an
                // fps filter, so the fallback can select a different frame set.
                let opencv_input_path = input_path.to_path_buf();
                let opencv_result = task::spawn_blocking(move || {
                    decode_video_with_opencv_logged(&opencv_input_path, input_bytes, cfg)
                })
                .await
                .map_err(MediaConnectorError::Blocking)?;

                match opencv_result {
                    Ok(frames) => Ok(frames),
                    Err(opencv_error) => {
                        if log_video_decode_timing_enabled() {
                            info!(
                                error = %opencv_error,
                                "smg_mm_timing video_decode_auto_opencv_fallback"
                            );
                        }

                        match decode_video_with_ffmpeg(input_path, input_bytes, input_data, cfg)
                            .await
                        {
                            Ok(frames) => Ok(frames),
                            Err(ffmpeg_error) => Err(MediaConnectorError::VideoDecode(format!(
                                "OpenCV decode failed: {opencv_error}; ffmpeg fallback failed: {ffmpeg_error}"
                            ))),
                        }
                    }
                }
            }

            #[cfg(not(feature = "opencv-video"))]
            {
                decode_video_with_ffmpeg(input_path, input_bytes, input_data, cfg).await
            }
        }
    }
}

fn validate_video_fetch_config(cfg: VideoFetchConfig) -> Result<(), MediaConnectorError> {
    if cfg.max_frames == 0 {
        return Err(MediaConnectorError::VideoDecode(
            "max_frames must be greater than 0".to_string(),
        ));
    }
    if cfg.min_frames == 0 {
        return Err(MediaConnectorError::VideoDecode(
            "min_frames must be greater than 0".to_string(),
        ));
    }
    if cfg.min_frames > cfg.max_frames {
        return Err(MediaConnectorError::VideoDecode(
            "min_frames must be less than or equal to max_frames".to_string(),
        ));
    }
    if !cfg.sample_fps.is_finite() || cfg.sample_fps <= 0.0 {
        return Err(MediaConnectorError::VideoDecode(
            "sample_fps must be finite and greater than 0".to_string(),
        ));
    }
    Ok(())
}

fn video_clip_from_decoded(
    decoded: DecodedVideoFrames,
    bytes: Bytes,
    source: VideoSource,
    hash: String,
) -> VideoClip {
    match decoded {
        DecodedVideoFrames::Images(frames) => VideoClip::new(frames, bytes, source, hash),
        DecodedVideoFrames::Rgb(rgb_video) => VideoClip::new_rgb(rgb_video, bytes, source, hash),
    }
}

#[cfg(feature = "opencv-video")]
fn decode_video_with_opencv_logged(
    input_path: &std::path::Path,
    input_bytes: usize,
    cfg: VideoFetchConfig,
) -> Result<DecodedVideoFrames, MediaConnectorError> {
    let started = video_decode_timing_started();
    let result = decode_video_with_opencv_file(input_path, cfg);
    match &result {
        Ok(_) => log_video_decode_backend_timing("opencv", started, input_bytes, cfg, None),
        Err(error) => {
            log_video_decode_backend_timing("opencv", started, input_bytes, cfg, Some(error));
        }
    }
    result
}

fn video_decode_backend_override() -> Option<&'static str> {
    VIDEO_DECODE_BACKEND
        .get_or_init(|| {
            let backend = std::env::var("SMG_VIDEO_DECODE_BACKEND")
                .ok()?
                .trim()
                .to_ascii_lowercase();
            match backend.as_str() {
                "" | "auto" => None,
                _ => Some(backend),
            }
        })
        .as_deref()
}

fn log_video_decode_timing_enabled() -> bool {
    *LOG_VIDEO_DECODE_TIMING.get_or_init(|| {
        std::env::var("SMG_LOG_MM_TIMING")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

fn video_decode_timing_started() -> Option<Instant> {
    log_video_decode_timing_enabled().then(Instant::now)
}

fn log_video_decode_backend_timing(
    backend: &str,
    started: Option<Instant>,
    input_bytes: usize,
    cfg: VideoFetchConfig,
    error: Option<&MediaConnectorError>,
) {
    if !log_video_decode_timing_enabled() {
        return;
    }
    let elapsed_ms = started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or_default();
    match error {
        Some(error) => info!(
            backend,
            ok = false,
            input_bytes,
            min_frames = cfg.min_frames,
            max_frames = cfg.max_frames,
            sample_fps = cfg.sample_fps,
            elapsed_ms,
            error = %error,
            "smg_mm_timing video_decode_backend"
        ),
        None => info!(
            backend,
            ok = true,
            input_bytes,
            min_frames = cfg.min_frames,
            max_frames = cfg.max_frames,
            sample_fps = cfg.sample_fps,
            elapsed_ms,
            "smg_mm_timing video_decode_backend"
        ),
    }
}

#[cfg(feature = "opencv-video")]
fn decode_video_with_opencv_file(
    input_path: &std::path::Path,
    cfg: VideoFetchConfig,
) -> Result<DecodedVideoFrames, MediaConnectorError> {
    let input = input_path.to_str().ok_or_else(|| {
        MediaConnectorError::VideoDecode(format!(
            "OpenCV video path is not valid UTF-8: {}",
            input_path.display()
        ))
    })?;

    let mut capture = open_opencv_video_capture(input)?;

    let total_frames = capture
        .get(videoio::CAP_PROP_FRAME_COUNT)
        .map_err(opencv_decode_error)?
        .round()
        .max(0.0) as usize;
    if total_frames == 0 {
        return Err(MediaConnectorError::VideoDecode(
            "OpenCV reported zero video frames".to_string(),
        ));
    }

    let fps = capture
        .get(videoio::CAP_PROP_FPS)
        .map_err(opencv_decode_error)?;
    let frame_indices = opencv_frame_indices(total_frames, fps, cfg);
    if frame_indices.is_empty() {
        return Err(MediaConnectorError::VideoDecode(
            "OpenCV video sampling produced no frame indices".to_string(),
        ));
    }

    let sampled_frame_counts = counted_frame_indices(&frame_indices);
    let unique_sampled_frames = sampled_frame_counts.len();
    let mut data = Vec::new();
    let mut frames = Vec::new();
    frames.try_reserve(frame_indices.len()).map_err(|e| {
        MediaConnectorError::VideoDecode(format!(
            "failed to reserve {} decoded video frame records: {e}",
            frame_indices.len()
        ))
    })?;
    let mut bgr_frame = Mat::default();
    let mut rgb_frame = Mat::default();

    let timeout = video_process_timeout();
    let started = Instant::now();
    // Advance to each sampled frame by SEQUENTIALLY grabbing the intervening frames
    // (cheap decode-without-retrieve) and `read`ing only the sampled ones, instead of
    // calling `set(CAP_PROP_POS_FRAMES)` per frame. OpenCV's POS_FRAMES set flushes/
    // re-seeks the decoder on every call (~10 ms/frame even for adjacent frames);
    // sequential grab is ~1-2 ms/frame. This is verified against the old
    // per-frame seek on both dense and sparse (non-keyframe) sampling, so
    // accuracy is unchanged. `sampled_frame_counts` is monotonic.
    // Index of the most recently decoded frame (-1 = nothing read yet).
    let mut decoded_pos: i64 = -1;
    for (idx, repeat_count) in sampled_frame_counts {
        if started.elapsed() >= timeout {
            return Err(MediaConnectorError::VideoDecode(format!(
                "OpenCV timed out after {:.3} seconds",
                timeout.as_secs_f64()
            )));
        }

        // Skip-decode the frames between the current position and `idx` so the
        // following `read` lands on `idx` without a decoder flush/seek.
        while decoded_pos + 1 < idx as i64 {
            if !capture.grab().map_err(opencv_decode_error)? {
                return Err(MediaConnectorError::VideoDecode(format!(
                    "OpenCV could not grab intervening frame to reach sampled frame {idx}"
                )));
            }
            decoded_pos += 1;
        }

        let read_successful = capture.read(&mut bgr_frame).map_err(opencv_decode_error)?;
        decoded_pos = idx as i64;
        if !read_successful || bgr_frame.empty() {
            continue;
        }

        imgproc::cvt_color_def(&bgr_frame, &mut rgb_frame, imgproc::COLOR_BGR2RGB)
            .map_err(opencv_decode_error)?;

        let decoded_width = u32::try_from(rgb_frame.cols()).map_err(|_| {
            MediaConnectorError::VideoDecode(format!(
                "OpenCV produced invalid RGB frame width: {}",
                rgb_frame.cols()
            ))
        })?;
        let decoded_height = u32::try_from(rgb_frame.rows()).map_err(|_| {
            MediaConnectorError::VideoDecode(format!(
                "OpenCV produced invalid RGB frame height: {}",
                rgb_frame.rows()
            ))
        })?;
        let frame_size = rawvideo_frame_size(decoded_width, decoded_height)?;
        if data.capacity() == 0 {
            let decoded_bytes = checked_decoded_rgb_bytes(unique_sampled_frames, frame_size)?;
            data.try_reserve_exact(decoded_bytes).map_err(|e| {
                MediaConnectorError::VideoDecode(format!(
                    "failed to reserve {decoded_bytes} decoded video bytes: {e}"
                ))
            })?;
        }
        let rgb_bytes = rgb_frame.data_bytes().map_err(opencv_decode_error)?;
        if rgb_bytes.len() < frame_size {
            return Err(MediaConnectorError::VideoDecode(format!(
                "OpenCV produced {} RGB bytes for {decoded_width}x{decoded_height} frame, expected {frame_size}",
                rgb_bytes.len()
            )));
        }
        let new_len = data.len().checked_add(frame_size).ok_or_else(|| {
            MediaConnectorError::VideoDecode(format!(
                "decoded video byte size overflow while appending {frame_size} bytes"
            ))
        })?;
        ensure_decoded_byte_limit(new_len)?;
        let offset = data.len();
        data.extend_from_slice(&rgb_bytes[..frame_size]);
        for _ in 0..repeat_count {
            frames.push(DecodedRgbFrame {
                width: decoded_width,
                height: decoded_height,
                offset,
                len: frame_size,
            });
        }
    }

    if frames.is_empty() {
        return Err(MediaConnectorError::VideoDecode(
            "OpenCV produced no readable sampled frames".to_string(),
        ));
    }
    if frames.len() != frame_indices.len() {
        return Err(MediaConnectorError::VideoDecode(format!(
            "OpenCV produced {} sampled frames, expected {}",
            frames.len(),
            frame_indices.len()
        )));
    }

    Ok(DecodedVideoFrames::Rgb(DecodedRgbVideo::new(
        Bytes::from(data),
        frames,
    )))
}

#[cfg(feature = "opencv-video")]
fn open_opencv_video_capture(input: &str) -> Result<videoio::VideoCapture, MediaConnectorError> {
    let capture = videoio::VideoCapture::from_file(input, videoio::CAP_FFMPEG)
        .map_err(opencv_decode_error)?;
    if capture.is_opened().map_err(opencv_decode_error)? {
        return Ok(capture);
    }

    let capture =
        videoio::VideoCapture::from_file(input, videoio::CAP_ANY).map_err(opencv_decode_error)?;
    if capture.is_opened().map_err(opencv_decode_error)? {
        return Ok(capture);
    }

    Err(MediaConnectorError::VideoDecode(format!(
        "OpenCV could not open video: {input}"
    )))
}

#[cfg(feature = "opencv-video")]
fn opencv_frame_indices(total_frames: usize, fps: f64, cfg: VideoFetchConfig) -> Vec<usize> {
    let mut target_frames = if fps.is_finite() && fps > 0.0 {
        let duration = total_frames as f64 / fps;
        (duration * cfg.sample_fps as f64).round() as usize
    } else {
        cfg.max_frames
    };
    target_frames = target_frames.clamp(cfg.min_frames, cfg.max_frames);
    target_frames = target_frames.max(1);
    if target_frames == 1 {
        return vec![0];
    }

    let last = (total_frames - 1) as f64;
    let denom = (target_frames - 1) as f64;
    (0..target_frames)
        .map(|idx| ((idx as f64 * last) / denom).floor() as usize)
        .collect()
}

#[cfg(feature = "opencv-video")]
fn counted_frame_indices(frame_indices: &[usize]) -> Vec<(usize, usize)> {
    let mut counts = Vec::new();
    for &idx in frame_indices {
        if let Some((last_idx, count)) = counts.last_mut() {
            if *last_idx == idx {
                *count += 1;
                continue;
            }
        }
        counts.push((idx, 1));
    }
    counts
}

#[cfg(feature = "opencv-video")]
fn opencv_decode_error(err: opencv::Error) -> MediaConnectorError {
    MediaConnectorError::VideoDecode(format!("OpenCV video decode failed: {err}"))
}

async fn decode_video_with_ffmpeg(
    input_path: &std::path::Path,
    input_bytes: usize,
    input_data: Option<&Bytes>,
    cfg: VideoFetchConfig,
) -> Result<DecodedVideoFrames, MediaConnectorError> {
    let duration_seconds = video_duration_seconds_for_input(input_path, input_data).await;

    let started = video_decode_timing_started();
    match decode_video_with_ffmpeg_ppm(input_path, cfg, duration_seconds).await {
        Ok(rgb_video) => {
            log_video_decode_backend_timing("ffmpeg_ppm_file", started, input_bytes, cfg, None);
            return Ok(DecodedVideoFrames::Rgb(rgb_video));
        }
        Err(error) => {
            log_video_decode_backend_timing(
                "ffmpeg_ppm_file",
                started,
                input_bytes,
                cfg,
                Some(&error),
            );
        }
    }

    if let Ok(metadata) = probe_video_metadata(input_path).await {
        let started = video_decode_timing_started();
        match decode_video_with_ffmpeg_raw(input_path, cfg, metadata).await {
            Ok(rgb_video) => {
                log_video_decode_backend_timing("ffmpeg_raw_file", started, input_bytes, cfg, None);
                return Ok(DecodedVideoFrames::Rgb(rgb_video));
            }
            Err(error) => {
                log_video_decode_backend_timing(
                    "ffmpeg_raw_file",
                    started,
                    input_bytes,
                    cfg,
                    Some(&error),
                );
            }
        }
    }

    let started = video_decode_timing_started();
    match decode_video_with_ffmpeg_png(input_path, cfg, duration_seconds).await {
        Ok(frames) => {
            log_video_decode_backend_timing("ffmpeg_png_file", started, input_bytes, cfg, None);
            Ok(DecodedVideoFrames::Images(frames))
        }
        Err(error) => {
            log_video_decode_backend_timing(
                "ffmpeg_png_file",
                started,
                input_bytes,
                cfg,
                Some(&error),
            );
            Err(error)
        }
    }
}

fn write_temp_video_file(bytes: &[u8]) -> Result<tempfile::NamedTempFile, MediaConnectorError> {
    let started = video_decode_timing_started();
    let mut input_file = tempfile::Builder::new()
        .prefix("smg-video-")
        .suffix(video_temp_suffix(bytes))
        .tempfile()?;
    input_file.write_all(bytes)?;
    input_file.flush()?;
    if log_video_decode_timing_enabled() {
        info!(
            nbytes = bytes.len(),
            elapsed_ms = started
                .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or_default(),
            suffix = video_temp_suffix(bytes),
            "smg_mm_timing video_tempfile_write"
        );
    }
    Ok(input_file)
}

fn video_temp_suffix(bytes: &[u8]) -> &'static str {
    if bytes.len() >= 12 && bytes.get(4..8) == Some(b"ftyp") {
        return ".mp4";
    }
    if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        return ".webm";
    }
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"AVI ") {
        return ".avi";
    }
    if bytes.starts_with(b"OggS") {
        return ".ogv";
    }
    if bytes.starts_with(&[0x00, 0x00, 0x01, 0xba]) {
        return ".mpg";
    }
    ".video"
}

fn video_process_timeout() -> Duration {
    *VIDEO_PROCESS_TIMEOUT.get_or_init(|| {
        std::env::var("SMG_VIDEO_PROCESS_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
            .map(Duration::from_secs_f64)
            .unwrap_or(DEFAULT_VIDEO_PROCESS_TIMEOUT)
    })
}

fn video_max_decoded_bytes() -> usize {
    *VIDEO_MAX_DECODED_BYTES.get_or_init(|| {
        std::env::var("SMG_VIDEO_MAX_DECODED_BYTES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|bytes| *bytes > 0)
            .unwrap_or(DEFAULT_VIDEO_MAX_DECODED_BYTES)
    })
}

fn ensure_decoded_byte_limit(bytes: usize) -> Result<(), MediaConnectorError> {
    let limit = video_max_decoded_bytes();
    if bytes > limit {
        return Err(MediaConnectorError::VideoDecode(format!(
            "decoded video RGB payload would be {bytes} bytes, exceeding SMG_VIDEO_MAX_DECODED_BYTES={limit}"
        )));
    }
    Ok(())
}

fn checked_decoded_rgb_bytes(
    frame_count: usize,
    frame_size: usize,
) -> Result<usize, MediaConnectorError> {
    let bytes = frame_count.checked_mul(frame_size).ok_or_else(|| {
        MediaConnectorError::VideoDecode(format!(
            "decoded video byte size overflow for {frame_count} frames of {frame_size} bytes"
        ))
    })?;
    ensure_decoded_byte_limit(bytes)?;
    Ok(bytes)
}

async fn run_video_command_output(
    command: Command,
    program: &'static str,
) -> Result<Output, MediaConnectorError> {
    let child = spawn_video_command(command, program)?;
    let timeout = video_process_timeout();
    match time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => Ok(output),
        Ok(Err(error)) => Err(MediaConnectorError::Io(error)),
        Err(_) => Err(MediaConnectorError::VideoDecode(format!(
            "{program} timed out after {:.3} seconds",
            timeout.as_secs_f64()
        ))),
    }
}

async fn run_video_command_output_with_stdout_capacity(
    command: Command,
    program: &'static str,
    stdout_capacity: usize,
) -> Result<Output, MediaConnectorError> {
    let child = spawn_video_command(command, program)?;
    let timeout = video_process_timeout();
    match time::timeout(
        timeout,
        collect_video_command_output(child, stdout_capacity),
    )
    .await
    {
        Ok(Ok(output)) => Ok(output),
        Ok(Err(error)) => Err(MediaConnectorError::Io(error)),
        Err(_) => Err(MediaConnectorError::VideoDecode(format!(
            "{program} timed out after {:.3} seconds",
            timeout.as_secs_f64()
        ))),
    }
}

fn spawn_video_command(
    mut command: Command,
    program: &'static str,
) -> Result<tokio::process::Child, MediaConnectorError> {
    command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    command.spawn().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            MediaConnectorError::VideoDecode(format!(
                "{program} executable not found; install {program} to decode video inputs"
            ))
        } else {
            MediaConnectorError::Io(e)
        }
    })
}

async fn collect_video_command_output(
    mut child: tokio::process::Child,
    stdout_capacity: usize,
) -> std::io::Result<Output> {
    let mut stdout_pipe = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("video command stdout was not piped"))?;
    let mut stderr_pipe = child
        .stderr
        .take()
        .ok_or_else(|| std::io::Error::other("video command stderr was not piped"))?;

    let stdout = async move {
        let mut bytes = Vec::with_capacity(stdout_capacity);
        stdout_pipe.read_to_end(&mut bytes).await?;
        Ok::<_, std::io::Error>(bytes)
    };
    let stderr = async move {
        let mut bytes = Vec::new();
        stderr_pipe.read_to_end(&mut bytes).await?;
        Ok::<_, std::io::Error>(bytes)
    };
    let status = child.wait();

    let (stdout, stderr, status) = tokio::try_join!(stdout, stderr, status)?;
    Ok(Output {
        status,
        stdout,
        stderr,
    })
}

async fn decode_video_with_ffmpeg_ppm(
    input_path: &std::path::Path,
    cfg: VideoFetchConfig,
    duration_seconds: Option<f64>,
) -> Result<DecodedRgbVideo, MediaConnectorError> {
    let fps_filter = fps_filter_for_optional_duration(duration_seconds, cfg);
    let max_frames = cfg.max_frames.to_string();
    let output_limit = video_max_decoded_bytes().to_string();
    let mut command = Command::new("ffmpeg");
    command
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-threads",
            "1",
            "-i",
        ])
        .arg(input_path)
        .args([
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            &fps_filter,
            "-frames:v",
            &max_frames,
            "-fs",
            &output_limit,
            "-f",
            "image2pipe",
            "-vcodec",
            "ppm",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]);
    let output = run_video_command_output_with_stdout_capacity(command, "ffmpeg", 0).await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(MediaConnectorError::VideoDecode(format!(
            "ffmpeg failed: {stderr}"
        )));
    }

    parse_ppm_rgb_video(Bytes::from(output.stdout))
}

async fn decode_video_with_ffmpeg_raw(
    input_path: &std::path::Path,
    cfg: VideoFetchConfig,
    metadata: VideoMetadata,
) -> Result<DecodedRgbVideo, MediaConnectorError> {
    let fps_filter = fps_filter_for_metadata(metadata, cfg);
    let max_frames = cfg.max_frames.to_string();
    let frame_size = rawvideo_frame_size(metadata.width, metadata.height)?;
    let target_frames = expected_sampled_frame_count(metadata, cfg);
    let decoded_bytes = checked_decoded_rgb_bytes(target_frames, frame_size)?;
    let output_limit = decoded_bytes.to_string();
    let mut command = Command::new("ffmpeg");
    // Rawvideo has no per-frame header, so we interpret stdout using ffprobe's
    // coded stream dimensions. Disable FFmpeg autorotation here; otherwise a
    // display-matrix rotation can swap output width/height and corrupt framing.
    command
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-threads",
            "1",
            "-noautorotate",
            "-i",
        ])
        .arg(input_path)
        .args([
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            &fps_filter,
            "-frames:v",
            &max_frames,
            "-fs",
            &output_limit,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]);
    let output = run_video_command_output_with_stdout_capacity(
        command,
        "ffmpeg",
        video_stdout_prealloc_capacity(metadata, decoded_bytes),
    )
    .await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(MediaConnectorError::VideoDecode(format!(
            "ffmpeg failed: {stderr}"
        )));
    }

    let frame_count = output.stdout.len() / frame_size;
    checked_decoded_rgb_bytes(frame_count, frame_size)?;
    let mut frames = Vec::new();
    frames.try_reserve(frame_count).map_err(|e| {
        MediaConnectorError::VideoDecode(format!(
            "failed to reserve {frame_count} decoded video frame records: {e}"
        ))
    })?;
    for idx in 0..frame_count {
        frames.push(DecodedRgbFrame {
            width: metadata.width,
            height: metadata.height,
            offset: idx * frame_size,
            len: frame_size,
        });
    }
    let remainder = output.stdout.len() % frame_size;
    if remainder != 0 {
        return Err(MediaConnectorError::VideoDecode(format!(
            "ffmpeg rawvideo output has trailing partial frame: {remainder} bytes"
        )));
    }
    if frames.is_empty() {
        return Err(MediaConnectorError::VideoDecode(
            "ffmpeg produced no frames".to_string(),
        ));
    }
    Ok(DecodedRgbVideo::new(Bytes::from(output.stdout), frames))
}

async fn decode_video_with_ffmpeg_png(
    input_path: &std::path::Path,
    cfg: VideoFetchConfig,
    duration_seconds: Option<f64>,
) -> Result<Vec<image::DynamicImage>, MediaConnectorError> {
    let fps_filter = fps_filter_for_optional_duration(duration_seconds, cfg);
    let max_frames = cfg.max_frames.to_string();
    let output_limit = video_max_decoded_bytes().to_string();
    let mut command = Command::new("ffmpeg");
    command
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-threads",
            "1",
            "-i",
        ])
        .arg(input_path)
        .args([
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            &fps_filter,
            "-frames:v",
            &max_frames,
            "-fs",
            &output_limit,
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ]);
    let output = run_video_command_output(command, "ffmpeg").await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(MediaConnectorError::VideoDecode(format!(
            "ffmpeg failed: {stderr}"
        )));
    }

    let pngs = split_png_stream(&output.stdout)?;
    let mut frames = Vec::with_capacity(pngs.len());
    let mut decoded_bytes = 0usize;
    for png in pngs {
        let image = image::load_from_memory(png)?;
        let frame_size = rawvideo_frame_size(image.width(), image.height())?;
        decoded_bytes = decoded_bytes.checked_add(frame_size).ok_or_else(|| {
            MediaConnectorError::VideoDecode("PNG decoded byte size overflow".to_string())
        })?;
        ensure_decoded_byte_limit(decoded_bytes)?;
        frames.push(image);
    }
    if frames.is_empty() {
        return Err(MediaConnectorError::VideoDecode(
            "ffmpeg produced no frames".to_string(),
        ));
    }
    Ok(frames)
}

#[derive(Debug, Clone, Copy)]
struct VideoMetadata {
    width: u32,
    height: u32,
    duration_seconds: Option<f64>,
}

async fn probe_video_metadata(
    input_path: &std::path::Path,
) -> Result<VideoMetadata, MediaConnectorError> {
    let mut command = Command::new("ffprobe");
    command
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(input_path);
    let output = run_video_command_output(command, "ffprobe").await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(MediaConnectorError::VideoDecode(format!(
            "ffprobe failed: {stderr}"
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut width = None;
    let mut height = None;
    let mut duration_seconds = None;
    for line in stdout.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        match key {
            "width" => width = value.parse::<u32>().ok(),
            "height" => height = value.parse::<u32>().ok(),
            "duration" if value != "N/A" => duration_seconds = value.parse::<f64>().ok(),
            _ => {}
        }
    }

    let width = width.ok_or_else(|| {
        MediaConnectorError::VideoDecode("ffprobe did not return video width".to_string())
    })?;
    let height = height.ok_or_else(|| {
        MediaConnectorError::VideoDecode("ffprobe did not return video height".to_string())
    })?;
    Ok(VideoMetadata {
        width,
        height,
        duration_seconds,
    })
}

fn fps_filter_for_metadata(metadata: VideoMetadata, cfg: VideoFetchConfig) -> String {
    fps_filter_for_optional_duration(metadata.duration_seconds, cfg)
}

fn fps_filter_for_optional_duration(
    duration_seconds: Option<f64>,
    cfg: VideoFetchConfig,
) -> String {
    if let Some(duration) = duration_seconds {
        if let Some(filter) = fps_filter_for_duration(duration, cfg) {
            return filter;
        }
    }

    format!("fps={}", cfg.sample_fps)
}

fn expected_sampled_frame_count(metadata: VideoMetadata, cfg: VideoFetchConfig) -> usize {
    if let Some(duration) = metadata.duration_seconds {
        if duration.is_finite() && duration > 0.0 {
            return (duration * cfg.sample_fps as f64)
                .round()
                .clamp(cfg.min_frames as f64, cfg.max_frames as f64) as usize;
        }
    }
    cfg.max_frames
}

fn video_stdout_prealloc_capacity(metadata: VideoMetadata, expected_bytes: usize) -> usize {
    match metadata.duration_seconds {
        Some(duration) if duration.is_finite() && duration > 0.0 => expected_bytes,
        _ => 0,
    }
}

fn fps_filter_for_duration(duration: f64, cfg: VideoFetchConfig) -> Option<String> {
    if !duration.is_finite() || duration <= 0.0 {
        return None;
    }
    let target_frames = (duration * cfg.sample_fps as f64)
        .round()
        .clamp(cfg.min_frames as f64, cfg.max_frames as f64);
    let fps = (target_frames / duration).max(f64::EPSILON);
    Some(format!("fps={fps:.6}"))
}

async fn video_duration_seconds_for_input(
    input_path: &std::path::Path,
    input_data: Option<&Bytes>,
) -> Option<f64> {
    if let Some(bytes) = input_data {
        if let Some(duration) = parse_mp4_duration_seconds(bytes.as_ref()) {
            return Some(duration);
        }
    }

    probe_video_duration_seconds(input_path).await.ok()
}

async fn probe_video_duration_seconds(
    input_path: &std::path::Path,
) -> Result<f64, MediaConnectorError> {
    let mut command = Command::new("ffprobe");
    command
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(input_path);
    match run_video_command_output(command, "ffprobe").await {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            stdout.trim().parse::<f64>().map_err(|err| {
                MediaConnectorError::VideoDecode(format!("failed to parse ffprobe duration: {err}"))
            })
        }
        Ok(_) | Err(_) => probe_video_duration_seconds_with_ffmpeg(input_path).await,
    }
}

async fn probe_video_duration_seconds_with_ffmpeg(
    input_path: &std::path::Path,
) -> Result<f64, MediaConnectorError> {
    let mut command = Command::new("ffmpeg");
    command
        .args(["-hide_banner", "-nostdin", "-i"])
        .arg(input_path);
    let output = run_video_command_output(command, "ffmpeg").await?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    parse_ffmpeg_duration_seconds(&stderr).ok_or_else(|| {
        MediaConnectorError::VideoDecode("failed to parse ffmpeg duration".to_string())
    })
}

fn parse_ffmpeg_duration_seconds(stderr: &str) -> Option<f64> {
    let marker = "Duration:";
    let start = stderr.find(marker)? + marker.len();
    let duration = stderr[start..].trim_start().split(',').next()?.trim();
    let mut parts = duration.split(':');
    let hours = parts.next()?.parse::<f64>().ok()?;
    let minutes = parts.next()?.parse::<f64>().ok()?;
    let seconds = parts.next()?.parse::<f64>().ok()?;
    Some(hours * 3600.0 + minutes * 60.0 + seconds)
}

fn parse_mp4_duration_seconds(bytes: &[u8]) -> Option<f64> {
    let mut pos = 0usize;
    while let Some((kind, payload_start, payload_end)) = read_mp4_box(bytes, pos, bytes.len()) {
        if kind == *b"moov" {
            return parse_mp4_moov_duration_seconds(bytes, payload_start, payload_end);
        }
        pos = payload_end;
    }
    None
}

fn parse_mp4_moov_duration_seconds(bytes: &[u8], start: usize, end: usize) -> Option<f64> {
    let mut pos = start;
    while let Some((kind, payload_start, payload_end)) = read_mp4_box(bytes, pos, end) {
        if kind == *b"mvhd" {
            return parse_mp4_mvhd_duration_seconds(&bytes[payload_start..payload_end]);
        }
        pos = payload_end;
    }
    None
}

fn read_mp4_box(bytes: &[u8], pos: usize, limit: usize) -> Option<([u8; 4], usize, usize)> {
    if pos.checked_add(8)? > limit || limit > bytes.len() {
        return None;
    }
    let size32 = u32::from_be_bytes(bytes[pos..pos + 4].try_into().ok()?);
    let kind: [u8; 4] = bytes[pos + 4..pos + 8].try_into().ok()?;
    let mut header_len = 8usize;
    let size = match size32 {
        0 => limit.checked_sub(pos)?,
        1 => {
            if pos.checked_add(16)? > limit {
                return None;
            }
            header_len = 16;
            usize::try_from(u64::from_be_bytes(
                bytes[pos + 8..pos + 16].try_into().ok()?,
            ))
            .ok()?
        }
        size => size as usize,
    };
    if size < header_len {
        return None;
    }
    let payload_start = pos.checked_add(header_len)?;
    let payload_end = pos.checked_add(size)?;
    if payload_end > limit || payload_start > payload_end {
        return None;
    }
    Some((kind, payload_start, payload_end))
}

fn parse_mp4_mvhd_duration_seconds(payload: &[u8]) -> Option<f64> {
    let version = *payload.first()?;
    let (timescale_offset, duration_offset, duration_len) = match version {
        0 => (12usize, 16usize, 4usize),
        1 => (20usize, 24usize, 8usize),
        _ => return None,
    };
    let timescale_end = timescale_offset.checked_add(4)?;
    let duration_end = duration_offset.checked_add(duration_len)?;
    if duration_end > payload.len() || timescale_end > payload.len() {
        return None;
    }
    let timescale = u32::from_be_bytes(payload[timescale_offset..timescale_end].try_into().ok()?);
    if timescale == 0 {
        return None;
    }
    let duration = if duration_len == 4 {
        u32::from_be_bytes(payload[duration_offset..duration_end].try_into().ok()?) as f64
    } else {
        u64::from_be_bytes(payload[duration_offset..duration_end].try_into().ok()?) as f64
    };
    let seconds = duration / timescale as f64;
    seconds
        .is_finite()
        .then_some(seconds)
        .filter(|value| *value > 0.0)
}

fn split_png_stream(bytes: &[u8]) -> Result<Vec<&[u8]>, MediaConnectorError> {
    const PNG_SIG: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
    const IEND: &[u8; 4] = b"IEND";

    let mut frames = Vec::new();
    let mut pos = 0;
    while pos < bytes.len() {
        let Some(rel_start) = bytes[pos..]
            .windows(PNG_SIG.len())
            .position(|w| w == PNG_SIG)
        else {
            break;
        };
        let start = pos + rel_start;
        let mut cursor = start + PNG_SIG.len();

        loop {
            let remaining = bytes.len() - cursor;
            if remaining < 12 {
                return Err(MediaConnectorError::VideoDecode(
                    "truncated PNG frame in ffmpeg output".to_string(),
                ));
            }
            let mut len_bytes = [0_u8; 4];
            len_bytes.copy_from_slice(&bytes[cursor..cursor + 4]);
            let len = u32::from_be_bytes(len_bytes) as usize;
            let chunk_type = &bytes[cursor + 4..cursor + 8];
            if remaining - 12 < len {
                return Err(MediaConnectorError::VideoDecode(
                    "truncated PNG chunk in ffmpeg output".to_string(),
                ));
            }
            cursor += 12 + len;
            if chunk_type == IEND {
                frames.push(&bytes[start..cursor]);
                pos = cursor;
                break;
            }
        }
    }

    Ok(frames)
}

#[cfg(test)]
fn parse_ppm_stream(bytes: &[u8]) -> Result<Vec<image::DynamicImage>, MediaConnectorError> {
    let layouts = parse_ppm_frame_layout(bytes)?;
    let mut frames = Vec::with_capacity(layouts.len());
    for layout in layouts {
        let end = layout.offset.checked_add(layout.len).ok_or_else(|| {
            MediaConnectorError::VideoDecode("PPM frame size overflow".to_string())
        })?;
        let image = image::RgbImage::from_raw(
            layout.width,
            layout.height,
            bytes[layout.offset..end].to_vec(),
        )
        .ok_or_else(|| {
            MediaConnectorError::VideoDecode(format!(
                "failed to build RGB frame from {} bytes for {}x{} video",
                layout.len, layout.width, layout.height
            ))
        })?;
        frames.push(image::DynamicImage::ImageRgb8(image));
    }
    Ok(frames)
}

fn parse_ppm_rgb_video(bytes: Bytes) -> Result<DecodedRgbVideo, MediaConnectorError> {
    let layouts = parse_ppm_frame_layout(&bytes)?;
    let decoded_bytes = layouts.iter().try_fold(0usize, |total, frame| {
        total.checked_add(frame.len).ok_or_else(|| {
            MediaConnectorError::VideoDecode("PPM decoded byte size overflow".to_string())
        })
    })?;
    ensure_decoded_byte_limit(decoded_bytes)?;
    Ok(DecodedRgbVideo::new(bytes, layouts))
}

fn parse_ppm_frame_layout(bytes: &[u8]) -> Result<Vec<DecodedRgbFrame>, MediaConnectorError> {
    let mut frames = Vec::new();
    let mut pos = 0;

    while pos < bytes.len() {
        skip_ppm_whitespace_and_comments(bytes, &mut pos);
        if pos >= bytes.len() {
            break;
        }

        let magic = read_ppm_token(bytes, &mut pos)?.ok_or_else(|| {
            MediaConnectorError::VideoDecode("truncated PPM frame header".to_string())
        })?;
        if magic != b"P6" {
            return Err(MediaConnectorError::VideoDecode(format!(
                "unsupported PPM magic: {}",
                String::from_utf8_lossy(magic)
            )));
        }
        let width = parse_ppm_u32(bytes, &mut pos, "width")?;
        let height = parse_ppm_u32(bytes, &mut pos, "height")?;
        let max_value = parse_ppm_u32(bytes, &mut pos, "max value")?;
        if width == 0 || height == 0 {
            return Err(MediaConnectorError::VideoDecode(
                "PPM frame dimensions must be non-zero".to_string(),
            ));
        }
        if max_value != 255 {
            return Err(MediaConnectorError::VideoDecode(format!(
                "unsupported PPM max value: {max_value}"
            )));
        }
        if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
            return Err(MediaConnectorError::VideoDecode(
                "PPM header is not followed by pixel data".to_string(),
            ));
        }
        pos += 1;

        let frame_size = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| {
                MediaConnectorError::VideoDecode(format!(
                    "PPM frame dimensions are too large: {width}x{height}"
                ))
            })?;
        let end = pos.checked_add(frame_size).ok_or_else(|| {
            MediaConnectorError::VideoDecode("PPM frame size overflow".to_string())
        })?;
        if end > bytes.len() {
            return Err(MediaConnectorError::VideoDecode(
                "truncated PPM frame pixel data".to_string(),
            ));
        }
        frames.push(DecodedRgbFrame {
            width,
            height,
            offset: pos,
            len: frame_size,
        });
        pos = end;
    }

    if frames.is_empty() {
        return Err(MediaConnectorError::VideoDecode(
            "ffmpeg produced no frames".to_string(),
        ));
    }

    Ok(frames)
}

fn rawvideo_frame_size(width: u32, height: u32) -> Result<usize, MediaConnectorError> {
    let frame_size = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| {
            MediaConnectorError::VideoDecode(format!(
                "video frame dimensions are too large: {width}x{height}"
            ))
        })?;
    if frame_size == 0 {
        return Err(MediaConnectorError::VideoDecode(
            "video frame dimensions must be non-zero".to_string(),
        ));
    }
    Ok(frame_size)
}

fn parse_ppm_u32(bytes: &[u8], pos: &mut usize, field: &str) -> Result<u32, MediaConnectorError> {
    let token = read_ppm_token(bytes, pos)?
        .ok_or_else(|| MediaConnectorError::VideoDecode(format!("truncated PPM {field} header")))?;
    std::str::from_utf8(token)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .ok_or_else(|| {
            MediaConnectorError::VideoDecode(format!(
                "invalid PPM {field}: {}",
                String::from_utf8_lossy(token)
            ))
        })
}

fn read_ppm_token<'a>(
    bytes: &'a [u8],
    pos: &mut usize,
) -> Result<Option<&'a [u8]>, MediaConnectorError> {
    skip_ppm_whitespace_and_comments(bytes, pos);
    if *pos >= bytes.len() {
        return Ok(None);
    }

    let start = *pos;
    while *pos < bytes.len() && !bytes[*pos].is_ascii_whitespace() {
        if bytes[*pos] == b'#' {
            return Err(MediaConnectorError::VideoDecode(
                "unexpected PPM comment inside token".to_string(),
            ));
        }
        *pos += 1;
    }
    Ok(Some(&bytes[start..*pos]))
}

fn skip_ppm_whitespace_and_comments(bytes: &[u8], pos: &mut usize) {
    loop {
        while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
        if *pos < bytes.len() && bytes[*pos] == b'#' {
            while *pos < bytes.len() && bytes[*pos] != b'\n' {
                *pos += 1;
            }
            continue;
        }
        break;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_ffmpeg_duration_seconds, parse_mp4_duration_seconds, parse_ppm_stream,
        split_png_stream, video_stdout_prealloc_capacity, video_temp_suffix, VideoMetadata,
    };

    const TINY_PNG: &[u8] = &[
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 4,
        0, 0, 0, 181, 28, 12, 2, 0, 0, 0, 11, 73, 68, 65, 84, 120, 218, 99, 96, 96, 0, 0, 0, 3, 0,
        1, 43, 9, 77, 132, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
    ];

    #[test]
    fn splits_concatenated_png_stream() {
        let mut stream = Vec::new();
        stream.extend_from_slice(TINY_PNG);
        stream.extend_from_slice(TINY_PNG);

        let frames = match split_png_stream(&stream) {
            Ok(frames) => frames,
            Err(err) => panic!("split png stream failed: {err}"),
        };
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], TINY_PNG);
        assert_eq!(frames[1], TINY_PNG);
    }

    #[test]
    fn parses_ffmpeg_duration() {
        let stderr = "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'video.mp4':\n  Duration: 00:01:23.45, start: 0.000000, bitrate: 123 kb/s";
        assert_eq!(parse_ffmpeg_duration_seconds(stderr), Some(83.45));
    }

    fn mp4_box(kind: [u8; 4], payload: Vec<u8>) -> Vec<u8> {
        let size = u32::try_from(payload.len() + 8).unwrap();
        let mut bytes = Vec::with_capacity(size as usize);
        bytes.extend_from_slice(&size.to_be_bytes());
        bytes.extend_from_slice(&kind);
        bytes.extend_from_slice(&payload);
        bytes
    }

    #[test]
    fn parses_mp4_mvhd_v0_duration() {
        let mut mvhd = vec![0; 20];
        mvhd[12..16].copy_from_slice(&1_000u32.to_be_bytes());
        mvhd[16..20].copy_from_slice(&2_000u32.to_be_bytes());
        let mut file = mp4_box(*b"ftyp", b"isom".to_vec());
        file.extend_from_slice(&mp4_box(*b"moov", mp4_box(*b"mvhd", mvhd)));

        assert_eq!(parse_mp4_duration_seconds(&file), Some(2.0));
    }

    #[test]
    fn parses_mp4_mvhd_v1_duration() {
        let mut mvhd = vec![0; 32];
        mvhd[0] = 1;
        mvhd[20..24].copy_from_slice(&90_000u32.to_be_bytes());
        mvhd[24..32].copy_from_slice(&180_000u64.to_be_bytes());
        let file = mp4_box(*b"moov", mp4_box(*b"mvhd", mvhd));

        assert_eq!(parse_mp4_duration_seconds(&file), Some(2.0));
    }

    #[test]
    fn preallocates_video_stdout_only_with_known_duration() {
        let known = VideoMetadata {
            width: 16,
            height: 16,
            duration_seconds: Some(1.0),
        };
        let unknown = VideoMetadata {
            width: 16,
            height: 16,
            duration_seconds: None,
        };

        assert_eq!(video_stdout_prealloc_capacity(known, 4096), 4096);
        assert_eq!(video_stdout_prealloc_capacity(unknown, 4096), 0);
    }

    #[test]
    fn detects_video_temp_suffix_from_container_header() {
        let mut mp4 = vec![0; 12];
        mp4[4..8].copy_from_slice(b"ftyp");
        assert_eq!(video_temp_suffix(&mp4), ".mp4");
        assert_eq!(video_temp_suffix(&[0x1a, 0x45, 0xdf, 0xa3]), ".webm");
        assert_eq!(video_temp_suffix(b"RIFF....AVI "), ".avi");
        assert_eq!(video_temp_suffix(b"OggS"), ".ogv");
        assert_eq!(video_temp_suffix(&[0x00, 0x00, 0x01, 0xba]), ".mpg");
        assert_eq!(video_temp_suffix(b"unknown"), ".video");
    }

    #[test]
    fn parses_concatenated_ppm_stream() {
        let stream = b"P6\n2 1\n255\n\x01\x02\x03\x04\x05\x06P6\n# comment\n1 2\n255\n\x07\x08\x09\x0a\x0b\x0c";

        let frames = match parse_ppm_stream(stream) {
            Ok(frames) => frames,
            Err(err) => panic!("parse ppm stream failed: {err}"),
        };
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].width(), 2);
        assert_eq!(frames[0].height(), 1);
        assert_eq!(frames[1].width(), 1);
        assert_eq!(frames[1].height(), 2);
    }

    #[test]
    fn rejects_truncated_ppm_stream() {
        assert!(parse_ppm_stream(b"P6\n2 1\n255\n\x01\x02").is_err());
    }

    #[test]
    fn rejects_invalid_ppm_header() {
        assert!(parse_ppm_stream(b"P3\n1 1\n255\n\x01\x02\x03").is_err());
        assert!(parse_ppm_stream(b"P6\n1 1\n65535\n\x01\x02\x03").is_err());
    }

    #[test]
    fn rejects_zero_dimension_ppm_stream() {
        assert!(parse_ppm_stream(b"P6\n0 1\n255\n").is_err());
        assert!(parse_ppm_stream(b"P6\n1 0\n255\n").is_err());
    }

    #[test]
    fn rejects_overflowing_ppm_frame_size() {
        assert!(parse_ppm_stream(b"P6\n4294967295 4294967295\n255\n").is_err());
    }

    #[cfg(feature = "opencv-video")]
    #[test]
    fn opencv_sampling_preserves_min_frames_for_short_clips() {
        let cfg = super::VideoFetchConfig {
            min_frames: 4,
            max_frames: 8,
            sample_fps: 2.0,
        };
        let indices = super::opencv_frame_indices(1, 30.0, cfg);
        assert_eq!(indices, vec![0, 0, 0, 0]);
    }
}
