use std::{
    collections::HashSet,
    io::{Read, Write},
    path::PathBuf,
    process::{Command, Output, Stdio},
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use bytes::Bytes;
#[cfg(feature = "opencv-video")]
use opencv::{
    core::{Mat, Scalar, CV_8UC3},
    imgproc,
    prelude::*,
    videoio,
};
use reqwest::Client;
use tokio::{fs, task};
use tracing::info;
use url::Url;

use super::{
    error::MediaConnectorError,
    types::{
        DecodedRgbFrame, DecodedRgbVideo, ImageDetail, ImageFrame, ImageSource, VideoClip,
        VideoSource,
    },
};

const DEFAULT_VIDEO_PROCESS_TIMEOUT: Duration = Duration::from_secs(30);
const VIDEO_PROCESS_POLL_INTERVAL: Duration = Duration::from_millis(10);
const VIDEO_PROCESS_TIMEOUT_ENV: &str = "SMG_VIDEO_PROCESS_TIMEOUT_SECS";

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
        let decoded = BASE64_STANDARD.decode(data)?;
        self.decode_image(decoded.into(), cfg.detail, ImageSource::DataUrl)
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
        let decoded = BASE64_STANDARD.decode(data)?;
        self.decode_video(decoded.into(), cfg, VideoSource::DataUrl)
            .await
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

        let bytes = fs::read(&canonical).await?;
        self.decode_video(bytes.into(), cfg, VideoSource::File { path: canonical })
            .await
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

        let cursor = std::io::Cursor::new(bytes.clone());
        let reader = image::ImageReader::new(cursor).with_guessed_format()?;

        let image = task::spawn_blocking(move || reader.decode())
            .await
            .map_err(MediaConnectorError::Blocking)??;

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
        if cfg.sample_fps <= 0.0 {
            return Err(MediaConnectorError::VideoDecode(
                "sample_fps must be greater than 0".to_string(),
            ));
        }

        let hash = crate::hasher::hash_video(&bytes);
        let input = bytes.clone();
        let source_path = match &source {
            VideoSource::File { path } => Some(path.clone()),
            _ => None,
        };
        let decoded =
            task::spawn_blocking(move || decode_video_frames(&input, cfg, source_path.as_deref()))
                .await
                .map_err(MediaConnectorError::Blocking)??;

        let clip = match decoded {
            DecodedVideoFrames::Images(frames) => VideoClip::new(frames, bytes, source, hash),
            DecodedVideoFrames::Rgb(rgb_video) => {
                VideoClip::new_rgb(rgb_video, bytes, source, hash)
            }
        };
        Ok(Arc::new(clip))
    }
}

enum DecodedVideoFrames {
    Images(Vec<image::DynamicImage>),
    Rgb(DecodedRgbVideo),
}

fn decode_video_frames(
    bytes: &[u8],
    cfg: VideoFetchConfig,
    source_path: Option<&std::path::Path>,
) -> Result<DecodedVideoFrames, MediaConnectorError> {
    match video_decode_backend_override().as_deref() {
        Some("ffmpeg") => return decode_video_with_ffmpeg(bytes, cfg, source_path),
        Some("opencv") => {
            #[cfg(feature = "opencv-video")]
            {
                return decode_video_with_opencv_logged(bytes, cfg, source_path);
            }
            #[cfg(not(feature = "opencv-video"))]
            {
                return Err(MediaConnectorError::VideoDecode(
                    "SMG_VIDEO_DECODE_BACKEND=opencv requires the opencv-video feature".to_string(),
                ));
            }
        }
        Some(backend) => {
            return Err(MediaConnectorError::VideoDecode(format!(
                "unsupported SMG_VIDEO_DECODE_BACKEND={backend}; expected auto, opencv, or ffmpeg"
            )));
        }
        None => {
            #[cfg(feature = "opencv-video")]
            {
                match decode_video_with_opencv_logged(bytes, cfg, source_path) {
                    Ok(frames) => return Ok(frames),
                    Err(opencv_error) => {
                        if log_video_decode_timing_enabled() {
                            info!(
                                error = %opencv_error,
                                "smg_mm_timing video_decode_auto_opencv_fallback"
                            );
                        }

                        return match decode_video_with_ffmpeg(bytes, cfg, source_path) {
                            Ok(frames) => Ok(frames),
                            Err(ffmpeg_error) => Err(MediaConnectorError::VideoDecode(format!(
                                "OpenCV decode failed: {opencv_error}; ffmpeg fallback failed: {ffmpeg_error}"
                            ))),
                        };
                    }
                }
            }

            #[cfg(not(feature = "opencv-video"))]
            {
                return decode_video_with_ffmpeg(bytes, cfg, source_path);
            }
        }
    }
}

#[cfg(feature = "opencv-video")]
fn decode_video_with_opencv_logged(
    bytes: &[u8],
    cfg: VideoFetchConfig,
    source_path: Option<&std::path::Path>,
) -> Result<DecodedVideoFrames, MediaConnectorError> {
    let started = Instant::now();
    let result = decode_video_with_opencv(bytes, cfg, source_path);
    match &result {
        Ok(_) => log_video_decode_backend_timing("opencv", started, bytes.len(), cfg, None),
        Err(error) => {
            log_video_decode_backend_timing("opencv", started, bytes.len(), cfg, Some(error))
        }
    }
    result
}

fn video_decode_backend_override() -> Option<String> {
    let backend = std::env::var("SMG_VIDEO_DECODE_BACKEND")
        .ok()?
        .trim()
        .to_ascii_lowercase();
    match backend.as_str() {
        "" | "auto" => None,
        _ => Some(backend),
    }
}

fn log_video_decode_timing_enabled() -> bool {
    std::env::var("SMG_LOG_MM_TIMING")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn log_video_decode_backend_timing(
    backend: &str,
    started: Instant,
    input_bytes: usize,
    cfg: VideoFetchConfig,
    error: Option<&MediaConnectorError>,
) {
    if !log_video_decode_timing_enabled() {
        return;
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
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
fn decode_video_with_opencv(
    bytes: &[u8],
    cfg: VideoFetchConfig,
    source_path: Option<&std::path::Path>,
) -> Result<DecodedVideoFrames, MediaConnectorError> {
    if let Some(path) = source_path {
        return decode_video_with_opencv_file(path, cfg);
    }

    let input_file = write_temp_video_file(bytes)?;
    decode_video_with_opencv_file(input_file.path(), cfg)
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

    let width = capture
        .get(videoio::CAP_PROP_FRAME_WIDTH)
        .map_err(opencv_decode_error)?
        .round()
        .max(0.0) as u32;
    let height = capture
        .get(videoio::CAP_PROP_FRAME_HEIGHT)
        .map_err(opencv_decode_error)?
        .round()
        .max(0.0) as u32;
    if width == 0 || height == 0 {
        return Err(MediaConnectorError::VideoDecode(format!(
            "OpenCV reported invalid video frame size: {width}x{height}"
        )));
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

    let frame_index_set = frame_indices.iter().copied().collect::<HashSet<_>>();
    let max_frame_index = *frame_indices.last().ok_or_else(|| {
        MediaConnectorError::VideoDecode("OpenCV video sampling produced no frames".to_string())
    })?;
    let frame_size = rawvideo_frame_size(width, height)?;
    let mut data = Vec::with_capacity(frame_indices.len() * frame_size);
    let mut frames = Vec::with_capacity(frame_indices.len());
    let mut bgr_frame = Mat::default();
    let mut rgb_frame =
        Mat::new_rows_cols_with_default(height as i32, width as i32, CV_8UC3, Scalar::default())
            .map_err(opencv_decode_error)?;

    for idx in 0..=max_frame_index {
        if !capture.grab().map_err(opencv_decode_error)? {
            continue;
        }

        if !frame_index_set.contains(&idx) {
            continue;
        }

        if !capture
            .retrieve(&mut bgr_frame, 0)
            .map_err(opencv_decode_error)?
            || bgr_frame.empty()
        {
            continue;
        }

        imgproc::cvt_color_def(&bgr_frame, &mut rgb_frame, imgproc::COLOR_BGR2RGB)
            .map_err(opencv_decode_error)?;

        let rgb_bytes = rgb_frame.data_bytes().map_err(opencv_decode_error)?;
        if rgb_bytes.len() < frame_size {
            return Err(MediaConnectorError::VideoDecode(format!(
                "OpenCV produced {} RGB bytes for {width}x{height} frame, expected {frame_size}",
                rgb_bytes.len()
            )));
        }
        let offset = data.len();
        data.extend_from_slice(&rgb_bytes[..frame_size]);
        frames.push(DecodedRgbFrame {
            width,
            height,
            offset,
            len: frame_size,
        });
    }

    if frames.is_empty() {
        return Err(MediaConnectorError::VideoDecode(
            "OpenCV produced no readable sampled frames".to_string(),
        ));
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
    target_frames = target_frames.clamp(1, total_frames);

    if target_frames >= total_frames {
        return (0..total_frames).collect();
    }
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
fn opencv_decode_error(err: opencv::Error) -> MediaConnectorError {
    MediaConnectorError::VideoDecode(format!("OpenCV video decode failed: {err}"))
}

fn decode_video_with_ffmpeg(
    bytes: &[u8],
    cfg: VideoFetchConfig,
    source_path: Option<&std::path::Path>,
) -> Result<DecodedVideoFrames, MediaConnectorError> {
    let input_path = match source_path {
        Some(path) => FfmpegInputPath::Existing(path),
        None => FfmpegInputPath::Temp(write_temp_video_file(bytes)?),
    };
    let input_path = input_path.path();

    if let Ok(metadata) = probe_video_metadata(input_path) {
        let started = Instant::now();
        match decode_video_with_ffmpeg_ppm(input_path, cfg, metadata) {
            Ok(frames) => {
                log_video_decode_backend_timing("ffmpeg_ppm_file", started, bytes.len(), cfg, None);
                return Ok(DecodedVideoFrames::Images(frames));
            }
            Err(error) => {
                log_video_decode_backend_timing(
                    "ffmpeg_ppm_file",
                    started,
                    bytes.len(),
                    cfg,
                    Some(&error),
                );
            }
        }

        let started = Instant::now();
        match decode_video_with_ffmpeg_raw(input_path, cfg, metadata) {
            Ok(frames) => {
                log_video_decode_backend_timing("ffmpeg_raw_file", started, bytes.len(), cfg, None);
                return Ok(DecodedVideoFrames::Images(frames));
            }
            Err(error) => {
                log_video_decode_backend_timing(
                    "ffmpeg_raw_file",
                    started,
                    bytes.len(),
                    cfg,
                    Some(&error),
                );
            }
        }
    }

    let started = Instant::now();
    match decode_video_with_ffmpeg_png(input_path, cfg) {
        Ok(frames) => {
            log_video_decode_backend_timing("ffmpeg_png_file", started, bytes.len(), cfg, None);
            Ok(DecodedVideoFrames::Images(frames))
        }
        Err(error) => {
            log_video_decode_backend_timing(
                "ffmpeg_png_file",
                started,
                bytes.len(),
                cfg,
                Some(&error),
            );
            Err(error)
        }
    }
}

enum FfmpegInputPath<'a> {
    Existing(&'a std::path::Path),
    Temp(tempfile::NamedTempFile),
}

impl<'a> FfmpegInputPath<'a> {
    fn path(&'a self) -> &'a std::path::Path {
        match self {
            Self::Existing(path) => path,
            Self::Temp(file) => file.path(),
        }
    }
}

fn write_temp_video_file(bytes: &[u8]) -> Result<tempfile::NamedTempFile, MediaConnectorError> {
    let started = Instant::now();
    let mut input_file = tempfile::Builder::new()
        .prefix("smg-video-")
        .suffix(video_temp_suffix(bytes))
        .tempfile()?;
    input_file.write_all(bytes)?;
    input_file.flush()?;
    if log_video_decode_timing_enabled() {
        info!(
            nbytes = bytes.len(),
            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
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

fn decode_video_with_ffmpeg_ppm(
    input_path: &std::path::Path,
    cfg: VideoFetchConfig,
    metadata: VideoMetadata,
) -> Result<Vec<image::DynamicImage>, MediaConnectorError> {
    let fps_filter = fps_filter_for_metadata(metadata, cfg);
    let max_frames = cfg.max_frames.to_string();
    let mut command = Command::new("ffmpeg");
    command
        .args(["-hide_banner", "-nostdin", "-loglevel", "error", "-i"])
        .arg(input_path)
        .args([
            "-vf",
            &fps_filter,
            "-frames:v",
            &max_frames,
            "-f",
            "image2pipe",
            "-vcodec",
            "ppm",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]);
    let output = run_video_command_output(command, "ffmpeg")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(MediaConnectorError::VideoDecode(format!(
            "ffmpeg failed: {stderr}"
        )));
    }

    parse_ppm_stream(&output.stdout)
}

fn decode_video_with_ffmpeg_raw(
    input_path: &std::path::Path,
    cfg: VideoFetchConfig,
    metadata: VideoMetadata,
) -> Result<Vec<image::DynamicImage>, MediaConnectorError> {
    let fps_filter = fps_filter_for_metadata(metadata, cfg);
    let max_frames = cfg.max_frames.to_string();
    let mut command = Command::new("ffmpeg");
    command
        .args(["-hide_banner", "-nostdin", "-loglevel", "error", "-i"])
        .arg(input_path)
        .args([
            "-vf",
            &fps_filter,
            "-frames:v",
            &max_frames,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]);
    let output = run_video_command_output(command, "ffmpeg")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(MediaConnectorError::VideoDecode(format!(
            "ffmpeg failed: {stderr}"
        )));
    }

    let frame_size = rawvideo_frame_size(metadata.width, metadata.height)?;
    let mut chunks = output.stdout.chunks_exact(frame_size);
    let mut frames = Vec::with_capacity(output.stdout.len() / frame_size);
    for chunk in chunks.by_ref() {
        let image = image::RgbImage::from_raw(metadata.width, metadata.height, chunk.to_vec())
            .ok_or_else(|| {
                MediaConnectorError::VideoDecode(format!(
                    "failed to build RGB frame from {} bytes for {}x{} video",
                    frame_size, metadata.width, metadata.height
                ))
            })?;
        frames.push(image::DynamicImage::ImageRgb8(image));
    }
    if !chunks.remainder().is_empty() {
        return Err(MediaConnectorError::VideoDecode(format!(
            "ffmpeg rawvideo output has trailing partial frame: {} bytes",
            chunks.remainder().len()
        )));
    }
    if frames.is_empty() {
        return Err(MediaConnectorError::VideoDecode(
            "ffmpeg produced no frames".to_string(),
        ));
    }
    Ok(frames)
}

fn decode_video_with_ffmpeg_png(
    input_path: &std::path::Path,
    cfg: VideoFetchConfig,
) -> Result<Vec<image::DynamicImage>, MediaConnectorError> {
    let fps_filter = fps_filter_for_video(input_path, cfg);
    let max_frames = cfg.max_frames.to_string();
    let mut command = Command::new("ffmpeg");
    command
        .args(["-hide_banner", "-nostdin", "-loglevel", "error", "-i"])
        .arg(input_path)
        .args([
            "-vf",
            &fps_filter,
            "-frames:v",
            &max_frames,
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ]);
    let output = run_video_command_output(command, "ffmpeg")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(MediaConnectorError::VideoDecode(format!(
            "ffmpeg failed: {stderr}"
        )));
    }

    let pngs = split_png_stream(&output.stdout)?;
    let mut frames = Vec::with_capacity(pngs.len());
    for png in pngs {
        frames.push(image::load_from_memory(png)?);
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

fn probe_video_metadata(
    input_path: &std::path::Path,
) -> Result<VideoMetadata, MediaConnectorError> {
    let mut command = Command::new("ffprobe");
    command
        .args([
            "-v",
            "error",
            "-nostdin",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(input_path);
    let output = run_video_command_output(command, "ffprobe")?;

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
    if let Some(duration) = metadata.duration_seconds {
        if let Some(filter) = fps_filter_for_duration(duration, cfg) {
            return filter;
        }
    }

    format!("fps={}", cfg.sample_fps)
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

fn fps_filter_for_video(input_path: &std::path::Path, cfg: VideoFetchConfig) -> String {
    if let Ok(duration) = probe_video_duration_seconds(input_path) {
        if let Some(filter) = fps_filter_for_duration(duration, cfg) {
            return filter;
        }
    }

    format!("fps={}", cfg.sample_fps)
}

fn probe_video_duration_seconds(input_path: &std::path::Path) -> Result<f64, MediaConnectorError> {
    let mut command = Command::new("ffprobe");
    command
        .args([
            "-v",
            "error",
            "-nostdin",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(input_path);
    match run_video_command_output(command, "ffprobe") {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            stdout.trim().parse::<f64>().map_err(|err| {
                MediaConnectorError::VideoDecode(format!("failed to parse ffprobe duration: {err}"))
            })
        }
        Ok(_) | Err(_) => probe_video_duration_seconds_with_ffmpeg(input_path),
    }
}

fn probe_video_duration_seconds_with_ffmpeg(
    input_path: &std::path::Path,
) -> Result<f64, MediaConnectorError> {
    let mut command = Command::new("ffmpeg");
    command
        .args(["-hide_banner", "-nostdin", "-i"])
        .arg(input_path);
    let output = run_video_command_output(command, "ffmpeg")?;

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

fn video_process_timeout() -> Duration {
    std::env::var(VIDEO_PROCESS_TIMEOUT_ENV)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_secs)
        .filter(|timeout| *timeout > Duration::ZERO)
        .unwrap_or(DEFAULT_VIDEO_PROCESS_TIMEOUT)
}

fn run_video_command_output(
    mut command: Command,
    program: &'static str,
) -> Result<Output, MediaConnectorError> {
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command.spawn().map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            MediaConnectorError::VideoDecode(format!(
                "{program} executable not found; install ffmpeg to decode video_url inputs"
            ))
        } else {
            MediaConnectorError::Io(error)
        }
    })?;

    let mut stdout = child.stdout.take().ok_or_else(|| {
        MediaConnectorError::VideoDecode(format!("{program} stdout pipe was not captured"))
    })?;
    let mut stderr = child.stderr.take().ok_or_else(|| {
        MediaConnectorError::VideoDecode(format!("{program} stderr pipe was not captured"))
    })?;

    let stdout_handle = thread::spawn(move || {
        let mut buffer = Vec::new();
        stdout.read_to_end(&mut buffer).map(|_| buffer)
    });
    let stderr_handle = thread::spawn(move || {
        let mut buffer = Vec::new();
        stderr.read_to_end(&mut buffer).map(|_| buffer)
    });

    let timeout = video_process_timeout();
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(MediaConnectorError::VideoDecode(format!(
                "{program} timed out after {:.1}s",
                timeout.as_secs_f64()
            )));
        }
        thread::sleep(VIDEO_PROCESS_POLL_INTERVAL);
    };

    let stdout = stdout_handle.join().map_err(|_| {
        MediaConnectorError::VideoDecode(format!("{program} stdout reader panicked"))
    })??;
    let stderr = stderr_handle.join().map_err(|_| {
        MediaConnectorError::VideoDecode(format!("{program} stderr reader panicked"))
    })??;

    Ok(Output {
        status,
        stdout,
        stderr,
    })
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
        parse_ffmpeg_duration_seconds, parse_ppm_stream, split_png_stream, video_temp_suffix,
    };

    const TINY_PNG: &[u8] = &[
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 4,
        0, 0, 0, 181, 28, 12, 2, 0, 0, 0, 11, 73, 68, 65, 84, 120, 218, 99, 96, 96, 0, 0, 0, 3, 0,
        1, 43, 9, 141, 84, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
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
}
