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
use reqwest::Client;
use tokio::{fs, task};
use url::Url;

use super::{
    error::MediaConnectorError,
    types::{ImageDetail, ImageFrame, ImageSource, VideoClip, VideoSource},
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
        let frames = task::spawn_blocking(move || decode_video_with_ffmpeg(&input, cfg))
            .await
            .map_err(MediaConnectorError::Blocking)??;

        Ok(Arc::new(VideoClip::new(frames, bytes, source, hash)))
    }
}

fn decode_video_with_ffmpeg(
    bytes: &[u8],
    cfg: VideoFetchConfig,
) -> Result<Vec<image::DynamicImage>, MediaConnectorError> {
    let mut input_file = tempfile::Builder::new()
        .prefix("smg-video-")
        .suffix(".mp4")
        .tempfile()?;
    input_file.write_all(bytes)?;
    input_file.flush()?;

    let fps_filter = fps_filter_for_video(input_file.path(), cfg);
    let max_frames = cfg.max_frames.to_string();
    let mut command = Command::new("ffmpeg");
    command
        .args(["-hide_banner", "-nostdin", "-loglevel", "error", "-i"])
        .arg(input_file.path())
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

fn fps_filter_for_video(input_path: &std::path::Path, cfg: VideoFetchConfig) -> String {
    if let Ok(duration) = probe_video_duration_seconds(input_path) {
        if duration.is_finite() && duration > 0.0 {
            let target_frames = (duration * cfg.sample_fps as f64)
                .round()
                .clamp(cfg.min_frames as f64, cfg.max_frames as f64);
            let fps = (target_frames / duration).max(f64::EPSILON);
            return format!("fps={fps:.6}");
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
    command.args(["-hide_banner", "-nostdin", "-i"]).arg(input_path);
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

#[cfg(test)]
mod tests {
    use super::{parse_ffmpeg_duration_seconds, split_png_stream};

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
}
