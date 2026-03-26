use std::{collections::HashSet, path::PathBuf, sync::Arc, time::Duration};

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use bytes::Bytes;
use reqwest::Client;
use tokio::{fs, task};
use url::Url;

use super::{
    error::MediaConnectorError,
    types::{ImageDetail, ImageFrame, ImageSource},
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

        let decode_bytes = bytes.clone();
        let image = task::spawn_blocking(move || Self::decode_bytes(decode_bytes))
            .await
            .map_err(MediaConnectorError::Blocking)??;

        Ok(Arc::new(ImageFrame::new(
            image, bytes, detail, source, hash,
        )))
    }

    /// Decode raw bytes into a [`DynamicImage`].
    ///
    /// When the `fast-jpeg` feature is enabled and the bytes start with the
    /// JPEG magic bytes (0xFF 0xD8), [`zune_jpeg`] is used for decoding which
    /// is significantly faster than the pure-Rust decoder bundled with the
    /// `image` crate.  On failure, we fall back to the generic decoder.
    fn decode_bytes(bytes: Bytes) -> Result<image::DynamicImage, image::ImageError> {
        #[cfg(feature = "fast-jpeg")]
        if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xD8 {
            if let Ok(img) = Self::decode_jpeg_fast(&bytes) {
                return Ok(img);
            }
            // Fall through to generic decoder on failure.
        }

        let cursor = std::io::Cursor::new(bytes);
        let reader = image::ImageReader::new(cursor).with_guessed_format()?;
        reader.decode()
    }

    /// Fast JPEG decoding via `zune-jpeg`.
    #[cfg(feature = "fast-jpeg")]
    fn decode_jpeg_fast(bytes: &[u8]) -> Result<image::DynamicImage, Box<dyn std::error::Error>> {
        use image::{DynamicImage, GrayImage, RgbImage};
        use zune_jpeg::JpegDecoder;

        let cursor = std::io::Cursor::new(bytes);
        let mut decoder = JpegDecoder::new(cursor);
        let pixels = decoder.decode()?;
        let info = decoder.info().ok_or("missing JPEG header info")?;
        let width = info.width as u32;
        let height = info.height as u32;
        let components = info.components as u32;

        match components {
            1 => {
                let img = GrayImage::from_raw(width, height, pixels)
                    .ok_or("gray buffer size mismatch")?;
                Ok(DynamicImage::ImageLuma8(img))
            }
            3 => {
                let img =
                    RgbImage::from_raw(width, height, pixels).ok_or("RGB buffer size mismatch")?;
                Ok(DynamicImage::ImageRgb8(img))
            }
            _ => Err(format!("unsupported JPEG component count: {components}").into()),
        }
    }
}
