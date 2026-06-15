use std::{collections::HashMap, fmt, path::PathBuf, sync::Arc};

use image::{DynamicImage, RgbImage};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Supported multimodal modalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Image,
    ImageEmbeds,
    Audio,
    Video,
}

impl fmt::Display for Modality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Modality::Image => write!(f, "image"),
            Modality::ImageEmbeds => write!(f, "image_embeds"),
            Modality::Audio => write!(f, "audio"),
            Modality::Video => write!(f, "video"),
        }
    }
}

/// Detail level passed by OpenAI style APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImageDetail {
    #[default]
    Auto,
    Low,
    High,
}

/// A normalized content part understood by the tracker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MediaContentPart {
    Text {
        text: String,
    },
    ImageUrl {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
    },
    ImageData {
        data: Vec<u8>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
    ImageEmbeds {
        payload: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
    },
    VideoUrl {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
    },
    VideoData {
        data: Vec<u8>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
    },
}

/// Image source metadata (useful for hashing & tracing).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageSource {
    Url { url: String },
    DataUrl,
    InlineBytes,
    File { path: PathBuf },
}

/// Video source metadata (useful for hashing & tracing).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VideoSource {
    Url { url: String },
    DataUrl,
    InlineBytes,
    File { path: PathBuf },
}

/// Concrete image payload captured by the media connector.
#[derive(Debug, Clone)]
pub struct ImageFrame {
    pub image: DynamicImage,
    pub raw_bytes: bytes::Bytes,
    pub detail: ImageDetail,
    pub source: ImageSource,
    /// Blake3 hex-digest of raw_bytes, computed at decode time.
    pub hash: String,
}

/// Decoded video payload captured by the media connector.
#[derive(Debug, Clone)]
pub struct VideoClip {
    pub frames: Vec<DynamicImage>,
    pub rgb_video: Option<DecodedRgbVideo>,
    pub raw_bytes: bytes::Bytes,
    pub source: VideoSource,
    /// Blake3 hex-digest of raw_bytes, computed at decode time.
    pub hash: String,
}

/// Borrowed RGB frame data for video preprocessors.
#[derive(Debug, Clone, Copy)]
pub struct RgbFrameRef<'a> {
    pub width: u32,
    pub height: u32,
    pub data: &'a [u8],
}

/// One decoded RGB frame inside a shared decoded-video byte buffer.
#[derive(Debug, Clone)]
pub struct DecodedRgbFrame {
    pub width: u32,
    pub height: u32,
    pub offset: usize,
    pub len: usize,
}

/// Decoded RGB video frames backed by one shared byte buffer.
#[derive(Debug, Clone)]
pub struct DecodedRgbVideo {
    pub data: bytes::Bytes,
    pub frames: Vec<DecodedRgbFrame>,
}

impl DecodedRgbVideo {
    pub fn new(data: bytes::Bytes, frames: Vec<DecodedRgbFrame>) -> Self {
        Self { data, frames }
    }

    pub fn frame_refs(&self) -> Result<Vec<RgbFrameRef<'_>>, String> {
        self.frames
            .iter()
            .map(|frame| {
                let end = frame
                    .offset
                    .checked_add(frame.len)
                    .ok_or_else(|| "decoded RGB frame offset overflow".to_string())?;
                let data = self
                    .data
                    .get(frame.offset..end)
                    .ok_or_else(|| "decoded RGB frame range is out of bounds".to_string())?;
                Ok(RgbFrameRef {
                    width: frame.width,
                    height: frame.height,
                    data,
                })
            })
            .collect()
    }

    pub fn to_dynamic_images(&self) -> Result<Vec<DynamicImage>, String> {
        let mut images = Vec::with_capacity(self.frames.len());
        for frame in &self.frames {
            let end = frame
                .offset
                .checked_add(frame.len)
                .ok_or_else(|| "decoded RGB frame offset overflow".to_string())?;
            let data = self
                .data
                .get(frame.offset..end)
                .ok_or_else(|| "decoded RGB frame range is out of bounds".to_string())?;
            let image =
                RgbImage::from_raw(frame.width, frame.height, data.to_vec()).ok_or_else(|| {
                    format!(
                        "failed to build RGB frame from {} bytes for {}x{} video",
                        frame.len, frame.width, frame.height
                    )
                })?;
            images.push(DynamicImage::ImageRgb8(image));
        }
        Ok(images)
    }
}

impl VideoClip {
    pub fn new(
        frames: Vec<DynamicImage>,
        raw_bytes: bytes::Bytes,
        source: VideoSource,
        hash: String,
    ) -> Self {
        Self {
            frames,
            rgb_video: None,
            raw_bytes,
            source,
            hash,
        }
    }

    pub fn new_rgb(
        rgb_video: DecodedRgbVideo,
        raw_bytes: bytes::Bytes,
        source: VideoSource,
        hash: String,
    ) -> Self {
        Self {
            frames: Vec::new(),
            rgb_video: Some(rgb_video),
            raw_bytes,
            source,
            hash,
        }
    }

    pub fn frames(&self) -> &[DynamicImage] {
        &self.frames
    }

    pub fn rgb_video(&self) -> Option<&DecodedRgbVideo> {
        self.rgb_video.as_ref()
    }

    pub fn materialized_frames(&self) -> Result<Vec<DynamicImage>, String> {
        if !self.frames.is_empty() {
            return Ok(self.frames.clone());
        }
        self.rgb_video
            .as_ref()
            .ok_or_else(|| "video clip has no decoded frames".to_string())?
            .to_dynamic_images()
    }

    pub fn raw_bytes(&self) -> &[u8] {
        &self.raw_bytes
    }

    pub fn source(&self) -> &VideoSource {
        &self.source
    }
}

impl ImageFrame {
    pub fn new(
        image: DynamicImage,
        raw_bytes: bytes::Bytes,
        detail: ImageDetail,
        source: ImageSource,
        hash: String,
    ) -> Self {
        Self {
            image,
            raw_bytes,
            detail,
            source,
            hash,
        }
    }

    pub fn data(&self) -> &DynamicImage {
        &self.image
    }

    pub fn raw_bytes(&self) -> &[u8] {
        &self.raw_bytes
    }

    pub fn source(&self) -> &ImageSource {
        &self.source
    }

    pub fn size(&self) -> ImageSize {
        ImageSize::new(self.image.width(), self.image.height())
    }
}

/// Container for all supported multimodal media objects.
#[derive(Debug, Clone)]
pub enum TrackedMedia {
    Image(Arc<ImageFrame>),
    Video(Arc<VideoClip>),
    /// Placeholder variants for future modalities.
    Audio,
    Embeddings,
}

pub type MultiModalData = HashMap<Modality, Vec<TrackedMedia>>;
pub type MultiModalUUIDs = HashMap<Modality, Vec<Option<String>>>;

pub type TokenId = i32;

/// Declares how a multimodal tensor's first dimension maps to media items.
///
/// Used by [`ModelProcessorSpec::field_layouts`] to tell the backend how to
/// split tensors for per-item scheduling (vLLM `MultiModalFieldConfig`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldLayout {
    /// First dimension equals number of media items (one slice per item).
    Batched,
    /// Variable-length slices per item. The sizes are stored in the tensor
    /// named by `sizes_key` (e.g. `"patches_per_image"` or `"patches_per_video"`).
    Flat { sizes_key: String },
}

impl FieldLayout {
    /// Convenience constructor for `Flat`.
    pub fn flat(sizes_key: impl Into<String>) -> Self {
        Self::Flat {
            sizes_key: sizes_key.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

impl ImageSize {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlaceholderRange {
    pub offset: usize,
    pub length: usize,
}

#[derive(Debug, Clone)]
pub struct PromptReplacement {
    pub modality: Modality,
    pub placeholder_token: String,
    pub tokens: Vec<TokenId>,
}

impl PromptReplacement {
    pub fn repeated(
        modality: Modality,
        placeholder_token: &str,
        token_id: TokenId,
        count: usize,
    ) -> Self {
        Self {
            modality,
            placeholder_token: placeholder_token.to_string(),
            tokens: vec![token_id; count],
        }
    }

    pub fn sequence(modality: Modality, placeholder_token: &str, sequence: Vec<TokenId>) -> Self {
        Self {
            modality,
            placeholder_token: placeholder_token.to_string(),
            tokens: sequence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_range_serializes() {
        let range = PlaceholderRange {
            offset: 10,
            length: 4,
        };
        let json = serde_json::to_string(&range).unwrap();
        assert!(json.contains("offset"));
    }

    #[test]
    fn prompt_replacement_builders() {
        let rep = PromptReplacement::repeated(Modality::Image, "<image>", 100, 3);
        assert_eq!(rep.tokens, vec![100, 100, 100]);
    }
}
