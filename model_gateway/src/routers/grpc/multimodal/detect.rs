//! Multimodal content detection and extraction.
//!
//! Both the chat completion pipeline (`ChatMessage`) and the Messages API
//! pipeline (`InputMessage`) funnel into the shared processing core; only the
//! detection and extraction differ, because the input message types differ.

use llm_multimodal::{ImageDetail, MediaContentPart, Modality};
use openai_protocol::{
    chat::{ChatMessage, MessageContent},
    common::ContentPart,
    messages::{ImageSource, InputContent, InputContentBlock, InputMessage, Role},
};

/// Return the multimodal modalities present in OpenAI chat messages.
pub(crate) fn chat_modalities(messages: &[ChatMessage]) -> Vec<Modality> {
    let mut modalities = Vec::new();
    let mut push_unique = |modality| {
        if !modalities.contains(&modality) {
            modalities.push(modality);
        }
    };

    for msg in messages {
        let content = match msg {
            ChatMessage::User { content, .. } => Some(content),
            ChatMessage::System { content, .. } => Some(content),
            ChatMessage::Developer { content, .. } => Some(content),
            ChatMessage::Tool { content, .. } => Some(content),
            _ => None,
        };

        if let Some(MessageContent::Parts(parts)) = content {
            for part in parts {
                match part {
                    ContentPart::ImageUrl { .. } => push_unique(Modality::Image),
                    ContentPart::VideoUrl { .. } => push_unique(Modality::Video),
                    ContentPart::Text { .. } => {}
                }
            }
        }
    }

    modalities
}

/// Check if any messages in the request contain multimodal content.
#[cfg(test)]
pub(crate) fn has_multimodal_content(messages: &[ChatMessage]) -> bool {
    !chat_modalities(messages).is_empty()
}

/// Extract multimodal content parts from OpenAI chat messages,
/// converting protocol `ContentPart` to multimodal crate `MediaContentPart`.
pub(super) fn extract_content_parts(messages: &[ChatMessage]) -> Vec<MediaContentPart> {
    let mut parts = Vec::new();

    for msg in messages {
        let content = match msg {
            ChatMessage::User { content, .. } => Some(content),
            ChatMessage::System { content, .. } => Some(content),
            ChatMessage::Developer { content, .. } => Some(content),
            ChatMessage::Tool { content, .. } => Some(content),
            _ => None,
        };

        if let Some(MessageContent::Parts(message_parts)) = content {
            for part in message_parts {
                match part {
                    ContentPart::ImageUrl { image_url } => {
                        let detail = image_url.detail.as_deref().and_then(parse_detail);
                        parts.push(MediaContentPart::ImageUrl {
                            url: image_url.url.clone(),
                            detail,
                            uuid: None,
                        });
                    }
                    ContentPart::Text { text } => {
                        parts.push(MediaContentPart::Text { text: text.clone() });
                    }
                    ContentPart::VideoUrl { video_url } => {
                        parts.push(MediaContentPart::VideoUrl {
                            url: video_url.url.clone(),
                            uuid: None,
                        });
                    }
                }
            }
        }
    }

    parts
}

/// Parse OpenAI detail string to multimodal ImageDetail enum.
fn parse_detail(detail: &str) -> Option<ImageDetail> {
    match detail.to_ascii_lowercase().as_str() {
        "auto" => Some(ImageDetail::Auto),
        "low" => Some(ImageDetail::Low),
        "high" => Some(ImageDetail::High),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Messages API multimodal detection and extraction
// ---------------------------------------------------------------------------

/// Check if any messages in a Messages API request contain multimodal content.
pub(crate) fn has_multimodal_content_messages(messages: &[InputMessage]) -> bool {
    messages.iter().any(|msg| {
        if msg.role != Role::User {
            return false;
        }
        match &msg.content {
            InputContent::Blocks(blocks) => blocks
                .iter()
                .any(|block| matches!(block, InputContentBlock::Image(_))),
            InputContent::String(_) => false,
        }
    })
}

/// Extract multimodal content parts from Messages API input messages,
/// converting `InputContentBlock::Image` to multimodal crate `MediaContentPart`.
pub(super) fn extract_content_parts_messages(messages: &[InputMessage]) -> Vec<MediaContentPart> {
    let mut parts = Vec::new();

    for msg in messages {
        if msg.role != Role::User {
            continue;
        }
        let blocks = match &msg.content {
            InputContent::Blocks(blocks) => blocks,
            InputContent::String(_) => continue,
        };

        for block in blocks {
            match block {
                InputContentBlock::Image(image_block) => match &image_block.source {
                    ImageSource::Base64 { media_type, data } => {
                        // Convert base64 to data URL for the media connector
                        let data_url = format!("data:{media_type};base64,{data}");
                        parts.push(MediaContentPart::ImageUrl {
                            url: data_url,
                            detail: None,
                            uuid: None,
                        });
                    }
                    ImageSource::Url { url } => {
                        parts.push(MediaContentPart::ImageUrl {
                            url: url.clone(),
                            detail: None,
                            uuid: None,
                        });
                    }
                },
                InputContentBlock::Text(text_block) => {
                    parts.push(MediaContentPart::Text {
                        text: text_block.text.clone(),
                    });
                }
                _ => {}
            }
        }
    }

    parts
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{ImageUrl, VideoUrl};

    use super::*;

    #[test]
    fn test_has_multimodal_content_with_images() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "What is this?".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/cat.jpg".to_string(),
                        detail: None,
                    },
                },
            ]),
            name: None,
        }];

        assert!(has_multimodal_content(&messages));
    }

    #[test]
    fn test_has_multimodal_content_with_video() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![ContentPart::VideoUrl {
                video_url: VideoUrl {
                    url: "https://example.com/clip.mp4".to_string(),
                },
            }]),
            name: None,
        }];

        assert!(has_multimodal_content(&messages));
        assert_eq!(chat_modalities(&messages), vec![Modality::Video]);
    }

    #[test]
    fn test_has_multimodal_content_text_only() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Text("Hello".to_string()),
            name: None,
        }];

        assert!(!has_multimodal_content(&messages));
    }

    #[test]
    fn test_has_multimodal_content_parts_text_only() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![ContentPart::Text {
                text: "Just text".to_string(),
            }]),
            name: None,
        }];

        assert!(!has_multimodal_content(&messages));
    }

    #[test]
    fn test_extract_content_parts() {
        let messages = vec![
            ChatMessage::System {
                content: MessageContent::Text("You are helpful".to_string()),
                name: None,
            },
            ChatMessage::User {
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "Describe this:".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image.jpg".to_string(),
                            detail: Some("high".to_string()),
                        },
                    },
                ]),
                name: None,
            },
        ];

        let parts = extract_content_parts(&messages);
        assert_eq!(parts.len(), 2);

        match &parts[0] {
            MediaContentPart::Text { text } => assert_eq!(text, "Describe this:"),
            _ => panic!("Expected Text part"),
        }

        match &parts[1] {
            MediaContentPart::ImageUrl { url, detail, .. } => {
                assert_eq!(url, "https://example.com/image.jpg");
                assert_eq!(*detail, Some(ImageDetail::High));
            }
            _ => panic!("Expected ImageUrl part"),
        }
    }

    #[test]
    fn test_extract_video_content_parts() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![ContentPart::VideoUrl {
                video_url: VideoUrl {
                    url: "https://example.com/video.mp4".to_string(),
                },
            }]),
            name: None,
        }];

        let parts = extract_content_parts(&messages);
        assert_eq!(parts.len(), 1);
        match &parts[0] {
            MediaContentPart::VideoUrl { url, .. } => {
                assert_eq!(url, "https://example.com/video.mp4");
            }
            _ => panic!("Expected VideoUrl part"),
        }
    }

    #[test]
    fn test_parse_detail() {
        assert_eq!(parse_detail("auto"), Some(ImageDetail::Auto));
        assert_eq!(parse_detail("Auto"), Some(ImageDetail::Auto));
        assert_eq!(parse_detail("LOW"), Some(ImageDetail::Low));
        assert_eq!(parse_detail("high"), Some(ImageDetail::High));
        assert_eq!(parse_detail("unknown"), None);
    }
}
