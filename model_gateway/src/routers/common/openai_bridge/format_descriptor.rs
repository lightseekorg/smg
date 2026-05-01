//! Per-format dispatch table — one row per `ResponseFormat`.
//!
//! Routers read fields off [`FormatDescriptor`] instead of matching on the
//! enum directly, so adding a new format is one row here and zero edits in
//! router code.

use openai_protocol::event_types::{
    CodeInterpreterCallEvent, FileSearchCallEvent, ImageGenerationCallEvent, ItemType, McpEvent,
    WebSearchCallEvent,
};

use super::ResponseFormat;

#[derive(Debug, Clone, Copy)]
pub struct FormatDescriptor {
    pub type_str: &'static str,
    pub id_prefix: &'static str,
    pub in_progress_event: &'static str,
    /// `None` for formats with no intermediate phase (e.g. Passthrough).
    pub searching_event: Option<&'static str>,
    pub completed_event: &'static str,
    pub streams_arguments: bool,
    /// `None` for formats without a partial-image-style intermediate frame.
    pub partial_image_event: Option<&'static str>,
}

pub const fn descriptor(format: ResponseFormat) -> FormatDescriptor {
    match format {
        ResponseFormat::WebSearchCall => FormatDescriptor {
            type_str: ItemType::WEB_SEARCH_CALL,
            id_prefix: "ws_",
            in_progress_event: WebSearchCallEvent::IN_PROGRESS,
            searching_event: Some(WebSearchCallEvent::SEARCHING),
            completed_event: WebSearchCallEvent::COMPLETED,
            streams_arguments: false,
            partial_image_event: None,
        },
        ResponseFormat::CodeInterpreterCall => FormatDescriptor {
            type_str: ItemType::CODE_INTERPRETER_CALL,
            id_prefix: "ci_",
            in_progress_event: CodeInterpreterCallEvent::IN_PROGRESS,
            searching_event: Some(CodeInterpreterCallEvent::INTERPRETING),
            completed_event: CodeInterpreterCallEvent::COMPLETED,
            streams_arguments: false,
            partial_image_event: None,
        },
        ResponseFormat::FileSearchCall => FormatDescriptor {
            type_str: ItemType::FILE_SEARCH_CALL,
            id_prefix: "fs_",
            in_progress_event: FileSearchCallEvent::IN_PROGRESS,
            searching_event: Some(FileSearchCallEvent::SEARCHING),
            completed_event: FileSearchCallEvent::COMPLETED,
            streams_arguments: false,
            partial_image_event: None,
        },
        ResponseFormat::ImageGenerationCall => FormatDescriptor {
            type_str: ItemType::IMAGE_GENERATION_CALL,
            id_prefix: "ig_",
            in_progress_event: ImageGenerationCallEvent::IN_PROGRESS,
            searching_event: Some(ImageGenerationCallEvent::GENERATING),
            completed_event: ImageGenerationCallEvent::COMPLETED,
            streams_arguments: false,
            partial_image_event: Some(ImageGenerationCallEvent::PARTIAL_IMAGE),
        },
        ResponseFormat::Passthrough => FormatDescriptor {
            type_str: ItemType::MCP_CALL,
            id_prefix: "mcp_",
            in_progress_event: McpEvent::CALL_IN_PROGRESS,
            searching_event: None,
            completed_event: McpEvent::CALL_COMPLETED,
            streams_arguments: true,
            partial_image_event: None,
        },
    }
}
