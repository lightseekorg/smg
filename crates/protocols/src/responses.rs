// OpenAI Responses API types
// https://platform.openai.com/docs/api-reference/responses

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::{Validate, ValidationError};

use super::{
    common::{
        default_true, validate_stop, ChatLogProbs, ContextManagementEntry, Detail, Function,
        FunctionChoice, GenerationRequest, PromptCacheRetention, PromptTokenUsageInfo,
        ResponsePrompt, StreamOptions, StringOrArray, ToolChoice as ChatToolChoice,
        ToolChoiceValue as ChatToolChoiceValue, ToolReference, UsageInfo,
    },
    sampling_params::{validate_top_k_value, validate_top_p_value},
};
use crate::{
    builders::ResponsesResponseBuilder, skills::ResponsesSkillEntry, validated::Normalizable,
};

// ============================================================================
// Responses API Tool Choice
// ============================================================================

/// Simple tool-choice option strings supported by the Responses API.
///
/// Spec: `tool_choice` may be the bare string `"none"`, `"auto"`, or `"required"`.
/// Shared chat/responses semantics but the Responses type owns its own enum so
/// chat-path validators cannot accept unknown Responses variants by accident.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceOptions {
    None,
    Auto,
    Required,
}

/// Single-value tag used to force the `"type": "function"` discriminator on the
/// flat function-selection variant so the untagged outer enum can distinguish
/// `Function` from `AllowedTools` / `Mcp` / `Custom` / `Types`.
///
/// Responses spec: `{"type": "function", "name": "..."}` — note the **flat**
/// shape (no nested `function` object). This differs from Chat Completions
/// which wraps the name in `{"function": {"name": "..."}}`.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, schemars::JsonSchema, PartialEq, Eq)]
pub enum FunctionToolChoiceTag {
    #[serde(rename = "function")]
    Function,
}

/// Tag enum forcing the `"type": "allowed_tools"` discriminator.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, schemars::JsonSchema, PartialEq, Eq)]
pub enum AllowedToolsToolChoiceTag {
    #[serde(rename = "allowed_tools")]
    AllowedTools,
}

/// Tag enum forcing the `"type": "mcp"` discriminator.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, schemars::JsonSchema, PartialEq, Eq)]
pub enum McpToolChoiceTag {
    #[serde(rename = "mcp")]
    Mcp,
}

/// Tag enum forcing the `"type": "custom"` discriminator.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, schemars::JsonSchema, PartialEq, Eq)]
pub enum CustomToolChoiceTag {
    #[serde(rename = "custom")]
    Custom,
}

/// Tag enum forcing the `"type": "apply_patch"` discriminator.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, schemars::JsonSchema, PartialEq, Eq)]
pub enum ApplyPatchToolChoiceTag {
    #[serde(rename = "apply_patch")]
    ApplyPatch,
}

/// Tag enum forcing the `"type": "shell"` discriminator.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, schemars::JsonSchema, PartialEq, Eq)]
pub enum ShellToolChoiceTag {
    #[serde(rename = "shell")]
    Shell,
}

/// Built-in (hosted) tool types that can be referenced directly by
/// `tool_choice: {"type": "..."}` in the Responses API.
///
/// Each variant is a distinct string per the spec — we keep multiple
/// `web_search_preview*` forms because the spec enumerates them separately
/// and older clients may still send the versioned aliases.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BuiltInToolChoiceType {
    FileSearch,
    WebSearch,
    WebSearchPreview,
    #[serde(rename = "web_search_preview_2025_03_11")]
    WebSearchPreview20250311,
    ImageGeneration,
    ComputerUsePreview,
    CodeInterpreter,
}

/// Canonical payload for the Responses API `Function` tool-choice variant.
///
/// Serializes as the spec-flat shape `{"type": "function", "name": "..."}`.
///
/// Deserialization accepts **both** wire shapes for backward compatibility
/// with smg clients written against the pre-split shared `ToolChoice` type
/// (which used the Chat-style nested `{"function": {"name": "..."}}` layout
/// on the Responses endpoint):
///
/// * Canonical flat: `{"type": "function", "name": "..."}`
/// * Legacy nested:  `{"type": "function", "function": {"name": "..."}}`
///
/// Either shape normalizes to `name: String` at deserialize time so the rest
/// of the gateway only ever sees the canonical form.
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct ResponsesFunctionToolChoice {
    #[serde(rename = "type")]
    pub tool_type: FunctionToolChoiceTag,
    pub name: String,
}

impl<'de> Deserialize<'de> for ResponsesFunctionToolChoice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Accept either the spec-flat `{type, name}` or the legacy
        // Chat-nested `{type, function: {name}}` shape. The helper lets
        // both fields be absent so serde can bind whichever wire form
        // the caller sent; we then pick one and fail loudly if neither
        // provides a function name.
        #[derive(Deserialize)]
        struct Helper {
            #[serde(rename = "type")]
            tool_type: FunctionToolChoiceTag,
            #[serde(default)]
            name: Option<String>,
            #[serde(default)]
            function: Option<FunctionChoice>,
        }

        let helper = Helper::deserialize(deserializer)?;
        let name = helper
            .name
            .or_else(|| helper.function.map(|f| f.name))
            .ok_or_else(|| {
                serde::de::Error::custom(
                    "tool_choice function requires a `name` field or a `function.name` field",
                )
            })?;
        Ok(Self {
            tool_type: helper.tool_type,
            name,
        })
    }
}

/// `tool_choice` accepted on the Responses API (`POST /v1/responses`).
///
/// The Responses spec enumerates eight concrete wire shapes, each with a
/// distinct discriminator — see `ResponsesToolChoice` variants below.
/// Deserialised via `#[serde(untagged)]` because the outermost JSON is
/// either a bare string (`Options`) or an object whose `"type"` picks
/// the variant.
///
/// Each object variant pins the discriminator through a single-value tag
/// enum (`FunctionToolChoiceTag`, etc.) so serde cannot match a payload
/// whose `type` does not belong to that variant. Without the tag pinning,
/// the `#[serde(untagged)]` enum would accept any object shape that
/// happened to fit the field set of an earlier variant.
///
/// This type deliberately does NOT live in `common.rs`: Chat Completions
/// has its own `ToolChoice` with a different `Function` wire shape
/// (nested `{"function": {"name": ...}}`) and does not accept the
/// `Types` / `Mcp` / `Custom` / `ApplyPatch` / `Shell` variants at all.
/// Sharing one enum across both APIs would silently accept spec-invalid
/// payloads on `/v1/chat/completions`.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ResponsesToolChoice {
    /// `"none"` | `"auto"` | `"required"`.
    Options(ToolChoiceOptions),

    /// `{"type": "file_search" | "web_search" | "web_search_preview" |
    /// "web_search_preview_2025_03_11" | "image_generation" |
    /// "computer_use_preview" | "code_interpreter"}` — select a built-in
    /// hosted tool by type alone (no additional payload).
    Types {
        #[serde(rename = "type")]
        tool_type: BuiltInToolChoiceType,
    },

    /// `{"type": "function", "name": "..."}` — Responses spec flat shape.
    ///
    /// Accepts both the spec-canonical flat wire shape and the legacy
    /// Chat-style nested shape (`{"type": "function", "function": {"name": "..."}}`)
    /// on deserialize to preserve backward compatibility with smg clients
    /// written against the pre-split shared `ToolChoice` type. Always
    /// serializes as the canonical flat shape per the OpenAI Responses spec
    /// — Postel's law: liberal on input, conservative on output.
    ///
    /// The nested legacy shape is gated behind a custom `Deserialize` impl on
    /// `ResponsesFunctionToolChoice`; the untagged outer enum still pins the
    /// `"type": "function"` discriminator via `FunctionToolChoiceTag` so
    /// payloads without that tag cannot reach this variant.
    Function(ResponsesFunctionToolChoice),

    /// `{"type": "allowed_tools", "mode": "auto"|"required", "tools": [...]}`.
    ///
    /// `tools` is an array of `ToolReference` items — the same type reused
    /// from Chat's Allowed Tools payload because the Responses spec also
    /// allows function / mcp / file_search / web_search_preview /
    /// computer_use_preview / code_interpreter / image_generation entries.
    AllowedTools {
        #[serde(rename = "type")]
        tool_type: AllowedToolsToolChoiceTag,
        /// `"auto"` or `"required"`. Validated at request-normalisation time
        /// (see `validate_tool_choice_with_tools`).
        mode: String,
        tools: Vec<ToolReference>,
    },

    /// `{"type": "mcp", "server_label": "...", "name"?: "..."}` — force
    /// routing to a specific MCP server, optionally pinning a tool name.
    Mcp {
        #[serde(rename = "type")]
        tool_type: McpToolChoiceTag,
        server_label: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },

    /// `{"type": "custom", "name": "..."}` — pin a user-registered
    /// custom tool by name.
    Custom {
        #[serde(rename = "type")]
        tool_type: CustomToolChoiceTag,
        name: String,
    },

    /// `{"type": "apply_patch"}` — force the built-in `apply_patch` tool.
    ApplyPatch {
        #[serde(rename = "type")]
        tool_type: ApplyPatchToolChoiceTag,
    },

    /// `{"type": "shell"}` — force the built-in `shell` tool.
    Shell {
        #[serde(rename = "type")]
        tool_type: ShellToolChoiceTag,
    },
}

impl Default for ResponsesToolChoice {
    fn default() -> Self {
        Self::Options(ToolChoiceOptions::Auto)
    }
}

impl ResponsesToolChoice {
    /// Serialize tool_choice to string for ResponsesResponse payloads.
    ///
    /// Returns the JSON-serialized tool_choice or `"auto"` as default.
    pub fn serialize_to_string(tool_choice: Option<&ResponsesToolChoice>) -> String {
        tool_choice
            .map(|tc| serde_json::to_string(tc).unwrap_or_else(|_| "auto".to_string()))
            .unwrap_or_else(|| "auto".to_string())
    }

    /// Return the pinned function name for the `Function` variant, regardless
    /// of which wire shape (spec-flat `name` or legacy nested `function.name`)
    /// was used at deserialize time. `None` for any non-`Function` variant.
    ///
    /// Consumers that need to project / validate the function name should go
    /// through this accessor rather than pattern-matching so future wire
    /// shapes can be added without touching call sites.
    pub fn function_name(&self) -> Option<&str> {
        match self {
            Self::Function(payload) => Some(payload.name.as_str()),
            _ => None,
        }
    }

    /// Project the Responses-level tool_choice onto a Chat Completions
    /// tool_choice when a Responses request is being routed through the
    /// Chat Completions gRPC pipeline.
    ///
    /// Mapping rules:
    /// - `Options(None|Auto|Required)` → `ChatToolChoice::Value(...)` — shared
    ///   semantics.
    /// - `Function { name }` → `ChatToolChoice::Function { nested name }` —
    ///   shape translation (flat Responses → nested Chat wire form).
    /// - `AllowedTools { mode, tools }` → `ChatToolChoice::AllowedTools {...}`.
    /// - Hosted / custom / apply_patch / shell / mcp — Chat Completions has no
    ///   equivalent spec variant; fall back to `Auto` so the downstream chat
    ///   backend still runs with tool-calling enabled.
    pub fn to_chat_tool_choice(&self) -> ChatToolChoice {
        match self {
            Self::Options(ToolChoiceOptions::None) => {
                ChatToolChoice::Value(ChatToolChoiceValue::None)
            }
            Self::Options(ToolChoiceOptions::Auto) => {
                ChatToolChoice::Value(ChatToolChoiceValue::Auto)
            }
            Self::Options(ToolChoiceOptions::Required) => {
                ChatToolChoice::Value(ChatToolChoiceValue::Required)
            }
            Self::Function(payload) => ChatToolChoice::Function {
                tool_type: "function".to_string(),
                function: FunctionChoice {
                    name: payload.name.clone(),
                },
            },
            Self::AllowedTools { mode, tools, .. } => ChatToolChoice::AllowedTools {
                tool_type: "allowed_tools".to_string(),
                mode: mode.clone(),
                tools: tools.clone(),
            },
            // No matching Chat spec variant — fall through to `auto` so
            // downstream Chat backends still see tool-calling enabled.
            Self::Types { .. }
            | Self::Mcp { .. }
            | Self::Custom { .. }
            | Self::ApplyPatch { .. }
            | Self::Shell { .. } => ChatToolChoice::Value(ChatToolChoiceValue::Auto),
        }
    }
}

// ============================================================================
// Response Tools (MCP and others)
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponseTool {
    /// Function tool.
    #[serde(rename = "function")]
    Function(FunctionTool),

    /// Built-in tool.
    #[serde(rename = "web_search_preview")]
    WebSearchPreview(WebSearchPreviewTool),

    /// Built-in tool.
    #[serde(rename = "code_interpreter")]
    CodeInterpreter(CodeInterpreterTool),

    /// MCP server tool.
    #[serde(rename = "mcp")]
    Mcp(McpTool),

    /// Built-in file search tool over vector stores.
    #[serde(rename = "file_search")]
    FileSearch(FileSearchTool),
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FunctionTool {
    /// Flatten to match Responses API tool JSON shape.
    #[serde(flatten)]
    pub function: Function,
}

/// File search tool configuration.
///
/// Spec: `{ type: "file_search", vector_store_ids, filters?, max_num_results?, ranking_options? }`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FileSearchTool {
    /// Vector store IDs to search over.
    pub vector_store_ids: Vec<String>,
    /// Optional filter applied to candidate documents.
    pub filters: Option<FileSearchFilter>,
    /// Maximum number of results to return.
    pub max_num_results: Option<u32>,
    /// Ranking options for the search.
    pub ranking_options: Option<FileSearchRankingOptions>,
}

/// Filter expression for file search.
///
/// Either a single comparison or a boolean compound (`and` / `or`) over nested filters.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
pub enum FileSearchFilter {
    #[serde(rename = "eq")]
    Eq(ComparisonFilter),
    #[serde(rename = "ne")]
    Ne(ComparisonFilter),
    #[serde(rename = "gt")]
    Gt(ComparisonFilter),
    #[serde(rename = "gte")]
    Gte(ComparisonFilter),
    #[serde(rename = "lt")]
    Lt(ComparisonFilter),
    #[serde(rename = "lte")]
    Lte(ComparisonFilter),
    #[serde(rename = "in")]
    In(ComparisonFilter),
    #[serde(rename = "nin")]
    Nin(ComparisonFilter),
    #[serde(rename = "and")]
    And(CompoundFilter),
    #[serde(rename = "or")]
    Or(CompoundFilter),
}

/// Key/value comparison used by the `eq`/`ne`/`gt`/`gte`/`lt`/`lte`/`in`/`nin` filter variants.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ComparisonFilter {
    pub key: String,
    /// Spec allows `string | number | boolean | array of string | number`.
    pub value: Value,
}

/// Boolean composition over nested filters (used by `and` / `or` variants).
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CompoundFilter {
    pub filters: Vec<FileSearchFilter>,
}

/// Ranking options for file search results.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FileSearchRankingOptions {
    pub hybrid_search: Option<HybridSearchOptions>,
    pub ranker: Option<FileSearchRanker>,
    pub score_threshold: Option<f64>,
}

/// Weights combining embedding-based and text-based similarity.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HybridSearchOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_weight: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_weight: Option<f64>,
}

/// Ranker selection for file search.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
pub enum FileSearchRanker {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "default-2024-11-15")]
    Default20241115,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct McpTool {
    pub server_url: Option<String>,
    pub authorization: Option<String>,
    /// Custom headers to send to MCP server (from request payload, not HTTP headers)
    pub headers: Option<HashMap<String, String>>,
    pub server_label: String,
    pub server_description: Option<String>,
    /// Approval requirement configuration for MCP tools.
    pub require_approval: Option<RequireApproval>,
    pub allowed_tools: Option<Vec<String>>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Default, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WebSearchPreviewTool {
    pub search_context_size: Option<String>,
    pub user_location: Option<Value>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Default, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CodeInterpreterTool {
    pub container: Option<Value>,
    pub environment: Option<ResponseToolEnvironment>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Default, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResponseToolEnvironment {
    pub skills: Option<Vec<ResponsesSkillEntry>>,
}

/// `require_approval` values.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(untagged)]
pub enum RequireApproval {
    Mode(RequireApprovalMode),
    Rules(RequireApprovalRules),
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RequireApprovalMode {
    Always,
    Never,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RequireApprovalRules {
    pub always: Option<RequireApprovalFilter>,
    pub never: Option<RequireApprovalFilter>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RequireApprovalFilter {
    pub tool_names: Option<Vec<String>>,
    pub read_only: Option<bool>,
}

// ============================================================================
// Reasoning Parameters
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ResponseReasoningParam {
    #[serde(default = "default_reasoning_effort")]
    pub effort: Option<ReasoningEffort>,
    pub summary: Option<ReasoningSummary>,
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default function must match field type Option<T>"
)]
fn default_reasoning_effort() -> Option<ReasoningEffort> {
    Some(ReasoningEffort::Medium)
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningSummary {
    Auto,
    Concise,
    Detailed,
}

// ============================================================================
// Input/Output Items
// ============================================================================

/// Content can be either a simple string or array of content parts (for SimpleInputMessage)
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum StringOrContentParts {
    String(String),
    Array(Vec<ResponseContentPart>),
}

/// Phase label for assistant messages in the Responses API.
///
/// For gpt-5.3-codex+ multi-turn conversations, preserving and resending the
/// original `phase` value avoids quality / latency degradation — the model
/// relies on it to disambiguate commentary-style reasoning from the final
/// answer. Opaque to SMG otherwise; preserved verbatim through store+retrieve,
/// SSE output, and upstream passthrough.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MessagePhase {
    Commentary,
    FinalAnswer,
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponseInputOutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        role: String,
        content: Vec<ResponseContentPart>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        /// Optional phase label, preserved from previous assistant output so
        /// gpt-5.3-codex+ multi-turn does not degrade (spec: ResponseOutputMessage.phase).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        phase: Option<MessagePhase>,
    },
    #[serde(rename = "reasoning")]
    #[non_exhaustive]
    Reasoning {
        id: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        summary: Vec<SummaryTextContent>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        #[serde(default)]
        content: Vec<ResponseReasoningContent>,
        /// Encrypted reasoning payload for gpt-5 / o-series round-trip via
        /// `previous_response_id`. Opaque to SMG; preserved verbatim.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(default)]
        encrypted_content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(rename = "function_call")]
    FunctionToolCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        id: Option<String>,
        call_id: String,
        output: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(rename = "mcp_approval_request")]
    McpApprovalRequest {
        id: String,
        server_label: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "mcp_approval_response")]
    McpApprovalResponse {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        approval_request_id: String,
        approve: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    #[serde(untagged)]
    SimpleInputMessage {
        content: StringOrContentParts,
        role: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(rename = "type")]
        r#type: Option<String>,
        /// Optional phase label (spec: EasyInputMessage.phase).
        ///
        /// Preserved through conversation storage so gpt-5.3-codex+ does not
        /// lose the commentary/final_answer distinction across turns.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        phase: Option<MessagePhase>,
    },
}

/// Detail level for [`ResponseContentPart::InputFile`]. Spec restricts this
/// to `"low" | "high"` (defaults to `low`); it is narrower than [`Detail`]
/// used for images which also admits `auto` / `original`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FileDetail {
    #[default]
    Low,
    High,
}

/// Typed annotation attached to [`ResponseContentPart::OutputText`]. Matches
/// the OpenAI Responses API `Annotation` union; a `type` discriminator selects
/// the variant on the wire.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Annotation {
    /// `type: "file_citation"` — points at a file previously uploaded.
    FileCitation {
        file_id: String,
        filename: String,
        index: u32,
    },
    /// `type: "url_citation"` — citation back to a URL in a web-search result.
    UrlCitation {
        url: String,
        title: String,
        start_index: u32,
        end_index: u32,
    },
    /// `type: "container_file_citation"` — citation to a file inside a
    /// code-interpreter / computer-use container.
    ContainerFileCitation {
        container_id: String,
        file_id: String,
        filename: String,
        start_index: u32,
        end_index: u32,
    },
    /// `type: "file_path"` — reference to a generated file path.
    FilePath { file_id: String, index: u32 },
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponseContentPart {
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        #[serde(default)]
        #[serde(skip_serializing_if = "Vec::is_empty")]
        annotations: Vec<Annotation>,
        #[serde(skip_serializing_if = "Option::is_none")]
        logprobs: Option<ChatLogProbs>,
    },
    #[serde(rename = "input_text")]
    InputText { text: String },
    /// `type: "input_image"` — reference to an image supplied by the client.
    /// Exactly one of `file_id` / `image_url` is typically set; both may be
    /// absent when only `detail` is being conveyed.
    #[serde(rename = "input_image")]
    InputImage {
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<Detail>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
    },
    /// `type: "input_file"` — reference to an attached file. `file_data` is a
    /// base64 blob; `file_url` / `file_id` reference external/uploaded files.
    #[serde(rename = "input_file")]
    InputFile {
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<FileDetail>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
    /// `type: "refusal"` — model refusal surfaced as a content part (spec's
    /// `ResponseOutputRefusal`).
    #[serde(rename = "refusal")]
    Refusal { refusal: String },
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponseReasoningContent {
    #[serde(rename = "reasoning_text")]
    ReasoningText { text: String },
}

/// Tagged content element carried in `Reasoning.summary`.
///
/// OpenAI spec: `summary: array of SummaryTextContent { text, type: "summary_text" }`.
/// Replaces the prior `Vec<String>` wire-type that broke bidirectional
/// interoperability with spec-compliant clients.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum SummaryTextContent {
    #[serde(rename = "summary_text")]
    SummaryText { text: String },
}

/// MCP Tool information for the mcp_list_tools output item
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct McpToolInfo {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
    pub annotations: Option<Value>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponseOutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        role: String,
        content: Vec<ResponseContentPart>,
        status: String,
        /// Optional phase label (spec: ResponseOutputMessage.phase).
        ///
        /// Labels assistant messages; for gpt-5.3-codex+ we must preserve and
        /// resend this on subsequent turns to avoid perf degradation.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        phase: Option<MessagePhase>,
    },
    #[serde(rename = "reasoning")]
    #[non_exhaustive]
    Reasoning {
        id: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        summary: Vec<SummaryTextContent>,
        content: Vec<ResponseReasoningContent>,
        /// Encrypted reasoning payload for gpt-5 / o-series round-trip.
        /// Opaque to SMG; preserved verbatim.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(default)]
        encrypted_content: Option<String>,
        status: Option<String>,
    },
    #[serde(rename = "function_call")]
    FunctionToolCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        output: Option<String>,
        status: String,
    },
    #[serde(rename = "mcp_list_tools")]
    McpListTools {
        id: String,
        server_label: String,
        tools: Vec<McpToolInfo>,
    },
    #[serde(rename = "mcp_call")]
    McpCall {
        id: String,
        status: String,
        approval_request_id: Option<String>,
        arguments: String,
        error: Option<String>,
        name: String,
        output: String,
        server_label: String,
    },
    #[serde(rename = "mcp_approval_request")]
    McpApprovalRequest {
        id: String,
        server_label: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "web_search_call")]
    WebSearchCall {
        id: String,
        status: WebSearchCallStatus,
        action: WebSearchAction,
    },
    #[serde(rename = "code_interpreter_call")]
    CodeInterpreterCall {
        id: String,
        status: CodeInterpreterCallStatus,
        container_id: String,
        code: Option<String>,
        outputs: Option<Vec<CodeInterpreterOutput>>,
    },
    #[serde(rename = "file_search_call")]
    FileSearchCall {
        id: String,
        status: FileSearchCallStatus,
        queries: Vec<String>,
        results: Option<Vec<FileSearchResult>>,
    },
}

// ============================================================================
// Built-in Tool Call Types
// ============================================================================

/// Status for web search tool calls.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchCallStatus {
    InProgress,
    Searching,
    Completed,
    Failed,
}

/// Action performed during a web search.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WebSearchAction {
    Search {
        #[serde(skip_serializing_if = "Option::is_none")]
        query: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        queries: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        sources: Vec<WebSearchSource>,
    },
    OpenPage {
        url: String,
    },
    Find {
        url: String,
        pattern: String,
    },
}

/// A source returned from web search.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct WebSearchSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub url: String,
}

/// Status for code interpreter tool calls.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CodeInterpreterCallStatus {
    InProgress,
    Completed,
    Incomplete,
    Interpreting,
    Failed,
}

/// Output from code interpreter execution.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CodeInterpreterOutput {
    Logs { logs: String },
    Image { url: String },
}

/// Status for file search tool calls.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FileSearchCallStatus {
    InProgress,
    Searching,
    Completed,
    Incomplete,
    Failed,
}

/// A result from file search.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct FileSearchResult {
    pub file_id: String,
    pub filename: String,
    pub text: Option<String>,
    pub score: Option<f32>,
    pub attributes: Option<Value>,
}

// ============================================================================
// Configuration Enums
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, Default, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(rename = "ResponsesServiceTier")]
pub enum ServiceTier {
    #[default]
    Auto,
    Default,
    Flex,
    Scale,
    Priority,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Truncation {
    Auto,
    #[default]
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ResponseStatus {
    Queued,
    InProgress,
    Completed,
    Incomplete,
    Failed,
    Cancelled,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ReasoningInfo {
    pub effort: Option<String>,
    pub summary: Option<String>,
}

// ============================================================================
// Text Format (structured outputs)
// ============================================================================

/// Text configuration for structured output requests
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct TextConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<TextFormat>,
}

/// Text format: text (default), json_object (legacy), or json_schema (recommended)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
pub enum TextFormat {
    #[serde(rename = "text")]
    Text,

    #[serde(rename = "json_object")]
    JsonObject,

    #[serde(rename = "json_schema")]
    JsonSchema {
        name: String,
        schema: Value,
        description: Option<String>,
        strict: Option<bool>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum IncludeField {
    #[serde(rename = "code_interpreter_call.outputs")]
    CodeInterpreterCallOutputs,
    #[serde(rename = "computer_call_output.output.image_url")]
    ComputerCallOutputImageUrl,
    #[serde(rename = "file_search_call.results")]
    FileSearchCallResults,
    #[serde(rename = "message.input_image.image_url")]
    MessageInputImageUrl,
    #[serde(rename = "message.output_text.logprobs")]
    MessageOutputTextLogprobs,
    #[serde(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
    #[serde(rename = "web_search_call.action.sources")]
    WebSearchCallActionSources,
    #[serde(rename = "web_search_call.results")]
    WebSearchCallResults,
}

// ============================================================================
// Usage Types (Responses API format)
// ============================================================================

/// OpenAI Responses API usage format (different from standard UsageInfo)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ResponseUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    pub input_tokens_details: Option<InputTokensDetails>,
    pub output_tokens_details: Option<OutputTokensDetails>,
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ResponsesUsage {
    Classic(UsageInfo),
    Modern(ResponseUsage),
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct InputTokensDetails {
    pub cached_tokens: u32,
}

impl From<&PromptTokenUsageInfo> for InputTokensDetails {
    fn from(d: &PromptTokenUsageInfo) -> Self {
        Self {
            cached_tokens: d.cached_tokens,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: u32,
}

impl UsageInfo {
    /// Convert to OpenAI Responses API format
    pub fn to_response_usage(&self) -> ResponseUsage {
        ResponseUsage {
            input_tokens: self.prompt_tokens,
            output_tokens: self.completion_tokens,
            total_tokens: self.total_tokens,
            input_tokens_details: self
                .prompt_tokens_details
                .as_ref()
                .map(InputTokensDetails::from),
            output_tokens_details: self.reasoning_tokens.map(|tokens| OutputTokensDetails {
                reasoning_tokens: tokens,
            }),
        }
    }
}

impl From<UsageInfo> for ResponseUsage {
    fn from(usage: UsageInfo) -> Self {
        usage.to_response_usage()
    }
}

impl ResponseUsage {
    /// Convert back to standard UsageInfo format
    pub fn to_usage_info(&self) -> UsageInfo {
        UsageInfo {
            prompt_tokens: self.input_tokens,
            completion_tokens: self.output_tokens,
            total_tokens: self.total_tokens,
            reasoning_tokens: self
                .output_tokens_details
                .as_ref()
                .map(|details| details.reasoning_tokens),
            prompt_tokens_details: self.input_tokens_details.as_ref().map(|details| {
                PromptTokenUsageInfo {
                    cached_tokens: details.cached_tokens,
                }
            }),
        }
    }
}

impl ResponsesUsage {
    pub fn to_response_usage(&self) -> ResponseUsage {
        match self {
            ResponsesUsage::Classic(usage) => usage.to_response_usage(),
            ResponsesUsage::Modern(usage) => usage.clone(),
        }
    }

    pub fn to_usage_info(&self) -> UsageInfo {
        match self {
            ResponsesUsage::Classic(usage) => usage.clone(),
            ResponsesUsage::Modern(usage) => usage.to_usage_info(),
        }
    }
}

// ============================================================================
// Helper Functions for Defaults
// ============================================================================

fn default_top_k() -> i32 {
    -1
}

fn default_repetition_penalty() -> f32 {
    1.0
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default function must match field type Option<T>"
)]
fn default_temperature() -> Option<f32> {
    Some(1.0)
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, Validate, schemars::JsonSchema)]
#[validate(schema(function = "validate_responses_cross_parameters"))]
pub struct ResponsesRequest {
    /// Run the request in the background
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,

    /// Fields to include in the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<IncludeField>>,

    /// Input content - can be string or structured items
    #[validate(custom(function = "validate_response_input"))]
    pub input: ResponseInput,

    /// System instructions for the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Maximum number of output tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 1))]
    pub max_output_tokens: Option<u32>,

    /// Maximum number of tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 1))]
    pub max_tool_calls: Option<u32>,

    /// Additional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, Value>>,

    /// Model to use
    pub model: String,

    /// Optional conversation id to persist input/output as items
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_conversation_id"))]
    pub conversation: Option<String>,

    /// Whether to enable parallel tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// ID of previous response to continue from
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,

    /// Reasoning configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponseReasoningParam>,

    /// Service tier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,

    /// Whether to store the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: Option<bool>,

    /// Temperature for sampling
    #[serde(
        default = "default_temperature",
        skip_serializing_if = "Option::is_none"
    )]
    #[validate(range(min = 0.0, max = 2.0))]
    pub temperature: Option<f32>,

    /// Tool choice behavior (Responses-spec enum — see `ResponsesToolChoice`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ResponsesToolChoice>,

    /// Available tools
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_response_tools"))]
    pub tools: Option<Vec<ResponseTool>>,

    /// Number of top logprobs to return
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 0, max = 20))]
    pub top_logprobs: Option<u32>,

    /// Top-p sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_top_p_value"))]
    pub top_p: Option<f32>,

    /// Truncation behavior
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Truncation>,

    /// Text format for structured outputs (text, json_object, json_schema)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_text_format"))]
    pub text: Option<TextConfig>,

    /// User identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Request ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,

    /// Request priority
    #[serde(default)]
    pub priority: i32,

    /// Frequency penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = -2.0, max = 2.0))]
    pub frequency_penalty: Option<f32>,

    /// Presence penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = -2.0, max = 2.0))]
    pub presence_penalty: Option<f32>,

    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_stop"))]
    pub stop: Option<StringOrArray>,

    /// Reference to a prompt template and its variables.
    /// Spec: body param `prompt` (ResponsePrompt).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<ResponsePrompt>,

    /// Stable cache key used by upstream to share prompt-prefix caches across
    /// requests. Spec: body param `prompt_cache_key` (replaces `user`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,

    /// Retention policy for prompt-cache entries.
    /// Spec: body param `prompt_cache_retention` (`"in-memory"` | `"24h"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<PromptCacheRetention>,

    /// Stable user identifier for policy/abuse detection (max 64 chars on the
    /// spec, but we do not enforce length here — routers may pass through).
    /// Spec: body param `safety_identifier` (replaces `user` on request side).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,

    /// Streaming-only options. Spec: body param `stream_options`.
    /// On the Responses API the only documented field is `include_obfuscation`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// Per-request context-management configuration.
    /// Spec: body param `context_management` — array of entries describing how
    /// the upstream should compact context for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_management: Option<Vec<ContextManagementEntry>>,

    /// Top-k sampling parameter (SGLang extension)
    #[serde(default = "default_top_k")]
    #[validate(custom(function = "validate_top_k_value"))]
    pub top_k: i32,

    /// Min-p sampling parameter (SGLang extension)
    #[serde(default)]
    #[validate(range(min = 0.0, max = 1.0))]
    pub min_p: f32,

    /// Repetition penalty (SGLang extension)
    #[serde(default = "default_repetition_penalty")]
    #[validate(range(min = 0.0, max = 2.0))]
    pub repetition_penalty: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ResponseInput {
    Items(Vec<ResponseInputOutputItem>),
    Text(String),
}

impl Default for ResponsesRequest {
    fn default() -> Self {
        Self {
            background: None,
            include: None,
            input: ResponseInput::Text(String::new()),
            instructions: None,
            max_output_tokens: None,
            max_tool_calls: None,
            metadata: None,
            model: String::new(),
            conversation: None,
            parallel_tool_calls: None,
            previous_response_id: None,
            reasoning: None,
            service_tier: None,
            store: None,
            stream: None,
            temperature: None,
            tool_choice: None,
            tools: None,
            top_logprobs: None,
            top_p: None,
            truncation: None,
            text: None,
            user: None,
            request_id: None,
            priority: 0,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            prompt: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            safety_identifier: None,
            stream_options: None,
            context_management: None,
            top_k: default_top_k(),
            min_p: 0.0,
            repetition_penalty: default_repetition_penalty(),
        }
    }
}

impl Normalizable for ResponsesRequest {
    /// Normalize the request by applying defaults:
    /// 1. Apply tool_choice defaults based on tools presence
    /// 2. Apply parallel_tool_calls defaults
    /// 3. Apply store field defaults
    fn normalize(&mut self) {
        // 1. Apply tool_choice defaults
        if self.tool_choice.is_none() {
            if let Some(tools) = &self.tools {
                let choice_value = if tools.is_empty() {
                    ToolChoiceOptions::None
                } else {
                    ToolChoiceOptions::Auto
                };
                self.tool_choice = Some(ResponsesToolChoice::Options(choice_value));
            }
            // If tools is None, leave tool_choice as None (don't set it)
        }

        // 2. Apply default for parallel_tool_calls if tools are present
        if self.parallel_tool_calls.is_none() && self.tools.is_some() {
            self.parallel_tool_calls = Some(true);
        }

        // 3. Ensure store defaults to true if not specified
        if self.store.is_none() {
            self.store = Some(true);
        }
    }
}

impl GenerationRequest for ResponsesRequest {
    fn is_stream(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    fn get_model(&self) -> Option<&str> {
        Some(self.model.as_str())
    }

    fn extract_text_for_routing(&self) -> String {
        match &self.input {
            ResponseInput::Text(text) => text.clone(),
            ResponseInput::Items(items) => {
                let mut result = String::with_capacity(256);
                let mut has_parts = false;

                let mut append_text = |text: &str| {
                    if has_parts {
                        result.push(' ');
                    }
                    has_parts = true;
                    result.push_str(text);
                };

                for item in items {
                    match item {
                        ResponseInputOutputItem::Message { content, .. } => {
                            for part in content {
                                let text = match part {
                                    ResponseContentPart::OutputText { text, .. } => {
                                        Some(text.as_str())
                                    }
                                    ResponseContentPart::InputText { text } => Some(text.as_str()),
                                    // Non-text parts (images, files, refusals) contribute no
                                    // prompt text; skip without appending.
                                    ResponseContentPart::InputImage { .. }
                                    | ResponseContentPart::InputFile { .. }
                                    | ResponseContentPart::Refusal { .. } => None,
                                };
                                if let Some(t) = text {
                                    append_text(t);
                                }
                            }
                        }
                        ResponseInputOutputItem::SimpleInputMessage { content, .. } => {
                            match content {
                                StringOrContentParts::String(s) => {
                                    append_text(s.as_str());
                                }
                                StringOrContentParts::Array(parts) => {
                                    for part in parts {
                                        let text = match part {
                                            ResponseContentPart::OutputText { text, .. } => {
                                                Some(text.as_str())
                                            }
                                            ResponseContentPart::InputText { text } => {
                                                Some(text.as_str())
                                            }
                                            ResponseContentPart::InputImage { .. }
                                            | ResponseContentPart::InputFile { .. }
                                            | ResponseContentPart::Refusal { .. } => None,
                                        };
                                        if let Some(t) = text {
                                            append_text(t);
                                        }
                                    }
                                }
                            }
                        }
                        ResponseInputOutputItem::Reasoning { content, .. } => {
                            for part in content {
                                match part {
                                    ResponseReasoningContent::ReasoningText { text } => {
                                        append_text(text.as_str());
                                    }
                                }
                            }
                        }
                        ResponseInputOutputItem::FunctionToolCall { .. }
                        | ResponseInputOutputItem::FunctionCallOutput { .. }
                        | ResponseInputOutputItem::McpApprovalRequest { .. }
                        | ResponseInputOutputItem::McpApprovalResponse { .. } => {}
                    }
                }

                result
            }
        }
    }
}

/// Validate conversation ID format
pub fn validate_conversation_id(conv_id: &str) -> Result<(), ValidationError> {
    if !conv_id.starts_with("conv_") {
        let mut error = ValidationError::new("invalid_conversation_id");
        error.message = Some(std::borrow::Cow::Owned(format!(
            "Invalid 'conversation': '{conv_id}'. Expected an ID that begins with 'conv_'."
        )));
        return Err(error);
    }

    // Check if the conversation ID contains only valid characters
    let is_valid = conv_id
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-');

    if !is_valid {
        let mut error = ValidationError::new("invalid_conversation_id");
        error.message = Some(std::borrow::Cow::Owned(format!(
            "Invalid 'conversation': '{conv_id}'. Expected an ID that contains letters, numbers, underscores, or dashes, but this value contained additional characters."
        )));
        return Err(error);
    }
    Ok(())
}

/// Validates tool_choice requires tools and references exist
fn validate_tool_choice_with_tools(request: &ResponsesRequest) -> Result<(), ValidationError> {
    let Some(tool_choice) = &request.tool_choice else {
        return Ok(());
    };

    let has_tools = request.tools.as_ref().is_some_and(|t| !t.is_empty());
    let is_some_choice = !matches!(
        tool_choice,
        ResponsesToolChoice::Options(ToolChoiceOptions::None)
    );

    // Check if tool_choice requires tools but none are provided
    if is_some_choice && !has_tools {
        let mut e = ValidationError::new("tool_choice_requires_tools");
        e.message = Some("Invalid value for 'tool_choice': 'tool_choice' is only allowed when 'tools' are specified.".into());
        return Err(e);
    }

    // Validate tool references exist when tools are present
    if !has_tools {
        return Ok(());
    }

    // Extract function tool names from ResponseTools
    // INVARIANT: has_tools is true here, so tools is Some and non-empty
    let Some(tools) = request.tools.as_ref() else {
        return Ok(());
    };
    let function_tool_names: Vec<&str> = tools
        .iter()
        .filter_map(|t| match t {
            ResponseTool::Function(ft) => Some(ft.function.name.as_str()),
            _ => None,
        })
        .collect();

    // Validate tool references exist
    match tool_choice {
        ResponsesToolChoice::Function(_) => {
            // Accessor goes through `function_name()` so we stay agnostic to
            // the underlying wire shape (flat vs. legacy nested) — both are
            // normalized at deserialize time.
            if let Some(name) = tool_choice.function_name() {
                if !function_tool_names.contains(&name) {
                    let mut e = ValidationError::new("tool_choice_function_not_found");
                    e.message = Some(
                        format!(
                            "Invalid value for 'tool_choice': function '{name}' not found in 'tools'.",
                        )
                        .into(),
                    );
                    return Err(e);
                }
            }
        }
        ResponsesToolChoice::AllowedTools {
            mode,
            tools: allowed_tools,
            ..
        } => {
            // Validate mode is "auto" or "required"
            if mode != "auto" && mode != "required" {
                let mut e = ValidationError::new("tool_choice_invalid_mode");
                e.message = Some(
                    format!(
                        "Invalid value for 'tool_choice.mode': must be 'auto' or 'required', got '{mode}'."
                    )
                    .into(),
                );
                return Err(e);
            }

            // Validate that all function tool references exist
            for tool_ref in allowed_tools {
                if let ToolReference::Function { name } = tool_ref {
                    if !function_tool_names.contains(&name.as_str()) {
                        let mut e = ValidationError::new("tool_choice_tool_not_found");
                        e.message = Some(
                            format!(
                                "Invalid value for 'tool_choice.tools': tool '{name}' not found in 'tools'."
                            )
                            .into(),
                        );
                        return Err(e);
                    }
                }
                // Note: MCP and hosted tools don't need existence validation here
                // as they are resolved dynamically at runtime
            }
        }
        // Remaining variants have no cross-field existence constraints —
        // hosted built-ins, MCP server selection, custom tool names, and
        // `apply_patch` / `shell` are resolved at routing time.
        ResponsesToolChoice::Options(_)
        | ResponsesToolChoice::Types { .. }
        | ResponsesToolChoice::Mcp { .. }
        | ResponsesToolChoice::Custom { .. }
        | ResponsesToolChoice::ApplyPatch { .. }
        | ResponsesToolChoice::Shell { .. } => {}
    }

    Ok(())
}

/// Schema-level validation for cross-field dependencies
fn validate_responses_cross_parameters(request: &ResponsesRequest) -> Result<(), ValidationError> {
    // 1. Validate tool_choice requires tools (enhanced)
    validate_tool_choice_with_tools(request)?;

    // 2. Validate top_logprobs requires include field
    if request.top_logprobs.is_some() {
        let has_logprobs_include = request
            .include
            .as_ref()
            .is_some_and(|inc| inc.contains(&IncludeField::MessageOutputTextLogprobs));

        if !has_logprobs_include {
            let mut e = ValidationError::new("top_logprobs_requires_include");
            e.message = Some(
                "top_logprobs requires include field with 'message.output_text.logprobs'".into(),
            );
            return Err(e);
        }
    }

    // 3. Validate background/stream conflict
    if request.background == Some(true) && request.stream == Some(true) {
        let mut e = ValidationError::new("background_conflicts_with_stream");
        e.message = Some("Cannot use background mode with streaming".into());
        return Err(e);
    }

    // 4. Validate conversation and previous_response_id are mutually exclusive
    if request.conversation.is_some() && request.previous_response_id.is_some() {
        let mut e = ValidationError::new("mutually_exclusive_parameters");
        e.message = Some("Mutually exclusive parameters. Ensure you are only providing one of: 'previous_response_id' or 'conversation'.".into());
        return Err(e);
    }

    // 5. Validate input items structure
    if let ResponseInput::Items(items) = &request.input {
        // Check for at least one valid input message
        let has_valid_input = items.iter().any(|item| {
            matches!(
                item,
                ResponseInputOutputItem::Message { .. }
                    | ResponseInputOutputItem::SimpleInputMessage { .. }
            )
        });

        if !has_valid_input {
            let mut e = ValidationError::new("input_missing_user_message");
            e.message = Some("Input items must contain at least one message".into());
            return Err(e);
        }
    }

    // 6. Validate text format conflicts (for future structured output constraints)
    // Currently, Responses API doesn't have regex/ebnf like Chat API,
    // but this is here for completeness and future-proofing

    Ok(())
}

// ============================================================================
// Field-Level Validation Functions
// ============================================================================

/// Validates response input is not empty and has valid content
fn validate_response_input(input: &ResponseInput) -> Result<(), ValidationError> {
    match input {
        ResponseInput::Text(text) => {
            if text.is_empty() {
                let mut e = ValidationError::new("input_text_empty");
                e.message = Some("Input text cannot be empty".into());
                return Err(e);
            }
        }
        ResponseInput::Items(items) => {
            if items.is_empty() {
                let mut e = ValidationError::new("input_items_empty");
                e.message = Some("Input items cannot be empty".into());
                return Err(e);
            }
            // Validate each item has valid content
            for item in items {
                validate_input_item(item)?;
            }
        }
    }
    Ok(())
}

/// Validates individual input items have valid content
fn validate_input_item(item: &ResponseInputOutputItem) -> Result<(), ValidationError> {
    match item {
        ResponseInputOutputItem::Message { content, .. } => {
            if content.is_empty() {
                let mut e = ValidationError::new("message_content_empty");
                e.message = Some("Message content cannot be empty".into());
                return Err(e);
            }
        }
        ResponseInputOutputItem::SimpleInputMessage { content, .. } => match content {
            StringOrContentParts::String(s) if s.is_empty() => {
                let mut e = ValidationError::new("message_content_empty");
                e.message = Some("Message content cannot be empty".into());
                return Err(e);
            }
            StringOrContentParts::Array(parts) if parts.is_empty() => {
                let mut e = ValidationError::new("message_content_empty");
                e.message = Some("Message content parts cannot be empty".into());
                return Err(e);
            }
            _ => {}
        },
        ResponseInputOutputItem::Reasoning { .. } => {
            // Reasoning content can be empty - no validation needed
        }
        ResponseInputOutputItem::FunctionCallOutput { output, .. } => {
            if output.is_empty() {
                let mut e = ValidationError::new("function_output_empty");
                e.message = Some("Function call output cannot be empty".into());
                return Err(e);
            }
        }
        ResponseInputOutputItem::FunctionToolCall { .. } => {}
        ResponseInputOutputItem::McpApprovalRequest { .. } => {}
        ResponseInputOutputItem::McpApprovalResponse { .. } => {}
    }
    Ok(())
}

/// Validates ResponseTool structure based on tool type
fn validate_response_tools(tools: &[ResponseTool]) -> Result<(), ValidationError> {
    // MCP server_label must be present and unique (case-insensitive).
    let mut seen_mcp_labels: HashSet<String> = HashSet::new();

    for (idx, tool) in tools.iter().enumerate() {
        if let ResponseTool::Mcp(mcp) = tool {
            let raw_label = mcp.server_label.as_str();
            if raw_label.is_empty() {
                let mut e = ValidationError::new("missing_required_parameter");
                e.message = Some(
                    format!("Missing required parameter: 'tools[{idx}].server_label'.").into(),
                );
                return Err(e);
            }

            // OpenAI spec-compatible validation: require a non-empty label that starts with a
            // letter and contains only letters, digits, '-' and '_'.
            let valid = raw_label.starts_with(|c: char| c.is_ascii_alphabetic())
                && raw_label
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
            if !valid {
                let mut e = ValidationError::new("invalid_server_label");
                e.message = Some(
                    format!(
                        "Invalid input {raw_label}: 'server_label' must start with a letter and consist of only letters, digits, '-' and '_'"
                    )
                    .into(),
                );
                return Err(e);
            }

            let normalized = raw_label.to_lowercase();
            if !seen_mcp_labels.insert(normalized) {
                let mut e = ValidationError::new("mcp_tool_duplicate_server_label");
                e.message = Some(
                    format!("Duplicate MCP server_label '{raw_label}' found in 'tools' parameter.")
                        .into(),
                );
                return Err(e);
            }
        }
    }
    Ok(())
}

/// Validates text format configuration (JSON schema name cannot be empty)
fn validate_text_format(text: &TextConfig) -> Result<(), ValidationError> {
    if let Some(TextFormat::JsonSchema { name, .. }) = &text.format {
        if name.is_empty() {
            let mut e = ValidationError::new("json_schema_name_empty");
            e.message = Some("JSON schema name cannot be empty".into());
            return Err(e);
        }
    }
    Ok(())
}

/// Normalize a SimpleInputMessage to a proper Message item
///
/// This helper converts SimpleInputMessage (which can have flexible content)
/// into a fully-structured Message item with a generated ID, role, and content array.
///
/// SimpleInputMessage items are converted to Message items with IDs generated using
/// the centralized ID generation pattern with "msg_" prefix for consistency.
///
/// # Arguments
/// * `item` - The input item to normalize
///
/// # Returns
/// A normalized ResponseInputOutputItem (either Message if converted, or original if not SimpleInputMessage)
pub fn normalize_input_item(item: &ResponseInputOutputItem) -> ResponseInputOutputItem {
    match item {
        ResponseInputOutputItem::SimpleInputMessage {
            content,
            role,
            phase,
            ..
        } => {
            let content_vec = match content {
                StringOrContentParts::String(s) => {
                    vec![ResponseContentPart::InputText { text: s.clone() }]
                }
                StringOrContentParts::Array(parts) => parts.clone(),
            };

            ResponseInputOutputItem::Message {
                id: generate_id("msg"),
                role: role.clone(),
                content: content_vec,
                status: Some("completed".to_string()),
                phase: *phase,
            }
        }
        _ => item.clone(),
    }
}

pub fn generate_id(prefix: &str) -> String {
    use rand::RngCore;
    let mut rng = rand::rng();
    // Generate exactly 50 hex characters (25 bytes) for the part after the underscore
    let mut bytes = [0u8; 25];
    rng.fill_bytes(&mut bytes);
    let hex_string: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("{prefix}_{hex_string}")
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[non_exhaustive]
pub struct ResponsesResponse {
    /// Response ID
    pub id: String,

    /// Object type
    #[serde(default = "default_object_type")]
    pub object: String,

    /// Creation timestamp (unix seconds)
    pub created_at: i64,

    /// Completion timestamp (unix seconds). `None` until the response reaches
    /// a terminal state (`completed`, `incomplete`, `failed`, `cancelled`).
    #[serde(default)]
    pub completed_at: Option<i64>,

    /// Whether the response was created in background mode.
    #[serde(default)]
    pub background: Option<bool>,

    /// Conversation this response is linked to, if any.
    #[serde(default)]
    pub conversation: Option<String>,

    /// Response status
    pub status: ResponseStatus,

    /// Error information if status is failed
    pub error: Option<Value>,

    /// Incomplete details if response was truncated
    pub incomplete_details: Option<Value>,

    /// System instructions used
    pub instructions: Option<String>,

    /// Max output tokens setting
    pub max_output_tokens: Option<u32>,

    /// Model name
    pub model: String,

    /// Output items
    #[serde(default)]
    pub output: Vec<ResponseOutputItem>,

    /// Whether parallel tool calls are enabled
    #[serde(default = "default_true")]
    pub parallel_tool_calls: bool,

    /// Previous response ID if this is a continuation
    pub previous_response_id: Option<String>,

    /// Reasoning information
    pub reasoning: Option<ReasoningInfo>,

    /// Whether the response is stored
    #[serde(default = "default_true")]
    pub store: bool,

    /// Temperature setting used
    pub temperature: Option<f32>,

    /// Text format settings
    pub text: Option<TextConfig>,

    /// Tool choice setting
    #[serde(default = "default_tool_choice")]
    pub tool_choice: String,

    /// Available tools
    #[serde(default)]
    pub tools: Vec<ResponseTool>,

    /// Top-p setting used
    pub top_p: Option<f32>,

    /// Truncation strategy used
    pub truncation: Option<String>,

    /// Usage information
    pub usage: Option<ResponsesUsage>,

    /// User identifier
    pub user: Option<String>,

    /// Safety identifier for content moderation
    pub safety_identifier: Option<String>,

    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

fn default_object_type() -> String {
    "response".to_string()
}

fn default_tool_choice() -> String {
    "auto".to_string()
}

impl ResponsesResponse {
    /// Create a builder for constructing a ResponsesResponse
    pub fn builder(id: impl Into<String>, model: impl Into<String>) -> ResponsesResponseBuilder {
        ResponsesResponseBuilder::new(id, model)
    }

    /// Check if the response is complete
    pub fn is_complete(&self) -> bool {
        matches!(self.status, ResponseStatus::Completed)
    }

    /// Check if the response is in progress
    pub fn is_in_progress(&self) -> bool {
        matches!(self.status, ResponseStatus::InProgress)
    }

    /// Check if the response failed
    pub fn is_failed(&self) -> bool {
        matches!(self.status, ResponseStatus::Failed)
    }

    /// Check if the response terminated as incomplete (max_output_tokens / content_filter)
    pub fn is_incomplete(&self) -> bool {
        matches!(self.status, ResponseStatus::Incomplete)
    }
}

impl ResponseInputOutputItem {
    /// Create a new reasoning input/output item.
    ///
    /// `encrypted_content` defaults to `None`; use
    /// [`Self::new_reasoning_encrypted`] when round-tripping gpt-5 /
    /// o-series encrypted reasoning.
    pub fn new_reasoning(
        id: String,
        summary: Vec<SummaryTextContent>,
        content: Vec<ResponseReasoningContent>,
        status: Option<String>,
    ) -> Self {
        Self::Reasoning {
            id,
            summary,
            content,
            encrypted_content: None,
            status,
        }
    }

    /// Create a new reasoning input/output item carrying an encrypted
    /// reasoning payload. The `encrypted_content` must be the opaque
    /// ciphertext.
    pub fn new_reasoning_encrypted(
        id: String,
        summary: Vec<SummaryTextContent>,
        content: Vec<ResponseReasoningContent>,
        encrypted_content: String,
        status: Option<String>,
    ) -> Self {
        Self::Reasoning {
            id,
            summary,
            content,
            encrypted_content: Some(encrypted_content),
            status,
        }
    }
}

impl ResponseOutputItem {
    /// Create a new message output item (no phase).
    pub fn new_message(
        id: String,
        role: String,
        content: Vec<ResponseContentPart>,
        status: String,
    ) -> Self {
        Self::Message {
            id,
            role,
            content,
            status,
            phase: None,
        }
    }

    /// Create a new reasoning output item.
    ///
    /// `encrypted_content` defaults to `None`; use
    /// [`Self::new_reasoning_encrypted`] when carrying gpt-5 / o-series
    /// encrypted reasoning.
    pub fn new_reasoning(
        id: String,
        summary: Vec<SummaryTextContent>,
        content: Vec<ResponseReasoningContent>,
        status: Option<String>,
    ) -> Self {
        Self::Reasoning {
            id,
            summary,
            content,
            encrypted_content: None,
            status,
        }
    }

    /// Create a new reasoning output item carrying an encrypted reasoning payload.
    ///
    /// The `encrypted_content` must be the opaque ciphertext; a `None` value
    /// would defeat the purpose of the `_encrypted` constructor — callers
    /// without ciphertext should use [`Self::new_reasoning`] instead.
    pub fn new_reasoning_encrypted(
        id: String,
        summary: Vec<SummaryTextContent>,
        content: Vec<ResponseReasoningContent>,
        encrypted_content: String,
        status: Option<String>,
    ) -> Self {
        Self::Reasoning {
            id,
            summary,
            content,
            encrypted_content: Some(encrypted_content),
            status,
        }
    }

    /// Create a new function tool call output item
    pub fn new_function_tool_call(
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        output: Option<String>,
        status: String,
    ) -> Self {
        Self::FunctionToolCall {
            id,
            call_id,
            name,
            arguments,
            output,
            status,
        }
    }
}

impl ResponseContentPart {
    /// Create a new `output_text` content part.
    pub fn new_text(
        text: String,
        annotations: Vec<Annotation>,
        logprobs: Option<ChatLogProbs>,
    ) -> Self {
        Self::OutputText {
            text,
            annotations,
            logprobs,
        }
    }
}

impl ResponseReasoningContent {
    /// Create a new reasoning text content
    pub fn new_reasoning_text(text: String) -> Self {
        Self::ReasoningText { text }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn summary_text_content_round_trips_spec_shape() {
        // Spec: `summary: array of SummaryTextContent { text, type: "summary_text" }`.
        let item: ResponseOutputItem = serde_json::from_value(json!({
            "type": "reasoning",
            "id": "r_1",
            "summary": [{"text": "step 1", "type": "summary_text"}],
            "content": [],
            "status": "completed",
        }))
        .expect("spec-shape reasoning should deserialize");

        match &item {
            ResponseOutputItem::Reasoning { summary, .. } => {
                assert_eq!(summary.len(), 1);
                match &summary[0] {
                    SummaryTextContent::SummaryText { text } => {
                        assert_eq!(text, "step 1");
                    }
                }
            }
            _ => panic!("expected Reasoning variant"),
        }

        let v = serde_json::to_value(&item).expect("serialize");
        assert_eq!(
            v["summary"],
            json!([{"text": "step 1", "type": "summary_text"}])
        );
    }

    #[test]
    fn legacy_vec_string_summary_fails_to_deserialize() {
        let legacy = r#"{"type":"reasoning","id":"r_x","summary":["text"]}"#;
        let result: Result<ResponseInputOutputItem, _> = serde_json::from_str(legacy);
        assert!(
            result.is_err(),
            "legacy Vec<String> summary must no longer deserialize"
        );
    }

    #[test]
    fn test_responses_request_omitted_top_p_deserializes_to_none() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5.4",
            "input": "hello"
        }))
        .expect("request should deserialize");

        assert_eq!(request.top_p, None);

        let serialized = serde_json::to_value(&request).expect("request should serialize");
        assert!(serialized.get("top_p").is_none());
    }

    #[test]
    fn test_responses_request_null_top_p_deserializes_to_none() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5.4",
            "input": "hello",
            "top_p": null
        }))
        .expect("request should deserialize");

        assert_eq!(request.top_p, None);

        let serialized = serde_json::to_value(&request).expect("request should serialize");
        assert!(serialized.get("top_p").is_none());
    }

    #[test]
    fn test_responses_request_explicit_top_p_preserved() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5.4",
            "input": "hello",
            "top_p": 0.9
        }))
        .expect("request should deserialize");

        assert_eq!(request.top_p, Some(0.9));
    }

    #[test]
    fn test_require_approval_string_round_trip() {
        let tool: McpTool = serde_json::from_value(json!({
            "server_label": "deepwiki",
            "require_approval": "never"
        }))
        .expect("mcp tool should deserialize");

        assert_eq!(
            tool.require_approval,
            Some(RequireApproval::Mode(RequireApprovalMode::Never))
        );

        let serialized = serde_json::to_value(&tool).expect("mcp tool should serialize");
        assert_eq!(serialized["require_approval"], json!("never"));
    }

    #[test]
    fn test_require_approval_object_round_trip() {
        let tool: McpTool = serde_json::from_value(json!({
            "server_label": "deepwiki",
            "require_approval": {
                "always": null,
                "never": {
                    "tool_names": ["ask_question", "read_wiki_structure"],
                    "read_only": null
                }
            }
        }))
        .expect("mcp tool should deserialize");

        assert_eq!(
            tool.require_approval,
            Some(RequireApproval::Rules(RequireApprovalRules {
                always: None,
                never: Some(RequireApprovalFilter {
                    tool_names: Some(vec![
                        "ask_question".to_string(),
                        "read_wiki_structure".to_string()
                    ]),
                    read_only: None,
                }),
            }))
        );

        let serialized = serde_json::to_value(&tool).expect("mcp tool should serialize");
        assert_eq!(
            serialized["require_approval"],
            json!({
                "never": {
                    "tool_names": ["ask_question", "read_wiki_structure"]
                }
            })
        );
    }

    #[test]
    fn test_include_field_web_search_call_variants_round_trip() {
        let fields: Vec<IncludeField> = serde_json::from_value(json!([
            "web_search_call.action.sources",
            "web_search_call.results",
        ]))
        .expect("include fields should deserialize");

        assert_eq!(
            fields,
            vec![
                IncludeField::WebSearchCallActionSources,
                IncludeField::WebSearchCallResults,
            ]
        );

        let serialized = serde_json::to_value(&fields).expect("include fields should serialize");
        assert_eq!(
            serialized,
            json!(["web_search_call.action.sources", "web_search_call.results"])
        );
    }

    #[test]
    fn test_file_search_tool_round_trip() {
        // Spec fixture (openai-responses-api-spec.md §tools): `file_search` with
        // vector_store_ids, compound filter, and ranking_options incl. hybrid_search,
        // ranker enum, and score_threshold.
        let payload = json!({
            "type": "file_search",
            "vector_store_ids": ["vs_1234567890"],
            "max_num_results": 20,
            "filters": {
                "type": "and",
                "filters": [
                    {"type": "eq", "key": "region", "value": "us-east-1"},
                    {"type": "in", "key": "tag", "value": ["alpha", "beta"]}
                ]
            },
            "ranking_options": {
                "ranker": "default-2024-11-15",
                "score_threshold": 0.5,
                "hybrid_search": {"embedding_weight": 0.7, "text_weight": 0.3}
            }
        });

        let tool: ResponseTool =
            serde_json::from_value(payload.clone()).expect("file_search tool should deserialize");
        assert!(matches!(tool, ResponseTool::FileSearch(_)));

        let serialized = serde_json::to_value(&tool).expect("file_search tool should serialize");
        assert_eq!(serialized, payload);
    }

    #[test]
    fn test_file_search_tool_round_trip_hybrid_weights_omitted() {
        // Spec (openai-responses-api-spec.md §tools): `embedding_weight` and
        // `text_weight` are optional. When omitted they must deserialize to
        // `None` and be absent again on re-serialization (absent→None→absent).
        let payload = json!({
            "type": "file_search",
            "vector_store_ids": ["vs_1234567890"],
            "ranking_options": {
                "hybrid_search": {}
            }
        });

        let tool: ResponseTool = serde_json::from_value(payload.clone())
            .expect("file_search tool with empty hybrid_search should deserialize");
        match &tool {
            ResponseTool::FileSearch(fs) => {
                let ranking = fs
                    .ranking_options
                    .as_ref()
                    .expect("ranking_options should be present");
                let hybrid = ranking
                    .hybrid_search
                    .as_ref()
                    .expect("hybrid_search should be present");
                assert!(hybrid.embedding_weight.is_none());
                assert!(hybrid.text_weight.is_none());
            }
            other => panic!("expected FileSearch variant, got {other:?}"),
        }

        let serialized = serde_json::to_value(&tool).expect("file_search tool should serialize");
        assert_eq!(serialized, payload);
    }

    // ------------------------------------------------------------------
    // P2: new top-level ResponsesRequest fields
    // ------------------------------------------------------------------

    /// Acceptance: the six new top-level fields deserialize and re-serialize
    /// without loss (`prompt`, `prompt_cache_key`, `prompt_cache_retention`,
    /// `safety_identifier`, `stream_options`, `context_management`).
    #[test]
    fn test_responses_request_new_top_level_fields_round_trip() {
        let payload = json!({
            "model": "gpt-5.4",
            "input": "hello",
            "prompt": {
                "id": "pmpt_abc",
                "variables": {
                    "name": "ada",
                    "picture": {
                        "type": "input_image",
                        "image_url": "https://example.com/pic.png",
                        "detail": "high"
                    }
                },
                "version": "1"
            },
            "prompt_cache_key": "pck-123",
            "prompt_cache_retention": "24h",
            "safety_identifier": "sid-123",
            "stream_options": { "include_obfuscation": false },
            "context_management": [{
                "type": "compaction",
                "compact_threshold": 4096
            }]
        });

        let request: ResponsesRequest =
            serde_json::from_value(payload.clone()).expect("request should deserialize");

        assert!(request.prompt.is_some());
        assert_eq!(
            request.prompt.as_ref().map(|p| p.id.as_str()),
            Some("pmpt_abc")
        );
        assert_eq!(
            request.prompt.as_ref().and_then(|p| p.version.as_deref()),
            Some("1")
        );
        assert_eq!(request.prompt_cache_key.as_deref(), Some("pck-123"));
        assert_eq!(
            request.prompt_cache_retention,
            Some(PromptCacheRetention::Duration24h)
        );
        assert_eq!(request.safety_identifier.as_deref(), Some("sid-123"));
        assert_eq!(
            request
                .stream_options
                .as_ref()
                .and_then(|s| s.include_obfuscation),
            Some(false)
        );
        let ctx = request
            .context_management
            .as_ref()
            .expect("context_management must round-trip");
        assert_eq!(ctx.len(), 1);
        assert_eq!(
            ctx[0].r#type,
            crate::common::ContextManagementType::Compaction
        );
        assert_eq!(ctx[0].compact_threshold, Some(4096));

        // Re-serialize and confirm the wire form matches the inputs.
        let reserialized = serde_json::to_value(&request).expect("should serialize");
        assert_eq!(reserialized["prompt"]["id"], "pmpt_abc");
        assert_eq!(reserialized["prompt_cache_key"], "pck-123");
        assert_eq!(reserialized["prompt_cache_retention"], "24h");
        assert_eq!(reserialized["safety_identifier"], "sid-123");
        assert_eq!(reserialized["stream_options"]["include_obfuscation"], false);
        assert_eq!(reserialized["context_management"][0]["type"], "compaction");
        assert_eq!(
            reserialized["context_management"][0]["compact_threshold"],
            4096
        );
    }

    /// `prompt_cache_retention` accepts the other spec value and serializes
    /// back with the hyphenated rename.
    #[test]
    fn test_prompt_cache_retention_in_memory_round_trip() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5.4",
            "input": "hello",
            "prompt_cache_retention": "in-memory"
        }))
        .expect("should deserialize");

        assert_eq!(
            request.prompt_cache_retention,
            Some(PromptCacheRetention::InMemory)
        );

        let reserialized = serde_json::to_value(&request).expect("should serialize");
        assert_eq!(reserialized["prompt_cache_retention"], "in-memory");
    }

    /// Absent fields must stay absent on the wire (no `"prompt": null` etc.),
    /// matching every other `Option<_>` field on `ResponsesRequest`.
    #[test]
    fn test_responses_request_new_fields_omitted_when_absent() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5.4",
            "input": "hello"
        }))
        .expect("should deserialize");

        assert!(request.prompt.is_none());
        assert!(request.prompt_cache_key.is_none());
        assert!(request.prompt_cache_retention.is_none());
        assert!(request.safety_identifier.is_none());
        assert!(request.stream_options.is_none());
        assert!(request.context_management.is_none());

        let serialized = serde_json::to_value(&request).expect("should serialize");
        for key in [
            "prompt",
            "prompt_cache_key",
            "prompt_cache_retention",
            "safety_identifier",
            "stream_options",
            "context_management",
        ] {
            assert!(
                serialized.get(key).is_none(),
                "field {key} should be skipped when absent"
            );
        }
    }

    // ---- P1: rich content parts + typed annotations ----

    #[test]
    fn content_part_input_image_roundtrip() {
        // `original` is the spec-new detail level; exercising it here
        // also covers the rest of the variant shape.
        let raw = json!({
            "type": "input_image",
            "detail": "original",
            "image_url": "https://example.com/cat.png"
        });
        let part: ResponseContentPart =
            serde_json::from_value(raw.clone()).expect("input_image should deserialize");
        match &part {
            ResponseContentPart::InputImage {
                detail,
                file_id,
                image_url,
            } => {
                assert!(matches!(detail, Some(Detail::Original)));
                assert!(file_id.is_none());
                assert_eq!(image_url.as_deref(), Some("https://example.com/cat.png"));
            }
            other => panic!("expected InputImage, got {other:?}"),
        }
        let roundtripped = serde_json::to_value(&part).unwrap();
        assert_eq!(roundtripped, raw);
    }

    #[test]
    fn content_part_input_file_roundtrip() {
        let raw = json!({
            "type": "input_file",
            "detail": "low",
            "file_data": "BASE64DATA",
            "filename": "report.pdf"
        });
        let part: ResponseContentPart =
            serde_json::from_value(raw.clone()).expect("input_file should deserialize");
        match &part {
            ResponseContentPart::InputFile {
                detail,
                file_data,
                file_id,
                file_url,
                filename,
            } => {
                assert!(matches!(detail, Some(FileDetail::Low)));
                assert_eq!(file_data.as_deref(), Some("BASE64DATA"));
                assert!(file_id.is_none());
                assert!(file_url.is_none());
                assert_eq!(filename.as_deref(), Some("report.pdf"));
            }
            other => panic!("expected InputFile, got {other:?}"),
        }
        let roundtripped = serde_json::to_value(&part).unwrap();
        assert_eq!(roundtripped, raw);
    }

    #[test]
    fn content_part_refusal_roundtrip() {
        let raw = json!({ "type": "refusal", "refusal": "I cannot help with that." });
        let part: ResponseContentPart =
            serde_json::from_value(raw.clone()).expect("refusal should deserialize");
        match &part {
            ResponseContentPart::Refusal { refusal } => {
                assert_eq!(refusal, "I cannot help with that.");
            }
            other => panic!("expected Refusal, got {other:?}"),
        }
        let roundtripped = serde_json::to_value(&part).unwrap();
        assert_eq!(roundtripped, raw);
    }

    #[test]
    fn content_part_output_text_with_typed_annotations() {
        let raw = json!({
            "type": "output_text",
            "text": "See [1] and [2].",
            "annotations": [
                {
                    "type": "file_citation",
                    "file_id": "file-abc",
                    "filename": "source.pdf",
                    "index": 5
                },
                {
                    "type": "url_citation",
                    "url": "https://example.com",
                    "title": "Example",
                    "start_index": 0,
                    "end_index": 18
                },
                {
                    "type": "container_file_citation",
                    "container_id": "cntr-1",
                    "file_id": "file-xyz",
                    "filename": "out.txt",
                    "start_index": 0,
                    "end_index": 4
                },
                {
                    "type": "file_path",
                    "file_id": "file-gen",
                    "index": 7
                }
            ]
        });
        let part: ResponseContentPart =
            serde_json::from_value(raw.clone()).expect("output_text should deserialize");
        match &part {
            ResponseContentPart::OutputText {
                text,
                annotations,
                logprobs,
            } => {
                assert_eq!(text, "See [1] and [2].");
                assert_eq!(annotations.len(), 4);
                assert!(logprobs.is_none());
                assert!(matches!(&annotations[0], Annotation::FileCitation { .. }));
                assert!(matches!(&annotations[1], Annotation::UrlCitation { .. }));
                assert!(matches!(
                    &annotations[2],
                    Annotation::ContainerFileCitation { .. }
                ));
                assert!(matches!(&annotations[3], Annotation::FilePath { .. }));
            }
            other => panic!("expected OutputText, got {other:?}"),
        }
        let roundtripped = serde_json::to_value(&part).unwrap();
        assert_eq!(roundtripped, raw);
    }

    #[test]
    fn content_part_unknown_type_fails_fast() {
        // Previously `#[serde(other)] Unknown` silently swallowed unknown
        // types; P1 removes that arm so spec-invalid payloads fail cleanly.
        let raw = json!({ "type": "totally_made_up", "text": "x" });
        let err = serde_json::from_value::<ResponseContentPart>(raw)
            .expect_err("unknown content-part type must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("totally_made_up") || msg.contains("unknown variant"),
            "error should mention the unknown variant, got: {msg}"
        );
    }

    // ------------------------------------------------------------------
    // P7: ResponsesToolChoice round-trip tests (all 8 spec variants)
    // ------------------------------------------------------------------

    /// Options: `"none"` | `"auto"` | `"required"` deserialize as bare strings
    /// and re-serialize identically.
    #[test]
    fn responses_tool_choice_options_round_trip() {
        for (payload, expected) in [
            (json!("none"), ToolChoiceOptions::None),
            (json!("auto"), ToolChoiceOptions::Auto),
            (json!("required"), ToolChoiceOptions::Required),
        ] {
            let choice: ResponsesToolChoice =
                serde_json::from_value(payload.clone()).expect("options string should deserialize");
            match &choice {
                ResponsesToolChoice::Options(opt) => assert_eq!(opt, &expected),
                other => panic!("expected Options, got {other:?}"),
            }
            let reserialized = serde_json::to_value(&choice).expect("serialize");
            assert_eq!(reserialized, payload);
        }
    }

    /// Types: `{"type": "<built-in>"}` for every hosted tool the spec enumerates.
    #[test]
    fn responses_tool_choice_types_round_trip() {
        for (payload, expected) in [
            (
                json!({"type": "file_search"}),
                BuiltInToolChoiceType::FileSearch,
            ),
            (
                json!({"type": "web_search"}),
                BuiltInToolChoiceType::WebSearch,
            ),
            (
                json!({"type": "web_search_preview"}),
                BuiltInToolChoiceType::WebSearchPreview,
            ),
            (
                json!({"type": "web_search_preview_2025_03_11"}),
                BuiltInToolChoiceType::WebSearchPreview20250311,
            ),
            (
                json!({"type": "image_generation"}),
                BuiltInToolChoiceType::ImageGeneration,
            ),
            (
                json!({"type": "computer_use_preview"}),
                BuiltInToolChoiceType::ComputerUsePreview,
            ),
            (
                json!({"type": "code_interpreter"}),
                BuiltInToolChoiceType::CodeInterpreter,
            ),
        ] {
            let choice: ResponsesToolChoice =
                serde_json::from_value(payload.clone()).expect("types object should deserialize");
            match &choice {
                ResponsesToolChoice::Types { tool_type } => assert_eq!(tool_type, &expected),
                other => panic!("expected Types, got {other:?}"),
            }
            let reserialized = serde_json::to_value(&choice).expect("serialize");
            assert_eq!(reserialized, payload);
        }
    }

    /// Function: `{"type": "function", "name": "..."}` — spec-canonical flat
    /// shape round-trips unchanged.
    #[test]
    fn responses_tool_choice_function_round_trip() {
        let payload = json!({"type": "function", "name": "get_weather"});
        let choice: ResponsesToolChoice = serde_json::from_value(payload.clone())
            .expect("function flat shape should deserialize");
        match &choice {
            ResponsesToolChoice::Function(payload) => assert_eq!(payload.name, "get_weather"),
            other => panic!("expected Function, got {other:?}"),
        }
        assert_eq!(choice.function_name(), Some("get_weather"));
        let reserialized = serde_json::to_value(&choice).expect("serialize");
        assert_eq!(reserialized, payload);
    }

    /// Backward compat: the Chat-style nested `{"function": {"name": ...}}`
    /// wire shape was accepted on the Responses endpoint before the P7
    /// split and existing smg clients + e2e tests rely on it. We accept it
    /// on deserialize (Postel: liberal on input) but always emit the
    /// canonical flat shape on serialize (conservative on output).
    #[test]
    fn responses_tool_choice_function_accepts_legacy_nested() {
        let legacy_nested = json!({
            "type": "function",
            "function": {"name": "search_web"}
        });
        let choice: ResponsesToolChoice = serde_json::from_value(legacy_nested)
            .expect("legacy nested function shape must deserialize for backward compat");

        // Internal state is normalized to the flat `name` field regardless of
        // which wire shape we read.
        match &choice {
            ResponsesToolChoice::Function(payload) => {
                assert_eq!(payload.name, "search_web");
                assert_eq!(payload.tool_type, FunctionToolChoiceTag::Function);
            }
            other => panic!("expected Function, got {other:?}"),
        }
        assert_eq!(choice.function_name(), Some("search_web"));

        // Serialize MUST emit the canonical flat shape, not the nested legacy
        // shape that came in on the wire.
        let reserialized = serde_json::to_value(&choice).expect("serialize");
        assert_eq!(
            reserialized,
            json!({"type": "function", "name": "search_web"}),
            "Responses `Function` must serialize as canonical flat even when \
             deserialized from the legacy nested shape"
        );
        assert!(
            reserialized.get("function").is_none(),
            "canonical flat serialize must not emit a nested `function` object"
        );
    }

    /// Payloads without either `name` or `function.name` must fail —
    /// we accept both wire shapes, but at least one of them must carry the
    /// function name. (The exact error string comes from serde's untagged
    /// enum routing which swallows inner errors, so we assert only that
    /// deserialization fails.)
    #[test]
    fn responses_tool_choice_function_rejects_missing_name() {
        let payload = json!({"type": "function"});
        assert!(
            serde_json::from_value::<ResponsesToolChoice>(payload).is_err(),
            "function without `name` or `function.name` must be rejected",
        );
    }

    /// AllowedTools: `{"type": "allowed_tools", "mode", "tools"}` with Function
    /// references; validates ToolReference interop.
    #[test]
    fn responses_tool_choice_allowed_tools_round_trip() {
        let payload = json!({
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [{"type": "function", "name": "get_weather"}]
        });
        let choice: ResponsesToolChoice =
            serde_json::from_value(payload.clone()).expect("allowed_tools should deserialize");
        match &choice {
            ResponsesToolChoice::AllowedTools { mode, tools, .. } => {
                assert_eq!(mode, "auto");
                assert_eq!(tools.len(), 1);
                match &tools[0] {
                    ToolReference::Function { name } => assert_eq!(name, "get_weather"),
                    other => panic!("expected Function reference, got {other:?}"),
                }
            }
            other => panic!("expected AllowedTools, got {other:?}"),
        }
        let reserialized = serde_json::to_value(&choice).expect("serialize");
        assert_eq!(reserialized, payload);
    }

    /// Mcp: `{"type": "mcp", "server_label", "name"?}` with and without the
    /// optional `name`.
    #[test]
    fn responses_tool_choice_mcp_round_trip() {
        // With `name`
        let payload_with_name = json!({
            "type": "mcp",
            "server_label": "my_server",
            "name": "search"
        });
        let choice: ResponsesToolChoice = serde_json::from_value(payload_with_name.clone())
            .expect("mcp with name should deserialize");
        match &choice {
            ResponsesToolChoice::Mcp {
                server_label, name, ..
            } => {
                assert_eq!(server_label, "my_server");
                assert_eq!(name.as_deref(), Some("search"));
            }
            other => panic!("expected Mcp, got {other:?}"),
        }
        let reserialized = serde_json::to_value(&choice).expect("serialize");
        assert_eq!(reserialized, payload_with_name);

        // Without `name`
        let payload_no_name = json!({
            "type": "mcp",
            "server_label": "my_server"
        });
        let choice: ResponsesToolChoice = serde_json::from_value(payload_no_name.clone())
            .expect("mcp without name should deserialize");
        match &choice {
            ResponsesToolChoice::Mcp {
                server_label, name, ..
            } => {
                assert_eq!(server_label, "my_server");
                assert!(name.is_none());
            }
            other => panic!("expected Mcp, got {other:?}"),
        }
        let reserialized = serde_json::to_value(&choice).expect("serialize");
        assert_eq!(reserialized, payload_no_name);
    }

    /// Custom: `{"type": "custom", "name": "..."}` — user-registered tool by name.
    #[test]
    fn responses_tool_choice_custom_round_trip() {
        let payload = json!({"type": "custom", "name": "my_custom_tool"});
        let choice: ResponsesToolChoice =
            serde_json::from_value(payload.clone()).expect("custom should deserialize");
        match &choice {
            ResponsesToolChoice::Custom { name, .. } => assert_eq!(name, "my_custom_tool"),
            other => panic!("expected Custom, got {other:?}"),
        }
        let reserialized = serde_json::to_value(&choice).expect("serialize");
        assert_eq!(reserialized, payload);
    }

    /// ApplyPatch: `{"type": "apply_patch"}`.
    #[test]
    fn responses_tool_choice_apply_patch_round_trip() {
        let payload = json!({"type": "apply_patch"});
        let choice: ResponsesToolChoice =
            serde_json::from_value(payload.clone()).expect("apply_patch should deserialize");
        assert!(matches!(choice, ResponsesToolChoice::ApplyPatch { .. }));
        let reserialized = serde_json::to_value(&choice).expect("serialize");
        assert_eq!(reserialized, payload);
    }

    /// Shell: `{"type": "shell"}`.
    #[test]
    fn responses_tool_choice_shell_round_trip() {
        let payload = json!({"type": "shell"});
        let choice: ResponsesToolChoice =
            serde_json::from_value(payload.clone()).expect("shell should deserialize");
        assert!(matches!(choice, ResponsesToolChoice::Shell { .. }));
        let reserialized = serde_json::to_value(&choice).expect("serialize");
        assert_eq!(reserialized, payload);
    }

    /// Projection onto Chat Completions' ToolChoice enum.
    /// - Options map to the same Value variants.
    /// - Function flattens to Chat's nested `{function: {name}}` wire form.
    /// - AllowedTools preserves mode and tool list.
    /// - Hosted / custom / apply_patch / shell / mcp have no Chat equivalent
    ///   and fall back to `auto`.
    #[test]
    fn responses_tool_choice_to_chat_projection() {
        assert!(matches!(
            ResponsesToolChoice::Options(ToolChoiceOptions::None).to_chat_tool_choice(),
            ChatToolChoice::Value(ChatToolChoiceValue::None)
        ));
        assert!(matches!(
            ResponsesToolChoice::Options(ToolChoiceOptions::Auto).to_chat_tool_choice(),
            ChatToolChoice::Value(ChatToolChoiceValue::Auto)
        ));
        assert!(matches!(
            ResponsesToolChoice::Options(ToolChoiceOptions::Required).to_chat_tool_choice(),
            ChatToolChoice::Value(ChatToolChoiceValue::Required)
        ));

        let fn_choice = ResponsesToolChoice::Function(ResponsesFunctionToolChoice {
            tool_type: FunctionToolChoiceTag::Function,
            name: "get_weather".to_string(),
        });
        match fn_choice.to_chat_tool_choice() {
            ChatToolChoice::Function {
                tool_type,
                function,
            } => {
                assert_eq!(tool_type, "function");
                assert_eq!(function.name, "get_weather");
            }
            other => panic!("expected Function, got {other:?}"),
        }

        let allowed = ResponsesToolChoice::AllowedTools {
            tool_type: AllowedToolsToolChoiceTag::AllowedTools,
            mode: "auto".to_string(),
            tools: vec![ToolReference::Function {
                name: "get_weather".to_string(),
            }],
        };
        match allowed.to_chat_tool_choice() {
            ChatToolChoice::AllowedTools {
                tool_type,
                mode,
                tools,
            } => {
                assert_eq!(tool_type, "allowed_tools");
                assert_eq!(mode, "auto");
                assert_eq!(tools.len(), 1);
            }
            other => panic!("expected AllowedTools, got {other:?}"),
        }

        // Responses-only variants collapse onto Chat's `auto` — there is no
        // spec-valid Chat projection for hosted / mcp / custom / apply_patch / shell.
        for variant in [
            ResponsesToolChoice::Types {
                tool_type: BuiltInToolChoiceType::FileSearch,
            },
            ResponsesToolChoice::Mcp {
                tool_type: McpToolChoiceTag::Mcp,
                server_label: "s".into(),
                name: None,
            },
            ResponsesToolChoice::Custom {
                tool_type: CustomToolChoiceTag::Custom,
                name: "c".into(),
            },
            ResponsesToolChoice::ApplyPatch {
                tool_type: ApplyPatchToolChoiceTag::ApplyPatch,
            },
            ResponsesToolChoice::Shell {
                tool_type: ShellToolChoiceTag::Shell,
            },
        ] {
            assert!(matches!(
                variant.to_chat_tool_choice(),
                ChatToolChoice::Value(ChatToolChoiceValue::Auto)
            ));
        }
    }

    /// The Default impl is `Options(Auto)`.
    #[test]
    fn responses_tool_choice_default_is_auto() {
        assert!(matches!(
            ResponsesToolChoice::default(),
            ResponsesToolChoice::Options(ToolChoiceOptions::Auto)
        ));
    }
}
