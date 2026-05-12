use openai_protocol::{
    chat::{ChatCompletionRequest, ChatMessage, MessageContent},
    common::{Tool, ToolChoice, ToolChoiceValue},
};
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BedrockConverseRequest {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub system: Vec<ContentBlock>,
    pub messages: Vec<BedrockMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_config: Option<InferenceConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
}

#[derive(Debug, Serialize)]
pub(crate) struct BedrockMessage {
    pub role: String,
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ContentBlock {
    pub text: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InferenceConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ToolConfig {
    pub tools: Vec<BedrockTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<BedrockToolChoice>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BedrockTool {
    pub tool_spec: BedrockToolSpec,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BedrockToolSpec {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: BedrockInputSchema,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BedrockInputSchema {
    pub json: Value,
}

/// Bedrock toolChoice is an object with exactly one key indicating the mode.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum BedrockToolChoice {
    Auto(EmptyObject),
    Any(EmptyObject),
    Tool(BedrockToolName),
}

#[derive(Debug, Serialize)]
pub(crate) struct EmptyObject {}

#[derive(Debug, Serialize)]
pub(crate) struct BedrockToolName {
    pub name: String,
}

pub(crate) fn map_chat_request(request: &ChatCompletionRequest) -> BedrockConverseRequest {
    let mut system = Vec::new();
    let mut messages = Vec::new();

    for msg in &request.messages {
        match msg {
            ChatMessage::System { content, .. } | ChatMessage::Developer { content, .. } => {
                push_text_blocks(&mut system, content);
            }
            ChatMessage::User { content, .. } => {
                let content = content_blocks_from_content(content);
                if !content.is_empty() {
                    messages.push(BedrockMessage {
                        role: "user".to_string(),
                        content,
                    });
                }
            }
            ChatMessage::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut blocks = Vec::new();
                if let Some(content) = content {
                    blocks.extend(content_blocks_from_content(content));
                }
                if let Some(calls) = tool_calls {
                    let tools_text = calls
                        .iter()
                        .map(|c| {
                            format!(
                                "tool_call {} {}",
                                c.function.name,
                                c.function.arguments.clone().unwrap_or_default()
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    if !tools_text.is_empty() {
                        blocks.push(ContentBlock { text: tools_text });
                    }
                }
                if !blocks.is_empty() {
                    messages.push(BedrockMessage {
                        role: "assistant".to_string(),
                        content: blocks,
                    });
                }
            }
            ChatMessage::Tool {
                content,
                tool_call_id,
            } => {
                let mut text = content.to_simple_string();
                if !tool_call_id.is_empty() {
                    text = format!("tool_result {tool_call_id}: {text}");
                }
                if !text.is_empty() {
                    messages.push(BedrockMessage {
                        role: "user".to_string(),
                        content: vec![ContentBlock { text }],
                    });
                }
            }
            ChatMessage::Function { content, name } => {
                if !content.is_empty() || !name.is_empty() {
                    messages.push(BedrockMessage {
                        role: "user".to_string(),
                        content: vec![ContentBlock {
                            text: format!("function {name}: {content}"),
                        }],
                    });
                }
            }
        }
    }

    if messages.is_empty() {
        messages.push(BedrockMessage {
            role: "user".to_string(),
            content: vec![ContentBlock {
                text: "[no input]".to_string(),
            }],
        });
    }

    let inference_config = Some(InferenceConfig {
        max_tokens: max_tokens(request),
        temperature: request.temperature,
        top_p: request.top_p,
        stop_sequences: stop_sequences_from_value(request.stop.as_ref()),
    });

    BedrockConverseRequest {
        system,
        messages,
        inference_config,
        tool_config: map_tool_config(request.tools.as_deref(), request.tool_choice.as_ref()),
    }
}

#[expect(deprecated)]
fn max_tokens(request: &ChatCompletionRequest) -> Option<u32> {
    request.max_completion_tokens.or(request.max_tokens)
}

fn map_tool_config(tools: Option<&[Tool]>, tool_choice: Option<&ToolChoice>) -> Option<ToolConfig> {
    if matches!(tool_choice, Some(ToolChoice::Value(ToolChoiceValue::None))) {
        return None;
    }

    let tools = tools?;
    if tools.is_empty() {
        return None;
    }

    let bedrock_tools: Vec<BedrockTool> = tools
        .iter()
        .map(|t| BedrockTool {
            tool_spec: BedrockToolSpec {
                name: t.function.name.clone(),
                description: t.function.description.clone(),
                input_schema: BedrockInputSchema {
                    json: t.function.parameters.clone(),
                },
            },
        })
        .collect();

    let bedrock_choice = tool_choice.and_then(map_tool_choice);

    Some(ToolConfig {
        tools: bedrock_tools,
        tool_choice: bedrock_choice,
    })
}

fn map_tool_choice(choice: &ToolChoice) -> Option<BedrockToolChoice> {
    match choice {
        ToolChoice::Value(ToolChoiceValue::Auto) => Some(BedrockToolChoice::Auto(EmptyObject {})),
        ToolChoice::Value(ToolChoiceValue::Required) => {
            Some(BedrockToolChoice::Any(EmptyObject {}))
        }
        ToolChoice::Value(ToolChoiceValue::None) => None,
        ToolChoice::Function { function, .. } => Some(BedrockToolChoice::Tool(BedrockToolName {
            name: function.name.clone(),
        })),
        ToolChoice::AllowedTools { .. } => Some(BedrockToolChoice::Auto(EmptyObject {})),
    }
}

fn push_text_blocks(dst: &mut Vec<ContentBlock>, content: &MessageContent) {
    dst.extend(content_blocks_from_content(content));
}

fn content_blocks_from_content(content: &MessageContent) -> Vec<ContentBlock> {
    let text = content.to_simple_string();
    if text.is_empty() {
        Vec::new()
    } else {
        vec![ContentBlock { text }]
    }
}

fn stop_sequences_from_value(
    v: Option<&openai_protocol::common::StringOrArray>,
) -> Option<Vec<String>> {
    match v {
        Some(openai_protocol::common::StringOrArray::String(s)) if !s.is_empty() => {
            Some(vec![s.clone()])
        }
        Some(openai_protocol::common::StringOrArray::Array(a)) => {
            let list = a
                .iter()
                .filter(|s| !s.is_empty())
                .cloned()
                .collect::<Vec<_>>();
            if list.is_empty() {
                None
            } else {
                Some(list)
            }
        }
        _ => None,
    }
}
