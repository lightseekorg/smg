//! gRPC service implementation for the Render service.
//!
//! Converts proto types to internal types and delegates to shared core logic.

use std::sync::Arc;

use llm_tokenizer::TokenizerRegistry;
use openai_protocol::{
    chat::{ChatMessage, MessageContent},
    common::{Function, FunctionCallResponse, Tool, ToolChoice, ToolChoiceValue},
};
use smg_grpc_client::render_service_proto::{self as proto, render_service_server::RenderService};
use tonic::{Request, Response, Status};

use super::handlers;

/// gRPC service for rendering (chat template + tokenization, no generation).
pub struct RenderGrpcService {
    tokenizer_registry: Arc<TokenizerRegistry>,
}

impl RenderGrpcService {
    pub fn new(tokenizer_registry: Arc<TokenizerRegistry>) -> Self {
        Self { tokenizer_registry }
    }
}

#[tonic::async_trait]
impl RenderService for RenderGrpcService {
    async fn render_chat(
        &self,
        request: Request<proto::RenderChatRequest>,
    ) -> Result<Response<proto::RenderResponse>, Status> {
        let req = request.into_inner();

        let chat_request = proto_to_chat_request(&req)
            .map_err(|e| Status::invalid_argument(format!("Invalid request: {e}")))?;

        let token_ids = handlers::render_chat_core(&self.tokenizer_registry, &chat_request)
            .map_err(Status::internal)?;

        Ok(Response::new(proto::RenderResponse {
            count: token_ids.len() as u32,
            token_ids,
        }))
    }

    async fn render_completion(
        &self,
        request: Request<proto::RenderCompletionRequest>,
    ) -> Result<Response<proto::RenderResponse>, Status> {
        let req = request.into_inner();

        let prompt = extract_prompt_text(&req)
            .map_err(|e| Status::invalid_argument(format!("Invalid prompt: {e}")))?;
        let add_special_tokens = req.add_special_tokens.unwrap_or(true);

        let token_ids = handlers::render_completion_core(
            &self.tokenizer_registry,
            &req.model,
            &prompt,
            add_special_tokens,
        )
        .map_err(Status::internal)?;

        Ok(Response::new(proto::RenderResponse {
            count: token_ids.len() as u32,
            token_ids,
        }))
    }
}

// ============================================================================
// Proto → Internal type conversions
// ============================================================================

fn proto_to_chat_request(
    req: &proto::RenderChatRequest,
) -> Result<openai_protocol::chat::ChatCompletionRequest, String> {
    let messages: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(proto_message_to_chat_message)
        .collect::<Result<Vec<_>, _>>()?;

    let tools: Option<Vec<Tool>> = if req.tools.is_empty() {
        None
    } else {
        Some(
            req.tools
                .iter()
                .map(|t| {
                    let func = t.function.as_ref().ok_or("Tool missing function")?;
                    let params = func
                        .parameters_json
                        .as_ref()
                        .and_then(|s| serde_json::from_str(s).ok());
                    Ok(Tool {
                        tool_type: "function".to_string(),
                        function: Function {
                            name: func.name.clone(),
                            description: func.description.clone(),
                            parameters: params
                                .unwrap_or(serde_json::Value::Object(Default::default())),
                            strict: func.strict,
                        },
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
        )
    };

    let tool_choice = req.tool_choice.as_deref().and_then(|s| match s {
        "none" => Some(ToolChoice::Value(ToolChoiceValue::None)),
        "auto" => Some(ToolChoice::Value(ToolChoiceValue::Auto)),
        "required" => Some(ToolChoice::Value(ToolChoiceValue::Required)),
        _ => serde_json::from_str(s).ok(),
    });

    let continue_final_message = req.continue_final_message.unwrap_or(false);

    Ok(openai_protocol::chat::ChatCompletionRequest {
        model: req.model.clone(),
        messages,
        tools,
        tool_choice,
        continue_final_message,
        ..Default::default()
    })
}

fn proto_message_to_chat_message(
    msg: &proto::ChatCompletionMessage,
) -> Result<ChatMessage, String> {
    let content_from_str = |s: &Option<String>| -> MessageContent {
        MessageContent::Text(s.clone().unwrap_or_default())
    };

    match msg.role.as_str() {
        "system" => Ok(ChatMessage::System {
            content: content_from_str(&msg.content),
            name: msg.name.clone(),
        }),
        "user" => Ok(ChatMessage::User {
            content: content_from_str(&msg.content),
            name: msg.name.clone(),
        }),
        "assistant" => {
            let tool_calls = if msg.tool_calls.is_empty() {
                None
            } else {
                Some(
                    msg.tool_calls
                        .iter()
                        .map(|tc| {
                            let func = tc.function.as_ref().ok_or("ToolCall missing function")?;
                            Ok(openai_protocol::common::ToolCall {
                                id: tc.id.clone(),
                                tool_type: "function".to_string(),
                                function: FunctionCallResponse {
                                    name: func.name.clone(),
                                    arguments: Some(func.arguments.clone()),
                                },
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?,
                )
            };
            Ok(ChatMessage::Assistant {
                content: msg
                    .content
                    .as_ref()
                    .map(|s| MessageContent::Text(s.clone())),
                name: msg.name.clone(),
                tool_calls,
                reasoning_content: msg.reasoning.clone(),
            })
        }
        "tool" => Ok(ChatMessage::Tool {
            content: content_from_str(&msg.content),
            tool_call_id: msg.tool_call_id.clone().unwrap_or_default(),
        }),
        "developer" => Ok(ChatMessage::Developer {
            content: content_from_str(&msg.content),
            tools: None,
            name: msg.name.clone(),
        }),
        other => Err(format!("Unknown role: {other}")),
    }
}

fn extract_prompt_text(req: &proto::RenderCompletionRequest) -> Result<String, String> {
    match &req.prompt {
        Some(prompt) => match &prompt.prompt {
            Some(proto::completion_prompt::Prompt::Text(s)) => Ok(s.clone()),
            Some(proto::completion_prompt::Prompt::Texts(texts)) => texts
                .texts
                .first()
                .cloned()
                .ok_or_else(|| "Empty texts array".to_string()),
            Some(proto::completion_prompt::Prompt::TokenIds(_)) => {
                Err("Token ID input not supported for render; use text input".to_string())
            }
            Some(proto::completion_prompt::Prompt::TokenIdBatches(_)) => {
                Err("Token ID batch input not supported for render; use text input".to_string())
            }
            None => Err("Empty prompt".to_string()),
        },
        None => Err("Missing prompt field".to_string()),
    }
}
