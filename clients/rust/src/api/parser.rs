use openai_protocol::parser::{
    ParseFunctionCallRequest, ParseFunctionCallResponse, SeparateReasoningRequest,
    SeparateReasoningResponse,
};

use crate::{transport::Transport, SmgError};

/// Parser API (`/parse/*`).
pub struct Parser {
    transport: Transport,
}

impl Parser {
    pub(crate) fn new(transport: Transport) -> Self {
        Self { transport }
    }

    /// Parse function/tool calls from model output text.
    pub async fn parse_function_call(
        &self,
        request: &ParseFunctionCallRequest,
    ) -> Result<ParseFunctionCallResponse, SmgError> {
        let resp = self.transport.post("/parse/function_call", request).await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }

    /// Separate reasoning content from normal text in model output.
    pub async fn separate_reasoning(
        &self,
        request: &SeparateReasoningRequest,
    ) -> Result<SeparateReasoningResponse, SmgError> {
        let resp = self.transport.post("/parse/reasoning", request).await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }
}
