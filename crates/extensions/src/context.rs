use std::sync::Arc;

use axum::http::HeaderMap;
use openai_protocol::responses::ResponsesRequest;
use serde_json::Value;
use smg_data_connector::{
    ConversationId, ConversationItemId, ConversationItemStorage, ResponseId,
};

use crate::metadata::{ConversationTurnInfo, RequestMetadata};

#[non_exhaustive]
pub struct BeforeModelCtx<'a> {
    pub headers: &'a HeaderMap,
    pub request: &'a mut ResponsesRequest,
    pub conversation_id: Option<&'a ConversationId>,
    pub history: Arc<dyn ConversationItemStorage>,
    pub turn_info: ConversationTurnInfo,
    pub request_metadata: &'a RequestMetadata,
}

#[non_exhaustive]
pub struct AfterPersistCtx<'a> {
    pub headers: &'a HeaderMap,
    pub request: &'a ResponsesRequest,
    pub response_json: Option<&'a Value>,
    pub response_id: Option<&'a ResponseId>,
    pub conversation_id: Option<&'a ConversationId>,
    pub turn_info: ConversationTurnInfo,
    pub persisted_item_ids: &'a [ConversationItemId],
    pub request_metadata: &'a RequestMetadata,
}
