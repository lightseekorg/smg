//! Thin wrapper around the cross-router Responses history loader.

use axum::response::Response;
use openai_protocol::responses::ResponsesRequest;

use super::ResponsesContext;
use crate::routers::common::responses_history::load_request_history as load_shared_request_history;
pub(crate) use crate::routers::common::responses_history::LoadedHistory;

pub(crate) async fn load_request_history(
    ctx: &ResponsesContext,
    request: &ResponsesRequest,
) -> Result<LoadedHistory, Response> {
    load_shared_request_history(
        &ctx.response_storage,
        &ctx.conversation_storage,
        &ctx.conversation_item_storage,
        request,
    )
    .await
}
