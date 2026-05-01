//! Thin wrapper around the cross-router Responses history preparer.

use axum::response::Response;
use openai_protocol::responses::ResponsesRequest;

use super::ResponsesContext;
use crate::routers::common::responses_history::{
    prepare_request_history as prepare_shared_request_history, PreparedRequestHistory,
};

pub(crate) async fn prepare_request_history(
    ctx: &ResponsesContext,
    request: &ResponsesRequest,
) -> Result<PreparedRequestHistory, Response> {
    prepare_shared_request_history(
        &ctx.response_storage,
        &ctx.conversation_storage,
        &ctx.conversation_item_storage,
        request,
    )
    .await
}
