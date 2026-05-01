//! Generic request-scoped bootstrap support for stateful Responses tools.
//!
//! The upstream runtime only knows that some tools require prepared state
//! before the first model call. Provider-specific lifecycle logic stays behind
//! the bootstrapper implementation, and the prepared state itself is stored as
//! opaque JSON so upstream does not learn OCI container/session semantics.

use std::{collections::BTreeSet, sync::Arc};

use async_trait::async_trait;
use axum::http::HeaderMap;
use openai_protocol::responses::{
    generate_id, ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseTool,
    ResponsesRequest,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use smg_data_connector::RequestContext as StorageRequestContext;

use crate::{memory::MemoryExecutionContext, middleware::TenantRequestMeta};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatefulToolKind {
    CodeInterpreter,
    Shell,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreparedToolState {
    pub kind: StatefulToolKind,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub value: Value,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StatefulToolBootstrapState {
    pub executed: bool,
    pub prepared_tools: Vec<PreparedToolState>,
}

impl StatefulToolBootstrapState {
    pub fn prepared_tool(&self, kind: StatefulToolKind) -> Option<&Value> {
        self.prepared_tools
            .iter()
            .find(|tool| tool.kind == kind)
            .map(|tool| &tool.value)
    }

    pub fn upsert_prepared_tool(&mut self, kind: StatefulToolKind, value: Value) {
        if let Some(existing) = self
            .prepared_tools
            .iter_mut()
            .find(|tool| tool.kind == kind)
        {
            existing.value = value;
            return;
        }
        self.prepared_tools.push(PreparedToolState { kind, value });
    }
}

#[derive(Debug, Default)]
pub struct StatefulToolBootstrapResult {
    pub prepared_tools: Vec<PreparedToolState>,
    pub injected_input_items: Vec<ResponseInputOutputItem>,
}

pub struct StatefulToolBootstrapContext<'a> {
    pub headers: Option<&'a HeaderMap>,
    pub storage_request_context: Option<&'a StorageRequestContext>,
    pub memory_execution_context: &'a MemoryExecutionContext,
    pub tenant_request_meta: Option<&'a TenantRequestMeta>,
}

#[async_trait]
pub trait StatefulToolBootstrapper: Send + Sync {
    async fn bootstrap(
        &self,
        request: &ResponsesRequest,
        context: StatefulToolBootstrapContext<'_>,
    ) -> Result<StatefulToolBootstrapResult, String>;
}

pub struct NoOpStatefulToolBootstrapper;

#[async_trait]
impl StatefulToolBootstrapper for NoOpStatefulToolBootstrapper {
    async fn bootstrap(
        &self,
        _request: &ResponsesRequest,
        _context: StatefulToolBootstrapContext<'_>,
    ) -> Result<StatefulToolBootstrapResult, String> {
        Ok(StatefulToolBootstrapResult::default())
    }
}

pub fn declared_stateful_tool_kinds(tools: Option<&[ResponseTool]>) -> Vec<StatefulToolKind> {
    let mut kinds = BTreeSet::new();

    for tool in tools.unwrap_or(&[]) {
        match tool {
            ResponseTool::CodeInterpreter(_) => {
                kinds.insert(StatefulToolKind::CodeInterpreter);
            }
            ResponseTool::Shell(_) => {
                kinds.insert(StatefulToolKind::Shell);
            }
            _ => {}
        }
    }

    kinds.into_iter().collect()
}

pub fn request_has_stateful_tools(request: &ResponsesRequest) -> bool {
    !declared_stateful_tool_kinds(request.tools.as_deref()).is_empty()
}

pub async fn ensure_stateful_tool_bootstrap(
    request: &mut ResponsesRequest,
    bootstrap_state: &mut StatefulToolBootstrapState,
    bootstrapper: &dyn StatefulToolBootstrapper,
    context: StatefulToolBootstrapContext<'_>,
) -> Result<(), String> {
    if bootstrap_state.executed || !request_has_stateful_tools(request) {
        return Ok(());
    }

    // INVARIANT: bootstrap failures short-circuit the route immediately, so
    // `executed` is only flipped after a successful injection/preparation.
    let result = bootstrapper.bootstrap(request, context).await?;
    prepend_injected_items(&mut request.input, result.injected_input_items);

    bootstrap_state.executed = true;
    for tool in result.prepared_tools {
        bootstrap_state.upsert_prepared_tool(tool.kind, tool.value);
    }

    Ok(())
}

fn prepend_injected_items(
    input: &mut ResponseInput,
    mut injected_items: Vec<ResponseInputOutputItem>,
) {
    if injected_items.is_empty() {
        return;
    }

    match input {
        ResponseInput::Text(text) => {
            injected_items.push(ResponseInputOutputItem::Message {
                id: generate_id("msg"),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText { text: text.clone() }],
                status: None,
                phase: None,
            });
            *input = ResponseInput::Items(injected_items);
        }
        ResponseInput::Items(existing_items) => {
            injected_items.append(existing_items);
            *existing_items = injected_items;
        }
    }
}

pub type SharedStatefulToolBootstrapper = Arc<dyn StatefulToolBootstrapper>;

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use openai_protocol::responses::{
        CodeInterpreterTool, ResponseToolEnvironment, ShellTool, WebSearchPreviewTool,
    };
    use serde_json::json;

    use super::*;

    struct CountingBootstrapper {
        calls: Arc<AtomicUsize>,
        result: StatefulToolBootstrapResult,
    }

    #[async_trait]
    impl StatefulToolBootstrapper for CountingBootstrapper {
        async fn bootstrap(
            &self,
            _request: &ResponsesRequest,
            _context: StatefulToolBootstrapContext<'_>,
        ) -> Result<StatefulToolBootstrapResult, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(StatefulToolBootstrapResult {
                prepared_tools: self.result.prepared_tools.clone(),
                injected_input_items: self.result.injected_input_items.clone(),
            })
        }
    }

    struct FailingBootstrapper;

    #[async_trait]
    impl StatefulToolBootstrapper for FailingBootstrapper {
        async fn bootstrap(
            &self,
            _request: &ResponsesRequest,
            _context: StatefulToolBootstrapContext<'_>,
        ) -> Result<StatefulToolBootstrapResult, String> {
            Err("boom".to_string())
        }
    }

    fn bootstrap_context<'a>(
        memory_execution_context: &'a MemoryExecutionContext,
    ) -> StatefulToolBootstrapContext<'a> {
        StatefulToolBootstrapContext {
            headers: None,
            storage_request_context: None,
            memory_execution_context,
            tenant_request_meta: None,
        }
    }

    #[test]
    fn declared_stateful_tool_kinds_deduplicates_supported_tools() {
        let tools = vec![
            ResponseTool::WebSearchPreview(WebSearchPreviewTool::default()),
            ResponseTool::Shell(ShellTool::default()),
            ResponseTool::CodeInterpreter(CodeInterpreterTool {
                container: None,
                environment: Some(ResponseToolEnvironment::default()),
            }),
            ResponseTool::Shell(ShellTool::default()),
        ];

        assert_eq!(
            declared_stateful_tool_kinds(Some(&tools)),
            vec![StatefulToolKind::CodeInterpreter, StatefulToolKind::Shell]
        );
    }

    #[tokio::test]
    async fn ensure_stateful_tool_bootstrap_skips_requests_without_stateful_tools() {
        let calls = Arc::new(AtomicUsize::new(0));
        let bootstrapper = CountingBootstrapper {
            calls: Arc::clone(&calls),
            result: StatefulToolBootstrapResult::default(),
        };
        let memory_execution_context = MemoryExecutionContext::default();
        let mut request = ResponsesRequest {
            input: ResponseInput::Text("hello".to_string()),
            tools: Some(vec![ResponseTool::WebSearchPreview(
                WebSearchPreviewTool::default(),
            )]),
            ..Default::default()
        };
        let mut bootstrap_state = StatefulToolBootstrapState::default();

        ensure_stateful_tool_bootstrap(
            &mut request,
            &mut bootstrap_state,
            &bootstrapper,
            bootstrap_context(&memory_execution_context),
        )
        .await
        .expect("bootstrap should succeed");

        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(!bootstrap_state.executed);
    }

    #[tokio::test]
    async fn ensure_stateful_tool_bootstrap_injects_context_and_runs_once() {
        let calls = Arc::new(AtomicUsize::new(0));
        let bootstrapper = CountingBootstrapper {
            calls: Arc::clone(&calls),
            result: StatefulToolBootstrapResult {
                prepared_tools: vec![PreparedToolState {
                    kind: StatefulToolKind::Shell,
                    value: json!({"session_id": "sess_123"}),
                }],
                injected_input_items: vec![ResponseInputOutputItem::Message {
                    id: "msg_bootstrap".to_string(),
                    role: "developer".to_string(),
                    content: vec![ResponseContentPart::InputText {
                        text: "Resolved shell session is available.".to_string(),
                    }],
                    status: Some("completed".to_string()),
                    phase: None,
                }],
            },
        };
        let memory_execution_context = MemoryExecutionContext::default();
        let mut request = ResponsesRequest {
            input: ResponseInput::Text("hello".to_string()),
            tools: Some(vec![ResponseTool::Shell(ShellTool::default())]),
            ..Default::default()
        };
        let mut bootstrap_state = StatefulToolBootstrapState::default();

        ensure_stateful_tool_bootstrap(
            &mut request,
            &mut bootstrap_state,
            &bootstrapper,
            bootstrap_context(&memory_execution_context),
        )
        .await
        .expect("first bootstrap should succeed");
        ensure_stateful_tool_bootstrap(
            &mut request,
            &mut bootstrap_state,
            &bootstrapper,
            bootstrap_context(&memory_execution_context),
        )
        .await
        .expect("second bootstrap should be a no-op");

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(bootstrap_state.executed);
        assert_eq!(
            bootstrap_state.prepared_tool(StatefulToolKind::Shell),
            Some(&json!({"session_id": "sess_123"}))
        );

        let ResponseInput::Items(items) = &request.input else {
            panic!("bootstrap should normalize text input to items");
        };
        assert_eq!(items.len(), 2);
        assert!(matches!(
            &items[0],
            ResponseInputOutputItem::Message { role, .. } if role == "developer"
        ));
        assert!(matches!(
            &items[1],
            ResponseInputOutputItem::Message { role, .. } if role == "user"
        ));
    }

    #[tokio::test]
    async fn ensure_stateful_tool_bootstrap_error_is_non_mutating() {
        let memory_execution_context = MemoryExecutionContext::default();
        let mut request = ResponsesRequest {
            input: ResponseInput::Text("hello".to_string()),
            tools: Some(vec![ResponseTool::Shell(ShellTool::default())]),
            ..Default::default()
        };
        let original_input = serde_json::to_value(&request.input).expect("serialize input");
        let mut bootstrap_state = StatefulToolBootstrapState::default();

        let err = ensure_stateful_tool_bootstrap(
            &mut request,
            &mut bootstrap_state,
            &FailingBootstrapper,
            bootstrap_context(&memory_execution_context),
        )
        .await
        .expect_err("bootstrap should fail");

        assert_eq!(err, "boom");
        assert!(!bootstrap_state.executed);
        assert_eq!(
            serde_json::to_value(&request.input).expect("serialize input"),
            original_input
        );
        assert!(bootstrap_state.prepared_tools.is_empty());
    }
}
