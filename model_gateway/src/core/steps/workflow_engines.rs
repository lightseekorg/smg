//! Typed workflow engines collection
//!
//! This module provides a collection of typed workflow engines for different workflow types.
//! Each workflow type has its own engine with compile-time type safety.
//!
//! When an Oracle state store config is provided, workflow state is persisted to Oracle.
//! Otherwise, an in-memory store is used (state is lost on restart).

use std::sync::Arc;

use wfaas::{
    ArcStateStore, EventSubscriber, OracleStateStore, OracleStateStoreConfig, WorkflowData,
    WorkflowEngine,
};

use super::{
    create_mcp_registration_workflow, create_tokenizer_registration_workflow,
    create_wasm_module_registration_workflow, create_wasm_module_removal_workflow,
    create_worker_registration_workflow, create_worker_removal_workflow,
    create_worker_update_workflow, McpWorkflowData, TokenizerWorkflowData,
    WasmRegistrationWorkflowData, WasmRemovalWorkflowData, WorkerRemovalWorkflowData,
    WorkerUpdateWorkflowData,
};
use crate::{config::RouterConfig, core::steps::workflow_data::WorkerWorkflowData};

/// Type alias for the unified worker registration workflow engine
pub type WorkerRegistrationEngine =
    WorkflowEngine<WorkerWorkflowData, ArcStateStore<WorkerWorkflowData>>;

/// Type alias for worker removal workflow engine
pub type WorkerRemovalEngine =
    WorkflowEngine<WorkerRemovalWorkflowData, ArcStateStore<WorkerRemovalWorkflowData>>;

/// Type alias for worker update workflow engine
pub type WorkerUpdateEngine =
    WorkflowEngine<WorkerUpdateWorkflowData, ArcStateStore<WorkerUpdateWorkflowData>>;

/// Type alias for MCP registration workflow engine
pub type McpEngine = WorkflowEngine<McpWorkflowData, ArcStateStore<McpWorkflowData>>;

/// Type alias for tokenizer registration workflow engine
pub type TokenizerEngine =
    WorkflowEngine<TokenizerWorkflowData, ArcStateStore<TokenizerWorkflowData>>;

/// Type alias for WASM registration workflow engine
pub type WasmRegistrationEngine =
    WorkflowEngine<WasmRegistrationWorkflowData, ArcStateStore<WasmRegistrationWorkflowData>>;

/// Type alias for WASM removal workflow engine
pub type WasmRemovalEngine =
    WorkflowEngine<WasmRemovalWorkflowData, ArcStateStore<WasmRemovalWorkflowData>>;

/// Collection of typed workflow engines
///
/// Each workflow type has its own engine with compile-time type safety.
/// This replaces the old `WorkflowEngine<AnyWorkflowData, ...>` approach.
#[derive(Clone, Debug)]
pub struct WorkflowEngines {
    /// Engine for unified worker registration workflows (local + external)
    pub worker_registration: Arc<WorkerRegistrationEngine>,
    /// Engine for worker removal workflows
    pub worker_removal: Arc<WorkerRemovalEngine>,
    /// Engine for worker update workflows
    pub worker_update: Arc<WorkerUpdateEngine>,
    /// Engine for MCP server registration workflows
    pub mcp: Arc<McpEngine>,
    /// Engine for tokenizer registration workflows
    pub tokenizer: Arc<TokenizerEngine>,
    /// Engine for WASM module registration workflows
    pub wasm_registration: Arc<WasmRegistrationEngine>,
    /// Engine for WASM module removal workflows
    pub wasm_removal: Arc<WasmRemovalEngine>,
}

impl WorkflowEngines {
    /// Create and initialize all workflow engines with their workflow definitions.
    ///
    /// When `oracle_config` is `Some`, workflow state is persisted to Oracle.
    /// Otherwise, an in-memory store is used.
    #[expect(
        clippy::expect_used,
        reason = "Workflow registration uses compile-time-known step/transition definitions that cannot fail at runtime — a failure here indicates a programming error in workflow construction"
    )]
    pub fn new(
        router_config: &RouterConfig,
        oracle_config: Option<&OracleStateStoreConfig>,
    ) -> Self {
        fn make_store<D: WorkflowData>(
            oracle_config: Option<&OracleStateStoreConfig>,
        ) -> ArcStateStore<D> {
            match oracle_config {
                Some(cfg) => match OracleStateStore::<D>::new(cfg) {
                    Ok(store) => ArcStateStore::oracle(store),
                    Err(e) => {
                        tracing::error!(
                            error = %e,
                            workflow_type = D::workflow_type(),
                            "Failed to create Oracle state store, falling back to in-memory"
                        );
                        ArcStateStore::memory()
                    }
                },
                None => ArcStateStore::memory(),
            }
        }

        // Create unified worker registration engine
        let worker_registration =
            WorkflowEngine::with_store(make_store::<WorkerWorkflowData>(oracle_config));
        worker_registration
            .register_workflow(create_worker_registration_workflow(router_config))
            .expect("worker_registration workflow should be valid");

        // Create worker removal engine
        let worker_removal =
            WorkflowEngine::with_store(make_store::<WorkerRemovalWorkflowData>(oracle_config));
        worker_removal
            .register_workflow(create_worker_removal_workflow())
            .expect("worker_removal workflow should be valid");

        // Create worker update engine
        let worker_update =
            WorkflowEngine::with_store(make_store::<WorkerUpdateWorkflowData>(oracle_config));
        worker_update
            .register_workflow(create_worker_update_workflow())
            .expect("worker_update workflow should be valid");

        // Create MCP engine
        let mcp = WorkflowEngine::with_store(make_store::<McpWorkflowData>(oracle_config));
        mcp.register_workflow(create_mcp_registration_workflow())
            .expect("mcp_registration workflow should be valid");

        // Create tokenizer engine
        let tokenizer =
            WorkflowEngine::with_store(make_store::<TokenizerWorkflowData>(oracle_config));
        tokenizer
            .register_workflow(create_tokenizer_registration_workflow())
            .expect("tokenizer_registration workflow should be valid");

        // Create WASM registration engine
        let wasm_registration =
            WorkflowEngine::with_store(make_store::<WasmRegistrationWorkflowData>(oracle_config));
        wasm_registration
            .register_workflow(create_wasm_module_registration_workflow())
            .expect("wasm_module_registration workflow should be valid");

        // Create WASM removal engine
        let wasm_removal =
            WorkflowEngine::with_store(make_store::<WasmRemovalWorkflowData>(oracle_config));
        wasm_removal
            .register_workflow(create_wasm_module_removal_workflow())
            .expect("wasm_module_removal workflow should be valid");

        Self {
            worker_registration: Arc::new(worker_registration),
            worker_removal: Arc::new(worker_removal),
            worker_update: Arc::new(worker_update),
            mcp: Arc::new(mcp),
            tokenizer: Arc::new(tokenizer),
            wasm_registration: Arc::new(wasm_registration),
            wasm_removal: Arc::new(wasm_removal),
        }
    }

    /// Subscribe an event subscriber to all workflow engines
    pub async fn subscribe_all<S: EventSubscriber + 'static>(&self, subscriber: Arc<S>) {
        self.worker_registration
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.worker_removal
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.worker_update
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.mcp.event_bus().subscribe(subscriber.clone()).await;
        self.tokenizer
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.wasm_registration
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.wasm_removal.event_bus().subscribe(subscriber).await;
    }
}
