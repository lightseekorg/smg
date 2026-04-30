use std::{
    sync::{Arc, OnceLock},
    time::Duration,
};

use llm_tokenizer::registry::TokenizerRegistry;
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use reqwest::Client;
use smg_blob_storage::create_blob_store;
use smg_data_connector::{
    backend_supports_memory_writer, create_storage, BackgroundResponseRepository,
    ConversationItemStorage, ConversationMemoryWriter, ConversationStorage, ResponseStorage,
    StorageFactoryConfig,
};
use smg_mcp::McpOrchestrator;
use smg_skills::{SkillService, SkillUploadLimits};
use tool_parser::ParserFactory as ToolParserFactory;
use tracing::debug;

use crate::{
    config::RouterConfig,
    middleware::TokenBucket,
    observability::inflight_tracker::InFlightRequestTracker,
    policies::PolicyRegistry,
    routers::{
        grpc::multimodal::MultimodalConfigRegistry, openai::realtime::RealtimeRegistry,
        router_manager::RouterManager,
    },
    wasm::{config::WasmRuntimeConfig, module_manager::WasmModuleManager},
    worker::{KvEventMonitor, WorkerMonitor, WorkerRegistry, WorkerService},
    workflow::{JobQueue, WorkflowEngines},
};

/// Error type for AppContext builder
#[derive(Debug)]
pub enum AppContextBuildError {
    MissingField(&'static str),
    InvalidConfig(String),
}

impl std::fmt::Display for AppContextBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingField(field) => write!(f, "Missing required field: {field}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid configuration: {msg}"),
        }
    }
}

impl std::error::Error for AppContextBuildError {}

#[derive(Clone)]
pub struct AppContext {
    pub client: Client,
    pub router_config: RouterConfig,
    pub rate_limiter: Option<Arc<TokenBucket>>,
    pub tokenizer_registry: Arc<TokenizerRegistry>,
    pub multimodal_config_registry: Arc<MultimodalConfigRegistry>,
    pub reasoning_parser_factory: Option<ReasoningParserFactory>,
    pub tool_parser_factory: Option<ToolParserFactory>,
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub router_manager: Option<Arc<RouterManager>>,
    pub response_storage: Arc<dyn ResponseStorage>,
    pub conversation_storage: Arc<dyn ConversationStorage>,
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,
    /// Writer used for long-term-memory persistence (NoOp when backend does not support writes).
    pub conversation_memory_writer: Arc<dyn ConversationMemoryWriter>,
    pub background_repository: Option<Arc<dyn BackgroundResponseRepository>>,
    pub worker_monitor: Option<Arc<WorkerMonitor>>,
    pub configured_reasoning_parser: Option<String>,
    pub configured_tool_parser: Option<String>,
    pub worker_job_queue: Arc<OnceLock<Arc<JobQueue>>>,
    pub workflow_engines: Arc<OnceLock<WorkflowEngines>>,
    pub mcp_orchestrator: Arc<OnceLock<Arc<McpOrchestrator>>>,
    pub skill_service: Option<Arc<SkillService>>,
    pub wasm_manager: Option<Arc<WasmModuleManager>>,
    pub worker_service: Arc<WorkerService>,
    pub inflight_tracker: Arc<InFlightRequestTracker>,
    pub kv_event_monitor: Option<Arc<KvEventMonitor>>,
    pub realtime_registry: Arc<RealtimeRegistry>,
    /// Bind address for WebRTC UDP sockets (`None` = `0.0.0.0`, auto-detect).
    pub webrtc_bind_addr: Option<std::net::IpAddr>,
    /// STUN server for ICE candidate gathering (`None` = `stun.l.google.com:19302`).
    pub webrtc_stun_server: Option<String>,
}

impl std::fmt::Debug for AppContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppContext")
            .field("router_config", &self.router_config)
            .finish_non_exhaustive()
    }
}

pub struct AppContextBuilder {
    client: Option<Client>,
    router_config: Option<RouterConfig>,
    rate_limiter: Option<Arc<TokenBucket>>,
    tokenizer_registry: Option<Arc<TokenizerRegistry>>,
    reasoning_parser_factory: Option<ReasoningParserFactory>,
    tool_parser_factory: Option<ToolParserFactory>,
    worker_registry: Option<Arc<WorkerRegistry>>,
    policy_registry: Option<Arc<PolicyRegistry>>,
    router_manager: Option<Arc<RouterManager>>,
    response_storage: Option<Arc<dyn ResponseStorage>>,
    conversation_storage: Option<Arc<dyn ConversationStorage>>,
    conversation_item_storage: Option<Arc<dyn ConversationItemStorage>>,
    conversation_memory_writer: Option<Arc<dyn ConversationMemoryWriter>>,
    background_repository: Option<Arc<dyn BackgroundResponseRepository>>,
    worker_monitor: Option<Arc<WorkerMonitor>>,
    worker_job_queue: Option<Arc<OnceLock<Arc<JobQueue>>>>,
    workflow_engines: Option<Arc<OnceLock<WorkflowEngines>>>,
    mcp_orchestrator: Option<Arc<OnceLock<Arc<McpOrchestrator>>>>,
    skill_service: Option<Arc<SkillService>>,
    wasm_manager: Option<Arc<WasmModuleManager>>,
    kv_event_monitor: Option<Arc<KvEventMonitor>>,
    webrtc_bind_addr: Option<std::net::IpAddr>,
    webrtc_stun_server: Option<String>,
}

impl AppContext {
    pub fn builder() -> AppContextBuilder {
        AppContextBuilder::new()
    }

    /// Create AppContext from config with all components initialized
    /// This is the main entry point that replaces ~194 lines of initialization in server.rs
    pub fn from_config(
        router_config: RouterConfig,
        request_timeout_secs: u64,
        webrtc_bind_addr: Option<std::net::IpAddr>,
        webrtc_stun_server: Option<String>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self, String>> + Send>> {
        Box::pin(async move {
            Box::pin(AppContextBuilder::from_config(
                router_config,
                request_timeout_secs,
                webrtc_bind_addr,
                webrtc_stun_server,
            ))
            .await?
            .build()
            .map_err(|e| e.to_string())
        })
    }
}

impl AppContextBuilder {
    pub fn new() -> Self {
        Self {
            client: None,
            router_config: None,
            rate_limiter: None,
            tokenizer_registry: None,
            reasoning_parser_factory: None,
            tool_parser_factory: None,
            worker_registry: None,
            policy_registry: None,
            router_manager: None,
            response_storage: None,
            conversation_storage: None,
            conversation_item_storage: None,
            conversation_memory_writer: None,
            background_repository: None,
            worker_monitor: None,
            worker_job_queue: None,
            workflow_engines: None,
            mcp_orchestrator: None,
            skill_service: None,
            wasm_manager: None,
            kv_event_monitor: None,
            webrtc_bind_addr: None,
            webrtc_stun_server: None,
        }
    }

    pub fn client(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    pub fn router_config(mut self, router_config: RouterConfig) -> Self {
        self.router_config = Some(router_config);
        self
    }

    pub fn rate_limiter(mut self, rate_limiter: Option<Arc<TokenBucket>>) -> Self {
        self.rate_limiter = rate_limiter;
        self
    }

    pub fn tokenizer_registry(mut self, tokenizer_registry: Arc<TokenizerRegistry>) -> Self {
        self.tokenizer_registry = Some(tokenizer_registry);
        self
    }

    pub fn reasoning_parser_factory(
        mut self,
        reasoning_parser_factory: Option<ReasoningParserFactory>,
    ) -> Self {
        self.reasoning_parser_factory = reasoning_parser_factory;
        self
    }

    pub fn tool_parser_factory(mut self, tool_parser_factory: Option<ToolParserFactory>) -> Self {
        self.tool_parser_factory = tool_parser_factory;
        self
    }

    pub fn worker_registry(mut self, worker_registry: Arc<WorkerRegistry>) -> Self {
        self.worker_registry = Some(worker_registry);
        self
    }

    pub fn policy_registry(mut self, policy_registry: Arc<PolicyRegistry>) -> Self {
        self.policy_registry = Some(policy_registry);
        self
    }

    pub fn router_manager(mut self, router_manager: Option<Arc<RouterManager>>) -> Self {
        self.router_manager = router_manager;
        self
    }

    pub fn response_storage(mut self, response_storage: Arc<dyn ResponseStorage>) -> Self {
        self.response_storage = Some(response_storage);
        self
    }

    pub fn conversation_storage(
        mut self,
        conversation_storage: Arc<dyn ConversationStorage>,
    ) -> Self {
        self.conversation_storage = Some(conversation_storage);
        self
    }

    pub fn conversation_item_storage(
        mut self,
        conversation_item_storage: Arc<dyn ConversationItemStorage>,
    ) -> Self {
        self.conversation_item_storage = Some(conversation_item_storage);
        self
    }

    /// Inject conversation memory writer for long-term-memory store operations.
    pub fn conversation_memory_writer(
        mut self,
        conversation_memory_writer: Arc<dyn ConversationMemoryWriter>,
    ) -> Self {
        self.conversation_memory_writer = Some(conversation_memory_writer);
        self
    }

    pub fn background_repository(
        mut self,
        background_repository: Option<Arc<dyn BackgroundResponseRepository>>,
    ) -> Self {
        self.background_repository = background_repository;
        self
    }

    pub fn worker_monitor(mut self, worker_monitor: Option<Arc<WorkerMonitor>>) -> Self {
        self.worker_monitor = worker_monitor;
        self
    }

    pub fn worker_job_queue(mut self, worker_job_queue: Arc<OnceLock<Arc<JobQueue>>>) -> Self {
        self.worker_job_queue = Some(worker_job_queue);
        self
    }

    pub fn workflow_engines(mut self, workflow_engines: Arc<OnceLock<WorkflowEngines>>) -> Self {
        self.workflow_engines = Some(workflow_engines);
        self
    }

    pub fn mcp_orchestrator(
        mut self,
        mcp_orchestrator: Arc<OnceLock<Arc<McpOrchestrator>>>,
    ) -> Self {
        self.mcp_orchestrator = Some(mcp_orchestrator);
        self
    }

    pub fn wasm_manager(mut self, wasm_manager: Option<Arc<WasmModuleManager>>) -> Self {
        self.wasm_manager = wasm_manager;
        self
    }

    pub fn kv_event_monitor(mut self, kv_event_monitor: Option<Arc<KvEventMonitor>>) -> Self {
        self.kv_event_monitor = kv_event_monitor;
        self
    }

    pub fn webrtc_bind_addr(mut self, addr: Option<std::net::IpAddr>) -> Self {
        self.webrtc_bind_addr = addr;
        self
    }

    pub fn webrtc_stun_server(mut self, server: Option<String>) -> Self {
        self.webrtc_stun_server = server;
        self
    }

    pub fn build(self) -> Result<AppContext, AppContextBuildError> {
        let router_config = self
            .router_config
            .ok_or(AppContextBuildError::MissingField("router_config"))?;
        let configured_reasoning_parser = router_config.reasoning_parser.clone();
        let configured_tool_parser = router_config.tool_call_parser.clone();

        // Validate configured parser names against their registries at startup
        if let (Some(name), Some(factory)) =
            (&configured_reasoning_parser, &self.reasoning_parser_factory)
        {
            if !factory.registry().has_parser(name) {
                tracing::error!(
                    parser = %name,
                    available = %factory.list_parsers().join(", "),
                    "Unknown reasoning parser"
                );
                return Err(AppContextBuildError::InvalidConfig(format!(
                    "unknown reasoning parser '{name}'"
                )));
            }
        }
        if let (Some(name), Some(factory)) = (&configured_tool_parser, &self.tool_parser_factory) {
            if !factory.has_parser(name) {
                tracing::error!(
                    parser = %name,
                    available = %factory.list_parsers().join(", "),
                    "Unknown tool-call parser"
                );
                return Err(AppContextBuildError::InvalidConfig(format!(
                    "unknown tool-call parser '{name}'"
                )));
            }
        }

        validate_memory_writer_configuration(&router_config)
            .map_err(AppContextBuildError::InvalidConfig)?;

        let worker_registry = self
            .worker_registry
            .ok_or(AppContextBuildError::MissingField("worker_registry"))?;
        let worker_job_queue = self
            .worker_job_queue
            .ok_or(AppContextBuildError::MissingField("worker_job_queue"))?;

        // Create WorkerService from the already-built components
        let worker_service = Arc::new(WorkerService::new(
            worker_registry.clone(),
            worker_job_queue.clone(),
            router_config.clone(),
        ));

        Ok(AppContext {
            client: self
                .client
                .ok_or(AppContextBuildError::MissingField("client"))?,
            router_config,
            rate_limiter: self.rate_limiter,
            tokenizer_registry: self
                .tokenizer_registry
                .ok_or(AppContextBuildError::MissingField("tokenizer_registry"))?,
            multimodal_config_registry: Arc::new(MultimodalConfigRegistry::new()),
            reasoning_parser_factory: self.reasoning_parser_factory,
            tool_parser_factory: self.tool_parser_factory,
            worker_registry,
            policy_registry: self
                .policy_registry
                .ok_or(AppContextBuildError::MissingField("policy_registry"))?,
            router_manager: self.router_manager,
            response_storage: self
                .response_storage
                .ok_or(AppContextBuildError::MissingField("response_storage"))?,
            conversation_storage: self
                .conversation_storage
                .ok_or(AppContextBuildError::MissingField("conversation_storage"))?,
            conversation_item_storage: self.conversation_item_storage.ok_or(
                AppContextBuildError::MissingField("conversation_item_storage"),
            )?,
            conversation_memory_writer: self.conversation_memory_writer.ok_or(
                AppContextBuildError::MissingField("conversation_memory_writer"),
            )?,
            background_repository: self.background_repository,
            worker_monitor: self.worker_monitor,
            configured_reasoning_parser,
            configured_tool_parser,
            worker_job_queue,
            workflow_engines: self
                .workflow_engines
                .ok_or(AppContextBuildError::MissingField("workflow_engines"))?,
            mcp_orchestrator: self
                .mcp_orchestrator
                .ok_or(AppContextBuildError::MissingField("mcp_orchestrator"))?,
            skill_service: self.skill_service,
            wasm_manager: self.wasm_manager,
            worker_service,
            inflight_tracker: InFlightRequestTracker::new(),
            kv_event_monitor: self.kv_event_monitor,
            realtime_registry: Arc::new(RealtimeRegistry::new()),
            webrtc_bind_addr: self.webrtc_bind_addr,
            webrtc_stun_server: self.webrtc_stun_server,
        })
    }

    /// Initialize AppContext from config - creates ALL components
    /// This replaces ~194 lines of initialization logic from server.rs
    pub async fn from_config(
        router_config: RouterConfig,
        request_timeout_secs: u64,
        webrtc_bind_addr: Option<std::net::IpAddr>,
        webrtc_stun_server: Option<String>,
    ) -> Result<Self, String> {
        // Fail fast before storage initialization to avoid side effects
        // (e.g., migrations) for invalid memory_runtime/backend combinations.
        validate_memory_writer_configuration(&router_config)?;

        Ok(Self::new()
            .with_client(&router_config, request_timeout_secs)?
            .maybe_rate_limiter(&router_config)
            .with_tokenizer_registry()
            .with_reasoning_parser_factory()
            .with_tool_parser_factory()
            .with_worker_registry()
            .with_policy_registry(&router_config)
            .with_storage(&router_config)
            .await?
            .with_worker_monitor(&router_config)?
            .with_worker_job_queue()
            .with_workflow_engines()
            .with_mcp_orchestrator(&router_config)
            .await?
            .with_skill_service(&router_config)?
            .with_wasm_manager(&router_config)
            .with_kv_event_monitor(&router_config)
            .webrtc_bind_addr(webrtc_bind_addr)
            .webrtc_stun_server(webrtc_stun_server)
            .router_config(router_config))
    }

    /// Create HTTP client with TLS/mTLS configuration
    fn with_client(mut self, config: &RouterConfig, timeout_secs: u64) -> Result<Self, String> {
        // FIXME: Current implementation creates a single HTTP client for all workers.
        // This works well for single security domain deployments where all workers share
        // the same CA and can accept the same client certificate.
        //
        // For multi-domain deployments (e.g., different model families with different CAs),
        // this architecture needs significant refactoring:
        // 1. Move client creation into worker registration workflow (per-worker clients)
        // 2. Store client per worker in WorkerRegistry
        // 3. Update PDRouter and other routers to fetch client from worker
        // 4. Add per-worker TLS spec in WorkerConfigRequest
        //
        // Current single-domain approach is sufficient for most deployments.
        //
        // Use rustls TLS backend when TLS/mTLS is configured (client cert or CA certs provided).
        // This ensures proper PKCS#8 key format support. For plain HTTP workers, use default
        // backend to avoid unnecessary TLS initialization overhead.
        let has_tls_config = config.client_identity.is_some() || !config.ca_certificates.is_empty();

        let mut client_builder = Client::builder()
            .pool_idle_timeout(Some(Duration::from_secs(50)))
            .pool_max_idle_per_host(500)
            .timeout(Duration::from_secs(timeout_secs))
            .connect_timeout(Duration::from_secs(10))
            .tcp_nodelay(true)
            .tcp_keepalive(Some(Duration::from_secs(30)));

        // Force rustls backend when TLS is configured
        if has_tls_config {
            client_builder = client_builder.use_rustls_tls();
            debug!("Using rustls TLS backend for TLS/mTLS connections");
        }

        // Configure mTLS client identity if provided (certificates already loaded during config creation)
        if let Some(identity_pem) = &config.client_identity {
            let identity = reqwest::Identity::from_pem(identity_pem)
                .map_err(|e| format!("Failed to create client identity: {e}"))?;
            client_builder = client_builder.identity(identity);
            debug!("mTLS client authentication enabled");
        }

        // Add CA certificates for verifying worker TLS (certificates already loaded during config creation)
        for ca_cert in &config.ca_certificates {
            let cert = reqwest::Certificate::from_pem(ca_cert)
                .map_err(|e| format!("Failed to add CA certificate: {e}"))?;
            client_builder = client_builder.add_root_certificate(cert);
        }
        if !config.ca_certificates.is_empty() {
            debug!(
                "Added {} CA certificate(s) for worker verification",
                config.ca_certificates.len()
            );
        }

        let client = client_builder
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {e}"))?;

        self.client = Some(client);
        Ok(self)
    }

    /// Create rate limiter based on config
    fn maybe_rate_limiter(mut self, config: &RouterConfig) -> Self {
        self.rate_limiter = match config.max_concurrent_requests {
            n if n <= 0 => None,
            n => {
                let rate_limit_tokens = config
                    .rate_limit_tokens_per_second
                    .filter(|&t| t > 0)
                    .unwrap_or(n);
                Some(Arc::new(TokenBucket::new(
                    n as usize,
                    rate_limit_tokens as usize,
                )))
            }
        };
        self
    }

    /// Create reasoning parser factory for gRPC mode or IGW mode
    fn with_reasoning_parser_factory(mut self) -> Self {
        // Initialize reasoning parser factory
        self.reasoning_parser_factory = Some(ReasoningParserFactory::new());
        self
    }

    /// Create tool parser factory for gRPC mode or IGW mode
    fn with_tool_parser_factory(mut self) -> Self {
        // Initialize tool parser factory
        self.tool_parser_factory = Some(ToolParserFactory::new());
        self
    }

    /// Create empty tokenizer registry
    ///
    /// Tokenizers are loaded via the tokenizer_registration workflow, which is triggered:
    /// - At startup (if --tokenizer-path or --model-path is provided)
    /// - When workers connect (registers under model_id)
    /// - Via POST /v1/tokenizers API (registers under user-specified name)
    ///
    /// This unified approach ensures consistent behavior (caching, validation) across all paths.
    fn with_tokenizer_registry(mut self) -> Self {
        self.tokenizer_registry = Some(Arc::new(TokenizerRegistry::new()));
        self
    }

    /// Create worker registry
    fn with_worker_registry(mut self) -> Self {
        self.worker_registry = Some(Arc::new(WorkerRegistry::new()));
        self
    }

    /// Create policy registry
    fn with_policy_registry(mut self, config: &RouterConfig) -> Self {
        self.policy_registry = Some(Arc::new(PolicyRegistry::new(config.policy.clone())));
        self
    }

    /// Create all storage backends using the factory function
    async fn with_storage(mut self, config: &RouterConfig) -> Result<Self, String> {
        let hook: Option<Arc<dyn smg_data_connector::hooks::StorageHook>> =
            match &config.storage_hook_wasm_path {
                Some(path) => {
                    let bytes = tokio::fs::read(path)
                        .await
                        .map_err(|e| format!("failed to read WASM storage hook at {path}: {e}"))?;
                    let wasm_hook =
                        tokio::task::spawn_blocking(move || smg_wasm::WasmStorageHook::new(&bytes))
                            .await
                            .map_err(|e| format!("WASM compilation task panicked: {e}"))?
                            .map_err(|e| {
                                format!("failed to compile WASM storage hook at {path}: {e}")
                            })?;
                    debug!("loaded WASM storage hook from {path}");
                    Some(Arc::new(wasm_hook))
                }
                None => None,
            };

        let storage_config = StorageFactoryConfig {
            backend: &config.history_backend,
            oracle: config.oracle.as_ref(),
            postgres: config.postgres.as_ref(),
            redis: config.redis.as_ref(),
            hook,
        };
        let bundle = create_storage(storage_config).await?;

        self.response_storage = Some(bundle.response_storage);
        self.conversation_storage = Some(bundle.conversation_storage);
        self.conversation_item_storage = Some(bundle.conversation_item_storage);
        self.conversation_memory_writer = Some(bundle.conversation_memory_writer);
        self.background_repository = bundle.background_repository;

        Ok(self)
    }

    /// Create load monitor
    fn with_worker_monitor(mut self, config: &RouterConfig) -> Result<Self, String> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| "client must be set before load monitor".to_string())?;
        self.worker_monitor = Some(Arc::new(WorkerMonitor::new(
            self.worker_registry
                .as_ref()
                .ok_or_else(|| "worker_registry must be set before load monitor".to_string())?
                .clone(),
            self.policy_registry
                .as_ref()
                .ok_or_else(|| "policy_registry must be set before load monitor".to_string())?
                .clone(),
            client.clone(),
            config.load_monitor_interval_secs,
        )));
        Ok(self)
    }

    /// Create worker job queue OnceLock container
    fn with_worker_job_queue(mut self) -> Self {
        self.worker_job_queue = Some(Arc::new(OnceLock::new()));
        self
    }

    /// Create workflow engines OnceLock container
    fn with_workflow_engines(mut self) -> Self {
        self.workflow_engines = Some(Arc::new(OnceLock::new()));
        self
    }

    /// Create and initialize MCP orchestrator with empty config
    ///
    /// This initializes the MCP orchestrator with an empty config and default settings.
    /// MCP servers will be registered later via the InitializeMcpServers job.
    async fn with_mcp_orchestrator(
        mut self,
        _router_config: &RouterConfig,
    ) -> Result<Self, String> {
        // Create OnceLock container
        let mcp_orchestrator_lock = Arc::new(OnceLock::new());

        // Always create with empty config and defaults
        debug!("Initializing MCP orchestrator with empty config and default settings (5 min TTL, 100 max connections)");

        let empty_config = smg_mcp::McpConfig {
            servers: Vec::new(),
            pool: Default::default(),
            proxy: None,
            warmup: Vec::new(),
            inventory: Default::default(),
            policy: Default::default(),
        };

        let orchestrator = McpOrchestrator::new(empty_config)
            .await
            .map_err(|e| format!("Failed to initialize MCP orchestrator: {e}"))?;

        // Store the initialized orchestrator in the OnceLock
        mcp_orchestrator_lock
            .set(Arc::new(orchestrator))
            .map_err(|_| "Failed to set MCP orchestrator in OnceLock".to_string())?;

        self.mcp_orchestrator = Some(mcp_orchestrator_lock);
        Ok(self)
    }

    fn with_skill_service(mut self, config: &RouterConfig) -> Result<Self, String> {
        if !config.skills_enabled {
            self.skill_service = None;
            return Ok(self);
        }

        let Some(skills_config) = config.skills.as_ref() else {
            return Err(
                "Skills are enabled but no validated skills config was loaded at startup"
                    .to_string(),
            );
        };

        let blob_store =
            create_blob_store(&skills_config.blob_store, Some(&skills_config.cache))
                .map_err(|error| format!("Failed to initialize skills blob store: {error}"))?;
        let upload_limits = SkillUploadLimits::from_config(skills_config)
            .map_err(|error| format!("Invalid skills upload limits: {error}"))?;
        self.skill_service = Some(Arc::new(SkillService::in_memory_with_limits(
            blob_store,
            upload_limits,
        )));
        Ok(self)
    }

    /// Create KV event monitor for event-driven cache-aware routing.
    ///
    /// The monitor is created when the default policy is cache_aware, regardless
    /// of connection mode. The monitor itself is cheap (empty DashMaps) and stays
    /// dormant until workers are added. The UpdatePoliciesStep gates subscriptions
    /// on `cache_aware && gRPC`, so HTTP workers are never subscribed.
    fn with_kv_event_monitor(mut self, config: &RouterConfig) -> Self {
        use crate::config::types::PolicyConfig;

        let is_cache_aware = matches!(config.policy, PolicyConfig::CacheAware { .. });

        if is_cache_aware {
            let monitor = Arc::new(KvEventMonitor::new(None));
            debug!("Created KV event monitor for event-driven cache-aware routing");

            // Inject monitor into PolicyRegistry — propagates to default_policy
            // and any other existing cache-aware policies.
            if let Some(ref registry) = self.policy_registry {
                registry.set_kv_event_monitor(Some(Arc::clone(&monitor)));
            }

            self.kv_event_monitor = Some(monitor);
        }

        self
    }

    /// Create wasm manager if enabled in config
    fn with_wasm_manager(mut self, config: &RouterConfig) -> Self {
        self.wasm_manager = if config.enable_wasm {
            Some(Arc::new(WasmModuleManager::new(
                WasmRuntimeConfig::default(),
            )))
        } else {
            None
        };
        self
    }
}

impl Default for AppContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Enforce runtime-aware constraints for conversation memory writer availability.
fn validate_memory_writer_configuration(config: &RouterConfig) -> Result<(), String> {
    let backend_supports_memory_writer = backend_supports_memory_writer(&config.history_backend);

    if config.memory_runtime.enabled && !backend_supports_memory_writer {
        return Err(
            "memory_runtime.enabled is true but selected storage backend does not support conversation memory writer".to_string(),
        );
    }

    if config.memory_runtime.enabled && config.storage_hook_wasm_path.is_some() {
        return Err(
            "memory_runtime.enabled cannot be used with storage_hook_wasm_path until conversation memory writer hooks are implemented".to_string(),
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use anyhow::{anyhow, Result};
    use smg_skills::{SkillServiceMode, SkillsConfig};
    use tempfile::TempDir;

    use super::AppContextBuilder;
    use crate::config::RouterConfig;

    #[test]
    fn with_skill_service_builds_single_process_service_when_enabled() -> Result<()> {
        let blob_root = TempDir::new()?;
        let cache_root = TempDir::new()?;
        let mut config = RouterConfig {
            skills_enabled: true,
            ..RouterConfig::default()
        };
        let mut skills = SkillsConfig::default();
        skills.blob_store.path = blob_root.path().display().to_string();
        skills.cache.path = cache_root.path().display().to_string();
        config.skills = Some(skills);

        let builder = AppContextBuilder::new()
            .with_skill_service(&config)
            .map_err(anyhow::Error::msg)?;
        let service = builder
            .skill_service
            .ok_or_else(|| anyhow!("skill service should be built"))?;

        assert_eq!(service.mode(), SkillServiceMode::SingleProcess);
        Ok(())
    }
}
