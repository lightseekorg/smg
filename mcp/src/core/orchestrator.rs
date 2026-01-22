//! MCP Orchestrator - Main entry point for all MCP operations.
//!
//! `McpOrchestrator` coordinates between:
//! - Server connections (static from config, dynamic from requests)
//! - Tool inventory with qualified names and aliasing
//! - Approval manager (interactive + policy-only modes)
//! - Response transformation (MCP → OpenAI formats)
//! - Metrics and monitoring
//!
//! ## Usage
//!
//! ```ignore
//! // Initialize orchestrator
//! let orchestrator = McpOrchestrator::new(config).await?;
//!
//! // Create per-request context
//! let request_ctx = orchestrator.create_request_context(tenant_ctx);
//!
//! // Call a tool
//! let result = orchestrator.call_tool(
//!     "brave",
//!     "web_search",
//!     json!({"query": "rust programming"}),
//!     &request_ctx,
//! ).await?;
//! ```

use std::{
    borrow::Cow,
    sync::Arc,
    time::{Duration, Instant},
};

use dashmap::DashMap;
use openai_protocol::responses::ResponseOutputItem;
use rmcp::{
    model::{CallToolRequestParam, CallToolResult},
    service::RunningService,
    RoleClient,
};
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use super::{
    config::{McpConfig, McpServerConfig, McpTransport},
    handler::{HandlerRequestContext, RefreshRequest, SmgClientHandler},
    metrics::McpMetrics,
    pool::McpConnectionPool,
};
use crate::{
    approval::{
        ApprovalDecision, ApprovalManager, ApprovalMode, ApprovalOutcome, ApprovalParams,
        McpApprovalRequest,
    },
    error::{McpError, McpResult},
    inventory::{
        AliasTarget, ArgMapping, QualifiedToolName, ToolCategory, ToolEntry, ToolInventory,
    },
    tenant::TenantContext,
    transform::{ResponseFormat, ResponseTransformer},
};

/// Type alias for MCP client with handler.
type McpClientWithHandler = RunningService<RoleClient, SmgClientHandler>;

/// Server entry with client and handler.
#[derive(Clone)]
struct ServerEntry {
    client: Arc<McpClientWithHandler>,
    handler: Arc<SmgClientHandler>,
}

/// Result of a tool call.
#[derive(Debug)]
pub enum ToolCallResult {
    /// Successfully executed and transformed.
    Success(ResponseOutputItem),
    /// Pending approval from user.
    PendingApproval(McpApprovalRequest),
}

/// Main orchestrator for MCP operations.
///
/// Thread-safe and designed for sharing across async tasks.
pub struct McpOrchestrator {
    /// Static servers (from config, never evicted).
    static_servers: DashMap<String, ServerEntry>,
    /// Tool inventory with qualified names.
    tool_inventory: Arc<ToolInventory>,
    /// Approval manager for interactive and policy-only modes.
    approval_manager: Arc<ApprovalManager>,
    /// Connection pool for dynamic servers.
    connection_pool: Arc<McpConnectionPool>,
    /// Metrics and monitoring.
    metrics: Arc<McpMetrics>,
    /// Channel for refresh requests from handlers.
    refresh_tx: mpsc::Sender<RefreshRequest>,
    /// Original config for reference.
    config: McpConfig,
}

impl McpOrchestrator {
    /// Create a new orchestrator with the given configuration.
    ///
    /// Policy is built from `config.policy`. Default policy allows all tools.
    pub async fn new(config: McpConfig) -> McpResult<Self> {
        let tool_inventory = Arc::new(ToolInventory::new());
        let metrics = Arc::new(McpMetrics::new());

        // Build approval manager from config
        let audit_log = Arc::new(crate::approval::audit::AuditLog::new());
        let policy_engine = Arc::new(crate::approval::policy::PolicyEngine::from_yaml_config(
            &config.policy,
            Arc::clone(&audit_log),
        ));
        let approval_manager = Arc::new(ApprovalManager::new(policy_engine, audit_log));

        // Create connection pool with eviction callback
        let mut connection_pool =
            McpConnectionPool::with_full_config(config.pool.max_connections, config.proxy.clone());

        let inventory_clone = Arc::clone(&tool_inventory);
        connection_pool.set_eviction_callback(move |server_key: &str| {
            debug!(
                "LRU evicted dynamic server '{}' - clearing tools from inventory",
                server_key
            );
            inventory_clone.clear_server_tools(server_key);
        });

        let connection_pool = Arc::new(connection_pool);

        // Create refresh channel
        let (refresh_tx, refresh_rx) = mpsc::channel(100);

        let orchestrator = Self {
            static_servers: DashMap::new(),
            tool_inventory,
            approval_manager,
            connection_pool,
            metrics,
            refresh_tx,
            config: config.clone(),
        };

        // Connect to static servers
        for server_config in &config.servers {
            if let Err(e) = orchestrator.connect_static_server(server_config).await {
                if server_config.required {
                    return Err(e);
                }
                error!(
                    "Failed to connect to optional server '{}': {}",
                    server_config.name, e
                );
            }
        }

        // Start background refresh task
        orchestrator.spawn_refresh_handler(refresh_rx);

        info!(
            "McpOrchestrator initialized with {} static servers",
            orchestrator.static_servers.len()
        );

        Ok(orchestrator)
    }

    /// Create a simplified orchestrator for testing.
    #[cfg(test)]
    pub fn new_test() -> Self {
        use crate::approval::{audit::AuditLog, policy::PolicyEngine};

        let (refresh_tx, _) = mpsc::channel(10);
        let audit_log = Arc::new(AuditLog::new());
        let policy_engine = Arc::new(PolicyEngine::new(Arc::clone(&audit_log)));
        let approval_manager = Arc::new(ApprovalManager::new(policy_engine, audit_log));

        Self {
            static_servers: DashMap::new(),
            tool_inventory: Arc::new(ToolInventory::new()),
            approval_manager,
            connection_pool: Arc::new(McpConnectionPool::new()),
            metrics: Arc::new(McpMetrics::new()),
            refresh_tx,
            config: McpConfig::default(),
        }
    }

    // ========================================================================
    // Server Connection
    // ========================================================================

    /// Connect to a static server from config.
    ///
    /// This method:
    /// 1. Establishes a connection to the MCP server
    /// 2. Loads tools, prompts, and resources from the server
    /// 3. Applies tool configurations (aliases, response formats)
    /// 4. Registers the server as a static server
    ///
    /// Static servers are never evicted from the connection pool.
    pub async fn connect_static_server(&self, config: &McpServerConfig) -> McpResult<()> {
        info!("Connecting to static server '{}'", config.name);

        let handler = Arc::new(
            SmgClientHandler::new(
                &config.name,
                Arc::clone(&self.approval_manager),
                Arc::clone(&self.tool_inventory),
            )
            .with_refresh_channel(self.refresh_tx.clone()),
        );

        let client = self.connect_server_impl(config, (*handler).clone()).await?;
        let client = Arc::new(client);

        // Load tools from server
        self.load_server_inventory(&config.name, &client).await;

        // Apply tool configs (aliases, response formats)
        self.apply_tool_configs(config);

        // Store server entry
        self.static_servers
            .insert(config.name.clone(), ServerEntry { client, handler });

        self.metrics.record_connection_opened();
        info!("Connected to static server '{}'", config.name);
        Ok(())
    }

    /// Apply tool configurations from server config (aliases, response formats, arg mappings).
    fn apply_tool_configs(&self, config: &McpServerConfig) {
        let Some(tools) = &config.tools else {
            return;
        };

        for (tool_name, tool_config) in tools {
            // Check if the tool exists
            if !self
                .tool_inventory
                .has_tool_qualified(&config.name, tool_name)
            {
                warn!(
                    "Tool config for '{}:{}' but tool not found on server",
                    config.name, tool_name
                );
                continue;
            }

            // Get the existing entry to update or create alias
            let response_format: ResponseFormat = tool_config.response_format.clone().into();

            // If there's an alias, register it
            if let Some(alias_name) = &tool_config.alias {
                let arg_mapping = tool_config.arg_mapping.as_ref().map(|cfg| {
                    let mut mapping = ArgMapping::new();
                    for (from, to) in &cfg.renames {
                        mapping = mapping.with_rename(from, to);
                    }
                    for (key, value) in &cfg.defaults {
                        mapping = mapping.with_default(key, value.clone());
                    }
                    mapping
                });

                if let Err(e) = self.register_alias(
                    alias_name,
                    &config.name,
                    tool_name,
                    arg_mapping,
                    response_format.clone(),
                ) {
                    warn!(
                        "Failed to register alias '{}' for '{}:{}': {}",
                        alias_name, config.name, tool_name, e
                    );
                } else {
                    info!(
                        "Registered alias '{}' → '{}:{}' with format {:?}",
                        alias_name, config.name, tool_name, response_format
                    );
                }
            } else if response_format != ResponseFormat::Passthrough {
                // No alias, but has custom response format - update the entry directly
                if let Some(mut entry) = self.tool_inventory.get_entry(&config.name, tool_name) {
                    entry.response_format = response_format.clone();
                    self.tool_inventory.insert_entry(entry);
                    info!(
                        "Set response format {:?} for '{}:{}'",
                        response_format, config.name, tool_name
                    );
                }
            }
        }
    }

    /// Internal server connection logic.
    async fn connect_server_impl(
        &self,
        config: &McpServerConfig,
        handler: SmgClientHandler,
    ) -> McpResult<McpClientWithHandler> {
        use rmcp::{
            transport::{
                sse_client::SseClientConfig,
                streamable_http_client::StreamableHttpClientTransportConfig, ConfigureCommandExt,
                SseClientTransport, StreamableHttpClientTransport, TokioChildProcess,
            },
            ServiceExt,
        };

        match &config.transport {
            McpTransport::Stdio {
                command,
                args,
                envs,
            } => {
                let transport = TokioChildProcess::new(
                    tokio::process::Command::new(command).configure(|cmd| {
                        cmd.args(args)
                            .envs(envs.iter())
                            .stderr(std::process::Stdio::inherit());
                    }),
                )
                .map_err(|e| McpError::Transport(format!("create stdio transport: {}", e)))?;

                handler.serve(transport).await.map_err(|e| {
                    McpError::ConnectionFailed(format!("initialize stdio client: {}", e))
                })
            }

            McpTransport::Sse { url, token } => {
                let proxy_config =
                    super::proxy::resolve_proxy_config(config, self.config.proxy.as_ref());

                let mut builder =
                    reqwest::Client::builder().connect_timeout(Duration::from_secs(10));

                if let Some(proxy_cfg) = proxy_config {
                    builder = super::proxy::apply_proxy_to_builder(builder, proxy_cfg)?;
                }

                if let Some(tok) = token {
                    builder = builder.default_headers({
                        let mut headers = reqwest::header::HeaderMap::new();
                        headers.insert(
                            reqwest::header::AUTHORIZATION,
                            format!("Bearer {}", tok)
                                .parse()
                                .map_err(|e| McpError::Transport(format!("auth token: {}", e)))?,
                        );
                        headers
                    });
                }

                let http_client = builder
                    .build()
                    .map_err(|e| McpError::Transport(format!("build HTTP client: {}", e)))?;

                let sse_config = SseClientConfig {
                    sse_endpoint: url.clone().into(),
                    ..Default::default()
                };

                let transport = SseClientTransport::start_with_client(http_client, sse_config)
                    .await
                    .map_err(|e| McpError::Transport(format!("create SSE transport: {}", e)))?;

                handler.serve(transport).await.map_err(|e| {
                    McpError::ConnectionFailed(format!("initialize SSE client: {}", e))
                })
            }

            McpTransport::Streamable { url, token } => {
                let transport = if let Some(tok) = token {
                    let mut cfg = StreamableHttpClientTransportConfig::with_uri(url.as_str());
                    cfg.auth_header = Some(tok.to_string());
                    StreamableHttpClientTransport::from_config(cfg)
                } else {
                    StreamableHttpClientTransport::from_uri(url.as_str())
                };

                handler.serve(transport).await.map_err(|e| {
                    McpError::ConnectionFailed(format!("initialize streamable client: {}", e))
                })
            }
        }
    }

    /// Load tools, prompts, and resources from a server into the inventory.
    async fn load_server_inventory(&self, server_key: &str, client: &Arc<McpClientWithHandler>) {
        // Load tools
        match client.peer().list_all_tools().await {
            Ok(tools) => {
                info!("Discovered {} tools from '{}'", tools.len(), server_key);
                for tool in tools {
                    let entry = ToolEntry::from_server_tool(server_key, tool)
                        .with_category(ToolCategory::Static);
                    self.tool_inventory.insert_entry(entry);
                }
            }
            Err(e) => warn!("Failed to list tools from '{}': {}", server_key, e),
        }

        // Load prompts
        match client.peer().list_all_prompts().await {
            Ok(prompts) => {
                info!("Discovered {} prompts from '{}'", prompts.len(), server_key);
                for prompt in prompts {
                    self.tool_inventory.insert_prompt(
                        prompt.name.clone(),
                        server_key.to_string(),
                        prompt,
                    );
                }
            }
            Err(e) => debug!("No prompts from '{}': {}", server_key, e),
        }

        // Load resources
        match client.peer().list_all_resources().await {
            Ok(resources) => {
                info!(
                    "Discovered {} resources from '{}'",
                    resources.len(),
                    server_key
                );
                for resource in resources {
                    self.tool_inventory.insert_resource(
                        resource.uri.clone(),
                        server_key.to_string(),
                        resource.raw,
                    );
                }
            }
            Err(e) => debug!("No resources from '{}': {}", server_key, e),
        }
    }

    /// Spawn background handler for inventory refresh requests.
    fn spawn_refresh_handler(&self, mut rx: mpsc::Receiver<RefreshRequest>) {
        let tool_inventory = Arc::clone(&self.tool_inventory);
        let static_servers = self.static_servers.clone();

        tokio::spawn(async move {
            while let Some(request) = rx.recv().await {
                debug!("Processing refresh request for '{}'", request.server_key);

                if let Some(entry) = static_servers.get(&request.server_key) {
                    // Clear existing tools for this server
                    tool_inventory.clear_server_tools(&request.server_key);

                    // Reload tools
                    match entry.client.peer().list_all_tools().await {
                        Ok(tools) => {
                            for tool in tools {
                                let entry = ToolEntry::from_server_tool(&request.server_key, tool)
                                    .with_category(ToolCategory::Static);
                                tool_inventory.insert_entry(entry);
                            }
                            info!(
                                "Refreshed inventory for '{}': {} tools",
                                request.server_key,
                                tool_inventory.counts().0
                            );
                        }
                        Err(e) => {
                            warn!(
                                "Failed to refresh tools for '{}': {}",
                                request.server_key, e
                            );
                        }
                    }
                }
            }
        });
    }

    // ========================================================================
    // Tool Execution
    // ========================================================================

    /// Call a tool with approval checking and response transformation.
    ///
    /// This is the main entry point for tool execution.
    pub async fn call_tool(
        &self,
        server_key: &str,
        tool_name: &str,
        arguments: Value,
        request_ctx: &McpRequestContext<'_>,
    ) -> McpResult<ToolCallResult> {
        let qualified = QualifiedToolName::new(server_key, tool_name);

        // Get tool entry
        let entry = self
            .tool_inventory
            .get_entry(server_key, tool_name)
            .ok_or_else(|| McpError::ToolNotFound(qualified.to_string()))?;

        // Record metrics start
        self.metrics.record_call_start(&qualified);
        let start_time = Instant::now();

        // Execute with approval flow
        let result = self
            .execute_tool_with_approval(&entry, arguments, request_ctx)
            .await;

        // Record metrics end
        let duration_ms = start_time.elapsed().as_millis() as u64;
        self.metrics
            .record_call_end(&qualified, result.is_ok(), duration_ms);

        result
    }

    /// Call a tool by name within a set of allowed servers.
    ///
    /// This is the recommended entry point for callers who have:
    /// - A tool name (from LLM response)
    /// - A list of allowed server keys (from request configuration)
    ///
    /// The method handles:
    /// - Looking up which server has the tool
    /// - Detecting collisions (same tool name on multiple allowed servers)
    /// - Returning proper errors for not-found or collision cases
    ///
    /// # Arguments
    /// * `tool_name` - The tool name to call
    /// * `arguments` - Tool arguments as JSON
    /// * `allowed_servers` - Server keys to search within (filters scope)
    /// * `request_ctx` - Request context for approval and tenant isolation
    ///
    /// # Errors
    /// * `ToolNotFound` - Tool doesn't exist on any allowed server
    /// * `ToolCollision` - Tool exists on multiple allowed servers
    pub async fn call_tool_by_name(
        &self,
        tool_name: &str,
        arguments: Value,
        allowed_servers: &[String],
        request_ctx: &McpRequestContext<'_>,
    ) -> McpResult<ToolCallResult> {
        // Find all matching tools in allowed servers
        let matching: Vec<_> = self
            .tool_inventory
            .list_tools()
            .into_iter()
            .filter(|(name, server_key, _)| {
                name == tool_name && allowed_servers.contains(server_key)
            })
            .collect();

        match matching.len() {
            0 => Err(McpError::ToolNotFound(tool_name.to_string())),
            1 => {
                let (_, server_key, _) = &matching[0];
                self.call_tool(server_key, tool_name, arguments, request_ctx)
                    .await
            }
            _ => {
                // Multiple servers have this tool - ambiguous
                let servers: Vec<String> = matching.iter().map(|(_, s, _)| s.to_string()).collect();
                warn!(
                    tool_name = tool_name,
                    servers = ?servers,
                    "Tool name collision detected"
                );
                Err(McpError::ToolCollision {
                    tool_name: tool_name.to_string(),
                    servers,
                })
            }
        }
    }

    /// Execute tool with approval checking.
    async fn execute_tool_with_approval(
        &self,
        entry: &ToolEntry,
        arguments: Value,
        request_ctx: &McpRequestContext<'_>,
    ) -> McpResult<ToolCallResult> {
        // Check if approval is needed
        let approval_params = ApprovalParams {
            request_id: &request_ctx.request_id,
            server_key: entry.server_key(),
            elicitation_id: &format!("tool-{}", entry.tool_name()),
            tool_name: entry.tool_name(),
            hints: &entry.annotations,
            message: &format!("Allow execution of '{}'?", entry.tool_name()),
            tenant_ctx: &request_ctx.tenant_ctx,
        };

        let outcome = self
            .approval_manager
            .handle_approval(request_ctx.approval_mode, approval_params)
            .await?;

        match outcome {
            ApprovalOutcome::Decided(decision) => {
                if !decision.is_allowed() {
                    self.metrics.record_approval_denied();
                    return Err(McpError::ToolDenied(entry.tool_name().to_string()));
                }
                self.metrics.record_approval_granted();

                // Execute the tool
                let result = self.execute_tool_impl(entry, arguments.clone()).await?;

                // Transform response
                let output = self.transform_result(
                    &result,
                    &entry.response_format,
                    &request_ctx.request_id,
                    entry.server_key(),
                    entry.tool_name(),
                    &arguments.to_string(),
                );

                Ok(ToolCallResult::Success(output))
            }
            ApprovalOutcome::Pending {
                approval_request,
                rx,
                ..
            } => {
                self.metrics.record_approval_requested();

                // In interactive mode, return pending approval
                // The caller should send this to the client and wait for response
                if request_ctx.approval_mode == ApprovalMode::Interactive {
                    return Ok(ToolCallResult::PendingApproval(approval_request));
                }

                // In policy-only mode (shouldn't happen), wait for decision
                match rx.await {
                    Ok(ApprovalDecision::Approved) => {
                        self.metrics.record_approval_granted();
                        let result = self.execute_tool_impl(entry, arguments.clone()).await?;
                        let output = self.transform_result(
                            &result,
                            &entry.response_format,
                            &request_ctx.request_id,
                            entry.server_key(),
                            entry.tool_name(),
                            &arguments.to_string(),
                        );
                        Ok(ToolCallResult::Success(output))
                    }
                    Ok(ApprovalDecision::Denied { reason }) => {
                        self.metrics.record_approval_denied();
                        Err(McpError::ToolDenied(reason))
                    }
                    Err(_) => Err(McpError::ToolDenied("Channel closed".to_string())),
                }
            }
        }
    }

    /// Execute tool without approval (internal use).
    async fn execute_tool_impl(
        &self,
        entry: &ToolEntry,
        mut arguments: Value,
    ) -> McpResult<CallToolResult> {
        // Resolve alias if needed
        let (target_server, target_tool) = if let Some(alias) = &entry.alias_target {
            // Apply argument mapping
            if let Some(mapping) = &alias.arg_mapping {
                arguments = self.apply_arg_mapping(arguments, mapping);
            }
            (
                alias.target.server_key().to_string(),
                alias.target.tool_name().to_string(),
            )
        } else {
            (
                entry.server_key().to_string(),
                entry.tool_name().to_string(),
            )
        };

        // Coerce argument types based on tool schema
        // LLMs often return numbers as strings (e.g., "5" instead of 5)
        Self::coerce_arg_types(&mut arguments, &entry.tool.input_schema);

        // Build request
        let args_map = if let Value::Object(map) = arguments {
            Some(map)
        } else {
            None
        };

        let request = CallToolRequestParam {
            name: Cow::Owned(target_tool),
            arguments: args_map,
        };

        // Execute on server
        self.execute_on_server(&target_server, request).await
    }

    /// Coerce argument types based on tool schema.
    ///
    /// LLMs often output numbers as strings, so we convert them based on the schema.
    fn coerce_arg_types(args: &mut Value, schema: &serde_json::Map<String, Value>) {
        let Some(props) = schema.get("properties").and_then(|p| p.as_object()) else {
            return;
        };
        let Some(args_map) = args.as_object_mut() else {
            return;
        };

        for (key, val) in args_map.iter_mut() {
            let should_be_number = props
                .get(key)
                .and_then(|s| s.get("type"))
                .and_then(|t| t.as_str())
                .is_some_and(|t| matches!(t, "number" | "integer"));

            if should_be_number {
                if let Some(s) = val.as_str() {
                    if let Ok(num) = s.parse::<f64>() {
                        *val = serde_json::json!(num);
                    }
                }
            }
        }
    }

    /// Apply argument mapping for aliased tools.
    fn apply_arg_mapping(&self, mut args: Value, mapping: &ArgMapping) -> Value {
        if let Value::Object(ref mut map) = args {
            // Apply renames
            for (from, to) in &mapping.renames {
                if let Some(value) = map.remove(from) {
                    map.insert(to.clone(), value);
                }
            }

            // Apply defaults
            for (key, default_value) in &mapping.defaults {
                map.entry(key.clone())
                    .or_insert_with(|| default_value.clone());
            }
        }
        args
    }

    /// Transform MCP result to OpenAI format.
    fn transform_result(
        &self,
        result: &CallToolResult,
        format: &ResponseFormat,
        tool_call_id: &str,
        server_label: &str,
        tool_name: &str,
        arguments: &str,
    ) -> ResponseOutputItem {
        // Convert CallToolResult content to JSON for transformation
        let result_json = self.call_result_to_json(result);

        ResponseTransformer::transform(
            &result_json,
            format,
            tool_call_id,
            server_label,
            tool_name,
            arguments,
        )
    }

    /// Convert CallToolResult to JSON value.
    fn call_result_to_json(&self, result: &CallToolResult) -> Value {
        // Serialize the CallToolResult content to JSON
        // The content is a Vec of annotated content items
        serde_json::to_value(&result.content).unwrap_or_else(|e| {
            warn!(
                "Failed to serialize CallToolResult to JSON: {}. Falling back to empty object.",
                e
            );
            Value::Object(serde_json::Map::new())
        })
    }

    /// Execute a tool call on a server.
    async fn execute_on_server(
        &self,
        server_key: &str,
        request: CallToolRequestParam,
    ) -> McpResult<CallToolResult> {
        // Check static servers first
        if let Some(entry) = self.static_servers.get(server_key) {
            return entry
                .client
                .call_tool(request)
                .await
                .map_err(|e| McpError::ToolExecution(format!("MCP call failed: {}", e)));
        }

        // Check connection pool (pool uses different handler type)
        if let Some(client) = self.connection_pool.get(server_key) {
            return client
                .call_tool(request)
                .await
                .map_err(|e| McpError::ToolExecution(format!("MCP call failed: {}", e)));
        }

        Err(McpError::ServerNotFound(server_key.to_string()))
    }

    // ========================================================================
    // Alias Registration
    // ========================================================================

    /// Register a tool alias (e.g., `web_search` → `brave:brave_web_search`).
    pub fn register_alias(
        &self,
        alias_name: &str,
        target_server: &str,
        target_tool: &str,
        arg_mapping: Option<ArgMapping>,
        response_format: ResponseFormat,
    ) -> McpResult<()> {
        // Verify target exists
        let target_entry = self
            .tool_inventory
            .get_entry(target_server, target_tool)
            .ok_or_else(|| McpError::ToolNotFound(format!("{}:{}", target_server, target_tool)))?;

        // Create alias entry
        let alias_target = AliasTarget {
            target: QualifiedToolName::new(target_server, target_tool),
            arg_mapping,
        };

        let alias_entry = ToolEntry::new(
            QualifiedToolName::new("alias", alias_name),
            target_entry.tool.clone(),
        )
        .with_alias(alias_target)
        .with_response_format(response_format);

        self.tool_inventory.insert_entry(alias_entry);

        // Also register in the aliases index
        self.tool_inventory.register_alias(
            alias_name.to_string(),
            QualifiedToolName::new(target_server, target_tool),
        );

        info!(
            "Registered alias '{}' → '{}:{}'",
            alias_name, target_server, target_tool
        );
        Ok(())
    }

    // ========================================================================
    // Request Context
    // ========================================================================

    /// Create a per-request context for tool execution.
    pub fn create_request_context<'a>(
        &'a self,
        request_id: impl Into<String>,
        tenant_ctx: TenantContext,
        approval_mode: ApprovalMode,
    ) -> McpRequestContext<'a> {
        McpRequestContext::new(self, request_id.into(), tenant_ctx, approval_mode)
    }

    /// Set request context on all static server handlers.
    pub fn set_handler_contexts(&self, ctx: &HandlerRequestContext) {
        for entry in self.static_servers.iter() {
            entry.handler.set_request_context(ctx.clone());
        }
    }

    /// Clear request context from all static server handlers.
    pub fn clear_handler_contexts(&self) {
        for entry in self.static_servers.iter() {
            entry.handler.clear_request_context();
        }
    }

    // ========================================================================
    // Dynamic Server Connection
    // ========================================================================

    /// Generate a unique key for a server configuration.
    ///
    /// The key is based on the transport URL, making it suitable for connection pooling.
    pub fn server_key(config: &McpServerConfig) -> String {
        match &config.transport {
            McpTransport::Streamable { url, .. } => url.clone(),
            McpTransport::Sse { url, .. } => url.clone(),
            McpTransport::Stdio { command, args, .. } => {
                format!("{}:{}", command, args.join(" "))
            }
        }
    }

    /// Connect to a dynamic server and add it to the connection pool.
    ///
    /// This is used for per-request MCP servers specified in tool configurations.
    /// Returns the server key that can be used to reference the connection.
    pub async fn connect_dynamic_server(&self, config: McpServerConfig) -> McpResult<String> {
        use rmcp::{
            transport::{
                sse_client::SseClientConfig,
                streamable_http_client::StreamableHttpClientTransportConfig, SseClientTransport,
                StreamableHttpClientTransport,
            },
            ServiceExt,
        };

        let server_key = Self::server_key(&config);

        // Check if already connected
        if self.connection_pool.get(&server_key).is_some() {
            return Ok(server_key);
        }

        // Connect via the pool
        let inventory_clone = Arc::clone(&self.tool_inventory);
        let global_proxy = self.config.proxy.clone();

        let client = self
            .connection_pool
            .get_or_create(&server_key, config.clone(), |cfg, _proxy| async move {
                match &cfg.transport {
                    McpTransport::Streamable { url, token } => {
                        let transport = if let Some(tok) = token {
                            let mut cfg_http =
                                StreamableHttpClientTransportConfig::with_uri(url.as_str());
                            cfg_http.auth_header = Some(tok.to_string());
                            StreamableHttpClientTransport::from_config(cfg_http)
                        } else {
                            StreamableHttpClientTransport::from_uri(url.as_str())
                        };

                        ().serve(transport)
                            .await
                            .map_err(|e| McpError::ConnectionFailed(format!("streamable: {}", e)))
                    }
                    McpTransport::Sse { url, token } => {
                        let proxy_config =
                            super::proxy::resolve_proxy_config(&cfg, global_proxy.as_ref());

                        let mut builder =
                            reqwest::Client::builder().connect_timeout(Duration::from_secs(10));

                        if let Some(proxy_cfg) = proxy_config {
                            builder = super::proxy::apply_proxy_to_builder(builder, proxy_cfg)?;
                        }

                        if let Some(tok) = token {
                            builder = builder.default_headers({
                                let mut headers = reqwest::header::HeaderMap::new();
                                headers.insert(
                                    reqwest::header::AUTHORIZATION,
                                    format!("Bearer {}", tok).parse().map_err(|e| {
                                        McpError::Transport(format!("auth token: {}", e))
                                    })?,
                                );
                                headers
                            });
                        }

                        let http_client = builder.build().map_err(|e| {
                            McpError::Transport(format!("build HTTP client: {}", e))
                        })?;

                        let sse_config = SseClientConfig {
                            sse_endpoint: url.clone().into(),
                            ..Default::default()
                        };

                        let transport =
                            SseClientTransport::start_with_client(http_client, sse_config)
                                .await
                                .map_err(|e| {
                                    McpError::Transport(format!("create SSE transport: {}", e))
                                })?;

                        ().serve(transport)
                            .await
                            .map_err(|e| McpError::ConnectionFailed(format!("SSE: {}", e)))
                    }
                    McpTransport::Stdio { .. } => Err(McpError::Transport(
                        "Stdio not supported for dynamic connections".to_string(),
                    )),
                }
            })
            .await?;

        // Load tools from the server
        // Use server_key (URL) as the tool's server identifier so it matches
        // what ensure_request_mcp_client adds to server_keys for filtering
        match client.peer().list_all_tools().await {
            Ok(tools) => {
                info!(
                    "Discovered {} tools from dynamic server '{}'",
                    tools.len(),
                    server_key
                );
                for tool in tools {
                    let entry = ToolEntry::from_server_tool(&server_key, tool)
                        .with_category(ToolCategory::Dynamic);
                    inventory_clone.insert_entry(entry);
                }
            }
            Err(e) => warn!("Failed to list tools from '{}': {}", server_key, e),
        }

        self.metrics.record_connection_opened();
        Ok(server_key)
    }

    // ========================================================================
    // Queries
    // ========================================================================

    /// List all tools visible to a tenant.
    pub fn list_tools(&self, _tenant_ctx: Option<&TenantContext>) -> Vec<ToolEntry> {
        self.tool_inventory
            .list_tools()
            .into_iter()
            .filter_map(|(tool_name, server_key, _)| {
                self.tool_inventory.get_entry(&server_key, &tool_name)
            })
            .collect()
    }

    /// List tools for specific servers.
    pub fn list_tools_for_servers(&self, server_keys: &[String]) -> Vec<ToolEntry> {
        self.tool_inventory
            .list_tools()
            .into_iter()
            .filter_map(|(tool_name, server_key, _)| {
                if server_keys.contains(&server_key) {
                    self.tool_inventory.get_entry(&server_key, &tool_name)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get a tool by qualified name.
    pub fn get_tool(&self, server_key: &str, tool_name: &str) -> Option<ToolEntry> {
        self.tool_inventory.get_entry(server_key, tool_name)
    }

    /// Check if a tool exists.
    pub fn has_tool(&self, server_key: &str, tool_name: &str) -> bool {
        self.tool_inventory
            .has_tool_qualified(server_key, tool_name)
    }

    /// List all connected servers.
    pub fn list_servers(&self) -> Vec<String> {
        let mut servers: Vec<_> = self
            .static_servers
            .iter()
            .map(|e| e.key().clone())
            .collect();
        servers.extend(self.connection_pool.list_server_keys());
        servers
    }

    /// Get the tool inventory.
    pub fn tool_inventory(&self) -> Arc<ToolInventory> {
        Arc::clone(&self.tool_inventory)
    }

    /// Get the approval manager.
    pub fn approval_manager(&self) -> Arc<ApprovalManager> {
        Arc::clone(&self.approval_manager)
    }

    /// Get metrics.
    pub fn metrics(&self) -> Arc<McpMetrics> {
        Arc::clone(&self.metrics)
    }

    // ========================================================================
    // Interactive Mode API (Issue #103)
    // ========================================================================

    /// Resolve a pending approval request.
    ///
    /// Called when the client responds to an approval request in interactive mode.
    /// This matches the OpenAI Responses API `mcp_approval_response` format.
    pub async fn resolve_approval(
        &self,
        request_id: &str,
        server_key: &str,
        elicitation_id: &str,
        approved: bool,
        reason: Option<String>,
        tenant_ctx: &TenantContext,
    ) -> McpResult<()> {
        self.approval_manager
            .resolve(
                request_id,
                server_key,
                elicitation_id,
                approved,
                reason,
                tenant_ctx,
            )
            .await
    }

    /// Get the count of pending approvals for a request.
    pub fn pending_approval_count(&self) -> usize {
        self.approval_manager.pending_count()
    }

    /// Determine the approval mode based on API type.
    ///
    /// | API                      | Mode         |
    /// |--------------------------|--------------|
    /// | OpenAI Responses API     | Interactive  |
    /// | OpenAI Chat Completions  | PolicyOnly   |
    /// | Anthropic Messages API   | PolicyOnly   |
    /// | Batch processing         | PolicyOnly   |
    pub fn determine_approval_mode(supports_mcp_approval: bool) -> ApprovalMode {
        if supports_mcp_approval {
            ApprovalMode::Interactive
        } else {
            ApprovalMode::PolicyOnly
        }
    }

    /// Call a tool and continue execution after approval (for continuing paused requests).
    ///
    /// This is called after the user approves a tool execution in interactive mode.
    /// The approval should already be resolved via `resolve_approval()`.
    pub async fn continue_tool_execution(
        &self,
        server_key: &str,
        tool_name: &str,
        arguments: Value,
        request_ctx: &McpRequestContext<'_>,
    ) -> McpResult<ToolCallResult> {
        // Get tool entry
        let entry = self
            .tool_inventory
            .get_entry(server_key, tool_name)
            .ok_or_else(|| McpError::ToolNotFound(format!("{}:{}", server_key, tool_name)))?;

        // Execute directly (approval already handled)
        let result = self.execute_tool_impl(&entry, arguments.clone()).await?;

        // Transform response
        let output = self.transform_result(
            &result,
            &entry.response_format,
            &request_ctx.request_id,
            entry.server_key(),
            entry.tool_name(),
            &arguments.to_string(),
        );

        Ok(ToolCallResult::Success(output))
    }

    // ========================================================================
    // Lifecycle
    // ========================================================================

    /// Shutdown the orchestrator gracefully.
    pub async fn shutdown(&self) {
        info!("Shutting down McpOrchestrator");

        // Close static server connections
        for _entry in self.static_servers.iter() {
            self.metrics.record_connection_closed();
        }
        self.static_servers.clear();

        // Clear connection pool
        self.connection_pool.clear();

        // Clear inventory
        self.tool_inventory.clear_all();

        info!("McpOrchestrator shutdown complete");
    }
}

/// Per-request context for MCP operations.
///
/// Holds request-specific state and provides access to the orchestrator
/// for tool execution with proper tenant isolation.
pub struct McpRequestContext<'a> {
    orchestrator: &'a McpOrchestrator,
    pub request_id: String,
    pub tenant_ctx: TenantContext,
    pub approval_mode: ApprovalMode,
    /// Dynamic tools added for this request only.
    dynamic_tools: DashMap<QualifiedToolName, ToolEntry>,
    /// Dynamic server clients for this request.
    dynamic_clients: DashMap<String, Arc<McpClientWithHandler>>,
}

impl<'a> McpRequestContext<'a> {
    fn new(
        orchestrator: &'a McpOrchestrator,
        request_id: String,
        tenant_ctx: TenantContext,
        approval_mode: ApprovalMode,
    ) -> Self {
        Self {
            orchestrator,
            request_id,
            tenant_ctx,
            approval_mode,
            dynamic_tools: DashMap::new(),
            dynamic_clients: DashMap::new(),
        }
    }

    /// Get the handler request context for setting on handlers.
    pub fn handler_context(&self) -> HandlerRequestContext {
        HandlerRequestContext::new(
            &self.request_id,
            self.approval_mode,
            self.tenant_ctx.clone(),
        )
    }

    /// Add a dynamic server for this request.
    pub async fn add_dynamic_server(&self, config: &McpServerConfig) -> McpResult<()> {
        let handler = SmgClientHandler::new(
            &config.name,
            Arc::clone(&self.orchestrator.approval_manager),
            Arc::clone(&self.orchestrator.tool_inventory),
        );

        let client = self
            .orchestrator
            .connect_server_impl(config, handler)
            .await?;
        let client = Arc::new(client);

        // Load tools
        match client.peer().list_all_tools().await {
            Ok(tools) => {
                for tool in tools {
                    let entry = ToolEntry::from_server_tool(&config.name, tool)
                        .with_category(ToolCategory::Dynamic);
                    self.dynamic_tools
                        .insert(entry.qualified_name.clone(), entry);
                }
            }
            Err(e) => warn!("Failed to list tools from '{}': {}", config.name, e),
        }

        self.dynamic_clients.insert(config.name.clone(), client);
        Ok(())
    }

    /// Call a tool in this request context.
    pub async fn call_tool(
        &self,
        server_key: &str,
        tool_name: &str,
        arguments: Value,
    ) -> McpResult<ToolCallResult> {
        // Check dynamic tools first
        let qualified = QualifiedToolName::new(server_key, tool_name);
        if let Some(entry) = self.dynamic_tools.get(&qualified) {
            return self.execute_dynamic_tool(&entry, arguments).await;
        }

        // Fall back to orchestrator
        self.orchestrator
            .call_tool(server_key, tool_name, arguments, self)
            .await
    }

    /// Execute a dynamic tool.
    async fn execute_dynamic_tool(
        &self,
        entry: &ToolEntry,
        arguments: Value,
    ) -> McpResult<ToolCallResult> {
        let client = self
            .dynamic_clients
            .get(entry.server_key())
            .ok_or_else(|| McpError::ServerNotFound(entry.server_key().to_string()))?;

        let args_map = if let Value::Object(map) = arguments.clone() {
            Some(map)
        } else {
            None
        };

        let request = CallToolRequestParam {
            name: Cow::Owned(entry.tool_name().to_string()),
            arguments: args_map,
        };

        let result = client
            .call_tool(request)
            .await
            .map_err(|e| McpError::ToolExecution(format!("MCP call failed: {}", e)))?;

        let output = self.orchestrator.transform_result(
            &result,
            &entry.response_format,
            &self.request_id,
            entry.server_key(),
            entry.tool_name(),
            &arguments.to_string(),
        );

        Ok(ToolCallResult::Success(output))
    }

    /// List all tools visible in this request context.
    pub fn list_tools(&self) -> Vec<ToolEntry> {
        let mut tools = self.orchestrator.list_tools(Some(&self.tenant_ctx));

        // Add dynamic tools
        for entry in self.dynamic_tools.iter() {
            tools.push(entry.value().clone());
        }

        tools
    }

    /// Get server keys for dynamic clients.
    pub fn dynamic_server_keys(&self) -> Vec<String> {
        self.dynamic_clients
            .iter()
            .map(|e| e.key().clone())
            .collect()
    }
}

impl<'a> Drop for McpRequestContext<'a> {
    fn drop(&mut self) {
        // Cleanup dynamic clients
        if !self.dynamic_clients.is_empty() {
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                // Collect keys first, then remove to get exclusive ownership of Arc
                let keys: Vec<_> = self
                    .dynamic_clients
                    .iter()
                    .map(|e| e.key().clone())
                    .collect();
                for key in keys {
                    if let Some((_, client)) = self.dynamic_clients.remove(&key) {
                        if let Some(client) = Arc::into_inner(client) {
                            handle.spawn(async move {
                                if let Err(e) = client.cancel().await {
                                    warn!("Error closing dynamic client: {}", e);
                                }
                            });
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_tool(name: &str) -> crate::core::config::Tool {
        use std::sync::Arc;

        crate::core::config::Tool {
            name: Cow::Owned(name.to_string()),
            title: None,
            description: Some(Cow::Owned(format!("Test tool: {}", name))),
            input_schema: Arc::new(serde_json::Map::new()),
            output_schema: None,
            annotations: None,
            icons: None,
        }
    }

    #[test]
    fn test_orchestrator_creation() {
        let orchestrator = McpOrchestrator::new_test();
        assert!(orchestrator.list_servers().is_empty());
    }

    #[test]
    fn test_request_context_creation() {
        let orchestrator = McpOrchestrator::new_test();
        let ctx = orchestrator.create_request_context(
            "req-1",
            TenantContext::new("tenant-1"),
            ApprovalMode::PolicyOnly,
        );

        assert_eq!(ctx.request_id, "req-1");
        assert_eq!(ctx.tenant_ctx.tenant_id.as_str(), "tenant-1");
    }

    #[test]
    fn test_handler_context() {
        let orchestrator = McpOrchestrator::new_test();
        let ctx = orchestrator.create_request_context(
            "req-1",
            TenantContext::new("tenant-1"),
            ApprovalMode::Interactive,
        );

        let handler_ctx = ctx.handler_context();
        assert_eq!(handler_ctx.request_id, "req-1");
        assert_eq!(handler_ctx.approval_mode, ApprovalMode::Interactive);
    }

    #[test]
    fn test_tool_inventory_access() {
        let orchestrator = McpOrchestrator::new_test();

        // Insert a test tool
        let tool = create_test_tool("test_tool");
        let entry = ToolEntry::from_server_tool("test_server", tool);
        orchestrator.tool_inventory.insert_entry(entry);

        assert!(orchestrator.has_tool("test_server", "test_tool"));
        assert!(!orchestrator.has_tool("other_server", "test_tool"));
    }

    #[test]
    fn test_alias_registration() {
        let orchestrator = McpOrchestrator::new_test();

        // Insert target tool
        let tool = create_test_tool("brave_web_search");
        let entry = ToolEntry::from_server_tool("brave", tool);
        orchestrator.tool_inventory.insert_entry(entry);

        // Register alias
        let result = orchestrator.register_alias(
            "web_search",
            "brave",
            "brave_web_search",
            None,
            ResponseFormat::WebSearchCall,
        );

        assert!(result.is_ok());
        assert!(orchestrator
            .tool_inventory
            .resolve_alias("web_search")
            .is_some());
    }

    #[test]
    fn test_alias_registration_missing_target() {
        let orchestrator = McpOrchestrator::new_test();

        let result = orchestrator.register_alias(
            "web_search",
            "missing_server",
            "missing_tool",
            None,
            ResponseFormat::Passthrough,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_arg_mapping() {
        let orchestrator = McpOrchestrator::new_test();

        let mapping = ArgMapping::new()
            .with_rename("query", "search_query")
            .with_default("limit", serde_json::json!(10));

        let args = serde_json::json!({
            "query": "rust programming"
        });

        let result = orchestrator.apply_arg_mapping(args, &mapping);

        let obj = result.as_object().unwrap();
        assert!(obj.contains_key("search_query"));
        assert!(!obj.contains_key("query"));
        assert_eq!(obj.get("limit"), Some(&serde_json::json!(10)));
    }

    #[test]
    fn test_metrics_access() {
        let orchestrator = McpOrchestrator::new_test();
        let metrics = orchestrator.metrics();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_calls, 0);
    }

    #[test]
    fn test_determine_approval_mode() {
        // OpenAI Responses API supports MCP approval
        assert_eq!(
            McpOrchestrator::determine_approval_mode(true),
            ApprovalMode::Interactive
        );

        // Other APIs don't support MCP approval
        assert_eq!(
            McpOrchestrator::determine_approval_mode(false),
            ApprovalMode::PolicyOnly
        );
    }

    #[test]
    fn test_pending_approval_count() {
        let orchestrator = McpOrchestrator::new_test();
        assert_eq!(orchestrator.pending_approval_count(), 0);
    }

    #[test]
    fn test_call_tool_by_name_finds_unique_tool() {
        let orchestrator = McpOrchestrator::new_test();

        // Insert a tool on server1
        let tool = create_test_tool("unique_tool");
        let entry = ToolEntry::from_server_tool("server1", tool);
        orchestrator.tool_inventory.insert_entry(entry);

        // Check that the tool is found in inventory
        let tools = orchestrator.tool_inventory.list_tools();
        let matching: Vec<_> = tools
            .into_iter()
            .filter(|(name, server_key, _)| name == "unique_tool" && server_key == "server1")
            .collect();

        assert_eq!(matching.len(), 1);
        assert_eq!(matching[0].1, "server1");
    }

    #[test]
    fn test_call_tool_by_name_collision_detection() {
        let orchestrator = McpOrchestrator::new_test();

        // Insert same tool name on two different servers
        let tool1 = create_test_tool("shared_tool");
        let entry1 = ToolEntry::from_server_tool("server1", tool1);
        orchestrator.tool_inventory.insert_entry(entry1);

        let tool2 = create_test_tool("shared_tool");
        let entry2 = ToolEntry::from_server_tool("server2", tool2);
        orchestrator.tool_inventory.insert_entry(entry2);

        // Check collision: both servers allowed
        let tools = orchestrator.tool_inventory.list_tools();
        let allowed_servers = ["server1", "server2"];
        let matching: Vec<_> = tools
            .into_iter()
            .filter(|(name, server_key, _)| {
                name == "shared_tool" && allowed_servers.contains(&server_key.as_str())
            })
            .collect();

        // Should find 2 matches (collision)
        assert_eq!(matching.len(), 2);
    }

    #[test]
    fn test_call_tool_by_name_no_collision_with_single_server() {
        let orchestrator = McpOrchestrator::new_test();

        // Insert same tool name on two different servers
        let tool1 = create_test_tool("shared_tool");
        let entry1 = ToolEntry::from_server_tool("server1", tool1);
        orchestrator.tool_inventory.insert_entry(entry1);

        let tool2 = create_test_tool("shared_tool");
        let entry2 = ToolEntry::from_server_tool("server2", tool2);
        orchestrator.tool_inventory.insert_entry(entry2);

        // Check no collision: only one server allowed
        let tools = orchestrator.tool_inventory.list_tools();
        let allowed_servers = ["server1"];
        let matching: Vec<_> = tools
            .into_iter()
            .filter(|(name, server_key, _)| {
                name == "shared_tool" && allowed_servers.contains(&server_key.as_str())
            })
            .collect();

        // Should find only 1 match (no collision)
        assert_eq!(matching.len(), 1);
        assert_eq!(matching[0].1, "server1");
    }

    #[test]
    fn test_call_tool_by_name_tool_not_found() {
        let orchestrator = McpOrchestrator::new_test();

        // Insert a tool
        let tool = create_test_tool("existing_tool");
        let entry = ToolEntry::from_server_tool("server1", tool);
        orchestrator.tool_inventory.insert_entry(entry);

        // Search for non-existent tool
        let tools = orchestrator.tool_inventory.list_tools();
        let allowed_servers = ["server1"];
        let matching: Vec<_> = tools
            .into_iter()
            .filter(|(name, server_key, _)| {
                name == "nonexistent_tool" && allowed_servers.contains(&server_key.as_str())
            })
            .collect();

        // Should find 0 matches
        assert_eq!(matching.len(), 0);
    }
}
