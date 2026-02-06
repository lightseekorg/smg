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
//!     "brave",           // server_key
//!     "web_search",      // tool_name
//!     json!({"query": "rust programming"}),
//!     "brave",           // server_label (user-facing)
//!     &request_ctx,
//! ).await?;
//! ```

use std::{
    borrow::Cow,
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
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
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use super::{
    config::{BuiltinToolType, McpConfig, McpProxyConfig, McpServerConfig, McpTransport},
    handler::{HandlerRequestContext, RefreshRequest, SmgClientHandler},
    metrics::McpMetrics,
    pool::{McpConnectionPool, PoolKey},
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

/// Build request headers from token and custom headers.
fn build_request_headers(
    token: &Option<String>,
    custom_headers: &HashMap<String, String>,
) -> McpResult<reqwest::header::HeaderMap> {
    let mut headers = reqwest::header::HeaderMap::new();

    if let Some(tok) = token {
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", tok)
                .parse()
                .map_err(|e| McpError::Transport(format!("auth token: {}", e)))?,
        );
    }

    for (key, value) in custom_headers {
        headers.insert(
            reqwest::header::HeaderName::from_bytes(key.as_bytes())
                .map_err(|e| McpError::Transport(format!("header name: {}", e)))?,
            value
                .parse()
                .map_err(|e| McpError::Transport(format!("header value: {}", e)))?,
        );
    }

    Ok(headers)
}

/// Build HTTP client with default headers.
fn build_http_client(
    proxy_config: Option<&McpProxyConfig>,
    token: &Option<String>,
    custom_headers: &HashMap<String, String>,
) -> McpResult<reqwest::Client> {
    let mut builder = reqwest::Client::builder().connect_timeout(Duration::from_secs(10));

    if let Some(proxy_cfg) = proxy_config {
        builder = super::proxy::apply_proxy_to_builder(builder, proxy_cfg)?;
    }

    let req_headers = build_request_headers(token, custom_headers)?;
    if !req_headers.is_empty() {
        builder = builder.default_headers(req_headers);
    }

    builder
        .build()
        .map_err(|e| McpError::Transport(format!("build HTTP client: {}", e)))
}

/// Type alias for MCP client with handler.
type McpClientWithHandler = RunningService<RoleClient, SmgClientHandler>;

/// Server entry with client, handler, and config.
#[derive(Clone)]
struct ServerEntry {
    client: Arc<McpClientWithHandler>,
    handler: Arc<SmgClientHandler>,
    config: McpServerConfig,
}

/// Result of a tool call.
#[derive(Debug)]
pub enum ToolCallResult {
    /// Successfully executed and transformed.
    Success(ResponseOutputItem),
    /// Pending approval from user.
    PendingApproval(McpApprovalRequest),
}

/// Internal result type for approval-checked execution.
///
/// Used to avoid code duplication between `execute_tool_with_approval` and
/// `execute_tool_with_approval_raw`. Contains either the raw result or
/// the pending approval request.
enum ApprovalExecutionResult {
    /// Tool executed successfully, contains raw result.
    Success(CallToolResult),
    /// Pending approval from user (interactive mode only).
    PendingApproval(McpApprovalRequest),
}

impl ToolCallResult {
    /// Get the transformed output item directly without serialization.
    ///
    /// Returns `Some(item)` for successful tool calls, `None` for pending approvals.
    /// This avoids the serialize/deserialize roundtrip when the caller needs
    /// the item as a Value or ResponseOutputItem.
    pub fn into_item(self) -> Option<ResponseOutputItem> {
        match self {
            ToolCallResult::Success(item) => Some(item),
            ToolCallResult::PendingApproval(_) => None,
        }
    }

    /// Convert the result to a serialized output tuple.
    ///
    /// Returns `(output_str, is_error, error_message)` suitable for
    /// recording in conversation history or emitting as events.
    ///
    /// This centralizes the result serialization logic that was previously
    /// duplicated across all routers.
    pub fn into_serialized(self) -> (String, bool, Option<String>) {
        match self {
            ToolCallResult::Success(item) => match serde_json::to_string(&item) {
                Ok(s) => (s, false, None),
                Err(e) => {
                    let err = format!("Failed to serialize tool result: {}", e);
                    (
                        serde_json::json!({"error": &err}).to_string(),
                        true,
                        Some(err),
                    )
                }
            },
            ToolCallResult::PendingApproval(_) => {
                let err = "Tool requires approval (not supported in this context)".to_string();
                (
                    serde_json::json!({"error": &err}).to_string(),
                    true,
                    Some(err),
                )
            }
        }
    }
}

// ============================================================================
// Batch Tool Execution Types
// ============================================================================

/// Input for batch tool execution.
///
/// This is a simplified input format that allows routers to batch
/// multiple tool calls without worrying about the underlying execution details.
#[derive(Debug, Clone)]
pub struct ToolExecutionInput {
    /// Unique identifier for this tool call (from LLM response).
    pub call_id: String,
    /// Name of the tool to execute.
    pub tool_name: String,
    /// Tool arguments as JSON.
    pub arguments: Value,
}

/// Output from batch tool execution.
///
/// Contains all information needed by routers to build responses,
/// record state, and emit events. The MCP crate handles:
/// - Tool lookup and execution
/// - Result serialization and transformation
/// - Error handling
#[derive(Debug, Clone)]
pub struct ToolExecutionOutput {
    /// The call_id from the input (for matching).
    pub call_id: String,
    /// Name of the tool that was executed.
    pub tool_name: String,
    /// Server key where the tool was executed (internal identifier, may be URL).
    pub server_key: String,
    /// User-facing server label for API responses.
    pub server_label: String,
    /// Original arguments JSON string (for conversation history).
    pub arguments_str: String,
    /// Raw tool output as Value (for conversation history and transformation).
    pub output: Value,
    /// Whether the execution resulted in an error.
    pub is_error: bool,
    /// Error message if `is_error` is true.
    pub error_message: Option<String>,
    /// Response format for transforming output to API-specific types.
    pub response_format: ResponseFormat,
    /// Execution duration.
    pub duration: Duration,
}

impl ToolExecutionOutput {
    /// Get the transformed ResponseOutputItem.
    ///
    /// Transforms the raw output to the appropriate ResponseOutputItem type
    /// based on the tool's configured response format (WebSearchCall,
    /// CodeInterpreterCall, FileSearchCall, or Passthrough/McpCall).
    ///
    /// Uses `server_label` (user-facing) for the output, not `server_key` (internal).
    pub fn to_response_item(&self) -> ResponseOutputItem {
        ResponseTransformer::transform(
            &self.output,
            &self.response_format,
            &self.call_id,
            &self.server_label,
            &self.tool_name,
            &self.arguments_str,
        )
    }
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
    active_executions: Arc<AtomicUsize>,
    shutdown_token: CancellationToken,
    /// Original config for reference.
    config: McpConfig,
}

impl McpOrchestrator {
    /// Create a new orchestrator with the given configuration.
    ///
    /// Policy is built from `config.policy`. Default policy allows all tools.
    pub async fn new(config: McpConfig) -> McpResult<Self> {
        // Validate configuration (server pairs, no duplicate builtin_types)
        config
            .validate()
            .map_err(|e| McpError::Config(e.to_string()))?;

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
        connection_pool.set_eviction_callback(move |key: &PoolKey| {
            debug!(
                "LRU evicted dynamic server '{}' (tenant: {:?}) - clearing tools from inventory",
                key.url, key.tenant_id
            );
            // Tools are registered by URL, so clear by URL
            inventory_clone.clear_server_tools(&key.url);
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
            active_executions: Arc::new(AtomicUsize::new(0)),
            shutdown_token: CancellationToken::new(),
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
            active_executions: Arc::new(AtomicUsize::new(0)),
            shutdown_token: CancellationToken::new(),
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
        // Skip if already connected (handles duplicate registrations from workflow)
        if self.static_servers.contains_key(&config.name) {
            debug!(
                "Server '{}' already connected, skipping duplicate registration",
                config.name
            );
            return Ok(());
        }

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

        // Apply builtin response format if server has builtin_type configured
        self.apply_builtin_response_format(config);

        // Store server entry with config for builtin lookups
        self.static_servers.insert(
            config.name.clone(),
            ServerEntry {
                client,
                handler,
                config: config.clone(),
            },
        );

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

    /// Apply builtin response format to the builtin_tool_name if not explicitly overridden.
    ///
    /// When a server is configured with `builtin_type` and `builtin_tool_name`, the
    /// corresponding tool should use the response format associated with the builtin type
    /// (e.g., WebSearchPreview -> WebSearchCall) unless explicitly overridden in the tools config.
    fn apply_builtin_response_format(&self, config: &McpServerConfig) {
        let Some(builtin_type) = &config.builtin_type else {
            return;
        };
        let Some(tool_name) = &config.builtin_tool_name else {
            return;
        };

        let has_explicit_config = config
            .tools
            .as_ref()
            .is_some_and(|tools| tools.contains_key(tool_name));

        if has_explicit_config {
            debug!(
                server = %config.name,
                tool = %tool_name,
                "Builtin tool has explicit config, skipping auto-apply of response_format"
            );
            return;
        }

        let response_format: ResponseFormat = builtin_type.response_format().into();

        let updated = self
            .tool_inventory
            .update_entry(&config.name, tool_name, |entry| {
                if entry.response_format != response_format {
                    info!(
                        server = %config.name,
                        tool = %tool_name,
                        builtin_type = %builtin_type,
                        format = ?response_format,
                        "Applied builtin response format"
                    );
                    entry.response_format = response_format.clone();
                }
            });

        if !updated {
            warn!(
                server = %config.name,
                tool = %tool_name,
                builtin_type = %builtin_type,
                "Builtin tool not found on server"
            );
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

            McpTransport::Sse {
                url,
                token,
                headers: custom_headers,
            } => {
                let proxy_config =
                    super::proxy::resolve_proxy_config(config, self.config.proxy.as_ref());
                let http_client = build_http_client(proxy_config, token, custom_headers)?;

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

            McpTransport::Streamable {
                url,
                token,
                headers: custom_headers,
            } => {
                let proxy_config =
                    super::proxy::resolve_proxy_config(config, self.config.proxy.as_ref());
                let http_client = build_http_client(proxy_config, token, custom_headers)?;
                let cfg = StreamableHttpClientTransportConfig::with_uri(url.as_str());

                let transport = StreamableHttpClientTransport::with_client(http_client, cfg);

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
        let token = self.shutdown_token.clone(); //
        let tool_inventory = Arc::clone(&self.tool_inventory);
        let static_servers = self.static_servers.clone();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    // Stop the loop if the shutdown token is triggered
                    _ = token.cancelled() => { //
                        debug!("Refresh handler shutting down");
                        break;
                    }
                    // Process refresh requests as they arrive
                    Some(request) = rx.recv() => {
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

                    else => break,
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
    ///
    /// # Arguments
    /// * `server_key` - Internal server identifier (may be URL for dynamic servers)
    /// * `tool_name` - The tool name to execute
    /// * `arguments` - Tool arguments as JSON
    /// * `server_label` - User-facing label for API responses
    /// * `request_ctx` - Request context for approval
    pub async fn call_tool(
        &self,
        server_key: &str,
        tool_name: &str,
        arguments: Value,
        server_label: &str,
        request_ctx: &McpRequestContext<'_>,
    ) -> McpResult<ToolCallResult> {
        self.active_executions.fetch_add(1, Ordering::SeqCst);
        let _guard = scopeguard::guard(Arc::clone(&self.active_executions), |count| {
            count.fetch_sub(1, Ordering::SeqCst);
        });
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
            .execute_tool_with_approval(&entry, arguments, server_label, request_ctx)
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
        server_label: &str,
        request_ctx: &McpRequestContext<'_>,
    ) -> McpResult<ToolCallResult> {
        // For small server lists (typical case: 1-5), linear scan is faster than HashSet
        let is_allowed =
            |server_key: &str| -> bool { allowed_servers.iter().any(|s| s == server_key) };

        // Find all matching tools in allowed servers
        let matching: Vec<_> = self
            .tool_inventory
            .list_tools()
            .into_iter()
            .filter(|(name, server_key, _)| name == tool_name && is_allowed(server_key))
            .collect();

        match matching.len() {
            0 => Err(McpError::ToolNotFound(tool_name.to_string())),
            1 => {
                let (_, server_key, _) = &matching[0];
                self.call_tool(server_key, tool_name, arguments, server_label, request_ctx)
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

    /// Find a tool entry by name within allowed servers.
    ///
    /// Use this when you need tool metadata (especially `response_format`)
    /// without executing the tool. Useful for streaming paths that execute
    /// tools individually but need format information for output transformation.
    ///
    /// Returns `None` if tool not found or collision detected.
    pub fn find_tool_by_name(
        &self,
        tool_name: &str,
        allowed_servers: &[String],
    ) -> Option<ToolEntry> {
        // For small server lists (typical case: 1-5), linear scan is faster than HashSet
        let is_allowed =
            |server_key: &str| -> bool { allowed_servers.iter().any(|s| s == server_key) };

        let matching: Vec<_> = self
            .tool_inventory
            .list_tools()
            .into_iter()
            .filter(|(name, server_key, _)| name == tool_name && is_allowed(server_key))
            .collect();

        // Only return if exactly one match (no collision)
        if matching.len() == 1 {
            let (_, server_key, _) = &matching[0];
            self.tool_inventory.get_entry(server_key, tool_name)
        } else {
            None
        }
    }

    /// Find the MCP server configured to handle a built-in tool type.
    ///
    /// When a request includes built-in tools like `{"type": "web_search_preview"}`,
    /// routers can use this method to find which MCP server should handle it.
    ///
    /// # Arguments
    /// * `builtin_type` - The built-in tool type to look up
    ///
    /// # Returns
    /// If a server is configured for this built-in type, returns:
    /// - `server_key` - Internal identifier for the server (used for `call_tool`)
    /// - `tool_name` - The MCP tool to call on that server
    /// - `response_format` - The format to use for response transformation
    ///
    /// Returns `None` if no server is configured for this built-in type.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Check if web_search_preview is configured
    /// if let Some((server_key, tool_name, format)) =
    ///     orchestrator.find_builtin_server(BuiltinToolType::WebSearchPreview)
    /// {
    ///     // Route to MCP server
    ///     let result = orchestrator.call_tool(
    ///         &server_key,
    ///         &tool_name,
    ///         arguments,
    ///         "web-search",  // user-facing label
    ///         &request_ctx,
    ///     ).await?;
    /// } else {
    ///     // No MCP configured - handle differently
    /// }
    /// ```
    pub fn find_builtin_server(
        &self,
        builtin_type: BuiltinToolType,
    ) -> Option<(String, String, ResponseFormat)> {
        // Helper to extract builtin info from a server config
        let extract_builtin = |server_config: &McpServerConfig| {
            if let (Some(cfg_type), Some(tool_name)) = (
                &server_config.builtin_type,
                &server_config.builtin_tool_name,
            ) {
                if *cfg_type == builtin_type {
                    // Determine response format from tool config or use builtin default
                    let response_format = server_config
                        .tools
                        .as_ref()
                        .and_then(|tools| tools.get(tool_name))
                        .map(|tc| tc.response_format.clone().into())
                        .unwrap_or_else(|| builtin_type.response_format().into());

                    return Some((
                        server_config.name.clone(),
                        tool_name.clone(),
                        response_format,
                    ));
                }
            }
            None
        };

        // First, search connected static servers (dynamically registered via connect_static_server).
        // This handles servers registered after orchestrator initialization.
        for entry in self.static_servers.iter() {
            if let Some(result) = extract_builtin(&entry.config) {
                return Some(result);
            }
        }

        // Fallback to initial config servers (handles edge cases where servers
        // from the initial config haven't connected yet, and supports tests).
        for server_config in &self.config.servers {
            if let Some(result) = extract_builtin(server_config) {
                return Some(result);
            }
        }

        None
    }

    /// Execute multiple tools with unified error handling.
    ///
    /// This is the recommended entry point for batch tool execution. It:
    /// - Executes each tool via `call_tool_by_name`
    /// - Handles result serialization uniformly
    /// - Returns fully-processed results ready for router use
    ///
    /// Routers should convert their tool calls to `ToolExecutionInput`,
    /// call this method, and use `ToolExecutionOutput::to_response_item()`
    /// to get correctly-typed output items.
    /// Metrics recording is left to callers (routers) since they have access
    /// to the model_gateway metrics infrastructure.
    ///
    /// # Arguments
    /// * `inputs` - Tool calls to execute
    /// * `allowed_servers` - Server keys to search within
    /// * `mcp_servers` - MCP servers for this request (label, server_key)
    /// * `request_ctx` - Request context for approval and tenant isolation
    ///
    /// # Returns
    /// Vector of outputs in the same order as inputs. Errors are returned
    /// as outputs with `is_error: true` rather than failing the entire batch.
    pub async fn execute_tools(
        &self,
        inputs: Vec<ToolExecutionInput>,
        allowed_servers: &[String],
        mcp_servers: &[(String, String)],
        request_ctx: &McpRequestContext<'_>,
    ) -> Vec<ToolExecutionOutput> {
        let fallback_label = mcp_servers
            .first()
            .map(|(label, _)| label.as_str())
            .unwrap_or("mcp");
        let server_label_map: HashMap<_, _> = mcp_servers
            .iter()
            .map(|(label, key)| (key.as_str(), label.as_str()))
            .collect();
        let mut results = Vec::with_capacity(inputs.len());

        for input in inputs {
            let start = Instant::now();

            // Preserve arguments string for conversation history
            let arguments_str = input.arguments.to_string();

            // Look up tool entry to get server_key, response_format, and entry itself
            let tool_entry = self.find_tool_by_name(&input.tool_name, allowed_servers);

            let (server_key, response_format, entry) = match tool_entry {
                Some(entry) => (
                    entry.server_key().to_string(),
                    entry.response_format.clone(),
                    Some(entry),
                ),
                None => {
                    // Tool not found - will error below
                    ("unknown".to_string(), ResponseFormat::Passthrough, None)
                }
            };

            let server_label = server_label_map
                .get(server_key.as_str())
                .copied()
                .unwrap_or(fallback_label);

            let (output, is_error, error_message) = {
                let handle_error = |err_msg: String| {
                    warn!("Tool execution failed: {}", err_msg);
                    let error_json = serde_json::json!({ "error": &err_msg });
                    (error_json, true, Some(err_msg))
                };

                match entry {
                    None => handle_error(format!("Tool '{}' not found", input.tool_name)),
                    Some(entry) => {
                        // Execute with approval, getting raw result
                        match self
                            .execute_tool_with_approval_raw(&entry, input.arguments, request_ctx)
                            .await
                        {
                            Ok(Some(raw_result)) => {
                                // Convert raw CallToolResult to JSON
                                let output = self.call_result_to_json(&raw_result);
                                (output, raw_result.is_error.unwrap_or(false), None)
                            }
                            Ok(None) => handle_error(
                                "Tool requires approval (not supported in this context)"
                                    .to_string(),
                            ),
                            Err(e) => handle_error(format!("Tool call failed: {}", e)),
                        }
                    }
                }
            };

            let duration = start.elapsed();

            results.push(ToolExecutionOutput {
                call_id: input.call_id,
                tool_name: input.tool_name,
                server_key,
                server_label: server_label.to_string(),
                arguments_str,
                output,
                is_error,
                error_message,
                response_format,
                duration,
            });
        }

        results
    }

    /// Execute tool with approval checking.
    ///
    /// Returns a transformed `ToolCallResult` ready for API responses.
    /// For raw results without transformation, use `execute_tool_with_approval_raw`.
    ///
    /// # Arguments
    /// * `entry` - Tool entry to execute
    /// * `arguments` - Tool arguments
    /// * `server_label` - User-facing server label for API responses
    /// * `request_ctx` - Request context for approval
    async fn execute_tool_with_approval(
        &self,
        entry: &ToolEntry,
        arguments: Value,
        server_label: &str,
        request_ctx: &McpRequestContext<'_>,
    ) -> McpResult<ToolCallResult> {
        // Delegate to raw implementation and transform result
        match self
            .execute_tool_with_approval_raw_internal(entry, arguments.clone(), request_ctx)
            .await?
        {
            ApprovalExecutionResult::Success(result) => {
                let output = self.transform_result(
                    &result,
                    &entry.response_format,
                    &request_ctx.request_id,
                    server_label,
                    entry.tool_name(),
                    &arguments.to_string(),
                );
                Ok(ToolCallResult::Success(output))
            }
            ApprovalExecutionResult::PendingApproval(approval_request) => {
                Ok(ToolCallResult::PendingApproval(approval_request))
            }
        }
    }

    /// Execute tool with approval, returning raw CallToolResult.
    ///
    /// This is used by `execute_tools` to get the raw output before transformation.
    /// Returns `Ok(Some(result))` on success, `Ok(None)` if pending approval,
    /// or `Err` on failure.
    async fn execute_tool_with_approval_raw(
        &self,
        entry: &ToolEntry,
        arguments: Value,
        request_ctx: &McpRequestContext<'_>,
    ) -> McpResult<Option<CallToolResult>> {
        match self
            .execute_tool_with_approval_raw_internal(entry, arguments, request_ctx)
            .await?
        {
            ApprovalExecutionResult::Success(result) => Ok(Some(result)),
            ApprovalExecutionResult::PendingApproval(_) => Ok(None),
        }
    }

    /// Internal implementation of approval-checked tool execution.
    ///
    /// Returns either the raw result or the pending approval request.
    /// Both public methods delegate to this to avoid code duplication.
    async fn execute_tool_with_approval_raw_internal(
        &self,
        entry: &ToolEntry,
        arguments: Value,
        request_ctx: &McpRequestContext<'_>,
    ) -> McpResult<ApprovalExecutionResult> {
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
                let result = self.execute_tool_impl(entry, arguments).await?;
                Ok(ApprovalExecutionResult::Success(result))
            }
            ApprovalOutcome::Pending {
                approval_request,
                rx,
                ..
            } => {
                self.metrics.record_approval_requested();

                // In interactive mode, return pending approval
                if request_ctx.approval_mode == ApprovalMode::Interactive {
                    return Ok(ApprovalExecutionResult::PendingApproval(approval_request));
                }

                // In policy-only mode, wait for decision
                match rx.await {
                    Ok(ApprovalDecision::Approved) => {
                        self.metrics.record_approval_granted();
                        let result = self.execute_tool_impl(entry, arguments).await?;
                        Ok(ApprovalExecutionResult::Success(result))
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

        // Check connection pool by URL (pool uses different handler type)
        // Note: This uses URL-based lookup for backward compatibility.
        // For full auth/tenant isolation, use PoolKey-based lookup.
        if let Some(client) = self.connection_pool.get_by_url(server_key) {
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
    /// Returns the server key (URL) that can be used to reference the connection.
    ///
    /// The connection is keyed by (url, auth_hash, tenant_id) to ensure:
    /// - Different auth tokens get different connections
    /// - Different tenants are isolated
    pub async fn connect_dynamic_server(&self, config: McpServerConfig) -> McpResult<String> {
        self.connect_dynamic_server_with_tenant(config, None).await
    }

    /// Connect to a dynamic server with tenant isolation.
    ///
    /// Like `connect_dynamic_server` but includes tenant_id in the pool key
    /// for proper tenant isolation.
    pub async fn connect_dynamic_server_with_tenant(
        &self,
        config: McpServerConfig,
        tenant_id: Option<String>,
    ) -> McpResult<String> {
        use rmcp::{
            transport::{
                sse_client::SseClientConfig,
                streamable_http_client::StreamableHttpClientTransportConfig, SseClientTransport,
                StreamableHttpClientTransport,
            },
            ServiceExt,
        };

        let pool_key = PoolKey::from_config(&config, tenant_id);

        // Check if already connected with same auth/tenant
        if self.connection_pool.contains(&pool_key) {
            return Ok(pool_key.url.clone());
        }

        // Extract server_key from pool_key to avoid double URL extraction
        let server_key = pool_key.url.clone();

        // Connect via the pool
        let inventory_clone = Arc::clone(&self.tool_inventory);
        let global_proxy = self.config.proxy.clone();

        let client = self
            .connection_pool
            .get_or_create(pool_key, config.clone(), |cfg, _proxy| async move {
                match &cfg.transport {
                    McpTransport::Streamable {
                        url,
                        token,
                        headers: custom_headers,
                    } => {
                        let proxy_config =
                            super::proxy::resolve_proxy_config(&cfg, global_proxy.as_ref());
                        let http_client = build_http_client(proxy_config, token, custom_headers)?;
                        let cfg_http = StreamableHttpClientTransportConfig::with_uri(url.as_str());

                        let transport =
                            StreamableHttpClientTransport::with_client(http_client, cfg_http);

                        ().serve(transport)
                            .await
                            .map_err(|e| McpError::ConnectionFailed(format!("streamable: {}", e)))
                    }
                    McpTransport::Sse {
                        url,
                        token,
                        headers: custom_headers,
                    } => {
                        let proxy_config =
                            super::proxy::resolve_proxy_config(&cfg, global_proxy.as_ref());
                        let http_client = build_http_client(proxy_config, token, custom_headers)?;

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
        // For small server lists (typical case: 1-5), linear scan is faster than HashSet
        let is_allowed = |server_key: &str| -> bool { server_keys.iter().any(|s| s == server_key) };

        self.tool_inventory
            .list_tools()
            .into_iter()
            .filter_map(|(tool_name, server_key, _)| {
                if is_allowed(&server_key) {
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
        // Extract URLs from pool keys (may have duplicates for different auth/tenants)
        servers.extend(self.connection_pool.list_keys().into_iter().map(|k| k.url));
        servers.sort();
        servers.dedup();
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
        tracing::info!("Starting graceful shutdown of McpOrchestrator");

        self.shutdown_token.cancel();

        // Wait for active executions (30s timeout)
        let start = Instant::now();
        let timeout = Duration::from_secs(30);
        while self.active_executions.load(Ordering::SeqCst) > 0 {
            if start.elapsed() >= timeout {
                tracing::warn!(
                    "Shutdown timeout reached; {} executions still active",
                    self.active_executions.load(Ordering::SeqCst)
                );
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Cancel pending approvals
        self.approval_manager.cancel_all_pending();

        for _ in self.static_servers.iter() {
            self.metrics.record_connection_closed();
        }
        self.static_servers.clear();
        self.connection_pool.clear();

        self.tool_inventory.clear_all();

        tracing::info!("McpOrchestrator shutdown complete");
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
    ///
    /// # Arguments
    /// * `server_key` - Internal server identifier
    /// * `tool_name` - Tool name to execute
    /// * `arguments` - Tool arguments as JSON
    /// * `server_label` - User-facing label for API responses
    pub async fn call_tool(
        &self,
        server_key: &str,
        tool_name: &str,
        arguments: Value,
        server_label: &str,
    ) -> McpResult<ToolCallResult> {
        self.orchestrator
            .active_executions
            .fetch_add(1, Ordering::SeqCst);
        let _guard = scopeguard::guard(Arc::clone(&self.orchestrator.active_executions), |count| {
            count.fetch_sub(1, Ordering::SeqCst);
        });
        // Check dynamic tools first
        let qualified = QualifiedToolName::new(server_key, tool_name);
        if let Some(entry) = self.dynamic_tools.get(&qualified) {
            return self
                .execute_dynamic_tool(&entry, arguments, server_label)
                .await;
        }

        // Fall back to orchestrator
        self.orchestrator
            .call_tool(server_key, tool_name, arguments, server_label, self)
            .await
    }

    /// Execute a dynamic tool.
    async fn execute_dynamic_tool(
        &self,
        entry: &ToolEntry,
        arguments: Value,
        server_label: &str,
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
            server_label,
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
    use std::time::Duration;

    use tokio::time::timeout;

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
    #[tokio::test]
    async fn test_orchestrator_graceful_shutdown_flow() {
        let orchestrator = McpOrchestrator::new_test();
        let tenant = TenantContext::new("test-tenant");
        let ctx = orchestrator.create_request_context(
            "req-shutdown-test",
            tenant,
            ApprovalMode::Interactive,
        );

        let hints = crate::annotations::ToolAnnotations::new();
        let params = ApprovalParams {
            request_id: "req-shutdown-test",
            server_key: "test-server",
            elicitation_id: "elicit-1",
            tool_name: "test_tool",
            hints: &hints,
            message: "Allow tool execution?",
            tenant_ctx: &ctx.tenant_ctx,
        };

        // Use the approval manager directly to simulate a real pending state
        let outcome = orchestrator
            .approval_manager()
            .handle_approval(ApprovalMode::Interactive, params)
            .await
            .expect("Failed to create pending approval");

        let rx = match outcome {
            ApprovalOutcome::Pending { rx, .. } => rx,
            _ => panic!("Expected the outcome to be Pending for interactive mode"),
        };

        let shutdown_result = timeout(Duration::from_secs(5), orchestrator.shutdown()).await;
        assert!(shutdown_result.is_ok(), "Orchestrator shutdown timed out");

        let decision = rx
            .await
            .expect("The approval channel should have received a response before closing");
        match decision {
            ApprovalDecision::Denied { reason } => {
                assert_eq!(
                    reason, "System shutdown",
                    "Denial reason should be 'System shutdown'"
                );
            }
            _ => panic!("Expected the tool call to be Denied, but it was Approved"),
        }

        assert_eq!(
            orchestrator.active_executions.load(Ordering::SeqCst),
            0,
            "Active execution counter must be zero"
        );
        assert_eq!(
            orchestrator.pending_approval_count(),
            0,
            "Pending approvals were not cleared"
        );
        assert!(
            orchestrator.list_servers().is_empty(),
            "Server registry was not cleared"
        );
        assert_eq!(
            orchestrator.tool_inventory().counts().0,
            0,
            "Tool inventory was not cleared"
        );
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

    #[test]
    fn test_find_builtin_server_web_search() {
        use std::collections::HashMap;

        use crate::approval::{audit::AuditLog, policy::PolicyEngine};

        // Create config with a server configured for web_search_preview
        let config = McpConfig {
            servers: vec![McpServerConfig {
                name: "brave".to_string(),
                transport: McpTransport::Sse {
                    url: "https://mcp.brave.com/sse".to_string(),
                    token: None,
                    headers: HashMap::new(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("brave_web_search".to_string()),
            }],
            ..Default::default()
        };

        let (refresh_tx, _) = mpsc::channel(10);
        let audit_log = Arc::new(AuditLog::new());
        let policy_engine = Arc::new(PolicyEngine::new(Arc::clone(&audit_log)));
        let approval_manager = Arc::new(ApprovalManager::new(policy_engine, audit_log));

        let orchestrator = McpOrchestrator {
            static_servers: DashMap::new(),
            tool_inventory: Arc::new(ToolInventory::new()),
            approval_manager,
            connection_pool: Arc::new(McpConnectionPool::new()),
            metrics: Arc::new(McpMetrics::new()),
            refresh_tx,
            active_executions: Arc::new(AtomicUsize::new(0)),
            shutdown_token: CancellationToken::new(),
            config,
        };

        // Should find the brave server for web_search_preview
        let result = orchestrator.find_builtin_server(BuiltinToolType::WebSearchPreview);
        assert!(result.is_some());

        let (server_key, tool_name, response_format) = result.unwrap();
        assert_eq!(server_key, "brave");
        assert_eq!(tool_name, "brave_web_search");
        assert_eq!(response_format, ResponseFormat::WebSearchCall);

        // Should NOT find a server for code_interpreter
        let result = orchestrator.find_builtin_server(BuiltinToolType::CodeInterpreter);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_builtin_server_with_custom_response_format() {
        use std::collections::HashMap;

        use crate::{
            approval::{audit::AuditLog, policy::PolicyEngine},
            core::config::{ResponseFormatConfig, ToolConfig},
        };

        // Create config with custom response_format for the tool
        let mut tools = HashMap::new();
        tools.insert(
            "my_search".to_string(),
            ToolConfig {
                alias: None,
                response_format: ResponseFormatConfig::Passthrough, // Override default
                arg_mapping: None,
            },
        );

        let config = McpConfig {
            servers: vec![McpServerConfig {
                name: "custom-search".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: HashMap::new(),
                },
                proxy: None,
                required: false,
                tools: Some(tools),
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("my_search".to_string()),
            }],
            ..Default::default()
        };

        let (refresh_tx, _) = mpsc::channel(10);
        let audit_log = Arc::new(AuditLog::new());
        let policy_engine = Arc::new(PolicyEngine::new(Arc::clone(&audit_log)));
        let approval_manager = Arc::new(ApprovalManager::new(policy_engine, audit_log));

        let orchestrator = McpOrchestrator {
            static_servers: DashMap::new(),
            tool_inventory: Arc::new(ToolInventory::new()),
            approval_manager,
            connection_pool: Arc::new(McpConnectionPool::new()),
            metrics: Arc::new(McpMetrics::new()),
            refresh_tx,
            active_executions: Arc::new(AtomicUsize::new(0)),
            shutdown_token: CancellationToken::new(),
            config,
        };

        let result = orchestrator.find_builtin_server(BuiltinToolType::WebSearchPreview);
        assert!(result.is_some());

        let (server_key, tool_name, response_format) = result.unwrap();
        assert_eq!(server_key, "custom-search");
        assert_eq!(tool_name, "my_search");
        // Should use the custom Passthrough format, not the default WebSearchCall
        assert_eq!(response_format, ResponseFormat::Passthrough);
    }

    #[test]
    fn test_find_builtin_server_no_config() {
        let orchestrator = McpOrchestrator::new_test();

        // No servers configured for any builtin type
        assert!(orchestrator
            .find_builtin_server(BuiltinToolType::WebSearchPreview)
            .is_none());
        assert!(orchestrator
            .find_builtin_server(BuiltinToolType::CodeInterpreter)
            .is_none());
        assert!(orchestrator
            .find_builtin_server(BuiltinToolType::FileSearch)
            .is_none());
    }

    #[test]
    fn test_apply_builtin_response_format() {
        use std::collections::HashMap;

        use crate::{
            approval::{audit::AuditLog, policy::PolicyEngine},
            inventory::types::ToolEntry,
        };

        // Create config with builtin_type but no explicit response_format for the tool
        let config = McpConfig {
            servers: vec![McpServerConfig {
                name: "brave".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: HashMap::new(),
                },
                proxy: None,
                required: false,
                tools: None, // No explicit tool config
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("brave_search".to_string()),
            }],
            ..Default::default()
        };

        let (refresh_tx, _) = mpsc::channel(10);
        let audit_log = Arc::new(AuditLog::new());
        let policy_engine = Arc::new(PolicyEngine::new(Arc::clone(&audit_log)));
        let approval_manager = Arc::new(ApprovalManager::new(policy_engine, audit_log));

        let orchestrator = McpOrchestrator {
            static_servers: DashMap::new(),
            tool_inventory: Arc::new(ToolInventory::new()),
            approval_manager,
            connection_pool: Arc::new(McpConnectionPool::new()),
            metrics: Arc::new(McpMetrics::new()),
            refresh_tx,
            active_executions: Arc::new(AtomicUsize::new(0)),
            shutdown_token: CancellationToken::new(),
            config,
        };

        // Simulate tool discovery - tool is registered with default Passthrough
        let tool = create_test_tool("brave_search");
        let entry = ToolEntry::from_server_tool("brave", tool);
        assert_eq!(entry.response_format, ResponseFormat::Passthrough); // Default
        orchestrator.tool_inventory.insert_entry(entry);

        // Apply builtin response format - should update to WebSearchCall
        orchestrator.apply_builtin_response_format(&orchestrator.config.servers[0]);

        // Verify the tool entry was updated
        let entry = orchestrator
            .tool_inventory
            .get_entry("brave", "brave_search")
            .expect("Tool should exist");
        assert_eq!(
            entry.response_format,
            ResponseFormat::WebSearchCall,
            "Builtin type should auto-apply WebSearchCall format"
        );
    }

    #[test]
    fn test_apply_builtin_response_format_with_explicit_override() {
        use std::collections::HashMap;

        use crate::{
            approval::{audit::AuditLog, policy::PolicyEngine},
            core::config::{ResponseFormatConfig, ToolConfig},
            inventory::types::ToolEntry,
        };

        // Create config with builtin_type AND explicit response_format override
        let mut tools = HashMap::new();
        tools.insert(
            "brave_search".to_string(),
            ToolConfig {
                alias: None,
                response_format: ResponseFormatConfig::Passthrough, // Explicit override
                arg_mapping: None,
            },
        );

        let config = McpConfig {
            servers: vec![McpServerConfig {
                name: "brave".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: HashMap::new(),
                },
                proxy: None,
                required: false,
                tools: Some(tools),
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("brave_search".to_string()),
            }],
            ..Default::default()
        };

        let (refresh_tx, _) = mpsc::channel(10);
        let audit_log = Arc::new(AuditLog::new());
        let policy_engine = Arc::new(PolicyEngine::new(Arc::clone(&audit_log)));
        let approval_manager = Arc::new(ApprovalManager::new(policy_engine, audit_log));

        let orchestrator = McpOrchestrator {
            static_servers: DashMap::new(),
            tool_inventory: Arc::new(ToolInventory::new()),
            approval_manager,
            connection_pool: Arc::new(McpConnectionPool::new()),
            metrics: Arc::new(McpMetrics::new()),
            refresh_tx,
            active_executions: Arc::new(AtomicUsize::new(0)),
            shutdown_token: CancellationToken::new(),
            config,
        };

        // Simulate tool discovery
        let tool = create_test_tool("brave_search");
        let entry = ToolEntry::from_server_tool("brave", tool);
        orchestrator.tool_inventory.insert_entry(entry);

        // Apply builtin response format - should NOT override because explicit config exists
        orchestrator.apply_builtin_response_format(&orchestrator.config.servers[0]);

        // Verify the tool entry kept Passthrough (explicit override)
        let entry = orchestrator
            .tool_inventory
            .get_entry("brave", "brave_search")
            .expect("Tool should exist");
        assert_eq!(
            entry.response_format,
            ResponseFormat::Passthrough,
            "Explicit override should be preserved"
        );
    }
}
