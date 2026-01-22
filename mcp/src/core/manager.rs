//! MCP client management and orchestration.
//!
//! Manages static MCP servers (from config) and dynamic MCP servers (from requests).
//! Static clients are never evicted; dynamic clients use LRU eviction via the connection pool.
//! Request-scoped tools are handled by `RequestMcpContext` and do not use the pool.

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

use backoff::ExponentialBackoffBuilder;
use dashmap::DashMap;
use openai_protocol::responses::{ResponseTool, ResponseToolType};
use rmcp::{
    model::{
        CallToolRequestParam, CallToolResult, GetPromptRequestParam, GetPromptResult,
        ReadResourceRequestParam, ReadResourceResult, SubscribeRequestParam,
        UnsubscribeRequestParam,
    },
    service::RunningService,
    transport::{
        sse_client::SseClientConfig, streamable_http_client::StreamableHttpClientTransportConfig,
        ConfigureCommandExt, SseClientTransport, StreamableHttpClientTransport, TokioChildProcess,
    },
    RoleClient, ServiceExt,
};
use serde_json::Map;
use tracing::{debug, error, info, warn};

use super::{
    config::{McpConfig, McpProxyConfig, McpServerConfig, McpTransport, Prompt, RawResource, Tool},
    pool::McpConnectionPool,
};
use crate::{
    error::{McpError, McpResult},
    inventory::{ToolArgs, ToolInventory},
};

/// Type alias for MCP client
type McpClient = RunningService<RoleClient, ()>;

pub struct McpManager {
    static_clients: Arc<DashMap<String, Arc<McpClient>>>,
    inventory: Arc<ToolInventory>,
    connection_pool: Arc<McpConnectionPool>,
    _config: McpConfig,
}

impl McpManager {
    const MAX_DYNAMIC_CLIENTS: usize = 200;

    pub async fn new(config: McpConfig, pool_max_connections: usize) -> McpResult<Self> {
        let inventory = Arc::new(ToolInventory::new());

        let mut connection_pool =
            McpConnectionPool::with_full_config(pool_max_connections, config.proxy.clone());

        let inventory_clone = Arc::clone(&inventory);
        connection_pool.set_eviction_callback(move |server_key: &str| {
            debug!(
                "LRU evicted dynamic server '{}' - clearing tools from inventory",
                server_key
            );
            inventory_clone.clear_server_tools(server_key);
        });

        let connection_pool = Arc::new(connection_pool);

        // Create storage for static clients
        let static_clients = Arc::new(DashMap::new());

        // Get global proxy config for all servers
        let global_proxy = config.proxy.as_ref();

        // Connect to all static servers from config
        for server_config in &config.servers {
            match Self::connect_server(server_config, global_proxy).await {
                Ok(client) => {
                    let client_arc = Arc::new(client);
                    // Load inventory for this server
                    Self::load_server_inventory(&inventory, &server_config.name, &client_arc).await;
                    static_clients.insert(server_config.name.clone(), client_arc);
                    info!("Connected to static server '{}'", server_config.name);
                }
                Err(e) => {
                    error!(
                        "Failed to connect to static server '{}': {}",
                        server_config.name, e
                    );
                }
            }
        }

        if static_clients.is_empty() {
            info!("No static MCP servers connected");
        }

        Ok(Self {
            static_clients,
            inventory,
            connection_pool,
            _config: config,
        })
    }

    pub async fn with_defaults(config: McpConfig) -> McpResult<Self> {
        Self::new(config, Self::MAX_DYNAMIC_CLIENTS).await
    }

    pub async fn get_client(&self, server_name: &str) -> Option<Arc<McpClient>> {
        if let Some(client) = self.static_clients.get(server_name) {
            return Some(Arc::clone(client.value()));
        }
        self.connection_pool.get(server_name)
    }

    /// Connect to an MCP server for a single request without using the pool.
    pub async fn connect_request_client(
        &self,
        server_config: &McpServerConfig,
    ) -> McpResult<Arc<McpClient>> {
        let client = Self::connect_server(server_config, self._config.proxy.as_ref()).await?;
        Ok(Arc::new(client))
    }

    pub async fn get_or_create_client(
        &self,
        server_config: McpServerConfig,
    ) -> McpResult<Arc<McpClient>> {
        let server_name = server_config.name.clone();

        if let Some(client) = self.static_clients.get(&server_name) {
            return Ok(Arc::clone(client.value()));
        }

        let server_key = Self::server_key(&server_config);
        let client = self
            .connection_pool
            .get_or_create(
                &server_key,
                server_config,
                |config, global_proxy| async move {
                    Self::connect_server(&config, global_proxy.as_ref()).await
                },
            )
            .await?;

        self.inventory.clear_server_tools(&server_key);
        Self::load_server_inventory(&self.inventory, &server_key, &client).await;
        Ok(client)
    }

    pub async fn create_request_context(
        &self,
        tools: Option<&[ResponseTool]>,
    ) -> Option<Arc<RequestMcpContext>> {
        let tools = tools?;
        let mut clients = HashMap::new();
        let inventory = Arc::new(ToolInventory::new());
        let mut has_valid_mcp_tools = false;

        for tool in tools {
            let Some(server_config) = self.parse_tool_server_config(tool) else {
                continue;
            };

            has_valid_mcp_tools = true;
            let server_key = Self::server_key(&server_config);

            if clients.contains_key(&server_key) {
                continue;
            }

            match self.connect_request_client(&server_config).await {
                Ok(client) => {
                    Self::load_server_inventory(&inventory, &server_key, &client).await;
                    clients.insert(server_key, client);
                }
                Err(err) => {
                    warn!(
                        "Failed to connect MCP request server {}: {}",
                        server_key, err
                    );
                }
            }
        }

        if !has_valid_mcp_tools {
            return None;
        }

        if clients.is_empty() {
            warn!("All MCP request tools failed to connect");
            return None;
        }

        Some(Arc::new(RequestMcpContext::new(inventory, clients)))
    }

    fn parse_tool_server_config(&self, tool: &ResponseTool) -> Option<McpServerConfig> {
        if !matches!(tool.r#type, ResponseToolType::Mcp) {
            return None;
        }

        let server_url = tool.server_url.as_ref().map(|s| s.trim().to_string())?;

        if !(server_url.starts_with("http://") || server_url.starts_with("https://")) {
            warn!(
                "Ignoring MCP server_url with unsupported scheme: {}",
                server_url
            );
            return None;
        }

        let name = tool
            .server_label
            .clone()
            .unwrap_or_else(|| "request-mcp".to_string());
        let token = tool.authorization.clone();

        let transport = if server_url.ends_with("/sse") {
            McpTransport::Sse {
                url: server_url,
                token,
            }
        } else {
            McpTransport::Streamable {
                url: server_url,
                token,
            }
        };

        Some(McpServerConfig {
            name,
            transport,
            proxy: None,
            required: false,
            tools: None,
        })
    }

    pub fn list_static_servers(&self) -> Vec<String> {
        self.static_clients
            .iter()
            .map(|e| e.key().clone())
            .collect()
    }

    pub fn is_static_server(&self, server_name: &str) -> bool {
        self.static_clients.contains_key(server_name)
    }

    pub fn register_static_server(&self, name: String, client: Arc<McpClient>) {
        self.static_clients.insert(name.clone(), client);
        info!("Registered static MCP server: {}", name);
    }

    /// List all available tools from all servers
    pub fn list_tools(&self) -> Vec<Tool> {
        self.inventory
            .list_tools()
            .into_iter()
            .map(|(_tool_name, _server_name, tool_info)| tool_info)
            .collect()
    }

    /// List tools only from specific servers plus all static servers
    ///
    /// This method filters tools to only include:
    /// 1. Tools from static servers (always visible)
    /// 2. Tools from the specified dynamic servers
    ///
    /// This provides request-scoped tool isolation while maintaining
    /// global visibility for static servers.
    pub fn list_tools_for_servers(&self, server_keys: &[String]) -> Vec<Tool> {
        self.inventory
            .list_tools()
            .into_iter()
            .filter(|(_tool_name, server_key, _tool_info)| {
                // Include if:
                // 1. It's a static server (check by name in static_clients)
                // 2. It's in the requested servers list
                self.is_static_server_by_key(server_key) || server_keys.contains(server_key)
            })
            .map(|(_tool_name, _server_key, tool_info)| tool_info)
            .collect()
    }

    /// Check if a server key belongs to a static server
    ///
    /// Static servers can be identified by checking if their name
    /// exists in the static_clients map. We need to handle the fact
    /// that static servers use name as key while dynamic use URL.
    fn is_static_server_by_key(&self, server_key: &str) -> bool {
        // For static servers, the server_key in inventory is the server name
        // Check if this key exists in static_clients
        self.static_clients.contains_key(server_key)
    }

    /// Call a tool by name with automatic type coercion.
    /// On collision, calls the first registered server's tool.
    pub async fn call_tool(
        &self,
        tool_name: &str,
        args: impl Into<ToolArgs>,
    ) -> McpResult<CallToolResult> {
        let (server_key, tool_info) = self
            .inventory
            .get_tool(tool_name)
            .ok_or_else(|| McpError::ToolNotFound(tool_name.to_string()))?;

        self.call_tool_impl(&server_key, tool_name, &tool_info, args)
            .await
    }

    /// Get a tool by name.
    pub fn get_tool(&self, tool_name: &str) -> Option<Tool> {
        self.inventory
            .get_tool(tool_name)
            .map(|(_server_name, tool_info)| tool_info)
    }

    /// Get a tool by qualified name (server + tool name).
    pub fn get_tool_qualified(&self, server_key: &str, tool_name: &str) -> Option<Tool> {
        self.inventory.get_tool_qualified(server_key, tool_name)
    }

    /// Call a tool by qualified name (no ambiguity from name collisions).
    pub async fn call_tool_qualified(
        &self,
        server_key: &str,
        tool_name: &str,
        args: impl Into<ToolArgs>,
    ) -> McpResult<CallToolResult> {
        let tool_info = self
            .inventory
            .get_tool_qualified(server_key, tool_name)
            .ok_or_else(|| McpError::ToolNotFound(format!("{}:{}", server_key, tool_name)))?;

        self.call_tool_impl(server_key, tool_name, &tool_info, args)
            .await
    }

    /// Check if a tool exists by qualified name.
    pub fn has_tool_qualified(&self, server_key: &str, tool_name: &str) -> bool {
        self.inventory.has_tool_qualified(server_key, tool_name)
    }

    /// Get all servers that have a tool with the given name.
    pub fn get_tool_servers(&self, tool_name: &str) -> Vec<String> {
        self.inventory.get_tool_servers(tool_name)
    }

    async fn call_tool_impl(
        &self,
        server_key: &str,
        tool_name: &str,
        tool_info: &Tool,
        args: impl Into<ToolArgs>,
    ) -> McpResult<CallToolResult> {
        let tool_schema = Some(serde_json::Value::Object((*tool_info.input_schema).clone()));
        let args_map = args
            .into()
            .into_map(tool_schema.as_ref())
            .map_err(McpError::InvalidArguments)?;

        let client = self
            .get_client(server_key)
            .await
            .ok_or_else(|| McpError::ServerNotFound(server_key.to_string()))?;

        let request = CallToolRequestParam {
            name: Cow::Owned(tool_name.to_string()),
            arguments: args_map,
        };

        client
            .call_tool(request)
            .await
            .map_err(|e| McpError::ToolExecution(format!("Failed to call tool: {}", e)))
    }

    /// Get a prompt by name
    pub async fn get_prompt(
        &self,
        prompt_name: &str,
        args: Option<Map<String, serde_json::Value>>,
    ) -> McpResult<GetPromptResult> {
        // Get server that owns this prompt
        let (server_name, _prompt_info) = self
            .inventory
            .get_prompt(prompt_name)
            .ok_or_else(|| McpError::PromptNotFound(prompt_name.to_string()))?;

        // Get client for that server
        let client = self
            .get_client(&server_name)
            .await
            .ok_or_else(|| McpError::ServerNotFound(server_name.clone()))?;

        // Get the prompt
        let request = GetPromptRequestParam {
            name: prompt_name.to_string(),
            arguments: args,
        };

        client
            .get_prompt(request)
            .await
            .map_err(|e| McpError::Transport(format!("Failed to get prompt: {}", e)))
    }

    /// List all available prompts
    pub fn list_prompts(&self) -> Vec<Prompt> {
        self.inventory
            .list_prompts()
            .into_iter()
            .map(|(_prompt_name, _server_name, prompt_info)| prompt_info)
            .collect()
    }

    /// Read a resource by URI
    pub async fn read_resource(&self, uri: &str) -> McpResult<ReadResourceResult> {
        // Get server that owns this resource
        let (server_name, _resource_info) = self
            .inventory
            .get_resource(uri)
            .ok_or_else(|| McpError::ResourceNotFound(uri.to_string()))?;

        // Get client for that server
        let client = self
            .get_client(&server_name)
            .await
            .ok_or_else(|| McpError::ServerNotFound(server_name.clone()))?;

        // Read the resource
        let request = ReadResourceRequestParam {
            uri: uri.to_string(),
        };

        client
            .read_resource(request)
            .await
            .map_err(|e| McpError::Transport(format!("Failed to read resource: {}", e)))
    }

    /// List all available resources
    pub fn list_resources(&self) -> Vec<RawResource> {
        self.inventory
            .list_resources()
            .into_iter()
            .map(|(_resource_uri, _server_name, resource_info)| resource_info)
            .collect()
    }

    /// Refresh inventory for a specific server
    pub async fn refresh_server_inventory(&self, server_name: &str) -> McpResult<()> {
        let client = self
            .get_client(server_name)
            .await
            .ok_or_else(|| McpError::ServerNotFound(server_name.to_string()))?;

        info!("Refreshing inventory for server: {}", server_name);
        Self::load_server_inventory(&self.inventory, server_name, &client).await;
        Ok(())
    }

    /// Start background refresh for ALL servers (static + dynamic)
    /// Refreshes every 10-15 minutes to keep tool inventory up-to-date
    pub fn spawn_background_refresh_all(
        self: Arc<Self>,
        refresh_interval: Duration,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(refresh_interval);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                interval.tick().await;

                // Get all static server keys
                // Note: Dynamic clients in the connection pool are refreshed on-demand
                // when they are accessed via get_or_create_client()
                let server_keys: Vec<String> = self
                    .static_clients
                    .iter()
                    .map(|e| e.key().clone())
                    .collect();

                if !server_keys.is_empty() {
                    debug!(
                        "Background refresh: Refreshing {} static server(s)",
                        server_keys.len()
                    );

                    for server_key in server_keys {
                        if let Err(e) = self.refresh_server_inventory(&server_key).await {
                            warn!("Background refresh failed for '{}': {}", server_key, e);
                        }
                    }

                    debug!("Background refresh: Completed refresh cycle");
                }
            }
        })
    }

    /// Check if a tool exists
    pub fn has_tool(&self, name: &str) -> bool {
        self.inventory.has_tool(name)
    }

    /// Get prompt info by name
    pub fn get_prompt_info(&self, name: &str) -> Option<Prompt> {
        self.inventory.get_prompt(name).map(|(_server, info)| info)
    }

    /// Get resource info by URI
    pub fn get_resource_info(&self, uri: &str) -> Option<RawResource> {
        self.inventory.get_resource(uri).map(|(_server, info)| info)
    }

    /// Subscribe to resource changes
    pub async fn subscribe_resource(&self, uri: &str) -> McpResult<()> {
        let (server_name, _resource_info) = self
            .inventory
            .get_resource(uri)
            .ok_or_else(|| McpError::ResourceNotFound(uri.to_string()))?;

        let client = self
            .get_client(&server_name)
            .await
            .ok_or_else(|| McpError::ServerNotFound(server_name.clone()))?;

        debug!("Subscribing to '{}' on '{}'", uri, server_name);

        client
            .peer()
            .subscribe(SubscribeRequestParam {
                uri: uri.to_string(),
            })
            .await
            .map_err(|e| McpError::ToolExecution(format!("Failed to subscribe: {}", e)))
    }

    /// Unsubscribe from resource changes
    pub async fn unsubscribe_resource(&self, uri: &str) -> McpResult<()> {
        let (server_name, _resource_info) = self
            .inventory
            .get_resource(uri)
            .ok_or_else(|| McpError::ResourceNotFound(uri.to_string()))?;

        let client = self
            .get_client(&server_name)
            .await
            .ok_or_else(|| McpError::ServerNotFound(server_name.clone()))?;

        debug!("Unsubscribing from '{}' on '{}'", uri, server_name);

        client
            .peer()
            .unsubscribe(UnsubscribeRequestParam {
                uri: uri.to_string(),
            })
            .await
            .map_err(|e| McpError::ToolExecution(format!("Failed to unsubscribe: {}", e)))
    }

    /// List all connected servers (static + dynamic)
    pub fn list_servers(&self) -> Vec<String> {
        let mut servers = Vec::new();

        // Add static servers
        servers.extend(self.static_clients.iter().map(|e| e.key().clone()));

        // Add dynamic servers from connection pool
        servers.extend(self.connection_pool.list_server_keys());

        servers
    }

    /// Disconnect from all servers (for cleanup)
    pub async fn shutdown(&self) {
        // Shutdown static servers
        let static_keys: Vec<String> = self
            .static_clients
            .iter()
            .map(|e| e.key().clone())
            .collect();
        for name in static_keys {
            if let Some((_key, client)) = self.static_clients.remove(&name) {
                // Try to unwrap Arc to call cancel
                match Arc::try_unwrap(client) {
                    Ok(client) => {
                        if let Err(e) = client.cancel().await {
                            warn!("Error disconnecting from static server '{}': {}", name, e);
                        }
                    }
                    Err(_) => {
                        warn!(
                            "Could not shutdown static server '{}': client still in use",
                            name
                        );
                    }
                }
            }
        }

        // Clear dynamic clients from connection pool
        // The pool will handle cleanup on drop
        self.connection_pool.clear();
    }

    /// Get statistics about the manager
    pub fn stats(&self) -> McpManagerStats {
        let (tools, prompts, resources) = self.inventory.counts();
        McpManagerStats {
            static_server_count: self.static_clients.len(),
            pool_stats: self.connection_pool.stats(),
            tool_count: tools,
            prompt_count: prompts,
            resource_count: resources,
        }
    }

    /// Get the shared tool inventory
    pub fn inventory(&self) -> Arc<ToolInventory> {
        Arc::clone(&self.inventory)
    }

    /// Get the connection pool
    pub fn connection_pool(&self) -> Arc<McpConnectionPool> {
        Arc::clone(&self.connection_pool)
    }

    // ========================================================================
    // Internal Helper Methods
    // ========================================================================

    /// Static helper for loading inventory (for new())
    /// Discover and cache tools/prompts/resources for a connected server
    ///
    /// This method is public to allow workflow-based inventory loading.
    /// It discovers all tools, prompts, and resources from the client and caches them in the inventory.
    pub async fn load_server_inventory(
        inventory: &Arc<ToolInventory>,
        server_key: &str,
        client: &Arc<McpClient>,
    ) {
        // Tools
        match client.peer().list_all_tools().await {
            Ok(ts) => {
                info!("Discovered {} tools from '{}'", ts.len(), server_key);
                for t in ts {
                    inventory.insert_tool(t.name.to_string(), server_key.to_string(), t);
                }
            }
            Err(e) => warn!("Failed to list tools from '{}': {}", server_key, e),
        }

        // Prompts
        match client.peer().list_all_prompts().await {
            Ok(ps) => {
                info!("Discovered {} prompts from '{}'", ps.len(), server_key);
                for p in ps {
                    inventory.insert_prompt(p.name.clone(), server_key.to_string(), p);
                }
            }
            Err(e) => debug!("No prompts or failed to list on '{}': {}", server_key, e),
        }

        // Resources
        match client.peer().list_all_resources().await {
            Ok(rs) => {
                info!("Discovered {} resources from '{}'", rs.len(), server_key);
                for r in rs {
                    inventory.insert_resource(r.uri.clone(), server_key.to_string(), r.raw);
                }
            }
            Err(e) => debug!("No resources or failed to list on '{}': {}", server_key, e),
        }
    }

    // ========================================================================
    // Connection Logic (from client_manager.rs)
    // ========================================================================

    /// Connect to an MCP server
    ///
    /// This method is public to allow workflow-based server registration at runtime.
    /// It handles connection with automatic retry for network-based transports (SSE/Streamable).
    pub async fn connect_server(
        config: &McpServerConfig,
        global_proxy: Option<&McpProxyConfig>,
    ) -> McpResult<McpClient> {
        let needs_retry = matches!(
            &config.transport,
            McpTransport::Sse { .. } | McpTransport::Streamable { .. }
        );
        if needs_retry {
            Self::connect_server_with_retry(config, global_proxy).await
        } else {
            Self::connect_server_impl(config, global_proxy).await
        }
    }

    /// Connect with exponential backoff retry for remote servers
    async fn connect_server_with_retry(
        config: &McpServerConfig,
        global_proxy: Option<&McpProxyConfig>,
    ) -> McpResult<McpClient> {
        let backoff = ExponentialBackoffBuilder::new()
            .with_initial_interval(Duration::from_secs(1))
            .with_max_interval(Duration::from_secs(30))
            .with_max_elapsed_time(Some(Duration::from_secs(30)))
            .build();

        backoff::future::retry(backoff, || async {
            match Self::connect_server_impl(config, global_proxy).await {
                Ok(client) => Ok(client),
                Err(e) => {
                    if Self::is_permanent_error(&e) {
                        error!(
                            "Permanent error connecting to '{}': {} - not retrying",
                            config.name, e
                        );
                        Err(backoff::Error::permanent(e))
                    } else {
                        warn!("Failed to connect to '{}', retrying: {}", config.name, e);
                        Err(backoff::Error::transient(e))
                    }
                }
            }
        })
        .await
    }

    /// Determine if an error is permanent (should not retry) or transient
    fn is_permanent_error(error: &McpError) -> bool {
        match error {
            McpError::Config(_) => true,
            McpError::Auth(_) => true,
            McpError::ServerNotFound(_) => true,
            McpError::Transport(_) => true,
            McpError::ConnectionFailed(msg) => {
                msg.contains("initialize")
                    || msg.contains("connection closed")
                    || msg.contains("connection refused")
                    || msg.contains("invalid URL")
                    || msg.contains("not found")
            }
            _ => false,
        }
    }

    /// Internal implementation of server connection (stdio/sse/streamable)
    async fn connect_server_impl(
        config: &McpServerConfig,
        global_proxy: Option<&McpProxyConfig>,
    ) -> McpResult<McpClient> {
        info!(
            "Connecting to MCP server '{}' via {:?}",
            config.name, config.transport
        );

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

                let client = ().serve(transport).await.map_err(|e| {
                    McpError::ConnectionFailed(format!("initialize stdio client: {}", e))
                })?;

                info!("Connected to stdio server '{}'", config.name);
                Ok(client)
            }

            McpTransport::Sse { url, token } => {
                // Resolve proxy configuration
                let proxy_config = super::proxy::resolve_proxy_config(config, global_proxy);

                // Create HTTP client with proxy support
                let client = if token.is_some() {
                    let mut builder =
                        reqwest::Client::builder().connect_timeout(Duration::from_secs(10));

                    // Apply proxy configuration using proxy.rs helper
                    if let Some(proxy_cfg) = proxy_config {
                        builder = super::proxy::apply_proxy_to_builder(builder, proxy_cfg)?;
                    }

                    // Add Authorization header
                    builder = builder.default_headers({
                        let mut headers = reqwest::header::HeaderMap::new();
                        headers.insert(
                            reqwest::header::AUTHORIZATION,
                            format!("Bearer {}", token.as_ref().unwrap())
                                .parse()
                                .map_err(|e| McpError::Transport(format!("auth token: {}", e)))?,
                        );
                        headers
                    });

                    builder
                        .build()
                        .map_err(|e| McpError::Transport(format!("build HTTP client: {}", e)))?
                } else {
                    super::proxy::create_http_client(proxy_config)?
                };

                let cfg = SseClientConfig {
                    sse_endpoint: url.clone().into(),
                    ..Default::default()
                };

                let transport = SseClientTransport::start_with_client(client, cfg)
                    .await
                    .map_err(|e| McpError::Transport(format!("create SSE transport: {}", e)))?;

                let client = ().serve(transport).await.map_err(|e| {
                    McpError::ConnectionFailed(format!("initialize SSE client: {}", e))
                })?;

                info!("Connected to SSE server '{}' at {}", config.name, url);
                Ok(client)
            }

            McpTransport::Streamable { url, token } => {
                // Note: Streamable transport doesn't support proxy yet
                let _proxy_config = super::proxy::resolve_proxy_config(config, global_proxy);
                if _proxy_config.is_some() {
                    warn!(
                        "Proxy configuration detected but not supported for Streamable transport on server '{}'",
                        config.name
                    );
                }

                let transport = if let Some(tok) = token {
                    let mut cfg = StreamableHttpClientTransportConfig::with_uri(url.as_str());
                    cfg.auth_header = Some(tok.to_string());
                    StreamableHttpClientTransport::from_config(cfg)
                } else {
                    StreamableHttpClientTransport::from_uri(url.as_str())
                };

                let client = ().serve(transport).await.map_err(|e| {
                    McpError::ConnectionFailed(format!("initialize streamable client: {}", e))
                })?;

                info!(
                    "Connected to streamable HTTP server '{}' at {}",
                    config.name, url
                );
                Ok(client)
            }
        }
    }

    /// Generate a unique key for a server config based on its transport
    pub fn server_key(config: &McpServerConfig) -> String {
        match &config.transport {
            McpTransport::Streamable { url, .. } => url.clone(),
            McpTransport::Sse { url, .. } => url.clone(),
            McpTransport::Stdio { command, .. } => command.clone(),
        }
    }
}

/// Request-scoped MCP context for Responses API.
///
/// Holds per-request clients and a private tool inventory, while still
/// allowing access to static tools managed by `McpManager`.
pub struct RequestMcpContext {
    inventory: Arc<ToolInventory>,
    clients: HashMap<String, Arc<McpClient>>,
}

impl RequestMcpContext {
    pub(crate) fn new(
        inventory: Arc<ToolInventory>,
        clients: HashMap<String, Arc<McpClient>>,
    ) -> Self {
        Self { inventory, clients }
    }

    pub fn server_keys(&self) -> Vec<String> {
        self.clients.keys().cloned().collect()
    }

    pub fn list_tools_for_servers(&self, server_keys: &[String]) -> Vec<Tool> {
        let server_keys_set: HashSet<&str> = server_keys.iter().map(String::as_str).collect();

        self.inventory
            .list_tools()
            .into_iter()
            .filter_map(|(_tool_name, server_key, tool_info)| {
                if server_keys_set.contains(server_key.as_str()) {
                    Some(tool_info)
                } else {
                    None
                }
            })
            .collect()
    }

    pub async fn call_tool(
        &self,
        tool_name: &str,
        args: impl Into<ToolArgs>,
    ) -> McpResult<CallToolResult> {
        let (server_key, tool_info) = self
            .inventory
            .get_tool(tool_name)
            .ok_or_else(|| McpError::ToolNotFound(tool_name.to_string()))?;

        let tool_schema = Some(serde_json::Value::Object((*tool_info.input_schema).clone()));
        let args_map = args
            .into()
            .into_map(tool_schema.as_ref())
            .map_err(McpError::InvalidArguments)?;

        let client = self
            .clients
            .get(&server_key)
            .ok_or_else(|| McpError::ServerNotFound(server_key.clone()))?;

        let request = CallToolRequestParam {
            name: Cow::Owned(tool_name.to_string()),
            arguments: args_map,
        };

        client
            .call_tool(request)
            .await
            .map_err(|e| McpError::ToolExecution(format!("Failed to call tool: {}", e)))
    }
}

impl Drop for RequestMcpContext {
    /// Best-effort cleanup for request-scoped clients.
    fn drop(&mut self) {
        if self.clients.is_empty() {
            return;
        }

        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            for (_, client) in self.clients.drain() {
                match Arc::try_unwrap(client) {
                    Ok(client) => {
                        handle.spawn(async move {
                            if let Err(err) = client.cancel().await {
                                warn!("Error closing request client: {}", err);
                            }
                        });
                    }
                    Err(_) => {
                        warn!("Request MCP client still has active references on drop");
                    }
                }
            }
        } else {
            warn!("No tokio runtime available for MCP client cleanup");
        }
    }
}

/// Statistics about the MCP manager
#[derive(Debug, Clone)]
pub struct McpManagerStats {
    /// Number of static servers registered
    pub static_server_count: usize,
    /// Connection pool statistics
    pub pool_stats: super::pool::PoolStats,
    /// Number of cached tools
    pub tool_count: usize,
    /// Number of cached prompts
    pub prompt_count: usize,
    /// Number of cached resources
    pub resource_count: usize,
}

#[cfg(test)]
mod tests {
    use std::{borrow::Cow, sync::Arc};

    use dashmap::DashMap;
    use rmcp::model::Tool;
    use serde_json::Map;

    use super::*;

    fn test_manager() -> McpManager {
        McpManager {
            static_clients: Arc::new(DashMap::new()),
            inventory: Arc::new(ToolInventory::new()),
            connection_pool: Arc::new(McpConnectionPool::new()),
            _config: McpConfig {
                servers: vec![],
                pool: Default::default(),
                proxy: None,
                warmup: vec![],
                inventory: Default::default(),
                policy: Default::default(),
            },
        }
    }

    fn test_tool(name: &str) -> Tool {
        Tool {
            name: Cow::Owned(name.to_string()),
            title: None,
            description: None,
            input_schema: Arc::new(Map::new()),
            output_schema: None,
            annotations: None,
            icons: None,
        }
    }

    #[tokio::test]
    async fn test_manager_creation() {
        let config = McpConfig {
            servers: vec![],
            pool: Default::default(),
            proxy: None,
            warmup: vec![],
            inventory: Default::default(),
            policy: Default::default(),
        };

        let manager = McpManager::new(config, 100).await.unwrap();
        assert_eq!(manager.list_static_servers().len(), 0);
    }

    #[test]
    fn test_parse_tool_server_config_rejects_invalid_scheme() {
        let manager = test_manager();
        let tool = ResponseTool {
            r#type: ResponseToolType::Mcp,
            server_url: Some("ftp://example.com/sse".to_string()),
            ..ResponseTool::default()
        };

        assert!(manager.parse_tool_server_config(&tool).is_none());
    }

    #[test]
    fn test_parse_tool_server_config_detects_sse_transport() {
        let manager = test_manager();
        let tool = ResponseTool {
            r#type: ResponseToolType::Mcp,
            server_url: Some("https://example.com/sse".to_string()),
            authorization: Some("token".to_string()),
            ..ResponseTool::default()
        };

        let config = manager.parse_tool_server_config(&tool).unwrap();
        match config.transport {
            McpTransport::Sse { url, token } => {
                assert_eq!(url, "https://example.com/sse");
                assert_eq!(token.as_deref(), Some("token"));
            }
            _ => panic!("expected SSE transport"),
        }
    }

    #[tokio::test]
    async fn test_create_request_context_returns_none_without_valid_mcp_tools() {
        let manager = test_manager();
        let tools = vec![ResponseTool {
            r#type: ResponseToolType::Function,
            ..ResponseTool::default()
        }];

        let ctx = manager.create_request_context(Some(&tools)).await;
        assert!(ctx.is_none());
    }

    #[test]
    fn test_list_tools_for_servers_filters() {
        let manager = test_manager();
        manager.inventory.insert_tool(
            "tool-a".to_string(),
            "server-a".to_string(),
            test_tool("tool-a"),
        );
        manager.inventory.insert_tool(
            "tool-b".to_string(),
            "server-b".to_string(),
            test_tool("tool-b"),
        );

        let filtered = manager.list_tools_for_servers(&["server-a".to_string()]);
        let names: Vec<String> = filtered
            .into_iter()
            .map(|tool| tool.name.to_string())
            .collect();

        assert_eq!(names, vec!["tool-a".to_string()]);
    }
}
