// tests/common/mock_mcp_server.rs - Mock MCP server for testing
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    service::RequestContext,
    tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpService,
    },
    ErrorData as McpError, RoleServer, ServerHandler,
};
use tokio::net::TcpListener;

/// Mock MCP server that returns hardcoded responses for testing
pub struct MockMCPServer {
    pub port: u16,
    pub server_handle: Option<tokio::task::JoinHandle<()>>,
}

/// Mock MCP server that always fails tool execution with a caller-provided marker.
pub struct MockFailingMCPServer {
    pub port: u16,
    pub server_handle: Option<tokio::task::JoinHandle<()>>,
}

/// Simple test server with mock search tools
#[derive(Clone)]
pub struct MockSearchServer {
    tool_router: ToolRouter<MockSearchServer>,
}

impl Default for MockSearchServer {
    fn default() -> Self {
        Self::new()
    }
}

/// Test server with a tool that always returns an MCP internal error.
#[derive(Clone)]
pub struct MockFailingSearchServer {
    error_marker: String,
    tool_router: ToolRouter<MockFailingSearchServer>,
}

impl MockFailingSearchServer {
    pub fn new(error_marker: impl Into<String>) -> Self {
        Self {
            error_marker: error_marker.into(),
            tool_router: Self::tool_router(),
        }
    }
}

#[allow(
    clippy::unused_self,
    clippy::unnecessary_wraps,
    reason = "proc macro generated"
)]
#[tool_router]
impl MockSearchServer {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Mock web search tool")]
    fn brave_web_search(
        &self,
        Parameters(params): Parameters<serde_json::Map<String, serde_json::Value>>,
    ) -> Result<CallToolResult, McpError> {
        let query = params
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("test");
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Mock search results for: {query}"
        ))]))
    }

    #[tool(description = "Mock local search tool")]
    fn brave_local_search(
        &self,
        Parameters(_params): Parameters<serde_json::Map<String, serde_json::Value>>,
    ) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![Content::text(
            "Mock local search results",
        )]))
    }
}

#[tool_handler]
impl ServerHandler for MockSearchServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation::from_build_env(),
            instructions: Some("Mock server for testing".to_string()),
        }
    }

    async fn initialize(
        &self,
        _request: InitializeRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<InitializeResult, McpError> {
        Ok(self.get_info())
    }
}

impl MockMCPServer {
    /// Start a mock MCP server on an available port
    #[expect(
        clippy::disallowed_methods,
        clippy::expect_used,
        reason = "test infrastructure"
    )]
    pub async fn start() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Find an available port
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();

        // Create the MCP service using rmcp's StreamableHttpService
        let service = StreamableHttpService::new(
            || Ok(MockSearchServer::new()),
            LocalSessionManager::default().into(),
            Default::default(),
        );

        let app = axum::Router::new().nest_service("/mcp", service);

        let server_handle = tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("Mock MCP server failed to start");
        });

        // Give the server a moment to start
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        Ok(MockMCPServer {
            port,
            server_handle: Some(server_handle),
        })
    }

    /// Get the full URL for this mock server
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}/mcp", self.port)
    }

    /// Stop the mock server
    pub async fn stop(&mut self) {
        if let Some(handle) = self.server_handle.take() {
            handle.abort();
            // Wait a moment for cleanup
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
    }
}

impl Drop for MockMCPServer {
    fn drop(&mut self) {
        if let Some(handle) = self.server_handle.take() {
            handle.abort();
        }
    }
}

#[allow(
    clippy::unused_self,
    clippy::unnecessary_wraps,
    reason = "proc macro generated"
)]
#[tool_router]
impl MockFailingSearchServer {
    #[tool(description = "Mock web search tool that always fails")]
    fn brave_web_search(
        &self,
        Parameters(_params): Parameters<serde_json::Map<String, serde_json::Value>>,
    ) -> Result<CallToolResult, McpError> {
        Err(McpError::internal_error(
            format!("mock internal MCP failure: {}", self.error_marker),
            None,
        ))
    }
}

#[tool_handler]
impl ServerHandler for MockFailingSearchServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation::from_build_env(),
            instructions: Some("Mock failing server for testing".to_string()),
        }
    }

    async fn initialize(
        &self,
        _request: InitializeRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<InitializeResult, McpError> {
        Ok(self.get_info())
    }
}

impl MockFailingMCPServer {
    #[expect(
        clippy::disallowed_methods,
        clippy::expect_used,
        reason = "test infrastructure"
    )]
    pub async fn start(
        error_marker: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();
        let error_marker = error_marker.to_string();

        let service = StreamableHttpService::new(
            move || Ok(MockFailingSearchServer::new(error_marker.clone())),
            LocalSessionManager::default().into(),
            Default::default(),
        );

        let app = axum::Router::new().nest_service("/mcp", service);

        let server_handle = tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("Mock failing MCP server failed to start");
        });

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        Ok(Self {
            port,
            server_handle: Some(server_handle),
        })
    }

    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}/mcp", self.port)
    }

    pub async fn stop(&mut self) {
        if let Some(handle) = self.server_handle.take() {
            handle.abort();
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
    }
}

impl Drop for MockFailingMCPServer {
    fn drop(&mut self) {
        if let Some(handle) = self.server_handle.take() {
            handle.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MockFailingMCPServer, MockMCPServer};

    #[tokio::test]
    async fn test_mock_server_startup() {
        let mut server = MockMCPServer::start().await.unwrap();
        assert!(server.port > 0);
        assert!(server.url().contains(&server.port.to_string()));
        server.stop().await;
    }

    #[tokio::test]
    async fn test_mock_server_with_rmcp_client() {
        let mut server = MockMCPServer::start().await.unwrap();

        use rmcp::{transport::StreamableHttpClientTransport, ServiceExt};

        let transport = StreamableHttpClientTransport::from_uri(server.url().as_str());
        let client = ().serve(transport).await;

        assert!(client.is_ok(), "Should be able to connect to mock server");

        if let Ok(client) = client {
            let tools = client.peer().list_all_tools().await;
            assert!(tools.is_ok(), "Should be able to list tools");

            if let Ok(tools) = tools {
                assert_eq!(tools.len(), 2, "Should have 2 tools");
                assert!(tools.iter().any(|t| t.name == "brave_web_search"));
                assert!(tools.iter().any(|t| t.name == "brave_local_search"));
            }

            // Shutdown by dropping the client
            drop(client);
        }

        server.stop().await;
    }

    #[tokio::test]
    async fn test_mock_failing_server_startup() {
        let mut server = MockFailingMCPServer::start("marker").await.unwrap();
        assert!(server.port > 0);
        assert!(server.url().contains(&server.port.to_string()));
        server.stop().await;
    }
}
