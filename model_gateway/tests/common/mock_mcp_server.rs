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
use tokio::{
    net::TcpListener,
    sync::oneshot,
    task::JoinHandle,
    time::{timeout, Duration},
};

struct MockServerHarness {
    port: u16,
    shutdown_tx: Option<oneshot::Sender<()>>,
    server_handle: Option<JoinHandle<Result<(), std::io::Error>>>,
}

impl MockServerHarness {
    #[expect(
        clippy::disallowed_methods,
        reason = "test infrastructure uses a background server task"
    )]
    async fn start(app: axum::Router) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let (ready_tx, ready_rx) = oneshot::channel::<Result<(), String>>();

        let server_handle = tokio::spawn(async move {
            let _ = ready_tx.send(Ok(()));
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await
        });

        match ready_rx.await {
            Ok(Ok(())) => {}
            Ok(Err(err)) => return Err(err.into()),
            Err(_) => return Err("mock server readiness channel dropped unexpectedly".into()),
        }

        Ok(Self {
            port,
            shutdown_tx: Some(shutdown_tx),
            server_handle: Some(server_handle),
        })
    }

    fn port(&self) -> u16 {
        self.port
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}/mcp", self.port)
    }

    async fn stop(&mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        if let Some(handle) = self.server_handle.take() {
            let mut handle = handle;
            match timeout(Duration::from_secs(2), &mut handle).await {
                Ok(join_result) => {
                    let _ = join_result;
                }
                Err(_) => {
                    handle.abort();
                    let _ = handle.await;
                }
            }
        }
    }
}

impl Drop for MockServerHarness {
    fn drop(&mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        if let Some(handle) = self.server_handle.take() {
            handle.abort();
        }
    }
}

/// Mock MCP server that returns hardcoded responses for testing
pub struct MockMCPServer {
    harness: MockServerHarness,
}

/// Mock MCP server that always fails tool execution with a caller-provided marker.
pub struct MockFailingMCPServer {
    harness: MockServerHarness,
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
    fn router() -> axum::Router {
        let service = StreamableHttpService::new(
            || Ok(MockSearchServer::new()),
            LocalSessionManager::default().into(),
            Default::default(),
        );

        axum::Router::new().nest_service("/mcp", service)
    }

    /// Start a mock MCP server on an available port
    pub async fn start() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Self {
            harness: MockServerHarness::start(Self::router()).await?,
        })
    }

    pub fn port(&self) -> u16 {
        self.harness.port()
    }

    /// Get the full URL for this mock server
    pub fn url(&self) -> String {
        self.harness.url()
    }

    /// Stop the mock server
    pub async fn stop(&mut self) {
        self.harness.stop().await;
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
    fn router(error_marker: String) -> axum::Router {
        let service = StreamableHttpService::new(
            move || Ok(MockFailingSearchServer::new(error_marker.clone())),
            LocalSessionManager::default().into(),
            Default::default(),
        );

        axum::Router::new().nest_service("/mcp", service)
    }

    pub async fn start(
        error_marker: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Self {
            harness: MockServerHarness::start(Self::router(error_marker.to_string())).await?,
        })
    }

    pub fn port(&self) -> u16 {
        self.harness.port()
    }

    pub fn url(&self) -> String {
        self.harness.url()
    }

    pub async fn stop(&mut self) {
        self.harness.stop().await;
    }
}

#[cfg(test)]
mod tests {
    use super::{MockFailingMCPServer, MockMCPServer};

    #[tokio::test]
    async fn test_mock_server_startup() {
        let mut server = MockMCPServer::start().await.unwrap();
        assert!(server.port() > 0);
        assert!(server.url().contains(&server.port().to_string()));
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
        use rmcp::{
            model::CallToolRequestParam, transport::StreamableHttpClientTransport, ServiceExt,
        };

        let mut server = MockFailingMCPServer::start("marker").await.unwrap();
        assert!(server.port() > 0);
        assert!(server.url().contains(&server.port().to_string()));

        let transport = StreamableHttpClientTransport::from_uri(server.url().as_str());
        let client = ().serve(transport).await.expect("connect failing mock server");

        let err = client
            .call_tool(CallToolRequestParam {
                name: "brave_web_search".into(),
                arguments: Some(
                    serde_json::json!({
                        "query": "smoke"
                    })
                    .as_object()
                    .unwrap()
                    .clone(),
                ),
            })
            .await
            .expect_err("failing mock tool call should error");
        assert!(err.to_string().contains("marker"));

        server.stop().await;
    }
}
