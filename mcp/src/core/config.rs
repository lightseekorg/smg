//! MCP configuration types and utilities.
//!
//! Defines configuration structures for MCP servers, transports, proxies, and inventory.

use std::{collections::HashMap, fmt};

pub use rmcp::model::{Prompt, RawResource, Tool};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct McpConfig {
    /// Static MCP servers (loaded at startup)
    pub servers: Vec<McpServerConfig>,

    /// Connection pool settings
    #[serde(default)]
    pub pool: McpPoolConfig,

    /// Global MCP proxy configuration (default for all servers)
    /// Can be overridden per-server
    #[serde(default)]
    pub proxy: Option<McpProxyConfig>,

    /// Pre-warm these connections at startup
    #[serde(default)]
    pub warmup: Vec<WarmupServer>,

    /// Tool inventory refresh settings
    #[serde(default)]
    pub inventory: InventoryConfig,

    /// Approval policy configuration
    /// Default: allow all tools
    #[serde(default)]
    pub policy: PolicyConfig,
}

/// Policy configuration for tool approval decisions.
///
/// Evaluation order:
/// 1. Explicit tool policies (server:tool → decision)
/// 2. Server policies with trust levels
/// 3. Default policy (fallback)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PolicyConfig {
    /// Default policy when no other rules match.
    /// Default: "allow"
    #[serde(default = "default_allow")]
    pub default: PolicyDecisionConfig,

    /// Per-server policies with trust levels.
    #[serde(default)]
    pub servers: HashMap<String, ServerPolicyConfig>,

    /// Explicit per-tool policies (qualified name: "server:tool").
    #[serde(default)]
    pub tools: HashMap<String, PolicyDecisionConfig>,
}

/// Server-level policy configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerPolicyConfig {
    /// Trust level for this server.
    /// - trusted: Allow all tools unconditionally
    /// - standard: Use default policy (default)
    /// - untrusted: Deny destructive operations
    /// - sandboxed: Only allow read-only, no external access
    #[serde(default)]
    pub trust_level: TrustLevelConfig,

    /// Default policy for tools on this server.
    #[serde(default = "default_allow")]
    pub default: PolicyDecisionConfig,
}

/// Trust level for an MCP server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TrustLevelConfig {
    Trusted,
    #[default]
    Standard,
    Untrusted,
    Sandboxed,
}

/// Policy decision configuration.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PolicyDecisionConfig {
    #[default]
    Allow,
    Deny,
    /// Deny with a specific reason message.
    DenyWithReason(String),
}

impl Serialize for PolicyDecisionConfig {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            PolicyDecisionConfig::Allow => serializer.serialize_str("allow"),
            PolicyDecisionConfig::Deny => serializer.serialize_str("deny"),
            PolicyDecisionConfig::DenyWithReason(reason) => {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("deny_with_reason", reason)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for PolicyDecisionConfig {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::{self, MapAccess, Visitor};

        struct PolicyDecisionVisitor;

        impl<'de> Visitor<'de> for PolicyDecisionVisitor {
            type Value = PolicyDecisionConfig;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("\"allow\", \"deny\", or {\"deny_with_reason\": \"...\"}")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                match v {
                    "allow" => Ok(PolicyDecisionConfig::Allow),
                    "deny" => Ok(PolicyDecisionConfig::Deny),
                    _ => Err(E::unknown_variant(v, &["allow", "deny"])),
                }
            }

            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                if let Some(key) = map.next_key::<&str>()? {
                    if key == "deny_with_reason" {
                        let reason: String = map.next_value()?;
                        return Ok(PolicyDecisionConfig::DenyWithReason(reason));
                    }
                }
                Err(de::Error::custom("expected deny_with_reason key"))
            }
        }

        deserializer.deserialize_any(PolicyDecisionVisitor)
    }
}

fn default_allow() -> PolicyDecisionConfig {
    PolicyDecisionConfig::Allow
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            default: PolicyDecisionConfig::Allow,
            servers: HashMap::new(),
            tools: HashMap::new(),
        }
    }
}

impl Default for ServerPolicyConfig {
    fn default() -> Self {
        Self {
            trust_level: TrustLevelConfig::Standard,
            default: PolicyDecisionConfig::Allow,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpServerConfig {
    pub name: String,
    #[serde(flatten)]
    pub transport: McpTransport,

    /// Per-server proxy override (overrides global proxy)
    /// Set to `null` in YAML to force direct connection (no proxy)
    #[serde(default)]
    pub proxy: Option<McpProxyConfig>,

    /// Whether this server is required for router startup
    /// - true: Router startup fails if this server cannot be reached
    /// - false: Log warning but continue (default)
    #[serde(default)]
    pub required: bool,

    /// Tool-level configuration (aliases, response formats, arg mappings)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<HashMap<String, ToolConfig>>,
}

/// Configuration for a specific tool on an MCP server.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ToolConfig {
    /// Optional alias name (e.g., "web_search" for "brave_web_search")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,

    /// Response format for transformation (default: passthrough)
    #[serde(default)]
    pub response_format: ResponseFormatConfig,

    /// Argument mapping configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arg_mapping: Option<ArgMappingConfig>,
}

/// Response format configuration (mirrors ResponseFormat but for config).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormatConfig {
    #[default]
    Passthrough,
    WebSearchCall,
    CodeInterpreterCall,
    FileSearchCall,
}

/// Argument mapping configuration for tool aliases.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ArgMappingConfig {
    /// Rename arguments: from -> to
    #[serde(default)]
    pub renames: HashMap<String, String>,

    /// Default values for arguments
    #[serde(default)]
    pub defaults: HashMap<String, serde_json::Value>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "protocol", rename_all = "lowercase")]
pub enum McpTransport {
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        envs: HashMap<String, String>,
    },
    Sse {
        url: String,
        /// Bearer token for Authorization header
        #[serde(skip_serializing_if = "Option::is_none")]
        token: Option<String>,
        /// Additional headers (e.g., X-API-Key, custom auth)
        /// These affect connection identity and will be hashed for pool keying
        #[serde(default, skip_serializing_if = "HashMap::is_empty")]
        headers: HashMap<String, String>,
    },
    Streamable {
        url: String,
        /// Bearer token for Authorization header
        #[serde(skip_serializing_if = "Option::is_none")]
        token: Option<String>,
        /// Additional headers (e.g., X-API-Key, custom auth)
        /// These affect connection identity and will be hashed for pool keying
        #[serde(default, skip_serializing_if = "HashMap::is_empty")]
        headers: HashMap<String, String>,
    },
}

impl fmt::Debug for McpTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            McpTransport::Stdio {
                command,
                args,
                envs,
            } => f
                .debug_struct("Stdio")
                .field("command", command)
                .field("args", args)
                .field("envs", envs)
                .finish(),
            McpTransport::Sse {
                url,
                token,
                headers,
            } => f
                .debug_struct("Sse")
                .field("url", url)
                .field("token", &token.as_ref().map(|_| "****"))
                .field("headers", &format!("{} headers", headers.len()))
                .finish(),
            McpTransport::Streamable {
                url,
                token,
                headers,
            } => f
                .debug_struct("Streamable")
                .field("url", url)
                .field("token", &token.as_ref().map(|_| "****"))
                .field("headers", &format!("{} headers", headers.len()))
                .finish(),
        }
    }
}

/// MCP-specific proxy configuration (does NOT affect LLM API traffic)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpProxyConfig {
    /// HTTP proxy URL (e.g., "http://proxy.internal:8080")
    pub http: Option<String>,

    /// HTTPS proxy URL
    pub https: Option<String>,

    /// Comma-separated hosts to exclude from proxying
    /// Example: "localhost,127.0.0.1,*.internal,10.*"
    pub no_proxy: Option<String>,

    /// Custom proxy authentication (if needed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub username: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub password: Option<String>,
}

/// Connection pool configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpPoolConfig {
    /// Maximum cached connections per server URL
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,

    /// Idle timeout before closing connection (seconds)
    #[serde(default = "default_idle_timeout")]
    pub idle_timeout: u64,
}

/// Tool inventory refresh configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InventoryConfig {
    /// Enable automatic tool inventory refresh
    #[serde(default = "default_true")]
    pub enable_refresh: bool,

    /// Tool cache TTL (seconds) - how long tools are considered fresh
    #[serde(default = "default_tool_ttl")]
    pub tool_ttl: u64,

    /// Background refresh interval (seconds) - proactive refresh
    #[serde(default = "default_refresh_interval")]
    pub refresh_interval: u64,

    /// Refresh on tool call failure (try refreshing if tool not found)
    #[serde(default = "default_true")]
    pub refresh_on_error: bool,
}

/// Pre-warm server connections at startup
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WarmupServer {
    /// Server URL
    pub url: String,

    /// Server label/name
    pub label: String,

    /// Optional authentication token
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
}

// Default value functions
fn default_max_connections() -> usize {
    100
}

fn default_idle_timeout() -> u64 {
    300 // 5 minutes
}

fn default_true() -> bool {
    true
}

fn default_tool_ttl() -> u64 {
    300 // 5 minutes
}

fn default_refresh_interval() -> u64 {
    60 // 1 minute
}

// Default implementations
impl Default for McpPoolConfig {
    fn default() -> Self {
        Self {
            max_connections: default_max_connections(),
            idle_timeout: default_idle_timeout(),
        }
    }
}

impl Default for InventoryConfig {
    fn default() -> Self {
        Self {
            enable_refresh: true,
            tool_ttl: default_tool_ttl(),
            refresh_interval: default_refresh_interval(),
            refresh_on_error: true,
        }
    }
}

impl McpProxyConfig {
    /// Load proxy config from standard environment variables
    pub fn from_env() -> Option<Self> {
        let http = std::env::var("MCP_HTTP_PROXY")
            .ok()
            .or_else(|| std::env::var("HTTP_PROXY").ok());

        let https = std::env::var("MCP_HTTPS_PROXY")
            .ok()
            .or_else(|| std::env::var("HTTPS_PROXY").ok());

        let no_proxy = std::env::var("MCP_NO_PROXY")
            .ok()
            .or_else(|| std::env::var("NO_PROXY").ok());

        if http.is_some() || https.is_some() {
            Some(Self {
                http,
                https,
                no_proxy,
                username: None,
                password: None,
            })
        } else {
            None
        }
    }
}

impl McpConfig {
    /// Load configuration from a YAML file
    pub async fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = tokio::fs::read_to_string(path).await?;
        let config: Self = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Load configuration from environment variables (optional)
    pub fn from_env() -> Option<Self> {
        // This could be expanded to read from env vars
        // For now, return None to indicate env config not implemented
        None
    }

    /// Merge with environment-based proxy config
    pub fn with_env_proxy(mut self) -> Self {
        if self.proxy.is_none() {
            self.proxy = McpProxyConfig::from_env();
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_pool_config() {
        let config = McpPoolConfig::default();
        assert_eq!(config.max_connections, 100);
        assert_eq!(config.idle_timeout, 300);
    }

    #[test]
    fn test_default_inventory_config() {
        let config = InventoryConfig::default();
        assert!(config.enable_refresh);
        assert_eq!(config.tool_ttl, 300);
        assert_eq!(config.refresh_interval, 60);
        assert!(config.refresh_on_error);
    }

    #[test]
    fn test_proxy_from_env_empty() {
        // Ensure no proxy env vars are set for this test
        std::env::remove_var("MCP_HTTP_PROXY");
        std::env::remove_var("MCP_HTTPS_PROXY");
        std::env::remove_var("HTTP_PROXY");
        std::env::remove_var("HTTPS_PROXY");

        let proxy = McpProxyConfig::from_env();
        assert!(proxy.is_none(), "Should return None when no env vars set");
    }

    #[test]
    fn test_proxy_from_env_with_vars() {
        std::env::set_var("MCP_HTTP_PROXY", "http://test-proxy:8080");
        std::env::set_var("MCP_NO_PROXY", "localhost,127.0.0.1");

        let proxy = McpProxyConfig::from_env();
        assert!(proxy.is_some(), "Should return Some when env vars set");

        let proxy = proxy.unwrap();
        assert_eq!(proxy.http.as_ref().unwrap(), "http://test-proxy:8080");
        assert_eq!(proxy.no_proxy.as_ref().unwrap(), "localhost,127.0.0.1");

        // Cleanup
        std::env::remove_var("MCP_HTTP_PROXY");
        std::env::remove_var("MCP_NO_PROXY");
    }

    #[tokio::test]
    async fn test_yaml_minimal_config() {
        let yaml = r#"
servers:
  - name: "test-server"
    protocol: sse
    url: "http://localhost:3000/sse"
"#;

        let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse YAML");
        assert_eq!(config.servers.len(), 1);
        assert_eq!(config.servers[0].name, "test-server");
        assert!(!config.servers[0].required); // Should default to false
        assert!(config.servers[0].proxy.is_none()); // Should default to None
        assert_eq!(config.pool.max_connections, 100); // Should use default
        assert_eq!(config.inventory.tool_ttl, 300); // Should use default
    }

    #[tokio::test]
    async fn test_yaml_full_config() {
        let yaml = r#"
# Global proxy configuration
proxy:
  http: "http://global-proxy:8080"
  https: "http://global-proxy:8080"
  no_proxy: "localhost,127.0.0.1,*.internal"

# Connection pool settings
pool:
  max_connections: 50
  idle_timeout: 600

# Tool inventory settings
inventory:
  enable_refresh: true
  tool_ttl: 600
  refresh_interval: 120
  refresh_on_error: true

# Static servers
servers:
  - name: "required-server"
    protocol: sse
    url: "https://api.example.com/sse"
    token: "secret-token"
    required: true

  - name: "optional-server"
    protocol: stdio
    command: "mcp-server"
    args: ["--port", "3000"]
    required: false
    proxy:
      http: "http://server-specific-proxy:9090"

# Pre-warm connections
warmup:
  - url: "http://localhost:3000/sse"
    label: "local-dev"
"#;

        let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse YAML");

        // Check global proxy
        assert!(config.proxy.is_some());
        let global_proxy = config.proxy.as_ref().unwrap();
        assert_eq!(
            global_proxy.http.as_ref().unwrap(),
            "http://global-proxy:8080"
        );

        // Check pool config
        assert_eq!(config.pool.max_connections, 50);
        assert_eq!(config.pool.idle_timeout, 600);

        // Check inventory config
        assert_eq!(config.inventory.tool_ttl, 600);
        assert_eq!(config.inventory.refresh_interval, 120);

        // Check servers
        assert_eq!(config.servers.len(), 2);

        // Required server
        assert_eq!(config.servers[0].name, "required-server");
        assert!(config.servers[0].required);
        assert!(config.servers[0].proxy.is_none()); // Inherits global proxy

        // Optional server with custom proxy
        assert_eq!(config.servers[1].name, "optional-server");
        assert!(!config.servers[1].required);
        assert!(config.servers[1].proxy.is_some());
        assert_eq!(
            config.servers[1]
                .proxy
                .as_ref()
                .unwrap()
                .http
                .as_ref()
                .unwrap(),
            "http://server-specific-proxy:9090"
        );

        // Check warmup
        assert_eq!(config.warmup.len(), 1);
        assert_eq!(config.warmup[0].label, "local-dev");
    }

    #[tokio::test]
    async fn test_yaml_backward_compatibility() {
        // Old config format should still work
        let yaml = r#"
servers:
  - name: "legacy-server"
    protocol: sse
    url: "http://localhost:3000/sse"
"#;

        let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse old format");
        assert_eq!(config.servers.len(), 1);
        assert_eq!(config.servers[0].name, "legacy-server");
        assert!(!config.servers[0].required); // New field defaults to false
        assert!(config.servers[0].proxy.is_none()); // New field defaults to None
        assert!(config.proxy.is_none()); // New field defaults to None
        assert!(config.warmup.is_empty()); // New field defaults to empty
    }

    #[tokio::test]
    async fn test_yaml_null_proxy_override() {
        // Test that explicit null in YAML sets proxy to None
        let yaml = r#"
proxy:
  http: "http://global-proxy:8080"

servers:
  - name: "direct-connection"
    protocol: sse
    url: "http://localhost:3000/sse"
    proxy: null
"#;

        let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse YAML");
        assert!(config.proxy.is_some()); // Global proxy set
        assert_eq!(config.servers.len(), 1);
        assert!(config.servers[0].proxy.is_none()); // Explicitly set to null
    }

    #[test]
    fn test_transport_stdio() {
        let yaml = r#"
name: "test"
protocol: stdio
command: "mcp-server"
args: ["--port", "3000"]
envs:
  VAR1: "value1"
  VAR2: "value2"
"#;

        let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse stdio");
        assert_eq!(config.name, "test");

        match config.transport {
            McpTransport::Stdio {
                command,
                args,
                envs,
            } => {
                assert_eq!(command, "mcp-server");
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], "--port");
                assert_eq!(envs.get("VAR1").unwrap(), "value1");
            }
            _ => panic!("Expected Stdio transport"),
        }
    }

    #[test]
    fn test_transport_sse() {
        let yaml = r#"
name: "test"
protocol: sse
url: "http://localhost:3000/sse"
token: "secret"
"#;

        let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse sse");
        assert_eq!(config.name, "test");

        match config.transport {
            McpTransport::Sse { url, token, .. } => {
                assert_eq!(url, "http://localhost:3000/sse");
                assert_eq!(token.unwrap(), "secret");
            }
            _ => panic!("Expected Sse transport"),
        }
    }

    #[test]
    fn test_transport_streamable() {
        let yaml = r#"
name: "test"
protocol: streamable
url: "http://localhost:3000"
"#;

        let config: McpServerConfig =
            serde_yaml::from_str(yaml).expect("Failed to parse streamable");
        assert_eq!(config.name, "test");

        match config.transport {
            McpTransport::Streamable { url, token, .. } => {
                assert_eq!(url, "http://localhost:3000");
                assert!(token.is_none());
            }
            _ => panic!("Expected Streamable transport"),
        }
    }

    #[test]
    fn test_tool_config_with_alias() {
        let yaml = r#"
name: "brave"
protocol: sse
url: "https://mcp.brave.com/sse"
tools:
  brave_web_search:
    alias: web_search
    response_format: web_search_call
    arg_mapping:
      renames:
        q: query
      defaults:
        count: 10
"#;

        let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
        assert_eq!(config.name, "brave");
        let tools = config.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        let tool_config = tools.get("brave_web_search").unwrap();
        assert_eq!(tool_config.alias, Some("web_search".to_string()));
        assert_eq!(
            tool_config.response_format,
            ResponseFormatConfig::WebSearchCall
        );

        let arg_mapping = tool_config.arg_mapping.as_ref().unwrap();
        assert_eq!(arg_mapping.renames.get("q").unwrap(), "query");
        assert_eq!(
            arg_mapping.defaults.get("count").unwrap(),
            &serde_json::json!(10)
        );
    }

    #[test]
    fn test_tool_config_format_only() {
        let yaml = r#"
name: "filesystem"
protocol: stdio
command: "npx"
args: ["-y", "@anthropic/mcp-server-filesystem"]
tools:
  search:
    response_format: file_search_call
"#;

        let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
        let tools = config.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        let tool_config = tools.get("search").unwrap();
        assert!(tool_config.alias.is_none());
        assert_eq!(
            tool_config.response_format,
            ResponseFormatConfig::FileSearchCall
        );
        assert!(tool_config.arg_mapping.is_none());
    }

    #[test]
    fn test_tool_config_defaults() {
        let yaml = r#"
name: "test"
protocol: sse
url: "http://localhost:3000/sse"
tools:
  my_tool: {}
"#;

        let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
        let tools = config.tools.as_ref().unwrap();
        let tool_config = tools.get("my_tool").unwrap();
        assert!(tool_config.alias.is_none());
        assert_eq!(
            tool_config.response_format,
            ResponseFormatConfig::Passthrough
        );
        assert!(tool_config.arg_mapping.is_none());
    }

    #[test]
    fn test_response_format_config_serde() {
        let formats = vec![
            (ResponseFormatConfig::Passthrough, "\"passthrough\""),
            (ResponseFormatConfig::WebSearchCall, "\"web_search_call\""),
            (
                ResponseFormatConfig::CodeInterpreterCall,
                "\"code_interpreter_call\"",
            ),
            (ResponseFormatConfig::FileSearchCall, "\"file_search_call\""),
        ];

        for (format, expected) in formats {
            let serialized = serde_json::to_string(&format).unwrap();
            assert_eq!(serialized, expected);

            let deserialized: ResponseFormatConfig = serde_json::from_str(&serialized).unwrap();
            assert_eq!(deserialized, format);
        }
    }

    #[test]
    fn test_multiple_tools_config() {
        let yaml = r#"
name: "multi-tool-server"
protocol: sse
url: "https://example.com/sse"
tools:
  tool_a:
    alias: a
    response_format: web_search_call
  tool_b:
    response_format: file_search_call
  tool_c:
    alias: c
"#;

        let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
        let tools = config.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 3);

        let tool_a = tools.get("tool_a").unwrap();
        assert_eq!(tool_a.alias, Some("a".to_string()));
        assert_eq!(tool_a.response_format, ResponseFormatConfig::WebSearchCall);

        let tool_b = tools.get("tool_b").unwrap();
        assert!(tool_b.alias.is_none());
        assert_eq!(tool_b.response_format, ResponseFormatConfig::FileSearchCall);

        let tool_c = tools.get("tool_c").unwrap();
        assert_eq!(tool_c.alias, Some("c".to_string()));
        assert_eq!(tool_c.response_format, ResponseFormatConfig::Passthrough);
    }

    #[test]
    fn test_policy_config_default() {
        let config = PolicyConfig::default();
        assert_eq!(config.default, PolicyDecisionConfig::Allow);
        assert!(config.servers.is_empty());
        assert!(config.tools.is_empty());
    }

    #[test]
    fn test_policy_config_yaml_minimal() {
        let yaml = r#"
servers: []
"#;

        let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
        assert_eq!(config.policy.default, PolicyDecisionConfig::Allow);
        assert!(config.policy.servers.is_empty());
    }

    #[test]
    fn test_policy_config_yaml_full() {
        let yaml = r#"
servers:
  - name: "test"
    protocol: sse
    url: "http://localhost:3000/sse"

policy:
  default: allow
  servers:
    brave:
      trust_level: trusted
    untrusted_server:
      trust_level: untrusted
      default: deny
    sandbox_server:
      trust_level: sandboxed
  tools:
    "dangerous_server:delete_all": deny
    "risky_server:format_disk":
      deny_with_reason: "This operation is too dangerous"
"#;

        let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse");

        // Check default policy
        assert_eq!(config.policy.default, PolicyDecisionConfig::Allow);

        // Check server policies
        assert_eq!(config.policy.servers.len(), 3);

        let brave = config.policy.servers.get("brave").unwrap();
        assert_eq!(brave.trust_level, TrustLevelConfig::Trusted);
        assert_eq!(brave.default, PolicyDecisionConfig::Allow);

        let untrusted = config.policy.servers.get("untrusted_server").unwrap();
        assert_eq!(untrusted.trust_level, TrustLevelConfig::Untrusted);
        assert_eq!(untrusted.default, PolicyDecisionConfig::Deny);

        let sandbox = config.policy.servers.get("sandbox_server").unwrap();
        assert_eq!(sandbox.trust_level, TrustLevelConfig::Sandboxed);

        // Check tool policies
        assert_eq!(config.policy.tools.len(), 2);
        assert_eq!(
            config.policy.tools.get("dangerous_server:delete_all"),
            Some(&PolicyDecisionConfig::Deny)
        );
        assert_eq!(
            config.policy.tools.get("risky_server:format_disk"),
            Some(&PolicyDecisionConfig::DenyWithReason(
                "This operation is too dangerous".to_string()
            ))
        );
    }

    #[test]
    fn test_trust_level_config_serde() {
        let levels = vec![
            (TrustLevelConfig::Trusted, "\"trusted\""),
            (TrustLevelConfig::Standard, "\"standard\""),
            (TrustLevelConfig::Untrusted, "\"untrusted\""),
            (TrustLevelConfig::Sandboxed, "\"sandboxed\""),
        ];

        for (level, expected) in levels {
            let serialized = serde_json::to_string(&level).unwrap();
            assert_eq!(serialized, expected);

            let deserialized: TrustLevelConfig = serde_json::from_str(&serialized).unwrap();
            assert_eq!(deserialized, level);
        }
    }

    #[test]
    fn test_policy_decision_config_serde() {
        let decisions = vec![
            (PolicyDecisionConfig::Allow, "\"allow\""),
            (PolicyDecisionConfig::Deny, "\"deny\""),
        ];

        for (decision, expected) in decisions {
            let serialized = serde_json::to_string(&decision).unwrap();
            assert_eq!(serialized, expected);

            let deserialized: PolicyDecisionConfig = serde_json::from_str(&serialized).unwrap();
            assert_eq!(deserialized, decision);
        }

        // Test deny_with_reason
        let deny_with_reason = PolicyDecisionConfig::DenyWithReason("Not allowed".to_string());
        let serialized = serde_json::to_string(&deny_with_reason).unwrap();
        assert!(serialized.contains("deny_with_reason"));
        assert!(serialized.contains("Not allowed"));
    }
}
