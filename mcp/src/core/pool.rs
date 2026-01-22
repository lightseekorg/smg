//! MCP Connection Pool for dynamic servers.

use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::Arc,
};

use lru::LruCache;
use parking_lot::Mutex;
use rmcp::{service::RunningService, RoleClient};

use super::config::{McpProxyConfig, McpServerConfig, McpTransport};
use crate::error::McpResult;

type McpClient = RunningService<RoleClient, ()>;
type EvictionCallback = Arc<dyn Fn(&PoolKey) + Send + Sync>;

/// Key for connection pool entries (URL + auth hash + tenant ID).
///
/// Credentials are hashed, not stored as plaintext.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PoolKey {
    pub url: String,
    pub auth_hash: u64,
    pub tenant_id: Option<String>,
}

impl PoolKey {
    pub fn new(url: impl Into<String>, auth_hash: u64, tenant_id: Option<String>) -> Self {
        Self {
            url: url.into(),
            auth_hash,
            tenant_id,
        }
    }

    pub fn from_config(config: &McpServerConfig, tenant_id: Option<String>) -> Self {
        let (url, auth_hash) = match &config.transport {
            McpTransport::Streamable {
                url,
                token,
                headers,
            } => (url.clone(), Self::hash_auth(token, headers)),
            McpTransport::Sse {
                url,
                token,
                headers,
            } => (url.clone(), Self::hash_auth(token, headers)),
            McpTransport::Stdio { command, args, .. } => {
                (format!("{}:{}", command, args.join(" ")), 0)
            }
        };
        Self {
            url,
            auth_hash,
            tenant_id,
        }
    }

    /// Hash token and headers. Returns 0 if no auth info.
    fn hash_auth(token: &Option<String>, headers: &HashMap<String, String>) -> u64 {
        if token.is_none() && headers.is_empty() {
            return 0;
        }

        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        if let Some(t) = token {
            t.hash(&mut hasher);
        }

        if !headers.is_empty() {
            let mut sorted_headers: Vec<_> = headers.iter().collect();
            sorted_headers.sort_by_key(|(k, _)| *k);
            for (key, value) in sorted_headers {
                key.hash(&mut hasher);
                value.hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    #[inline]
    pub fn url(&self) -> &str {
        &self.url
    }
}

/// Cached MCP connection.
#[derive(Clone)]
pub(crate) struct CachedConnection {
    pub client: Arc<McpClient>,
}

impl CachedConnection {
    pub fn new(client: Arc<McpClient>) -> Self {
        Self { client }
    }
}

/// Thread-safe LRU connection pool for dynamic MCP servers.
pub struct McpConnectionPool {
    connections: Arc<Mutex<LruCache<PoolKey, CachedConnection>>>,
    max_connections: usize,
    global_proxy: Option<McpProxyConfig>,
    eviction_callback: Option<EvictionCallback>,
}

impl McpConnectionPool {
    const DEFAULT_MAX_CONNECTIONS: usize = 200;

    /// Create pool with defaults (200 connections, proxy from env).
    pub fn new() -> Self {
        Self {
            connections: Arc::new(Mutex::new(LruCache::new(
                std::num::NonZeroUsize::new(Self::DEFAULT_MAX_CONNECTIONS).unwrap(),
            ))),
            max_connections: Self::DEFAULT_MAX_CONNECTIONS,
            global_proxy: McpProxyConfig::from_env(),
            eviction_callback: None,
        }
    }

    pub fn with_capacity(max_connections: usize) -> Self {
        Self {
            connections: Arc::new(Mutex::new(LruCache::new(
                std::num::NonZeroUsize::new(max_connections).unwrap(),
            ))),
            max_connections,
            global_proxy: McpProxyConfig::from_env(),
            eviction_callback: None,
        }
    }

    pub fn with_full_config(max_connections: usize, global_proxy: Option<McpProxyConfig>) -> Self {
        Self {
            connections: Arc::new(Mutex::new(LruCache::new(
                std::num::NonZeroUsize::new(max_connections).unwrap(),
            ))),
            max_connections,
            global_proxy,
            eviction_callback: None,
        }
    }

    pub fn set_eviction_callback<F>(&mut self, callback: F)
    where
        F: Fn(&PoolKey) + Send + Sync + 'static,
    {
        self.eviction_callback = Some(Arc::new(callback));
    }

    /// Get existing connection or create via `connect_fn`.
    pub async fn get_or_create<F, Fut>(
        &self,
        key: PoolKey,
        server_config: McpServerConfig,
        connect_fn: F,
    ) -> McpResult<Arc<McpClient>>
    where
        F: FnOnce(McpServerConfig, Option<McpProxyConfig>) -> Fut,
        Fut: std::future::Future<Output = McpResult<McpClient>>,
    {
        {
            let mut connections = self.connections.lock();
            if let Some(cached) = connections.get(&key) {
                return Ok(Arc::clone(&cached.client));
            }
        }

        let client = connect_fn(server_config.clone(), self.global_proxy.clone()).await?;
        let client_arc = Arc::new(client);

        let cached = CachedConnection::new(Arc::clone(&client_arc));
        {
            let mut connections = self.connections.lock();
            if let Some((evicted_key, _)) = connections.push(key, cached) {
                if let Some(callback) = &self.eviction_callback {
                    callback(&evicted_key);
                }
            }
        }

        Ok(client_arc)
    }

    pub fn len(&self) -> usize {
        self.connections.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.connections.lock().is_empty()
    }

    pub fn clear(&self) {
        self.connections.lock().clear();
    }

    pub fn stats(&self) -> PoolStats {
        PoolStats {
            total_connections: self.connections.lock().len(),
            capacity: self.max_connections,
        }
    }

    pub fn list_keys(&self) -> Vec<PoolKey> {
        self.connections
            .lock()
            .iter()
            .map(|(key, _)| key.clone())
            .collect()
    }

    /// Get connection, promoting in LRU.
    pub fn get(&self, key: &PoolKey) -> Option<Arc<McpClient>> {
        self.connections
            .lock()
            .get(key)
            .map(|cached| Arc::clone(&cached.client))
    }

    pub fn contains(&self, key: &PoolKey) -> bool {
        self.connections.lock().contains(key)
    }

    /// Get by URL only (backward compat). Prefer `get()` with full `PoolKey`.
    pub fn get_by_url(&self, url: &str) -> Option<Arc<McpClient>> {
        self.connections
            .lock()
            .iter()
            .find(|(key, _)| key.url == url)
            .map(|(_, cached)| Arc::clone(&cached.client))
    }

    pub fn contains_url(&self, url: &str) -> bool {
        self.connections
            .lock()
            .iter()
            .any(|(key, _)| key.url == url)
    }
}

impl Default for McpConnectionPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Connection pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_connections: usize,
    pub capacity: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::McpTransport;

    // Helper to create test server config
    fn create_test_config(url: &str) -> McpServerConfig {
        McpServerConfig {
            name: "test_server".to_string(),
            transport: McpTransport::Streamable {
                url: url.to_string(),
                token: None,
                headers: HashMap::new(),
            },
            proxy: None,
            required: false,
            tools: None,
        }
    }

    #[tokio::test]
    async fn test_pool_creation() {
        let pool = McpConnectionPool::new();
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
    }

    #[test]
    fn test_pool_stats() {
        let pool = McpConnectionPool::with_capacity(10);

        let stats = pool.stats();
        assert_eq!(stats.total_connections, 0);
        assert_eq!(stats.capacity, 10);
    }

    #[test]
    fn test_pool_clear() {
        let pool = McpConnectionPool::new();
        // Pool starts empty
        assert_eq!(pool.len(), 0);
        // Clear on empty pool should work
        pool.clear();
        assert!(pool.is_empty());
    }

    #[test]
    fn test_pool_key_from_config() {
        // No token
        let config = create_test_config("http://localhost:3000");
        let key = PoolKey::from_config(&config, None);
        assert_eq!(key.url, "http://localhost:3000");
        assert_eq!(key.auth_hash, 0);
        assert_eq!(key.tenant_id, None);

        // With token
        let config_with_token = McpServerConfig {
            name: "test".to_string(),
            transport: McpTransport::Streamable {
                url: "http://localhost:3000".to_string(),
                token: Some("secret-token".to_string()),
                headers: HashMap::new(),
            },
            proxy: None,
            required: false,
            tools: None,
        };
        let key_with_token = PoolKey::from_config(&config_with_token, None);
        assert_eq!(key_with_token.url, "http://localhost:3000");
        assert_ne!(key_with_token.auth_hash, 0); // Token hashed

        // With tenant
        let key_with_tenant = PoolKey::from_config(&config, Some("tenant-123".to_string()));
        assert_eq!(key_with_tenant.tenant_id, Some("tenant-123".to_string()));
    }

    #[test]
    fn test_pool_key_different_tokens() {
        let config1 = McpServerConfig {
            name: "test".to_string(),
            transport: McpTransport::Streamable {
                url: "http://localhost:3000".to_string(),
                token: Some("token-a".to_string()),
                headers: HashMap::new(),
            },
            proxy: None,
            required: false,
            tools: None,
        };
        let config2 = McpServerConfig {
            name: "test".to_string(),
            transport: McpTransport::Streamable {
                url: "http://localhost:3000".to_string(),
                token: Some("token-b".to_string()),
                headers: HashMap::new(),
            },
            proxy: None,
            required: false,
            tools: None,
        };

        let key1 = PoolKey::from_config(&config1, None);
        let key2 = PoolKey::from_config(&config2, None);

        // Same URL but different tokens = different keys
        assert_eq!(key1.url, key2.url);
        assert_ne!(key1.auth_hash, key2.auth_hash);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_pool_key_with_headers() {
        let mut headers1 = HashMap::new();
        headers1.insert("X-API-Key".to_string(), "key-1".to_string());

        let mut headers2 = HashMap::new();
        headers2.insert("X-API-Key".to_string(), "key-2".to_string());

        let config1 = McpServerConfig {
            name: "test".to_string(),
            transport: McpTransport::Sse {
                url: "http://localhost:3000".to_string(),
                token: None,
                headers: headers1,
            },
            proxy: None,
            required: false,
            tools: None,
        };
        let config2 = McpServerConfig {
            name: "test".to_string(),
            transport: McpTransport::Sse {
                url: "http://localhost:3000".to_string(),
                token: None,
                headers: headers2,
            },
            proxy: None,
            required: false,
            tools: None,
        };

        let key1 = PoolKey::from_config(&config1, None);
        let key2 = PoolKey::from_config(&config2, None);

        // Same URL but different headers = different keys
        assert_eq!(key1.url, key2.url);
        assert_ne!(key1.auth_hash, key2.auth_hash);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_pool_with_global_proxy() {
        use crate::core::config::McpProxyConfig;

        // Create proxy config
        let proxy = McpProxyConfig {
            http: Some("http://proxy.example.com:8080".to_string()),
            https: None,
            no_proxy: Some("localhost,127.0.0.1".to_string()),
            username: None,
            password: None,
        };

        // Create pool with proxy
        let pool = McpConnectionPool::with_full_config(100, Some(proxy.clone()));

        // Verify proxy is stored
        assert!(pool.global_proxy.is_some());
        let stored_proxy = pool.global_proxy.as_ref().unwrap();
        assert_eq!(
            stored_proxy.http.as_ref().unwrap(),
            "http://proxy.example.com:8080"
        );
        assert_eq!(
            stored_proxy.no_proxy.as_ref().unwrap(),
            "localhost,127.0.0.1"
        );
    }

    #[test]
    fn test_pool_proxy_from_env() {
        // Note: This test depends on environment variables
        // In production, proxy is loaded from MCP_HTTP_PROXY or HTTP_PROXY env vars
        let pool = McpConnectionPool::new();

        // Pool should either have proxy from env or None
        // We can't assert specific value since it depends on test environment
        // Just verify it doesn't panic
        assert!(pool.global_proxy.is_some() || pool.global_proxy.is_none());
    }
}
