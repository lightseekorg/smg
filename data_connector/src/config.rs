//! Storage backend configuration types.

use serde::{Deserialize, Serialize};
use url::Url;

/// History backend configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum HistoryBackend {
    #[default]
    Memory,
    None,
    Oracle,
    Postgres,
    Redis,
}

/// Oracle history backend configuration
#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct OracleConfig {
    /// ATP wallet or TLS config files directory
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wallet_path: Option<String>,
    /// DSN (e.g. `tcps://host:port/service`)
    pub connect_descriptor: String,
    pub username: String,
    pub password: String,
    #[serde(default = "default_pool_min")]
    pub pool_min: usize,
    #[serde(default = "default_pool_max")]
    pub pool_max: usize,
    #[serde(default = "default_pool_timeout_secs")]
    pub pool_timeout_secs: u64,
}

impl OracleConfig {
    pub fn default_pool_min() -> usize {
        default_pool_min()
    }

    pub fn default_pool_max() -> usize {
        default_pool_max()
    }

    pub fn default_pool_timeout_secs() -> u64 {
        default_pool_timeout_secs()
    }
}

fn default_pool_min() -> usize {
    1
}

fn default_pool_max() -> usize {
    16
}

fn default_pool_timeout_secs() -> u64 {
    30
}

impl std::fmt::Debug for OracleConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OracleConfig")
            .field("wallet_path", &self.wallet_path)
            .field("connect_descriptor", &self.connect_descriptor)
            .field("username", &self.username)
            .field("pool_min", &self.pool_min)
            .field("pool_max", &self.pool_max)
            .field("pool_timeout_secs", &self.pool_timeout_secs)
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PostgresConfig {
    // Database connection URL,
    // postgres://[user[:password]@][netloc][:port][/dbname][?param1=value1&...]
    pub db_url: String,
    // Database pool max size
    pub pool_max: usize,
}

impl PostgresConfig {
    pub fn default_pool_max() -> usize {
        16
    }

    pub fn validate(&self) -> Result<(), String> {
        let s = self.db_url.trim();
        if s.is_empty() {
            return Err("is it db-url should be not empty".to_string());
        }

        let url = Url::parse(s).map_err(|e| format!("invalid db_url: {}", e))?;

        let scheme = url.scheme();
        if scheme != "postgres" && scheme != "postgresql" {
            return Err(format!("don't support URL scheme: {}", scheme));
        }

        if url.host().is_none() {
            return Err("db_url must need host".to_string());
        }

        let path = url.path();
        let dbname = path
            .strip_prefix('/')
            .filter(|p| !p.is_empty())
            .map(|s| s.to_string());
        if dbname.is_none() {
            return Err("db_url must need database name".to_string());
        }

        if self.pool_max == 0 {
            return Err("pool_max must be greater 1, default is 16".to_string());
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RedisConfig {
    // Redis connection URL
    // redis://[:password@]host[:port][/db]
    pub url: String,
    // Connection pool max size
    #[serde(default = "default_redis_pool_max")]
    pub pool_max: usize,
    // Data retention in days. If None, data persists indefinitely.
    #[serde(default = "default_redis_retention_days")]
    pub retention_days: Option<u64>,
}

fn default_redis_pool_max() -> usize {
    16
}

fn default_redis_retention_days() -> Option<u64> {
    Some(30)
}

impl RedisConfig {
    pub fn validate(&self) -> Result<(), String> {
        let s = self.url.trim();
        if s.is_empty() {
            return Err("redis url should not be empty".to_string());
        }

        let url = Url::parse(s).map_err(|e| format!("invalid redis url: {}", e))?;

        let scheme = url.scheme();
        if scheme != "redis" && scheme != "rediss" {
            return Err(format!("unsupported URL scheme: {}", scheme));
        }

        if url.host().is_none() {
            return Err("redis url must have a host".to_string());
        }

        if self.pool_max == 0 {
            return Err("pool_max must be greater than 0".to_string());
        }

        Ok(())
    }
}
