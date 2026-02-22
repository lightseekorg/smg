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
    #[serde(default)]
    pub external_auth: bool,
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
            .field("external_auth", &self.external_auth)
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
            return Err("db_url should not be empty".to_string());
        }

        let url = Url::parse(s).map_err(|e| format!("invalid db_url: {e}"))?;

        let scheme = url.scheme();
        if scheme != "postgres" && scheme != "postgresql" {
            return Err(format!("unsupported URL scheme: {scheme}"));
        }

        if url.host().is_none() {
            return Err("db_url must have a host".to_string());
        }

        let path = url.path();
        let dbname = path
            .strip_prefix('/')
            .filter(|p| !p.is_empty())
            .map(|s| s.to_string());
        if dbname.is_none() {
            return Err("db_url must include a database name".to_string());
        }

        if self.pool_max == 0 {
            return Err("pool_max must be greater than 0".to_string());
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

#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default function must match field type Option<u64>"
)]
fn default_redis_retention_days() -> Option<u64> {
    Some(30)
}

impl RedisConfig {
    pub fn validate(&self) -> Result<(), String> {
        let s = self.url.trim();
        if s.is_empty() {
            return Err("redis url should not be empty".to_string());
        }

        let url = Url::parse(s).map_err(|e| format!("invalid redis url: {e}"))?;

        let scheme = url.scheme();
        if scheme != "redis" && scheme != "rediss" {
            return Err(format!("unsupported URL scheme: {scheme}"));
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── PostgresConfig::validate ────────────────────────────────────────

    #[test]
    fn postgres_valid_url_succeeds() {
        let cfg = PostgresConfig {
            db_url: "postgres://user:pass@localhost:5432/mydb".to_string(),
            pool_max: 16,
        };
        cfg.validate()
            .expect("valid postgres URL should pass validation");
    }

    #[test]
    fn postgres_postgresql_scheme_succeeds() {
        let cfg = PostgresConfig {
            db_url: "postgresql://user:pass@localhost/mydb".to_string(),
            pool_max: 8,
        };
        cfg.validate()
            .expect("postgresql:// scheme should also be accepted");
    }

    #[test]
    fn postgres_empty_url_fails() {
        let cfg = PostgresConfig {
            db_url: "  ".to_string(),
            pool_max: 16,
        };
        let err = cfg.validate().expect_err("empty URL should fail");
        assert!(
            err.contains("not be empty"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn postgres_non_postgres_scheme_fails() {
        let cfg = PostgresConfig {
            db_url: "mysql://user:pass@localhost/mydb".to_string(),
            pool_max: 16,
        };
        let err = cfg.validate().expect_err("mysql scheme should be rejected");
        assert!(
            err.contains("unsupported URL scheme"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn postgres_missing_host_fails() {
        // `postgres:///mydb` is a valid URL with no host
        let cfg = PostgresConfig {
            db_url: "postgres:///mydb".to_string(),
            pool_max: 16,
        };
        let err = cfg.validate().expect_err("missing host should fail");
        assert!(
            err.contains("must have a host"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn postgres_missing_database_name_fails() {
        let cfg = PostgresConfig {
            db_url: "postgres://user:pass@localhost".to_string(),
            pool_max: 16,
        };
        let err = cfg
            .validate()
            .expect_err("missing database name should fail");
        assert!(
            err.contains("database name"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn postgres_pool_max_zero_fails() {
        let cfg = PostgresConfig {
            db_url: "postgres://user:pass@localhost/mydb".to_string(),
            pool_max: 0,
        };
        let err = cfg.validate().expect_err("pool_max=0 should fail");
        assert!(
            err.contains("greater than 0"),
            "unexpected error message: {err}"
        );
    }

    // ── RedisConfig::validate ───────────────────────────────────────────

    #[test]
    fn redis_valid_url_succeeds() {
        let cfg = RedisConfig {
            url: "redis://:password@localhost:6379/0".to_string(),
            pool_max: 16,
            retention_days: Some(30),
        };
        cfg.validate()
            .expect("valid redis URL should pass validation");
    }

    #[test]
    fn redis_rediss_scheme_succeeds() {
        let cfg = RedisConfig {
            url: "rediss://:password@redis.example.com:6380".to_string(),
            pool_max: 8,
            retention_days: None,
        };
        cfg.validate()
            .expect("rediss:// scheme should also be accepted");
    }

    #[test]
    fn redis_empty_url_fails() {
        let cfg = RedisConfig {
            url: String::new(),
            pool_max: 16,
            retention_days: Some(30),
        };
        let err = cfg.validate().expect_err("empty URL should fail");
        assert!(
            err.contains("not be empty"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn redis_non_redis_scheme_fails() {
        let cfg = RedisConfig {
            url: "http://localhost:6379".to_string(),
            pool_max: 16,
            retention_days: Some(30),
        };
        let err = cfg.validate().expect_err("http scheme should be rejected");
        assert!(
            err.contains("unsupported URL scheme"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn redis_missing_host_fails() {
        let cfg = RedisConfig {
            url: "redis:///0".to_string(),
            pool_max: 16,
            retention_days: Some(30),
        };
        let err = cfg.validate().expect_err("missing host should fail");
        assert!(
            err.contains("must have a host"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn redis_pool_max_zero_fails() {
        let cfg = RedisConfig {
            url: "redis://localhost:6379".to_string(),
            pool_max: 0,
            retention_days: Some(30),
        };
        let err = cfg.validate().expect_err("pool_max=0 should fail");
        assert!(
            err.contains("greater than 0"),
            "unexpected error message: {err}"
        );
    }

    // ── HistoryBackend ──────────────────────────────────────────────────

    #[test]
    fn history_backend_default_is_memory() {
        assert_eq!(HistoryBackend::default(), HistoryBackend::Memory);
    }

    #[test]
    fn history_backend_serde_roundtrip_memory() {
        let backend = HistoryBackend::Memory;
        let json = serde_json::to_string(&backend).expect("serialize Memory");
        assert_eq!(json, r#""memory""#);
        let deserialized: HistoryBackend = serde_json::from_str(&json).expect("deserialize Memory");
        assert_eq!(deserialized, HistoryBackend::Memory);
    }

    #[test]
    fn history_backend_serde_roundtrip_none() {
        let backend = HistoryBackend::None;
        let json = serde_json::to_string(&backend).expect("serialize None");
        assert_eq!(json, r#""none""#);
        let deserialized: HistoryBackend = serde_json::from_str(&json).expect("deserialize None");
        assert_eq!(deserialized, HistoryBackend::None);
    }

    #[test]
    fn history_backend_serde_roundtrip_oracle() {
        let backend = HistoryBackend::Oracle;
        let json = serde_json::to_string(&backend).expect("serialize Oracle");
        assert_eq!(json, r#""oracle""#);
        let deserialized: HistoryBackend = serde_json::from_str(&json).expect("deserialize Oracle");
        assert_eq!(deserialized, HistoryBackend::Oracle);
    }

    #[test]
    fn history_backend_serde_roundtrip_postgres() {
        let backend = HistoryBackend::Postgres;
        let json = serde_json::to_string(&backend).expect("serialize Postgres");
        assert_eq!(json, r#""postgres""#);
        let deserialized: HistoryBackend =
            serde_json::from_str(&json).expect("deserialize Postgres");
        assert_eq!(deserialized, HistoryBackend::Postgres);
    }

    #[test]
    fn history_backend_serde_roundtrip_redis() {
        let backend = HistoryBackend::Redis;
        let json = serde_json::to_string(&backend).expect("serialize Redis");
        assert_eq!(json, r#""redis""#);
        let deserialized: HistoryBackend = serde_json::from_str(&json).expect("deserialize Redis");
        assert_eq!(deserialized, HistoryBackend::Redis);
    }
}
