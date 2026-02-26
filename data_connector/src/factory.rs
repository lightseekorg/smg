//! Factory function to create storage backends based on configuration.

use std::sync::Arc;

use tracing::info;
use url::Url;

use crate::{
    config::{HistoryBackend, OracleConfig, PostgresConfig, RedisConfig},
    core::{ConversationItemStorage, ConversationStorage, ResponseStorage},
    hooked::{HookedConversationItemStorage, HookedConversationStorage, HookedResponseStorage},
    hooks::StorageHook,
    memory::{MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage},
    noop::{NoOpConversationItemStorage, NoOpConversationStorage, NoOpResponseStorage},
    oracle::{OracleConversationItemStorage, OracleConversationStorage, OracleResponseStorage},
    postgres::{
        PostgresConversationItemStorage, PostgresConversationStorage, PostgresResponseStorage,
        PostgresStore,
    },
    redis::{
        RedisConversationItemStorage, RedisConversationStorage, RedisResponseStorage, RedisStore,
    },
};

/// Type alias for the storage tuple returned by factory functions.
/// This avoids clippy::type_complexity warnings while keeping Arc explicit.
pub type StorageTuple = (
    Arc<dyn ResponseStorage>,
    Arc<dyn ConversationStorage>,
    Arc<dyn ConversationItemStorage>,
);

/// Configuration for creating storage backends
pub struct StorageFactoryConfig<'a> {
    pub backend: &'a HistoryBackend,
    pub oracle: Option<&'a OracleConfig>,
    pub postgres: Option<&'a PostgresConfig>,
    pub redis: Option<&'a RedisConfig>,
    /// Optional storage hook. When provided, all three storage backends are
    /// wrapped in `Hooked*Storage` that runs before/after hooks around every
    /// storage operation.
    pub hook: Option<Arc<dyn StorageHook>>,
}

/// Create all three storage backends based on configuration.
///
/// # Arguments
/// * `config` - Storage factory configuration
///
/// # Returns
/// Tuple of (response_storage, conversation_storage, conversation_item_storage)
///
/// # Errors
/// Returns error string if required configuration is missing or initialization fails
pub async fn create_storage(config: StorageFactoryConfig<'_>) -> Result<StorageTuple, String> {
    let (resp, conv, items): StorageTuple = match config.backend {
        HistoryBackend::Memory => {
            info!("Initializing data connector: Memory");
            (
                Arc::new(MemoryResponseStorage::new()),
                Arc::new(MemoryConversationStorage::new()),
                Arc::new(MemoryConversationItemStorage::new()),
            )
        }
        HistoryBackend::None => {
            info!("Initializing data connector: None (no persistence)");
            (
                Arc::new(NoOpResponseStorage::new()),
                Arc::new(NoOpConversationStorage::new()),
                Arc::new(NoOpConversationItemStorage::new()),
            )
        }
        HistoryBackend::Oracle => {
            let oracle_cfg = config
                .oracle
                .ok_or("oracle configuration is required when history_backend=oracle")?;

            info!(
                "Initializing data connector: Oracle ATP (pool: {}-{})",
                oracle_cfg.pool_min, oracle_cfg.pool_max
            );

            let storages = create_oracle_storage(oracle_cfg)?;

            info!("Data connector initialized successfully: Oracle ATP");
            storages
        }
        HistoryBackend::Postgres => {
            let postgres_cfg = config
                .postgres
                .ok_or("Postgres configuration is required when history_backend=postgres")?;

            let log_db_url = match Url::parse(&postgres_cfg.db_url) {
                Ok(mut url) => {
                    if url.password().is_some() {
                        let _ = url.set_password(Some("****"));
                    }
                    url.to_string()
                }
                Err(_) => "<redacted>".to_string(),
            };

            info!(
                "Initializing data connector: Postgres (db_url: {}, pool_max: {})",
                log_db_url, postgres_cfg.pool_max
            );

            let storages = create_postgres_storage(postgres_cfg).await?;

            info!("Data connector initialized successfully: Postgres");
            storages
        }
        HistoryBackend::Redis => {
            let redis_cfg = config
                .redis
                .ok_or("Redis configuration is required when history_backend=redis")?;

            let log_redis_url = match Url::parse(&redis_cfg.url) {
                Ok(mut url) => {
                    if url.password().is_some() {
                        let _ = url.set_password(Some("****"));
                    }
                    url.to_string()
                }
                Err(_) => "<redacted>".to_string(),
            };

            info!(
                "Initializing data connector: Redis (url: {}, pool_max: {})",
                log_redis_url, redis_cfg.pool_max
            );

            let storages = create_redis_storage(redis_cfg)?;

            info!("Data connector initialized successfully: Redis");
            storages
        }
    };

    // Wrap backends in hooked storage when a hook is provided
    if let Some(hook) = config.hook {
        info!("Wrapping storage backends with hook");
        Ok((
            Arc::new(HookedResponseStorage::new(resp, hook.clone())),
            Arc::new(HookedConversationStorage::new(conv, hook.clone())),
            Arc::new(HookedConversationItemStorage::new(items, hook)),
        ))
    } else {
        Ok((resp, conv, items))
    }
}

/// Create Oracle storage backends with a single shared connection pool.
fn create_oracle_storage(oracle_cfg: &OracleConfig) -> Result<StorageTuple, String> {
    use crate::oracle::OracleStore;

    let store = OracleStore::new(
        oracle_cfg,
        &[
            OracleConversationStorage::init_schema,
            OracleConversationItemStorage::init_schema,
            OracleResponseStorage::init_schema,
        ],
    )?;

    Ok((
        Arc::new(OracleResponseStorage::new(store.clone())),
        Arc::new(OracleConversationStorage::new(store.clone())),
        Arc::new(OracleConversationItemStorage::new(store)),
    ))
}

async fn create_postgres_storage(postgres_cfg: &PostgresConfig) -> Result<StorageTuple, String> {
    let store = PostgresStore::new(postgres_cfg.clone())?;
    let postgres_resp = PostgresResponseStorage::new(store.clone())
        .await
        .map_err(|err| format!("failed to initialize Postgres response storage: {err}"))?;
    let postgres_conv = PostgresConversationStorage::new(store.clone())
        .await
        .map_err(|err| format!("failed to initialize Postgres conversation storage: {err}"))?;
    let postgres_item = PostgresConversationItemStorage::new(store.clone())
        .await
        .map_err(|err| format!("failed to initialize Postgres conversation item storage: {err}"))?;

    // Run versioned migrations after all tables are created
    let applied = store.run_migrations().await?;

    // Re-create indexes that were deferred during init because
    // migration-added columns did not yet exist.
    if !applied.is_empty() {
        store.ensure_response_indexes().await?;
    }

    Ok((
        Arc::new(postgres_resp),
        Arc::new(postgres_conv),
        Arc::new(postgres_item),
    ))
}

fn create_redis_storage(redis_cfg: &RedisConfig) -> Result<StorageTuple, String> {
    let store = RedisStore::new(redis_cfg.clone())?;
    let redis_resp = RedisResponseStorage::new(store.clone());
    let redis_conv = RedisConversationStorage::new(store.clone());
    let redis_item = RedisConversationItemStorage::new(store);

    Ok((
        Arc::new(redis_resp),
        Arc::new(redis_conv),
        Arc::new(redis_item),
    ))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::core::{NewConversation, NewConversationItem, StoredResponse};

    #[tokio::test]
    async fn test_create_storage_memory() {
        let config = StorageFactoryConfig {
            backend: &HistoryBackend::Memory,
            oracle: None,
            postgres: None,
            redis: None,
            hook: None,
        };
        let (resp, conv, items) = create_storage(config).await.unwrap();

        // Verify they work end-to-end
        let mut response = StoredResponse::new(None);
        response.input = json!("hello");
        let id = resp.store_response(response).await.unwrap();
        assert!(resp.get_response(&id).await.unwrap().is_some());

        let conversation = conv
            .create_conversation(NewConversation {
                id: None,
                metadata: None,
            })
            .await
            .unwrap();
        assert!(conv
            .get_conversation(&conversation.id)
            .await
            .unwrap()
            .is_some());

        let item = items
            .create_item(NewConversationItem {
                id: None,
                response_id: None,
                item_type: "message".to_string(),
                role: Some("user".to_string()),
                content: json!([]),
                status: Some("completed".to_string()),
            })
            .await
            .unwrap();
        assert!(items.get_item(&item.id).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_create_storage_none() {
        let config = StorageFactoryConfig {
            backend: &HistoryBackend::None,
            oracle: None,
            postgres: None,
            redis: None,
            hook: None,
        };
        let (resp, conv, _items) = create_storage(config).await.unwrap();

        // NoOp storage should accept writes but return nothing on reads
        let mut response = StoredResponse::new(None);
        response.input = json!("hello");
        let id = resp.store_response(response).await.unwrap();
        assert!(resp.get_response(&id).await.unwrap().is_none());
        assert!(conv
            .get_conversation(&"nonexistent".into())
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn test_create_storage_oracle_missing_config() {
        let config = StorageFactoryConfig {
            backend: &HistoryBackend::Oracle,
            oracle: None,
            postgres: None,
            redis: None,
            hook: None,
        };
        let err = create_storage(config).await.err().expect("should fail");
        assert!(err.contains("oracle configuration is required"));
    }

    #[tokio::test]
    async fn test_create_storage_postgres_missing_config() {
        let config = StorageFactoryConfig {
            backend: &HistoryBackend::Postgres,
            oracle: None,
            postgres: None,
            redis: None,
            hook: None,
        };
        let err = create_storage(config).await.err().expect("should fail");
        assert!(err.contains("Postgres configuration is required"));
    }

    #[tokio::test]
    async fn test_create_storage_redis_missing_config() {
        let config = StorageFactoryConfig {
            backend: &HistoryBackend::Redis,
            oracle: None,
            postgres: None,
            redis: None,
            hook: None,
        };
        let err = create_storage(config).await.err().expect("should fail");
        assert!(err.contains("Redis configuration is required"));
    }

    #[tokio::test]
    async fn test_create_storage_with_hook() {
        use std::sync::Arc;

        use async_trait::async_trait;

        use crate::{
            context::RequestContext,
            hooks::{BeforeHookResult, ExtraColumns, HookError, StorageHook, StorageOperation},
        };

        struct NoOpHook;

        #[async_trait]
        impl StorageHook for NoOpHook {
            async fn before(
                &self,
                _op: StorageOperation,
                _ctx: Option<&RequestContext>,
                _payload: &serde_json::Value,
            ) -> Result<BeforeHookResult, HookError> {
                Ok(BeforeHookResult::default())
            }

            async fn after(
                &self,
                _op: StorageOperation,
                _ctx: Option<&RequestContext>,
                _payload: &serde_json::Value,
                _result: &serde_json::Value,
                extra: &ExtraColumns,
            ) -> Result<ExtraColumns, HookError> {
                Ok(extra.clone())
            }
        }

        let config = StorageFactoryConfig {
            backend: &HistoryBackend::Memory,
            oracle: None,
            postgres: None,
            redis: None,
            hook: Some(Arc::new(NoOpHook)),
        };
        let (resp, conv, items) = create_storage(config).await.unwrap();

        // Verify hooked storage works end-to-end
        let mut response = StoredResponse::new(None);
        response.input = json!("hello");
        let id = resp.store_response(response).await.unwrap();
        assert!(resp.get_response(&id).await.unwrap().is_some());

        let conversation = conv
            .create_conversation(NewConversation {
                id: None,
                metadata: None,
            })
            .await
            .unwrap();
        assert!(conv
            .get_conversation(&conversation.id)
            .await
            .unwrap()
            .is_some());

        let item = items
            .create_item(NewConversationItem {
                id: None,
                response_id: None,
                item_type: "message".to_string(),
                role: Some("user".to_string()),
                content: json!([]),
                status: Some("completed".to_string()),
            })
            .await
            .unwrap();
        assert!(items.get_item(&item.id).await.unwrap().is_some());
    }
}
