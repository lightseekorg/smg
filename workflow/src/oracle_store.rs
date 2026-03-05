//! Oracle-backed [`StateStore`] implementation.
//!
//! Stores workflow state in an Oracle table (`WORKFLOW_STATES` by default).
//! The full [`WorkflowState<D>`] is serialised as JSON into a CLOB column,
//! with `status` and `workflow_type` denormalised into their own columns for
//! efficient filtering.
//!
//! All Oracle operations use [`tokio::task::block_in_place`] because the
//! `oracle` crate provides only a synchronous driver.

use std::{marker::PhantomData, sync::Arc, time::Duration};

use async_trait::async_trait;
use chrono::Utc;
use deadpool::managed::{Manager, Metrics, Pool, RecycleError, RecycleResult};
use oracle::{Connection, Connector};

use crate::{
    state::StateStore,
    types::{
        WorkflowContext, WorkflowData, WorkflowError, WorkflowInstanceId, WorkflowResult,
        WorkflowState,
    },
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for creating an [`OracleStateStore`].
#[derive(Debug, Clone)]
pub struct OracleStateStoreConfig {
    /// TNS alias or full connect descriptor (e.g. `tcps://host:1522/service`)
    pub connect_descriptor: String,
    pub username: String,
    pub password: String,
    pub external_auth: bool,
    /// Path to Oracle ATP wallet directory (sets `TNS_ADMIN`)
    pub wallet_path: Option<String>,
    /// Maximum connection pool size (default: 8)
    pub pool_max: usize,
    /// Pool wait timeout in seconds (default: 30, 0 = no timeout)
    pub pool_timeout_secs: u64,
    /// Table name override (default: `WORKFLOW_STATES`)
    pub table_name: Option<String>,
    /// Schema owner for qualified table references (e.g. `ADMIN`)
    pub owner: Option<String>,
}

impl OracleStateStoreConfig {
    fn table_name_upper(&self) -> String {
        self.table_name
            .as_deref()
            .unwrap_or("WORKFLOW_STATES")
            .to_ascii_uppercase()
    }

    fn qualified_table(&self) -> String {
        let table = self.table_name_upper();
        match &self.owner {
            Some(o) => format!("{o}.\"{table}\""),
            None => table,
        }
    }
}

impl Default for OracleStateStoreConfig {
    fn default() -> Self {
        Self {
            connect_descriptor: String::new(),
            username: String::new(),
            password: String::new(),
            external_auth: false,
            wallet_path: None,
            pool_max: 8,
            pool_timeout_secs: 30,
            table_name: None,
            owner: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Connection management (mirrors data_connector/src/oracle.rs pattern)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ConnectParams {
    username: String,
    password: String,
    connect_descriptor: String,
    external_auth: bool,
}

#[derive(Clone)]
pub(crate) struct WfOracleConnectionManager {
    params: Arc<ConnectParams>,
}

impl std::fmt::Debug for WfOracleConnectionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WfOracleConnectionManager")
            .field("connect_descriptor", &self.params.connect_descriptor)
            .finish()
    }
}

#[async_trait]
impl Manager for WfOracleConnectionManager {
    type Type = Connection;
    type Error = oracle::Error;

    fn create(
        &self,
    ) -> impl std::future::Future<Output = Result<Connection, oracle::Error>> + Send {
        let params = self.params.clone();
        async move {
            let mut conn = connect_oracle(
                params.external_auth,
                &params.username,
                &params.password,
                &params.connect_descriptor,
            )?;
            conn.set_autocommit(true);
            Ok(conn)
        }
    }

    #[expect(clippy::manual_async_fn)]
    fn recycle(
        &self,
        conn: &mut Connection,
        _: &Metrics,
    ) -> impl std::future::Future<Output = RecycleResult<Self::Error>> + Send {
        async move { conn.ping().map_err(RecycleError::Backend) }
    }
}

fn connect_oracle(
    external_auth: bool,
    username: &str,
    password: &str,
    connect_descriptor: &str,
) -> Result<Connection, oracle::Error> {
    if external_auth {
        Connector::new("", "", connect_descriptor)
            .external_auth(true)
            .connect()
    } else {
        Connection::connect(username, password, connect_descriptor)
    }
}

fn map_oracle_error(err: oracle::Error) -> String {
    if let Some(db_err) = err.db_error() {
        format!(
            "Oracle error (code {}): {}",
            db_err.code(),
            db_err.message()
        )
    } else {
        err.to_string()
    }
}

/// Convert an Oracle error to a WorkflowError, treating ORA-01403 as NotFound.
fn to_wf_error(e: String, instance_id: WorkflowInstanceId) -> WorkflowError {
    if e == "not_found" {
        WorkflowError::NotFound(instance_id)
    } else {
        WorkflowError::Storage(e)
    }
}

// ---------------------------------------------------------------------------
// OracleStateStore
// ---------------------------------------------------------------------------

/// Oracle-backed state store for workflow instances.
///
/// Multiple workflow engine types (worker registration, MCP, tokenizer, …)
/// share the same table, isolated by the `WORKFLOW_TYPE` column which is set
/// to [`WorkflowData::workflow_type()`].
#[derive(Clone)]
#[expect(
    dead_code,
    reason = "table_upper and owner retained for future migration support"
)]
pub struct OracleStateStore<D: WorkflowData> {
    pool: Pool<WfOracleConnectionManager>,
    table: String,
    table_upper: String,
    owner: Option<String>,
    _phantom: PhantomData<D>,
}

impl<D: WorkflowData> std::fmt::Debug for OracleStateStore<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OracleStateStore")
            .field("table", &self.table)
            .field("workflow_type", &D::workflow_type())
            .finish()
    }
}

impl<D: WorkflowData> OracleStateStore<D> {
    /// Create a new Oracle state store and ensure the backing table exists.
    pub fn new(config: &OracleStateStoreConfig) -> Result<Self, String> {
        // Configure wallet env if needed
        if let Some(wallet_path) = &config.wallet_path {
            let path = std::path::Path::new(wallet_path);
            if !path.is_dir() {
                return Err(format!(
                    "Oracle wallet path '{wallet_path}' is not a directory"
                ));
            }
            std::env::set_var("TNS_ADMIN", wallet_path);
        }

        // Bootstrap connection to create table
        let conn = connect_oracle(
            config.external_auth,
            &config.username,
            &config.password,
            &config.connect_descriptor,
        )
        .map_err(map_oracle_error)?;

        let table_upper = config.table_name_upper();
        let qualified = config.qualified_table();
        init_table(&conn, &table_upper, &qualified, config.owner.as_deref())?;
        drop(conn);

        // Build connection pool
        let params = Arc::new(ConnectParams {
            username: config.username.clone(),
            password: config.password.clone(),
            connect_descriptor: config.connect_descriptor.clone(),
            external_auth: config.external_auth,
        });
        let manager = WfOracleConnectionManager { params };

        let mut builder = Pool::builder(manager)
            .max_size(config.pool_max)
            .runtime(deadpool::Runtime::Tokio1);

        if config.pool_timeout_secs > 0 {
            builder = builder.wait_timeout(Some(Duration::from_secs(config.pool_timeout_secs)));
        }

        let pool = builder
            .build()
            .map_err(|e| format!("Failed to build Oracle workflow state pool: {e}"))?;

        Ok(Self {
            pool,
            table: qualified,
            table_upper,
            owner: config.owner.clone(),
            _phantom: PhantomData,
        })
    }

    /// Get a connection from the pool.
    async fn conn(&self) -> Result<deadpool::managed::Object<WfOracleConnectionManager>, String> {
        self.pool
            .get()
            .await
            .map_err(|e| format!("Failed to get Oracle connection: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Table initialisation
// ---------------------------------------------------------------------------

fn init_table(
    conn: &Connection,
    table_upper: &str,
    qualified: &str,
    owner: Option<&str>,
) -> Result<(), String> {
    let check_sql = match owner {
        Some(o) => format!(
            "SELECT COUNT(*) FROM all_tables WHERE owner = '{}' AND table_name = '{table_upper}'",
            o.to_ascii_uppercase()
        ),
        None => format!("SELECT COUNT(*) FROM user_tables WHERE table_name = '{table_upper}'"),
    };
    let exists: i64 = conn
        .query_row_as(&check_sql, &[])
        .map_err(map_oracle_error)?;

    if exists == 0 {
        let ddl = format!(
            "CREATE TABLE {qualified} (\
                INSTANCE_ID    VARCHAR2(64) PRIMARY KEY, \
                DEFINITION_ID  VARCHAR2(128) NOT NULL, \
                WORKFLOW_TYPE  VARCHAR2(64) NOT NULL, \
                STATUS         VARCHAR2(32) NOT NULL, \
                STATE_JSON     CLOB NOT NULL, \
                CREATED_AT     TIMESTAMP WITH TIME ZONE NOT NULL, \
                UPDATED_AT     TIMESTAMP WITH TIME ZONE NOT NULL, \
                EXPIRES_AT     TIMESTAMP WITH TIME ZONE \
                    DEFAULT (SYSTIMESTAMP + INTERVAL '7' DAY) NOT NULL\
            )"
        );
        conn.execute(&ddl, &[]).map_err(map_oracle_error)?;
        tracing::info!(table = qualified, "Created workflow states table");
    }

    create_index_if_missing(
        conn,
        table_upper,
        "IDX_WS_STATUS",
        &format!("CREATE INDEX IDX_WS_STATUS ON {qualified} (STATUS)"),
        owner,
    );
    create_index_if_missing(
        conn,
        table_upper,
        "IDX_WS_TYPE_STATUS",
        &format!("CREATE INDEX IDX_WS_TYPE_STATUS ON {qualified} (WORKFLOW_TYPE, STATUS)"),
        owner,
    );
    create_index_if_missing(
        conn,
        table_upper,
        "IDX_WS_UPDATED",
        &format!("CREATE INDEX IDX_WS_UPDATED ON {qualified} (UPDATED_AT)"),
        owner,
    );

    Ok(())
}

fn create_index_if_missing(
    conn: &Connection,
    table_upper: &str,
    index_name: &str,
    ddl: &str,
    owner: Option<&str>,
) {
    let check_sql = match owner {
        Some(o) => format!(
            "SELECT COUNT(*) FROM all_indexes WHERE owner = '{}' AND table_name = '{table_upper}' AND index_name = '{index_name}'",
            o.to_ascii_uppercase()
        ),
        None => format!(
            "SELECT COUNT(*) FROM user_indexes WHERE table_name = '{table_upper}' AND index_name = '{index_name}'"
        ),
    };
    let count: i64 = match conn.query_row_as(&check_sql, &[]) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(
                "Failed to check index {index_name}: {}",
                map_oracle_error(e)
            );
            return;
        }
    };
    if count == 0 {
        if let Err(err) = conn.execute(ddl, &[]) {
            if let Some(db_err) = err.db_error() {
                // ORA-00955: name already used, ORA-01408: column list already indexed
                if db_err.code() == 955 || db_err.code() == 1408 {
                    return;
                }
            }
            tracing::warn!(
                "Failed to create index {index_name}: {}",
                map_oracle_error(err)
            );
        }
    }
}

// ---------------------------------------------------------------------------
// StateStore implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl<D: WorkflowData> StateStore<D> for OracleStateStore<D> {
    async fn save(&self, state: WorkflowState<D>) -> WorkflowResult<()> {
        let id = state.instance_id.to_string();
        let def_id = state.definition_id.to_string();
        let wf_type = D::workflow_type().to_string();
        let status = format!("{:?}", state.status);
        let json = serde_json::to_string(&state)
            .map_err(|e| WorkflowError::Storage(format!("Failed to serialize state: {e}")))?;
        let created_at = state.created_at;
        let updated_at = state.updated_at;

        let conn = self.conn().await.map_err(WorkflowError::Storage)?;
        let table = &self.table;

        tokio::task::block_in_place(|| {
            let sql = format!(
                "MERGE INTO {table} dst \
                 USING (SELECT :id AS INSTANCE_ID FROM DUAL) src \
                 ON (dst.INSTANCE_ID = src.INSTANCE_ID) \
                 WHEN MATCHED THEN UPDATE SET \
                     STATUS = :status, STATE_JSON = :json, UPDATED_AT = :upd \
                 WHEN NOT MATCHED THEN INSERT \
                     (INSTANCE_ID, DEFINITION_ID, WORKFLOW_TYPE, STATUS, STATE_JSON, CREATED_AT, UPDATED_AT) \
                     VALUES (:id, :def, :wft, :status, :json, :crt, :upd)"
            );
            conn.execute_named(
                &sql,
                &[
                    ("id", &id),
                    ("status", &status),
                    ("json", &json),
                    ("upd", &updated_at),
                    ("def", &def_id),
                    ("wft", &wf_type),
                    ("crt", &created_at),
                ],
            )
            .map_err(map_oracle_error)
            .map_err(WorkflowError::Storage)?;
            Ok(())
        })
    }

    async fn load(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<WorkflowState<D>> {
        let id = instance_id.to_string();
        let conn = self.conn().await.map_err(WorkflowError::Storage)?;
        let table = &self.table;

        tokio::task::block_in_place(|| {
            let sql = format!("SELECT STATE_JSON FROM {table} WHERE INSTANCE_ID = :1");
            let json: String = conn
                .query_row_as(&sql, &[&id])
                .map_err(|e| to_wf_error(ora_to_string(e), instance_id))?;
            serde_json::from_str(&json)
                .map_err(|e| WorkflowError::Storage(format!("Failed to deserialize state: {e}")))
        })
    }

    async fn update<F>(&self, instance_id: WorkflowInstanceId, f: F) -> WorkflowResult<()>
    where
        F: FnOnce(&mut WorkflowState<D>) + Send,
    {
        let id = instance_id.to_string();
        let mut conn = self.conn().await.map_err(WorkflowError::Storage)?;
        let table = &self.table;

        tokio::task::block_in_place(|| {
            // Begin transaction
            conn.set_autocommit(false);

            let result: Result<(), WorkflowError> = (|| {
                // Lock the row
                let select_sql =
                    format!("SELECT STATE_JSON FROM {table} WHERE INSTANCE_ID = :1 FOR UPDATE");
                let json: String = conn
                    .query_row_as(&select_sql, &[&id])
                    .map_err(|e| to_wf_error(ora_to_string(e), instance_id))?;

                let mut state: WorkflowState<D> = serde_json::from_str(&json).map_err(|e| {
                    WorkflowError::Storage(format!("Failed to deserialize state: {e}"))
                })?;

                f(&mut state);
                state.updated_at = Utc::now();

                let new_json = serde_json::to_string(&state).map_err(|e| {
                    WorkflowError::Storage(format!("Failed to serialize state: {e}"))
                })?;
                let new_status = format!("{:?}", state.status);

                let update_sql = format!(
                    "UPDATE {table} SET STATE_JSON = :1, STATUS = :2, UPDATED_AT = :3 \
                     WHERE INSTANCE_ID = :4"
                );
                conn.execute(
                    &update_sql,
                    &[
                        &new_json as &dyn oracle::sql_type::ToSql,
                        &new_status as &dyn oracle::sql_type::ToSql,
                        &state.updated_at as &dyn oracle::sql_type::ToSql,
                        &id as &dyn oracle::sql_type::ToSql,
                    ],
                )
                .map_err(|e| WorkflowError::Storage(map_oracle_error(e)))?;

                conn.commit()
                    .map_err(|e| WorkflowError::Storage(map_oracle_error(e)))?;
                Ok(())
            })();

            // Always restore autocommit
            if result.is_err() {
                let _ = conn.rollback();
            }
            conn.set_autocommit(true);

            result
        })
    }

    async fn delete(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<()> {
        let id = instance_id.to_string();
        let conn = self.conn().await.map_err(WorkflowError::Storage)?;
        let table = &self.table;

        tokio::task::block_in_place(|| {
            let sql = format!("DELETE FROM {table} WHERE INSTANCE_ID = :1");
            conn.execute(&sql, &[&id])
                .map_err(|e| WorkflowError::Storage(map_oracle_error(e)))?;
            Ok(())
        })
    }

    async fn list_active(&self) -> WorkflowResult<Vec<WorkflowState<D>>> {
        let wf_type = D::workflow_type().to_string();
        let conn = self.conn().await.map_err(WorkflowError::Storage)?;
        let table = &self.table;

        tokio::task::block_in_place(|| {
            let sql = format!(
                "SELECT STATE_JSON FROM {table} \
                 WHERE STATUS IN ('Running', 'Pending') AND WORKFLOW_TYPE = :1"
            );
            query_states::<D>(&conn, &sql, &[&wf_type]).map_err(WorkflowError::Storage)
        })
    }

    async fn list_all(&self) -> WorkflowResult<Vec<WorkflowState<D>>> {
        let wf_type = D::workflow_type().to_string();
        let conn = self.conn().await.map_err(WorkflowError::Storage)?;
        let table = &self.table;

        tokio::task::block_in_place(|| {
            let sql = format!("SELECT STATE_JSON FROM {table} WHERE WORKFLOW_TYPE = :1");
            query_states::<D>(&conn, &sql, &[&wf_type]).map_err(WorkflowError::Storage)
        })
    }

    async fn is_cancelled(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<bool> {
        let id = instance_id.to_string();
        let conn = self.conn().await.map_err(WorkflowError::Storage)?;
        let table = &self.table;

        tokio::task::block_in_place(|| {
            let sql = format!("SELECT STATUS FROM {table} WHERE INSTANCE_ID = :1");
            let status: String = conn
                .query_row_as(&sql, &[&id])
                .map_err(|e| to_wf_error(ora_to_string(e), instance_id))?;
            Ok(status == "Cancelled")
        })
    }

    async fn cleanup_old_workflows(&self, ttl: Duration) -> usize {
        let cutoff = Utc::now()
            - chrono::Duration::from_std(ttl).unwrap_or_else(|_| chrono::Duration::hours(1));
        let wf_type = D::workflow_type().to_string();

        let conn = match self.conn().await {
            Ok(c) => c,
            Err(e) => {
                tracing::error!(error = %e, "cleanup_old_workflows: pool error");
                return 0;
            }
        };
        let table = &self.table;

        match tokio::task::block_in_place(|| {
            let sql = format!(
                "DELETE FROM {table} \
                 WHERE STATUS IN ('Completed', 'Failed', 'Cancelled') \
                   AND UPDATED_AT < :1 \
                   AND WORKFLOW_TYPE = :2"
            );
            let stmt = conn
                .execute(
                    &sql,
                    &[
                        &cutoff as &dyn oracle::sql_type::ToSql,
                        &wf_type as &dyn oracle::sql_type::ToSql,
                    ],
                )
                .map_err(map_oracle_error)?;
            let count = stmt.row_count().map_err(map_oracle_error)?;
            Ok::<_, String>(count as usize)
        }) {
            Ok(n) => {
                if n > 0 {
                    tracing::info!(
                        removed = n,
                        workflow_type = D::workflow_type(),
                        "Cleaned up old workflow states from Oracle"
                    );
                }
                n
            }
            Err(e) => {
                tracing::error!(error = %e, "cleanup_old_workflows: query failed");
                0
            }
        }
    }

    async fn get_context(
        &self,
        instance_id: WorkflowInstanceId,
    ) -> WorkflowResult<WorkflowContext<D>> {
        let id = instance_id.to_string();
        let conn = self.conn().await.map_err(WorkflowError::Storage)?;
        let table = &self.table;

        tokio::task::block_in_place(|| {
            let sql = format!("SELECT STATE_JSON FROM {table} WHERE INSTANCE_ID = :1");
            let json: String = conn
                .query_row_as(&sql, &[&id])
                .map_err(|e| to_wf_error(ora_to_string(e), instance_id))?;

            // Extract just the "context" field from the full state JSON
            let value: serde_json::Value = serde_json::from_str(&json)
                .map_err(|e| WorkflowError::Storage(format!("Failed to parse JSON: {e}")))?;
            let ctx_val = value
                .get("context")
                .ok_or_else(|| WorkflowError::Storage("Missing 'context' in state JSON".into()))?;
            serde_json::from_value(ctx_val.clone())
                .map_err(|e| WorkflowError::Storage(format!("Failed to deserialize context: {e}")))
        })
    }

    async fn cleanup_if_terminal(&self, instance_id: WorkflowInstanceId) -> bool {
        let id = instance_id.to_string();
        let conn = match self.conn().await {
            Ok(c) => c,
            Err(_) => return false,
        };
        let table = &self.table;

        tokio::task::block_in_place(|| {
            let sql = format!(
                "DELETE FROM {table} \
                 WHERE INSTANCE_ID = :1 \
                   AND STATUS IN ('Completed', 'Failed', 'Cancelled')"
            );
            match conn.execute(&sql, &[&id]) {
                Ok(stmt) => stmt.row_count().unwrap_or(0) > 0,
                Err(_) => false,
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn query_states<D: WorkflowData>(
    conn: &Connection,
    sql: &str,
    params: &[&dyn oracle::sql_type::ToSql],
) -> Result<Vec<WorkflowState<D>>, String> {
    let mut stmt = conn.statement(sql).build().map_err(map_oracle_error)?;
    let rows = stmt.query(params).map_err(map_oracle_error)?;
    let mut result = Vec::new();
    for row_result in rows {
        let row = row_result.map_err(map_oracle_error)?;
        let json: String = row.get("STATE_JSON").map_err(map_oracle_error)?;
        let state: WorkflowState<D> =
            serde_json::from_str(&json).map_err(|e| format!("Failed to deserialize state: {e}"))?;
        result.push(state);
    }
    Ok(result)
}

/// Convert an oracle error to a string, using "not_found" for ORA-01403.
fn ora_to_string(e: oracle::Error) -> String {
    if let Some(db_err) = e.db_error() {
        if db_err.code() == 1403 {
            return "not_found".to_string();
        }
    }
    map_oracle_error(e)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{WorkflowId, WorkflowStatus};

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct TestData {
        pub value: String,
        pub count: u32,
    }

    impl WorkflowData for TestData {
        fn workflow_type() -> &'static str {
            "test_workflow"
        }
    }

    #[test]
    fn test_config_defaults() {
        let config = OracleStateStoreConfig::default();
        assert_eq!(config.table_name_upper(), "WORKFLOW_STATES");
        assert_eq!(config.qualified_table(), "WORKFLOW_STATES");
        assert_eq!(config.pool_max, 8);
    }

    #[test]
    fn test_config_qualified_table_with_owner() {
        let config = OracleStateStoreConfig {
            owner: Some("ADMIN".to_string()),
            table_name: Some("WF_STATES".to_string()),
            ..Default::default()
        };
        assert_eq!(config.qualified_table(), "ADMIN.\"WF_STATES\"");
    }

    #[test]
    fn test_workflow_state_json_roundtrip() {
        let data = TestData {
            value: "hello".to_string(),
            count: 42,
        };
        let state =
            WorkflowState::new(WorkflowInstanceId::new(), WorkflowId::new("test_def"), data);

        let json = serde_json::to_string(&state).unwrap();
        let restored: WorkflowState<TestData> = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.instance_id, state.instance_id);
        assert_eq!(restored.context.data.value, "hello");
        assert_eq!(restored.context.data.count, 42);
        assert_eq!(restored.status, WorkflowStatus::Pending);
    }

    #[test]
    fn test_status_debug_format() {
        assert_eq!(format!("{:?}", WorkflowStatus::Pending), "Pending");
        assert_eq!(format!("{:?}", WorkflowStatus::Running), "Running");
        assert_eq!(format!("{:?}", WorkflowStatus::Completed), "Completed");
        assert_eq!(format!("{:?}", WorkflowStatus::Failed), "Failed");
        assert_eq!(format!("{:?}", WorkflowStatus::Cancelled), "Cancelled");
        assert_eq!(format!("{:?}", WorkflowStatus::Paused), "Paused");
    }

    #[test]
    fn test_context_extraction_from_json() {
        let data = TestData {
            value: "ctx_test".to_string(),
            count: 7,
        };
        let state =
            WorkflowState::new(WorkflowInstanceId::new(), WorkflowId::new("test_def"), data);

        let json = serde_json::to_string(&state).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let ctx_value = value.get("context").unwrap();
        let ctx: WorkflowContext<TestData> = serde_json::from_value(ctx_value.clone()).unwrap();

        assert_eq!(ctx.data.value, "ctx_test");
        assert_eq!(ctx.data.count, 7);
    }
}
