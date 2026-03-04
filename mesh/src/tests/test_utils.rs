//! Test utilities for mesh module
use std::{
    collections::{BTreeMap, HashMap},
    net::SocketAddr,
    sync::Arc,
    time::Duration,
};

use parking_lot::RwLock;
use tokio::net::TcpListener;

use crate::{
    service::{gossip::NodeState, ClusterState},
    stores::StateStores,
    sync::MeshSyncManager,
};

/// Bind to an ephemeral port and return the listener + address.
/// The caller must keep the listener alive and pass it to `mesh_run!`
/// to avoid a TOCTOU port race.
pub async fn bind_node() -> (TcpListener, SocketAddr) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tracing::debug!("Bound node to {}", addr);
    (listener, addr)
}

/// Poll `condition` every 100ms until it returns true or `timeout` expires.
pub async fn wait_for<F>(condition: F, timeout: Duration, msg: &str)
where
    F: Fn() -> bool,
{
    let start = std::time::Instant::now();
    while !condition() {
        assert!(
            start.elapsed() <= timeout,
            "Timeout after {timeout:?} waiting for: {msg}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Create test StateStores with a given node name
pub fn create_test_stores(self_name: String) -> Arc<StateStores> {
    Arc::new(StateStores::with_self_name(self_name))
}

/// Create test MeshSyncManager
pub fn create_test_sync_manager(self_name: String) -> Arc<MeshSyncManager> {
    let stores = create_test_stores(self_name.clone());
    Arc::new(MeshSyncManager::new(stores, self_name))
}

/// Create test cluster state with given nodes
pub fn create_test_cluster_state(
    nodes: Vec<(String, String, i32)>, // (name, address, status)
) -> ClusterState {
    let mut state = BTreeMap::new();
    for (name, address, status) in nodes {
        state.insert(
            name.clone(),
            NodeState {
                name: name.clone(),
                address,
                status,
                version: 1,
                metadata: HashMap::new(),
            },
        );
    }
    Arc::new(RwLock::new(state))
}

#[cfg(test)]
mod test_utils_tests {
    use super::*;

    #[test]
    fn test_create_test_stores() {
        let stores = create_test_stores("test_node".to_string());
        assert!(!stores.rate_limit.is_owner("key1"));
    }

    #[test]
    fn test_create_test_sync_manager() {
        let manager = create_test_sync_manager("test_node".to_string());
        assert_eq!(manager.self_name(), "test_node");
    }

    #[test]
    fn test_create_test_cluster_state() {
        let state = create_test_cluster_state(vec![
            ("node1".to_string(), "127.0.0.1:8000".to_string(), 1),
            ("node2".to_string(), "127.0.0.1:8001".to_string(), 1),
        ]);
        let read_state = state.read();
        assert_eq!(read_state.len(), 2);
        assert!(read_state.contains_key("node1"));
        assert!(read_state.contains_key("node2"));
    }
}
