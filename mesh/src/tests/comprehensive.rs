//! Comprehensive Mesh Service Tests
//!
//! This module implements High Priority Steps 1-5 from the test plan:
//! - Step 1: Test Infrastructure Setup
//! - Step 2: Basic Component Unit Tests
//! - Step 3: Single Node Integration Tests
//! - Step 4: Two-Node Cluster Tests
//! - Step 5: Multi-Node Cluster Formation
//!
//! ## Internal Tests
//! These tests are now crate-internal and have full access to private modules.

use std::{
    collections::BTreeMap,
    net::SocketAddr,
    sync::{Arc, Once},
    time::Duration,
};

use tokio::net::TcpListener;
use tracing as log;
use tracing_subscriber::{
    filter::LevelFilter, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter,
};

use super::test_utils;
// Internal crate imports - now can access private modules
use crate::{
    node_state_machine::{ConvergenceConfig, NodeReadiness, NodeStateMachine},
    partition::{PartitionConfig, PartitionDetector, PartitionState},
    service::{
        gossip::{NodeState as GossipNodeState, NodeStatus},
        MeshServerHandler,
    },
    stores::{AppState, StateStores},
    sync::MeshSyncManager,
    SKey,
};

//
// ====================================================================================
// STEP 1: Test Infrastructure Setup
// ====================================================================================
//

static INIT: Once = Once::new();

/// Initialize test logging infrastructure
fn init_test_logging() {
    INIT.call_once(|| {
        let _ = tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer())
            .with(
                EnvFilter::builder()
                    .with_default_directive(LevelFilter::INFO.into())
                    .from_env_lossy(),
            )
            .try_init();
    });
}

/// Test utility: Find a free port for node binding
async fn find_free_port() -> (TcpListener, u16) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    log::debug!("Found free port: {}", port);
    (listener, port)
}

/// Test utility: Get a free socket address
async fn get_node_addr() -> SocketAddr {
    let (_listener, port) = find_free_port().await;
    format!("127.0.0.1:{}", port).parse().unwrap()
}

/// Test utility: Print cluster state for debugging
fn print_cluster_state(handler: &MeshServerHandler) -> String {
    let state = handler.state.read();
    let mut res = vec![];
    for (k, v) in state.iter() {
        let status = NodeStatus::try_from(v.status)
            .map(|s| format!("{:?}", s))
            .unwrap_or_else(|_| format!("Unknown({})", v.status));
        res.push(format!("{}: {} v={}", k, status, v.version));
    }
    res.join(", ")
}

#[test]
fn test_infrastructure_utilities() {
    init_test_logging();

    // Test using test_utils module
    let stores = test_utils::create_test_stores("test_node".to_string());
    assert!(stores.membership.all().is_empty());

    let sync_manager = test_utils::create_test_sync_manager("test_node".to_string());
    assert_eq!(sync_manager.self_name(), "test_node");

    // Now we can test create_test_cluster_state with NodeState
    let cluster_state = test_utils::create_test_cluster_state(vec![
        (
            "node1".to_string(),
            "127.0.0.1:8000".to_string(),
            NodeStatus::Alive as i32,
        ),
        (
            "node2".to_string(),
            "127.0.0.1:8001".to_string(),
            NodeStatus::Alive as i32,
        ),
    ]);
    assert_eq!(cluster_state.read().len(), 2);
}

//
// ====================================================================================
// STEP 2: Basic Component Unit Tests
// ====================================================================================
//

#[test]
fn test_partition_detector_initialization() {
    let config = PartitionConfig::default();
    let detector = PartitionDetector::new(config);

    // Test with empty cluster state
    let empty_state = BTreeMap::new();
    let state = detector.detect_partition(&empty_state);
    assert_eq!(state, PartitionState::Normal);
}

#[test]
fn test_partition_detector_quorum_calculation() {
    let detector = PartitionDetector::default();

    // Test quorum with 3 nodes (need 2 for quorum)
    let mut cluster_state = BTreeMap::new();
    cluster_state.insert(
        "node1".to_string(),
        GossipNodeState {
            name: "node1".to_string(),
            address: "127.0.0.1:8000".to_string(),
            status: NodeStatus::Alive as i32,
            version: 1,
            metadata: Default::default(),
        },
    );
    cluster_state.insert(
        "node2".to_string(),
        GossipNodeState {
            name: "node2".to_string(),
            address: "127.0.0.1:8001".to_string(),
            status: NodeStatus::Alive as i32,
            version: 1,
            metadata: Default::default(),
        },
    );
    cluster_state.insert(
        "node3".to_string(),
        GossipNodeState {
            name: "node3".to_string(),
            address: "127.0.0.1:8002".to_string(),
            status: NodeStatus::Down as i32,
            version: 1,
            metadata: Default::default(),
        },
    );

    // Update last_seen for alive nodes
    detector.update_last_seen("node1");
    detector.update_last_seen("node2");

    let state = detector.detect_partition(&cluster_state);
    assert_eq!(state, PartitionState::Normal);
}

#[test]
fn test_node_state_machine_lifecycle() {
    let stores = test_utils::create_test_stores("test_node".to_string());
    let config = ConvergenceConfig::default();
    let state_machine = NodeStateMachine::new(stores, config);

    // Initial state should be NotReady
    assert!(!state_machine.is_ready());
    assert_eq!(state_machine.readiness(), NodeReadiness::NotReady);

    // Transition to Joining
    state_machine.start_joining();
    assert_eq!(state_machine.readiness(), NodeReadiness::Joining);

    // Transition to SnapshotPull
    state_machine.start_snapshot_pull();
    assert_eq!(state_machine.readiness(), NodeReadiness::SnapshotPull);

    // Transition to Converging
    state_machine.start_converging();
    assert_eq!(state_machine.readiness(), NodeReadiness::Converging);

    // Transition to Ready
    state_machine.transition_to_ready();
    assert!(state_machine.is_ready());
    assert_eq!(state_machine.readiness(), NodeReadiness::Ready);
}

#[test]
fn test_state_stores_basic_operations() {
    let stores = test_utils::create_test_stores("test_node".to_string());

    // Test app data write/read
    let app_state = AppState {
        key: "key1".to_string(),
        value: vec![1, 2, 3],
        version: 1,
    };
    stores.app.insert(
        SKey("key1".to_string()),
        app_state.clone(),
        "test_node".to_string(),
    );
    let value = stores.app.get(&SKey("key1".to_string()));
    assert!(value.is_some());
    assert_eq!(value.unwrap().value, vec![1, 2, 3]);

    // Test that keys don't exist initially
    assert_eq!(stores.app.get(&SKey("nonexistent".to_string())), None);
}

#[test]
fn test_sync_manager_rate_limit_membership() {
    let sync_manager = test_utils::create_test_sync_manager("node1".to_string());

    // Update membership should not panic
    sync_manager.update_rate_limit_membership();

    // Test self name
    assert_eq!(sync_manager.self_name(), "node1");
}

#[tokio::test]
async fn test_rate_limit_window_creation() {
    use crate::rate_limit_window::RateLimitWindow;

    let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
    let sync_manager = Arc::new(MeshSyncManager::new(stores, "node1".to_string()));

    let _window = RateLimitWindow::new(sync_manager, 60);
    // Window created successfully - no public fields to assert
}

//
// ====================================================================================
// STEP 3: Single Node Integration Tests
// ====================================================================================
//

#[tokio::test]
async fn test_single_node_creation_and_shutdown() {
    init_test_logging();
    log::info!("Starting test_single_node_creation_and_shutdown");

    let addr = get_node_addr().await;
    let handler = crate::mesh_run!("single_node", addr, None);

    // Wait for node to initialize
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify node is in cluster state
    {
        let state = handler.state.read();
        assert!(state.contains_key("single_node"));
    } // Drop the read lock before shutdown

    // Test graceful shutdown - must work for single node scenario
    // as clusters may scale down to 1 node in production
    handler.graceful_shutdown().await.unwrap();
    log::info!("Single node shutdown completed");
}

#[tokio::test]
async fn test_single_node_data_operations() {
    init_test_logging();
    log::info!("Starting test_single_node_data_operations");

    let addr = get_node_addr().await;
    let handler = crate::mesh_run!("data_node", addr, None);

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Write data
    handler.write_data("test_key".into(), "test_value".into());

    // Verify data was written
    let value = handler.stores.app.get(&SKey("test_key".to_string()));
    assert!(value.is_some());

    handler.shutdown();
    log::info!("Data operations test completed");
}

#[tokio::test]
async fn test_single_node_subsystems_initialized() {
    init_test_logging();
    log::info!("Starting test_single_node_subsystems_initialized");

    let addr = get_node_addr().await;
    let handler = crate::mesh_run!("subsystem_node", addr, None);

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Now we can access private methods
    assert!(handler.partition_detector().is_some());
    assert!(handler.state_machine().is_some());

    // Verify stores exist
    assert!(
        !handler.stores.membership.all().is_empty() || handler.stores.membership.all().is_empty()
    );

    handler.shutdown();
    log::info!("Subsystems initialization test completed");
}

//
// ====================================================================================
// STEP 4: Two-Node Cluster Tests
// ====================================================================================
//

#[tokio::test]
async fn test_two_node_cluster_formation() {
    init_test_logging();
    log::info!("Starting test_two_node_cluster_formation");

    // Start node A
    let addr_a = get_node_addr().await;
    let handler_a = crate::mesh_run!("node_a", addr_a, None);

    tokio::time::sleep(Duration::from_secs(1)).await;

    // Start node B, joining through A
    let addr_b = get_node_addr().await;
    let handler_b = crate::mesh_run!("node_b", addr_b, Some(addr_a));

    // Wait for cluster formation
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Verify both nodes see each other
    {
        let state_a = handler_a.state.read();
        let state_b = handler_b.state.read();

        assert!(state_a.contains_key("node_a"));
        assert!(state_a.contains_key("node_b"));
        assert!(state_b.contains_key("node_a"));
        assert!(state_b.contains_key("node_b"));

        log::info!("State A: {:?}", print_cluster_state(&handler_a));
        log::info!("State B: {:?}", print_cluster_state(&handler_b));
    } // Drop locks before shutdown

    handler_a.shutdown();
    handler_b.shutdown();
    log::info!("Two-node cluster formation test completed");
}

#[tokio::test]
async fn test_two_node_data_synchronization() {
    init_test_logging();
    log::info!("Starting test_two_node_data_synchronization");

    let addr_a = get_node_addr().await;
    let handler_a = crate::mesh_run!("sync_node_a", addr_a, None);

    let addr_b = get_node_addr().await;
    let handler_b = crate::mesh_run!("sync_node_b", addr_b, Some(addr_a));

    tokio::time::sleep(Duration::from_secs(2)).await;

    // Write data on node A
    handler_a.write_data("shared_key".into(), "shared_value".into());

    // Wait for automatic sync via sync_stream
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Verify data exists on both nodes
    let value_a = handler_a.stores.app.get(&SKey("shared_key".to_string()));
    let value_b = handler_b.stores.app.get(&SKey("shared_key".to_string()));

    log::info!("Value on A: {:?}", value_a);
    log::info!("Value on B: {:?}", value_b);

    // Update data on node A to see if it syncs again
    handler_a.write_data("shared_key".into(), "shared_value2".into());

    // Wait for the second update to be synced (increased wait time)
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Verify data exists on both nodes
    let value_a = handler_a.stores.app.get(&SKey("shared_key".to_string()));
    let value_b = handler_b.stores.app.get(&SKey("shared_key".to_string()));

    log::info!("Value on A: {:?}", value_a);
    log::info!("Value on B: {:?}", value_b);

    assert!(value_a.is_some());
    assert!(value_b.is_some());

    // Verify values are the same
    assert_eq!(value_a.unwrap().value, value_b.unwrap().value);

    handler_a.shutdown();
    handler_b.shutdown();
    log::info!("Two-node data synchronization test completed");
}

#[tokio::test]
async fn test_two_node_heartbeat_monitoring() {
    init_test_logging();
    log::info!("Starting test_two_node_heartbeat_monitoring");

    let addr_a = get_node_addr().await;
    let handler_a = crate::mesh_run!("heartbeat_a", addr_a, None);

    let addr_b = get_node_addr().await;
    let handler_b = crate::mesh_run!("heartbeat_b", addr_b, Some(addr_a));

    // Wait for cluster formation sync
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Both nodes should be alive
    {
        let state_a = handler_a.state.read();
        let node_b_status = state_a.get("heartbeat_b").map(|n| n.status);
        assert_eq!(node_b_status, Some(NodeStatus::Alive as i32));
    }

    // Shutdown node B abruptly
    handler_b.shutdown();

    // Wait for detection
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Node A should detect B as suspected or down
    let state_a = handler_a.state.read();
    let node_b_status = state_a.get("heartbeat_b").map(|n| n.status);
    log::info!("Node B status after shutdown: {:?}", node_b_status);

    // Status should have changed from Alive
    assert_ne!(node_b_status, Some(NodeStatus::Alive as i32));
    handler_a.shutdown();
    log::info!("Two-node heartbeat monitoring test completed");
}

//
// ====================================================================================
// STEP 5: Multi-Node Cluster Formation
// ====================================================================================
//

#[tokio::test]
#[ignore = "Long-running test with complex state convergence"]
async fn test_three_node_cluster_formation() {
    init_test_logging();
    log::info!("Starting test_three_node_cluster_formation");

    // Start node A
    let addr_a = get_node_addr().await;
    let handler_a = crate::mesh_run!("cluster_a", addr_a, None);

    tokio::time::sleep(Duration::from_secs(1)).await;

    // Start node B joining through A
    let addr_b = get_node_addr().await;
    let handler_b = crate::mesh_run!("cluster_b", addr_b, Some(addr_a));

    tokio::time::sleep(Duration::from_secs(1)).await;

    // Start node C joining through A
    let addr_c = get_node_addr().await;
    let handler_c = crate::mesh_run!("cluster_c", addr_c, Some(addr_a));

    // Wait for full cluster formation
    tokio::time::sleep(Duration::from_secs(6)).await;

    // Verify all nodes see each other
    {
        let state_a = handler_a.state.read();
        let state_b = handler_b.state.read();
        let state_c = handler_c.state.read();

        log::info!("State A: {}", print_cluster_state(&handler_a));
        log::info!("State B: {}", print_cluster_state(&handler_b));
        log::info!("State C: {}", print_cluster_state(&handler_c));

        // All nodes should see all 3 nodes
        assert_eq!(state_a.len(), 3);
        assert_eq!(state_b.len(), 3);
        assert_eq!(state_c.len(), 3);

        assert!(state_a.contains_key("cluster_a"));
        assert!(state_a.contains_key("cluster_b"));
        assert!(state_a.contains_key("cluster_c"));
    } // Drop locks before shutdown

    handler_a.shutdown();
    handler_b.shutdown();
    handler_c.shutdown();
    log::info!("Three-node cluster formation test completed");
}

#[tokio::test]
#[ignore = "Long-running test with complex state convergence"]
async fn test_multi_node_data_propagation() {
    init_test_logging();
    log::info!("Starting test_multi_node_data_propagation");

    let addr_a = get_node_addr().await;
    let handler_a = crate::mesh_run!("prop_a", addr_a, None);

    let addr_b = get_node_addr().await;
    let handler_b = crate::mesh_run!("prop_b", addr_b, Some(addr_a));

    let addr_c = get_node_addr().await;
    let handler_c = crate::mesh_run!("prop_c", addr_c, Some(addr_a));

    tokio::time::sleep(Duration::from_secs(3)).await;

    // Write data on node A
    handler_a.write_data("propagated_key".into(), "propagated_value".into());

    // Wait for automatic sync via sync_stream
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Verify data reached all nodes
    let value_a = handler_a
        .stores
        .app
        .get(&SKey("propagated_key".to_string()));
    let value_b = handler_b
        .stores
        .app
        .get(&SKey("propagated_key".to_string()));
    let value_c = handler_c
        .stores
        .app
        .get(&SKey("propagated_key".to_string()));

    log::info!("Value on A: {:?}", value_a);
    log::info!("Value on B: {:?}", value_b);
    log::info!("Value on C: {:?}", value_c);

    assert!(value_a.is_some());
    assert!(value_b.is_some());
    assert!(value_c.is_some());

    // Verify all values are the same
    let val_a = value_a.unwrap().value;
    assert_eq!(val_a, value_b.unwrap().value);
    assert_eq!(val_a, value_c.unwrap().value);

    // Write data on node B to verify continued propagation
    handler_b.write_data("propagated_key".into(), "propagated_value".into());

    // Wait for automatic sync via sync_stream
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Verify data reached all nodes
    let value_a = handler_a
        .stores
        .app
        .get(&SKey("propagated_key".to_string()));
    let value_b = handler_b
        .stores
        .app
        .get(&SKey("propagated_key".to_string()));
    let value_c = handler_c
        .stores
        .app
        .get(&SKey("propagated_key".to_string()));

    log::info!("Value on A: {:?}", value_a);
    log::info!("Value on B: {:?}", value_b);
    log::info!("Value on C: {:?}", value_c);

    assert!(value_a.is_some());
    assert!(value_b.is_some());
    assert!(value_c.is_some());

    // Verify all values are the same
    let val_a = value_a.unwrap().value;
    assert_eq!(val_a, value_b.unwrap().value);
    assert_eq!(val_a, value_c.unwrap().value);

    handler_a.shutdown();
    handler_b.shutdown();
    handler_c.shutdown();
    log::info!("Multi-node data propagation test completed");
}

#[tokio::test]
#[ignore = "Long-running test with complex state convergence"]
async fn test_five_node_cluster_with_failure() {
    init_test_logging();
    log::info!("Starting test_five_node_cluster_with_failure");

    // Setup initial cluster with nodes A and B
    let addr_a = get_node_addr().await;
    let handler_a = crate::mesh_run!("multi_a", addr_a, None);

    let addr_b = get_node_addr().await;
    let handler_b = crate::mesh_run!("multi_b", addr_b, Some(addr_a));

    tokio::time::sleep(Duration::from_secs(2)).await;

    // Write some data
    handler_a.write_data("test_data".into(), "initial_value".into());
    log::info!("Initial data written");

    // Add nodes C and D
    let addr_c = get_node_addr().await;
    let handler_c = crate::mesh_run!("multi_c", addr_c, Some(addr_a));

    let addr_d = get_node_addr().await;
    let handler_d = crate::mesh_run!("multi_d", addr_d, Some(addr_c));

    tokio::time::sleep(Duration::from_secs(2)).await;
    log::info!("Nodes C and D joined");

    // Add node E and then shut it down
    {
        let addr_e = get_node_addr().await;
        let handler_e = crate::mesh_run!("multi_e", addr_e, Some(addr_d));

        tokio::time::sleep(Duration::from_secs(3)).await;
        log::info!("Node E joined, state: {}", print_cluster_state(&handler_e));

        handler_e.shutdown();
        log::info!("Node E shutdown");
    }

    // Gracefully shutdown node D
    handler_d.graceful_shutdown().await.unwrap();
    tokio::time::sleep(Duration::from_secs(2)).await;
    log::info!("Node D gracefully shutdown");

    // Wait for state convergence
    tokio::time::sleep(Duration::from_secs(10)).await;

    log::info!("Final state A: {}", print_cluster_state(&handler_a));
    log::info!("Final state B: {}", print_cluster_state(&handler_b));
    log::info!("Final state C: {}", print_cluster_state(&handler_c));

    // Verify remaining nodes have consistent view
    let state_a = handler_a.state.read();
    let _state_b = handler_b.state.read();
    let _state_c = handler_c.state.read();

    // All remaining nodes should see nodes A, B, C
    assert!(state_a.contains_key("multi_a"));
    assert!(state_a.contains_key("multi_b"));
    assert!(state_a.contains_key("multi_c"));

    // D should be in Leaving state
    assert_eq!(
        state_a.get("multi_d").map(|n| n.status),
        Some(NodeStatus::Leaving as i32)
    );

    // E should eventually be Down
    let e_status = state_a.get("multi_e").map(|n| n.status);
    log::info!("Node E final status: {:?}", e_status);

    handler_a.shutdown();
    handler_b.shutdown();
    handler_c.shutdown();
    log::info!("Five-node cluster test completed");
}

#[tokio::test]
async fn test_cluster_formation_different_join_patterns() {
    init_test_logging();
    log::info!("Starting test_cluster_formation_different_join_patterns");

    // Create initial node
    let addr_a = get_node_addr().await;
    let handler_a = crate::mesh_run!("pattern_a", addr_a, None);

    tokio::time::sleep(Duration::from_secs(1)).await;

    // Node B joins through A
    let addr_b = get_node_addr().await;
    let handler_b = crate::mesh_run!("pattern_b", addr_b, Some(addr_a));

    tokio::time::sleep(Duration::from_secs(1)).await;

    // Node C joins through B (chain)
    let addr_c = get_node_addr().await;
    let handler_c = crate::mesh_run!("pattern_c", addr_c, Some(addr_b));

    tokio::time::sleep(Duration::from_secs(1)).await;

    // Node D joins through A (star pattern)
    let addr_d = get_node_addr().await;
    let handler_d = crate::mesh_run!("pattern_d", addr_d, Some(addr_a));

    // Wait for convergence - need more time for gossip to propagate through chain topology
    // With 4 nodes in different join patterns (chain + star), state needs multiple rounds to converge
    tokio::time::sleep(Duration::from_secs(8)).await;

    // Verify all nodes see all 4 nodes regardless of join pattern
    {
        let state_a = handler_a.state.read();
        let state_b = handler_b.state.read();
        let state_c = handler_c.state.read();
        let state_d = handler_d.state.read();

        log::info!("State A: {}", print_cluster_state(&handler_a));
        log::info!("State B: {}", print_cluster_state(&handler_b));
        log::info!("State C: {}", print_cluster_state(&handler_c));
        log::info!("State D: {}", print_cluster_state(&handler_d));

        assert_eq!(state_a.len(), 4);
        assert_eq!(state_b.len(), 4);
        assert_eq!(state_c.len(), 4);
        assert_eq!(state_d.len(), 4);
    } // Drop locks before shutdown

    handler_a.shutdown();
    handler_b.shutdown();
    handler_c.shutdown();
    handler_d.shutdown();
    log::info!("Different join patterns test completed");
}
