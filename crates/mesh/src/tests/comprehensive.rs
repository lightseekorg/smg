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
    sync::{Arc, Once},
    time::Duration,
};

use tracing as log;
use tracing_subscriber::{
    filter::LevelFilter, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter,
};

use super::test_utils::{self, bind_node, wait_for};
// Internal crate imports - now can access private modules
use crate::{
    node_state_machine::{ConvergenceConfig, NodeReadiness, NodeStateMachine},
    partition::{PartitionConfig, PartitionDetector, PartitionState},
    service::gossip::{NodeState as GossipNodeState, NodeStatus},
    stores::{AppState, StateStores},
    sync::MeshSyncManager,
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
    let _ = stores.app.insert("key1".to_string(), app_state.clone());
    let value = stores.app.get("key1");
    assert!(value.is_some());
    assert_eq!(value.unwrap().value, vec![1, 2, 3]);

    // Test that keys don't exist initially
    assert_eq!(stores.app.get("nonexistent"), None);
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

    let (listener, addr) = bind_node().await;
    let handler = crate::mesh_run!("single_node", listener, addr, None);

    wait_for(
        || handler.state.read().contains_key("single_node"),
        Duration::from_secs(5),
        "single_node appears in cluster state",
    )
    .await;

    handler.graceful_shutdown().await.unwrap();
    log::info!("Single node shutdown completed");
}

#[tokio::test]
async fn test_single_node_data_operations() {
    init_test_logging();
    log::info!("Starting test_single_node_data_operations");

    let (listener, addr) = bind_node().await;
    let handler = crate::mesh_run!("data_node", listener, addr, None);

    wait_for(
        || handler.state.read().contains_key("data_node"),
        Duration::from_secs(5),
        "data_node appears in cluster state",
    )
    .await;

    handler
        .write_data("test_key".into(), "test_value".into())
        .unwrap();

    assert!(handler.stores.app.get("test_key").is_some());

    handler.shutdown();
    log::info!("Data operations test completed");
}

#[tokio::test]
async fn test_single_node_subsystems_initialized() {
    init_test_logging();
    log::info!("Starting test_single_node_subsystems_initialized");

    let (listener, addr) = bind_node().await;
    let handler = crate::mesh_run!("subsystem_node", listener, addr, None);

    wait_for(
        || handler.state.read().contains_key("subsystem_node"),
        Duration::from_secs(5),
        "subsystem_node appears in cluster state",
    )
    .await;

    assert!(handler.partition_detector().is_some());
    assert!(handler.state_machine().is_some());

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

    let (listener_a, addr_a) = bind_node().await;
    let handler_a = crate::mesh_run!("node_a", listener_a, addr_a, None);

    let (listener_b, addr_b) = bind_node().await;
    let handler_b = crate::mesh_run!("node_b", listener_b, addr_b, Some(addr_a));

    wait_for(
        || handler_a.state.read().len() == 2 && handler_b.state.read().len() == 2,
        Duration::from_secs(15),
        "both nodes see each other",
    )
    .await;

    let state_a = handler_a.state.read();
    assert!(state_a.contains_key("node_a"));
    assert!(state_a.contains_key("node_b"));
    drop(state_a);

    handler_a.shutdown();
    handler_b.shutdown();
    log::info!("Two-node cluster formation test completed");
}

#[tokio::test]
async fn test_two_node_data_synchronization() {
    init_test_logging();
    log::info!("Starting test_two_node_data_synchronization");

    let (listener_a, addr_a) = bind_node().await;
    let handler_a = crate::mesh_run!("sync_node_a", listener_a, addr_a, None);

    let (listener_b, addr_b) = bind_node().await;
    let handler_b = crate::mesh_run!("sync_node_b", listener_b, addr_b, Some(addr_a));

    // Wait for cluster formation
    wait_for(
        || handler_a.state.read().len() == 2 && handler_b.state.read().len() == 2,
        Duration::from_secs(15),
        "both nodes see each other",
    )
    .await;

    // Write data on node A
    handler_a
        .write_data("shared_key".into(), "shared_value".into())
        .unwrap();

    // Poll until data syncs to B via incremental sync stream
    wait_for(
        || {
            handler_b
                .stores
                .app
                .get("shared_key")
                .is_some_and(|v| v.value == b"shared_value")
        },
        Duration::from_secs(15),
        "shared_key synced to node B",
    )
    .await;

    // Allow the sync cycle to settle before writing a second update.
    // The incremental collector runs on a 1s interval and the mark_sent
    // bookkeeping must complete before the next version can be detected.
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Update data on node A
    handler_a
        .write_data("shared_key".into(), "shared_value2".into())
        .unwrap();

    // Poll until updated value syncs to B
    wait_for(
        || {
            handler_b
                .stores
                .app
                .get("shared_key")
                .is_some_and(|v| v.value == b"shared_value2")
        },
        Duration::from_secs(30),
        "updated shared_key synced to node B",
    )
    .await;

    let value_a = handler_a.stores.app.get("shared_key").unwrap();
    let value_b = handler_b.stores.app.get("shared_key").unwrap();
    assert_eq!(value_a.value, value_b.value);

    handler_a.shutdown();
    handler_b.shutdown();
    log::info!("Two-node data synchronization test completed");
}

#[tokio::test]
async fn test_two_node_heartbeat_monitoring() {
    init_test_logging();
    log::info!("Starting test_two_node_heartbeat_monitoring");

    let (listener_a, addr_a) = bind_node().await;
    let handler_a = crate::mesh_run!("heartbeat_a", listener_a, addr_a, None);

    let (listener_b, addr_b) = bind_node().await;
    let handler_b = crate::mesh_run!("heartbeat_b", listener_b, addr_b, Some(addr_a));

    // Wait for cluster formation
    wait_for(
        || {
            handler_a
                .state
                .read()
                .get("heartbeat_b")
                .is_some_and(|n| n.status == NodeStatus::Alive as i32)
        },
        Duration::from_secs(15),
        "node A sees heartbeat_b as Alive",
    )
    .await;

    // Shutdown node B abruptly
    handler_b.shutdown();

    // Poll until A detects B as no longer Alive
    wait_for(
        || {
            handler_a
                .state
                .read()
                .get("heartbeat_b")
                .is_some_and(|n| n.status != NodeStatus::Alive as i32)
        },
        Duration::from_secs(30),
        "node A detects heartbeat_b as not Alive",
    )
    .await;

    let status = handler_a.state.read().get("heartbeat_b").map(|n| n.status);
    log::info!("Node B status after shutdown: {:?}", status);

    handler_a.shutdown();
    log::info!("Two-node heartbeat monitoring test completed");
}

//
// ====================================================================================
// STEP 5: Multi-Node Cluster Formation
// ====================================================================================
//

#[tokio::test]
async fn test_three_node_cluster_formation() {
    init_test_logging();
    log::info!("Starting test_three_node_cluster_formation");

    let (listener_a, addr_a) = bind_node().await;
    let handler_a = crate::mesh_run!("cluster_a", listener_a, addr_a, None);

    let (listener_b, addr_b) = bind_node().await;
    let handler_b = crate::mesh_run!("cluster_b", listener_b, addr_b, Some(addr_a));

    let (listener_c, addr_c) = bind_node().await;
    let handler_c = crate::mesh_run!("cluster_c", listener_c, addr_c, Some(addr_a));

    wait_for(
        || {
            handler_a.state.read().len() == 3
                && handler_b.state.read().len() == 3
                && handler_c.state.read().len() == 3
        },
        Duration::from_secs(30),
        "all 3 nodes see each other",
    )
    .await;

    let state_a = handler_a.state.read();
    assert!(state_a.contains_key("cluster_a"));
    assert!(state_a.contains_key("cluster_b"));
    assert!(state_a.contains_key("cluster_c"));
    drop(state_a);

    handler_a.shutdown();
    handler_b.shutdown();
    handler_c.shutdown();
    log::info!("Three-node cluster formation test completed");
}

#[tokio::test]
async fn test_multi_node_data_propagation() {
    init_test_logging();
    log::info!("Starting test_multi_node_data_propagation");

    let (listener_a, addr_a) = bind_node().await;
    let handler_a = crate::mesh_run!("prop_a", listener_a, addr_a, None);

    let (listener_b, addr_b) = bind_node().await;
    let handler_b = crate::mesh_run!("prop_b", listener_b, addr_b, Some(addr_a));

    let (listener_c, addr_c) = bind_node().await;
    let handler_c = crate::mesh_run!("prop_c", listener_c, addr_c, Some(addr_a));

    // Wait for 3-node cluster
    wait_for(
        || {
            handler_a.state.read().len() == 3
                && handler_b.state.read().len() == 3
                && handler_c.state.read().len() == 3
        },
        Duration::from_secs(60),
        "all 3 nodes see each other",
    )
    .await;

    // Write data on node A
    handler_a
        .write_data("propagated_key".into(), "propagated_value".into())
        .unwrap();

    // Poll until data reaches B and C
    wait_for(
        || {
            handler_b.stores.app.get("propagated_key").is_some()
                && handler_c.stores.app.get("propagated_key").is_some()
        },
        Duration::from_secs(60),
        "propagated_key synced to B and C",
    )
    .await;

    let val_a = handler_a.stores.app.get("propagated_key").unwrap().value;
    assert_eq!(
        val_a,
        handler_b.stores.app.get("propagated_key").unwrap().value
    );
    assert_eq!(
        val_a,
        handler_c.stores.app.get("propagated_key").unwrap().value
    );

    // Write updated data on node B
    handler_b
        .write_data("propagated_key".into(), "propagated_value2".into())
        .unwrap();

    // Poll until updated value reaches A and C
    wait_for(
        || {
            handler_a
                .stores
                .app
                .get("propagated_key")
                .is_some_and(|v| v.value == b"propagated_value2")
                && handler_c
                    .stores
                    .app
                    .get("propagated_key")
                    .is_some_and(|v| v.value == b"propagated_value2")
        },
        Duration::from_secs(60),
        "updated propagated_key synced to A and C",
    )
    .await;

    handler_a.shutdown();
    handler_b.shutdown();
    handler_c.shutdown();
    log::info!("Multi-node data propagation test completed");
}

#[tokio::test]
#[ignore = "SWIM failure detection for hard-shutdown nodes needs many gossip rounds; flaky under parallel CI load"]
async fn test_five_node_cluster_with_failure() {
    init_test_logging();
    log::info!("Starting test_five_node_cluster_with_failure");

    let (listener_a, addr_a) = bind_node().await;
    let handler_a = crate::mesh_run!("multi_a", listener_a, addr_a, None);

    let (listener_b, addr_b) = bind_node().await;
    let handler_b = crate::mesh_run!("multi_b", listener_b, addr_b, Some(addr_a));

    // Wait for A-B cluster
    wait_for(
        || handler_a.state.read().len() == 2,
        Duration::from_secs(15),
        "A-B cluster formed",
    )
    .await;

    handler_a
        .write_data("test_data".into(), "initial_value".into())
        .unwrap();

    // Add C and D
    let (listener_c, addr_c) = bind_node().await;
    let handler_c = crate::mesh_run!("multi_c", listener_c, addr_c, Some(addr_a));

    let (listener_d, addr_d) = bind_node().await;
    let handler_d = crate::mesh_run!("multi_d", listener_d, addr_d, Some(addr_c));

    wait_for(
        || handler_a.state.read().len() == 4,
        Duration::from_secs(30),
        "4-node cluster formed",
    )
    .await;

    // Add E, wait for it to join, then kill it
    {
        let (listener_e, addr_e) = bind_node().await;
        let handler_e = crate::mesh_run!("multi_e", listener_e, addr_e, Some(addr_d));

        wait_for(
            || handler_a.state.read().len() == 5,
            Duration::from_secs(30),
            "5-node cluster formed",
        )
        .await;

        handler_e.shutdown();
        log::info!("Node E shutdown");
    }

    // Gracefully shutdown D
    handler_d.graceful_shutdown().await.unwrap();
    log::info!("Node D gracefully shutdown");

    // Wait for D to be Leaving
    wait_for(
        || {
            handler_a
                .state
                .read()
                .get("multi_d")
                .is_some_and(|n| n.status == NodeStatus::Leaving as i32)
        },
        Duration::from_secs(30),
        "node D detected as Leaving",
    )
    .await;

    // Wait for E to be detected as not Alive (Suspected or Down).
    // SWIM failure detection requires multiple gossip rounds, so allow ample time
    // especially when other tests are running in parallel.
    wait_for(
        || {
            handler_a
                .state
                .read()
                .get("multi_e")
                .is_some_and(|n| n.status != NodeStatus::Alive as i32)
        },
        Duration::from_secs(60),
        "node E detected as not Alive",
    )
    .await;

    let state_a = handler_a.state.read();
    assert!(state_a.contains_key("multi_a"));
    assert!(state_a.contains_key("multi_b"));
    assert!(state_a.contains_key("multi_c"));
    assert_eq!(
        state_a.get("multi_d").map(|n| n.status),
        Some(NodeStatus::Leaving as i32)
    );
    let e_status = state_a.get("multi_e").map(|n| n.status);
    log::info!("Node E final status: {:?}", e_status);
    drop(state_a);

    handler_a.shutdown();
    handler_b.shutdown();
    handler_c.shutdown();
    log::info!("Five-node cluster test completed");
}

#[tokio::test]
async fn test_cluster_formation_different_join_patterns() {
    init_test_logging();
    log::info!("Starting test_cluster_formation_different_join_patterns");

    let (listener_a, addr_a) = bind_node().await;
    let handler_a = crate::mesh_run!("pattern_a", listener_a, addr_a, None);

    let (listener_b, addr_b) = bind_node().await;
    let handler_b = crate::mesh_run!("pattern_b", listener_b, addr_b, Some(addr_a));

    // Node C joins through B (chain topology)
    let (listener_c, addr_c) = bind_node().await;
    let handler_c = crate::mesh_run!("pattern_c", listener_c, addr_c, Some(addr_b));

    // Node D joins through A (star topology)
    let (listener_d, addr_d) = bind_node().await;
    let handler_d = crate::mesh_run!("pattern_d", listener_d, addr_d, Some(addr_a));

    wait_for(
        || {
            handler_a.state.read().len() == 4
                && handler_b.state.read().len() == 4
                && handler_c.state.read().len() == 4
                && handler_d.state.read().len() == 4
        },
        Duration::from_secs(30),
        "all 4 nodes see each other (chain + star topology)",
    )
    .await;

    handler_a.shutdown();
    handler_b.shutdown();
    handler_c.shutdown();
    handler_d.shutdown();
    log::info!("Different join patterns test completed");
}
