use std::{sync::Once, thread};

use tracing::info;
use tracing_subscriber::{
    filter::LevelFilter, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter,
};

use super::{crdt::CrdtOrMap, operation::OperationLog};
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

// ============================================================================
// Basic Functionality Tests
// ============================================================================

#[test]
fn test_basic_insert_and_get() {
    init_test_logging();
    let map = CrdtOrMap::new();

    // Insert data
    map.insert("key1".to_string(), b"value1".to_vec());
    map.insert("key2".to_string(), b"value2".to_vec());

    // Verify retrieval
    assert_eq!(map.get("key1"), Some(b"value1".to_vec()));
    assert_eq!(map.get("key2"), Some(b"value2".to_vec()));
    assert_eq!(map.get("key3"), None);
}

#[test]
fn test_basic_remove() {
    init_test_logging();
    let map = CrdtOrMap::new();

    map.insert("key1".to_string(), b"value1".to_vec());
    assert!(map.contains_key("key1"));

    map.remove("key1");
    assert!(!map.contains_key("key1"));
    assert_eq!(map.get("key1"), None);
}

#[test]
fn test_update_value() {
    init_test_logging();
    let map = CrdtOrMap::new();

    map.insert("key1".to_string(), b"value1".to_vec());
    assert_eq!(map.get("key1"), Some(b"value1".to_vec()));

    map.insert("key1".to_string(), b"value2".to_vec());
    assert_eq!(map.get("key1"), Some(b"value2".to_vec()));
}

// ============================================================================
// Concurrency Tests
// ============================================================================

#[test]
fn test_concurrent_inserts() {
    init_test_logging();
    let map = CrdtOrMap::new();
    let mut handles = vec![];

    // 10 threads inserting concurrently
    for i in 0..10 {
        let map_clone = map.clone();
        let handle = thread::spawn(move || {
            for j in 0..100 {
                let key = format!("key_{}_{}", i, j);
                let value = format!("value_{}_{}", i, j).into_bytes();
                map_clone.insert(key, value);
            }
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all data was inserted successfully
    for i in 0..10 {
        for j in 0..100 {
            let key = format!("key_{}_{}", i, j);
            assert!(map.contains_key(&key));
        }
    }
}

// ============================================================================
// CRDT Merge Tests
// ============================================================================

#[test]
fn test_merge_two_replicas() {
    init_test_logging();
    let replica1 = CrdtOrMap::new();
    let replica2 = CrdtOrMap::new();

    // Replica 1 inserts data
    replica1.insert("key1".to_string(), b"value1_from_r1".to_vec());
    replica1.insert("key2".to_string(), b"value2_from_r1".to_vec());

    // Replica 2 inserts data
    replica2.insert("key3".to_string(), b"value3_from_r2".to_vec());
    replica2.insert("key4".to_string(), b"value4_from_r2".to_vec());
    replica2.remove("key3");

    // Get replica 2's operation log and merge into replica 1
    let log2 = replica2.get_operation_log();

    info!(
        "Replica 1 merging Replica 2's log with \n====\n{:?}\n====",
        log2
    );
    replica1.merge(&log2);

    // Verify merged data
    assert_eq!(replica1.get("key1"), Some(b"value1_from_r1".to_vec()));
    assert_eq!(replica1.get("key2"), Some(b"value2_from_r1".to_vec()));
    // assert_eq!(replica1.get("key3"), Some(b"value3_from_r2".to_vec()));
    assert_eq!(replica1.get("key4"), Some(b"value4_from_r2".to_vec()));
}

#[test]
fn test_concurrent_insert_same_key() {
    init_test_logging();
    let replica1 = CrdtOrMap::new();
    let replica2 = CrdtOrMap::new();

    // Two replicas insert the same key concurrently
    replica1.insert("key1".to_string(), b"value_from_r1".to_vec());
    replica2.insert("key1".to_string(), b"value_from_r2".to_vec());

    // Get replica 2's log and merge
    let log2 = replica2.get_operation_log();
    info!(
        "Replica 1 merging Replica 2's log with \n====\n{:?}\n====",
        log2
    );
    replica1.merge(&log2);

    // Add-wins semantic: both values should be preserved, but only one is displayed
    // Due to timestamp and replica ID ordering, one will be selected
    info!("{:?}", String::from_utf8(replica1.get("key1").unwrap()));
    assert!(replica1.contains_key("key1"));
}

#[test]
fn test_remove_after_insert() {
    init_test_logging();
    let replica1 = CrdtOrMap::new();
    let replica2 = CrdtOrMap::new();

    // Replica 1 inserts
    replica1.insert("key1".to_string(), b"value1".to_vec());

    // Replica 2 also inserts the same key
    replica2.insert("key1".to_string(), b"value1".to_vec());

    // Replica 1 removes
    replica1.remove("key1");

    // Get replica 2's log and merge into replica 1
    let log2 = replica2.get_operation_log();
    replica1.merge(&log2);

    // Remove operation should win (because remove has newer timestamp)
    assert!(!replica1.contains_key("key1"));
}

// ============================================================================
// Serialization Tests
// ============================================================================

#[test]
fn test_operation_log_json_serialization() {
    init_test_logging();
    let map = CrdtOrMap::new();

    map.insert("key1".to_string(), b"value1".to_vec());
    map.insert("key2".to_string(), b"value2".to_vec());
    map.remove("key1");

    let log = map.get_operation_log();

    // Serialize to JSON
    let json = log.to_json().unwrap();
    println!("JSON: {}", json);

    // Deserialize
    let deserialized_log = OperationLog::from_json(&json).unwrap();
    assert_eq!(log.len(), deserialized_log.len());
}

#[test]
fn test_operation_log_binary_serialization() {
    init_test_logging();
    let map = CrdtOrMap::new();

    map.insert("key1".to_string(), b"value1".to_vec());
    map.insert("key2".to_string(), b"value2".to_vec());
    map.remove("key1");

    let log = map.get_operation_log();

    // Serialize to binary
    let bytes = log.to_bytes().unwrap();
    println!("Binary size: {} bytes", bytes.len());

    // Deserialize
    let deserialized_log = OperationLog::from_bytes(&bytes).unwrap();
    assert_eq!(log.len(), deserialized_log.len());
}

#[test]
fn test_apply_operation_log() {
    init_test_logging();
    let replica1 = CrdtOrMap::new();
    let replica2 = CrdtOrMap::new();

    // Replica 1 executes operations
    replica1.insert("key1".to_string(), b"value1".to_vec());
    replica1.insert("key2".to_string(), b"value2".to_vec());
    replica1.remove("key1");

    // Get operation log
    let log = replica1.get_operation_log();

    // Replica 2 merges operation log
    replica2.merge(&log);

    // Verify replica 2's state matches replica 1
    assert!(!replica2.contains_key("key1"));
    assert_eq!(replica2.get("key2"), Some(b"value2".to_vec()));
}

// ============================================================================
// Complex Scenario Tests
// ============================================================================

#[test]
fn test_distributed_scenario() {
    init_test_logging();
    // Simulate distributed scenario: 3 replicas operate independently then merge
    let replica1 = CrdtOrMap::new();
    let replica2 = CrdtOrMap::new();
    let replica3 = CrdtOrMap::new();

    // Replica 1 operations
    replica1.insert("user:1".to_string(), b"Alice".to_vec());
    replica1.insert("user:2".to_string(), b"Bob".to_vec());

    // Replica 2 operations
    replica2.insert("user:3".to_string(), b"Charlie".to_vec());
    replica2.insert("user:1".to_string(), b"Alice_Updated".to_vec());

    // Replica 3 operations
    replica3.insert("user:4".to_string(), b"David".to_vec());
    replica3.remove("user:2");

    // Merge all replicas into replica 1
    let log2 = replica2.get_operation_log();
    let log3 = replica3.get_operation_log();

    // Idempotent + unordered merge
    replica1.merge(&log3);
    replica1.merge(&log2);
    replica1.merge(&log3);

    // Verify final state
    assert!(replica1.contains_key("user:1")); // Exists (updated)
    assert!(!replica1.contains_key("user:2")); // Removed
    assert!(replica1.contains_key("user:3")); // Exists
    assert!(replica1.contains_key("user:4")); // Exists

    println!("Final state:");
    println!(
        "user:1 = {:?}",
        String::from_utf8(replica1.get("user:1").unwrap())
    );
    println!(
        "user:3 = {:?}",
        String::from_utf8(replica1.get("user:3").unwrap())
    );
    println!(
        "user:4 = {:?}",
        String::from_utf8(replica1.get("user:4").unwrap())
    );
}
