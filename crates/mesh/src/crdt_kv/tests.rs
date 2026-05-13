use std::{sync::Once, thread, time::Duration};

use tracing::info;
use tracing_subscriber::{
    filter::LevelFilter, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter,
};

use super::{
    crdt::CrdtOrMap,
    epoch_max_wins::{decode, encode, EpochCount},
    merge_strategy::MergeStrategy,
    operation::{Operation, OperationLog},
    replica::ReplicaId,
};
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
                let key = format!("key_{i}_{j}");
                let value = format!("value_{i}_{j}").into_bytes();
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
            let key = format!("key_{i}_{j}");
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
    assert_eq!(replica1.get("key3"), None);
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

    // LWW semantic: conflicts resolve by (timestamp, replica_id), so one value wins.
    // The winner displayed here is deterministic under that ordering.
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

#[test]
fn test_older_insert_applied_later_does_not_overwrite_winner() {
    init_test_logging();
    let source = CrdtOrMap::new();

    source.insert("key1".to_string(), b"older_value".to_vec());
    source.insert("key1".to_string(), b"newer_value".to_vec());

    let full_log = source.get_operation_log();
    let stale_insert = full_log
        .operations()
        .iter()
        .find_map(|op| match op {
            Operation::Insert { value, .. } if value.as_slice() == b"older_value" => {
                Some(op.clone())
            }
            _ => None,
        })
        .unwrap();

    let replica = CrdtOrMap::new();
    replica.merge(&full_log);
    assert_eq!(replica.get("key1"), Some(b"newer_value".to_vec()));

    let mut stale_log = OperationLog::new();
    stale_log.append(stale_insert);
    replica.merge(&stale_log);

    assert_eq!(replica.get("key1"), Some(b"newer_value".to_vec()));
}

#[test]
fn test_epoch_max_wins_compaction_uses_value_epoch() {
    init_test_logging();
    let replica = CrdtOrMap::new();
    replica.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);

    let key = "rl:global:node-a";
    let older_reset =
        Operation::insert(key.to_string(), encode(6, 0).to_vec(), 1, ReplicaId::new());
    let newer_stale_count = Operation::insert(
        key.to_string(),
        encode(5, 100).to_vec(),
        2,
        ReplicaId::new(),
    );

    let mut log = OperationLog::new();
    log.append(newer_stale_count);
    log.append(older_reset);

    replica.merge(&log);

    let value = replica.get(key).expect("rate-limit shard should exist");
    assert_eq!(decode(&value), Some(EpochCount { epoch: 6, count: 0 }));
}

#[test]
fn test_epoch_max_wins_preserves_newer_tombstone() {
    init_test_logging();
    let replica = CrdtOrMap::new();
    replica.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);

    let key = "rl:global:dead-node";
    let stale_insert =
        Operation::insert(key.to_string(), encode(6, 50).to_vec(), 1, ReplicaId::new());
    let tombstone = Operation::remove(key.to_string(), 2, ReplicaId::new());

    let mut log = OperationLog::new();
    log.append(stale_insert);
    log.append(tombstone);

    replica.merge(&log);

    assert_eq!(replica.get(key), None);
}

#[test]
fn test_epoch_max_wins_local_write_cannot_rewind_epoch() {
    init_test_logging();
    let replica = CrdtOrMap::new();
    replica.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);

    let key = "rl:global:node-a";
    replica.insert(key.to_string(), encode(6, 0).to_vec());
    replica.insert(key.to_string(), encode(5, 100).to_vec());

    let value = replica.get(key).expect("rate-limit shard should exist");
    assert_eq!(decode(&value), Some(EpochCount { epoch: 6, count: 0 }));
}

#[test]
fn test_epoch_max_wins_tombstone_compares_against_newest_live_version() {
    init_test_logging();
    let replica = CrdtOrMap::new();
    replica.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);

    let key = "rl:global:node-a";
    let stale_newer_timestamp = Operation::insert(
        key.to_string(),
        encode(5, 100).to_vec(),
        100,
        ReplicaId::new(),
    );
    let epoch_winner_older_timestamp =
        Operation::insert(key.to_string(), encode(6, 0).to_vec(), 90, ReplicaId::new());
    let tombstone_after_epoch_winner = Operation::remove(key.to_string(), 95, ReplicaId::new());

    let mut stale_log = OperationLog::new();
    stale_log.append(stale_newer_timestamp);
    replica.merge(&stale_log);

    let mut reset_log = OperationLog::new();
    reset_log.append(epoch_winner_older_timestamp);
    replica.merge(&reset_log);
    assert_eq!(
        decode(&replica.get(key).expect("reset should win stale count")),
        Some(EpochCount { epoch: 6, count: 0 }),
    );

    let mut tombstone_log = OperationLog::new();
    tombstone_log.append(tombstone_after_epoch_winner);
    replica.merge(&tombstone_log);

    assert_eq!(
        decode(
            &replica
                .get(key)
                .expect("newer live version suppresses tombstone")
        ),
        Some(EpochCount { epoch: 6, count: 0 }),
    );
}

#[test]
fn test_epoch_max_wins_snapshot_only_propagation_preserves_tombstone_boundary() {
    // Snapshot-only path: the source replica compacts its log so a
    // peer receives just one Insert per key (with the shard's
    // `tombstone_version` embedded), never the original Remove op.
    // A late peer that still holds the pre-tombstone high-epoch
    // insert must not be able to resurrect it.
    init_test_logging();
    let key = "rl:global:node-a";

    // Source: pre-tombstone high-epoch insert, then tombstone, then
    // post-tombstone lower-epoch insert. After merge+compact, the
    // log holds a single shard insert with tombstone_version=65.
    let source = CrdtOrMap::new();
    source.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);
    let mut source_log = OperationLog::new();
    source_log.append(Operation::insert(
        key.to_string(),
        encode(7, 99).to_vec(),
        60,
        ReplicaId::new(),
    ));
    source_log.append(Operation::remove(key.to_string(), 65, ReplicaId::new()));
    source_log.append(Operation::insert(
        key.to_string(),
        encode(6, 1).to_vec(),
        70,
        ReplicaId::new(),
    ));
    source.merge(&source_log);

    let snapshot_log = source.get_operation_log();
    assert_eq!(
        snapshot_log.operations().len(),
        1,
        "compaction must reduce to a single shard insert",
    );

    // Receiver applies the snapshot — gets the shard with
    // tombstone_version embedded but no Remove op in its log.
    let receiver = CrdtOrMap::new();
    receiver.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);
    receiver.merge(&snapshot_log);
    assert_eq!(
        decode(&receiver.get(key).expect("post-tombstone insert applied")),
        Some(EpochCount { epoch: 6, count: 1 }),
    );

    // Late peer that never saw the Remove gossips the original
    // pre-tombstone high-epoch insert. The receiver must reject it
    // — the shard's embedded tombstone_version (65) > the late
    // insert's version (60), so it gets filtered.
    let mut late_log = OperationLog::new();
    late_log.append(Operation::insert(
        key.to_string(),
        encode(7, 99).to_vec(),
        60,
        ReplicaId::new(),
    ));
    receiver.merge(&late_log);

    assert_eq!(
        decode(
            &receiver
                .get(key)
                .expect("post-tombstone state must survive late pre-tombstone insert")
        ),
        Some(EpochCount { epoch: 6, count: 1 }),
        "pre-tombstone insert must not resurrect when only the snapshot \
         (no Remove op) has reached the receiver",
    );
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

    // Serialize to bytes
    let bytes = log.to_bytes().unwrap();

    // Deserialize
    let deserialized_log = OperationLog::from_bytes(&bytes).unwrap();
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
    assert!(!bytes.is_empty());

    // Deserialize
    let deserialized_log = OperationLog::from_bytes(&bytes).unwrap();
    assert_eq!(log.len(), deserialized_log.len());
}

#[test]
fn test_operation_log_merge_deduplicates() {
    init_test_logging();
    let map = CrdtOrMap::new();

    map.insert("key1".to_string(), b"value1".to_vec());
    map.insert("key2".to_string(), b"value2".to_vec());
    map.remove("key1");

    let log = map.get_operation_log();

    let mut merged_log = OperationLog::new();
    merged_log.merge(&log);
    let merged_once_len = merged_log.len();

    // Re-merging the same log should be a no-op for log length.
    merged_log.merge(&log);
    assert_eq!(merged_log.len(), merged_once_len);
}

#[test]
fn test_operation_log_snapshot_uses_merge_strategy() {
    let key = "rl:global:node-a";
    let stale_newer_timestamp = Operation::insert(
        key.to_string(),
        encode(5, 100).to_vec(),
        2,
        ReplicaId::new(),
    );
    let epoch_winner_older_timestamp =
        Operation::insert(key.to_string(), encode(6, 0).to_vec(), 1, ReplicaId::new());

    let mut log = OperationLog::new();
    log.append(stale_newer_timestamp);
    log.append(epoch_winner_older_timestamp);

    let snapshot = log.snapshot_and_truncate(|key| {
        if key.starts_with("rl:") {
            MergeStrategy::EpochMaxWins
        } else {
            MergeStrategy::LastWriterWins
        }
    });

    let Operation::Insert { value, .. } = snapshot.get(key).expect("snapshot keeps rl shard")
    else {
        panic!("snapshot should keep an insert");
    };
    assert_eq!(decode(value), Some(EpochCount { epoch: 6, count: 0 }));
    assert!(log.is_empty(), "snapshot truncates the source log");
}

#[test]
fn test_operation_log_epoch_max_wins_tombstone_selection_is_order_independent() {
    let key = "rl:global:node-a";
    let stale_lower_epoch = Operation::insert(
        key.to_string(),
        encode(5, 100).to_vec(),
        80,
        ReplicaId::new(),
    );
    let epoch_winner_older_timestamp =
        Operation::insert(key.to_string(), encode(6, 0).to_vec(), 90, ReplicaId::new());
    let tombstone_after_epoch_winner = Operation::remove(key.to_string(), 95, ReplicaId::new());
    let orders = [
        [
            stale_lower_epoch.clone(),
            epoch_winner_older_timestamp.clone(),
            tombstone_after_epoch_winner.clone(),
        ],
        [
            stale_lower_epoch.clone(),
            tombstone_after_epoch_winner.clone(),
            epoch_winner_older_timestamp.clone(),
        ],
        [
            epoch_winner_older_timestamp.clone(),
            stale_lower_epoch.clone(),
            tombstone_after_epoch_winner.clone(),
        ],
        [
            epoch_winner_older_timestamp.clone(),
            tombstone_after_epoch_winner.clone(),
            stale_lower_epoch.clone(),
        ],
        [
            tombstone_after_epoch_winner.clone(),
            stale_lower_epoch.clone(),
            epoch_winner_older_timestamp.clone(),
        ],
        [
            tombstone_after_epoch_winner.clone(),
            epoch_winner_older_timestamp.clone(),
            stale_lower_epoch.clone(),
        ],
    ];

    for order in orders {
        let mut log = OperationLog::new();
        for operation in order {
            log.append(operation);
        }

        let snapshot = log.snapshot_and_truncate(|key| {
            if key.starts_with("rl:") {
                MergeStrategy::EpochMaxWins
            } else {
                MergeStrategy::LastWriterWins
            }
        });

        let Some(Operation::Remove { timestamp, .. }) = snapshot.get(key) else {
            panic!("tombstone should win consistently for order {snapshot:?}");
        };
        assert_eq!(*timestamp, 95);
    }
}

#[test]
fn test_operation_log_epoch_max_wins_post_tombstone_insert_revives_key() {
    let key = "rl:global:node-a";
    let pre_tombstone_higher_epoch =
        Operation::insert(key.to_string(), encode(7, 0).to_vec(), 90, ReplicaId::new());
    let tombstone = Operation::remove(key.to_string(), 95, ReplicaId::new());
    let post_tombstone_lower_epoch = Operation::insert(
        key.to_string(),
        encode(6, 0).to_vec(),
        100,
        ReplicaId::new(),
    );
    let orders = [
        [
            pre_tombstone_higher_epoch.clone(),
            tombstone.clone(),
            post_tombstone_lower_epoch.clone(),
        ],
        [
            pre_tombstone_higher_epoch.clone(),
            post_tombstone_lower_epoch.clone(),
            tombstone.clone(),
        ],
        [
            tombstone.clone(),
            pre_tombstone_higher_epoch.clone(),
            post_tombstone_lower_epoch.clone(),
        ],
        [
            tombstone.clone(),
            post_tombstone_lower_epoch.clone(),
            pre_tombstone_higher_epoch.clone(),
        ],
        [
            post_tombstone_lower_epoch.clone(),
            pre_tombstone_higher_epoch.clone(),
            tombstone.clone(),
        ],
        [
            post_tombstone_lower_epoch.clone(),
            tombstone.clone(),
            pre_tombstone_higher_epoch.clone(),
        ],
    ];

    for order in orders {
        let mut log = OperationLog::new();
        for operation in order {
            log.append(operation);
        }

        let snapshot = log.snapshot_and_truncate(|key| {
            if key.starts_with("rl:") {
                MergeStrategy::EpochMaxWins
            } else {
                MergeStrategy::LastWriterWins
            }
        });

        let Some(Operation::Insert {
            value, timestamp, ..
        }) = snapshot.get(key)
        else {
            panic!("post-tombstone insert should revive key for order {snapshot:?}");
        };
        assert_eq!(*timestamp, 100);
        assert_eq!(decode(value), Some(EpochCount { epoch: 6, count: 0 }));
    }
}

#[test]
fn test_operation_log_epoch_max_wins_post_tombstone_insert_wins_over_pre_tombstone_equal_epoch() {
    let key = "rl:global:node-a";
    let newer_insert = Operation::insert(
        key.to_string(),
        encode(6, 0).to_vec(),
        100,
        ReplicaId::new(),
    );
    let older_equal_insert =
        Operation::insert(key.to_string(), encode(6, 0).to_vec(), 10, ReplicaId::new());
    let tombstone_between = Operation::remove(key.to_string(), 50, ReplicaId::new());

    let mut log = OperationLog::new();
    log.append(older_equal_insert);
    log.append(tombstone_between);
    log.append(newer_insert);

    let snapshot = log.snapshot_and_truncate(|key| {
        if key.starts_with("rl:") {
            MergeStrategy::EpochMaxWins
        } else {
            MergeStrategy::LastWriterWins
        }
    });

    let Some(Operation::Insert {
        value, timestamp, ..
    }) = snapshot.get(key)
    else {
        panic!("newer equal-value insert should win over intermediate tombstone");
    };
    assert_eq!(*timestamp, 100);
    assert_eq!(decode(value), Some(EpochCount { epoch: 6, count: 0 }));
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
    // OR-Map remove only applies to observed keys, so replica3 first observes replica1 state.
    let log1 = replica1.get_operation_log();
    replica3.merge(&log1);
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

    assert_eq!(replica1.get("user:1"), Some(b"Alice_Updated".to_vec()));
    assert_eq!(replica1.get("user:3"), Some(b"Charlie".to_vec()));
    assert_eq!(replica1.get("user:4"), Some(b"David".to_vec()));
}

// ============================================================================
// Tombstone GC Grace Period Tests
// ============================================================================

#[test]
fn test_gc_tombstones_respects_grace_period() {
    init_test_logging();
    let map = CrdtOrMap::new();

    map.insert("key1".to_string(), b"value1".to_vec());
    map.remove("key1");

    // GC with a long grace period — tombstone is too young, should NOT be collected.
    let removed = map.gc_tombstones_with_grace(Duration::from_secs(3600));
    assert_eq!(removed, 0, "Young tombstone should not be GC'd");

    // GC with zero grace period — tombstone should be collected immediately.
    let removed = map.gc_tombstones_with_grace(Duration::ZERO);
    assert_eq!(removed, 1, "Expired tombstone should be GC'd");
}

#[test]
fn test_gc_tombstones_does_not_remove_live_keys() {
    init_test_logging();
    let map = CrdtOrMap::new();

    map.insert("key1".to_string(), b"value1".to_vec());
    map.insert("key2".to_string(), b"value2".to_vec());

    // GC should not remove live (non-tombstoned) keys.
    let removed = map.gc_tombstones_with_grace(Duration::ZERO);
    assert_eq!(removed, 0);
    assert_eq!(map.get("key1"), Some(b"value1".to_vec()));
    assert_eq!(map.get("key2"), Some(b"value2".to_vec()));
}

#[test]
fn test_gc_tombstones_multiple_keys() {
    init_test_logging();
    let map = CrdtOrMap::new();

    map.insert("key1".to_string(), b"v1".to_vec());
    map.insert("key2".to_string(), b"v2".to_vec());
    map.insert("key3".to_string(), b"v3".to_vec());

    map.remove("key1");
    map.remove("key3");
    // key2 stays alive.

    let removed = map.gc_tombstones_with_grace(Duration::ZERO);
    assert_eq!(removed, 2, "Two tombstoned keys should be GC'd");
    assert_eq!(map.get("key2"), Some(b"v2".to_vec()));
}
