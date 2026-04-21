use std::{
    collections::HashMap,
    net::SocketAddr,
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Result;
use futures::Stream;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tonic::{
    transport::{server::TcpIncoming, Server},
    Response, Status,
};
use tracing as log;
use tracing::instrument;

use super::{
    chunking::{
        build_stream_batches, chunk_value, dispatch_stream_batch, next_generation,
        DEFAULT_MAX_CHUNKS_PER_BATCH, MAX_STREAM_CHUNK_BYTES,
    },
    flow_control::{MessageSizeValidator, MAX_MESSAGE_SIZE},
    metrics::{
        record_ack, record_batch_sent, record_nack, record_peer_reconnect, record_snapshot_bytes,
        record_snapshot_duration, record_snapshot_trigger, update_peer_connections,
        ConvergenceTracker,
    },
    mtls::MTLSManager,
    node_state_machine::NodeStateMachine,
    partition::PartitionDetector,
    service::{
        gossip::{
            self,
            gossip_server::{Gossip, GossipServer},
            GossipMessage, IncrementalUpdate, NodeState, NodeStatus, NodeUpdate, PingReq,
            SnapshotChunk, SnapshotRequest, StateUpdate, StreamAck, StreamMessage,
            StreamMessageType,
        },
        try_ping, ClusterState,
    },
    stores::{StateStores, StoreType as LocalStoreType},
    sync::MeshSyncManager,
};
use crate::collector::{PeerWatermark, RoundBatch};

#[derive(Debug)]
pub struct GossipService {
    state: ClusterState,
    listen_addr: SocketAddr,
    advertise_addr: SocketAddr,
    self_name: String,
    stores: Option<Arc<StateStores>>, // Optional state stores for CRDT-based sync
    sync_manager: Option<Arc<MeshSyncManager>>, // Optional sync manager for applying remote updates
    state_machine: Option<Arc<NodeStateMachine>>,
    partition_detector: Option<Arc<PartitionDetector>>,
    mtls_manager: Option<Arc<MTLSManager>>,
    /// Shared reference to the current RoundBatch, updated once per round by
    /// the MeshController's central collector. Server-side sync_stream handlers
    /// read from this and apply per-peer watermark filtering.
    current_batch: Option<Arc<parking_lot::RwLock<Arc<RoundBatch>>>>,
    /// Shared reference to the current stream RoundBatch, drained once
    /// per round by the MeshController. Server-side handlers read
    /// broadcast drain_entries and also emit targeted_entries addressed
    /// to the remote peer learned from the first inbound message, so
    /// publish_to(peer) works in both directions of a peer pair.
    current_stream_batch: Option<Arc<parking_lot::RwLock<Arc<crate::kv::RoundBatch>>>>,
    /// Node-wide MeshKV handle. Owns the stream buffers, subscriber
    /// registry, and chunk assembler shared with the client-side
    /// SyncStream handlers.
    mesh_kv: Option<Arc<crate::kv::MeshKV>>,
}

impl GossipService {
    /// Create snapshot chunks for a store
    #[expect(
        clippy::expect_used,
        reason = "system clock before UNIX epoch is a fatal misconfiguration that must not silently produce timestamp=0"
    )]
    pub fn create_snapshot_chunks(
        &self,
        store_type: LocalStoreType,
        chunk_size: usize,
    ) -> Vec<SnapshotChunk> {
        let stores = match self.stores.as_ref() {
            Some(s) => s,
            None => {
                log::warn!("State stores not available for snapshot generation");
                return vec![];
            }
        };

        let proto_store_type = store_type.to_proto();

        // Get all entries from the store
        let entries: Vec<(String, Vec<u8>)> = match store_type {
            LocalStoreType::Membership => stores
                .membership
                .all()
                .into_iter()
                .map(|(k, v)| {
                    let serialized = bincode::serialize(&v).unwrap_or_else(|e| {
                        log::error!("Failed to serialize membership state: {}", e);
                        vec![]
                    });
                    (k, serialized)
                })
                .collect(),
            LocalStoreType::App => stores
                .app
                .all()
                .into_iter()
                .map(|(k, v)| {
                    let serialized = bincode::serialize(&v).unwrap_or_else(|e| {
                        log::error!("Failed to serialize app state: {}", e);
                        vec![]
                    });
                    (k, serialized)
                })
                .collect(),
            LocalStoreType::Worker => stores
                .worker
                .all()
                .into_iter()
                .map(|(k, v)| {
                    let serialized = bincode::serialize(&v).unwrap_or_else(|e| {
                        log::error!("Failed to serialize worker state: {}", e);
                        vec![]
                    });
                    (k, serialized)
                })
                .collect(),
            LocalStoreType::Policy => {
                let entries: Vec<(String, Vec<u8>)> = stores
                    .policy
                    .all()
                    .into_iter()
                    .filter(|(k, _)| {
                        // Tree configs are handled separately below via
                        // stores.tree_configs — skip stale CRDT policy
                        // entries with "tree:" keys.
                        !k.starts_with("tree:")
                    })
                    .map(|(k, v)| {
                        let serialized = bincode::serialize(&v).unwrap_or_else(|e| {
                            log::error!("Failed to serialize policy state: {}", e);
                            vec![]
                        });
                        (k, serialized)
                    })
                    .collect();

                // Tree data is synced via Layer 1 (tenant deltas) and Layer 2
                // (periodic compressed snapshots). No longer include tree_configs
                // in the snapshot exchange — cloning large TreeState bytes on
                // every ping round caused multi-GB memory growth.
                entries
            }
            LocalStoreType::RateLimit => {
                // For rate limit, serialize all counters from owners
                stores
                    .rate_limit
                    .keys()
                    .into_iter()
                    .filter_map(|key| {
                        if stores.rate_limit.is_owner(&key) {
                            stores.rate_limit.get_counter(&key).map(|counter_value| {
                                let serialized =
                                    bincode::serialize(&counter_value).unwrap_or_else(|e| {
                                        log::error!(
                                            "Failed to serialize rate limit counter: {}",
                                            e
                                        );
                                        vec![]
                                    });
                                (key.clone(), serialized)
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            }
        };

        if entries.is_empty() {
            return vec![];
        }

        // Split entries into chunks
        let mut chunks = Vec::new();
        let total_chunks = entries.len().div_ceil(chunk_size);

        for (chunk_idx, chunk_entries) in entries.chunks(chunk_size).enumerate() {
            let state_updates: Vec<StateUpdate> = chunk_entries
                .iter()
                .map(|(key, value)| {
                    // Get actual version from CRDT metadata
                    let version = match store_type {
                        LocalStoreType::Membership => {
                            stores.membership.get(key).map(|s| s.version).unwrap_or(1)
                        }
                        LocalStoreType::App => stores.app.get(key).map(|s| s.version).unwrap_or(1),
                        LocalStoreType::Worker => {
                            stores.worker.get(key).map(|s| s.version).unwrap_or(1)
                        }
                        LocalStoreType::Policy => {
                            // For tree keys, version comes from tree_configs
                            // (not the CRDT policy store).
                            if key.starts_with("tree:") {
                                stores
                                    .tree_configs
                                    .get(key)
                                    .and_then(|bytes| {
                                        super::tree_ops::TreeState::from_bytes(&bytes)
                                            .ok()
                                            .map(|ts| ts.version)
                                    })
                                    .unwrap_or(1)
                            } else {
                                stores.policy.get(key).map(|s| s.version).unwrap_or(1)
                            }
                        }
                        LocalStoreType::RateLimit => {
                            // For rate limit, use timestamp as version
                            {
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .expect("system clock before UNIX_EPOCH; cannot generate valid timestamps")
                                    .as_nanos() as u64
                            }
                        }
                    };

                    let timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .expect("system clock before UNIX_EPOCH; cannot generate valid timestamps")
                        .as_nanos() as u64;

                    StateUpdate {
                        key: key.clone(),
                        value: value.clone(),
                        version,
                        actor: self.self_name.clone(),
                        timestamp,
                    }
                })
                .collect();

            // Calculate checksum for integrity verification
            use std::{
                collections::hash_map::DefaultHasher,
                hash::{Hash, Hasher},
            };
            let mut hasher = DefaultHasher::new();
            for update in &state_updates {
                update.key.hash(&mut hasher);
                update.value.hash(&mut hasher);
            }
            let checksum = hasher.finish().to_le_bytes().to_vec();

            chunks.push(SnapshotChunk {
                store: proto_store_type,
                chunk_index: chunk_idx as u64,
                total_chunks: total_chunks as u64,
                entries: state_updates,
                checksum,
            });
        }

        log::info!(
            "Generated {} snapshot chunks for store {:?}",
            chunks.len(),
            store_type
        );
        chunks
    }
}

impl GossipService {
    pub fn new(
        state: ClusterState,
        listen_addr: SocketAddr,
        advertise_addr: SocketAddr,
        self_name: &str,
    ) -> Self {
        Self {
            state,
            listen_addr,
            advertise_addr,
            self_name: self_name.to_string(),
            stores: None,
            sync_manager: None,
            state_machine: None,
            partition_detector: None,
            mtls_manager: None,
            current_batch: None,
            current_stream_batch: None,
            mesh_kv: None,
        }
    }

    /// Attach the shared RoundBatch reference from the MeshController.
    /// Server-side sync_stream handlers read from this single batch
    /// produced by the CentralCollector rather than re-collecting per peer.
    pub fn with_current_batch(
        mut self,
        current_batch: Arc<parking_lot::RwLock<Arc<RoundBatch>>>,
    ) -> Self {
        self.current_batch = Some(current_batch);
        self
    }

    /// Attach the shared stream RoundBatch reference. Server-side
    /// handlers emit broadcast drain_entries plus targeted_entries
    /// whose target matches the remote peer learned from the first
    /// inbound StreamMessage, so publish_to() works in both directions
    /// of a peer pair.
    pub fn with_current_stream_batch(
        mut self,
        current_stream_batch: Arc<parking_lot::RwLock<Arc<crate::kv::RoundBatch>>>,
    ) -> Self {
        self.current_stream_batch = Some(current_stream_batch);
        self
    }

    /// Attach the node-wide MeshKV handle. Plumbed from the server
    /// builder so stream buffers, subscribers, and the chunk assembler
    /// are shared between the client-side (outbound) and server-side
    /// (inbound) SyncStream handlers.
    pub fn with_mesh_kv(mut self, mesh_kv: Arc<crate::kv::MeshKV>) -> Self {
        self.mesh_kv = Some(mesh_kv);
        self
    }

    pub fn with_stores(mut self, stores: Arc<StateStores>) -> Self {
        self.stores = Some(stores.clone());
        // Create state machine if stores are provided
        if self.state_machine.is_none() {
            use super::node_state_machine::ConvergenceConfig;
            self.state_machine = Some(Arc::new(NodeStateMachine::new(
                stores,
                ConvergenceConfig::default(),
            )));
        }
        self
    }

    pub fn with_sync_manager(mut self, sync_manager: Arc<MeshSyncManager>) -> Self {
        self.sync_manager = Some(sync_manager);
        self
    }

    pub fn with_partition_detector(mut self, partition_detector: Arc<PartitionDetector>) -> Self {
        self.partition_detector = Some(partition_detector);
        self
    }

    pub fn with_mtls_manager(mut self, mtls_manager: Arc<MTLSManager>) -> Self {
        self.mtls_manager = Some(mtls_manager);
        self
    }

    pub async fn serve_ping_with_shutdown<F: std::future::Future<Output = ()>>(
        self,
        signal: F,
    ) -> Result<()> {
        let listen_addr = self.listen_addr;
        let service = GossipServer::new(self)
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE)
            .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
            .send_compressed(tonic::codec::CompressionEncoding::Gzip);

        Server::builder()
            .add_service(service)
            .serve_with_shutdown(listen_addr, signal)
            .await?;
        Ok(())
    }

    pub async fn serve_ping_with_listener<F: std::future::Future<Output = ()>>(
        self,
        listener: tokio::net::TcpListener,
        signal: F,
    ) -> Result<()> {
        let incoming = TcpIncoming::from(listener);
        let service = GossipServer::new(self)
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE)
            .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
            .send_compressed(tonic::codec::CompressionEncoding::Gzip);
        Server::builder()
            .add_service(service)
            .serve_with_incoming_shutdown(incoming, signal)
            .await?;
        Ok(())
    }

    fn merge_state(&self, incoming_nodes: Vec<NodeState>) -> bool {
        let mut state = self.state.write();
        let mut updated = false;
        for node in incoming_nodes {
            state
                .entry(node.name.clone())
                .and_modify(|entry| {
                    if node.version > entry.version {
                        *entry = node.clone();
                        updated = true;
                    }
                })
                .or_insert_with(|| {
                    updated = true;
                    node
                });
        }
        if updated {
            log::info!("Cluster state updated. Current nodes: {}", state.len());
        }
        updated
    }
}

#[tonic::async_trait]
impl Gossip for GossipService {
    type SyncStreamStream =
        Pin<Box<dyn Stream<Item = Result<StreamMessage, Status>> + Send + 'static>>;

    #[instrument(fields(name = %self.self_name), skip(self, request))]
    async fn ping_server(
        &self,
        request: tonic::Request<GossipMessage>,
    ) -> std::result::Result<Response<NodeUpdate>, Status> {
        let message = request.into_inner();
        match message.payload {
            Some(gossip::gossip_message::Payload::Ping(ping)) => {
                log::info!("Received {:?}", ping);
                if let Some(stat_sync) = ping.state_sync {
                    log::info!("Merging state from Ping: {} nodes", stat_sync.nodes.len());
                    self.merge_state(stat_sync.nodes);
                }
                // Return current status of self node (could be Alive or Leaving)
                let current_status = {
                    let state = self.state.read();
                    state
                        .get(&self.self_name)
                        .map(|n| n.status)
                        .unwrap_or(NodeStatus::Alive as i32)
                };
                Ok(Response::new(NodeUpdate {
                    name: self.self_name.clone(),
                    address: self.advertise_addr.to_string(),
                    status: current_status,
                }))
            }
            Some(gossip::gossip_message::Payload::PingReq(PingReq { node: Some(node) })) => {
                log::info!("PingReq to node {} addr:{}", node.name, node.address);
                let res = try_ping(&node, None, self.mtls_manager.clone()).await?;
                Ok(Response::new(res))
            }
            _ => Err(Status::invalid_argument("Invalid message payload")),
        }
    }

    #[instrument(fields(name = %self.self_name), skip(self, request))]
    async fn sync_stream(
        &self,
        request: tonic::Request<tonic::Streaming<StreamMessage>>,
    ) -> Result<Response<Self::SyncStreamStream>, Status> {
        let mut incoming = request.into_inner();
        let self_name = self.self_name.clone();
        let state = self.state.clone();
        let stores = self.stores.clone();
        let sync_manager = self.sync_manager.clone();
        let mesh_kv = self.mesh_kv.clone();

        // Create output stream with flow control
        const CHANNEL_CAPACITY: usize = 128;
        let (tx, rx) = mpsc::channel::<Result<StreamMessage, Status>>(CHANNEL_CAPACITY);
        let size_validator = MessageSizeValidator::default();

        // Remote peer identity, discovered from inbound StreamMessage.peer_id.
        // Shared between the inbound handler (writer) and the server-side
        // sender task (reader) so the sender can emit targeted_entries whose
        // `target` matches the learned peer. Before the first inbound message
        // this is `None` and the sender emits only broadcast drain_entries —
        // the targeted entries for that first round are dropped under
        // at-most-once semantics and the application retries.
        let learned_peer: Arc<parking_lot::RwLock<Option<String>>> =
            Arc::new(parking_lot::RwLock::new(None));

        // Spawn task to periodically send incremental updates.
        // Uses PeerWatermark reading from the shared RoundBatch (central collector).
        // If current_batch is not set (e.g., temporary GossipService for snapshots),
        // skip the sender task.
        let incremental_sender_handle = if let Some(batch_handle) = self.current_batch.clone() {
            let tx_incremental = tx.clone();
            let self_name_incremental = self_name.clone();
            let size_validator_clone = size_validator.clone();
            // The remote peer's name isn't known until the first stream message
            // arrives. Use a placeholder label for debug output — this doesn't
            // affect filtering correctness (each task has its own watermark).
            let peer_name_for_watermark = "server-inbound".to_string();
            let stream_batch_handle = self.current_stream_batch.clone();
            let learned_peer_sender = learned_peer.clone();
            #[expect(
                clippy::disallowed_methods,
                reason = "server-side incremental sender that runs for the lifetime of the sync_stream; terminates when the channel closes or handle is aborted"
            )]
            Some(tokio::spawn(async move {
                let mut watermark = PeerWatermark::new(peer_name_for_watermark);
                let mut interval = tokio::time::interval(Duration::from_secs(1));
                let mut sequence_counter: u64 = 0;
                let mut last_stream_batch: Option<Arc<crate::kv::RoundBatch>> = None;

                loop {
                    interval.tick().await;

                    // Read the centrally collected batch and filter by this peer's watermark.
                    let batch = batch_handle.read().clone();
                    let all_updates = watermark.filter(&batch);

                    if !all_updates.is_empty() {
                        for (store_type, updates) in all_updates {
                            let proto_store_type = store_type.to_proto();

                            sequence_counter += 1;
                            let batch_size: usize = updates.iter().map(|u| u.value.len()).sum();

                            // Validate message size
                            if let Err(e) = size_validator_clone.validate(batch_size) {
                                log::warn!(
                                    "Incremental update too large, skipping store {:?}: {} (max: {} bytes)",
                                    store_type,
                                    e,
                                    size_validator_clone.max_size()
                                );
                                // Mark as sent to prevent infinite retry loop.
                                watermark.mark_sent(store_type, &updates);
                                continue;
                            }

                            let incremental_update = StreamMessage {
                                message_type: StreamMessageType::IncrementalUpdate as i32,
                                payload: Some(gossip::stream_message::Payload::Incremental(
                                    IncrementalUpdate {
                                        store: proto_store_type,
                                        updates: updates.clone(),
                                        version: 0, // Version is tracked per key in StateUpdate
                                    },
                                )),
                                sequence: sequence_counter,
                                peer_id: self_name_incremental.clone(),
                            };

                            // Check backpressure using try_send
                            match tx_incremental.try_send(Ok(incremental_update)) {
                                Ok(()) => {
                                    record_batch_sent(&self_name_incremental, batch_size);
                                    watermark.mark_sent(store_type, &updates);
                                }
                                Err(mpsc::error::TrySendError::Full(_)) => {
                                    log::debug!(
                                        "Backpressure: channel full, skipping send (will retry next interval)"
                                    );
                                    continue;
                                }
                                Err(mpsc::error::TrySendError::Closed(_)) => {
                                    log::warn!(
                                        "Channel closed, stopping incremental update sender"
                                    );
                                    break;
                                }
                            }

                            log::debug!(
                                "Sent incremental update: store={:?}, {} updates",
                                store_type,
                                updates.len()
                            );
                        }
                    }

                    // Server-side stream emission: broadcast drain_entries
                    // plus targeted_entries addressed to the learned remote
                    // peer. Covers the non-initiator → initiator direction
                    // of each peer pair; the client-side sender in
                    // controller.rs handles the other direction. Before
                    // learned_peer is set (first inbound message not yet
                    // received), targeted entries in this round are dropped
                    // under at-most-once. On channel full, drop without retry.
                    if let Some(sbh) = &stream_batch_handle {
                        let stream_batch = sbh.read().clone();
                        let fresh_batch = last_stream_batch
                            .as_ref()
                            .is_none_or(|last| !Arc::ptr_eq(last, &stream_batch));
                        if fresh_batch {
                            // Advance the tracker for every fresh Arc, not just
                            // ones that produced work — otherwise a batch with
                            // empty drain_entries stays "fresh" across ticks and
                            // we keep re-checking it.
                            last_stream_batch = Some(stream_batch.clone());
                        }
                        let peer_for_targeted = learned_peer_sender.read().clone();
                        let has_targeted = peer_for_targeted.as_ref().is_some_and(|p| {
                            stream_batch.targeted_entries.iter().any(|(t, _, _)| t == p)
                        });
                        if fresh_batch && (!stream_batch.drain_entries.is_empty() || has_targeted) {
                            let mut entries = Vec::new();
                            // Generation is per-value so concurrent publishes
                            // to the same key get distinct tags.
                            for (key, value) in &stream_batch.drain_entries {
                                entries.extend(chunk_value(
                                    key.clone(),
                                    next_generation(),
                                    value.clone(),
                                    MAX_STREAM_CHUNK_BYTES,
                                ));
                            }
                            if let Some(ref peer) = peer_for_targeted {
                                for (target, key, value) in &stream_batch.targeted_entries {
                                    if target == peer {
                                        entries.extend(chunk_value(
                                            key.clone(),
                                            next_generation(),
                                            value.clone(),
                                            MAX_STREAM_CHUNK_BYTES,
                                        ));
                                    }
                                }
                            }
                            if !entries.is_empty() {
                                for batch in build_stream_batches(
                                    entries,
                                    DEFAULT_MAX_CHUNKS_PER_BATCH,
                                    MAX_STREAM_CHUNK_BYTES,
                                ) {
                                    sequence_counter += 1;
                                    let msg = StreamMessage {
                                        message_type: StreamMessageType::StreamBatch as i32,
                                        payload: Some(
                                            gossip::stream_message::Payload::StreamBatch(batch),
                                        ),
                                        sequence: sequence_counter,
                                        peer_id: self_name_incremental.clone(),
                                    };
                                    match tx_incremental.try_send(Ok(msg)) {
                                        Ok(()) => {}
                                        Err(mpsc::error::TrySendError::Full(_)) => {
                                            log::debug!(
                                                "server-side stream batch dropped on backpressure"
                                            );
                                            // TODO(metrics): bump
                                            // stream_dropped_on_backpressure
                                            break;
                                        }
                                        Err(mpsc::error::TrySendError::Closed(_)) => {
                                            log::warn!(
                                                "server-side stream sender: channel closed, stopping"
                                            );
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                    } else if last_stream_batch.is_some() {
                        // If handle became None (shouldn't normally), clear state.
                        last_stream_batch = None;
                    }
                }
            }))
        } else {
            None
        };

        // Spawn task to handle incoming messages
        let mut sequence: u64 = 0;
        let _convergence_tracker = ConvergenceTracker::new();

        // Track snapshot reception state: store_type -> (received_chunks, expected_total)
        // Keyed by store_type only — a new snapshot request for the same store
        // replaces any incomplete previous attempt (prevents stale chunk mixing).
        let mut snapshot_state: HashMap<LocalStoreType, (Vec<SnapshotChunk>, u64)> = HashMap::new();

        let learned_peer_inbound = learned_peer.clone();
        #[expect(
            clippy::disallowed_methods,
            reason = "server-side stream handler that runs for the lifetime of the sync_stream gRPC connection; terminates when the stream closes"
        )]
        tokio::spawn(async move {
            let mut peer_id = String::new();
            update_peer_connections(&peer_id, true);

            // Check if we need to request snapshots on connection
            // This happens when:
            // 1. We're a new node joining (stores are empty or very small)
            // 2. We detect a version gap
            if let Some(ref stores) = stores {
                for store_type in [
                    LocalStoreType::Membership,
                    LocalStoreType::App,
                    LocalStoreType::Worker,
                    LocalStoreType::Policy,
                    LocalStoreType::RateLimit,
                ] {
                    let store_len = match store_type {
                        LocalStoreType::Membership => stores.membership.len(),
                        LocalStoreType::App => stores.app.len(),
                        LocalStoreType::Worker => stores.worker.len(),
                        LocalStoreType::Policy => stores.policy.len(),
                        LocalStoreType::RateLimit => stores.rate_limit.keys().len(),
                    };

                    // If store is empty or very small, request snapshot
                    if store_len == 0 {
                        log::info!(
                            "Store {:?} is empty, requesting snapshot from {}",
                            store_type,
                            peer_id
                        );
                        let proto_store_type = store_type.to_proto();

                        let snapshot_request = StreamMessage {
                            message_type: StreamMessageType::SnapshotRequest as i32,
                            payload: Some(gossip::stream_message::Payload::SnapshotRequest(
                                SnapshotRequest {
                                    store: proto_store_type,
                                    from_version: 0, // Request from beginning
                                },
                            )),
                            sequence: 0,
                            peer_id: self_name.clone(),
                        };

                        if tx.send(Ok(snapshot_request)).await.is_err() {
                            log::warn!("Failed to send snapshot request");
                        }
                    }
                }
            }

            const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(60);
            // Short-circuits the RwLock::read() once we've accepted an
            // identity for this stream — learned_peer_inbound is
            // write-once-per-connection, so after the first learn every
            // subsequent message would otherwise take a pointless lock.
            let mut peer_learned = false;
            loop {
                match tokio::time::timeout(STREAM_IDLE_TIMEOUT, incoming.next()).await {
                    Ok(Some(Ok(msg))) => {
                        sequence += 1;
                        // Accept the claimed peer_id before copying it into
                        // the task-local `peer_id`: the task-local drives
                        // the teardown log and connection-gauge decrement
                        // below, and if we wrote a mismatched value here
                        // before the reject, the cleanup would attribute
                        // the close to the spoofed peer and leak the real
                        // peer's connection gauge.
                        //
                        // Reject mid-stream peer_id changes: the stream's
                        // remote identity is fixed for the connection, so
                        // a different peer_id on a later message is either
                        // a bug or a peer trying to claim another node's
                        // targeted entries. Close the stream and let the
                        // client reconnect. Pre-mTLS-binding defence;
                        // mTLS-derived identity is the authoritative
                        // long-term fix.
                        if !peer_learned && !msg.peer_id.is_empty() {
                            let mut learned = learned_peer_inbound.write();
                            match learned.as_ref() {
                                None => {
                                    *learned = Some(msg.peer_id.clone());
                                    peer_id.clone_from(&msg.peer_id);
                                    peer_learned = true;
                                }
                                Some(existing) if existing == &msg.peer_id => {
                                    peer_id.clone_from(existing);
                                    peer_learned = true;
                                }
                                Some(existing) => {
                                    log::warn!(
                                        expected_peer_id = %existing,
                                        received_peer_id = %msg.peer_id,
                                        "peer_id changed mid-stream; closing sync_stream"
                                    );
                                    break;
                                }
                            }
                        } else if peer_learned && msg.peer_id != peer_id {
                            // Empty peer_id after learn is also an identity
                            // change — a stream bound to a learned peer
                            // shouldn't accept unowned frames.
                            log::warn!(
                                expected_peer_id = %peer_id,
                                received_peer_id = %msg.peer_id,
                                "peer_id changed mid-stream; closing sync_stream"
                            );
                            break;
                        }

                        match msg.message_type() {
                            StreamMessageType::IncrementalUpdate => {
                                if let Some(gossip::stream_message::Payload::Incremental(update)) =
                                    &msg.payload
                                {
                                    // Validate message size
                                    let msg_size: usize =
                                        update.updates.iter().map(|u| u.value.len()).sum();
                                    if let Err(e) = size_validator.validate(msg_size) {
                                        log::warn!(
                                            "Received oversized incremental update from {}: {} (max: {} bytes), rejecting",
                                            peer_id, e, size_validator.max_size()
                                        );
                                        let nack = StreamMessage {
                                            message_type: StreamMessageType::Nack as i32,
                                            payload: Some(gossip::stream_message::Payload::Ack(
                                                StreamAck {
                                                    sequence: msg.sequence,
                                                    success: false,
                                                    error_message: format!(
                                                        "Message too large: {e}"
                                                    ),
                                                },
                                            )),
                                            sequence,
                                            peer_id: self_name.clone(),
                                        };
                                        if tx.send(Ok(nack)).await.is_err() {
                                            break;
                                        }
                                        record_nack(&peer_id);
                                        continue;
                                    }

                                    let store_type = LocalStoreType::from_proto(update.store);
                                    log::debug!("Received incremental update from {}: store={:?}, {} updates",
                                        peer_id, store_type, update.updates.len());

                                    // Apply incremental updates to state stores
                                    // This will be handled by the sync manager if available
                                    // For now, we acknowledge and the sync manager will handle it
                                    if let Some(ref sync_manager) = sync_manager {
                                        for state_update in &update.updates {
                                            match store_type {
                                                LocalStoreType::Worker => {
                                                    // Deserialize and apply worker state
                                                    if let Ok(worker_state) = bincode::deserialize::<
                                                        super::stores::WorkerState,
                                                    >(
                                                        &state_update.value
                                                    ) {
                                                        // Extract actor from StateUpdate
                                                        let actor =
                                                            Some(state_update.actor.clone());
                                                        sync_manager.apply_remote_worker_state(
                                                            worker_state,
                                                            actor,
                                                        );
                                                    }
                                                }
                                                LocalStoreType::Policy => {
                                                    // Deserialize and apply policy state
                                                    if let Ok(policy_state) = bincode::deserialize::<
                                                        super::stores::PolicyState,
                                                    >(
                                                        &state_update.value
                                                    ) {
                                                        let actor =
                                                            Some(state_update.actor.clone());

                                                        if policy_state.policy_type
                                                            == "tenant_delta"
                                                        {
                                                            // Lightweight tenant delta — no tree structure, no prompt text
                                                            match super::tree_ops::TenantDelta::from_bytes(
                                                                    &policy_state.config,
                                                                )
                                                            {
                                                                Ok(delta) => {
                                                                    sync_manager
                                                                        .apply_remote_tenant_delta(
                                                                            delta, actor,
                                                                        );
                                                                }
                                                                Err(e) => {
                                                                    log::warn!(
                                                                        "Failed to deserialize tenant delta for model {}: {e}",
                                                                        policy_state.model_id
                                                                    );
                                                                }
                                                            }
                                                        } else if policy_state.policy_type
                                                            == "tree_state_delta"
                                                        {
                                                            // Operation-level delta: apply only the new operations
                                                            match super::tree_ops::TreeStateDelta::from_bytes(
                                                                    &policy_state.config,
                                                                )
                                                            {
                                                                Ok(delta) => {
                                                                    sync_manager
                                                                        .apply_remote_tree_delta(
                                                                            delta, actor,
                                                                        );
                                                                }
                                                                Err(e) => {
                                                                    log::warn!(
                                                                        "Failed to deserialize tree state delta for model {}: {e}",
                                                                        policy_state.model_id
                                                                    );
                                                                }
                                                            }
                                                        } else if policy_state.policy_type
                                                            == "tree_state_lz4"
                                                        {
                                                            // LZ4-compressed snapshot (TreeState or TreeSnapshot bytes)
                                                            match super::tree_ops::lz4_decompress(
                                                                &policy_state.config,
                                                            ) {
                                                                Ok(decompressed) => {
                                                                    // Try TreeState first (backward compat)
                                                                    if let Ok(tree_state) =
                                                                        super::tree_ops::TreeState::from_bytes(
                                                                            &decompressed,
                                                                        )
                                                                    {
                                                                        sync_manager
                                                                            .apply_remote_tree_operation(
                                                                                policy_state
                                                                                    .model_id
                                                                                    .clone(),
                                                                                tree_state,
                                                                                actor.clone(),
                                                                            );
                                                                    } else if let Ok(snap) =
                                                                        kv_index::snapshot::TreeSnapshot::from_bytes(
                                                                            &decompressed,
                                                                        )
                                                                    {
                                                                        // Compact TreeSnapshot — convert to TreeState
                                                                        let tree_state =
                                                                            super::tree_ops::TreeState::from_snapshot(
                                                                                policy_state
                                                                                    .model_id
                                                                                    .clone(),
                                                                                &snap,
                                                                                policy_state.version,
                                                                            );
                                                                        sync_manager
                                                                            .apply_remote_tree_operation(
                                                                                policy_state
                                                                                    .model_id
                                                                                    .clone(),
                                                                                tree_state,
                                                                                actor.clone(),
                                                                            );
                                                                    } else {
                                                                        log::warn!(
                                                                            "Failed to deserialize tree_state_lz4 payload for model {}",
                                                                            policy_state.model_id
                                                                        );
                                                                    }
                                                                }
                                                                Err(e) => {
                                                                    log::warn!(
                                                                        "Failed to LZ4-decompress tree state for model {}: {e}",
                                                                        policy_state.model_id
                                                                    );
                                                                }
                                                            }
                                                        } else if policy_state.policy_type
                                                            == "tree_state"
                                                        {
                                                            // Uncompressed full state (backward compatible)
                                                            if let Ok(tree_state) =
                                                                super::tree_ops::TreeState::from_bytes(
                                                                    &policy_state.config,
                                                                )
                                                            {
                                                                sync_manager
                                                                    .apply_remote_tree_operation(
                                                                        policy_state
                                                                            .model_id
                                                                            .clone(),
                                                                        tree_state,
                                                                        actor,
                                                                    );
                                                            }
                                                        } else {
                                                            // Regular policy state update
                                                            sync_manager.apply_remote_policy_state(
                                                                policy_state,
                                                                actor,
                                                            );
                                                        }
                                                    }
                                                }
                                                LocalStoreType::App => {
                                                    // Deserialize and apply app state
                                                    if let Ok(app_state) = bincode::deserialize::<
                                                        super::stores::AppState,
                                                    >(
                                                        &state_update.value
                                                    ) {
                                                        // Apply app state directly to the store, skipping stale versions
                                                        if let Some(ref stores) = stores {
                                                            let dominated = stores
                                                                .app
                                                                .get(&app_state.key)
                                                                .is_some_and(|existing| {
                                                                    existing.version
                                                                        >= app_state.version
                                                                });
                                                            if !dominated {
                                                                if let Err(err) = stores.app.insert(
                                                                    app_state.key.clone(),
                                                                    app_state,
                                                                ) {
                                                                    log::warn!(error = %err, "Failed to apply app state update");
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                                LocalStoreType::Membership => {
                                                    // Deserialize and apply membership state
                                                    if let Ok(membership_state) =
                                                        bincode::deserialize::<
                                                            super::stores::MembershipState,
                                                        >(
                                                            &state_update.value
                                                        )
                                                    {
                                                        // Apply membership state directly to the store
                                                        if let Some(ref stores) = stores {
                                                            if let Err(err) =
                                                                stores.membership.insert(
                                                                    membership_state.name.clone(),
                                                                    membership_state,
                                                                )
                                                            {
                                                                log::warn!(error = %err, "Failed to apply membership state update");
                                                            }
                                                        }
                                                    }
                                                }
                                                LocalStoreType::RateLimit => {
                                                    if let Ok(op_log) = bincode::deserialize::<
                                                        super::crdt_kv::OperationLog,
                                                    >(
                                                        &state_update.value
                                                    ) {
                                                        if let Some(counter_value) = op_log
                                                            .latest_counter_value(&state_update.key)
                                                            .or_else(|| {
                                                                op_log.latest_counter_value_any()
                                                            })
                                                        {
                                                            sync_manager
                                                                .apply_remote_rate_limit_counter_value_with_actor_and_timestamp(
                                                                    state_update.key.clone(),
                                                                    state_update.actor.clone(),
                                                                    counter_value,
                                                                    state_update.timestamp,
                                                                );
                                                        } else {
                                                            log::warn!(
                                                                key = %state_update.key,
                                                                "Rate-limit OperationLog does not contain a decodable counter value"
                                                            );
                                                        }
                                                    } else if let Ok(counter_value) =
                                                        bincode::deserialize::<i64>(
                                                            &state_update.value,
                                                        )
                                                    {
                                                        sync_manager
                                                            .apply_remote_rate_limit_counter_value_with_actor_and_timestamp(
                                                                state_update.key.clone(),
                                                                state_update.actor.clone(),
                                                                counter_value,
                                                                state_update.timestamp,
                                                            );
                                                    } else {
                                                        log::warn!(
                                                            key = %state_update.key,
                                                            "Failed to decode rate-limit update as OperationLog or i64"
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    let ack = StreamMessage {
                                        message_type: StreamMessageType::Ack as i32,
                                        payload: Some(gossip::stream_message::Payload::Ack(
                                            StreamAck {
                                                sequence: msg.sequence,
                                                success: true,
                                                error_message: String::new(),
                                            },
                                        )),
                                        sequence,
                                        peer_id: self_name.clone(),
                                    };
                                    if tx.send(Ok(ack)).await.is_err() {
                                        break;
                                    }
                                }
                            }
                            StreamMessageType::SnapshotRequest => {
                                if let Some(gossip::stream_message::Payload::SnapshotRequest(req)) =
                                    &msg.payload
                                {
                                    let store_type = LocalStoreType::from_proto(req.store);
                                    let store_name = store_type.as_str();
                                    log::info!("Received snapshot request from {}: store={:?}, from_version={}",
                                        peer_id, store_type, req.from_version);

                                    record_snapshot_trigger(store_name, "request");
                                    let snapshot_start = Instant::now();

                                    // Generate and send snapshot chunks
                                    let service = GossipService {
                                        state: state.clone(),
                                        listen_addr: SocketAddr::from(([0, 0, 0, 0], 0)), // Not used in snapshot generation
                                        advertise_addr: SocketAddr::from(([0, 0, 0, 0], 0)), // Not used in snapshot generation
                                        self_name: self_name.clone(),
                                        stores: stores.clone(),
                                        sync_manager: sync_manager.clone(),
                                        state_machine: None,
                                        partition_detector: None,
                                        mtls_manager: None,
                                        current_batch: None,
                                        current_stream_batch: None,
                                        mesh_kv: None,
                                    };
                                    let chunks = service.create_snapshot_chunks(store_type, 100); // chunk_size = 100 entries
                                    let total_chunks = chunks.len() as u64;
                                    let mut total_bytes = 0;

                                    for (idx, chunk) in chunks.into_iter().enumerate() {
                                        let chunk_bytes = chunk
                                            .entries
                                            .iter()
                                            .map(|e| e.value.len())
                                            .sum::<usize>();
                                        total_bytes += chunk_bytes;

                                        let mut chunk_msg = StreamMessage {
                                            message_type: StreamMessageType::SnapshotChunk as i32,
                                            payload: Some(
                                                gossip::stream_message::Payload::SnapshotChunk(
                                                    chunk,
                                                ),
                                            ),
                                            sequence: sequence + idx as u64 + 1,
                                            peer_id: self_name.clone(),
                                        };
                                        // Update chunk metadata
                                        if let Some(
                                            gossip::stream_message::Payload::SnapshotChunk(
                                                ref mut c,
                                            ),
                                        ) = chunk_msg.payload
                                        {
                                            c.chunk_index = idx as u64;
                                            c.total_chunks = total_chunks;
                                        }

                                        // Check backpressure using try_send
                                        match tx.try_send(Ok(chunk_msg)) {
                                            Ok(()) => {
                                                // Successfully queued
                                            }
                                            Err(mpsc::error::TrySendError::Full(msg)) => {
                                                log::debug!(
                                                    "Backpressure: channel full, waiting for drain"
                                                );
                                                // Wait a bit for channel to drain, then use blocking send
                                                tokio::time::sleep(Duration::from_millis(100))
                                                    .await;
                                                if tx.send(msg).await.is_err() {
                                                    log::warn!("Backpressure: channel closed, stopping snapshot");
                                                    break;
                                                }
                                            }
                                            Err(mpsc::error::TrySendError::Closed(_)) => {
                                                log::warn!("Channel closed, stopping snapshot");
                                                break;
                                            }
                                        }
                                    }

                                    record_snapshot_duration(store_name, snapshot_start.elapsed());
                                    record_snapshot_bytes(store_name, "sent", total_bytes);

                                    // Send snapshot complete message
                                    let complete = StreamMessage {
                                        message_type: StreamMessageType::SnapshotComplete as i32,
                                        payload: None,
                                        sequence: sequence + total_chunks + 1,
                                        peer_id: self_name.clone(),
                                    };
                                    if tx.send(Ok(complete)).await.is_err() {
                                        break;
                                    }

                                    // Send ACK
                                    let ack = StreamMessage {
                                        message_type: StreamMessageType::Ack as i32,
                                        payload: Some(gossip::stream_message::Payload::Ack(
                                            StreamAck {
                                                sequence: msg.sequence,
                                                success: true,
                                                error_message: String::new(),
                                            },
                                        )),
                                        sequence,
                                        peer_id: self_name.clone(),
                                    };
                                    record_ack(&peer_id, true);
                                    if tx.send(Ok(ack)).await.is_err() {
                                        break;
                                    }
                                }
                            }
                            StreamMessageType::SnapshotChunk => {
                                if let Some(gossip::stream_message::Payload::SnapshotChunk(chunk)) =
                                    &msg.payload
                                {
                                    let store_type = LocalStoreType::from_proto(chunk.store);
                                    let store_name = store_type.as_str();
                                    log::info!(
                                        "Received snapshot chunk from {}: store={:?}, chunk={}/{}",
                                        peer_id,
                                        store_type,
                                        chunk.chunk_index,
                                        chunk.total_chunks
                                    );

                                    // Record metrics
                                    let chunk_bytes: usize =
                                        chunk.entries.iter().map(|e| e.value.len()).sum();
                                    record_snapshot_bytes(store_name, "received", chunk_bytes);

                                    // Store chunk. Reset on chunk_index == 0 (start of a
                                    // new snapshot transfer) to prevent stale chunks from a
                                    // previous attempt mixing with new ones — even if
                                    // total_chunks is the same.
                                    let (chunks, expected) = snapshot_state
                                        .entry(store_type)
                                        .or_insert_with(|| (Vec::new(), chunk.total_chunks));
                                    if chunk.chunk_index == 0 && !chunks.is_empty() {
                                        log::info!(
                                            "New snapshot transfer for {:?}, discarding {} partial chunks",
                                            store_type, chunks.len()
                                        );
                                        chunks.clear();
                                    }
                                    *expected = chunk.total_chunks;
                                    chunks.push(chunk.clone());

                                    // Check if we've received all chunks with valid indices
                                    if let Some((received_chunks, total)) =
                                        snapshot_state.get(&store_type)
                                    {
                                        if received_chunks.len() as u64 == *total {
                                            // Verify all indices 0..total are present (no duplicates/gaps)
                                            let mut sorted_chunks = received_chunks.to_vec();
                                            sorted_chunks.sort_by_key(|c| c.chunk_index);
                                            let indices_valid = sorted_chunks
                                                .iter()
                                                .enumerate()
                                                .all(|(i, c)| c.chunk_index == i as u64);
                                            if !indices_valid {
                                                log::warn!(
                                                    "Snapshot for {:?} has {} chunks but indices are not contiguous 0..{}, discarding",
                                                    store_type, sorted_chunks.len(), total
                                                );
                                                snapshot_state.remove(&store_type);
                                                continue;
                                            }

                                            log::info!("All {} chunks received for store {:?}, applying snapshot",
                                                total, store_type);

                                            if let Some(ref stores) = stores {
                                                // Apply all entries from chunks
                                                for chunk in &sorted_chunks {
                                                    for entry in &chunk.entries {
                                                        let key = entry.key.clone();

                                                        match store_type {
                                                            LocalStoreType::Membership => {
                                                                if let Ok(membership_state) = bincode::deserialize::<super::stores::MembershipState>(&entry.value) {
                                                                    let _ = stores.membership.insert(key, membership_state);
                                                                }
                                                            }
                                                            LocalStoreType::App => {
                                                                if let Ok(app_state) = bincode::deserialize::<super::stores::AppState>(&entry.value) {
                                                                    let dominated = stores.app.get(&key)
                                                                        .is_some_and(|existing| existing.version >= app_state.version);
                                                                    if !dominated {
                                                                        let _ = stores.app.insert(key, app_state);
                                                                    }
                                                                }
                                                            }
                                                            LocalStoreType::Worker => {
                                                                if let Ok(worker_state) = bincode::deserialize::<super::stores::WorkerState>(&entry.value) {
                                                                    let _ = stores.worker.insert(key, worker_state.clone());
                                                                    // Also update sync manager if available
                                                                    if let Some(ref sync_manager) = sync_manager {
                                                                        sync_manager.apply_remote_worker_state(worker_state, Some(entry.actor.clone()));
                                                                    }
                                                                }
                                                            }
                                                            LocalStoreType::Policy => {
                                                                if let Ok(policy_state) = bincode::deserialize::<super::stores::PolicyState>(&entry.value) {
                                                                    if policy_state.policy_type == "tree_state" {
                                                                        // Tree state entries go to tree_configs
                                                                        // (plain DashMap, no CRDT operation log).
                                                                        if let Some(ref sync_manager) = sync_manager {
                                                                            if let Ok(tree_state) = super::tree_ops::TreeState::from_bytes(
                                                                                &policy_state.config
                                                                            ) {
                                                                                sync_manager.apply_remote_tree_operation(
                                                                                    policy_state.model_id.clone(),
                                                                                    tree_state,
                                                                                    Some(entry.actor.clone()),
                                                                                );
                                                                            }
                                                                        } else {
                                                                            // No sync manager — write directly to tree_configs.
                                                                            stores.tree_configs.insert(key, policy_state.config.clone());
                                                                        }
                                                                    } else {
                                                                        let _ = stores.policy.insert(key, policy_state.clone());
                                                                        if let Some(ref sync_manager) = sync_manager {
                                                                            sync_manager.apply_remote_policy_state(policy_state, Some(entry.actor.clone()));
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                            LocalStoreType::RateLimit => {
                                                                if let Some(ref sync_manager) = sync_manager {
                                                                    if let Ok(op_log) = bincode::deserialize::<super::crdt_kv::OperationLog>(&entry.value) {
                                                                        if let Some(counter_value) = op_log
                                                                            .latest_counter_value(&entry.key)
                                                                            .or_else(|| op_log.latest_counter_value_any())
                                                                        {
                                                                            sync_manager
                                                                                .apply_remote_rate_limit_counter_value_with_actor_and_timestamp(
                                                                                    entry.key.clone(),
                                                                                    entry.actor.clone(),
                                                                                    counter_value,
                                                                                    entry.timestamp,
                                                                                );
                                                                        } else {
                                                                            log::warn!(
                                                                                key = %entry.key,
                                                                                "Snapshot OperationLog does not contain a decodable rate-limit counter"
                                                                            );
                                                                        }
                                                                    } else if let Ok(counter_value) = bincode::deserialize::<i64>(&entry.value) {
                                                                        sync_manager
                                                                            .apply_remote_rate_limit_counter_value_with_actor_and_timestamp(
                                                                                entry.key.clone(),
                                                                                entry.actor.clone(),
                                                                                counter_value,
                                                                                entry.timestamp,
                                                                            );
                                                                    } else {
                                                                        log::warn!(
                                                                            key = %entry.key,
                                                                            "Failed to decode snapshot rate-limit entry as i64 or OperationLog"
                                                                        );
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }

                                                // Clear snapshot state
                                                snapshot_state.remove(&store_type);
                                                log::info!(
                                                    "Snapshot applied successfully for store {:?}",
                                                    store_type
                                                );
                                            }
                                        }
                                    }

                                    let ack = StreamMessage {
                                        message_type: StreamMessageType::Ack as i32,
                                        payload: Some(gossip::stream_message::Payload::Ack(
                                            StreamAck {
                                                sequence: msg.sequence,
                                                success: true,
                                                error_message: String::new(),
                                            },
                                        )),
                                        sequence,
                                        peer_id: self_name.clone(),
                                    };
                                    record_ack(&peer_id, true);
                                    if tx.send(Ok(ack)).await.is_err() {
                                        break;
                                    }
                                }
                            }
                            StreamMessageType::Ack => {
                                log::debug!(
                                    "Received ACK from {}: sequence={}",
                                    peer_id,
                                    msg.sequence
                                );
                                if let Some(gossip::stream_message::Payload::Ack(ack)) =
                                    &msg.payload
                                {
                                    record_ack(&peer_id, ack.success);
                                }
                            }
                            StreamMessageType::Heartbeat => {
                                // Send heartbeat back
                                let heartbeat = StreamMessage {
                                    message_type: StreamMessageType::Heartbeat as i32,
                                    payload: None,
                                    sequence,
                                    peer_id: self_name.clone(),
                                };
                                if tx.send(Ok(heartbeat)).await.is_err() {
                                    break;
                                }
                            }
                            StreamMessageType::StreamBatch => {
                                if let Some(mesh_kv) = &mesh_kv {
                                    if let Some(gossip::stream_message::Payload::StreamBatch(
                                        batch,
                                    )) = msg.payload
                                    {
                                        dispatch_stream_batch(mesh_kv, &msg.peer_id, batch.entries);
                                    }
                                }
                            }
                            _ => {
                                log::warn!(
                                    "Unknown message type from {}: {:?}",
                                    peer_id,
                                    msg.message_type
                                );
                            }
                        }
                    }
                    Ok(Some(Err(e))) => {
                        log::error!("Error receiving stream message: {}", e);
                        record_nack(&peer_id);
                        update_peer_connections(&peer_id, false);
                        record_peer_reconnect(&peer_id);
                        break;
                    }
                    Ok(None) => break,
                    Err(_) => {
                        tracing::warn!(
                            "sync_stream idle timeout ({STREAM_IDLE_TIMEOUT:?}) — closing"
                        );
                        break;
                    }
                }
            }
            // Abort the incremental sender task to release the tx
            // channel and allow the stream to close cleanly.
            if let Some(handle) = incremental_sender_handle {
                handle.abort();
                let _ = handle.await;
            }
            log::info!("Stream from {} closed", peer_id);
            update_peer_connections(&peer_id, false);
        });

        // Convert receiver to stream
        let output_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(
            Box::pin(output_stream) as Self::SyncStreamStream
        ))
    }
}
