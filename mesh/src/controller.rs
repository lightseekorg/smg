use std::{
    collections::{BTreeMap, HashMap},
    net::SocketAddr,
    sync::Arc,
    time::Duration,
};

use anyhow::Result;
use rand::seq::{IndexedRandom, SliceRandom};
use tokio::sync::Mutex;
use tracing as log;
use tracing::{instrument, Instrument};

use super::{
    flow_control::RetryManager,
    service::{
        broadcast_node_states,
        gossip::{
            gossip_client::GossipClient, gossip_message, stream_message::Payload as StreamPayload,
            NodeState, NodeStatus, Ping, PingReq, StateSync, StreamMessage, StreamMessageType,
        },
        try_ping, ClusterState,
    },
    stores::StateStores,
    sync::MeshSyncManager,
};

pub struct MeshController {
    state: ClusterState,
    self_name: String,
    self_addr: SocketAddr,
    init_peer: Option<SocketAddr>,
    stores: Option<Arc<StateStores>>,
    sync_manager: Option<Arc<MeshSyncManager>>,
    // Track active sync_stream connections
    sync_connections: Arc<Mutex<HashMap<String, tokio::task::JoinHandle<()>>>>,
}

impl MeshController {
    /// Create a new MeshController with stores and sync manager
    pub fn new(
        state: ClusterState,
        self_addr: SocketAddr,
        self_name: &str,
        init_peer: Option<SocketAddr>,
        stores: Arc<StateStores>,
        sync_manager: Arc<MeshSyncManager>,
    ) -> Self {
        Self {
            state,
            self_name: self_name.to_string(),
            self_addr,
            init_peer,
            stores: Some(stores),
            sync_manager: Some(sync_manager),
            sync_connections: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    #[instrument(fields(name = %self.self_name), skip(self, signal))]
    pub async fn event_loop(self, mut signal: tokio::sync::watch::Receiver<()>) -> Result<()> {
        let init_state = self.state.clone();
        let read_state = self.state.clone();
        let mut cnt: u64 = 0;

        // Track retry managers for each peer
        use std::collections::HashMap;
        let mut retry_managers: HashMap<String, RetryManager> = HashMap::new();

        loop {
            log::info!("Round {} Status:{:?}", cnt, read_state.read());

            // Clean up finished sync_stream connections
            {
                let mut connections = self.sync_connections.lock().await;
                connections.retain(|peer_name, handle| {
                    if handle.is_finished() {
                        log::info!(
                            "Sync stream connection to {} has finished, removing",
                            peer_name
                        );
                        false
                    } else {
                        true
                    }
                });
            }

            // Get available peers from cluster state
            let mut map = init_state.read().clone();
            map.retain(|k, v| {
                k.ne(&self.self_name.to_string())
                    && v.status != NodeStatus::Down as i32
                    && v.status != NodeStatus::Leaving as i32
            });

            let peer = if cnt == 0 && map.is_empty() {
                // Only use init_peer if cluster state is empty (no service discovery)
                self.init_peer.map(|init_peer| NodeState {
                    name: "init_peer".to_string(),
                    address: init_peer.to_string(),
                    status: NodeStatus::Suspected as i32,
                    version: 1,
                    metadata: HashMap::new(),
                })
            } else {
                // Use nodes from cluster state (from service discovery or gossip)
                let random_nodes = get_random_values_refs(&map, 1);
                random_nodes.first().map(|&node| node.clone())
            };
            cnt += 1;

            tokio::select! {

                _ = signal.changed() => {
                    log::info!("Gossip app_server {} at {} is shutting down", self.self_name, self.self_addr);
                    break;
                }

                _ = tokio::time::sleep(Duration::from_secs(1)) => {
                    if let Some(peer) = peer {
                        let peer_name = peer.name.clone();

                        // Get or create retry manager for this peer
                        let retry_manager = retry_managers
                            .entry(peer_name.clone())
                            .or_default();

                        // Check if we should retry based on backoff
                        if retry_manager.should_retry() {
                            match self.connect_to_peer(peer.clone()).await {
                                Ok(_) => {
                                    // Success - reset retry state
                                    retry_manager.reset();
                                    log::info!("Successfully connected to peer {}", peer_name);
                                }
                                Err(e) => {
                                    // Failure - record attempt and calculate next delay
                                    retry_manager.record_attempt();
                                    let next_delay = retry_manager.next_delay();
                                    let attempt = retry_manager.attempt_count();
                                    log::warn!(
                                        "Error connecting to peer {} (attempt {}): {}. Next retry in {:?}",
                                        peer_name,
                                        attempt,
                                        e,
                                        next_delay
                                    );
                                }
                            }
                        } else {
                            // Still in backoff period, skip this attempt
                            let next_delay = retry_manager.next_delay();
                            log::debug!(
                                "Skipping connection to peer {} (backoff: {:?} remaining)",
                                peer_name,
                                next_delay
                            );
                        }
                    } else {
                        log::info!("No peer address available to connect");
                    }
                }
            }
        }
        Ok(())
    }

    async fn connect_to_peer(&self, peer: NodeState) -> Result<()> {
        log::info!("Connecting to peer {} at {}", peer.name, peer.address);

        let read_state = self.state.clone();

        // TODO: Maybe we don't need to send the whole state.
        let state_sync = StateSync {
            nodes: read_state.read().values().cloned().collect(),
        };
        let peer_addr = peer.address.parse::<SocketAddr>()?;
        let peer_name = peer.name.clone();
        match try_ping(
            &peer,
            Some(gossip_message::Payload::Ping(Ping {
                state_sync: Some(state_sync),
            })),
        )
        .await
        {
            Ok(node_update) => {
                log::info!("Received NodeUpdate from peer: {:?}", node_update);
                // Update state for Alive or Leaving status
                if node_update.status == NodeStatus::Alive as i32
                    || node_update.status == NodeStatus::Leaving as i32
                {
                    {
                        let mut s = read_state.write();
                        s.entry(node_update.name.clone())
                            .and_modify(|e| e.status = node_update.status)
                            .or_insert(NodeState {
                                name: node_update.name.clone(),
                                address: node_update.address.clone(),
                                status: node_update.status,
                                version: 1,
                                metadata: HashMap::new(),
                            });
                    } // Lock is released here

                    // If node is Alive and we have stores/sync_manager, establish sync_stream connection
                    if node_update.status == NodeStatus::Alive as i32
                        && self.stores.is_some()
                        && self.sync_manager.is_some()
                    {
                        if let Err(e) = self.start_sync_stream_connection(peer.clone()).await {
                            log::warn!("Failed to start sync_stream to {}: {}", peer.name, e);
                            // Connection failure doesn't affect ping flow, will retry in next cycle
                        }
                    }
                }
            }
            Err(e) => {
                log::info!("Failed to connect to peer: {}, now try ping-req", e);
                let mut map = read_state.read().clone();
                map.retain(|k, v| {
                    k.ne(&self.self_name)
                        && k.ne(&peer_name)
                        && v.status == NodeStatus::Alive as i32
                        && v.status != NodeStatus::Leaving as i32
                });
                let random_nodes = get_random_values_refs(&map, 3);
                let mut reachable = false;
                for node in random_nodes {
                    log::info!(
                        "Trying to ping-req node {}, req target: {}",
                        node.address,
                        peer_addr
                    );
                    if try_ping(
                        node,
                        Some(gossip_message::Payload::PingReq(PingReq {
                            node: Some(peer.clone()),
                        })),
                    )
                    .await
                    .is_ok()
                    {
                        reachable = true;
                        break;
                    }
                }
                if !reachable {
                    let mut target = read_state.read().clone();

                    // Broadcast only the unreachable node's status is enough.
                    if let Some(mut unreachable_node) = target.remove(&peer_name) {
                        if unreachable_node.status == NodeStatus::Suspected as i32 {
                            unreachable_node.status = NodeStatus::Down as i32
                        } else {
                            unreachable_node.status = NodeStatus::Suspected as i32
                        }
                        unreachable_node.version += 1;

                        // Broadcast target nodes should include self.
                        let target_nodes: Vec<NodeState> = target
                            .values()
                            .filter(|v| {
                                v.name.ne(&peer_name)
                                    && v.status == NodeStatus::Alive as i32
                                    && v.status != NodeStatus::Leaving as i32
                            })
                            .cloned()
                            .collect();

                        log::info!(
                            "Broadcasting node status to {} alive nodes, new_state: {:?}",
                            target_nodes.len(),
                            unreachable_node
                        );

                        let (success_count, total_count) = broadcast_node_states(
                            vec![unreachable_node],
                            target_nodes,
                            None, // Use default timeout
                        )
                        .await;

                        log::info!(
                            "Broadcast node status: {}/{} successful",
                            success_count,
                            total_count
                        );
                    }
                    return Err(anyhow::anyhow!(
                        "Failed to connect to peer {}: direct ping and ping-req both failed",
                        peer_name
                    ));
                }
            }
        }

        log::info!("Successfully connected to peer {}", peer_addr);

        Ok(())
    }

    /// Determine if this node should initiate sync_stream connection
    /// Use lexicographic ordering to avoid duplicate connections
    fn should_initiate_connection(&self, peer_name: &str) -> bool {
        self.self_name.as_str() < peer_name
    }

    /// Spawn a task to handle sync_stream messages
    fn spawn_sync_stream_handler(
        &self,
        mut incoming_stream: tonic::Streaming<StreamMessage>,
        tx: tokio::sync::mpsc::Sender<StreamMessage>,
        self_name: String,
        peer_name: String,
    ) -> tokio::task::JoinHandle<()> {
        let stores = self.stores.clone();
        let sync_manager = self.sync_manager.clone();

        // Create a span for the spawned task
        let span = tracing::info_span!(
            "sync_stream_handler",
            peer = %peer_name
        );

        tokio::spawn(
            async move {
                use tokio_stream::StreamExt;

                log::info!("Sync stream handler started for peer {}", peer_name);

                let mut sequence: u64 = 0;

                // Send initial heartbeat
                let heartbeat = StreamMessage {
                    message_type: StreamMessageType::Heartbeat as i32,
                    payload: None,
                    sequence,
                    peer_id: self_name.clone(),
                };
                if tx.send(heartbeat).await.is_err() {
                    log::warn!("Failed to send initial heartbeat to {}", peer_name);
                    return;
                }

                // Spawn a task to periodically send incremental updates (client-side sender)
                if let (Some(stores), Some(_sync_manager)) = (&stores, &sync_manager) {
                    use super::{
                        incremental::IncrementalUpdateCollector, service::gossip::IncrementalUpdate,
                        stores::StoreType as LocalStoreType,
                    };

                    let collector = Arc::new(IncrementalUpdateCollector::new(
                        stores.clone(),
                        self_name.clone(),
                    ));
                    let tx_incremental = tx.clone();
                    let self_name_incremental = self_name.clone();
                    let peer_name_incremental = peer_name.clone();

                    tokio::spawn(async move {
                        let mut interval = tokio::time::interval(Duration::from_secs(1));
                        let mut seq_counter: u64 = 1000; // Start from 1000 to avoid conflicts with main sequence

                        loop {
                            interval.tick().await;

                            // Collect all incremental updates
                            let all_updates = collector.collect_all_updates();

                            if !all_updates.is_empty() {
                                for (store_type, updates) in all_updates {
                                    let proto_store_type = match store_type {
                                        LocalStoreType::Membership => {
                                            super::service::gossip::StoreType::Membership as i32
                                        }
                                        LocalStoreType::App => {
                                            super::service::gossip::StoreType::App as i32
                                        }
                                        LocalStoreType::Worker => {
                                            super::service::gossip::StoreType::Worker as i32
                                        }
                                        LocalStoreType::Policy => {
                                            super::service::gossip::StoreType::Policy as i32
                                        }
                                        LocalStoreType::RateLimit => {
                                            super::service::gossip::StoreType::RateLimit as i32
                                        }
                                    };

                                    seq_counter += 1;

                                    let incremental_update = StreamMessage {
                                        message_type: StreamMessageType::IncrementalUpdate as i32,
                                        payload: Some(
                                            super::service::gossip::stream_message::Payload::Incremental(
                                                IncrementalUpdate {
                                                    store: proto_store_type,
                                                    updates: updates.clone(),
                                                    version: 0,
                                                },
                                            ),
                                        ),
                                        sequence: seq_counter,
                                        peer_id: self_name_incremental.clone(),
                                    };

                                    log::info!(
                                        "Sending incremental update to {}: store={:?}, {} updates, versions: {:?}",
                                        peer_name_incremental,
                                        store_type,
                                        updates.len(),
                                        updates.iter().map(|u| (u.key.clone(), u.version)).collect::<Vec<_>>()
                                    );

                                    match tx_incremental.try_send(incremental_update) {
                                        Ok(_) => {
                                            // Mark as sent after successful transmission
                                            collector.mark_sent(store_type, &updates);
                                        }
                                        Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                                            log::debug!(
                                                "Backpressure: channel full, skipping send (will retry next interval)"
                                            );
                                            continue;
                                        }
                                        Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                                            log::warn!(
                                                "Channel closed, stopping incremental update sender"
                                            );
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    });
                }

                // Handle incoming messages
                while let Some(msg_result) = incoming_stream.next().await {
                    match msg_result {
                        Ok(msg) => {
                            sequence += 1;

                            match msg.message_type() {
                                StreamMessageType::IncrementalUpdate => {
                                    log::info!(
                                        "[CLIENT] Received incremental update from {} (seq: {})",
                                        peer_name,
                                        msg.sequence
                                    );

                                    // Apply incremental updates to local stores
                                    if let Some(
                                        super::service::gossip::stream_message::Payload::Incremental(
                                            update,
                                        ),
                                    ) = &msg.payload
                                    {
                                        use super::stores::StoreType as LocalStoreType;

                                        let store_type = LocalStoreType::from_proto(update.store);
                                        log::info!(
                                            "[CLIENT] Applying incremental update from {}: store={:?}, {} updates",
                                            peer_name,
                                            store_type,
                                            update.updates.len()
                                        );

                                        // Apply updates based on store type
                                        for state_update in &update.updates {
                                            match store_type {
                                                LocalStoreType::App => {
                                                    // Deserialize and apply app state
                                                    if let Ok(app_state) = serde_json::from_slice::<
                                                        super::stores::AppState,
                                                    >(
                                                        &state_update.value
                                                    ) {
                                                        if let (Some(ref stores), Some(_)) =
                                                            (&stores, &sync_manager)
                                                        {
                                                            stores.app.insert(
                                                                super::crdt::SKey(
                                                                    app_state.key.clone(),
                                                                ),
                                                                app_state,
                                                                state_update.actor.clone(),
                                                            );
                                                        }
                                                    }
                                                }
                                                LocalStoreType::Membership => {
                                                    // Deserialize and apply membership state
                                                    if let Ok(membership_state) = serde_json::from_slice::<
                                                        super::stores::MembershipState,
                                                    >(
                                                        &state_update.value
                                                    ) {
                                                        if let (Some(ref stores), Some(_)) =
                                                            (&stores, &sync_manager)
                                                        {
                                                            stores.membership.insert(
                                                                super::crdt::SKey(
                                                                    membership_state.name.clone(),
                                                                ),
                                                                membership_state,
                                                                state_update.actor.clone(),
                                                            );
                                                        }
                                                    }
                                                }
                                                LocalStoreType::Worker => {
                                                    // Deserialize and apply worker state
                                                    if let Ok(worker_state) = serde_json::from_slice::<
                                                        super::stores::WorkerState,
                                                    >(
                                                        &state_update.value
                                                    ) {
                                                        if let Some(ref sync_mgr) = sync_manager {
                                                            let actor =
                                                                Some(state_update.actor.clone());
                                                            sync_mgr.apply_remote_worker_state(
                                                                worker_state,
                                                                actor,
                                                            );
                                                        }
                                                    }
                                                }
                                                LocalStoreType::Policy => {
                                                    // Deserialize and apply policy state
                                                    if let Ok(policy_state) = serde_json::from_slice::<
                                                        super::stores::PolicyState,
                                                    >(
                                                        &state_update.value
                                                    ) {
                                                        if let Some(ref sync_mgr) = sync_manager {
                                                            let actor =
                                                                Some(state_update.actor.clone());
                                                            sync_mgr.apply_remote_policy_state(
                                                                policy_state,
                                                                actor,
                                                            );
                                                        }
                                                    }
                                                }
                                                LocalStoreType::RateLimit => {
                                                    // Deserialize and apply rate limit counter
                                                    if let Ok(counter) = serde_json::from_slice::<
                                                        super::crdt::CRDTPNCounter,
                                                    >(
                                                        &state_update.value
                                                    ) {
                                                        if let Some(ref sync_mgr) = sync_manager {
                                                            let sync_counter =
                                                                super::crdt::SyncPNCounter::new();
                                                            sync_counter.merge(&counter);
                                                            sync_mgr.apply_remote_rate_limit_counter(
                                                                state_update.key.clone(),
                                                                &sync_counter,
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Send ACK
                                    let ack = StreamMessage {
                                        message_type: StreamMessageType::Ack as i32,
                                        payload: Some(StreamPayload::Ack(
                                            super::service::gossip::StreamAck {
                                                sequence: msg.sequence,
                                                success: true,
                                                error_message: String::new(),
                                            },
                                        )),
                                        sequence,
                                        peer_id: self_name.clone(),
                                    };
                                    if tx.send(ack).await.is_err() {
                                        log::warn!("Failed to send ACK to {}", peer_name);
                                        break;
                                    }
                                }
                                StreamMessageType::SnapshotChunk => {
                                    log::info!(
                                        "Received snapshot chunk from {} (seq: {})",
                                        peer_name,
                                        msg.sequence
                                    );
                                    // Server side handles snapshot assembly
                                    // Send ACK
                                    let ack = StreamMessage {
                                        message_type: StreamMessageType::Ack as i32,
                                        payload: Some(StreamPayload::Ack(
                                            super::service::gossip::StreamAck {
                                                sequence: msg.sequence,
                                                success: true,
                                                error_message: String::new(),
                                            },
                                        )),
                                        sequence,
                                        peer_id: self_name.clone(),
                                    };
                                    if tx.send(ack).await.is_err() {
                                        log::warn!("Failed to send ACK to {}", peer_name);
                                        break;
                                    }
                                }
                                StreamMessageType::Heartbeat => {
                                    log::trace!("Received heartbeat from {}", peer_name);
                                    // Send heartbeat back
                                    let heartbeat = StreamMessage {
                                        message_type: StreamMessageType::Heartbeat as i32,
                                        payload: None,
                                        sequence,
                                        peer_id: self_name.clone(),
                                    };
                                    if tx.send(heartbeat).await.is_err() {
                                        log::warn!("Failed to send heartbeat to {}", peer_name);
                                        break;
                                    }
                                }
                                StreamMessageType::SnapshotRequest => {
                                    log::info!("Received snapshot request from {}", peer_name);
                                    // Handle snapshot request - generate and send snapshot using GossipService
                                    if let (Some(ref stores), Some(ref sync_manager)) =
                                        (&stores, &sync_manager)
                                    {
                                        if let Some(StreamPayload::SnapshotRequest(req)) = &msg.payload
                                        {
                                            use std::net::SocketAddr;

                                            use super::{
                                                ping_server::GossipService,
                                                stores::StoreType as LocalStoreType,
                                            };

                                            let store_type = LocalStoreType::from_proto(req.store);
                                            log::info!(
                                                "Generating snapshot for store {:?}",
                                                store_type
                                            );

                                            // Create a temporary GossipService to generate snapshot chunks
                                            let service = GossipService::new(
                                                Arc::new(parking_lot::RwLock::new(
                                                    std::collections::BTreeMap::new(),
                                                )),
                                                SocketAddr::from(([0, 0, 0, 0], 0)),
                                                &self_name,
                                            )
                                            .with_stores(stores.clone())
                                            .with_sync_manager(sync_manager.clone());

                                            let chunks =
                                                service.create_snapshot_chunks(store_type, 100).await;
                                            let total_chunks = chunks.len() as u64;

                                            log::info!(
                                                "Sending {} snapshot chunks for store {:?}",
                                                total_chunks,
                                                store_type
                                            );

                                            for (idx, chunk) in chunks.into_iter().enumerate() {
                                                let snapshot_chunk = StreamMessage {
                                                    message_type: StreamMessageType::SnapshotChunk
                                                        as i32,
                                                    payload: Some(StreamPayload::SnapshotChunk(chunk)),
                                                    sequence: sequence + idx as u64,
                                                    peer_id: self_name.clone(),
                                                };

                                                if tx.send(snapshot_chunk).await.is_err() {
                                                    log::warn!(
                                                        "Failed to send snapshot chunk {} to {}",
                                                        idx,
                                                        peer_name
                                                    );
                                                    break;
                                                }
                                            }

                                            log::info!(
                                                "Sent all snapshot chunks for store {:?} to {}",
                                                store_type,
                                                peer_name
                                            );
                                        }
                                    }
                                }
                                StreamMessageType::Ack => {
                                    log::trace!(
                                        "Received ACK from {} (seq: {})",
                                        peer_name,
                                        msg.sequence
                                    );
                                }
                                StreamMessageType::Nack => {
                                    log::warn!(
                                        "Received NACK from {} (seq: {})",
                                        peer_name,
                                        msg.sequence
                                    );
                                }
                                _ => {
                                    log::debug!(
                                        "Received message type {:?} from {}",
                                        msg.message_type,
                                        peer_name
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            log::error!("Error receiving from sync_stream with {}: {}", peer_name, e);
                            break;
                        }
                    }
                }

                log::info!("Sync stream handler stopped for peer {}", peer_name);
            }
            .instrument(span),
        )
    }

    /// Start a sync_stream connection to a peer
    async fn start_sync_stream_connection(&self, peer: NodeState) -> Result<()> {
        let peer_name = peer.name.clone();
        let peer_addr = peer.address.clone();

        // Check if connection already exists
        {
            let connections = self.sync_connections.lock().await;
            if connections.contains_key(&peer_name) {
                log::debug!("Sync stream connection to {} already exists", peer_name);
                return Ok(());
            }
        }

        // Check if we should initiate connection (avoid duplicates)
        if !self.should_initiate_connection(&peer_name) {
            log::debug!(
                "Skipping sync_stream to {} (peer should initiate)",
                peer_name
            );
            return Ok(());
        }

        log::info!(
            "Starting sync_stream connection to peer {} at address {}",
            peer_name,
            peer_addr
        );

        // Connect to peer's gRPC service
        let connect_url = format!("http://{}", peer_addr);
        log::info!("Connecting to URL: {}", connect_url);
        let mut client = GossipClient::connect(connect_url).await.map_err(|e| {
            log::warn!(
                "Failed to connect to peer {} for sync_stream: {}",
                peer_name,
                e
            );
            anyhow::anyhow!("Connection failed: {}", e)
        })?;

        // Create bidirectional stream
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamMessage>(128);
        let outgoing_stream = tokio_stream::wrappers::ReceiverStream::new(rx);

        let response = client.sync_stream(outgoing_stream).await.map_err(|e| {
            log::error!("Failed to establish sync_stream with {}: {}", peer_name, e);
            anyhow::anyhow!("sync_stream RPC failed: {}", e)
        })?;

        let incoming_stream = response.into_inner();

        // Spawn task to handle the bidirectional stream
        let self_name = self.self_name.clone();
        let peer_name_clone = peer_name.clone();

        let handle =
            self.spawn_sync_stream_handler(incoming_stream, tx, self_name, peer_name_clone);

        // Store the task handle
        {
            let mut connections = self.sync_connections.lock().await;
            connections.insert(peer_name.clone(), handle);
        }

        log::info!("Sync stream connection to {} established", peer_name);
        Ok(())
    }
}

// TODO: Support weighted random selection. e.g. nodes in INIT state should be more likely to be selected.
fn get_random_values_refs<K, V>(map: &BTreeMap<K, V>, k: usize) -> Vec<&V> {
    let values: Vec<&V> = map.values().collect();

    if k >= values.len() {
        let mut all_values = values;
        all_values.shuffle(&mut rand::rng());
        return all_values;
    }

    let mut rng = rand::rng();

    values.choose_multiple(&mut rng, k).cloned().collect()
}
