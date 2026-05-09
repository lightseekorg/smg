use std::{
    collections::{BTreeMap, HashMap},
    net::SocketAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::Result;
use rand::seq::{IndexedRandom, SliceRandom};
use tokio::sync::{mpsc, watch, Mutex};
use tonic::transport::{ClientTlsConfig, Endpoint};
use tracing as log;
use tracing::{instrument, Instrument};

use super::{
    flow_control::{MessageSizeValidator, RetryManager, MAX_MESSAGE_SIZE},
    metrics,
    mtls::MTLSManager,
    service::{
        broadcast_node_states,
        gossip::{
            gossip_client::GossipClient, gossip_message, stream_message::Payload as StreamPayload,
            NodeState, NodeStatus, Ping, PingReq, StateSync, StreamMessage, StreamMessageType,
        },
        try_ping, ClusterState,
    },
    sync_stream_messages,
};

pub struct MeshController {
    state: ClusterState,
    self_name: String,
    self_addr: SocketAddr,
    init_peer: Option<SocketAddr>,
    mtls_manager: Option<Arc<MTLSManager>>,
    sync_connections: Arc<Mutex<HashMap<String, tokio::task::JoinHandle<()>>>>,
    /// Current stream round batch, drained once per round from MeshKV.
    /// Per-peer senders read this and filter targeted entries to their
    /// peer; drain entries are broadcast to every connected peer.
    current_stream_batch: Arc<parking_lot::RwLock<Arc<crate::kv::RoundBatch>>>,
    mesh_kv: Arc<crate::kv::MeshKV>,
}

impl MeshController {
    pub fn new(
        state: ClusterState,
        self_addr: SocketAddr,
        self_name: &str,
        init_peer: Option<SocketAddr>,
        mtls_manager: Option<Arc<MTLSManager>>,
        mesh_kv: Arc<crate::kv::MeshKV>,
    ) -> Self {
        Self {
            state,
            self_name: self_name.to_string(),
            self_addr,
            init_peer,
            mtls_manager,
            sync_connections: Arc::new(Mutex::new(HashMap::new())),
            current_stream_batch: Arc::new(parking_lot::RwLock::new(Arc::new(
                crate::kv::RoundBatch::default(),
            ))),
            mesh_kv,
        }
    }

    /// Get a handle to the shared stream RoundBatch. Used by GossipService
    /// so server-side sync_stream handlers see the same drained stream
    /// entries as client-side handlers.
    pub fn current_stream_batch(&self) -> Arc<parking_lot::RwLock<Arc<crate::kv::RoundBatch>>> {
        self.current_stream_batch.clone()
    }

    #[instrument(fields(name = %self.self_name), skip(self, signal))]
    pub async fn event_loop(self, mut signal: watch::Receiver<bool>) -> Result<()> {
        let init_state = self.state.clone();
        let read_state = self.state.clone();
        let mut cnt: u64 = 0;
        let mut retry_managers: HashMap<String, RetryManager> = HashMap::new();

        loop {
            log::info!("Round {} Status:{:?}", cnt, read_state.read());

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

            let mut map = init_state.read().clone();
            map.retain(|k, v| {
                k.ne(&self.self_name)
                    && v.status != NodeStatus::Down as i32
                    && v.status != NodeStatus::Leaving as i32
            });

            let peer = if cnt == 0 && map.is_empty() {
                self.init_peer.map(|init_peer| NodeState {
                    name: "init_peer".to_string(),
                    address: init_peer.to_string(),
                    status: NodeStatus::Suspected as i32,
                    version: 1,
                    metadata: HashMap::new(),
                })
            } else {
                get_random_values_refs(&map, 1)
                    .first()
                    .map(|&node| node.clone())
            };
            cnt += 1;

            if cnt.is_multiple_of(5) {
                self.mesh_kv.chunk_assembler().gc(Duration::from_secs(30));
            }

            if cnt.is_multiple_of(60) {
                let removed = self.mesh_kv.gc_tombstones();
                if removed > 0 {
                    log::info!("GC: removed {removed} MeshKV tombstoned CRDT metadata entries");
                }
                retry_managers.retain(|peer_name, _| map.contains_key(peer_name));
            }

            let stream_batch = self.mesh_kv.collect_round_batch();
            *self.current_stream_batch.write() = Arc::new(stream_batch);

            tokio::select! {
                _ = signal.changed() => {
                    log::info!("Gossip app_server {} at {} is shutting down", self.self_name, self.self_addr);
                    break;
                }

                () = tokio::time::sleep(Duration::from_secs(1)) => {
                    if let Some(peer) = peer {
                        let peer_name = peer.name.clone();
                        let retry_manager = retry_managers
                            .entry(peer_name.clone())
                            .or_default();

                        if retry_manager.should_retry() {
                            match self.connect_to_peer(peer.clone()).await {
                                Ok(()) => {
                                    retry_manager.reset();
                                    log::info!("Successfully connected to peer {}", peer_name);
                                }
                                Err(e) => {
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
            self.mtls_manager.clone(),
        )
        .await
        {
            Ok(node_update) => {
                log::info!("Received NodeUpdate from peer: {:?}", node_update);
                if node_update.status == NodeStatus::Alive as i32
                    || node_update.status == NodeStatus::Leaving as i32
                {
                    let updated_peer = {
                        let mut s = read_state.write();
                        let entry = s
                            .entry(node_update.name.clone())
                            .and_modify(|e| {
                                e.status = node_update.status;
                                e.address.clone_from(&node_update.address);
                            })
                            .or_insert_with(|| NodeState {
                                name: node_update.name.clone(),
                                address: node_update.address.clone(),
                                status: node_update.status,
                                version: 1,
                                metadata: HashMap::new(),
                            });
                        entry.clone()
                    };

                    if node_update.status == NodeStatus::Alive as i32 {
                        if let Err(e) = self
                            .start_sync_stream_connection(updated_peer.clone())
                            .await
                        {
                            log::warn!(
                                "Failed to start sync_stream to {}: {}",
                                updated_peer.name,
                                e
                            );
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
                        self.mtls_manager.clone(),
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

                    if let Some(mut unreachable_node) = target.remove(&peer_name) {
                        if unreachable_node.status == NodeStatus::Suspected as i32 {
                            unreachable_node.status = NodeStatus::Down as i32;
                        } else {
                            unreachable_node.status = NodeStatus::Suspected as i32;
                        }
                        unreachable_node.version += 1;

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

                        let (success_count, total_count) =
                            broadcast_node_states(vec![unreachable_node], target_nodes, None).await;

                        log::info!(
                            "Broadcast node status: {}/{} successful",
                            success_count,
                            total_count
                        );
                    }
                    return Err(anyhow::anyhow!(
                        "Failed to connect to peer {peer_name}: direct ping and ping-req both failed"
                    ));
                }
            }
        }

        log::info!("Successfully connected to peer {}", peer_addr);
        Ok(())
    }

    fn should_initiate_connection(&self, peer_name: &str) -> bool {
        self.self_name.as_str() < peer_name
    }

    fn spawn_sync_stream_handler(
        &self,
        mut incoming_stream: tonic::Streaming<StreamMessage>,
        tx: mpsc::Sender<StreamMessage>,
        self_name: String,
        peer_name: String,
    ) -> tokio::task::JoinHandle<()> {
        let sync_connections = self.sync_connections.clone();
        let current_stream_batch = self.current_stream_batch.clone();
        let mesh_kv = self.mesh_kv.clone();

        log::debug!(
            peer = %peer_name,
            "spawn_sync_stream_handler called"
        );

        let span = tracing::info_span!("sync_stream_handler", peer = %peer_name);

        #[expect(
            clippy::disallowed_methods,
            reason = "handle is stored in sync_connections for lifecycle tracking"
        )]
        tokio::spawn(
            async move {
                use tokio_stream::StreamExt;

                let sequence = Arc::new(AtomicU64::new(0));
                if tx
                    .send(sync_stream_messages::heartbeat(&self_name, &sequence))
                    .await
                    .is_err()
                {
                    log::warn!("Failed to send initial heartbeat to {}", peer_name);
                    return;
                }

                let sender_handle = {
                    let tx_sender = tx.clone();
                    let self_name_sender = self_name.clone();
                    let peer_name_sender = peer_name.clone();
                    let sequence_sender = sequence.clone();
                    let batch_handle = current_stream_batch.clone();
                    let mesh_kv_sender = mesh_kv.clone();
                    let size_validator = MessageSizeValidator::default();

                    #[expect(
                        clippy::disallowed_methods,
                        reason = "sender handle is aborted when sync_stream exits"
                    )]
                    tokio::spawn(async move {
                        let mut interval = tokio::time::interval(Duration::from_secs(1));
                        let mut last_stream_batch: Option<Arc<crate::kv::RoundBatch>> = None;
                        let mut last_crdt_generation: Option<u64> = None;

                        loop {
                            interval.tick().await;
                            let round_start = std::time::Instant::now();

                            let crdt_generation = mesh_kv_sender.crdt_generation();
                            if last_crdt_generation != Some(crdt_generation) {
                                last_crdt_generation = Some(crdt_generation);
                                if let Some(msg) = sync_stream_messages::crdt_batch_message(
                                    &mesh_kv_sender,
                                    &self_name_sender,
                                    &sequence_sender,
                                ) {
                                    let batch_size = match &msg.payload {
                                        Some(StreamPayload::CrdtBatch(batch)) => {
                                            sync_stream_messages::crdt_batch_encoded_len(batch)
                                        }
                                        _ => 0,
                                    };
                                    if let Err(err) = size_validator.validate(batch_size) {
                                        log::warn!(
                                            peer = %peer_name_sender,
                                            %err,
                                            max_bytes = size_validator.max_size(),
                                            "CRDT batch too large to send"
                                        );
                                    } else if tx_sender.try_send(msg).is_err() {
                                        log::debug!(
                                            peer = %peer_name_sender,
                                            "CRDT batch dropped on backpressure"
                                        );
                                    }
                                }
                            }

                            let stream_batch = batch_handle.read().clone();
                            let fresh_batch = last_stream_batch
                                .as_ref()
                                .is_none_or(|last| !Arc::ptr_eq(last, &stream_batch));
                            if fresh_batch {
                                last_stream_batch = Some(stream_batch.clone());
                                for msg in sync_stream_messages::build_stream_messages(
                                    &stream_batch,
                                    Some(&peer_name_sender),
                                    &self_name_sender,
                                    &sequence_sender,
                                ) {
                                    if tx_sender.try_send(msg).is_err() {
                                        log::debug!(
                                            peer = %peer_name_sender,
                                            "stream batch dropped on backpressure"
                                        );
                                        break;
                                    }
                                }
                            }

                            metrics::record_sync_round_duration(
                                &peer_name_sender,
                                round_start.elapsed(),
                            );
                        }
                    })
                };

                const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(60);
                loop {
                    match tokio::time::timeout(STREAM_IDLE_TIMEOUT, incoming_stream.next()).await {
                        Ok(Some(Ok(msg))) => {
                            sequence.fetch_add(1, Ordering::Relaxed);

                            match msg.message_type() {
                                StreamMessageType::CrdtBatch => {
                                    if let Some(StreamPayload::CrdtBatch(batch)) = &msg.payload {
                                        sync_stream_messages::apply_crdt_batch(&mesh_kv, batch);
                                    }
                                    let ack = sync_stream_messages::ack(
                                        &self_name,
                                        &sequence,
                                        msg.sequence,
                                        true,
                                        String::new(),
                                    );
                                    if tx.send(ack).await.is_err() {
                                        break;
                                    }
                                }
                                StreamMessageType::StreamBatch => {
                                    if let Some(StreamPayload::StreamBatch(batch)) = msg.payload {
                                        sync_stream_messages::dispatch_stream_payload(
                                            &mesh_kv,
                                            &msg.peer_id,
                                            batch,
                                        );
                                    }
                                    let ack = sync_stream_messages::ack(
                                        &self_name,
                                        &sequence,
                                        msg.sequence,
                                        true,
                                        String::new(),
                                    );
                                    if tx.send(ack).await.is_err() {
                                        break;
                                    }
                                }
                                StreamMessageType::Heartbeat => {
                                    if tx
                                        .send(sync_stream_messages::heartbeat(
                                            &self_name, &sequence,
                                        ))
                                        .await
                                        .is_err()
                                    {
                                        break;
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
                            }
                        }
                        Ok(Some(Err(e))) => {
                            log::error!(
                                "Error receiving from sync_stream with {}: {}",
                                peer_name,
                                e
                            );
                            break;
                        }
                        Ok(None) => break,
                        Err(_) => {
                            log::warn!(
                                "sync_stream to {peer_name} idle timeout ({STREAM_IDLE_TIMEOUT:?})"
                            );
                            break;
                        }
                    }
                }

                sender_handle.abort();
                let _ = sender_handle.await;
                sync_connections.lock().await.remove(&peer_name);
                log::debug!(peer = %peer_name, "sync_stream_handler exited");
            }
            .instrument(span),
        )
    }

    async fn start_sync_stream_connection(&self, peer: NodeState) -> Result<()> {
        let peer_name = peer.name.clone();
        let peer_addr = peer.address.clone();

        {
            let connections = self.sync_connections.lock().await;
            if connections.contains_key(&peer_name) {
                log::debug!("Sync stream connection to {} already exists", peer_name);
                return Ok(());
            }
        }

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

        let connect_url = if self.mtls_manager.is_some() {
            format!("https://{peer_addr}")
        } else {
            format!("http://{peer_addr}")
        };
        let mut endpoint = Endpoint::from_shared(connect_url.clone())
            .map_err(|e| anyhow::anyhow!("Invalid peer endpoint {connect_url}: {e}"))?;

        if let Some(mtls_manager) = self.mtls_manager.clone() {
            let tls_domain = endpoint
                .uri()
                .host()
                .map(str::to_owned)
                .unwrap_or_else(|| peer_name.clone());
            let ca_certificate = mtls_manager
                .load_ca_certificate()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to load mTLS CA certificate: {e}"))?;

            endpoint = endpoint
                .tls_config(
                    ClientTlsConfig::new()
                        .domain_name(tls_domain)
                        .ca_certificate(ca_certificate),
                )
                .map_err(|e| anyhow::anyhow!("Failed to configure TLS endpoint: {e}"))?;
        }

        let channel = endpoint.connect().await.map_err(|e| {
            log::warn!(
                "Failed to connect to peer {} for sync_stream: {}",
                peer_name,
                e
            );
            anyhow::anyhow!("Connection failed: {e}")
        })?;
        let mut client = GossipClient::new(channel)
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE)
            .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
            .send_compressed(tonic::codec::CompressionEncoding::Gzip);

        let (tx, rx) = mpsc::channel::<StreamMessage>(128);
        let outgoing_stream = tokio_stream::wrappers::ReceiverStream::new(rx);

        let response = client.sync_stream(outgoing_stream).await.map_err(|e| {
            log::error!("Failed to establish sync_stream with {}: {}", peer_name, e);
            anyhow::anyhow!("sync_stream RPC failed: {e}")
        })?;

        let incoming_stream = response.into_inner();
        let handle = self.spawn_sync_stream_handler(
            incoming_stream,
            tx,
            self.self_name.clone(),
            peer_name.clone(),
        );

        self.sync_connections
            .lock()
            .await
            .insert(peer_name.clone(), handle);

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

    values.choose_multiple(&mut rng, k).copied().collect()
}
