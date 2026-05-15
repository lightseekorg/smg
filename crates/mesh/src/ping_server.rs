use std::{net::SocketAddr, pin::Pin, sync::Arc, time::Duration};

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
    flow_control::MAX_MESSAGE_SIZE,
    metrics::{record_ack, record_nack, record_peer_reconnect, update_peer_connections},
    mtls::MTLSManager,
    partition::PartitionDetector,
    service::{
        gossip::{
            self,
            gossip_server::{Gossip, GossipServer},
            GossipMessage, NodeState, NodeStatus, NodeUpdate, PingReq, StreamMessage,
            StreamMessageType,
        },
        try_ping, ClusterState,
    },
};

#[derive(Debug)]
pub struct GossipService {
    state: ClusterState,
    listen_addr: SocketAddr,
    advertise_addr: SocketAddr,
    self_name: String,
    partition_detector: Option<Arc<PartitionDetector>>,
    mtls_manager: Option<Arc<MTLSManager>>,
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
            partition_detector: None,
            mtls_manager: None,
            current_stream_batch: None,
            mesh_kv: None,
        }
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
        let mesh_kv = self.mesh_kv.clone();

        const CHANNEL_CAPACITY: usize = 128;
        let (tx, rx) = mpsc::channel::<Result<StreamMessage, Status>>(CHANNEL_CAPACITY);

        // Remote peer identity, learned from the first inbound message and
        // used by the sender to filter targeted_entries.
        let learned_peer: Arc<parking_lot::RwLock<Option<String>>> =
            Arc::new(parking_lot::RwLock::new(None));

        // Server-side stream sender: periodically emit fresh stream batches
        // (broadcast drain_entries + targeted entries addressed to the
        // learned peer). Skipped when no current_stream_batch is attached.
        let sender_handle = if let Some(stream_batch_handle) = self.current_stream_batch.clone() {
            let tx_sender = tx.clone();
            let self_name_sender = self_name.clone();
            let learned_peer_sender = learned_peer.clone();
            #[expect(
                clippy::disallowed_methods,
                reason = "server-side sender bound to sync_stream lifetime; terminates when channel closes or handle is aborted on disconnect"
            )]
            Some(tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(1));
                let mut sequence_counter: u64 = 0;
                let mut last_stream_batch: Option<Arc<crate::kv::RoundBatch>> = None;

                loop {
                    interval.tick().await;

                    let stream_batch = stream_batch_handle.read().clone();
                    let fresh = last_stream_batch
                        .as_ref()
                        .is_none_or(|last| !Arc::ptr_eq(last, &stream_batch));
                    if !fresh {
                        continue;
                    }
                    last_stream_batch = Some(stream_batch.clone());

                    let peer_for_targeted = learned_peer_sender.read().clone();
                    let has_targeted = peer_for_targeted.as_ref().is_some_and(|p| {
                        stream_batch.targeted_entries.iter().any(|(t, _, _)| t == p)
                    });
                    if stream_batch.drain_entries.is_empty() && !has_targeted {
                        continue;
                    }

                    let mut entries = Vec::new();
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
                    if entries.is_empty() {
                        continue;
                    }

                    for batch in build_stream_batches(
                        entries,
                        DEFAULT_MAX_CHUNKS_PER_BATCH,
                        MAX_STREAM_CHUNK_BYTES,
                    ) {
                        sequence_counter += 1;
                        let msg = StreamMessage {
                            message_type: StreamMessageType::StreamBatch as i32,
                            payload: Some(gossip::stream_message::Payload::StreamBatch(batch)),
                            sequence: sequence_counter,
                            peer_id: self_name_sender.clone(),
                        };
                        match tx_sender.try_send(Ok(msg)) {
                            Ok(()) => {}
                            Err(mpsc::error::TrySendError::Full(_)) => {
                                log::debug!("server-side stream batch dropped on backpressure");
                                break;
                            }
                            Err(mpsc::error::TrySendError::Closed(_)) => return,
                        }
                    }
                }
            }))
        } else {
            None
        };

        let learned_peer_inbound = learned_peer.clone();
        #[expect(
            clippy::disallowed_methods,
            reason = "server-side inbound handler bound to sync_stream lifetime; terminates when the stream closes"
        )]
        tokio::spawn(async move {
            // Close the stream if no inbound message arrives within
            // this window — protects against idle clients pinning the
            // server-side task and mpsc channel indefinitely.
            const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(60);

            let mut peer_id = String::new();
            update_peer_connections(&peer_id, true);
            let mut sequence: u64 = 0;

            loop {
                let msg = match tokio::time::timeout(STREAM_IDLE_TIMEOUT, incoming.next()).await {
                    Ok(Some(Ok(msg))) => msg,
                    Ok(Some(Err(e))) => {
                        log::error!("Error receiving stream message: {}", e);
                        break;
                    }
                    Ok(None) => break,
                    Err(_) => {
                        log::warn!(
                            peer = %peer_id,
                            "sync_stream idle timeout ({STREAM_IDLE_TIMEOUT:?}) — closing"
                        );
                        break;
                    }
                };

                // Bind peer_id to the first non-empty inbound id. A later
                // frame whose msg.peer_id (empty or otherwise) doesn't
                // match is treated as identity change and closes the
                // stream. Pre-mTLS-binding defence; mTLS-derived
                // identity is the authoritative long-term fix.
                if peer_id.is_empty() {
                    if !msg.peer_id.is_empty() {
                        peer_id = msg.peer_id.clone();
                        update_peer_connections(&peer_id, true);
                        *learned_peer_inbound.write() = Some(peer_id.clone());
                    }
                } else if msg.peer_id != peer_id {
                    log::warn!(
                        expected_peer_id = %peer_id,
                        received_peer_id = %msg.peer_id,
                        "peer_id changed mid-stream; closing sync_stream"
                    );
                    break;
                }
                sequence = sequence.max(msg.sequence);

                match msg.message_type() {
                    StreamMessageType::Heartbeat => {
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
                    StreamMessageType::Ack => {
                        if let Some(gossip::stream_message::Payload::Ack(ack)) = &msg.payload {
                            record_ack(&peer_id, ack.success);
                        }
                    }
                    StreamMessageType::Nack => record_nack(&peer_id),
                    StreamMessageType::StreamBatch => {
                        if let (
                            Some(mesh_kv),
                            Some(gossip::stream_message::Payload::StreamBatch(batch)),
                        ) = (&mesh_kv, msg.payload)
                        {
                            dispatch_stream_batch(mesh_kv, &msg.peer_id, batch.entries);
                        }
                    }
                    StreamMessageType::IncrementalUpdate
                    | StreamMessageType::SnapshotRequest
                    | StreamMessageType::SnapshotChunk
                    | StreamMessageType::SnapshotComplete => {
                        log::debug!(
                            peer = %peer_id,
                            message_type = ?msg.message_type(),
                            "ignoring v1 wire message (state-sync removed)",
                        );
                    }
                }
            }

            update_peer_connections(&peer_id, false);
            record_peer_reconnect(&peer_id);
            if let Some(handle) = sender_handle {
                handle.abort();
            }
        });

        let output_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(
            Box::pin(output_stream) as Self::SyncStreamStream
        ))
    }
}
