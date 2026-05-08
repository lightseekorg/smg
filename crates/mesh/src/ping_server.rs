use std::{
    net::SocketAddr,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
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
    flow_control::{MessageSizeValidator, MAX_MESSAGE_SIZE},
    metrics::{record_ack, record_nack, update_peer_connections},
    mtls::MTLSManager,
    partition::PartitionDetector,
    service::{
        gossip::{
            self,
            gossip_server::{Gossip, GossipServer},
            stream_message::Payload as StreamPayload,
            GossipMessage, NodeStatus, NodeUpdate, PingReq, StreamMessage, StreamMessageType,
        },
        try_ping, ClusterState,
    },
    stream_sync,
};

#[derive(Debug)]
pub struct GossipService {
    state: ClusterState,
    listen_addr: SocketAddr,
    advertise_addr: SocketAddr,
    self_name: String,
    _partition_detector: Option<Arc<PartitionDetector>>,
    mtls_manager: Option<Arc<MTLSManager>>,
    /// Shared stream batch drained once per round by MeshController.
    current_stream_batch: Option<Arc<parking_lot::RwLock<Arc<crate::kv::RoundBatch>>>>,
    /// Node-wide MeshKV handle used for CRDT merge and stream dispatch.
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
            _partition_detector: None,
            mtls_manager: None,
            current_stream_batch: None,
            mesh_kv: None,
        }
    }

    /// Attach the shared stream RoundBatch reference. Server-side handlers
    /// emit broadcast drain entries plus targeted entries whose target
    /// matches the remote peer learned from inbound messages.
    pub fn with_current_stream_batch(
        mut self,
        current_stream_batch: Arc<parking_lot::RwLock<Arc<crate::kv::RoundBatch>>>,
    ) -> Self {
        self.current_stream_batch = Some(current_stream_batch);
        self
    }

    pub fn with_mesh_kv(mut self, mesh_kv: Arc<crate::kv::MeshKV>) -> Self {
        self.mesh_kv = Some(mesh_kv);
        self
    }

    pub fn with_partition_detector(mut self, partition_detector: Arc<PartitionDetector>) -> Self {
        self._partition_detector = Some(partition_detector);
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

    fn merge_state(&self, incoming_nodes: Vec<gossip::NodeState>) -> bool {
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

        let (tx, rx) = mpsc::channel::<Result<StreamMessage, Status>>(128);
        let size_validator = MessageSizeValidator::default();

        let learned_peer: Arc<parking_lot::RwLock<Option<String>>> =
            Arc::new(parking_lot::RwLock::new(None));

        let sender_handle = if let (Some(batch_handle), Some(mesh_kv)) =
            (self.current_stream_batch.clone(), self.mesh_kv.clone())
        {
            let tx_sender = tx.clone();
            let self_name_sender = self_name.clone();
            let learned_peer_sender = learned_peer.clone();

            #[expect(
                clippy::disallowed_methods,
                reason = "server-side sender runs for the lifetime of sync_stream and is aborted on stream close"
            )]
            Some(tokio::spawn(async move {
                let sequence = AtomicU64::new(0);
                let mut interval = tokio::time::interval(Duration::from_secs(1));
                let mut last_stream_batch: Option<Arc<crate::kv::RoundBatch>> = None;
                let mut last_crdt_generation: Option<u64> = None;

                loop {
                    interval.tick().await;

                    let crdt_generation = mesh_kv.crdt_generation();
                    if last_crdt_generation != Some(crdt_generation) {
                        last_crdt_generation = Some(crdt_generation);
                        if let Some(msg) =
                            stream_sync::crdt_batch_message(&mesh_kv, &self_name_sender, &sequence)
                        {
                            let batch_size = match &msg.payload {
                                Some(StreamPayload::CrdtBatch(batch)) => {
                                    stream_sync::crdt_batch_encoded_len(batch)
                                }
                                _ => 0,
                            };
                            if let Err(err) = size_validator.validate(batch_size) {
                                log::warn!(
                                    %err,
                                    max_bytes = size_validator.max_size(),
                                    "server-side CRDT batch too large to send"
                                );
                            } else if tx_sender.try_send(Ok(msg)).is_err() {
                                log::debug!("server-side CRDT batch dropped on backpressure");
                            }
                        }
                    }

                    let stream_batch = batch_handle.read().clone();
                    let fresh_batch = last_stream_batch
                        .as_ref()
                        .is_none_or(|last| !Arc::ptr_eq(last, &stream_batch));
                    if fresh_batch {
                        last_stream_batch = Some(stream_batch.clone());
                        let peer_for_targeted = learned_peer_sender.read().clone();
                        for msg in stream_sync::build_stream_messages(
                            &stream_batch,
                            peer_for_targeted.as_deref(),
                            &self_name_sender,
                            &sequence,
                        ) {
                            if tx_sender.try_send(Ok(msg)).is_err() {
                                log::debug!("server-side stream batch dropped on backpressure");
                                break;
                            }
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
            reason = "server-side stream handler runs for the lifetime of the gRPC stream"
        )]
        tokio::spawn(async move {
            let sequence = AtomicU64::new(0);
            let mut peer_id = String::new();
            let mut peer_learned = false;

            const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(60);
            loop {
                match tokio::time::timeout(STREAM_IDLE_TIMEOUT, incoming.next()).await {
                    Ok(Some(Ok(msg))) => {
                        sequence.fetch_add(1, Ordering::Relaxed);

                        if !peer_learned && !msg.peer_id.is_empty() {
                            let mut learned = learned_peer_inbound.write();
                            match learned.as_ref() {
                                None => {
                                    *learned = Some(msg.peer_id.clone());
                                    peer_id.clone_from(&msg.peer_id);
                                    peer_learned = true;
                                    update_peer_connections(&peer_id, true);
                                }
                                Some(existing) if existing == &msg.peer_id => {
                                    peer_id.clone_from(existing);
                                    peer_learned = true;
                                    update_peer_connections(&peer_id, true);
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
                            log::warn!(
                                expected_peer_id = %peer_id,
                                received_peer_id = %msg.peer_id,
                                "peer_id changed mid-stream; closing sync_stream"
                            );
                            break;
                        }

                        match msg.message_type() {
                            StreamMessageType::CrdtBatch => {
                                if let (Some(mesh_kv), Some(StreamPayload::CrdtBatch(batch))) =
                                    (&mesh_kv, &msg.payload)
                                {
                                    stream_sync::apply_crdt_batch(mesh_kv, batch);
                                }
                                let ack = stream_sync::ack(
                                    &self_name,
                                    &sequence,
                                    msg.sequence,
                                    true,
                                    String::new(),
                                );
                                record_ack(&peer_id, true);
                                if tx.send(Ok(ack)).await.is_err() {
                                    break;
                                }
                            }
                            StreamMessageType::StreamBatch => {
                                if let (Some(mesh_kv), Some(StreamPayload::StreamBatch(batch))) =
                                    (&mesh_kv, msg.payload)
                                {
                                    stream_sync::dispatch_stream_payload(
                                        mesh_kv,
                                        &msg.peer_id,
                                        batch,
                                    );
                                }
                                let ack = stream_sync::ack(
                                    &self_name,
                                    &sequence,
                                    msg.sequence,
                                    true,
                                    String::new(),
                                );
                                record_ack(&peer_id, true);
                                if tx.send(Ok(ack)).await.is_err() {
                                    break;
                                }
                            }
                            StreamMessageType::Heartbeat => {
                                let heartbeat = stream_sync::heartbeat(&self_name, &sequence);
                                if tx.send(Ok(heartbeat)).await.is_err() {
                                    break;
                                }
                            }
                            StreamMessageType::Ack => {
                                if let Some(StreamPayload::Ack(ack)) = &msg.payload {
                                    record_ack(&peer_id, ack.success);
                                }
                            }
                            StreamMessageType::Nack => {
                                record_nack(&peer_id);
                            }
                        }
                    }
                    Ok(Some(Err(e))) => {
                        log::error!("Error receiving stream message: {}", e);
                        record_nack(&peer_id);
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

            if let Some(handle) = sender_handle {
                handle.abort();
                let _ = handle.await;
            }
            if !peer_id.is_empty() {
                update_peer_connections(&peer_id, false);
            }
            log::info!("Stream from {} closed", peer_id);
        });

        let output_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(
            Box::pin(output_stream) as Self::SyncStreamStream
        ))
    }
}
