//! Node-wide gossip event loop and outbound stream management.
//!
//! Despite the name, this file does more than outbound gossip. It owns the
//! single per-node tick (1 Hz) that drives several responsibilities:
//!
//! 1. **SWIM-style peer probing.** Picks a peer from cluster state, sends a
//!    `Ping` (with `ping_req` indirect-probe fallback), updates membership.
//! 2. **Outbound `sync_stream` lifecycle.** Dials peers that need a stream,
//!    spawns the per-peer outbound sender task, retries with exponential
//!    backoff on failure, garbage-collects finished connections.
//! 3. **Round collection (node-wide housekeeping).** Drains the local
//!    [`MeshKV`](crate::kv::MeshKV) into a fresh [`RoundBatch`](crate::kv::RoundBatch)
//!    and publishes it into a shared `Arc<RwLock<Arc<RoundBatch>>>` slot
//!    that is read by BOTH the outbound senders here AND the inbound
//!    senders in [`gossip_service`](crate::gossip_service). This step is
//!    not outbound-specific — it produces the shared per-round data that
//!    every outgoing stream (in either direction) consumes.
//! 4. **Periodic housekeeping**: chunk-assembler GC, retry-manager pruning.
//!
//! Per-peer outbound sender tasks also live here. They are spawned by
//! [`GossipController::event_loop`] when this node initiates a stream to a
//! peer. The peer's name is captured as a `String` at task-spawn time, so
//! these senders never need to learn peer identity at runtime — contrast
//! with the inbound senders in `gossip_service.rs`, which must learn the
//! counterparty from the first inbound frame.

use std::{
    collections::{BTreeMap, HashMap},
    net::SocketAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use anyhow::Result;
use parking_lot::RwLock;
use rand::seq::{IndexedRandom, SliceRandom};
use tokio::sync::{mpsc, watch, Mutex};
use tonic::transport::{ClientTlsConfig, Endpoint};
use tracing as log;
use tracing::{instrument, Instrument};

use super::{
    mtls::MTLSManager,
    service::{
        broadcast_node_states,
        gossip::{
            gossip_client::GossipClient, gossip_message, stream_message::Payload as StreamPayload,
            NodeState, NodeStatus, Ping, PingReq, StateSync, StreamMessage, StreamMessageType,
        },
        try_ping, ClusterState,
    },
};
use crate::{
    crdt_kv::CrdtWatermark,
    metrics,
    transport::{
        crdt_batch::{
            build_crdt_batches, crdt_ack_to_watermark, dispatch_crdt_batch, wrap_crdt_ack,
            wrap_crdt_batch,
        },
        limits::{MAX_MESSAGE_SIZE, MAX_STREAM_CHUNK_BYTES, STREAM_IDLE_TIMEOUT},
        sync_stream::{
            build_heartbeat, build_peer_stream_batches, dispatch_stream_batch, wrap_stream_batch,
        },
    },
};

/// The per-node event loop driver. Holds the cluster-state reference,
/// self-identity, init-peer address, mTLS config, the live set of
/// outbound sync_stream task handles, and the shared `RoundBatch` slot.
///
/// One instance per mesh node. See module docs for the full set of
/// responsibilities driven by [`event_loop`](Self::event_loop).
pub struct GossipController {
    state: ClusterState,
    self_name: String,
    self_addr: SocketAddr,
    init_peer: Option<SocketAddr>,
    mtls_manager: Option<Arc<MTLSManager>>,
    // Track active sync_stream connections
    sync_connections: Arc<Mutex<HashMap<String, tokio::task::JoinHandle<()>>>>,
    /// Current stream round batch, drained once per round from MeshKV.
    /// Per-peer senders read this and filter targeted entries to their
    /// own peer; drain_entries are broadcast to every peer.
    current_stream_batch: Arc<RwLock<Arc<crate::kv::RoundBatch>>>,
    /// Node-wide MeshKV handle. Owns the stream buffers, subscriber
    /// registry, and chunk assembler shared with the server-side
    /// SyncStream handlers.
    mesh_kv: Option<Arc<crate::kv::MeshKV>>,
    /// How long a node may stay Down before it is removed from the
    /// cluster and its keys are swept (SWIM §5.2).
    dead_timeout: Duration,
}

/// SWIM §5.2 default: Down nodes are removed after 60s.
const DEFAULT_DEAD_TIMEOUT: Duration = Duration::from_secs(60);

/// Track how long each peer has been Down and return those past
/// `dead_timeout`, due for removal. `down_since` keeps the first moment a
/// peer was seen Down; entries are dropped when the peer revives (its next
/// Down restarts the clock) or leaves the state map.
fn expire_down_nodes(
    down_since: &mut HashMap<String, Instant>,
    state: &BTreeMap<String, NodeState>,
    self_name: &str,
    dead_timeout: Duration,
    now: Instant,
) -> Vec<String> {
    down_since.retain(|name, _| {
        state
            .get(name)
            .is_some_and(|node| node.status == NodeStatus::Down as i32)
    });
    let mut expired = Vec::new();
    for (name, node) in state {
        if name == self_name || node.status != NodeStatus::Down as i32 {
            continue;
        }
        let since = down_since.entry(name.clone()).or_insert(now);
        if now.saturating_duration_since(*since) >= dead_timeout {
            expired.push(name.clone());
        }
    }
    expired
}

/// A removed node's holddown record: when it was removed and the membership
/// version it carried at removal, the freshness bar for lifting the hold.
struct Holddown {
    since: Instant,
    removed_version: u64,
}

/// Removal holddown: a removed node re-offered by a slower survivor's
/// full-state gossip is purged again each round instead of re-arming a
/// fresh `dead_timeout` (and re-firing the sweep). Only a freshness proof
/// lifts the hold: an Alive entry whose version exceeds the removed
/// version (a genuine return produces one via SWIM refutation, which bumps
/// past the Down rumor). A stale Alive re-offer from a survivor that never
/// saw the Down transition is purged like any other stale rumor. Holds
/// expire after `holddown`, which must exceed the slowest survivor's own
/// removal lag.
fn purge_held_reinsertions(
    held: &mut HashMap<String, Holddown>,
    state: &mut BTreeMap<String, NodeState>,
    holddown: Duration,
    now: Instant,
) {
    held.retain(|_, hold| now.saturating_duration_since(hold.since) < holddown);
    held.retain(|name, hold| match state.get(name) {
        // Genuine return: lift the hold so normal tracking resumes.
        Some(node)
            if node.status == NodeStatus::Alive as i32 && node.version > hold.removed_version =>
        {
            false
        }
        // Stale re-offer (Down, or an Alive predating the removal): purge
        // again quietly, keep holding.
        Some(_) => {
            state.remove(name);
            true
        }
        None => true,
    });
}

impl GossipController {
    pub fn new(
        state: ClusterState,
        self_addr: SocketAddr,
        self_name: &str,
        init_peer: Option<SocketAddr>,
        mtls_manager: Option<Arc<MTLSManager>>,
    ) -> Self {
        Self {
            state,
            self_name: self_name.to_string(),
            self_addr,
            init_peer,
            mtls_manager,
            sync_connections: Arc::new(Mutex::new(HashMap::new())),
            current_stream_batch: Arc::new(RwLock::new(Arc::new(crate::kv::RoundBatch::default()))),
            mesh_kv: None,
            dead_timeout: DEFAULT_DEAD_TIMEOUT,
        }
    }

    /// Attach the node-wide MeshKV handle. Plumbed from the server
    /// builder so stream buffers, subscribers, and the chunk assembler
    /// are shared between client-side (outbound) and server-side
    /// (inbound) SyncStream handlers.
    pub fn with_mesh_kv(mut self, mesh_kv: Arc<crate::kv::MeshKV>) -> Self {
        self.mesh_kv = Some(mesh_kv);
        self
    }

    /// Get a handle to the shared stream RoundBatch. Used by GossipService
    /// so server-side sync_stream handlers see the same drained stream
    /// entries as client-side handlers.
    pub fn current_stream_batch(&self) -> Arc<RwLock<Arc<crate::kv::RoundBatch>>> {
        self.current_stream_batch.clone()
    }

    #[instrument(fields(name = %self.self_name), skip(self, signal))]
    pub async fn event_loop(self, mut signal: watch::Receiver<bool>) -> Result<()> {
        let init_state = self.state.clone();
        let read_state = self.state.clone();
        let mut cnt: u64 = 0;

        // Track retry managers for each peer
        use std::collections::HashMap;
        let mut retry_managers: HashMap<String, RetryManager> = HashMap::new();

        // First moment each peer was seen Down, for dead_timeout removal.
        let mut down_since: HashMap<String, Instant> = HashMap::new();
        // Recently removed nodes, held against gossip re-insertion.
        let mut removed_holddown: HashMap<String, Holddown> = HashMap::new();

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

            // SWIM §5.2: remove nodes Down for longer than dead_timeout and
            // sweep their keys (replica registry, registered namespaces).
            purge_held_reinsertions(
                &mut removed_holddown,
                &mut init_state.write(),
                self.dead_timeout * 3,
                Instant::now(),
            );
            let expired = expire_down_nodes(
                &mut down_since,
                &init_state.read(),
                &self.self_name,
                self.dead_timeout,
                Instant::now(),
            );
            for name in expired {
                // Re-check under the write lock: `expired` came from a read
                // snapshot, and merge_state (on a gRPC task) may have revived
                // this peer to Alive in the gap. Remove + sweep ONLY if it is
                // still departed — otherwise this is the destructive live-node
                // sweep the rest of this PR exists to prevent.
                let removed_version = {
                    let mut state = init_state.write();
                    match state.get(&name) {
                        Some(node) if node.status == NodeStatus::Down as i32 => {
                            let version = node.version;
                            state.remove(&name);
                            Some(version)
                        }
                        _ => None,
                    }
                };
                let Some(removed_version) = removed_version else {
                    // Revived or already gone: stop tracking so it re-arms
                    // cleanly if it goes Down again.
                    down_since.remove(&name);
                    continue;
                };
                log::warn!("Removing node {name} after dead_timeout; sweeping its keys");
                retry_managers.remove(&name);
                down_since.remove(&name);
                removed_holddown.insert(
                    name.clone(),
                    Holddown {
                        since: Instant::now(),
                        removed_version,
                    },
                );
                // Reap the dead peer's stream task now rather than letting it
                // tick into a dead channel until the idle timeout.
                if let Some(handle) = self.sync_connections.lock().await.remove(&name) {
                    handle.abort();
                }
                if let Some(mesh_kv) = &self.mesh_kv {
                    let swept = mesh_kv.handle_node_removed(&name);
                    log::info!("Dead-node sweep for {name} tombstoned {swept} keys");
                }
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

            // Chunk assembler GC: every 5 rounds (~5s), drop partial
            // assemblies older than 30s. Partial chunks the receiver has
            // been holding for a full assembly timeout are assumed lost;
            // the sender will re-publish on its own retry cycle with a
            // fresh generation.
            if cnt.is_multiple_of(5) {
                if let Some(mesh_kv) = &self.mesh_kv {
                    mesh_kv.chunk_assembler().gc(Duration::from_secs(30));
                }
            }

            // Periodic retry-manager cleanup every 60 rounds (~60s).
            if cnt.is_multiple_of(60) {
                retry_managers.retain(|peer_name, _| map.contains_key(peer_name));
                // Owner-side replica-registry upkeep: re-assert this
                // incarnation's entry, retire prior incarnations'.
                if let Some(mesh_kv) = &self.mesh_kv {
                    mesh_kv.reconcile_replica_registry();
                }
            }

            // Stream round collection: drain stream namespace buffers and
            // drain callbacks exactly once per round (destructive). Per-peer
            // senders filter targeted_entries by their own peer_id and
            // broadcast drain_entries to all peers. Empty batch if no
            // MeshKV is attached (legacy path pre-Step 3).
            if let Some(mesh_kv) = &self.mesh_kv {
                let stream_batch = mesh_kv.collect_round_batch();
                *self.current_stream_batch.write() = Arc::new(stream_batch);
            }

            tokio::select! {

                _ = signal.changed() => {
                    log::info!("Gossip app_server {} at {} is shutting down", self.self_name, self.self_addr);
                    break;
                }

                () = tokio::time::sleep(Duration::from_secs(1)) => {
                    if let Some(peer) = peer {
                        let peer_name = peer.name.clone();

                        // Get or create retry manager for this peer
                        let retry_manager = retry_managers
                            .entry(peer_name.clone())
                            .or_default();

                        // Check if we should retry based on backoff
                        if retry_manager.should_retry() {
                            match self.connect_to_peer(peer.clone()).await {
                                Ok(()) => {
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
            self.mtls_manager.clone(),
        )
        .await
        {
            Ok(node_update) => {
                log::info!("Received NodeUpdate from peer: {:?}", node_update);
                // Update state for Alive or Leaving status
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
                    }; // Lock is released here

                    // If node is Alive, establish sync_stream connection with freshest address.
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

                    // Broadcast only the unreachable node's status is enough.
                    if let Some(mut unreachable_node) = target.remove(&peer_name) {
                        if unreachable_node.status == NodeStatus::Suspected as i32 {
                            unreachable_node.status = NodeStatus::Down as i32;
                        } else {
                            unreachable_node.status = NodeStatus::Suspected as i32;
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
                        "Failed to connect to peer {peer_name}: direct ping and ping-req both failed"
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
        tx: mpsc::Sender<StreamMessage>,
        self_name: String,
        peer_name: String,
    ) -> tokio::task::JoinHandle<()> {
        let sync_connections = self.sync_connections.clone();
        let current_stream_batch = self.current_stream_batch.clone();
        let mesh_kv = self.mesh_kv.clone();

        // Log connection lifecycle: spawn
        log::debug!(
            peer = %peer_name,
            "spawn_sync_stream_handler called — spawning handler task"
        );

        // Create a span for the spawned task
        let span = tracing::info_span!(
            "sync_stream_handler",
            peer = %peer_name
        );

        #[expect(clippy::disallowed_methods, reason = "handle is returned to caller (spawn_sync_stream_handler) and stored in sync_connections map for lifecycle tracking")]
        tokio::spawn(
            async move {
                use tokio_stream::StreamExt;

                // Log active connection count at handler start
                let active_connections = sync_connections.lock().await.len();
                log::debug!(
                    peer = %peer_name,
                    active_connections,
                    "Sync stream handler started"
                );

                let sequence = Arc::new(AtomicU64::new(0));

                // Per-peer CRDT send watermark (per-key acked versions). The
                // incremental sender filters by it; the inbound loop advances it
                // on CrdtAck and emits acks for received batches.
                let acked: Arc<RwLock<CrdtWatermark>> =
                    Arc::new(RwLock::new(CrdtWatermark::new()));

                // Send initial heartbeat
                let heartbeat =
                    build_heartbeat(sequence.fetch_add(1, Ordering::Relaxed), &self_name);
                if tx.send(heartbeat).await.is_err() {
                    log::warn!("Failed to send initial heartbeat to {}", peer_name);
                    return;
                }

                // Spawn a task to periodically broadcast v2 stream batches.
                let incremental_sender_handle = {
                    let tx_incremental = tx.clone();
                    let self_name_incremental = self_name.clone();
                    let peer_name_incremental = peer_name.clone();
                    let shared_sequence = sequence.clone();
                    let stream_batch_handle = current_stream_batch.clone();
                    let acked_incremental = acked.clone();

                    #[expect(clippy::disallowed_methods, reason = "incremental sender handle is stored and aborted when the parent sync_stream handler exits")]
                    tokio::spawn(async move {
                        let mut interval = tokio::time::interval(Duration::from_secs(1));
                        // Skip re-emission of an unchanged stream batch (main
                        // loop hasn't collected a new one since last tick).
                        let mut last_stream_batch: Option<Arc<crate::kv::RoundBatch>> = None;

                        loop {
                            interval.tick().await;

                            let round_start = Instant::now();

                            // Stream batches: drain-portion (broadcast) +
                            // targeted entries addressed to this peer. On
                            // channel full, the round's stream traffic for
                            // this peer is dropped — no retry (at-most-once).
                            // Application regenerates on its own retry cycle.
                            let stream_batch = stream_batch_handle.read().clone();
                            let fresh_batch = last_stream_batch
                                .as_ref()
                                .is_none_or(|last| !Arc::ptr_eq(last, &stream_batch));
                            if fresh_batch {
                                last_stream_batch = Some(stream_batch.clone());
                                for batch in build_peer_stream_batches(
                                    &stream_batch,
                                    &peer_name_incremental,
                                ) {
                                    let msg = wrap_stream_batch(
                                        batch,
                                        shared_sequence.fetch_add(1, Ordering::Relaxed),
                                        &self_name_incremental,
                                    );
                                    match tx_incremental.try_send(msg) {
                                        Ok(()) => {}
                                        Err(mpsc::error::TrySendError::Full(_)) => {
                                            log::debug!(
                                                peer = %peer_name_incremental,
                                                "stream batch dropped on backpressure"
                                            );
                                            // TODO(metrics): bump
                                            // stream_dropped_on_backpressure
                                            break;
                                        }
                                        Err(mpsc::error::TrySendError::Closed(_)) => {
                                            log::warn!(
                                                peer = %peer_name_incremental,
                                                "stream sender: channel closed, stopping"
                                            );
                                            return;
                                        }
                                    }
                                }
                            }

                            // CRDT op-log: evaluated every tick (acks shrink the
                            // delta even when the stream batch is unchanged).
                            // Send only ops this peer has not acked; the
                            // watermark advances solely on CrdtAck, so unacked
                            // keys retry next round.
                            let crdt_ops: Vec<_> = {
                                let acked = acked_incremental.read();
                                stream_batch
                                    .crdt_ops
                                    .operations()
                                    .iter()
                                    .filter(|op| acked.allows(op))
                                    .cloned()
                                    .collect()
                            };
                            for crdt_batch in build_crdt_batches(&crdt_ops, MAX_STREAM_CHUNK_BYTES) {
                                let msg = wrap_crdt_batch(
                                    crdt_batch,
                                    shared_sequence.fetch_add(1, Ordering::Relaxed),
                                    &self_name_incremental,
                                );
                                match tx_incremental.try_send(msg) {
                                    Ok(()) => {}
                                    Err(mpsc::error::TrySendError::Full(_)) => {
                                        log::debug!(
                                            peer = %peer_name_incremental,
                                            "crdt batch dropped on backpressure"
                                        );
                                        break;
                                    }
                                    Err(mpsc::error::TrySendError::Closed(_)) => {
                                        log::warn!(
                                            peer = %peer_name_incremental,
                                            "crdt sender: channel closed, stopping"
                                        );
                                        return;
                                    }
                                }
                            }

                            let round_elapsed = round_start.elapsed();
                            metrics::record_sync_round_duration(
                                &peer_name_incremental,
                                round_elapsed,
                            );
                            if round_elapsed.as_millis() > 10 {
                                log::info!(
                                    peer = %peer_name_incremental,
                                    round_ms = round_elapsed.as_millis(),
                                    "mesh sync round"
                                );
                            }
                        }
                    })
                };

                // Handle incoming messages
                loop {
                    match tokio::time::timeout(STREAM_IDLE_TIMEOUT, incoming_stream.next()).await {
                        Ok(Some(Ok(msg))) => {
                            sequence.fetch_add(1, Ordering::Relaxed);

                            match msg.message_type() {
                                StreamMessageType::Heartbeat => {
                                    log::trace!("Received heartbeat from {}", peer_name);
                                    let heartbeat = build_heartbeat(
                                        sequence.fetch_add(1, Ordering::Relaxed),
                                        &self_name,
                                    );
                                    if tx.send(heartbeat).await.is_err() {
                                        log::warn!("Failed to send heartbeat to {}", peer_name);
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
                                StreamMessageType::IncrementalUpdate
                                | StreamMessageType::SnapshotRequest
                                | StreamMessageType::SnapshotChunk
                                | StreamMessageType::SnapshotComplete => {
                                    log::debug!(
                                        peer = %peer_name,
                                        message_type = ?msg.message_type(),
                                        "ignoring v1 wire message (state-sync removed)",
                                    );
                                }
                                StreamMessageType::StreamBatch => {
                                    if let Some(mesh_kv) = &mesh_kv {
                                        if let Some(StreamPayload::StreamBatch(batch)) =
                                            msg.payload
                                        {
                                            dispatch_stream_batch(
                                                mesh_kv,
                                                &msg.peer_id,
                                                batch.entries,
                                            );
                                        }
                                    }
                                }
                                StreamMessageType::CrdtBatch => {
                                    if let Some(mesh_kv) = &mesh_kv {
                                        if let Some(StreamPayload::CrdtBatch(batch)) = msg.payload {
                                            // Merge, then ack the per-key
                                            // versions so the peer can advance
                                            // its send watermark. Ack loss is
                                            // fine (peer resends), so drop on a
                                            // full channel rather than block.
                                            let ack = dispatch_crdt_batch(mesh_kv, batch);
                                            if !ack.is_empty() {
                                                let _ = tx.try_send(wrap_crdt_ack(
                                                    &ack,
                                                    sequence.fetch_add(1, Ordering::Relaxed),
                                                    &self_name,
                                                ));
                                            }
                                        }
                                    }
                                }
                                // CRDT delivery ack: advance this peer's send watermark.
                                StreamMessageType::CrdtAck => {
                                    if let Some(StreamPayload::CrdtAck(ack)) = msg.payload {
                                        acked.write().merge_max(&crdt_ack_to_watermark(ack));
                                    }
                                }
                            }
                        }
                        Ok(Some(Err(e))) => {
                            log::error!("Error receiving from sync_stream with {}: {}", peer_name, e);
                            break;
                        }
                        Ok(None) => break,
                        Err(_) => {
                            log::warn!(
                                "sync_stream to {peer_name} idle timeout ({STREAM_IDLE_TIMEOUT:?}) — closing"
                            );
                            break;
                        }
                    }
                }

                incremental_sender_handle.abort();
                let _ = incremental_sender_handle.await;
                log::debug!(
                    peer = %peer_name,
                    "sync_stream_handler exited — handler dropped"
                );
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

        // Connect to peer's gRPC service via Endpoint so TLS can be configured.
        let connect_url = if self.mtls_manager.is_some() {
            format!("https://{peer_addr}")
        } else {
            format!("http://{peer_addr}")
        };
        log::info!("Connecting to URL: {}", connect_url);

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

        // Create bidirectional stream
        let (tx, rx) = mpsc::channel::<StreamMessage>(128);
        let outgoing_stream = tokio_stream::wrappers::ReceiverStream::new(rx);

        let response = client.sync_stream(outgoing_stream).await.map_err(|e| {
            log::error!("Failed to establish sync_stream with {}: {}", peer_name, e);
            anyhow::anyhow!("sync_stream RPC failed: {e}")
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

    values.sample(&mut rng, k).copied().collect()
}

/// Exponential backoff calculator used by the per-peer reconnect loop.
#[derive(Debug, Clone)]
struct ExponentialBackoff {
    initial_delay: Duration,
    max_delay: Duration,
    multiplier: f64,
}

impl ExponentialBackoff {
    fn new(initial_delay: Duration, max_delay: Duration, multiplier: f64) -> Self {
        Self {
            initial_delay,
            max_delay,
            multiplier,
        }
    }

    /// Delay for attempt number (0-indexed).
    fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let max_delay_secs = self.max_delay.as_secs_f64();
        let delay_secs = self.initial_delay.as_secs_f64()
            * self.multiplier.powi(attempt.min(i32::MAX as u32) as i32);
        // Guard against f64 overflow to infinity which would panic in
        // Duration::from_secs_f64.
        let capped = if delay_secs.is_finite() && delay_secs >= 0.0 {
            delay_secs.min(max_delay_secs)
        } else {
            max_delay_secs
        };
        Duration::from_secs_f64(capped)
    }
}

impl Default for ExponentialBackoff {
    fn default() -> Self {
        Self::new(Duration::from_secs(1), Duration::from_secs(60), 2.0)
    }
}

/// Per-peer reconnect state tracker with exponential backoff. Owned
/// by the controller's `HashMap<peer_name, RetryManager>` and only
/// touched on the controller's task, so plain `&mut self` mutation
/// is sufficient — no interior mutability needed.
#[derive(Debug, Default)]
struct RetryManager {
    backoff: ExponentialBackoff,
    last_attempt: Option<Instant>,
    attempt_count: u32,
}

impl RetryManager {
    /// Whether enough time has elapsed since the last attempt to retry.
    /// `attempt_count` counts *completed* attempts; the next retry's
    /// delay slot is therefore the zero-indexed `attempt_count - 1`.
    fn should_retry(&self) -> bool {
        match self.last_attempt {
            Some(last_attempt) => last_attempt.elapsed() >= self.next_delay(),
            None => true,
        }
    }

    fn record_attempt(&mut self) {
        self.last_attempt = Some(Instant::now());
        self.attempt_count = self.attempt_count.saturating_add(1);
    }

    /// Reset on successful connection.
    fn reset(&mut self) {
        self.last_attempt = None;
        self.attempt_count = 0;
    }

    fn attempt_count(&self) -> u32 {
        self.attempt_count
    }

    fn next_delay(&self) -> Duration {
        // `attempt_count` counts completed attempts; the upcoming retry
        // is in the zero-indexed slot one below it.
        self.backoff
            .delay_for_attempt(self.attempt_count.saturating_sub(1))
    }
}

#[cfg(test)]
mod retry_manager_tests {
    use super::*;

    #[test]
    fn first_retry_uses_initial_delay() {
        let mut mgr = RetryManager::default();
        mgr.record_attempt();
        // ExponentialBackoff::default() = (1s, 60s, 2.0)
        assert_eq!(mgr.next_delay(), Duration::from_secs(1));
    }

    #[test]
    fn subsequent_retries_double_until_capped() {
        let mut mgr = RetryManager::default();
        mgr.record_attempt();
        assert_eq!(mgr.next_delay(), Duration::from_secs(1));
        mgr.record_attempt();
        assert_eq!(mgr.next_delay(), Duration::from_secs(2));
        mgr.record_attempt();
        assert_eq!(mgr.next_delay(), Duration::from_secs(4));
        // Cap is 60s with the default config: 1 * 2^6 = 64 -> clamped.
        for _ in 0..10 {
            mgr.record_attempt();
        }
        assert_eq!(mgr.next_delay(), Duration::from_secs(60));
    }

    #[test]
    fn reset_returns_to_first_retry_state() {
        let mut mgr = RetryManager::default();
        for _ in 0..5 {
            mgr.record_attempt();
        }
        assert_ne!(mgr.next_delay(), Duration::from_secs(1));
        mgr.reset();
        assert!(mgr.should_retry(), "post-reset should always allow retry");
        mgr.record_attempt();
        assert_eq!(mgr.next_delay(), Duration::from_secs(1));
    }

    #[test]
    fn should_retry_before_any_attempt() {
        let mgr = RetryManager::default();
        assert!(mgr.should_retry());
    }
}

#[cfg(test)]
mod dead_node_tests {
    use super::*;

    fn node(name: &str, status: NodeStatus) -> NodeState {
        NodeState {
            name: name.to_string(),
            address: format!("{name}:50051"),
            status: status as i32,
            version: 1,
            metadata: HashMap::new(),
        }
    }

    fn state_of(nodes: Vec<NodeState>) -> BTreeMap<String, NodeState> {
        nodes.into_iter().map(|n| (n.name.clone(), n)).collect()
    }

    const TIMEOUT: Duration = Duration::from_secs(60);

    fn hold(t0: Instant, removed_version: u64) -> Holddown {
        Holddown {
            since: t0,
            removed_version,
        }
    }

    fn node_v(name: &str, status: NodeStatus, version: u64) -> NodeState {
        NodeState {
            version,
            ..node(name, status)
        }
    }

    #[test]
    fn down_node_expires_only_after_dead_timeout() {
        let mut down_since = HashMap::new();
        let state = state_of(vec![
            node("a", NodeStatus::Alive),
            node("b", NodeStatus::Down),
        ]);
        let t0 = Instant::now();

        let expired = expire_down_nodes(&mut down_since, &state, "self", TIMEOUT, t0);
        assert!(expired.is_empty(), "fresh Down node survives");

        let expired = expire_down_nodes(
            &mut down_since,
            &state,
            "self",
            TIMEOUT,
            t0 + Duration::from_secs(59),
        );
        assert!(expired.is_empty(), "still inside dead_timeout");

        let expired = expire_down_nodes(
            &mut down_since,
            &state,
            "self",
            TIMEOUT,
            t0 + Duration::from_secs(60),
        );
        assert_eq!(expired, vec!["b".to_string()], "expired at dead_timeout");
    }

    #[test]
    fn revival_restarts_the_clock() {
        let mut down_since = HashMap::new();
        let down = state_of(vec![node("b", NodeStatus::Down)]);
        let alive = state_of(vec![node("b", NodeStatus::Alive)]);
        let t0 = Instant::now();

        expire_down_nodes(&mut down_since, &down, "self", TIMEOUT, t0);
        expire_down_nodes(
            &mut down_since,
            &alive,
            "self",
            TIMEOUT,
            t0 + Duration::from_secs(30),
        );
        assert!(down_since.is_empty(), "revival clears tracking");

        // Down again: the clock restarts from the new observation.
        let expired = expire_down_nodes(
            &mut down_since,
            &down,
            "self",
            TIMEOUT,
            t0 + Duration::from_secs(61),
        );
        assert!(expired.is_empty(), "second Down gets a fresh dead_timeout");
    }

    #[test]
    fn self_is_never_expired() {
        let mut down_since = HashMap::new();
        let state = state_of(vec![node("self", NodeStatus::Down)]);
        let t0 = Instant::now();
        expire_down_nodes(&mut down_since, &state, "self", TIMEOUT, t0);
        let expired = expire_down_nodes(
            &mut down_since,
            &state,
            "self",
            TIMEOUT,
            t0 + Duration::from_secs(120),
        );
        assert!(expired.is_empty());
        assert!(down_since.is_empty(), "self is never tracked");
    }

    #[test]
    fn holddown_purges_reinserted_down_entry_without_retracking() {
        let mut held = HashMap::new();
        let t0 = Instant::now();
        held.insert("d".to_string(), hold(t0, 3));

        // A slower survivor's gossip re-inserted d(Down): purged, still held.
        let mut state = state_of(vec![node("d", NodeStatus::Down)]);
        purge_held_reinsertions(&mut held, &mut state, TIMEOUT, t0 + Duration::from_secs(1));
        assert!(!state.contains_key("d"), "stale re-offer purged");
        assert!(held.contains_key("d"), "hold persists");
    }

    #[test]
    fn holddown_lifts_on_alive_newer_than_removal() {
        let mut held = HashMap::new();
        let t0 = Instant::now();
        held.insert("d".to_string(), hold(t0, 3));

        // A genuine return carries a refutation version past the removal.
        let mut state = state_of(vec![node_v("d", NodeStatus::Alive, 4)]);
        purge_held_reinsertions(&mut held, &mut state, TIMEOUT, t0 + Duration::from_secs(1));
        assert!(state.contains_key("d"), "fresh return kept in membership");
        assert!(held.is_empty(), "hold lifted on fresh alive return");
    }

    #[test]
    fn holddown_keeps_holding_on_stale_alive_rumor() {
        let mut held = HashMap::new();
        let t0 = Instant::now();
        held.insert("d".to_string(), hold(t0, 3));

        // A survivor that never saw the Down transition re-offers a stale
        // Alive; lifting on it would park the dead node in membership for
        // another Suspected -> Down -> dead_timeout walk.
        let mut state = state_of(vec![node_v("d", NodeStatus::Alive, 1)]);
        purge_held_reinsertions(&mut held, &mut state, TIMEOUT, t0 + Duration::from_secs(1));
        assert!(!state.contains_key("d"), "stale alive re-offer purged");
        assert!(held.contains_key("d"), "hold persists against stale alive");
    }

    #[test]
    fn holddown_expires_after_window() {
        let mut held = HashMap::new();
        let t0 = Instant::now();
        held.insert("d".to_string(), hold(t0, 3));

        let mut state = state_of(vec![node("d", NodeStatus::Down)]);
        purge_held_reinsertions(&mut held, &mut state, TIMEOUT, t0 + TIMEOUT);
        assert!(held.is_empty(), "hold expires after the window");
        assert!(
            state.contains_key("d"),
            "post-holddown re-offer re-enters normal tracking"
        );
    }

    #[test]
    fn departed_node_is_dropped_from_tracking() {
        let mut down_since = HashMap::new();
        let down = state_of(vec![node("b", NodeStatus::Down)]);
        let t0 = Instant::now();
        expire_down_nodes(&mut down_since, &down, "self", TIMEOUT, t0);
        assert_eq!(down_since.len(), 1);

        let empty = state_of(vec![]);
        expire_down_nodes(
            &mut down_since,
            &empty,
            "self",
            TIMEOUT,
            t0 + Duration::from_secs(1),
        );
        assert!(down_since.is_empty(), "node gone from state is untracked");
    }
}
