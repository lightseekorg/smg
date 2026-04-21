use std::{
    collections::{BTreeMap, HashMap},
    net::SocketAddr,
    str::FromStr,
    sync::Arc,
    time::Duration,
};

use anyhow::Result;
use parking_lot::RwLock;
use tokio::sync::watch;
use tonic::{
    transport::{ClientTlsConfig, Endpoint},
    Request,
};
use tracing as log;

use crate::flow_control::MAX_MESSAGE_SIZE;

pub mod gossip {
    #![allow(unused_qualifications, clippy::absolute_paths)]
    #![allow(clippy::trivially_copy_pass_by_ref, clippy::allow_attributes)]
    tonic::include_proto!("mesh.gossip");
}
use gossip::{
    gossip_client, gossip_message, GossipMessage, NodeState, NodeStatus, NodeUpdate, Ping,
    StateSync,
};

use crate::{
    controller::MeshController,
    mtls::{MTLSConfig, MTLSManager},
    node_state_machine::{ConvergenceConfig, NodeStateMachine},
    partition::PartitionDetector,
    ping_server::GossipService,
    stores::{AppState, StateStores},
    sync::MeshSyncManager,
};

pub type ClusterState = Arc<RwLock<BTreeMap<String, NodeState>>>;

pub struct MeshServerConfig {
    pub self_name: String,
    pub bind_addr: SocketAddr,
    pub advertise_addr: SocketAddr,
    pub init_peer: Option<SocketAddr>,
    pub mtls_config: Option<MTLSConfig>,
}

/// MeshServerHandler
/// It is the handler for the mesh server, which is responsible for the node management.
/// Includes some basic node management logic, like shutdown,
/// node discovery(TODO), node status update(TODO), etc.
pub struct MeshServerHandler {
    pub state: ClusterState,
    pub stores: Arc<StateStores>,
    pub sync_manager: Arc<MeshSyncManager>,
    pub self_name: String,
    _self_addr: SocketAddr,
    signal_tx: watch::Sender<bool>,
    partition_detector: Option<Arc<PartitionDetector>>,
    state_machine: Option<Arc<NodeStateMachine>>,
    rate_limit_task_handle: std::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
    /// Shared with the MeshServer so adapters can subscribe to stream
    /// namespaces (broadcast/targeted) and publish values that reach
    /// peers via the gossip loop.
    mesh_kv: Arc<crate::kv::MeshKV>,
}

impl MeshServerHandler {
    /// Get partition detector
    pub fn partition_detector(&self) -> Option<&Arc<PartitionDetector>> {
        self.partition_detector.as_ref()
    }

    /// Get state machine
    pub fn state_machine(&self) -> Option<&Arc<NodeStateMachine>> {
        self.state_machine.as_ref()
    }

    /// Check if node is ready
    pub fn is_ready(&self) -> bool {
        self.state_machine
            .as_ref()
            .map(|sm| sm.is_ready())
            .unwrap_or(true) // If no state machine, consider ready
    }

    /// Check if we should serve (have quorum)
    pub fn should_serve(&self) -> bool {
        self.partition_detector
            .as_ref()
            .map(|pd| pd.should_serve())
            .unwrap_or(true) // If no partition detector, consider should serve
    }

    /// Start rate limit window reset task
    /// This task will periodically reset the global rate limit counter
    pub fn start_rate_limit_task(&self, window_seconds: u64) {
        use crate::rate_limit_window::RateLimitWindow;

        let window_manager = RateLimitWindow::new(self.sync_manager.clone(), window_seconds);
        let shutdown_rx = self.signal_tx.subscribe();

        #[expect(
            clippy::disallowed_methods,
            reason = "handle is stored in rate_limit_task_handle and awaited on shutdown via stop_rate_limit_task"
        )]
        let handle = tokio::spawn(async move {
            window_manager.start_reset_task(shutdown_rx).await;
        });

        if let Ok(mut task_handle) = self.rate_limit_task_handle.lock() {
            *task_handle = Some(handle);
        }
    }

    /// Stop rate limit window reset task
    pub fn stop_rate_limit_task(&self) {
        self.signal_tx.send(true).ok();
        if let Ok(mut task_handle) = self.rate_limit_task_handle.lock() {
            if let Some(handle) = task_handle.take() {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "short-lived join task that awaits the rate_limit_task handle during shutdown; completes when the inner task finishes"
                )]
                tokio::spawn(async move {
                    if let Err(err) = handle.await {
                        log::warn!("Rate limit task shutdown failed: {}", err);
                    }
                });
            }
        }
    }

    /// Shutdown immediately without graceful shutdown
    pub fn shutdown(&self) {
        self.stop_rate_limit_task();
    }

    /// Graceful shutdown: broadcast LEAVING status to all alive nodes,
    /// wait for propagation, then shutdown
    pub async fn graceful_shutdown(&self) -> Result<()> {
        log::info!("Graceful shutdown for node {}", self.self_name);

        let maybe_leaving = {
            let state = self.state.read();

            if let Some(self_node) = state.get(&self.self_name) {
                let mut self_node = self_node.clone();
                if self_node.status == NodeStatus::Leaving as i32 {
                    None
                } else {
                    self_node.status = NodeStatus::Leaving as i32;
                    self_node.version += 1;

                    let alive_nodes = state
                        .values()
                        .filter(|node| {
                            node.status == NodeStatus::Alive as i32 && node.name != self.self_name
                            // exclude self from broadcast targets
                        })
                        .cloned()
                        .collect::<Vec<NodeState>>();

                    Some((self_node, alive_nodes))
                }
            } else {
                None
            }
        };
        let (leaving_node, alive_nodes) = match maybe_leaving {
            Some(values) => values,
            None => {
                self.stop_rate_limit_task();
                return Ok(());
            }
        };

        log::info!(
            "Broadcasting LEAVING status to {} alive nodes",
            alive_nodes.len()
        );

        // Broadcast LEAVING status to all alive nodes
        let (success_count, total_count) = broadcast_node_states(
            vec![leaving_node],
            alive_nodes,
            Some(Duration::from_secs(3)),
        )
        .await;

        log::info!(
            "Broadcast LEAVING status: {}/{} successful",
            success_count,
            total_count
        );

        // Wait a bit more for state propagation
        let propagation_delay = Duration::from_secs(1);
        log::info!(
            "Waiting {} seconds for LEAVING status propagation",
            propagation_delay.as_secs()
        );
        tokio::time::sleep(propagation_delay).await;

        log::info!("Stopping rate limit task and signaling shutdown");
        self.stop_rate_limit_task();
        Ok(())
    }

    /// Calculate the next version for a key
    /// If the key exists, increment its version by 1
    /// If the key doesn't exist, start with version 1
    fn next_version(&self, key: &str) -> u64 {
        self.stores
            .app
            .get(key)
            .map(|app_state| app_state.version + 1)
            .unwrap_or(1)
    }

    pub fn write_data(&self, key: String, value: Vec<u8>) -> Result<()> {
        // Keep app store write and metadata/version update in one lock scope.
        let mut state = self.state.write();
        let node = state.get_mut(&self.self_name).ok_or_else(|| {
            anyhow::anyhow!(
                "Node {} not found in cluster state during write_data",
                self.self_name
            )
        })?;

        let version = self.next_version(&key);
        let app_state = AppState {
            key: key.clone(),
            value: value.clone(),
            version,
        };
        self.stores
            .app
            .insert(key.clone(), app_state)
            .map_err(|err| anyhow::anyhow!("Failed to persist app state for key {key}: {err}"))?;

        node.metadata.insert(key, value);
        node.version += 1;
        Ok(())
    }

    pub fn read_data(&self, key: String) -> Option<Vec<u8>> {
        // Read from the app store
        self.stores
            .app
            .get(&key)
            .map(|app_state| app_state.value.clone())
    }

    /// Get operation log of the app store for synchronization
    /// Returns an operation log that can be merged into other nodes
    pub fn get_operation_log(&self) -> crate::crdt_kv::OperationLog {
        self.stores.app.get_operation_log()
    }

    /// Sync app store data from an operation log (for testing and manual sync)
    /// This will be replaced by automatic sync stream in the future
    pub fn sync_app_from_log(&self, log: &crate::crdt_kv::OperationLog) {
        // Merge operation log into our app store using CRDT merge
        self.stores.app.merge(log);
    }

    /// Shared MeshKV handle — adapters subscribe to stream namespaces
    /// and publish values through this. The handle is Arc-cloned, so
    /// subscribers created here see the same events as the gossip loop.
    pub fn mesh_kv(&self) -> &Arc<crate::kv::MeshKV> {
        &self.mesh_kv
    }
}

pub struct MeshServerBuilder {
    state: ClusterState,
    stores: Arc<StateStores>,
    self_name: String,
    bind_addr: SocketAddr,
    advertise_addr: SocketAddr,
    init_peer: Option<SocketAddr>,
    mtls_manager: Option<Arc<MTLSManager>>,
}

impl MeshServerBuilder {
    pub fn new(
        self_name: String,
        bind_addr: SocketAddr,
        advertise_addr: SocketAddr,
        init_peer: Option<SocketAddr>,
    ) -> Self {
        let state = Arc::new(RwLock::new(BTreeMap::from([(
            self_name.clone(),
            NodeState {
                name: self_name.clone(),
                address: advertise_addr.to_string(),
                status: NodeStatus::Alive as i32,
                version: 1,
                metadata: HashMap::new(),
            },
        )])));
        let stores = Arc::new(StateStores::with_self_name(self_name.clone()));
        Self {
            state,
            stores,
            self_name,
            bind_addr,
            advertise_addr,
            init_peer,
            mtls_manager: None,
        }
    }

    pub fn with_mtls(mut self, mtls_config: MTLSConfig) -> Self {
        self.mtls_manager = Some(Arc::new(MTLSManager::new(mtls_config)));
        self
    }

    pub fn build(&self) -> (MeshServer, MeshServerHandler) {
        let (signal_tx, signal_rx) = watch::channel(false);
        let partition_detector = Arc::new(PartitionDetector::default());
        let sync_manager = Arc::new(MeshSyncManager::new(
            self.stores.clone(),
            self.self_name.clone(),
        ));
        let state_machine = Arc::new(NodeStateMachine::new(
            self.stores.clone(),
            ConvergenceConfig::default(),
        ));
        // Initialize rate-limit hash ring with current membership
        sync_manager.update_rate_limit_membership();
        let mesh_kv = Arc::new(crate::kv::MeshKV::new(self.self_name.clone()));
        (
            MeshServer {
                state: self.state.clone(),
                stores: self.stores.clone(),
                sync_manager: sync_manager.clone(),
                self_name: self.self_name.clone(),
                bind_addr: self.bind_addr,
                advertise_addr: self.advertise_addr,
                init_peer: self.init_peer,
                signal_rx,
                partition_detector: Some(partition_detector.clone()),
                mtls_manager: self.mtls_manager.clone(),
                mesh_kv: mesh_kv.clone(),
            },
            MeshServerHandler {
                state: self.state.clone(),
                stores: self.stores.clone(),
                sync_manager,
                self_name: self.self_name.clone(),
                _self_addr: self.advertise_addr,
                signal_tx,
                partition_detector: Some(partition_detector),
                state_machine: Some(state_machine),
                rate_limit_task_handle: std::sync::Mutex::new(None),
                mesh_kv,
            },
        )
    }
}

impl From<&MeshServerConfig> for MeshServerBuilder {
    fn from(value: &MeshServerConfig) -> Self {
        let mut builder = MeshServerBuilder::new(
            value.self_name.clone(),
            value.bind_addr,
            value.advertise_addr,
            value.init_peer,
        );
        if let Some(mtls_config) = &value.mtls_config {
            builder = builder.with_mtls(mtls_config.clone());
        }
        builder
    }
}

pub struct MeshServer {
    state: ClusterState,
    stores: Arc<StateStores>,
    sync_manager: Arc<MeshSyncManager>,
    self_name: String,
    bind_addr: SocketAddr,
    advertise_addr: SocketAddr,
    init_peer: Option<SocketAddr>,
    signal_rx: watch::Receiver<bool>,
    partition_detector: Option<Arc<PartitionDetector>>,
    mtls_manager: Option<Arc<MTLSManager>>,
    /// Node-wide MeshKV handle shared by controller + ping_server.
    mesh_kv: Arc<crate::kv::MeshKV>,
}

impl MeshServer {
    fn build_ping_server(&self) -> GossipService {
        GossipService::new(
            self.state.clone(),
            self.bind_addr,
            self.advertise_addr,
            &self.self_name,
        )
        .with_mesh_kv(self.mesh_kv.clone())
    }

    fn build_controller(&self) -> MeshController {
        MeshController::new(
            self.state.clone(),
            self.advertise_addr,
            &self.self_name,
            self.init_peer,
            self.stores.clone(),
            self.sync_manager.clone(),
            self.mtls_manager.clone(),
        )
        .with_mesh_kv(self.mesh_kv.clone())
    }

    pub async fn start(self) -> Result<()> {
        self.start_inner(None).await
    }

    pub async fn start_with_listener(self, listener: tokio::net::TcpListener) -> Result<()> {
        let bound_addr = listener
            .local_addr()
            .map_err(|e| anyhow::anyhow!("Failed to read listener local addr: {e}"))?;
        if bound_addr != self.bind_addr {
            return Err(anyhow::anyhow!(
                "Listener/bind_addr mismatch: listener={}, bind_addr={}",
                bound_addr,
                self.bind_addr
            ));
        }
        self.start_inner(Some(listener)).await
    }

    async fn start_inner(self, listener: Option<tokio::net::TcpListener>) -> Result<()> {
        log::info!(
            "Mesh server listening on {} and advertising {}",
            self.bind_addr,
            self.advertise_addr
        );
        let self_name = self.self_name.clone();
        let advertise_address = self.advertise_addr;

        #[expect(
            clippy::expect_used,
            reason = "partition_detector is always set to Some by MeshServerBuilder::build() before start() is called"
        )]
        let partition_detector = self
            .partition_detector
            .clone()
            .expect("partition detector missing");

        // Build controller first so we can share its current_batch with the
        // server-side sync_stream handlers. This ensures both client-side
        // (outgoing connections) and server-side (incoming connections) use
        // the same centrally collected RoundBatch.
        let controller = self.build_controller();

        let mut service = self.build_ping_server();
        service = service.with_stores(self.stores.clone());

        service = service.with_sync_manager(self.sync_manager.clone());

        service = service.with_partition_detector(partition_detector);

        // Share the controller's current_batch so server-side sync_stream
        // handlers use the same centrally collected data as client-side.
        service = service.with_current_batch(controller.current_batch());
        service = service.with_current_stream_batch(controller.current_stream_batch());

        // Add mTLS support if configured
        if let Some(mtls_manager) = self.mtls_manager.clone() {
            service = service.with_mtls_manager(mtls_manager);
        }

        let mut service_shutdown = self.signal_rx.clone();

        #[expect(
            clippy::disallowed_methods,
            reason = "handle is awaited immediately below via tokio::select!, bounded by shutdown signal"
        )]
        let server_handle = if let Some(tcp_listener) = listener {
            tokio::spawn(service.serve_ping_with_listener(tcp_listener, async move {
                _ = service_shutdown.changed().await;
            }))
        } else {
            tokio::spawn(service.serve_ping_with_shutdown(async move {
                _ = service_shutdown.changed().await;
            }))
        };
        tokio::time::sleep(Duration::from_secs(1)).await;
        #[expect(
            clippy::disallowed_methods,
            reason = "handle is awaited immediately below via tokio::select!, bounded by shutdown signal"
        )]
        let app_handle = tokio::spawn(controller.event_loop(self.signal_rx.clone()));

        tokio::select! {
            res = server_handle => res??,
            res = app_handle => res??,
        }

        log::info!(
            "Mesh server {} at {} is shutting down",
            self_name,
            advertise_address
        );
        Ok(())
    }
}

/// Broadcast node state updates to target nodes
/// Returns (success_count, total_count)
pub async fn broadcast_node_states(
    nodes_to_broadcast: Vec<NodeState>,
    target_nodes: Vec<NodeState>,
    timeout: Option<Duration>,
) -> (usize, usize) {
    if nodes_to_broadcast.is_empty() || target_nodes.is_empty() {
        log::debug!(
            "Nothing to broadcast: nodes_to_broadcast={}, target_nodes={}",
            nodes_to_broadcast.len(),
            target_nodes.len()
        );
        return (0, target_nodes.len());
    }

    let mut broadcast_tasks = Vec::new();
    for target_node in &target_nodes {
        let target_node_clone = target_node.clone();
        let nodes_for_task = nodes_to_broadcast.clone();
        #[expect(
            clippy::disallowed_methods,
            reason = "broadcast tasks are collected and awaited via join_all with a timeout immediately below"
        )]
        let task = tokio::spawn(async move {
            let state_sync = StateSync {
                nodes: nodes_for_task,
            };
            let ping_payload = gossip_message::Payload::Ping(Ping {
                state_sync: Some(state_sync),
            });
            match try_ping(&target_node_clone, Some(ping_payload), None).await {
                Ok(_) => {
                    log::debug!("Successfully broadcasted to {}", target_node_clone.name);
                    Ok(())
                }
                Err(e) => {
                    log::warn!("Failed to broadcast to {}: {}", target_node_clone.name, e);
                    Err(e)
                }
            }
        });
        broadcast_tasks.push(task);
    }

    let timeout_duration = timeout.unwrap_or(Duration::from_secs(3));
    let broadcast_result = tokio::time::timeout(timeout_duration, async {
        futures::future::join_all(broadcast_tasks).await
    })
    .await;

    match broadcast_result {
        Ok(results) => {
            let success_count = results.iter().filter(|r| matches!(r, Ok(Ok(())))).count();
            let total_count = target_nodes.len();
            log::info!(
                "Broadcast completed: {}/{} successful",
                success_count,
                total_count
            );
            (success_count, total_count)
        }
        Err(_) => {
            log::warn!(
                "Broadcast timeout after {} seconds",
                timeout_duration.as_secs()
            );
            (0, target_nodes.len())
        }
    }
}

pub async fn try_ping(
    peer_node: &NodeState,
    payload: Option<gossip_message::Payload>,
    mtls_manager: Option<Arc<MTLSManager>>,
) -> Result<NodeUpdate, tonic::Status> {
    let peer_name = peer_node.name.clone();

    let peer_addr = SocketAddr::from_str(&peer_node.address).map_err(|e| {
        tonic::Status::invalid_argument(format!(
            "Invalid address for node {}: {}, {}",
            peer_name, peer_node.address, e
        ))
    })?;

    let connect_url = if mtls_manager.is_some() {
        format!("https://{peer_addr}")
    } else {
        format!("http://{peer_addr}")
    };

    let mut endpoint = Endpoint::from_shared(connect_url.clone())
        .map_err(|e| {
            tonic::Status::invalid_argument(format!(
                "Invalid endpoint for node {peer_name}: {connect_url}, {e}"
            ))
        })?
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(10));

    if let Some(mtls_manager) = mtls_manager {
        mtls_manager.load_client_config().await.map_err(|e| {
            tonic::Status::unavailable(format!(
                "Failed to load mTLS client config for {peer_name}: {e}"
            ))
        })?;

        let tls_domain = endpoint
            .uri()
            .host()
            .map(str::to_owned)
            .unwrap_or_else(|| peer_name.clone());
        let ca_certificate = mtls_manager.load_ca_certificate().await.map_err(|e| {
            tonic::Status::unavailable(format!(
                "Failed to load mTLS CA certificate for {peer_name}: {e}"
            ))
        })?;

        endpoint = endpoint
            .tls_config(
                ClientTlsConfig::new()
                    .domain_name(tls_domain)
                    .ca_certificate(ca_certificate),
            )
            .map_err(|e| {
                tonic::Status::unavailable(format!(
                    "Failed to configure TLS endpoint for {peer_name}: {e}"
                ))
            })?;
    }

    let channel = endpoint.connect().await.map_err(|e| {
        log::warn!(
            "Failed to connect to peer {} {}: {}.",
            peer_name,
            peer_addr,
            e
        );
        tonic::Status::unavailable("Failed to connect to peer")
    })?;
    let mut client = gossip_client::GossipClient::new(channel)
        .max_decoding_message_size(MAX_MESSAGE_SIZE)
        .max_encoding_message_size(MAX_MESSAGE_SIZE)
        .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
        .send_compressed(tonic::codec::CompressionEncoding::Gzip);

    let ping_message = GossipMessage { payload };
    let response = client.ping_server(Request::new(ping_message)).await?;

    Ok(response.into_inner())
}

#[macro_export]
macro_rules! mesh_run {
    ($addr:expr, $init_peer:expr) => {{
        mesh_run!($addr.to_string(), $addr, $init_peer)
    }};

    ($name:expr, $addr:expr, $init_peer:expr) => {{
        tracing::info!("Starting mesh server : {}", $addr);
        use $crate::MeshServerBuilder;
        let (server, handler) =
            MeshServerBuilder::new($name.to_string(), $addr, $addr, $init_peer).build();
        #[expect(clippy::disallowed_methods, reason = "test macro: spawned server runs for the test lifetime and handler is returned for assertions")]
        tokio::spawn(async move {
            if let Err(e) = server.start().await {
                tracing::error!("Mesh server failed: {}", e);
            }
        });
        handler
    }};

    ($name:expr, $listener:expr, $addr:expr, $init_peer:expr) => {{
        tracing::info!("Starting mesh server : {}", $addr);
        use $crate::MeshServerBuilder;
        let (server, handler) =
            MeshServerBuilder::new($name.to_string(), $addr, $addr, $init_peer).build();
        #[expect(clippy::disallowed_methods, reason = "test macro: spawned server runs for the test lifetime and handler is returned for assertions")]
        tokio::spawn(async move {
            if let Err(e) = server.start_with_listener($listener).await {
                tracing::error!("Mesh server failed: {}", e);
            }
        });
        handler
    }};
}

#[cfg(test)]
mod tests {
    use std::sync::Once;

    use tracing as log;
    use tracing_subscriber::{
        filter::LevelFilter, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter,
    };

    use super::*;
    use crate::tests::test_utils::{bind_node, wait_for};

    static INIT: Once = Once::new();
    fn init() {
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

    #[tokio::test]
    async fn test_ping_advertises_configured_address() {
        init();

        let (listener, bind_addr) = bind_node().await;
        let advertise_addr = SocketAddr::from(([10, 20, 30, 40], bind_addr.port()));
        let (server, handler) =
            MeshServerBuilder::new("A".to_string(), bind_addr, advertise_addr, None).build();

        #[expect(
            clippy::disallowed_methods,
            reason = "test server runs in the background for the duration of the assertion"
        )]
        tokio::spawn(async move {
            if let Err(e) = server.start_with_listener(listener).await {
                tracing::error!("Mesh server failed: {}", e);
            }
        });

        wait_for(
            || std::net::TcpStream::connect(bind_addr).is_ok(),
            Duration::from_secs(5),
            "mesh listener started",
        )
        .await;

        let response = try_ping(
            &NodeState {
                name: "A".to_string(),
                address: bind_addr.to_string(),
                status: NodeStatus::Alive as i32,
                version: 1,
                metadata: HashMap::new(),
            },
            Some(gossip_message::Payload::Ping(Ping {
                state_sync: Some(StateSync { nodes: vec![] }),
            })),
            None,
        )
        .await
        .unwrap();

        assert_eq!(response.address, advertise_addr.to_string());
        handler.shutdown();
    }

    #[tokio::test]
    #[ignore = "SWIM failure detection for hard-shutdown nodes needs many gossip rounds; flaky under parallel CI load"]
    async fn test_state_synchronization() {
        init();
        log::info!("Starting test_state_synchronization");

        // 1. Setup A-B cluster
        let (listener_a, addr_a) = bind_node().await;
        let handler_a = mesh_run!("A", listener_a, addr_a, None);
        let (listener_b, addr_b) = bind_node().await;
        let handler_b = mesh_run!("B", listener_b, addr_b, Some(addr_a));

        wait_for(
            || handler_a.state.read().len() == 2,
            Duration::from_secs(15),
            "A-B cluster formed",
        )
        .await;

        handler_a
            .write_data("hello".into(), "world".into())
            .unwrap();

        // 2. Add C and D
        let (listener_c, addr_c) = bind_node().await;
        let handler_c = mesh_run!("C", listener_c, addr_c, Some(addr_a));
        let (listener_d, addr_d) = bind_node().await;
        let handler_d = mesh_run!("D", listener_d, addr_d, Some(addr_c));

        wait_for(
            || handler_a.state.read().len() == 4,
            Duration::from_secs(30),
            "4-node cluster formed",
        )
        .await;

        // 3. Add E, let it join, then kill it
        {
            let (listener_e, addr_e) = bind_node().await;
            let handler_e = mesh_run!("E", listener_e, addr_e, Some(addr_d));

            wait_for(
                || handler_a.state.read().len() == 5,
                Duration::from_secs(30),
                "E joined cluster",
            )
            .await;

            handler_e.shutdown();
        }

        // 4. Gracefully shutdown D
        handler_d.graceful_shutdown().await.unwrap();

        // 5. Wait for D=Leaving and E=Down (not Alive) on all remaining nodes
        let check_statuses = |handler: &MeshServerHandler| {
            let state = handler.state.read();
            let d_leaving = state
                .get("D")
                .is_some_and(|n| n.status == NodeStatus::Leaving as i32);
            let e_not_alive = state
                .get("E")
                .is_some_and(|n| n.status != NodeStatus::Alive as i32);
            d_leaving && e_not_alive
        };

        for (handler, name) in [(&handler_a, "A"), (&handler_b, "B"), (&handler_c, "C")] {
            wait_for(
                || check_statuses(handler),
                Duration::from_secs(60),
                &format!("D=Leaving, E not Alive on node {name}"),
            )
            .await;
        }

        log::info!("All nodes converged to expected state");
    }
}
