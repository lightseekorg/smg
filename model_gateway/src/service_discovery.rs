use std::{
    collections::{HashMap, HashSet},
    // std::sync::Mutex is intentional: all critical sections are tiny
    // (HashSet insert/remove/contains) and never cross .await boundaries.
    // See: https://docs.rs/tokio/latest/tokio/sync/struct.Mutex.html#which-kind-of-mutex-should-you-use
    sync::{Arc, Mutex},
    time::Duration,
};

use futures::StreamExt;
use k8s_openapi::api::core::v1::Pod;
use kube::{
    api::{Api, ListParams},
    runtime::watcher::{watcher, Config, Event},
    Client,
};
use openai_protocol::worker::{WorkerSpec, WorkerType};
use rustls::crypto::ring;
use smg_mesh::{
    gossip::{NodeState, NodeStatus},
    ClusterState,
};
use tokio::{task, time};
use tracing::{debug, error, info, warn};

use crate::{
    app_context::AppContext,
    observability::metrics::{metrics_labels, Metrics},
    workflow::Job,
};

type TrackedPods = HashMap<String, PodInfo>;
type PodDiff = Vec<(String, PodInfo)>;
type TrackedRouterNodes = HashMap<String, String>;

/// Source for per-worker model_id override during Kubernetes service discovery.
#[derive(Debug, Clone)]
pub enum ModelIdSource {
    /// Use the pod's namespace as the model_id.
    Namespace,
    /// Use a specific pod label value as the model_id.
    Label(String),
    /// Use a specific pod annotation value as the model_id.
    Annotation(String),
}

impl ModelIdSource {
    /// Parse a CLI string like `"namespace"`, `"label:key"`, or `"annotation:key"`.
    pub fn parse(s: &str) -> Result<Self, String> {
        if s.eq_ignore_ascii_case("namespace") {
            Ok(Self::Namespace)
        } else if let Some(key) = s.strip_prefix("label:") {
            if key.is_empty() {
                Err("label: requires a key name".to_string())
            } else {
                Ok(Self::Label(key.to_string()))
            }
        } else if let Some(key) = s.strip_prefix("annotation:") {
            if key.is_empty() {
                Err("annotation: requires a key name".to_string())
            } else {
                Ok(Self::Annotation(key.to_string()))
            }
        } else {
            Err(format!(
                "Invalid model-id-from value '{s}'. Expected: namespace, label:<key>, or annotation:<key>"
            ))
        }
    }

    /// Extract the model_id value from a Kubernetes Pod object.
    pub fn extract(&self, pod: &Pod) -> Option<String> {
        match self {
            Self::Namespace => pod.metadata.namespace.clone(),
            Self::Label(key) => pod
                .metadata
                .labels
                .as_ref()
                .and_then(|labels| labels.get(key).cloned()),
            Self::Annotation(key) => pod
                .metadata
                .annotations
                .as_ref()
                .and_then(|annotations| annotations.get(key).cloned()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServiceDiscoveryConfig {
    pub enabled: bool,
    pub selector: HashMap<String, String>,
    pub check_interval: Duration,
    pub port: u16,
    pub namespace: Option<String>,
    // PD mode specific configuration
    pub pd_mode: bool,
    pub prefill_selector: HashMap<String, String>,
    pub decode_selector: HashMap<String, String>,
    // Bootstrap port annotation specific to mooncake implementation
    pub bootstrap_port_annotation: String,
    // Router node discovery for mesh
    pub router_selector: HashMap<String, String>,
    pub router_mesh_port_annotation: String,
    /// Per-worker model_id override source from pod metadata.
    pub model_id_source: Option<ModelIdSource>,
}

impl ServiceDiscoveryConfig {
    /// Build a label selector string for K8s list calls.
    ///
    /// In regular mode, uses the worker selector directly.
    /// In PD mode, uses labels common to both prefill and decode selectors
    /// so a single list call covers both pod types. If there are no common
    /// labels, returns an empty string (no server-side filtering).
    fn list_label_selector(&self) -> String {
        if self.pd_mode {
            self.prefill_selector
                .iter()
                .filter(|(k, v)| self.decode_selector.get(*k) == Some(*v))
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(",")
        } else {
            self.selector
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(",")
        }
    }

    /// Build a label selector string for router pod K8s list/watch calls.
    /// Returns an empty string when the router selector is unset, in which
    /// case the watcher should fall back to listing without server-side
    /// label filtering.
    fn router_label_selector(&self) -> String {
        self.router_selector
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Build a kube watcher Config that pushes the given label selector down to
/// the API server, logging the start of a new watcher iteration at INFO.
/// An empty selector falls back to `Config::default()` (no server-side
/// label filtering) so the watcher still functions when no selector is set.
fn build_watcher_config(watcher_kind: &str, label_selector: &str) -> Config {
    info!(
        "Starting K8s {} watcher | selector: '{}'",
        watcher_kind, label_selector
    );
    if label_selector.is_empty() {
        Config::default()
    } else {
        Config::default().labels(label_selector)
    }
}

fn tracking_key(pod: &Pod) -> String {
    let namespace = pod.metadata.namespace.as_deref().unwrap_or("");
    let name = pod.metadata.name.as_deref().unwrap_or("");
    format!("{namespace}/{name}")
}

impl Default for ServiceDiscoveryConfig {
    fn default() -> Self {
        ServiceDiscoveryConfig {
            enabled: false,
            selector: HashMap::new(),
            check_interval: Duration::from_secs(60),
            port: 8000,
            namespace: None,
            pd_mode: false,
            prefill_selector: HashMap::new(),
            decode_selector: HashMap::new(),
            bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
            router_selector: HashMap::new(),
            router_mesh_port_annotation: "sglang.ai/mesh-port".to_string(),
            model_id_source: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PodType {
    Prefill,
    Decode,
    Regular,
}

#[derive(Debug, Clone)]
pub struct PodInfo {
    pub name: String,
    pub uid: String,
    pub ip: String,
    pub status: String,
    pub is_ready: bool,
    pub pod_type: Option<PodType>,
    pub bootstrap_port: Option<u16>,
    pub is_router: bool,
    pub mesh_port: Option<u16>,
    pub model_id_override: Option<String>,
}

// Identity is (name, uid) — the uid changes on every pod restart, so
// StatefulSet pods that keep the same name+IP across restarts are still
// detected as different entities.  Mutable fields like status and is_ready
// must not affect set membership, otherwise reconciliation produces false
// diffs when a pod's readiness changes between watcher events and a full list.
impl PartialEq for PodInfo {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.uid == other.uid
    }
}

impl Eq for PodInfo {}

impl std::hash::Hash for PodInfo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.uid.hash(state);
    }
}

impl PodInfo {
    fn matches_selector(pod: &Pod, selector: &HashMap<String, String>) -> bool {
        if selector.is_empty() {
            return false;
        }

        pod.metadata
            .labels
            .as_ref()
            .is_some_and(|labels| selector.iter().all(|(k, v)| labels.get(k) == Some(v)))
    }

    pub fn should_include(pod: &Pod, config: &ServiceDiscoveryConfig) -> bool {
        if config.pd_mode {
            if config.prefill_selector.is_empty() && config.decode_selector.is_empty() {
                warn!("PD mode enabled but both prefill_selector and decode_selector are empty");
                return false;
            }
            Self::matches_selector(pod, &config.prefill_selector)
                || Self::matches_selector(pod, &config.decode_selector)
        } else {
            if config.selector.is_empty() {
                warn!("Regular mode enabled but selector is empty");
                return false;
            }
            Self::matches_selector(pod, &config.selector)
        }
    }

    pub fn from_pod(pod: &Pod, config: Option<&ServiceDiscoveryConfig>) -> Option<Self> {
        let name = pod.metadata.name.clone()?;
        let uid = match pod.metadata.uid.clone() {
            Some(uid) => uid,
            None => {
                warn!(
                    "Pod {} has no UID, skipping -- cannot track identity for reconciliation",
                    name
                );
                return None;
            }
        };
        let status = pod.status.clone()?;
        let pod_ip = status.pod_ip?;

        let is_ready = if let Some(conditions) = &status.conditions {
            conditions
                .iter()
                .any(|condition| condition.type_ == "Ready" && condition.status == "True")
        } else {
            false
        };

        let pod_status = status.phase.unwrap_or_else(|| "Unknown".to_string());

        let pod_type = if let Some(config) = config {
            if config.pd_mode {
                if Self::matches_selector(pod, &config.prefill_selector) {
                    Some(PodType::Prefill)
                } else if Self::matches_selector(pod, &config.decode_selector) {
                    Some(PodType::Decode)
                } else {
                    Some(PodType::Regular)
                }
            } else {
                Some(PodType::Regular)
            }
        } else {
            None
        };

        let bootstrap_port = if matches!(pod_type, Some(PodType::Prefill)) {
            if let Some(config) = config {
                pod.metadata
                    .annotations
                    .as_ref()
                    .and_then(|annotations| annotations.get(&config.bootstrap_port_annotation))
                    .and_then(|port_str| port_str.parse::<u16>().ok())
            } else {
                None
            }
        } else {
            None
        };

        // Check if this is a router pod
        let is_router = if let Some(config) = config {
            !config.router_selector.is_empty()
                && Self::matches_selector(pod, &config.router_selector)
        } else {
            false
        };

        // Extract mesh port from annotation if this is a router pod
        let mesh_port = if is_router {
            if let Some(config) = config {
                pod.metadata
                    .annotations
                    .as_ref()
                    .and_then(|annotations| annotations.get(&config.router_mesh_port_annotation))
                    .and_then(|port_str| port_str.parse::<u16>().ok())
            } else {
                None
            }
        } else {
            None
        };

        // Extract model_id override from pod metadata if source is configured
        let model_id_override = config
            .and_then(|c| c.model_id_source.as_ref())
            .and_then(|source| source.extract(pod));

        Some(PodInfo {
            name,
            uid,
            ip: pod_ip,
            status: pod_status,
            is_ready,
            pod_type,
            bootstrap_port,
            is_router,
            mesh_port,
            model_id_override,
        })
    }

    pub fn is_healthy(&self) -> bool {
        self.is_ready && self.status == "Running"
    }

    pub fn worker_url(&self, port: u16) -> String {
        // Default to http:// prefix; workflow will detect actual protocol (HTTP vs gRPC)
        format!("http://{}:{}", self.ip, port)
    }
}

pub async fn start_service_discovery(
    config: ServiceDiscoveryConfig,
    app_context: Arc<AppContext>,
    mesh_cluster_state: Option<ClusterState>,
    mesh_port: Option<u16>,
) -> Result<task::JoinHandle<()>, kube::Error> {
    if !config.enabled {
        return Err(kube::Error::Api(
            kube::core::Status::failure("Service discovery is disabled", "ConfigurationError")
                .with_code(400)
                .boxed(),
        ));
    }

    let _ = ring::default_provider().install_default();

    let client = Client::try_default().await?;

    // Log the appropriate selectors based on mode
    if config.pd_mode {
        let prefill_selector = config
            .prefill_selector
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",");

        let decode_selector = config
            .decode_selector
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",");

        info!(
            "Starting K8s service discovery | PD mode | prefill: '{}' | decode: '{}'",
            prefill_selector, decode_selector
        );
    } else {
        let label_selector = config
            .selector
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",");

        info!(
            "Starting K8s service discovery | selector: '{}'",
            label_selector
        );
    }

    // Log router discovery if enabled
    if !config.router_selector.is_empty() {
        let router_selector = config
            .router_selector
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",");
        info!(
            "Router node discovery enabled | selector: '{}' | mesh port annotation: '{}'",
            router_selector, config.router_mesh_port_annotation
        );
    }

    #[expect(
        clippy::disallowed_methods,
        reason = "service discovery runs for the lifetime of the server; shutdown is handled by dropping the handle"
    )]
    let handle = task::spawn(async move {
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));

        let pods: Api<Pod> = if let Some(namespace) = &config.namespace {
            Api::namespaced(client, namespace)
        } else {
            Api::all(client)
        };

        debug!("K8s service discovery initialized");

        let config_arc = Arc::new(config.clone());
        let port = config.port;

        // Spawn router discovery task if enabled and mesh is available
        // Router discovery requires mesh to be enabled to update cluster state
        // If mesh is not enabled, router discovery is skipped and service discovery works independently
        if !config_arc.router_selector.is_empty() {
            if let (Some(cluster_state), Some(mesh_port)) = (mesh_cluster_state.clone(), mesh_port)
            {
                let router_config = config_arc.clone();
                let router_pods = pods.clone();
                #[expect(
                    clippy::disallowed_methods,
                    reason = "router discovery runs for the lifetime of the server alongside worker discovery"
                )]
                let router_handle = tokio::spawn(async move {
                    start_router_discovery(router_config, router_pods, cluster_state, mesh_port)
                        .await;
                });
                #[expect(
                    clippy::disallowed_methods,
                    reason = "monitor task runs for the lifetime of the server"
                )]
                tokio::spawn(async move {
                    if let Err(e) = router_handle.await {
                        error!(
                            "Router discovery task panicked and is no longer running: {}",
                            e
                        );
                    }
                });
                info!("Router discovery enabled (requires mesh to be enabled)");
            } else {
                warn!(
                    "Router selector configured but mesh is not enabled (mesh cluster state or mesh port not provided). \
                    Router discovery requires mesh to be enabled. Skipping router discovery."
                );
            }
        }

        // Spawn a supervisor that runs periodic reconciliation and restarts it
        // on panic. This is a safety net independent of the watcher: it catches
        // missed events regardless of whether the watcher is healthy, restarting,
        // or erroring out.
        {
            let reconcile_pods_api = pods.clone();
            let reconcile_config = Arc::clone(&config_arc);
            let reconcile_tracked = Arc::clone(&tracked_pods);
            let reconcile_ctx = Arc::clone(&app_context);
            let reconcile_interval = config.check_interval;
            #[expect(
                clippy::disallowed_methods,
                reason = "reconciliation supervisor runs for the lifetime of the server"
            )]
            tokio::spawn(async move {
                loop {
                    let api = reconcile_pods_api.clone();
                    let cfg = Arc::clone(&reconcile_config);
                    let trk = Arc::clone(&reconcile_tracked);
                    let ctx = Arc::clone(&reconcile_ctx);
                    let handle = tokio::spawn(async move {
                        // Delay the first tick so the watcher has time to populate initial state.
                        let start = time::Instant::now() + reconcile_interval;
                        let mut interval = time::interval_at(start, reconcile_interval);
                        loop {
                            interval.tick().await;
                            reconcile_pods(
                                &api,
                                Arc::clone(&cfg),
                                Arc::clone(&trk),
                                Arc::clone(&ctx),
                                port,
                            )
                            .await;
                        }
                    });
                    if let Err(e) = handle.await {
                        error!(
                            "Periodic reconciliation task panicked: {} -- restarting after {}s",
                            e,
                            reconcile_interval.as_secs()
                        );
                        time::sleep(reconcile_interval).await;
                    } else {
                        break;
                    }
                }
            });
            info!(
                "Periodic reconciliation enabled | interval: {}s",
                config.check_interval.as_secs()
            );
        }

        let mut retry_delay = Duration::from_secs(1);
        const MAX_RETRY_DELAY: Duration = Duration::from_secs(300);

        loop {
            let watcher_config = build_watcher_config("worker", &config_arc.list_label_selector());
            let mut watcher_stream = std::pin::pin!(watcher(pods.clone(), watcher_config));
            let mut init_snapshot = HashSet::new();
            let mut watcher_ok = Ok(());

            while let Some(event_result) = watcher_stream.next().await {
                match event_result {
                    Ok(Event::Apply(pod)) => {
                        let key = tracking_key(&pod);
                        handle_pod_apply_event(
                            pod,
                            key,
                            Arc::clone(&tracked_pods),
                            Arc::clone(&app_context),
                            Arc::clone(&config_arc),
                            port,
                        )
                        .await;
                    }
                    Ok(Event::Delete(pod)) => {
                        let key = tracking_key(&pod);
                        handle_pod_deletion_by_key(
                            &key,
                            Arc::clone(&tracked_pods),
                            Arc::clone(&app_context),
                            port,
                            metrics_labels::DEREGISTRATION_POD_DELETED,
                        )
                        .await;
                    }
                    Ok(Event::Init) => {
                        init_snapshot.clear();
                    }
                    Ok(Event::InitApply(pod)) => {
                        let key = tracking_key(&pod);
                        init_snapshot.insert(key.clone());
                        handle_pod_apply_event(
                            pod,
                            key,
                            Arc::clone(&tracked_pods),
                            Arc::clone(&app_context),
                            Arc::clone(&config_arc),
                            port,
                        )
                        .await;
                    }
                    Ok(Event::InitDone) => {
                        reconcile_stale_snapshot(
                            Arc::clone(&tracked_pods),
                            &init_snapshot,
                            Arc::clone(&app_context),
                            port,
                        )
                        .await;
                        init_snapshot.clear();
                    }
                    Err(err) => {
                        watcher_ok = Err(err);
                        break;
                    }
                }
            }

            match watcher_ok {
                Ok(()) => {
                    retry_delay = Duration::from_secs(1);
                }
                Err(err) => {
                    error!("Error in Kubernetes watcher: {}", err);
                    warn!(
                        "Retrying in {} seconds with exponential backoff",
                        retry_delay.as_secs()
                    );
                    time::sleep(retry_delay).await;

                    retry_delay = std::cmp::min(retry_delay * 2, MAX_RETRY_DELAY);
                }
            }

            warn!(
                "Kubernetes watcher exited, restarting in {} seconds",
                config_arc.check_interval.as_secs()
            );
            time::sleep(config_arc.check_interval).await;
        }
    });

    Ok(handle)
}

async fn handle_pod_apply_event(
    pod: Pod,
    key: String,
    tracked_pods: Arc<Mutex<TrackedPods>>,
    app_context: Arc<AppContext>,
    config: Arc<ServiceDiscoveryConfig>,
    port: u16,
) {
    if pod.metadata.deletion_timestamp.is_some() || !PodInfo::should_include(&pod, &config) {
        handle_pod_deletion_by_key(
            &key,
            tracked_pods,
            app_context,
            port,
            metrics_labels::DEREGISTRATION_POD_DELETED,
        )
        .await;
        return;
    }

    let Some(pod_info) = PodInfo::from_pod(&pod, Some(&config)) else {
        return;
    };

    handle_pod_event(
        &key,
        &pod_info,
        tracked_pods,
        app_context,
        port,
        config.pd_mode,
    )
    .await;
}

async fn handle_pod_event(
    key: &str,
    pod_info: &PodInfo,
    tracked_pods: Arc<Mutex<TrackedPods>>,
    app_context: Arc<AppContext>,
    port: u16,
    pd_mode: bool,
) {
    let worker_url = pod_info.worker_url(port);

    if pod_info.is_healthy() {
        let (should_add, evicted, tracked_count) = {
            let mut tracker = match tracked_pods.lock() {
                Ok(tracker) => tracker,
                Err(e) => {
                    error!("Failed to acquire tracked_pods lock: {}", e);
                    return;
                }
            };

            match tracker.get(key).cloned() {
                Some(old)
                    if old.uid == pod_info.uid
                        && old.worker_url(port) == pod_info.worker_url(port) =>
                {
                    (false, None, tracker.len())
                }
                old => {
                    tracker.insert(key.to_string(), pod_info.clone());
                    (true, old, tracker.len())
                }
            }
        };

        // Submit RemoveWorker for the old entry outside the lock.
        if let Some(ref old) = evicted {
            let old_url = old.worker_url(port);
            info!(
                "Removing previous pod entry {} (old uid={}) | url: {}",
                old.name, old.uid, old_url
            );
            let job = Job::RemoveWorker {
                url: old_url.clone(),
                expected_revision: None,
            };
            if let Some(job_queue) = app_context.worker_job_queue.get() {
                if let Err(e) = job_queue.submit(job).await {
                    error!(
                        "Failed to submit removal for evicted pod {}: {}",
                        old_url, e
                    );
                }
            }
        }

        if should_add {
            info!(
                "Adding pod: {} | type: {:?} | url: {}",
                pod_info.name, pod_info.pod_type, worker_url
            );

            let worker_type = if pd_mode {
                match &pod_info.pod_type {
                    Some(PodType::Prefill) => WorkerType::Prefill,
                    Some(PodType::Decode) => WorkerType::Decode,
                    _ => WorkerType::Regular,
                }
            } else {
                WorkerType::Regular
            };

            let bootstrap_port = if pd_mode {
                match &pod_info.pod_type {
                    Some(PodType::Prefill) => pod_info.bootstrap_port,
                    _ => None,
                }
            } else {
                None
            };

            let mut spec = WorkerSpec::new(worker_url.clone());
            spec.worker_type = worker_type;
            spec.bootstrap_port = bootstrap_port;
            // Inject pod-metadata model_id as a label so the existing
            // resolution chain in create_worker.rs picks it up at
            // priority #2 (served_model_name).
            if let Some(ref override_id) = pod_info.model_id_override {
                spec.labels
                    .insert("served_model_name".to_string(), override_id.clone());
            }
            spec.api_key.clone_from(&app_context.router_config.api_key);
            // Health config is resolved at worker build time from router
            // defaults + per-worker overrides (spec.health).
            spec.max_connection_attempts = app_context
                .router_config
                .health_check
                .success_threshold
                .max(1)
                * 20;

            let config = spec;

            let job = Job::AddWorker {
                config: Box::new(config.clone()),
            };

            if let Some(job_queue) = app_context.worker_job_queue.get() {
                match job_queue.submit(job).await {
                    Ok(()) => {
                        debug!("Worker addition job submitted for: {}", worker_url);

                        // Layer 4: Record successful registration from K8s discovery
                        Metrics::record_discovery_registration(
                            metrics_labels::DISCOVERY_KUBERNETES,
                            metrics_labels::REGISTRATION_SUCCESS,
                        );

                        // Update workers discovered gauge (using count from initial lock)
                        Metrics::set_discovery_workers_discovered(
                            metrics_labels::DISCOVERY_KUBERNETES,
                            tracked_count,
                        );
                    }
                    Err(e) => {
                        error!(
                            "Failed to submit worker addition job for {}: {}",
                            worker_url, e
                        );

                        // Layer 4: Record failed registration
                        Metrics::record_discovery_registration(
                            metrics_labels::DISCOVERY_KUBERNETES,
                            metrics_labels::REGISTRATION_FAILED,
                        );

                        match tracked_pods.lock() {
                            Ok(mut tracker) => {
                                if tracker
                                    .get(key)
                                    .is_some_and(|tracked| tracked.uid == pod_info.uid)
                                {
                                    tracker.remove(key);
                                }
                            }
                            Err(e) => {
                                error!(
                                    "Lock poisoned during rollback for {}: {} -- tracked state is now inconsistent",
                                    worker_url, e
                                );
                            }
                        }
                    }
                }
            } else {
                debug!(
                    "JobQueue not initialized, skipping async worker addition for: {}",
                    worker_url
                );
            }
        } else {
            // Pod already tracked - this is a duplicate event
            Metrics::record_discovery_registration(
                metrics_labels::DISCOVERY_KUBERNETES,
                metrics_labels::REGISTRATION_DUPLICATE,
            );
        }
    }
}

async fn handle_pod_deletion(
    key: &str,
    pod_info: &PodInfo,
    tracked_pods: Arc<Mutex<TrackedPods>>,
    app_context: Arc<AppContext>,
    port: u16,
    reason: &'static str,
) {
    let worker_url = pod_info.worker_url(port);

    let was_tracked = {
        let tracked = match tracked_pods.lock() {
            Ok(tracked) => tracked,
            Err(e) => {
                error!("Failed to acquire tracked_pods lock during deletion: {}", e);
                return;
            }
        };

        tracked.get(key).is_some_and(|tracked_pod| {
            tracked_pod.uid == pod_info.uid && tracked_pod.worker_url(port) == worker_url
        })
    };

    if was_tracked {
        info!(
            "Removing pod: {} | type: {:?} | url: {}",
            pod_info.name, pod_info.pod_type, worker_url
        );

        let job = Job::RemoveWorker {
            url: worker_url.clone(),
            expected_revision: None,
        };

        if let Some(job_queue) = app_context.worker_job_queue.get() {
            if let Err(e) = job_queue.submit(job).await {
                error!(
                    "Failed to submit worker removal job for {}; keeping pod tracked for retry: {}",
                    worker_url, e
                );
            } else {
                debug!("Submitted worker removal job for {}", worker_url);

                let remaining_count = match tracked_pods.lock() {
                    Ok(mut tracked) => {
                        if tracked.get(key).is_some_and(|tracked_pod| {
                            tracked_pod.uid == pod_info.uid
                                && tracked_pod.worker_url(port) == worker_url
                        }) {
                            tracked.remove(key);
                        }
                        tracked.len()
                    }
                    Err(e) => {
                        error!(
                            "Failed to acquire tracked_pods lock after deletion submit: {}",
                            e
                        );
                        return;
                    }
                };

                // Layer 4: Record deregistration from K8s pod deletion
                Metrics::record_discovery_deregistration(
                    metrics_labels::DISCOVERY_KUBERNETES,
                    reason,
                );

                // Update workers discovered gauge (using count from initial lock)
                Metrics::set_discovery_workers_discovered(
                    metrics_labels::DISCOVERY_KUBERNETES,
                    remaining_count,
                );
            }
        } else {
            error!(
                "JobQueue not initialized, cannot remove worker {}; keeping pod tracked for retry",
                worker_url
            );
        }
    } else {
        debug!(
            "Pod deletion event for untracked/already removed pod: {} (type: {:?}). Worker URL: {}",
            pod_info.name, pod_info.pod_type, worker_url
        );
    }
}

async fn handle_pod_deletion_by_key(
    key: &str,
    tracked_pods: Arc<Mutex<TrackedPods>>,
    app_context: Arc<AppContext>,
    port: u16,
    reason: &'static str,
) {
    let stored = match tracked_pods.lock() {
        Ok(tracked) => tracked.get(key).cloned(),
        Err(e) => {
            error!(
                "Failed to acquire tracked_pods lock during deletion lookup: {}",
                e
            );
            return;
        }
    };

    if let Some(pod_info) = stored {
        handle_pod_deletion(key, &pod_info, tracked_pods, app_context, port, reason).await;
    } else {
        debug!("Pod deletion event for untracked pod key: {}", key);
    }
}

async fn reconcile_stale_snapshot(
    tracked_pods: Arc<Mutex<TrackedPods>>,
    init_snapshot: &HashSet<String>,
    app_context: Arc<AppContext>,
    port: u16,
) {
    let stale_keys: Vec<String> = match tracked_pods.lock() {
        Ok(tracked) => tracked
            .keys()
            .filter(|key| !init_snapshot.contains(*key))
            .cloned()
            .collect(),
        Err(e) => {
            error!(
                "Failed to acquire tracked_pods lock during watcher reconciliation: {}",
                e
            );
            return;
        }
    };

    for key in stale_keys {
        handle_pod_deletion_by_key(
            &key,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
            metrics_labels::DEREGISTRATION_RECONCILIATION,
        )
        .await;
    }
}

/// Build the set of live pods from a K8s pod list, filtering by config selectors
/// and excluding pods with a deletion timestamp.
fn build_live_pod_set(pod_list: &[Pod], config: &ServiceDiscoveryConfig) -> TrackedPods {
    let mut live_pods = HashMap::new();
    for pod in pod_list {
        if !PodInfo::should_include(pod, config) {
            continue;
        }
        if pod.metadata.deletion_timestamp.is_some() {
            continue;
        }
        if let Some(info) = PodInfo::from_pod(pod, Some(config)) {
            live_pods.insert(tracking_key(pod), info);
        }
    }
    live_pods
}

/// Compute the reconciliation diff between tracked and live pod sets.
///
/// Returns `(stale, missing)` where:
/// - `stale`: pods in `tracked` but not in `live` (should be removed)
/// - `missing`: pods in `live` but not in `tracked` that are healthy (should be added)
fn compute_reconciliation_diff(tracked: &TrackedPods, live: &TrackedPods) -> (PodDiff, PodDiff) {
    let stale: PodDiff = tracked
        .iter()
        .filter_map(|(key, tracked_pod)| match live.get(key) {
            Some(live_pod) if live_pod.uid == tracked_pod.uid => None,
            _ => Some((key.clone(), tracked_pod.clone())),
        })
        .collect();
    let missing: PodDiff = live
        .iter()
        .filter_map(|(key, live_pod)| match tracked.get(key) {
            Some(tracked_pod) if tracked_pod.uid == live_pod.uid => None,
            _ if live_pod.is_healthy() => Some((key.clone(), live_pod.clone())),
            _ => None,
        })
        .collect();
    (stale, missing)
}

/// Reconcile the tracked pod set with actual Kubernetes state.
///
/// Performs a full pod list via the K8s API and compares with `tracked_pods`:
/// - Pods in `tracked_pods` but no longer in K8s → submit `RemoveWorker` job
/// - Healthy pods in K8s but missing from `tracked_pods` → submit `AddWorker` job
///
/// This closes two gaps in the event-driven watcher:
/// 1. Missed deletion events (pod force-deleted, watcher down during delete)
/// 2. Missed creation events (pod created while watcher was restarting)
async fn reconcile_pods(
    pods: &Api<Pod>,
    config: Arc<ServiceDiscoveryConfig>,
    tracked_pods: Arc<Mutex<TrackedPods>>,
    app_context: Arc<AppContext>,
    port: u16,
) {
    let reconcile_start = time::Instant::now();
    let label_selector = config.list_label_selector();
    let list_params = if label_selector.is_empty() {
        ListParams::default()
    } else {
        ListParams::default().labels(&label_selector)
    };
    let pod_list = match pods.list(&list_params).await {
        Ok(list) => list,
        Err(e) => {
            error!("Reconciliation: failed to list pods: {}", e);
            return;
        }
    };

    // Build the set of live pods that match our selectors.
    // Include all non-deleted pods regardless of health: the router's own health
    // checker handles unhealthy workers. Only pods completely gone from K8s are stale.
    let live_pods = build_live_pod_set(&pod_list.items, &config);

    // Diff: stale = tracked but not live, missing = live-and-healthy but not tracked
    let (stale, missing) = {
        let tracked = match tracked_pods.lock() {
            Ok(t) => t,
            Err(e) => {
                error!("Reconciliation: failed to acquire lock: {}", e);
                return;
            }
        };
        compute_reconciliation_diff(&tracked, &live_pods)
    };

    if stale.is_empty() && missing.is_empty() {
        debug!("Reconciliation: tracked state is consistent with K8s");
        return;
    }

    info!(
        "Reconciliation: removing {} stale, adding {} missing pods",
        stale.len(),
        missing.len()
    );

    for (key, pod_info) in &stale {
        let worker_url = pod_info.worker_url(port);
        info!(
            "Reconciliation: removing stale pod {} (uid={}) | url: {}",
            pod_info.name, pod_info.uid, worker_url
        );
        // Use the snapshotted PodInfo (with its UID/url) rather than
        // re-resolving by key, so that a concurrent watcher update that
        // replaced the entry with a new healthy pod cannot cause us to
        // deregister the replacement instead of the stale one.
        handle_pod_deletion(
            key,
            pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
            metrics_labels::DEREGISTRATION_RECONCILED,
        )
        .await;
    }

    // Add missing workers, tracking how many succeed.
    let mut added = 0usize;
    for (key, pod_info) in &missing {
        let pre = tracked_pods.lock().map(|t| t.len()).unwrap_or(0);
        handle_pod_event(
            key,
            pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
            config.pd_mode,
        )
        .await;
        let post = tracked_pods.lock().map(|t| t.len()).unwrap_or(0);
        if post > pre {
            added += 1;
        }
    }
    if !missing.is_empty() && added < missing.len() {
        warn!(
            "Reconciliation: only {}/{} missing pods were successfully added",
            added,
            missing.len()
        );
    }

    // Update gauge to final post-reconciliation count
    match tracked_pods.lock() {
        Ok(tracker) => {
            Metrics::set_discovery_workers_discovered(
                metrics_labels::DISCOVERY_KUBERNETES,
                tracker.len(),
            );
        }
        Err(e) => {
            error!("Reconciliation: lock poisoned during gauge update: {}", e);
        }
    }

    // Record reconciliation cycle duration for observability.
    Metrics::record_discovery_sync_duration(
        metrics_labels::DISCOVERY_KUBERNETES,
        reconcile_start.elapsed(),
    );
}

/// Start router node discovery for mesh cluster
async fn start_router_discovery(
    config: Arc<ServiceDiscoveryConfig>,
    pods: Api<Pod>,
    cluster_state: ClusterState,
    default_mesh_port: u16,
) {
    let mut retry_delay = Duration::from_secs(1);
    let mut tracked_router_nodes = HashMap::new();
    const MAX_RETRY_DELAY: Duration = Duration::from_secs(300);

    loop {
        let watcher_config = build_watcher_config("router", &config.router_label_selector());
        let mut watcher_stream = std::pin::pin!(watcher(pods.clone(), watcher_config));
        let mut init_snapshot = HashSet::new();
        let mut watcher_ok = Ok(());

        while let Some(event_result) = watcher_stream.next().await {
            match event_result {
                Ok(Event::Apply(pod)) => {
                    let key = tracking_key(&pod);
                    handle_router_apply_event(
                        pod,
                        &key,
                        Arc::clone(&config),
                        cluster_state.clone(),
                        default_mesh_port,
                        &mut tracked_router_nodes,
                    );
                }
                Ok(Event::Delete(pod)) => {
                    let key = tracking_key(&pod);
                    handle_router_delete_event(
                        &key,
                        cluster_state.clone(),
                        &mut tracked_router_nodes,
                    );
                }
                Ok(Event::Init) => {
                    init_snapshot.clear();
                }
                Ok(Event::InitApply(pod)) => {
                    let key = tracking_key(&pod);
                    init_snapshot.insert(key.clone());
                    handle_router_apply_event(
                        pod,
                        &key,
                        Arc::clone(&config),
                        cluster_state.clone(),
                        default_mesh_port,
                        &mut tracked_router_nodes,
                    );
                }
                Ok(Event::InitDone) => {
                    reconcile_router_snapshot(
                        &mut tracked_router_nodes,
                        &init_snapshot,
                        cluster_state.clone(),
                    );
                    init_snapshot.clear();
                }
                Err(err) => {
                    watcher_ok = Err(err);
                    break;
                }
            }
        }

        match watcher_ok {
            Ok(()) => {
                retry_delay = Duration::from_secs(1);
            }
            Err(err) => {
                error!("Error in router discovery watcher: {}", err);
                warn!(
                    "Retrying router discovery in {} seconds with exponential backoff",
                    retry_delay.as_secs()
                );
                time::sleep(retry_delay).await;

                retry_delay = std::cmp::min(retry_delay * 2, MAX_RETRY_DELAY);
            }
        }

        warn!(
            "Router discovery watcher exited, restarting in {} seconds",
            config.check_interval.as_secs()
        );
        time::sleep(config.check_interval).await;
    }
}

fn handle_router_apply_event(
    pod: Pod,
    key: &str,
    config: Arc<ServiceDiscoveryConfig>,
    cluster_state: ClusterState,
    default_mesh_port: u16,
    tracked_router_nodes: &mut TrackedRouterNodes,
) {
    if pod.metadata.deletion_timestamp.is_some()
        || !PodInfo::matches_selector(&pod, &config.router_selector)
    {
        handle_router_delete_event(key, cluster_state, tracked_router_nodes);
        return;
    }

    let Some(pod_info) = PodInfo::from_pod(&pod, Some(&config)) else {
        return;
    };

    if !pod_info.is_router {
        return;
    }

    if pod_info.is_healthy() {
        let mesh_port = pod_info.mesh_port.unwrap_or(default_mesh_port);
        let node_address = format!("{}:{}", pod_info.ip, mesh_port);
        let node_name = pod_info.name.clone();
        let mut state = cluster_state.write();
        if let Some(previous_node_name) = tracked_router_nodes.get(key) {
            if previous_node_name != &node_name {
                if let Some(node) = state.get_mut(previous_node_name) {
                    node.status = NodeStatus::Down as i32;
                    node.version += 1;
                }
            }
        }
        let existing_version = state.get(&node_name).map(|n| n.version).unwrap_or(0);

        let node_state = NodeState {
            name: node_name.clone(),
            address: node_address,
            status: NodeStatus::Alive as i32,
            version: existing_version + 1,
            metadata: HashMap::new(),
        };

        state.insert(node_name.clone(), node_state.clone());
        tracked_router_nodes.insert(key.to_string(), node_name.clone());
        info!(
            "Router node {} added/updated in mesh cluster from pod {} (address: {})",
            node_name, key, node_state.address
        );
    } else if let Some(node_name) = tracked_router_nodes.get(key) {
        let mut state = cluster_state.write();
        if let Some(node) = state.get_mut(node_name) {
            if node.status != NodeStatus::Down as i32 {
                node.status = NodeStatus::Suspected as i32;
                node.version += 1;
                debug!(
                    "Router node {} marked as Suspected (pod {} not healthy)",
                    node_name, key
                );
            }
        }
    }
}

fn handle_router_delete_event(
    key: &str,
    cluster_state: ClusterState,
    tracked_router_nodes: &mut TrackedRouterNodes,
) {
    let Some(node_name) = tracked_router_nodes.remove(key) else {
        debug!("Router node {} not tracked by this discovery loop", key);
        return;
    };

    let mut state = cluster_state.write();
    if let Some(node) = state.get_mut(&node_name) {
        node.status = NodeStatus::Down as i32;
        node.version += 1;
        info!(
            "Router node {} marked as Down (pod {} deleted)",
            node_name, key
        );
    } else {
        debug!("Router node {} not found in cluster state", node_name);
    }
}

fn reconcile_router_snapshot(
    tracked_router_nodes: &mut TrackedRouterNodes,
    init_snapshot: &HashSet<String>,
    cluster_state: ClusterState,
) {
    let stale_keys: Vec<String> = tracked_router_nodes
        .keys()
        .filter(|key| !init_snapshot.contains(*key))
        .cloned()
        .collect();

    for key in stale_keys {
        handle_router_delete_event(&key, cluster_state.clone(), tracked_router_nodes);
    }
}

#[cfg(test)]
mod tests {
    use k8s_openapi::{
        api::core::v1::{Pod, PodCondition, PodSpec, PodStatus},
        apimachinery::pkg::apis::meta::v1::{ObjectMeta, Time},
    };
    use openai_protocol::worker::WorkerStatus;
    use tracing_test::traced_test;

    use super::*;
    use crate::{
        routers::{common::openai_bridge, grpc::multimodal::MultimodalConfigRegistry},
        worker::{BasicWorkerBuilder, Worker},
        workflow::{JobQueue, JobQueueConfig, WorkflowEngines},
    };

    fn create_k8s_pod(
        name: Option<&str>,
        ip: Option<&str>,
        phase: Option<&str>,
        ready_status: Option<&str>,
        deletion_timestamp: Option<Time>,
    ) -> Pod {
        let mut pod = Pod {
            metadata: ObjectMeta {
                name: name.map(String::from),
                uid: name.map(|n| format!("uid-{n}")),
                deletion_timestamp,
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: None,
        };

        if ip.is_some() || phase.is_some() || ready_status.is_some() {
            let mut pod_status = PodStatus {
                pod_ip: ip.map(String::from),
                phase: phase.map(String::from),
                conditions: None,
                ..Default::default()
            };

            if let Some(status_str) = ready_status {
                let condition = PodCondition {
                    type_: "Ready".to_string(),
                    status: status_str.to_string(),
                    last_probe_time: None,
                    last_transition_time: None,
                    message: None,
                    reason: None,
                    observed_generation: None,
                };
                pod_status.conditions = Some(vec![condition]);
            }
            pod.status = Some(pod_status);
        }
        pod
    }

    fn create_pd_k8s_pod(name: &str, ip: &str, pod_type: &str, bootstrap_port: Option<u16>) -> Pod {
        let mut labels = std::collections::BTreeMap::new();
        labels.insert("app".to_string(), "sglang".to_string());
        labels.insert("component".to_string(), pod_type.to_string());

        let mut annotations = std::collections::BTreeMap::new();
        if let Some(port) = bootstrap_port {
            annotations.insert("sglang.ai/bootstrap-port".to_string(), port.to_string());
        }

        Pod {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                uid: Some(format!("uid-{name}")),
                labels: Some(labels),
                annotations: Some(annotations),
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: Some(PodStatus {
                pod_ip: Some(ip.to_string()),
                phase: Some("Running".to_string()),
                conditions: Some(vec![PodCondition {
                    type_: "Ready".to_string(),
                    status: "True".to_string(),
                    last_probe_time: None,
                    last_transition_time: None,
                    message: None,
                    reason: None,
                    observed_generation: None,
                }]),
                ..Default::default()
            }),
        }
    }

    fn create_test_app_context() -> Arc<AppContext> {
        use crate::{
            config::RouterConfig, middleware::TokenBucket,
            observability::inflight_tracker::InFlightRequestTracker,
            routers::openai::realtime::RealtimeRegistry, worker::WorkerService,
        };

        let router_config = RouterConfig::builder()
            .worker_startup_timeout_secs(1)
            .build_unchecked();

        let worker_registry = Arc::new(crate::worker::WorkerRegistry::new());
        let worker_job_queue = Arc::new(std::sync::OnceLock::new());

        // Note: Using uninitialized queue for tests to avoid spawning background workers
        // Jobs submitted during tests will queue but not be processed
        Arc::new(AppContext {
            client: reqwest::Client::new(),
            router_config: router_config.clone(),
            rate_limiter: Some(Arc::new(TokenBucket::new(1000, 1000))),
            worker_registry: worker_registry.clone(),
            policy_registry: Arc::new(crate::policies::PolicyRegistry::new(
                router_config.policy.clone(),
            )),
            reasoning_parser_factory: None,
            tool_parser_factory: None,
            router_manager: None,
            response_storage: Arc::new(smg_data_connector::MemoryResponseStorage::new()),
            conversation_storage: Arc::new(smg_data_connector::MemoryConversationStorage::new()),
            conversation_item_storage: Arc::new(
                smg_data_connector::MemoryConversationItemStorage::new(),
            ),
            conversation_memory_writer: Arc::new(
                smg_data_connector::NoOpConversationMemoryWriter::new(),
            ),
            background_repository: None,
            worker_monitor: None,
            configured_reasoning_parser: None,
            configured_tool_parser: None,
            worker_job_queue: worker_job_queue.clone(),
            workflow_engines: Arc::new(std::sync::OnceLock::new()),
            mcp_orchestrator: Arc::new(std::sync::OnceLock::new()),
            mcp_format_registry: openai_bridge::FormatRegistry::new(),
            tokenizer_registry: Arc::new(llm_tokenizer::registry::TokenizerRegistry::new()),
            multimodal_config_registry: Arc::new(MultimodalConfigRegistry::new()),
            skill_service: None,
            wasm_manager: None,
            worker_service: Arc::new(WorkerService::new(
                worker_registry,
                worker_job_queue,
                router_config,
            )),
            inflight_tracker: InFlightRequestTracker::new(),
            kv_event_monitor: None,
            realtime_registry: Arc::new(RealtimeRegistry::new()),
            webrtc_bind_addr: None,
            webrtc_stun_server: None,
        })
    }

    fn create_test_app_context_with_job_queue() -> Arc<AppContext> {
        use std::sync::OnceLock;

        use crate::{
            config::RouterConfig, middleware::TokenBucket,
            observability::inflight_tracker::InFlightRequestTracker,
            routers::openai::realtime::RealtimeRegistry, worker::WorkerService,
        };

        let router_config = RouterConfig::builder()
            .worker_startup_timeout_secs(1)
            .health_check_config(crate::config::HealthCheckConfig {
                disable_health_check: true,
                drain_settle_secs: 0,
                ..Default::default()
            })
            .build_unchecked();
        let worker_registry = Arc::new(crate::worker::WorkerRegistry::new());
        let worker_job_queue = Arc::new(OnceLock::new());
        let workflow_engines = Arc::new(OnceLock::new());
        workflow_engines
            .set(WorkflowEngines::new(&router_config))
            .expect("workflow engines should be set once");

        Arc::new_cyclic(|weak| {
            let job_queue = JobQueue::new(
                JobQueueConfig {
                    queue_capacity: 32,
                    max_concurrent_jobs: 1,
                },
                weak.clone(),
            );
            worker_job_queue
                .set(job_queue)
                .expect("job queue should be set once");

            AppContext {
                client: reqwest::Client::new(),
                router_config: router_config.clone(),
                rate_limiter: Some(Arc::new(TokenBucket::new(1000, 1000))),
                worker_registry: Arc::clone(&worker_registry),
                policy_registry: Arc::new(crate::policies::PolicyRegistry::new(
                    router_config.policy.clone(),
                )),
                reasoning_parser_factory: None,
                tool_parser_factory: None,
                router_manager: None,
                response_storage: Arc::new(smg_data_connector::MemoryResponseStorage::new()),
                conversation_storage: Arc::new(smg_data_connector::MemoryConversationStorage::new()),
                conversation_item_storage: Arc::new(
                    smg_data_connector::MemoryConversationItemStorage::new(),
                ),
                conversation_memory_writer: Arc::new(
                    smg_data_connector::NoOpConversationMemoryWriter::new(),
                ),
                background_repository: None,
                worker_monitor: None,
                configured_reasoning_parser: None,
                configured_tool_parser: None,
                worker_job_queue: Arc::clone(&worker_job_queue),
                workflow_engines: Arc::clone(&workflow_engines),
                mcp_orchestrator: Arc::new(OnceLock::new()),
                mcp_format_registry: openai_bridge::FormatRegistry::new(),
                tokenizer_registry: Arc::new(llm_tokenizer::registry::TokenizerRegistry::new()),
                multimodal_config_registry: Arc::new(MultimodalConfigRegistry::new()),
                skill_service: None,
                wasm_manager: None,
                worker_service: Arc::new(WorkerService::new(
                    Arc::clone(&worker_registry),
                    Arc::clone(&worker_job_queue),
                    router_config.clone(),
                )),
                inflight_tracker: InFlightRequestTracker::new(),
                kv_event_monitor: None,
                realtime_registry: Arc::new(RealtimeRegistry::new()),
                webrtc_bind_addr: None,
                webrtc_stun_server: None,
            }
        })
    }

    fn register_test_worker(app_context: &Arc<AppContext>, url: &str, worker_type: WorkerType) {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(worker_type)
                .health_config(openai_protocol::worker::HealthCheckConfig {
                    disable_health_check: true,
                    drain_settle_secs: 0,
                    ..Default::default()
                })
                .status(WorkerStatus::Ready)
                .build(),
        );
        app_context.worker_registry.register_or_replace(worker);
    }

    async fn wait_for_worker_absent(app_context: &Arc<AppContext>, url: &str) {
        let deadline = time::Instant::now() + Duration::from_secs(5);
        while time::Instant::now() < deadline {
            if app_context.worker_registry.get_by_url(url).is_none() {
                return;
            }
            time::sleep(Duration::from_millis(20)).await;
        }
        panic!("worker {url} was not removed");
    }

    fn create_pd_config() -> ServiceDiscoveryConfig {
        let mut prefill_selector = HashMap::new();
        prefill_selector.insert("app".to_string(), "sglang".to_string());
        prefill_selector.insert("component".to_string(), "prefill".to_string());

        let mut decode_selector = HashMap::new();
        decode_selector.insert("app".to_string(), "sglang".to_string());
        decode_selector.insert("component".to_string(), "decode".to_string());

        ServiceDiscoveryConfig {
            enabled: true,
            selector: HashMap::new(),
            check_interval: Duration::from_secs(60),
            port: 8080,
            namespace: None,
            pd_mode: true,
            prefill_selector,
            decode_selector,
            bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
            router_selector: HashMap::new(),
            router_mesh_port_annotation: "sglang.ai/mesh-port".to_string(),
            model_id_source: None,
        }
    }

    #[test]
    fn test_pod_info_should_include() {
        let config = create_pd_config();

        let prefill_pod = create_pd_k8s_pod("prefill-pod", "10.0.0.1", "prefill", Some(8081));
        assert!(PodInfo::should_include(&prefill_pod, &config));

        let decode_pod = create_pd_k8s_pod("decode-pod", "10.0.0.2", "decode", None);
        assert!(PodInfo::should_include(&decode_pod, &config));

        let unmatched_pod = create_pd_k8s_pod("other-pod", "10.0.0.3", "other", None);
        assert!(!PodInfo::should_include(&unmatched_pod, &config));

        let mut regular_config = ServiceDiscoveryConfig::default();
        regular_config
            .selector
            .insert("app".to_string(), "sglang".to_string());
        regular_config.pd_mode = false;

        let regular_pod = create_pd_k8s_pod("worker-pod", "10.0.0.4", "worker", None);
        assert!(PodInfo::should_include(&regular_pod, &regular_config));
    }

    #[test]
    fn test_service_discovery_config_default() {
        let config = ServiceDiscoveryConfig::default();
        assert!(!config.enabled);
        assert!(config.selector.is_empty());
        assert_eq!(config.check_interval, Duration::from_secs(60));
        assert_eq!(config.port, 8000);
        assert!(config.namespace.is_none());
        assert!(!config.pd_mode);
        assert!(config.prefill_selector.is_empty());
        assert!(config.decode_selector.is_empty());
        assert_eq!(config.bootstrap_port_annotation, "sglang.ai/bootstrap-port");
    }

    #[test]
    fn test_pod_type_enum() {
        let prefill = PodType::Prefill;
        let decode = PodType::Decode;
        let regular = PodType::Regular;

        assert_eq!(format!("{prefill:?}"), "Prefill");
        assert_eq!(format!("{decode:?}"), "Decode");
        assert_eq!(format!("{regular:?}"), "Regular");
    }

    #[test]
    fn test_pod_info_from_pod_valid() {
        let k8s_pod = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            Some("True"),
            None,
        );
        let pod_info = PodInfo::from_pod(&k8s_pod, None).unwrap();
        assert_eq!(pod_info.name, "test-pod");
        assert_eq!(pod_info.ip, "10.0.0.1");
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert!(pod_info.pod_type.is_none());
        assert!(pod_info.bootstrap_port.is_none());
    }

    #[test]
    fn test_pod_info_from_pod_with_pd_config_prefill() {
        let k8s_pod = create_pd_k8s_pod("prefill-pod", "10.0.0.1", "prefill", Some(8081));
        let config = create_pd_config();

        let pod_info = PodInfo::from_pod(&k8s_pod, Some(&config)).unwrap();
        assert_eq!(pod_info.name, "prefill-pod");
        assert_eq!(pod_info.ip, "10.0.0.1");
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert_eq!(pod_info.pod_type, Some(PodType::Prefill));
        assert_eq!(pod_info.bootstrap_port, Some(8081));
    }

    #[test]
    fn test_pod_info_from_pod_with_pd_config_decode() {
        let k8s_pod = create_pd_k8s_pod("decode-pod", "10.0.0.2", "decode", None);
        let config = create_pd_config();

        let pod_info = PodInfo::from_pod(&k8s_pod, Some(&config)).unwrap();
        assert_eq!(pod_info.name, "decode-pod");
        assert_eq!(pod_info.ip, "10.0.0.2");
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert_eq!(pod_info.pod_type, Some(PodType::Decode));
        assert!(pod_info.bootstrap_port.is_none());
    }

    #[test]
    fn test_pod_info_from_pod_with_pd_config_regular_mode() {
        let k8s_pod = create_pd_k8s_pod("regular-pod", "10.0.0.3", "worker", None);
        let mut config = create_pd_config();
        config.pd_mode = false;

        let pod_info = PodInfo::from_pod(&k8s_pod, Some(&config)).unwrap();
        assert_eq!(pod_info.name, "regular-pod");
        assert_eq!(pod_info.ip, "10.0.0.3");
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert_eq!(pod_info.pod_type, Some(PodType::Regular));
        assert!(pod_info.bootstrap_port.is_none());
    }

    #[test]
    fn test_pod_info_from_pod_with_pd_config_unmatched_labels() {
        let k8s_pod = create_pd_k8s_pod("unknown-pod", "10.0.0.4", "unknown", None);
        let config = create_pd_config();

        let pod_info = PodInfo::from_pod(&k8s_pod, Some(&config)).unwrap();
        assert_eq!(pod_info.name, "unknown-pod");
        assert_eq!(pod_info.ip, "10.0.0.4");
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert_eq!(pod_info.pod_type, Some(PodType::Regular));
        assert!(pod_info.bootstrap_port.is_none());
    }

    #[test]
    fn test_pod_info_from_pod_with_pd_config_invalid_bootstrap_port() {
        let mut pod = create_pd_k8s_pod("prefill-pod", "10.0.0.1", "prefill", None);
        pod.metadata.annotations.as_mut().unwrap().insert(
            "sglang.ai/bootstrap-port".to_string(),
            "invalid".to_string(),
        );
        let config = create_pd_config();

        let pod_info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
        assert_eq!(pod_info.pod_type, Some(PodType::Prefill));
        assert!(pod_info.bootstrap_port.is_none());
    }

    #[test]
    fn test_pod_info_from_pod_not_ready() {
        let k8s_pod = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            Some("False"),
            None,
        );
        let pod_info = PodInfo::from_pod(&k8s_pod, None).unwrap();
        assert!(!pod_info.is_ready);
    }

    #[test]
    fn test_pod_info_from_pod_no_conditions() {
        let k8s_pod = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            None,
            None,
        );
        let pod_info = PodInfo::from_pod(&k8s_pod, None).unwrap();
        assert!(!pod_info.is_ready);
    }

    #[test]
    fn test_pod_info_from_pod_missing_name() {
        let k8s_pod = create_k8s_pod(None, Some("10.0.0.1"), Some("Running"), Some("True"), None);
        assert!(PodInfo::from_pod(&k8s_pod, None).is_none());
    }

    #[test]
    fn test_pod_info_from_pod_missing_ip() {
        let k8s_pod = create_k8s_pod(Some("test-pod"), None, Some("Running"), Some("True"), None);
        assert!(PodInfo::from_pod(&k8s_pod, None).is_none());
    }

    #[test]
    fn test_pod_info_from_pod_missing_status_phase() {
        let k8s_pod = create_k8s_pod(Some("test-pod"), Some("10.0.0.1"), None, Some("True"), None);
        let pod_info = PodInfo::from_pod(&k8s_pod, None).unwrap();
        assert_eq!(pod_info.status, "Unknown");
    }

    #[test]
    fn test_pod_info_from_pod_no_status_object() {
        let mut k8s_pod = create_k8s_pod(Some("test-pod"), None, None, None, None);
        k8s_pod.status = None;
        assert!(PodInfo::from_pod(&k8s_pod, None).is_none());
    }

    #[test]
    fn test_pod_info_is_healthy() {
        let healthy_pod = PodInfo {
            name: "p1".into(),
            uid: "uid-p1".into(),
            ip: "1.1.1.1".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };
        assert!(healthy_pod.is_healthy());

        let not_ready_pod = PodInfo {
            name: "p2".into(),
            uid: "uid-p2".into(),
            ip: "1.1.1.2".into(),
            status: "Running".into(),
            is_ready: false,
            pod_type: None,
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };
        assert!(!not_ready_pod.is_healthy());

        let not_running_pod = PodInfo {
            name: "p3".into(),
            uid: "uid-p3".into(),
            ip: "1.1.1.3".into(),
            status: "Pending".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };
        assert!(!not_running_pod.is_healthy());
    }

    #[test]
    fn test_pod_info_identity_based_equality() {
        // PodInfo equality is based on (name, uid) only — mutable fields like
        // status, is_ready, ip, and pod_type do not affect identity.
        let pod1 = PodInfo {
            name: "pod1".into(),
            uid: "uid-1".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Prefill),
            bootstrap_port: Some(8081),
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };

        let pod2_same_identity = PodInfo {
            name: "pod1".into(),
            uid: "uid-1".into(),
            ip: "1.2.3.4".into(),
            status: "Pending".into(),
            is_ready: false,
            pod_type: Some(PodType::Decode),
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };

        let pod3_different_identity = PodInfo {
            name: "pod2".into(),
            uid: "uid-2".into(),
            ip: "1.2.3.5".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Prefill),
            bootstrap_port: Some(8081),
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };

        // Same (name, uid) → equal, even with different status/type/readiness
        assert_eq!(pod1, pod2_same_identity);
        // Different (name, uid) → not equal
        assert_ne!(pod1, pod3_different_identity);
    }

    #[tokio::test]
    async fn test_handle_pod_event_add_unhealthy_pod() {
        let app_context = create_test_app_context();
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "pod1".into(),
            uid: "uid-pod1".into(),
            ip: "1.2.3.4".into(),
            status: "Pending".into(),
            is_ready: false,
            pod_type: None,
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };
        let port = 8080u16;

        handle_pod_event(
            "default/pod1",
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
            false, // pd_mode = false
        )
        .await;

        assert!(!tracked_pods.lock().unwrap().contains_key("default/pod1"));
    }

    #[tokio::test]
    async fn test_handle_pod_deletion_non_existing_pod() {
        let app_context = create_test_app_context();
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "pod1".into(),
            uid: "uid-pod1".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };
        let port = 8080u16;

        handle_pod_deletion(
            "default/pod1",
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
            metrics_labels::DEREGISTRATION_POD_DELETED,
        )
        .await;

        assert!(tracked_pods.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_handle_pd_pod_event_prefill_pod() {
        let app_context = create_test_app_context();
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "prefill-pod".into(),
            uid: "uid-prefill-pod".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Prefill),
            bootstrap_port: Some(8081),
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };
        let port = 8080u16;

        handle_pod_event(
            "default/prefill-pod",
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
            true, // pd_mode = true for PD pod
        )
        .await;

        // With fully async control plane, pod is tracked and job is queued
        // Worker registration and validation happen in background job
        assert_eq!(
            tracked_pods.lock().unwrap().get("default/prefill-pod"),
            Some(&pod_info)
        );

        // Note: In tests with uninitialized queue, background jobs don't process
        // Worker won't appear in registry until background job runs (in production)
    }

    #[tokio::test]
    async fn test_handle_pd_pod_event_decode_pod() {
        let app_context = create_test_app_context();
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "decode-pod".into(),
            uid: "uid-decode-pod".into(),
            ip: "1.2.3.5".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Decode),
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };
        let port = 8080u16;

        handle_pod_event(
            "default/decode-pod",
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
            true, // pd_mode = true for PD pod
        )
        .await;

        // With fully async control plane, pod is tracked and job is queued
        // Worker registration and validation happen in background job
        assert_eq!(
            tracked_pods.lock().unwrap().get("default/decode-pod"),
            Some(&pod_info)
        );

        // Note: In tests with uninitialized queue, background jobs don't process
        // Worker won't appear in registry until background job runs (in production)
    }

    #[tokio::test]
    async fn test_handle_pd_pod_deletion_tracked_pod() {
        let app_context = create_test_app_context_with_job_queue();
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "test-pod".into(),
            uid: "uid-test-pod".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Prefill),
            bootstrap_port: Some(8081),
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };

        // Add pod to tracked set first
        {
            let mut tracked = tracked_pods.lock().unwrap();
            tracked.insert("default/test-pod".to_string(), pod_info.clone());
        }

        let port = 8080u16;
        let worker_url = pod_info.worker_url(port);
        register_test_worker(&app_context, &worker_url, WorkerType::Prefill);

        handle_pod_deletion(
            "default/test-pod",
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
            metrics_labels::DEREGISTRATION_POD_DELETED,
        )
        .await;

        // Pod should be removed from tracking
        assert!(!tracked_pods
            .lock()
            .unwrap()
            .contains_key("default/test-pod"));
        wait_for_worker_absent(&app_context, &worker_url).await;
    }

    #[tokio::test]
    async fn test_handle_pod_deletion_keeps_tracked_pod_when_submit_fails() {
        let app_context = create_test_app_context();
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info =
            make_pod_info_with_uid("retry-pod", "uid-retry-pod", "1.2.3.4", "Running", true);
        tracked_pods
            .lock()
            .unwrap()
            .insert("default/retry-pod".to_string(), pod_info.clone());

        handle_pod_deletion(
            "default/retry-pod",
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            8080,
            metrics_labels::DEREGISTRATION_RECONCILED,
        )
        .await;

        assert_eq!(
            tracked_pods.lock().unwrap().get("default/retry-pod"),
            Some(&pod_info)
        );
    }

    #[tokio::test]
    async fn test_handle_pd_pod_deletion_untracked_pod() {
        let app_context = create_test_app_context();
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "untracked-pod".into(),
            uid: "uid-untracked-pod".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Decode),
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };
        let port = 8080u16;

        // Don't add pod to tracked set

        handle_pod_deletion(
            "default/untracked-pod",
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
            metrics_labels::DEREGISTRATION_POD_DELETED,
        )
        .await;

        // Tracked set should remain empty
        assert!(tracked_pods.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_unified_handler_regular_mode() {
        let app_context = create_test_app_context();
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "regular-pod".into(),
            uid: "uid-regular-pod".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Regular),
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };
        let port = 8080u16;

        handle_pod_event(
            "default/regular-pod",
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
            false, // pd_mode = false
        )
        .await;

        // With fully async control plane, pod is tracked and job is queued
        // In regular mode (pd_mode=false), worker_type defaults to Regular
        // Worker registration and validation happen in background job
        assert_eq!(
            tracked_pods.lock().unwrap().get("default/regular-pod"),
            Some(&pod_info)
        );

        // Note: In tests with uninitialized queue, background jobs don't process
        // Worker won't appear in registry until background job runs (in production)
    }

    #[tokio::test]
    async fn test_unified_handler_pd_mode_with_prefill() {
        let app_context = create_test_app_context();
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "prefill-pod".into(),
            uid: "uid-prefill-pod-2".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Prefill),
            bootstrap_port: Some(8081),
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };
        let port = 8080u16;

        handle_pod_event(
            "default/prefill-pod",
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
            true, // pd_mode = true
        )
        .await;

        // With fully async control plane, pod is tracked and job is queued
        // Worker registration and validation happen in background job
        assert_eq!(
            tracked_pods.lock().unwrap().get("default/prefill-pod"),
            Some(&pod_info)
        );

        // Note: In tests with uninitialized queue, background jobs don't process
        // Worker won't appear in registry until background job runs (in production)
    }

    #[tokio::test]
    async fn test_unified_handler_deletion_with_pd_mode() {
        let app_context = create_test_app_context_with_job_queue();
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "decode-pod".into(),
            uid: "uid-decode-pod-2".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Decode),
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };

        // Add pod to tracked set first
        {
            let mut tracked = tracked_pods.lock().unwrap();
            tracked.insert("default/decode-pod".to_string(), pod_info.clone());
        }

        let port = 8080u16;
        let worker_url = pod_info.worker_url(port);
        register_test_worker(&app_context, &worker_url, WorkerType::Decode);

        handle_pod_deletion(
            "default/decode-pod",
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
            metrics_labels::DEREGISTRATION_POD_DELETED,
        )
        .await;

        // Pod should be removed from tracking
        assert!(!tracked_pods
            .lock()
            .unwrap()
            .contains_key("default/decode-pod"));
        wait_for_worker_absent(&app_context, &worker_url).await;
    }

    #[tokio::test]
    async fn test_delete_event_with_missing_pod_ip_uses_stored_worker_url() {
        let app_context = create_test_app_context_with_job_queue();
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let port = 8080u16;
        let pod_info = PodInfo {
            name: "worker-0".into(),
            uid: "uid-worker-0".into(),
            ip: "10.0.0.10".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Regular),
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };
        let key = "team-a/worker-0";
        let worker_url = pod_info.worker_url(port);
        tracked_pods
            .lock()
            .unwrap()
            .insert(key.to_string(), pod_info);
        register_test_worker(&app_context, &worker_url, WorkerType::Regular);

        let mut delete_pod = create_k8s_pod(Some("worker-0"), None, None, None, None);
        delete_pod.metadata.namespace = Some("team-a".to_string());
        assert!(PodInfo::from_pod(&delete_pod, None).is_none());

        handle_pod_deletion_by_key(
            &tracking_key(&delete_pod),
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
            metrics_labels::DEREGISTRATION_POD_DELETED,
        )
        .await;

        assert!(!tracked_pods.lock().unwrap().contains_key(key));
        wait_for_worker_absent(&app_context, &worker_url).await;
    }

    #[tokio::test]
    async fn test_init_done_reconciles_stale_tracked_pods() {
        let app_context = create_test_app_context_with_job_queue();
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let port = 8080u16;
        let stale = make_pod_info_with_uid("stale", "uid-stale", "10.0.0.20", "Running", true);
        let live = make_pod_info_with_uid("live", "uid-live", "10.0.0.21", "Running", true);
        let stale_url = stale.worker_url(port);
        let live_url = live.worker_url(port);
        tracked_pods
            .lock()
            .unwrap()
            .extend(pod_map(vec![("ns/stale", stale), ("ns/live", live)]));
        register_test_worker(&app_context, &stale_url, WorkerType::Regular);
        register_test_worker(&app_context, &live_url, WorkerType::Regular);

        let init_snapshot = HashSet::from(["ns/live".to_string()]);
        reconcile_stale_snapshot(
            Arc::clone(&tracked_pods),
            &init_snapshot,
            Arc::clone(&app_context),
            port,
        )
        .await;
        reconcile_stale_snapshot(
            Arc::clone(&tracked_pods),
            &init_snapshot,
            Arc::clone(&app_context),
            port,
        )
        .await;

        assert!(!tracked_pods.lock().unwrap().contains_key("ns/stale"));
        assert!(tracked_pods.lock().unwrap().contains_key("ns/live"));
        wait_for_worker_absent(&app_context, &stale_url).await;
        assert!(app_context.worker_registry.get_by_url(&live_url).is_some());
    }

    #[tokio::test]
    async fn test_apply_url_change_removes_old_and_tracks_new_url() {
        let app_context = create_test_app_context_with_job_queue();
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let port = 8080u16;
        let old = make_pod_info_with_uid("worker-0", "uid-old", "10.0.0.30", "Running", true);
        let new = make_pod_info_with_uid("worker-0", "uid-new", "10.0.0.31", "Running", true);
        let old_url = old.worker_url(port);
        let new_url = new.worker_url(port);
        tracked_pods
            .lock()
            .unwrap()
            .insert("ns/worker-0".to_string(), old);
        register_test_worker(&app_context, &old_url, WorkerType::Regular);

        handle_pod_event(
            "ns/worker-0",
            &new,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
            false,
        )
        .await;

        assert_eq!(tracked_pods.lock().unwrap().get("ns/worker-0"), Some(&new));
        wait_for_worker_absent(&app_context, &old_url).await;
        let new_status = app_context
            .worker_job_queue
            .get()
            .and_then(|queue| queue.get_status(&new_url));
        assert!(
            new_status.is_some(),
            "new worker add job should be submitted"
        );
    }

    #[tokio::test]
    async fn test_apply_with_selector_removed_deregisters_tracked_pod() {
        let app_context = create_test_app_context_with_job_queue();
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let mut config = make_regular_config();
        config.port = 8080;
        let config = Arc::new(config);
        let port = config.port;
        let pod_info =
            make_pod_info_with_uid("worker-0", "uid-worker-0", "10.0.0.40", "Running", true);
        let worker_url = pod_info.worker_url(port);
        tracked_pods
            .lock()
            .unwrap()
            .insert("ns/worker-0".to_string(), pod_info);
        register_test_worker(&app_context, &worker_url, WorkerType::Regular);

        let mut pod = make_labeled_pod("worker-0", "10.0.0.40", &[("app", "other")]);
        pod.metadata.namespace = Some("ns".to_string());

        handle_pod_apply_event(
            pod,
            "ns/worker-0".to_string(),
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            config,
            port,
        )
        .await;

        assert!(!tracked_pods.lock().unwrap().contains_key("ns/worker-0"));
        wait_for_worker_absent(&app_context, &worker_url).await;
    }

    // ========== ModelIdSource tests ==========

    #[test]
    fn test_model_id_source_parse_namespace() {
        let source = ModelIdSource::parse("namespace").unwrap();
        assert!(matches!(source, ModelIdSource::Namespace));
    }

    #[test]
    fn test_model_id_source_parse_namespace_case_insensitive() {
        let source = ModelIdSource::parse("Namespace").unwrap();
        assert!(matches!(source, ModelIdSource::Namespace));
    }

    #[test]
    fn test_model_id_source_parse_label() {
        let source = ModelIdSource::parse("label:model-name").unwrap();
        match source {
            ModelIdSource::Label(key) => assert_eq!(key, "model-name"),
            _ => panic!("Expected Label variant"),
        }
    }

    #[test]
    fn test_model_id_source_parse_annotation() {
        let source = ModelIdSource::parse("annotation:serving.example.com/model-id").unwrap();
        match source {
            ModelIdSource::Annotation(key) => {
                assert_eq!(key, "serving.example.com/model-id");
            }
            _ => panic!("Expected Annotation variant"),
        }
    }

    #[test]
    fn test_model_id_source_parse_label_empty_key() {
        assert!(ModelIdSource::parse("label:").is_err());
    }

    #[test]
    fn test_model_id_source_parse_annotation_empty_key() {
        assert!(ModelIdSource::parse("annotation:").is_err());
    }

    #[test]
    fn test_model_id_source_parse_invalid() {
        assert!(ModelIdSource::parse("hostname").is_err());
        assert!(ModelIdSource::parse("").is_err());
    }

    #[test]
    fn test_model_id_source_extract_namespace() {
        let source = ModelIdSource::Namespace;
        let pod = Pod {
            metadata: ObjectMeta {
                name: Some("pod1".to_string()),
                namespace: Some("team-a-serving".to_string()),
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: None,
        };
        assert_eq!(source.extract(&pod), Some("team-a-serving".to_string()));
    }

    #[test]
    fn test_model_id_source_extract_namespace_missing() {
        let source = ModelIdSource::Namespace;
        let pod = Pod {
            metadata: ObjectMeta {
                name: Some("pod1".to_string()),
                namespace: None,
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: None,
        };
        assert_eq!(source.extract(&pod), None);
    }

    #[test]
    fn test_model_id_source_extract_label() {
        let source = ModelIdSource::Label("model-name".to_string());
        let mut labels = std::collections::BTreeMap::new();
        labels.insert("model-name".to_string(), "llama-70b".to_string());
        let pod = Pod {
            metadata: ObjectMeta {
                name: Some("pod1".to_string()),
                labels: Some(labels),
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: None,
        };
        assert_eq!(source.extract(&pod), Some("llama-70b".to_string()));
    }

    #[test]
    fn test_model_id_source_extract_label_missing() {
        let source = ModelIdSource::Label("model-name".to_string());
        let pod = Pod {
            metadata: ObjectMeta {
                name: Some("pod1".to_string()),
                labels: None,
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: None,
        };
        assert_eq!(source.extract(&pod), None);
    }

    #[test]
    fn test_model_id_source_extract_annotation() {
        let source = ModelIdSource::Annotation("serving.example.com/model-id".to_string());
        let mut annotations = std::collections::BTreeMap::new();
        annotations.insert(
            "serving.example.com/model-id".to_string(),
            "my-model".to_string(),
        );
        let pod = Pod {
            metadata: ObjectMeta {
                name: Some("pod1".to_string()),
                annotations: Some(annotations),
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: None,
        };
        assert_eq!(source.extract(&pod), Some("my-model".to_string()));
    }

    #[test]
    fn test_pod_info_from_pod_with_model_id_override() {
        let mut pod = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            Some("True"),
            None,
        );
        pod.metadata.namespace = Some("team-a".to_string());

        let config = ServiceDiscoveryConfig {
            model_id_source: Some(ModelIdSource::Namespace),
            ..Default::default()
        };

        let info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
        assert_eq!(info.model_id_override, Some("team-a".to_string()));
    }

    #[test]
    fn test_pod_info_from_pod_without_model_id_source() {
        let pod = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            Some("True"),
            None,
        );

        let config = ServiceDiscoveryConfig::default();
        let info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
        assert_eq!(info.model_id_override, None);
    }

    // ========== Reconciliation helper tests ==========

    fn make_pod_info(name: &str, ip: &str, status: &str, is_ready: bool) -> PodInfo {
        make_pod_info_with_uid(name, name, ip, status, is_ready)
    }

    fn make_pod_info_with_uid(
        name: &str,
        uid: &str,
        ip: &str,
        status: &str,
        is_ready: bool,
    ) -> PodInfo {
        PodInfo {
            name: name.into(),
            uid: uid.into(),
            ip: ip.into(),
            status: status.into(),
            is_ready,
            pod_type: Some(PodType::Regular),
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        }
    }

    fn pod_map(entries: Vec<(&str, PodInfo)>) -> TrackedPods {
        entries
            .into_iter()
            .map(|(key, pod)| (key.to_string(), pod))
            .collect()
    }

    fn make_regular_config() -> ServiceDiscoveryConfig {
        let mut selector = HashMap::new();
        selector.insert("app".to_string(), "sglang".to_string());
        ServiceDiscoveryConfig {
            enabled: true,
            selector,
            pd_mode: false,
            ..Default::default()
        }
    }

    fn make_labeled_pod(name: &str, ip: &str, labels: &[(&str, &str)]) -> Pod {
        let mut label_map = std::collections::BTreeMap::new();
        for &(k, v) in labels {
            label_map.insert(k.to_string(), v.to_string());
        }
        Pod {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                uid: Some(format!("uid-{name}")),
                labels: Some(label_map),
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: Some(PodStatus {
                pod_ip: Some(ip.to_string()),
                phase: Some("Running".to_string()),
                conditions: Some(vec![PodCondition {
                    type_: "Ready".to_string(),
                    status: "True".to_string(),
                    last_probe_time: None,
                    last_transition_time: None,
                    message: None,
                    reason: None,
                    observed_generation: None,
                }]),
                ..Default::default()
            }),
        }
    }

    fn make_router_config() -> ServiceDiscoveryConfig {
        let mut router_selector = HashMap::new();
        router_selector.insert("app".to_string(), "smg-router".to_string());
        ServiceDiscoveryConfig {
            router_selector,
            router_mesh_port_annotation: "sglang.ai/mesh-port".to_string(),
            ..Default::default()
        }
    }

    fn make_router_pod(name: &str, namespace: &str, ip: &str, healthy: bool) -> Pod {
        let mut pod = make_labeled_pod(name, ip, &[("app", "smg-router")]);
        pod.metadata.namespace = Some(namespace.to_string());
        let mut annotations = std::collections::BTreeMap::new();
        annotations.insert("sglang.ai/mesh-port".to_string(), "29001".to_string());
        pod.metadata.annotations = Some(annotations);
        if !healthy {
            pod.status.as_mut().unwrap().conditions = Some(vec![PodCondition {
                type_: "Ready".to_string(),
                status: "False".to_string(),
                last_probe_time: None,
                last_transition_time: None,
                message: None,
                reason: None,
                observed_generation: None,
            }]);
        }
        pod
    }

    #[test]
    fn test_build_live_pod_set_includes_matching_pods() {
        let config = make_regular_config();
        let pods = vec![
            make_labeled_pod("pod-a", "10.0.0.1", &[("app", "sglang")]),
            make_labeled_pod("pod-b", "10.0.0.2", &[("app", "sglang")]),
        ];

        let live = build_live_pod_set(&pods, &config);
        assert_eq!(live.len(), 2);
        assert!(live.values().any(|p| p.name == "pod-a"));
        assert!(live.values().any(|p| p.name == "pod-b"));
    }

    #[test]
    fn test_tracking_key_includes_namespace() {
        let mut pod_a = make_labeled_pod("worker-0", "10.0.0.1", &[("app", "sglang")]);
        pod_a.metadata.namespace = Some("team-a".to_string());
        let mut pod_b = make_labeled_pod("worker-0", "10.0.0.2", &[("app", "sglang")]);
        pod_b.metadata.namespace = Some("team-b".to_string());

        assert_eq!(tracking_key(&pod_a), "team-a/worker-0");
        assert_eq!(tracking_key(&pod_b), "team-b/worker-0");
        assert_ne!(tracking_key(&pod_a), tracking_key(&pod_b));
    }

    #[test]
    fn test_build_live_pod_set_excludes_non_matching_pods() {
        let config = make_regular_config();
        let pods = vec![
            make_labeled_pod("pod-a", "10.0.0.1", &[("app", "sglang")]),
            make_labeled_pod("pod-b", "10.0.0.2", &[("app", "other")]),
        ];

        let live = build_live_pod_set(&pods, &config);
        assert_eq!(live.len(), 1);
        assert!(live.values().any(|p| p.name == "pod-a"));
    }

    #[test]
    fn test_build_live_pod_set_excludes_pods_with_deletion_timestamp() {
        let config = make_regular_config();
        let mut deleted_pod = make_labeled_pod("pod-a", "10.0.0.1", &[("app", "sglang")]);
        deleted_pod.metadata.deletion_timestamp = Some(Time(k8s_openapi::jiff::Timestamp::now()));
        let live_pod = make_labeled_pod("pod-b", "10.0.0.2", &[("app", "sglang")]);

        let live = build_live_pod_set(&[deleted_pod, live_pod], &config);
        assert_eq!(live.len(), 1);
        assert!(live.values().any(|p| p.name == "pod-b"));
    }

    #[test]
    fn test_build_live_pod_set_empty_list() {
        let config = make_regular_config();
        let live = build_live_pod_set(&[], &config);
        assert!(live.is_empty());
    }

    #[test]
    fn test_build_live_pod_set_pd_mode() {
        let config = create_pd_config();
        let pods = vec![
            create_pd_k8s_pod("prefill-0", "10.0.0.1", "prefill", Some(8081)),
            create_pd_k8s_pod("decode-0", "10.0.0.2", "decode", None),
            create_pd_k8s_pod("other-0", "10.0.0.3", "other", None),
        ];

        let live = build_live_pod_set(&pods, &config);
        // Only prefill and decode selectors match; "other" does not
        assert_eq!(live.len(), 2);
        assert!(live.values().any(|p| p.name == "prefill-0"));
        assert!(live.values().any(|p| p.name == "decode-0"));
    }

    #[test]
    fn test_compute_reconciliation_diff_no_changes() {
        let pod = make_pod_info("pod-a", "10.0.0.1", "Running", true);
        let tracked = pod_map(vec![("default/pod-a", pod.clone())]);
        let live = pod_map(vec![("default/pod-a", pod)]);

        let (stale, missing) = compute_reconciliation_diff(&tracked, &live);
        assert!(stale.is_empty());
        assert!(missing.is_empty());
    }

    #[test]
    fn test_compute_reconciliation_diff_stale_pod() {
        let tracked_pod = make_pod_info("pod-a", "10.0.0.1", "Running", true);
        let tracked = pod_map(vec![("default/pod-a", tracked_pod.clone())]);
        let live = HashMap::new();

        let (stale, missing) = compute_reconciliation_diff(&tracked, &live);
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].1.name, "pod-a");
        assert!(missing.is_empty());
    }

    #[test]
    fn test_compute_reconciliation_diff_missing_healthy_pod() {
        let tracked = HashMap::new();
        let live_pod = make_pod_info("pod-b", "10.0.0.2", "Running", true);
        let live = pod_map(vec![("default/pod-b", live_pod)]);

        let (stale, missing) = compute_reconciliation_diff(&tracked, &live);
        assert!(stale.is_empty());
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].1.name, "pod-b");
    }

    #[test]
    fn test_compute_reconciliation_diff_missing_unhealthy_pod_excluded() {
        let tracked = HashMap::new();
        // Pod exists in K8s but is not ready — should NOT be added
        let unhealthy_pod = make_pod_info("pod-c", "10.0.0.3", "Running", false);
        let live = pod_map(vec![("default/pod-c", unhealthy_pod)]);

        let (stale, missing) = compute_reconciliation_diff(&tracked, &live);
        assert!(stale.is_empty());
        assert!(missing.is_empty());
    }

    #[test]
    fn test_compute_reconciliation_diff_missing_pending_pod_excluded() {
        let tracked = HashMap::new();
        // Pod is "Pending" with is_ready=true — is_healthy() returns false
        let pending_pod = make_pod_info("pod-d", "10.0.0.4", "Pending", true);
        let live = pod_map(vec![("default/pod-d", pending_pod)]);

        let (stale, missing) = compute_reconciliation_diff(&tracked, &live);
        assert!(stale.is_empty());
        assert!(missing.is_empty());
    }

    #[test]
    fn test_compute_reconciliation_diff_mixed() {
        // tracked: {A, B}, live: {B, C (healthy), D (unhealthy)}
        // Expected: stale=[A], missing=[C]
        let pod_a = make_pod_info("pod-a", "10.0.0.1", "Running", true);
        let pod_b = make_pod_info("pod-b", "10.0.0.2", "Running", true);
        let pod_c = make_pod_info("pod-c", "10.0.0.3", "Running", true);
        let pod_d = make_pod_info("pod-d", "10.0.0.4", "Running", false);

        let tracked = pod_map(vec![
            ("default/pod-a", pod_a.clone()),
            ("default/pod-b", pod_b.clone()),
        ]);
        let live = pod_map(vec![
            ("default/pod-b", pod_b),
            ("default/pod-c", pod_c.clone()),
            ("default/pod-d", pod_d),
        ]);

        let (stale, missing) = compute_reconciliation_diff(&tracked, &live);
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].1.name, "pod-a");
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].1.name, "pod-c");
    }

    #[test]
    fn test_compute_reconciliation_diff_both_empty() {
        let tracked = HashMap::new();
        let live = HashMap::new();

        let (stale, missing) = compute_reconciliation_diff(&tracked, &live);
        assert!(stale.is_empty());
        assert!(missing.is_empty());
    }

    #[test]
    fn test_build_live_pod_set_includes_unhealthy_non_deleted_pods() {
        // Reconciliation should include unhealthy pods in live set (they are not stale),
        // but compute_reconciliation_diff will filter them out of "missing" additions.
        let config = make_regular_config();
        let mut unhealthy_pod = make_labeled_pod("pod-a", "10.0.0.1", &[("app", "sglang")]);
        // Make it not-ready
        if let Some(ref mut status) = unhealthy_pod.status {
            status.conditions = Some(vec![PodCondition {
                type_: "Ready".to_string(),
                status: "False".to_string(),
                last_probe_time: None,
                last_transition_time: None,
                message: None,
                reason: None,
                observed_generation: None,
            }]);
        }

        let live = build_live_pod_set(&[unhealthy_pod], &config);
        // Pod should still be in the live set (not considered stale)
        assert_eq!(live.len(), 1);
        assert!(!live.values().next().unwrap().is_ready);
    }

    #[test]
    fn test_reconciliation_readiness_change_not_considered_stale() {
        // A tracked pod that exists in K8s but changed readiness should NOT be
        // considered stale — PodInfo identity is (name, uid), not full state.
        let tracked_pod = make_pod_info("pod-a", "10.0.0.1", "Running", true);
        let live_pod = make_pod_info("pod-a", "10.0.0.1", "Running", false);

        let tracked = pod_map(vec![("default/pod-a", tracked_pod)]);
        let live = pod_map(vec![("default/pod-a", live_pod)]);

        let (stale, missing) = compute_reconciliation_diff(&tracked, &live);
        // Same (name, uid) means the pod is recognized as the same entity.
        // Not stale (still in K8s), not missing (already tracked).
        assert!(stale.is_empty());
        assert!(missing.is_empty());
    }

    #[test]
    fn test_reconciliation_detects_pod_restart_same_name_and_ip() {
        // LWS / StatefulSet pods keep the same name across restarts,
        // and hostNetwork pods keep the same IP (the node IP).
        // The uid changes on every restart, so reconciliation must
        // detect the old instance as stale and the new one as missing.
        let old_pod = make_pod_info_with_uid("worker-0", "uid-old", "10.0.0.1", "Running", true);
        let new_pod = make_pod_info_with_uid("worker-0", "uid-new", "10.0.0.1", "Running", true);

        let tracked = pod_map(vec![("default/worker-0", old_pod)]);
        let live = pod_map(vec![("default/worker-0", new_pod)]);

        let (stale, missing) = compute_reconciliation_diff(&tracked, &live);
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].1.uid, "uid-old");
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].1.uid, "uid-new");
    }

    #[test]
    fn test_reconciliation_distinguishes_same_name_different_namespaces() {
        let team_a = make_pod_info_with_uid("worker-0", "uid-a", "10.0.0.1", "Running", true);
        let team_b = make_pod_info_with_uid("worker-0", "uid-b", "10.0.0.2", "Running", true);
        let tracked = pod_map(vec![("team-a/worker-0", team_a.clone())]);
        let live = pod_map(vec![
            ("team-a/worker-0", team_a),
            ("team-b/worker-0", team_b),
        ]);

        let (stale, missing) = compute_reconciliation_diff(&tracked, &live);
        assert!(stale.is_empty());
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].0, "team-b/worker-0");
        assert_eq!(missing[0].1.ip, "10.0.0.2");
    }

    #[test]
    fn test_pod_info_from_pod_missing_uid() {
        let mut k8s_pod = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            Some("True"),
            None,
        );
        k8s_pod.metadata.uid = None;
        assert!(PodInfo::from_pod(&k8s_pod, None).is_none());
    }

    #[test]
    fn test_pod_info_hash_consistent_with_eq() {
        use std::{
            collections::hash_map::DefaultHasher,
            hash::{Hash, Hasher},
        };

        let pod1 = PodInfo {
            name: "pod1".into(),
            uid: "uid-1".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Prefill),
            bootstrap_port: Some(8081),
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };
        let pod2 = PodInfo {
            name: "pod1".into(),
            uid: "uid-1".into(),
            ip: "5.6.7.8".into(),
            status: "Pending".into(),
            is_ready: false,
            pod_type: Some(PodType::Decode),
            bootstrap_port: None,
            is_router: true,
            mesh_port: Some(9090),
            model_id_override: Some("model".into()),
        };

        assert_eq!(pod1, pod2);

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        pod1.hash(&mut h1);
        pod2.hash(&mut h2);
        assert_eq!(
            h1.finish(),
            h2.finish(),
            "Equal PodInfos must produce the same hash"
        );
    }

    #[test]
    fn test_router_delete_marks_tracked_node_down() {
        let config = Arc::new(make_router_config());
        let cluster_state: ClusterState = Arc::new(parking_lot::RwLock::new(Default::default()));
        let mut tracked_router_nodes = HashMap::new();
        let pod = make_router_pod("router-0", "ns", "10.0.1.10", true);
        let key = tracking_key(&pod);

        handle_router_apply_event(
            pod,
            &key,
            Arc::clone(&config),
            Arc::clone(&cluster_state),
            29000,
            &mut tracked_router_nodes,
        );
        handle_router_delete_event(&key, Arc::clone(&cluster_state), &mut tracked_router_nodes);

        let state = cluster_state.read();
        assert_eq!(
            state.get("router-0").unwrap().status,
            NodeStatus::Down as i32
        );
        assert!(!tracked_router_nodes.contains_key(&key));
        assert!(!state.contains_key(&key));
    }

    #[test]
    fn test_router_apply_uses_pod_name_for_mesh_node_identity() {
        let config = Arc::new(make_router_config());
        let cluster_state: ClusterState = Arc::new(parking_lot::RwLock::new(Default::default()));
        let mut tracked_router_nodes = HashMap::new();
        let pod = make_router_pod("router-0", "ns", "10.0.1.10", true);
        let key = tracking_key(&pod);

        handle_router_apply_event(
            pod,
            &key,
            Arc::clone(&config),
            Arc::clone(&cluster_state),
            29000,
            &mut tracked_router_nodes,
        );

        let state = cluster_state.read();
        let node = state.get("router-0").unwrap();
        assert_eq!(node.name, "router-0");
        assert_eq!(
            tracked_router_nodes.get(&key),
            Some(&"router-0".to_string())
        );
        assert!(!state.contains_key(&key));
    }

    #[test]
    fn test_router_init_done_marks_absent_tracked_nodes_down() {
        let config = Arc::new(make_router_config());
        let cluster_state: ClusterState = Arc::new(parking_lot::RwLock::new(Default::default()));
        let mut tracked_router_nodes = HashMap::new();
        let stale_pod = make_router_pod("router-stale", "ns", "10.0.1.11", true);
        let live_pod = make_router_pod("router-live", "ns", "10.0.1.12", true);
        let stale_key = tracking_key(&stale_pod);
        let live_key = tracking_key(&live_pod);

        handle_router_apply_event(
            stale_pod,
            &stale_key,
            Arc::clone(&config),
            Arc::clone(&cluster_state),
            29000,
            &mut tracked_router_nodes,
        );
        handle_router_apply_event(
            live_pod,
            &live_key,
            Arc::clone(&config),
            Arc::clone(&cluster_state),
            29000,
            &mut tracked_router_nodes,
        );

        reconcile_router_snapshot(
            &mut tracked_router_nodes,
            &HashSet::from([live_key.clone()]),
            Arc::clone(&cluster_state),
        );

        let state = cluster_state.read();
        assert_eq!(
            state.get("router-stale").unwrap().status,
            NodeStatus::Down as i32
        );
        assert_eq!(
            state.get("router-live").unwrap().status,
            NodeStatus::Alive as i32
        );
        assert!(!tracked_router_nodes.contains_key(&stale_key));
        assert!(tracked_router_nodes.contains_key(&live_key));
        assert!(!state.contains_key(&stale_key));
        assert!(!state.contains_key(&live_key));
    }

    #[tokio::test]
    async fn test_handle_pod_event_evicts_old_uid_on_restart() {
        // When a StatefulSet pod restarts with same name but new UID,
        // handle_pod_event should evict the old entry and insert the new one.
        let app_context = create_test_app_context();
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let port = 8080u16;

        let old_pod = PodInfo {
            name: "worker-0".into(),
            uid: "uid-old".into(),
            ip: "10.0.0.1".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Regular),
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };
        // Pre-populate tracked set with the old pod.
        tracked_pods
            .lock()
            .unwrap()
            .insert("default/worker-0".to_string(), old_pod.clone());

        let new_pod = PodInfo {
            name: "worker-0".into(),
            uid: "uid-new".into(),
            ip: "10.0.0.1".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Regular),
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };

        handle_pod_event(
            "default/worker-0",
            &new_pod,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
            false,
        )
        .await;

        let tracker = tracked_pods.lock().unwrap();
        // Old pod should be evicted, new pod should be present.
        assert_eq!(tracker.len(), 1);
        assert_eq!(tracker.get("default/worker-0"), Some(&new_pod));
        assert_ne!(tracker.get("default/worker-0"), Some(&old_pod));
    }

    #[test]
    fn test_list_label_selector_regular_mode() {
        let mut selector = HashMap::new();
        selector.insert("app".to_string(), "sglang".to_string());
        let config = ServiceDiscoveryConfig {
            selector,
            pd_mode: false,
            ..Default::default()
        };
        assert_eq!(config.list_label_selector(), "app=sglang");
    }

    #[test]
    fn test_list_label_selector_pd_mode_common_labels() {
        let mut prefill = HashMap::new();
        prefill.insert("app".to_string(), "sglang".to_string());
        prefill.insert("component".to_string(), "prefill".to_string());
        let mut decode = HashMap::new();
        decode.insert("app".to_string(), "sglang".to_string());
        decode.insert("component".to_string(), "decode".to_string());
        let config = ServiceDiscoveryConfig {
            pd_mode: true,
            prefill_selector: prefill,
            decode_selector: decode,
            ..Default::default()
        };
        // Only the common label "app=sglang" should be in the selector.
        assert_eq!(config.list_label_selector(), "app=sglang");
    }

    #[test]
    fn test_list_label_selector_pd_mode_no_common_labels() {
        let mut prefill = HashMap::new();
        prefill.insert("role".to_string(), "prefill".to_string());
        let mut decode = HashMap::new();
        decode.insert("role".to_string(), "decode".to_string());
        let config = ServiceDiscoveryConfig {
            pd_mode: true,
            prefill_selector: prefill,
            decode_selector: decode,
            ..Default::default()
        };
        // No common labels → empty selector (falls back to listing all pods).
        assert!(config.list_label_selector().is_empty());
    }

    #[test]
    fn test_deregistration_reconciled_metric_label() {
        // Verify the metric label constant exists and has expected value
        assert_eq!(metrics_labels::DEREGISTRATION_RECONCILED, "reconciled");
    }

    #[test]
    fn test_build_watcher_config_with_selector_pushes_label_selector() {
        let cfg = build_watcher_config("worker", "app=sglang");
        assert_eq!(cfg.label_selector.as_deref(), Some("app=sglang"));
    }

    #[test]
    fn test_build_watcher_config_empty_selector_falls_back_to_default() {
        let cfg = build_watcher_config("worker", "");
        assert!(cfg.label_selector.is_none());
    }

    #[test]
    fn test_build_watcher_config_for_regular_mode_pushes_worker_selector() {
        let mut selector = HashMap::new();
        selector.insert("app".to_string(), "sglang".to_string());
        let config = ServiceDiscoveryConfig {
            selector,
            pd_mode: false,
            ..Default::default()
        };
        let watcher_config = build_watcher_config("worker", &config.list_label_selector());
        assert_eq!(watcher_config.label_selector.as_deref(), Some("app=sglang"));
    }

    #[test]
    fn test_build_watcher_config_for_pd_mode_pushes_intersection() {
        let mut prefill = HashMap::new();
        prefill.insert("app".to_string(), "sglang".to_string());
        prefill.insert("component".to_string(), "prefill".to_string());
        let mut decode = HashMap::new();
        decode.insert("app".to_string(), "sglang".to_string());
        decode.insert("component".to_string(), "decode".to_string());
        let config = ServiceDiscoveryConfig {
            pd_mode: true,
            prefill_selector: prefill,
            decode_selector: decode,
            ..Default::default()
        };
        let watcher_config = build_watcher_config("worker", &config.list_label_selector());
        assert_eq!(watcher_config.label_selector.as_deref(), Some("app=sglang"));
    }

    #[test]
    fn test_build_watcher_config_for_pd_mode_no_common_labels_omits_filter() {
        let mut prefill = HashMap::new();
        prefill.insert("role".to_string(), "prefill".to_string());
        let mut decode = HashMap::new();
        decode.insert("role".to_string(), "decode".to_string());
        let config = ServiceDiscoveryConfig {
            pd_mode: true,
            prefill_selector: prefill,
            decode_selector: decode,
            ..Default::default()
        };
        let watcher_config = build_watcher_config("worker", &config.list_label_selector());
        assert!(watcher_config.label_selector.is_none());
    }

    #[test]
    fn test_router_label_selector_serializes_router_selector() {
        let mut router = HashMap::new();
        router.insert("app".to_string(), "smg".to_string());
        let config = ServiceDiscoveryConfig {
            router_selector: router,
            ..Default::default()
        };
        assert_eq!(config.router_label_selector(), "app=smg");
    }

    #[test]
    fn test_router_label_selector_empty_when_unset() {
        let config = ServiceDiscoveryConfig::default();
        assert!(config.router_label_selector().is_empty());
    }

    #[test]
    fn test_build_watcher_config_for_router_pushes_router_selector() {
        let mut router = HashMap::new();
        router.insert("app".to_string(), "smg".to_string());
        let config = ServiceDiscoveryConfig {
            router_selector: router,
            ..Default::default()
        };
        let watcher_config = build_watcher_config("router", &config.router_label_selector());
        assert_eq!(watcher_config.label_selector.as_deref(), Some("app=smg"));
    }

    #[traced_test]
    #[test]
    fn test_build_watcher_config_logs_selector_at_info_level() {
        let _ = build_watcher_config("worker", "app=sglang");
        assert!(logs_contain("Starting K8s worker watcher"));
        assert!(logs_contain("app=sglang"));
    }

    #[traced_test]
    #[test]
    fn test_build_watcher_config_logs_router_kind_with_empty_selector() {
        let _ = build_watcher_config("router", "");
        assert!(logs_contain("Starting K8s router watcher"));
        assert!(logs_contain("selector: ''"));
    }
}
